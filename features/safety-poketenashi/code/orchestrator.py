"""Poketenashi orchestrator — detects 5 prohibited/required behaviors.

Behaviors:
  phone_usage          — ML detection (external sub-model, passed as phone_detections)
  hands_in_pockets     — pose rule
  stair_diagonal       — pose + trajectory rule
  no_handrail          — pose + zone rule
  no_pointing_calling  — pose rule

Run smoke test:
    uv run features/safety-poketenashi/code/orchestrator.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
_CODE_DIR = Path(__file__).parent

import sys

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(_CODE_DIR))

from utils.config import load_config
from _base import RuleResult  # noqa: E402
from hands_in_pockets_detector import HandsInPocketsDetector  # noqa: E402
from stair_safety_detector import StairSafetyDetector  # noqa: E402
from handrail_detector import HandrailDetector  # noqa: E402
from pointing_calling_detector import PointingCallingDetector  # noqa: E402

# COCO-17 skeleton connections for visualization.
_SKELETON_17 = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4),
]

_PRETRAIN = REPO / "pretrained" / "safety-poketenashi"
_DWPOSE_ONNX = _PRETRAIN / "dw-ll_ucoco_384.onnx"

# WholeBody body slice (first 17 kps).
_WB_BODY = slice(0, 17)


# ---------------------------------------------------------------------------
# DWPose ONNX wrapper (RTMPose SimCC head — reused from eval_sota.py)
# ---------------------------------------------------------------------------

class _DWPose:
    INPUT_HW = (384, 288)  # H, W

    def __init__(self, onnx_path: Path) -> None:
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            self._sess = ort.InferenceSession(str(onnx_path), providers=providers)
        except Exception:
            self._sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self._in_name = self._sess.get_inputs()[0].name

    def _affine(self, box_xyxy: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = box_xyxy
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        bw, bh = x1 - x0, y1 - y0
        oh, ow = self.INPUT_HW
        aspect = ow / oh
        if bw / (bh + 1e-9) > aspect:
            bh = bw / aspect
        else:
            bw = bh * aspect
        bw *= 1.25
        bh *= 1.25
        src = np.array([[cx, cy], [cx + bw / 2, cy], [cx, cy + bh / 2]], dtype=np.float32)
        dst = np.array([[ow / 2, oh / 2], [ow, oh / 2], [ow / 2, oh]], dtype=np.float32)
        return cv2.getAffineTransform(src, dst)

    def __call__(self, img_bgr: np.ndarray, box_xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        oh, ow = self.INPUT_HW
        M = self._affine(box_xyxy)
        crop = cv2.warpAffine(img_bgr, M, (ow, oh), flags=cv2.INTER_LINEAR)
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        x = (crop.astype(np.float32) - mean) / std
        x = x.transpose(2, 0, 1)[None]
        simcc_x, simcc_y = self._sess.run(None, {self._in_name: x})
        Minv = cv2.invertAffineTransform(M)

        # SimCC decode + affine un-warp on GPU (falls back to CPU cleanly).
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tx = torch.from_numpy(simcc_x[0]).to(device)
        ty = torch.from_numpy(simcc_y[0]).to(device)

        max_x = torch.max(tx, dim=-1)
        max_y = torch.max(ty, dim=-1)
        sx = max_x.indices.to(torch.float32) / 2.0
        sy = max_y.indices.to(torch.float32) / 2.0
        scores_t = torch.minimum(max_x.values, max_y.values)

        ones = torch.ones_like(sx)
        pts_in = torch.stack([sx, sy, ones], dim=1)  # (K, 3)
        Minv_t = torch.from_numpy(Minv).to(device=device, dtype=torch.float32)
        pts_orig_t = pts_in @ Minv_t.T  # (K, 2)

        pts_orig = pts_orig_t.cpu().numpy().astype(np.float32)
        scores = scores_t.cpu().numpy().astype(np.float32)
        return pts_orig, scores


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PersonBehavior:
    track_id: int  # -1 if no tracking
    behaviors: dict[str, RuleResult]
    keypoints: np.ndarray  # (K, 2)
    kp_scores: np.ndarray  # (K,)


@dataclass
class OrchestratorResult:
    alerts: list[str]  # triggered behavior names
    persons: list[PersonBehavior]
    latency_ms: float


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class PoketanashiOrchestrator:
    """Detect 5 prohibited/required behaviors using pose estimation + pose rules."""

    def __init__(self, config_path: str | Path) -> None:
        self._cfg = load_config(config_path)
        self._pose_model, self._pose_backend = self._load_pose_model()
        self._rules = self._build_rules()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_pose_model(self) -> tuple[Any, str]:
        if _DWPOSE_ONNX.exists():
            try:
                model = _DWPose(_DWPOSE_ONNX)
                print(f"[poketenashi] DWPose ONNX loaded: {_DWPOSE_ONNX.name}")
                return model, "dwpose"
            except Exception as exc:
                print(f"[poketenashi] DWPose load failed ({exc}), falling back to whole-frame mode")

        return None, "none"

    # ------------------------------------------------------------------
    # Rule construction from config
    # ------------------------------------------------------------------

    def _build_rules(self) -> list:
        pr = self._cfg.get("pose_rules", {})

        hip_cfg = pr.get("hands_in_pockets", {})
        stair_cfg = pr.get("stair_diagonal", {})
        rail_cfg = pr.get("no_handrail", {})
        point_cfg = pr.get("no_pointing_calling", {})

        handrail_zones = self._cfg.get("handrail_zones", [])

        return [
            HandsInPocketsDetector(
                wrist_below_hip_ratio=hip_cfg.get("wrist_below_hip_ratio", 0.05),
                wrist_inside_torso_margin=hip_cfg.get("wrist_inside_torso_margin", 0.08),
            ),
            StairSafetyDetector(
                max_diagonal_angle_deg=stair_cfg.get("max_diagonal_angle_deg", 20.0),
            ),
            HandrailDetector(
                handrail_zones=handrail_zones,
                hand_to_railing_px=float(rail_cfg.get("hand_to_railing_px", 60)),
            ),
            PointingCallingDetector(
                elbow_wrist_angle_min_deg=float(point_cfg.get("elbow_wrist_angle_min_deg", 150.0)),
                pointing_duration_frames=int(point_cfg.get("pointing_duration_frames", 20)),
            ),
        ]

    # ------------------------------------------------------------------
    # Pose estimation per frame
    # ------------------------------------------------------------------

    def _run_pose(self, image_bgr: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Return list of (keypoints_17, scores_17, box_xyxy) per detected person."""
        persons = []

        if self._pose_backend == "dwpose":
            # Need person boxes first — use a lightweight YOLO detector or whole-frame fallback.
            boxes = self._detect_persons(image_bgr)
            for box in boxes:
                kpts, scores = self._pose_model(image_bgr, box)
                # DWPose returns 133 kps; take first 17 (body).
                kpts17 = kpts[_WB_BODY]
                sc17 = scores[_WB_BODY]
                persons.append((kpts17, sc17, box))
        else:
            # YOLOv8n-pose returns all persons in one call.
            results = self._pose_model.predict(image_bgr, conf=0.35, verbose=False)[0]
            if results.keypoints is None or len(results.keypoints) == 0:
                return []
            kps_xy = results.keypoints.xy.cpu().numpy()  # (N, 17, 2)
            kps_conf = results.keypoints.conf.cpu().numpy()  # (N, 17)
            boxes = results.boxes.xyxy.cpu().numpy()  # (N, 4)
            for i in range(len(kps_xy)):
                persons.append((kps_xy[i], kps_conf[i], boxes[i]))

        return persons

    def _detect_persons(self, image_bgr: np.ndarray) -> list[np.ndarray]:
        """Detect person bboxes for top-down DWPose inference."""
        # Lazy-load a lightweight person detector (reuse yolo if available, else whole frame).
        if not hasattr(self, "_person_detector"):
            try:
                from ultralytics import YOLO
                _pt = Path(__file__).resolve().parents[3] / "pretrained" / "access-zone_intrusion" / "yolo11n.pt"
                self._person_detector = YOLO(str(_pt))
            except Exception:
                self._person_detector = None

        if self._person_detector is None:
            h, w = image_bgr.shape[:2]
            return [np.array([0, 0, w, h], dtype=np.float32)]

        det = self._person_detector.predict(image_bgr, classes=[0], conf=0.35, verbose=False)[0]
        if det.boxes is None or len(det.boxes) == 0:
            h, w = image_bgr.shape[:2]
            return [np.array([0, 0, w, h], dtype=np.float32)]
        return [b for b in det.boxes.xyxy.cpu().numpy()]

    # ------------------------------------------------------------------
    # Phone overlap check
    # ------------------------------------------------------------------

    def _check_phone_overlap(
        self,
        person_box: np.ndarray,
        phone_detections: list[dict],
    ) -> bool:
        """Return True if any phone detection overlaps with the person bbox."""
        px1, py1, px2, py2 = person_box[:4]
        for det in phone_detections:
            bx1, by1, bx2, by2 = det.get("bbox", [0, 0, 0, 0])
            # IoU-like: check intersection.
            ix1, iy1 = max(px1, bx1), max(py1, by1)
            ix2, iy2 = min(px2, bx2), min(py2, by2)
            if ix2 > ix1 and iy2 > iy1:
                return True
        return False

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_frame(
        self,
        image_bgr: np.ndarray,
        phone_detections: list[dict] | None = None,
    ) -> OrchestratorResult:
        t0 = time.perf_counter()

        pose_persons = self._run_pose(image_bgr)
        person_behaviors: list[PersonBehavior] = []
        all_alerts: list[str] = []

        for i, (kpts, scores, box) in enumerate(pose_persons):
            behaviors: dict[str, RuleResult] = {}

            for rule in self._rules:
                result = rule.check(kpts, scores)
                behaviors[result.behavior] = result
                if result.triggered and result.behavior not in all_alerts:
                    all_alerts.append(result.behavior)

            # Phone usage from external sub-model.
            if phone_detections:
                phone_hit = self._check_phone_overlap(box, phone_detections)
                phone_conf = max(
                    (d.get("confidence", 0.0) for d in phone_detections), default=0.0
                )
                behaviors["phone_usage"] = RuleResult(
                    triggered=phone_hit,
                    confidence=float(phone_conf),
                    behavior="phone_usage",
                )
                if phone_hit and "phone_usage" not in all_alerts:
                    all_alerts.append("phone_usage")

            person_behaviors.append(
                PersonBehavior(track_id=-1, behaviors=behaviors, keypoints=kpts, kp_scores=scores)
            )

        latency_ms = (time.perf_counter() - t0) * 1000
        return OrchestratorResult(alerts=all_alerts, persons=person_behaviors, latency_ms=latency_ms)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def draw(self, image_bgr: np.ndarray, result: OrchestratorResult) -> np.ndarray:
        out = image_bgr.copy()
        conf_th = 0.3

        for person in result.persons:
            kpts = person.keypoints
            scores = person.kp_scores

            # Skeleton.
            for a, b in _SKELETON_17:
                if a < len(kpts) and b < len(kpts) and scores[a] > conf_th and scores[b] > conf_th:
                    pa = tuple(kpts[a].astype(int))
                    pb = tuple(kpts[b].astype(int))
                    cv2.line(out, pa, pb, (0, 200, 255), 2)
            for i, pt in enumerate(kpts):
                if scores[i] > conf_th:
                    cv2.circle(out, tuple(pt.astype(int)), 3, (0, 0, 255), -1)

            # Triggered behaviors near the person's head (kp 0 = nose).
            triggered = [b for b, r in person.behaviors.items() if r.triggered]
            if triggered and scores[0] > conf_th:
                head = kpts[0].astype(int)
                for j, label in enumerate(triggered):
                    y = head[1] - 15 - j * 18
                    cv2.putText(out, label, (head[0], max(y, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1)

        # Alert summary overlay.
        if result.alerts:
            overlay_h = 16 + len(result.alerts) * 20
            cv2.rectangle(out, (0, 0), (320, overlay_h), (30, 30, 30), -1)
            for j, alert in enumerate(result.alerts):
                cv2.putText(out, f"ALERT: {alert}", (8, 14 + j * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 50, 255), 1)

        # Latency.
        cv2.putText(out, f"{result.latency_ms:.1f}ms", (out.shape[1] - 80, out.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
        return out


# ---------------------------------------------------------------------------
# Smoke test CLI
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    feat = REPO / "features" / "safety-poketenashi"
    config_path = feat / "configs" / "10_inference.yaml"
    samples_dir = feat / "samples"
    eval_dir = feat / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    orchestrator = PoketanashiOrchestrator(config_path)

    image_paths = sorted(samples_dir.glob("*.jpg"))
    assert image_paths, f"No sample images found in {samples_dir}"

    results: list[dict] = []
    print(f"\n{'Image':<30} {'Alerts':<50} {'ms':>6}")
    print("-" * 90)

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  skip {img_path.name}: unreadable")
            continue

        result = orchestrator.process_frame(img)

        # Serialize per-image result.
        persons_serialized = []
        for p in result.persons:
            b = {name: {"triggered": r.triggered, "confidence": r.confidence, **r.debug_info}
                 for name, r in p.behaviors.items()}
            persons_serialized.append({"track_id": p.track_id, "behaviors": b})

        entry = {
            "image": img_path.name,
            "alerts": result.alerts,
            "persons": persons_serialized,
            "latency_ms": round(result.latency_ms, 2),
        }
        results.append(entry)

        alert_str = ", ".join(result.alerts) if result.alerts else "none"
        print(f"  {img_path.name:<28} {alert_str:<50} {result.latency_ms:>6.1f}")

        # Save annotated image.
        annotated = orchestrator.draw(img, result)
        out_path = eval_dir / f"smoke_{img_path.name}"
        cv2.imwrite(str(out_path), annotated)

    report_path = eval_dir / "orchestrator_smoke_test.json"
    report_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {len(results)} results → {report_path}")
    print(f"Annotated images → {eval_dir}/smoke_*.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poketenashi orchestrator")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test on sample images")
    args = parser.parse_args()

    if args.smoke_test:
        _smoke_test()
    else:
        parser.print_help()

"""no_handrail predictor — DWPose + HandrailDetector orchestrator.

V1 -- pretrained-only, per-frame. Loads a DWPose ONNX backend, runs
person detection + top-down keypoint inference, then evaluates
``HandrailDetector`` against the configured zone polygons.

Run smoke test:
    uv run features/safety-poketenashi_no_handrail/code/predictor.py --smoke-test

Run on a video:
    uv run features/safety-poketenashi_no_handrail/code/predictor.py \\
        --video features/safety-poketenashi_no_handrail/samples/<clip>.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import supervision as sv
import torch

REPO = Path(__file__).resolve().parents[3]
_CODE_DIR = Path(__file__).parent

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(_CODE_DIR))

from utils.config import load_config  # noqa: E402
from utils.viz import VizStyle, annotate_keypoints, classification_banner  # noqa: E402

from _base import RuleResult  # noqa: E402
from handrail_detector import HandrailDetector  # noqa: E402

# COCO-17 skeleton edges for visualisation.
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
# DWPose ONNX wrapper (RTMPose SimCC head — copied from safety-poketenashi)
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

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tx = torch.from_numpy(simcc_x[0]).to(device)
        ty = torch.from_numpy(simcc_y[0]).to(device)

        max_x = torch.max(tx, dim=-1)
        max_y = torch.max(ty, dim=-1)
        sx = max_x.indices.to(torch.float32) / 2.0
        sy = max_y.indices.to(torch.float32) / 2.0
        scores_t = torch.minimum(max_x.values, max_y.values)

        ones = torch.ones_like(sx)
        pts_in = torch.stack([sx, sy, ones], dim=1)
        Minv_t = torch.from_numpy(Minv).to(device=device, dtype=torch.float32)
        pts_orig_t = pts_in @ Minv_t.T

        pts_orig = pts_orig_t.cpu().numpy().astype(np.float32)
        scores = scores_t.cpu().numpy().astype(np.float32)
        return pts_orig, scores


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PersonResult:
    track_id: int
    rule_result: RuleResult
    keypoints: np.ndarray
    kp_scores: np.ndarray
    box_xyxy: np.ndarray


@dataclass
class FrameResult:
    alerts: list[str] = field(default_factory=list)
    persons: list[PersonResult] = field(default_factory=list)
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class NoHandrailPredictor:
    """DWPose + HandrailDetector per-frame orchestrator."""

    def __init__(self, config_path: str | Path) -> None:
        self._cfg = load_config(config_path)
        self._pose_model = self._load_pose_model()
        self._person_detector: Any = None
        self._person_detector_loaded = False
        self._rule = self._build_rule()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_pose_model(self) -> _DWPose | None:
        if not _DWPOSE_ONNX.exists():
            print(f"[no_handrail] WARN: DWPose ONNX missing at {_DWPOSE_ONNX}")
            return None
        try:
            model = _DWPose(_DWPOSE_ONNX)
            print(f"[no_handrail] DWPose ONNX loaded: {_DWPOSE_ONNX.name}")
            return model
        except Exception as exc:
            print(f"[no_handrail] DWPose load failed ({exc}); whole-frame fallback")
            return None

    def _build_rule(self) -> HandrailDetector:
        pr = self._cfg.get("pose_rules", {}).get("no_handrail", {})
        zones = pr.get("handrail_zones", [])
        return HandrailDetector(
            handrail_zones=zones,
            hand_to_railing_px=float(pr.get("hand_to_railing_px", 60)),
        )

    # ------------------------------------------------------------------
    # Person detection
    # ------------------------------------------------------------------

    def _detect_persons(self, image_bgr: np.ndarray) -> list[np.ndarray]:
        if not self._person_detector_loaded:
            self._person_detector_loaded = True
            try:
                from ultralytics import YOLO

                pt = REPO / "pretrained" / "access-zone_intrusion" / "yolo11n.pt"
                if pt.exists():
                    self._person_detector = YOLO(str(pt))
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
    # Per-frame entry point
    # ------------------------------------------------------------------

    def process_frame(self, image_bgr: np.ndarray) -> FrameResult:
        t0 = time.perf_counter()
        result = FrameResult()

        if self._pose_model is None:
            result.latency_ms = (time.perf_counter() - t0) * 1000
            return result

        boxes = self._detect_persons(image_bgr)
        for box in boxes:
            kpts_full, scores_full = self._pose_model(image_bgr, box)
            kpts = kpts_full[_WB_BODY]
            scores = scores_full[_WB_BODY]
            rule_res = self._rule.check(kpts, scores)
            if rule_res.triggered and rule_res.behavior not in result.alerts:
                result.alerts.append(rule_res.behavior)
            result.persons.append(
                PersonResult(
                    track_id=-1,
                    rule_result=rule_res,
                    keypoints=kpts,
                    kp_scores=scores,
                    box_xyxy=np.asarray(box, dtype=np.float32),
                )
            )

        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def draw(self, image_bgr: np.ndarray, result: FrameResult) -> np.ndarray:
        conf_th = 0.3
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        body_style = VizStyle(kpt_visibility_threshold=conf_th,
                              skeleton_color_rgb=(255, 200, 0))

        xyxys: list[list[float]] = []
        labels: list[str] = []
        for person in result.persons:
            rgb = annotate_keypoints(rgb, person.keypoints, skeleton_edges=_SKELETON_17,
                                     confidence=person.kp_scores, style=body_style,
                                     color=sv.Color(r=0, g=0, b=255))
            if person.rule_result.triggered and person.kp_scores[0] > conf_th:
                head = person.keypoints[0]
                x, y = float(head[0]), float(head[1])
                yy = max(y - 15, 10)
                xyxys.append([x, yy, x + 1, yy + 1])
                labels.append(person.rule_result.behavior)

        if labels:
            dets = sv.Detections(
                xyxy=np.asarray(xyxys, dtype=np.float32),
                class_id=np.zeros(len(labels), dtype=int),
            )
            lbl_ann = sv.LabelAnnotator(
                color=sv.Color(r=220, g=0, b=0),
                text_scale=0.45, text_padding=2,
                text_position=sv.Position.TOP_LEFT,
            )
            rgb = lbl_ann.annotate(scene=rgb, detections=dets, labels=labels)

        if result.alerts:
            alert_text = "  |  ".join(f"ALERT: {a}" for a in result.alerts)
            rgb = classification_banner(
                rgb, alert_text,
                style=VizStyle(banner_height=24, banner_text_scale=0.5),
                position="overlay_top",
                bg_color_rgb=(30, 30, 30),
                text_color_rgb=(255, 50, 0),
            )

        rgb = classification_banner(
            rgb, f"{result.latency_ms:.1f}ms",
            style=VizStyle(banner_height=18, banner_text_scale=0.4),
            position="bottom",
            bg_color_rgb=(20, 20, 20),
            text_color_rgb=(160, 160, 160),
        )
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _serialize(result: FrameResult) -> dict:
    persons = []
    for p in result.persons:
        rr = p.rule_result
        persons.append({
            "track_id": p.track_id,
            "behavior": rr.behavior,
            "triggered": rr.triggered,
            "confidence": rr.confidence,
            "debug": rr.debug_info,
        })
    return {"alerts": result.alerts, "persons": persons,
            "latency_ms": round(result.latency_ms, 2)}


def _smoke_test() -> None:
    feat = REPO / "features" / "safety-poketenashi_no_handrail"
    config_path = feat / "configs" / "10_inference.yaml"
    samples_dir = feat / "samples"
    eval_dir = feat / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    predictor = NoHandrailPredictor(config_path)

    image_paths = sorted(p for p in samples_dir.glob("*.jpg"))
    if not image_paths:
        print(f"[smoke] no .jpg samples in {samples_dir} — copy a stair clip frame in to test")
        return

    results: list[dict] = []
    print(f"\n{'Image':<30} {'Alerts':<30} {'ms':>6}")
    print("-" * 70)
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  skip {img_path.name}: unreadable")
            continue
        result = predictor.process_frame(img)
        entry = {"image": img_path.name, **_serialize(result)}
        results.append(entry)
        alert_str = ", ".join(result.alerts) if result.alerts else "none"
        print(f"  {img_path.name:<28} {alert_str:<30} {result.latency_ms:>6.1f}")

        annotated = predictor.draw(img, result)
        cv2.imwrite(str(eval_dir / f"smoke_{img_path.name}"), annotated)

    report = eval_dir / "predictor_smoke_test.json"
    report.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {len(results)} results -> {report}")


def _video(video_path: Path) -> None:
    feat = REPO / "features" / "safety-poketenashi_no_handrail"
    config_path = feat / "configs" / "10_inference.yaml"
    eval_dir = feat / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    predictor = NoHandrailPredictor(config_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[video] cannot open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_mp4 = eval_dir / f"smoke_{video_path.stem}.mp4"
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))

    timeline: list[dict] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = predictor.process_frame(frame)
        timeline.append({"frame": frame_idx, "t_sec": frame_idx / fps,
                         **_serialize(result)})
        annotated = predictor.draw(frame, result)
        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()
    out_json = eval_dir / f"smoke_{video_path.stem}.json"
    out_json.write_text(json.dumps(timeline, indent=2))
    print(f"\n{frame_idx} frames -> {out_mp4}\nTimeline -> {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="no_handrail predictor")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run on samples/*.jpg, write eval/smoke_*.jpg")
    parser.add_argument("--video", type=Path, default=None,
                        help="Run on a video file, write eval/smoke_<basename>.mp4")
    args = parser.parse_args()
    if args.smoke_test:
        _smoke_test()
    elif args.video is not None:
        _video(args.video)
    else:
        parser.print_help()

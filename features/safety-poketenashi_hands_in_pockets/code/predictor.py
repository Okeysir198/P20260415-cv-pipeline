"""Hands-in-pockets predictor (single-rule orchestrator).

V1 -- pretrained-only. Wires:
  person detector (YOLO11n, lazy) -> DWPose ONNX -> COCO-17 keypoints
  -> HandsInPocketsDetector -> per-frame triggered flag.

Public API:
    HandsInPocketsPredictor(config_path).process_frame(image_bgr)
        -> {"alerts": [...], "persons": [...], "latency_ms": float}

CLI:
    uv run features/safety-poketenashi_hands_in_pockets/code/predictor.py --smoke-test
    uv run features/safety-poketenashi_hands_in_pockets/code/predictor.py --video <path>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
_CODE_DIR = Path(__file__).parent

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(_CODE_DIR))

from utils.config import load_config  # noqa: E402

from _base import RuleResult  # noqa: E402
from hands_in_pockets_detector import HandsInPocketsDetector  # noqa: E402

# COCO body slice when reading the first 17 of 133 wholebody keypoints.
_WB_BODY = slice(0, 17)


# ---------------------------------------------------------------------------
# Person detector (lazy YOLO11n; whole-frame fallback)
# ---------------------------------------------------------------------------

class _PersonDetector:
    """Lazy YOLO11n person detector. Falls back to a whole-frame box.

    Adapted from ``safety-point_and_call/code/pose_backend.py::_PersonDetector``
    -- duplicated per project rule: ``code/`` may not import from another
    feature's ``code/``.
    """

    _DEFAULT_PT = REPO / "pretrained" / "access-zone_intrusion" / "yolo11n.pt"

    def __init__(self, weights_path: Path | None = None, conf: float = 0.35) -> None:
        self._weights = Path(weights_path) if weights_path else self._DEFAULT_PT
        self._conf = float(conf)
        self._model: Any = None
        self._tried_load = False

    def _ensure_loaded(self) -> None:
        if self._tried_load:
            return
        self._tried_load = True
        if not self._weights.exists():
            self._model = None
            return
        try:
            from ultralytics import YOLO

            self._model = YOLO(str(self._weights))
        except Exception:
            self._model = None

    def __call__(self, image_bgr: np.ndarray) -> list[np.ndarray]:
        self._ensure_loaded()
        h, w = image_bgr.shape[:2]
        whole = [np.array([0.0, 0.0, float(w), float(h)], dtype=np.float32)]
        if self._model is None:
            return whole
        try:
            det = self._model.predict(
                image_bgr, classes=[0], conf=self._conf, verbose=False
            )[0]
        except Exception:
            return whole
        if det.boxes is None or len(det.boxes) == 0:
            return whole
        return [b.astype(np.float32) for b in det.boxes.xyxy.cpu().numpy()]


# ---------------------------------------------------------------------------
# DWPose ONNX adapter (RTMPose SimCC head)
# ---------------------------------------------------------------------------

class _DWPoseAdapter:
    """DWPose top-down ONNX adapter.

    Adapted from ``safety-point_and_call/code/pose_backend.py::_DWPoseAdapter``
    -- duplicated per project rule.
    """

    INPUT_HW = (384, 288)  # (H, W)

    def __init__(self, onnx_path: Path, person_detector: _PersonDetector) -> None:
        import onnxruntime as ort

        try:
            self._sess = ort.InferenceSession(
                str(onnx_path), providers=["CUDAExecutionProvider"]
            )
        except Exception:
            self._sess = ort.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
            )
        self._in_name = self._sess.get_inputs()[0].name
        self._person_detector = person_detector

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
        src = np.array(
            [[cx, cy], [cx + bw / 2, cy], [cx, cy + bh / 2]], dtype=np.float32
        )
        dst = np.array(
            [[ow / 2, oh / 2], [ow, oh / 2], [ow / 2, oh]], dtype=np.float32
        )
        return cv2.getAffineTransform(src, dst)

    def _infer_one(
        self, img_bgr: np.ndarray, box_xyxy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        oh, ow = self.INPUT_HW
        M = self._affine(box_xyxy)
        crop = cv2.warpAffine(img_bgr, M, (ow, oh), flags=cv2.INTER_LINEAR)
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        x = (crop.astype(np.float32) - mean) / std
        x = x.transpose(2, 0, 1)[None]
        simcc_x, simcc_y = self._sess.run(None, {self._in_name: x})
        sx = simcc_x[0].argmax(axis=-1).astype(np.float32) / 2.0
        sy = simcc_y[0].argmax(axis=-1).astype(np.float32) / 2.0
        scores = np.minimum(simcc_x[0].max(axis=-1), simcc_y[0].max(axis=-1))
        Minv = cv2.invertAffineTransform(M)
        ones = np.ones((sx.shape[0], 1), dtype=np.float32)
        pts_in = np.concatenate([sx[:, None], sy[:, None], ones], axis=1)
        pts_orig = pts_in @ Minv.T
        return pts_orig.astype(np.float32), scores.astype(np.float32)

    def __call__(
        self, image_bgr: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        out: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for box in self._person_detector(image_bgr):
            kpts, scores = self._infer_one(image_bgr, box)
            out.append((kpts[_WB_BODY], scores[_WB_BODY], box.astype(np.float32)))
        return out


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PersonBehavior:
    keypoints: np.ndarray  # (17, 2)
    kp_scores: np.ndarray  # (17,)
    box_xyxy: np.ndarray   # (4,)
    result: RuleResult


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

def _resolve_path(p: str | Path, base_dir: Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


class HandsInPocketsPredictor:
    """Single-rule pose predictor for the hands-in-pockets behavior."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.cfg = load_config(self.config_path)
        cfg_dir = self.config_path.parent

        pose_cfg = self.cfg.get("pose", {}) or {}
        if pose_cfg.get("backend") not in (None, "dwpose_onnx"):
            raise ValueError(
                f"Only 'dwpose_onnx' backend is supported in v1; "
                f"got {pose_cfg.get('backend')!r}"
            )
        weights = pose_cfg.get("weights")
        if not weights:
            raise ValueError("pose.weights (path to DWPose ONNX) is required")
        onnx_path = _resolve_path(weights, cfg_dir)

        det_weights_field = pose_cfg.get("person_detector")
        det_weights_path = (
            _resolve_path(det_weights_field, cfg_dir) if det_weights_field else None
        )
        det_conf = float(pose_cfg.get("person_conf", 0.35))
        person_detector = _PersonDetector(weights_path=det_weights_path, conf=det_conf)
        self._pose = _DWPoseAdapter(onnx_path=onnx_path, person_detector=person_detector)

        rule_cfg = (self.cfg.get("pose_rules", {}) or {}).get("hands_in_pockets", {}) or {}
        self._rule = HandsInPocketsDetector(
            wrist_below_hip_ratio=float(rule_cfg.get("wrist_below_hip_ratio", 0.05)),
            wrist_inside_torso_margin=float(rule_cfg.get("wrist_inside_torso_margin", 0.08)),
        )

    # ------------------------------------------------------------------

    def process_frame(self, image_bgr: np.ndarray) -> dict:
        """Return {alerts, persons, latency_ms} for a single frame."""
        t0 = time.perf_counter()
        pose_out = self._pose(image_bgr)

        persons: list[PersonBehavior] = []
        alerts: list[str] = []
        for kpts, scores, box in pose_out:
            res = self._rule.check(kpts, scores)
            persons.append(
                PersonBehavior(keypoints=kpts, kp_scores=scores, box_xyxy=box, result=res)
            )
            if res.triggered and self._rule.behavior not in alerts:
                alerts.append(self._rule.behavior)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {"alerts": alerts, "persons": persons, "latency_ms": latency_ms}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run_smoke_test(predictor: HandsInPocketsPredictor, out_dir: Path) -> dict:
    samples = (predictor.cfg.get("samples", {}) or {}).get("images", []) or []
    cfg_dir = predictor.config_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for rel in samples:
        img_path = _resolve_path(rel, cfg_dir)
        if not img_path.exists():
            results.append({"image": str(img_path), "error": "missing"})
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            results.append({"image": str(img_path), "error": "unreadable"})
            continue
        out = predictor.process_frame(img)
        results.append(
            {
                "image": str(img_path),
                "alerts": out["alerts"],
                "n_persons": len(out["persons"]),
                "latency_ms": round(out["latency_ms"], 2),
                "any_triggered": any(p.result.triggered for p in out["persons"]),
            }
        )

    summary = {
        "n_samples": len(samples),
        "n_triggered": sum(1 for r in results if r.get("any_triggered")),
        "results": results,
    }
    (out_dir / "smoke_test.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def _run_video(predictor: HandsInPocketsPredictor, video_path: Path, out_dir: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    timeline: list[dict] = []
    frame_idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out = predictor.process_frame(frame)
        timeline.append(
            {
                "frame": frame_idx,
                "t_sec": round(frame_idx / fps, 3),
                "alerts": out["alerts"],
                "latency_ms": round(out["latency_ms"], 2),
            }
        )
        frame_idx += 1
    cap.release()

    summary = {
        "video": str(video_path),
        "n_frames": frame_idx,
        "n_triggered_frames": sum(1 for f in timeline if f["alerts"]),
        "timeline": timeline,
    }
    out_path = out_dir / f"video_{video_path.stem}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_path}  ({summary['n_triggered_frames']}/{frame_idx} triggered)")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="path to 10_inference.yaml")
    parser.add_argument("--smoke-test", action="store_true", help="run on configured sample images")
    parser.add_argument("--video", type=Path, default=None, help="path to a video file")
    args = parser.parse_args()

    cfg_path = args.config or (
        Path(__file__).resolve().parents[1] / "configs" / "10_inference.yaml"
    )
    predictor = HandsInPocketsPredictor(cfg_path)
    out_dir = Path(__file__).resolve().parents[1] / "eval"

    if args.smoke_test:
        _run_smoke_test(predictor, out_dir)
    if args.video:
        _run_video(predictor, args.video, out_dir)
    if not (args.smoke_test or args.video):
        parser.error("specify --smoke-test or --video <path>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Thin orchestrator for safety-poketenashi_stair_diagonal.

Pipeline per frame:
  YOLO11n person detector → top-down box(es) → DWPose-L ONNX (384x288)
  → COCO-17 keypoints → per-track StairSafetyDetector → triggered flag.

State: StairSafetyDetector buffers hip positions across `min_frames`. With more
than one person in frame ByteTrack must drive `track_id`s — each id keeps its
own detector instance (mapping below). v1 single-track mode keys on track_id=0.

Run:
    uv run features/safety-poketenashi_stair_diagonal/code/predictor.py --smoke-test
    uv run features/safety-poketenashi_stair_diagonal/code/predictor.py --video clip.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
_CODE_DIR = Path(__file__).parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(_CODE_DIR))

from _base import RuleResult  # noqa: E402
from stair_safety_detector import StairSafetyDetector  # noqa: E402

from utils.config import load_config  # noqa: E402

_PRETRAIN = REPO / "pretrained" / "safety-poketenashi"
_DWPOSE_ONNX = _PRETRAIN / "dw-ll_ucoco_384.onnx"
_PERSON_DETECTOR = REPO / "pretrained" / "access-zone_intrusion" / "yolo11n.pt"

_WB_BODY = slice(0, 17)  # DWPose returns 133 wholebody kps; first 17 are COCO-17 body.


# ---------------------------------------------------------------------------
# DWPose ONNX wrapper (RTMPose SimCC head)
# ---------------------------------------------------------------------------

class _DWPose:
    INPUT_HW = (384, 288)  # H, W

    def __init__(self, onnx_path: Path) -> None:
        import onnxruntime as ort

        # GPU-only per project policy; ORT auto-falls back inside CUDA EP setup
        # if libs are missing — but we don't add CPU EP.
        self._sess = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider"])
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

        # SimCC argmax decode + affine un-warp (numpy keeps the predictor light).
        sx = simcc_x[0].argmax(axis=-1).astype(np.float32) / 2.0
        sy = simcc_y[0].argmax(axis=-1).astype(np.float32) / 2.0
        scores = np.minimum(simcc_x[0].max(axis=-1), simcc_y[0].max(axis=-1)).astype(np.float32)

        ones = np.ones_like(sx)
        pts_in = np.stack([sx, sy, ones], axis=1)  # (K, 3)
        pts_orig = (pts_in @ Minv.T).astype(np.float32)
        return pts_orig, scores


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TrackResult:
    track_id: int
    behavior: RuleResult
    keypoints: np.ndarray  # (17, 2)
    kp_scores: np.ndarray  # (17,)
    box_xyxy: np.ndarray  # (4,)


@dataclass
class FrameResult:
    triggered_track_ids: list[int]
    tracks: list[TrackResult]
    latency_ms: float


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class StairDiagonalPredictor:
    """DWPose ONNX + per-track StairSafetyDetector instances."""

    def __init__(self, config_path: str | Path) -> None:
        self._cfg = load_config(config_path)
        self._pose_model = self._load_pose_model()
        self._person_detector = self._load_person_detector()

        rule_cfg = self._cfg.get("pose_rules", {}).get("stair_diagonal", {})
        self._max_angle_deg = float(rule_cfg.get("max_diagonal_angle_deg", 20.0))
        self._min_frames = int(rule_cfg.get("min_frames", 5))

        # Per-track stateful detectors. v1 single-track flow uses track_id=0.
        self._detectors: dict[int, StairSafetyDetector] = {}

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_pose_model(self) -> _DWPose | None:
        if not _DWPOSE_ONNX.exists():
            print(f"[stair_diagonal] DWPose ONNX missing: {_DWPOSE_ONNX} — pose disabled")
            return None
        try:
            model = _DWPose(_DWPOSE_ONNX)
            print(f"[stair_diagonal] DWPose ONNX loaded: {_DWPOSE_ONNX.name}")
            return model
        except Exception as exc:
            print(f"[stair_diagonal] DWPose load failed ({exc}) — pose disabled")
            return None

    def _load_person_detector(self) -> Any | None:
        if not _PERSON_DETECTOR.exists():
            print(f"[stair_diagonal] person detector missing: {_PERSON_DETECTOR}"
                  " — whole-frame mode")
            return None
        try:
            from ultralytics import YOLO

            return YOLO(str(_PERSON_DETECTOR))
        except Exception as exc:
            print(f"[stair_diagonal] person detector load failed ({exc}) — whole-frame mode")
            return None

    # ------------------------------------------------------------------
    # Per-track detector access
    # ------------------------------------------------------------------

    def _get_detector(self, track_id: int) -> StairSafetyDetector:
        det = self._detectors.get(track_id)
        if det is None:
            det = StairSafetyDetector(
                max_diagonal_angle_deg=self._max_angle_deg,
                min_frames=self._min_frames,
            )
            self._detectors[track_id] = det
        return det

    def reset(self, track_id: int | None = None) -> None:
        """Clear buffered hip positions. ``None`` resets every track."""
        if track_id is None:
            for det in self._detectors.values():
                det.reset()
            self._detectors.clear()
        elif track_id in self._detectors:
            self._detectors[track_id].reset()
            self._detectors.pop(track_id, None)

    # ------------------------------------------------------------------
    # Person detection
    # ------------------------------------------------------------------

    def _detect_persons(self, image_bgr: np.ndarray) -> list[np.ndarray]:
        if self._person_detector is None:
            h, w = image_bgr.shape[:2]
            return [np.array([0, 0, w, h], dtype=np.float32)]

        det = self._person_detector.predict(image_bgr, classes=[0], conf=0.35, verbose=False)[0]
        if det.boxes is None or len(det.boxes) == 0:
            h, w = image_bgr.shape[:2]
            return [np.array([0, 0, w, h], dtype=np.float32)]
        return [b for b in det.boxes.xyxy.cpu().numpy()]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_frame(
        self,
        image_bgr: np.ndarray,
        track_ids: list[int] | None = None,
    ) -> FrameResult:
        """Run pose + rule on a frame.

        ``track_ids``: optional one-per-detected-person id list. When omitted,
        v1 single-track mode keys every detection on track_id=0 — this is only
        safe with a single person in frame.
        """
        t0 = time.perf_counter()
        tracks: list[TrackResult] = []
        triggered_ids: list[int] = []

        if self._pose_model is None:
            latency_ms = (time.perf_counter() - t0) * 1000
            return FrameResult([], [], latency_ms)

        boxes = self._detect_persons(image_bgr)
        if track_ids is None:
            track_ids = [0] * len(boxes)
        if len(track_ids) != len(boxes):
            raise ValueError(f"track_ids length {len(track_ids)} != boxes {len(boxes)}")

        for box, tid in zip(boxes, track_ids, strict=True):
            kpts, scores = self._pose_model(image_bgr, box)
            kpts17 = kpts[_WB_BODY]
            sc17 = scores[_WB_BODY]
            detector = self._get_detector(tid)
            result = detector.check(kpts17, sc17)
            tracks.append(TrackResult(tid, result, kpts17, sc17, box))
            if result.triggered:
                triggered_ids.append(tid)

        latency_ms = (time.perf_counter() - t0) * 1000
        return FrameResult(triggered_ids, tracks, latency_ms)


# ---------------------------------------------------------------------------
# CLIs
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    """Synthetic-keypoint smoke: build a stationary then diagonal trajectory.

    Bypasses pose model + person detector — exercises the per-track buffer
    and rule firing only. Mirrors the pattern in tests/.
    """
    feature_dir = REPO / "features" / "safety-poketenashi_stair_diagonal"
    config_path = feature_dir / "configs" / "10_inference.yaml"
    eval_dir = feature_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_path)
    rule_cfg = cfg.get("pose_rules", {}).get("stair_diagonal", {})
    max_angle = float(rule_cfg.get("max_diagonal_angle_deg", 20.0))
    min_frames = int(rule_cfg.get("min_frames", 5))

    detector = StairSafetyDetector(max_diagonal_angle_deg=max_angle, min_frames=min_frames)
    scores = np.full(17, 0.9, dtype=np.float32)

    print(f"[smoke] config: max_angle={max_angle}°, min_frames={min_frames}")

    # Diagonal trajectory: hip moves 100px right + 100px down per step (45°).
    fired_at = -1
    for i in range(min_frames + 2):
        kpts = np.zeros((17, 2), dtype=np.float32)
        kpts[11] = [100 + i * 100, 200 + i * 100]  # left hip
        kpts[12] = [110 + i * 100, 200 + i * 100]  # right hip
        result = detector.check(kpts, scores)
        print(f"  frame {i}: triggered={result.triggered}, debug={result.debug_info}")
        if result.triggered and fired_at < 0:
            fired_at = i

    summary = {
        "config": {"max_diagonal_angle_deg": max_angle, "min_frames": min_frames},
        "first_triggered_frame": fired_at,
        "expected_first_trigger_frame": min_frames - 1,
        "result": "ok" if fired_at == min_frames - 1 else "unexpected",
    }
    out = eval_dir / "predictor_smoke_test.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary → {out}")


def _video(video_path: Path) -> None:
    feature_dir = REPO / "features" / "safety-poketenashi_stair_diagonal"
    config_path = feature_dir / "configs" / "10_inference.yaml"
    eval_dir = feature_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    predictor = StairDiagonalPredictor(config_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[stair_diagonal] cannot open video: {video_path}")
        return

    frame_log: list[dict] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = predictor.process_frame(frame)
        if result.tracks:
            t = result.tracks[0]
            frame_log.append({
                "frame": frame_idx,
                "track_id": t.track_id,
                "triggered": t.behavior.triggered,
                "latency_ms": round(result.latency_ms, 2),
                **t.behavior.debug_info,
            })
        frame_idx += 1
    cap.release()

    out = eval_dir / f"video_{video_path.stem}.json"
    out.write_text(json.dumps(frame_log, indent=2))
    n_trig = sum(1 for r in frame_log if r.get("triggered"))
    print(f"[stair_diagonal] {frame_idx} frames, {n_trig} triggered → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="safety-poketenashi_stair_diagonal predictor")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Synthetic-keypoint smoke (no media required)")
    parser.add_argument("--video", type=Path, default=None,
                        help="Run live inference on a video file (single-track mode)")
    args = parser.parse_args()

    if args.smoke_test:
        _smoke_test()
    elif args.video is not None:
        _video(args.video)
    else:
        parser.print_help()

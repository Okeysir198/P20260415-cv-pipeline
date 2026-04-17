#!/usr/bin/env python3
"""Benchmark pose estimation models for the poketenashi orchestrator.

Evaluates models from:
  pretrained/safety-poketenashi/   — DWPose ONNX + RTMPose-S/L wholebody .pth
  pretrained/safety-fall_pose_estimation/ — RTMPose-S body, RTMO-S/L, DWPose, YOLO-NAS, MediaPipe

For each model:
  1. Runs pose inference on all images in features/safety-poketenashi/samples/
  2. Measures latency and detection rate
  3. Applies the 4 pose rules (hands_in_pockets, stair_diagonal, no_handrail,
     no_pointing_calling) and accumulates behavior trigger counts

Outputs:
  features/safety-poketenashi/eval/benchmark_results.json
  features/safety-poketenashi/eval/benchmark_report.md

Usage:
    uv run features/safety-poketenashi/code/benchmark.py
"""

from __future__ import annotations

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

SAMPLES_DIR = REPO / "features" / "safety-poketenashi" / "samples"
EVAL_DIR = REPO / "features" / "safety-poketenashi" / "eval"
PRETRAIN_POKE = REPO / "pretrained" / "safety-poketenashi"
PRETRAIN_POSE = REPO / "pretrained" / "safety-fall_pose_estimation"

# COCO-17 body keypoint slice from wholebody 133-kp models
_WB_BODY = slice(0, 17)

# Behavior names for table formatting
_BEHAVIORS = ["hands_in_pockets", "stair_diagonal", "no_handrail", "no_pointing_calling"]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    model: str
    status: str  # "ok" | "skipped" | "error"
    error: str = ""
    latency_mean_ms: float | None = None
    latency_std_ms: float | None = None
    detection_rate: float | None = None
    behavior_triggers: dict[str, int] | None = None
    note: str = ""


# ---------------------------------------------------------------------------
# Pose rule helpers — re-use existing detectors from code/
# ---------------------------------------------------------------------------

def _build_rules() -> list:
    """Build all 4 pose rules with default config."""
    from hands_in_pockets_detector import HandsInPocketsDetector
    from stair_safety_detector import StairSafetyDetector
    from handrail_detector import HandrailDetector
    from pointing_calling_detector import PointingCallingDetector

    return [
        HandsInPocketsDetector(wrist_below_hip_ratio=0.05, wrist_inside_torso_margin=0.08),
        StairSafetyDetector(max_diagonal_angle_deg=20.0),
        HandrailDetector(handrail_zones=[]),  # no zones configured → never triggers
        PointingCallingDetector(elbow_wrist_angle_min_deg=150.0, pointing_duration_frames=20),
    ]


def _apply_rules(rules: list, kpts17: np.ndarray, scores17: np.ndarray) -> dict[str, bool]:
    """Run all pose rules on a single person's 17 keypoints. Returns {behavior: triggered}."""
    results = {}
    for rule in rules:
        r = rule.check(kpts17, scores17)
        results[r.behavior] = r.triggered
    return results


def _accumulate_triggers(
    behavior_triggers: dict[str, int],
    per_image_triggered: set[str],
) -> None:
    for b in per_image_triggered:
        behavior_triggers[b] = behavior_triggers.get(b, 0) + 1


# ---------------------------------------------------------------------------
# DWPose ONNX inference (reused from orchestrator.py)
# ---------------------------------------------------------------------------

class _DWPose:
    INPUT_HW = (384, 288)

    def __init__(self, onnx_path: Path) -> None:
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            self._sess = ort.InferenceSession(str(onnx_path), providers=providers)
        except Exception:
            self._sess = ort.InferenceSession(str(onnx_path),
                                              providers=["CPUExecutionProvider"])
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

    def infer(self, img_bgr: np.ndarray,
              box_xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def _detect_persons_yolo(img_bgr: np.ndarray) -> list[np.ndarray]:
    """Lightweight person detector — returns list of xyxy boxes."""
    try:
        from ultralytics import YOLO
        if not hasattr(_detect_persons_yolo, "_model"):
            _pt = Path(__file__).resolve().parents[3] / "pretrained" / "access-zone_intrusion" / "yolo11n.pt"
            _detect_persons_yolo._model = YOLO(str(_pt))
        det = _detect_persons_yolo._model.predict(img_bgr, classes=[0], conf=0.35,
                                                   verbose=False)[0]
        if det.boxes is not None and len(det.boxes) > 0:
            return [b for b in det.boxes.xyxy.cpu().numpy()]
    except Exception:
        pass
    # Fallback: whole frame as single box
    h, w = img_bgr.shape[:2]
    return [np.array([0, 0, w, h], dtype=np.float32)]


# ---------------------------------------------------------------------------
# Per-backend inference functions — return list of (kpts17, scores17) per image
# ---------------------------------------------------------------------------

def _infer_dwpose_onnx(
    model: _DWPose, img_bgr: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run DWPose on all detected persons in an image."""
    boxes = _detect_persons_yolo(img_bgr)
    persons = []
    for box in boxes:
        kpts, scores = model.infer(img_bgr, box)
        persons.append((kpts[_WB_BODY], scores[_WB_BODY]))
    return persons


def _infer_yolo_nas_onnx(
    sess: Any, img_bgr: np.ndarray, input_hw: tuple[int, int]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run YOLO-NAS-Pose ONNX on image — returns (kpts17, scores17) per person."""
    in_name = sess.get_inputs()[0].name
    oh, ow = input_hw
    resized = cv2.resize(img_bgr, (ow, oh))
    # YOLO-NAS-Pose ONNX: shape (1,3,H,W) uint8 BGR
    x = resized[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.uint8)  # NCHW uint8
    outputs = sess.run(None, {in_name: x})
    # YOLO-NAS-Pose outputs: [num_predictions, boxes(1,N,4), scores(1,N), joints(1,N,17,3)]
    persons = []
    if len(outputs) >= 4:
        joints = outputs[3]   # (1, N, 17, 3)
        scores_out = outputs[2]  # (1, N)
        h, w = img_bgr.shape[:2]
        n_det = joints.shape[1] if joints.ndim >= 2 else 0
        for i in range(n_det):
            det_score = float(scores_out[0, i]) if scores_out is not None else 1.0
            if det_score < 0.3:
                continue
            kp = joints[0, i]  # (17, 3)
            kpts = kp[:, :2].copy().astype(np.float32)
            # Scale from model-input to original image coords
            kpts[:, 0] *= w / ow
            kpts[:, 1] *= h / oh
            kp_scores = kp[:, 2].astype(np.float32)
            persons.append((kpts, kp_scores))
    return persons


def _infer_mediapipe_task(
    landmarker: Any, img_bgr: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run MediaPipe PoseLandmarker on image — converts to COCO-17 subset."""
    import mediapipe as mp
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    result = landmarker.detect(mp_image)
    if not result.pose_landmarks:
        return []
    h, w = img_bgr.shape[:2]
    # MediaPipe BlazePose has 33 landmarks; COCO-17 mapping (approximate):
    # MP indices → COCO: 0→0(nose), 2→1(l_eye), 5→2(r_eye), 7→3(l_ear), 8→4(r_ear)
    # 11→5(l_sh), 12→6(r_sh), 13→7(l_el), 14→8(r_el), 15→9(l_wr), 16→10(r_wr)
    # 23→11(l_hip), 24→12(r_hip), 25→13(l_kn), 26→14(r_kn), 27→15(l_ank), 28→16(r_ank)
    _MP_TO_COCO17 = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    persons = []
    for pose in result.pose_landmarks:
        kpts = np.zeros((17, 2), dtype=np.float32)
        kp_scores = np.zeros(17, dtype=np.float32)
        for coco_idx, mp_idx in enumerate(_MP_TO_COCO17):
            if mp_idx < len(pose):
                lm = pose[mp_idx]
                kpts[coco_idx] = [lm.x * w, lm.y * h]
                kp_scores[coco_idx] = lm.visibility
        persons.append((kpts, kp_scores))
    return persons


# ---------------------------------------------------------------------------
# Benchmark a single model
# ---------------------------------------------------------------------------

def _benchmark_model(
    label: str,
    infer_fn: Any,  # callable(img_bgr) -> list[(kpts17, scores17)]
    img_paths: list[Path],
    rules: list,
) -> ModelResult:
    """Run inference + rule application on all sample images."""
    latencies: list[float] = []
    detected_count = 0
    behavior_triggers: dict[str, int] = {b: 0 for b in _BEHAVIORS}

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        t0 = time.perf_counter()
        try:
            persons = infer_fn(img)
        except Exception as exc:
            return ModelResult(model=label, status="error", error=str(exc))
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        if persons:
            detected_count += 1

        # Per-image: collect which behaviors fired on at least one person
        img_triggers: set[str] = set()
        for kpts17, scores17 in persons:
            triggered_map = _apply_rules(rules, kpts17, scores17)
            for b, fired in triggered_map.items():
                if fired:
                    img_triggers.add(b)
        _accumulate_triggers(behavior_triggers, img_triggers)

    n = len(img_paths)
    return ModelResult(
        model=label,
        status="ok",
        latency_mean_ms=float(np.mean(latencies)) if latencies else None,
        latency_std_ms=float(np.std(latencies)) if latencies else None,
        detection_rate=detected_count / n if n > 0 else 0.0,
        behavior_triggers=behavior_triggers,
    )


# ---------------------------------------------------------------------------
# Model catalogue — build (label, loader) pairs
# ---------------------------------------------------------------------------

def _load_dwpose_onnx(onnx_path: Path) -> Any:
    return _DWPose(onnx_path)


def _collect_models() -> list[tuple[str, str, Path]]:
    """Return (label, backend, path) for all candidate models."""
    entries: list[tuple[str, str, Path]] = []

    # ── safety-poketenashi pretrained ──────────────────────────────────────
    onnx_poke = PRETRAIN_POKE / "dw-ll_ucoco_384.onnx"
    if onnx_poke.exists():
        entries.append(("dwpose_384_poke", "dwpose_onnx", onnx_poke))

    for pth_name in ["rtmpose-s_coco-wholebody.pth", "rtmw-l_cocktail14_384x288.pth"]:
        p = PRETRAIN_POKE / pth_name
        if p.exists():
            entries.append((p.stem, "mmpose_pth", p))

    # ── safety-fall_pose_estimation pretrained ─────────────────────────────
    onnx_pose = PRETRAIN_POSE / "dw-ll_ucoco_384.onnx"
    if onnx_pose.exists():
        entries.append(("dwpose_384_pose", "dwpose_onnx", onnx_pose))

    for pth_name in [
        "rtmpose-s_coco_256x192.pth",
        "rtmo-s_body7_640x640.pth",
        "rtmo-l_body7_640x640.pth",
    ]:
        p = PRETRAIN_POSE / pth_name
        if p.exists():
            entries.append((p.stem, "mmpose_pth", p))

    for onnx_name in ["yolo_nas_pose_s.onnx", "yolo_nas_pose_m.onnx", "yolo_nas_pose_l.onnx"]:
        p = PRETRAIN_POSE / onnx_name
        if p.exists():
            entries.append((p.stem, "yolo_nas_onnx", p))

    for task_name in [
        "pose_landmarker_lite.task",
        "pose_landmarker_full.task",
        "pose_landmarker_heavy.task",
    ]:
        p = PRETRAIN_POSE / task_name
        if p.exists():
            entries.append((p.stem, "mediapipe_task", p))

    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import importlib

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(
        p for p in SAMPLES_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not img_paths:
        print(f"No sample images found in {SAMPLES_DIR}")
        sys.exit(1)
    print(f"Found {len(img_paths)} sample images in {SAMPLES_DIR}")

    rules = _build_rules()
    model_entries = _collect_models()
    print(f"Found {len(model_entries)} candidate models\n")

    all_results: list[ModelResult] = []

    for label, backend, model_path in model_entries:
        print(f"--- {label} ({backend}) ---")

        if backend == "dwpose_onnx":
            try:
                import onnxruntime  # noqa: F401
            except ImportError:
                r = ModelResult(model=label, status="skipped",
                                error="onnxruntime not installed")
                print(f"  skipped: {r.error}")
                all_results.append(r)
                continue
            try:
                dwpose = _load_dwpose_onnx(model_path)
                infer_fn = lambda img, _m=dwpose: _infer_dwpose_onnx(_m, img)
                result = _benchmark_model(label, infer_fn, img_paths, rules)
            except Exception as exc:
                result = ModelResult(model=label, status="error", error=str(exc))

        elif backend == "mmpose_pth":
            try:
                import mmpose  # noqa: F401
                from mmpose.apis import init_model, inference_topdown
            except ImportError:
                r = ModelResult(
                    model=label, status="skipped",
                    error="mmpose not installed — install via: pip install mmpose",
                )
                print(f"  skipped: {r.error}")
                all_results.append(r)
                continue
            # MMPose requires a config file alongside the checkpoint.
            # Without it we cannot instantiate the model architecture.
            # The .pth files in pretrained/ are bare checkpoints without config.
            result = ModelResult(
                model=label,
                status="skipped",
                error=(
                    "MMPose .pth requires a matching model config file "
                    "(e.g. rtmpose-s_8xb256-420e_coco-256x192.py) to init_model(). "
                    "Download the full MMPose config repo and pass --cfg to use these."
                ),
            )

        elif backend == "yolo_nas_onnx":
            try:
                import onnxruntime as ort
            except ImportError:
                r = ModelResult(model=label, status="skipped",
                                error="onnxruntime not installed")
                print(f"  skipped: {r.error}")
                all_results.append(r)
                continue
            try:
                # Force CPU for YOLO-NAS to avoid GPU memory fragmentation after DWPose
                sess = ort.InferenceSession(str(model_path),
                                            providers=["CPUExecutionProvider"])
                # Determine input spatial size from model metadata
                inp = sess.get_inputs()[0]
                shape = inp.shape  # e.g. [1, 3, H, W]
                try:
                    ih, iw = int(shape[2]), int(shape[3])
                except (TypeError, IndexError):
                    ih, iw = 640, 640
                infer_fn = lambda img, _s=sess, _hw=(ih, iw): _infer_yolo_nas_onnx(_s, img, _hw)
                result = _benchmark_model(label, infer_fn, img_paths, rules)
            except Exception as exc:
                result = ModelResult(model=label, status="error", error=str(exc))

        elif backend == "mediapipe_task":
            try:
                import mediapipe as mp
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python import vision as mp_vision
            except ImportError:
                r = ModelResult(model=label, status="skipped",
                                error="mediapipe not installed")
                print(f"  skipped: {r.error}")
                all_results.append(r)
                continue
            try:
                base_opts = mp_python.BaseOptions(model_asset_path=str(model_path))
                options = mp_vision.PoseLandmarkerOptions(
                    base_options=base_opts,
                    num_poses=5,
                    min_pose_detection_confidence=0.3,
                    min_pose_presence_confidence=0.3,
                    min_tracking_confidence=0.3,
                )
                landmarker = mp_vision.PoseLandmarker.create_from_options(options)
                infer_fn = lambda img, _lm=landmarker: _infer_mediapipe_task(_lm, img)
                result = _benchmark_model(label, infer_fn, img_paths, rules)
                landmarker.close()
            except Exception as exc:
                result = ModelResult(model=label, status="error", error=str(exc))

        else:
            result = ModelResult(model=label, status="error",
                                 error=f"unknown backend: {backend}")

        if result.status == "ok":
            print(
                f"  detection_rate={result.detection_rate:.2f}  "
                f"latency={result.latency_mean_ms:.1f}±{result.latency_std_ms:.1f}ms  "
                f"triggers={result.behavior_triggers}"
            )
        else:
            print(f"  {result.status}: {result.error}")

        all_results.append(result)

    # ── Write JSON ─────────────────────────────────────────────────────────
    def _to_native(obj: Any) -> Any:
        if hasattr(obj, "item"):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_native(v) for v in obj]
        return obj

    serialisable = [
        _to_native({
            "model": r.model,
            "status": r.status,
            "error": r.error,
            "latency_mean_ms": r.latency_mean_ms,
            "latency_std_ms": r.latency_std_ms,
            "detection_rate": r.detection_rate,
            "behavior_triggers": r.behavior_triggers,
            "note": r.note,
        })
        for r in all_results
    ]

    json_path = EVAL_DIR / "benchmark_results.json"
    json_path.write_text(json.dumps(serialisable, indent=2))
    print(f"\nJSON results → {json_path}")

    _write_report(all_results, EVAL_DIR)


def _write_report(results: list[ModelResult], out_dir: Path) -> None:
    """Write markdown benchmark report."""
    ok = [r for r in results if r.status == "ok"]
    skipped = [r for r in results if r.status != "ok"]

    # Sort ok by detection_rate desc, then latency asc
    ok_sorted = sorted(ok, key=lambda r: (-(r.detection_rate or 0), r.latency_mean_ms or 9999))

    def _fmt(v: float | None, decimals: int = 2) -> str:
        return f"{v:.{decimals}f}" if v is not None else "—"

    lines = [
        "# Poketenashi Pose Estimation — Model Benchmark",
        "",
        f"**Sample images:** `features/safety-poketenashi/samples/` ({len(list(SAMPLES_DIR.glob('*.jpg')))} images)  ",
        "**Metrics:** detection rate (≥1 person detected), mean latency, behavior trigger counts  ",
        "**Pose rules applied:** hands_in_pockets, stair_diagonal, no_handrail (no zones → 0), "
        "no_pointing_calling",
        "",
        "## Results",
        "",
        "| Rank | Model | Det Rate | Latency ms | hands_in_pockets | stair_diagonal"
        " | no_pointing_calling | Status |",
        "|------|-------|----------|------------|------------------|----------------|"
        "---------------------|--------|",
    ]

    for rank, r in enumerate(ok_sorted, 1):
        bt = r.behavior_triggers or {}
        lines.append(
            f"| {rank} | `{r.model}` "
            f"| {_fmt(r.detection_rate)} "
            f"| {_fmt(r.latency_mean_ms, 1)}±{_fmt(r.latency_std_ms, 1)} "
            f"| {bt.get('hands_in_pockets', 0)} "
            f"| {bt.get('stair_diagonal', 0)} "
            f"| {bt.get('no_pointing_calling', 0)} "
            f"| ok |"
        )

    if skipped:
        lines += ["", "## Skipped / Errors", ""]
        for r in skipped:
            lines.append(f"- `{r.model}` ({r.status}): {r.error}")

    # Recommendation
    lines += ["", "## Recommendation", ""]
    if ok_sorted:
        best = ok_sorted[0]
        lines.append(
            f"**Best model for poketenashi:** `{best.model}`  \n"
            f"Detection rate: {_fmt(best.detection_rate)} | "
            f"Latency: {_fmt(best.latency_mean_ms, 1)} ms  \n"
            "Criteria: highest detection rate, then lowest latency. "
            "DWPose ONNX is preferred when available — no mmpose dependency, "
            "runs directly via onnxruntime, and the orchestrator already has a "
            "built-in DWPose loader."
        )
    else:
        lines.append(
            "No models evaluated successfully. "
            "Ensure onnxruntime is installed and `dw-ll_ucoco_384.onnx` is present in "
            "`pretrained/safety-poketenashi/`."
        )

    lines += ["", f"*Generated {time.strftime('%Y-%m-%d %H:%M')}*", ""]

    report_path = out_dir / "benchmark_report.md"
    report_path.write_text("\n".join(lines))
    print(f"Report → {report_path}")


if __name__ == "__main__":
    main()

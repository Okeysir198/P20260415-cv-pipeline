"""Pretrained model benchmark for safety-fall_pose_estimation.

No labeled val/test dataset exists yet for this feature.
Evaluation is latency-focused with qualitative metrics on sample images.

Metrics collected per model:
  - latency_mean_ms / latency_std_ms  (inference time per image)
  - detection_rate                    (fraction of images with >= 1 person detected)
  - fall_trigger_rate                 (fraction triggering simple geometric fall rule)

Fall rule: torso angle from vertical > 45° OR any wrist keypoint below hip keypoint.

Framework routing:
  .pth  → MMPose (skipped with note if not installed)
  .onnx → ONNX Runtime
  .task → MediaPipe PoseLandmarker

Output:
    features/safety-fall_pose_estimation/eval/benchmark_results.json
    features/safety-fall_pose_estimation/eval/benchmark_report.md
"""

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRETRAINED_DIR = REPO / "pretrained" / "safety-fall_pose_estimation"
SAMPLES_DIR = REPO / "features" / "safety-fall_pose_estimation" / "samples"
EVAL_DIR = REPO / "features" / "safety-fall_pose_estimation" / "eval"

# COCO 17-keypoint indices
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10

FALL_ANGLE_THRESHOLD_DEG = 45.0
INPUT_SIZE_HW = (640, 640)  # default for RTMO/YOLO-NAS
RTMPOSE_INPUT_HW = (192, 256)  # RTMPose: (W, H) → stored as (H=256, W=192)

# Models to probe (existence checked at runtime)
ONNX_MODELS = [
    {"name": "dwpose", "path": PRETRAINED_DIR / "dw-ll_ucoco_384.onnx",
     "input_hw": (384, 288), "num_keypoints": 133},
    {"name": "yolo_nas_pose_s", "path": PRETRAINED_DIR / "yolo_nas_pose_s.onnx",
     "input_hw": (640, 640), "num_keypoints": 17},
    {"name": "yolo_nas_pose_m", "path": PRETRAINED_DIR / "yolo_nas_pose_m.onnx",
     "input_hw": (640, 640), "num_keypoints": 17},
    {"name": "yolo_nas_pose_l", "path": PRETRAINED_DIR / "yolo_nas_pose_l.onnx",
     "input_hw": (640, 640), "num_keypoints": 17},
]

MEDIAPIPE_MODELS = [
    {"name": "mediapipe_lite", "path": PRETRAINED_DIR / "pose_landmarker_lite.task"},
    {"name": "mediapipe_full", "path": PRETRAINED_DIR / "pose_landmarker_full.task"},
    {"name": "mediapipe_heavy", "path": PRETRAINED_DIR / "pose_landmarker_heavy.task"},
]

PTH_MODELS = [
    {"name": "rtmpose-s", "path": PRETRAINED_DIR / "rtmpose-s_coco_256x192.pth",
     "arch": "rtmpose-s", "input_hw": (256, 192)},
    {"name": "rtmo-s", "path": PRETRAINED_DIR / "rtmo-s_body7_640x640.pth",
     "arch": "rtmo-s", "input_hw": (640, 640)},
    {"name": "rtmo-l", "path": PRETRAINED_DIR / "rtmo-l_body7_640x640.pth",
     "arch": "rtmo-l", "input_hw": (640, 640)},
    {"name": "hrnet_w48", "path": PRETRAINED_DIR / "hrnet_w48_coco_256x192.pth",
     "arch": "hrnet-w48", "input_hw": (256, 192)},
    {"name": "vitpose_small", "path": PRETRAINED_DIR / "vitpose-plus-small.pth",
     "arch": "vitpose-plus-small", "input_hw": (256, 192)},
    {"name": "vitpose_base", "path": PRETRAINED_DIR / "vitpose-plus-base.pth",
     "arch": "vitpose-plus-base", "input_hw": (256, 192)},
]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def torso_angle_deg(keypoints: np.ndarray) -> float | None:
    """Compute torso vertical angle in degrees from 17-kp array (N, 3).

    Returns None if required keypoints are invisible.
    Keypoints: x, y, visibility (or confidence).
    """
    if keypoints.shape[0] < 13:
        return None

    def midpoint(a_idx: int, b_idx: int) -> np.ndarray | None:
        a, b = keypoints[a_idx], keypoints[b_idx]
        if a[2] < 0.3 and b[2] < 0.3:
            return None
        return (a[:2] + b[:2]) / 2.0

    shoulder_mid = midpoint(KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER)
    hip_mid = midpoint(KP_LEFT_HIP, KP_RIGHT_HIP)
    if shoulder_mid is None or hip_mid is None:
        return None

    delta = shoulder_mid - hip_mid  # (dx, dy) — y increases downward
    # Angle from vertical (straight up = 0°)
    angle = abs(np.degrees(np.arctan2(abs(delta[0]), abs(delta[1]) + 1e-6)))
    return float(angle)


def wrist_below_hip(keypoints: np.ndarray) -> bool:
    """Return True if any visible wrist is below any visible hip (y-axis)."""
    if keypoints.shape[0] < 13:
        return False

    hip_ys = []
    for idx in [KP_LEFT_HIP, KP_RIGHT_HIP]:
        if keypoints[idx, 2] >= 0.3:
            hip_ys.append(keypoints[idx, 1])

    wrist_ys = []
    for idx in [KP_LEFT_WRIST, KP_RIGHT_WRIST]:
        if keypoints[idx, 2] >= 0.3:
            wrist_ys.append(keypoints[idx, 1])

    if not hip_ys or not wrist_ys:
        return False

    return any(w > h for w in wrist_ys for h in hip_ys)


def is_fall(keypoints: np.ndarray) -> bool:
    """Apply geometric fall rule to a single person's keypoints."""
    angle = torso_angle_deg(keypoints)
    if angle is not None and angle > FALL_ANGLE_THRESHOLD_DEG:
        return True
    return wrist_below_hip(keypoints)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def load_sample_images() -> list[np.ndarray]:
    """Load all images from SAMPLES_DIR. Returns list of BGR arrays."""
    images = []
    if not SAMPLES_DIR.exists():
        print(f"[warn] samples dir not found: {SAMPLES_DIR}")
        return images
    for p in sorted(SAMPLES_DIR.iterdir()):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            img = cv2.imread(str(p))
            if img is not None:
                images.append(img)
    print(f"Loaded {len(images)} sample images from {SAMPLES_DIR}")
    return images


def preprocess_image(
    img: np.ndarray, target_hw: tuple[int, int]
) -> np.ndarray:
    """Resize + normalize to float32 CHW for ONNX/PyTorch inference."""
    h, w = target_hw
    resized = cv2.resize(img, (w, h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    return rgb.transpose(2, 0, 1)  # CHW


# ---------------------------------------------------------------------------
# ONNX inference
# ---------------------------------------------------------------------------


def _ort_session(onnx_path: Path) -> Any:
    import onnxruntime as ort  # type: ignore[import]

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(str(onnx_path), providers=providers)
    except Exception:
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return sess


def _extract_keypoints_from_onnx_output(
    outputs: list[np.ndarray], num_keypoints: int
) -> list[np.ndarray]:
    """Best-effort extraction of keypoints from arbitrary ONNX output shapes.

    Returns list of (num_keypoints, 3) arrays (x, y, conf), one per detected person.
    Returns empty list on failure.
    """
    for out in outputs:
        out = out.squeeze()  # remove batch dim
        # Shape (N, K, 3) — N persons, K keypoints
        if out.ndim == 3 and out.shape[1] == num_keypoints and out.shape[2] == 3:
            return [out[i] for i in range(out.shape[0])]
        # Shape (K, 3) — single person
        if out.ndim == 2 and out.shape[0] == num_keypoints and out.shape[1] == 3:
            return [out]
        # Shape (K*3,) — flat single person
        if out.ndim == 1 and out.shape[0] == num_keypoints * 3:
            return [out.reshape(num_keypoints, 3)]
    return []


def benchmark_onnx_model(spec: dict, images: list[np.ndarray]) -> dict[str, Any]:
    name = spec["name"]
    onnx_path = spec["path"]
    input_hw = spec["input_hw"]
    num_kp = spec["num_keypoints"]

    result: dict[str, Any] = {"name": name, "framework": "onnxruntime", "checkpoint": str(onnx_path)}

    if not onnx_path.exists():
        result["status"] = "error"
        result["error"] = f"file not found: {onnx_path}"
        return result

    if not images:
        result["status"] = "error"
        result["error"] = "no sample images"
        return result

    try:
        import onnxruntime  # noqa: F401  -- check availability
    except ImportError:
        result["status"] = "skipped"
        result["error"] = "onnxruntime not installed"
        return result

    try:
        print(f"  [{name}] Loading ONNX session...")
        sess = _ort_session(onnx_path)
        input_name = sess.get_inputs()[0].name

        latencies: list[float] = []
        detected_count = 0
        fall_count = 0

        for img in images:
            blob = preprocess_image(img, input_hw)
            blob = blob[np.newaxis]  # (1, C, H, W)

            t0 = time.perf_counter()
            try:
                outputs = sess.run(None, {input_name: blob})
            except Exception:
                # Some models expect uint8 or different normalization; try raw resize
                raw = cv2.resize(img, (input_hw[1], input_hw[0]))
                raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                blob_u8 = raw.transpose(2, 0, 1)[np.newaxis].astype(np.uint8)
                outputs = sess.run(None, {input_name: blob_u8})
            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)

            persons = _extract_keypoints_from_onnx_output(outputs, num_kp)
            if persons:
                detected_count += 1
                # Use first 17 keypoints for fall rule (DWPose has 133)
                person_kp = persons[0]
                kp = person_kp[:17] if person_kp.shape[0] >= 17 else person_kp
                if is_fall(kp):
                    fall_count += 1

        n = len(images)
        result.update({
            "status": "ok",
            "num_images": n,
            "latency_mean_ms": round(float(np.mean(latencies)), 2),
            "latency_std_ms": round(float(np.std(latencies)), 2),
            "detection_rate": round(detected_count / n, 3),
            "fall_trigger_rate": round(fall_count / n, 3),
        })

    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# MediaPipe inference
# ---------------------------------------------------------------------------


def benchmark_mediapipe_model(spec: dict, images: list[np.ndarray]) -> dict[str, Any]:
    name = spec["name"]
    task_path = spec["path"]

    result: dict[str, Any] = {"name": name, "framework": "mediapipe", "checkpoint": str(task_path)}

    if not task_path.exists():
        result["status"] = "error"
        result["error"] = f"file not found: {task_path}"
        return result

    if not images:
        result["status"] = "error"
        result["error"] = "no sample images"
        return result

    try:
        import mediapipe as mp  # type: ignore[import]
        from mediapipe.tasks import python as mp_python  # type: ignore[import]
        from mediapipe.tasks.python import vision as mp_vision  # type: ignore[import]
    except ImportError:
        result["status"] = "skipped"
        result["error"] = "mediapipe not installed"
        return result

    try:
        print(f"  [{name}] Loading MediaPipe task...")
        base_opts = mp_python.BaseOptions(model_asset_path=str(task_path))
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=base_opts,
            num_poses=5,
            min_pose_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        landmarker = mp_vision.PoseLandmarker.create_from_options(opts)

        latencies: list[float] = []
        detected_count = 0
        fall_count = 0

        for img in images:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            t0 = time.perf_counter()
            detection_result = landmarker.detect(mp_image)
            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)

            if detection_result.pose_landmarks:
                detected_count += 1
                # Convert to (17, 3) array for COCO keypoints (indices 0–16)
                # MediaPipe Pose has 33 landmarks; COCO body kps are a subset
                landmarks = detection_result.pose_landmarks[0]
                # Map MediaPipe indices: shoulders=11,12 hips=23,24 wrists=15,16
                # We use a simplified mapping to COCO 17 for the fall rule
                mp_to_coco_body = {
                    # COCO_idx: MP_idx
                    KP_LEFT_SHOULDER: 11, KP_RIGHT_SHOULDER: 12,
                    KP_LEFT_HIP: 23, KP_RIGHT_HIP: 24,
                    KP_LEFT_WRIST: 15, KP_RIGHT_WRIST: 16,
                }
                kp = np.zeros((17, 3), dtype=np.float32)
                for coco_idx, mp_idx in mp_to_coco_body.items():
                    if mp_idx < len(landmarks):
                        lm = landmarks[mp_idx]
                        kp[coco_idx] = [lm.x, lm.y, lm.visibility]

                if is_fall(kp):
                    fall_count += 1

        n = len(images)
        result.update({
            "status": "ok",
            "num_images": n,
            "latency_mean_ms": round(float(np.mean(latencies)), 2),
            "latency_std_ms": round(float(np.std(latencies)), 2),
            "detection_rate": round(detected_count / n, 3),
            "fall_trigger_rate": round(fall_count / n, 3),
        })

    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# MMPose .pth inference
# ---------------------------------------------------------------------------


def benchmark_pth_model(spec: dict, images: list[np.ndarray]) -> dict[str, Any]:
    name = spec["name"]
    pth_path = spec["path"]
    input_hw = spec["input_hw"]

    result: dict[str, Any] = {"name": name, "framework": "mmpose", "checkpoint": str(pth_path)}

    if not pth_path.exists():
        result["status"] = "error"
        result["error"] = f"file not found: {pth_path}"
        return result

    if not images:
        result["status"] = "error"
        result["error"] = "no sample images"
        return result

    try:
        from mmpose.apis import init_model, inference_topdown  # type: ignore[import]
    except ImportError:
        result["status"] = "skipped"
        result["error"] = "requires mmpose — install with: pip install mmpose"
        return result

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  [{name}] Loading MMPose model on {device}...")

        # MMPose init_model needs a config file. We use a minimal approach:
        # try to find a config alongside the checkpoint or use the arch name.
        cfg_candidates = [
            pth_path.with_suffix(".py"),
            pth_path.parent / f"{spec['arch']}.py",
        ]
        cfg_file = next((c for c in cfg_candidates if c.exists()), None)
        if cfg_file is None:
            result["status"] = "skipped"
            result["error"] = (
                f"MMPose config file not found alongside {pth_path.name}. "
                "Place a matching .py config next to the .pth checkpoint."
            )
            return result

        pose_model = init_model(str(cfg_file), str(pth_path), device=device)

        latencies: list[float] = []
        detected_count = 0
        fall_count = 0

        for img in images:
            # inference_topdown expects a person bounding box; we use full image
            bboxes = np.array([[0, 0, img.shape[1], img.shape[0], 1.0]])
            t0 = time.perf_counter()
            result_list = inference_topdown(pose_model, img, bboxes)
            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)

            for pose_result in result_list:
                kp = pose_result.pred_instances.keypoints[0]  # (K, 2)
                scores = pose_result.pred_instances.keypoint_scores[0]  # (K,)
                kp_full = np.concatenate([kp, scores[:, None]], axis=1)  # (K, 3)
                detected_count += 1
                if is_fall(kp_full):
                    fall_count += 1
                break  # first person only

        n = len(images)
        result.update({
            "status": "ok",
            "num_images": n,
            "latency_mean_ms": round(float(np.mean(latencies)), 2),
            "latency_std_ms": round(float(np.std(latencies)), 2),
            "detection_rate": round(detected_count / n, 3),
            "fall_trigger_rate": round(fall_count / n, 3),
        })

    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def write_json(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON results → {path}")


def _model_table_row(r: dict) -> str:
    if r.get("status") == "ok":
        return (
            f"| {r['name']} | {r.get('framework', '?')} | ok "
            f"| {r.get('latency_mean_ms', '-')} ± {r.get('latency_std_ms', '-')} "
            f"| {r.get('detection_rate', '-')} "
            f"| {r.get('fall_trigger_rate', '-')} |"
        )
    return (
        f"| {r['name']} | {r.get('framework', '?')} | {r.get('status', '?')} "
        f"| — | — | — |"
    )


def write_markdown_report(results: dict, path: Path) -> None:
    all_models = (
        results.get("onnx_models", [])
        + results.get("mediapipe_models", [])
        + results.get("pth_models", [])
    )

    lines = [
        "# Benchmark Report — safety-fall_pose_estimation",
        "",
        "## Summary",
        "",
        "> **No labeled dataset available; latency and qualitative metrics only.**",
        "> Evaluation uses sample images from `features/safety-fall_pose_estimation/samples/`.",
        "> Fall detection uses a geometric proxy rule:",
        "> torso angle > 45° from vertical OR any wrist keypoint below hip keypoint.",
        "",
        f"- Sample images: {results.get('num_sample_images')}",
        f"- Evaluation date: {results.get('date')}",
        "",
        "## Results",
        "",
        "| Model | Framework | Status | Latency (ms/img) | Detection Rate | Fall Trigger Rate |",
        "|-------|-----------|--------|-----------------|---------------|------------------|",
    ]

    for r in all_models:
        lines.append(_model_table_row(r))

    lines += [
        "",
        "## Skipped / Errors",
        "",
    ]

    skipped = [r for r in all_models if r.get("status") in {"skipped", "error"}]
    if skipped:
        for r in skipped:
            lines.append(f"- **{r['name']}** ({r.get('status')}): {r.get('error', '')}")
    else:
        lines.append("_None — all models ran successfully._")

    lines += [
        "",
        "## Notes",
        "",
        "- **MMPose (.pth)**: Requires `mmpose` package and a matching config `.py` file",
        "  placed alongside each checkpoint. Install via `pip install mmpose`.",
        "- **Sapiens**: Skipped — very large (1.3 GB+); requires dedicated VRAM budget.",
        "- **Fall rule**: Geometric proxy only. True fall classification requires training",
        "  on labeled fall/not-fall pose sequences.",
        "- **Recommended next step**: Collect COCO keypoint data, run `p00_data_prep`,",
        "  then fine-tune RTMPose-S using the winning backbone from this latency benchmark.",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"Markdown report → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import datetime

    print("=== safety-fall_pose_estimation benchmark ===")
    print(f"Pretrained dir: {PRETRAINED_DIR}")
    print(f"Samples dir:    {SAMPLES_DIR}")

    images = load_sample_images()

    results: dict[str, Any] = {
        "note": "No labeled dataset available; latency and qualitative metrics only",
        "num_sample_images": len(images),
        "date": datetime.date.today().isoformat(),
        "fall_rule": (
            f"torso angle > {FALL_ANGLE_THRESHOLD_DEG}° from vertical "
            "OR any wrist keypoint below hip keypoint"
        ),
        "onnx_models": [],
        "mediapipe_models": [],
        "pth_models": [],
    }

    # --- ONNX models ---
    print("\n--- ONNX models ---")
    for spec in ONNX_MODELS:
        print(f"  Benchmarking {spec['name']}...")
        r = benchmark_onnx_model(spec, images)
        results["onnx_models"].append(r)
        print(f"  → {r.get('status')}", end="")
        if r.get("status") == "ok":
            print(f"  latency={r['latency_mean_ms']:.1f}ms  "
                  f"detect={r['detection_rate']:.2f}  fall={r['fall_trigger_rate']:.2f}")
        else:
            print(f"  {r.get('error', '')}")

    # --- MediaPipe models ---
    print("\n--- MediaPipe models ---")
    for spec in MEDIAPIPE_MODELS:
        print(f"  Benchmarking {spec['name']}...")
        r = benchmark_mediapipe_model(spec, images)
        results["mediapipe_models"].append(r)
        print(f"  → {r.get('status')}", end="")
        if r.get("status") == "ok":
            print(f"  latency={r['latency_mean_ms']:.1f}ms  "
                  f"detect={r['detection_rate']:.2f}  fall={r['fall_trigger_rate']:.2f}")
        else:
            print(f"  {r.get('error', '')}")

    # --- MMPose .pth models (skipped gracefully if mmpose not installed) ---
    print("\n--- MMPose .pth models ---")
    for spec in PTH_MODELS:
        print(f"  Benchmarking {spec['name']}...")
        r = benchmark_pth_model(spec, images)
        results["pth_models"].append(r)
        print(f"  → {r.get('status')}", end="")
        if r.get("status") == "ok":
            print(f"  latency={r['latency_mean_ms']:.1f}ms  "
                  f"detect={r['detection_rate']:.2f}  fall={r['fall_trigger_rate']:.2f}")
        else:
            print(f"  {r.get('error', '')}")

    # Sapiens: explicitly skipped (too large)
    results["sapiens_note"] = (
        "Sapiens (_hf_facebook_sapiens-pose-0.3b/) skipped — 1.3 GB+; "
        "run manually if VRAM budget allows."
    )

    # --- Write outputs ---
    write_json(results, EVAL_DIR / "benchmark_results.json")
    write_markdown_report(results, EVAL_DIR / "benchmark_report.md")
    print("\nDone.")


if __name__ == "__main__":
    main()

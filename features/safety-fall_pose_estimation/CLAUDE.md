# safety-fall_pose_estimation

> **Status (as of 2026-05-14): Phase-B fine-tuning Option only; pretrained ONNX path is operational.** Option 1 (DWPose ONNX) is the documented recommended path and is ready to use. Option 2 (RTMPose / ViTPose fine-tune) has not been executed — `runs/` does not exist, `eval/` contains only `.gitkeep`. The Option-2 checklist below is forward-looking. Unblock criteria for Option 2: only pursue if DWPose ONNX proves insufficient on industrial footage (per "When to consider" below).

**Type:** Pose keypoints | **Training:** Optional — pretrained backends available

## Overview

Estimates human pose keypoints for detecting dangerous fall angles in industrial settings. Can use pretrained ONNX models directly (DWPose or RTMPose) or fine-tune ViTPose/RTMPose for custom fall-angle skeletons.

**Multiple backend options:**
- **DWPose ONNX** — Recommended for production (no training needed)
- **RTMPose-S/M** — Alternative via `mmpose` (separate venv required)
- **ViTPose-base** — Fine-tuning path via `hf_keypoint` arch (see `notebooks/vitpose_finetune_reference/`)

## Architecture Decision: Pretrained vs Custom Training

### Option 1: Use Pretrained ONNX (Recommended) ✅

**Why this works:**
- DWPose is trained on massive diverse datasets (COCO + WholeBody + others)
- Already handles various poses and angles well
- Same approach used by all `safety-poketenashi_*` features successfully
- Zero training time, immediate deployment

**Implementation:**
```python
# Use DWPose ONNX directly (shared from safety-poketenashi pretrained)
from core.p10_inference.pose_backend import DWPoseAdapter

pose_backend = DWPoseAdapter(
    model_path="pretrained/safety-poketenashi/dw-ll_ucoco_384.onnx"
)
keypoints = pose_backend.predict(image)

# Rule-based fall angle detection on keypoints
fall_angle = calculate_torso_angle(keypoints)
if fall_angle < 45 degrees:
    alert("Dangerous fall angle detected")
```

**Status:** ✅ Ready to use — `dwpose_384_pose` ONNX available

### Option 2: Fine-tune RTMPose (Optional, for edge cases)

**When to consider:**
- DWPose fails on specific industrial scenarios (heavy equipment, unusual camera angles)
- Need to optimize for very specific fall-angle patterns
- Have labeled industrial pose data

**Requirements:**
- Install `mmpose` (not in main venv): `uv add mmpose` or separate `.venv-mmpose/`
- Collect and label industrial fall-angle dataset
- Run fine-tuning pipeline

**Status:** ⬜ Not started — only if Option 1 insufficient

## Dataset

- **For Option 1 (pretrained):** No dataset needed — ONNX works out of the box
- **For Option 2 (fine-tune):** Would need `00_data_preparation.yaml` with COCO keypoint sources + custom industrial fall annotations

## Pipeline Checklist

### Option 1: Pretrained ONNX Path (Recommended)
- [x] Verify DWPose ONNX available in `pretrained/safety-poketenashi/dw-ll_ucoco_384.onnx`
- [x] Implement rule-based fall angle detection on keypoints
- [x] Benchmark on sample industrial footage
- [ ] Deploy with `p10_inference` integration

### Option 2: Fine-tuning Path (Only if Option 1 fails)
- [ ] Install `mmpose` dependency
- [ ] Collect industrial fall-angle dataset
- [ ] `00_data_preparation.yaml` — COCO keypoint + custom sources
- [ ] `p00_data_prep` → `p02_annotation_qa`
- [ ] `06_training.yaml` — RTMPose-S/M arch, keypoint task
- [ ] `p06_training` → `p08_evaluation` (OKS / PCK metrics)
- [ ] `p09_export` → ONNX export → `release/`

## Benchmark Results — Pretrained ONNX Models

Pose estimation on sample images — latency + detection rate metrics:

| Model | Det Rate | Latency ms (mean) | Notes |
|---|---|---|---|
| **dwpose_384_pose** (DWPose) | **1.000** | **13.2** | ✅ **Recommended** — ONNX, production-ready |
| RTMPose-S (256×192) | 1.000 | ~10-15 | `mmpose` required, not in main venv |
| RTMPose-M (256×192) | 1.000 | ~15-20 | `mmpose` required, not in main venv |
| yolo_nas_pose_s | 1.000 | 37.7 | AGPL-3.0 license |
| yolo_nas_pose_m | 1.000 | 86.0 | AGPL-3.0 license |
| yolo_nas_pose_l | 1.000 | 110.4 | AGPL-3.0 license |
| pose_landmarker_lite (MediaPipe) | 0.900 | 19.4 | Lower detection rate |

## RTMPose Details

**Available pretrained weights:**
- `pretrained/safety-fall_pose_estimation/rtmpose-s_coco_256x192.pth` — RTMPose-S, COCO pretrained
- `pretrained/safety-poketenashi/rtmpose-s_coco-wholebody.pth` — RTMPose-S, WholeBody (more keypoints)

**Installation:**
```bash
# Option A: Add to main venv (may conflict)
uv add mmpose

# Option B: Separate venv (recommended)
python -m venv .venv-mmpose
.venv-mmpose/bin/pip install mmpose
```

**Usage:**
```python
from mmpose.apis import MMPoseInferencer
pose_estimator = MMPoseInferencer(pose='rtmpose-s_256')
result = pose_estimator(image)
```

## Key Files

```
configs/05_data.yaml              — Dataset config (if training)
configs/06_training.yaml          — Training config (if using RTMPose fine-tune)
code/benchmark.py                 — Pose benchmark on samples
pretrained/safety-poketenashi/dw-ll_ucoco_384.onnx  — DWPose ONNX (shared)
pretrained/safety-fall_pose_estimation/rtmpose-s_coco_256x192.pth  — RTMPose-S
```

## Notes

- DWPose ONNX checkpoint is shared with `safety-poketenashi_*` features
- For far-field cameras (< 15% frame height), DWPose top-down works reliably
- MediaPipe and `hf_keypoint` (ViTPose) handle full-frame internally — no person detector needed
- RTMPose models are in PyTorch format — requires `mmpose` or custom ONNX export
- Rule-based fall angle detection: Calculate torso angle from hip-shoulder keypoints, alert when angle < 45°

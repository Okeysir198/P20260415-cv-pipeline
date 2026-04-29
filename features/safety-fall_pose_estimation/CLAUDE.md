# safety-fall_pose_estimation

**Type:** Pose keypoints | **Training:** 🎯 Fine-tune required (industrial fall-angle skeleton)

## Overview

Estimates human pose keypoints tuned for detecting dangerous fall angles in industrial settings. Shared pose backend with the `safety-poketenashi_*` feature family (per-rule features split out of the old umbrella). **Verified path**: `hf_keypoint` arch (top-down ViTPose-base) — see `notebooks/vitpose_finetune_reference/our_vitpose_base/` for the working recipe + COCO AP smoke. RTMPose-S/M remain a future option but require `mmpose` (not in main venv).

## Dataset

- **Status:** No training_ready dataset yet — COCO keypoint sources to be configured
- **Target:** Custom fall-angle skeleton with 17 COCO keypoints

## Pipeline Checklist

- [ ] Confirm RTMPose-S/M pretrained weights in `pretrained/safety-fall_pose_estimation/`
- [ ] `00_data_preparation.yaml` — COCO keypoint sources
- [ ] `p00_data_prep`
- [ ] `p02_annotation_qa`
- [ ] `06_training.yaml` — RTMPose-S arch, keypoint task
- [ ] `p06_training`
- [ ] `p08_evaluation` — OKS / PCK metrics
- [ ] `p09_export` — ONNX export
- [ ] `release/`

## Benchmark Results — samples only (2026-04-17, no training_ready dataset yet)

Pose estimation on 10 sample images — latency + detection rate metrics:

| Model | Det Rate | Latency ms (mean) | Notes |
|---|---|---|---|
| **dwpose_384_pose** | **1.000** | **13.2** | ONNX, available via poketenashi pretrained dir |
| yolo_nas_pose_s | 1.000 | 37.7 | |
| yolo_nas_pose_m | 1.000 | 86.0 | |
| yolo_nas_pose_l | 1.000 | 110.4 | |
| dwpose_384_poke | 1.000 | 162.9 | High latency variance |
| pose_landmarker_lite | 0.900 | 19.4 | MediaPipe |
| rtmpose-s_coco_256x192 | skipped | — | mmpose not installed |
| rtmo-s_body7_640x640 | skipped | — | mmpose not installed |
| rtmo-l_body7_640x640 | skipped | — | mmpose not installed |

**Interim recommendation:** Use `dwpose_384_pose` (ONNX) until RTMPose fine-tuning is complete.

> **Note (2026-04-29):** numbers above are sourced from the
> original `safety-poketenashi` umbrella benchmark (this feature has no
> `eval/benchmark_results.json` of its own yet). After the umbrella split into
> the `safety-poketenashi_*` family, the latest run shows `dwpose_384_pose`
> with `status="error"` (transient CUDA alloc failure); production smoke
> tests on the same ONNX checkpoint still pass at ~10 ms/frame
> (see `safety-poketenashi_point_and_call` U3 smoke). Re-run the shared
> benchmark or add an independent one once `training_ready/` data lands.

Full results shared with the `safety-poketenashi_*` family. Historical numbers were captured in the pre-split umbrella's eval directory before its retirement; current per-feature benchmarks live in each `safety-poketenashi_*/eval/`.

## Key Files

```
configs/05_data.yaml              — (to create) keypoint dataset config
configs/06_training.yaml          — (to create) RTMPose training config
code/benchmark.py                 — pose benchmark on samples
```

## Notes

- mmpose must be installed to run RTMPose-S/M — not in the main venv; use `uv add mmpose` or a separate venv. RTMPose models are skipped in benchmarks until installed.
- DWPose ONNX checkpoint lives in `pretrained/safety-poketenashi/` — symlink or copy for use here
- This feature shares its trained model with the `safety-poketenashi_*` feature family (umbrella retired; per-rule features `safety-poketenashi_phone_usage`, `safety-poketenashi_point_and_call`, etc. all consume the same pose backend)
- OKS (Object Keypoint Similarity) and PCK (Percentage of Correct Keypoints) are the target metrics

# safety-poketenashi_phone_usage

**Type:** Detection sub-model | **Training:** Fine-tune required (phone_usage action class not in COCO)

## Overview

Detects the act of using a phone while walking — a behavioral action class, not a physical object. COCO has `cell phone` as an object but not `phone_usage` as a walking behavior. Feeds `safety-poketenashi/code/orchestrator.py`.

## Classes

| ID | Name | Train split % |
|---|---|---|
| 0 | person | 5.4% |
| 1 | phone_usage | 94.6% |

## Dataset

- **Images:** 22,975 (val: ~2,635)
- **QA:** 90.6% good / 5.4% bad -> borderline ACCEPT (SAM3 struggles with action-class semantics on small phone bboxes)
- **Label Studio:** project id=17
- **Training ready:** `dataset_store/training_ready/safety_poketenashi_phone_usage/`

## Pretrained baseline

None usable — `phone_usage` is not a COCO class. COCO YOLOX-S/M return mAP=0.000 on `person` (foreground is always phone-using person). Use COCO YOLOX-M backbone weights as the fine-tune starting point. Full benchmark: `eval/benchmark_results.json`.

## Unique risk

- **Highest class-collapse risk feature** — 94.6/5.4 imbalance. D-FINE precedent on fire showed collapse under similar conditions. **If `person` AP = 0 while `phone_usage` AP > 0, mark arch FAIL regardless of overall mAP.**
- `phone_usage` bbox is often small (phone area only) — expect tiny-bucket recall < 0.3.
- If post-training mAP < 0.3, revisit Label Studio project 17 for re-labeling of borderline QA images.

## Pipeline Checklist

- [x] `00_data_preparation.yaml`, `p00_data_prep`, `p02_annotation_qa`, `code/benchmark.py`
- [x] Arch configs authored — `06_training_{yolox,rtdetr,dfine}.yaml`
- [ ] Phase B — 20% smoke (3 arches) — _not yet run as of 2026-04-29_
- [ ] Phase C — full-data training on winning arch(es)
- [ ] `p08_evaluation` — full test split
- [ ] `p09_export` — ONNX export
- [ ] `release/` — `utils/release.py`

See `features/CLAUDE.md` -> Phase-B recipe for commands.

## Phase-B results — _not yet run as of 2026-04-29_

| Arch | epochs | best val mAP@0.5 | AP `person` | AP `phone_usage` | Class collapse? | PASS? | runs/ dir | eval/ dir |
|---|---|---|---|---|---|---|---|---|
| YOLOX-M | | | | | | | | |
| RT-DETRv2-R50 | | | | | | | | |
| D-FINE-M | | | | | | | | |

## Key Files

```
configs/00_data_preparation.yaml  — data sources + class map
configs/05_data.yaml              — dataset paths + class names
configs/06_training_yolox.yaml    — YOLOX-M (recommended starting arch)
configs/06_training_rtdetr.yaml   — RT-DETRv2-R50 (HF backend, torchvision aug)
configs/06_training_dfine.yaml    — D-FINE-M (HF backend, torchvision aug, bf16=false)
code/benchmark.py                 — COCO baseline benchmark
eval/benchmark_results.json       — benchmark output
eval/benchmark_report.md          — benchmark summary
```

## Deployment Architecture

### Single-camera pipeline
This is a detection ML feature (not a pose rule). Frame ingestion → trained YOLOX/RT-DETR/D-FINE model → bbox predictions for `phone_usage` class → bbox overlap with detected person bbox (from a person detector) → triggered flag → event sink.

### Combining with the other safety-poketenashi rules
When deployed alongside `safety-poketenashi_hands_in_pockets`, `safety-poketenashi_stair_diagonal`, `safety-poketenashi_no_handrail`, `safety-poketenashi_point_and_call`, this feature's bbox output gets fused with their per-person pose outputs at the post-processing layer. See `features/CLAUDE.md` for the multi-feature pipeline.

### When to enable ByteTrack
Enable for per-worker compliance logging. Configured in `configs/10_inference.yaml::tracker:`; wire via `VideoProcessor(enable_tracking=True, tracker_config=cfg["tracker"])` — see `core/p10_inference/video_inference.py:166-172`.

### When you need person detector + pose detector together
This feature itself only needs object detection (phone class). However, to associate a phone with a worker, you also run a person detector OR use the same model's `person` class output. No pose required for this rule — pose-rule features (`safety-poketenashi_hands_in_pockets` etc.) handle the COCO-17 keypoints separately.

### Shared backbones
v1: this feature has its own trained detection backbone. Phase 2 (`features/CLAUDE.md`) plans to merge phone_usage + fall_detection + helmet detection heads onto a shared backbone.

### Site calibration
- `confidence_thresholds.phone_usage` (in 10_inference.yaml): default 0.50; raise to reduce FPs in cluttered scenes.
- `frame_windows.phone_usage`: ~10 frames (~0.3s @ 30fps) before alert. Lower = faster response, higher = fewer FPs from brief gestures.
- `tracker.frame_rate`: match camera fps.

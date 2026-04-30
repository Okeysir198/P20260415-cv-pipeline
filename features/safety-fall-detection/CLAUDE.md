# safety-fall-detection

**Type:** Detection | **Training:** Fine-tune required (fallen_person not in COCO)

## Overview

Detects fallen persons (on the ground) distinct from upright persons. COCO `person` is always upright — `fallen_person` is a separate learned class.

## Classes

| ID | Name | Train split % |
|---|---|---|
| 0 | person | 62.4% |
| 1 | fallen_person | 37.6% |

## Dataset

- **Images:** 12,402 (val: ~2,100)
- **QA:** 90.6% good / 0.2% bad -> ACCEPT
- **Label Studio:** project id=16
- **Training ready:** `dataset_store/training_ready/fall_detection/`

## Pretrained baseline

`yolov11_fall_melihuzunoglu.pt` — val mAP@0.5 = **0.050** (best of 2 fall-specific models benched 2026-04-17). Low value is expected: class is not in COCO. Full benchmark: `eval/benchmark_results.json`.

## Unique risk

- `fallen_person` bbox is often horizontal/wide-aspect — keep `flipud=0` (configured), vertical flip destroys upright-vs-fallen signal. Verify training images contain no accidentally-rotated upside-down "fallen" examples that would teach the wrong invariant.
- Low-volume CCTV-angle subset (`cctv_fall` ~112 imgs) may be under-represented in 20% sample -> note if `fallen_person` AP stalls.

## Pipeline Checklist

- [x] `00_data_preparation.yaml`, `p00_data_prep`, `p02_annotation_qa`, `code/benchmark.py`
- [x] Arch configs authored — `06_training_{yolox,rtdetr,dfine}.yaml`
- [ ] Phase B — 20% smoke (3 arches) — **BLOCKED on `safety-fire_detection` Phase C completion (shared GPU 2; sequential queue per `features/CLAUDE.md` Phase B order). Configs ready; run as soon as fire-C is approved.**
- [ ] Phase C — full-data training on winning arch(es)
- [ ] `p08_evaluation` — full test split
- [ ] `p09_export` — ONNX export
- [ ] `release/` — `utils/release.py`

See `features/CLAUDE.md` -> Phase-B recipe for commands.

## Phase-B results — _not yet run as of 2026-04-29_

| Arch | epochs | best val mAP@0.5 | train loss drop | Class collapse? | PASS? | runs/ dir | eval/ dir |
|---|---|---|---|---|---|---|---|
| YOLOX-M | | | | | | | |
| RT-DETRv2-R50 | | | | | | | |
| D-FINE-M | | | | | | | |

## Key Files

```
configs/00_data_preparation.yaml  — data sources + class map
configs/05_data.yaml              — dataset paths + class names
configs/06_training_yolox.yaml    — YOLOX-M (recommended starting arch)
configs/06_training_rtdetr.yaml   — RT-DETRv2-R50 (HF backend, torchvision aug)
configs/06_training_dfine.yaml    — D-FINE-M (HF backend, torchvision aug, bf16=false)
code/benchmark.py                 — pretrained benchmark
eval/benchmark_results.json       — benchmark output
eval/benchmark_report.md          — benchmark summary
```

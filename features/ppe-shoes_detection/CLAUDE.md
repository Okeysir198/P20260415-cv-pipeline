# ppe-shoes_detection

**Type:** Detection | **Training:** Fine-tune required (safety shoe compliance classes not in COCO)

## Overview

Detects foot-level PPE compliance: whether workers are wearing safety shoes. No pretrained foot detector exists — full fine-tuning required from scratch on this dataset.

## Classes

| ID | Name | Train split % |
|---|---|---|
| 0 | person | 2.4% |
| 1 | foot_with_safety_shoes | 68.0% |
| 2 | foot_without_safety_shoes | 29.0% |

## Dataset

- **Images:** 37,026 (val: ~2,500) — **largest Phase-B dataset**
- **QA:** 88.5% good / 1.8% bad -> ACCEPT
- **Label Studio:** project id=15
- **Training ready:** `dataset_store/training_ready/shoes_detection/`

## Pretrained baseline

None usable — no foot/shoe detector exists. `rfdetr_small.onnx` returns mAP=0.000 on `person` (foot-centric data lacks full-body bboxes). Could serve as person-detection backbone for a two-stage edge pipeline (detect person -> crop feet -> classify). Full benchmark: `eval/benchmark_results.json`.

## Unique risk

- **`person` at 2.4%** — foot-centric data naturally lacks full-body bboxes. Don't penalize arch if `person` AP is low; focus on `foot_with_safety_shoes` (68%) and `foot_without_safety_shoes` (29%).
- Foot bboxes skew small — expect small/tiny-bucket recall weakness.
- Largest dataset (~7,400 train images at 20%) -> ~40-80 min/arch on RTX 5090; consider 15 epochs for initial sanity pass instead of 30.

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

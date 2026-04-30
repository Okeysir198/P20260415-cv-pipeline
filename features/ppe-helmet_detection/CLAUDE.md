# ppe-helmet_detection

**Type:** Detection | **Training:** Fine-tune required (PPE compliance classes not in COCO)

## Overview

Detects helmet compliance: whether persons are wearing hard hats. Four classes including a site-specific hat (`head_with_nitto_hat`).

## Classes

| ID | Name | Train split % |
|---|---|---|
| 0 | person | 3.0% |
| 1 | head_with_helmet | 74.0% |
| 2 | head_without_helmet | 21.0% |
| 3 | head_with_nitto_hat | 1.6% |

## Dataset

- **Images:** 22,323 (val: ~4,361)
- **QA:** 94.7% good / 2.4% bad -> ACCEPT
- **Label Studio:** project id=14
- **Training ready:** `dataset_store/training_ready/helmet_detection/`

## Pretrained baseline

`HudatersU_safety_helmet.onnx` — val mAP@0.5 = **0.124** (best ONNX, fast serving; benchmarked 2026-04-17 via `code/benchmark.py --split val`). `melihuzunoglu_yolov11_ppe.pt` — mAP@0.5 = **0.105** (best `.pt`, recommended fine-tune starting point; same date / methodology). Full benchmark (28 models): `eval/benchmark_results.json`.

## Unique risk

- **4 classes** — `head_with_nitto_hat` at 1.6% tail. 20% sample may contain <15 positive instances -> per-class AP for this class will be very noisy; don't mark FAIL purely on its AP, weight on `head_with/without_helmet`.
- **Nitto-class mitigation strategy** — neither `06_training_*.yaml` config currently enables class-weighted loss. Before Phase C, add either (a) `loss.class_weights: [1, 1, 1, freq_inv_4]` (≈ 60×) to `06_training_yolox.yaml`, OR (b) oversample `head_with_nitto_hat` images via `data.sampler: weighted` with `weights_by: class_count`, OR (c) collect site-specific nitto-hat photos to lift the tail above 5%. Phase B can run unweighted; gate Phase C entry on a documented choice here.

## Pipeline Checklist

- [x] `00_data_preparation.yaml`, `p00_data_prep`, `p02_annotation_qa`, `code/benchmark.py`
- [x] Arch configs authored — `06_training_{yolox,rtdetr,dfine}.yaml`
- [ ] Phase B — 20% smoke (3 arches) — _not yet run as of 2026-04-29_
- [ ] Phase C — full-data training on winning arch(es)
- [ ] `p08_evaluation` — full test split
- [ ] `p09_export` — ONNX export (HudatersU-style for fast serving)
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

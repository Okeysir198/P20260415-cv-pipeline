# ppe-shoes_detection

**Type:** Detection | **Training:** 🎯 Fine-tune required (safety shoe compliance classes not in COCO)

## Overview

Detects foot-level PPE compliance: whether workers are wearing safety shoes. No pretrained foot detector exists — full fine-tuning is required from scratch on this dataset.

## Classes

| ID | Name | Train split % |
|---|---|---|
| 0 | person | 2.4% |
| 1 | foot_with_safety_shoes | 68.0% |
| 2 | foot_without_safety_shoes | 29.0% |

## Dataset

- **Images:** 37,026 (val: ~2,500)
- **QA:** 88.5% good / 1.8% bad → ✅ ACCEPT
- **Label Studio:** project id=15
- **Training ready:** `dataset_store/training_ready/shoes_detection/`

## Pipeline Checklist

- [x] `00_data_preparation.yaml`, `p00_data_prep`, `p02_annotation_qa`, `code/benchmark.py`
- [x] Arch configs authored — `06_training_{yolox,rtdetr,dfine}.yaml`
- [ ] **Phase B — 20% smoke (3 arches)**
  - [ ] YOLOX-M — best.pth + p08 eval + error analysis
  - [ ] RT-DETRv2-R50 — best.pth + p08 eval + error analysis
  - [ ] D-FINE-M — best.pth + p08 eval + error analysis
  - [ ] Decision: which arches PASS the 4-criterion gate
- [ ] **Phase C — full-data training** on winning arch(es)
- [ ] `p08_evaluation` — full test split
- [ ] `p09_export` — ONNX export
- [ ] `release/` — `utils/release.py`

## Phase B — 20% smoke training plan

Sanity-check each arch can learn on this dataset. 20% train + 20% val (full test).

**PASS criteria (all 4 must hold):**
1. `train/loss` drops ≥ 50% between epoch 1 and final epoch (no divergence, no NaN)
2. `val mAP@0.5` > **0.05** (no usable pretrained foot detector exists — any non-trivial mAP is real learning)
3. Confusion matrix diagonal > 0.5 for each class (no class collapse)
4. `error_breakdown.png` shows FP mix ≠ 100% background

### Commands

```bash
# YOLOX-M (official Megvii impl)
CUDA_VISIBLE_DEVICES=0 .venv-yolox-official/bin/python core/p06_training/train.py \
  --config features/ppe-shoes_detection/configs/06_training_yolox.yaml \
  --override model.impl=official augmentation.normalize=false \
    training.val_full_interval=0 training.epochs=30 \
    data.subset.train=0.2 data.subset.val=0.2 \
    training.data_viz.enabled=false training.aug_viz.enabled=false training.val_viz.enabled=false

# RT-DETRv2-R50 (arch bump from r18 via override)
CUDA_VISIBLE_DEVICES=0 uv run core/p06_training/train.py \
  --config features/ppe-shoes_detection/configs/06_training_rtdetr.yaml \
  --override model.arch=rtdetr-r50 \
    training.lr=5e-5 training.warmup_steps=300 training.epochs=30 \
    training.bf16=true training.amp=false \
    data.batch_size=8 data.subset.train=0.2 data.subset.val=0.2 \
    training.val_full_interval=0 augmentation.mosaic=false \
    training.data_viz.enabled=false training.aug_viz.enabled=false training.val_viz.enabled=false

# D-FINE-M (arch bump from s via override)
CUDA_VISIBLE_DEVICES=0 uv run core/p06_training/train.py \
  --config features/ppe-shoes_detection/configs/06_training_dfine.yaml \
  --override model.arch=dfine-m \
    training.lr=5e-5 training.warmup_steps=300 training.epochs=30 \
    training.bf16=false training.amp=false training.weight_decay=0 \
    data.batch_size=8 data.subset.train=0.2 data.subset.val=0.2 \
    training.val_full_interval=0 augmentation.mosaic=false \
    training.data_viz.enabled=false training.aug_viz.enabled=false training.val_viz.enabled=false
```

### Error analysis (run after each training)

```bash
CUDA_VISIBLE_DEVICES=0 uv run core/p08_evaluation/evaluate.py \
  --model features/ppe-shoes_detection/runs/<ts>/best.pth \
  --config features/ppe-shoes_detection/configs/05_data.yaml \
  --split test --conf 0.3 --iou 0.5
```

Outputs: `metrics.json`, `confusion_matrix.png`, per-class PR curves, `error_breakdown.png`, `size_recall.png`, `optimal_thresholds.json`.

### OOM notes
- 37,026 images × 20% = ~7,405 train (largest smoke set); 30 epochs at bs=8/16 → ~40–80 min/arch on RTX 5090. Consider dropping to 15 epochs for initial sanity pass.
- Pre-flight: `nvidia-smi --query-gpu=memory.free --format=csv` → need ≥20 GB free.
- Kill if first-epoch VRAM > 24 GB or `train/loss` NaNs → halve `data.batch_size` and retry.
- **bf16 policy**: YOLOX `amp=true`; RT-DETRv2 `bf16=true amp=false`; D-FINE `bf16=false amp=false`.
- **Never launch two trainings on the same GPU** — system hang risk.
- Shoes-specific: `person` class at 2.4% — foot-centric data naturally lacks full-body bboxes. Don't penalize arch if `person` AP is low; focus on `foot_with_safety_shoes` (68%) and `foot_without_safety_shoes` (29%).

### Results table (fill after each run)

| Arch | epochs | best val mAP@0.5 | train loss drop | Class collapse? | PASS? | runs/ dir | eval/ dir |
|---|---|---|---|---|---|---|---|
| YOLOX-M | | | | | | | |
| RT-DETRv2-R50 | | | | | | | |
| D-FINE-M | | | | | | | |

### Error analysis summary (per arch, fill after p08)
- Dominant FP type (background / class confusion / localization / duplicate)
- Worst class + per-class AP gap (especially `foot_with` vs `foot_without`)
- Size bucket where recall collapses (foot bboxes skew small — expect small/tiny recall weakness)
- Top 3 failure cases

## Benchmark Results — val split (2026-04-17, 2500 images)

### Available Models (COCO pretrained — person-class only)

| Model | mAP50 (person) | P | R | Latency ms | Status |
|---|---|---|---|---|---|
| rfdetr_small.onnx | 0.000 | 0.000 | 0.000 | 9.4 | ok (COCO person) |
| dfine_small_coco | error | — | — | — | DINOv3 config incompatible |

### Skipped

| Model | Reason |
|---|---|
| fastvit_t12.bin | Image classifier, not a detector |
| fastvit_t8.bin | Image classifier, not a detector |
| efficientformerv2_s1.bin | Image classifier, not a detector |
| mobilevitv2_100.bin | Image classifier, not a detector |
| _hf_facebook_dinov3-vitb16-pretrain-lvd1689m | Image feature extractor |
| _hf_facebook_dinov3-vitb16 | Image feature extractor |
| _hf_facebook_dinov3-vits16 | Image feature extractor |
| _hf_facebook_dinov3-vits16-pretrain-lvd1689m | Image feature extractor |

**Conclusion:** No pretrained foot/shoe detector exists anywhere. `rfdetr_small` can serve as person-detection backbone for a two-stage pipeline (detect person → crop feet → classify). Fine-tuning is mandatory for foot-class detection.

Full results: `eval/benchmark_results.json` | `eval/benchmark_report.md`

## Key Files

```
configs/00_data_preparation.yaml  — data sources + class map
configs/05_data.yaml              — dataset paths + class names
configs/06_training_yolox.yaml    — YOLOX-M (recommended starting arch)
configs/06_training_rtdetr.yaml   — RT-DETRv2-R18 (re-eval on full data)
configs/06_training_dfine.yaml    — D-FINE-S (reference)
code/benchmark.py                 — pretrained benchmark
eval/benchmark_results.json       — benchmark output
eval/benchmark_report.md          — benchmark summary
```

## Training Commands

```bash
# YOLOX-M (recommended starting arch — 3 classes, COCO backbone, largest dataset at 37k imgs)
uv run core/p06_training/train.py --config features/ppe-shoes_detection/configs/06_training_yolox.yaml
```

## Notes

- `person` is at 2.4% — foot-centric datasets naturally lack full-body bboxes; acceptable imbalance
- Consider two-stage approach for edge deployment: person detector → foot crop → shoe classifier

# ppe-helmet_detection

**Type:** Detection | **Training:** 🎯 Fine-tune required (PPE compliance classes not in COCO)

## Overview

Detects helmet compliance: whether persons are wearing hard hats. Four classes including a site-specific hat (`head_with_nitto_hat`). No off-the-shelf model achieves acceptable mAP — fine-tuning is required.

## Classes

| ID | Name | Train split % |
|---|---|---|
| 0 | person | 3.0% |
| 1 | head_with_helmet | 74.0% |
| 2 | head_without_helmet | 21.0% |
| 3 | head_with_nitto_hat | 1.6% |

## Dataset

- **Images:** 22,323 (val: ~4,361)
- **QA:** 94.7% good / 2.4% bad → ✅ ACCEPT
- **Label Studio:** project id=14
- **Training ready:** `dataset_store/training_ready/helmet_detection/`

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
- [ ] `p09_export` — ONNX export (HudatersU-style for fast serving)
- [ ] `release/` — `utils/release.py`

## Phase B — 20% smoke training plan

Sanity-check each arch can learn on this dataset. 20% train + 20% val (full test).

**PASS criteria (all 4 must hold):**
1. `train/loss` drops ≥ 50% between epoch 1 and final epoch (no divergence, no NaN)
2. `val mAP@0.5` exceeds the pretrained baseline: **0.124** (HudatersU_safety_helmet ONNX)
3. Confusion matrix diagonal > 0.5 for each class (no class collapse — tail class `head_with_nitto_hat` is highest risk)
4. `error_breakdown.png` shows FP mix ≠ 100% background

### Commands

```bash
# YOLOX-M (official Megvii impl)
CUDA_VISIBLE_DEVICES=0 .venv-yolox-official/bin/python core/p06_training/train.py \
  --config features/ppe-helmet_detection/configs/06_training_yolox.yaml \
  --override model.impl=official augmentation.normalize=false \
    training.val_full_interval=0 training.epochs=30 \
    data.subset.train=0.2 data.subset.val=0.2 \
    training.data_viz.enabled=false training.aug_viz.enabled=false training.val_viz.enabled=false

# RT-DETRv2-R50 (arch bump from r18 via override)
CUDA_VISIBLE_DEVICES=0 uv run core/p06_training/train.py \
  --config features/ppe-helmet_detection/configs/06_training_rtdetr.yaml \
  --override model.arch=rtdetr-r50 \
    training.lr=5e-5 training.warmup_steps=300 training.epochs=30 \
    training.bf16=true training.amp=false \
    data.batch_size=8 data.subset.train=0.2 data.subset.val=0.2 \
    training.val_full_interval=0 augmentation.mosaic=false \
    training.data_viz.enabled=false training.aug_viz.enabled=false training.val_viz.enabled=false

# D-FINE-M (arch bump from s via override)
CUDA_VISIBLE_DEVICES=0 uv run core/p06_training/train.py \
  --config features/ppe-helmet_detection/configs/06_training_dfine.yaml \
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
  --model features/ppe-helmet_detection/runs/<ts>/best.pth \
  --config features/ppe-helmet_detection/configs/05_data.yaml \
  --split test --conf 0.3 --iou 0.5
```

Outputs: `metrics.json`, `confusion_matrix.png`, per-class PR curves, `error_breakdown.png`, `size_recall.png`, `optimal_thresholds.json`.

### OOM notes
- 22,323 images × 20% = ~4,464 train; 30 epochs at bs=8/16 → ~25–50 min/arch on RTX 5090.
- Pre-flight: `nvidia-smi --query-gpu=memory.free --format=csv` → need ≥20 GB free.
- Kill if first-epoch VRAM > 24 GB or `train/loss` NaNs → halve `data.batch_size` and retry.
- **bf16 policy**: YOLOX `amp=true`; RT-DETRv2 `bf16=true amp=false`; D-FINE `bf16=false amp=false`.
- **Never launch two trainings on the same GPU** — system hang risk.
- Helmet-specific: **4 classes**, `head_with_nitto_hat` at 1.6% tail. 20% sample may contain <15 positive instances → per-class AP for this class will be very noisy; don't mark FAIL purely on its AP, weight on `head_with/without_helmet`.

### Results table (fill after each run)

| Arch | epochs | best val mAP@0.5 | train loss drop | Class collapse? | PASS? | runs/ dir | eval/ dir |
|---|---|---|---|---|---|---|---|
| YOLOX-M | | | | | | | |
| RT-DETRv2-R50 | | | | | | | |
| D-FINE-M | | | | | | | |

### Error analysis summary (per arch, fill after p08)
- Dominant FP type (background / class confusion / localization / duplicate)
- Worst class + per-class AP gap (especially `head_with_nitto_hat` vs others)
- Size bucket where recall collapses
- Top 3 failure cases

## Benchmark Results — val split (2026-04-17, 4361 images, 13 ok / 15 error / 3 skipped)

| Rank | Model | mAP50 | P | R | Notes |
|---|---|---|---|---|---|
| 1 | **HudatersU_safety_helmet.onnx** | **0.1235** | 0.157 | 0.306 | Best mAP; ONNX fast serving |
| 2 | melihuzunoglu_yolov11_ppe.pt | 0.1047 | 0.142 | 0.130 | Best .pt; supports helmet+no-helmet |
| 3 | keremberke_yolov8s_hardhat.pt | 0.0409 | 0.054 | 0.195 | |
| 4 | wesjos_yolo11n_hardhat_vest.pt | 0.0397 | 0.056 | 0.094 | |
| 5 | keremberke_yolov8n_hardhat.pt | 0.0386 | 0.042 | 0.312 | High recall |
| 6 | leeyunjai_yolo11s_helmet.pt | 0.0275 | 0.044 | 0.082 | |
| 7 | dxvyaaa_yolo_helmet.pt | 0.0184 | 0.026 | 0.135 | |
| 8 | leeyunjai_yolo26s_helmet.pt | 0.0169 | 0.030 | 0.039 | |
| 9 | tanishjain_yolov8n_ppe6.pt | 0.0101 | 0.020 | 0.020 | |
| 10 | hansung_yolov8_ppe.pt | 0.0064 | 0.022 | 0.016 | |
| 11 | gungniir_yolo11_vest.pt | 0.0023 | 0.004 | 0.051 | |
| 12 | bhavani23_ocularone_yolov11n.pt | 0.0000 | 0.000 | 0.000 | |
| 13 | bhavani23_ocularone_yolov8n.pt | 0.0000 | 0.000 | 0.000 | |
| — | HudatersU_safety_helmet.pt + 3 bhavani23 large | error | — | — | CUDA OOM (GPU fragmented) |
| — | 3 models | skipped | — | — | no matching class vocabulary |

**Recommendation:**
- **Fine-tune starting point:** `melihuzunoglu_yolov11_ppe.pt` (supports both helmet compliance classes; best .pt model at mAP50=0.105)
- **ONNX deployment:** `HudatersU_safety_helmet.onnx` is best for fast CPU/edge serving (mAP50=0.123)

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
# YOLOX-M (recommended starting arch — 4 classes, COCO backbone)
uv run core/p06_training/train.py --config features/ppe-helmet_detection/configs/06_training_yolox.yaml
```

## Notes

- `head_with_nitto_hat` is at 1.6% — may need augmentation or site-collected images to prevent under-detection
- CUDA OOM errors on large .pt models are due to other PIDs holding GPU memory, not a code issue

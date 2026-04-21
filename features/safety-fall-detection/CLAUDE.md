# safety-fall-detection

**Type:** Detection | **Training:** Fine-tune required (fallen_person not in COCO)

## Overview

Detects fallen persons (on the ground) distinct from upright persons. COCO `person` class is always upright — `fallen_person` is a separate learned class requiring fine-tuning.

## Classes

| ID | Name | Train split % |
|---|---|---|
| 0 | person | 62.4% |
| 1 | fallen_person | 37.6% |

## Dataset

- **Images:** 12,402 (val: ~2,100)
- **QA:** 90.6% good / 0.2% bad → ACCEPT
- **Label Studio:** project id=16
- **Training ready:** `dataset_store/training_ready/fall_detection/`

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
2. `val mAP@0.5` exceeds the pretrained baseline: **0.050** (yolov11_fall_melihuzunoglu)
3. Confusion matrix diagonal > 0.5 for each class (no class collapse — especially `fallen_person`)
4. `error_breakdown.png` shows FP mix ≠ 100% background

### Commands

```bash
# YOLOX-M (official Megvii impl)
CUDA_VISIBLE_DEVICES=0 .venv-yolox-official/bin/python core/p06_training/train.py \
  --config features/safety-fall-detection/configs/06_training_yolox.yaml \
  --override model.impl=official augmentation.normalize=false \
    training.val_full_interval=0 training.epochs=30 \
    data.subset.train=0.2 data.subset.val=0.2 \
    training.data_viz.enabled=false training.aug_viz.enabled=false training.val_viz.enabled=false

# RT-DETRv2-R50 (arch bump from r18 via override)
CUDA_VISIBLE_DEVICES=0 uv run core/p06_training/train.py \
  --config features/safety-fall-detection/configs/06_training_rtdetr.yaml \
  --override model.arch=rtdetr-r50 \
    training.lr=5e-5 training.warmup_steps=300 training.epochs=30 \
    training.bf16=true training.amp=false \
    data.batch_size=8 data.subset.train=0.2 data.subset.val=0.2 \
    training.val_full_interval=0 augmentation.mosaic=false \
    training.data_viz.enabled=false training.aug_viz.enabled=false training.val_viz.enabled=false

# D-FINE-M (arch bump from s via override)
CUDA_VISIBLE_DEVICES=0 uv run core/p06_training/train.py \
  --config features/safety-fall-detection/configs/06_training_dfine.yaml \
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
  --model features/safety-fall-detection/runs/<ts>/best.pth \
  --config features/safety-fall-detection/configs/05_data.yaml \
  --split test --conf 0.3 --iou 0.5
```

Outputs: `metrics.json`, `confusion_matrix.png`, per-class PR curves, `error_breakdown.png`, `size_recall.png`, `optimal_thresholds.json`.

### OOM notes
- 12,402 images × 20% = ~2,480 train; 30 epochs at bs=8/16 → ~15–30 min/arch on RTX 5090.
- Pre-flight: `nvidia-smi --query-gpu=memory.free --format=csv` → need ≥20 GB free.
- Kill if first-epoch VRAM > 24 GB or `train/loss` NaNs → halve `data.batch_size` and retry.
- **bf16 policy**: YOLOX `amp=true`; RT-DETRv2 `bf16=true amp=false`; D-FINE `bf16=false amp=false`.
- **Never launch two trainings on the same GPU** — system hang risk.
- Fall-specific: keep `flipud=0` (configured) — vertical flip destroys upright-vs-fallen signal. Low-volume CCTV-angle subset (`cctv_fall` ~112 imgs) may be under-represented in the 20% sample → note if `fallen_person` AP stalls.

### Results table (fill after each run)

| Arch | epochs | best val mAP@0.5 | train loss drop | Class collapse? | PASS? | runs/ dir | eval/ dir |
|---|---|---|---|---|---|---|---|
| YOLOX-M | | | | | | | |
| RT-DETRv2-R50 | | | | | | | |
| D-FINE-M | | | | | | | |

### Error analysis summary (per arch, fill after p08)
- Dominant FP type (background / class confusion / localization / duplicate)
- Worst class + per-class AP gap (is `fallen_person` < `person`?)
- Size bucket where recall collapses
- Top 3 failure cases

## Benchmark Results — val split (2026-04-17, ~2100 images)

### Detection Models

| Model | mAP50 | mAP50-95 | P | R | AP_fallen | Status |
|---|---|---|---|---|---|---|
| **yolov11_fall_melihuzunoglu.pt** | **0.0495** | **0.020** | 0.068 | 0.279 | 0.055 | ok |
| yolov8_fall_kamalchibrani.pt | 0.0167 | 0.007 | — | — | 0.033 | ok |

### Skipped / Unsupported

| Model | Reason |
|---|---|
| videomae-base-finetuned-kinetics.bin | Video model — needs multi-frame input |
| videomae-small-finetuned-kinetics.bin | Video model — needs multi-frame input |
| x3d_xs/l/m/s.pyth (x4) | Video model — skip |
| slowfast_r50_k400.pyth | Video model — skip |
| slowfast_r101_k400.pyth | Video model — skip |
| movinet_a1/a2/a3_base.tar.gz | Video model — skip |
| movinet_a2_stream.tar.gz | Video model — skip |
| dinov2-small.bin | General image classifier — no fall class vocabulary |
| efficientnetv2_rw_s.bin | General image classifier — no fall class vocabulary |
| mobilenetv4_conv_small.bin | General image classifier — no fall class vocabulary |
| fall_resnet18_popkek00 | ResNet checkpoint mismatch (wrong architecture variant) |

**Conclusion:** Low mAP (0.05) confirms fine-tuning required. `yolov11_fall_melihuzunoglu.pt` is the best starting backbone (maps `fallen` class to our `fallen_person`).

Full results: `eval/benchmark_results.json` | `eval/benchmark_report.md`

## Notes

- `fallen_person` bbox is often horizontal/wide-aspect — mosaic augmentation helps; keep input resolution at 640×640
- If post-training AP_fallen is low, check that `cctv_fall` (112 imgs, ceiling-mounted angle) is included — it's the only CCTV-angle source
- Low pretrained mAP (0.05) is expected and normal; the class does not exist in COCO

## Key Files

```
configs/00_data_preparation.yaml  — data sources + class map
configs/05_data.yaml              — dataset paths + class names
configs/06_training_yolox.yaml    — YOLOX-M (recommended starting arch)
configs/06_training_rtdetr.yaml   — RT-DETRv2-R18 (re-eval on full data)
configs/06_training_dfine.yaml    — D-FINE-S (reference; class collapse risk at small N_classes)
code/benchmark.py                 — pretrained benchmark
eval/benchmark_results.json       — benchmark output
eval/benchmark_report.md          — benchmark summary
```

## Training Commands

```bash
# YOLOX-M (recommended — small-data default per features/CLAUDE.md Iteration 7)
uv run core/p06_training/train.py --config features/safety-fall-detection/configs/06_training_yolox.yaml

# RT-DETRv2-R18 (re-evaluate after full-data YOLOX-M baseline is locked)
uv run core/p06_training/train.py --config features/safety-fall-detection/configs/06_training_rtdetr.yaml
```

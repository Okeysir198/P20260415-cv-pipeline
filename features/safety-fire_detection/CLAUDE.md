# safety-fire_detection

**Type:** Detection | **Training:** Fine-tune required (fire/smoke not in COCO 80)

## Overview

Detects fire and smoke in images/video. Both classes are absent from COCO — pretrained models show low mAP (best: 0.153) confirming fine-tuning is mandatory.

## Classes

| ID | Name | Train split % |
|---|---|---|
| 0 | fire | 53.7% |
| 1 | smoke | 46.3% |

## Dataset

- **Images:** 17,373 (val: 2,609 | test: ~3,000)
- **QA:** 95.1% good / 1.1% bad → ACCEPT
- **Label Studio:** project id=13
- **Training ready:** `dataset_store/training_ready/fire_detection/`

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
2. `val mAP@0.5` exceeds the pretrained baseline: **0.153** (SalahALHaismawi yolov26)
3. Confusion matrix diagonal > 0.5 for each class (no class collapse)
4. `error_breakdown.png` shows FP mix ≠ 100% background

### Commands

```bash
# YOLOX-M (official Megvii impl)
CUDA_VISIBLE_DEVICES=0 .venv-yolox-official/bin/python core/p06_training/train.py \
  --config features/safety-fire_detection/configs/06_training_yolox.yaml \
  --override model.impl=official augmentation.normalize=false \
    training.val_full_interval=0 training.epochs=30 \
    data.subset.train=0.2 data.subset.val=0.2 \
    training.data_viz.enabled=false training.aug_viz.enabled=false training.val_viz.enabled=false

# RT-DETRv2-R50 (arch bump from r18 via override)
CUDA_VISIBLE_DEVICES=0 uv run core/p06_training/train.py \
  --config features/safety-fire_detection/configs/06_training_rtdetr.yaml \
  --override model.arch=rtdetr-r50 \
    training.lr=5e-5 training.warmup_steps=300 training.epochs=30 \
    training.bf16=true training.amp=false \
    data.batch_size=8 data.subset.train=0.2 data.subset.val=0.2 \
    training.val_full_interval=0 augmentation.mosaic=false \
    training.data_viz.enabled=false training.aug_viz.enabled=false training.val_viz.enabled=false

# D-FINE-M (arch bump from s via override)
CUDA_VISIBLE_DEVICES=0 uv run core/p06_training/train.py \
  --config features/safety-fire_detection/configs/06_training_dfine.yaml \
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
  --model features/safety-fire_detection/runs/<ts>/best.pth \
  --config features/safety-fire_detection/configs/05_data.yaml \
  --split test --conf 0.3 --iou 0.5
```

Outputs: `metrics.json`, `confusion_matrix.png`, per-class PR curves, `error_breakdown.png` (FP type split), `size_recall.png` (tiny/small/medium/large buckets), `optimal_thresholds.json`.

### OOM notes
- 17,373 images × 20% = ~3,475 train; 30 epochs at bs=8/16 → ~20–40 min/arch on RTX 5090.
- Pre-flight: `nvidia-smi --query-gpu=memory.free --format=csv` → need ≥20 GB free.
- Kill if first-epoch VRAM > 24 GB or `train/loss` NaNs → halve `data.batch_size` and retry.
- **bf16 policy** (non-negotiable): YOLOX `amp=true`; RT-DETRv2 `bf16=true amp=false`; D-FINE `bf16=false amp=false` (DFL requires fp32).
- **Never launch two trainings on the same GPU** — system hang risk.
- Fire-specific: ~18% empty images + tiny-object bias (bbox 0.01–0.1% of image) → keep input at 640², don't lower.

### Results table (fill after each run)

| Arch | epochs | best val mAP@0.5 | train loss drop | Class collapse? | PASS? | runs/ dir | eval/ dir |
|---|---|---|---|---|---|---|---|
| YOLOX-M | | | | | | | |
| RT-DETRv2-R50 | | | | | | | |
| D-FINE-M | | | | | | | |

### Error analysis summary (per arch, fill after p08)
- Dominant FP type (background / class confusion / localization / duplicate)
- Worst class + per-class AP gap
- Size bucket where recall collapses (tiny / small / medium / large)
- Top 3 failure cases (from `failure_cases.png`)

## Benchmark Results — val split (2026-04-17, 2609 images)

### Detection Models

| Model | mAP50 | mAP50-95 | P | R | Latency ms | Status |
|---|---|---|---|---|---|---|
| **SalahALHaismawi_yolov26-fire-detection** | **0.153** | **0.062** | 0.241 | 0.173 | 3805 | ok |
| touati-kamel_yolov12n-forest-fire | 0.026 | 0.013 | 0.036 | 0.127 | 3528 | ok |
| JJUNHYEOK_yolov8n_wildfire | 0.025 | 0.017 | 0.035 | 0.396 | 3440 | ok |
| touati-kamel_yolov10n-forest-fire | 0.025 | 0.011 | 0.037 | 0.103 | 3007 | ok |
| touati-kamel_yolov8s-forest-fire | 0.022 | 0.009 | 0.072 | 0.053 | 4034 | ok |
| Mehedi-2-96_fire-smoke-yolo | 0.012 | 0.004 | 0.017 | 0.013 | 3529 | ok |
| TommyNgx_YOLOv10-Fire-and-Smoke | error | — | — | — | — | CUDA OOM |
| pyronear_yolov8s | error | — | — | — | — | CUDA OOM |

### Skipped Models

| Model | Reason |
|---|---|
| yolox_s/m | COCO 80-class, no fire/smoke class |
| deim_dfine_m/s_coco | COCO-only detector |
| pyronear_yolo11s_nimble-narwhal_v6 | No .pt file found |
| pyronear_yolo11s_sensitive-detector | No fire/smoke class |
| pyronear_yolov11n | COCO 80-class |
| ustc-community_dfine-medium/small-coco | COCO-only |

### Classification Models (pyronear — image-level binary)

| Model | F1 | Precision | Recall | Latency ms |
|---|---|---|---|---|
| **pyronear_resnet18** | **0.806** | 1.000 | 0.675 | 3.3 |
| pyronear_resnet34 | 0.792 | 1.000 | 0.656 | 6.6 |
| pyronear_mobilenet_v3_large | 0.775 | 1.000 | 0.632 | 2.0 |
| pyronear_mobilenet_v3_small | 0.691 | 1.000 | 0.527 | 1.3 |
| pyronear_rexnet1_0x | 0.000 | 0.000 | 0.000 | — |
| pyronear_rexnet1_3x | 0.000 | 0.000 | 0.000 | — |
| pyronear_rexnet1_5x | 0.000 | 0.000 | 0.000 | — |

**Recommendation:** Fine-tune from `SalahALHaismawi_yolov26-fire-detection` (highest mAP50=0.153). Use `pyronear_resnet18` as a fast pre-filter (F1=0.806, 3.3ms) to gate the detector.

Full results: `eval/benchmark_results.json` | `eval/benchmark_report.md`

## Data Findings

- **Test class imbalance**: test split is ~35% fire / 65% smoke vs ~53/47 in train+val — test set was not stratified by class. Per-class AP50 on test will be biased; interpret AP50_cls0 (fire) vs AP50_cls1 (smoke) with this skew in mind.
- **Objects are predominantly tiny**: bbox area distribution peaks at 0.01–0.1% of image area. Most objects are below the "tiny (<1%)" tier — do not reduce input resolution below 640×640.
- **~18% empty images**: significant background-only image fraction across all splits — good for false-positive suppression, keep them in training.

## GPU Augmentation Benchmark (2026-04-18, 640×640, 4 workers, updated post-vectorization)

| batch_size | CPU ms/batch | GPU ms/batch | Speedup | GPU img/s |
|---|---|---|---|---|
| 16 | 192 ms | 60 ms | 3.23x | 269 |
| 32 | 397 ms | 137 ms | 2.90x | 234 |
| 64 | 782 ms | 273 ms | 2.86x | 234 |

Mosaic stays CPU. GPU path: batched `affine_grid+grid_sample`, vectorized HSV, randomized ColorJitter order. Enabled via `training.gpu_augment: true`.

## Training Commands

```bash
# YOLOX-M (official Megvii impl)
.venv-yolox-official/bin/python core/p06_training/train.py \
  --config features/safety-fire_detection/configs/06_training_yolox.yaml \
  --override model.impl=official augmentation.normalize=false training.val_full_interval=0

# RT-DETRv2
uv run core/p06_training/train.py --config features/safety-fire_detection/configs/06_training_rtdetr.yaml

# D-FINE-S
uv run core/p06_training/train.py --config features/safety-fire_detection/configs/06_training_dfine.yaml
```

## Key Files

```
configs/00_data_preparation.yaml   — data sources + class map
configs/05_data.yaml               — dataset paths + class names
configs/06_training_yolox.yaml     — YOLOX-M
configs/06_training_rtdetr.yaml    — RT-DETRv2-R18
configs/06_training_dfine.yaml     — D-FINE-S
code/benchmark.py                  — pretrained benchmark
eval/benchmark_results.json        — benchmark output
eval/benchmark_report.md           — benchmark summary
```

## Gotchas

- **D-FINE/RT-DETR require `amp: false`** — HF DETR decoder overflows in fp16, producing NaN `pred_boxes` that crash `generalized_box_iou` on first forward pass. Both configs already set this.
- **`ValPredictionLogger` class names** — reads `trainer._loaded_data_cfg` (the resolved `05_data.yaml`). Using `trainer._data_cfg` (the `data:` section of the training YAML) returns no `names` key and labels show as IDs only. Fixed in `callbacks.py`.
- **`gpu_augment: true` is mandatory** for all three arch configs — always verify it's present. DETR-family (D-FINE, RT-DETRv2) additionally require `augmentation.mosaic: false` since DETR does not support mosaic.
- **HPO commands** (parallel on two GPUs):
  ```bash
  # GPU 0 — YOLOX
  CUDA_VISIBLE_DEVICES=0 uv run core/p07_hpo/run_hpo.py \
    --config features/safety-fire_detection/configs/06_training_yolox.yaml \
    --hpo-config configs/_shared/08_hpo_yolox.yaml \
    --override data.subset.train=0.05 data.subset.val=0.10 data.batch_size=32 \
      augmentation.mosaic=false \
      training.data_viz.enabled=false training.aug_viz.enabled=false training.val_viz.enabled=false
  # GPU 1 — D-FINE (then RT-DETRv2 sequentially)
  CUDA_VISIBLE_DEVICES=1 uv run core/p07_hpo/run_hpo.py \
    --config features/safety-fire_detection/configs/06_training_dfine.yaml \
    --hpo-config configs/_shared/08_hpo_detr.yaml \
    --override data.subset.train=0.05 data.subset.val=0.10 data.batch_size=32 \
      training.data_viz.enabled=false training.aug_viz.enabled=false training.val_viz.enabled=false
  ```

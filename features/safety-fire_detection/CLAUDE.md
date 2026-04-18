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

- [x] `00_data_preparation.yaml` — sources locked, class map verified
- [x] `p00_data_prep` — 17,373 imgs, DATASET_REPORT
- [x] `p02_annotation_qa` — LS project 13
- [x] `code/benchmark.py` — pretrained benchmark complete
- [x] `06_training.yaml` — gpu_augment enabled; data_viz/aug_viz callbacks wired; subset config added
- [x] Arch comparison configs — `06_training_dfine.yaml`, `06_training_rtdetr.yaml`, `06_training_yolox.yaml` created for arch selection
- [ ] `p06_training` — freeze backbone 5 epochs, then full fine-tune
- [ ] `p08_evaluation` — evaluate on test split
- [ ] `p09_export` — ONNX export
- [ ] `release/` — `utils/release.py`

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
# Train with default config
uv run core/p06_training/train.py --config features/safety-fire_detection/configs/06_training.yaml

# Arch comparison (run each, pick best mAP50, then full train)
uv run core/p06_training/train.py --config features/safety-fire_detection/configs/06_training_yolox.yaml
uv run core/p06_training/train.py --config features/safety-fire_detection/configs/06_training_dfine.yaml
uv run core/p06_training/train.py --config features/safety-fire_detection/configs/06_training_rtdetr.yaml
```

## Key Files

```
configs/00_data_preparation.yaml  — data sources + class map
configs/05_data.yaml              — dataset paths + class names
configs/06_training.yaml          — (to create) training config
code/benchmark.py                 — pretrained benchmark
eval/benchmark_results.json       — benchmark output
eval/benchmark_report.md          — benchmark summary
```

## Training Config Template

```yaml
# features/safety-fire_detection/configs/06_training.yaml
model:
  arch: yolox-s          # start with s; upgrade to m if mAP plateaus
  num_classes: 2
  pretrained: ../../../pretrained/yolox_m.pth   # SalahALHaismawi is Ultralytics DetectionModel (YOLO v2.6) — incompatible with custom YOLOX; use COCO yolox_m.pth backbone

training:
  epochs: 100
  freeze_backbone_epochs: 5
  lr: 0.001
  lr_backbone: 0.0001
  gpu_augment: true

augmentation:
  mosaic: true
  contrast: 0.4          # vectorized luma-contrast on GPU (fire/smoke are low-contrast)
  hsv_h: 0.015
  hsv_s: 0.5
  hsv_v: 0.4
  fliplr: 0.5
  scale: [0.5, 1.5]
  degrees: 10.0
  translate: 0.1
  shear: 2.0

data:
  batch_size: 16
  num_workers: 8
  prefetch_factor: 4     # Mosaic does 4 disk reads/sample — deep prefetch critical
  pin_memory: true
```

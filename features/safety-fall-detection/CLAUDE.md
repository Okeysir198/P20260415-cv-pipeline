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

- [x] `00_data_preparation.yaml` — sources locked
- [x] `p00_data_prep` — 12,402 imgs, DATASET_REPORT
- [x] `p02_annotation_qa` — LS project 16
- [x] `code/benchmark.py` — pretrained benchmark complete
- [ ] `06_training.yaml` — use yolov11_fall_melihuzunoglu as starting weights
- [ ] `p06_training` — freeze backbone 5 epochs, then full fine-tune
- [ ] `p08_evaluation` — evaluate on test split
- [ ] `p09_export` — ONNX export
- [ ] `release/` — `utils/release.py`

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
# features/safety-fall-detection/configs/06_training.yaml
model:
  arch: yolox-s
  num_classes: 2
  pretrained: true
  pretrained_path: pretrained/safety-fall-detection/yolov11_fall_melihuzunoglu.pt

training:
  epochs: 100
  freeze_backbone_epochs: 5
  lr: 0.001
  lr_backbone: 0.0001
  batch_size: 16
```

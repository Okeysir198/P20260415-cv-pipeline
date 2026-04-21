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
- [x] Arch-specific training configs created — `06_training_{yolox,rtdetr,dfine}.yaml`
- [ ] Arch comparison on 10% data (see `safety-fire_detection` Iteration 5 for recipe); start with `06_training_yolox.yaml` (YOLOX-M is the small-data default per features/CLAUDE.md)
- [ ] `p06_training` — full fine-tune on winning arch; `yolov11_fall_melihuzunoglu.pt` is a YOLOv11 checkpoint (not directly loadable into YOLOX), use for reference only
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

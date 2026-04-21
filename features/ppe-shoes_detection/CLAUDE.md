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

- [x] `00_data_preparation.yaml` — sources locked
- [x] `p00_data_prep` — 37,026 imgs, DATASET_REPORT ✅
- [x] `p02_annotation_qa` — LS project 15
- [x] `code/benchmark.py` — pretrained benchmark complete
- [x] Arch-specific training configs created — `06_training_{yolox,rtdetr,dfine}.yaml`
- [ ] Arch comparison on 10% data; start with `06_training_yolox.yaml` (YOLOX-M, COCO backbone). Person class at 2.4% — foot-centric data naturally lacks full-body bboxes.
- [ ] `p06_training` — full fine-tune on winning arch. Consider two-stage (person detector → foot crop → shoe classifier) for edge deployment.
- [ ] `p08_evaluation` — evaluate on test split
- [ ] `p09_export` — ONNX export
- [ ] `release/` — `utils/release.py`

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

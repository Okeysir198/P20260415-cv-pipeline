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
- [ ] `06_training.yaml` — start from COCO YOLOX-S/M backbone
- [ ] `p06_training` — full fine-tune on foot classes
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
configs/06_training.yaml          — (to create) training config
code/benchmark.py                 — pretrained benchmark
eval/benchmark_results.json       — benchmark output
eval/benchmark_report.md          — benchmark summary
```

## Training Config Template

```yaml
# features/ppe-shoes_detection/configs/06_training.yaml
model:
  arch: yolox-s
  num_classes: 3
  pretrained: true       # start from COCO YOLOX-S backbone

training:
  epochs: 100
  freeze_backbone_epochs: 5
  lr: 0.001
  lr_backbone: 0.0001
  batch_size: 16
```

## Notes

- `person` is at 2.4% — foot-centric datasets naturally lack full-body bboxes; acceptable imbalance
- Consider two-stage approach for edge deployment: person detector → foot crop → shoe classifier

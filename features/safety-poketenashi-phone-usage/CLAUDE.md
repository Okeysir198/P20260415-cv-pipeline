# safety-poketenashi-phone-usage

**Type:** Detection sub-model | **Training:** 🎯 Fine-tune required (phone_usage action class not in COCO)

## Overview

Detects the act of using a phone while walking — a behavioral action class, not a physical object. COCO has `cell phone` as an object but not `phone_usage` as a walking behavior. Full fine-tuning required.

## Classes

| ID | Name | Train split % |
|---|---|---|
| 0 | person | 5.4% |
| 1 | phone_usage | 94.6% |

## Dataset

- **Images:** 22,975 (val: ~2,635)
- **QA:** 90.6% good / 5.4% bad → ⚠️ borderline ACCEPT
  - SAM3 struggles with action-class semantics on small phone bboxes
  - Re-label only if post-training mAP is low
- **Label Studio:** project id=17
- **Training ready:** `dataset_store/training_ready/safety_poketenashi_phone_usage/`

## Pipeline Checklist

- [x] `00_data_preparation.yaml` — sources locked
- [x] `p00_data_prep` — 22,975 imgs, DATASET_REPORT ✅
- [x] `p02_annotation_qa` — LS project 17 (borderline)
- [x] `code/benchmark.py` — COCO baseline benchmark complete
- [ ] `06_training.yaml` — start from COCO YOLOX-S backbone
- [ ] `p06_training` — full fine-tune (no domain-specific pretrained backbone available)
- [ ] `p08_evaluation` — evaluate on test split
- [ ] `p09_export` — ONNX export
- [ ] `release/` — `utils/release.py`

## Benchmark Results — val split (2026-04-17, 2635 images, CPU inference)

COCO pretrained baselines — `phone_usage` class has no COCO equivalent; only `person` class evaluated:

| Model | person mAP@50 | person mAP@50:95 | ms/img | Status |
|---|---|---|---|---|
| yolox_s (COCO 80-class) | 0.000 | 0.000 | 72.2 | ok |
| yolox_m (COCO 80-class) | 0.000 | 0.000 | 160.2 | ok |

No .pt files found in pretrained dir for this feature.

**Conclusion:** Zero mAP expected and confirmed — `phone_usage` is not a COCO class. No transfer possible without fine-tuning. Start from COCO YOLOX-S/M backbone weights.

Full results: `eval/benchmark_results.json` | `eval/benchmark_report.md`

## Key Files

```
configs/00_data_preparation.yaml  — data sources + class map
configs/05_data.yaml              — dataset paths + class names
configs/06_training.yaml          — (to create) training config
code/benchmark.py                 — COCO baseline benchmark
eval/benchmark_results.json       — benchmark output
eval/benchmark_report.md          — benchmark summary
```

## Training Config Template

```yaml
# features/safety-poketenashi-phone-usage/configs/06_training.yaml
model:
  arch: yolox-s
  num_classes: 2
  pretrained: true       # COCO YOLOX-S — no domain-specific backbone available

training:
  epochs: 100
  freeze_backbone_epochs: 5
  lr: 0.001
  lr_backbone: 0.0001
  batch_size: 16
```

## Notes

- `phone_usage` bbox is often small (phone area only) — consider multi-scale training or mosaic augmentation
- If post-training mAP < 0.3, revisit Label Studio project 17 for re-labeling of borderline QA images
- This sub-model feeds results into `safety-poketenashi/code/orchestrator.py`

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

- [x] `00_data_preparation.yaml` — sources locked; 4 classes confirmed
- [x] `p00_data_prep` — 22,323 imgs, DATASET_REPORT ✅
- [x] `p02_annotation_qa` — LS project 14
- [x] `code/benchmark.py` — pretrained benchmark complete
- [ ] `06_training.yaml` — use melihuzunoglu_yolov11_ppe as starting weights
- [ ] `p06_training` — freeze backbone 5 epochs, then full fine-tune
- [ ] `p08_evaluation` — evaluate on test split
- [ ] `p09_export` — export HudatersU-style ONNX for fast serving
- [ ] `release/` — `utils/release.py`

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
configs/06_training.yaml          — (to create) training config
code/benchmark.py                 — pretrained benchmark
eval/benchmark_results.json       — benchmark output
eval/benchmark_report.md          — benchmark summary
```

## Training Config Template

```yaml
# features/ppe-helmet_detection/configs/06_training.yaml
model:
  arch: yolox-m          # larger model needed for 4-class fine-grained detection
  num_classes: 4
  pretrained: true
  pretrained_path: pretrained/ppe-helmet_detection/melihuzunoglu_yolov11_ppe.pt

training:
  epochs: 100
  freeze_backbone_epochs: 5
  lr: 0.001
  lr_backbone: 0.0001
  batch_size: 16
```

## Notes

- `head_with_nitto_hat` is at 1.6% — may need augmentation or site-collected images to prevent under-detection
- CUDA OOM errors on large .pt models are due to other PIDs holding GPU memory, not a code issue

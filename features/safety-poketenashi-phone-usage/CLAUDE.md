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
2. `val mAP@0.5` > **0.05** (no pretrained backbone — any non-trivial mAP is real learning)
3. Confusion matrix diagonal > 0.5 for each class (no class collapse — **highest risk feature** due to 94.6/5.4 imbalance)
4. `error_breakdown.png` shows FP mix ≠ 100% background

### Commands

```bash
# YOLOX-M (config defaults: impl=official, library=torchvision, mosaic=true,
# mixup=false, normalize=false, lr=0.0025, val_full_interval=0 — Phase B only
# overrides epochs→30 + subset)
CUDA_VISIBLE_DEVICES=0 .venv-yolox-official/bin/python core/p06_training/train.py \
  --config features/safety-poketenashi-phone-usage/configs/06_training_yolox.yaml \
  --override training.epochs=30 \
    data.subset.train=0.2 data.subset.val=0.2 \
    training.data_viz.enabled=false training.aug_viz.enabled=false training.val_viz.enabled=false

# RT-DETRv2-R50 (config defaults: arch=r50, pretrained=r50vd, bf16=true, amp=false,
# mosaic=false, warmup_steps=300, lr=1e-4 — Phase B overrides lr→5e-5 + bs→8 + epochs→30)
CUDA_VISIBLE_DEVICES=0 uv run core/p06_training/train.py \
  --config features/safety-poketenashi-phone-usage/configs/06_training_rtdetr.yaml \
  --override training.lr=5e-5 training.epochs=30 \
    data.batch_size=8 data.subset.train=0.2 data.subset.val=0.2 \
    training.data_viz.enabled=false training.aug_viz.enabled=false training.val_viz.enabled=false

# D-FINE-M (config defaults match the reference recipe; Phase B only overrides epochs→30.
# HIGHEST class-collapse risk on this feature due to 94.6/5.4 imbalance — if `person` AP=0
# while `phone_usage` AP>0, mark arch FAIL regardless of mAP.)
CUDA_VISIBLE_DEVICES=0 uv run core/p06_training/train.py \
  --config features/safety-poketenashi-phone-usage/configs/06_training_dfine.yaml \
  --override training.epochs=30 \
    data.subset.train=0.2 data.subset.val=0.2 \
    training.data_viz.enabled=false training.aug_viz.enabled=false training.val_viz.enabled=false
```

### Error analysis (run after each training)

```bash
CUDA_VISIBLE_DEVICES=0 uv run core/p08_evaluation/evaluate.py \
  --model features/safety-poketenashi-phone-usage/runs/<ts>/best.pth \
  --config features/safety-poketenashi-phone-usage/configs/05_data.yaml \
  --split test --conf 0.3 --iou 0.5
```

Outputs: `metrics.json`, `confusion_matrix.png`, per-class PR curves, `error_breakdown.png`, `size_recall.png`, `optimal_thresholds.json`.

### OOM notes
- 22,975 images × 20% = ~4,595 train; 30 epochs at bs=8/16 → ~25–50 min/arch on RTX 5090.
- Pre-flight: `nvidia-smi --query-gpu=memory.free --format=csv` → need ≥20 GB free.
- Kill if first-epoch VRAM > 24 GB or `train/loss` NaNs → halve `data.batch_size` and retry.
- **bf16 policy**: YOLOX `amp=true`; RT-DETRv2 `bf16=true amp=false`; D-FINE `bf16=false amp=false`.
- **Never launch two trainings on the same GPU** — system hang risk.
- Phone-usage-specific: **extreme 94.6/5.4 class imbalance** — D-FINE is the highest-risk arch (class-collapse precedent on fire). If `person` AP = 0 while `phone_usage` AP > 0, that's class collapse — mark arch as FAIL regardless of overall mAP. QA was borderline (5.4% bad) — small phone bboxes may confuse the matcher.

### Results table (fill after each run)

| Arch | epochs | best val mAP@0.5 | AP `person` | AP `phone_usage` | Class collapse? | PASS? | runs/ dir | eval/ dir |
|---|---|---|---|---|---|---|---|---|
| YOLOX-M | | | | | | | | |
| RT-DETRv2-R50 | | | | | | | | |
| D-FINE-M | | | | | | | | |

### Error analysis summary (per arch, fill after p08)
- Dominant FP type (background / class confusion / localization / duplicate)
- Class-collapse check (is `person` AP ≈ 0?)
- Size bucket where recall collapses (phone bboxes are tiny — expect tiny-bucket recall < 0.3)
- Top 3 failure cases

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
configs/06_training_yolox.yaml    — YOLOX-M (recommended starting arch)
configs/06_training_rtdetr.yaml   — RT-DETRv2-R50 (HF backend, torchvision aug)
configs/06_training_dfine.yaml    — D-FINE-M (HF backend, torchvision aug, bf16=false)
code/benchmark.py                 — COCO baseline benchmark
eval/benchmark_results.json       — benchmark output
eval/benchmark_report.md          — benchmark summary
```

## Training Commands

```bash
# YOLOX-M (recommended starting arch; COCO backbone)
uv run core/p06_training/train.py --config features/safety-poketenashi-phone-usage/configs/06_training_yolox.yaml
```

## Notes

- `phone_usage` bbox is often small (phone area only) — consider multi-scale training or mosaic augmentation
- If post-training mAP < 0.3, revisit Label Studio project 17 for re-labeling of borderline QA images
- This sub-model feeds results into `safety-poketenashi/code/orchestrator.py`

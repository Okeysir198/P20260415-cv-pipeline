# safety-fire_detection

**Type:** Detection | **Training:** Fine-tune required (fire/smoke not in COCO 80)

## 🔥 Status (2026-05-04) — must-read before retraining

### Load-bearing facts

1. **Dataset is per-source-temporal split (active 2026-05-04)** — `dataset_store/training_ready/fire_detection/` produced by `core/p00_data_prep/run.py` with `dedup.split_strategy: per_source_with_temporal` at `hamming_thresh: 6`, ratios `[0.70, 0.15, 0.15]`. Final counts:

   | Split | Imgs | Fire boxes | Smoke boxes | Fire % | Boxes/img |
   |---|---:|---:|---:|---:|---:|
   | Train | 12,161 | 13,717 | 11,232 | 55.0% | 2.05 |
   | Val | **2,606** | 2,567 | 2,525 | 50.4% | 1.95 |
   | Test | **2,606** | 2,615 | 2,557 | 50.6% | 1.98 |

   Cross-split leakage at hamming ≤ 6: **0**. Per-source 70/15/15 within ±0.2pp across all 3 raw sources (d_fire, zenodo_indoor_fire, industrial_hazards). Video pHash groups (AoF aerial sequences, industrial Chimney smokehazard sequences) are split temporally with a 5% buffer gap to test temporal generalization rather than scene memorization.

2. **ImageNet normalization HURTS this dataset** — fire imagery mean=[0.392, 0.360, 0.340] is ~10% darker than ImageNet's [0.485, 0.456, 0.406]. `tensor_prep.normalize: false` lifted test mAP@0.5 from 0.320 → 0.409 (+0.089). All configs default to `normalize: false`.

3. **HF default loss params are RT-DETRv2-specific, NOT vanilla DETR** (footgun, fixed 2026-05-03). Real HF `RTDetrV2Config` defaults:
   - `eos_coefficient: 0.0001` (not 0.1 — that's vanilla DETR / cross-entropy world)
   - `focal_loss_alpha: 0.75` (not 0.25)
   - `focal_loss_gamma: 2.0`
   - `box_noise_scale: 1.0`
   - `label_noise_ratio: 0.5`
   - `num_queries: 300`, `num_denoising: 100`

   Our configs deviate only on `num_queries: 30` (fire p99 = 9 boxes/img, 300 wastes capacity). Everything else now matches HF defaults.

4. **Per-source LR split is paper-mandated for R50, not R18**:
   - R50: head `lr=5e-5`, backbone `lr=1e-5` (5× lower) — paper Table 1 × 0.5
   - R18: single `lr=5e-5` (no split — paper Table 1 row S × 0.5)
   - D-FINE: flat `lr=5e-5` (no backbone split) — reference baseline recipe (see below)

5. **Long DETR fine-tunes overfit + miscalibrate** — train loss keeps falling but val mAP peaks at ep 3-15 then degrades. Late-epoch scores collapse into [0, 0.05] making the model effectively undeployable (no predictions clear default 0.05 confidence threshold). Cause: focal loss with `α=0.75, γ=2` rewards "be conservative when unsure" once positives are easy to recognize. **Stop training at peak val mAP** (or use temperature scaling at inference).

6. **`val_loss` is misleading as best-checkpoint selector for DETR** — declines smoothly even when mAP plateaus, picks late over-regularized checkpoint. Use `metric: eval_map_50` (or `eval_map`). Never `eval_loss` for DETR.

7. **F1-optimal inference threshold ≈ 0.075 for fire, ≈ 0.05 for smoke** — DETR sigmoid scores cap around 0.2. Default 0.05 in `10_inference.yaml` floods FPs. Verify post-training with `scripts/threshold_sweep.py --run <ckpt-dir>`.

### What did NOT help (don't re-test on clean dataset)

- 1280×1280 resolution — GPU memory 27 GB, ep1 mAP near-zero. R50 uses 1024², R18 uses 960². Don't go higher.
- `eval_loss` as metric — pick val-best at ep3 (model barely trained), see Phase D fire runs.
- box_noise_scale=2.0 + label_noise_ratio=0.1 ("learn through label noise") — neutral on clean dataset; revert to HF defaults.

### What MIGHT help (untested on clean dataset)

- Per-class loss weighting (smoke 2× fire) — needs custom loss subclass, ~30 LOC.
- Temperature scaling at inference (`scripts/fit_rtdetr_temperature.py`) — fights calibration drift post-hoc.
- Shorter training (epochs 80 → 30) — both R18 and R50 peak before ep 15 on the clean dataset.
- Smoothed best-ckpt selector (rolling-3-epoch mAP mean) — current single-epoch mAP@50 has ±0.02 variance even on val=2606.

## Overview

Detects fire and smoke in images/video. Both classes are absent from COCO — pretrained models show low mAP (best: 0.153) confirming fine-tuning is mandatory.

## Classes

| ID | Name |
|---|---|
| 0 | fire |
| 1 | smoke |

Most images contain both classes. Per-class fire/smoke ratio is balanced within ±5pp across train/val/test (55.0 / 50.4 / 50.6 % fire).

## Dataset

- **Images:** ~17,400 raw → train=12,161 / val=2,606 / test=2,606 after per-source-temporal split
- **QA:** 95.1% good / 1.1% bad → ACCEPT
- **Label Studio:** project id=13
- **Re-prep:** `uv run core/p00_data_prep/run.py --config features/safety-fire_detection/configs/00_data_preparation.yaml` (uses `dedup.split_strategy: per_source_with_temporal` from the YAML)
- **Verify**: `dataset_store/training_ready/fire_detection/DATASET_REPORT.md` shows source/class breakdown after re-prep

## Pipeline Checklist

- [x] `00_data_preparation.yaml` (per_source_with_temporal split — 2026-05-04)
- [x] `p00_data_prep`, `p02_annotation_qa`, `code/benchmark.py`
- [x] Arch configs authored — 7 configs (rtdetr_r18/r50, dfine_n/s/m, yolox_s/m), D-FINE aligned to reference baseline, RT-DETR to paper×0.5
- [ ] **Phase D — full-data on clean per-source-temporal split** — PENDING
- [ ] `p08_evaluation` — full test split on best clean-dataset checkpoint
- [ ] `p09_export` — ONNX export
- [ ] `release/` — `utils/release.py`

## Best Results (PRIOR DATASET — reference only)

| Run | Dataset | Test mAP@50 | Notes |
|---|---|---|---|
| RT-DETR R50 (val-best ep14) | leaked-val (2026-05-01) | 0.576 | val was inflated 2× by leakage |
| D-FINE-S ep32 ckpt | leaked-val (2026-05-02) | 0.648 | smoke AP 0.293 (best smoke seen) |
| RT-DETR R50 (val-best ep3) | clean v1 (2026-05-03, max_per_group_eval=50) | 0.543 | val=593, test=765, single seed |
| RT-DETR R18 (val-best ep14) | clean v1 (2026-05-03) | 0.446 | val=593, test=765 |

These all predate the per-source-temporal split. Phase D will produce the new authoritative numbers.

## Current Results (2026-05-10, per-source-temporal dataset)

| Run | Recipe | best val mAP@50 | test mAP@50 | fire test AP | smoke test AP | small mAP |
|---|---|---:|---:|---:|---:|---:|
| RT-DETR R18 | 960² + optimized loss recipe | 0.565 (ep14) | **0.555** | 0.249 | 0.308 | 0.112 |
| RT-DETR R50 | same + R50 backbone (1:5 LR split) | 0.622 (ep3, ES@ep18) | **0.607** | 0.252 | 0.333 | 0.148 |
| D-FINE-N | 640² + reference baseline (EMA on) | 0.414 (ep28) | **0.412** | 0.158 | 0.242 | 0.060 |
| D-FINE-S | 640² + reference baseline (EMA off) | 0.403 (ep15, ES@ep23) | **0.402** | 0.162 | 0.249 | 0.084 |

R50 is the current best across all metrics. D-FINE variants underperform RT-DETRv2 by ~0.15–0.20 mAP@50 on this dataset; D-FINE-N (4M params, EMA) edges D-FINE-S (16M params, no EMA) by +0.01 mAP@50 but D-FINE-S has better small-object recall (+0.024 small mAP).

### Optimized loss-recipe deltas vs HF defaults (committed in `06_training_rtdetr_{r18,r50}.yaml`)

| Field | HF default | optimized | Why |
|---|---:|---:|---|
| `lr` | 1e-4 | 5e-5 | 1e-4 collapsed mAP from 0.445→0.19 by ep14 (overshoot past optimum) |
| `num_queries` | 300 | 30 | 100 surplus queries fired on background; 30 = 3.3× p99 |
| `eos_coefficient` | 0.0001 | 0.1 | Penalises orphan-query background FPs (textbook DETR fix; prior docs warned against this but observed behaviour says otherwise) |
| `focal_loss_alpha` | 0.75 | 0.5 | Removes positive-class bias driving score collapse |
| `focal_loss_gamma` | 2.0 | 1.5 | Smoother gradients on hard examples → less oscillation |
| `box_noise_scale` | 1.0 | 1.0 | (was experimentally bumped to 1.5; reverted) |
| `label_noise_ratio` | 0.5 | 0.3 | Over-regularising at this dataset size |
| `input_size` | 640 | 960 | Small-tier (15.4% of GTs) recall lifted 0.30 → 0.33 |

### CPU-RAM leak fixed during this work

RT-DETR @ 960² OOMs by epoch 3 on a 128 GB box without the eval-mode `ModelOutput` strip in `core/p06_models/hf_model.py::HFDetectionModel.forward`. See `core/p06_models/CLAUDE.md` for the invariant + `core/p06_training/CLAUDE.md` for the eval/CPU-RAM pair fix. Required even though `eval_accumulation_steps=4` is already set.

## Training (Phase D — clean per-source-temporal dataset)

Recommended baseline: RT-DETR R50, paper-aligned, ~25 epochs.

```bash
# R18 + R50 simultaneously on 2 GPUs (one Bash background task per GPU so
# stdout shows in the Claude Code UI)
CUDA_VISIBLE_DEVICES=0 uv run core/p06_training/train.py \
  --config features/safety-fire_detection/configs/06_training_rtdetr_r18.yaml

CUDA_VISIBLE_DEVICES=1 uv run core/p06_training/train.py \
  --config features/safety-fire_detection/configs/06_training_rtdetr_r50.yaml
```

Run dirs land in `features/safety-fire_detection/runs/<arch>_<ts>/`.

## Config Summary (2026-05-10)

| Config | Arch | Params | Backend | epochs | lr | lr (backbone) | scheduler | EMA | input | metric |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `06_training_dfine_n.yaml` | dfine-n | 4M | hf | 30 | 5e-5 | — | linear | true | 640 | eval_map |
| `06_training_dfine_m.yaml` | dfine-m | 31M | hf | 30 | 5e-5 | — | linear | true | 640 | eval_map |
| `06_training_rtdetr_r18.yaml` | rtdetr-r18 | 20M | hf | 30 | 5e-5 | — | cosine_with_min_lr | true | 960 | eval_map_50 |
| `06_training_rtdetr_r50.yaml` | rtdetr-r50 | 42M | hf | 30 | 5e-5 | 1e-5 | cosine_with_min_lr | true | 1024 | eval_map_50 |
| `06_training_yolox_s.yaml` | yolox-s | 9M | pytorch | 80 | 1e-3 | — | cosine | false | 640 | val/mAP50 |
| `06_training_yolox_m.yaml` | yolox-m | 25M | pytorch | 80 | 1e-3 | — | cosine | false | 640 | val/mAP50 |

### Recipe sources

**D-FINE** — reference baseline from `notebooks/detr_finetune_reference/our_dfine_torchvision/` (CPPE-5 ablation, best-performing norm=false configs). Proven: dfine-n=0.710, dfine-s=0.603, dfine-m=0.431 on CPPE-5. Flat LR, no backbone split, wd=0, linear schedule, 30 ep. Previous paper-aligned recipe (lr=2.5e-4 + 1:20 LR split + constant schedule) produced worse results on fire data (dfine-n peaked at 0.365 vs RT-DETR R50's 0.607).

**RT-DETR** — paper Table 1 × 0.5 safety factor. Loss params deviate from HF defaults (see v2 deltas table below).

**YOLOX** — Megvii recipe. Pytorch backend, `.venv-yolox-official/` required.

Common: `normalize=false`, `score_threshold=0.0` (canonical mAP), `seed=42`, `save_interval=1`.

D-FINE invariants: `bf16=false` (mandatory — DFL stalls under bf16), `weight_decay=0`, EMA decay 0.9999 / warmup 1000.
RT-DETRv2 invariants: `bf16=true`. `num_queries=30` (r18) / `50` (r50) — fire-density justified (paper default 300).
YOLOX invariants: pytorch backend, sgd, `mosaic=true`.

## Evaluation utilities

- **Evaluate any HF Trainer checkpoint on test split** — `scripts/eval_hf_checkpoint.py` (added 2026-05-04). Uses `AutoModelForObjectDetection.from_pretrained(<checkpoint-N>)` + `strip_hf_prefix` for the wrapper-prefix issue. Useful for comparing best.pt vs last.pt or comparing across runs.
  ```bash
  uv run python scripts/eval_hf_checkpoint.py \
    --ckpt features/safety-fire_detection/runs/rtdetr_r50_<ts>/checkpoint-2523 \
    --data-config features/safety-fire_detection/configs/05_data.yaml --split test
  ```
- **Threshold sweep / F1-optimal operating point** — `scripts/threshold_sweep.py --run <ckpt-dir> --split val` finds per-class F1-optimal threshold. Use BEFORE setting `model.conf` in `10_inference.yaml`.
- **Cross-checkpoint test eval** — root `pytorch_model.bin` is bit-identical to `checkpoint-<best_step>/pytorch_model.bin` (verified via md5sum 2026-05-04 — `_load_best_model_at_end` works correctly despite the wrapper prefix).

## Key Files

```
configs/00_data_preparation.yaml      — per_source_with_temporal split, hamming=6, 70/15/15
configs/05_data.yaml                  — dataset path → fire_detection/, names: {0: fire, 1: smoke}
configs/06_training_dfine_n.yaml      — D-FINE-n (4M, hf, reference baseline, EMA on)
configs/06_training_dfine_s.yaml      — D-FINE-s (16M, hf, reference baseline, EMA off)
configs/06_training_dfine_m.yaml      — D-FINE-m (31M, hf, reference baseline, EMA on)
configs/06_training_rtdetr_r50.yaml   — RT-DETRv2-R50 (42M, 1:5 LR split, 1024²)
configs/06_training_rtdetr_r18.yaml   — RT-DETRv2-R18 (20M, single LR, 960²)
configs/06_training_yolox_s.yaml      — YOLOX-S (9M, pytorch, sgd+mosaic)
configs/06_training_yolox_m.yaml      — YOLOX-M (25M)
configs/10_inference.yaml             — per-class deployment thresholds
runs/<arch>_<ts>/                     — run artifacts
runs/_logs/                           — training stdout logs
```

## Gotchas

- **D-FINE/RT-DETR require `amp: false`** — fp16 overflows decoder, NaN pred_boxes on first forward.
- **D-FINE additionally requires `bf16: false`** — DFL distribution-focused loss stalls val mAP at ~0.15 under bf16. RT-DETRv2 is bf16-neutral.
- **D-FINE wrong pretrained = bad convergence** — `dfine_m_coco` weights in `dfine-n/s` architecture cause 52+ mismatched-layer reinits. Each variant must use its matching `ustc-community/dfine_<n|s|m>_coco`.
- **Per-source LR is paper-correct, missing it is a footgun** — RT-DETRv2-R50 needs `lr_backbone=1e-5` (10× lower than head). D-FINE needs `lr_backbone=1.25e-5` (20× lower). R18 (RT-DETR S row) uses single LR — no backbone split needed.
- **HF defaults footgun (eos / focal_alpha)** — `eos_coefficient=0.0001` and `focal_loss_alpha=0.75` are RT-DETRv2 defaults. The "vanilla DETR" values (0.1, 0.25) are for cross-entropy world, NOT focal-loss world. Using vanilla values increases FPs and overfits the no-object class.
- **`val_loss` as `metric` for best-ckpt selection picks ep3 model on DETR** — train loss keeps falling, val_loss looks smoother than mAP, but the ep3 model has barely been trained. Use `metric: eval_map_50` (or `eval_map` for less noise).
- **mAP@50 has ±0.02 variance even on val=2606** — single-epoch peak is noisy. For the final release checkpoint, smoothed selection or visual inspection of `val_predictions/epochs/*.png` beats single-metric automation.
- **HF `load_best_model_at_end` works correctly** (verified via md5sum 2026-05-04). Root `pytorch_model.bin` IS the val-best checkpoint, bit-identical. Earlier doc claiming silent failure was incorrect.
- **`EarlyStoppingCallback` warning during final test eval** — fires `eval_map_50` not found because test eval uses `test_*` prefix. `train_with_hf` strips the callback before test eval; no action needed if you see legacy logs.
- **DETR sigmoid scores cap ~0.2** — set `score_threshold: 0.0` in TRAINING configs (canonical mAP); set per-class threshold in `10_inference.yaml` from `scripts/threshold_sweep.py` output.
- **DETR calibration drift over long fine-tunes** — late epochs become underconfident; scores collapse into [0, 0.05] even on real detections. Stop early or apply temperature scaling.
- **Never launch two trainings on the same GPU** — system hang risk (confirmed 2026-05-01).
- **Launch training as a Bash background task** (`run_in_background: true`) — stdout streams into the Claude Code UI so the user can watch progress live. Avoid detached `nohup` for in-session runs.

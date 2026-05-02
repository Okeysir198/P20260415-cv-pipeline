# safety-fire_detection

**Type:** Detection | **Training:** Fine-tune required (fire/smoke not in COCO 80)

## 🔥 Findings (2026-05-01 / 05-02) — must-read before retraining

### Load-bearing facts

1. **Dataset double-deduped (2026-05-02, active)** — `fire_detection` at `dataset_store/training_ready/fire_detection/` is now the authoritative dataset. Two-pass cleanup:
   - Pass 1 (2026-04-30, hamming ≤ 4): eliminated 74,707 cross-split pairs from Roboflow split-after-augmentation leakage.
   - Pass 2 (2026-05-02, hamming ≤ 6): eliminated residual 584 train↔test + 365 train↔val pairs. Additionally applied `--max-per-group-eval 200` to remove within-val redundancy (101k near-duplicate pairs from video sequences → 21k). Cross-split leakage = **0**.
   - Final counts: **train=13,428 | val=887 | test=1,858**. Val shrank from 1,970 → 887 (one representative per video-sequence group, max 200).

2. **ImageNet normalization HURTS this dataset** — fire imagery mean=[0.392, 0.360, 0.340] is ~10% darker than ImageNet's [0.485, 0.456, 0.406]. Setting `tensor_prep.normalize: false` lifted test mAP@0.5 from 0.320 → 0.409 (+0.089). All configs default to `normalize: false`.

3. **D-FINE-S is the current best model** (2026-05-02) — test mAP@50 = **0.648** (ep32 checkpoint from old leaked dataset run). Better than RT-DETR R50 (0.576) despite having half the params (~22M vs 42M). DFL convergence is slow (25–26 epochs of instability before breakthrough) but delivers superior smoke AP (0.293 vs 0.221).

4. **F1-optimal inference threshold = 0.075** — DETR sigmoid scores cap ~0.2, so default 0.05 floods FPs. Updated in `10_inference.yaml`.

5. **D-FINE-S pretrained must be `dfine_s_coco`** — using `dfine_m_coco` (wrong size) causes partial reinit and worse convergence. Config now defaults to `ustc-community/dfine_s_coco`.

6. **DFL convergence pattern** — D-FINE-S oscillates wildly (mAP 0.04–0.33) for eps 1–24, then breaks through sharply at ep25–26 (mAP jumps to 0.52→0.66 in 2 epochs, loss drops 3.9→1.7). Do NOT stop early before ep25. `patience: 15` is fine given 40 epochs; `save_interval: 1` is mandatory to avoid losing the ep26 best on crash.

### What did NOT help (don't re-test)
- `box_noise_scale` sweep (0.5/1.0/1.5/2.0) — monotonically worse on test beyond 1.0. HF default 1.0 wins.
- Removing all aug — smoke val AP 0.18 → 0.12. Aug helps generalization.
- EMA + patience=20 alone (without normalize fix) — val 0.526, worse than no-normalize.
- 1280×1280 resolution — GPU memory 27 GB, ep1 mAP near-zero (feature map recalibration takes too many epochs to pay off at this dataset size). Stick to 640×640.
- YOLOX-M full data — peaked ep8 (0.385) then steadily declined. LR=0.0025 too hot; lr=0.001 is better but still plateaued. YOLOX struggles with smoke on this dataset.

### Smoke — still the binding constraint
Per-class on old val (no-normalize, threshold=0.075):
- fire:  TP=84  FP=37   FN=21  →  P=0.79  R=0.69  F1=0.73
- smoke: TP=98  FP=256  FN=79  →  **P=0.28  R=0.55  F1=0.43**

Model hallucinates smoke on grey/cloudy backgrounds (256 FPs, 232 at large-bbox scale). Fixes tried: `eos_coefficient=0.4` (added to RT-DETR config, wired through HF kwargs automatically). D-FINE-S naturally handles this better (test smoke AP 0.293 vs 0.221 for RT-DETR).

## Overview

Detects fire and smoke in images/video. Both classes are absent from COCO — pretrained models show low mAP (best: 0.153) confirming fine-tuning is mandatory.

## Classes

| ID | Name | Train split % (post-dedup) |
|---|---|---|
| 0 | fire  | ~54% (imgs) |
| 1 | smoke | ~80% (imgs) |

Most images contain BOTH classes. Val/test are NOT class-balanced — structural cost of group-aware splitting.

## Dataset

- **Images:** 17,373 total → train=13,428 | val=887 | test=1,858 (post double-dedup)
- **QA:** 95.1% good / 1.1% bad → ACCEPT
- **Label Studio:** project id=13
- **Training ready:** `dataset_store/training_ready/fire_detection/` (hamming ≤ 6, max-200/group-eval)
- **Dedup script:** `scripts/dedup_split.py --name <dataset> --thresh 6 --max-per-group-eval 200`

## Pipeline Checklist

- [x] `00_data_preparation.yaml`, `p00_data_prep`, `p02_annotation_qa`, `code/benchmark.py`
- [x] Arch configs authored — `06_training_{yolox,rtdetr,dfine}.yaml`
- [x] **Phase B — 20% smoke** — all 3 arches PASS on old dataset
- [x] **Phase C — full-data (old dataset, leaked)** — best: RT-DETR R50 val 0.684, D-FINE-S test 0.648
- [ ] **Phase D — full-data (clean dataset, 2026-05-02)** — PENDING: retrain D-FINE-S + RT-DETR R50 + RT-DETR R18
- [ ] `p08_evaluation` — full test split on best clean-dataset checkpoint
- [ ] `p09_export` — ONNX export
- [ ] `release/` — `utils/release.py`

## Best Results (old leaked dataset — reference only)

| Arch | Val mAP@50 | Test mAP@50 | Fire AP | Smoke AP | Run |
|------|-----------|------------|---------|---------|-----|
| **D-FINE-S** | 0.658 @ ep26 | **0.648** | 0.361 | **0.293** | `2026-05-02_061317` (ep32 ckpt) |
| RT-DETR R50 | 0.673 @ ep8 | 0.576 | 0.325 | 0.221 | `2026-05-01_120040` |
| RT-DETR R50 (warm-start) | 0.643 @ ep25 | 0.559 | 0.324 | 0.238 | `2026-05-01_201738` |
| RT-DETR R18 | 0.538 @ ep3 | 0.440 | — | — | `2026-05-02_031518` |
| YOLOX-M | 0.395 @ ep11 | 0.218 | 0.328 | 0.109 | `2026-05-01_165844` |

Note: these runs used the leaked dataset (val inflated ~2×). Clean-dataset results will differ.

## Training Commands (clean dataset, Phase D)

```bash
# D-FINE-S — GPU 0 (50 epochs, ~5–6 h, nohup to survive task wrapper)
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 uv run core/p06_training/train.py \
  --config features/safety-fire_detection/configs/06_training_dfine.yaml \
  > /tmp/fire_dfine_s.log 2>&1' &

# RT-DETRv2-R50 — GPU 1 (60 epochs, ~3 h)
nohup bash -c 'CUDA_VISIBLE_DEVICES=1 uv run core/p06_training/train.py \
  --config features/safety-fire_detection/configs/06_training_rtdetr.yaml \
  > /tmp/fire_rtdetr_r50.log 2>&1' &

# RT-DETRv2-R18 — GPU 1 after R50 completes (60 epochs, ~1.5 h)
nohup bash -c 'CUDA_VISIBLE_DEVICES=1 uv run core/p06_training/train.py \
  --config features/safety-fire_detection/configs/06_training_rtdetr.yaml \
  --override model.arch=rtdetr-r18 model.pretrained=PekingU/rtdetr_v2_r18vd \
  > /tmp/fire_rtdetr_r18.log 2>&1' &
```

**IMPORTANT**: Use `nohup ... &` (not background task) to prevent training from being killed on session timeout.

## Config Highlights (2026-05-02)

### 06_training_dfine.yaml
- `arch: dfine-s`, `pretrained: ustc-community/dfine_s_coco` — S not M
- `save_interval: 1` — mandatory; DFL best checkpoint (ep26) was lost on crash with save_interval=10
- `scheduler: constant_with_warmup` — official D-FINE recipe; do NOT change to cosine
- `bf16: false` — DFL stalls under bf16 (val collapses to ~0.15)
- `patience: 15` — safe given DFL breakthrough at ep25–26; epochs=40 gives 15 ep post-breakthrough

### 06_training_rtdetr.yaml
- `arch: rtdetr-r50`, `pretrained: PekingU/rtdetr_v2_r50vd`
- `epochs: 60`, `scheduler: cosine`, `patience: 20`
- `eos_coefficient: 0.4` — cuts large-bbox smoke FPs (wired through HF kwargs automatically)
- `num_denoising: 100` — 2× num_queries (training-only, no inference cost)
- `normalize: false` — fire dataset off-distribution from ImageNet

### For RT-DETR R18
- Same config as R50, override: `model.arch=rtdetr-r18 model.pretrained=PekingU/rtdetr_v2_r18vd`
- Faster per epoch (~1.5× R50) but ~13 pp lower val mAP on old dataset

## Key Files

```
configs/05_data.yaml               — dataset path (→ fire_detection, double-deduped)
configs/06_training_yolox.yaml     — YOLOX-M (lr=0.001, warmup=3, full aug parity)
configs/06_training_rtdetr.yaml    — RT-DETRv2-R50 (cosine, 60ep, eos_coef=0.4)
configs/06_training_dfine.yaml     — D-FINE-S (constant_with_warmup, 40ep, save_interval=1)
runs/2026-05-02_data_preview/      — data_preview on clean dataset (duplicates chart = 0 leakage)
```

## Gotchas

- **D-FINE-S DFL breakthrough at ep25–26** — mAP oscillates 0.04–0.33 for first 24 epochs then jumps to 0.65+. This is expected; do not kill early. EarlyStopping with patience=15 + epochs=40 gives the model time to converge.
- **D-FINE-S `save_interval: 1` is non-negotiable** — the best checkpoint is at ep26; without frequent saves, a crash at post-training test eval loses it (observed 2026-05-02). HF Trainer's `checkpoint-N` dirs only save every `save_interval` epochs; with save_interval=10, ep26 gets overwritten by ep30.
- **D-FINE-S wrong pretrained = bad convergence** — `dfine_m_coco` weights loaded into `dfine-s` architecture cause 52+ size-mismatched layers to reinitialize randomly. Always verify `config_resolved.yaml::model.pretrained` matches the arch size after training starts.
- **HF checkpoint prefix stripping** — our `_DetectionTrainer._save` writes `hf_model.model.*` keys. To use a checkpoint as `model.pretrained` for warm-start, strip with: `utils.checkpoint.strip_hf_prefix` and save to a temp dir. See fire session 2026-05-01 for the pattern.
- **RT-DETR warm-start from previous run** — strip prefix from `pytorch_model.bin`, pass temp dir as `model.pretrained`, halve LR (`training.lr=0.00005`). Previous run's val mAP restored at ep1 (confirmed 0.657 on ep1 with ep8 weights as warm-start).
- **val=887 is intentional** — the clean dataset's val is smaller than before (was 1,970). The reduction removes video-sequence near-duplicates. Val mAP will likely be lower but more honest than the leaked-dataset val (which was inflated by factor ~2.2× from 101k within-val near-duplicate pairs).
- **D-FINE/RT-DETR require `amp: false`** — fp16 overflows both architectures.
- **Never launch two trainings on the same GPU** — system hang risk (confirmed 2026-05-01).
- **Use `nohup` not background tasks** — Claude task wrappers timeout and kill training. Use `nohup bash -c '... > log 2>&1' &` and monitor via `tail -f /tmp/fire_*.log`.

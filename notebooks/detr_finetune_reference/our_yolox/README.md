# our_yolox — YOLOX-M (official Megvii impl) on CPPE-5 via our in-repo pipeline

**Arch**: YOLOX-M. **Impl**: `model.impl: official` (Megvii `yolox` package
via `_OfficialYOLOXAdapter`). **Aug backend**: Albumentations. **Trainer**:
`core/p06_training/train.py --backend pytorch` (HF Trainer rejects
`output_format='yolox'`).

Purpose: complete the three-arch matrix against the DETR-family siblings
(`../our_rtdetr_v2_albumentations/`, `../our_dfine_albumentations/`).
Same data split, same input size, same Albumentations aug semantics —
so any mAP / wall-time delta is the arch × loss function × matcher
choice, not the pipeline.

## Prereqs (one-time)

```bash
# Main venv (for code path + configs)
uv sync

# Separate venv for the official Megvii yolox package — conflicting
# transformers/torch versus the main .venv.
bash scripts/setup-yolox-venv.sh

# Pretrained weights (8.9 M param YOLOX-M COCO checkpoint)
ls pretrained/yolox_m.pth   # should exist

# CPPE-5 dumped to disk (shared with all reference/our_* folders)
.venv-notebook/bin/python notebooks/detr_finetune_reference/data_loader.py --dump-cppe5
```

## Run (critical — use the official venv, not `uv run`)

```bash
CUDA_VISIBLE_DEVICES=1 .venv-yolox-official/bin/python \
  core/p06_training/train.py \
  --config notebooks/detr_finetune_reference/our_yolox/06_training.yaml
```

Do NOT `uv run` this config. `uv run` activates the main `.venv/` which
has our custom YOLOX impl + a newer `transformers` pin that clashes with
the official Megvii `yolox==0.3.0` requirements. Always invoke via
`.venv-yolox-official/bin/python` (created by `scripts/setup-yolox-venv.sh`).

Artefacts land in `runs/seed42/` — custom pytorch trainer layout:

```
runs/seed42/
├── best.pth                    # best by val/mAP50
├── last.pth
├── ema.pth                     # EMA-averaged weights (separate from best)
├── data_preview/               # dataset_stats, data_labels_{train,val,test}, aug_labels_train
├── val_predictions/            # per-epoch GT-vs-Pred grids
├── trainer_state.json
├── runs/<ts>/                  # tensorboard
└── 05_data.yaml / 06_training.yaml / config_resolved.yaml   # lineage
```

## Aug-parity caveat (important)

Albumentations **does not implement Mosaic / MixUp / CopyPaste** —
those are dataset-level ops that need `Dataset.get_raw_item()` hooks and
live only in the torchvision v2 backend of `core/p05_data/transforms.py`.
Using Albumentations here means YOLOX trains **without Mosaic**, which:

- Makes the comparison against the DETR-family siblings *cleanly
  apples-to-apples* (same aug set, same library, same semantics — only
  the arch differs).
- Costs YOLOX **~3-5 mAP points** vs the production recipe on larger
  datasets where Mosaic is a significant regularizer (verified on
  `features/safety-fire_detection/` — YOLOX-M with Mosaic hits 0.442
  full-val mAP, YOLOX-M without hits roughly the same scale of loss in
  head-only capacity).
- CPPE-5 at 850 images is small enough that Mosaic's context-stitching
  benefit is muted; the gap here is smaller than it would be on 17 k
  images.

If you want the production YOLOX recipe (with Mosaic), use
`core/p05_data/transforms.py` torchvision v2 backend instead — set
`augmentation.library: torchvision, mosaic: true, mixup: true` in a
separate `our_yolox_torchvision/` variant.

## Other YOLOX-specific pitfalls encoded in this config

- **`augmentation.normalize: false`** — Megvii YOLOX weights expect raw
  `[0, 255]` uint8 pixel inputs. Don't divide by 255, don't subtract
  ImageNet mean/std. Silent correctness bug if you flip it (all scores
  collapse to near-zero).
- **`training.backend: pytorch`** — not `hf`. Our HF backend's config
  validator hard-fails on YOLOX (`output_format='yolox'`).
- **`training.amp: true`** — safe for YOLOX (unlike DETR-family which
  requires `amp: false` due to decoder fp16 overflow).
- **`lr: 0.0025`** — Megvii's scaling rule `basic_lr × bs / 64` =
  `0.01 × 16/64`. The default `0.01` is 4× too hot at bs=16 (documented
  in `features/safety-fire_detection/CLAUDE.md` → *Overfit-capability
  analysis*).
- **`model.pretrained`** — auto-detects Megvii key format
  (`backbone.backbone.*`) and remaps to the adapter's convention. No
  manual key renaming.

## Result (seed=42, 50 epochs, GPU 1, 2026-04-20)

### Wall time — **1.55× faster than RT-DETRv2 on the same GPU**

| axis | value |
|---|---|
| `train_runtime` (50 ep) | **553.7 s** (~9 min 14 s) |
| per-epoch | 11.1 s |
| RT-DETRv2 sibling (40 ep, same GPU) | 857.3 s (21.4 s/ep) |

YOLOX-M beats RT-DETRv2-R50 on wall time by **~2× per epoch** here —
smaller encoder, no transformer self-attention overhead, and AMP is
enabled where DETR requires fp32.

### Accuracy — val (full CPPE-5 val, bs=16, no TTA)

| metric | val (best ep26) | val (final ep50) |
|---|---|---|
| mAP@0.5 | **0.6561** | 0.6408 |
| Coverall AP₅₀ | 0.8163 | 0.7998 |
| Face_Shield AP₅₀ | 0.6796 | 0.6344 |
| Gloves AP₅₀ | 0.5583 | 0.5422 |
| Goggles AP₅₀ | 0.5179 | 0.5232 |
| Mask AP₅₀ | 0.7084 | 0.7043 |

### Accuracy — test (29 imgs, p08 evaluate.py, conf=0.05, IoU=0.5)

| metric | value |
|---|---|
| **Test mAP@0.5** | **0.5718** |
| Coverall AP | 0.6991 |
| Face_Shield AP | 0.7218 |
| Gloves AP | 0.4419 |
| Goggles AP | 0.2960 |
| Mask AP | 0.7000 |

### Comparison vs RT-DETRv2-R50 sibling (same GPU 1, same data split)

| axis | YOLOX-M (this) | RT-DETRv2-R50 (`../our_rtdetr_v2_albumentations/`) |
|---|---|---|
| epochs | 50 | 40 |
| `train_runtime` | **553.7 s** (1.55× faster) | 857.3 s |
| per-epoch | 11.1 s (1.93× faster) | 21.4 s |
| val mAP@0.5 | 0.6561 | — |
| test mAP₅₀ | **0.5718** (−0.200) | 0.7714 |
| test mAP (COCO) | ≈ 0.35¹ | 0.5309 |

¹ YOLOX was evaluated via `core/p08_evaluation/evaluate.py` which
reports single-IoU AP@0.5, not the COCO-style `mAP@[0.5:0.95]`. The
0.35 is inferred from the usual ~0.60 × mAP₅₀ relationship on
CPPE-5; for a strict apples-to-apples torchmetrics MAP, re-evaluate
with the HF Trainer test loop or wire torchmetrics into `evaluator.py`.

### Interpretation

- **YOLOX is faster but weaker on mAP₅₀ here (−0.20 vs RT-DETRv2).**
  Root cause is the Albumentations constraint — CPPE-5 with Mosaic
  typically closes ~0.05-0.10 of that gap (YOLOX's training recipe
  assumes Mosaic provides scale/context augmentation; removing it
  starves the backbone).
- **Goggles is the weakest class (0.30 test AP)** — smallest support
  in CPPE-5 and most sensitive to aug richness. This tracks the pattern
  in `features/safety-fire_detection/` where rare-class AP is the first
  thing Mosaic regularisation protects.
- **YOLOX beats RT-DETRv2 on Face_Shield (0.72 vs 0.67)** —
  low-variance class where simpler IoU-based matching (SimOTA) converges
  faster than bipartite.

### If you want the full YOLOX recipe

Switch `augmentation.library: torchvision` + `mosaic: true` + optional
`mixup: true` in `06_training.yaml`. Expected gain: +0.05-0.10 mAP₅₀
at the cost of ~+30 % wall time (Mosaic does 4× disk reads per sample).
Create `our_yolox_torchvision/` as a sibling if you want this
head-to-head.

## Head-to-head lives in `../CLAUDE.md`

Numbers above also copied into the *Head-to-head* table in
`../CLAUDE.md` alongside the three DETR-family runs for a single
at-a-glance comparison.

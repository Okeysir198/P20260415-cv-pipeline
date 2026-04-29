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

## Result (seed=42, 50 epochs, 2026-04-22 rerun)

### Accuracy — val (full CPPE-5 val, bs=16, no TTA)

Final `runs/seed42/test_results.json`:

| metric | value |
|---|---|
| **val/mAP50** | **0.7388** |
| Coverall AP₅₀ (cls0) | 0.7983 |
| Face_Shield AP₅₀ (cls1) | 0.9274 |
| Gloves AP₅₀ (cls2) | 0.5283 |
| Goggles AP₅₀ (cls3) | 0.6802 |
| Mask AP₅₀ (cls4) | 0.7596 |
| precision | 0.8520 |
| recall | 0.7299 |

Matches the `notebooks/detr_finetune_reference/CLAUDE.md` head-to-head
table (`our_yolox` row, val mAP₅₀ 0.7388).

### Comparison vs DETR-family siblings (CPPE-5 val, same data split)

| run | arch | aug | epochs | val mAP₅₀ |
|---|---|---|---|---|
| **our_yolox (this)** | yolox-m (official) | Albumentations | 50 | **0.7388** |
| our_yolox_torchvision | yolox-m (official) | TV + Mosaic + MixUp | 100 | 0.8668 |
| our_rtdetr_v2_albumentations | RT-DETRv2-R50 | Albumentations | 40 | 0.8264 |
| our_rtdetr_v2_torchvision | RT-DETRv2-R50 | torchvision v2 | 40 | 0.8231 |
| our_dfine_albumentations | dfine-n | Albumentations | 30 | 0.6778 |
| our_dfine_torchvision | dfine-n | torchvision v2 | 30 | 0.6828 |

### Interpretation

- **YOLOX is weaker on mAP₅₀ here (−0.09 vs RT-DETRv2-R50, −0.13 vs
  our_yolox_torchvision).** Root cause is the no-Mosaic constraint —
  the Albumentations backend can't do dataset-level Mosaic/MixUp, so
  this run is YOLOX-without-its-training-recipe. The `our_yolox_torchvision/`
  sibling at 100 ep with TV + Mosaic + MixUp closes the gap and beats
  RT-DETRv2.
- **Gloves is the weakest class (val AP₅₀ 0.53)** — smallest objects
  on average; YOLOX without Mosaic loses scale-augmentation richness.
- **Face_Shield strongest at 0.93** — low-variance class where SimOTA's
  IoU-based matching converges quickly.

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

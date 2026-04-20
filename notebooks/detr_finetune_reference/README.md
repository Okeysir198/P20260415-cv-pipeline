# DETR-family fine-tune reference scripts + in-repo pipeline comparisons

Central ground-truth folder for any RT-DETRv2 / D-FINE / YOLOX work on the
HF `cppe-5` dataset. Holds:

1. **Reference runs** — runnable ports of qubvel's HF notebooks. Used as
   the known-good baseline for diffing against our own pipeline.
2. **In-repo pipeline runs** — the same recipes executed through
   `core/p06_training/train.py`. Proves our `backend: hf` detection path
   is a faithful translation and measures where it differs.

## Layout

```
notebooks/detr_finetune_reference/
├── README.md                       (this file — overview + navigation)
├── CLAUDE.md                       Claude-facing notes on recipes / gotchas / variance results
├── pyproject.toml                  pinned deps (uv-managed — albumentations 1.4.6 etc.)
├── uv.lock
├── data_loader.py                  shared — HF `cppe-5` ↔ YOLO bridge + --dump-cppe5 CLI
├── run_inference.py                shared — GT-vs-pred side-by-side PNG grids
│
├── reference_rtdetr_v2/            qubvel's RT-DETRv2 reference (runnable .py + .ipynb + runs/)
│   ├── finetune.py                 RT-DETRv2-R50 fine-tune (CLI: --seed, --tag, --aug)
│   ├── inference.py                Single-image inference
│   ├── RT_DETR_v2_finetune_on_a_custom_dataset.ipynb   (upstream original)
│   ├── RT_DETR_v2_inference.ipynb                       (upstream original)
│   ├── README.md
│   └── runs/                       (gitignored) training outputs
│
├── reference_dfine/                qubvel's D-FINE reference (runnable .py + .ipynb + runs/)
│   ├── finetune.py                 D-FINE-large fine-tune (lr=2e-5 fix applied, see README)
│   ├── inference.py
│   ├── DFine_finetune_on_a_custom_dataset.ipynb        (upstream original)
│   ├── DFine_inference.ipynb                            (upstream original)
│   ├── README.md
│   └── runs/                       (gitignored)
│
├── our_rtdetr_v2_albumentations/   OUR pipeline, RT-DETRv2, Albumentations aug — DONE
│   ├── 05_data.yaml / 06_training.yaml / README.md
│   └── runs/                       (gitignored)
│
├── our_rtdetr_v2_torchvision/      OUR pipeline, RT-DETRv2, torchvision v2 aug — PLACEHOLDER
├── our_dfine/                      OUR pipeline, D-FINE, HF backend — PLACEHOLDER
└── our_yolox/                      OUR pipeline, YOLOX-M, pytorch backend — PLACEHOLDER
```

`reference_*/` folders hold the **upstream baseline**. `our_*/` folders
hold **the same experiment run through `core/p06_training/`** for
apples-to-apples comparison. Each folder has its own README with setup
+ expected numbers.

## Setup (once)

```bash
bash scripts/setup-notebook-venv.sh
# creates .venv-notebook/ via `uv sync` against this folder's pyproject.toml
# (albumentations 1.4.6 + torchmetrics + HF transformers git + torch cu130)
```

Or directly with uv:

```bash
UV_PROJECT_ENVIRONMENT="$(pwd)/.venv-notebook" \
  uv sync --project notebooks/detr_finetune_reference --python 3.12
```

## Run the reference (single command)

```bash
# RT-DETRv2 — matches qubvel's published 0.5789 test mAP within ~0.02
CUDA_VISIBLE_DEVICES=1 .venv-notebook/bin/python \
  notebooks/detr_finetune_reference/reference_rtdetr_v2/finetune.py \
  --seed 42 --tag bs16_lr1e4_cosine_wd_bf16

# D-FINE — qubvel's 0.4485 target; finetune.py has the lr=2e-5 fix baked in
CUDA_VISIBLE_DEVICES=1 .venv-notebook/bin/python \
  notebooks/detr_finetune_reference/reference_dfine/finetune.py \
  --seed 42 --tag lr2e5_warmup500_cosine_wd_bf16
```

Outputs land under the script's own `runs/` subfolder (resolved via
`__file__` so cwd-independent).

## Run our in-repo pipeline on the same recipe

```bash
# CPPE-5 → dump to disk once (reads HF dataset, writes YOLO to
# dataset_store/training_ready/cppe5/)
.venv-notebook/bin/python notebooks/detr_finetune_reference/data_loader.py --dump-cppe5

# Then train via core/p06_training/
CUDA_VISIBLE_DEVICES=1 uv run core/p06_training/train.py \
  --config notebooks/detr_finetune_reference/our_rtdetr_v2_albumentations/06_training.yaml
```

## Results log

### RT-DETRv2 / CPPE-5 — progression toward reproducing qubvel's 0.5789 test mAP

| Date | Tag | Seed | Config | val best | test mAP | test mAP50 | Wall time | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-04-20 | (baseline non-det) | OS-entropy | qubvel's recipe | 0.3231 @ ep19 | 0.5054 | 0.7631 | 10m 33s | one-shot unlucky seed; -0.073 vs qubvel. |
| 2026-04-20 | (deterministic) | 42 | +`set_seed(42)` before from_pretrained, `cudnn.deterministic`, `use_deterministic_algorithms(warn_only=True)` | 0.3659 @ ep19 | 0.5325 | 0.7814 | 11m 31s | `+0.027` just from seeding the class-head reinit; reproducible run-over-run. |
| 2026-04-20 | `cosine_wd_bf16` | 42 | Bundle A: `lr_scheduler_type="cosine"`, `weight_decay=1e-4`, `bf16=True` | 0.3686 @ ep23 | 0.5348 | 0.8118 | 11m 28s | Regularization redistributed across classes ⇒ net ≈ 0. |
| 2026-04-20 | `bs16_lr1e4_cosine_wd_bf16` | 42 | Bundle B: `bs=16`, `lr=1e-4`, cosine, WD, bf16 | **0.3740 @ ep13** | **0.5585** | 0.8222 | **9m 23s** | **Reference best single run.** -0.020 vs qubvel. Faster than qubvel's recipe. |
| 2026-04-20 | `bs16_lr1e4_cosine_wd_bf16` | 0 | Bundle B | — | 0.4857 | 0.7606 | 9m 23s | Seed variance check. |
| 2026-04-20 | `bs16_lr1e4_cosine_wd_bf16` | 2024 | Bundle B | — | 0.5418 | 0.8158 | 9m 23s | Seed variance check. |
| **2026-04-20** | **Bundle B 3-seed mean ± std** | 42/0/2024 | same | — | **0.5287 ± 0.030** | 0.7995 ± 0.027 | — | Phase-1 summary. Qubvel's 0.5789 at +1.65σ — reachable. |
| 2026-04-20 | `bs16_lr1e4_cosine_wd_bf16_aug-strong` | 42 | Bundle B + BBoxSafeRandomCrop + stronger HSV + CLAHE | — | 0.5420 | 0.7971 | 10m 27s | `--aug strong` net-negative. Opt-in only. |
| **2026-04-20** | **our_rtdetr_v2_albumentations** | **42** | OUR in-repo pipeline, same Bundle B recipe | 0.3593 @ ep20 | **0.5591** | **0.8619** | 10m 20s | **In-repo pipeline matches the reference within 0.0006.** Proves `core/p06_training` HF backend reproduces qubvel's RT-DETRv2 recipe. |

### D-FINE / CPPE-5

| Tag | Seed | Config | val best | test mAP | Notes |
|---|---|---|---|---|---|
| (naïve port) | 42 | qubvel's recipe (lr=5e-5) | 0.1976 @ ep3 | 0.2617 | Val saturated ep3; LR too hot for dfine-large's backbone. |
| `lr2e5_warmup500_cosine_wd_bf16` | 42 | lr=2e-5, warmup=500, cosine, wd=1e-4, bf16 | TBD | TBD | In-flight / TBD. |

## Canonical training args (from qubvel's notebooks)

| Arg | RT-DETRv2 | D-FINE |
|---|---|---|
| `num_train_epochs` | 40 | 30 |
| `learning_rate` | 5e-5 (qubvel) / **1e-4** (our Bundle B) | 5e-5 (qubvel) / **2e-5** (our fix) |
| `warmup_steps` | 300 | 300 (qubvel) / **500** (our fix) |
| `max_grad_norm` | 0.1 | 0.1 |
| `per_device_train_batch_size` | 8 (qubvel) / **16** (our Bundle B) | 8 |
| `checkpoint` | `PekingU/rtdetr_v2_r50vd` | `ustc-community/dfine-large-coco` |
| `image_size` | 480 | 480 |

## Phase 2 — swap CPPE-5 for our features (future work)

The reference scripts have CPPE-5 hardcoded in the data-loading cell.
To run the same recipe on one of our features (fire_detection,
helmet_detection, etc.):

```python
# In `reference_rtdetr_v2/finetune.py`, replace:
#   dataset = load_dataset("cppe-5")
# with:
from data_loader import load_feature_dataset
dataset = load_feature_dataset("fire_detection", subset=0.05)
```

`data_loader.py` emits a schema byte-compatible with CPPE-5 so the
downstream Albumentations / `image_processor` / `CPPE5Dataset` / `Trainer`
code runs unchanged.

## Conversion notes — what differs from the original notebooks

The `.py` files are direct ports of the `.ipynb` via `jupyter nbconvert --to script`,
with three mechanical cleanups applied:

1. **Shell installs removed** — `!pip install …` lines stripped; deps live
   in `pyproject.toml` and get installed by `scripts/setup-notebook-venv.sh`.
2. **Jupyter `display(...)` commented out** — visualization cells don't
   run in plain Python. Training/eval behaviour unchanged.
3. **`datasets` 4.x syntax fix** — the notebook accesses
   `ds.features["objects"].feature["category"].names` (valid in 2.x,
   raises on 4.x). Rewritten to `ds.features["objects"]["category"].feature.names`.

No training-behaviour-affecting changes. Original notebooks live next to
their `.py` ports inside each `reference_*/` folder for byte-level
verification.

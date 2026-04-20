# DETR-family fine-tune reference scripts

Direct ports of qubvel's HF reference notebooks for RT-DETRv2 and D-FINE,
converted to plain Python so they can be run, diffed, and compared against our
in-repo `DetectionTrainer` pipeline as a known-good baseline.

## Layout

```
notebooks/detr_finetune_reference/
├── README.md                       (this file)
├── pyproject.toml                  pinned deps (uv-managed)
├── rtdetr_v2_finetune_cppe5.py     RT-DETRv2 fine-tune on CPPE-5 (runnable)
├── dfine_finetune_cppe5.py         D-FINE fine-tune on CPPE-5 (runnable)
├── rtdetr_v2_inference.py          RT-DETRv2 inference (runnable)
├── dfine_inference.py              D-FINE inference (runnable)
├── data_loader.py                  YOLO → HF Dataset bridge for when we later
│                                   want to swap CPPE-5 → our features
└── reference/                      Untouched original .ipynb notebooks
    ├── RT_DETR_v2_finetune_on_a_custom_dataset.ipynb
    ├── DFine_finetune_on_a_custom_dataset.ipynb
    ├── RT_DETR_v2_inference.ipynb
    └── DFine_inference.ipynb
```

## Setup

```bash
bash scripts/setup-notebook-venv.sh
# creates .venv-notebook/ via `uv sync` against notebooks/detr_finetune_reference/pyproject.toml
# (albumentations==1.4.6 + torchmetrics + HF transformers git + torch/torchvision cu130)
```

Or directly with uv from repo root:

```bash
UV_PROJECT_ENVIRONMENT="$(pwd)/.venv-notebook" \
  uv sync --project notebooks/detr_finetune_reference --python 3.12
```

## Phase 1 — replicate the reference result on CPPE-5

Run the notebooks AS-IS on their native CPPE-5 dataset (~1k training images,
5 classes). Goal: confirm the reference code trains cleanly in our environment,
so any gap vs our in-repo pipeline is attributable to our code, not the
notebook recipe.

```bash
.venv-notebook/bin/python notebooks/detr_finetune_reference/rtdetr_v2_finetune_cppe5.py
.venv-notebook/bin/python notebooks/detr_finetune_reference/dfine_finetune_cppe5.py
```

Expected outcome (from qubvel's published results):
- **RT-DETRv2-R50** on CPPE-5 @ 40 epochs: val `mAP` ≈ 0.34
- **D-FINE** on CPPE-5 @ 30 epochs: val `mAP` ≈ 0.33

## Phase 2 — swap CPPE-5 for our features (optional, after Phase 1 passes)

Only after the reference scripts reproduce the expected CPPE-5 numbers:

1. In each script, replace the CPPE-5 loading block (around line 55-65):
   ```python
   # REMOVE:
   # from datasets import load_dataset
   # dataset = load_dataset("cppe-5")
   # if "validation" not in dataset: ...

   # ADD:
   from data_loader import load_feature_dataset
   dataset = load_feature_dataset("fire_detection", subset=0.05)
   ```
2. Nothing else changes. `data_loader.py` emits a schema byte-compatible with CPPE-5
   (COCO bbox format, `ClassLabel` category feature), so the downstream
   Albumentations / `image_processor` / `CPPE5Dataset` / `Trainer` code all work
   verbatim.

## Conversion notes — what differs from the original notebooks

The `.py` files are direct ports of the `.ipynb` via `jupyter nbconvert --to script`,
with three mechanical cleanups applied:

1. **Shell installs removed** — `!pip install …` / `get_ipython().system(…)` lines
   stripped. Deps live in `pyproject.toml` and are installed by
   `scripts/setup-notebook-venv.sh` via `uv sync`.
2. **Jupyter `display(...)` commented out** — visualization cells don't run in
   plain Python. Training/eval behavior unchanged.
3. **`datasets` 4.x syntax fix** — the notebook accesses
   `ds.features["objects"].feature["category"].names` (valid in datasets 2.x),
   which raises `AttributeError` on datasets 4.x. The `.py` uses
   `ds.features["objects"]["category"].feature.names` instead. Same result.

No training-behavior-affecting changes. Original notebooks are preserved under
`reference/` for byte-level verification.

## Canonical training args (from qubvel's notebooks — DO NOT change)

| Arg | RT-DETRv2 | D-FINE |
|---|---|---|
| `num_train_epochs` | 40 | 30 |
| `learning_rate` | 5e-5 | 5e-5 |
| `warmup_steps` | 300 | 300 |
| `max_grad_norm` | 0.1 | 0.1 |
| `per_device_train_batch_size` | 8 | 8 |
| `checkpoint` | `PekingU/rtdetr_v2_r50vd` | `ustc-community/dfine-large-coco` |
| `image_size` | 480 | 480 |

## Results log

### RT-DETRv2 / CPPE-5 — progression toward reproducing qubvel's 0.5789 test mAP

| Date | Tag | Seed | Config | val best | test mAP | test mAP50 | Wall time | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-04-20 | (baseline non-det) | OS-entropy | qubvel's recipe | 0.3231 @ ep19 | 0.5054 | 0.7631 | 10m 33s | one-shot unlucky seed; -0.073 vs qubvel. |
| 2026-04-20 | (deterministic) | 42 | +`set_seed(42)` before from_pretrained, `cudnn.deterministic`, `use_deterministic_algorithms(warn_only=True)` | 0.3659 @ ep19 | 0.5325 | 0.7814 | 11m 31s | `+0.027` just from seeding the class-head reinit; reproducible run-over-run. |
| 2026-04-20 | `cosine_wd_bf16` | 42 | Bundle A: `lr_scheduler_type="cosine"`, `weight_decay=1e-4`, `bf16=True` | 0.3686 @ ep23 | 0.5348 | 0.8118 | 11m 28s | Regularization redistributed: Face_Shield +0.06 / Goggles +0.03 / Gloves -0.04 / Mask -0.04 ⇒ net ≈ 0. |
| 2026-04-20 | `bs16_lr1e4_cosine_wd_bf16` | 42 | Bundle B: `bs=16`, `lr=1e-4`, cosine, WD, bf16 | **0.3740 @ ep13** | **0.5585** | 0.8222 | **9m 23s** | Closes to **-0.020 of qubvel**. Beats qubvel on Coverall / Gloves / small objects. Remaining gap ≈ Goggles class (-0.10). **Faster than qubvel's recipe**. |
| — | `dfine` on CPPE-5 | 42 | qubvel's recipe | 0.1976 @ ep3 | 0.2617 | 0.3691 | 9m 16s | Val saturated ep3, stuck for 27 more epochs — LR too hot for dfine-large's backbone; not used as Phase-2 reference until re-tuned (lr=2e-5, warmup=500). |

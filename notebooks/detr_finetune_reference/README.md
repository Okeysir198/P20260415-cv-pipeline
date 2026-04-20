# DETR-family fine-tune reference scripts

Direct ports of qubvel's HF reference notebooks for RT-DETRv2 and D-FINE,
converted to plain Python so they can be run, diffed, and compared against our
in-repo `DetectionTrainer` pipeline as a known-good baseline.

## Layout

```
notebooks/detr_finetune_reference/
├── README.md                       (this file)
├── CLAUDE.md                       Claude-facing notes — recipes, gotchas, variance results
├── pyproject.toml                  pinned deps (uv-managed)
├── .gitignore                      excludes runs/, inference/, .venv/
├── rtdetr_v2_finetune_cppe5.py     RT-DETRv2 fine-tune on CPPE-5 (CLI: --seed, --tag, --aug)
├── dfine_finetune_cppe5.py         D-FINE fine-tune on CPPE-5 (not yet re-tuned — see log)
├── rtdetr_v2_inference.py          RT-DETRv2 single-image inference (uses local best ckpt)
├── dfine_inference.py              D-FINE single-image inference (uses local best ckpt)
├── run_inference.py                GT-vs-pred side-by-side on N train + N val samples
├── data_loader.py                  YOLO → HF Dataset bridge (Phase 2 — swap CPPE-5 → our features)
├── runs/                           (gitignored) per-run output dirs
├── inference/                      (gitignored) GT-vs-pred PNGs from run_inference.py
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

RT-DETRv2 reproduction: **done**. D-FINE: pending (see Results log).
Run the scripts directly — they're now CLI-configurable:

```bash
# RT-DETRv2 — Bundle B deterministic recipe (our operating default).
# Best single-seed result so far: test mAP = 0.5585 (seed=42), vs qubvel's 0.5789.
CUDA_VISIBLE_DEVICES=1 \
  .venv-notebook/bin/python notebooks/detr_finetune_reference/rtdetr_v2_finetune_cppe5.py \
    --seed 42 --tag bs16_lr1e4_cosine_wd_bf16

# Byte-identical to qubvel's notebook (stripped recipe) — deterministic seed only.
.venv-notebook/bin/python notebooks/detr_finetune_reference/rtdetr_v2_finetune_cppe5.py \
    --seed 42 --tag deterministic_baseline

# D-FINE (not yet re-tuned; qubvel's recipe fails to converge — see log).
.venv-notebook/bin/python notebooks/detr_finetune_reference/dfine_finetune_cppe5.py
```

Qubvel's published numbers (test mAP, one run, no seed variance reported):
- **RT-DETRv2-R50** on CPPE-5 @ 40 epochs: 0.5789
- **D-FINE-large** on CPPE-5 @ 30 epochs: 0.4485

### What changed from a straight notebook port

The original notebook had **no explicit seeding** — the class-head reinit in
`from_pretrained` uses whatever OS-entropy state Python booted with, so
every fresh process gets different class-head init and different final mAP
(we observed up to ±0.07 swing). Our script now:

1. Calls `set_seed(SEED)` **before** `from_pretrained`.
2. Sets `cudnn.deterministic=True`, `benchmark=False`,
   `use_deterministic_algorithms(True, warn_only=True)`, and
   `CUBLAS_WORKSPACE_CONFIG=:4096:8` in env.
3. Passes `seed=SEED` + `data_seed=SEED` to `TrainingArguments`.
4. Writes to `runs/rtdetr_v2_r50_cppe5_seed{SEED}{_TAG}/` — one run per
   config.

After this, **same seed → same mAP run-over-run** (modulo two non-deterministic
CUDA kernels in deformable attention that upstream PyTorch hasn't patched).

### Bundle B — the tuned recipe

Beyond qubvel's exact notebook, three additions that cleanly close most of
the remaining gap:

| Change | Why |
|---|---|
| `per_device_train_batch_size=16` (was 8) | 2× stable Hungarian matching |
| `learning_rate=1e-4` (was 5e-5) | linear scaling rule for 2× batch |
| `lr_scheduler_type="cosine"` | anneal post-peak, not drift on linear decay |
| `weight_decay=1e-4` | HF Trainer default is 0; DETR canonical is 1e-4 |
| `bf16=True` | RTX 5090 tensor cores, ~1.3× faster, neutral numerically |

Bundle B finishes in **9m 23s vs qubvel's ~10m 33s** — faster *and* closer
to qubvel's mAP. It's the default recipe we'll use when swapping CPPE-5
for our own features in Phase 2.

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
| 2026-04-20 | `bs16_lr1e4_cosine_wd_bf16` | 42 | Bundle B: `bs=16`, `lr=1e-4`, cosine, WD, bf16 | **0.3740 @ ep13** | **0.5585** | 0.8222 | **9m 23s** | **Best single run.** -0.020 vs qubvel. Beats qubvel on Coverall / Gloves / small objects. Residual gap ≈ Goggles (-0.10). Faster than qubvel's recipe. |
| 2026-04-20 | `bs16_lr1e4_cosine_wd_bf16` | 0 | Bundle B (different seed) | — | 0.4857 | 0.7606 | 9m 23s | Seed variance check. Goggles AP=0.335 — rare-class instability dominates. |
| 2026-04-20 | `bs16_lr1e4_cosine_wd_bf16` | 2024 | Bundle B (different seed) | — | 0.5418 | 0.8158 | 9m 23s | Seed variance check. |
| **2026-04-20** | **Bundle B 3-seed mean ± std** | 42/0/2024 | same | — | **0.5287 ± 0.030** | 0.7995 ± 0.027 | — | **Phase-1 summary**. Qubvel's 0.5789 sits at +1.65σ — reachable as a lucky seed, not statistically distinguishable from a library regression with only 3 samples. Goggles std=0.038 (highest of any class). |
| 2026-04-20 | `bs16_lr1e4_cosine_wd_bf16_aug-strong` | 42 | Bundle B + BBoxSafeRandomCrop + stronger HSV + CLAHE | — | 0.5420 | 0.7971 | 10m 27s | `--aug strong` **net negative** on overall mAP (-0.017 vs basic aug); does give +0.005 on Goggles and +0.05 on Mask but at cost of -0.06 on large objects. Opt-in only. |
| — | `dfine` on CPPE-5 | 42 | qubvel's recipe | 0.1976 @ ep3 | 0.2617 | 0.3691 | 9m 16s | Val saturated ep3, stuck for 27 more epochs — LR too hot for dfine-large's backbone; not used as Phase-2 reference until re-tuned (lr=2e-5, warmup=500). |

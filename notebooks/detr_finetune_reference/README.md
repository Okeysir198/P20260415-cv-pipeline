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

## Results, recipes, and gotchas

Full per-run metrics, the reference-vs-in-repo head-to-head comparison,
Bundle B hyperparameter rationale, conversion notes, canonical training
args, Phase 2 plans, and per-arch history live in
[`CLAUDE.md`](./CLAUDE.md) — this file is deliberately setup-oriented to
avoid duplicating that reference.

Quick headline from the latest seed=42 Bundle B runs on parallel GPUs,
plus qubvel's originally-published numbers from the upstream `.ipynb`:

| Axis | qubvel published | `reference_rtdetr_v2/` | `our_rtdetr_v2_albumentations/` |
|---|---|---|---|
| `train_runtime` | — | 617.1 s | 615.1 s |
| Test mAP | 0.5789 | 0.5464 | **0.5577** |
| Test mAP₅₀ | 0.8674 | 0.8043 | 0.8285 |

See `CLAUDE.md` → *Head-to-head* for per-class breakdown + Δ + notes on
qubvel's test-set variance (single run, ±0.03 σ on 29-image test).

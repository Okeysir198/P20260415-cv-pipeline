# Image-classification fine-tune reference scripts + in-repo pipeline comparisons

Central ground-truth folder for any image-classification (timm / HF
`AutoModelForImageClassification`) work on HF public datasets, starting with
**EuroSAT** (10-class satellite land-use, 64×64 RGB, 27k images — public, no
gated terms).

Mirrors the layout of `notebooks/detr_finetune_reference/`:

1. **Reference runs** — runnable ports of the upstream HF cookbook notebook
   ([`image_classification.ipynb`](https://github.com/huggingface/notebooks/blob/main/examples/image_classification.ipynb)).
   Used as the known-good baseline for diffing against our own pipeline.
2. **In-repo pipeline runs** — the same recipes executed through
   `core/p06_training/train.py` (backend `hf` or `pytorch` + timm).

## Layout

```
notebooks/image_classification_finetune_reference/
├── README.md                       (this file)
├── CLAUDE.md                       Claude-facing notes on recipes / gotchas
├── data_loader.py                  EuroSAT → ImageFolder dump CLI
├── .gitignore                      ignores runs/, .venv*/, .ipynb_checkpoints
│
├── reference_swin_tiny/            (pending) upstream cookbook .py port
│                                    — microsoft/swin-tiny-patch4-window7-224
│
└── our_swin_tiny/                  (pending) same recipe via
                                    core/p06_training/ --backend hf
```

`reference_*/` folders hold the **upstream baseline**. `our_*/` folders hold
**the same experiment run through `core/p06_training/`** for apples-to-apples
comparison. Each folder will have its own README with setup + expected numbers.

## Setup (once)

Same venv as the DETR reference folder — `albumentations` pin doesn't matter
for image classification, but we reuse the notebook venv for consistency:

```bash
bash scripts/setup-notebook-venv.sh
# creates .venv-notebook/ via `uv sync` against the DETR-reference pyproject.toml
```

> **⚠️ CRITICAL:** Reference `.py` scripts should run in `.venv-notebook/`,
> NOT the main `.venv/`. Always invoke via `.venv-notebook/bin/python ...`.

## Dump EuroSAT to disk (once)

```bash
.venv-notebook/bin/python \
  notebooks/image_classification_finetune_reference/data_loader.py \
  --dump-eurosat
```

Writes `dataset_store/training_ready/eurosat/{train,val}/<class>/*.jpg`
(ImageFolder layout) + `id2label.json`. Stratified 80/20 split per class,
seed 42. Rerunning skips the dump if file counts match (use `--force` to
override, or `--limit N` for a smoke dump).

## Run the reference (pending)

```bash
# Upstream HF cookbook recipe — microsoft/swin-tiny, EuroSAT, 5 epochs
CUDA_VISIBLE_DEVICES=0 .venv-notebook/bin/python \
  notebooks/image_classification_finetune_reference/reference_swin_tiny/finetune.py \
  --seed 42
```

## Run our in-repo pipeline on the same recipe (pending)

```bash
CUDA_VISIBLE_DEVICES=0 uv run core/p06_training/train.py \
  --config notebooks/image_classification_finetune_reference/our_swin_tiny/06_training.yaml
```

## Results — pending

Run once reference + `our_*` scripts land. Expected columns mirror the DETR
README's head-to-head table:

| Axis | HF cookbook published | `reference_swin_tiny/` | `our_swin_tiny/` |
|---|---|---|---|
| Epochs | 5 | — | — |
| Eval accuracy | ~0.97 (expected) | — | — |
| Eval top-1 | ~0.97 | — | — |

## Gotchas

Full per-recipe notes + conversion gotchas live in [`CLAUDE.md`](./CLAUDE.md).

## Upstream references

- HF cookbook notebook:
  https://github.com/huggingface/notebooks/blob/main/examples/image_classification.ipynb
- EuroSAT (public, no gated terms): https://huggingface.co/datasets/jonathan-roberts1/EuroSAT
- Swin-T checkpoint: https://huggingface.co/microsoft/swin-tiny-patch4-window7-224

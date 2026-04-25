# Segformer fine-tune reference scripts + in-repo pipeline comparisons

Central ground-truth folder for any semantic-segmentation (HF
`AutoModelForSemanticSegmentation`, SegFormer family) work on HF public
datasets.

**Experiment: sidewalk-semantic** — the recipe from the upstream HF cookbook /
blog post (`segments/sidewalk-semantic`, 35 classes, gated — requires `HF_TOKEN`).

Mirrors the layout of `notebooks/detr_finetune_reference/` and
`notebooks/image_classification_finetune_reference/`:

1. **Reference run** — runnable port of the upstream HF cookbook
   ([`semantic_segmentation.ipynb`](https://github.com/huggingface/notebooks/blob/main/examples/semantic_segmentation.ipynb)
   + [blog post](https://huggingface.co/blog/fine-tune-segformer)).
   Known-good baseline for diffing against our own pipeline.
2. **In-repo pipeline run** — the same recipe executed through
   `core/p06_training/train.py --backend hf`.

## Layout

```
notebooks/segformer_finetune_reference/
├── README.md                            (this file)
├── CLAUDE.md                            Claude-facing notes on recipes / gotchas
├── .gitignore                           ignores runs/, .venv*/, .ipynb_checkpoints
│
├── reference_segformer_b0/              upstream cookbook/blog .py port
│   ├── semantic_segmentation.ipynb      frozen upstream notebook
│   ├── finetune.py                      sidewalk-semantic recipe (gated; HF_TOKEN required)
│   └── inference.py                     val-prediction overlay grid
│
└── our_segformer_b0/                    same recipe via core/p06_training/
    └── 05_data.yaml   06_training.yaml
```

`reference_*/` holds the **upstream baseline**. `our_*/` holds **the same
experiment run through `core/p06_training/`** for apples-to-apples comparison.

## Setup (once)

Same venv as the other reference folders:

```bash
bash scripts/setup-notebook-venv.sh
```

> **⚠️ CRITICAL:** Reference `.py` scripts run in `.venv-notebook/`, NOT
> the main `.venv/`. Always invoke via `.venv-notebook/bin/python ...`.

The sidewalk-semantic dataset is gated on the Hub. Set `HF_TOKEN` in the
repo's `.env` (loaded automatically by the launch scripts) or export it
manually before running.

## Run

```bash
# Reference
.venv-notebook/bin/python \
  notebooks/segformer_finetune_reference/reference_segformer_b0/finetune.py --seed 42

# Ours (HF backend)
uv run core/p06_training/train.py \
  --config notebooks/segformer_finetune_reference/our_segformer_b0/06_training.yaml
```

## Upstream references

- HF cookbook notebook:
  https://github.com/huggingface/notebooks/blob/main/examples/semantic_segmentation.ipynb
- HF blog post: https://huggingface.co/blog/fine-tune-segformer
- SegFormer-B0 checkpoint: https://huggingface.co/nvidia/mit-b0
- sidewalk-semantic (gated): https://huggingface.co/datasets/segments/sidewalk-semantic

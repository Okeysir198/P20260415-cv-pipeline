# reference_segformer_b0 — HF SegFormer semantic-segmentation reference

Runnable `.py` port of HF's semantic-segmentation notebook, plus the frozen
upstream `.ipynb` in the same folder. Used as the known-good baseline to diff
against our in-repo semantic-segmentation training path.

## Contents

| File | Purpose |
|---|---|
| `semantic_segmentation.ipynb` | Upstream original — frozen reference, never edited. Download command below. |
| `finetune.py` | SegFormer-B0 fine-tune on `segments/sidewalk-semantic-2`. Takes `--seed`, `--tag`, `--epochs`, `--output-dir`. Writes to `runs/seed{SEED}/` (or `runs/{TAG}_seed{SEED}/`). |
| `inference.py` | Post-training viz — loads best checkpoint, overlays pred + GT masks for N val samples in a single PNG. |

Upstream sources:
- Notebook: <https://github.com/huggingface/notebooks/blob/main/examples/semantic_segmentation.ipynb>
- Blog post: <https://huggingface.co/blog/fine-tune-segformer>

To re-download the upstream `.ipynb` (frozen — do not edit):
```bash
curl -L -o semantic_segmentation.ipynb \
  https://raw.githubusercontent.com/huggingface/notebooks/main/examples/semantic_segmentation.ipynb
```

## Run

```bash
# Step 1 — train (wall-clock pending full run; upstream ran on Colab T4)
.venv-notebook/bin/python \
  notebooks/segformer_finetune_reference/reference_segformer_b0/finetune.py \
  --seed 42

# Step 2 — val-prediction grid
.venv-notebook/bin/python \
  notebooks/segformer_finetune_reference/reference_segformer_b0/inference.py \
  --run-dir notebooks/segformer_finetune_reference/reference_segformer_b0/runs/seed42 \
  --n 16
```

Headline numbers: **pending full run — see CLAUDE.md.**

## Hyperparameter recipe (upstream, baked into `finetune.py`)

| Knob | Value | Note |
|---|---|---|
| `model` | `nvidia/mit-b0` | SegFormer-B0 |
| `dataset` | `segments/sidewalk-semantic-2` | public variant (non-gated) |
| `epochs` | 50 | upstream default |
| `lr` | 6e-5 | |
| `per_device_train_batch_size` | 2 | |
| `per_device_eval_batch_size` | 2 | |
| augment | `ColorJitter(0.25, 0.25, 0.25, 0.1)` (train only) | torchvision |
| normalize | ImageNet mean/std via `SegformerImageProcessor` | |
| metric | `evaluate.load("mean_iou")` with `ignore_index=0` | index 0 = unlabeled |
| `push_to_hub` | False | reference port is local-only |
| `report_to` | none | no wandb/tb auth required (see top-level CLAUDE.md) |

## Conversion notes vs `.ipynb`

Applied edits when porting to runnable `.py`:
- Shell installs (`!pip`, `!git lfs install`) stripped.
- `notebook_login()` + `push_to_hub=True` + `trainer.push_to_hub(...)` removed.
- `display(...)` calls commented out.
- `SegformerFeatureExtractor` → `SegformerImageProcessor` (upstream alias is deprecated).
- `feature_extractor.reduce_labels` → `feature_extractor.do_reduce_labels`
  (attribute rename in current transformers).
- Dataset swapped from gated `segments/sidewalk-semantic` to the public
  `segments/sidewalk-semantic-2` so the script runs anonymously.

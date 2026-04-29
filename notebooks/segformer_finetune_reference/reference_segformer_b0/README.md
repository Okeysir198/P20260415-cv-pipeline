# reference_segformer_b0 — HF SegFormer semantic-segmentation reference

Runnable `.py` port of HF's semantic-segmentation notebook, plus the frozen
upstream `.ipynb` in the same folder. Used as the known-good baseline to diff
against our in-repo semantic-segmentation training path.

## Contents

| File | Purpose |
|---|---|
| `semantic_segmentation.ipynb` | Upstream original — frozen reference, never edited. Download command below. |
| `finetune.py` | SegFormer-B0 fine-tune on `segments/sidewalk-semantic` (gated — needs HF_TOKEN). Takes `--seed`, `--tag`, `--epochs`, `--output-dir`, `--batch-size`, `--num-workers`, `--bf16`. Writes to `runs/seed{SEED}/` (or `runs/{TAG}_seed{SEED}/`). |
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

Headline numbers (seed42, 50 epochs, bs=8, bf16, fp32 cookbook recipe otherwise unchanged):

| Metric | Value | Step / Epoch |
|---|---|---|
| best `eval_loss` (drives `load_best_model_at_end`) | 0.604 | step 1800 / ep 36 |
| `eval_mean_iou` at best ckpt | 0.310 | step 1800 / ep 36 |
| `eval_overall_accuracy` at best ckpt | 0.842 | step 1800 / ep 36 |
| final `eval_mean_iou` | 0.311 | step 2500 / ep 50 |

Note: `metric_for_best_model` is unset, so HF Trainer selects on `eval_loss`,
not mIoU. Set `metric_for_best_model="mean_iou", greater_is_better=True` if
you want mIoU-driven checkpoint selection.

## Hyperparameter recipe (upstream, baked into `finetune.py`)

| Knob | Value | Note |
|---|---|---|
| `model` | `nvidia/mit-b0` | SegFormer-B0 |
| `dataset` | `segments/sidewalk-semantic` | gated — requires HF_TOKEN |
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
- Dataset kept as gated `segments/sidewalk-semantic` (matches CLAUDE.md;
  `runs/seed42/` was trained on this). Set `HF_TOKEN` in the env before running.

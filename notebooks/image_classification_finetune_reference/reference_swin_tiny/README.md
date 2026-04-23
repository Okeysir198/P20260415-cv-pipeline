# reference_swin_tiny — HF cookbook Swin-tiny on EuroSAT

Byte-for-byte port of the upstream HF image-classification cookbook.

- **Upstream notebook**: <https://github.com/huggingface/notebooks/blob/main/examples/image_classification.ipynb>
- **Checkpoint**: `microsoft/swin-tiny-patch4-window7-224`
- **Dataset**: `jonathan-roberts1/EuroSAT` (10-class land cover)
- **Hyperparams** (preserved): `lr=5e-5`, `epochs=3`, `batch_size=32`,
  `gradient_accumulation_steps=4`, `warmup_ratio=0.1`,
  `metric_for_best_model="accuracy"`.

## Files

| Path | Purpose |
|---|---|
| `image_classification.ipynb` | Frozen upstream notebook (unmodified). |
| `finetune.py` | `.py` port of the notebook's training cells. Seeded + deterministic, writes to `runs/seed<SEED>/`. |
| `inference.py` | Loads `runs/seed<SEED>/best/`, runs on the val split, writes `val_report/{confusion_matrix.png, top_misclassifications.png, report.json}`. |

## Run

```bash
# Reference script (uses .venv-notebook/ — NOT `uv run`)
.venv-notebook/bin/python \
  notebooks/image_classification_finetune_reference/reference_swin_tiny/finetune.py \
  --seed 42

# Post-train error analysis on the val split
.venv-notebook/bin/python \
  notebooks/image_classification_finetune_reference/reference_swin_tiny/inference.py \
  --seed 42
```

All artefacts land under `runs/seed42/` (self-contained, resolved via `__file__`).

## Conversion notes vs upstream notebook

- `!pip`, `!sudo apt`, `!git config` shell installs stripped.
- `notebook_login()` + `push_to_hub=True` removed (local reproduction only).
- Jupyter-only `display()` / image-preview cells dropped.
- `report_to="none"` — avoids the HF-Trainer-hard-fails-without-wandb-login
  footgun documented in `../../detr_finetune_reference/CLAUDE.md`.
- Split `train_test_split(test_size=0.1, seed=SEED)` is seeded (upstream omitted
  the seed; we fix it so runs are reproducible).

## Pending results

Training not run yet under this worktree. Expected upstream ballpark
(3 epochs, seed=42): val accuracy ~0.98 on EuroSAT's 10-class split.
Fill in after the first run alongside `val_report/report.json`.

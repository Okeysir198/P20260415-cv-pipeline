# our_swin_tiny — Swin-tiny on EuroSAT via our in-repo pipeline

Companion to `../reference_swin_tiny/` (qubvel/nielsr's HF image-classification
cookbook, byte-for-byte port). Purpose: run the same recipe (model, dataset,
hyperparams) through `core/p06_training/train.py --backend hf` so we can
diff our pipeline against a known-good baseline.

## Run

```bash
uv run core/p06_training/train.py \
  --config notebooks/image_classification_finetune_reference/our_swin_tiny/06_training.yaml
```

All artefacts land under `runs/seed42/` — the standard HF-Trainer layout
(`checkpoint-*/`, `runs/<ts>_<host>/` tensorboard placeholders,
`trainer_state.json`) plus our 3-axis observability tree (data_preview/,
val_predictions/, test_predictions/) written by the HF-backend callbacks.

## Invariants (vs `../reference_swin_tiny/`)

| Item | Value |
|---|---|
| Checkpoint | `microsoft/swin-tiny-patch4-window7-224` |
| Dataset | EuroSAT (10-class), ImageFolder at `dataset_store/training_ready/eurosat/` |
| Input size | 224×224 |
| Epochs | 3 |
| LR | 5e-5 |
| Scheduler | linear, `warmup_ratio=0.1` |
| Batch size | 32 (× 4 grad-accum = effective 128) |
| Train aug | RandomResizedCrop + RandomHorizontalFlip |
| Val transform | Resize + CenterCrop |
| Normalize | ImageNet (mean/std from Swin image_processor) |
| Precision | fp32 (upstream + our recipe — Swin is not bf16-tuned here) |
| best-metric | `accuracy` |

## Pending results

Training not run yet under this worktree. Expected upstream ballpark
(3 epochs, seed=42): val accuracy ~0.98 on EuroSAT's 10-class split.
Fill in after the first run.

## Gotchas already avoided

- No `logging.run_name` key — HF Trainer would use it as the feature
  folder name and create ghost dirs.
- `logging.report_to: none` — HF Trainer's wandb callback hard-fails
  without `wandb login`.
- `data.names` is the 10-class EuroSAT dict in `05_data.yaml` —
  `core/p06_models/hf_model.py::build_hf_model` reads it to populate
  `id2label` / `label2id`, which HF image-classification heads need for
  correct class-embedding initialisation.

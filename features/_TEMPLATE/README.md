# <feature_name>

One-line purpose (what does this feature detect / classify / predict?).

| Field | Value |
|-------|-------|
| Task | Detection / Classification / Segmentation / Keypoint |
| Recommended model | e.g. YOLOX-M, D-FINE-S, timm-mobilenetv3 |
| Classes | `class_a`, `class_b` |
| Dataset | `dataset_store/training_ready/<dataset_name>/` |

## Layout

```
features/<name>/
  configs/        phase YAMLs (00_data_preparation, 05_data, 06_training,
                  08_evaluation, 09_export, 10_inference)
  code/           OPTIONAL custom code (custom_trainer.py, train.py, …)
  samples/        small smoke-test images & clips (tracked)
  notebooks/      exploration notebooks
  tests/          per-feature pytest (optional)
  runs/           training checkpoints (gitignored; DVC-tracked best.pt)
  eval/           evaluation reports (gitignored)
  export/         exported ONNX/TFLite (gitignored)
  predict/        inference outputs (gitignored)
  release/        versioned deploy bundles (v<semver>/, symlink latest/)
```

## Commands

```bash
cd edge_ai/ai

# Train (core pipeline)
uv run python core/p06_training/train.py \
  --config features/<name>/configs/06_training.yaml

# Evaluate
uv run python core/p08_evaluation/evaluate.py \
  --model features/<name>/runs/best.pt \
  --config features/<name>/configs/05_data.yaml

# Export
uv run python core/p09_export/export.py \
  --model features/<name>/runs/best.pt \
  --training-config features/<name>/configs/06_training.yaml

# Inference with alert logic
uv run python core/p10_inference/video.py \
  --config features/<name>/configs/10_inference.yaml \
  --video features/<name>/samples/demo.mp4
```

## Custom code (optional)

When the stock `core/` pipeline isn't enough, drop code into `code/` and
reference it from your config:

```yaml
# features/<name>/configs/06_training.yaml
training:
  backend: custom
  custom_trainer_class: features.<name>.code.custom_trainer.MyTrainer
```

See `ai/README.md` → "Custom feature code" for the three escape hatches.

## Release

```
release/
  v0.1.0/
    model.onnx       # DVC-tracked
    config.yaml      # deploy config snapshot (tracked)
    classes.txt      # class names (tracked)
    metrics.json     # eval scores at release (tracked)
    MANIFEST.md      # git sha, dataset hash, notes (tracked)
  latest -> v0.1.0
```

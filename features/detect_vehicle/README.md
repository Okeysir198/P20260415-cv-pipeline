# Detect Vehicle

Generic vehicle detection (car / bus / truck / motorcycle / van).

| Field | Value |
|-------|-------|
| Task | Detection |
| Recommended model | YOLOX-M |
| Classes | `car`, `bus`, `truck`, `motorcycle`, `van` |
| Dataset | `dataset_store/training_ready/vehicle_detection/` |
| Dataset report | `../../dataset_store/training_ready/vehicle_detection/DATASET_REPORT.md` |

## Layout

```
features/detect_vehicle/
  configs/     phase YAMLs (00_data_preparation, 05_data, 06_training,
               08_evaluation, 09_export, 10_inference)
  code/        OPTIONAL custom code (custom_trainer.py, train.py, …)
  samples/     small smoke-test images & clips (tracked)
  notebooks/   exploration notebooks
  tests/       per-feature pytest (optional)
  runs/        training checkpoints (gitignored; DVC-tracked best.pt)
  eval/        evaluation reports (gitignored)
  export/      exported ONNX / TFLite (gitignored)
  predict/     inference outputs (gitignored)
  release/     versioned deploy bundles (v<semver>/, latest/ symlink)
```

## Commands

```bash
cd edge_ai/ai

# Train
uv run python core/p06_training/train.py \
  --config features/detect_vehicle/configs/06_training.yaml

# Evaluate
uv run python core/p08_evaluation/evaluate.py \
  --model features/detect_vehicle/runs/best.pt \
  --config features/detect_vehicle/configs/05_data.yaml

# Export to ONNX
uv run python core/p09_export/export.py \
  --model features/detect_vehicle/runs/best.pt \
  --training-config features/detect_vehicle/configs/06_training.yaml

# Inference with alert logic
uv run python core/p10_inference/video.py \
  --config features/detect_vehicle/configs/10_inference.yaml \
  --video features/detect_vehicle/samples/demo.mp4
```

## Custom code (optional)

When the stock `core/` pipeline isn't enough, drop code into `code/` and
reference it from your config:

```yaml
# features/detect_vehicle/configs/06_training.yaml
training:
  backend: custom
  custom_trainer_class: features.detect_vehicle.code.custom_trainer.MyTrainer
```

See `ai/CLAUDE.md` → "Custom feature code" for the three escape hatches.

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

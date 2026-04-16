# p03_training — Training Loop + Losses + Callbacks

## Purpose

Config-driven training with automatic model type detection, loss computation, checkpointing, and learning rate scheduling.

## Files

| File | Purpose |
|---|---|
| `trainer.py` | `DetectionTrainer` — main training loop with `ModelEMA`, gradient clipping, auto-detection of HF vs custom model path |
| `train.py` | CLI entry point (`--config`, `--override`, `--resume`) |
| `losses.py` | `DetectionLoss` ABC, `YOLOXLoss` (SimOTA assignment), `FocalLoss`, `IoULoss`, `_DETRPassthroughLoss`, loss registry (`build_loss`) |
| `lr_scheduler.py` | `WarmupScheduler`, `CosineScheduler`, `PlateauScheduler`, `StepScheduler`, `OneCycleScheduler`, `build_scheduler()` — all with linear warmup support |
| `callbacks.py` | `Callback` base class, `CheckpointSaver`, `EarlyStopping`, `WandBLogger`, `CallbackRunner` |
| `postprocess.py` | `POSTPROCESSOR_REGISTRY`, `register_postprocessor()`, `postprocess()` — YOLOX-only output decoding (HF models use built-in `post_process_object_detection`) |
| `metrics_registry.py` | `METRICS_REGISTRY`, `register_metrics()`, `compute_metrics()` — per-format validation metrics dispatch |

## Two Training Paths

The trainer auto-detects which path to use:

1. **Custom models (YOLOX)**: `model.forward()` → external `loss_fn()` → backprop
2. **HF models (D-FINE, RT-DETRv2)**: `model.forward_with_loss()` → built-in loss → backprop

## Adding a New Loss Function

```python
from core.p03_training.losses import DetectionLoss, register_loss

@register_loss("my_loss")
class MyLoss(DetectionLoss):
    def forward(self, predictions, targets):
        # Return (total_loss, {"cls_loss": ..., "reg_loss": ...})
        ...
```

## CLI

```bash
uv run core/p06_training/train.py --config features/safety-fire_detection/configs/06_training.yaml
uv run core/p06_training/train.py --config features/safety-fire_detection/configs/06_training.yaml --override training.lr=0.005
uv run core/p06_training/train.py --config features/safety-fire_detection/configs/06_training.yaml --resume runs/fire_detection/last.pth
```

## Config Reference

- `features/<name>/configs/06_training.yaml` — optimizer, scheduler, epochs, EMA, gradient clipping, augmentation, loss settings

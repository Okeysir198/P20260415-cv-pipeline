"""Training pipeline: trainer, losses, schedulers, callbacks, and metrics."""

from core.p06_training import metrics_registry  # noqa: F401 — triggers @register_metrics decorators
from core.p06_training.trainer import DetectionTrainer, ModelEMA
from core.p06_training.hf_trainer import train_with_hf
from core.p06_training.losses import LOSS_REGISTRY, build_loss
from core.p06_training.lr_scheduler import build_scheduler
from core.p06_training.postprocess import POSTPROCESSOR_REGISTRY, postprocess
from core.p06_training.callbacks import (
    CallbackRunner,
    CheckpointSaver,
    EarlyStopping,
    WandBLogger,
)

__all__ = [
    "DetectionTrainer",
    "ModelEMA",
    "train_with_hf",
    "LOSS_REGISTRY",
    "build_loss",
    "build_scheduler",
    "POSTPROCESSOR_REGISTRY",
    "postprocess",
    "CallbackRunner",
    "CheckpointSaver",
    "EarlyStopping",
    "WandBLogger",
]

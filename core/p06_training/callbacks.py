"""Training callbacks for checkpointing, early stopping, and logging.

Provides:
- Callback: Base class with training lifecycle hooks.
- CheckpointSaver: Save best and periodic model checkpoints.
- EarlyStopping: Stop training when metric plateaus.
- WandBLogger: Log metrics, config, and artifacts to Weights & Biases.
- CallbackRunner: Manages and invokes a list of callbacks.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

logger = logging.getLogger(__name__)


class Callback:
    """Base callback with training lifecycle hooks.

    Override any of these methods to inject custom behavior at
    specific points in the training loop.
    """

    def on_train_start(self, trainer: Any) -> None:
        """Called once at the beginning of training.

        Args:
            trainer: The DetectionTrainer instance.
        """

    def on_train_end(self, trainer: Any) -> None:
        """Called once at the end of training.

        Args:
            trainer: The DetectionTrainer instance.
        """

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        """Called at the beginning of each epoch.

        Args:
            trainer: The DetectionTrainer instance.
            epoch: Current epoch number (0-indexed).
        """

    def on_epoch_end(
        self, trainer: Any, epoch: int, metrics: Dict[str, float]
    ) -> None:
        """Called at the end of each epoch.

        Args:
            trainer: The DetectionTrainer instance.
            epoch: Current epoch number (0-indexed).
            metrics: Dictionary of epoch metrics (train + val).
        """

    def on_batch_end(
        self, trainer: Any, batch_idx: int, metrics: Dict[str, float]
    ) -> None:
        """Called at the end of each training batch.

        Args:
            trainer: The DetectionTrainer instance.
            batch_idx: Current batch index within the epoch.
            metrics: Dictionary of batch-level metrics.
        """


class CheckpointSaver(Callback):
    """Save model checkpoints: best model and periodic saves.

    Checkpoints include model state dict, optimizer state, scheduler state,
    epoch number, and metrics for seamless training resumption.

    Args:
        save_dir: Directory to save checkpoints.
        metric: Metric key to track for best model (e.g. "val/mAP50").
        mode: "max" if higher metric is better, "min" if lower. Default: "max".
        save_interval: Save periodic checkpoint every N epochs. 0 to disable.
            Default: 10.
        save_best: Whether to save the best model checkpoint. Default: True.
    """

    def __init__(
        self,
        save_dir: str,
        metric: str = "val/mAP50",
        mode: str = "max",
        save_interval: int = 10,
        save_best: bool = True,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.metric = metric
        self.mode = mode
        self.save_interval = save_interval
        self.save_best = save_best

        self._best_value: Optional[float] = None
        self._best_epoch: int = 0

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_train_start(self, trainer: Any) -> None:
        """Copy training + data configs to the run directory for reproducibility."""
        import shutil
        import yaml

        config_path = getattr(trainer, "config_path", None)

        # Copy the original training YAML (named to match releases/ convention)
        if config_path is not None and Path(config_path).exists():
            shutil.copy2(config_path, self.save_dir / "06_training.yaml")
            logger.info("Saved training config to %s", self.save_dir / "06_training.yaml")

        # Copy the data config YAML (resolved from dataset_config reference)
        data_cfg = getattr(trainer, "_data_cfg", None)
        if data_cfg and config_path is not None:
            dataset_config_ref = data_cfg.get("dataset_config")
            if dataset_config_ref:
                data_config_path = Path(dataset_config_ref)
                if not data_config_path.is_absolute():
                    data_config_path = (Path(config_path).parent / dataset_config_ref).resolve()
                if data_config_path.exists():
                    shutil.copy2(data_config_path, self.save_dir / "05_data.yaml")
                    logger.info("Saved data config to %s", self.save_dir / "05_data.yaml")

        # Dump the resolved config (with CLI overrides applied)
        config = getattr(trainer, "config", None)
        if config is not None:
            resolved_path = self.save_dir / "config_resolved.yaml"
            with open(resolved_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def on_epoch_end(
        self, trainer: Any, epoch: int, metrics: Dict[str, float]
    ) -> None:
        """Save checkpoints at epoch end.

        Args:
            trainer: The DetectionTrainer instance.
            epoch: Current epoch number (0-indexed).
            metrics: Dictionary of epoch metrics.
        """
        checkpoint = self._build_checkpoint(trainer, epoch, metrics)

        # Save best model
        if self.save_best and self.metric in metrics:
            current_value = metrics[self.metric]
            is_best = False

            if self._best_value is None:
                is_best = True
            elif self.mode == "max" and current_value > self._best_value:
                is_best = True
            elif self.mode == "min" and current_value < self._best_value:
                is_best = True

            if is_best:
                self._best_value = current_value
                self._best_epoch = epoch
                best_path = self.save_dir / "best.pth"
                torch.save(checkpoint, best_path)
                logger.info(
                    "Saved best model: %s=%.4f at epoch %d -> %s",
                    self.metric, current_value, epoch + 1, best_path,
                )

        # Save periodic checkpoint
        if self.save_interval > 0 and (epoch + 1) % self.save_interval == 0:
            periodic_path = self.save_dir / f"epoch_{epoch + 1}.pth"
            torch.save(checkpoint, periodic_path)
            logger.info("Saved periodic checkpoint: epoch %d -> %s", epoch + 1, periodic_path)

        # Always save last checkpoint for resume
        last_path = self.save_dir / "last.pth"
        torch.save(checkpoint, last_path)

    @staticmethod
    def _build_checkpoint(
        trainer: Any, epoch: int, metrics: Dict[str, float]
    ) -> dict:
        """Build a checkpoint dictionary from trainer state.

        Args:
            trainer: The DetectionTrainer instance.
            epoch: Current epoch number.
            metrics: Current epoch metrics.

        Returns:
            Checkpoint dictionary.
        """
        checkpoint: Dict[str, Any] = {
            "epoch": epoch,
            "metrics": metrics,
            "config": trainer.config,
        }

        # Model architecture metadata for reproducibility
        if hasattr(trainer, "_model_cfg") and trainer._model_cfg is not None:
            checkpoint["model_arch"] = {
                "depth": trainer._model_cfg.get("depth"),
                "width": trainer._model_cfg.get("width"),
                "num_classes": trainer._model_cfg.get("num_classes"),
                "arch": trainer._model_cfg.get("arch"),
                "input_size": trainer._model_cfg.get("input_size"),
            }
        # Output format for postprocessor dispatch
        if hasattr(trainer, "model") and trainer.model is not None:
            base = trainer.model
            if isinstance(base, nn.DataParallel):
                base = base.module
            checkpoint["output_format"] = getattr(base, "output_format", "yolox")

        if hasattr(trainer, "_data_cfg") and trainer._data_cfg is not None:
            checkpoint["dataset_name"] = trainer._data_cfg.get(
                "dataset_config", trainer._data_cfg.get("dataset_name", "unknown")
            )

        # Model state
        if hasattr(trainer, "model") and trainer.model is not None:
            model = trainer.model
            if isinstance(model, nn.DataParallel):
                checkpoint["model_state_dict"] = model.module.state_dict()
            else:
                checkpoint["model_state_dict"] = model.state_dict()

        # Optimizer state
        if hasattr(trainer, "optimizer") and trainer.optimizer is not None:
            checkpoint["optimizer_state_dict"] = trainer.optimizer.state_dict()

        # Scheduler state
        if hasattr(trainer, "scheduler") and trainer.scheduler is not None:
            if hasattr(trainer.scheduler, "state_dict"):
                checkpoint["scheduler_state_dict"] = trainer.scheduler.state_dict()

        # Scaler state (AMP)
        if hasattr(trainer, "scaler") and trainer.scaler is not None:
            checkpoint["scaler_state_dict"] = trainer.scaler.state_dict()

        # EMA state
        if hasattr(trainer, "ema") and trainer.ema is not None:
            checkpoint["ema_state_dict"] = trainer.ema.state_dict()

        # RNG states for exact reproducibility on resume
        import random as _random

        import numpy as _np

        checkpoint["rng_states"] = {
            "python": _random.getstate(),
            "numpy": _np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
        }

        # Callback states for correct resume behavior
        callback_runner = getattr(trainer, "callback_runner", None)
        if callback_runner is not None:
            cb_states = {}
            ckpt_cb = callback_runner.get_callback(CheckpointSaver)
            if ckpt_cb is not None:
                cb_states["checkpoint_saver"] = {
                    "best_value": ckpt_cb._best_value,
                    "best_epoch": ckpt_cb._best_epoch,
                }
            es_cb = callback_runner.get_callback(EarlyStopping)
            if es_cb is not None:
                cb_states["early_stopping"] = {
                    "best_value": es_cb._best_value,
                    "counter": es_cb._counter,
                }
            if cb_states:
                checkpoint["callback_states"] = cb_states

        return checkpoint


class EarlyStopping(Callback):
    """Stop training when a monitored metric stops improving.

    Args:
        metric: Metric key to monitor (e.g. "val/mAP50").
        mode: "max" if higher is better, "min" if lower. Default: "max".
        patience: Number of epochs to wait for improvement. Default: 50.
        min_delta: Minimum change to qualify as improvement. Default: 0.0.
    """

    def __init__(
        self,
        metric: str = "val/mAP50",
        mode: str = "max",
        patience: int = 50,
        min_delta: float = 0.0,
    ) -> None:
        self.metric = metric
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta

        self._best_value: Optional[float] = None
        self._counter: int = 0
        self._best_epoch: int = 0
        self.should_stop: bool = False

    def on_epoch_end(
        self, trainer: Any, epoch: int, metrics: Dict[str, float]
    ) -> None:
        """Check if training should stop.

        Sets self.should_stop = True if patience is exhausted.

        Args:
            trainer: The DetectionTrainer instance.
            epoch: Current epoch number (0-indexed).
            metrics: Dictionary of epoch metrics.
        """
        if self.metric not in metrics:
            return

        current_value = metrics[self.metric]
        improved = False

        if self._best_value is None:
            improved = True
        elif self.mode == "max" and current_value > self._best_value + self.min_delta:
            improved = True
        elif self.mode == "min" and current_value < self._best_value - self.min_delta:
            improved = True

        if improved:
            self._best_value = current_value
            self._best_epoch = epoch
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "Early stopping triggered at epoch %d. "
                    "Best %s=%.4f at epoch %d. No improvement for %d epochs.",
                    epoch + 1, self.metric, self._best_value,
                    self._best_epoch + 1, self.patience,
                )


class WandBLogger(Callback):
    """Log training metrics, config, and artifacts to Weights & Biases.

    Handles wandb initialization, per-epoch metric logging, and cleanup.
    Gracefully degrades if wandb is not installed or login fails.

    Args:
        project: W&B project name. Default: "smart-camera".
        run_name: Name for this training run. Default: None (auto-generated).
        config: Full training config dict to log. Default: None.
        log_interval: Log batch metrics every N batches. 0 for epoch-only. Default: 0.
        tags: Optional list of tags for the run.
    """

    def __init__(
        self,
        project: str = "smart-camera",
        run_name: Optional[str] = None,
        config: Optional[dict] = None,
        log_interval: int = 0,
        tags: Optional[List[str]] = None,
    ) -> None:
        self.project = project
        self.run_name = run_name
        self._config = config
        self.log_interval = log_interval
        self.tags = tags
        self._run = None
        self._wandb = None
        self._enabled = False

    def on_train_start(self, trainer: Any) -> None:
        """Initialize W&B run.

        Args:
            trainer: The DetectionTrainer instance.
        """
        try:
            self._wandb = wandb
            self._run = wandb.init(
                project=self.project,
                name=self.run_name,
                config=self._config or trainer.config,
                tags=self.tags,
                reinit="finish_previous",
            )
            self._enabled = True
            logger.info("W&B logging enabled: project=%s, run=%s", self.project, self.run_name)
        except Exception as e:
            logger.warning("Failed to initialize W&B: %s. Continuing without logging.", e)

    def on_epoch_end(
        self, trainer: Any, epoch: int, metrics: Dict[str, float]
    ) -> None:
        """Log epoch-level metrics to W&B.

        Args:
            trainer: The DetectionTrainer instance.
            epoch: Current epoch number (0-indexed).
            metrics: Dictionary of epoch metrics.
        """
        if not self._enabled:
            return

        # Add epoch and learning rate info
        log_data = {"epoch": epoch + 1, **metrics}

        if hasattr(trainer, "optimizer") and trainer.optimizer is not None:
            log_data["lr"] = trainer.optimizer.param_groups[0]["lr"]

        self._wandb.log(log_data, step=epoch + 1)

    def on_batch_end(
        self, trainer: Any, batch_idx: int, metrics: Dict[str, float]
    ) -> None:
        """Log batch-level metrics to W&B (if log_interval > 0).

        Args:
            trainer: The DetectionTrainer instance.
            batch_idx: Current batch index within the epoch.
            metrics: Dictionary of batch-level metrics.
        """
        if not self._enabled or self.log_interval <= 0:
            return

        if (batch_idx + 1) % self.log_interval == 0:
            self._wandb.log(metrics)

    def on_train_end(self, trainer: Any) -> None:
        """Finalize and close the W&B run.

        Args:
            trainer: The DetectionTrainer instance.
        """
        if self._enabled and self._run is not None:
            self._run.finish()
            logger.info("W&B run finished.")
            self._enabled = False


class CallbackRunner:
    """Manages and invokes a list of training callbacks.

    Iterates through registered callbacks and calls the appropriate
    hook method on each.

    Args:
        callbacks: List of Callback instances.
    """

    def __init__(self, callbacks: Optional[List[Callback]] = None) -> None:
        self.callbacks: List[Callback] = callbacks or []

    def add(self, callback: Callback) -> None:
        """Register a new callback.

        Args:
            callback: Callback instance to add.
        """
        self.callbacks.append(callback)

    def on_train_start(self, trainer: Any) -> None:
        """Fire on_train_start for all callbacks.

        Args:
            trainer: The DetectionTrainer instance.
        """
        for cb in self.callbacks:
            cb.on_train_start(trainer)

    def on_train_end(self, trainer: Any) -> None:
        """Fire on_train_end for all callbacks.

        Args:
            trainer: The DetectionTrainer instance.
        """
        for cb in self.callbacks:
            cb.on_train_end(trainer)

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        """Fire on_epoch_start for all callbacks.

        Args:
            trainer: The DetectionTrainer instance.
            epoch: Current epoch number.
        """
        for cb in self.callbacks:
            cb.on_epoch_start(trainer, epoch)

    def on_epoch_end(
        self, trainer: Any, epoch: int, metrics: Dict[str, float]
    ) -> None:
        """Fire on_epoch_end for all callbacks.

        Args:
            trainer: The DetectionTrainer instance.
            epoch: Current epoch number.
            metrics: Dictionary of epoch metrics.
        """
        for cb in self.callbacks:
            cb.on_epoch_end(trainer, epoch, metrics)

    def on_batch_end(
        self, trainer: Any, batch_idx: int, metrics: Dict[str, float]
    ) -> None:
        """Fire on_batch_end for all callbacks.

        Args:
            trainer: The DetectionTrainer instance.
            batch_idx: Current batch index.
            metrics: Dictionary of batch-level metrics.
        """
        for cb in self.callbacks:
            cb.on_batch_end(trainer, batch_idx, metrics)

    def get_callback(self, callback_type: type) -> Optional[Callback]:
        """Find a callback by type.

        Args:
            callback_type: The class type to search for.

        Returns:
            First matching callback instance, or None.
        """
        for cb in self.callbacks:
            if isinstance(cb, callback_type):
                return cb
        return None

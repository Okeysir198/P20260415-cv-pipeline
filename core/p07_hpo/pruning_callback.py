"""Optuna pruning callback for detection model training integration."""

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

import optuna

from loguru import logger
from core.p06_training.callbacks import Callback


class OptunaPruningCallback(Callback):
    """Callback that reports metrics to Optuna and prunes unpromising trials.

    Integrates with the training loop via the Callback interface.
    At each epoch end, reports the tracked metric to the Optuna trial and
    raises TrialPruned if the trial should be stopped early.

    Args:
        trial: The Optuna trial object.
        metric: Metric key to report (e.g. "val/mAP50").
        warmup_epochs: Number of initial epochs to skip before pruning.
            Default: 10.
    """

    def __init__(
        self,
        trial: "optuna.Trial",
        metric: str = "val/mAP50",
        warmup_epochs: int = 10,
    ) -> None:
        self.trial = trial
        self.metric = metric
        self.warmup_epochs = warmup_epochs

    def on_epoch_end(
        self, trainer: Any, epoch: int, metrics: dict[str, float]
    ) -> None:
        """Report metric to Optuna and check for pruning.

        Args:
            trainer: The DetectionTrainer instance.
            epoch: Current epoch number (0-indexed).
            metrics: Dictionary of epoch metrics.

        Raises:
            optuna.TrialPruned: If the trial should be pruned.
        """
        value = metrics.get(self.metric)
        if value is None:
            logger.warning(
                "Metric '%s' not found in epoch %d metrics. "
                "Available: %s. Skipping report.",
                self.metric, epoch + 1, list(metrics.keys()),
            )
            return

        # Report to Optuna
        self.trial.report(value, epoch)

        # Check pruning (skip during warmup)
        if epoch >= self.warmup_epochs and self.trial.should_prune():
            logger.info(
                "Trial %d pruned at epoch %d (%s=%.4f).",
                self.trial.number, epoch + 1, self.metric, value,
            )
            raise optuna.TrialPruned(
                f"Trial {self.trial.number} pruned at epoch {epoch + 1} "
                f"({self.metric}={value:.4f})"
            )

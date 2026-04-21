"""Progress tracking utilities for training and evaluation loops."""

import time
from typing import Any

from tqdm import tqdm


def _fmt(v: Any) -> Any:
    return f"{v:.4f}" if isinstance(v, float) else v


class ProgressBar:
    """Thin wrapper around tqdm for consistent progress bar styling.

    Provides a unified interface for iteration progress with optional
    metrics display (loss, mAP, etc.).

    Args:
        total: Total number of iterations.
        desc: Description prefix for the progress bar.
        unit: Unit name for iterations (default: "batch").
        leave: Whether to keep the progress bar after completion.
        ncols: Fixed width of the progress bar. None for auto.

    Examples:
        >>> pbar = ProgressBar(total=100, desc="Training")
        >>> for batch in dataloader:
        ...     loss = train_step(batch)
        ...     pbar.update(metrics={"loss": loss})
        >>> pbar.close()
    """

    def __init__(
        self,
        total: int,
        desc: str = "",
        unit: str = "batch",
        leave: bool = True,
        ncols: int | None = None,
    ) -> None:
        self._bar = tqdm(
            total=total,
            desc=desc,
            unit=unit,
            leave=leave,
            ncols=ncols,
            bar_format="{l_bar}{bar:30}{r_bar}",
        )

    def update(self, n: int = 1, metrics: dict[str, float] | None = None) -> None:
        """Advance the progress bar and optionally display metrics.

        Args:
            n: Number of steps to advance.
            metrics: Dictionary of metric name -> value to display as postfix.
        """
        if metrics:
            self._bar.set_postfix({k: _fmt(v) for k, v in metrics.items()})
        self._bar.update(n)

    def set_description(self, desc: str) -> None:
        """Update the progress bar description.

        Args:
            desc: New description string.
        """
        self._bar.set_description(desc)

    def close(self) -> None:
        """Close the progress bar."""
        self._bar.close()

    def __enter__(self) -> "ProgressBar":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class TrainingProgress:
    """Multi-epoch training progress tracker.

    Tracks epoch-level and batch-level progress, elapsed time,
    and best metric values across the full training run.

    Args:
        total_epochs: Total number of training epochs.
        batches_per_epoch: Number of batches per epoch (for nested progress).

    Examples:
        >>> tracker = TrainingProgress(total_epochs=200, batches_per_epoch=500)
        >>> for epoch in range(200):
        ...     tracker.start_epoch(epoch)
        ...     for batch in dataloader:
        ...         loss = train_step(batch)
        ...         tracker.update_batch(metrics={"loss": loss})
        ...     tracker.end_epoch(metrics={"val/mAP50": 0.85, "val/loss": 0.3})
        >>> tracker.close()
    """

    def __init__(self, total_epochs: int, batches_per_epoch: int = 0) -> None:
        self.total_epochs = total_epochs
        self.batches_per_epoch = batches_per_epoch
        self._start_time = time.time()
        self._epoch_start_time: float = 0.0
        self._current_epoch: int = 0
        self._best_metric: float | None = None
        self._best_epoch: int = 0

        self._epoch_bar = tqdm(
            total=total_epochs,
            desc="Training",
            unit="epoch",
            bar_format="{l_bar}{bar:30}{r_bar}",
        )
        self._batch_bar: tqdm | None = None

    @property
    def elapsed_seconds(self) -> float:
        """Total elapsed time since training started, in seconds."""
        return time.time() - self._start_time

    @property
    def elapsed_str(self) -> str:
        """Human-readable elapsed time string."""
        elapsed = int(self.elapsed_seconds)
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @property
    def best_metric(self) -> float | None:
        """Best metric value seen so far."""
        return self._best_metric

    @property
    def best_epoch(self) -> int:
        """Epoch at which the best metric was recorded."""
        return self._best_epoch

    def start_epoch(self, epoch: int) -> None:
        """Signal the start of a new epoch.

        Args:
            epoch: Current epoch number (0-indexed).
        """
        self._current_epoch = epoch
        self._epoch_start_time = time.time()
        self._epoch_bar.set_description(f"Epoch {epoch + 1}/{self.total_epochs}")

        if self.batches_per_epoch > 0:
            self._batch_bar = tqdm(
                total=self.batches_per_epoch,
                desc=f"  Epoch {epoch + 1}",
                unit="batch",
                leave=False,
                bar_format="{l_bar}{bar:20}{r_bar}",
            )

    def update_batch(self, n: int = 1, metrics: dict[str, float] | None = None) -> None:
        """Update batch-level progress within the current epoch.

        Args:
            n: Number of batches to advance.
            metrics: Dictionary of metric name -> value to display.
        """
        if self._batch_bar is not None:
            if metrics:
                self._batch_bar.set_postfix({k: _fmt(v) for k, v in metrics.items()})
            self._batch_bar.update(n)

    def end_epoch(
        self,
        metrics: dict[str, float] | None = None,
        track_metric: str | None = None,
        mode: str = "max",
    ) -> bool:
        """Signal the end of an epoch and update epoch-level progress.

        Args:
            metrics: Dictionary of epoch-level metrics (e.g. val/mAP50).
            track_metric: Key in metrics to track for best model selection.
            mode: "max" if higher is better, "min" if lower is better.

        Returns:
            True if this epoch achieved a new best metric value.
        """
        # Close batch bar
        if self._batch_bar is not None:
            self._batch_bar.close()
            self._batch_bar = None

        # Epoch timing
        epoch_time = time.time() - self._epoch_start_time

        # Update epoch bar
        postfix: dict[str, Any] = {"time": f"{epoch_time:.1f}s", "total": self.elapsed_str}
        is_best = False

        if metrics:
            for k, v in metrics.items():
                postfix[k] = _fmt(v)

            # Track best metric
            if track_metric and track_metric in metrics:
                value = metrics[track_metric]
                if self._best_metric is None:
                    is_best = True
                elif mode == "max" and value > self._best_metric:
                    is_best = True
                elif mode == "min" and value < self._best_metric:
                    is_best = True

                if is_best:
                    self._best_metric = value
                    self._best_epoch = self._current_epoch
                    postfix["best"] = f"{value:.4f}@{self._current_epoch + 1}"

        self._epoch_bar.set_postfix(postfix)
        self._epoch_bar.update(1)

        return is_best

    def close(self) -> None:
        """Close all progress bars and print summary."""
        if self._batch_bar is not None:
            self._batch_bar.close()
        self._epoch_bar.close()

        # Print final summary
        print(f"\nTraining complete in {self.elapsed_str}")
        if self._best_metric is not None:
            print(f"Best metric: {self._best_metric:.4f} at epoch {self._best_epoch + 1}")

    def __enter__(self) -> "TrainingProgress":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

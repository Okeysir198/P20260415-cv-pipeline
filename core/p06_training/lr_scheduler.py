"""Learning rate schedulers with warmup support for detection model training.

Provides:
- WarmupScheduler: Linear warmup wrapper for any base scheduler.
- CosineScheduler: Cosine annealing with optional warmup.
- PlateauScheduler: ReduceLROnPlateau with optional warmup.
- build_scheduler: Factory function to build a scheduler from config.
"""

import math
import sys
from pathlib import Path

import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root


def _warmup_factor(epoch: int, warmup_epochs: int, start_factor: float) -> float:
    """Compute linear warmup multiplier for the given epoch.

    Args:
        epoch: Current epoch number.
        warmup_epochs: Total number of warmup epochs.
        start_factor: Starting LR multiplier (e.g. 0.001).

    Returns:
        LR multiplier in [start_factor, 1.0].
    """
    alpha = epoch / max(warmup_epochs, 1)
    return start_factor + alpha * (1.0 - start_factor)


class _WarmupMixin:
    """Shared warmup logic for epoch-based schedulers.

    Subclasses must set ``self.optimizer``, ``self.warmup_epochs``,
    ``self.warmup_start_factor``, ``self._base_lrs``, and
    ``self._current_epoch`` in their ``__init__``.
    """

    def _advance_epoch(self, epoch: int | None = None) -> None:
        """Update the internal epoch counter."""
        if epoch is not None:
            self._current_epoch = epoch
        else:
            self._current_epoch += 1

    def _apply_warmup(self) -> bool:
        """Apply linear warmup if still in the warmup phase.

        Returns:
            True if warmup was applied (caller should skip base scheduler step).
        """
        if self._current_epoch < self.warmup_epochs:
            factor = _warmup_factor(
                self._current_epoch, self.warmup_epochs, self.warmup_start_factor,
            )
            for param_group, base_lr in zip(
                self.optimizer.param_groups, self._base_lrs, strict=True
            ):
                param_group["lr"] = base_lr * factor
            return True
        return False

    def get_last_lr(self) -> list:
        """Get the last computed learning rates.

        Returns:
            List of current learning rates for each parameter group.
        """
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        """Return scheduler state for checkpointing.

        Returns:
            Dictionary with current epoch.
        """
        return {"current_epoch": self._current_epoch}

    def load_state_dict(self, state_dict: dict) -> None:
        """Load scheduler state from checkpoint.

        Args:
            state_dict: Dictionary with scheduler state.
        """
        self._current_epoch = state_dict["current_epoch"]


class WarmupScheduler(LRScheduler):
    """Linear warmup wrapper for any PyTorch learning rate scheduler.

    During warmup, the learning rate linearly increases from a small initial
    value to the optimizer's base LR. After warmup completes, the base
    scheduler takes over.

    Args:
        optimizer: Wrapped optimizer.
        base_scheduler: Scheduler to use after warmup completes.
        warmup_epochs: Number of warmup epochs. Default: 5.
        warmup_start_factor: Starting LR multiplier (relative to base LR).
            Default: 0.001 (starts at 0.1% of base LR).
        last_epoch: The index of the last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        base_scheduler: LRScheduler,
        warmup_epochs: int = 5,
        warmup_start_factor: float = 0.001,
        last_epoch: int = -1,
    ) -> None:
        self.base_scheduler = base_scheduler
        self.warmup_epochs = warmup_epochs
        self.warmup_start_factor = warmup_start_factor
        self._base_lrs = [group["lr"] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list:
        """Compute learning rates for the current epoch.

        Returns:
            List of learning rates for each parameter group.
        """
        if self.last_epoch < self.warmup_epochs:
            factor = _warmup_factor(self.last_epoch, self.warmup_epochs, self.warmup_start_factor)
            return [base_lr * factor for base_lr in self._base_lrs]
        return self.base_scheduler.get_last_lr()

    def step(self, epoch: int | None = None, metrics: float | None = None) -> None:
        """Advance the scheduler by one epoch.

        During warmup, updates LR linearly. After warmup, delegates
        to the base scheduler.

        Args:
            epoch: Optional epoch number. If None, uses internal counter.
            metrics: Optional metric value for ReduceLROnPlateau-style schedulers.
        """
        if epoch is not None:
            self.last_epoch = epoch
        else:
            self.last_epoch += 1

        if self.last_epoch < self.warmup_epochs:
            factor = _warmup_factor(self.last_epoch, self.warmup_epochs, self.warmup_start_factor)
            for param_group, base_lr in zip(
                self.optimizer.param_groups, self._base_lrs, strict=True
            ):
                param_group["lr"] = base_lr * factor
        else:
            if metrics is not None:
                self.base_scheduler.step(metrics)
            else:
                self.base_scheduler.step()


class CosineScheduler(_WarmupMixin):
    """Cosine annealing learning rate scheduler with warmup.

    LR follows a cosine decay from base LR to min_lr after warmup.

    Args:
        optimizer: Wrapped optimizer.
        total_epochs: Total training epochs (including warmup).
        warmup_epochs: Number of linear warmup epochs. Default: 5.
        min_lr: Minimum learning rate at end of schedule. Default: 1e-6.
        warmup_start_factor: Starting LR multiplier during warmup. Default: 0.001.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        total_epochs: int,
        warmup_epochs: int = 5,
        min_lr: float = 1e-6,
        warmup_start_factor: float = 0.001,
    ) -> None:
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.warmup_start_factor = warmup_start_factor
        self._base_lrs = [group["lr"] for group in optimizer.param_groups]
        self._current_epoch = 0

    def step(self, epoch: int | None = None, metrics: float | None = None) -> None:
        """Advance the scheduler by one epoch.

        Args:
            epoch: Optional epoch number. If None, uses internal counter.
            metrics: Unused, present for API compatibility.
        """
        self._advance_epoch(epoch)

        for param_group, base_lr in zip(self.optimizer.param_groups, self._base_lrs, strict=True):
            param_group["lr"] = self._compute_lr(base_lr)

    def _compute_lr(self, base_lr: float) -> float:
        """Compute learning rate for the current epoch.

        Args:
            base_lr: Base learning rate for the parameter group.

        Returns:
            Computed learning rate.
        """
        epoch = self._current_epoch

        if epoch < self.warmup_epochs:
            return base_lr * _warmup_factor(epoch, self.warmup_epochs, self.warmup_start_factor)

        # Cosine annealing after warmup
        cosine_epochs = self.total_epochs - self.warmup_epochs
        progress = (epoch - self.warmup_epochs) / max(cosine_epochs, 1)
        progress = min(progress, 1.0)
        return self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))


class PlateauScheduler(_WarmupMixin):
    """ReduceLROnPlateau with linear warmup.

    During warmup, LR increases linearly. After warmup, reduces LR
    when a tracked metric plateaus.

    Args:
        optimizer: Wrapped optimizer.
        warmup_epochs: Number of linear warmup epochs. Default: 5.
        warmup_start_factor: Starting LR multiplier during warmup. Default: 0.001.
        mode: "min" or "max" — direction for improvement detection. Default: "max".
        factor: Factor by which to reduce LR. Default: 0.1.
        patience: Epochs to wait before reducing. Default: 10.
        min_lr: Minimum learning rate floor. Default: 1e-6.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int = 5,
        warmup_start_factor: float = 0.001,
        mode: str = "max",
        factor: float = 0.1,
        patience: int = 10,
        min_lr: float = 1e-6,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_start_factor = warmup_start_factor
        self._base_lrs = [group["lr"] for group in optimizer.param_groups]
        self._current_epoch = 0

        self._plateau = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )

    def step(self, epoch: int | None = None, metrics: float | None = None) -> None:
        """Advance the scheduler by one epoch.

        Args:
            epoch: Optional epoch number. If None, uses internal counter.
            metrics: Metric value for plateau detection (required after warmup).
        """
        self._advance_epoch(epoch)

        if not self._apply_warmup() and metrics is not None:
            self._plateau.step(metrics)

    def state_dict(self) -> dict:
        """Return scheduler state for checkpointing.

        Returns:
            Dictionary with scheduler state.
        """
        return {
            "current_epoch": self._current_epoch,
            "plateau_state": self._plateau.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load scheduler state from checkpoint.

        Args:
            state_dict: Dictionary with scheduler state.
        """
        self._current_epoch = state_dict["current_epoch"]
        self._plateau.load_state_dict(state_dict["plateau_state"])


class StepScheduler(_WarmupMixin):
    """StepLR with linear warmup.

    Reduces LR by a factor every ``step_size`` epochs, after warmup.

    Args:
        optimizer: Wrapped optimizer.
        step_size: Epochs between LR reductions. Default: 30.
        gamma: Multiplicative LR decay factor. Default: 0.1.
        warmup_epochs: Number of linear warmup epochs. Default: 5.
        warmup_start_factor: Starting LR multiplier during warmup. Default: 0.001.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        step_size: int = 30,
        gamma: float = 0.1,
        warmup_epochs: int = 5,
        warmup_start_factor: float = 0.001,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_start_factor = warmup_start_factor
        self._base_lrs = [group["lr"] for group in optimizer.param_groups]
        self._current_epoch = 0
        self._step_lr = optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma,
        )

    def step(self, epoch: int | None = None, metrics: float | None = None) -> None:
        """Advance the scheduler by one epoch."""
        self._advance_epoch(epoch)

        if not self._apply_warmup():
            self._step_lr.step()

    def state_dict(self) -> dict:
        """Return scheduler state for checkpointing."""
        return {
            "current_epoch": self._current_epoch,
            "step_lr_state": self._step_lr.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load scheduler state from checkpoint."""
        self._current_epoch = state_dict["current_epoch"]
        self._step_lr.load_state_dict(state_dict["step_lr_state"])


class OneCycleScheduler:
    """OneCycleLR wrapper with interface matching other schedulers.

    OneCycleLR has built-in warmup (pct_start) and annealing. This
    wrapper adapts it to our per-epoch step() API.

    Args:
        optimizer: Wrapped optimizer.
        max_lr: Peak learning rate. Default: 0.01.
        total_epochs: Total number of training epochs. Default: 200.
        steps_per_epoch: Batches per epoch (for per-step scheduling).
            Default: 1 (per-epoch stepping).
        pct_start: Fraction of cycle spent increasing LR. Default: 0.3.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        max_lr: float = 0.01,
        total_epochs: int = 200,
        steps_per_epoch: int = 1,
        pct_start: float = 0.3,
    ) -> None:
        self.optimizer = optimizer
        self._current_epoch = 0
        self._scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_epochs * steps_per_epoch,
            pct_start=pct_start,
        )

    def step(self, epoch: int | None = None, metrics: float | None = None) -> None:
        """Advance the scheduler by one epoch."""
        if epoch is not None:
            self._current_epoch = epoch
        else:
            self._current_epoch += 1
        self._scheduler.step()

    def get_last_lr(self) -> list:
        """Get the last computed learning rates."""
        return self._scheduler.get_last_lr()

    def state_dict(self) -> dict:
        """Return scheduler state for checkpointing."""
        return {
            "current_epoch": self._current_epoch,
            "onecycle_state": self._scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load scheduler state from checkpoint."""
        self._current_epoch = state_dict["current_epoch"]
        self._scheduler.load_state_dict(state_dict["onecycle_state"])


def build_scheduler(optimizer: optim.Optimizer, config: dict) -> object:
    """Build a learning rate scheduler from training config.

    Reads the scheduler type and parameters from the training config
    section and constructs the appropriate scheduler with warmup.

    Args:
        optimizer: Optimizer to schedule.
        config: Training config dictionary. Expected keys:
            - training.scheduler: "cosine" or "plateau". Default: "cosine".
            - training.epochs: Total number of training epochs.
            - training.warmup_epochs: Number of warmup epochs. Default: 5.
            - training.min_lr: Minimum LR (for cosine). Default: 1e-6.
            - training.warmup_start_factor: Starting warmup factor. Default: 0.001.
            - training.patience: Patience epochs (for plateau). Default: 10.

    Returns:
        Configured scheduler instance.

    Raises:
        ValueError: If an unknown scheduler type is specified.
    """
    training_config = config.get("training", {})

    scheduler_type = training_config.get("scheduler", "cosine")
    total_epochs = training_config.get("epochs", 200)
    warmup_epochs = training_config.get("warmup_epochs", 5)
    warmup_start_factor = training_config.get("warmup_start_factor", 0.001)
    min_lr = training_config.get("min_lr", 1e-6)

    if scheduler_type == "cosine":
        return CosineScheduler(
            optimizer=optimizer,
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
            min_lr=min_lr,
            warmup_start_factor=warmup_start_factor,
        )
    elif scheduler_type == "plateau":
        patience = training_config.get("patience", 10)
        factor = training_config.get("scheduler_factor", 0.1)
        mode = config.get("checkpoint", {}).get("mode", "max")
        return PlateauScheduler(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            warmup_start_factor=warmup_start_factor,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )
    elif scheduler_type == "step":
        step_size = training_config.get("step_size", 30)
        gamma = training_config.get("gamma", 0.1)
        return StepScheduler(
            optimizer=optimizer,
            step_size=step_size,
            gamma=gamma,
            warmup_epochs=warmup_epochs,
            warmup_start_factor=warmup_start_factor,
        )
    elif scheduler_type == "onecycle":
        max_lr = training_config.get("lr", 0.01)
        return OneCycleScheduler(
            optimizer=optimizer,
            max_lr=max_lr,
            total_epochs=total_epochs,
            pct_start=training_config.get("pct_start", 0.3),
        )
    else:
        raise ValueError(
            f"Unknown scheduler type: '{scheduler_type}'. "
            f"Must be 'cosine', 'plateau', 'step', or 'onecycle'."
        )

"""Abstract base class for all models (detection, classification, segmentation).

All model architectures in the registry must inherit from DetectionModel
and implement the required abstract methods.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class DetectionModel(ABC, nn.Module):
    """Abstract base class for all registered models.

    Subclasses must implement ``forward`` and ``output_format``.
    Detection models should override ``strides`` to return their FPN strides;
    non-detection models (classification, segmentation) inherit the default
    empty list.

    A default ``get_param_groups`` splits parameters into decay / no-decay
    groups; subclasses may override for finer-grained control (e.g.
    per-component learning rates).
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference on a batch of images.

        Args:
            x: Input images of shape ``(B, 3, H, W)``.

        Returns:
            Predictions whose layout is described by :pyattr:`output_format`.
        """

    @property
    @abstractmethod
    def output_format(self) -> str:
        """String describing the output tensor layout (e.g. ``"yolox"``)."""

    @property
    def strides(self) -> list[int]:
        """Detection strides (e.g. ``[8, 16, 32]``).

        Non-detection models return an empty list (default).
        """
        return []

    def get_param_groups(
        self, lr: float, weight_decay: float
    ) -> list[dict]:
        """Return optimizer parameter groups.

        The default implementation splits into two groups:

        * **decay** -- weight tensors from Conv/Linear layers.
        * **no_decay** -- biases and normalization parameters.

        Subclasses may override to provide per-component groups (backbone,
        neck, head) with different learning rates.

        Args:
            lr: Base learning rate.
            weight_decay: Weight-decay coefficient for the decay group.

        Returns:
            List of parameter-group dicts suitable for
            ``torch.optim.Optimizer``.
        """
        pg_decay: list[nn.Parameter] = []
        pg_no_decay: list[nn.Parameter] = []

        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                pg_no_decay.extend(module.parameters())
            elif hasattr(module, "weight") and isinstance(
                module.weight, nn.Parameter
            ):
                pg_decay.append(module.weight)
            if hasattr(module, "bias") and isinstance(
                module.bias, nn.Parameter
            ):
                pg_no_decay.append(module.bias)

        return [
            {"params": pg_decay, "lr": lr, "weight_decay": weight_decay},
            {"params": pg_no_decay, "lr": lr, "weight_decay": 0.0},
        ]

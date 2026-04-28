"""Thin adapter for PaddleClas classification models.

Bridges PaddleClas' paddle.nn models into our PyTorch-native classification
trainer interface. Heavy ``paddle`` / ``paddleclas`` imports stay inside the
builder functions so the registry import stays cheap on minimal venvs.

Mirrors the I/O contract of :mod:`core.p06_models.timm_model` and
:mod:`core.p06_models.hf_classification_variants`:

* ``forward(x)`` returns ``(B, num_classes)`` logits as a torch tensor
* ``forward_with_loss(images, targets)`` returns
  ``(loss, loss_dict, logits)`` — same as the timm wrapper

Only registry entries are exposed at import time:

* ``ppclas-lcnet``        — PP-LCNet (mobile-friendly)
* ``ppclas-hgnet``        — PP-HGNet (server-side accuracy)
* ``ppclas-mobilenetv3``  — MobileNetV3 (PaddleClas variant)
* ``ppclas-pphgnetv2``    — PP-HGNetV2 (latest)

Pretrained weights are pulled from the PaddleClas model zoo via
``paddleclas`` defaults; users may override via ``model.pretrained``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from transformers.utils import ModelOutput

from core.p06_models.base import DetectionModel
from core.p06_models.registry import register_model

# Maps our arch key → PaddleClas internal model name (case matters in ppcls.arch)
_PPCLAS_ARCH_MAP: dict[str, str] = {
    "ppclas-lcnet": "PPLCNet_x1_0",
    "ppclas-hgnet": "PPHGNet_small",
    "ppclas-mobilenetv3": "MobileNetV3_small_x1_0",
    "ppclas-pphgnetv2": "PPHGNetV2_B4",
}


def _resolve_num_classes(config: dict) -> int:
    """Read ``model.num_classes`` or derive from ``data.names``.

    Mirrors :mod:`core.p06_models.hf_classification_variants` /
    :mod:`core.p06_models.hf_model` lookup pattern.
    """
    model_cfg = config.get("model", {})
    if "num_classes" in model_cfg:
        return int(model_cfg["num_classes"])
    names = config.get("data", {}).get("names")
    if isinstance(names, dict) and len(names) > 0:
        return len(names)
    if isinstance(names, list) and len(names) > 0:
        return len(names)
    raise ValueError(
        "PaddleClas model: config['model']['num_classes'] is required, or "
        "config['data']['names'] must be a non-empty dict/list."
    )


class PaddleClassificationModel(DetectionModel):
    """Adapter wrapping a paddle.nn classification module.

    Bridges torch tensors → paddle tensors → torch tensors so the rest of
    the trainer (loss accumulation, AMP autocast, EMA) sees only torch.

    Args:
        backbone: A ``paddle.nn.Layer`` instance (PaddleClas arch).
        num_classes: Number of output classes.
    """

    def __init__(self, backbone: Any, num_classes: int) -> None:
        super().__init__()
        # Stored as a plain attribute (not a child Module) — paddle Layers
        # are not nn.Module instances. State-dict / .to() / param iteration
        # on this wrapper deliberately skip the paddle backbone; the paddle
        # trainer (Unit 7) drives optimization on paddle params directly.
        self._paddle_backbone = backbone
        self.num_classes = num_classes

    @property
    def output_format(self) -> str:
        return "classification"

    @property
    def backbone(self) -> Any:
        """Expose underlying paddle.nn.Layer for the paddle trainer."""
        return self._paddle_backbone

    @staticmethod
    def _torch_to_paddle(x: torch.Tensor) -> Any:
        import paddle

        return paddle.to_tensor(x.detach().cpu().numpy())

    @staticmethod
    def _paddle_to_torch(x: Any, device: torch.device) -> torch.Tensor:
        return torch.as_tensor(x.numpy(), device=device)

    def forward(self, x: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """Run inference, return ``(B, num_classes)`` logits.

        Accepts both positional ``forward(x)`` and keyword
        ``forward(images=x, targets=...)`` calling conventions to match the
        timm wrapper. When targets are present, returns a HF-Trainer-shaped
        ``ModelOutput(loss=..., logits=...)``.
        """
        if x is None:
            x = kwargs.get("images")
        if x is None:
            x = kwargs.get("pixel_values")
        targets = kwargs.get("targets")
        labels = kwargs.get("labels")
        if targets is not None or labels is not None:
            t = targets if targets is not None else labels
            loss, _, logits = self.forward_with_loss(x, t)
            return ModelOutput(loss=loss, logits=logits)

        device = x.device
        px = self._torch_to_paddle(x)
        py = self._paddle_backbone(px)
        return self._paddle_to_torch(py, device)

    def forward_with_loss(
        self, images: torch.Tensor, targets: list | torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """Forward pass with cross-entropy loss (PaddleClas default).

        Args:
            images: ``(B, 3, H, W)`` torch tensor.
            targets: List of ``B`` scalar tensors (class indices) OR a
                pre-stacked ``(B,)`` tensor.

        Returns:
            ``(loss, loss_dict, logits)`` — same shape as the timm wrapper.
        """
        logits = self.forward(images)  # (B, C) torch tensor

        if isinstance(targets, torch.Tensor):
            labels = targets.to(images.device)
        else:
            labels = torch.stack(list(targets)).to(images.device)

        loss = F.cross_entropy(logits, labels)
        loss_dict = {"cls_loss": loss.detach()}
        return loss, loss_dict, logits

    def get_param_groups(
        self, lr: float, weight_decay: float,
    ) -> list[dict]:
        """No torch params — paddle backbone is opaque to the torch optimizer.

        The paddle trainer (Unit 7) builds its own paddle optimizer over
        ``self.backbone.parameters()`` directly. Returning an empty list
        keeps the torch path safe (no params → no torch optim is built for
        this wrapper).
        """
        return []


def _build_ppclas_backbone(
    arch: str, num_classes: int, pretrained: bool | str,
) -> Any:
    """Instantiate a PaddleClas arch and load pretrained weights.

    Imports of ``paddle`` / ``ppcls`` are deferred until call time so the
    registry stays importable in venvs without paddle installed.
    """
    try:
        # Most PaddleClas archs live under ppcls.arch.backbone
        from ppcls.arch import backbone as ppcls_backbone  # type: ignore
    except ImportError as e:
        raise ImportError(
            "PaddleClas is required for ppclas-* models. Install via "
            "`bash scripts/setup-paddle-venv.sh` and run with "
            "`.venv-paddle/bin/python`."
        ) from e

    pp_name = _PPCLAS_ARCH_MAP[arch]
    if not hasattr(ppcls_backbone, pp_name):
        raise ValueError(
            f"PaddleClas arch '{pp_name}' not found in ppcls.arch.backbone. "
            f"Check your paddleclas install."
        )
    ModelCls = getattr(ppcls_backbone, pp_name)

    # PaddleClas arches accept class_num + pretrained at construct time.
    use_pretrained: bool | str
    if pretrained is True or pretrained == "true":
        use_pretrained = True
    elif pretrained in (False, None, "false"):
        use_pretrained = False
    else:
        # Path / URL passed through as-is; paddleclas resolves both.
        use_pretrained = str(pretrained)

    logger.info(
        "Building PaddleClas model: arch=%s (%s), num_classes=%d, pretrained=%s",
        arch, pp_name, num_classes, use_pretrained,
    )
    return ModelCls(class_num=num_classes, pretrained=use_pretrained)


@register_model("ppclas-lcnet")
@register_model("ppclas-hgnet")
@register_model("ppclas-mobilenetv3")
@register_model("ppclas-pphgnetv2")
def build_ppclas_model(config: dict) -> PaddleClassificationModel:
    """Shared builder for every ``ppclas-*`` arch.

    Dispatch to the concrete PaddleClas class is keyed off
    ``config['model']['arch']`` via ``_PPCLAS_ARCH_MAP``.
    """
    model_cfg = config.get("model", {})
    arch = model_cfg["arch"].lower()
    if arch not in _PPCLAS_ARCH_MAP:
        raise ValueError(
            f"Unknown PaddleClas arch '{arch}'. Supported: {sorted(_PPCLAS_ARCH_MAP)}"
        )
    num_classes = _resolve_num_classes(config)
    pretrained = model_cfg.get("pretrained", True)
    backbone = _build_ppclas_backbone(arch, num_classes, pretrained)
    return PaddleClassificationModel(backbone, num_classes)

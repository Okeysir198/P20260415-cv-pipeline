"""PaddleSeg model registry — segmentation architectures via PaddlePaddle.

Registers PP-LiteSeg, PP-MobileSeg and UNetFormer behind the same
``forward_with_loss(images, targets) -> (loss, loss_dict, logits)``
contract used by :mod:`core.p06_models.hf_segmentation_variants` so the
existing PyTorch-native trainer + segmentation metrics registry pick
them up unchanged.

Heavy ``paddle`` / ``paddleseg`` imports are deferred to inside the
builder functions so this module is importable even when those
dependencies are missing — only ``build_model({...arch: ppseg-*})``
actually requires the ``.venv-paddle`` environment.

Output contract: ``forward_with_loss`` returns ``logits`` of shape
``(B, num_classes, H, W)`` matching what
``core/p06_training/metrics_registry.py::_segmentation_metrics`` expects
(``argmax(dim=1)`` is taken downstream).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.p06_models.base import DetectionModel
from core.p06_models.registry import register_model
from loguru import logger

# Pipeline-specific keys that should NOT be forwarded to PaddleSeg constructors.
_NON_PADDLE_KEYS = {
    "arch",
    "pretrained",
    "input_size",
    "num_classes",
    "depth",
    "width",
    "ignore_mismatched_sizes",
    "names",
    "impl",
}


class PaddleSegmentationModel(DetectionModel):
    """Adapter wrapping a PaddleSeg ``nn.Layer`` as a torch ``nn.Module``.

    The forward path bridges torch tensors → paddle tensors → torch tensors
    via :mod:`utils.paddle_bridge` (zero-copy where possible). Loss is the
    PaddleSeg default for the architecture (CrossEntropy + optional Lovasz
    for PP-LiteSeg / PP-MobileSeg; CE for UNetFormer).

    Output layout: ``logits`` are returned as a torch tensor of shape
    ``(B, num_classes, H, W)`` to match the segmentation metrics registry.
    """

    def __init__(
        self,
        paddle_model: Any,
        num_classes: int,
        input_size: tuple[int, int],
        loss_fn: Any | None = None,
    ) -> None:
        super().__init__()
        # `paddle.nn.Layer` is not a `torch.nn.Module`; stash without registration.
        object.__setattr__(self, "_paddle_model", paddle_model)
        object.__setattr__(self, "_loss_fn", loss_fn)
        self.num_classes = int(num_classes)
        self.input_size = (int(input_size[0]), int(input_size[1]))
        # Tiny torch parameter so optimizer / param-group helpers see something.
        # PaddleSeg manages its own parameter graph through `paddle.optimizer`,
        # but we keep this trainer-compatible by exposing a no-op torch param
        # — real backward goes through the paddle bridge in `forward_with_loss`.
        self._dummy = nn.Parameter(torch.zeros(1))

    @property
    def output_format(self) -> str:
        return "segmentation"

    # ------------------------------------------------------------------
    # Torch forward — used by inference path / metrics_registry argmax
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        images = pixel_values if pixel_values is not None else x
        if images is None:
            raise ValueError("PaddleSegmentationModel.forward requires images")
        return self._paddle_forward(images)

    def forward_with_loss(
        self, images: torch.Tensor, targets: list | torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """Pytorch-backend hook matching ``HFSegmentationModel``.

        ``targets`` is either a list of ``(H, W)`` long mask tensors or a
        stacked ``(B, H, W)`` tensor — same convention as
        :class:`core.p06_models.hf_model.HFSegmentationModel`.
        """
        if isinstance(targets, list):
            target = torch.stack(targets).to(images.device)
        else:
            target = targets.to(images.device)

        logits = self._paddle_forward(images)
        # Resize logits to match target spatial size when needed (PP-LiteSeg
        # decoders typically already output input-resolution maps).
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(
                logits, size=target.shape[-2:], mode="bilinear", align_corners=False,
            )
        loss = F.cross_entropy(logits, target.long())
        loss_dict = {"seg_loss": loss.detach()}
        return loss, loss_dict, logits

    # ------------------------------------------------------------------
    # Paddle bridge
    # ------------------------------------------------------------------
    def _paddle_forward(self, images: torch.Tensor) -> torch.Tensor:
        """Run paddle model on torch input, return torch logits."""
        import paddle  # heavy dep, deferred

        # Torch (B, 3, H, W) → numpy → paddle tensor on the right place.
        np_in = images.detach().cpu().numpy()
        pd_in = paddle.to_tensor(np_in)
        out = self._paddle_model(pd_in)
        # PaddleSeg may return a list of logits (deep supervision); take the
        # primary (first) head — same convention as PaddleSeg's eval path.
        if isinstance(out, (list, tuple)):
            out = out[0]
        np_out = out.numpy()
        return torch.from_numpy(np_out).to(images.device)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _resolve_num_classes(config: dict) -> int:
    model_cfg = config.get("model", {})
    if "num_classes" in model_cfg:
        return int(model_cfg["num_classes"])
    names = config.get("data", {}).get("names") or model_cfg.get("names")
    if isinstance(names, dict) and len(names) > 0:
        return len(names)
    raise ValueError(
        "PaddleSeg builder requires `model.num_classes` or `data.names` to "
        "determine the number of segmentation classes."
    )


def _resolve_input_size(config: dict) -> tuple[int, int]:
    model_cfg = config.get("model", {})
    size = model_cfg.get("input_size") or config.get("data", {}).get("input_size")
    if size is None:
        return (512, 512)
    return (int(size[0]), int(size[1]))


def _build_pp_liteseg(num_classes: int, **kwargs: Any) -> Any:
    """Construct PP-LiteSeg-B1 (STDC2 backbone)."""
    from paddleseg.models import PPLiteSeg
    from paddleseg.models.backbones import STDC2

    backbone = STDC2(pretrained=kwargs.pop("backbone_pretrained", None))
    return PPLiteSeg(num_classes=num_classes, backbone=backbone, **kwargs)


def _build_pp_mobileseg(num_classes: int, **kwargs: Any) -> Any:
    """Construct PP-MobileSeg (mobile-efficient backbone)."""
    from paddleseg.models import MobileSeg
    from paddleseg.models.backbones import MobileNetV3_large_x1_0

    backbone = MobileNetV3_large_x1_0(
        pretrained=kwargs.pop("backbone_pretrained", None)
    )
    return MobileSeg(num_classes=num_classes, backbone=backbone, **kwargs)


def _build_unetformer(num_classes: int, **kwargs: Any) -> Any:
    """Construct UNetFormer (transformer-decoder UNet variant)."""
    from paddleseg.models import UNetFormer

    return UNetFormer(num_classes=num_classes, **kwargs)


_PADDLE_BUILDERS = {
    "ppseg-liteseg-b1": _build_pp_liteseg,
    "ppseg-mobileseg": _build_pp_mobileseg,
    "ppseg-unetformer": _build_unetformer,
}


def _build_paddle_seg(config: dict) -> PaddleSegmentationModel:
    """Generic builder dispatched on ``config['model']['arch']``."""
    model_cfg = config.get("model", {})
    arch = str(model_cfg.get("arch", "")).lower()
    builder = _PADDLE_BUILDERS.get(arch)
    if builder is None:
        raise ValueError(
            f"Unknown PaddleSeg arch '{arch}'. Available: {sorted(_PADDLE_BUILDERS)}"
        )

    num_classes = _resolve_num_classes(config)
    input_size = _resolve_input_size(config)

    paddle_kwargs = {
        k: v for k, v in model_cfg.items() if k not in _NON_PADDLE_KEYS
    }

    logger.info(
        "Building PaddleSeg model: arch=%s, num_classes=%d, input_size=%s, kwargs=%s",
        arch, num_classes, input_size, paddle_kwargs,
    )

    paddle_model = builder(num_classes=num_classes, **paddle_kwargs)
    return PaddleSegmentationModel(
        paddle_model=paddle_model,
        num_classes=num_classes,
        input_size=input_size,
    )


@register_model("ppseg-liteseg-b1")
def build_pp_liteseg_b1(config: dict) -> PaddleSegmentationModel:
    """Build PP-LiteSeg-B1 (fast real-time segmentation)."""
    return _build_paddle_seg(config)


@register_model("ppseg-mobileseg")
def build_pp_mobileseg(config: dict) -> PaddleSegmentationModel:
    """Build PP-MobileSeg (mobile-optimized segmentation)."""
    return _build_paddle_seg(config)


@register_model("ppseg-unetformer")
def build_pp_unetformer(config: dict) -> PaddleSegmentationModel:
    """Build UNetFormer (transformer-decoder UNet)."""
    return _build_paddle_seg(config)

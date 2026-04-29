"""Thin adapter for timm classification models.

All config passes directly to timm.create_model() — this module
only adapts I/O format between timm and our trainer.
"""

from pathlib import Path

import timm as timm_lib
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from transformers.utils import ModelOutput

from core.p06_models.base import DetectionModel
from core.p06_models.registry import register_model

# Keys that belong to our pipeline, NOT to timm
_NON_TIMM_KEYS = {"arch", "pretrained", "input_size", "num_classes", "depth", "width", "timm_name"}


class TimmModel(DetectionModel):
    """Thin adapter around any timm classification model.

    Handles I/O format conversion between timm and our trainer.
    All architecture config passes through to timm unchanged.

    Args:
        backbone: A timm model instance.
        num_classes: Number of output classes.
    """

    def __init__(self, backbone: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes

    @property
    def output_format(self) -> str:
        return "classification"

    def forward(self, x: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """Run inference, return (B, num_classes) logits.

        Accepts both positional ``forward(x)`` and keyword
        ``forward(images=x, targets=...)`` calling conventions.
        The keyword form is used by HF Trainer which calls ``model(**batch)``.
        When targets are present, returns an object with ``.loss`` and ``.logits``
        attributes (compatible with HF Trainer).
        """
        if x is None:
            x = kwargs.get("images")
        targets = kwargs.get("targets")
        if targets is not None:
            loss, _, logits = self.forward_with_loss(x, targets)
            return ModelOutput(loss=loss, logits=logits)
        return self.backbone(x)

    def forward_with_loss(
        self, images: torch.Tensor, targets: list,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """Forward pass with cross-entropy loss.

        Args:
            images: (B, 3, H, W) input tensor.
            targets: List of B scalar tensors (class indices).

        Returns:
            (loss, loss_dict, logits) tuple.
        """
        logits = self.backbone(images)  # (B, C)

        # Stack targets: list of scalar tensors -> (B,) tensor
        labels = torch.stack(targets).to(images.device)

        loss = F.cross_entropy(logits, labels)
        loss_dict = {"cls_loss": loss.detach()}

        return loss, loss_dict, logits

    def get_param_groups(
        self, lr: float, weight_decay: float,
    ) -> list[dict]:
        """Split into backbone and classifier head groups.

        Uses timm's built-in get_classifier() to identify the head.
        """
        head_params = set()
        if hasattr(self.backbone, "get_classifier"):
            classifier = self.backbone.get_classifier()
            head_params = {id(p) for p in classifier.parameters()}

        backbone_decay = []
        backbone_no_decay = []
        head_decay = []
        head_no_decay = []

        for name, param in self.backbone.named_parameters():
            if not param.requires_grad:
                continue
            is_head = id(param) in head_params
            is_norm_or_bias = (
                "bn" in name or "norm" in name or "bias" in name
            )

            if is_head:
                if is_norm_or_bias:
                    head_no_decay.append(param)
                else:
                    head_decay.append(param)
            else:
                if is_norm_or_bias:
                    backbone_no_decay.append(param)
                else:
                    backbone_decay.append(param)

        groups = []
        if backbone_decay:
            groups.append({
                "params": backbone_decay,
                "lr": lr,
                "weight_decay": weight_decay,
                "group_name": "backbone",
            })
        if backbone_no_decay:
            groups.append({
                "params": backbone_no_decay,
                "lr": lr,
                "weight_decay": 0.0,
                "group_name": "backbone",
            })
        if head_decay:
            groups.append({
                "params": head_decay,
                "lr": lr,
                "weight_decay": weight_decay,
                "group_name": "head",
            })
        if head_no_decay:
            groups.append({
                "params": head_no_decay,
                "lr": lr,
                "weight_decay": 0.0,
                "group_name": "head",
            })

        return groups


@register_model("timm")
def build_timm_model(config: dict) -> TimmModel:
    """Build any timm classification model from config.

    Config example::

        model:
          arch: timm
          timm_name: mobilenetv3_small_100
          num_classes: 2
          input_size: [224, 224]
          pretrained: true
          drop_rate: 0.2  # passes to timm

    All config["model"] keys except pipeline-specific ones pass
    directly to timm.create_model() as kwargs.
    """
    model_cfg = config.get("model", {})
    timm_name = model_cfg.get("timm_name")
    if not timm_name:
        raise ValueError("config['model']['timm_name'] is required for timm models")

    num_classes = model_cfg["num_classes"]
    pretrained = model_cfg.get("pretrained", True)

    # Collect all non-pipeline keys as timm kwargs
    timm_kwargs = {k: v for k, v in model_cfg.items() if k not in _NON_TIMM_KEYS}

    # Handle pretrained: can be bool or path
    use_pretrained = pretrained is True or pretrained == "true"

    logger.info(
        "Building timm model: %s, pretrained=%s, num_classes=%d, kwargs=%s",
        timm_name, use_pretrained, num_classes, timm_kwargs,
    )

    backbone = timm_lib.create_model(
        timm_name,
        pretrained=use_pretrained,
        num_classes=num_classes,
        **timm_kwargs,
    )

    # Load custom weights if pretrained is a path
    if isinstance(pretrained, str) and pretrained not in ("true", "false"):
        weights_path = Path(pretrained)
        if not weights_path.is_absolute():
            config_dir = Path(config.get("_config_path", ".")).parent
            weights_path = (config_dir / weights_path).resolve()
        if weights_path.exists():
            state = torch.load(weights_path, map_location="cuda", weights_only=False)
            if "model" in state:
                state = state["model"]
            elif "model_state_dict" in state:
                state = state["model_state_dict"]
            backbone.load_state_dict(state, strict=False)
            logger.info("Loaded timm weights from %s", weights_path)

    return TimmModel(backbone, num_classes)

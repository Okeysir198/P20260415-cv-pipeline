"""timm variant aliases -> timm adapter."""

from core.p06_models.registry import _VARIANT_MAP

for _v in ("timm", "mobilenetv3", "efficientnet", "resnet", "vit", "convnext"):
    _VARIANT_MAP[_v] = "timm"

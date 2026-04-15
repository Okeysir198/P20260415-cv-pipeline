"""HF classification variant aliases → hf_classification builder."""

from core.p06_models.registry import _VARIANT_MAP

for _v in ("hf-vit-cls", "hf-dinov2-cls", "hf-resnet-cls", "hf-mobilenet-cls", "hf-swin-cls"):
    _VARIANT_MAP[_v] = "hf_classification"

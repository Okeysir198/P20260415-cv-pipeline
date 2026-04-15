"""HF segmentation variant aliases → hf_segmentation builder."""

from core.p06_models.registry import _VARIANT_MAP

for _v in ("hf-segformer", "hf-mask2former", "hf-dinov2-seg"):
    _VARIANT_MAP[_v] = "hf_segmentation"

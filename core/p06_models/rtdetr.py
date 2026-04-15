"""RT-DETRv2 variant aliases → generic HF adapter."""

from core.p06_models.registry import _VARIANT_MAP

for _v in ("rtdetr", "rtdetr-r18", "rtdetr-r50"):
    _VARIANT_MAP[_v] = "hf_detection"

"""D-FINE variant aliases → generic HF adapter."""

from core.p06_models.registry import _VARIANT_MAP

for _v in ("dfine", "dfine-s", "dfine-n", "dfine-m", "dfine-l", "dfine-large"):
    _VARIANT_MAP[_v] = "hf_detection"

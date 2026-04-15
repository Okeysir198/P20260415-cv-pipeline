"""Pipeline utilities for the object detection training pipeline.

Core utilities (config) are always available. Device and progress
utilities require torch and tqdm respectively — they are imported
lazily to avoid hard failures when dependencies are not yet installed.

Note: sv_metrics, visualization, supervision_bridge, and label_studio_bridge
have been relocated to their consuming pipeline steps:
  - sv_metrics          → core.p08_evaluation.sv_metrics
  - visualization       → core.p08_evaluation.visualization
  - supervision_bridge  → core.p10_inference.supervision_bridge
  - label_studio_bridge → core.p04_label_studio.bridge
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import load_config, merge_configs, validate_config

if TYPE_CHECKING:
    from .device import get_device, set_seed, get_gpu_info  # noqa: F401
    from .metrics import (  # noqa: F401
        compute_iou,
        xywh_to_xyxy,
        cxcywh_to_xyxy,
        xyxy_to_xywh,
        nms_numpy,
        nms_torch,
    )
    from .progress import ProgressBar, TrainingProgress  # noqa: F401

__all__ = [
    "load_config",
    "merge_configs",
    "validate_config",
    "get_device",
    "set_seed",
    "get_gpu_info",
    "ProgressBar",
    "TrainingProgress",
    "compute_iou",
    "xywh_to_xyxy",
    "cxcywh_to_xyxy",
    "xyxy_to_xywh",
    "nms_numpy",
    "nms_torch",
]


_DEVICE_NAMES = frozenset(("get_device", "set_seed", "get_gpu_info"))
_PROGRESS_NAMES = frozenset(("ProgressBar", "TrainingProgress"))
_METRICS_NAMES = frozenset((
    "compute_iou", "xywh_to_xyxy", "cxcywh_to_xyxy", "xyxy_to_xywh", "nms_numpy", "nms_torch",
))


def __getattr__(name: str):
    """Lazy imports for modules with heavy dependencies (torch, tqdm)."""
    if name in _DEVICE_NAMES:
        from . import device as _device
        return getattr(_device, name)
    if name in _PROGRESS_NAMES:
        from . import progress as _progress
        return getattr(_progress, name)
    if name in _METRICS_NAMES:
        from . import metrics as _metrics
        return getattr(_metrics, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

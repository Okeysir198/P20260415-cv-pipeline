"""Lazy-loaded SAM3 model singletons with per-model locks."""

from __future__ import annotations

import threading
from typing import Any

import torch

from src.config import config, logger

# Lazy-loaded singletons
_text_model = None
_text_processor = None
_tracker_model = None
_tracker_processor = None
_video_model = None
_video_processor = None
_text_lock = threading.Lock()
_tracker_lock = threading.Lock()
_video_lock = threading.Lock()


_device: str = config.get("model", {}).get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
_dtype_name: str = config.get("model", {}).get("dtype", "bfloat16")
_dtype: torch.dtype = getattr(torch, _dtype_name, torch.bfloat16)


def device() -> str:
    """Return configured or auto-detected device."""
    return _device


def dtype() -> torch.dtype:
    """Return configured torch dtype."""
    return _dtype


def loaded_models() -> dict[str, bool]:
    """Return which model singletons have been initialized."""
    return {
        "text": _text_model is not None,
        "tracker": _tracker_model is not None,
        "video": _video_model is not None,
    }


def get_text() -> tuple[Any, Any]:
    """Return (Sam3Model, Sam3Processor) for text/image segmentation."""
    global _text_model, _text_processor
    if _text_model is None:
        with _text_lock:
            if _text_model is None:
                from transformers import Sam3Model, Sam3Processor
                name = config["model"]["name"]
                logger.info("Loading Sam3Model (text/image) from %s", name)
                _text_processor = Sam3Processor.from_pretrained(name)
                _text_model = Sam3Model.from_pretrained(name).to(device(), dtype=dtype())
                _text_model.eval()
    return _text_model, _text_processor


def get_tracker() -> tuple[Any, Any]:
    """Return (Sam3TrackerVideoModel, Sam3TrackerVideoProcessor) for interactive tracking."""
    global _tracker_model, _tracker_processor
    if _tracker_model is None:
        with _tracker_lock:
            if _tracker_model is None:
                from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
                name = config["model"]["name"]
                logger.info("Loading Sam3TrackerVideoModel from %s", name)
                _tracker_processor = Sam3TrackerVideoProcessor.from_pretrained(name)
                _tracker_model = Sam3TrackerVideoModel.from_pretrained(name).to(device(), dtype=dtype())
                _tracker_model.eval()
    return _tracker_model, _tracker_processor


def get_video() -> tuple[Any, Any]:
    """Return (Sam3VideoModel, Sam3VideoProcessor) for text-driven video detection."""
    global _video_model, _video_processor
    if _video_model is None:
        with _video_lock:
            if _video_model is None:
                from transformers import Sam3VideoModel, Sam3VideoProcessor
                name = config["model"]["name"]
                logger.info("Loading Sam3VideoModel from %s", name)
                _video_processor = Sam3VideoProcessor.from_pretrained(name)
                _video_model = Sam3VideoModel.from_pretrained(name).to(device(), dtype=dtype())
                _video_model.eval()
    return _video_model, _video_processor

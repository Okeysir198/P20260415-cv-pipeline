"""Checkpoint loading helpers shared across p08/p09/p10."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_HF_PREFIX = "hf_model."


def strip_hf_prefix(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Remove the ``hf_model.`` key prefix written by HF-backend training.

    `core/p06_training/hf_trainer.py::_DetectionTrainer._save` saves the
    wrapper's state_dict, so every key starts with ``hf_model.``. Loading
    that into a bare HF model (or our `HFDetectionModel` via load_state_dict)
    silently keeps every weight randomly initialised when ``strict=False``.
    Strip the prefix here before any ``load_state_dict`` call on a
    HF-backend checkpoint. No-op for non-HF checkpoints.
    """
    if not any(k.startswith(_HF_PREFIX) for k in state_dict):
        return dict(state_dict)
    return {k.removeprefix(_HF_PREFIX): v for k, v in state_dict.items()}

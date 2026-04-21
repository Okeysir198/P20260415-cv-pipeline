"""Model registry for architecture selection via config.

Models register themselves with :func:`register_model`.  At build time,
:func:`build_model` looks up ``config["model"]["arch"]`` and dispatches to the
corresponding builder function.
"""

from collections.abc import Callable

import torch.nn as nn

from utils.registry import Registry

# Backward-compat alias — pose_registry / face_registry still import this name.
GenericRegistry = Registry

# ---------------------------------------------------------------------------
# Detection model registry (backward-compatible module-level API)
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, Callable] = {}
_VARIANT_MAP: dict[str, str] = {}

_detection_registry = Registry(
    entity_name="model",
    config_key="model",
    default_arch="yolox-m",
    registry=MODEL_REGISTRY,
    variant_map=_VARIANT_MAP,
)


def register_model(name: str):
    """Decorator that registers a model builder function.

    Args:
        name: Architecture name used in config (e.g. ``"yolox-m"``).

    Returns:
        Decorator that stores *fn* in :data:`MODEL_REGISTRY`.
    """
    return _detection_registry.register(name)


def build_model(config: dict) -> nn.Module:
    """Build a detection model from a config dictionary.

    Looks up ``config["model"]["arch"]`` (default ``"yolox-m"``) in the
    registry and calls the corresponding builder.

    Args:
        config: Full training config with a ``"model"`` section.

    Returns:
        Instantiated model (on CPU).

    Raises:
        ValueError: If the architecture is not registered.
    """
    return _detection_registry.build(config)

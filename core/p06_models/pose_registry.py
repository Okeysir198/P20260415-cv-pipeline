"""Pose model registry for architecture selection via config.

Pose models register themselves with :func:`register_pose_model`.  At build
time, :func:`build_pose_model` looks up ``config["pose_model"]["arch"]`` and
dispatches to the corresponding builder function.
"""

from collections.abc import Callable

from core.p06_models.pose_base import PoseModel
from utils.registry import Registry

POSE_MODEL_REGISTRY: dict[str, Callable] = {}
_POSE_VARIANT_MAP: dict[str, str] = {}

_pose_registry = Registry(
    entity_name="pose model",
    config_key="pose_model",
    default_arch="rtmpose-s",
    registry=POSE_MODEL_REGISTRY,
    variant_map=_POSE_VARIANT_MAP,
)


def register_pose_model(name: str):
    """Decorator that registers a pose model builder function.

    Args:
        name: Architecture name used in config (e.g. ``"rtmpose-s"``).

    Returns:
        Decorator that stores *fn* in :data:`POSE_MODEL_REGISTRY`.
    """
    return _pose_registry.register(name)


def build_pose_model(config: dict) -> PoseModel:
    """Build a pose model from a config dictionary.

    Looks up ``config["pose_model"]["arch"]`` (default ``"rtmpose-s"``) in
    the registry and calls the corresponding builder.

    Args:
        config: Full pose config with a ``"pose_model"`` section.

    Returns:
        Instantiated :class:`PoseModel`.

    Raises:
        ValueError: If the architecture is not registered.
    """
    return _pose_registry.build(config)

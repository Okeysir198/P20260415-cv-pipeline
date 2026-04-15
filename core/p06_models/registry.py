"""Model registry for architecture selection via config.

Models register themselves with :func:`register_model`.  At build time,
:func:`build_model` looks up ``config["model"]["arch"]`` and dispatches to the
corresponding builder function.
"""

import logging
from typing import Callable, Dict, Optional

import torch.nn as nn

logger = logging.getLogger(__name__)


class GenericRegistry:
    """Reusable registry with variant aliases and a build function.

    Subclasses or instances provide the registry dict, variant map,
    default architecture, config key, and an optional entity name
    (for error messages and log lines).
    """

    def __init__(
        self,
        registry: Dict[str, Callable],
        variant_map: Dict[str, str],
        default_arch: str = "",
        config_key: str = "model",
        entity_name: str = "model",
    ) -> None:
        self.registry = registry
        self.variant_map = variant_map
        self.default_arch = default_arch
        self.config_key = config_key
        self.entity_name = entity_name

    def register(self, name: str):
        """Decorator that registers a builder function under *name*."""

        def wrapper(fn: Callable) -> Callable:
            self.registry[name] = fn
            return fn

        return wrapper

    def build(self, config: dict, _base_type: Optional[type] = None) -> nn.Module:
        """Look up the architecture in the config and call the builder.

        Args:
            config: Config dictionary containing ``self.config_key`` section.
            base_type: Optional expected return type (not enforced, for
                documentation only).

        Returns:
            Whatever the registered builder returns.

        Raises:
            ValueError: If the architecture is not registered.
        """
        section = config.get(self.config_key, {})
        arch = section.get("arch", self.default_arch).lower()

        canonical = self.variant_map.get(arch, arch)

        if canonical not in self.registry:
            available = sorted(
                set(list(self.registry.keys()) + list(self.variant_map.keys()))
            )
            raise ValueError(
                f"Unknown {self.entity_name} architecture '{arch}'. "
                f"Available: {available}"
            )

        builder = self.registry[canonical]
        logger.info(
            "Building %s: arch=%s (builder=%s)", self.entity_name, arch, canonical
        )
        return builder(config)


# ---------------------------------------------------------------------------
# Detection model registry (backward-compatible module-level API)
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, Callable] = {}
_VARIANT_MAP: Dict[str, str] = {}

_detection_registry = GenericRegistry(
    registry=MODEL_REGISTRY,
    variant_map=_VARIANT_MAP,
    default_arch="yolox-m",
    config_key="model",
    entity_name="model",
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

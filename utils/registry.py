"""Generic registry + build helper for pipeline dispatch tables.

Used by :mod:`core.p06_models` (detection / pose / face), :mod:`core.p06_training`
(losses / metrics / postprocessors) to map a config string (e.g.
``model.arch=yolox-m`` or ``loss.type=yolox``) onto a builder / class.

The class has no dependency on torch so it can be imported from anywhere.
Downstream registries expose thin module-level aliases
(``register_model``, ``MODEL_REGISTRY``, ...) for backward compatibility —
this class is an implementation detail, not part of the public API.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class Registry:
    """String-keyed registry with optional variant aliases and config-driven build.

    Subclass-free by design: users construct a single instance per registry,
    expose :meth:`register` as a module-level decorator, and dispatch via
    :meth:`build` (config-driven) or :meth:`get` (direct name lookup).

    Args:
        entity_name: Human-readable label for log / error messages
            (e.g. ``"model"``, ``"loss"``).
        config_key: Top-level config section this registry dispatches on
            (e.g. ``"model"``, ``"face_detector"``).  Unused when the caller
            only uses :meth:`get` / :meth:`register`.
        default_arch: Architecture used when the config section has no
            ``arch`` key (e.g. ``"yolox-m"``).  Empty string means no default.
        registry: Optional pre-existing dict to store entries into.  Pass a
            module-level dict (e.g. ``MODEL_REGISTRY``) to preserve the legacy
            public name.
        variant_map: Optional pre-existing alias map.  Keys are aliases, values
            are canonical names already in :attr:`registry`.
    """

    def __init__(
        self,
        entity_name: str = "entry",
        config_key: str = "",
        default_arch: str = "",
        registry: dict[str, Callable] | None = None,
        variant_map: dict[str, str] | None = None,
    ) -> None:
        self.entity_name = entity_name
        self.config_key = config_key
        self.default_arch = default_arch
        self.registry: dict[str, Callable] = registry if registry is not None else {}
        self.variant_map: dict[str, str] = variant_map if variant_map is not None else {}

    def register(self, name: str) -> Callable[[Callable], Callable]:
        """Decorator that stores *fn* under *name*."""

        def wrapper(fn: Callable) -> Callable:
            self.registry[name] = fn
            return fn

        return wrapper

    def get(self, name: str, qualifier: str = "") -> Callable:
        """Look up a builder by name, resolving variant aliases.

        Args:
            name: Registry key (or alias) to look up.
            qualifier: Optional word inserted between the entity name and
                the quoted key in the error message — e.g. passing
                ``"architecture"`` yields ``"Unknown model architecture
                'foo'"`` to match legacy error text.

        Raises:
            ValueError: If *name* is not registered and has no alias.
        """
        canonical = self.variant_map.get(name, name)
        if canonical not in self.registry:
            available = sorted(set(list(self.registry) + list(self.variant_map)))
            prefix = (
                f"Unknown {self.entity_name} {qualifier}".rstrip()
                if qualifier
                else f"Unknown {self.entity_name}"
            )
            raise ValueError(f"{prefix} '{name}'. Available: {available}")
        return self.registry[canonical]

    def build(self, config: dict, **kwargs: Any) -> Any:
        """Build an instance from a config dictionary.

        Reads ``config[self.config_key]["arch"]`` (lowercased), resolves any
        variant alias, and calls the registered builder with *config*.

        Raises:
            ValueError: If the architecture key is not registered.
        """
        section = config.get(self.config_key, {})
        arch = section.get("arch", self.default_arch).lower()
        builder = self.get(arch, qualifier="architecture")
        logger.info(
            "Building %s: arch=%s (builder=%s)",
            self.entity_name,
            arch,
            builder.__name__ if hasattr(builder, "__name__") else arch,
        )
        return builder(config, **kwargs) if kwargs else builder(config)

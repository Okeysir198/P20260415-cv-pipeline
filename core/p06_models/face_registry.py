"""Face model registry for architecture selection via config.

Face detector and embedder models register themselves with
:func:`register_face_detector` and :func:`register_face_embedder`.  At build
time, :func:`build_face_detector` looks up ``config["face_detector"]["arch"]``
and :func:`build_face_embedder` looks up ``config["face_embedder"]["arch"]`,
dispatching to the corresponding builder function.
"""

from collections.abc import Callable

from core.p06_models.face_base import FaceDetector, FaceEmbedder
from utils.registry import Registry

# ---------------------------------------------------------------------------
# Face detector registry
# ---------------------------------------------------------------------------

FACE_DETECTOR_REGISTRY: dict[str, Callable] = {}
_FACE_DETECTOR_VARIANT_MAP: dict[str, str] = {}

_detector_registry = Registry(
    entity_name="face detector",
    config_key="face_detector",
    default_arch="scrfd",
    registry=FACE_DETECTOR_REGISTRY,
    variant_map=_FACE_DETECTOR_VARIANT_MAP,
)


def register_face_detector(name: str):
    """Decorator that registers a face detector builder function."""
    return _detector_registry.register(name)


def build_face_detector(config: dict) -> FaceDetector:
    """Build a face detector from a config dictionary."""
    return _detector_registry.build(config)


# ---------------------------------------------------------------------------
# Face embedder registry
# ---------------------------------------------------------------------------

FACE_EMBEDDER_REGISTRY: dict[str, Callable] = {}
_FACE_EMBEDDER_VARIANT_MAP: dict[str, str] = {}

_embedder_registry = Registry(
    entity_name="face embedder",
    config_key="face_embedder",
    default_arch="mobilefacenet",
    registry=FACE_EMBEDDER_REGISTRY,
    variant_map=_FACE_EMBEDDER_VARIANT_MAP,
)


def register_face_embedder(name: str):
    """Decorator that registers a face embedder builder function."""
    return _embedder_registry.register(name)


def build_face_embedder(config: dict) -> FaceEmbedder:
    """Build a face embedder from a config dictionary."""
    return _embedder_registry.build(config)

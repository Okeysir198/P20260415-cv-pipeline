"""Model registry and architecture factories.

Import :func:`build_model` to instantiate a model from config, or inspect
:data:`MODEL_REGISTRY` to see all registered architectures.

Example::

    from core.p06_models import build_model, MODEL_REGISTRY

    print(MODEL_REGISTRY.keys())
    model = build_model(config)
"""

import contextlib

# Import model modules to trigger registration.
# yolox is always available (only needs torch). HF-backed modules are
# soft-imported so minimal venvs (e.g. .venv-yolox-official/) that omit
# transformers / timm still get a usable yolox registry.
import core.p06_models.yolox  # noqa: F401
from core.p06_models.base import DetectionModel
from core.p06_models.face_base import FaceDetector, FaceEmbedder
from core.p06_models.face_registry import (
    FACE_DETECTOR_REGISTRY,
    FACE_EMBEDDER_REGISTRY,
    build_face_detector,
    build_face_embedder,
)
from core.p06_models.pose_base import PoseModel
from core.p06_models.pose_registry import POSE_MODEL_REGISTRY, build_pose_model
from core.p06_models.registry import MODEL_REGISTRY, build_model

with contextlib.suppress(ImportError):
    import core.p06_models.dfine  # noqa: F401
    import core.p06_models.hf_classification_variants  # noqa: F401
    import core.p06_models.hf_model  # noqa: F401
    import core.p06_models.hf_segmentation_variants  # noqa: F401
    import core.p06_models.rtdetr  # noqa: F401

# PaddleDetection backend (PicoDet, PP-YOLOE) — registry registration is
# numpy/torch only, so this should always succeed even when paddle itself
# is not installed (paddle imports are deferred to first build).
with contextlib.suppress(ImportError):
    import core.p06_models.paddle_model  # noqa: F401

# Import timm model module to trigger registration (optional dep)
with contextlib.suppress(ImportError):
    import core.p06_models.timm_model  # noqa: F401
    import core.p06_models.timm_variants  # noqa: F401

# Import PaddleClas classification module to trigger registration. The module
# itself defers heavy paddle imports to build-time, so this block only guards
# against missing torch-side deps in unusually minimal venvs (mirrors Unit 4's
# paddle_detection block).
with contextlib.suppress(ImportError):
    import core.p06_models.paddle_classification  # noqa: F401

# Import pose model modules to trigger registration (optional deps)
with contextlib.suppress(ImportError):
    import core.p06_models.rtmpose  # noqa: F401
with contextlib.suppress(ImportError):
    import core.p06_models.mediapipe_pose  # noqa: F401

# Import face model modules to trigger registration (optional deps)
with contextlib.suppress(ImportError):
    import core.p06_models.scrfd  # noqa: F401
with contextlib.suppress(ImportError):
    import core.p06_models.mobilefacenet  # noqa: F401

__all__ = [
    "build_model",
    "MODEL_REGISTRY",
    "DetectionModel",
    "build_pose_model",
    "POSE_MODEL_REGISTRY",
    "PoseModel",
    "build_face_detector",
    "build_face_embedder",
    "FACE_DETECTOR_REGISTRY",
    "FACE_EMBEDDER_REGISTRY",
    "FaceDetector",
    "FaceEmbedder",
]

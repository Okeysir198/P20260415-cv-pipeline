"""Object detection and pose estimation inference pipeline for factory safety cameras.

Submodules are soft-imported so minimal venvs (e.g. ``.venv-yolox-official/``)
that omit mediapipe / av / onnxruntime can still import sibling submodules
like ``supervision_bridge`` directly without dragging in the full pipeline.
"""

__all__ = []

try:
    from core.p10_inference.predictor import DetectionPredictor  # noqa: F401
    __all__.append("DetectionPredictor")
except ImportError:
    pass

try:
    from core.p10_inference.pose_predictor import PosePredictor  # noqa: F401
    __all__.append("PosePredictor")
except ImportError:
    pass

try:
    from core.p10_inference.video_inference import VideoProcessor  # noqa: F401
    __all__.append("VideoProcessor")
except ImportError:
    pass

"""Object detection and pose estimation inference pipeline for factory safety cameras."""

from core.p10_inference.predictor import DetectionPredictor
from core.p10_inference.pose_predictor import PosePredictor
from core.p10_inference.video_inference import VideoProcessor

__all__ = ["DetectionPredictor", "PosePredictor", "VideoProcessor"]

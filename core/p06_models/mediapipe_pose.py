"""MediaPipe Pose estimation model (33 3D landmarks).

Wraps ``mediapipe.solutions.pose.Pose`` behind the :class:`PoseModel` ABC so
it can be used interchangeably with RTMPose in the fall-detection and
poketenashi pipelines.  Supports three complexity levels (lite/full/heavy)
controlled via YAML config.
"""

import cv2
import mediapipe as mp
import numpy as np
from loguru import logger

from core.p06_models.pose_base import (
    MEDIAPIPE_KEYPOINT_NAMES,
    MEDIAPIPE_SKELETON,
    MEDIAPIPE_TO_COCO,
    PoseModel,
)
from core.p06_models.pose_registry import _POSE_VARIANT_MAP, register_pose_model


class MediaPipePoseModel(PoseModel):
    """MediaPipe Pose estimation model (33 landmarks).

    Uses ``mp.solutions.pose.Pose`` for single-person pose estimation.
    The model runs entirely on CPU via MediaPipe's TFLite backend.

    Args:
        model_complexity: MediaPipe model complexity (0=lite, 1=full, 2=heavy).
        min_detection_confidence: Minimum confidence for person detection.
        min_tracking_confidence: Minimum confidence for landmark tracking.
    """

    # Major joint indices used for overall pose score (shoulders, hips, knees).
    _MAJOR_JOINTS: list[int] = [11, 12, 23, 24, 25, 26]

    def __init__(
        self,
        model_complexity: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        logger.info(
            "MediaPipe Pose initialised: complexity=%d, det_conf=%.2f, track_conf=%.2f",
            model_complexity,
            min_detection_confidence,
            min_tracking_confidence,
        )

    # ------------------------------------------------------------------
    # PoseModel interface
    # ------------------------------------------------------------------

    def predict_keypoints(
        self, image: np.ndarray, bbox: np.ndarray
    ) -> dict[str, np.ndarray | float]:
        """Estimate 33 keypoints for a single person crop.

        Args:
            image: Full BGR image ``(H, W, 3)`` uint8.
            bbox: Person bounding box ``[x1, y1, x2, y2]`` in image coords.

        Returns:
            Dict with ``keypoints`` ``(33, 3)`` float32 (x, y, visibility)
            and ``score`` float (mean visibility of major joints).
        """
        x1, y1, x2, y2 = bbox.astype(int)
        h, w = image.shape[:2]

        # Pad bbox by 10% for context
        pad_w = int(0.1 * (x2 - x1))
        pad_h = int(0.1 * (y2 - y1))
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)

        crop = image[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = self._pose.process(crop_rgb)

        if not results.pose_landmarks:
            return {
                "keypoints": np.zeros((33, 3), dtype=np.float32),
                "score": 0.0,
            }

        crop_h, crop_w = crop.shape[:2]
        keypoints = np.zeros((33, 3), dtype=np.float32)
        for i, lm in enumerate(results.pose_landmarks.landmark):
            keypoints[i, 0] = lm.x * crop_w + x1  # map to full image coords
            keypoints[i, 1] = lm.y * crop_h + y1
            keypoints[i, 2] = lm.visibility

        # Overall score from major joints (shoulders, hips, knees)
        score = float(np.mean(keypoints[self._MAJOR_JOINTS, 2]))

        return {"keypoints": keypoints, "score": score}

    def to_coco(self, keypoints: np.ndarray) -> np.ndarray:
        """Convert 33 MediaPipe landmarks to COCO 17-keypoint format.

        Args:
            keypoints: ``(33, 3)`` MediaPipe keypoint array.

        Returns:
            ``(17, 3)`` COCO-format keypoints. Unmapped keypoints get
            ``score=0``.
        """
        coco_kpts = np.zeros((17, 3), dtype=np.float32)
        for mp_idx, coco_idx in MEDIAPIPE_TO_COCO.items():
            coco_kpts[coco_idx] = keypoints[mp_idx]
        return coco_kpts

    @property
    def keypoint_names(self) -> list[str]:
        """Ordered list of 33 MediaPipe landmark names."""
        return MEDIAPIPE_KEYPOINT_NAMES

    @property
    def num_keypoints(self) -> int:
        """Number of keypoints (33)."""
        return 33

    @property
    def skeleton(self) -> list[tuple[int, int]]:
        """Skeleton bone pairs for MediaPipe 33-landmark topology."""
        return MEDIAPIPE_SKELETON

    @property
    def input_size(self) -> tuple[int, int]:
        """MediaPipe internal default crop size ``(H, W)``."""
        return (256, 256)

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._pose.close()
        logger.info("MediaPipe Pose resources released.")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_COMPLEXITY_MAP: dict[str, int] = {
    "mediapipe-lite": 0,
    "mediapipe-full": 2,
    "mediapipe_pose": 2,
}


@register_pose_model("mediapipe_pose")
def build_mediapipe_pose(config: dict) -> MediaPipePoseModel:
    """Build a :class:`MediaPipePoseModel` from config.

    Config keys (under ``pose_model``):
        - ``arch``: One of ``mediapipe-lite``, ``mediapipe-full``.
        - ``model_complexity``: Override complexity (0/1/2).
        - ``min_detection_confidence``: Person detection threshold.
        - ``min_tracking_confidence``: Landmark tracking threshold.
    """
    pose_cfg = config.get("pose_model", {})
    arch = pose_cfg.get("arch", "mediapipe-full").lower()

    model_complexity = pose_cfg.get(
        "model_complexity", _COMPLEXITY_MAP.get(arch, 2)
    )
    min_det_conf = pose_cfg.get("min_detection_confidence", 0.5)
    min_track_conf = pose_cfg.get("min_tracking_confidence", 0.5)

    return MediaPipePoseModel(model_complexity, min_det_conf, min_track_conf)


_POSE_VARIANT_MAP["mediapipe-full"] = "mediapipe_pose"
_POSE_VARIANT_MAP["mediapipe-lite"] = "mediapipe_pose"

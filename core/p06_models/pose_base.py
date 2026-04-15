"""Abstract base class for pose estimation models (inference-only).

All pose model implementations must inherit from PoseModel and implement
the required abstract methods. This module also defines standard keypoint
schemas (COCO 17, MediaPipe 33) and mapping constants.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# COCO 17-keypoint schema (used by RTMPose, most top-down estimators)
# ---------------------------------------------------------------------------

COCO_KEYPOINT_NAMES: List[str] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COCO_SKELETON: List[Tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),  # head
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),  # upper body
    (5, 11),
    (6, 12),
    (11, 12),  # torso
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),  # lower body
]

# ---------------------------------------------------------------------------
# MediaPipe 33-landmark schema
# ---------------------------------------------------------------------------

MEDIAPIPE_KEYPOINT_NAMES: List[str] = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

MEDIAPIPE_SKELETON: List[Tuple[int, int]] = [
    (0, 2),
    (0, 5),
    (2, 7),
    (5, 8),  # head
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),  # upper body
    (11, 23),
    (12, 24),
    (23, 24),  # torso
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),  # lower body
    (27, 29),
    (29, 31),
    (28, 30),
    (30, 32),  # feet
    (15, 17),
    (15, 19),
    (15, 21),
    (16, 18),
    (16, 20),
    (16, 22),  # hands
]

# Mapping from MediaPipe landmark indices to COCO keypoint indices.
# Used by ``PoseModel.to_coco()`` for 33→17 conversion.
MEDIAPIPE_TO_COCO: Dict[int, int] = {
    0: 0,  # nose → nose
    2: 1,  # left_eye → left_eye
    5: 2,  # right_eye → right_eye
    7: 3,  # left_ear → left_ear
    8: 4,  # right_ear → right_ear
    11: 5,  # left_shoulder → left_shoulder
    12: 6,  # right_shoulder → right_shoulder
    13: 7,  # left_elbow → left_elbow
    14: 8,  # right_elbow → right_elbow
    15: 9,  # left_wrist → left_wrist
    16: 10,  # right_wrist → right_wrist
    23: 11,  # left_hip → left_hip
    24: 12,  # right_hip → right_hip
    25: 13,  # left_knee → left_knee
    26: 14,  # right_knee → right_knee
    27: 15,  # left_ankle → left_ankle
    28: 16,  # right_ankle → right_ankle
}


# ---------------------------------------------------------------------------
# PoseModel ABC
# ---------------------------------------------------------------------------


class PoseModel(ABC):
    """Abstract base class for pose estimation models.

    Subclasses must implement :meth:`predict_keypoints`, :attr:`keypoint_names`,
    :attr:`skeleton`, and :attr:`input_size`.

    This is inference-only — no ``nn.Module`` inheritance, no ``forward()``
    method, no training support. Models load ONNX or framework-specific
    weights directly.
    """

    @abstractmethod
    def predict_keypoints(
        self, image: np.ndarray, bbox: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Estimate keypoints for a single person.

        Args:
            image: Full BGR image ``(H, W, 3)`` uint8.
            bbox: Person bounding box ``[x1, y1, x2, y2]`` in image coords.

        Returns:
            Dict with:
                - ``keypoints``: ``(K, 3)`` float32 — x, y in image coords,
                  per-keypoint confidence score.
                - ``score``: float — overall pose confidence.
        """

    def predict_keypoints_batch(
        self, image: np.ndarray, bboxes: np.ndarray
    ) -> List[Dict[str, np.ndarray]]:
        """Estimate keypoints for multiple persons.

        Default implementation loops over bboxes. Subclasses may override
        for batched inference (e.g. stacking crops in a single ONNX call).

        Args:
            image: Full BGR image ``(H, W, 3)`` uint8.
            bboxes: ``(N, 4)`` float32 array of ``[x1, y1, x2, y2]``.

        Returns:
            List of *N* prediction dicts (same format as
            :meth:`predict_keypoints`).
        """
        results = []
        for i in range(len(bboxes)):
            results.append(self.predict_keypoints(image, bboxes[i]))
        return results

    @property
    @abstractmethod
    def keypoint_names(self) -> List[str]:
        """Ordered list of keypoint names."""

    @property
    @abstractmethod
    def num_keypoints(self) -> int:
        """Number of keypoints this model predicts."""

    @property
    @abstractmethod
    def skeleton(self) -> List[Tuple[int, int]]:
        """List of ``(idx_a, idx_b)`` pairs defining skeleton bones."""

    @property
    @abstractmethod
    def input_size(self) -> Tuple[int, int]:
        """Expected crop input size ``(H, W)``."""

    def to_coco(self, keypoints: np.ndarray) -> np.ndarray:
        """Convert native keypoints to COCO 17-keypoint format.

        Args:
            keypoints: ``(K, 3)`` native keypoint array.

        Returns:
            ``(17, 3)`` COCO-format keypoints. Missing keypoints get
            ``score=0``.
        """
        if keypoints.shape[0] == 17:
            return keypoints
        raise NotImplementedError(
            f"to_coco() not implemented for {keypoints.shape[0]}-keypoint model. "
            "Subclasses with non-COCO outputs must override this method."
        )

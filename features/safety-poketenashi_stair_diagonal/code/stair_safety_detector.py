"""Stair safety pose rule — diagonal body trajectory detection.

Rule: if the mid-hip position traces a diagonal path (crossing angle > threshold from
the stair axis) across consecutive frames, the person is not ascending/descending
stairs in a controlled, aligned manner.

Config keys from 10_inference.yaml pose_rules.stair_diagonal:
  max_diagonal_angle_deg — crossing angle > this from stair axis = diagonal (default 20°)

The stair axis is assumed horizontal (x-axis). A trajectory angle is computed from
the vector spanning the oldest→newest hip positions in the buffer.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _base import PoseRule, RuleResult  # noqa: E402

_L_HIP, _R_HIP = 11, 12
_MIN_SCORE = 0.3
_MIN_FRAMES = 5  # minimum buffer length before evaluating trajectory


class StairSafetyDetector(PoseRule):
    """Detect unsafe diagonal stair traversal via hip trajectory angle."""

    behavior = "stair_diagonal"

    def __init__(
        self,
        max_diagonal_angle_deg: float = 20.0,
        min_frames: int = _MIN_FRAMES,
    ) -> None:
        self._angle_threshold = max_diagonal_angle_deg
        self._min_frames = min_frames
        self._hip_positions: list[np.ndarray] = []

    def check(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        frame_buffer: list[np.ndarray] | None = None,
    ) -> RuleResult:
        if keypoints is None or len(keypoints) < 13:
            return RuleResult(False, 0.0, self.behavior, {"reason": "insufficient keypoints"})

        l_hip_s, r_hip_s = scores[_L_HIP], scores[_R_HIP]
        if l_hip_s < _MIN_SCORE and r_hip_s < _MIN_SCORE:
            return RuleResult(False, 0.0, self.behavior, {"reason": "hips not visible"})

        # Compute mid-hip, weighting by visibility.
        if l_hip_s >= _MIN_SCORE and r_hip_s >= _MIN_SCORE:
            mid_hip = (keypoints[_L_HIP] + keypoints[_R_HIP]) / 2.0
        elif l_hip_s >= _MIN_SCORE:
            mid_hip = keypoints[_L_HIP].copy()
        else:
            mid_hip = keypoints[_R_HIP].copy()

        self._hip_positions.append(mid_hip)

        if len(self._hip_positions) < self._min_frames:
            return RuleResult(
                False, 0.0, self.behavior,
                {"reason": f"buffering ({len(self._hip_positions)}/{self._min_frames})"},
            )

        # Trajectory vector: oldest → newest position.
        start = self._hip_positions[0]
        end = self._hip_positions[-1]
        dx = float(end[0] - start[0])
        dy = float(end[1] - start[1])

        # Angle from horizontal (stair axis assumed roughly horizontal).
        trajectory_angle_deg = abs(math.degrees(math.atan2(dy, dx + 1e-9)))
        # Normalise to [0, 90] — we care about deviation from horizontal.
        if trajectory_angle_deg > 90:
            trajectory_angle_deg = 180 - trajectory_angle_deg

        triggered = trajectory_angle_deg > self._angle_threshold

        # Keep buffer bounded — slide window.
        if len(self._hip_positions) > 30:
            self._hip_positions.pop(0)

        debug = {
            "trajectory_angle_deg": round(trajectory_angle_deg, 2),
            "threshold_deg": self._angle_threshold,
            "buffer_len": len(self._hip_positions),
        }
        confidence = 1.0 if triggered else 0.0
        return RuleResult(triggered, confidence, self.behavior, debug)

    def reset(self) -> None:
        self._hip_positions.clear()

"""Pointing-and-calling pose rule.

Rule: arm is extended (elbow-wrist-shoulder angle > threshold) with the arm roughly
horizontal (shoulder-to-wrist elevation below threshold). This models the Japanese
safety practice of "shisa kanko" (指差呼称) — pointing at a safety checkpoint while
calling out loud. A MISSING pointing gesture at a designated checkpoint is the violation.

Config keys from 10_inference.yaml pose_rules.no_pointing_calling:
  elbow_wrist_angle_min_deg — arm must be nearly straight (default 150°)
  pointing_duration_frames  — must hold for this many frames (temporal, via frame_buffer length)

The rule fires (violation = True) when NO extended pointing arm is detected.
"""

from __future__ import annotations

import math

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _base import PoseRule, RuleResult  # noqa: E402

# COCO-17 indices
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_ELBOW, _R_ELBOW = 7, 8
_L_WRIST, _R_WRIST = 9, 10

_MIN_SCORE = 0.3
_ARM_ELEVATION_MAX_DEG = 45.0  # wrist must not be too high (pointing up ≠ shisa kanko)


def _angle_deg(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> float:
    """Angle at `vertex` in the triangle a-vertex-b, in degrees."""
    va = a - vertex
    vb = b - vertex
    cos_a = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9)
    return float(math.degrees(math.acos(float(np.clip(cos_a, -1.0, 1.0)))))


def _arm_elevation_deg(shoulder: np.ndarray, wrist: np.ndarray) -> float:
    """Angle of the shoulder→wrist vector from horizontal (positive = upward)."""
    dx = float(wrist[0] - shoulder[0])
    dy = float(wrist[1] - shoulder[1])  # y increases downward in image coords
    return abs(math.degrees(math.atan2(-dy, abs(dx) + 1e-9)))


class PointingCallingDetector(PoseRule):
    """Detect missing pointing-and-calling gesture.

    Violation = no arm is extended horizontally at the checkpoint moment.
    """

    behavior = "no_pointing_calling"

    def __init__(
        self,
        elbow_wrist_angle_min_deg: float = 150.0,
        pointing_duration_frames: int = 20,
    ) -> None:
        self._min_elbow_angle = elbow_wrist_angle_min_deg
        self._duration_frames = pointing_duration_frames
        self._pointing_history: list[bool] = []  # True = pointing detected that frame

    def check(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        frame_buffer: list[np.ndarray] | None = None,
    ) -> RuleResult:
        if keypoints is None or len(keypoints) < 11:
            return RuleResult(False, 0.0, self.behavior, {"reason": "insufficient keypoints"})

        arm_pairs = [
            ("left", _L_SHOULDER, _L_ELBOW, _L_WRIST),
            ("right", _R_SHOULDER, _R_ELBOW, _R_WRIST),
        ]

        pointing_detected = False
        arm_debug: dict[str, dict] = {}

        for side, s_idx, e_idx, w_idx in arm_pairs:
            if scores[s_idx] < _MIN_SCORE or scores[e_idx] < _MIN_SCORE or scores[w_idx] < _MIN_SCORE:
                arm_debug[side] = {"skip": "low score"}
                continue

            shoulder = keypoints[s_idx]
            elbow = keypoints[e_idx]
            wrist = keypoints[w_idx]

            # Angle at elbow: shoulder–elbow–wrist.
            elbow_angle = _angle_deg(shoulder, elbow, wrist)
            # Elevation: shoulder→wrist vector from horizontal.
            elevation = _arm_elevation_deg(shoulder, wrist)

            is_extended = elbow_angle >= self._min_elbow_angle
            is_horizontal = elevation <= _ARM_ELEVATION_MAX_DEG

            arm_debug[side] = {
                "elbow_angle_deg": round(elbow_angle, 1),
                "arm_elevation_deg": round(elevation, 1),
                "extended": is_extended,
                "horizontal": is_horizontal,
            }

            if is_extended and is_horizontal:
                pointing_detected = True

        # Track history for temporal assessment.
        self._pointing_history.append(pointing_detected)
        if len(self._pointing_history) > self._duration_frames:
            self._pointing_history.pop(0)

        # Violation: pointing gesture has NOT been held long enough.
        # We flag True (violation) when pointing is absent in current frame.
        triggered = not pointing_detected

        debug = {"arms": arm_debug, "pointing_this_frame": pointing_detected}
        confidence = 1.0 if triggered else 0.0
        return RuleResult(triggered, confidence, self.behavior, debug)

    def reset(self) -> None:
        self._pointing_history.clear()

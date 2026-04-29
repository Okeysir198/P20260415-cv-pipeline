"""Per-frame pointing-direction labeller.

Outputs one of ``{point_left, point_right, point_front, neutral, invalid}``
for the (single) person in the frame. The detector does NOT itself fire
``triggered=True``; the sequence matcher consumes the frame-level labels
and decides when the full shisa-kanko gesture has occurred.

Direction conventions (body-relative, see ``_geometry.py``):
    azimuth ~= +90 deg  -> point_front
    azimuth ~=   0 deg  -> point_right
    azimuth ~= +-180    -> point_left
    azimuth ~= -90 deg  -> arm hanging down / behind body -> invalid
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from _base import PoseRule, RuleResult  # noqa: E402
from _geometry import (  # noqa: E402
    arm_azimuth_torso_frame,
    arm_elevation_deg,
    elbow_angle_deg,
    torso_frame_basis,
)

# COCO-17 indices used by this rule.
_NOSE = 0
_L_EAR, _R_EAR = 3, 4
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_ELBOW, _R_ELBOW = 7, 8
_L_WRIST, _R_WRIST = 9, 10
_L_HIP, _R_HIP = 11, 12

_REQUIRED = (5, 6, 7, 8, 9, 10, 11, 12)


def _angular_distance_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two angles (degrees)."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


class PointingDirectionDetector(PoseRule):
    """Per-frame label-only detector. ``triggered`` is always ``False``."""

    behavior = "pointing_direction"

    def __init__(
        self,
        elbow_angle_min_deg: float = 150.0,
        arm_elevation_max_deg: float = 45.0,
        front_half_angle_deg: float = 30.0,
        side_half_angle_deg: float = 45.0,
        min_keypoint_score: float = 0.3,
        min_wrist_ear_distance_ratio: float = 0.0,
    ) -> None:
        self._elbow_angle_min_deg = float(elbow_angle_min_deg)
        self._arm_elevation_max_deg = float(arm_elevation_max_deg)
        self._front_half_angle_deg = float(front_half_angle_deg)
        self._side_half_angle_deg = float(side_half_angle_deg)
        self._min_keypoint_score = float(min_keypoint_score)
        # Reject "phone-to-ear" / hand-near-face poses: require wrist to be at
        # least N * shoulder_width away from the nearest ear. 0 disables the
        # check (default; tests don't supply ear-distant fixtures).
        self._min_wrist_ear_dist_ratio = float(min_wrist_ear_distance_ratio)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def check(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        frame_buffer: list[np.ndarray] | None = None,
    ) -> RuleResult:
        debug: dict = {
            "label": "invalid",
            "side": None,
            "elbow_angle": None,
            "arm_elevation": None,
            "azimuth": None,
        }

        if keypoints is None or scores is None:
            return RuleResult(False, 0.0, self.behavior, debug)
        if len(keypoints) <= max(_REQUIRED) or len(scores) <= max(_REQUIRED):
            return RuleResult(False, 0.0, self.behavior, debug)

        for idx in _REQUIRED:
            if float(scores[idx]) < self._min_keypoint_score:
                return RuleResult(False, 0.0, self.behavior, debug)

        # Per-side extension check.
        # Elevation + azimuth use the FOREARM vector (elbow->wrist): more robust
        # to bent-elbow real-world pointing where the upper arm is dropped but
        # the forearm is the actual pointing vector. Elbow-angle gate still uses
        # the full shoulder-elbow-wrist triangle (a true straight-arm criterion).
        sides = {
            "left": (_L_SHOULDER, _L_ELBOW, _L_WRIST),
            "right": (_R_SHOULDER, _R_ELBOW, _R_WRIST),
        }
        side_metrics: dict[str, dict] = {}
        extending: list[str] = []

        # Reject phone-to-ear: wrist within R*shoulder_width of nearest ear.
        sw = float(np.linalg.norm(keypoints[_R_SHOULDER] - keypoints[_L_SHOULDER])) + 1e-6
        for name, (s_idx, e_idx, w_idx) in sides.items():
            ea = elbow_angle_deg(
                keypoints[s_idx], keypoints[e_idx], keypoints[w_idx]
            )
            el = arm_elevation_deg(keypoints[e_idx], keypoints[w_idx])
            side_metrics[name] = {"elbow_angle": ea, "arm_elevation": el}
            if not (
                ea >= self._elbow_angle_min_deg
                and el <= self._arm_elevation_max_deg
            ):
                continue
            if self._min_wrist_ear_dist_ratio > 0.0:
                d_l_ear = float(np.linalg.norm(keypoints[w_idx] - keypoints[_L_EAR]))
                d_r_ear = float(np.linalg.norm(keypoints[w_idx] - keypoints[_R_EAR]))
                if min(d_l_ear, d_r_ear) / sw < self._min_wrist_ear_dist_ratio:
                    continue
            extending.append(name)

        if not extending:
            debug["label"] = "neutral"
            return RuleResult(False, 0.0, self.behavior, debug)

        # Both arms extending -> pick the side with the lower confidence
        # sum first (per spec); ties go to right.
        if len(extending) == 2:
            l_sum = float(scores[_L_WRIST] + scores[_L_ELBOW] + scores[_L_SHOULDER])
            r_sum = float(scores[_R_WRIST] + scores[_R_ELBOW] + scores[_R_SHOULDER])
            side = "left" if l_sum < r_sum else "right"
        else:
            side = extending[0]

        s_idx, e_idx, w_idx = sides[side]

        _, e_x, e_y = torso_frame_basis(
            keypoints[_L_SHOULDER],
            keypoints[_R_SHOULDER],
            keypoints[_L_HIP],
            keypoints[_R_HIP],
        )
        # Forearm vector (elbow->wrist) is what actually points at the target.
        azimuth = arm_azimuth_torso_frame(
            keypoints[e_idx], keypoints[w_idx], e_x, e_y
        )

        debug["side"] = side
        debug["elbow_angle"] = round(side_metrics[side]["elbow_angle"], 2)
        debug["arm_elevation"] = round(side_metrics[side]["arm_elevation"], 2)
        debug["azimuth"] = round(float(azimuth), 2)

        label = self._azimuth_to_label(float(azimuth))
        debug["label"] = label

        # Confidence = min of wrist+elbow+shoulder scores for the chosen side.
        conf = float(min(scores[s_idx], scores[e_idx], scores[w_idx]))
        # Detector never fires triggered itself.
        return RuleResult(False, conf, self.behavior, debug)

    # ------------------------------------------------------------------
    # Azimuth mapping
    # ------------------------------------------------------------------

    def _azimuth_to_label(self, az: float) -> str:
        """Bucket a torso-frame azimuth into a direction label."""
        # Front = within +/- front_half_angle_deg of +90 deg.
        if _angular_distance_deg(az, 90.0) <= self._front_half_angle_deg:
            return "point_front"
        # Right = around 0 deg, on the right half (cos(az) > 0).
        if _angular_distance_deg(az, 0.0) <= self._side_half_angle_deg:
            return "point_right"
        # Left = around +/-180 deg, on the left half (cos(az) < 0).
        if _angular_distance_deg(az, 180.0) <= self._side_half_angle_deg:
            return "point_left"
        # Anything else (mostly behind / down) -> invalid.
        return "invalid"

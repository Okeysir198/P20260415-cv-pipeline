"""Hands-in-pockets pose rule.

Rule: wrist y-position is below hip y AND wrist is inside the torso horizontal band
(i.e. wrist x is close to the torso centerline). Both conditions must hold for the
same side to trigger. Either side triggers the alert.

Config keys from 10_inference.yaml pose_rules.hands_in_pockets:
  wrist_below_hip_ratio    — wrist y > hip y + ratio * torso_height
  wrist_inside_torso_margin — |wrist x - torso_cx| < margin * torso_width
  elbow_inside_torso_margin — low-wrist-score fallback: |elbow x - torso_cx|
                              < margin * torso_width (default 1.0 = ±1 torso width)
"""

from __future__ import annotations

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _base import PoseRule, RuleResult  # noqa: E402

# COCO-17 keypoint indices
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_ELBOW, _R_ELBOW = 7, 8
_L_WRIST, _R_WRIST = 9, 10
_L_HIP, _R_HIP = 11, 12

_MIN_SCORE = 0.3  # minimum keypoint confidence to use


class HandsInPocketsDetector(PoseRule):
    """Detect hands hidden in pockets via wrist-below-hip + torso-centerline proximity."""

    behavior = "hands_in_pockets"

    def __init__(
        self,
        wrist_below_hip_ratio: float = 0.05,
        wrist_inside_torso_margin: float = 0.08,
        elbow_inside_torso_margin: float = 0.7,
        require_both_sides: bool = False,
    ) -> None:
        self._below_ratio = wrist_below_hip_ratio
        self._torso_margin = wrist_inside_torso_margin
        # Used only when the wrist is low-confidence (hand likely occluded inside
        # the pocket). In-pocket pose drops the elbow toward hip height AND keeps
        # it tucked near the body. Pointing/gesturing raises the elbow toward
        # shoulder height and/or swings it out wide. Two-part gate:
        #   (1) elbow Y below the shoulder-hip midpoint (arm dropped, not raised)
        #   (2) elbow X within `elbow_inside_torso_margin` torso widths of the
        #       centerline (forward bend OK, sideways arms-at-side rejected)
        self._elbow_torso_margin = elbow_inside_torso_margin
        # The violation is `hands_in_pockets` (plural). When True, BOTH sides
        # must satisfy the in-pocket gate simultaneously. This rejects the
        # common FP shape "one arm at side + other arm gesturing" — the
        # gesturing arm's elbow is raised so it fails the dropped gate.
        self._require_both_sides = require_both_sides

    def check(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        frame_buffer: list[np.ndarray] | None = None,
    ) -> RuleResult:
        if keypoints is None or len(keypoints) < 17:
            return RuleResult(False, 0.0, self.behavior, {"reason": "insufficient keypoints"})

        l_hip = keypoints[_L_HIP]
        r_hip = keypoints[_R_HIP]
        l_shoulder = keypoints[_L_SHOULDER]
        r_shoulder = keypoints[_R_SHOULDER]
        l_wrist = keypoints[_L_WRIST]
        r_wrist = keypoints[_R_WRIST]
        l_elbow = keypoints[_L_ELBOW]
        r_elbow = keypoints[_R_ELBOW]

        hip_scores = (scores[_L_HIP], scores[_R_HIP])
        shoulder_scores = (scores[_L_SHOULDER], scores[_R_SHOULDER])

        # Need at least one hip + one shoulder visible to estimate torso.
        if max(hip_scores) < _MIN_SCORE or max(shoulder_scores) < _MIN_SCORE:
            return RuleResult(False, 0.0, self.behavior, {"reason": "hips/shoulders not visible"})

        mid_hip = (l_hip + r_hip) / 2.0
        mid_shoulder = (l_shoulder + r_shoulder) / 2.0
        torso_height = float(np.linalg.norm(mid_shoulder - mid_hip)) + 1e-6
        torso_width = float(abs(l_hip[0] - r_hip[0])) + 1e-6
        torso_cx = float(mid_hip[0])
        torso_mid_y = float((mid_shoulder[1] + mid_hip[1]) * 0.5)

        def _wrist_in_pocket(
            wrist: np.ndarray,
            wrist_score: float,
            hip: np.ndarray,
            elbow: np.ndarray,
            elbow_score: float,
        ) -> bool:
            if wrist_score >= _MIN_SCORE:
                # High-confidence path: trust wrist position directly.
                below_hip = wrist[1] > hip[1] + self._below_ratio * torso_height
                inside_torso = abs(wrist[0] - torso_cx) < self._torso_margin * torso_width
                return bool(below_hip and inside_torso)
            # Low-confidence wrist: hand likely occluded inside the pocket. Use
            # elbow as proxy. Reject the gesturing/pointing pose (elbow raised
            # toward shoulder) and the reaching pose (elbow swung out wide).
            if elbow_score < _MIN_SCORE:
                return False  # neither wrist nor elbow visible — abstain
            elbow_dropped = elbow[1] > torso_mid_y
            elbow_inside = abs(elbow[0] - torso_cx) < self._elbow_torso_margin * torso_width
            return bool(elbow_dropped and elbow_inside)

        left_hit = _wrist_in_pocket(
            l_wrist, scores[_L_WRIST], l_hip, l_elbow, scores[_L_ELBOW]
        )
        right_hit = _wrist_in_pocket(
            r_wrist, scores[_R_WRIST], r_hip, r_elbow, scores[_R_ELBOW]
        )
        if self._require_both_sides:
            triggered = left_hit and right_hit
        else:
            triggered = left_hit or right_hit

        debug = {
            "left_wrist_below_hip": bool(l_wrist[1] > l_hip[1]),
            "right_wrist_below_hip": bool(r_wrist[1] > r_hip[1]),
            "left_wrist_score": float(scores[_L_WRIST]),
            "right_wrist_score": float(scores[_R_WRIST]),
            "left_elbow_score": float(scores[_L_ELBOW]),
            "right_elbow_score": float(scores[_R_ELBOW]),
            "left_hit": left_hit,
            "right_hit": right_hit,
        }
        confidence = 1.0 if triggered else 0.0
        return RuleResult(triggered, confidence, self.behavior, debug)

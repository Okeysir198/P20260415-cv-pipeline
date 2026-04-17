"""Handrail detection pose rule.

Rule: person's wrists are NOT within reach distance of any configured handrail zone
polygon. A wrist that is visible (score > threshold) and farther than
`hand_to_railing_px` pixels from all zone polygons triggers the alert.

Config keys from 10_inference.yaml pose_rules.no_handrail:
  hand_to_railing_px — pixel distance threshold (default 60 px at 640px width)

Handrail zones are normalized [0,1] polygons passed at construction time and
scaled to frame pixel dimensions at check time.
"""

from __future__ import annotations

import cv2
import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _base import PoseRule, RuleResult  # noqa: E402

_L_WRIST, _R_WRIST = 9, 10
_MIN_SCORE = 0.3


class HandrailDetector(PoseRule):
    """Detect absence of handrail contact via wrist-to-zone proximity."""

    behavior = "no_handrail"

    def __init__(
        self,
        handrail_zones: list[list[list[float]]] | None = None,
        hand_to_railing_px: float = 60.0,
    ) -> None:
        # Normalized [0,1] polygons; each zone is [[x,y], ...].
        self._zones: list[list[list[float]]] = handrail_zones or []
        self._reach_px = hand_to_railing_px

    def check(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        frame_buffer: list[np.ndarray] | None = None,
    ) -> RuleResult:
        if not self._zones:
            return RuleResult(False, 0.0, self.behavior, {"reason": "no handrail zones configured"})

        if keypoints is None or len(keypoints) < 11:
            return RuleResult(False, 0.0, self.behavior, {"reason": "insufficient keypoints"})

        # Derive frame dimensions from keypoint range (fallback: 640x480).
        frame_w = max(float(keypoints[:, 0].max()), 640.0)
        frame_h = max(float(keypoints[:, 1].max()), 480.0)

        # Scale zone polygons to pixel coords.
        pixel_zones = [
            np.array([[pt[0] * frame_w, pt[1] * frame_h] for pt in zone], dtype=np.float32)
            for zone in self._zones
        ]

        wrist_results: dict[str, float] = {}
        both_far = True  # assume no contact until proven otherwise

        for side, idx in [("left", _L_WRIST), ("right", _R_WRIST)]:
            if scores[idx] < _MIN_SCORE:
                wrist_results[side] = -1.0  # not visible
                continue

            wrist_pt = keypoints[idx].astype(np.float32)
            min_dist = float("inf")
            for zone_pts in pixel_zones:
                # Negative = inside polygon; 0 = on edge; positive = outside.
                dist = cv2.pointPolygonTest(zone_pts, (wrist_pt[0], wrist_pt[1]), measureDist=True)
                # Distance to nearest edge (treat inside as dist=0).
                edge_dist = 0.0 if dist >= 0 else abs(dist)
                min_dist = min(min_dist, edge_dist)

            wrist_results[side] = round(min_dist, 2)
            if min_dist <= self._reach_px:
                both_far = False  # at least one wrist is near a railing

        triggered = both_far and any(v >= 0 for v in wrist_results.values())

        debug = {
            "left_dist_px": wrist_results.get("left"),
            "right_dist_px": wrist_results.get("right"),
            "reach_threshold_px": self._reach_px,
            "zones_count": len(self._zones),
        }
        confidence = 1.0 if triggered else 0.0
        return RuleResult(triggered, confidence, self.behavior, debug)

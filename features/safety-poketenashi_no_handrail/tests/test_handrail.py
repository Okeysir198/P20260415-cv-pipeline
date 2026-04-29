"""Unit tests for HandrailDetector — synthetic COCO-17 keypoints, no GPU/data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_CODE_DIR))

from handrail_detector import HandrailDetector  # noqa: E402

# COCO-17 indices used by the rule.
_L_WRIST, _R_WRIST = 9, 10


def _make_keypoints(
    l_wrist: tuple[float, float] = (320.0, 240.0),
    r_wrist: tuple[float, float] = (320.0, 240.0),
    frame_w: float = 640.0,
    frame_h: float = 480.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (17, 2) keypoints + (17,) scores. Anchors frame extent via face kps.

    The detector derives frame_w/h from `keypoints[:, 0/1].max()` (with 640/480
    floors). We ensure those match `frame_w/frame_h` by placing a "head" point.
    """
    kpts = np.zeros((17, 2), dtype=np.float32)
    scores = np.full(17, 0.9, dtype=np.float32)
    # Set extent anchors so frame_w/h derive cleanly to provided values.
    kpts[0] = (frame_w, frame_h)  # nose at far corner — establishes frame extent
    kpts[_L_WRIST] = l_wrist
    kpts[_R_WRIST] = r_wrist
    return kpts, scores


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_zones_configured_disables_rule() -> None:
    """Empty handrail_zones → rule reports not triggered (disabled)."""
    detector = HandrailDetector(handrail_zones=[], hand_to_railing_px=60.0)
    kpts, scores = _make_keypoints()
    result = detector.check(kpts, scores)
    assert result.triggered is False
    assert result.behavior == "no_handrail"
    assert "no handrail zones configured" in result.debug_info.get("reason", "")


def test_no_zones_default_disabled() -> None:
    """Default constructor (None zones) is also disabled."""
    detector = HandrailDetector()
    kpts, scores = _make_keypoints()
    result = detector.check(kpts, scores)
    assert result.triggered is False


def test_wrist_inside_zone_not_triggered() -> None:
    """Wrist inside the handrail zone polygon → rule does NOT trigger."""
    # Zone: vertical strip on the left edge, normalised to [0,1].
    # In a 640x480 frame, this maps to x in [64, 96], y in [96, 432].
    zone = [[0.10, 0.20], [0.15, 0.20], [0.15, 0.90], [0.10, 0.90]]
    detector = HandrailDetector(handrail_zones=[zone], hand_to_railing_px=60.0)

    # Place left wrist inside the zone (x=80 ∈ [64, 96], y=240 ∈ [96, 432]).
    kpts, scores = _make_keypoints(l_wrist=(80.0, 240.0), r_wrist=(80.0, 240.0))
    result = detector.check(kpts, scores)
    assert result.triggered is False, result.debug_info
    assert result.debug_info["left_dist_px"] == 0.0
    assert result.debug_info["right_dist_px"] == 0.0


def test_wrists_far_from_zone_triggered() -> None:
    """Both wrists far from every zone → rule triggers (no_handrail)."""
    zone = [[0.10, 0.20], [0.15, 0.20], [0.15, 0.90], [0.10, 0.90]]
    detector = HandrailDetector(handrail_zones=[zone], hand_to_railing_px=60.0)

    # Wrists at x=500: distance to zone right edge (x=96) is 404 px, far above 60 threshold.
    kpts, scores = _make_keypoints(l_wrist=(500.0, 240.0), r_wrist=(500.0, 240.0))
    result = detector.check(kpts, scores)
    assert result.triggered is True
    assert result.confidence == 1.0
    assert result.debug_info["left_dist_px"] > 60.0
    assert result.debug_info["right_dist_px"] > 60.0


def test_one_wrist_near_zone_suppresses_alert() -> None:
    """At least one wrist within reach distance → rule does NOT trigger."""
    zone = [[0.10, 0.20], [0.15, 0.20], [0.15, 0.90], [0.10, 0.90]]
    detector = HandrailDetector(handrail_zones=[zone], hand_to_railing_px=60.0)

    # Left wrist near the zone (x=120, ~24 px from x=96 right edge → within 60 px),
    # right wrist far away (x=500).
    kpts, scores = _make_keypoints(l_wrist=(120.0, 240.0), r_wrist=(500.0, 240.0))
    result = detector.check(kpts, scores)
    assert result.triggered is False
    assert result.debug_info["left_dist_px"] <= 60.0


def test_low_score_wrists_marked_invisible() -> None:
    """Wrists below score threshold are not used (-1.0 in debug)."""
    zone = [[0.10, 0.20], [0.15, 0.20], [0.15, 0.90], [0.10, 0.90]]
    detector = HandrailDetector(handrail_zones=[zone], hand_to_railing_px=60.0)
    kpts, scores = _make_keypoints(l_wrist=(500.0, 240.0), r_wrist=(500.0, 240.0))
    scores[_L_WRIST] = 0.05  # below MIN_SCORE=0.3
    scores[_R_WRIST] = 0.05
    result = detector.check(kpts, scores)
    # Both wrists invisible -> all values negative -> not triggered.
    assert result.triggered is False
    assert result.debug_info["left_dist_px"] == -1.0
    assert result.debug_info["right_dist_px"] == -1.0


def test_insufficient_keypoints_short_circuits() -> None:
    detector = HandrailDetector(handrail_zones=[[[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]]])
    kpts = np.zeros((5, 2), dtype=np.float32)
    scores = np.ones(5, dtype=np.float32)
    result = detector.check(kpts, scores)
    assert result.triggered is False
    assert "insufficient" in result.debug_info.get("reason", "")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

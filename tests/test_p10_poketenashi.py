"""Test: Poketenashi pose rule detectors — hands-in-pockets, stair safety,
handrail, and RuleResult dataclass.

Real COCO-17 keypoints as numpy arrays — no mocks, no synthetic random data.
"""

import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import passed, failed, errors, run_test, run_all

# After the safety-poketenashi umbrella was split into per-rule features, each
# new feature folder ships its own copy of `_base.py` alongside its detector
# module. Add all three `code/` dirs to sys.path so `from _base import ...` and
# the detector imports below resolve. (All `_base.py` copies are identical, so
# whichever is found first is fine.)
_NEW_FEAT_BASE = ROOT / "features"
for _sub in (
    "safety-poketenashi_hands_in_pockets",
    "safety-poketenashi_stair_diagonal",
    "safety-poketenashi_no_handrail",
):
    sys.path.insert(0, str(_NEW_FEAT_BASE / _sub / "code"))

from _base import RuleResult, PoseRule  # noqa: E402
from hands_in_pockets_detector import HandsInPocketsDetector  # noqa: E402
from stair_safety_detector import StairSafetyDetector  # noqa: E402
from handrail_detector import HandrailDetector  # noqa: E402

# ---------------------------------------------------------------------------
# COCO-17 keypoint index constants
# ---------------------------------------------------------------------------
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
N_KEYPOINTS = 17


# ---------------------------------------------------------------------------
# Keypoint builders — anatomically plausible standing person in image coords
# (origin top-left, y increases downward).
# ---------------------------------------------------------------------------
def _standing_keypoints(
    *,
    cx: float = 320.0,
    shoulder_y: float = 180.0,
    hip_y: float = 300.0,
    l_wrist_x: float | None = None,
    l_wrist_y: float | None = None,
    r_wrist_x: float | None = None,
    r_wrist_y: float | None = None,
    l_elbow_x: float | None = None,
    l_elbow_y: float | None = None,
    r_elbow_x: float | None = None,
    r_elbow_y: float | None = None,
    score: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a (17,2) keypoint array + (17,) score array for a front-facing person.

    Defaults place arms at sides (wrists at hip level, elbows midway).
    Override specific joint positions to create different poses.
    """
    kps = np.zeros((N_KEYPOINTS, 2), dtype=np.float64)
    scores = np.full(N_KEYPOINTS, score, dtype=np.float64)

    # Head
    kps[NOSE] = [cx, shoulder_y - 60]

    # Shoulders — ~60 px apart, centred on cx
    kps[L_SHOULDER] = [cx - 30, shoulder_y]
    kps[R_SHOULDER] = [cx + 30, shoulder_y]

    # Hips — ~50 px apart
    kps[L_HIP] = [cx - 25, hip_y]
    kps[R_HIP] = [cx + 25, hip_y]

    # Elbows — default: at sides, midway between shoulder and hip
    mid_y = (shoulder_y + hip_y) / 2
    kps[L_ELBOW] = [l_elbow_x if l_elbow_x is not None else cx - 35,
                     l_elbow_y if l_elbow_y is not None else mid_y]
    kps[R_ELBOW] = [r_elbow_x if r_elbow_x is not None else cx + 35,
                     r_elbow_y if r_elbow_y is not None else mid_y]

    # Wrists — default: at sides, near hip level
    kps[L_WRIST] = [l_wrist_x if l_wrist_x is not None else cx - 30,
                     l_wrist_y if l_wrist_y is not None else hip_y + 10]
    kps[R_WRIST] = [r_wrist_x if r_wrist_x is not None else cx + 30,
                     r_wrist_y if r_wrist_y is not None else hip_y + 10]

    return kps, scores


def _standing_keypoints_custom_scores(
    *,
    score_map: dict[int, float] | None = None,
    default_score: float = 0.9,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Like _standing_keypoints but with per-keypoint score overrides."""
    kps, scores = _standing_keypoints(**kwargs, score=default_score)
    if score_map:
        for idx, s in score_map.items():
            scores[idx] = s
    return kps, scores


# ===================================================================
# RuleResult
# ===================================================================

def test_rule_result_fields():
    r = RuleResult(
        triggered=True,
        confidence=0.85,
        behavior="hands_in_pockets",
        debug_info={"left_hit": True},
    )
    assert r.triggered is True
    assert r.confidence == 0.85
    assert r.behavior == "hands_in_pockets"
    assert r.debug_info == {"left_hit": True}

    # Default debug_info is empty dict
    r2 = RuleResult(False, 0.0, "test")
    assert r2.debug_info == {}


# ===================================================================
# HandsInPocketsDetector
# ===================================================================

def test_hands_in_pockets_triggered_left():
    """Left wrist below left hip and inside torso centerline → triggered."""
    kps, scores = _standing_keypoints(
        l_wrist_x=320.0,   # at torso center (cx)
        l_wrist_y=320.0,   # below hip_y=300
    )
    det = HandsInPocketsDetector(wrist_below_hip_ratio=0.05, wrist_inside_torso_margin=0.08)
    result = det.check(kps, scores)
    assert result.triggered is True
    assert result.behavior == "hands_in_pockets"
    assert result.confidence == 1.0
    assert result.debug_info["left_hit"] is True


def test_hands_in_pockets_triggered_right():
    """Right wrist below right hip and inside torso centerline → triggered."""
    kps, scores = _standing_keypoints(
        r_wrist_x=320.0,   # at torso center
        r_wrist_y=320.0,   # below hip_y=300
    )
    det = HandsInPocketsDetector(wrist_below_hip_ratio=0.05, wrist_inside_torso_margin=0.08)
    result = det.check(kps, scores)
    assert result.triggered is True
    assert result.debug_info["right_hit"] is True


def test_hands_in_pockets_not_triggered():
    """Wrists above shoulders — clearly not in pockets."""
    kps, scores = _standing_keypoints(
        l_wrist_x=200.0, l_wrist_y=120.0,   # left arm raised high
        r_wrist_x=440.0, r_wrist_y=120.0,   # right arm raised high
    )
    det = HandsInPocketsDetector()
    result = det.check(kps, scores)
    assert result.triggered is False
    assert result.confidence == 0.0


def test_hands_in_pockets_insufficient_keypoints():
    """Fewer than 17 keypoints → not triggered."""
    kps = np.zeros((10, 2), dtype=np.float64)
    scores = np.full(10, 0.9, dtype=np.float64)
    det = HandsInPocketsDetector()
    result = det.check(kps, scores)
    assert result.triggered is False
    assert "insufficient keypoints" in result.debug_info["reason"]


def test_hands_in_pockets_low_hip_score():
    """Hips invisible (low score) → not triggered."""
    kps, scores = _standing_keypoints(
        l_wrist_x=320.0, l_wrist_y=320.0,
    )
    scores[L_HIP] = 0.1
    scores[R_HIP] = 0.1
    det = HandsInPocketsDetector()
    result = det.check(kps, scores)
    assert result.triggered is False
    assert "hips/shoulders not visible" in result.debug_info["reason"]


# ===================================================================
# StairSafetyDetector
# ===================================================================

def test_stair_horizontal_trajectory():
    """Horizontal hip movement (walking along corridor) → not triggered."""
    det = StairSafetyDetector(max_diagonal_angle_deg=20.0, min_frames=3)
    for x in range(300, 303):
        kps, scores = _standing_keypoints(cx=float(x))
        result = det.check(kps, scores)
    # Last result: angle should be ~0° (horizontal)
    assert result.triggered is False
    assert "trajectory_angle_deg" in result.debug_info
    assert result.debug_info["trajectory_angle_deg"] < 20.0


def test_stair_diagonal_trajectory():
    """Diagonal hip movement > threshold → triggered."""
    det = StairSafetyDetector(max_diagonal_angle_deg=20.0, min_frames=3)
    # Simulate 3 frames moving diagonally: +20px right, +30px down per frame
    # Total displacement: (60, 90) → angle ≈ atan2(90, 60) ≈ 56°
    for i in range(3):
        kps, scores = _standing_keypoints(
            cx=300.0 + i * 20,
            hip_y=300.0 + i * 30,
        )
        result = det.check(kps, scores)
    assert result.triggered is True
    assert result.debug_info["trajectory_angle_deg"] > 20.0


def test_stair_buffering():
    """Fewer than min_frames → not triggered (buffering)."""
    det = StairSafetyDetector(max_diagonal_angle_deg=5.0, min_frames=5)
    kps, scores = _standing_keypoints(cx=300.0, hip_y=300.0)
    # Only feed 4 frames (min_frames=5)
    for i in range(4):
        result = det.check(kps, scores)
    assert result.triggered is False
    assert "buffering" in result.debug_info["reason"]


def test_stair_reset():
    """Reset clears buffer — next check starts fresh."""
    det = StairSafetyDetector(max_diagonal_angle_deg=20.0, min_frames=3)
    kps, scores = _standing_keypoints(cx=300.0, hip_y=300.0)
    det.check(kps, scores)
    det.check(kps, scores)
    det.reset()
    # After reset, one more frame should still be buffering
    result = det.check(kps, scores)
    assert result.triggered is False
    assert "buffering" in result.debug_info["reason"]


# ===================================================================
# HandrailDetector
# ===================================================================

def test_handrail_wrist_near_zone():
    """Wrist within reach_px of a handrail zone → not triggered."""
    # Zone covers left side of frame, normalized [0,1]
    zones = [[[0.0, 0.5], [0.15, 0.5], [0.15, 1.0], [0.0, 1.0]]]
    # Place left wrist at (48, 310) in a ~640-wide frame
    # Scaled zone left edge at x=0, right edge at x=96 → wrist at 48 is inside
    kps, scores = _standing_keypoints(l_wrist_x=48.0, l_wrist_y=310.0)
    det = HandrailDetector(handrail_zones=zones, hand_to_railing_px=60.0)
    result = det.check(kps, scores)
    assert result.triggered is False


def test_handrail_wrist_far_from_zone():
    """Wrist far from all handrail zones → triggered."""
    # Zone on the far left; person centred on the right
    zones = [[[0.0, 0.5], [0.1, 0.5], [0.1, 1.0], [0.0, 1.0]]]
    # Person at cx=500; left wrist at 470, right wrist at 530
    # Zone in pixel space: x from 0 to 64 → wrist at 470 is far away
    kps, scores = _standing_keypoints(cx=500.0)
    det = HandrailDetector(handrail_zones=zones, hand_to_railing_px=60.0)
    result = det.check(kps, scores)
    assert result.triggered is True
    assert result.confidence == 1.0


def test_handrail_no_zones_configured():
    """Empty zones list → not triggered."""
    kps, scores = _standing_keypoints()
    det = HandrailDetector(handrail_zones=[], hand_to_railing_px=60.0)
    result = det.check(kps, scores)
    assert result.triggered is False
    assert "no handrail zones configured" in result.debug_info["reason"]


def test_handrail_inside_polygon():
    """Wrist inside polygon → distance=0 → not triggered."""
    # Large zone covering most of the frame
    zones = [[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]]
    kps, scores = _standing_keypoints(cx=320.0)
    det = HandrailDetector(handrail_zones=zones, hand_to_railing_px=60.0)
    result = det.check(kps, scores)
    assert result.triggered is False
    # Wrist should be inside → distance 0
    assert result.debug_info["left_dist_px"] == 0.0


# ===================================================================
# Main runner
# ===================================================================
#
# NOTE: the legacy `PointingCallingDetector` rule (and its 5 tests) was
# dropped as part of the safety-poketenashi split — the directional
# `safety-poketenashi_point_and_call` feature subsumes it. Tests for that
# feature live in their own file under `features/safety-poketenashi_point_and_call/tests/`.

ALL_TESTS = [
    # RuleResult
    ("rule_result_fields", test_rule_result_fields),
    # HandsInPocketsDetector
    ("hands_in_pockets_triggered_left", test_hands_in_pockets_triggered_left),
    ("hands_in_pockets_triggered_right", test_hands_in_pockets_triggered_right),
    ("hands_in_pockets_not_triggered", test_hands_in_pockets_not_triggered),
    ("hands_in_pockets_insufficient_keypoints", test_hands_in_pockets_insufficient_keypoints),
    ("hands_in_pockets_low_hip_score", test_hands_in_pockets_low_hip_score),
    # StairSafetyDetector
    ("stair_horizontal_trajectory", test_stair_horizontal_trajectory),
    ("stair_diagonal_trajectory", test_stair_diagonal_trajectory),
    ("stair_buffering", test_stair_buffering),
    ("stair_reset", test_stair_reset),
    # HandrailDetector
    ("handrail_wrist_near_zone", test_handrail_wrist_near_zone),
    ("handrail_wrist_far_from_zone", test_handrail_wrist_far_from_zone),
    ("handrail_no_zones_configured", test_handrail_no_zones_configured),
    ("handrail_inside_polygon", test_handrail_inside_polygon),
]

if __name__ == "__main__":
    run_all(ALL_TESTS, title="Poketenashi Pose Rule Detectors")

"""Tests for the hands-in-pockets pose rule (synthetic COCO-17 keypoints)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_FEAT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_FEAT / "code"))

from hands_in_pockets_detector import HandsInPocketsDetector  # noqa: E402

# COCO-17 indices used by the rule.
_NOSE = 0
_L_SH, _R_SH = 5, 6
_L_EL, _R_EL = 7, 8
_L_WR, _R_WR = 9, 10
_L_HIP, _R_HIP = 11, 12
_L_KNEE, _R_KNEE = 13, 14
_L_ANK, _R_ANK = 15, 16


def _baseline_person() -> tuple[np.ndarray, np.ndarray]:
    """Standing person facing the camera, arms hanging at the sides.

    Image x increases to the right, y increases downward. Hips are at y=260,
    so wrists below hip line means y > 260 + small margin.
    """
    kpts = np.zeros((17, 2), dtype=np.float32)
    cx = 200.0
    kpts[_NOSE] = [cx, 100.0]
    kpts[_L_SH] = [cx - 40, 150.0]
    kpts[_R_SH] = [cx + 40, 150.0]
    kpts[_L_HIP] = [cx - 25, 260.0]
    kpts[_R_HIP] = [cx + 25, 260.0]
    kpts[_L_KNEE] = [cx - 25, 360.0]
    kpts[_R_KNEE] = [cx + 25, 360.0]
    kpts[_L_ANK] = [cx - 25, 450.0]
    kpts[_R_ANK] = [cx + 25, 450.0]
    # Arms hanging straight down, wrists OUTSIDE the torso band (next to hips).
    kpts[_L_EL] = [cx - 40, 220.0]
    kpts[_R_EL] = [cx + 40, 220.0]
    kpts[_L_WR] = [cx - 40, 290.0]
    kpts[_R_WR] = [cx + 40, 290.0]

    scores = np.full(17, 0.9, dtype=np.float32)
    return kpts, scores


def test_arms_at_sides_does_not_trigger():
    """Wrists below hips but OUTSIDE the torso band -> no trigger."""
    kpts, scores = _baseline_person()
    det = HandsInPocketsDetector()
    res = det.check(kpts, scores)
    assert res.triggered is False
    assert res.behavior == "hands_in_pockets"
    assert res.debug_info["left_hit"] is False
    assert res.debug_info["right_hit"] is False


def test_right_wrist_in_pocket_per_side_triggers():
    """Right wrist below hip AND near torso centerline -> per-side hit.

    The default require_both_sides=True means trigger only when BOTH sides
    are in the pocket (the violation `hands_in_pockets` is plural). This
    test isolates the per-side gate by setting require_both_sides=False.
    """
    kpts, scores = _baseline_person()
    # torso_cx = (l_hip.x + r_hip.x)/2 = 200; torso_width = 50;
    # margin = 0.08 * 50 = 4 px. Place wrist at cx (centerline).
    kpts[_R_WR] = [200.0, 320.0]  # x = torso_cx, y well below hip
    det = HandsInPocketsDetector(require_both_sides=False)
    res = det.check(kpts, scores)
    assert res.triggered is True
    assert res.debug_info["right_hit"] is True
    assert res.confidence == 1.0


def test_left_wrist_in_pocket_per_side_triggers():
    kpts, scores = _baseline_person()
    kpts[_L_WR] = [200.0, 320.0]
    det = HandsInPocketsDetector(require_both_sides=False)
    res = det.check(kpts, scores)
    assert res.triggered is True
    assert res.debug_info["left_hit"] is True


def test_require_both_sides_flag():
    """require_both_sides=True flips the combiner from OR to AND.

    Default is OR (either hand in pocket → trigger) for recall robustness.
    Strict deployments can opt into AND semantics where both hands are
    required.
    """
    kpts, scores = _baseline_person()
    # One hand in pocket, the other dangling at the side.
    kpts[_R_WR] = [200.0, 320.0]

    or_det = HandsInPocketsDetector(require_both_sides=False)  # default
    assert or_det.check(kpts, scores).triggered is True

    and_det = HandsInPocketsDetector(require_both_sides=True)
    assert and_det.check(kpts, scores).triggered is False

    # Both hands in pocket: AND-mode trigger.
    kpts[_L_WR] = [200.0, 320.0]
    assert and_det.check(kpts, scores).triggered is True


def test_wrist_above_hip_does_not_trigger():
    """Wrist near torso centerline but ABOVE the hip line -> no trigger."""
    kpts, scores = _baseline_person()
    # x at torso centerline, y above hip y (smaller image-y == higher up).
    kpts[_R_WR] = [200.0, 200.0]
    det = HandsInPocketsDetector()
    res = det.check(kpts, scores)
    assert res.debug_info["right_hit"] is False


def test_low_hip_score_returns_no_trigger():
    """If both hips are unreliable, rule abstains (cannot estimate torso)."""
    kpts, scores = _baseline_person()
    scores[_L_HIP] = 0.0
    scores[_R_HIP] = 0.0
    det = HandsInPocketsDetector()
    res = det.check(kpts, scores)
    assert res.triggered is False
    assert "hips/shoulders not visible" in res.debug_info.get("reason", "")


def test_too_few_keypoints_returns_no_trigger():
    kpts = np.zeros((4, 2), dtype=np.float32)
    scores = np.zeros(4, dtype=np.float32)
    det = HandsInPocketsDetector()
    res = det.check(kpts, scores)
    assert res.triggered is False
    assert "insufficient keypoints" in res.debug_info.get("reason", "")


def test_low_wrist_score_uses_elbow_proxy():
    """Low-confidence wrist falls back to elbow position as the in-pocket signal.

    In-pocket pose drops the elbow to ~hip height and keeps it tucked near the
    body. Pointing/gesturing raises the elbow toward shoulder height. Reaching
    swings it out to the side. The elbow check rejects the latter two.
    """
    kpts, scores = _baseline_person()
    scores[_R_WR] = 0.05  # wrist occluded
    # Tuck the elbow inward (forward-bent in-pocket pose). torso_cx=200,
    # torso_width=50, default elbow margin 0.7 → ±35 px around centerline.
    # Place at x=215 (15 px from cx, inside band) and y=240 (below midpoint).
    kpts[_R_EL] = [215.0, 240.0]

    # In-pocket: elbow dropped + tucked. Use require_both_sides=False to
    # isolate the per-side gate.
    det = HandsInPocketsDetector(require_both_sides=False)
    res = det.check(kpts, scores)
    assert res.debug_info["right_hit"] is True

    # Raise the elbow to shoulder height (pointing pose).
    kpts[_R_EL] = [215.0, 150.0]
    res = det.check(kpts, scores)
    assert res.debug_info["right_hit"] is False


def test_low_wrist_and_low_elbow_abstains():
    """If both wrist and elbow are low-confidence, the rule abstains (no trigger)."""
    kpts, scores = _baseline_person()
    scores[_R_WR] = 0.05
    scores[_R_EL] = 0.05
    det = HandsInPocketsDetector()
    res = det.check(kpts, scores)
    assert res.debug_info["right_hit"] is False


def test_widening_torso_margin_admits_wrist():
    """Wider margin -> a wrist that was just outside the band now triggers."""
    kpts, scores = _baseline_person()
    # Place wrist at x = torso_cx + 6 (margin default 0.08 * torso_width(50) = 4 px -> outside).
    kpts[_R_WR] = [206.0, 320.0]
    strict = HandsInPocketsDetector(
        wrist_inside_torso_margin=0.08, require_both_sides=False
    )
    permissive = HandsInPocketsDetector(
        wrist_inside_torso_margin=0.20, require_both_sides=False
    )
    assert strict.check(kpts, scores).debug_info["right_hit"] is False
    assert permissive.check(kpts, scores).debug_info["right_hit"] is True

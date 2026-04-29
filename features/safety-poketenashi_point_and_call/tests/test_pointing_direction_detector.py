"""Tests for the per-frame pointing-direction detector."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_FEAT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_FEAT / "code"))

from pointing_direction_detector import PointingDirectionDetector  # noqa: E402


# COCO-17 indices used by the rule.
_NOSE = 0
_L_EYE, _R_EYE, _L_EAR, _R_EAR = 1, 2, 3, 4
_L_SH, _R_SH = 5, 6
_L_EL, _R_EL = 7, 8
_L_WR, _R_WR = 9, 10
_L_HIP, _R_HIP = 11, 12
_L_KNEE, _R_KNEE = 13, 14
_L_ANK, _R_ANK = 15, 16


def _baseline_person() -> tuple[np.ndarray, np.ndarray]:
    """Return COCO-17 keypoints for a person standing facing the camera.

    Image x increases to the right, y increases downward.
    Layout: nose top-center, shoulders below, hips below shoulders,
    arms hanging straight down at the body's sides.
    """
    kpts = np.zeros((17, 2), dtype=np.float32)
    cx = 200.0  # image center x for the person
    # Head
    kpts[_NOSE] = [cx, 100.0]
    kpts[_L_EYE] = [cx - 8, 95.0]
    kpts[_R_EYE] = [cx + 8, 95.0]
    kpts[_L_EAR] = [cx - 14, 100.0]
    kpts[_R_EAR] = [cx + 14, 100.0]
    # Shoulders: L on the LEFT side of the image (smaller x),
    # R on the RIGHT side (larger x).
    kpts[_L_SH] = [cx - 40, 150.0]
    kpts[_R_SH] = [cx + 40, 150.0]
    # Hips, knees, ankles (downward).
    kpts[_L_HIP] = [cx - 25, 260.0]
    kpts[_R_HIP] = [cx + 25, 260.0]
    kpts[_L_KNEE] = [cx - 25, 360.0]
    kpts[_R_KNEE] = [cx + 25, 360.0]
    kpts[_L_ANK] = [cx - 25, 450.0]
    kpts[_R_ANK] = [cx + 25, 450.0]
    # Arms hanging straight down.
    kpts[_L_EL] = [cx - 40, 220.0]
    kpts[_R_EL] = [cx + 40, 220.0]
    kpts[_L_WR] = [cx - 40, 280.0]
    kpts[_R_WR] = [cx + 40, 280.0]

    scores = np.full(17, 0.9, dtype=np.float32)
    return kpts, scores


def test_rest_arm_at_side_is_neutral():
    kpts, scores = _baseline_person()
    det = PointingDirectionDetector()
    res = det.check(kpts, scores)
    assert res.triggered is False
    assert res.debug_info["label"] == "neutral"


def test_right_arm_extended_to_image_right_is_point_right():
    kpts, scores = _baseline_person()
    # Right shoulder is at (cx+40, 150). Extend wrist to far-right side,
    # roughly horizontal, with elbow on the path so elbow angle ~180.
    r_sh = kpts[_R_SH].copy()
    kpts[_R_EL] = [r_sh[0] + 70.0, r_sh[1]]
    kpts[_R_WR] = [r_sh[0] + 140.0, r_sh[1]]
    det = PointingDirectionDetector()
    res = det.check(kpts, scores)
    assert res.debug_info["label"] == "point_right", res.debug_info


def test_left_arm_extended_to_image_left_is_point_left():
    kpts, scores = _baseline_person()
    l_sh = kpts[_L_SH].copy()
    kpts[_L_EL] = [l_sh[0] - 70.0, l_sh[1]]
    kpts[_L_WR] = [l_sh[0] - 140.0, l_sh[1]]
    det = PointingDirectionDetector()
    res = det.check(kpts, scores)
    assert res.debug_info["label"] == "point_left", res.debug_info


def test_arm_extended_toward_body_centerline_is_point_front():
    """Arm extended forward (toward camera) is foreshortened: the wrist
    projects above the shoulder (toward the head) when the arm points
    out of the image plane. In the body-relative torso frame this is
    azimuth ~+90 deg (along -e_y, "up" the body).
    """
    kpts, scores = _baseline_person()
    r_sh = kpts[_R_SH].copy()
    # Forward-pointing arm modelled via heavy upward foreshortening:
    # in body-relative azimuth, "front" is along -e_y (~+90 deg).
    # Wrist primarily above the shoulder (small x offset, large -y),
    # so the projection onto -e_y dominates -> azimuth ~+90 deg.
    # arm_elevation = atan2(60, 5) ~ 85 deg, so we also relax the
    # elevation cap for this configuration.
    kpts[_R_EL] = [r_sh[0] - 3.0, r_sh[1] - 30.0]
    kpts[_R_WR] = [r_sh[0] - 5.0, r_sh[1] - 60.0]
    det = PointingDirectionDetector(arm_elevation_max_deg=89.0)
    res = det.check(kpts, scores)
    assert res.debug_info["label"] == "point_front", res.debug_info


def test_bent_arm_is_neutral():
    kpts, scores = _baseline_person()
    r_sh = kpts[_R_SH].copy()
    # Elbow extended to the side, wrist back near shoulder -> elbow angle ~90
    kpts[_R_EL] = [r_sh[0] + 70.0, r_sh[1]]
    kpts[_R_WR] = [r_sh[0] + 70.0, r_sh[1] - 70.0]  # 90 deg bend at elbow
    det = PointingDirectionDetector()
    res = det.check(kpts, scores)
    assert res.debug_info["label"] == "neutral"


def test_arm_raised_vertical_above_threshold_is_neutral():
    kpts, scores = _baseline_person()
    r_sh = kpts[_R_SH].copy()
    # Arm pointing straight up (elevation ~90 deg).
    kpts[_R_EL] = [r_sh[0], r_sh[1] - 70.0]
    kpts[_R_WR] = [r_sh[0], r_sh[1] - 140.0]
    det = PointingDirectionDetector(arm_elevation_max_deg=45.0)
    res = det.check(kpts, scores)
    # Elevation > threshold, so arm not "extending" -> neutral.
    assert res.debug_info["label"] == "neutral"


def test_low_confidence_keypoints_yield_invalid():
    kpts, scores = _baseline_person()
    scores[_R_WR] = 0.05  # below default min_keypoint_score
    det = PointingDirectionDetector()
    res = det.check(kpts, scores)
    assert res.debug_info["label"] == "invalid"


def test_too_few_keypoints_returns_invalid():
    kpts = np.zeros((4, 2), dtype=np.float32)
    scores = np.zeros(4, dtype=np.float32)
    det = PointingDirectionDetector()
    res = det.check(kpts, scores)
    assert res.debug_info["label"] == "invalid"

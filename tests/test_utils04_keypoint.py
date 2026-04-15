"""Test 04: Utils — keypoint_utils geometry helpers (real numpy, no mocks)."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import run_all
from core.p06_models.pose_base import COCO_KEYPOINT_NAMES as COCO_NAMES
from utils.keypoint_utils import (
    _compute_torso_data,
    body_orientation,
    get_angle,
    get_distance,
    get_keypoint,
    get_midpoint,
    ground_proximity,
    hip_shoulder_ratio,
)


def _upright_keypoints(score: float = 0.9) -> np.ndarray:
    """Synthetic upright pose: shoulders y=100, hips y=200, image coords."""
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[:, 2] = score
    # shoulders
    kp[5] = [95, 100, score]   # left_shoulder
    kp[6] = [105, 100, score]  # right_shoulder
    # hips (same x as shoulders so torso_length is exactly 100)
    kp[11] = [95, 200, score]  # left_hip
    kp[12] = [105, 200, score] # right_hip
    return kp


def test_get_keypoint():
    kp = _upright_keypoints()
    ls = get_keypoint(kp, COCO_NAMES, "left_shoulder")
    assert np.allclose(ls, [95, 100, 0.9])

    with pytest.raises(KeyError):
        get_keypoint(kp, COCO_NAMES, "tail")


def test_get_midpoint():
    kp = _upright_keypoints()
    mid = get_midpoint(kp, COCO_NAMES, "left_shoulder", "right_shoulder")
    assert np.allclose(mid[:2], [100, 100])
    assert math.isclose(mid[2], 0.9, rel_tol=1e-5)


def test_get_distance():
    kp = _upright_keypoints()
    d = get_distance(kp, COCO_NAMES, "left_shoulder", "left_hip")
    assert math.isclose(d, 100.0, rel_tol=1e-3), d

    # Low confidence → 0
    kp2 = _upright_keypoints()
    kp2[5, 2] = 0.0
    assert get_distance(kp2, COCO_NAMES, "left_shoulder", "left_hip") == 0.0


def test_get_angle_right_angle():
    kp = _upright_keypoints()
    # Build explicit 90° at left_elbow
    kp[5] = [0, 0, 0.9]    # left_shoulder (A)
    kp[7] = [100, 0, 0.9]  # left_elbow    (B)
    kp[9] = [100, 100, 0.9]  # left_wrist  (C)
    angle = get_angle(kp, COCO_NAMES, "left_shoulder", "left_elbow", "left_wrist")
    assert math.isclose(angle, 90.0, abs_tol=1e-3), angle

    # Missing conf → -1.0
    kp[7, 2] = 0.0
    assert get_angle(kp, COCO_NAMES, "left_shoulder", "left_elbow", "left_wrist") == -1.0


def test_compute_torso_data():
    kp = _upright_keypoints()
    data = _compute_torso_data(kp, COCO_NAMES, min_score=0.3)
    assert data is not None
    assert math.isclose(data["torso_length"], 100.0, rel_tol=1e-3)
    assert math.isclose(data["vertical_diff"], 100.0, rel_tol=1e-3)

    # Insufficient confidence
    kp_bad = _upright_keypoints(score=0.1)
    assert _compute_torso_data(kp_bad, COCO_NAMES, min_score=0.3) is None


def test_body_orientation_upright():
    kp = _upright_keypoints()
    assert body_orientation(kp, COCO_NAMES) == "upright"


def test_body_orientation_horizontal():
    kp = _upright_keypoints()
    # Move hips to same height as shoulders → ratio ~0 → horizontal
    kp[11] = [95, 100, 0.9]
    kp[12] = [105, 100, 0.9]
    # Need torso_length > 0 — stretch left_hip horizontally away from left_shoulder
    kp[11] = [195, 100, 0.9]
    assert body_orientation(kp, COCO_NAMES) == "horizontal"


def test_body_orientation_inverted():
    kp = _upright_keypoints()
    # swap: shoulders below hips
    kp[5] = [90, 200, 0.9]; kp[6] = [110, 200, 0.9]
    kp[11] = [95, 100, 0.9]; kp[12] = [105, 100, 0.9]
    assert body_orientation(kp, COCO_NAMES) == "inverted"


def test_body_orientation_unknown_low_conf():
    kp = _upright_keypoints(score=0.1)
    assert body_orientation(kp, COCO_NAMES) == "unknown"


def test_hip_shoulder_ratio():
    kp = _upright_keypoints()
    ratio = hip_shoulder_ratio(kp, COCO_NAMES)
    assert ratio is not None
    assert math.isclose(ratio, 1.0, rel_tol=1e-3), ratio

    kp_bad = _upright_keypoints(score=0.1)
    assert hip_shoulder_ratio(kp_bad, COCO_NAMES) is None


def test_ground_proximity():
    kp = _upright_keypoints()
    # hip_y = 200, image_height=400 → 0.5
    frac = ground_proximity(kp, COCO_NAMES, image_height=400)
    assert frac is not None and math.isclose(frac, 0.5, rel_tol=1e-3), frac

    # Insufficient confidence
    kp_bad = _upright_keypoints(score=0.1)
    assert ground_proximity(kp_bad, COCO_NAMES, image_height=400) is None

    # Zero image height → None
    assert ground_proximity(kp, COCO_NAMES, image_height=0) is None


if __name__ == "__main__":
    run_all([
        ("get_keypoint", test_get_keypoint),
        ("get_midpoint", test_get_midpoint),
        ("get_distance", test_get_distance),
        ("get_angle_right_angle", test_get_angle_right_angle),
        ("compute_torso_data", test_compute_torso_data),
        ("body_orientation_upright", test_body_orientation_upright),
        ("body_orientation_horizontal", test_body_orientation_horizontal),
        ("body_orientation_inverted", test_body_orientation_inverted),
        ("body_orientation_unknown", test_body_orientation_unknown_low_conf),
        ("hip_shoulder_ratio", test_hip_shoulder_ratio),
        ("ground_proximity", test_ground_proximity),
    ], title="Test utils04: keypoint_utils")

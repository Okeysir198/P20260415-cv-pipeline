"""Keypoint access utilities for pose-based rules.

All functions reference keypoints by name, never by raw index, so they
work with any pose model (COCO 17, MediaPipe 33, etc.).
"""

import math
from typing import List, Optional

import numpy as np


def get_keypoint(
    keypoints: np.ndarray,
    keypoint_names: List[str],
    name: str,
) -> np.ndarray:
    """Get a single keypoint by name.

    Args:
        keypoints: (K, 3) array [x, y, score].
        keypoint_names: Ordered list of K keypoint names.
        name: Keypoint name to look up.

    Returns:
        (3,) array [x, y, score].

    Raises:
        KeyError: If name not found in keypoint_names.
    """
    if name not in keypoint_names:
        raise KeyError(f"Keypoint '{name}' not found. Available: {keypoint_names}")
    idx = keypoint_names.index(name)
    return keypoints[idx]


def get_midpoint(
    keypoints: np.ndarray,
    keypoint_names: List[str],
    name_a: str,
    name_b: str,
) -> np.ndarray:
    """Get midpoint between two keypoints.

    Returns:
        (3,) array [x, y, min_score].
    """
    kpt_a = get_keypoint(keypoints, keypoint_names, name_a)
    kpt_b = get_keypoint(keypoints, keypoint_names, name_b)
    mid = (kpt_a[:2] + kpt_b[:2]) / 2.0
    score = min(kpt_a[2], kpt_b[2])
    return np.array([mid[0], mid[1], score], dtype=np.float32)


def get_distance(
    keypoints: np.ndarray,
    keypoint_names: List[str],
    name_a: str,
    name_b: str,
) -> float:
    """Euclidean distance between two keypoints.

    Returns:
        Distance in pixels. Returns 0.0 if either keypoint has score <= 0.
    """
    kpt_a = get_keypoint(keypoints, keypoint_names, name_a)
    kpt_b = get_keypoint(keypoints, keypoint_names, name_b)
    if kpt_a[2] <= 0 or kpt_b[2] <= 0:
        return 0.0
    return float(np.linalg.norm(kpt_a[:2] - kpt_b[:2]))


def get_angle(
    keypoints: np.ndarray,
    keypoint_names: List[str],
    name_a: str,
    name_b: str,
    name_c: str,
) -> float:
    """Compute angle at joint B between segments A-B and B-C.

    Args:
        name_a, name_b, name_c: Keypoint names forming the angle.

    Returns:
        Angle in degrees [0, 180]. Returns -1.0 if any keypoint has score <= 0.
    """
    kpt_a = get_keypoint(keypoints, keypoint_names, name_a)
    kpt_b = get_keypoint(keypoints, keypoint_names, name_b)
    kpt_c = get_keypoint(keypoints, keypoint_names, name_c)

    if kpt_a[2] <= 0 or kpt_b[2] <= 0 or kpt_c[2] <= 0:
        return -1.0

    vec_ba = kpt_a[:2] - kpt_b[:2]
    vec_bc = kpt_c[:2] - kpt_b[:2]

    cos_angle = np.dot(vec_ba, vec_bc) / (
        np.linalg.norm(vec_ba) * np.linalg.norm(vec_bc) + 1e-8
    )
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(math.degrees(math.acos(cos_angle)))


def _compute_torso_data(
    keypoints: np.ndarray,
    keypoint_names: List[str],
    min_score: float,
) -> Optional[dict]:
    """Compute shared torso geometry data for body_orientation and hip_shoulder_ratio.

    Returns:
        Dict with mid_shoulder, mid_hip, torso_length, vertical_diff,
        or None if keypoints are missing or confidence is too low.
    """
    try:
        mid_shoulder = get_midpoint(keypoints, keypoint_names, "left_shoulder", "right_shoulder")
        mid_hip = get_midpoint(keypoints, keypoint_names, "left_hip", "right_hip")
    except KeyError:
        return None

    if mid_shoulder[2] < min_score or mid_hip[2] < min_score:
        return None

    torso_length = get_distance(keypoints, keypoint_names, "left_shoulder", "left_hip")
    if torso_length < 1e-3:
        return None

    vertical_diff = mid_hip[1] - mid_shoulder[1]  # image coords: y increases downward
    return {
        "mid_shoulder": mid_shoulder,
        "mid_hip": mid_hip,
        "torso_length": torso_length,
        "vertical_diff": vertical_diff,
    }


def body_orientation(
    keypoints: np.ndarray,
    keypoint_names: List[str],
    min_score: float = 0.3,
) -> str:
    """Determine body orientation from keypoints.

    Uses the hip-shoulder vertical relationship:
    - "upright": shoulders significantly above hips
    - "horizontal": shoulders roughly at same height as hips
    - "inverted": shoulders below hips
    - "unknown": insufficient keypoint confidence

    Args:
        keypoints: (K, 3) keypoints.
        keypoint_names: Ordered keypoint names.
        min_score: Minimum keypoint confidence to use.

    Returns:
        One of "upright", "horizontal", "inverted", "unknown".
    """
    torso = _compute_torso_data(keypoints, keypoint_names, min_score)
    if torso is None:
        return "unknown"

    # Vertical difference: positive = shoulders above hips (upright)
    ratio = torso["vertical_diff"] / torso["torso_length"]

    if ratio > 0.5:
        return "upright"
    elif ratio < -0.5:
        return "inverted"
    else:
        return "horizontal"


def hip_shoulder_ratio(
    keypoints: np.ndarray,
    keypoint_names: List[str],
    min_score: float = 0.3,
) -> Optional[float]:
    """Compute hip-shoulder height ratio for fall detection.

    From the technical spec: ratio >= 0.05 indicates standing, lower
    values indicate potential fall.

    Returns:
        Ratio of vertical distance / torso length, or None if
        insufficient confidence.
    """
    torso = _compute_torso_data(keypoints, keypoint_names, min_score)
    if torso is None:
        return None

    return float(torso["vertical_diff"] / torso["torso_length"])


def ground_proximity(
    keypoints: np.ndarray,
    keypoint_names: List[str],
    image_height: int,
    min_score: float = 0.3,
) -> Optional[float]:
    """Compute hip vertical position as fraction of image height.

    From the technical spec: hip_y > 0.7 indicates ground proximity
    (potential fall).

    Returns:
        Hip y-position as fraction of image height [0, 1], or None
        if insufficient confidence.
    """
    try:
        mid_hip = get_midpoint(keypoints, keypoint_names, "left_hip", "right_hip")
    except KeyError:
        return None

    if mid_hip[2] < min_score:
        return None

    return float(mid_hip[1] / image_height) if image_height > 0 else None

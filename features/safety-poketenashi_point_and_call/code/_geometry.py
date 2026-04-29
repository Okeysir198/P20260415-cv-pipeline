"""Pure-numpy geometric helpers for pointing-direction analysis.

Image coords: x increases to the right, y increases downward.
Body-relative ("torso") frame:
    origin = midpoint of left + right shoulder
    e_x    = unit vector pointing from LEFT shoulder to RIGHT shoulder
             (so +e_x is the "right side of the body" in the image)
    e_y    = orthogonal direction, oriented to point DOWN the body
             (toward hips in image y).

In that body frame, the arm azimuth is measured CCW from +e_x in the
(e_x, -e_y) plane, so:
    0   degrees -> +e_x      (right side of body)
    90  degrees -> -e_y      (up / front-of-camera in body frame)
    180 degrees -> -e_x      (left side of body)
   -90  degrees -> +e_y      (down the body / behind)

This convention makes the front-pointing direction a single threshold
around 90 degrees regardless of which way the worker faces the camera.
"""

from __future__ import annotations

import math

import numpy as np


def _angle_at_vertex_deg(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> float:
    """Angle at ``vertex`` in the triangle a-vertex-b, in degrees."""
    va = np.asarray(a, dtype=np.float64) - np.asarray(vertex, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64) - np.asarray(vertex, dtype=np.float64)
    denom = float(np.linalg.norm(va)) * float(np.linalg.norm(vb)) + 1e-9
    cos_a = float(np.dot(va, vb)) / denom
    return float(math.degrees(math.acos(float(np.clip(cos_a, -1.0, 1.0)))))


def elbow_angle_deg(
    shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray
) -> float:
    """Angle at the elbow vertex (shoulder-elbow-wrist), in degrees."""
    return _angle_at_vertex_deg(shoulder, elbow, wrist)


def arm_elevation_deg(shoulder: np.ndarray, wrist: np.ndarray) -> float:
    """Absolute elevation of the shoulder->wrist vector from horizontal.

    Image y is inverted (down = positive y). A wrist at the same height as
    the shoulder gives 0 degrees; straight up or straight down gives 90.
    """
    dx = float(wrist[0] - shoulder[0])
    dy = float(wrist[1] - shoulder[1])
    # -dy because image y increases downward; we want "up = positive elevation".
    return abs(math.degrees(math.atan2(-dy, abs(dx) + 1e-9)))


def torso_frame_basis(
    l_shoulder: np.ndarray,
    r_shoulder: np.ndarray,
    l_hip: np.ndarray,
    r_hip: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(mid_shoulder, e_x, e_y)`` for a body-relative 2-D frame.

    ``e_x`` points from the LEFT shoulder to the RIGHT shoulder.
    ``e_y`` is orthogonal to ``e_x`` and oriented so the hip midpoint lies
    in the +e_y half-plane (i.e. e_y points DOWN the body in image space).
    Both basis vectors are unit-length; ``mid_shoulder`` is the origin.
    """
    l_sh = np.asarray(l_shoulder, dtype=np.float64)
    r_sh = np.asarray(r_shoulder, dtype=np.float64)
    l_hp = np.asarray(l_hip, dtype=np.float64)
    r_hp = np.asarray(r_hip, dtype=np.float64)

    mid_shoulder = (l_sh + r_sh) * 0.5
    mid_hip = (l_hp + r_hp) * 0.5

    raw_x = r_sh - l_sh
    nx = float(np.linalg.norm(raw_x))
    if nx < 1e-6:
        # Degenerate: shoulders coincide. Default to image-x.
        e_x = np.array([1.0, 0.0], dtype=np.float64)
    else:
        e_x = raw_x / nx

    # Orthogonal candidate; pick the sign that aligns with mid_shoulder->mid_hip.
    perp = np.array([-e_x[1], e_x[0]], dtype=np.float64)
    down = mid_hip - mid_shoulder
    if float(np.dot(perp, down)) < 0.0:
        perp = -perp
    npn = float(np.linalg.norm(perp)) + 1e-9
    e_y = perp / npn

    return mid_shoulder, e_x.astype(np.float64), e_y.astype(np.float64)


def arm_azimuth_torso_frame(
    shoulder: np.ndarray,
    wrist: np.ndarray,
    e_x: np.ndarray,
    e_y: np.ndarray,
) -> float:
    """Azimuth (degrees) of the shoulder->wrist vector in the torso frame.

    Convention: 0 deg = +e_x (right side of body), measured CCW in the
    (e_x, -e_y) plane. So +90 deg points opposite to e_y -- i.e. UP the
    body / forward-of-camera in the body-relative sense -- and we use
    that as the 'front' direction.
    """
    v = np.asarray(wrist, dtype=np.float64) - np.asarray(shoulder, dtype=np.float64)
    proj_x = float(np.dot(v, np.asarray(e_x, dtype=np.float64)))
    proj_y = float(np.dot(v, np.asarray(e_y, dtype=np.float64)))
    # CCW in (e_x, -e_y) means y axis is flipped.
    return float(math.degrees(math.atan2(-proj_y, proj_x)))

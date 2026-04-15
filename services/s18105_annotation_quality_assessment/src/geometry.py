"""Image decode helpers, IoU computation, bbox conversion, and polygon geometry."""

from __future__ import annotations

import numpy as np
from PIL import Image

from _shared.image_utils import decode_image, strip_data_uri


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of xyxy boxes. Returns N x M matrix.

    Args:
        boxes_a: (N, 4) array of [x1, y1, x2, y2] boxes.
        boxes_b: (M, 4) array of [x1, y1, x2, y2] boxes.

    Returns:
        (N, M) IoU matrix.
    """
    if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float64)

    # Expand dims for broadcasting: (N, 1, 4) vs (1, M, 4)
    a = boxes_a[:, np.newaxis, :]  # (N, 1, 4)
    b = boxes_b[np.newaxis, :, :]  # (1, M, 4)

    xx1 = np.maximum(a[..., 0], b[..., 0])
    yy1 = np.maximum(a[..., 1], b[..., 1])
    xx2 = np.minimum(a[..., 2], b[..., 2])
    yy2 = np.minimum(a[..., 3], b[..., 3])

    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter + 1e-6

    return inter / union


def compute_single_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU between two xyxy boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter + 1e-6
    return inter / union


def norm_cxcywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> list[float]:
    """Convert normalized [cx, cy, w, h] to normalized [x1, y1, x2, y2]."""
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def norm_xyxy_to_pixel(xyxy: list[float], img_w: int, img_h: int) -> list[int]:
    """Convert normalized xyxy to pixel xyxy."""
    return [
        int(round(xyxy[0] * img_w)),
        int(round(xyxy[1] * img_h)),
        int(round(xyxy[2] * img_w)),
        int(round(xyxy[3] * img_h)),
    ]


def pixel_xyxy_to_norm_cxcywh(xyxy: list[int], img_w: int, img_h: int) -> list[float]:
    """Convert pixel [x1, y1, x2, y2] to normalized [cx, cy, w, h]."""
    x1, y1, x2, y2 = xyxy
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return [round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)]


def shoelace_area(vertices: list[tuple[float, float]]) -> float:
    """Compute polygon area using the Shoelace formula.

    Args:
        vertices: List of (x, y) coordinate pairs.

    Returns:
        Absolute area of the polygon.
    """
    n = len(vertices)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    return abs(area) / 2.0


def segments_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> bool:
    """Check if line segment (p1, p2) intersects with (p3, p4) using cross products.

    Does not count shared endpoints as intersections.
    """
    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    return False


def polygon_self_intersects(vertices: list[tuple[float, float]]) -> bool:
    """Check if a polygon has any self-intersecting edges.

    Uses a simplified O(n^2) approach checking all non-adjacent edge pairs.
    """
    n = len(vertices)
    if n < 4:
        return False

    edges = []
    for i in range(n):
        j = (i + 1) % n
        edges.append((vertices[i], vertices[j]))

    for i in range(n):
        for j in range(i + 2, n):
            # Skip adjacent edges (they share a vertex)
            if i == 0 and j == n - 1:
                continue
            if segments_intersect(edges[i][0], edges[i][1], edges[j][0], edges[j][1]):
                return True

    return False

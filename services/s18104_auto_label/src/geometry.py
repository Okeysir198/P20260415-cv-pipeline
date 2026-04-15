"""Image decoding, polygon extraction, bbox helpers, and NMS."""

from __future__ import annotations

import base64
import io

import cv2
import numpy as np
from PIL import Image

from src.config import (
    CROSS_CLASS_NMS_ENABLED,
    CROSS_CLASS_NMS_THRESHOLD,
    MIN_POLYGON_VERTICES,
    SIMPLIFY_TOLERANCE,
)
from src.schemas import Detection


# ---------------------------------------------------------------------------
# Image decoding
# ---------------------------------------------------------------------------


def strip_data_uri(b64: str) -> str:
    """Remove data URI prefix if present."""
    if "," in b64 and b64.split(",")[0].startswith("data:"):
        return b64.split(",", 1)[1]
    return b64


def decode_image(b64: str) -> Image.Image:
    """Decode base64 string to PIL Image (RGB)."""
    return Image.open(io.BytesIO(base64.b64decode(strip_data_uri(b64)))).convert("RGB")


# ---------------------------------------------------------------------------
# Mask → Polygon
# ---------------------------------------------------------------------------


def mask_to_polygon(
    mask_b64: str,
    img_h: int,
    img_w: int,
    simplify_tolerance: float = SIMPLIFY_TOLERANCE,
    min_vertices: int = MIN_POLYGON_VERTICES,
) -> list[list[float]]:
    """Decode base64 PNG mask → boolean array → cv2.findContours → normalized polygon.

    Returns list of [x_norm, y_norm] pairs from the largest contour.
    """
    raw = base64.b64decode(mask_b64)
    mask_arr = np.array(Image.open(io.BytesIO(raw)).convert("L"))
    binary = (mask_arr > 127).astype(np.uint8) * 255

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # Use largest contour by area
    largest = max(contours, key=cv2.contourArea)

    # Simplify polygon
    epsilon = simplify_tolerance
    approx = cv2.approxPolyDP(largest, epsilon, True)

    # Fall back to original if simplified has too few vertices
    if len(approx) < min_vertices and len(largest) >= min_vertices:
        approx = largest

    if len(approx) < 3:
        return []

    # Normalize coordinates to [0, 1]
    polygon = []
    for point in approx:
        x = float(point[0][0]) / img_w
        y = float(point[0][1]) / img_h
        polygon.append([round(x, 6), round(y, 6)])

    return polygon


# ---------------------------------------------------------------------------
# Bbox helpers
# ---------------------------------------------------------------------------


def compute_bbox_norm(bbox_xyxy: list[int], img_w: int, img_h: int) -> list[float]:
    """Convert pixel [x1, y1, x2, y2] to normalized [cx, cy, w, h]."""
    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return [round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)]


def bbox_from_sam3(bbox_dict: dict) -> list[int]:
    """Extract [x1, y1, x2, y2] pixel coords from SAM3 bbox dict."""
    return [
        int(bbox_dict.get("x1", 0)),
        int(bbox_dict.get("y1", 0)),
        int(bbox_dict.get("x2", 0)),
        int(bbox_dict.get("y2", 0)),
    ]


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------


def nms_numpy(
    boxes_xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float,
) -> list[int]:
    """Pure numpy greedy NMS. Returns list of kept indices."""
    if len(boxes_xyxy) == 0:
        return []
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while len(order) > 0:
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes_xyxy[i, 0], boxes_xyxy[rest, 0])
        yy1 = np.maximum(boxes_xyxy[i, 1], boxes_xyxy[rest, 1])
        xx2 = np.minimum(boxes_xyxy[i, 2], boxes_xyxy[rest, 2])
        yy2 = np.minimum(boxes_xyxy[i, 3], boxes_xyxy[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes_xyxy[i, 2] - boxes_xyxy[i, 0]) * (boxes_xyxy[i, 3] - boxes_xyxy[i, 1])
        area_rest = (boxes_xyxy[rest, 2] - boxes_xyxy[rest, 0]) * (boxes_xyxy[rest, 3] - boxes_xyxy[rest, 1])
        iou = inter / (area_i + area_rest - inter + 1e-6)
        order = rest[iou <= iou_threshold]
    return keep


def apply_nms(
    detections: list[Detection],
    iou_threshold: float,
    cross_class_enabled: bool = CROSS_CLASS_NMS_ENABLED,
    cross_class_threshold: float = CROSS_CLASS_NMS_THRESHOLD,
) -> list[Detection]:
    """Apply per-class NMS, then optional cross-class NMS."""
    if not detections:
        return []

    # Per-class NMS
    by_class: dict[int, list[Detection]] = {}
    for det in detections:
        by_class.setdefault(det.class_id, []).append(det)

    kept: list[Detection] = []
    for _, cls_dets in by_class.items():
        boxes = np.array([d.bbox_xyxy for d in cls_dets], dtype=np.float64)
        scores = np.array([d.score for d in cls_dets], dtype=np.float64)
        indices = nms_numpy(boxes, scores, iou_threshold)
        kept.extend(cls_dets[i] for i in indices)

    # Optional cross-class NMS
    if cross_class_enabled and len(kept) > 1:
        boxes = np.array([d.bbox_xyxy for d in kept], dtype=np.float64)
        scores = np.array([d.score for d in kept], dtype=np.float64)
        indices = nms_numpy(boxes, scores, cross_class_threshold)
        kept = [kept[i] for i in indices]

    # Sort by score descending
    kept.sort(key=lambda d: d.score, reverse=True)
    return kept

"""Label parsing functions — YOLO, YOLO-seg, and COCO format parsers."""

from __future__ import annotations

import logging

from fastapi import HTTPException

from src.schemas import ParsedAnnotation

logger = logging.getLogger(__name__)


def parse_yolo(labels: list[str]) -> list[ParsedAnnotation]:
    """Parse YOLO format labels: 'class_id cx cy w h'."""
    annotations: list[ParsedAnnotation] = []
    for line in labels:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        annotations.append(ParsedAnnotation(
            class_id=class_id,
            bbox_norm=[cx, cy, w, h],
            polygon_norm=[],
            source_format="yolo",
        ))
    return annotations


def parse_yolo_seg(labels: list[str]) -> list[ParsedAnnotation]:
    """Parse YOLO-seg format labels: 'class_id x1 y1 x2 y2 ... xN yN'.

    Derives bbox from polygon min/max.
    """
    annotations: list[ParsedAnnotation] = []
    for line in labels:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7:  # class_id + at least 3 points (6 coords)
            continue
        class_id = int(parts[0])
        coords = [float(p) for p in parts[1:]]
        if len(coords) % 2 != 0:
            logger.warning("YOLO-seg label has odd coordinate count (%d), dropping last value", len(coords))
            coords = coords[:-1]
        if len(coords) < 6:
            continue

        # Extract polygon vertices
        polygon_norm = coords  # flat [x1,y1,x2,y2,...]

        # Derive bbox from polygon min/max
        xs = coords[0::2]
        ys = coords[1::2]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        w = x_max - x_min
        h = y_max - y_min
        cx = x_min + w / 2
        cy = y_min + h / 2

        annotations.append(ParsedAnnotation(
            class_id=class_id,
            bbox_norm=[cx, cy, w, h],
            polygon_norm=polygon_norm,
            source_format="yolo_seg",
        ))
    return annotations


def parse_coco(labels: list[dict], img_w: int, img_h: int) -> list[ParsedAnnotation]:
    """Parse COCO format labels.

    Each dict has: category_id, bbox [x, y, w, h] in pixels, segmentation [[x1,y1,x2,y2,...]] in pixels.
    """
    annotations: list[ParsedAnnotation] = []
    for ann in labels:
        class_id = int(ann.get("category_id", 0))

        # Parse bbox [x, y, w, h] in pixels -> normalized [cx, cy, w, h]
        bbox = ann.get("bbox", [0, 0, 0, 0])
        if len(bbox) < 4:
            continue
        px_x, px_y, px_w, px_h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        cx = (px_x + px_w / 2) / img_w if img_w > 0 else 0.0
        cy = (px_y + px_h / 2) / img_h if img_h > 0 else 0.0
        w = px_w / img_w if img_w > 0 else 0.0
        h = px_h / img_h if img_h > 0 else 0.0

        # Parse segmentation (optional)
        polygon_norm: list[float] = []
        seg = ann.get("segmentation", [])
        if seg and isinstance(seg, list) and len(seg) > 0:
            # Take first polygon (COCO allows multiple)
            first_poly = seg[0] if isinstance(seg[0], list) else seg
            if isinstance(first_poly, list) and len(first_poly) >= 6:
                # Convert pixel coords to normalized
                polygon_norm = []
                for i in range(0, len(first_poly), 2):
                    if i + 1 < len(first_poly):
                        norm_x = float(first_poly[i]) / img_w if img_w > 0 else 0.0
                        norm_y = float(first_poly[i + 1]) / img_h if img_h > 0 else 0.0
                        polygon_norm.extend([norm_x, norm_y])

        annotations.append(ParsedAnnotation(
            class_id=class_id,
            bbox_norm=[cx, cy, w, h],
            polygon_norm=polygon_norm,
            source_format="coco",
        ))
    return annotations


def parse_labels(
    labels: list[str] | list[dict],
    label_format: str,
    img_w: int,
    img_h: int,
) -> list[ParsedAnnotation]:
    """Parse labels into normalized ParsedAnnotation objects.

    Args:
        labels: Raw labels in the specified format.
        label_format: 'yolo', 'yolo_seg', or 'coco'.
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        List of parsed annotations.
    """
    fmt = label_format.lower().strip()
    if fmt == "yolo":
        return parse_yolo(labels)  # type: ignore[arg-type]
    elif fmt == "yolo_seg":
        return parse_yolo_seg(labels)  # type: ignore[arg-type]
    elif fmt == "coco":
        return parse_coco(labels, img_w, img_h)  # type: ignore[arg-type]
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown label_format '{label_format}'. Supported: yolo, yolo_seg, coco",
        )

"""Structural validation of parsed annotations."""

from __future__ import annotations

import numpy as np

from src.config import (
    DEFAULT_MIN_BOX_SIZE,
    DEFAULT_MAX_BOX_SIZE,
    DEFAULT_DUPLICATE_IOU_THRESH,
    DEFAULT_MAX_ASPECT_RATIO,
    DEFAULT_POLY_BBOX_AREA_MIN,
    DEFAULT_POLY_BBOX_AREA_MAX,
)
from src.geometry import compute_iou_matrix, norm_cxcywh_to_xyxy, shoelace_area, polygon_self_intersects
from src.schemas import ParsedAnnotation, ValidationIssue, SuggestedFix


def validate_annotations(
    annotations: list[ParsedAnnotation],
    classes: dict[int, str],
    cfg: dict,
) -> tuple[list[ValidationIssue], list[SuggestedFix]]:
    """Structural validation of parsed annotations.

    Checks:
        1. Out-of-bounds coordinates
        2. Invalid class IDs
        3. Degenerate boxes (too small)
        4. Large boxes
        5. Duplicate annotations (high-IoU same-class pairs)
        6. Extreme aspect ratios
        7. Polygon vertex count < 3
        8. Polygon self-intersection
        9. Polygon-bbox consistency
        10. Polygon out-of-bounds vertices

    Args:
        annotations: Parsed annotations list.
        classes: Valid class mapping.
        cfg: Optional config overrides for thresholds.

    Returns:
        Tuple of (issues, suggested_fixes).
    """
    min_box_size: float = cfg.get("min_box_size", DEFAULT_MIN_BOX_SIZE)
    max_box_size: float = cfg.get("max_box_size", DEFAULT_MAX_BOX_SIZE)
    duplicate_iou_thresh: float = cfg.get("duplicate_iou_threshold", DEFAULT_DUPLICATE_IOU_THRESH)
    max_aspect_ratio: float = cfg.get("max_aspect_ratio", DEFAULT_MAX_ASPECT_RATIO)
    poly_bbox_area_min: float = cfg.get("polygon_bbox_area_ratio_min", DEFAULT_POLY_BBOX_AREA_MIN)
    poly_bbox_area_max: float = cfg.get("polygon_bbox_area_ratio_max", DEFAULT_POLY_BBOX_AREA_MAX)

    min_aspect = 1.0 / max_aspect_ratio
    max_aspect = max_aspect_ratio

    # Normalize classes keys to int
    valid_classes = {int(k): v for k, v in classes.items()}

    issues: list[ValidationIssue] = []
    fixes: list[SuggestedFix] = []

    for idx, ann in enumerate(annotations):
        cx, cy, w, h = ann.bbox_norm

        # Check 1: out-of-bounds bbox coordinates
        oob = False
        for name, val in [("cx", cx), ("cy", cy), ("w", w), ("h", h)]:
            if val < 0.0 or val > 1.0:
                issues.append(ValidationIssue(
                    type="out_of_bounds",
                    severity="high",
                    annotation_idx=idx,
                    detail=f"{name}={val:.4f} outside [0, 1]",
                ))
                oob = True

        if oob:
            # Generate clip fix
            x1 = max(0.0, cx - w / 2)
            y1 = max(0.0, cy - h / 2)
            x2 = min(1.0, cx + w / 2)
            y2 = min(1.0, cy + h / 2)
            new_w = x2 - x1
            new_h = y2 - y1
            new_cx = x1 + new_w / 2
            new_cy = y1 + new_h / 2
            fixes.append(SuggestedFix(
                type="clip_bbox",
                annotation_idx=idx,
                original={"class_id": ann.class_id, "bbox_norm": ann.bbox_norm},
                suggested={"class_id": ann.class_id, "bbox_norm": [
                    round(new_cx, 6), round(new_cy, 6), round(new_w, 6), round(new_h, 6),
                ]},
                reason="Coordinates out of [0, 1] range — clipped.",
            ))

        # Check 2: invalid class ID
        if ann.class_id not in valid_classes:
            issues.append(ValidationIssue(
                type="invalid_class",
                severity="high",
                annotation_idx=idx,
                detail=f"Class ID {ann.class_id} not in valid classes {list(valid_classes.keys())}",
            ))

        # Check 3: degenerate boxes
        if w < min_box_size or h < min_box_size:
            issues.append(ValidationIssue(
                type="degenerate_box",
                severity="medium",
                annotation_idx=idx,
                detail=f"Box too small: w={w:.4f}, h={h:.4f} (min={min_box_size})",
            ))
            fixes.append(SuggestedFix(
                type="remove_degenerate",
                annotation_idx=idx,
                original={"class_id": ann.class_id, "bbox_norm": ann.bbox_norm},
                suggested=None,
                reason="Bounding box too small (near-zero area).",
            ))

        # Check 4: large boxes
        if w > max_box_size or h > max_box_size:
            issues.append(ValidationIssue(
                type="large_box",
                severity="low",
                annotation_idx=idx,
                detail=f"Box too large: w={w:.4f}, h={h:.4f} (max={max_box_size})",
            ))

        # Check 6: extreme aspect ratio
        if w > 0 and h > 0:
            aspect = w / h
            if aspect < min_aspect or aspect > max_aspect:
                issues.append(ValidationIssue(
                    type="extreme_aspect_ratio",
                    severity="low",
                    annotation_idx=idx,
                    detail=f"Aspect ratio {aspect:.2f} outside [{min_aspect:.2f}, {max_aspect:.2f}]",
                ))

        # Polygon-specific checks (only when polygon data present)
        if ann.polygon_norm:
            poly_coords = ann.polygon_norm
            num_vertices = len(poly_coords) // 2

            # Check 7: polygon vertex count < 3
            if num_vertices < 3:
                issues.append(ValidationIssue(
                    type="polygon_too_few_vertices",
                    severity="high",
                    annotation_idx=idx,
                    detail=f"Polygon has {num_vertices} vertices (minimum 3)",
                ))
            else:
                # Build vertex list for further checks
                vertices: list[tuple[float, float]] = []
                for i in range(0, len(poly_coords) - 1, 2):
                    vertices.append((poly_coords[i], poly_coords[i + 1]))

                # Check 8: polygon self-intersection
                if polygon_self_intersects(vertices):
                    issues.append(ValidationIssue(
                        type="polygon_self_intersection",
                        severity="medium",
                        annotation_idx=idx,
                        detail="Polygon has self-intersecting edges",
                    ))

                # Check 9: polygon-bbox area consistency
                poly_area = shoelace_area(vertices)
                bbox_area = w * h
                if bbox_area > 0:
                    ratio = poly_area / bbox_area
                    if ratio < poly_bbox_area_min or ratio > poly_bbox_area_max:
                        issues.append(ValidationIssue(
                            type="polygon_bbox_inconsistency",
                            severity="medium",
                            annotation_idx=idx,
                            detail=f"Polygon/bbox area ratio {ratio:.2f} outside [{poly_bbox_area_min}, {poly_bbox_area_max}]",
                        ))

                # Check 10: polygon out-of-bounds vertices
                for vi, (vx, vy) in enumerate(vertices):
                    if vx < 0.0 or vx > 1.0 or vy < 0.0 or vy > 1.0:
                        issues.append(ValidationIssue(
                            type="polygon_out_of_bounds",
                            severity="high",
                            annotation_idx=idx,
                            detail=f"Polygon vertex {vi}: ({vx:.4f}, {vy:.4f}) outside [0, 1]",
                        ))
                        break  # Report once per annotation

    # Check 5: duplicate annotations (high-IoU same-class pairs)
    if len(annotations) >= 2:
        # Convert to xyxy for IoU computation
        boxes_xyxy = np.array(
            [norm_cxcywh_to_xyxy(*a.bbox_norm) for a in annotations],
            dtype=np.float64,
        )
        iou_matrix = compute_iou_matrix(boxes_xyxy, boxes_xyxy)

        for i in range(len(annotations)):
            for j in range(i + 1, len(annotations)):
                if annotations[i].class_id != annotations[j].class_id:
                    continue
                if iou_matrix[i, j] >= duplicate_iou_thresh:
                    issues.append(ValidationIssue(
                        type="duplicate",
                        severity="medium",
                        annotation_idx=j,
                        detail=f"Annotation {j} duplicates {i} (class={annotations[i].class_id}, IoU={iou_matrix[i, j]:.3f})",
                    ))
                    fixes.append(SuggestedFix(
                        type="remove_duplicate",
                        annotation_idx=j,
                        original={"class_id": annotations[j].class_id, "bbox_norm": annotations[j].bbox_norm},
                        suggested=None,
                        reason="Duplicate annotation detected.",
                    ))

    return issues, fixes

"""Rule-based classification: derive final classes from intermediate detections."""

from __future__ import annotations

import numpy as np

from src.schemas import Detection


def _compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute N x M IoU matrix from xyxy boxes."""
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-6)


def apply_rule_classification(
    detections: list[Detection],
    class_rules: list[dict],
    detection_class_map: dict[str, int],
    final_classes: dict[int, str],
) -> list[Detection]:
    """Apply rule-based classification to remap intermediate detections to final classes.

    Args:
        detections: Detections with intermediate class IDs (from detection_class_map).
        class_rules: List of rule dicts, each with keys:
            - output_class_id: int — final class ID to assign
            - source: str — intermediate class name to match
            - condition: "direct" | "overlap" | "no_overlap"
            - target: str — (for overlap/no_overlap) intermediate class name to check IoU against
            - min_iou: float — (for overlap/no_overlap) IoU threshold (default 0.3)
        detection_class_map: Maps intermediate class name -> temp class ID.
        final_classes: Maps final class ID -> final class name.

    Returns:
        New list of Detection objects with class_id/class_name remapped to final classes.
    """
    if not detections or not class_rules:
        return detections

    # Build reverse map: temp_id -> intermediate class name
    id_to_name = {tid: name for name, tid in detection_class_map.items()}

    # Group detections by intermediate class name
    by_class: dict[str, list[Detection]] = {}
    for det in detections:
        name = id_to_name.get(det.class_id, "")
        by_class.setdefault(name, []).append(det)

    result: list[Detection] = []

    for rule in class_rules:
        output_class_id = int(rule["output_class_id"])
        source = rule["source"]
        condition = rule.get("condition", "direct")
        output_class_name = final_classes.get(output_class_id, str(output_class_id))

        source_dets = by_class.get(source, [])
        if not source_dets:
            continue

        if condition == "direct":
            # Simple remap: source intermediate -> final class
            for det in source_dets:
                result.append(
                    det.model_copy(
                        update={"class_id": output_class_id, "class_name": output_class_name}
                    )
                )

        elif condition == "overlap":
            # Source detections that overlap with target detections
            target = rule.get("target", "")
            min_iou = float(rule.get("min_iou", 0.3))
            target_dets = by_class.get(target, [])

            if not target_dets:
                continue

            source_boxes = np.array([d.bbox_xyxy for d in source_dets], dtype=np.float64)
            target_boxes = np.array([d.bbox_xyxy for d in target_dets], dtype=np.float64)
            iou_matrix = _compute_iou_matrix(source_boxes, target_boxes)

            for i, det in enumerate(source_dets):
                if iou_matrix[i].max() >= min_iou:
                    result.append(
                        det.model_copy(
                            update={"class_id": output_class_id, "class_name": output_class_name}
                        )
                    )

        elif condition == "no_overlap":
            # Source detections that do NOT overlap with target detections
            target = rule.get("target", "")
            min_iou = float(rule.get("min_iou", 0.3))
            target_dets = by_class.get(target, [])

            if not target_dets:
                # No targets -> all source dets qualify (no overlap by definition)
                for det in source_dets:
                    result.append(
                        det.model_copy(
                            update={"class_id": output_class_id, "class_name": output_class_name}
                        )
                    )
                continue

            source_boxes = np.array([d.bbox_xyxy for d in source_dets], dtype=np.float64)
            target_boxes = np.array([d.bbox_xyxy for d in target_dets], dtype=np.float64)
            iou_matrix = _compute_iou_matrix(source_boxes, target_boxes)

            for i, det in enumerate(source_dets):
                if iou_matrix[i].max() < min_iou:
                    result.append(
                        det.model_copy(
                            update={"class_id": output_class_id, "class_name": output_class_name}
                        )
                    )

    return result

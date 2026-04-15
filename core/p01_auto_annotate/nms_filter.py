"""NMS filter for auto-annotation pipeline.

Applies per-class and optional cross-class non-maximum suppression
to remove overlapping detections.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from utils.metrics import nms_numpy, xywh_to_xyxy

logger = logging.getLogger(__name__)


class NMSFilter:
    """Non-maximum suppression filter for auto-generated annotations.

    Applies per-class NMS (required) and optional cross-class NMS
    to remove overlapping detections. Polygons are preserved through
    filtering by index.

    Args:
        per_class_iou_threshold: IoU threshold for per-class NMS.
        cross_class_enabled: Whether to apply cross-class NMS.
        cross_class_iou_threshold: IoU threshold for cross-class NMS.
    """

    def __init__(
        self,
        per_class_iou_threshold: float = 0.5,
        cross_class_enabled: bool = False,
        cross_class_iou_threshold: float = 0.8,
    ) -> None:
        self.per_class_iou_threshold = per_class_iou_threshold
        self.cross_class_enabled = cross_class_enabled
        self.cross_class_iou_threshold = cross_class_iou_threshold

    def filter(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply NMS filtering to a list of detections.

        Args:
            detections: List of detection dicts with keys:
                class_id, cx, cy, w, h, score, and optionally polygon.

        Returns:
            Filtered list of detections after NMS.
        """
        if len(detections) <= 1:
            return detections

        # Per-class NMS
        kept = self._per_class_nms(detections)

        # Cross-class NMS (optional)
        if self.cross_class_enabled and len(kept) > 1:
            kept = self._cross_class_nms(kept)

        return kept

    def _per_class_nms(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply NMS independently for each class.

        Args:
            detections: List of detection dicts.

        Returns:
            Filtered detections after per-class NMS.
        """
        # Group by class
        by_class: Dict[int, List[int]] = {}
        for idx, det in enumerate(detections):
            cls_id = det["class_id"]
            by_class.setdefault(cls_id, []).append(idx)

        kept_indices: List[int] = []

        for cls_id, indices in by_class.items():
            if len(indices) <= 1:
                kept_indices.extend(indices)
                continue

            cls_dets = [detections[i] for i in indices]
            boxes_xywh = np.array(
                [[d["cx"], d["cy"], d["w"], d["h"]] for d in cls_dets],
                dtype=np.float64,
            )
            scores = np.array(
                [d.get("score", 0.0) for d in cls_dets],
                dtype=np.float64,
            )
            boxes_xyxy = xywh_to_xyxy(boxes_xywh)

            keep = nms_numpy(boxes_xyxy, scores, self.per_class_iou_threshold)

            for k in keep:
                kept_indices.append(indices[int(k)])

        # Sort to maintain original order
        kept_indices.sort()
        return [detections[i] for i in kept_indices]

    def _cross_class_nms(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply cross-class NMS (class-agnostic).

        Uses a higher IoU threshold to only suppress near-duplicates
        across different classes.

        Args:
            detections: List of detection dicts.

        Returns:
            Filtered detections after cross-class NMS.
        """
        boxes_xywh = np.array(
            [[d["cx"], d["cy"], d["w"], d["h"]] for d in detections],
            dtype=np.float64,
        )
        scores = np.array(
            [d.get("score", 0.0) for d in detections],
            dtype=np.float64,
        )
        boxes_xyxy = xywh_to_xyxy(boxes_xywh)

        keep = nms_numpy(boxes_xyxy, scores, self.cross_class_iou_threshold)

        return [detections[int(k)] for k in keep]

    @staticmethod
    def merge_with_existing(
        new_detections: List[Dict[str, Any]],
        existing_annotations: List[tuple],
        iou_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Merge new detections with existing YOLO annotations.

        Removes new detections that overlap significantly with existing
        annotations to avoid duplicates when re-annotating.

        Args:
            new_detections: New detections from SAM3.
            existing_annotations: Existing YOLO annotations as
                ``(class_id, cx, cy, w, h)`` tuples.
            iou_threshold: IoU threshold for considering overlap.

        Returns:
            New detections that don't overlap with existing annotations.
        """
        if not existing_annotations or not new_detections:
            return new_detections

        existing_xywh = np.array(
            [[cx, cy, w, h] for (_, cx, cy, w, h) in existing_annotations],
            dtype=np.float64,
        )
        existing_xyxy = xywh_to_xyxy(existing_xywh)

        from utils.metrics import compute_iou

        kept: List[Dict[str, Any]] = []
        for det in new_detections:
            det_xywh = np.array(
                [[det["cx"], det["cy"], det["w"], det["h"]]],
                dtype=np.float64,
            )
            det_xyxy = xywh_to_xyxy(det_xywh)
            ious = compute_iou(det_xyxy, existing_xyxy)
            if float(ious.max()) < iou_threshold:
                kept.append(det)

        return kept

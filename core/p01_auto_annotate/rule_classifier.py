"""Rule-based overlap classifier for auto-annotation.

Takes intermediate SAM3 detections (e.g. ``head``, ``helmet``) and applies
IoU overlap rules to derive final output classes (e.g. ``head_with_helmet``,
``head_without_helmet``).

Example rule set for helmet detection::

    class_rules = [
        {"output_class_id": 0, "source": "person",  "condition": "direct"},
        {"output_class_id": 1, "source": "head",    "condition": "overlap",    "target": "helmet", "min_iou": 0.3},
        {"output_class_id": 2, "source": "head",    "condition": "no_overlap", "target": "helmet", "min_iou": 0.3},
    ]
    detection_class_map = {"person": 100, "head": 101, "helmet": 102}
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from utils.metrics import compute_iou, xywh_to_xyxy


class RuleClassifier:
    """Apply ordered overlap rules to reclassify intermediate detections.

    Args:
        class_rules: Ordered list of rule dicts.  Each dict contains:
            - ``output_class_id`` (int): Final class ID to assign.
            - ``source`` (str): Intermediate class name of the source detection.
            - ``condition`` (str): One of ``"direct"``, ``"overlap"``, ``"no_overlap"``.
            - ``target`` (str, optional): Intermediate class name to compare against
              (required for ``overlap`` / ``no_overlap``).
            - ``min_iou`` (float, optional): IoU threshold, default ``0.3``.
        detection_class_map: Mapping from intermediate class name to temp class ID
            (e.g. ``{"person": 100, "head": 101, "helmet": 102}``).
    """

    def __init__(
        self,
        class_rules: list[dict],
        detection_class_map: dict[str, int],
    ) -> None:
        self.class_rules = class_rules
        self.detection_class_map = detection_class_map
        # Reverse map: temp class ID -> name
        self._id_to_name: dict[int, str] = {v: k for k, v in detection_class_map.items()}

    def classify(self, detections: list[dict]) -> list[dict]:
        """Classify detections according to the configured rules.

        Args:
            detections: List of detection dicts with keys ``class_id``, ``cx``,
                ``cy``, ``w``, ``h``, ``score``, and optionally ``polygon``.

        Returns:
            New list of detection dicts with ``class_id`` set to the matched
            ``output_class_id``.  Source detections are consumed at most once.
            Target detections are never emitted.
        """
        # Index detections by their intermediate class name.
        by_name: dict[str, list[tuple[int, dict]]] = {}
        for idx, det in enumerate(detections):
            name = self._id_to_name.get(det["class_id"])
            if name is not None:
                by_name.setdefault(name, []).append((idx, det))

        consumed: set[int] = set()
        results: list[dict] = []

        # Collect all names used only as targets so they are never emitted.
        target_names: set[str] = set()
        for rule in self.class_rules:
            if rule["condition"] in ("overlap", "no_overlap"):
                target_names.add(rule["target"])

        for rule in self.class_rules:
            source_name: str = rule["source"]
            condition: str = rule["condition"]
            output_id: int = rule["output_class_id"]

            sources = by_name.get(source_name, [])

            if condition == "direct":
                for idx, det in sources:
                    if idx not in consumed:
                        consumed.add(idx)
                        results.append(self._remap(det, output_id))

            elif condition in ("overlap", "no_overlap"):
                target_name: str = rule["target"]
                min_iou: float = rule.get("min_iou", 0.3)
                targets = by_name.get(target_name, [])

                # If no target detections exist, every source matches "no_overlap".
                if not targets:
                    if condition == "no_overlap":
                        for idx, det in sources:
                            if idx not in consumed:
                                consumed.add(idx)
                                results.append(self._remap(det, output_id))
                    continue

                # Build IoU matrix between source and target boxes.
                src_boxes = self._to_xyxy([d for _, d in sources])
                tgt_boxes = self._to_xyxy([d for _, d in targets])
                iou_matrix = compute_iou(src_boxes, tgt_boxes)  # (S, T)

                for si, (idx, det) in enumerate(sources):
                    if idx in consumed:
                        continue
                    max_iou = float(iou_matrix[si].max()) if iou_matrix.size else 0.0
                    match = (condition == "overlap" and max_iou >= min_iou) or (
                        condition == "no_overlap" and max_iou < min_iou
                    )
                    if match:
                        consumed.add(idx)
                        results.append(self._remap(det, output_id))

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_xyxy(dets: list[dict]) -> np.ndarray:
        """Convert detection dicts to (N, 4) xyxy array."""
        if not dets:
            return np.zeros((0, 4), dtype=np.float64)
        boxes = np.array([[d["cx"], d["cy"], d["w"], d["h"]] for d in dets], dtype=np.float64)
        return xywh_to_xyxy(boxes)

    @staticmethod
    def _remap(det: dict, output_class_id: int) -> dict:
        """Return a copy of *det* with ``class_id`` replaced."""
        out = dict(det)
        out["class_id"] = output_class_id
        return out

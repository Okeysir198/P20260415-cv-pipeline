#!/usr/bin/env python3
"""Per-image annotation quality scoring.

Computes a 0.0-1.0 quality score for each image based on structural
validation issues, SAM3 verification results, and annotation
consistency.  Grades images as "good", "review", or "bad" and
generates auto-fixable suggestions.

Usage (as library):
    from scorer import QualityScorer
    scorer = QualityScorer(qa_config)
    result = scorer.score_image(image_result)
"""

import sys
from pathlib import Path
from typing import Any

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root


class QualityScorer:
    """Per-image annotation quality scoring.

    Combines structural validation, SAM3 bbox IoU, classification
    accuracy, and coverage into a single 0-1 score, then assigns a
    letter grade and generates actionable fix suggestions.
    """

    def __init__(self, config: dict) -> None:
        """Initialize the scorer.

        Args:
            config: QA config dictionary. Must contain a ``scoring``
                section with ``weights`` (dict of component weights) and
                ``thresholds`` (dict with ``good`` and ``review`` cutoffs).

        Expected config structure::

            scoring:
              weights:
                structural: 0.3
                bbox_quality: 0.3
                classification: 0.2
                coverage: 0.2
              thresholds:
                good: 0.85
                review: 0.60
        """
        scoring = config["scoring"]
        self.weights: dict[str, float] = dict(scoring["weights"])
        self.thresholds: dict[str, float] = dict(scoring["thresholds"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_image(self, image_result: dict) -> dict:
        """Compute quality score for a single image.

        Args:
            image_result: Dict with keys:
                - ``annotations``: list of ``(class_id, cx, cy, w, h)``
                - ``validation_issues``: list of issue dicts
                - ``sam3_verification``: dict with ``box_ious``,
                  ``misclassified``, ``missing`` (may be empty if SAM3
                  was skipped)

        Returns:
            The same ``image_result`` dict, updated with:
                - ``quality_score``: float in [0.0, 1.0]
                - ``grade``: ``"good"`` | ``"review"`` | ``"bad"``
                - ``suggested_fixes``: list of fix dicts
        """
        annotations = image_result.get("annotations", [])
        issues = image_result.get("validation_issues", [])
        sam3 = image_result.get("sam3_verification", {})

        num_ann = max(len(annotations), 1)

        # --- Component penalties ---
        structural_penalty = len(issues) / num_ann
        bbox_penalty = self._bbox_quality_penalty(sam3)
        classification_penalty = self._classification_penalty(sam3, num_ann)
        coverage_penalty = self._coverage_penalty(sam3, len(annotations))

        # --- Weighted score ---
        score = 1.0
        score -= self.weights.get("structural", 0.0) * structural_penalty
        score -= self.weights.get("bbox_quality", 0.0) * bbox_penalty
        score -= self.weights.get("classification", 0.0) * classification_penalty
        score -= self.weights.get("coverage", 0.0) * coverage_penalty
        score = max(0.0, min(1.0, score))

        # --- Grade ---
        grade = self._assign_grade(score)

        # --- Fixes ---
        fixes = self.generate_fixes(image_result)

        image_result["quality_score"] = round(score, 4)
        image_result["grade"] = grade
        image_result["suggested_fixes"] = fixes
        return image_result

    def generate_fixes(self, image_result: dict) -> list[dict[str, Any]]:
        """Generate auto-fixable suggestions for an image.

        Fix types:
            - ``clip_bbox``: Out-of-bounds coords clipped to [0, 1].
            - ``remove_duplicate``: Duplicate annotation removed.
            - ``remove_degenerate``: Tiny bbox removed.
            - ``tighten_bbox``: SAM3 suggests tighter bbox.
            - ``remove_annotation``: Misclassified annotation removed.

        Args:
            image_result: Dict with annotation and issue data (same
                structure as accepted by :meth:`score_image`).

        Returns:
            List of fix dicts, each containing:
                - ``type`` (str)
                - ``annotation_idx`` (int)
                - ``original`` (tuple or None)
                - ``suggested`` (tuple or None)
                - ``reason`` (str)
        """
        fixes: list[dict[str, Any]] = []
        annotations = image_result.get("annotations", [])
        issues = image_result.get("validation_issues", [])
        sam3 = image_result.get("sam3_verification", {})

        # --- Structural fixes from validation issues ---
        for issue in issues:
            issue_type = issue.get("type", "")
            idx = issue.get("annotation_idx", -1)
            original = self._get_annotation(annotations, idx)

            if issue_type == "out_of_bounds":
                suggested = self._clip_annotation(original) if original else None
                fixes.append({
                    "type": "clip_bbox",
                    "annotation_idx": idx,
                    "original": original,
                    "suggested": suggested,
                    "reason": "Coordinates out of [0, 1] range — clipped.",
                })
            elif issue_type == "duplicate":
                fixes.append({
                    "type": "remove_duplicate",
                    "annotation_idx": idx,
                    "original": original,
                    "suggested": None,
                    "reason": "Duplicate annotation detected.",
                })
            elif issue_type == "degenerate_box":
                fixes.append({
                    "type": "remove_degenerate",
                    "annotation_idx": idx,
                    "original": original,
                    "suggested": None,
                    "reason": "Bounding box too small (near-zero area).",
                })

        # --- SAM3-based fixes ---
        # misclassified is a flat list of annotation indices
        misclassified = sam3.get("misclassified", [])
        for idx in misclassified:
            original = self._get_annotation(annotations, idx)
            fixes.append({
                "type": "remove_annotation",
                "annotation_idx": idx,
                "original": original,
                "suggested": None,
                "reason": "SAM3 verification: likely misclassified.",
            })

        # box_ious is a flat list of floats (one per annotation)
        box_ious = sam3.get("box_ious", [])
        for idx, iou_val in enumerate(box_ious):
            iou = float(iou_val)
            if iou < self.thresholds.get("review", 0.6):
                original = self._get_annotation(annotations, idx)
                fixes.append({
                    "type": "tighten_bbox",
                    "annotation_idx": idx,
                    "original": original,
                    "suggested": None,
                    "reason": f"SAM3 suggests tighter bbox (IoU={iou:.2f}).",
                })

        return fixes

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _bbox_quality_penalty(self, sam3: dict) -> float:
        """Compute bbox quality penalty from SAM3 IoUs.

        Args:
            sam3: SAM3 verification results dict.

        Returns:
            Penalty value in [0.0, 1.0]. Returns 0 if no SAM3 data.
        """
        box_ious = sam3.get("box_ious", [])
        if not box_ious:
            return 0.0
        # box_ious is a flat list of float IoU values
        mean_iou = sum(float(v) for v in box_ious) / len(box_ious)
        return 1.0 - mean_iou

    def _classification_penalty(self, sam3: dict, num_ann: int) -> float:
        """Compute classification penalty from misclassified count.

        Args:
            sam3: SAM3 verification results dict.
            num_ann: Number of annotations (clamped to >= 1).

        Returns:
            Penalty value in [0.0, 1.0].
        """
        # misclassified is a list of annotation indices
        misclassified = sam3.get("misclassified", [])
        return min(len(misclassified) / num_ann, 1.0)

    def _coverage_penalty(self, sam3: dict, raw_num_ann: int) -> float:
        """Compute coverage penalty from missing detections.

        Args:
            sam3: SAM3 verification results dict.
            raw_num_ann: Raw annotation count (before clamping).

        Returns:
            Penalty value in [0.0, 1.0].
        """
        missing = sam3.get("missing_masks", sam3.get("missing", []))
        num_missing = len(missing)
        denominator = max(raw_num_ann + num_missing, 1)
        return num_missing / denominator

    def _assign_grade(self, score: float) -> str:
        """Map a numeric score to a letter grade.

        Args:
            score: Quality score in [0.0, 1.0].

        Returns:
            ``"good"``, ``"review"``, or ``"bad"``.
        """
        if score >= self.thresholds.get("good", 0.85):
            return "good"
        if score >= self.thresholds.get("review", 0.60):
            return "review"
        return "bad"

    @staticmethod
    def _get_annotation(
        annotations: list, idx: int
    ) -> tuple[int, float, float, float, float] | None:
        """Safely retrieve an annotation by index.

        Args:
            annotations: List of annotation tuples.
            idx: Index to retrieve.

        Returns:
            The annotation tuple, or ``None`` if out of range.
        """
        if 0 <= idx < len(annotations):
            return tuple(annotations[idx])  # type: ignore[return-value]
        return None

    @staticmethod
    def _clip_annotation(
        ann: tuple[int, float, float, float, float],
    ) -> tuple[int, float, float, float, float]:
        """Clip annotation coordinates to the [0, 1] range.

        Adjusts center and size so that the resulting box lies entirely
        within the normalized image bounds.

        Args:
            ann: ``(class_id, cx, cy, w, h)`` tuple.

        Returns:
            Clipped ``(class_id, cx, cy, w, h)`` tuple.
        """
        cls_id, cx, cy, w, h = ann

        # Compute corners, clip, then recompute center/size
        x1 = max(0.0, cx - w / 2)
        y1 = max(0.0, cy - h / 2)
        x2 = min(1.0, cx + w / 2)
        y2 = min(1.0, cy + h / 2)

        new_w = x2 - x1
        new_h = y2 - y1
        new_cx = x1 + new_w / 2
        new_cy = y1 + new_h / 2

        return (cls_id, round(new_cx, 6), round(new_cy, 6),
                round(new_w, 6), round(new_h, 6))

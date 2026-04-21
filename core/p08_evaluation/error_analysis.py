"""Error analysis for object detection models.

Classifies detection errors (FP types, FN types), computes per-class
optimal thresholds, identifies hardest images, and produces structured
reports — all powered by supervision.

Usage:
    from core.p08_evaluation.error_analysis import ErrorAnalyzer

    analyzer = ErrorAnalyzer(class_names={0: "fire", 1: "smoke"})
    report = analyzer.analyze(predictions, ground_truths)
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.p08_evaluation.sv_metrics import _gt_to_sv, _preds_to_sv

# ---------------------------------------------------------------------------
# COCO size thresholds (pixel area)
# ---------------------------------------------------------------------------

_SMALL_AREA = 32 ** 2    # < 1024 px
_LARGE_AREA = 96 ** 2    # >= 9216 px


def _size_category(area: float) -> str:
    """Classify box area into COCO size bucket."""
    if area < _SMALL_AREA:
        return "small"
    if area < _LARGE_AREA:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ErrorCase:
    """Single detection error with full context."""

    image_idx: int
    error_type: str           # background_fp | class_confusion | localization | duplicate | missed
    box: np.ndarray           # xyxy (4,)
    class_id: int             # predicted class (FP) or GT class (FN)
    score: float | None    # confidence (None for missed)
    size_category: str        # small | medium | large
    iou: float = 0.0         # best IoU with a GT (0 for missed with no pred)
    gt_class_id: int | None = None  # for class_confusion: the true class


@dataclass
class ErrorReport:
    """Full error analysis output."""

    errors: list[ErrorCase] = field(default_factory=list)
    per_image_error_count: dict[int, int] = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    optimal_thresholds: dict[int, dict] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ErrorAnalyzer
# ---------------------------------------------------------------------------


class ErrorAnalyzer:
    """Analyze detection errors using supervision APIs.

    Args:
        class_names: Mapping class_id -> display name.
        iou_threshold: IoU threshold for matching (default 0.5).
        localization_iou_low: Lower IoU bound for localization errors (default 0.3).
    """

    def __init__(
        self,
        class_names: dict[int, str],
        iou_threshold: float = 0.5,
        localization_iou_low: float = 0.3,
    ) -> None:
        self.class_names = class_names
        self.iou_threshold = iou_threshold
        self.localization_iou_low = localization_iou_low

    # ----- main entry point -----

    def analyze(
        self,
        predictions: list[dict],
        ground_truths: list[dict],
        image_paths: list[str] | None = None,
    ) -> ErrorReport:
        """Run full error analysis.

        Args:
            predictions: Per-image prediction dicts (boxes, scores, labels).
            ground_truths: Per-image GT dicts (boxes, labels).
            image_paths: Optional list of image file paths (same length).

        Returns:
            ErrorReport with errors, per-image counts, summary, and thresholds.
        """
        errors = self.classify_errors(predictions, ground_truths)
        summary = self.error_summary(errors, predictions, ground_truths)
        thresholds = self.compute_optimal_thresholds(predictions, ground_truths)

        per_image = {}
        for err in errors:
            per_image[err.image_idx] = per_image.get(err.image_idx, 0) + 1

        return ErrorReport(
            errors=errors,
            per_image_error_count=per_image,
            summary=summary,
            optimal_thresholds=thresholds,
        )

    # ----- error classification -----

    def classify_errors(
        self,
        predictions: list[dict],
        ground_truths: list[dict],
    ) -> list[ErrorCase]:
        """Classify every detection into an error type.

        Uses sv.box_iou_batch() for IoU computation and sv.Detections
        filtering for size categorization.

        Returns:
            List of ErrorCase objects (all FP and FN across all images).
        """
        all_errors: list[ErrorCase] = []

        for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths, strict=True)):
            pred_sv = _preds_to_sv(pred)
            gt_sv = _gt_to_sv(gt)

            n_pred = len(pred_sv)
            n_gt = len(gt_sv)

            if n_pred == 0 and n_gt == 0:
                continue

            # --- All predictions are FP when no GT ---
            if n_gt == 0:
                for i in range(n_pred):
                    area = float(pred_sv.area[i]) if pred_sv.area is not None else 0.0
                    all_errors.append(ErrorCase(
                        image_idx=img_idx,
                        error_type="background_fp",
                        box=pred_sv.xyxy[i],
                        class_id=int(pred_sv.class_id[i]),
                        score=float(pred_sv.confidence[i]) if pred_sv.confidence is not None else None,
                        size_category=_size_category(area),
                    ))
                continue

            # --- All GT are missed when no predictions ---
            if n_pred == 0:
                for j in range(n_gt):
                    area = float(gt_sv.area[j]) if gt_sv.area is not None else 0.0
                    all_errors.append(ErrorCase(
                        image_idx=img_idx,
                        error_type="missed",
                        box=gt_sv.xyxy[j],
                        class_id=int(gt_sv.class_id[j]),
                        score=None,
                        size_category=_size_category(area),
                    ))
                continue

            # --- Compute IoU matrix using supervision ---
            iou_matrix = sv.box_iou_batch(
                pred_sv.xyxy.astype(np.float32),
                gt_sv.xyxy.astype(np.float32),
            )  # (n_pred, n_gt)

            # Greedy matching: sort predictions by confidence descending
            scores = pred_sv.confidence if pred_sv.confidence is not None else np.ones(n_pred)
            sorted_pred_idx = np.argsort(-scores)

            gt_matched = np.full(n_gt, False)
            pred_errors: list[ErrorCase | None] = [None] * n_pred

            for pi in sorted_pred_idx:
                pred_cls = int(pred_sv.class_id[pi])
                pred_score = float(scores[pi])
                pred_area = float(pred_sv.area[pi]) if pred_sv.area is not None else 0.0
                pred_size = _size_category(pred_area)
                best_iou = float(np.max(iou_matrix[pi])) if n_gt > 0 else 0.0
                best_gt_idx = int(np.argmax(iou_matrix[pi]))

                # Try to find a matching GT (same class, IoU >= threshold, unmatched)
                matched = False
                match_candidates = np.argsort(-iou_matrix[pi])
                for gi in match_candidates:
                    if gt_matched[gi]:
                        continue
                    iou_val = float(iou_matrix[pi, gi])
                    gt_cls = int(gt_sv.class_id[gi])

                    if iou_val < self.localization_iou_low:
                        break  # sorted descending, no point continuing

                    if iou_val >= self.iou_threshold and pred_cls == gt_cls:
                        # True positive — mark GT matched, no error
                        gt_matched[gi] = True
                        matched = True
                        break

                    if iou_val >= self.iou_threshold and pred_cls != gt_cls:
                        # Class confusion
                        gt_matched[gi] = True
                        pred_errors[pi] = ErrorCase(
                            image_idx=img_idx,
                            error_type="class_confusion",
                            box=pred_sv.xyxy[pi],
                            class_id=pred_cls,
                            score=pred_score,
                            size_category=pred_size,
                            iou=iou_val,
                            gt_class_id=gt_cls,
                        )
                        matched = True
                        break

                    if (self.localization_iou_low <= iou_val < self.iou_threshold
                            and pred_cls == gt_cls and not gt_matched[gi]):
                        # Localization error — found it but box is poor
                        pred_errors[pi] = ErrorCase(
                            image_idx=img_idx,
                            error_type="localization",
                            box=pred_sv.xyxy[pi],
                            class_id=pred_cls,
                            score=pred_score,
                            size_category=pred_size,
                            iou=iou_val,
                            gt_class_id=gt_cls,
                        )
                        matched = True
                        break

                if not matched:
                    # Check if it overlaps a matched GT (duplicate)
                    if best_iou >= self.iou_threshold:
                        gt_cls = int(gt_sv.class_id[best_gt_idx])
                        if pred_cls == gt_cls and gt_matched[best_gt_idx]:
                            pred_errors[pi] = ErrorCase(
                                image_idx=img_idx,
                                error_type="duplicate",
                                box=pred_sv.xyxy[pi],
                                class_id=pred_cls,
                                score=pred_score,
                                size_category=pred_size,
                                iou=best_iou,
                            )
                            continue

                    # Background false positive
                    pred_errors[pi] = ErrorCase(
                        image_idx=img_idx,
                        error_type="background_fp",
                        box=pred_sv.xyxy[pi],
                        class_id=pred_cls,
                        score=pred_score,
                        size_category=pred_size,
                        iou=best_iou,
                    )

            # Collect prediction errors
            for err in pred_errors:
                if err is not None:
                    all_errors.append(err)

            # Missed GT (false negatives)
            for gi in range(n_gt):
                if not gt_matched[gi]:
                    gt_area = float(gt_sv.area[gi]) if gt_sv.area is not None else 0.0
                    all_errors.append(ErrorCase(
                        image_idx=img_idx,
                        error_type="missed",
                        box=gt_sv.xyxy[gi],
                        class_id=int(gt_sv.class_id[gi]),
                        score=None,
                        size_category=_size_category(gt_area),
                    ))

        return all_errors

    # ----- summary aggregation -----

    def error_summary(
        self,
        errors: list[ErrorCase],
        predictions: list[dict],
        ground_truths: list[dict],
    ) -> dict:
        """Aggregate error statistics into a JSON-serializable dict.

        Returns:
            Dict with keys: total_images, total_errors, per_class, error_types,
            confusion_pairs, size_breakdown, hardest_images.
        """
        num_classes = max(self.class_names.keys()) + 1 if self.class_names else 0

        # --- Per-class TP/FP/FN ---
        per_class: dict[int, dict] = {}
        for cls_id in range(num_classes):
            per_class[cls_id] = {"tp": 0, "fp": 0, "fn": 0, "n_gt": 0}

        # Count GT per class
        for gt in ground_truths:
            labels = np.asarray(gt["labels"], dtype=np.int64).ravel()
            for lbl in labels:
                cls_id = int(lbl)
                if cls_id in per_class:
                    per_class[cls_id]["n_gt"] += 1

        # Count TP from matched predictions
        for pred, gt in zip(predictions, ground_truths, strict=True):
            pred_sv = _preds_to_sv(pred)
            gt_sv = _gt_to_sv(gt)
            if len(pred_sv) > 0 and len(gt_sv) > 0:
                iou_matrix = sv.box_iou_batch(
                    pred_sv.xyxy.astype(np.float32),
                    gt_sv.xyxy.astype(np.float32),
                )
                scores = pred_sv.confidence if pred_sv.confidence is not None else np.ones(len(pred_sv))
                gt_matched = np.full(len(gt_sv), False)
                for pi in np.argsort(-scores):
                    for gi in np.argsort(-iou_matrix[pi]):
                        if gt_matched[gi]:
                            continue
                        if (float(iou_matrix[pi, gi]) >= self.iou_threshold
                                and int(pred_sv.class_id[pi]) == int(gt_sv.class_id[gi])):
                            gt_matched[gi] = True
                            cls_id = int(pred_sv.class_id[pi])
                            if cls_id in per_class:
                                per_class[cls_id]["tp"] += 1
                            break

        # Count FP/FN from errors
        for err in errors:
            cls_id = err.class_id
            if cls_id not in per_class:
                per_class[cls_id] = {"tp": 0, "fp": 0, "fn": 0, "n_gt": 0}
            if err.error_type == "missed":
                per_class[cls_id]["fn"] += 1
            else:
                per_class[cls_id]["fp"] += 1

        # --- Error type breakdown ---
        error_types: dict[str, int] = {}
        for err in errors:
            error_types[err.error_type] = error_types.get(err.error_type, 0) + 1

        # --- Confusion pairs ---
        confusion_pairs: dict[tuple[int, int], int] = {}
        for err in errors:
            if err.error_type == "class_confusion" and err.gt_class_id is not None:
                key = (err.class_id, err.gt_class_id)
                confusion_pairs[key] = confusion_pairs.get(key, 0) + 1
        sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: -x[1])

        # --- Size breakdown ---
        size_breakdown: dict[str, dict[str, int]] = {
            "small": {"fp": 0, "fn": 0},
            "medium": {"fp": 0, "fn": 0},
            "large": {"fp": 0, "fn": 0},
        }
        for err in errors:
            cat = err.size_category
            if err.error_type == "missed":
                size_breakdown[cat]["fn"] += 1
            else:
                size_breakdown[cat]["fp"] += 1

        # --- Hardest images ---
        per_image_errors: dict[int, int] = {}
        for err in errors:
            per_image_errors[err.image_idx] = per_image_errors.get(err.image_idx, 0) + 1
        hardest = sorted(per_image_errors.items(), key=lambda x: -x[1])[:20]

        # Build serializable per-class dict
        per_class_serializable = {}
        for cls_id, counts in per_class.items():
            name = self.class_names.get(cls_id, f"class_{cls_id}")
            per_class_serializable[name] = counts

        return {
            "total_images": len(predictions),
            "total_errors": len(errors),
            "error_types": error_types,
            "per_class": per_class_serializable,
            "confusion_pairs": [
                {
                    "predicted": self.class_names.get(p, str(p)),
                    "ground_truth": self.class_names.get(g, str(g)),
                    "count": c,
                }
                for (p, g), c in sorted_pairs
            ],
            "size_breakdown": size_breakdown,
            "hardest_images": [
                {"image_idx": idx, "error_count": cnt} for idx, cnt in hardest
            ],
        }

    # ----- optimal threshold computation -----

    def compute_optimal_thresholds(
        self,
        predictions: list[dict],
        ground_truths: list[dict],
        threshold_steps: int = 50,
    ) -> dict[int, dict]:
        """Find per-class optimal confidence thresholds.

        Sweeps thresholds and computes precision/recall/F1 at each point
        using supervision's box_iou_batch for matching.

        Args:
            predictions: Per-image prediction dicts.
            ground_truths: Per-image GT dicts.
            threshold_steps: Number of threshold values to test.

        Returns:
            Dict mapping class_id to {best_f1, best_threshold,
            threshold_for_precision_95, threshold_for_recall_90,
            f1_curve: [(threshold, f1), ...]}.
        """
        thresholds = np.linspace(0.05, 0.95, threshold_steps)
        num_classes = max(self.class_names.keys()) + 1 if self.class_names else 0
        results: dict[int, dict] = {}

        # Pre-compute sv objects and per-class IoU matrices (threshold-independent)
        _cached: list[tuple[sv.Detections, sv.Detections, dict[int, tuple]]] = []
        for pred, gt in zip(predictions, ground_truths, strict=True):
            pred_sv = _preds_to_sv(pred)
            gt_sv = _gt_to_sv(gt)
            per_cls_iou: dict[int, tuple] = {}
            for cls_id in range(num_classes):
                pred_mask = (
                    pred_sv.class_id == cls_id
                ) if len(pred_sv) > 0 else np.array([], dtype=bool)
                gt_mask = gt_sv.class_id == cls_id if len(gt_sv) > 0 else np.array([], dtype=bool)
                n_pred = int(pred_mask.sum()) if pred_mask.size > 0 else 0
                n_gt = int(gt_mask.sum()) if gt_mask.size > 0 else 0
                if n_pred > 0 and n_gt > 0:
                    iou = sv.box_iou_batch(
                        pred_sv.xyxy[pred_mask].astype(np.float32),
                        gt_sv.xyxy[gt_mask].astype(np.float32),
                    )
                    confs = pred_sv.confidence[pred_mask] if pred_sv.confidence is not None else np.ones(n_pred)
                    per_cls_iou[cls_id] = (iou, confs, n_gt)
                else:
                    per_cls_iou[cls_id] = (None, None, n_gt if n_pred == 0 else 0, n_pred if n_gt == 0 else 0)
            _cached.append((pred_sv, gt_sv, per_cls_iou))

        for cls_id in range(num_classes):
            f1_curve = []

            for thresh in thresholds:
                tp, fp, fn = 0, 0, 0

                for _pred_sv, _gt_sv, per_cls_iou in _cached:
                    entry = per_cls_iou[cls_id]
                    if len(entry) == 4:
                        # No overlap case: (None, None, n_gt_only, n_pred_only)
                        _, _, n_gt_only, n_pred_only = entry
                        fn += n_gt_only
                        # For pred-only, need to filter by threshold
                        if n_pred_only > 0 and _pred_sv.confidence is not None:
                            mask = (_pred_sv.class_id == cls_id) & (_pred_sv.confidence >= thresh)
                            fp += int(mask.sum())
                        else:
                            fp += n_pred_only
                        continue

                    iou, confs, n_gt_cls = entry
                    # Filter by threshold
                    thresh_mask = confs >= thresh
                    n_above = int(thresh_mask.sum())
                    if n_above == 0:
                        fn += n_gt_cls
                        continue

                    iou_filtered = iou[thresh_mask]
                    gt_matched = np.full(n_gt_cls, False)
                    for pi in range(n_above):
                        best_gi = -1
                        best_val = self.iou_threshold
                        for gi in range(n_gt_cls):
                            if gt_matched[gi]:
                                continue
                            if iou_filtered[pi, gi] >= best_val:
                                best_val = iou_filtered[pi, gi]
                                best_gi = gi
                        if best_gi >= 0:
                            gt_matched[best_gi] = True
                            tp += 1
                        else:
                            fp += 1
                    fn += int((~gt_matched).sum())

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                f1_curve.append((float(thresh), precision, recall, f1))

            if not f1_curve:
                results[cls_id] = {
                    "best_f1": 0.0,
                    "best_threshold": 0.5,
                    "threshold_for_precision_95": None,
                    "threshold_for_recall_90": None,
                    "f1_curve": [],
                }
                continue

            # Find best F1
            best_idx = max(range(len(f1_curve)), key=lambda i: f1_curve[i][3])
            best_f1 = f1_curve[best_idx][3]
            best_thresh = f1_curve[best_idx][0]

            # Find threshold for precision >= 0.95
            thresh_p95 = None
            for t, p, r, _f1 in f1_curve:
                if p >= 0.95 and r > 0:
                    thresh_p95 = t
                    break

            # Find threshold for recall >= 0.90 (search from low to high)
            thresh_r90 = None
            for t, p, r, _f1 in reversed(f1_curve):
                if r >= 0.90:
                    thresh_r90 = t
                    break

            results[cls_id] = {
                "best_f1": round(best_f1, 4),
                "best_threshold": round(best_thresh, 3),
                "threshold_for_precision_95": round(thresh_p95, 3) if thresh_p95 else None,
                "threshold_for_recall_90": round(thresh_r90, 3) if thresh_r90 else None,
                "f1_curve": [(round(t, 3), round(f, 4)) for t, _, _, f in f1_curve],
            }

        return results

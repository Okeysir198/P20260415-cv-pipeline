"""Detection metrics powered by supervision.

Drop-in replacement for the evaluation functions previously in utils/metrics.py.
Uses sv.metrics.MeanAveragePrecision and sv.metrics.ConfusionMatrix for
COCO-style mAP, per-class AP, precision/recall, and confusion matrices.

The API matches the original compute_map() contract so callers need only
change the import path.
"""

import sys
from pathlib import Path

import numpy as np
import supervision as sv
from supervision.metrics import MeanAveragePrecision

# Allow imports from pipeline root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Helpers: convert pipeline dicts → sv.Detections
# ---------------------------------------------------------------------------


def _preds_to_sv(pred: dict) -> sv.Detections:
    """Convert a single image's prediction dict to sv.Detections."""
    boxes = np.asarray(pred["boxes"], dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(pred["scores"], dtype=np.float32).ravel()
    labels = np.asarray(pred["labels"], dtype=np.int64).ravel()
    return sv.Detections(
        xyxy=boxes,
        confidence=scores,
        class_id=labels.astype(int),
    )


def _gt_to_sv(gt: dict) -> sv.Detections:
    """Convert a single image's ground-truth dict to sv.Detections."""
    boxes = np.asarray(gt["boxes"], dtype=np.float32).reshape(-1, 4)
    labels = np.asarray(gt["labels"], dtype=np.int64).ravel()
    return sv.Detections(
        xyxy=boxes,
        class_id=labels.astype(int),
    )


# ---------------------------------------------------------------------------
# Public API — drop-in replacements
# ---------------------------------------------------------------------------


def compute_map_coco(
    predictions: list[dict],
    ground_truths: list[dict],
    num_classes: int = 2,
) -> dict:
    """Compute COCO-standard mAP@[0.5:0.95] averaged over 10 IoU thresholds.

    Uses supervision's MeanAveragePrecision with IoU thresholds from
    0.50 to 0.95 in 0.05 steps (COCO standard).

    Args:
        predictions: List of dicts per image with keys
            ``boxes`` (N, 4), ``scores`` (N,), ``labels`` (N,).
        ground_truths: List of dicts per image with keys
            ``boxes`` (M, 4), ``labels`` (M,).
        num_classes: Number of object classes.

    Returns:
        Dictionary with:
            - ``mAP_50_95``: COCO-standard mAP@[0.5:0.95] (float)
            - ``mAP_50``: mAP@0.5 (float)
            - ``mAP_75``: mAP@0.75 (float)
            - ``per_class_ap_50_95``: Dict[int, float] per-class AP@[0.5:0.95]
    """
    map_metric = MeanAveragePrecision(metric_target=sv.metrics.MetricTarget.BOXES)
    for pred, gt in zip(predictions, ground_truths, strict=True):
        map_metric.update(_preds_to_sv(pred), _gt_to_sv(gt))
    map_result = map_metric.compute()

    # Per-class AP averaged across all IoU thresholds
    per_class_ap_50_95: dict[int, float] = {}
    per_class_ap_50: dict[int, float] = {}
    per_class_ap_75: dict[int, float] = {}

    # Find IoU indices for 0.5 and 0.75
    iou_idx_50 = 0
    iou_idx_75 = 5  # 0.75 is at index 5 in [0.5, 0.55, ..., 0.95]
    if map_result.iou_thresholds is not None:
        iou_idx_50 = int(np.argmin(np.abs(map_result.iou_thresholds - 0.5)))
        iou_idx_75 = int(np.argmin(np.abs(map_result.iou_thresholds - 0.75)))

    for i, cls_id in enumerate(map_result.matched_classes):
        cls_id = int(cls_id)
        # Average across all IoU thresholds for COCO metric
        ap_all = map_result.ap_per_class[i, :]
        per_class_ap_50_95[cls_id] = max(float(np.mean(ap_all)), 0.0)
        per_class_ap_50[cls_id] = max(float(ap_all[iou_idx_50]), 0.0)
        per_class_ap_75[cls_id] = max(float(ap_all[iou_idx_75]), 0.0)

    # Fill missing classes with 0
    for cls_id in range(num_classes):
        per_class_ap_50_95.setdefault(cls_id, 0.0)
        per_class_ap_50.setdefault(cls_id, 0.0)
        per_class_ap_75.setdefault(cls_id, 0.0)

    ap_50_95_vals = list(per_class_ap_50_95.values())
    ap_50_vals = list(per_class_ap_50.values())
    ap_75_vals = list(per_class_ap_75.values())

    return {
        "mAP_50_95": float(np.mean(ap_50_95_vals)) if ap_50_95_vals else 0.0,
        "mAP_50": float(np.mean(ap_50_vals)) if ap_50_vals else 0.0,
        "mAP_75": float(np.mean(ap_75_vals)) if ap_75_vals else 0.0,
        "per_class_ap_50_95": per_class_ap_50_95,
    }


def compute_map(
    predictions: list[dict],
    ground_truths: list[dict],
    iou_threshold: float = 0.5,
    num_classes: int = 2,
) -> dict:
    """Compute mean Average Precision and per-class metrics using supervision.

    Drop-in replacement for ``utils.metrics.compute_map()``.

    Args:
        predictions: List of dicts per image with keys
            ``boxes`` (N, 4), ``scores`` (N,), ``labels`` (N,).
        ground_truths: List of dicts per image with keys
            ``boxes`` (M, 4), ``labels`` (M,).
        iou_threshold: IoU threshold for matching predictions to GT.
        num_classes: Number of object classes.

    Returns:
        Dictionary with:
            - ``mAP``: Mean AP across classes (float)
            - ``per_class_ap``: Dict[int, float] mapping class_id -> AP
            - ``precision``: Dict[int, float] at best F1 per class
            - ``recall``: Dict[int, float] at best F1 per class
            - ``confusion_matrix``: np.ndarray of shape (num_classes+1, num_classes+1)
    """
    # --- mAP via supervision ---
    map_metric = MeanAveragePrecision(metric_target=sv.metrics.MetricTarget.BOXES)
    for pred, gt in zip(predictions, ground_truths, strict=True):
        map_metric.update(_preds_to_sv(pred), _gt_to_sv(gt))
    map_result = map_metric.compute()

    # Extract per-class AP at the requested IoU threshold
    per_class_ap: dict[int, float] = {}
    per_class_precision: dict[int, float] = {}
    per_class_recall: dict[int, float] = {}

    # Find which IoU index corresponds to our threshold
    iou_idx = 0  # default to first (0.5)
    if map_result.iou_thresholds is not None:
        diffs = np.abs(map_result.iou_thresholds - iou_threshold)
        iou_idx = int(np.argmin(diffs))

    for i, cls_id in enumerate(map_result.matched_classes):
        cls_id = int(cls_id)
        ap_val = float(map_result.ap_per_class[i, iou_idx])
        per_class_ap[cls_id] = max(ap_val, 0.0)

    # Fill missing classes with 0
    for cls_id in range(num_classes):
        if cls_id not in per_class_ap:
            per_class_ap[cls_id] = 0.0

    # --- Per-class precision/recall at best F1 ---
    for cls_id in range(num_classes):
        prec, rec, _ = compute_precision_recall(
            predictions, ground_truths, cls_id, iou_threshold
        )
        if prec.size == 0:
            per_class_precision[cls_id] = 0.0
            per_class_recall[cls_id] = 0.0
            continue

        f1 = np.where(
            (prec + rec) > 0,
            2 * prec * rec / (prec + rec + 1e-16),
            0.0,
        )
        best_idx = int(np.argmax(f1))
        per_class_precision[cls_id] = float(prec[best_idx])
        per_class_recall[cls_id] = float(rec[best_idx])

    # Mean AP
    ap_values = list(per_class_ap.values())
    mean_ap = float(np.mean(ap_values)) if ap_values else 0.0

    # --- Confusion matrix via supervision ---
    cm = compute_confusion_matrix(
        predictions, ground_truths, iou_threshold, num_classes
    )

    return {
        "mAP": mean_ap,
        "per_class_ap": per_class_ap,
        "precision": per_class_precision,
        "recall": per_class_recall,
        "confusion_matrix": cm,
    }


def compute_confusion_matrix(
    predictions: list[dict],
    ground_truths: list[dict],
    iou_threshold: float = 0.5,
    num_classes: int = 2,
) -> np.ndarray:
    """Compute confusion matrix using supervision.

    Args:
        predictions: List of prediction dicts per image.
        ground_truths: List of ground-truth dicts per image.
        iou_threshold: IoU threshold for matching.
        num_classes: Number of object classes.

    Returns:
        Confusion matrix of shape (num_classes+1, num_classes+1).
        Last row/col is background (unmatched GT / false positives).
    """
    classes = [str(i) for i in range(num_classes)]
    pred_dets = [_preds_to_sv(p) for p in predictions]
    gt_dets = [_gt_to_sv(g) for g in ground_truths]

    cm = sv.ConfusionMatrix.from_detections(
        predictions=pred_dets,
        targets=gt_dets,
        classes=classes,
        conf_threshold=0.0,
        iou_threshold=iou_threshold,
    )
    return cm.matrix


def _compute_precision_recall_from_iou(
    predictions: list[dict],
    ground_truths: list[dict],
    iou_matrices: list[np.ndarray],
    class_id: int,
    iou_threshold: float = 0.5,
) -> tuple:
    """PR curve for a single class using pre-computed per-image IoU matrices.

    Each ``iou_matrices[i]`` is the full (N_pred, M_gt) IoU matrix for image i
    across ALL classes. Rows/cols are filtered by ``class_id`` inside.
    """
    all_scores = []
    all_tp = []
    n_gt = 0

    for pred, gt, iou_full in zip(predictions, ground_truths, iou_matrices, strict=True):
        pred_scores = np.asarray(pred["scores"], dtype=np.float64).ravel()
        pred_labels = np.asarray(pred["labels"], dtype=np.int64).ravel()
        gt_labels = np.asarray(gt["labels"], dtype=np.int64).ravel()

        pred_mask = pred_labels == class_id
        gt_mask = gt_labels == class_id
        cls_pred_scores = pred_scores[pred_mask]
        n_gt_cls = int(gt_mask.sum())
        n_gt += n_gt_cls

        if cls_pred_scores.size == 0:
            continue

        # Slice the full IoU matrix to this class's rows and cols
        cls_iou = (
            iou_full[pred_mask][:, gt_mask]
            if iou_full.size
            else np.zeros((cls_pred_scores.size, n_gt_cls), dtype=np.float64)
        )

        order = np.argsort(-cls_pred_scores)
        cls_pred_scores = cls_pred_scores[order]
        cls_iou = cls_iou[order]

        gt_matched = np.zeros(n_gt_cls, dtype=bool)

        for i in range(cls_pred_scores.size):
            all_scores.append(cls_pred_scores[i])

            if n_gt_cls == 0:
                all_tp.append(0)
                continue

            row = np.where(gt_matched, -1.0, cls_iou[i])
            best_gt = int(np.argmax(row))
            best_iou = float(row[best_gt])
            if best_iou >= iou_threshold:
                gt_matched[best_gt] = True
                all_tp.append(1)
            else:
                all_tp.append(0)

    if n_gt == 0 or len(all_scores) == 0:
        return np.array([]), np.array([]), np.array([])

    all_scores = np.array(all_scores)
    all_tp = np.array(all_tp)
    order = np.argsort(-all_scores)
    all_tp = all_tp[order]
    all_scores = all_scores[order]

    cum_tp = np.cumsum(all_tp)
    cum_fp = np.cumsum(1 - all_tp)
    precision = cum_tp / (cum_tp + cum_fp + 1e-16)
    recall = cum_tp / (n_gt + 1e-16)

    return precision, recall, all_scores


def compute_precision_recall(
    predictions: list[dict],
    ground_truths: list[dict],
    class_id: int,
    iou_threshold: float = 0.5,
) -> tuple:
    """Compute precision-recall curve for a single class.

    Uses greedy matching at the given IoU threshold (same logic as COCO).

    Args:
        predictions: List of prediction dicts per image.
        ground_truths: List of ground-truth dicts per image.
        class_id: Class ID to compute PR for.
        iou_threshold: IoU threshold for matching.

    Returns:
        Tuple of (precision, recall, thresholds) as numpy arrays.
    """
    all_scores = []
    all_tp = []
    n_gt = 0

    for pred, gt in zip(predictions, ground_truths, strict=True):
        pred_boxes = np.asarray(pred["boxes"], dtype=np.float64).reshape(-1, 4)
        pred_scores = np.asarray(pred["scores"], dtype=np.float64).ravel()
        pred_labels = np.asarray(pred["labels"], dtype=np.int64).ravel()
        gt_boxes = np.asarray(gt["boxes"], dtype=np.float64).reshape(-1, 4)
        gt_labels = np.asarray(gt["labels"], dtype=np.int64).ravel()

        # Filter to this class
        pred_mask = pred_labels == class_id
        gt_mask = gt_labels == class_id
        cls_pred_boxes = pred_boxes[pred_mask]
        cls_pred_scores = pred_scores[pred_mask]
        cls_gt_boxes = gt_boxes[gt_mask]

        n_gt += len(cls_gt_boxes)

        if len(cls_pred_boxes) == 0:
            continue

        # Sort by score descending
        order = np.argsort(-cls_pred_scores)
        cls_pred_boxes = cls_pred_boxes[order]
        cls_pred_scores = cls_pred_scores[order]

        gt_matched = np.zeros(len(cls_gt_boxes), dtype=bool)

        # Compute full IoU matrix once (vectorized via utils.metrics.compute_iou)
        if len(cls_gt_boxes) > 0:
            from utils.metrics import compute_iou as _compute_iou
            iou_matrix = _compute_iou(cls_pred_boxes, cls_gt_boxes)
        else:
            iou_matrix = np.zeros((len(cls_pred_boxes), 0), dtype=np.float64)

        for i in range(len(cls_pred_boxes)):
            all_scores.append(cls_pred_scores[i])

            if len(cls_gt_boxes) == 0:
                all_tp.append(0)
                continue

            # Masked argmax: already-matched GT rows are set to -1.0 so they
            # never win. Keep the outer for-i loop because match order depends
            # on score descending (gt_matched mutates each iteration).
            row = np.where(gt_matched, -1.0, iou_matrix[i])
            best_gt = int(np.argmax(row))
            best_iou = float(row[best_gt])
            if best_iou >= iou_threshold:
                gt_matched[best_gt] = True
                all_tp.append(1)
            else:
                all_tp.append(0)

    if n_gt == 0 or len(all_scores) == 0:
        return np.array([]), np.array([]), np.array([])

    # Sort by score
    all_scores = np.array(all_scores)
    all_tp = np.array(all_tp)
    order = np.argsort(-all_scores)
    all_tp = all_tp[order]
    all_scores = all_scores[order]

    cum_tp = np.cumsum(all_tp)
    cum_fp = np.cumsum(1 - all_tp)
    precision = cum_tp / (cum_tp + cum_fp + 1e-16)
    recall = cum_tp / (n_gt + 1e-16)

    return precision, recall, all_scores

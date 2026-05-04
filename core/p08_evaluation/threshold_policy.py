"""Per-class F1-optimal threshold derivation for error analysis.

Surfaces the same sweep that ``15_threshold_analysis`` renders so the
threshold-sensitive charts (08 / 10 / 11 / 12 / 13) operate at the
deployment-relevant cutoff instead of a hand-picked ``conf_threshold``.

Three policies::

    "fixed"                  — caller-supplied scalar (back-compat).
    "f1_optimal"             — single global F1-max threshold pooled across
                               all classes.
    "f1_optimal_per_class"   — dict ``{cls_id: thr}`` (default).

In policy mode, ``conf_threshold`` becomes the eps floor for raw prediction
collection (default ``1e-3``); the matcher then re-filters per class.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

EPS_FLOOR = 1e-3

# Charts that are filtered through the operating-point threshold.
APPLIED_TO_CHARTS: tuple[str, ...] = (
    "07_per_class_performance",
    "08_confusion_matrix",
    "08_top_confused_pairs",
    "10_failure_mode_contribution",
    "11_failure_by_attribute",
    "12_hardest_images",
    "13_failure_mode_examples",
)

VALID_POLICIES: tuple[str, ...] = ("fixed", "f1_optimal", "f1_optimal_per_class")


def _smooth_curve(values: Iterable[float], window: int) -> np.ndarray:
    """3-point (or N-point) edge-padded moving average to suppress jitter."""
    arr = np.asarray(list(values), dtype=np.float64)
    if window <= 1 or arr.size < window:
        return arr
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: arr.size]


def _f1_curve(class_dets: list[dict], gt_count: int, thresholds: np.ndarray,
              iou_thr: float) -> list[float]:
    out: list[float] = []
    for thr in thresholds:
        kept = [d for d in class_dets if d["score"] >= thr]
        tp = sum(1 for d in kept if d["best_iou_same_class"] >= iou_thr)
        fp = len(kept) - tp
        fn = gt_count - tp
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        out.append(2 * p * r / (p + r) if (p + r) else 0.0)
    return out


def compute_f1_optimal_thresholds(
    detections: list[dict],
    gt_per_class: dict[int, int],
    class_names: dict[int, str],
    *,
    iou_thr: float = 0.5,
    smooth: int = 3,
    policy: str = "f1_optimal_per_class",
    eps: float = EPS_FLOOR,
) -> dict[int, float] | float:
    """Sweep the F1 curve and return the operating-point threshold(s).

    Args:
        detections: per-prediction dicts with ``pred_cls``, ``score``,
            ``best_iou_same_class`` (as collected by the analyzer at the
            eps floor).
        gt_per_class: GT counts keyed by class id.
        class_names: id → name (only its keys are used).
        iou_thr: IoU cutoff for treating a prediction as TP.
        smooth: moving-average window over the F1 curve before arg-max.
            Set to 1 to disable smoothing.
        policy: one of ``"f1_optimal_per_class"`` or ``"f1_optimal"``.
        eps: lower sweep bound; predictions below this were never collected.

    Returns:
        ``dict[int, float]`` for per-class policy, ``float`` for global.
    """
    if policy not in {"f1_optimal", "f1_optimal_per_class"}:
        raise ValueError(f"Unknown policy: {policy!r}")

    thresholds = np.round(np.arange(eps, 1.0, 0.01), 3)

    if policy == "f1_optimal_per_class":
        out: dict[int, float] = {}
        for cid in class_names:
            class_dets = [d for d in detections if d["pred_cls"] == cid]
            gt_count = int(gt_per_class.get(cid, 0))
            if not class_dets or gt_count == 0:
                out[int(cid)] = float(eps)
                continue
            f1s = _f1_curve(class_dets, gt_count, thresholds, iou_thr)
            f1_smooth = _smooth_curve(f1s, smooth)
            out[int(cid)] = float(thresholds[int(np.argmax(f1_smooth))])
        return out

    # f1_optimal: pool across classes (per-class TPs sum, per-class FPs sum)
    total_gt = int(sum(gt_per_class.values()))
    if not detections or total_gt == 0:
        return float(eps)
    f1s: list[float] = []
    for thr in thresholds:
        kept = [d for d in detections if d["score"] >= thr]
        tp = sum(1 for d in kept if d["best_iou_same_class"] >= iou_thr)
        fp = len(kept) - tp
        fn = total_gt - tp
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
    f1_smooth = _smooth_curve(f1s, smooth)
    return float(thresholds[int(np.argmax(f1_smooth))])


def threshold_for(thresholds, cls_id: int, default: float = 0.0) -> float:
    """Look up the threshold for one class.

    Accepts a per-class dict, a scalar, or None.
    """
    if isinstance(thresholds, dict):
        return float(thresholds.get(int(cls_id), default))
    if thresholds is None:
        return float(default)
    return float(thresholds)

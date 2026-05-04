"""Unit tests for the F1-optimal threshold helper used by error analysis.

Synthetic two-class fixture exercising the per-class policy: class A's
TPs cluster at high scores, class B's at low scores. The expected behaviour
is two distinct thresholds, with B < A.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from _runner import run_all  # noqa: E402

from core.p08_evaluation.threshold_policy import (  # noqa: E402
    EPS_FLOOR,
    compute_f1_optimal_thresholds,
)


def _make_synthetic():
    """Two classes with very different score regimes.

    Class 0: 5 TPs at scores 0.6..0.9, 5 FPs at 0.05..0.25.
    Class 1: 5 TPs at scores 0.1..0.2, 5 FPs at 0.005..0.05.
    """
    rng = np.random.default_rng(0)
    detections = []

    # Class 0 — high-score regime. Optimal threshold should be in [0.3, 0.6).
    tp_scores_0 = np.linspace(0.6, 0.9, 5)
    fp_scores_0 = np.linspace(0.05, 0.25, 5)
    for s in tp_scores_0:
        detections.append({"pred_cls": 0, "score": float(s),
                            "best_iou_same_class": 0.9})
    for s in fp_scores_0:
        detections.append({"pred_cls": 0, "score": float(s),
                            "best_iou_same_class": 0.0})

    # Class 1 — low-score regime. Optimal threshold should be much lower.
    tp_scores_1 = np.linspace(0.10, 0.20, 5)
    fp_scores_1 = np.linspace(0.005, 0.05, 5)
    for s in tp_scores_1:
        detections.append({"pred_cls": 1, "score": float(s),
                            "best_iou_same_class": 0.9})
    for s in fp_scores_1:
        detections.append({"pred_cls": 1, "score": float(s),
                            "best_iou_same_class": 0.0})
    rng.shuffle(detections)

    gt_per_class = {0: 5, 1: 5}
    class_names = {0: "high", 1: "low"}
    return detections, gt_per_class, class_names


def test_per_class_returns_dict():
    detections, gt_per_class, class_names = _make_synthetic()
    out = compute_f1_optimal_thresholds(
        detections, gt_per_class, class_names, iou_thr=0.5, smooth=3,
        policy="f1_optimal_per_class",
    )
    assert isinstance(out, dict), f"expected dict, got {type(out)}"
    assert set(out.keys()) == {0, 1}
    assert all(isinstance(v, float) for v in out.values())


def test_per_class_thresholds_separate_regimes():
    detections, gt_per_class, class_names = _make_synthetic()
    out = compute_f1_optimal_thresholds(
        detections, gt_per_class, class_names, iou_thr=0.5, smooth=3,
        policy="f1_optimal_per_class",
    )
    # Class 0's TPs sit at 0.6+ → threshold should be high enough to drop FPs.
    # Class 1's TPs sit at 0.10..0.20 → threshold must be low.
    assert out[0] >= 0.20, f"class 0 thr {out[0]} too low"
    assert out[1] <= 0.15, f"class 1 thr {out[1]} too high"
    assert out[0] > out[1], (
        f"per-class policy collapsed regimes (cls0={out[0]} cls1={out[1]})"
    )


def test_global_policy_returns_scalar():
    detections, gt_per_class, class_names = _make_synthetic()
    out = compute_f1_optimal_thresholds(
        detections, gt_per_class, class_names, iou_thr=0.5,
        policy="f1_optimal",
    )
    assert isinstance(out, float)
    assert EPS_FLOOR <= out < 1.0


def test_empty_class_falls_back_to_eps():
    detections = [
        {"pred_cls": 0, "score": 0.5, "best_iou_same_class": 0.9},
    ]
    gt_per_class = {0: 1, 1: 0}
    class_names = {0: "a", 1: "b"}
    out = compute_f1_optimal_thresholds(
        detections, gt_per_class, class_names,
        policy="f1_optimal_per_class",
    )
    assert out[1] == EPS_FLOOR


def test_smoothing_does_not_break_argmax():
    """Smoothing window=1 disables; window=5 still finds a sane peak."""
    detections, gt_per_class, class_names = _make_synthetic()
    raw = compute_f1_optimal_thresholds(
        detections, gt_per_class, class_names, smooth=1,
        policy="f1_optimal_per_class",
    )
    smoothed = compute_f1_optimal_thresholds(
        detections, gt_per_class, class_names, smooth=5,
        policy="f1_optimal_per_class",
    )
    # Both should keep the per-class regime separation.
    assert raw[0] > raw[1]
    assert smoothed[0] > smoothed[1]


def test_invalid_policy_raises():
    try:
        compute_f1_optimal_thresholds(
            [], {0: 0}, {0: "a"}, policy="bogus",
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown policy")


if __name__ == "__main__":
    run_all([
        ("per_class_returns_dict",            test_per_class_returns_dict),
        ("per_class_thresholds_separate",     test_per_class_thresholds_separate_regimes),
        ("global_policy_returns_scalar",      test_global_policy_returns_scalar),
        ("empty_class_falls_back_to_eps",     test_empty_class_falls_back_to_eps),
        ("smoothing_does_not_break_argmax",   test_smoothing_does_not_break_argmax),
        ("invalid_policy_raises",             test_invalid_policy_raises),
    ], title="p08 threshold-policy")

"""Test 06: sv_metrics — compute_map and compute_precision_recall."""

import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import passed, failed, errors, run_test, run_all
from core.p08_evaluation.sv_metrics import compute_map, compute_precision_recall

OUTPUTS = Path(__file__).resolve().parent / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


def test_compute_map_perfect():
    """Perfect prediction on 1 image should give mAP close to 1.0."""
    predictions = [
        {"boxes": np.array([[10, 10, 50, 50]], dtype=np.float32),
         "scores": np.array([1.0], dtype=np.float32),
         "labels": np.array([0], dtype=np.int64)},
    ]
    ground_truths = [
        {"boxes": np.array([[10, 10, 50, 50]], dtype=np.float32),
         "labels": np.array([0], dtype=np.int64)},
    ]
    result = compute_map(predictions, ground_truths, iou_threshold=0.5, num_classes=1)
    assert abs(result["mAP"] - 1.0) < 0.01, f"Expected mAP ~1.0, got {result['mAP']}"
    assert result["per_class_ap"][0] > 0.9, f"Expected per_class_ap[0] > 0.9, got {result['per_class_ap'][0]}"


def test_compute_map_empty():
    """Empty predictions should give mAP == 0.0."""
    predictions = [
        {"boxes": np.zeros((0, 4), dtype=np.float32),
         "scores": np.array([], dtype=np.float32),
         "labels": np.array([], dtype=np.int64)},
    ]
    ground_truths = [
        {"boxes": np.array([[10, 10, 50, 50]], dtype=np.float32),
         "labels": np.array([0], dtype=np.int64)},
    ]
    result = compute_map(predictions, ground_truths, iou_threshold=0.5, num_classes=1)
    assert result["mAP"] == 0.0, f"Expected mAP == 0.0, got {result['mAP']}"


def test_compute_map_two_classes():
    """Two images, two classes, perfect predictions — mAP should be high."""
    predictions = [
        {"boxes": np.array([[10, 10, 50, 50]], dtype=np.float32),
         "scores": np.array([0.9], dtype=np.float32),
         "labels": np.array([0], dtype=np.int64)},
        {"boxes": np.array([[20, 20, 60, 60]], dtype=np.float32),
         "scores": np.array([0.8], dtype=np.float32),
         "labels": np.array([1], dtype=np.int64)},
    ]
    ground_truths = [
        {"boxes": np.array([[10, 10, 50, 50]], dtype=np.float32),
         "labels": np.array([0], dtype=np.int64)},
        {"boxes": np.array([[20, 20, 60, 60]], dtype=np.float32),
         "labels": np.array([1], dtype=np.int64)},
    ]
    result = compute_map(predictions, ground_truths, iou_threshold=0.5, num_classes=2)
    assert result["mAP"] > 0.5, f"Expected mAP > 0.5, got {result['mAP']}"
    assert len(result["per_class_ap"]) == 2, f"Expected 2 per_class_ap entries, got {len(result['per_class_ap'])}"


def test_precision_recall_shape():
    """Precision and recall arrays should be numpy arrays with matching length > 0."""
    predictions = [
        {"boxes": np.array([[10, 10, 50, 50]], dtype=np.float32),
         "scores": np.array([0.9], dtype=np.float32),
         "labels": np.array([0], dtype=np.int64)},
        {"boxes": np.array([[20, 20, 60, 60]], dtype=np.float32),
         "scores": np.array([0.8], dtype=np.float32),
         "labels": np.array([1], dtype=np.int64)},
    ]
    ground_truths = [
        {"boxes": np.array([[10, 10, 50, 50]], dtype=np.float32),
         "labels": np.array([0], dtype=np.int64)},
        {"boxes": np.array([[20, 20, 60, 60]], dtype=np.float32),
         "labels": np.array([1], dtype=np.int64)},
    ]
    precision, recall, thresholds = compute_precision_recall(
        predictions, ground_truths, class_id=0, iou_threshold=0.5
    )
    assert isinstance(precision, np.ndarray), f"Expected numpy array, got {type(precision)}"
    assert isinstance(recall, np.ndarray), f"Expected numpy array, got {type(recall)}"
    assert len(precision) > 0, "Precision array is empty"
    assert len(precision) == len(recall), f"Length mismatch: precision={len(precision)}, recall={len(recall)}"


if __name__ == "__main__":
    print("\n=== Test 06: sv_metrics ===\n")

    run_all([
        ("compute_map_perfect", test_compute_map_perfect),
        ("compute_map_empty", test_compute_map_empty),
        ("compute_map_two_classes", test_compute_map_two_classes),
        ("precision_recall_shape", test_precision_recall_shape),
    ], exit_on_fail=False)

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import run_all
from core.p06_training.metrics_registry import compute_metrics

H, W = 32, 32
NUM_CLASSES = 3
RS = np.random.RandomState(42)


def test_perfect_prediction():
    mask = RS.randint(0, NUM_CLASSES, size=(H, W))
    preds = [mask.copy()]
    targets = [mask.copy()]
    result = compute_metrics("segmentation", preds, targets, num_classes=NUM_CLASSES)
    assert result["val/mIoU"] > 0.99, f"Expected mIoU ~1.0, got {result['val/mIoU']}"
    for c in range(NUM_CLASSES):
        key = f"val/IoU_cls{c}"
        assert result[key] > 0.99, f"Expected {key} ~1.0, got {result[key]}"


def test_partial_match():
    gt = np.zeros((H, W), dtype=np.int64)
    gt[:, W // 2:] = 1
    pred = np.zeros((H, W), dtype=np.int64)
    pred[H // 2:, :] = 1
    result = compute_metrics("segmentation", [pred], [gt], num_classes=2)
    miou = result["val/mIoU"]
    assert 0.0 < miou < 1.0, f"Expected 0 < mIoU < 1, got {miou}"


def test_per_class_iou_keys():
    mask = RS.randint(0, NUM_CLASSES, size=(H, W))
    result = compute_metrics("segmentation", [mask], [mask], num_classes=NUM_CLASSES)
    for c in range(NUM_CLASSES):
        key = f"val/IoU_cls{c}"
        assert key in result, f"Missing key {key} in {list(result.keys())}"
    assert "val/mIoU" in result


def test_torch_tensor_inputs():
    mask_np = RS.randint(0, NUM_CLASSES, size=(H, W)).astype(np.int64)
    mask_t = torch.from_numpy(mask_np)
    result = compute_metrics("segmentation", [mask_t], [mask_t], num_classes=NUM_CLASSES)
    assert result["val/mIoU"] > 0.99, f"Expected mIoU ~1.0, got {result['val/mIoU']}"


def test_empty_predictions():
    result = compute_metrics("segmentation", [], [], num_classes=NUM_CLASSES)
    assert result["val/mIoU"] == 0.0, f"Expected mIoU=0.0, got {result['val/mIoU']}"


if __name__ == "__main__":
    run_all([
        ("test_perfect_prediction", test_perfect_prediction),
        ("test_partial_match", test_partial_match),
        ("test_per_class_iou_keys", test_per_class_iou_keys),
        ("test_torch_tensor_inputs", test_torch_tensor_inputs),
        ("test_empty_predictions", test_empty_predictions),
    ], title="Test: Segmentation Metrics")

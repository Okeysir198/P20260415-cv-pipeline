"""Test 00: Utils — test critical utility functions (config, device, metrics, visualization)."""

import sys
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import passed, failed, errors, run_test, run_all
import torch
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for CI
import matplotlib.figure
import matplotlib.pyplot as plt
from utils.config import load_config, merge_configs, validate_config, resolve_path, _resolve_variables
from utils.device import get_device, set_seed
from utils.metrics import xywh_to_xyxy, xyxy_to_xywh, compute_iou, nms_numpy
from core.p08_evaluation.visualization import draw_bboxes, plot_confusion_matrix, plot_pr_curve, plot_training_curves

OUTPUTS = Path(__file__).resolve().parent / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


def test_load_and_merge_config():
    """Load real fire.yaml, merge with override, verify merge worked and original keys preserved."""
    config = load_config(str(ROOT / "features" / "safety-fire_detection" / "configs" / "05_data.yaml"))
    assert "dataset_name" in config, "Missing dataset_name in fire.yaml"
    assert "num_classes" in config, "Missing num_classes in fire.yaml"
    original_name = config["dataset_name"]

    merged = merge_configs(config, {"num_classes": 5})
    assert merged["num_classes"] == 5, f"Override failed: num_classes={merged['num_classes']}"
    assert merged["dataset_name"] == original_name, "Original key lost after merge"


def test_validate_config():
    """Validate real config passes; missing dataset_name raises ValueError."""
    config = load_config(str(ROOT / "configs" / "_test" / "05_data.yaml"))
    result = validate_config(config, "data")
    assert result is True, f"validate_config returned {result}"

    bad_config = {"num_classes": 2, "path": "some/path"}
    try:
        validate_config(bad_config, "data")
        raise AssertionError("validate_config should have raised ValueError for missing dataset_name")
    except ValueError:
        pass  # expected


def test_resolve_path():
    """Resolve relative path, verify absolute. Verify absolute path passes through unchanged."""
    base_dir = ROOT / "configs" / "data"
    resolved = resolve_path("../../dataset_store/fire_detection", base_dir)
    assert resolved.is_absolute(), f"Resolved path is not absolute: {resolved}"

    abs_path = Path("/tmp/some/absolute/path")
    resolved_abs = resolve_path(str(abs_path), base_dir)
    assert resolved_abs == abs_path, f"Absolute path changed: {resolved_abs} != {abs_path}"


def test_get_device():
    """get_device() returns a CUDA torch.device; CPU/MPS requests raise."""
    device = get_device()
    assert isinstance(device, torch.device), f"Expected torch.device, got {type(device)}"
    assert device.type == "cuda", f"Expected cuda, got {device}"

    try:
        get_device("cpu")
    except RuntimeError:
        pass
    else:
        raise AssertionError("get_device('cpu') should raise under GPU-only policy")


def test_set_seed():
    """set_seed(42) produces reproducible torch.rand results."""
    device = get_device()
    set_seed(42)
    t1 = torch.rand(5, device=device)
    set_seed(42)
    t2 = torch.rand(5, device=device)
    assert torch.allclose(t1, t2), f"Seeds not reproducible:\n  {t1}\n  {t2}"


def test_xywh_xyxy_roundtrip():
    """xywh -> xyxy -> xywh roundtrip preserves values within tolerance."""
    boxes = np.array([[0.5, 0.5, 0.2, 0.3]])
    xyxy = xywh_to_xyxy(boxes)
    back = xyxy_to_xywh(xyxy)
    assert np.allclose(boxes, back, atol=1e-6), f"Roundtrip mismatch:\n  {boxes}\n  {back}"


def test_compute_iou():
    """Identical box IoU=1.0; disjoint boxes IoU=0.0."""
    box = np.array([[0, 0, 10, 10]], dtype=np.float32)
    iou_same = compute_iou(box, box)
    assert abs(iou_same[0, 0] - 1.0) < 1e-6, f"IoU of identical boxes: {iou_same[0, 0]}"

    box_disjoint = np.array([[20, 20, 30, 30]], dtype=np.float32)
    iou_disjoint = compute_iou(box, box_disjoint)
    assert abs(iou_disjoint[0, 0]) < 1e-6, f"IoU of disjoint boxes: {iou_disjoint[0, 0]}"


def test_nms_numpy():
    """NMS with overlapping boxes: suppress index 1, keep 0 and 2."""
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.8], dtype=np.float32)
    kept = nms_numpy(boxes, scores, iou_threshold=0.5)
    kept_set = set(kept.tolist())
    assert 0 in kept_set, f"Index 0 should be kept, got {kept}"
    assert 2 in kept_set, f"Index 2 should be kept, got {kept}"
    assert 1 not in kept_set, f"Index 1 should be suppressed, got {kept}"


def test_draw_bboxes():
    """draw_bboxes on a real image returns same shape and uint8 dtype."""
    from fixtures import real_image
    image = cv2.resize(real_image(), (640, 480))
    boxes = np.array([[10, 10, 100, 100], [200, 200, 300, 300]], dtype=np.float32)
    labels = np.array([0, 1])
    result = draw_bboxes(image, boxes, labels)
    assert result.shape == image.shape, f"Shape mismatch: {result.shape} != {image.shape}"
    assert result.dtype == np.uint8, f"Dtype mismatch: {result.dtype}"
    assert not np.array_equal(result, image), "Boxes should be visible on real image"


def test_variable_interpolation():
    """Test ${var} interpolation resolves correctly."""
    config = {
        "model": {"arch": "yolox-m"},
        "logging": {"run_name": "${model.arch}_experiment"},
        "reference": "${model.arch}",
    }
    resolved = _resolve_variables(config, config)
    assert resolved["logging"]["run_name"] == "yolox-m_experiment", (
        f"Partial interpolation failed: {resolved['logging']['run_name']}"
    )
    assert resolved["reference"] == "yolox-m", (
        f"Full interpolation failed: {resolved['reference']}"
    )
    print(f"    Resolved: run_name={resolved['logging']['run_name']}, reference={resolved['reference']}")


def test_plot_confusion_matrix():
    """plot_confusion_matrix with 2x2 matrix returns Figure."""
    cm = np.array([[10, 2], [1, 8]])
    fig = plot_confusion_matrix(cm, class_names=["fire", "smoke"])
    assert isinstance(fig, matplotlib.figure.Figure), f"Expected Figure, got {type(fig)}"
    plt.close(fig)
    print(f"    Confusion matrix plot created successfully")


def test_plot_pr_curve():
    """plot_pr_curve with synthetic data returns Figure."""
    precision = np.array([1.0, 0.8, 0.6, 0.5])
    recall = np.array([0.25, 0.5, 0.75, 1.0])
    fig = plot_pr_curve(precision, recall, ap=0.72, class_name="fire")
    assert isinstance(fig, matplotlib.figure.Figure), f"Expected Figure, got {type(fig)}"
    plt.close(fig)
    print(f"    PR curve plot created successfully")


def test_plot_training_curves():
    """plot_training_curves with loss history returns Figure."""
    history = {
        "train_loss": [2.0, 1.5, 1.2, 1.0, 0.8],
        "val_loss": [2.5, 1.8, 1.4, 1.1, 0.9],
        "val_mAP50": [0.1, 0.3, 0.5, 0.6, 0.65],
    }
    fig = plot_training_curves(history)
    assert isinstance(fig, matplotlib.figure.Figure), f"Expected Figure, got {type(fig)}"
    plt.close(fig)
    print(f"    Training curves plot created successfully")


if __name__ == "__main__":
    run_all([
        ("load_and_merge_config", test_load_and_merge_config),
        ("validate_config", test_validate_config),
        ("resolve_path", test_resolve_path),
        ("get_device", test_get_device),
        ("set_seed", test_set_seed),
        ("xywh_xyxy_roundtrip", test_xywh_xyxy_roundtrip),
        ("compute_iou", test_compute_iou),
        ("nms_numpy", test_nms_numpy),
        ("draw_bboxes", test_draw_bboxes),
        ("variable_interpolation", test_variable_interpolation),
        ("plot_confusion_matrix", test_plot_confusion_matrix),
        ("plot_pr_curve", test_plot_pr_curve),
        ("plot_training_curves", test_plot_training_curves),
    ], title="Test 00: Utils")

"""Test 10: Evaluation — evaluate trained model on val set, save metrics."""

import sys
import json
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import passed, failed, errors, run_test, run_all
from core.p08_evaluation.evaluator import ModelEvaluator
from core.p08_evaluation.error_analysis import ErrorAnalyzer, ErrorReport
from core.p06_training.postprocess import postprocess, POSTPROCESSOR_REGISTRY
from core.p06_models import build_model
from utils.config import load_config

OUTPUTS = Path(__file__).resolve().parent / "outputs" / "10_evaluation"
OUTPUTS.mkdir(parents=True, exist_ok=True)

TRAINING_OUTPUTS = Path(__file__).resolve().parent / "outputs" / "08_training"
DATA_CONFIG_PATH = str(ROOT / "configs" / "_test" / "05_data.yaml")
TRAIN_CONFIG_PATH = str(ROOT / "configs" / "_test" / "06_training.yaml")


def _get_checkpoint_path():
    """Find the best checkpoint from training."""
    for name in ["best.pth", "best.pt", "last.pth", "last.pt"]:
        p = TRAINING_OUTPUTS / name
        if p.exists():
            return p
    return None


def _load_model(ckpt_path):
    """Load model from checkpoint."""
    config = load_config(TRAIN_CONFIG_PATH)
    model = build_model(config)

    # Load weights
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_key = "model_state_dict" if "model_state_dict" in ckpt else "model" if "model" in ckpt else "state_dict"
    model.load_state_dict(ckpt[state_key], strict=False)
    model.eval()
    return model


def test_evaluate_val():
    """Evaluate trained model on val set."""
    ckpt_path = _get_checkpoint_path()
    assert ckpt_path is not None, (
        f"No checkpoint found in {TRAINING_OUTPUTS}. Run test_core13_training.py first."
    )

    model = _load_model(ckpt_path)
    data_config = load_config(DATA_CONFIG_PATH)
    data_config["_config_dir"] = Path(DATA_CONFIG_PATH).parent

    evaluator = ModelEvaluator(
        model=model,
        data_config=data_config,
        conf_threshold=0.01,  # low threshold to get some detections
        iou_threshold=0.5,
        batch_size=4,
        num_workers=0,
    )

    results = evaluator.evaluate(split="val")
    assert isinstance(results, dict), f"evaluate() returned {type(results)}"
    print(f"    Result keys: {list(results.keys())}")

    # Check expected keys
    if "mAP" in results:
        print(f"    mAP@0.5: {results['mAP']:.4f}")
    if "per_class_ap" in results:
        print(f"    Per-class AP: {results['per_class_ap']}")

    # Save metrics
    metrics_path = OUTPUTS / "metrics.json"
    # Convert numpy arrays to lists for JSON serialization
    serializable = {}
    for k, v in results.items():
        if hasattr(v, 'tolist'):
            serializable[k] = v.tolist()
        elif hasattr(v, 'item'):
            serializable[k] = v.item()
        else:
            serializable[k] = v
    with open(metrics_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"    Saved: {metrics_path}")


def test_evaluate_returns_metrics():
    """Verify evaluation returns expected metric fields."""
    metrics_path = OUTPUTS / "metrics.json"
    assert metrics_path.exists(), "metrics.json not found — test_evaluate_val must pass first"

    with open(metrics_path) as f:
        results = json.load(f)

    # At minimum, should have mAP
    assert "mAP" in results or "mAP50" in results, f"No mAP in results: {list(results.keys())}"


def test_evaluate_per_class():
    """Verify per_class_ap exists in saved metrics and has entries for each class."""
    metrics_path = OUTPUTS / "metrics.json"
    assert metrics_path.exists(), "metrics.json not found — test_evaluate_val must pass first"

    with open(metrics_path) as f:
        results = json.load(f)

    assert "per_class_ap" in results, (
        f"'per_class_ap' key missing from metrics. Keys: {list(results.keys())}"
    )

    per_class_ap = results["per_class_ap"]
    # per_class_ap may be a dict (class_name -> ap) or a list
    if isinstance(per_class_ap, dict):
        num_entries = len(per_class_ap)
    elif isinstance(per_class_ap, list):
        num_entries = len(per_class_ap)
    else:
        raise AssertionError(
            f"per_class_ap has unexpected type {type(per_class_ap)}: {per_class_ap}"
        )

    assert num_entries >= 2, (
        f"Expected at least 2 per-class AP entries (fire dataset has fire+smoke), "
        f"got {num_entries}: {per_class_ap}"
    )
    print(f"    per_class_ap ({num_entries} classes): {per_class_ap}")


def test_postprocess():
    """Test postprocess with real model output on a real image."""
    ckpt_path = _get_checkpoint_path()
    assert ckpt_path is not None, (
        f"No checkpoint found in {TRAINING_OUTPUTS}. Run test_core13_training.py first."
    )

    model = _load_model(ckpt_path)

    # Load a real image and preprocess it
    from fixtures import real_image_bgr_640
    import numpy as np
    image = real_image_bgr_640(idx=0, split="val")
    image_rgb = image[:, :, ::-1].copy()
    image_tensor = torch.from_numpy(
        image_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    ).unsqueeze(0)

    # Get real model raw output
    with torch.no_grad():
        outputs = model(image_tensor)

    results = postprocess("yolox", model, predictions=outputs, conf_threshold=0.1, nms_threshold=0.5)
    assert isinstance(results, list), f"Expected list, got {type(results)}"
    assert len(results) == 1, f"Expected 1 batch result, got {len(results)}"

    det = results[0]
    assert isinstance(det, dict), f"Expected dict, got {type(det)}"
    assert "boxes" in det and "scores" in det and "labels" in det
    print(f"    Detections: {det['boxes'].shape[0]} boxes")


def test_error_analyzer_classify():
    """Test ErrorAnalyzer.classify_errors with synthetic data."""
    import numpy as np

    class_names = {0: "fire", 1: "smoke"}
    analyzer = ErrorAnalyzer(class_names=class_names, iou_threshold=0.5)

    # Image 0: one TP, one background FP
    predictions = [
        {
            "boxes": np.array([[10, 10, 50, 50], [200, 200, 250, 250]], dtype=np.float64),
            "scores": np.array([0.9, 0.7], dtype=np.float64),
            "labels": np.array([0, 0], dtype=np.int64),
        },
    ]
    ground_truths = [
        {
            "boxes": np.array([[10, 10, 50, 50]], dtype=np.float64),
            "labels": np.array([0], dtype=np.int64),
        },
    ]

    errors_list = analyzer.classify_errors(predictions, ground_truths)
    assert len(errors_list) >= 1, f"Expected at least 1 error, got {len(errors_list)}"

    error_types = {e.error_type for e in errors_list}
    assert "background_fp" in error_types, f"Expected background_fp, got {error_types}"
    print(f"    Error types found: {error_types}")


def test_error_analyzer_missed():
    """Test ErrorAnalyzer detects missed GT (false negatives)."""
    import numpy as np

    class_names = {0: "fire"}
    analyzer = ErrorAnalyzer(class_names=class_names, iou_threshold=0.5)

    # No predictions, one GT → should be a missed error
    predictions = [
        {
            "boxes": np.zeros((0, 4), dtype=np.float64),
            "scores": np.array([], dtype=np.float64),
            "labels": np.array([], dtype=np.int64),
        },
    ]
    ground_truths = [
        {
            "boxes": np.array([[10, 10, 100, 100]], dtype=np.float64),
            "labels": np.array([0], dtype=np.int64),
        },
    ]

    errors_list = analyzer.classify_errors(predictions, ground_truths)
    assert len(errors_list) == 1, f"Expected 1 missed error, got {len(errors_list)}"
    assert errors_list[0].error_type == "missed"
    print(f"    Correctly detected missed GT")


def test_error_analyzer_full_report():
    """Test ErrorAnalyzer.analyze returns a complete ErrorReport."""
    import numpy as np

    class_names = {0: "fire", 1: "smoke"}
    analyzer = ErrorAnalyzer(class_names=class_names, iou_threshold=0.5)

    predictions = [
        {
            "boxes": np.array([[10, 10, 50, 50], [200, 200, 250, 250]], dtype=np.float64),
            "scores": np.array([0.9, 0.7], dtype=np.float64),
            "labels": np.array([0, 1], dtype=np.int64),
        },
    ]
    ground_truths = [
        {
            "boxes": np.array([[10, 10, 50, 50], [60, 60, 120, 120]], dtype=np.float64),
            "labels": np.array([0, 1], dtype=np.int64),
        },
    ]

    report = analyzer.analyze(predictions, ground_truths)
    assert isinstance(report, ErrorReport)
    assert "total_errors" in report.summary
    assert "error_types" in report.summary
    assert "per_class" in report.summary
    assert "size_breakdown" in report.summary
    assert isinstance(report.optimal_thresholds, dict)
    print(f"    Report: {report.summary['total_errors']} errors, "
          f"{len(report.optimal_thresholds)} class thresholds")


def test_error_analyzer_optimal_thresholds():
    """Test optimal threshold computation returns valid results."""
    import numpy as np

    class_names = {0: "fire"}
    analyzer = ErrorAnalyzer(class_names=class_names, iou_threshold=0.5)

    # Multiple images with varying confidence
    predictions = []
    ground_truths = []
    for i in range(10):
        score = 0.1 + i * 0.08
        predictions.append({
            "boxes": np.array([[10, 10, 50, 50]], dtype=np.float64),
            "scores": np.array([score], dtype=np.float64),
            "labels": np.array([0], dtype=np.int64),
        })
        ground_truths.append({
            "boxes": np.array([[10, 10, 50, 50]], dtype=np.float64),
            "labels": np.array([0], dtype=np.int64),
        })

    thresholds = analyzer.compute_optimal_thresholds(predictions, ground_truths, threshold_steps=20)
    assert 0 in thresholds, f"Expected class 0 in thresholds, got {list(thresholds.keys())}"
    assert "best_f1" in thresholds[0]
    assert "best_threshold" in thresholds[0]
    assert thresholds[0]["best_f1"] > 0, "Best F1 should be > 0 for matching data"
    print(f"    Class 0: best_threshold={thresholds[0]['best_threshold']}, "
          f"best_f1={thresholds[0]['best_f1']}")


if __name__ == "__main__":
    run_all([
        ("evaluate_val", test_evaluate_val),
        ("evaluate_returns_metrics", test_evaluate_returns_metrics),
        ("evaluate_per_class", test_evaluate_per_class),
        ("postprocess", test_postprocess),
        ("error_analyzer_classify", test_error_analyzer_classify),
        ("error_analyzer_missed", test_error_analyzer_missed),
        ("error_analyzer_full_report", test_error_analyzer_full_report),
        ("error_analyzer_optimal_thresholds", test_error_analyzer_optimal_thresholds),
    ], title="Test 05: Evaluation")

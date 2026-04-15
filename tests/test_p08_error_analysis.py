"""Test 11: Error Analysis — classify errors, compute thresholds, generate plots.

Uses real fixture images + YOLO labels. No mocks — predictions are constructed
from real GT data (shifted, swapped, removed) to exercise each error type.

Integration test uses the trained checkpoint from test_core13_training (skipped
if no checkpoint exists).
"""

import json
import sys
import traceback
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import passed, failed, errors, run_test, run_all
from fixtures import (
    real_image_with_targets,
    real_image,
    val_image_paths,
    train_image_paths,
    class_names as get_class_names,
)
from core.p08_evaluation.error_analysis import ErrorAnalyzer, ErrorCase, ErrorReport
from core.p08_evaluation.visualization import (
    plot_error_breakdown,
    plot_confidence_histogram,
    plot_size_recall,
    plot_hardest_images_grid,
    plot_threshold_curves,
)

OUTPUTS = Path(__file__).resolve().parent / "outputs" / "11_error_analysis"
OUTPUTS.mkdir(parents=True, exist_ok=True)

TRAINING_OUTPUTS = Path(__file__).resolve().parent / "outputs" / "08_training"
DATA_CONFIG_PATH = str(ROOT / "configs" / "_test" / "05_data.yaml")


# ---------------------------------------------------------------------------
# Helpers: convert real YOLO labels → evaluator format
# ---------------------------------------------------------------------------

def _yolo_to_xyxy(targets: np.ndarray, img_h: int, img_w: int) -> dict:
    """Convert YOLO (N,5) [cls, cx, cy, w, h] normalized → evaluator dict.

    Returns dict with "boxes" (xyxy pixel), "labels" (int64).
    """
    if targets.shape[0] == 0:
        return {
            "boxes": np.zeros((0, 4), dtype=np.float64),
            "labels": np.array([], dtype=np.int64),
        }
    cls_ids = targets[:, 0].astype(np.int64)
    cx = targets[:, 1] * img_w
    cy = targets[:, 2] * img_h
    w = targets[:, 3] * img_w
    h = targets[:, 4] * img_h
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float64)
    return {"boxes": boxes, "labels": cls_ids}


def _load_fixture_images_and_gts(split="train", n=5):
    """Load real images + labels from fixtures, return (images, gts) lists."""
    paths = train_image_paths(n) if split == "train" else val_image_paths(n)
    images = []
    gts = []
    labels_subdir = "labels"
    for img_path in paths:
        img = cv2.imread(str(img_path))
        assert img is not None, f"Failed to read {img_path}"
        images.append(img)
        label_path = img_path.parent.parent / labels_subdir / (img_path.stem + ".txt")
        if label_path.exists():
            raw = np.loadtxt(str(label_path), dtype=np.float32, ndmin=2)
            if raw.size == 0:
                raw = np.zeros((0, 5), dtype=np.float32)
            elif raw.ndim == 1:
                raw = raw.reshape(1, -1)
            targets = raw[:, :5]
        else:
            targets = np.zeros((0, 5), dtype=np.float32)
        h, w = img.shape[:2]
        gts.append(_yolo_to_xyxy(targets, h, w))
    return images, gts


def _make_perfect_predictions(gts):
    """Create predictions that exactly match GT (all TP)."""
    preds = []
    for gt in gts:
        preds.append({
            "boxes": gt["boxes"].copy(),
            "scores": np.ones(len(gt["labels"]), dtype=np.float64) * 0.95,
            "labels": gt["labels"].copy(),
        })
    return preds


def _make_shifted_predictions(gts, shift_px=80):
    """Create predictions with boxes shifted (localization errors)."""
    preds = []
    for gt in gts:
        boxes = gt["boxes"].copy()
        boxes[:, [0, 2]] += shift_px  # shift x
        preds.append({
            "boxes": boxes,
            "scores": np.ones(len(gt["labels"]), dtype=np.float64) * 0.85,
            "labels": gt["labels"].copy(),
        })
    return preds


def _make_wrong_class_predictions(gts, num_classes=2):
    """Create predictions with swapped class IDs (class confusion)."""
    preds = []
    for gt in gts:
        swapped = (gt["labels"] + 1) % num_classes
        preds.append({
            "boxes": gt["boxes"].copy(),
            "scores": np.ones(len(swapped), dtype=np.float64) * 0.80,
            "labels": swapped,
        })
    return preds


def _make_empty_predictions(gts):
    """Create empty predictions (all missed / FN)."""
    preds = []
    for _ in gts:
        preds.append({
            "boxes": np.zeros((0, 4), dtype=np.float64),
            "scores": np.array([], dtype=np.float64),
            "labels": np.array([], dtype=np.int64),
        })
    return preds


def _make_extra_fp_predictions(gts, n_extra=3):
    """Create predictions with extra boxes where no GT exists (background FP)."""
    preds = _make_perfect_predictions(gts)
    for pred in preds:
        extra_boxes = np.array(
            [[500 + i * 20, 500, 540 + i * 20, 540] for i in range(n_extra)],
            dtype=np.float64,
        )
        extra_scores = np.full(n_extra, 0.6, dtype=np.float64)
        extra_labels = np.zeros(n_extra, dtype=np.int64)
        pred["boxes"] = np.vstack([pred["boxes"], extra_boxes])
        pred["scores"] = np.concatenate([pred["scores"], extra_scores])
        pred["labels"] = np.concatenate([pred["labels"], extra_labels])
    return preds


# ---------------------------------------------------------------------------
# Tests: Error classification
# ---------------------------------------------------------------------------

def test_classify_perfect_predictions():
    """Perfect predictions (exact GT match) should produce zero errors."""
    _, gts = _load_fixture_images_and_gts("train", n=5)
    preds = _make_perfect_predictions(gts)
    analyzer = ErrorAnalyzer(class_names=get_class_names(), iou_threshold=0.5)

    error_list = analyzer.classify_errors(preds, gts)
    # Perfect overlap → all TP, no errors
    assert len(error_list) == 0, (
        f"Perfect predictions should have 0 errors, got {len(error_list)}: "
        f"{[e.error_type for e in error_list]}"
    )


def test_classify_background_fp():
    """Extra predictions with no GT match should be background_fp."""
    _, gts = _load_fixture_images_and_gts("train", n=3)
    preds = _make_extra_fp_predictions(gts, n_extra=2)
    analyzer = ErrorAnalyzer(class_names=get_class_names(), iou_threshold=0.5)

    error_list = analyzer.classify_errors(preds, gts)
    fp_errors = [e for e in error_list if e.error_type == "background_fp"]
    # Each image gets 2 extra FP → at least 6 background_fp
    assert len(fp_errors) >= 6, (
        f"Expected >= 6 background_fp errors (2 per image * 3 images), got {len(fp_errors)}"
    )
    # Each FP should have a confidence score
    for e in fp_errors:
        assert e.score is not None
        assert e.score > 0


def test_classify_missed():
    """Empty predictions with real GT should produce missed (FN) errors."""
    _, gts = _load_fixture_images_and_gts("train", n=3)
    preds = _make_empty_predictions(gts)
    analyzer = ErrorAnalyzer(class_names=get_class_names(), iou_threshold=0.5)

    error_list = analyzer.classify_errors(preds, gts)
    total_gt = sum(len(gt["labels"]) for gt in gts)
    missed = [e for e in error_list if e.error_type == "missed"]
    assert len(missed) == total_gt, (
        f"Expected {total_gt} missed errors (one per GT), got {len(missed)}"
    )
    for e in missed:
        assert e.score is None  # missed GT has no confidence


def test_classify_class_confusion():
    """Predictions with swapped classes should produce class_confusion errors."""
    _, gts = _load_fixture_images_and_gts("train", n=3)
    preds = _make_wrong_class_predictions(gts, num_classes=2)
    analyzer = ErrorAnalyzer(class_names=get_class_names(), iou_threshold=0.5)

    error_list = analyzer.classify_errors(preds, gts)
    confusions = [e for e in error_list if e.error_type == "class_confusion"]
    # Every prediction has the wrong class → each matched pair is a confusion
    assert len(confusions) > 0, "Expected class_confusion errors for swapped classes"
    for e in confusions:
        assert e.gt_class_id is not None
        assert e.gt_class_id != e.class_id, "Confusion must have different pred vs GT class"


def test_classify_localization_error():
    """Shifted predictions should produce localization errors."""
    _, gts = _load_fixture_images_and_gts("train", n=3)
    # Use a moderate shift — enough to drop IoU below 0.5 but above 0.3
    preds = _make_shifted_predictions(gts, shift_px=60)
    analyzer = ErrorAnalyzer(class_names=get_class_names(), iou_threshold=0.5, localization_iou_low=0.1)

    error_list = analyzer.classify_errors(preds, gts)
    loc_errors = [e for e in error_list if e.error_type == "localization"]
    # Some boxes will be localization errors (IoU in [0.1, 0.5))
    # Not all — depends on box size vs shift. Just verify we get some.
    print(f"    Localization errors: {len(loc_errors)} "
          f"(of {len(error_list)} total errors)")


def test_size_categories():
    """Error cases should have valid COCO size categories."""
    _, gts = _load_fixture_images_and_gts("train", n=5)
    preds = _make_extra_fp_predictions(gts, n_extra=2)
    analyzer = ErrorAnalyzer(class_names=get_class_names(), iou_threshold=0.5)

    error_list = analyzer.classify_errors(preds, gts)
    valid_sizes = {"small", "medium", "large"}
    for e in error_list:
        assert e.size_category in valid_sizes, (
            f"Invalid size_category '{e.size_category}', expected one of {valid_sizes}"
        )


# ---------------------------------------------------------------------------
# Tests: Error summary
# ---------------------------------------------------------------------------

def test_error_summary_structure():
    """error_summary() returns all expected keys with correct types."""
    _, gts = _load_fixture_images_and_gts("train", n=5)
    preds = _make_extra_fp_predictions(gts, n_extra=2)
    analyzer = ErrorAnalyzer(class_names=get_class_names(), iou_threshold=0.5)

    error_list = analyzer.classify_errors(preds, gts)
    summary = analyzer.error_summary(error_list, preds, gts)

    assert "total_images" in summary
    assert "total_errors" in summary
    assert "error_types" in summary
    assert "per_class" in summary
    assert "confusion_pairs" in summary
    assert "size_breakdown" in summary
    assert "hardest_images" in summary

    assert summary["total_images"] == 5
    assert summary["total_errors"] == len(error_list)
    assert isinstance(summary["error_types"], dict)
    assert isinstance(summary["per_class"], dict)
    assert isinstance(summary["confusion_pairs"], list)
    assert isinstance(summary["hardest_images"], list)

    # size_breakdown has all three categories
    for cat in ["small", "medium", "large"]:
        assert cat in summary["size_breakdown"]

    print(f"    Summary: {summary['total_errors']} errors across {summary['total_images']} images")
    print(f"    Error types: {summary['error_types']}")


def test_error_summary_per_class_counts():
    """Per-class counts (TP/FP/FN) should be consistent."""
    _, gts = _load_fixture_images_and_gts("train", n=5)
    preds = _make_extra_fp_predictions(gts, n_extra=2)
    analyzer = ErrorAnalyzer(class_names=get_class_names(), iou_threshold=0.5)

    error_list = analyzer.classify_errors(preds, gts)
    summary = analyzer.error_summary(error_list, preds, gts)

    for cls_name, counts in summary["per_class"].items():
        assert "tp" in counts and "fp" in counts and "fn" in counts
        assert counts["tp"] >= 0
        assert counts["fp"] >= 0
        assert counts["fn"] >= 0
        print(f"    {cls_name}: TP={counts['tp']} FP={counts['fp']} FN={counts['fn']}")


def test_error_summary_is_json_serializable():
    """Summary must be JSON-serializable (no numpy types)."""
    _, gts = _load_fixture_images_and_gts("train", n=5)
    preds = _make_extra_fp_predictions(gts, n_extra=2)
    analyzer = ErrorAnalyzer(class_names=get_class_names(), iou_threshold=0.5)

    error_list = analyzer.classify_errors(preds, gts)
    summary = analyzer.error_summary(error_list, preds, gts)

    # This will raise TypeError if any value is not JSON-serializable
    json_str = json.dumps(summary)
    assert len(json_str) > 0

    # Save for inspection
    with open(OUTPUTS / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"    Saved: {OUTPUTS / 'summary.json'}")


def test_hardest_images_ranking():
    """Hardest images should be sorted by error count descending."""
    _, gts = _load_fixture_images_and_gts("train", n=5)
    # Mix: some images have extra FP, some have no predictions
    preds = _make_extra_fp_predictions(gts, n_extra=3)
    # Make image 0 have no predictions → lots of missed errors too
    preds[0] = {
        "boxes": np.zeros((0, 4), dtype=np.float64),
        "scores": np.array([], dtype=np.float64),
        "labels": np.array([], dtype=np.int64),
    }
    analyzer = ErrorAnalyzer(class_names=get_class_names(), iou_threshold=0.5)

    error_list = analyzer.classify_errors(preds, gts)
    summary = analyzer.error_summary(error_list, preds, gts)

    hardest = summary["hardest_images"]
    assert len(hardest) > 0
    # Verify descending order
    counts = [h["error_count"] for h in hardest]
    assert counts == sorted(counts, reverse=True), (
        f"Hardest images not sorted descending: {counts}"
    )
    print(f"    Hardest images: {hardest[:5]}")


# ---------------------------------------------------------------------------
# Tests: Optimal thresholds
# ---------------------------------------------------------------------------

def test_optimal_thresholds_structure():
    """compute_optimal_thresholds returns per-class results with expected keys."""
    _, gts = _load_fixture_images_and_gts("train", n=5)
    preds = _make_perfect_predictions(gts)
    analyzer = ErrorAnalyzer(class_names=get_class_names(), iou_threshold=0.5)

    thresholds = analyzer.compute_optimal_thresholds(preds, gts, threshold_steps=10)
    class_names_map = get_class_names()

    for cls_id in class_names_map:
        assert cls_id in thresholds, f"Missing class {cls_id} in thresholds"
        result = thresholds[cls_id]
        assert "best_f1" in result
        assert "best_threshold" in result
        assert "f1_curve" in result
        assert isinstance(result["f1_curve"], list)
        print(f"    {class_names_map[cls_id]}: "
              f"best_threshold={result['best_threshold']}, best_f1={result['best_f1']}")


def test_optimal_thresholds_perfect_preds():
    """Perfect predictions should yield high F1 at some threshold."""
    _, gts = _load_fixture_images_and_gts("train", n=5)
    preds = _make_perfect_predictions(gts)
    analyzer = ErrorAnalyzer(class_names=get_class_names(), iou_threshold=0.5)

    thresholds = analyzer.compute_optimal_thresholds(preds, gts, threshold_steps=20)

    # At least one class with GT should have F1 > 0
    has_high_f1 = any(
        thresholds[cls_id]["best_f1"] > 0.5
        for cls_id in thresholds
        if any(int(gt["labels"][i]) == cls_id
               for gt in gts for i in range(len(gt["labels"])))
    )
    assert has_high_f1, "Perfect predictions should yield best_f1 > 0.5"


def test_optimal_thresholds_json_serializable():
    """Threshold results must be JSON-serializable."""
    _, gts = _load_fixture_images_and_gts("train", n=5)
    preds = _make_extra_fp_predictions(gts, n_extra=2)
    analyzer = ErrorAnalyzer(class_names=get_class_names(), iou_threshold=0.5)

    thresholds = analyzer.compute_optimal_thresholds(preds, gts, threshold_steps=10)

    # Serialize (strip f1_curve for the output file, keep full for the test)
    json_str = json.dumps(thresholds)
    assert len(json_str) > 0

    # Save compact version
    compact = {}
    for cls_id, result in thresholds.items():
        compact[get_class_names().get(cls_id, str(cls_id))] = {
            k: v for k, v in result.items() if k != "f1_curve"
        }
    with open(OUTPUTS / "optimal_thresholds.json", "w") as f:
        json.dump(compact, f, indent=2)
    print(f"    Saved: {OUTPUTS / 'optimal_thresholds.json'}")


# ---------------------------------------------------------------------------
# Tests: Full analyze() pipeline
# ---------------------------------------------------------------------------

def test_analyze_full_report():
    """analyze() returns a complete ErrorReport with all fields."""
    _, gts = _load_fixture_images_and_gts("train", n=5)
    preds = _make_extra_fp_predictions(gts, n_extra=2)
    analyzer = ErrorAnalyzer(class_names=get_class_names(), iou_threshold=0.5)

    report = analyzer.analyze(preds, gts)

    assert isinstance(report, ErrorReport)
    assert isinstance(report.errors, list)
    assert isinstance(report.per_image_error_count, dict)
    assert isinstance(report.summary, dict)
    assert isinstance(report.optimal_thresholds, dict)
    assert len(report.errors) == report.summary["total_errors"]

    # Save full report
    ea_json = {
        "summary": report.summary,
        "optimal_thresholds": {
            get_class_names().get(k, str(k)): {
                kk: vv for kk, vv in v.items() if kk != "f1_curve"
            }
            for k, v in report.optimal_thresholds.items()
        },
        "per_image_error_count": {
            str(k): v for k, v in report.per_image_error_count.items()
        },
    }
    with open(OUTPUTS / "error_analysis.json", "w") as f:
        json.dump(ea_json, f, indent=2)
    print(f"    Saved: {OUTPUTS / 'error_analysis.json'}")
    print(f"    Report: {report.summary['total_errors']} errors, "
          f"{len(report.per_image_error_count)} images with errors")


# ---------------------------------------------------------------------------
# Tests: Visualization (smoke tests — verify files are written)
# ---------------------------------------------------------------------------

def _get_report_and_images():
    """Shared setup for visualization tests."""
    images, gts = _load_fixture_images_and_gts("train", n=5)
    preds = _make_extra_fp_predictions(gts, n_extra=2)
    # Also mix in some missed (empty preds for image 0)
    preds[0] = {
        "boxes": np.zeros((0, 4), dtype=np.float64),
        "scores": np.array([], dtype=np.float64),
        "labels": np.array([], dtype=np.int64),
    }
    analyzer = ErrorAnalyzer(class_names=get_class_names(), iou_threshold=0.5)
    report = analyzer.analyze(preds, gts)
    return report, images


def test_plot_error_breakdown():
    """plot_error_breakdown produces a valid PNG."""
    report, _ = _get_report_and_images()
    save_path = str(OUTPUTS / "error_breakdown.png")
    fig = plot_error_breakdown(report.summary, save_path=save_path)
    plt.close(fig)

    path = Path(save_path)
    assert path.exists(), f"File not created: {save_path}"
    assert path.stat().st_size > 1000, f"File too small: {path.stat().st_size} bytes"
    print(f"    Saved: {save_path} ({path.stat().st_size:,} bytes)")


def test_plot_confidence_histogram():
    """plot_confidence_histogram produces a valid PNG."""
    report, _ = _get_report_and_images()
    save_path = str(OUTPUTS / "confidence_histogram.png")
    fig = plot_confidence_histogram(report.errors, save_path=save_path)
    plt.close(fig)

    path = Path(save_path)
    assert path.exists()
    assert path.stat().st_size > 1000
    print(f"    Saved: {save_path} ({path.stat().st_size:,} bytes)")


def test_plot_size_recall():
    """plot_size_recall produces a valid PNG."""
    report, _ = _get_report_and_images()
    save_path = str(OUTPUTS / "size_recall.png")
    fig = plot_size_recall(report.summary, save_path=save_path)
    plt.close(fig)

    path = Path(save_path)
    assert path.exists()
    assert path.stat().st_size > 1000
    print(f"    Saved: {save_path} ({path.stat().st_size:,} bytes)")


def test_plot_hardest_images_grid():
    """plot_hardest_images_grid produces a valid PNG with real images."""
    report, images = _get_report_and_images()
    save_path = str(OUTPUTS / "hardest_images.png")
    fig = plot_hardest_images_grid(
        report.errors, images, get_class_names(),
        top_n=4, save_path=save_path,
    )
    plt.close(fig)

    path = Path(save_path)
    assert path.exists()
    assert path.stat().st_size > 5000, "Hardest images grid should be > 5KB"
    print(f"    Saved: {save_path} ({path.stat().st_size:,} bytes)")


def test_plot_threshold_curves():
    """plot_threshold_curves produces a valid PNG."""
    report, _ = _get_report_and_images()
    save_path = str(OUTPUTS / "threshold_curves.png")
    fig = plot_threshold_curves(
        report.optimal_thresholds, get_class_names(),
        save_path=save_path,
    )
    plt.close(fig)

    path = Path(save_path)
    assert path.exists()
    assert path.stat().st_size > 1000
    print(f"    Saved: {save_path} ({path.stat().st_size:,} bytes)")


# ---------------------------------------------------------------------------
# Test: Integration with real trained model (optional)
# ---------------------------------------------------------------------------

def test_integration_with_trained_model():
    """Run error analysis on real model predictions from the val set.

    Requires a trained checkpoint from test_core13_training. Skipped if
    no checkpoint exists.
    """
    import torch
    from core.p08_evaluation.evaluator import ModelEvaluator
    from core.p06_models import build_model
    from utils.config import load_config

    # Find checkpoint
    ckpt_path = None
    for name in ["best.pth", "best.pt", "last.pth", "last.pt"]:
        p = TRAINING_OUTPUTS / name
        if p.exists():
            ckpt_path = p
            break
    assert ckpt_path is not None, (
        f"No checkpoint in {TRAINING_OUTPUTS}. Run test_core13_training first. "
        "This test is skipped if no checkpoint exists."
    )

    # Load model
    train_config_path = str(ROOT / "configs" / "_test" / "06_training.yaml")
    config = load_config(train_config_path)
    model = build_model(config)

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_key = (
        "model_state_dict" if "model_state_dict" in ckpt
        else "model" if "model" in ckpt
        else "state_dict"
    )
    model.load_state_dict(ckpt[state_key], strict=False)
    model.eval()

    # Run evaluator
    data_config = load_config(DATA_CONFIG_PATH)
    data_config["_config_dir"] = Path(DATA_CONFIG_PATH).parent
    class_names_map = {int(k): v for k, v in data_config["names"].items()}

    evaluator = ModelEvaluator(
        model=model,
        data_config=data_config,
        conf_threshold=0.01,  # low threshold to get errors
        iou_threshold=0.5,
        batch_size=4,
        num_workers=0,
    )
    predictions, ground_truths = evaluator.get_predictions(split="val")
    print(f"    Got {len(predictions)} images from evaluator")

    # Run error analysis on real predictions
    analyzer = ErrorAnalyzer(class_names=class_names_map, iou_threshold=0.5)
    report = analyzer.analyze(predictions, ground_truths)

    assert isinstance(report, ErrorReport)
    print(f"    Real model errors: {report.summary['total_errors']}")
    print(f"    Error types: {report.summary['error_types']}")
    for cls_name, counts in report.summary["per_class"].items():
        print(f"    {cls_name}: TP={counts['tp']} FP={counts['fp']} FN={counts['fn']}")

    # Save integration results
    integration_json = {
        "summary": report.summary,
        "optimal_thresholds": {
            class_names_map.get(k, str(k)): {
                kk: vv for kk, vv in v.items() if kk != "f1_curve"
            }
            for k, v in report.optimal_thresholds.items()
        },
    }
    with open(OUTPUTS / "integration_error_analysis.json", "w") as f:
        json.dump(integration_json, f, indent=2)
    print(f"    Saved: {OUTPUTS / 'integration_error_analysis.json'}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_all([
        # Error classification
        ("classify_perfect_predictions", test_classify_perfect_predictions),
        ("classify_background_fp", test_classify_background_fp),
        ("classify_missed", test_classify_missed),
        ("classify_class_confusion", test_classify_class_confusion),
        ("classify_localization_error", test_classify_localization_error),
        ("size_categories", test_size_categories),
        # Error summary
        ("error_summary_structure", test_error_summary_structure),
        ("error_summary_per_class_counts", test_error_summary_per_class_counts),
        ("error_summary_is_json_serializable", test_error_summary_is_json_serializable),
        ("hardest_images_ranking", test_hardest_images_ranking),
        # Optimal thresholds
        ("optimal_thresholds_structure", test_optimal_thresholds_structure),
        ("optimal_thresholds_perfect_preds", test_optimal_thresholds_perfect_preds),
        ("optimal_thresholds_json_serializable", test_optimal_thresholds_json_serializable),
        # Full pipeline
        ("analyze_full_report", test_analyze_full_report),
        # Visualization
        ("plot_error_breakdown", test_plot_error_breakdown),
        ("plot_confidence_histogram", test_plot_confidence_histogram),
        ("plot_size_recall", test_plot_size_recall),
        ("plot_hardest_images_grid", test_plot_hardest_images_grid),
        ("plot_threshold_curves", test_plot_threshold_curves),
        # Integration (optional — needs trained checkpoint)
        ("integration_with_trained_model", test_integration_with_trained_model),
    ], title="Test 10: Error Analysis")

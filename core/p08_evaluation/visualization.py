"""Visualization utilities for detection results and training diagnostics.

Uses matplotlib for all plotting (unified rcParams via
:func:`utils.viz.apply_plot_style`). Images are NumPy BGR arrays (OpenCV
convention). Supervision annotators are used for bounding box drawing.

The plot functions here are consumed by :mod:`evaluate.py`,
:mod:`analyze_errors.py`, and the test suite. For the modern numbered-chart
scheme (``01_overview.png`` … ``14_size_recall.png``), callers should use
:func:`core.p08_evaluation.error_analysis_runner.run_error_analysis` and
look up target filenames in
:data:`core.p08_evaluation.error_analysis_runner.CHART_FILENAMES`.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv

import core.p10_inference.supervision_bridge as _sv_bridge
from utils.viz import apply_plot_style, new_figure, shorten_label

if TYPE_CHECKING:
    from core.p08_evaluation.error_analysis import ErrorCase


apply_plot_style()


# Rotate tick labels when there are many categorical items — avoids overlap.
_TICK_ROT_THRESHOLD = 5
_TICK_ROT_DEG = 40


def _rotate_xticks(ax, labels: list[str]) -> None:
    if len(labels) > _TICK_ROT_THRESHOLD:
        ax.set_xticklabels(labels, rotation=_TICK_ROT_DEG, ha="right")
    else:
        ax.set_xticklabels(labels)


def _save(fig: matplotlib.figure.Figure, save_path: str | None) -> None:
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Bounding box drawing
# ---------------------------------------------------------------------------


def draw_bboxes(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray | None = None,
    class_names: dict[int, str] | None = None,
    colors: dict[int, tuple[int, int, int]] | None = None,
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Draw bounding boxes on an image with labels and confidence scores.

    Args:
        image: BGR image array of shape (H, W, 3).
        boxes: Array of shape (N, 4) in [x1, y1, x2, y2] format.
        labels: Array of shape (N,) with integer class IDs.
        scores: Optional array of shape (N,) with confidence scores.
        class_names: Optional mapping from class ID to display name.
        colors: Optional mapping from class ID to BGR color tuple.
        thickness: Box line thickness in pixels.
        font_scale: Font scale for labels.

    Returns:
        Copy of the image with drawn boxes (original is not modified).
    """
    img = image.copy()
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    labels = np.asarray(labels, dtype=np.int64).ravel()

    if scores is not None:
        scores = np.asarray(scores, dtype=np.float64).ravel()

    if boxes.shape[0] == 0:
        return img

    predictions = {
        "boxes": boxes,
        "scores": scores if scores is not None else np.ones(boxes.shape[0], dtype=np.float64),
        "labels": labels,
    }
    detections = _sv_bridge.to_sv_detections(predictions)

    annotators = _sv_bridge.build_annotators({
        "supervision": {
            "bbox": {"thickness": thickness},
            "label": {"text_scale": font_scale, "text_thickness": 1},
        }
    })

    img = annotators["box"].annotate(scene=img, detections=detections)

    label_class_names = (
        class_names if class_names
        else {int(i): str(i) for i in np.unique(labels)}
    )
    sv_labels = _sv_bridge.build_labels(detections, label_class_names)
    # If original scores were None, strip the trailing " 1.00" from labels
    if scores is None:
        sv_labels = [lbl.rsplit(" ", 1)[0] if " " in lbl else lbl for lbl in sv_labels]
    img = annotators["label"].annotate(scene=img, detections=detections, labels=sv_labels)

    return img


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: str | None = None,
    normalize: bool = True,
    figsize: tuple[int, int] = (10, 8),
) -> matplotlib.figure.Figure:
    """Plot confusion matrix as a heatmap."""
    display_names = [shorten_label(n) for n in class_names]
    if cm.shape[0] > len(display_names):
        display_names.append("background")

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True).astype(np.float64)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_display = cm.astype(np.float64) / row_sums
        fmt = ".2f"
        title = "Confusion Matrix (normalized)"
    else:
        cm_display = cm.astype(np.float64)
        fmt = ".0f"
        title = "Confusion Matrix"

    fig, ax = new_figure(figsize=figsize)
    im = ax.imshow(cm_display, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        yticklabels=display_names,
        ylabel="Ground Truth",
        xlabel="Predicted",
        title=title,
    )
    _rotate_xticks(ax, display_names)
    plt.setp(ax.get_xticklabels(), rotation_mode="anchor")

    # Text annotations
    thresh = cm_display.max() / 2.0
    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            val = cm_display[i, j]
            text = f"{val:{fmt}}"
            ax.text(
                j, i, text,
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=9,
            )

    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# PR curve plot
# ---------------------------------------------------------------------------


def plot_pr_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    ap: float,
    class_name: str,
    save_path: str | None = None,
    figsize: tuple[int, int] = (8, 6),
) -> matplotlib.figure.Figure:
    """Plot precision-recall curve for a single class."""
    short = shorten_label(class_name)
    fig, ax = new_figure(figsize=figsize)
    ax.plot(recall, precision, linewidth=2, label=f"{short} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve: {short}")
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------


def plot_training_curves(
    history: dict[str, list[float]],
    save_path: str | None = None,
    figsize: tuple[int, int] = (14, 5),
) -> matplotlib.figure.Figure:
    """Plot training/validation loss and mAP curves."""
    has_loss = "train_loss" in history or "val_loss" in history
    has_map = "val_mAP50" in history
    n_plots = int(has_loss) + int(has_map)
    if n_plots == 0:
        n_plots = 1

    fig, axes = new_figure(nrows=1, ncols=max(n_plots, 1), figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    if has_loss:
        ax = axes[plot_idx]
        if "train_loss" in history:
            epochs = range(1, len(history["train_loss"]) + 1)
            ax.plot(epochs, history["train_loss"], label="Train Loss", linewidth=1.5)
        if "val_loss" in history:
            epochs = range(1, len(history["val_loss"]) + 1)
            ax.plot(epochs, history["val_loss"], label="Val Loss", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    if has_map:
        ax = axes[plot_idx]
        epochs = range(1, len(history["val_mAP50"]) + 1)
        ax.plot(epochs, history["val_mAP50"], color="green", linewidth=1.5, label="mAP@0.5")
        if "val_mAP50_95" in history:
            epochs2 = range(1, len(history["val_mAP50_95"]) + 1)
            ax.plot(epochs2, history["val_mAP50_95"], color="orange", linewidth=1.5,
                    label="mAP@0.5:0.95")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("mAP")
        ax.set_title("Validation mAP")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    if plot_idx == 0:
        ax = axes[0]
        for key, values in history.items():
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, label=key, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_title("Training Metrics")
        ax.legend()
        ax.grid(True, alpha=0.3)

    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Class distribution
# ---------------------------------------------------------------------------


def plot_class_distribution(
    class_counts: dict[str, int],
    title: str = "",
    save_path: str | None = None,
    figsize: tuple[int, int] = (10, 5),
) -> matplotlib.figure.Figure:
    """Bar chart of class instance distribution."""
    if not class_counts:
        fig, ax = new_figure(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        _save(fig, save_path)
        return fig

    names = [shorten_label(n) for n in class_counts.keys()]
    counts = list(class_counts.values())

    fig, ax = new_figure(figsize=figsize)
    bars = ax.bar(range(len(names)), counts,
                  color=plt.cm.Set2(np.linspace(0, 1, len(names))))
    ax.set_xticks(range(len(names)))
    _rotate_xticks(ax, names)
    ax.set_ylabel("Count")
    ax.set_title(title or "Class Distribution")

    for bar, count in zip(bars, counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{count:,}", ha="center", va="bottom", fontsize=9,
        )

    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Detection grid
# ---------------------------------------------------------------------------


def create_detection_grid(
    images: list[np.ndarray],
    predictions: list[dict],
    class_names: dict[int, str],
    nrows: int = 2,
    ncols: int = 4,
    save_path: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> matplotlib.figure.Figure:
    """Create a grid of images with detection results drawn."""
    n = min(len(images), nrows * ncols)
    if figsize is None:
        figsize = (ncols * 4, nrows * 3.5)

    fig, axes = new_figure(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = np.asarray(axes).ravel()

    for i in range(nrows * ncols):
        ax = axes[i]
        ax.axis("off")
        if i >= n:
            continue

        pred = predictions[i]
        boxes = np.asarray(pred.get("boxes", []), dtype=np.float64).reshape(-1, 4)
        labels = np.asarray(pred.get("labels", []), dtype=np.int64).ravel()
        scores = np.asarray(pred.get("scores", []), dtype=np.float64).ravel()
        scores = scores if scores.size > 0 else None

        img_drawn = draw_bboxes(images[i], boxes, labels, scores, class_names)
        ax.imshow(cv2.cvtColor(img_drawn, cv2.COLOR_BGR2RGB))

    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Error analysis visualizations
# ---------------------------------------------------------------------------


def plot_error_breakdown(
    summary: dict,
    save_path: str | None = None,
    figsize: tuple[int, int] = (12, 6),
) -> matplotlib.figure.Figure:
    """Stacked bar chart of TP/FP/FN per class."""
    per_class = summary.get("per_class", {})
    if not per_class:
        fig, ax = new_figure(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        _save(fig, save_path)
        return fig

    class_names_list = list(per_class.keys())
    short_names = [shorten_label(c) for c in class_names_list]
    tp_vals = [per_class[c].get("tp", 0) for c in class_names_list]
    fp_vals = [per_class[c].get("fp", 0) for c in class_names_list]
    fn_vals = [per_class[c].get("fn", 0) for c in class_names_list]

    x = np.arange(len(class_names_list))
    width = 0.25

    fig, ax = new_figure(figsize=figsize)
    ax.bar(x - width, tp_vals, width, label="TP", color="#2ecc71")
    ax.bar(x, fp_vals, width, label="FP", color="#e74c3c")
    ax.bar(x + width, fn_vals, width, label="FN", color="#f39c12")

    ax.set_xticks(x)
    _rotate_xticks(ax, short_names)
    ax.set_ylabel("Count")
    ax.set_title("Error Breakdown by Class")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    error_types = summary.get("error_types", {})
    if error_types:
        text_lines = [f"{k}: {v}" for k, v in sorted(error_types.items(),
                                                     key=lambda item: -item[1])]
        ax.text(
            0.98, 0.98, "\n".join(text_lines),
            transform=ax.transAxes, fontsize=8, verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    _save(fig, save_path)
    return fig


def plot_confidence_histogram(
    errors: list["ErrorCase"],
    save_path: str | None = None,
    figsize: tuple[int, int] = (10, 5),
) -> matplotlib.figure.Figure:
    """Histogram of false positive confidence scores."""
    fp_scores = [e.score for e in errors if e.error_type != "missed" and e.score is not None]
    fn_count = sum(1 for e in errors if e.error_type == "missed")

    fig, ax = new_figure(figsize=figsize)

    if fp_scores:
        ax.hist(
            fp_scores, bins=30, alpha=0.7, color="#e74c3c",
            label=f"FP ({len(fp_scores)})", edgecolor="black", linewidth=0.5,
        )

    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Count")
    ax.set_title(f"False Positive Confidence Distribution (FN count: {fn_count})")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    _save(fig, save_path)
    return fig


def plot_size_recall(
    summary: dict,
    save_path: str | None = None,
    figsize: tuple[int, int] = (8, 5),
) -> matplotlib.figure.Figure:
    """Bar chart of FP/FN counts by object size category (COCO)."""
    size_data = summary.get("size_breakdown", {})
    categories = ["small", "medium", "large"]
    fp_vals = [size_data.get(c, {}).get("fp", 0) for c in categories]
    fn_vals = [size_data.get(c, {}).get("fn", 0) for c in categories]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = new_figure(figsize=figsize)
    ax.bar(x - width / 2, fp_vals, width, label="FP", color="#e74c3c")
    ax.bar(x + width / 2, fn_vals, width, label="FN (missed)", color="#f39c12")

    ax.set_xticks(x)
    ax.set_xticklabels([
        "small\n(<32px)", "medium\n(<96px)", "large\n(>=96px)",
    ])
    ax.set_ylabel("Count")
    ax.set_title("Errors by Object Size (COCO categories)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    _save(fig, save_path)
    return fig


def plot_hardest_images_grid(
    errors: list["ErrorCase"],
    images: list[np.ndarray],
    class_names: dict[int, str],
    top_n: int = 8,
    save_path: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> matplotlib.figure.Figure:
    """Grid of hardest images with error boxes drawn via supervision annotators."""
    per_image: dict[int, int] = {}
    for err in errors:
        per_image[err.image_idx] = per_image.get(err.image_idx, 0) + 1
    ranked = sorted(per_image.items(), key=lambda item: -item[1])[:top_n]

    if not ranked:
        fig, ax = new_figure()
        ax.text(0.5, 0.5, "No errors found", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        _save(fig, save_path)
        return fig

    n = len(ranked)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    if figsize is None:
        figsize = (ncols * 5, nrows * 4)

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.4, text_thickness=1)

    fig, axes = new_figure(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = np.asarray(axes).ravel()

    for ax_idx, (img_idx, err_count) in enumerate(ranked):
        ax = axes[ax_idx]
        ax.axis("off")

        if img_idx >= len(images):
            continue

        img = images[img_idx].copy()
        img_errors = [e for e in errors if e.image_idx == img_idx]

        fn_errors = [e for e in img_errors if e.error_type == "missed"]
        if fn_errors:
            fn_boxes = np.array([e.box for e in fn_errors], dtype=np.float32)
            fn_cls = np.array([e.class_id for e in fn_errors], dtype=int)
            fn_dets = sv.Detections(xyxy=fn_boxes, class_id=fn_cls)
            fn_labels = [
                f"FN: {shorten_label(class_names.get(e.class_id, str(e.class_id)))}"
                for e in fn_errors
            ]
            img = box_annotator.annotate(scene=img, detections=fn_dets)
            img = label_annotator.annotate(scene=img, detections=fn_dets, labels=fn_labels)

        fp_errors = [e for e in img_errors if e.error_type != "missed"]
        if fp_errors:
            fp_boxes = np.array([e.box for e in fp_errors], dtype=np.float32)
            fp_cls = np.array([e.class_id for e in fp_errors], dtype=int)
            fp_scores = np.array(
                [e.score if e.score is not None else 0.0 for e in fp_errors],
                dtype=np.float32,
            )
            fp_dets = sv.Detections(xyxy=fp_boxes, class_id=fp_cls, confidence=fp_scores)
            fp_labels = [
                (f"{e.error_type}: "
                 f"{shorten_label(class_names.get(e.class_id, str(e.class_id)))} "
                 f"{e.score:.2f}")
                if e.score is not None else e.error_type
                for e in fp_errors
            ]
            img = box_annotator.annotate(scene=img, detections=fp_dets)
            img = label_annotator.annotate(scene=img, detections=fp_dets, labels=fp_labels)

        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Image {img_idx} ({err_count} errors)", fontsize=9)

    for ax_idx in range(n, len(axes)):
        axes[ax_idx].axis("off")

    fig.suptitle("Hardest Images (ranked by error count)", fontsize=12)

    _save(fig, save_path)
    return fig


def plot_threshold_curves(
    threshold_results: dict[int, dict],
    class_names: dict[int, str],
    save_path: str | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> matplotlib.figure.Figure:
    """F1 vs confidence threshold curves per class with optimal points.

    No numbered counterpart in ``CHART_FILENAMES`` — threshold sweep is
    unique to this CLI. Filename is chosen by the caller.
    """
    fig, ax = new_figure(figsize=figsize)

    colors = plt.cm.Set1(np.linspace(0, 1, max(len(threshold_results), 1)))

    for idx, (cls_id, result) in enumerate(sorted(threshold_results.items())):
        curve = result.get("f1_curve", [])
        if not curve:
            continue

        thresholds = [pt[0] for pt in curve]
        f1_values = [pt[1] for pt in curve]
        name = shorten_label(class_names.get(cls_id, f"class_{cls_id}"))
        best_t = result.get("best_threshold", 0)
        best_f1 = result.get("best_f1", 0)

        color = colors[idx % len(colors)]
        ax.plot(
            thresholds, f1_values,
            label=f"{name} (best={best_t:.2f}, F1={best_f1:.3f})",
            color=color, linewidth=1.5,
        )
        ax.axvline(x=best_t, color=color, linestyle="--", alpha=0.4)

    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score vs Confidence Threshold")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)

    _save(fig, save_path)
    return fig

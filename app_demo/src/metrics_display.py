"""Metrics display component for demo tabs.

Provides per-class precision/recall/F1 computation and display utilities
using supervision-based metrics from core/p08_evaluation/sv_metrics.py.
"""

import sys
from io import StringIO
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.p08_evaluation.sv_metrics import compute_map


class MetricsDisplay:
    """Metrics computation and display utilities for detection results.

    Provides static methods to compute metrics from detections/ground-truth pairs
    and display them as ASCII tables or matplotlib charts.
    """

    @staticmethod
    def compute_from_detections(
        predictions: List[Dict],
        ground_truths: List[Dict],
        class_names: Dict[int, str],
        iou_threshold: float = 0.5,
    ) -> Dict:
        """Compute per-class precision/recall/F1 from detection results.

        Args:
            predictions: List of prediction dicts per image with keys
                ``boxes`` (N, 4), ``scores`` (N,), ``labels`` (N,).
            ground_truths: List of ground-truth dicts per image with keys
                ``boxes`` (M, 4), ``labels`` (M,).
            class_names: Mapping from class_id to display name.
            iou_threshold: IoU threshold for matching predictions to GT.

        Returns:
            Dictionary with:
                - ``mAP``: Mean AP across classes (float)
                - ``per_class``: Dict[class_name, Dict] with keys:
                    - ``ap``: Average Precision (float)
                    - ``precision``: Precision at best F1 (float)
                    - ``recall``: Recall at best F1 (float)
                    - ``f1``: F1 score (float)
                - ``num_classes``: Total number of classes (int)
        """
        num_classes = len(class_names)
        metrics = compute_map(
            predictions=predictions,
            ground_truths=ground_truths,
            iou_threshold=iou_threshold,
            num_classes=num_classes,
        )

        # Reorganize by class name instead of class ID
        per_class = {}
        for cls_id, name in class_names.items():
            ap = metrics["per_class_ap"].get(cls_id, 0.0)
            prec = metrics["precision"].get(cls_id, 0.0)
            rec = metrics["recall"].get(cls_id, 0.0)
            f1 = (
                2 * prec * rec / (prec + rec)
                if (prec + rec) > 0
                else 0.0
            )
            per_class[name] = {
                "ap": ap,
                "precision": prec,
                "recall": rec,
                "f1": f1,
            }

        return {
            "mAP": metrics["mAP"],
            "per_class": per_class,
            "num_classes": num_classes,
        }

    @staticmethod
    def create_metrics_table(metrics: Dict, title: str = "Detection Metrics") -> str:
        """Create an ASCII table formatted metrics display.

        Args:
            metrics: Metrics dict from ``compute_from_detections()``.
            title: Table title.

        Returns:
            String containing ASCII table with per-class metrics.
        """
        output = StringIO()

        # Header
        output.write(f"\n{title}\n")
        output.write("=" * 80 + "\n")
        output.write(f"{'Class':<25} {'AP':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}\n")
        output.write("-" * 80 + "\n")

        # Per-class rows
        for class_name, cls_metrics in metrics["per_class"].items():
            output.write(
                f"{class_name:<25} "
                f"{cls_metrics['ap']:>10.4f} "
                f"{cls_metrics['precision']:>10.4f} "
                f"{cls_metrics['recall']:>10.4f} "
                f"{cls_metrics['f1']:>10.4f}\n"
            )

        # Mean AP row
        output.write("-" * 80 + "\n")
        output.write(f"{'Mean AP':<25} {metrics['mAP']:>10.4f}\n")
        output.write("=" * 80 + "\n")

        return output.getvalue()

    @staticmethod
    def create_metrics_chart(metrics: Dict, figsize: tuple = (10, 6)) -> plt.Figure:
        """Create a matplotlib bar chart of per-class metrics.

        Args:
            metrics: Metrics dict from ``compute_from_detections()``.
            figsize: Figure size (width, height) in inches.

        Returns:
            matplotlib Figure object with grouped bar chart.
        """
        class_names = list(metrics["per_class"].keys())
        n_classes = len(class_names)

        # Extract metric values
        ap_vals = [metrics["per_class"][name]["ap"] for name in class_names]
        prec_vals = [metrics["per_class"][name]["precision"] for name in class_names]
        rec_vals = [metrics["per_class"][name]["recall"] for name in class_names]
        f1_vals = [metrics["per_class"][name]["f1"] for name in class_names]

        # Create grouped bar chart
        x = np.arange(n_classes)
        width = 0.2

        fig, ax = plt.subplots(figsize=figsize)
        bars1 = ax.bar(x - 1.5 * width, ap_vals, width, label="AP", color="#1f77b4")
        bars2 = ax.bar(x - 0.5 * width, prec_vals, width, label="Precision", color="#ff7f0e")
        bars3 = ax.bar(x + 0.5 * width, rec_vals, width, label="Recall", color="#2ca02c")
        bars4 = ax.bar(x + 1.5 * width, f1_vals, width, label="F1", color="#d62728")

        # Formatting
        ax.set_xlabel("Class")
        ax.set_ylabel("Score")
        ax.set_title("Per-Class Detection Metrics")
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        return fig

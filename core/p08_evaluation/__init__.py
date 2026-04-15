"""Model evaluation pipeline: metrics, error analysis, and visualization."""

from core.p08_evaluation.evaluator import ModelEvaluator
from core.p08_evaluation.sv_metrics import (
    compute_map,
    compute_map_coco,
    compute_precision_recall,
    compute_confusion_matrix,
)

__all__ = [
    "ModelEvaluator",
    "compute_map",
    "compute_map_coco",
    "compute_precision_recall",
    "compute_confusion_matrix",
]

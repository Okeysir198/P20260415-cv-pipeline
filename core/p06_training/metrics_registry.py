"""Metrics registry: maps output_format -> metrics computation function.

New frameworks can register their own metrics without editing trainer.py:

    from core.p06_training.metrics_registry import register_metrics

    @register_metrics("my_format")
    def my_metrics(predictions, targets, **kwargs):
        return {"val/mAP50": ...}
"""

from typing import Callable, Dict, Any, List

import numpy as np
import torch

from core.p08_evaluation.sv_metrics import compute_map

METRICS_REGISTRY: Dict[str, Callable] = {}


def register_metrics(output_format: str) -> Callable:
    """Decorator to register a metrics function for a given output_format.

    Args:
        output_format: The model output format this function handles
            (e.g. ``"yolox"``, ``"detr"``, ``"classification"``).

    Returns:
        Decorator that stores the function in :data:`METRICS_REGISTRY`.
    """
    def decorator(fn: Callable) -> Callable:
        METRICS_REGISTRY[output_format] = fn
        return fn
    return decorator


def compute_metrics(
    output_format: str,
    predictions: List,
    targets: List,
    **kwargs: Any,
) -> Dict[str, float]:
    """Dispatch to the registered metrics function for this output_format.

    Args:
        output_format: The model output format key (e.g. ``"yolox"``).
        predictions: Collected predictions from the validation loop.
            Detection: list of dicts with ``boxes``, ``scores``, ``labels``.
            Classification: list of (N, C) logit tensors.
        targets: Collected ground truths from the validation loop.
            Detection: list of dicts with ``boxes``, ``labels``.
            Classification: list of scalar label tensors.
        **kwargs: Extra keyword arguments forwarded to the registered function
            (e.g. ``num_classes`` for detection, ``model_cfg`` for config access).

    Returns:
        Dictionary of metric name → float value. Keys must not change across
        runs because loggers and checkpoint savers depend on them.

    Raises:
        ValueError: If no function is registered for ``output_format``.
    """
    if output_format not in METRICS_REGISTRY:
        raise ValueError(
            f"No metrics registered for output_format='{output_format}'. "
            f"Register with @register_metrics('{output_format}'). "
            f"Available: {sorted(METRICS_REGISTRY.keys())}"
        )
    return METRICS_REGISTRY[output_format](predictions, targets, **kwargs)


# ---------------------------------------------------------------------------
# Built-in metrics functions
# ---------------------------------------------------------------------------


def _detection_metrics(
    all_predictions: List[Dict],
    all_ground_truths: List[Dict],
    num_classes: int = 2,
    **kwargs: Any,
) -> Dict[str, float]:
    """Compute COCO-style mAP@0.5 for detection models.

    Args:
        all_predictions: List of prediction dicts per image with keys
            ``boxes`` (N, 4), ``scores`` (N,), ``labels`` (N,).
        all_ground_truths: List of ground-truth dicts per image with keys
            ``boxes`` (M, 4), ``labels`` (M,).
        num_classes: Number of object classes.

    Returns:
        Dictionary with ``val/mAP50``, ``val/AP50_cls{id}``,
        ``val/precision``, ``val/recall``.
    """
    if not all_predictions:
        return {"val/mAP50": 0.0}

    map_results = compute_map(
        all_predictions,
        all_ground_truths,
        iou_threshold=0.5,
        num_classes=num_classes,
    )
    metrics: Dict[str, float] = {"val/mAP50": map_results["mAP"]}
    for cls_id, ap in map_results.get("per_class_ap", {}).items():
        metrics[f"val/AP50_cls{cls_id}"] = ap
    prec_vals = list(map_results.get("precision", {}).values())
    rec_vals = list(map_results.get("recall", {}).values())
    if prec_vals:
        metrics["val/precision"] = float(np.mean(prec_vals))
    if rec_vals:
        metrics["val/recall"] = float(np.mean(rec_vals))
    return metrics


METRICS_REGISTRY["yolox"] = _detection_metrics
METRICS_REGISTRY["detr"] = _detection_metrics


@register_metrics("classification")
def _classification_metrics(
    predictions: List,
    targets: List,
    **_kwargs: Any,
) -> Dict[str, float]:
    """Classification metrics: top-1 accuracy and top-5 accuracy.

    Args:
        predictions: List of (B, C) logit tensors collected during validation.
        targets: List of scalar class-id tensors.

    Returns:
        Dictionary with ``val/accuracy`` and optionally ``val/top5_accuracy``.
    """
    if not predictions:
        return {"val/accuracy": 0.0}

    all_preds_cat = torch.cat(predictions)
    all_gts_cat = torch.stack(targets)

    pred_classes = all_preds_cat.argmax(dim=1)
    correct = (pred_classes == all_gts_cat).float()
    metrics: Dict[str, float] = {"val/accuracy": correct.mean().item()}

    num_classes = all_preds_cat.shape[1]
    if num_classes >= 5:
        top5_preds = all_preds_cat.topk(5, dim=1).indices
        top5_correct = (top5_preds == all_gts_cat.unsqueeze(1)).any(dim=1).float()
        metrics["val/top5_accuracy"] = top5_correct.mean().item()

    return metrics


@register_metrics("segmentation")
def _segmentation_metrics(
    predictions: List,
    targets: List,
    **kwargs: Any,
) -> Dict[str, float]:
    """Segmentation metrics: mean IoU.

    Args:
        predictions: List of (H, W) predicted class-id tensors or (C, H, W) logits.
        targets: List of (H, W) integer mask tensors.

    Returns:
        Dictionary with ``val/mIoU``.
    """
    if not predictions:
        return {"val/mIoU": 0.0}

    num_classes = kwargs.get("num_classes", 2)
    intersection_sum = np.zeros(num_classes)
    union_sum = np.zeros(num_classes)

    for pred, target in zip(predictions, targets):
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()

        for cls_id in range(num_classes):
            pred_mask = pred == cls_id
            gt_mask = target == cls_id
            intersection_sum[cls_id] += np.logical_and(pred_mask, gt_mask).sum()
            union_sum[cls_id] += np.logical_or(pred_mask, gt_mask).sum()

    iou_per_class = np.where(
        union_sum > 0,
        intersection_sum / (union_sum + 1e-10),
        0.0,
    )
    metrics: Dict[str, float] = {"val/mIoU": float(np.mean(iou_per_class))}
    for cls_id in range(num_classes):
        metrics[f"val/IoU_cls{cls_id}"] = float(iou_per_class[cls_id])
    return metrics

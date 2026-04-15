"""Postprocessor registry for detection model output decoding.

Each model architecture registers a postprocessor that converts raw model
outputs into filtered detections (boxes, scores, labels).  The dispatcher
:func:`postprocess` first checks whether the model exposes its own
``postprocess()`` method (highest priority — used by HF models), then falls
back to the registered function for ``output_format``.

Output contract
---------------
Every postprocessor (whether registered here or implemented on the model)
must return ``List[Dict[str, np.ndarray]]`` with exactly three keys:

* ``"boxes"``  — ``(N, 4)`` float32 xyxy coordinates.
* ``"scores"`` — ``(N,)`` float32 confidence scores.
* ``"labels"`` — ``(N,)`` int64 class indices.

Example::

    from core.p06_training.postprocess import postprocess, POSTPROCESSOR_REGISTRY

    results = postprocess(
        output_format="yolox",
        model=model,
        predictions=outputs,
        conf_threshold=0.5,
        nms_threshold=0.45,
        target_sizes=None,
    )
    # results[0]["boxes"]  → np.ndarray (N, 4) float32 xyxy
    # results[0]["scores"] → np.ndarray (N,)  float32
    # results[0]["labels"] → np.ndarray (N,)  int64
"""

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torchvision.ops import nms

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

POSTPROCESSOR_REGISTRY: Dict[str, Callable] = {}


def register_postprocessor(output_format: str):
    """Decorator that registers a postprocessor function.

    Args:
        output_format: Output format name (e.g. ``"yolox"``).

    Returns:
        Decorator that stores *fn* in :data:`POSTPROCESSOR_REGISTRY`.
    """

    def decorator(fn: Callable) -> Callable:
        POSTPROCESSOR_REGISTRY[output_format] = fn
        return fn

    return decorator


def postprocess(
    output_format: str,
    model: Any,
    predictions: Any,
    conf_threshold: float = 0.5,
    nms_threshold: float = 0.45,
    target_sizes: Optional[Any] = None,
) -> List[Dict[str, np.ndarray]]:
    """Dispatch postprocessing based on ``output_format``.

    Priority:
    1. ``model.postprocess()`` — HF / custom models that own their decode.
    2. :data:`POSTPROCESSOR_REGISTRY` lookup by ``output_format``.

    Args:
        output_format: String key (e.g. ``"yolox"``, ``"detr"``).
        model: The model that produced *predictions*.  Inspected for a
            ``postprocess`` method.  Pass ``None`` to skip step 1.
        predictions: Raw model output (tensor or numpy array).
        conf_threshold: Minimum score to keep a detection.
        nms_threshold: IoU threshold for non-maximum suppression.
        target_sizes: Optional tensor / list of ``[H, W]`` per image, used
            by HF models for coordinate rescaling.

    Returns:
        List of per-image result dicts with keys ``"boxes"``, ``"scores"``,
        ``"labels"`` (all numpy arrays).

    Raises:
        ValueError: If ``output_format`` is not registered and the model has
            no ``postprocess`` method.
    """
    base_model = model.module if hasattr(model, "module") else model
    if base_model is not None and hasattr(base_model, "postprocess"):
        return base_model.postprocess(predictions, conf_threshold, target_sizes)

    if output_format not in POSTPROCESSOR_REGISTRY:
        available = sorted(POSTPROCESSOR_REGISTRY.keys())
        raise ValueError(
            f"No postprocessor registered for '{output_format}'. "
            f"Available: {available}"
        )

    return POSTPROCESSOR_REGISTRY[output_format](
        predictions, conf_threshold, nms_threshold, target_sizes
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_result() -> Dict[str, np.ndarray]:
    """Return an empty detection result dict."""
    return {
        "boxes": np.empty((0, 4), dtype=np.float32),
        "scores": np.empty((0,), dtype=np.float32),
        "labels": np.empty((0,), dtype=np.int64),
    }


# ---------------------------------------------------------------------------
# YOLOX postprocessor
# ---------------------------------------------------------------------------


@register_postprocessor("yolox")
def _postprocess_yolox(
    predictions: torch.Tensor,
    conf_threshold: float = 0.5,
    nms_threshold: float = 0.45,
    _target_sizes: Optional[Any] = None,
) -> List[Dict[str, np.ndarray]]:
    """Decode raw YOLOX outputs into a list of result dicts.

    Args:
        predictions: ``(B, N, 5+C)`` tensor — ``[cx, cy, w, h, obj_logit,
            cls_logits…]``.
        conf_threshold: Minimum ``objectness * class_prob`` to keep.
        nms_threshold: IoU threshold for NMS.
        target_sizes: Unused for YOLOX (boxes are already in pixel space).
            Accepted for a uniform registry signature.

    Returns:
        List of B dicts with ``"boxes"`` (xyxy float32), ``"scores"``
        (float32), ``"labels"`` (int64).
    """
    results: List[Dict[str, np.ndarray]] = []
    batch_size = predictions.shape[0]

    for b in range(batch_size):
        pred = predictions[b]  # (N, 5+C)

        obj_conf = pred[:, 4].sigmoid()
        cls_probs = pred[:, 5:].sigmoid()
        cls_conf, cls_id = cls_probs.max(dim=1)
        scores = obj_conf * cls_conf

        mask = scores >= conf_threshold
        if not mask.any():
            results.append(_empty_result())
            continue

        pred_filt = pred[mask]
        scores_filt = scores[mask]
        cls_id_filt = cls_id[mask]

        # cxcywh → xyxy
        cx, cy, w, h = (
            pred_filt[:, 0], pred_filt[:, 1],
            pred_filt[:, 2], pred_filt[:, 3],
        )
        boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)

        # Per-class NMS
        keep_indices: List[torch.Tensor] = []
        for c in cls_id_filt.unique():
            c_mask = cls_id_filt == c
            c_boxes = boxes[c_mask]
            c_scores = scores_filt[c_mask]
            c_indices = torch.where(c_mask)[0]
            keep = nms(c_boxes, c_scores, nms_threshold)
            keep_indices.append(c_indices[keep])

        if not keep_indices:
            results.append(_empty_result())
            continue

        keep = torch.cat(keep_indices)
        results.append({
            "boxes": boxes[keep].cpu().numpy().astype(np.float32),
            "scores": scores_filt[keep].cpu().numpy().astype(np.float32),
            "labels": cls_id_filt[keep].cpu().numpy().astype(np.int64),
        })

    return results


# ---------------------------------------------------------------------------
# DETR postprocessor
# ---------------------------------------------------------------------------


@register_postprocessor("segmentation")
def _postprocess_segmentation(
    predictions: torch.Tensor,
    _conf_threshold: float = 0.5,
    _nms_threshold: float = 0.45,
    _target_sizes: Optional[Any] = None,
) -> List[Dict[str, np.ndarray]]:
    """Decode segmentation logits into per-pixel class maps.

    Args:
        predictions: ``(B, C, H, W)`` logits tensor.
        _conf_threshold: Unused (no confidence filtering for segmentation).
        _nms_threshold: Unused (no NMS for segmentation).
        _target_sizes: Unused.

    Returns:
        List of B dicts with ``"class_map"`` — ``(H, W)`` int64 array.
    """
    class_maps = predictions.argmax(dim=1)  # (B, H, W)
    results: List[Dict[str, np.ndarray]] = []
    for i in range(class_maps.shape[0]):
        results.append({"class_map": class_maps[i].cpu().numpy().astype(np.int64)})
    return results


@register_postprocessor("detr")
def _postprocess_detr(
    predictions: Any,
    _conf_threshold: float = 0.5,
    _nms_threshold: float = 0.45,
    _target_sizes: Optional[Any] = None,
) -> List[Dict[str, np.ndarray]]:
    """Decode HF DETR-family outputs into a list of result dicts.

    This is a fallback path for DETR-style models that do NOT implement their
    own ``model.postprocess()`` method.  In practice, :class:`HFDetectionModel`
    exposes ``postprocess()`` so the registry is not reached for those models.
    This registration ensures an explicit, informative error is never raised
    for ``output_format="detr"`` in future custom models.

    Args:
        predictions: Raw model output (implementation-defined).
        conf_threshold: Minimum score to keep.
        nms_threshold: IoU threshold for NMS (informational — DETR outputs
            are already NMS-free; kept for signature uniformity).
        target_sizes: Passed through to model-level postprocessors when
            called indirectly.

    Returns:
        List of result dicts (empty when predictions cannot be decoded here).
    """
    batch_size = (
        predictions.shape[0] if hasattr(predictions, "shape") else len(predictions)
    )
    logger.warning(
        "DETR postprocessor fallback: model has no postprocess() method. "
        "Returning empty detections for %d images.",
        batch_size,
    )
    return [_empty_result() for _ in range(batch_size)]

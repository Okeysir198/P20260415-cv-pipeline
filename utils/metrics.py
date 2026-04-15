"""Box utility functions, IoU, and NMS.

Coordinate conversions, IoU computation, and non-maximum suppression.
NumPy-based by default — torch variants provided as optional fallbacks.

NOTE: Evaluation metrics (mAP, precision-recall, confusion matrix) have
moved to utils/sv_metrics.py which uses the supervision library.
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Box format conversions
# ---------------------------------------------------------------------------


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2].

    Args:
        boxes: Array of shape (N, 4) in [cx, cy, w, h] format.

    Returns:
        Array of shape (N, 4) in [x1, y1, x2, y2] format.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim == 1:
        boxes = boxes[np.newaxis, :]
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float64)

    result = np.empty_like(boxes)
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    return result


# Alias: cxcywh_to_xyxy is the same conversion as xywh_to_xyxy
# (both interpret input as center-x, center-y, width, height).
cxcywh_to_xyxy = xywh_to_xyxy


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """Convert boxes from [x1, y1, x2, y2] to [cx, cy, w, h].

    Args:
        boxes: Array of shape (N, 4) in [x1, y1, x2, y2] format.

    Returns:
        Array of shape (N, 4) in [cx, cy, w, h] format.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim == 1:
        boxes = boxes[np.newaxis, :]
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float64)

    result = np.empty_like(boxes)
    result[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # cx
    result[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # cy
    result[:, 2] = boxes[:, 2] - boxes[:, 0]         # w
    result[:, 3] = boxes[:, 3] - boxes[:, 1]         # h
    return result


# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------


def compute_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of boxes.

    Args:
        boxes1: Array of shape (N, 4) in [x1, y1, x2, y2] format.
        boxes2: Array of shape (M, 4) in [x1, y1, x2, y2] format.

    Returns:
        IoU matrix of shape (N, M).
    """
    boxes1 = np.asarray(boxes1, dtype=np.float64)
    boxes2 = np.asarray(boxes2, dtype=np.float64)

    if boxes1.ndim == 1:
        boxes1 = boxes1[np.newaxis, :]
    if boxes2.ndim == 1:
        boxes2 = boxes2[np.newaxis, :]

    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float64)

    # Intersection coordinates
    x1 = np.maximum(boxes1[:, 0][:, np.newaxis], boxes2[:, 0][np.newaxis, :])
    y1 = np.maximum(boxes1[:, 1][:, np.newaxis], boxes2[:, 1][np.newaxis, :])
    x2 = np.minimum(boxes1[:, 2][:, np.newaxis], boxes2[:, 2][np.newaxis, :])
    y2 = np.minimum(boxes1[:, 3][:, np.newaxis], boxes2[:, 3][np.newaxis, :])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union = area1[:, np.newaxis] + area2[np.newaxis, :] - intersection

    # Avoid division by zero
    iou = np.where(union > 0, intersection / union, 0.0)
    return iou


# ---------------------------------------------------------------------------
# Non-maximum suppression
# ---------------------------------------------------------------------------


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Non-maximum suppression in pure NumPy.

    Greedy NMS that iteratively selects the highest-scoring box and
    removes overlapping boxes exceeding the IoU threshold.

    Args:
        boxes: (N, 4) array of [x1, y1, x2, y2].
        scores: (N,) confidence scores.
        iou_threshold: IoU threshold for suppression.

    Returns:
        Array of kept indices (int64).
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-16)

        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    return np.array(keep, dtype=np.int64)


def nms_torch(boxes: "torch.Tensor", scores: "torch.Tensor",
              iou_threshold: float) -> "torch.Tensor":
    """Non-maximum suppression using PyTorch tensors.

    Greedy NMS fallback when ``torchvision.ops.nms`` is not available.

    Args:
        boxes: (N, 4) tensor in [x1, y1, x2, y2] format.
        scores: (N,) tensor of confidence scores.
        iou_threshold: IoU threshold for suppression.

    Returns:
        Indices of kept boxes (long tensor on the same device as *boxes*).

    Raises:
        RuntimeError: If PyTorch is not installed.
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)

    keep: list[int] = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break

        remaining = order[1:]
        xx1 = torch.maximum(x1[i].unsqueeze(0), x1[remaining])
        yy1 = torch.maximum(y1[i].unsqueeze(0), y1[remaining])
        xx2 = torch.minimum(x2[i].unsqueeze(0), x2[remaining])
        yy2 = torch.minimum(y2[i].unsqueeze(0), y2[remaining])

        inter = torch.clamp(xx2 - xx1, min=0) * torch.clamp(yy2 - yy1, min=0)
        union = areas[i] + areas[remaining] - inter
        iou = inter / union.clamp(min=1e-6)

        mask = iou <= iou_threshold
        order = remaining[mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)

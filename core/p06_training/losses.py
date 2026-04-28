"""Loss functions for object detection training.

Provides:
- DetectionLoss: Abstract base class for detection losses.
- FocalLoss: Classification loss with class imbalance handling.
- IoULoss: Bounding box regression loss with IoU variants.
- YOLOXLoss: Combined loss with SimOTA dynamic label assignment.
- Loss registry: ``LOSS_REGISTRY``, ``@register_loss``, ``build_loss``.
"""

import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from loguru import logger
from utils.registry import Registry  # noqa: E402

# ---------------------------------------------------------------------------
# Loss registry
# ---------------------------------------------------------------------------

LOSS_REGISTRY: dict[str, Callable] = {}

# Maps model arch names to their default loss type.
_ARCH_LOSS_MAP: dict[str, str] = {}

_loss_registry = Registry(entity_name="loss", registry=LOSS_REGISTRY)


def register_loss(name: str, arch_aliases: list | None = None):
    """Decorator that registers a loss builder function.

    Args:
        name: Loss type name used in config (e.g. ``"yolox"``).
        arch_aliases: Optional list of model arch names that default
            to this loss (e.g. ``["yolox-m", "yolox-s"]``).

    Returns:
        Decorator that stores *cls* in :data:`LOSS_REGISTRY`.
    """
    base_decorator = _loss_registry.register(name)

    def wrapper(cls):
        base_decorator(cls)
        if arch_aliases:
            for alias in arch_aliases:
                _ARCH_LOSS_MAP[alias] = name
        return cls

    return wrapper


def build_loss(config: dict) -> nn.Module:
    """Build a detection loss from config.

    Looks up ``config["loss"]["type"]`` first.  If absent, infers from
    ``config["model"]["arch"]`` via ``_ARCH_LOSS_MAP``.  Falls back to
    ``"yolox"`` if nothing matches.

    Args:
        config: Full training config dict.

    Returns:
        Instantiated loss module.

    Raises:
        ValueError: If the loss type is not registered.
    """
    loss_cfg = config.get("loss", {})
    loss_type = loss_cfg.get("type")

    if loss_type is None:
        arch = config.get("model", {}).get("arch", "yolox-m").lower()
        loss_type = _ARCH_LOSS_MAP.get(arch)
        if loss_type is None:
            # Pattern-match Paddle archs (picodet-*, ppyoloe-*, ppclas-*,
            # ppseg-*, pp-tinypose-*) to the paddle passthrough loss — Paddle
            # models compute their own loss internally, mirroring HF detection.
            paddle_prefixes = ("picodet-", "ppyoloe-", "ppclas-", "ppseg-", "pp-tinypose-")
            if arch.startswith(paddle_prefixes):
                loss_type = "paddle-passthrough"
            else:
                loss_type = "yolox"

    if loss_type not in LOSS_REGISTRY:
        available = sorted(LOSS_REGISTRY.keys())
        raise ValueError(
            f"Unknown loss type '{loss_type}'. Available: {available}"
        )

    logger.info("Building loss: type=%s", loss_type)
    return LOSS_REGISTRY[loss_type](config)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class DetectionLoss(ABC, nn.Module):
    """Abstract base class for detection losses.

    Subclasses must implement ``forward`` returning ``(total_loss, loss_dict)``.
    Provides a concrete ``set_epoch`` for warmup scheduling.
    """

    def __init__(self) -> None:
        super().__init__()
        self._current_epoch: int = 0

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch for loss warmup scaling."""
        self._current_epoch = epoch

    @abstractmethod
    def forward(
        self,
        predictions: torch.Tensor,
        targets: list,
        grids: list | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute loss.

        Args:
            predictions: Model output tensor.
            targets: List of ground truth tensors.
            grids: Optional grid tensors (model-specific).

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict has
            string keys and detached tensor values for logging.
        """


def _reduce(tensor: torch.Tensor, reduction: str) -> torch.Tensor:
    """Apply reduction to a loss tensor.

    Args:
        tensor: Per-element loss tensor.
        reduction: "none", "mean", or "sum".

    Returns:
        Reduced loss tensor.
    """
    if reduction == "mean":
        return tensor.mean()
    if reduction == "sum":
        return tensor.sum()
    return tensor


class FocalLoss(nn.Module):
    """Focal loss for classification to handle class imbalance.

    Focal loss down-weights well-classified examples and focuses on
    hard, misclassified examples.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor for the rare class. Default: 0.25.
        gamma: Focusing parameter. Higher values focus more on hard examples.
            Default: 2.0.
        reduction: Reduction mode — "none", "mean", or "sum". Default: "mean".
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            pred: Predicted logits of shape (N, C) or (N,).
            target: Ground truth labels of shape (N, C) or (N,).
                For multi-class, should be one-hot or soft labels.

        Returns:
            Focal loss scalar (if reduced) or per-element tensor.
        """
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p_t = torch.sigmoid(pred)
        p_t = target * p_t + (1 - target) * (1 - p_t)
        modulating_factor = (1 - p_t) ** self.gamma
        alpha_factor = target * self.alpha + (1 - target) * (1 - self.alpha)
        focal_loss = alpha_factor * modulating_factor * bce_loss
        return _reduce(focal_loss, self.reduction)


class IoULoss(nn.Module):
    """IoU-based loss for bounding box regression.

    Supports multiple IoU variants for progressively better convergence:
    - IoU: Standard Intersection over Union.
    - GIoU: Generalized IoU — penalizes non-overlapping area.
    - DIoU: Distance IoU — penalizes center distance.
    - CIoU: Complete IoU — penalizes center distance + aspect ratio.

    Args:
        variant: IoU variant — "iou", "giou", "diou", or "ciou". Default: "giou".
        reduction: Reduction mode — "none", "mean", or "sum". Default: "mean".
        eps: Small value for numerical stability. Default: 1e-7.
    """

    def __init__(
        self,
        variant: str = "giou",
        reduction: str = "mean",
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        valid_variants = ("iou", "giou", "diou", "ciou")
        if variant not in valid_variants:
            raise ValueError(f"IoU variant must be one of {valid_variants}, got '{variant}'")
        self.variant = variant
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute IoU-based loss.

        Args:
            pred: Predicted boxes (N, 4) in [x1, y1, x2, y2] format.
            target: Ground truth boxes (N, 4) in [x1, y1, x2, y2] format.

        Returns:
            Loss value (1 - IoU variant).
        """
        iou = self._compute_iou(pred, target)

        loss = 1.0 - iou
        return _reduce(loss, self.reduction)

    def _compute_iou(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute IoU variant between predicted and target boxes.

        Args:
            pred: Predicted boxes (N, 4) in xyxy format.
            target: Target boxes (N, 4) in xyxy format.

        Returns:
            IoU values of shape (N,).
        """
        # Intersection coordinates
        inter_x1 = torch.max(pred[:, 0], target[:, 0])
        inter_y1 = torch.max(pred[:, 1], target[:, 1])
        inter_x2 = torch.min(pred[:, 2], target[:, 2])
        inter_y2 = torch.min(pred[:, 3], target[:, 3])

        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

        # Union — clamp dims to ≥0 so degenerate predictions don't produce NaN
        pred_area = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
        target_area = (target[:, 2] - target[:, 0]).clamp(min=0) * (target[:, 3] - target[:, 1]).clamp(min=0)
        union_area = pred_area + target_area - inter_area + self.eps

        iou = inter_area / union_area

        if self.variant == "iou":
            return iou

        # Enclosing box
        enclose_x1 = torch.min(pred[:, 0], target[:, 0])
        enclose_y1 = torch.min(pred[:, 1], target[:, 1])
        enclose_x2 = torch.max(pred[:, 2], target[:, 2])
        enclose_y2 = torch.max(pred[:, 3], target[:, 3])

        if self.variant == "giou":
            enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1) + self.eps
            return iou - (enclose_area - union_area) / enclose_area

        # Center distances for DIoU / CIoU
        pred_cx = (pred[:, 0] + pred[:, 2]) / 2
        pred_cy = (pred[:, 1] + pred[:, 3]) / 2
        target_cx = (target[:, 0] + target[:, 2]) / 2
        target_cy = (target[:, 1] + target[:, 3]) / 2

        center_dist_sq = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
        enclose_diag_sq = (
            (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + self.eps
        )

        if self.variant == "diou":
            return iou - center_dist_sq / enclose_diag_sq

        # CIoU: add aspect ratio penalty
        pred_w = (pred[:, 2] - pred[:, 0]).clamp(min=self.eps)
        pred_h = (pred[:, 3] - pred[:, 1]).clamp(min=self.eps)
        target_w = (target[:, 2] - target[:, 0]).clamp(min=self.eps)
        target_h = (target[:, 3] - target[:, 1]).clamp(min=self.eps)

        v = (4 / (torch.pi ** 2)) * (
            torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h)
        ) ** 2
        with torch.no_grad():
            alpha = v / (1 - iou + v + self.eps)

        return iou - center_dist_sq / enclose_diag_sq - alpha * v


class YOLOXLoss(DetectionLoss):
    """Combined YOLOX loss with SimOTA dynamic label assignment.

    Computes:
    - Classification loss: BCE (or Focal) for class predictions.
    - Objectness loss: BCE for objectness scores.
    - Regression loss: IoU-based loss for bounding box predictions.

    SimOTA selects the best prediction-to-GT assignments dynamically
    per image based on a cost matrix (cls_cost + reg_cost).

    Args:
        num_classes: Number of object classes.
        strides: Feature map strides for multi-scale detection. Default: [8, 16, 32].
        use_focal: Whether to use focal loss for classification. Default: False.
        iou_variant: IoU loss variant. Default: "giou".
        cls_weight: Weight for classification loss. Default: 1.0.
        obj_weight: Weight for objectness loss. Default: 1.0.
        reg_weight: Weight for regression loss. Default: 5.0.
        simota_top_k: Number of candidates per GT in SimOTA. Default: 10.
        focal_alpha: Focal loss alpha (only if use_focal=True). Default: 0.25.
        focal_gamma: Focal loss gamma (only if use_focal=True). Default: 2.0.
    """

    def __init__(
        self,
        num_classes: int = 80,
        strides: list | None = None,
        use_focal: bool = False,
        iou_variant: str = "giou",
        cls_weight: float = 1.0,
        obj_weight: float = 1.0,
        reg_weight: float = 5.0,
        simota_top_k: int = 10,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        warmup_epochs: int = 0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides or [8, 16, 32]
        self.cls_weight = cls_weight
        self.obj_weight = obj_weight
        self.reg_weight = reg_weight
        self.simota_top_k = simota_top_k
        self.warmup_epochs = warmup_epochs

        if use_focal:
            self.cls_loss_fn = FocalLoss(
                alpha=focal_alpha, gamma=focal_gamma, reduction="none"
            )
        else:
            self.cls_loss_fn = None  # Use BCE directly

        self.iou_loss_fn = IoULoss(variant=iou_variant, reduction="none")
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self._anchor_cache: dict = {}  # (num_anchors, device_str, dtype) -> (centers, strides)

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch for loss warmup scaling."""
        self._current_epoch = epoch

    def forward(
        self,
        predictions: torch.Tensor,
        targets: list,
        grids: list | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute YOLOX loss with SimOTA assignment.

        Args:
            predictions: Decoded predictions of shape (B, N, 5 + num_classes)
                where N is total anchors across all scales.
                Format per anchor: [cx, cy, w, h, obj, cls_0, ..., cls_C].
            targets: List of B tensors, each (M_i, 5) with [cls, cx, cy, w, h]
                where M_i is the number of GTs in image i.
                Coordinates are in input image scale (pixels).
            grids: Optional list of grid tensors per scale for center priors.
                If None, assignment uses raw prediction positions.

        Returns:
            Tuple of:
                - total_loss: Scalar loss tensor.
                - loss_dict: Dictionary with individual loss components for logging:
                    {"cls_loss", "obj_loss", "reg_loss", "num_fg"}.
        """
        batch_size = predictions.shape[0]
        device = predictions.device
        dtype = predictions.dtype

        total_cls_loss = torch.tensor(0.0, device=device, dtype=dtype)
        total_obj_loss = torch.tensor(0.0, device=device, dtype=dtype)
        total_reg_loss = torch.tensor(0.0, device=device, dtype=dtype)
        total_num_fg = 0
        total_num_anchors = 0

        for b in range(batch_size):
            pred = predictions[b]  # (N, 5 + C)
            gt = targets[b]  # (M, 5) [cls, cx, cy, w, h]
            num_anchors = pred.shape[0]
            total_num_anchors += num_anchors

            # Objectness target: all zeros initially
            obj_target = torch.zeros(pred.shape[0], 1, device=device, dtype=dtype)

            if gt.shape[0] == 0:
                # No GT in this image — only objectness loss
                obj_loss = self.bce_loss(pred[:, 4:5], obj_target).sum()
                total_obj_loss = total_obj_loss + obj_loss
                continue

            # Decode prediction boxes to xyxy
            pred_boxes = self._cxcywh_to_xyxy(pred[:, :4])  # (N, 4)
            gt_boxes = self._cxcywh_to_xyxy(gt[:, 1:5])  # (M, 4)
            gt_classes = gt[:, 0].long()  # (M,)

            # SimOTA assignment (with center-prior geometry filter)
            fg_mask, matched_gt_inds, matched_ious = self._simota_assign(
                pred, pred_boxes, gt_boxes, gt_classes, gt[:, 1:3]
            )

            num_fg = fg_mask.sum().item()
            total_num_fg += num_fg

            if num_fg == 0:
                obj_loss = self.bce_loss(pred[:, 4:5], obj_target).sum()
                total_obj_loss = total_obj_loss + obj_loss
                continue

            # Matched predictions and targets
            fg_pred_boxes = pred_boxes[fg_mask]  # (num_fg, 4)
            fg_pred_cls = pred[fg_mask, 5:]  # (num_fg, C)
            fg_gt_boxes = gt_boxes[matched_gt_inds]  # (num_fg, 4)
            fg_gt_classes = gt_classes[matched_gt_inds]  # (num_fg,)

            # Classification loss
            cls_target = F.one_hot(fg_gt_classes, self.num_classes).float()
            if self.cls_loss_fn is not None:
                cls_loss = self.cls_loss_fn(fg_pred_cls, cls_target).sum()
            else:
                cls_loss = self.bce_loss(
                    fg_pred_cls, cls_target
                ).sum()

            # Objectness loss
            obj_target[fg_mask] = 1.0
            obj_loss = self.bce_loss(pred[:, 4:5], obj_target).sum()

            # Regression loss (IoU-based); nan_to_num guards against fp16 overflow edge cases
            reg_loss = self.iou_loss_fn(fg_pred_boxes, fg_gt_boxes)
            reg_loss = torch.nan_to_num(reg_loss, nan=0.0, posinf=0.0, neginf=0.0).sum()

            total_cls_loss = total_cls_loss + cls_loss
            total_obj_loss = total_obj_loss + obj_loss
            total_reg_loss = total_reg_loss + reg_loss

        # Normalize cls and reg by num_fg, obj by total_num_anchors
        num_fg_safe = max(total_num_fg, 1)
        num_anchors_safe = max(total_num_anchors, 1)

        cls_loss_norm = total_cls_loss / num_fg_safe
        obj_loss_norm = total_obj_loss / num_anchors_safe
        reg_loss_norm = total_reg_loss / num_fg_safe

        # Loss warmup: scale cls and reg losses during early epochs
        if self.warmup_epochs > 0 and self._current_epoch < self.warmup_epochs:
            warmup_factor = (self._current_epoch + 1) / self.warmup_epochs
            cls_loss_norm = cls_loss_norm * warmup_factor
            reg_loss_norm = reg_loss_norm * warmup_factor

        total_loss = (
            self.cls_weight * cls_loss_norm
            + self.obj_weight * obj_loss_norm
            + self.reg_weight * reg_loss_norm
        )

        loss_dict = {
            "cls_loss": cls_loss_norm.detach(),
            "obj_loss": obj_loss_norm.detach(),
            "reg_loss": reg_loss_norm.detach(),
            "num_fg": total_num_fg / batch_size,
        }

        return total_loss, loss_dict

    def _build_anchor_metadata(
        self,
        num_anchors: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build per-anchor cell-center pixel coords + stride.

        Assumes a square input shape and anchor ordering stride-ascending
        (matches ``YOLOXModel.forward``'s iteration over ``self.strides``).

        Args:
            num_anchors: Total anchor count across all scales, ``N``.
            device: Target device.
            dtype: Target dtype (always float32 internally; cast on return).

        Returns:
            Tuple of ``(anchor_centers, anchor_strides)`` with shapes
            ``(N, 2)`` and ``(N,)`` — pixel-space cell centers and stride
            per anchor.
        """
        key = (num_anchors, str(device), dtype)
        cached = self._anchor_cache.get(key)
        if cached is not None:
            return cached

        # Solve square input size S from: N = sum_i (S/stride_i)^2
        sum_inv_sq = sum(1.0 / (s * s) for s in self.strides)
        s_float = (num_anchors / sum_inv_sq) ** 0.5
        s_int = int(round(s_float))
        if abs(s_int - s_float) > 1e-3 or any(s_int % s != 0 for s in self.strides):
            raise RuntimeError(
                f"Cannot infer square input size for num_anchors={num_anchors} "
                f"with strides={self.strides} (solved S={s_float})"
            )

        centers_parts, strides_parts = [], []
        for stride in self.strides:
            h = w = s_int // stride
            gy, gx = torch.meshgrid(
                torch.arange(h, device=device, dtype=torch.float32),
                torch.arange(w, device=device, dtype=torch.float32),
                indexing="ij",
            )
            cx = ((gx + 0.5) * stride).reshape(-1)
            cy = ((gy + 0.5) * stride).reshape(-1)
            centers_parts.append(torch.stack([cx, cy], dim=-1))
            strides_parts.append(
                torch.full((h * w,), float(stride), device=device, dtype=torch.float32)
            )

        anchor_centers = torch.cat(centers_parts, dim=0).to(dtype)  # (N, 2)
        anchor_strides = torch.cat(strides_parts, dim=0).to(dtype)  # (N,)
        self._anchor_cache[key] = (anchor_centers, anchor_strides)
        return anchor_centers, anchor_strides

    def _simota_assign(
        self,
        pred: torch.Tensor,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_classes: torch.Tensor,
        gt_cxcy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """SimOTA dynamic label assignment with center-prior geometry filter.

        Adds the upstream Megvii YOLOXHead.get_geometry_constraint step
        (``center_radius=1.5 * stride``) so positive-anchor selection only
        considers anchors whose cell center lies near the GT center. Without
        it, topk over all ~8400 anchors lets distant anchors win positive
        slots by virtue of slightly lower raw-logit BCE cost, driving
        training to divergence.

        Args:
            pred: Full predictions (N, 5 + C).
            pred_boxes: Decoded prediction boxes (N, 4) in xyxy format.
            gt_boxes: Ground truth boxes (M, 4) in xyxy format.
            gt_classes: Ground truth class indices (M,).
            gt_cxcy: Ground truth centers (M, 2) in pixel space.

        Returns:
            Tuple of:
                - fg_mask: Boolean mask (N,) marking foreground predictions.
                - matched_gt_inds: GT index for each foreground prediction.
                - matched_ious: IoU value for each foreground match.
        """
        device = pred.device
        num_pred = pred.shape[0]
        num_gt = gt_boxes.shape[0]

        anchor_centers, anchor_strides = self._build_anchor_metadata(
            num_pred, device, torch.float32
        )

        # Center-prior geometry filter: anchor cell center within
        # center_radius * stride (L-inf) of GT center.
        center_radius = 1.5
        center_dist = (anchor_strides * center_radius).unsqueeze(0)  # (1, N)
        gt_cxcy_f = gt_cxcy.float()
        dx = (anchor_centers[:, 0].unsqueeze(0) - gt_cxcy_f[:, 0].unsqueeze(1)).abs()
        dy = (anchor_centers[:, 1].unsqueeze(0) - gt_cxcy_f[:, 1].unsqueeze(1)).abs()
        pair_geom_mask = (dx < center_dist) & (dy < center_dist)  # (M, N)
        anchor_candidates = pair_geom_mask.any(dim=0)  # (N,)
        num_cand = int(anchor_candidates.sum().item())

        empty_fg = torch.zeros(num_pred, dtype=torch.bool, device=device)
        empty_inds = torch.zeros(0, dtype=torch.long, device=device)
        empty_ious = torch.zeros(0, device=device, dtype=pred_boxes.dtype)

        if num_cand == 0:
            return empty_fg, empty_inds, empty_ious

        cand_pred_boxes = pred_boxes[anchor_candidates]           # (C, 4)
        cand_pred_cls = pred[anchor_candidates, 5:].float()       # (C, nc)
        cand_geom_mask = pair_geom_mask[:, anchor_candidates]     # (M, C)

        # Pairwise IoU over candidates only
        pair_iou = self._pairwise_iou(gt_boxes, cand_pred_boxes)  # (M, C)

        # cls cost: BCE of each candidate against each GT class
        gt_onehot = F.one_hot(gt_classes, self.num_classes).float()  # (M, nc)
        cls_cost = torch.zeros(num_gt, num_cand, device=device)
        for i in range(num_gt):
            gt_cls_expanded = gt_onehot[i].unsqueeze(0).expand(num_cand, -1)
            cls_cost[i] = F.binary_cross_entropy_with_logits(
                cand_pred_cls, gt_cls_expanded, reduction="none"
            ).sum(dim=-1)

        # Combined cost; penalise (gt, anchor) pairs that fail the per-GT
        # geometry check so topk never picks them.
        cost_matrix = (
            cls_cost
            + 3.0 * (1.0 - pair_iou)
            + 1e5 * (~cand_geom_mask).float()
        )

        # Dynamic k per GT based on IoU sum of top candidates
        top_k = min(self.simota_top_k, num_cand)
        topk_ious, _ = torch.topk(pair_iou, top_k, dim=1)
        dynamic_ks = topk_ious.sum(dim=1).int().clamp(min=1)

        matching_matrix = torch.zeros(num_gt, num_cand, device=device, dtype=torch.bool)
        for gt_idx in range(num_gt):
            k = min(int(dynamic_ks[gt_idx].item()), num_cand)
            _, topk_inds = torch.topk(cost_matrix[gt_idx], k, largest=False)
            matching_matrix[gt_idx, topk_inds] = True

        # Resolve conflicts: anchor matched to multiple GTs → keep lowest-cost GT
        anchor_match_count = matching_matrix.sum(dim=0)  # (C,)
        conflict_mask = anchor_match_count > 1
        if conflict_mask.any():
            conflict_inds = conflict_mask.nonzero(as_tuple=True)[0]
            for idx in conflict_inds:
                gt_candidates = matching_matrix[:, idx].nonzero(as_tuple=True)[0]
                best_gt = gt_candidates[cost_matrix[gt_candidates, idx].argmin()]
                matching_matrix[:, idx] = False
                matching_matrix[best_gt, idx] = True

        # Map candidate-space results back to full anchor-space
        cand_fg_mask = matching_matrix.any(dim=0)  # (C,)
        if not cand_fg_mask.any():
            return empty_fg, empty_inds, empty_ious

        cand_abs_inds = anchor_candidates.nonzero(as_tuple=True)[0]  # (C,)
        fg_abs_inds = cand_abs_inds[cand_fg_mask]                    # (num_fg,)
        fg_mask = torch.zeros(num_pred, dtype=torch.bool, device=device)
        fg_mask[fg_abs_inds] = True

        matched_gt_inds = matching_matrix.float().argmax(dim=0)[cand_fg_mask]
        fg_pair_iou = pair_iou[:, cand_fg_mask]  # (M, num_fg)
        matched_ious = fg_pair_iou.gather(0, matched_gt_inds.unsqueeze(0)).squeeze(0)

        return fg_mask, matched_gt_inds, matched_ious

    @staticmethod
    def _pairwise_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
        """Compute pairwise IoU between two sets of boxes.

        Args:
            boxes_a: (M, 4) boxes in xyxy format.
            boxes_b: (N, 4) boxes in xyxy format.

        Returns:
            IoU matrix of shape (M, N).
        """
        eps = 1e-7

        # Intersection
        inter_x1 = torch.max(boxes_a[:, 0].unsqueeze(1), boxes_b[:, 0].unsqueeze(0))
        inter_y1 = torch.max(boxes_a[:, 1].unsqueeze(1), boxes_b[:, 1].unsqueeze(0))
        inter_x2 = torch.min(boxes_a[:, 2].unsqueeze(1), boxes_b[:, 2].unsqueeze(0))
        inter_y2 = torch.min(boxes_a[:, 3].unsqueeze(1), boxes_b[:, 3].unsqueeze(0))

        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

        area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp(min=0) * (boxes_a[:, 3] - boxes_a[:, 1]).clamp(min=0)
        area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp(min=0) * (boxes_b[:, 3] - boxes_b[:, 1]).clamp(min=0)

        union = area_a.unsqueeze(1) + area_b.unsqueeze(0) - inter_area + eps

        return inter_area / union

    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2].

        Args:
            boxes: (N, 4) tensor in center format.

        Returns:
            (N, 4) tensor in corner format.
        """
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)


# ---------------------------------------------------------------------------
# Register YOLOXLoss factory in the loss registry
# ---------------------------------------------------------------------------

def _build_yolox_loss_from_config(config: dict) -> YOLOXLoss:
    """Factory function for building YOLOXLoss from a full training config."""
    model_cfg = config.get("model", {})
    loss_cfg = config.get("loss", {})
    train_cfg = config.get("training", {})
    warmup_epochs = loss_cfg.get(
        "warmup_epochs", train_cfg.get("warmup_epochs", 0)
    )
    return YOLOXLoss(
        num_classes=model_cfg.get("num_classes", 80),
        strides=loss_cfg.get("strides", [8, 16, 32]),
        use_focal=loss_cfg.get("use_focal", False),
        iou_variant=loss_cfg.get("iou_variant", "giou"),
        cls_weight=loss_cfg.get("cls_weight", 1.0),
        obj_weight=loss_cfg.get("obj_weight", 1.0),
        reg_weight=loss_cfg.get("reg_weight", 5.0),
        simota_top_k=loss_cfg.get("simota_top_k", 10),
        focal_alpha=loss_cfg.get("focal_alpha", 0.25),
        focal_gamma=loss_cfg.get("focal_gamma", 2.0),
        warmup_epochs=warmup_epochs,
    )


LOSS_REGISTRY["yolox"] = _build_yolox_loss_from_config
for _alias in ("yolox-tiny", "yolox-s", "yolox-m", "yolox-l", "yolox-x", "yolox-nano"):
    _ARCH_LOSS_MAP[_alias] = "yolox"


# ---------------------------------------------------------------------------
# DETR passthrough loss (never called — HF models compute loss internally)
# ---------------------------------------------------------------------------


class _DETRPassthroughLoss(DetectionLoss):
    """Passthrough loss for DETR-family models.

    Never called during training because the trainer uses forward_with_loss()
    when available. Registered so build_loss() doesn't error when config
    specifies a DETR architecture.
    """

    def forward(self, predictions, targets, grids=None):
        raise RuntimeError(
            "DETR passthrough loss should never be called. "
            "Ensure the model provides forward_with_loss()."
        )


def _build_detr_loss(config: dict) -> _DETRPassthroughLoss:
    return _DETRPassthroughLoss()


LOSS_REGISTRY["detr"] = _build_detr_loss
for _alias in ("dfine", "dfine-s", "dfine-n", "dfine-m", "rtdetr", "rtdetr-r18", "rtdetr-r50"):
    _ARCH_LOSS_MAP[_alias] = "detr"


# ---------------------------------------------------------------------------
# Segmentation passthrough loss (never called — HF models compute loss internally)
# ---------------------------------------------------------------------------


class _SegmentationPassthroughLoss(DetectionLoss):
    """Passthrough loss for segmentation models.

    Never called during training because HF segmentation models provide
    forward_with_loss(). Registered so build_loss() doesn't wastefully
    construct a YOLOXLoss when config specifies a segmentation architecture.
    """

    def forward(self, predictions, targets, grids=None):
        raise RuntimeError(
            "Segmentation passthrough loss should never be called. "
            "Ensure the model provides forward_with_loss()."
        )


def _build_segmentation_loss(config: dict) -> _SegmentationPassthroughLoss:
    return _SegmentationPassthroughLoss()


LOSS_REGISTRY["segmentation"] = _build_segmentation_loss
for _alias in ("hf-segformer", "hf-mask2former", "hf-dinov2-seg"):
    _ARCH_LOSS_MAP[_alias] = "segmentation"


# ---------------------------------------------------------------------------
# Paddle passthrough loss (never called — Paddle models compute loss internally)
# ---------------------------------------------------------------------------


class _PaddlePassthroughLoss(DetectionLoss):
    """Passthrough loss for Paddle-family models (PicoDet, PP-YOLOE, PP-Cls,
    PP-Seg, PP-TinyPose).

    Never called during training because the trainer uses ``forward_with_loss()``
    when available — Paddle wrappers compute their own loss internally,
    mirroring how HF detection works via ``detr-passthrough``. Registered
    so :func:`build_loss` doesn't error when a config specifies a Paddle arch.
    """

    def forward(self, predictions, targets, grids=None):
        raise RuntimeError(
            "Paddle passthrough loss should never be called. "
            "Ensure the model provides forward_with_loss()."
        )


def _build_paddle_loss(config: dict) -> _PaddlePassthroughLoss:
    return _PaddlePassthroughLoss()


LOSS_REGISTRY["paddle-passthrough"] = _build_paddle_loss
# Arch dispatch is pattern-matched in build_loss() (paddle prefixes:
# picodet-, ppyoloe-, ppclas-, ppseg-, pp-tinypose-) rather than enumerated
# here, so new variants pick up the passthrough automatically.

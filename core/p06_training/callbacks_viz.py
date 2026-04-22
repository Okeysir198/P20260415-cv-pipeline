"""Extra viz callbacks that fire **only** on ``on_train_start``.

Currently provides :class:`NormalizedInputPreviewCallback` — the stage-3
sanity check from the reference notebook
(``notebooks/detr_finetune_reference/custom_rtdetr_v2/05_RTDERTV2_pytorch.ipynb``).

It pulls a handful of samples from the exact pipeline that feeds the model
(DataLoader + collate + processor), denormalizes, overlays GT targets, and
saves one PNG under ``<save_dir>/data_preview/normalized_input_preview.png``
before epoch 1. Catches the silent failure mode where:

- Processor normalization is on **and** our transforms also add ``v2.Normalize``
  (double-normalize — inputs go negative / near-zero).
- Processor ``do_normalize=False`` **and** transforms also skip Normalize
  (inputs arrive un-normalized — ImageNet-pretrained backbones mispredict).
- Box format drifts from cxcywh-normalized to pixel space or similar.

Written once per run, ~1 s, never touches the training loop.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from core.p05_data.base_dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    denormalize_tensor,
)

logger = logging.getLogger(__name__)


def render_normalized_input_preview(
    sample_batch: dict,
    out_path: Path,
    *,
    class_names: dict[int, str] | None = None,
    mean: list[float] = IMAGENET_MEAN,
    std: list[float] = IMAGENET_STD,
    num_samples: int = 8,
    grid_cols: int = 4,
    task: str = "detection",
    title: str = "Normalized input preview (denormalized, GT overlay)",
) -> Path:
    """Render a denormalize(pixel_values) grid with target overlays.

    ``sample_batch`` is the output of the collate function as it reaches the
    model — i.e. ``{"pixel_values": (B, 3, H, W), "labels": [...]}``  for
    detection or ``{"pixel_values": (B, 3, H, W), "labels": LongTensor}``
    for classification.
    """
    from core.p06_training.post_train import _save_grid  # reuse shared grid renderer
    from core.p10_inference.supervision_bridge import VizStyle, annotate_gt_pred
    import supervision as sv

    pixel_values = sample_batch.get("pixel_values")
    if pixel_values is None:
        return out_path  # nothing to render

    B = pixel_values.shape[0]
    take = min(num_samples, B)
    style = VizStyle()
    rows: list[np.ndarray] = []

    for i in range(take):
        # Denormalize single image back to uint8 HWC BGR
        image = denormalize_tensor(pixel_values[i], mean=mean, std=std, to_bgr=True)
        h, w = image.shape[:2]

        if task == "detection":
            labels = sample_batch.get("labels", [])
            if isinstance(labels, list) and i < len(labels):
                tgt = labels[i]
                cls_lab = tgt.get("class_labels") if isinstance(tgt, dict) else None
                boxes = tgt.get("boxes") if isinstance(tgt, dict) else None
                if boxes is None and hasattr(tgt, "class_labels"):
                    cls_lab = tgt.class_labels
                    boxes = tgt.boxes
                if boxes is not None and len(boxes) > 0:
                    boxes_np = boxes.detach().cpu().numpy() if hasattr(boxes, "detach") else np.asarray(boxes)
                    cx, cy, bw, bh = boxes_np[:, 0], boxes_np[:, 1], boxes_np[:, 2], boxes_np[:, 3]
                    gt_xyxy = np.stack([
                        (cx - bw / 2) * w, (cy - bh / 2) * h,
                        (cx + bw / 2) * w, (cy + bh / 2) * h,
                    ], axis=1).astype(np.float32)
                    gt_cls = (cls_lab.detach().cpu().numpy() if hasattr(cls_lab, "detach")
                              else np.asarray(cls_lab)).astype(np.int64)
                else:
                    gt_xyxy, gt_cls = None, None
            else:
                gt_xyxy, gt_cls = None, None
            pred_dets = sv.Detections.empty()
            rows.append(annotate_gt_pred(
                image, gt_xyxy, gt_cls, pred_dets,
                class_names or {}, style=style,
            ))
        elif task == "classification":
            label = sample_batch.get("labels")
            bar = np.full((28, image.shape[1], 3), 30, dtype=np.uint8)
            if label is not None and hasattr(label, "__getitem__"):
                try:
                    cid = int(label[i])
                except Exception:
                    cid = -1
                cname = (class_names or {}).get(cid, str(cid))
                text = f"GT: {cname}"
            else:
                text = "GT: -"
            cv2.putText(bar, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            rows.append(np.vstack([bar, image]))
        else:
            # Segmentation / keypoint: just show the image (target encoding varies)
            rows.append(image)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_grid(rows, out_path, title, ncols=grid_cols, dpi=130)
    return out_path


try:
    from transformers import TrainerCallback as _HFTrainerCallback
except Exception:  # pragma: no cover
    class _HFTrainerCallback:  # type: ignore[no-redef]
        """Fallback stub when `transformers` is unavailable — keeps the hook
        signatures so our pytorch-backend path still imports cleanly."""

        def on_train_begin(self, *args, **kwargs): return None


class NormalizedInputPreviewCallback(_HFTrainerCallback):
    """Task-agnostic callback for both backends.

    Pulls one batch from the train dataloader on `on_train_start` (pytorch)
    or `on_train_begin` (HF), denormalizes it, and writes
    ``data_preview/normalized_input_preview.png``. Inherits from
    :class:`transformers.TrainerCallback` so the HF CallbackHandler has the
    full default hook surface (``on_init_end`` etc.) without us needing to
    define no-op implementations for each one.

    For the pytorch backend, :class:`core.p06_training.callbacks.CallbackRunner`
    duck-types on ``on_train_start(trainer)`` so the HF base-class methods
    are ignored and ours takes effect.
    """

    def __init__(
        self,
        save_dir: str | Path,
        *,
        class_names: dict[int, str] | None = None,
        mean: list[float] = IMAGENET_MEAN,
        std: list[float] = IMAGENET_STD,
        num_samples: int = 8,
        grid_cols: int = 4,
        task: str = "detection",
    ) -> None:
        self.save_dir = Path(save_dir)
        self.class_names = class_names or {}
        self.mean = mean
        self.std = std
        self.num_samples = num_samples
        self.grid_cols = grid_cols
        self.task = task

    # ---------------- pytorch-backend entry point ----------------
    def on_train_start(self, trainer):
        """Pytorch-backend: grabs first batch from trainer.train_loader."""
        try:
            loader = getattr(trainer, "train_loader", None)
            if loader is None:
                return
            batch = next(iter(loader))
            sample = self._batch_to_sample_dict(batch)
            render_normalized_input_preview(
                sample, self.save_dir / "data_preview" / "normalized_input_preview.png",
                class_names=self.class_names,
                mean=self.mean, std=self.std,
                num_samples=self.num_samples, grid_cols=self.grid_cols,
                task=self.task,
            )
        except Exception as e:  # pragma: no cover — never block training
            logger.warning("NormalizedInputPreviewCallback skipped (pytorch): %s", e)

    # ---------------- HF-backend entry point ----------------
    def on_train_begin(self, args, state, control, **kwargs):
        """HF-backend: pulls first batch from the train_dataloader kwarg."""
        try:
            train_dl = kwargs.get("train_dataloader")
            if train_dl is None:
                return control
            batch = next(iter(train_dl))
            sample = self._batch_to_sample_dict(batch)
            render_normalized_input_preview(
                sample, self.save_dir / "data_preview" / "normalized_input_preview.png",
                class_names=self.class_names,
                mean=self.mean, std=self.std,
                num_samples=self.num_samples, grid_cols=self.grid_cols,
                task=self.task,
            )
        except Exception as e:  # pragma: no cover
            logger.warning("NormalizedInputPreviewCallback skipped (hf): %s", e)
        return control

    # ---------------- batch normalization helper ----------------
    def _batch_to_sample_dict(self, batch: Any) -> dict:
        """Coerce any backend's batch into ``{"pixel_values", "labels"}``."""
        if isinstance(batch, dict):
            if "pixel_values" in batch:
                return batch
            if "image" in batch:
                return {"pixel_values": batch["image"], "labels": batch.get("labels")}
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 2 and isinstance(batch[0], torch.Tensor):
                return {"pixel_values": batch[0], "labels": batch[1]}
        # Fallback: take first tensor we see
        if isinstance(batch, torch.Tensor):
            return {"pixel_values": batch, "labels": None}
        return {"pixel_values": None, "labels": None}

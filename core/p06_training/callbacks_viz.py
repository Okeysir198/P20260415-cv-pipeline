"""Extra viz callbacks that fire **only** on ``on_train_start``.

Currently provides :class:`NormalizedInputPreviewCallback` — the stage-3
sanity check from the reference notebook
(``notebooks/detr_finetune_reference/custom_rtdetr_v2/05_RTDERTV2_pytorch.ipynb``).

It pulls a handful of samples from the exact pipeline that feeds the model
(DataLoader + collate + processor), denormalizes, overlays GT targets, and
saves one PNG under ``<save_dir>/data_preview/04_normalized_input_preview.png``
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

import numpy as np
import torch

from core.p05_data.base_dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    denormalize_tensor,
)
from utils.viz import VizStyle, classification_banner

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
    import supervision as sv

    from core.p06_training.post_train import _save_grid  # reuse shared grid renderer
    from core.p10_inference.supervision_bridge import VizStyle, annotate_gt_pred

    pixel_values = sample_batch.get("pixel_values")
    if pixel_values is None:
        logger.warning("normalized_input_preview skipped — no pixel_values in batch")
        return out_path

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
            if label is not None and hasattr(label, "__getitem__"):
                try:
                    cid = int(label[i])
                except Exception:
                    cid = -1
                cname = (class_names or {}).get(cid, str(cid))
                text = f"GT: {cname}"
            else:
                text = "GT: -"
            # Preserve the original look: 28-px dark bar, white text.
            banner_style = VizStyle(
                banner_height=28,
                banner_bg_rgb=(30, 30, 30),
                banner_text_rgb=(255, 255, 255),
                banner_text_scale=0.5,
            )
            rows.append(classification_banner(image, text, style=banner_style, position="top"))
        else:
            # Segmentation / keypoint: just show the image (target encoding varies)
            rows.append(image)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_grid(rows, out_path, title, ncols=grid_cols, dpi=130)
    logger.info("NormalizedInputPreviewCallback: saved %s (%d samples)",
                out_path, len(rows))
    return out_path


class _AnyHook:
    """Base that turns every attribute access into a permissive no-op method.

    Lets one callback class satisfy both the pytorch CallbackRunner surface
    (``on_train_start(trainer)``, ``on_epoch_end(trainer, epoch, metrics)``,
    ``on_batch_end(trainer, i, metrics)``) AND the HF ``TrainerCallback``
    surface (``on_init_end(args, state, control, **kwargs)``,
    ``on_pre_optimizer_step(...)``, etc.) without tracking either API's
    growing list. Only the hooks we actually implement override this.
    """

    def __getattr__(self, name):
        if name.startswith("on_"):
            def _noop(*args, **kwargs):
                return kwargs.get("control")  # HF expects control back; pytorch ignores
            return _noop
        raise AttributeError(name)


class NormalizedInputPreviewCallback(_AnyHook):
    """Task-agnostic callback for both backends.

    Real work happens in two hook methods below — everything else is swallowed
    by :class:`_AnyHook`'s permissive no-op so the CallbackHandler of either
    backend (pytorch CallbackRunner or transformers.CallbackHandler) can
    dispatch without crashing on attribute lookups.
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
                sample, self.save_dir / "data_preview" / "04_normalized_input_preview.png",
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
                sample, self.save_dir / "data_preview" / "04_normalized_input_preview.png",
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
        """Coerce any backend's batch into ``{"pixel_values", "labels"}``.

        Handles:
          * HF detection collator → ``{"pixel_values", "labels": [{class_labels, boxes}]}``
          * pytorch YOLOX collator → ``(images_tensor, targets_list, paths_list)``
          * classification collator → ``(images, labels_tensor)``
          * segmentation collator → ``(images, masks_tensor, paths)``
        """
        if isinstance(batch, dict):
            if "pixel_values" in batch:
                return batch
            if "image" in batch:
                return {"pixel_values": batch["image"], "labels": batch.get("labels")}
            # Pytorch detection collator: {"images", "targets": [..], "paths": [..]}
            if "images" in batch:
                labels = batch.get("targets") or batch.get("labels")
                if self.task == "detection" and isinstance(labels, (list, tuple)) and labels and isinstance(labels[0], torch.Tensor):
                    converted = []
                    for t in labels:
                        if t.numel() == 0:
                            converted.append({
                                "class_labels": torch.zeros(0, dtype=torch.long),
                                "boxes": torch.zeros(0, 4, dtype=torch.float32),
                            })
                        else:
                            converted.append({
                                "class_labels": t[:, 0].long(),
                                "boxes": t[:, 1:5].float(),
                            })
                    labels = converted
                return {"pixel_values": batch["images"], "labels": labels}
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 2 and isinstance(batch[0], torch.Tensor):
                images = batch[0]
                labels = batch[1]
                # Detection's collated targets is usually a list of (N,5)
                # tensors in YOLO cxcywh-normalized space. Convert each row
                # to an HF-shaped dict so the detection render path works.
                if self.task == "detection" and isinstance(labels, (list, tuple)) and labels and isinstance(labels[0], torch.Tensor):
                    converted = []
                    for t in labels:
                        if t.numel() == 0:
                            converted.append({
                                "class_labels": torch.zeros(0, dtype=torch.long),
                                "boxes": torch.zeros(0, 4, dtype=torch.float32),
                            })
                        else:
                            converted.append({
                                "class_labels": t[:, 0].long(),
                                "boxes": t[:, 1:5].float(),
                            })
                    labels = converted
                return {"pixel_values": images, "labels": labels}
        # Fallback: take first tensor we see
        if isinstance(batch, torch.Tensor):
            return {"pixel_values": batch, "labels": None}
        return {"pixel_values": None, "labels": None}


class TransformPipelineCallback(_AnyHook):
    """Dual-backend callback that renders ``data_preview/05_transform_pipeline.png``.

    Fires once on train-start (pytorch ``on_train_start`` / HF
    ``on_train_begin``). Walks a single dataset sample through every CPU
    transform step and composes a step-by-step grid that makes
    normalize/denormalize round-trip bugs visually obvious.
    """

    def __init__(
        self,
        save_dir: str | Path,
        *,
        data_config: dict,
        training_config: dict,
        base_dir: str,
        class_names: dict[int, str] | None = None,
        gallery_samples: int = 4,
        style: Any | None = None,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.data_config = data_config
        self.training_config = training_config
        self.base_dir = base_dir
        self.class_names = class_names or {}
        self.gallery_samples = gallery_samples
        self.style = style

    def _render(self) -> None:
        from core.p05_data.detection_dataset import YOLOXDataset
        from core.p05_data.transform_pipeline_viz import render_transform_pipeline

        try:
            dataset = YOLOXDataset(
                self.data_config, split="train",
                base_dir=self.base_dir, transforms=None,
            )
        except Exception as e:
            logger.warning("TransformPipelineCallback: dataset build failed — %s", e)
            return
        if len(dataset) == 0:
            logger.warning("TransformPipelineCallback: empty train dataset")
            return
        render_transform_pipeline(
            out_path=self.save_dir / "data_preview" / "05_transform_pipeline.png",
            dataset=dataset,
            data_config=self.data_config,
            training_config=self.training_config,
            base_dir=self.base_dir,
            class_names=self.class_names,
            style=self.style,
            gallery_samples=self.gallery_samples,
        )

    # -------- pytorch backend --------
    def on_train_start(self, trainer):  # noqa: ARG002
        try:
            self._render()
        except Exception as e:  # pragma: no cover — never block training
            logger.warning("TransformPipelineCallback skipped (pytorch): %s", e)

    # -------- HF backend --------
    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ARG002
        try:
            self._render()
        except Exception as e:  # pragma: no cover
            logger.warning("TransformPipelineCallback skipped (hf): %s", e)
        return control

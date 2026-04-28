"""Native `transformers.TrainerCallback` subclasses for the HF detection backend.

Replaces the earlier `_HFVizBridge` attribute-proxy adapter — these are
first-class `TrainerCallback`s that read everything they need from HF's
documented callback kwargs (model, train_dataloader, eval_dataloader,
state.log_history) instead of synthesising a fake trainer object. Safer
against future HF Trainer API changes.

Four callbacks, one per viz we emit:

- :class:`HFDatasetStatsCallback`   — on_train_begin: `00_dataset_info.{md,json}` + `01_dataset_stats.{png,json}`
- :class:`HFDataLabelGridCallback`  — on_train_begin: `02_data_labels_<split>.png` per split
- :class:`HFAugLabelGridCallback`   — on_train_begin: `03_aug_labels_train.png`
- :class:`HFValPredictionCallback`  — on_epoch_end: `val_predictions/epoch_<N>.png`

Each takes all the data/config it needs at `__init__` so no trainer-proxy
attribute fetching is needed at hook time. Rendering helpers are imported
directly from the internal `callbacks` module — the module-level functions
there are pure (no trainer dependency).
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import supervision as sv
from transformers import TrainerCallback

from core.p05_data.transforms import build_transforms
from core.p06_training._common import (
    build_dataset_for_viz,
    task_from_output_format,
    yolo_targets_to_xyxy,
)
from utils.viz import (
    VizStyle,
    annotate_detections,
    annotate_keypoints,
    classification_banner,
    save_image_grid,
)


def _is_topdown_keypoint_dataset(ds: Any) -> bool:
    """Return True for ``KeypointTopDownDataset`` (or a Subset wrapping one)."""
    inner = getattr(ds, "dataset", ds)
    return type(inner).__name__ == "KeypointTopDownDataset"


def _denormalize_pixel_values(
    pixel_values: np.ndarray,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
) -> np.ndarray:
    """Invert ImageNet normalization on a (3, H, W) tensor → uint8 RGB (H, W, 3)."""
    px = pixel_values.astype(np.float32)
    if px.ndim == 3 and px.shape[0] == 3:
        px = px.transpose(1, 2, 0)
    px = px * np.asarray(std, dtype=np.float32) + np.asarray(mean, dtype=np.float32)
    px = np.clip(px * 255.0, 0, 255).astype(np.uint8)
    return px


def _render_topdown_keypoint_grid(
    model: Any,
    ds: Any,
    indices: list[int],
    out_path: Path,
    *,
    title: str,
    grid_cols: int = 4,
    dpi: int = 150,
    pred_score_threshold: float = 0.3,
) -> None:
    """Render a GT+Pred skeleton grid for a top-down keypoint dataset.

    For each sampled index we pull one (pixel_values, target_heatmap,
    target_weight) triple from the dataset, run the model on the stacked
    pixel_values, decode peaks for both pred + GT heatmaps, and draw both
    skeletons on the denormalized crop (GT in purple, pred in green).
    """
    import torch
    from core.p08_evaluation.keypoint_metrics import decode_heatmaps_to_xy

    inner = getattr(ds, "dataset", ds)
    index_map = (
        (lambda i: ds.indices[i]) if hasattr(ds, "indices") else (lambda i: i)
    )

    samples = []
    for idx in indices:
        try:
            real_idx = index_map(indices.index(idx))  # subset → underlying idx
        except (TypeError, ValueError):
            real_idx = idx
        try:
            item = ds[idx]  # respects Subset transparently
        except Exception:
            continue
        samples.append((real_idx, item))

    if not samples:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return

    device = next(model.parameters()).device
    pixel_batch = torch.stack([s[1]["pixel_values"] for s in samples]).to(device)
    target_hms = np.stack([s[1]["target_heatmap"].cpu().numpy() for s in samples])
    target_ws = np.stack([s[1]["target_weight"].cpu().numpy() for s in samples])

    with torch.inference_mode():
        out = model(pixel_values=pixel_batch)
    pred_hms_t = out["heatmaps"] if isinstance(out, dict) and "heatmaps" in out \
        else getattr(out, "heatmaps", None)
    if pred_hms_t is None:
        pred_hms_t = getattr(out, "logits", None)
    pred_hms = pred_hms_t.detach().float().cpu().numpy()

    stride = int(inner.input_hw[0] // pred_hms.shape[2])
    pred_xy, pred_scores = decode_heatmaps_to_xy(pred_hms, stride=stride)
    gt_xy, _ = decode_heatmaps_to_xy(target_hms, stride=stride)

    style = VizStyle()
    panels: list[np.ndarray] = []
    for i, (real_idx, _) in enumerate(samples):
        crop = _denormalize_pixel_values(samples[i][1]["pixel_values"].cpu().numpy())
        # GT in purple — gate by visibility weight
        gt_kp = np.concatenate(
            [gt_xy[i], (target_ws[i] > 0).astype(np.float32)[:, None] * 2.0],
            axis=1,
        )
        pred_kp = np.concatenate(
            [pred_xy[i], pred_scores[i, :, None]], axis=1,
        )
        from core.p10_inference.supervision_bridge import _draw_keypoints_panel
        panel = _draw_keypoints_panel(crop, gt_kp, style.gt_color_rgb, style)
        # Override threshold for pred dots: model heatmap-peak scores aren't
        # in the same range as visibility {0,1,2}; use a configured cutoff.
        pred_style = VizStyle(kpt_visibility_threshold=pred_score_threshold)
        panel = _draw_keypoints_panel(panel, pred_kp, style.pred_color_rgb, pred_style)
        # Caption: idx
        from utils.viz import classification_banner as _banner
        panel = _banner(panel, f"#{real_idx}", style=style, position="top",
                       bg_color_rgb=(0, 0, 0), text_color_rgb=(255, 255, 255))
        panels.append(panel)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image_grid(
        panels, out_path, cols=min(grid_cols, len(panels)), header=title,
    )


def _draw_gt_boxes(
    image: np.ndarray,
    targets: np.ndarray,
    class_names: dict,
    thickness: int = 2,
    text_scale: float = 0.5,
) -> np.ndarray:
    """BGR-in / BGR-out wrapper around ``utils.viz.annotate_detections``.

    Targets are YOLO normalized cxcywh in rows ``[cls, cx, cy, w, h]``.
    Mirrors the helper in ``callbacks.py`` so HF-backend viz output is
    byte-identical to pytorch-backend output.
    """
    if len(targets) == 0:
        return image.copy()
    h, w = image.shape[:2]
    xyxy, class_ids = yolo_targets_to_xyxy(targets, w, h)
    dets = sv.Detections(xyxy=xyxy.astype(np.float64), class_id=class_ids)
    style = VizStyle(box_thickness=thickness, label_text_scale=text_scale)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    annotated_rgb = annotate_detections(image_rgb, dets, class_names=class_names, style=style)
    return cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)


def _render_gt_panel(
    image_bgr: np.ndarray,
    target: Any,
    task: str,
    class_names: dict[int, str],
    thickness: int = 2,
    text_scale: float = 0.4,
) -> np.ndarray:
    """Task-aware single-panel GT rendering on a BGR image.

    Dispatches to the correct primitive per task and returns a BGR uint8
    image. Used by ``HFDataLabelGridCallback`` and ``HFAugLabelGridCallback``
    so the ``02_data_labels_*.png`` / ``03_aug_labels_*.png`` grids reflect
    each task's actual GT structure (boxes / mask / class label / keypoints).

    Target contract per task:
    - detection:     YOLO-normalized targets ``(N, 5)`` = ``[cls, cx, cy, w, h]``
    - segmentation:  ``(H, W)`` uint8 mask (pixel values = class id)
    - classification:scalar ``int`` class id
    - keypoint:      ``(K, 2)`` or ``(K, 3)`` keypoint array
    """
    if task == "detection":
        targets_np = (
            target if isinstance(target, np.ndarray)
            else np.zeros((0, 5), dtype=np.float32)
        )
        return _draw_gt_boxes(image_bgr, targets_np, class_names, thickness, text_scale)

    from core.p10_inference.supervision_bridge import _draw_keypoints_panel, _mask_overlay

    style = VizStyle(box_thickness=thickness, label_text_scale=text_scale)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if task == "segmentation":
        mask = target if isinstance(target, np.ndarray) else np.zeros(image_bgr.shape[:2], np.uint8)
        annotated_rgb = _mask_overlay(image_rgb, mask, style.gt_color_rgb, style.mask_alpha)
    elif task == "classification":
        try:
            cid = int(target)
        except (TypeError, ValueError):
            cid = -1
        label = class_names.get(cid, str(cid)) if cid >= 0 else "?"
        annotated_rgb = classification_banner(
            image_rgb, f"Label: {label}", style=style, position="top",
            bg_color_rgb=style.gt_color_rgb, text_color_rgb=(255, 255, 255),
        )
    elif task == "keypoint":
        kpts = target if isinstance(target, np.ndarray) else None
        # KeypointDataset (multi-instance YOLO-pose) emits (N, K, 3); top-down
        # emits (1, K, 3). For now we render the first instance per sample —
        # top-down has exactly one anyway, and multi-instance datasets get a
        # representative skeleton. TODO: per-instance overlay loop.
        if isinstance(kpts, np.ndarray) and kpts.ndim == 3 and kpts.shape[0] >= 1:
            kpts = kpts[0]
        annotated_rgb = _draw_keypoints_panel(image_rgb, kpts, style.gt_color_rgb, style)
    else:
        annotated_rgb = image_rgb

    return cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)


def _extract_raw_target(ds, idx: int, task: str):
    """Pull the per-task GT target for sample ``idx`` from a p05 Dataset.

    Returns the target in the shape expected by :func:`_render_gt_panel`.
    """
    if task == "detection":
        # YOLOXDataset: load YOLO normalized (N, 5) from the label .txt.
        labels = ds._load_label(ds.img_paths[idx])
        if labels is None or len(labels) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        return labels
    item = ds.get_raw_item(idx)
    if task == "keypoint":
        # `targets` here holds bboxes (compat with detection-style schema);
        # the GT skeleton lives under `keypoints`. Multi-instance YOLO-pose
        # `KeypointDataset` returns normalized [0,1] coords — denormalize to
        # pixel space for the renderer (which expects pixel coords).
        # Top-down `KeypointTopDownDataset` already returns pixel coords; we
        # detect the convention by max-coord magnitude rather than dataset
        # type so both work without coupling.
        kpts = item.get("keypoints")
        if isinstance(kpts, np.ndarray) and kpts.size and kpts[..., :2].max() <= 1.5:
            img = item.get("image")
            if img is not None:
                ih, iw = img.shape[:2]
                kpts = kpts.copy()
                kpts[..., 0] *= iw
                kpts[..., 1] *= ih
        return kpts
    # Segmentation / classification: get_raw_item returns the target directly.
    return item.get("targets")


def _save_image_grid(
    annotated: list[np.ndarray],
    grid_cols: int,
    title: str,
    out_path,
    dpi: int,
) -> None:
    """BGR-in wrapper around ``utils.viz.save_image_grid``."""
    if not annotated:
        return
    rgb_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in annotated]
    save_image_grid(
        rgb_imgs, out_path,
        cols=min(grid_cols, len(rgb_imgs)),
        header=title,
    )

from loguru import logger


def _build_class_names(data_config: dict) -> dict[int, str]:
    return {int(k): str(v) for k, v in data_config.get("names", {}).items()}


def _build_task_transforms(
    task: str,
    is_train: bool,
    aug_config: dict,
    input_size: tuple[int, int],
    mean: list | None,
    std: list | None,
):
    """Return a task-aware transform callable compatible with the p05 Dataset.

    - detection: :class:`DetectionTransform` (via core.p05_data.transforms.build_transforms)
    - classification: :func:`build_classification_transforms`
    - segmentation: :func:`build_segmentation_transforms`
    - keypoint: :func:`build_keypoint_transforms`

    Returns ``None`` if the task has no dedicated builder (lets the dataset
    fall back to its internal default).
    """
    if task == "detection":
        return build_transforms(
            config=aug_config, is_train=is_train, input_size=input_size,
            mean=mean, std=std,
        )
    if task == "classification":
        from core.p05_data.classification_dataset import build_classification_transforms
        return build_classification_transforms(
            is_train=is_train, input_size=input_size, mean=mean, std=std,
        )
    if task == "segmentation":
        try:
            from core.p05_data.segmentation_dataset import build_segmentation_transforms
            return build_segmentation_transforms(
                is_train=is_train, input_size=input_size, mean=mean, std=std,
            )
        except Exception:
            return None
    if task == "keypoint":
        try:
            from core.p05_data.keypoint_dataset import build_keypoint_transforms
            return build_keypoint_transforms(
                is_train=is_train, input_size=input_size, mean=mean, std=std,
            )
        except Exception:
            return None
    return None


def _extract_target_for_panel(targets_tensor, task: str):
    """Coerce a post-transform target into the shape :func:`_render_gt_panel` expects."""
    if task == "detection":
        if hasattr(targets_tensor, "numpy"):
            arr = targets_tensor.numpy()
        else:
            arr = np.asarray(targets_tensor)
        return arr if len(arr) > 0 else np.zeros((0, 5), dtype=np.float32)
    if task == "segmentation":
        if hasattr(targets_tensor, "numpy"):
            return targets_tensor.numpy().astype(np.uint8)
        return np.asarray(targets_tensor, dtype=np.uint8)
    if task == "classification":
        if hasattr(targets_tensor, "item"):
            return int(targets_tensor.item())
        return int(targets_tensor)
    # keypoint
    if hasattr(targets_tensor, "numpy"):
        return targets_tensor.numpy()
    if isinstance(targets_tensor, dict):
        # KeypointDataset may return {"keypoints": tensor, ...}
        kpts = targets_tensor.get("keypoints")
        if hasattr(kpts, "numpy"):
            return kpts.numpy()
        return np.asarray(kpts) if kpts is not None else None
    return np.asarray(targets_tensor)


def _tensor_to_denorm_bgr(
    tensor,
    mean: np.ndarray,
    std: np.ndarray,
    normalize_applied: bool,
) -> np.ndarray:
    """Convert a post-transform CHW float tensor → HWC uint8 BGR.

    If ``normalize_applied`` the tensor is denormalized via ``* std + mean``
    before clamp+scale. Otherwise it is already in ``[0, 1]``.
    """
    arr = tensor.detach().cpu().float().numpy().transpose(1, 2, 0)
    if normalize_applied:
        arr = np.clip(arr * std + mean, 0, 1)
    else:
        arr = np.clip(arr, 0, 1)
    return (arr[:, :, ::-1] * 255).astype(np.uint8)


class HFDatasetStatsCallback(TrainerCallback):
    """Emits `data_preview/00_dataset_info.{md,json}` + `01_dataset_stats.{png,json}`.

    Fires once at train-begin. Takes all inputs at init — doesn't need
    model/dataloader/trainer access.
    """

    def __init__(
        self,
        save_dir: str,
        data_config: dict,
        base_dir: str,
        splits: list[str],
        subsets: dict[str, list[int] | None] | None = None,
        dpi: int = 120,
        training_config: dict | None = None,
        training_config_path: str | None = None,
        data_config_path: str | None = None,
        feature_name: str | None = None,
        full_sizes: dict[str, int] | None = None,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.data_config = data_config
        self.base_dir = base_dir or ""
        self.splits = splits
        self.subsets = subsets or {s: None for s in splits}
        self.dpi = dpi
        self.training_config = training_config
        self.training_config_path = training_config_path
        self.data_config_path = data_config_path
        self.feature_name = feature_name
        self.full_sizes = full_sizes or {}

    def on_train_begin(self, args, state, control, **kwargs):
        from core.p05_data.run_viz import (
            _load_cached_stats,
            generate_dataset_stats,
            write_dataset_info,
        )

        out_dir = self.save_dir / "data_preview"
        class_names = _build_class_names(self.data_config)

        try:
            split_sizes = {
                s: (len(idxs) if idxs is not None else int(self.full_sizes.get(s, 0)))
                for s, idxs in self.subsets.items()
            }
            full_sizes = {s: int(v) for s, v in self.full_sizes.items()} if self.full_sizes else None
            write_dataset_info(
                out_dir,
                feature_name=self.feature_name,
                data_config_path=self.data_config_path,
                training_config_path=self.training_config_path,
                data_cfg=self.data_config,
                training_cfg=self.training_config,
                class_names=class_names,
                split_sizes=split_sizes,
                full_sizes=full_sizes,
            )
            if full_sizes and any(
                idxs is not None and len(idxs) < int(full_sizes.get(s, 0))
                for s, idxs in self.subsets.items()
            ):
                logger.warning(
                    "HFDatasetStatsCallback: data.subset.* active — dataset stats reflect subset, "
                    "not full splits. full=%s used=%s",
                    full_sizes, split_sizes,
                )
        except Exception as e:  # pragma: no cover
            logger.warning("HFDatasetStatsCallback: write_dataset_info failed — %s", e)

        if _load_cached_stats(out_dir):
            logger.info("HFDatasetStatsCallback: cache hit — skipping recompute (%s)", out_dir)
            return control

        try:
            # Compute subset_active / subset_pct from full vs used split sizes.
            subset_active = False
            subset_pct: float | None = None
            if self.full_sizes:
                ratios = []
                for s, idxs in self.subsets.items():
                    full_n = int(self.full_sizes.get(s, 0))
                    if full_n <= 0:
                        continue
                    used_n = len(idxs) if idxs is not None else full_n
                    if used_n < full_n:
                        subset_active = True
                    ratios.append(used_n / full_n)
                if subset_active and ratios:
                    subset_pct = round(float(np.mean(ratios)) * 100)
            generate_dataset_stats(
                self.data_config, self.base_dir, class_names,
                self.splits, out_dir, self.dpi,
                subset_indices=self.subsets,
                subset_active=subset_active,
                subset_pct=subset_pct,
            )
        except Exception as e:  # pragma: no cover
            logger.warning("HFDatasetStatsCallback failed: %s", e)
        return control


class HFDataLabelGridCallback(TrainerCallback):
    """Emits `data_preview/02_data_labels_<split>.png` once at training start.

    Task-aware: dispatches to the right p05 Dataset via
    :func:`build_dataset_for_viz` and renders GT per task via
    :func:`_render_gt_panel` — bbox overlays for detection, mask overlays for
    segmentation, class banners for classification, keypoint dots for
    keypoint.
    """

    def __init__(
        self,
        save_dir: str,
        splits: list[str],
        data_config: dict,
        base_dir: str,
        task: str = "detection",
        subsets: dict[str, list[int] | None] | None = None,
        num_samples: int = 16,
        grid_cols: int = 4,
        thickness: int = 2,
        text_scale: float = 0.4,
        dpi: int = 120,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.splits = splits
        self.data_config = data_config
        self.base_dir = base_dir or ""
        self.task = task
        self.subsets = subsets or {s: None for s in splits}
        self.num_samples = num_samples
        self.grid_cols = grid_cols
        self.thickness = thickness
        self.text_scale = text_scale
        self.dpi = dpi

    def on_train_begin(self, args, state, control, **kwargs):
        class_names = _build_class_names(self.data_config)
        for split in self.splits:
            try:
                ds = build_dataset_for_viz(
                    self.task, split, self.data_config, self.base_dir, transforms=None,
                )
            except Exception as e:
                logger.info("HFDataLabelGridCallback: skip split %s — %s", split, e)
                continue

            subset = self.subsets.get(split)
            pool = list(range(len(ds))) if subset is None else list(subset)
            n = min(self.num_samples, len(pool))
            if n == 0:
                continue
            indices = sorted(random.sample(pool, n))

            annotated: list[np.ndarray] = []
            for idx in indices:
                try:
                    item = ds.get_raw_item(idx)
                    target = _extract_raw_target(ds, idx, self.task)
                except Exception as e:
                    logger.warning("HFDataLabelGridCallback: failed idx %d — %s", idx, e)
                    continue
                annotated.append(_render_gt_panel(
                    item["image"], target, self.task, class_names,
                    self.thickness, self.text_scale,
                ))
            if not annotated:
                continue

            out_path = self.save_dir / "data_preview" / f"02_data_labels_{split}.png"
            _save_image_grid(
                annotated, self.grid_cols,
                f"Data + Labels [{split}] — {n} samples ({self.task})",
                out_path, self.dpi,
            )
            logger.info("HFDataLabelGridCallback: saved %s", out_path)
        return control


class HFAugLabelGridCallback(TrainerCallback):
    """Emits `data_preview/03_aug_labels_train.png` (augmented GT grid) at start.

    Applies `is_train=True` transforms with mosaic/mixup/copypaste disabled so
    each cell shows a single identifiable image — makes the HSV/affine/flip
    parameters visually verifiable. Mirrors the pytorch-backend
    :class:`AugLabelGridLogger`.
    """

    def __init__(
        self,
        save_dir: str,
        splits: list[str],
        data_config: dict,
        aug_config: dict,
        base_dir: str,
        input_size: tuple[int, int],
        task: str = "detection",
        subsets: dict[str, list[int] | None] | None = None,
        num_samples: int = 16,
        grid_cols: int = 4,
        thickness: int = 2,
        text_scale: float = 0.4,
        dpi: int = 120,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.splits = splits
        self.data_config = data_config
        self.aug_config = aug_config or {}
        self.base_dir = base_dir or ""
        self.input_size = tuple(input_size)
        self.task = task
        self.subsets = subsets or {s: None for s in splits}
        self.num_samples = num_samples
        self.grid_cols = grid_cols
        self.thickness = thickness
        self.text_scale = text_scale
        self.dpi = dpi

    def on_train_begin(self, args, state, control, **kwargs):
        class_names = _build_class_names(self.data_config)
        mean = np.asarray(
            self.data_config.get("mean", [0.485, 0.456, 0.406]),
            dtype=np.float32,
        ).reshape(1, 1, 3)
        std = np.asarray(
            self.data_config.get("std", [0.229, 0.224, 0.225]),
            dtype=np.float32,
        ).reshape(1, 1, 3)

        # Drop batch-level ops so each rendered cell is one clear augmented image.
        simple_cfg = {
            **self.aug_config, "mosaic": False, "mixup": False, "copypaste": False,
        }
        try:
            transforms = _build_task_transforms(
                task=self.task, is_train=True, aug_config=simple_cfg,
                input_size=self.input_size,
                mean=self.data_config.get("mean"), std=self.data_config.get("std"),
            )
        except Exception as e:
            logger.info(
                "HFAugLabelGridCallback: transform-build failed for task=%s "
                "(%s) — falling back to dataset default.", self.task, e,
            )
            transforms = None

        for split in self.splits:
            if split != "train":
                continue
            try:
                ds = build_dataset_for_viz(
                    self.task, split, self.data_config, self.base_dir,
                    transforms=transforms,
                )
            except Exception as e:
                logger.info("HFAugLabelGridCallback: skip %s — %s", split, e)
                continue

            subset = self.subsets.get(split)
            pool = list(range(len(ds))) if subset is None else list(subset)
            n = min(self.num_samples, len(pool))
            if n == 0:
                continue
            indices = sorted(random.sample(pool, n))

            annotated: list[np.ndarray] = []
            for i in indices:
                try:
                    result = ds[i]
                    aug_tensor, targets_tensor = result[0], result[1]
                except Exception as e:
                    logger.warning("HFAugLabelGridCallback: failed idx %d — %s", i, e)
                    continue
                aug_bgr = _tensor_to_denorm_bgr(
                    aug_tensor, mean, std,
                    normalize_applied=self.aug_config.get("normalize", True),
                )
                target = _extract_target_for_panel(targets_tensor, self.task)
                annotated.append(_render_gt_panel(
                    aug_bgr, target, self.task, class_names,
                    self.thickness, self.text_scale,
                ))
            if not annotated:
                continue

            out_path = self.save_dir / "data_preview" / f"03_aug_labels_{split}.png"
            _save_image_grid(
                annotated, self.grid_cols,
                f"Augmented + Labels [{split}] — {n} samples ({self.task})",
                out_path, self.dpi,
            )
            logger.info("HFAugLabelGridCallback: saved %s", out_path)
        return control


class HFValPredictionCallback(TrainerCallback):
    """Per-epoch val grids + (on_train_end) best-checkpoint val/test grids.

    Uses the HF `eval_dataloader` (passed via hook kwargs by HF Trainer) for
    per-epoch grids. Samples a fixed pool of indices on the first epoch so the
    same images appear across every epoch's grid for easy before/after
    comparison.

    On `on_train_end` HF has just reloaded the best checkpoint (via
    ``load_best_model_at_end=True``). We use that moment to render one final
    grid from the best weights — the same weights that produced the reported
    ``test_map`` — and save to ``{val,test}_predictions/best.png``. Test-set
    rendering fires only when a ``test_dataset`` is passed at init.
    """

    def __init__(
        self,
        save_dir: str,
        class_names: dict[int, str],
        input_size: tuple[int, int],
        num_samples: int = 12,
        conf_threshold: float = 0.05,
        grid_cols: int = 2,
        gt_thickness: int = 2,
        pred_thickness: int = 1,
        text_scale: float = 0.4,
        dpi: int = 150,
        test_dataset: Any = None,
        train_dataset: Any = None,
        best_num_samples: int = 16,
        best_conf_threshold: float = 0.1,
        enable_epoch_end: bool = True,
        enable_train_end: bool = True,
        data_config: dict | None = None,
        base_dir: str | None = None,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.class_names = class_names
        self.input_size = tuple(input_size)
        self.num_samples = num_samples
        self.conf_threshold = conf_threshold
        self.grid_cols = grid_cols
        self.gt_thickness = gt_thickness
        self.pred_thickness = pred_thickness
        self.text_scale = text_scale
        self.dpi = dpi
        self.test_dataset = test_dataset
        self.train_dataset = train_dataset
        self.best_num_samples = best_num_samples
        self.best_conf_threshold = best_conf_threshold
        self.enable_epoch_end = enable_epoch_end
        self.enable_train_end = enable_train_end
        self._loaded_data_cfg = data_config
        self._config_dir = base_dir
        self._sample_indices: list[int] | None = None

    def on_epoch_end(self, args, state, control, **kwargs):
        """Per-epoch val grid. Delegates rendering to the shared
        :func:`core.p06_training.post_train.render_prediction_grid` so the
        per-epoch grid is byte-consistent with best.png and the error-analysis
        galleries."""
        if not self.enable_epoch_end:
            return control
        eval_loader = kwargs.get("eval_dataloader")
        model = kwargs.get("model")
        if eval_loader is None or model is None:
            return control

        ds = eval_loader.dataset
        if self._sample_indices is None:
            n = len(ds)
            if n == 0:
                return control
            self._sample_indices = sorted(random.sample(range(n), min(self.num_samples, n)))

        was_training = model.training
        model.eval()

        epoch_idx = int(round(state.epoch or 0.0))

        # Top-down keypoint short-circuit — the shared grid path uses the
        # detection postprocess and doesn't fit our (pixel_values,
        # target_heatmap, target_weight) sample shape.
        task = _infer_task_from_model(model)
        if task == "keypoint" and _is_topdown_keypoint_dataset(ds):
            ap_val = 0.0
            if state.log_history:
                for entry in reversed(state.log_history):
                    if "eval_AP" in entry:
                        ap_val = float(entry["eval_AP"]); break
            out_path = (
                self.save_dir / "val_predictions" / "epochs"
                / f"epoch_{epoch_idx:03d}.png"
            )
            try:
                _render_topdown_keypoint_grid(
                    model, ds, self._sample_indices, out_path,
                    title=f"Epoch {epoch_idx} — AP: {ap_val:.4f}",
                    grid_cols=self.grid_cols, dpi=self.dpi,
                )
                logger.info(
                    "HFValPredictionCallback: saved epochs/epoch_%03d.png",
                    epoch_idx,
                )
            except Exception as e:
                logger.warning("per-epoch val grid skipped: %s", e)
            if was_training:
                model.train()
            return control

        map_val = 0.0
        if state.log_history:
            for entry in reversed(state.log_history):
                if "eval_map_50" in entry:
                    map_val = float(entry["eval_map_50"]); break

        from core.p06_training.post_train import render_prediction_grid
        from core.p10_inference.supervision_bridge import VizStyle
        out_path = self.save_dir / "val_predictions" / "epochs" / f"epoch_{epoch_idx:03d}.png"
        try:
            render_prediction_grid(
                model, ds, self._sample_indices, out_path,
                title=f"Epoch {epoch_idx} — mAP50: {map_val:.4f}",
                class_names=self.class_names, input_size=self.input_size,
                style=VizStyle(), task=task,
                conf_threshold=self.conf_threshold, grid_cols=self.grid_cols,
                dpi=self.dpi,
            )
            logger.info("HFValPredictionCallback: saved epochs/epoch_%03d.png", epoch_idx)
        except Exception as e:
            logger.warning("per-epoch val grid skipped: %s", e)

        if was_training:
            model.train()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        """Render best-checkpoint val+test grids + full error analysis.

        HF's ``load_best_model_at_end=True`` has already reloaded best weights
        by the time this hook fires — so the artifacts reflect the same
        checkpoint that produced the reported ``test_map``.

        Entirely delegated to :func:`core.p06_training.post_train.run_post_train_artifacts`
        so the pytorch backend and HF backend produce byte-identical artifact
        trees.
        """
        if not self.enable_train_end:
            return control
        model = kwargs.get("model")
        if model is None:
            return control

        from core.p06_training.post_train import run_post_train_artifacts
        from core.p10_inference.supervision_bridge import VizStyle

        best_map = 0.0
        for entry in state.log_history:
            if "eval_map_50" in entry:
                best_map = max(best_map, float(entry["eval_map_50"]))
        test_map = None
        for entry in reversed(state.log_history):
            if "test_map_50" in entry:
                test_map = float(entry["test_map_50"])
                break

        val_loader = kwargs.get("eval_dataloader")
        val_ds = val_loader.dataset if val_loader is not None else None

        try:
            training_config = _build_hf_training_config(args, state, model, best_map, test_map)
            # Plumb resolved data config + base_dir so post_train analyzers
            # (duplicates_leakage, etc.) can resolve split paths.
            if self._loaded_data_cfg is not None:
                training_config["_loaded_data_cfg"] = self._loaded_data_cfg
            if self._config_dir is not None:
                training_config["_config_dir"] = self._config_dir
            run_post_train_artifacts(
                model=model,
                save_dir=self.save_dir,
                val_dataset=val_ds,
                test_dataset=self.test_dataset,
                train_dataset=self.train_dataset,
                val_loader=val_loader,
                task=_infer_task_from_model(model),
                class_names=self.class_names,
                input_size=self.input_size,
                style=VizStyle(),
                best_num_samples=self.best_num_samples,
                best_conf_threshold=self.best_conf_threshold,
                log_history_best_map=best_map if best_map > 0 else None,
                log_history_test_map=test_map,
                training_config=training_config,
            )
        except Exception as e:
            logger.warning("post-train artifacts skipped: %s", e, exc_info=True)
        return control


def _build_hf_training_config(args, state, model, best_map: float, test_map: float | None) -> dict:
    """Extract a compact training-config snapshot from the HF trainer state.

    Shape matches the contract in the plan: model / training / augmentation / run.
    Missing fields are set to None — summary.md prints what's available.
    """
    inner = getattr(model, "hf_model", None)
    arch = None
    params = None
    if inner is not None:
        arch = getattr(getattr(inner, "config", None), "model_type", None) or \
               type(inner).__name__
        try:
            params = int(sum(p.numel() for p in inner.parameters() if p.requires_grad))
        except Exception:
            params = None
    best_epoch = None
    for e in state.log_history:
        if "eval_map_50" in e and float(e["eval_map_50"]) >= best_map - 1e-9:
            best_epoch = e.get("epoch"); break
    return {
        "model": {"arch": arch, "trainable_params": params,
                  "input_size": getattr(args, "_input_size", None)},
        "training": {
            "backend": "hf",
            "epochs": getattr(args, "num_train_epochs", None),
            "batch_size": getattr(args, "per_device_train_batch_size", None),
            "lr": getattr(args, "learning_rate", None),
            "optimizer": getattr(args, "optim", None),
            "scheduler": getattr(args, "lr_scheduler_type", None),
            "warmup_steps": getattr(args, "warmup_steps", None),
            "weight_decay": getattr(args, "weight_decay", None),
            "bf16": getattr(args, "bf16", None),
            "fp16": getattr(args, "fp16", None),
            "seed": getattr(args, "seed", None),
            "max_grad_norm": getattr(args, "max_grad_norm", None),
        },
        "run": {
            "best_val_map_50": round(float(best_map), 4) if best_map else None,
            "best_epoch": best_epoch,
            "total_epochs": state.epoch,
            "test_map_50": round(float(test_map), 4) if test_map is not None else None,
        },
    }


def _infer_task_from_model(model) -> str:
    """Map ``model.output_format`` → canonical task for the post-train runner."""
    return task_from_output_format(getattr(model, "output_format", None))


class HFNormalizedInputPreviewCallback(TrainerCallback):
    """Emits flat ``data_preview/05_normalized_input_preview.png`` at train-begin.

    Renders an ``N`` sample grid of the post-normalize tensor denormalized
    back to RGB, with task-aware GT overlays (detection=boxes,
    classification=banner, segmentation=mask, keypoint=dots+skeleton).

    Uses the ``is_train=False`` transform pipeline (resize + normalize, no
    aug) so the grid shows exactly what the model sees at val/test time.
    A colour cast or clamped-to-grey output here indicates a
    rescale/normalize mismatch between the training pipeline and model
    processor — same footgun ``04_transform_pipeline.png`` catches, but
    also visible for non-detection tasks.
    """

    def __init__(
        self,
        save_dir: str,
        data_config: dict,
        training_config: dict,
        base_dir: str,
        input_size: tuple[int, int],
        task: str = "detection",
        subsets: dict[str, list[int] | None] | None = None,
        num_samples: int = 16,
        grid_cols: int = 4,
        thickness: int = 2,
        text_scale: float = 0.4,
        dpi: int = 120,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.data_config = data_config
        self.training_config = training_config or {}
        self.base_dir = base_dir or ""
        self.input_size = tuple(input_size)
        self.task = task
        self.subsets = subsets or {}
        self.num_samples = num_samples
        self.grid_cols = grid_cols
        self.thickness = thickness
        self.text_scale = text_scale
        self.dpi = dpi

    def on_train_begin(self, args, state, control, **kwargs):
        class_names = _build_class_names(self.data_config)
        mean = np.asarray(
            self.data_config.get("mean", [0.485, 0.456, 0.406]),
            dtype=np.float32,
        ).reshape(1, 1, 3)
        std = np.asarray(
            self.data_config.get("std", [0.229, 0.224, 0.225]),
            dtype=np.float32,
        ).reshape(1, 1, 3)

        # Val pipeline — resize + normalize, no aug. For detection we pass an
        # augmentation dict with the stateless flags off; for other tasks
        # _build_task_transforms picks the task-specific val builder.
        aug_cfg = dict(self.training_config.get("augmentation", {}) or {})
        aug_cfg["mosaic"] = False
        aug_cfg["mixup"] = False
        aug_cfg["copypaste"] = False
        normalize_applied = bool(aug_cfg.get("normalize", True))

        try:
            transforms = _build_task_transforms(
                task=self.task, is_train=False, aug_config=aug_cfg,
                input_size=self.input_size,
                mean=self.data_config.get("mean"),
                std=self.data_config.get("std"),
            )
        except Exception as e:
            logger.info(
                "HFNormalizedInputPreviewCallback: transform-build failed for "
                "task=%s (%s) — using dataset default.", self.task, e,
            )
            transforms = None

        # Prefer the train split so 05_ aligns with 02_data_labels_train /
        # 03_aug_labels_train. Falls back to the first available split.
        split = "train" if "train" in (self.subsets or {"train": None}) else None
        if split is None:
            split = next(iter(self.subsets or ["train"]), "train")

        try:
            ds = build_dataset_for_viz(
                self.task, split, self.data_config, self.base_dir,
                transforms=transforms,
            )
        except Exception as e:
            logger.info(
                "HFNormalizedInputPreviewCallback: skip (dataset build failed "
                "for split=%s): %s", split, e,
            )
            return control

        subset = self.subsets.get(split)
        pool = list(range(len(ds))) if subset is None else list(subset)
        n = min(self.num_samples, len(pool))
        if n == 0:
            return control
        indices = sorted(random.sample(pool, n))

        annotated: list[np.ndarray] = []
        for i in indices:
            try:
                result = ds[i]
                img_tensor = result[0]
                targets_tensor = result[1]
            except Exception as e:
                logger.warning(
                    "HFNormalizedInputPreviewCallback: failed idx %d — %s", i, e,
                )
                continue
            try:
                img_bgr = _tensor_to_denorm_bgr(
                    img_tensor, mean, std, normalize_applied=normalize_applied,
                )
                target = _extract_target_for_panel(targets_tensor, self.task)
                annotated.append(_render_gt_panel(
                    img_bgr, target, self.task, class_names,
                    self.thickness, self.text_scale,
                ))
            except Exception as e:
                logger.warning(
                    "HFNormalizedInputPreviewCallback: render failed idx %d — %s",
                    i, e,
                )
                continue

        if not annotated:
            return control

        out_path = self.save_dir / "data_preview" / "05_normalized_input_preview.png"
        _save_image_grid(
            annotated, self.grid_cols,
            f"Normalized input (denormalized) + GT [{split}] — "
            f"{len(annotated)} samples ({self.task})",
            out_path, self.dpi,
        )
        logger.info("HFNormalizedInputPreviewCallback: saved %s", out_path)
        return control

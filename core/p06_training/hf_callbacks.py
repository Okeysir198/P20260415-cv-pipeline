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

import logging
import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import supervision as sv
from transformers import TrainerCallback

from core.p05_data.detection_dataset import YOLOXDataset
from core.p05_data.transforms import build_transforms
from utils.viz import (
    VizStyle,
    annotate_detections,
    save_image_grid,
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
    cx, cy, bw, bh = targets[:, 1], targets[:, 2], targets[:, 3], targets[:, 4]
    xyxy = np.stack([
        (cx - bw / 2) * w, (cy - bh / 2) * h,
        (cx + bw / 2) * w, (cy + bh / 2) * h,
    ], axis=1).astype(np.float64)
    class_ids = targets[:, 0].astype(np.int64)
    dets = sv.Detections(xyxy=xyxy, class_id=class_ids)
    style = VizStyle(box_thickness=thickness, label_text_scale=text_scale)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    annotated_rgb = annotate_detections(image_rgb, dets, class_names=class_names, style=style)
    return cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)


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

logger = logging.getLogger(__name__)


def _build_class_names(data_config: dict) -> dict[int, str]:
    return {int(k): str(v) for k, v in data_config.get("names", {}).items()}


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
            write_dataset_info(
                out_dir,
                feature_name=self.feature_name,
                data_config_path=self.data_config_path,
                training_config_path=self.training_config_path,
                data_cfg=self.data_config,
                training_cfg=self.training_config,
                class_names=class_names,
                split_sizes=split_sizes,
            )
        except Exception as e:  # pragma: no cover
            logger.warning("HFDatasetStatsCallback: write_dataset_info failed — %s", e)

        if _load_cached_stats(out_dir):
            logger.info("HFDatasetStatsCallback: cache hit — skipping recompute (%s)", out_dir)
            return control

        try:
            generate_dataset_stats(
                self.data_config, self.base_dir, class_names,
                self.splits, out_dir, self.dpi,
                subset_indices=self.subsets,
            )
        except Exception as e:  # pragma: no cover
            logger.warning("HFDatasetStatsCallback failed: %s", e)
        return control


class HFDataLabelGridCallback(TrainerCallback):
    """Emits `data_preview/02_data_labels_<split>.png` once at training start."""

    def __init__(
        self,
        save_dir: str,
        splits: list[str],
        data_config: dict,
        base_dir: str,
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
                ds = YOLOXDataset(
                    data_config=self.data_config, split=split,
                    transforms=None, base_dir=self.base_dir,
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
                item = ds.get_raw_item(idx)
                targets = ds._load_label(ds.img_paths[idx])
                if targets is None or len(targets) == 0:
                    targets = np.zeros((0, 5), dtype=np.float32)
                annotated.append(_draw_gt_boxes(
                    item["image"], targets, class_names,
                    self.thickness, self.text_scale,
                ))
            if not annotated:
                continue

            out_path = self.save_dir / "data_preview" / f"02_data_labels_{split}.png"
            _save_image_grid(
                annotated, self.grid_cols,
                f"Data + Labels [{split}] — {n} samples",
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
        transforms = build_transforms(
            config=simple_cfg, is_train=True, input_size=self.input_size,
            mean=self.data_config.get("mean"), std=self.data_config.get("std"),
        )

        for split in self.splits:
            if split != "train":
                continue
            try:
                ds = YOLOXDataset(
                    data_config=self.data_config, split=split,
                    transforms=transforms, base_dir=self.base_dir,
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
                aug_np = aug_tensor.numpy().transpose(1, 2, 0)
                if self.aug_config.get("normalize", True):
                    aug_np = np.clip(aug_np * std + mean, 0, 1)
                else:
                    aug_np = np.clip(aug_np, 0, 1)
                aug_bgr = (aug_np[:, :, ::-1] * 255).astype(np.uint8)
                targets_np = (
                    targets_tensor.numpy() if len(targets_tensor) > 0
                    else np.zeros((0, 5), dtype=np.float32)
                )
                annotated.append(_draw_gt_boxes(
                    aug_bgr, targets_np, class_names,
                    self.thickness, self.text_scale,
                ))
            if not annotated:
                continue

            out_path = self.save_dir / "data_preview" / f"03_aug_labels_{split}.png"
            _save_image_grid(
                annotated, self.grid_cols,
                f"Augmented + Labels [{split}] — {n} samples",
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
        best_num_samples: int = 16,
        best_conf_threshold: float = 0.1,
        enable_epoch_end: bool = True,
        enable_train_end: bool = True,
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
        self.best_num_samples = best_num_samples
        self.best_conf_threshold = best_conf_threshold
        self.enable_epoch_end = enable_epoch_end
        self.enable_train_end = enable_train_end
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

        if self._sample_indices is None:
            n = len(eval_loader.dataset)
            if n == 0:
                return control
            self._sample_indices = sorted(random.sample(range(n), min(self.num_samples, n)))

        was_training = model.training
        model.eval()

        epoch_idx = int(round(state.epoch or 0.0))
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
                model, eval_loader.dataset, self._sample_indices, out_path,
                title=f"Epoch {epoch_idx} — mAP50: {map_val:.4f}",
                class_names=self.class_names, input_size=self.input_size,
                style=VizStyle(), task=_infer_task_from_model(model),
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
            run_post_train_artifacts(
                model=model,
                save_dir=self.save_dir,
                val_dataset=val_ds,
                test_dataset=self.test_dataset,
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
    of = getattr(model, "output_format", None) or "detection"
    of = of.lower()
    if of in {"detr", "yolox", "detection"}:
        return "detection"
    if of in {"classification", "cls"}:
        return "classification"
    if of in {"segmentation", "seg"}:
        return "segmentation"
    if of in {"keypoint", "pose"}:
        return "keypoint"
    return "detection"

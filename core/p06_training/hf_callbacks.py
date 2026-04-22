"""Native `transformers.TrainerCallback` subclasses for the HF detection backend.

Replaces the earlier `_HFVizBridge` attribute-proxy adapter — these are
first-class `TrainerCallback`s that read everything they need from HF's
documented callback kwargs (model, train_dataloader, eval_dataloader,
state.log_history) instead of synthesising a fake trainer object. Safer
against future HF Trainer API changes.

Four callbacks, one per viz we emit:

- :class:`HFDatasetStatsCallback`   — on_train_begin: `dataset_stats.{json,png}`
- :class:`HFDataLabelGridCallback`  — on_train_begin: `data_labels_<split>.png` per split
- :class:`HFAugLabelGridCallback`   — on_train_begin: `aug_labels_train.png`
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

import numpy as np
from transformers import TrainerCallback

from core.p05_data.detection_dataset import YOLOXDataset
from core.p05_data.transforms import build_transforms
from core.p06_training.callbacks import (
    _draw_gt_boxes,
    _save_image_grid,
)

logger = logging.getLogger(__name__)


def _build_class_names(data_config: dict) -> dict[int, str]:
    return {int(k): str(v) for k, v in data_config.get("names", {}).items()}


class HFDatasetStatsCallback(TrainerCallback):
    """Emits `data_preview/dataset_stats.{json,png}` once at training start.

    Takes all inputs at init — doesn't need model/dataloader/trainer access.
    """

    def __init__(
        self,
        save_dir: str,
        data_config: dict,
        base_dir: str,
        splits: list[str],
        subsets: dict[str, list[int] | None] | None = None,
        dpi: int = 120,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.data_config = data_config
        self.base_dir = base_dir or ""
        self.splits = splits
        self.subsets = subsets or {s: None for s in splits}
        self.dpi = dpi

    def on_train_begin(self, args, state, control, **kwargs):
        from core.p05_data.run_viz import _load_cached_stats, generate_dataset_stats

        out_dir = self.save_dir / "data_preview"
        if _load_cached_stats(out_dir):
            logger.info("HFDatasetStatsCallback: cache hit — skipping recompute (%s)", out_dir)
            return control

        try:
            generate_dataset_stats(
                self.data_config, self.base_dir, _build_class_names(self.data_config),
                self.splits, out_dir, self.dpi,
                subset_indices=self.subsets,
            )
        except Exception as e:  # pragma: no cover
            logger.warning("HFDatasetStatsCallback failed: %s", e)
        return control


class HFDataLabelGridCallback(TrainerCallback):
    """Emits `data_preview/data_labels_<split>.png` once at training start."""

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

            out_path = self.save_dir / "data_preview" / f"data_labels_{split}.png"
            _save_image_grid(
                annotated, self.grid_cols,
                f"Data + Labels [{split}] — {n} samples",
                out_path, self.dpi,
            )
            logger.info("HFDataLabelGridCallback: saved %s", out_path)
        return control


class HFAugLabelGridCallback(TrainerCallback):
    """Emits `data_preview/aug_labels_train.png` (augmented GT grid) at start.

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

            out_path = self.save_dir / "data_preview" / f"aug_labels_{split}.png"
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
        best_conf_threshold: float = 0.3,
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
        self._sample_indices: list[int] | None = None

    @staticmethod
    def _unwrap(dataset: Any):
        if hasattr(dataset, "indices"):  # torch.utils.data.Subset
            indices = dataset.indices
            return dataset.dataset, (lambda i: indices[i])
        return dataset, (lambda i: i)

    def _forward_and_render_rows(self, model, dataset, indices, conf_threshold):
        """Run best-checkpoint forward on dataset[indices] and return annotated
        rows (list of BGR ndarrays). Shared by on_epoch_end and on_train_end
        so the per-epoch and best-checkpoint grids use identical rendering.
        """
        import cv2
        import supervision as sv
        import torch
        from core.p10_inference.supervision_bridge import annotate_gt_pred

        raw_dataset, idx_map = self._unwrap(dataset)
        device = next(model.parameters()).device
        input_h, input_w = self.input_size

        samples = []
        for idx in indices:
            real_idx = idx_map(idx)
            img_path = raw_dataset.img_paths[real_idx]
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            raw_img = raw_dataset.get_raw_item(real_idx)["image"]
            resized = cv2.resize(raw_img, (input_w, input_h))
            tensor = torch.from_numpy(
                np.ascontiguousarray(
                    (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)
                )
            )
            samples.append((real_idx, image, tensor))
        if not samples or not hasattr(model, "postprocess"):
            return []

        batch = torch.stack([t for _, _, t in samples]).to(device)
        with torch.no_grad():
            preds_raw = model(pixel_values=batch)
        target_sizes = torch.tensor([[input_h, input_w]] * batch.shape[0], device=device)
        all_decoded = model.postprocess(preds_raw, conf_threshold, target_sizes)

        rows: list[np.ndarray] = []
        for i, (real_idx, image, _) in enumerate(samples):
            orig_h, orig_w = image.shape[:2]
            gt_xyxy, gt_class_ids = None, None
            gt_targets = raw_dataset._load_label(raw_dataset.img_paths[real_idx])
            if gt_targets is not None and len(gt_targets) > 0:
                cx, cy, w, h = (gt_targets[:, 1], gt_targets[:, 2],
                                gt_targets[:, 3], gt_targets[:, 4])
                gt_xyxy = np.stack([
                    (cx - w / 2) * orig_w, (cy - h / 2) * orig_h,
                    (cx + w / 2) * orig_w, (cy + h / 2) * orig_h,
                ], axis=1)
                gt_class_ids = gt_targets[:, 0].astype(np.int64)

            pred = all_decoded[i] if i < len(all_decoded) else {}
            pred_boxes = np.asarray(pred.get("boxes", []), dtype=np.float64).reshape(-1, 4)
            pred_labels = np.asarray(pred.get("labels", []), dtype=np.int64).ravel()
            pred_scores = np.asarray(pred.get("scores", []), dtype=np.float64).ravel()
            if pred_boxes.shape[0] > 0:
                pred_boxes[:, [0, 2]] *= orig_w / input_w
                pred_boxes[:, [1, 3]] *= orig_h / input_h
            pred_dets = sv.Detections(xyxy=pred_boxes, class_id=pred_labels, confidence=pred_scores)

            rows.append(annotate_gt_pred(
                image, gt_xyxy, gt_class_ids, pred_dets, self.class_names,
                gt_thickness=self.gt_thickness, pred_thickness=self.pred_thickness,
                text_scale=self.text_scale, draw_legend=True,
            ))
        return rows

    def _save_grid(self, rows, out_path, title, ncols):
        import cv2, matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        nrows = math.ceil(len(rows) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5))
        axes = np.asarray(axes).ravel()
        for i in range(nrows * ncols):
            axes[i].axis("off")
            if i < len(rows):
                axes[i].imshow(cv2.cvtColor(rows[i], cv2.COLOR_BGR2RGB))
        fig.suptitle(title, fontsize=14)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def on_epoch_end(self, args, state, control, **kwargs):
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
        rows = self._forward_and_render_rows(
            model, eval_loader.dataset, self._sample_indices, self.conf_threshold,
        )
        if not rows:
            if was_training: model.train()
            return control

        epoch_idx = int(round(state.epoch or 0.0))
        map_val = 0.0
        if state.log_history:
            for entry in reversed(state.log_history):
                if "eval_map_50" in entry:
                    map_val = float(entry["eval_map_50"]); break

        # Per-epoch grids land under val_predictions/epochs/ to keep
        # val_predictions/ itself short (just epochs/, best.png, error_analysis/).
        out_dir = self.save_dir / "val_predictions" / "epochs"
        self._save_grid(rows, out_dir / f"epoch_{epoch_idx:03d}.png",
                        f"Epoch {epoch_idx} — mAP50: {map_val:.4f}", self.grid_cols)
        logger.info("HFValPredictionCallback: saved epochs/epoch_%03d.png", epoch_idx)

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
            )
        except Exception as e:
            logger.warning("post-train artifacts skipped: %s", e, exc_info=True)
        return control


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

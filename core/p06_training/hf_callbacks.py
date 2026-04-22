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

        out_dir = self.save_dir / "val_predictions"
        self._save_grid(rows, out_dir / f"epoch_{epoch_idx:03d}.png",
                        f"Epoch {epoch_idx} — mAP50: {map_val:.4f}", self.grid_cols)
        logger.info("HFValPredictionCallback: saved epoch_%03d.png", epoch_idx)

        if was_training:
            model.train()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        """Render best-checkpoint predictions on val + test (HF has just
        reloaded best weights via load_best_model_at_end=True).

        Also dumps per-class TP/FP/FN counts + hardest-10 images as
        ``test_predictions/error_analysis.json``. Runs at end only — does not
        touch training.
        """
        model = kwargs.get("model")
        if model is None:
            return control
        was_training = model.training
        model.eval()

        best_map = 0.0
        for entry in state.log_history:
            if "eval_map_50" in entry:
                best_map = max(best_map, float(entry["eval_map_50"]))
        test_map = None
        for entry in reversed(state.log_history):
            if "test_map_50" in entry:
                test_map = float(entry["test_map_50"]); break

        val_loader = kwargs.get("eval_dataloader")
        val_ds = val_loader.dataset if val_loader is not None else None

        if val_ds is not None:
            indices = self._sample_indices or sorted(random.sample(
                range(len(val_ds)), min(self.num_samples, len(val_ds))))
            rows = self._forward_and_render_rows(model, val_ds, indices, self.best_conf_threshold)
            if rows:
                self._save_grid(rows, self.save_dir / "val_predictions" / "best.png",
                                f"Best checkpoint (val) — mAP50: {best_map:.4f}",
                                self.grid_cols)
                logger.info("HFValPredictionCallback: saved val_predictions/best.png")

            # Val-set error analysis on best checkpoint — same breakdown as
            # test, but scoped to the val split so train/val/test all have
            # matching FP/FN/TP diagnostics.
            try:
                self._write_error_analysis(
                    model, val_ds,
                    self.save_dir / "val_predictions" / "error_analysis.json")
            except Exception as e:
                logger.warning("val error_analysis skipped: %s", e)

        if self.test_dataset is not None and len(self.test_dataset) > 0:
            n = len(self.test_dataset)
            k = min(self.best_num_samples, n)
            test_indices = sorted(random.sample(range(n), k))
            rows = self._forward_and_render_rows(
                model, self.test_dataset, test_indices, self.best_conf_threshold)
            if rows:
                title = (f"Best checkpoint (test) — mAP50: {test_map:.4f}"
                         if test_map is not None else "Best checkpoint (test)")
                self._save_grid(rows, self.save_dir / "test_predictions" / "best.png",
                                title, self.grid_cols)
                logger.info("HFValPredictionCallback: saved test_predictions/best.png")

            try:
                self._write_error_analysis(
                    model, self.test_dataset,
                    self.save_dir / "test_predictions" / "error_analysis.json")
            except Exception as e:
                logger.warning("error_analysis skipped: %s", e)

        if was_training:
            model.train()
        return control

    def _write_error_analysis(self, model, dataset, out_path,
                               iou_threshold=0.5, conf_threshold=0.3):
        """Per-class TP / FP / FN on dataset using best weights. One JSON
        file; ~3s on a 29-image test split.
        """
        import cv2, json, torch
        raw_dataset, idx_map = self._unwrap(dataset)
        device = next(model.parameters()).device
        input_h, input_w = self.input_size

        per_class = {int(cid): {"tp": 0, "fp": 0, "fn": 0} for cid in self.class_names}
        per_image = []

        for i in range(len(dataset)):
            real_idx = idx_map(i)
            raw_img = raw_dataset.get_raw_item(real_idx)["image"]
            resized = cv2.resize(raw_img, (input_w, input_h))
            tensor = torch.from_numpy(np.ascontiguousarray(
                (resized.astype(np.float32) / 255.0).transpose(2, 0, 1))).unsqueeze(0).to(device)
            with torch.no_grad():
                preds_raw = model(pixel_values=tensor)
            decoded = model.postprocess(
                preds_raw, conf_threshold,
                torch.tensor([[input_h, input_w]], device=device))[0]

            gt = raw_dataset._load_label(raw_dataset.img_paths[real_idx])
            if gt is not None and len(gt) > 0:
                cx, cy, w, h = gt[:,1], gt[:,2], gt[:,3], gt[:,4]
                gt_xyxy = np.stack([(cx-w/2)*input_w, (cy-h/2)*input_h,
                                     (cx+w/2)*input_w, (cy+h/2)*input_h], axis=1)
                gt_cls = gt[:,0].astype(np.int64)
            else:
                gt_xyxy = np.zeros((0,4)); gt_cls = np.zeros(0, dtype=np.int64)

            pb = np.asarray(decoded.get("boxes", []), dtype=np.float64).reshape(-1,4)
            pl = np.asarray(decoded.get("labels", []), dtype=np.int64).ravel()
            matched = np.zeros(len(gt_xyxy), dtype=bool)
            img_tp = img_fp = 0
            for bi in range(len(pb)):
                best_iou, best_j = 0.0, -1
                for j in range(len(gt_xyxy)):
                    if matched[j] or gt_cls[j] != pl[bi]: continue
                    xa=max(pb[bi,0],gt_xyxy[j,0]); ya=max(pb[bi,1],gt_xyxy[j,1])
                    xb=min(pb[bi,2],gt_xyxy[j,2]); yb=min(pb[bi,3],gt_xyxy[j,3])
                    inter = max(0, xb-xa) * max(0, yb-ya)
                    union = ((pb[bi,2]-pb[bi,0])*(pb[bi,3]-pb[bi,1]) +
                             (gt_xyxy[j,2]-gt_xyxy[j,0])*(gt_xyxy[j,3]-gt_xyxy[j,1]) - inter)
                    iou = inter/union if union > 0 else 0
                    if iou > best_iou: best_iou, best_j = iou, j
                if best_iou >= iou_threshold:
                    matched[best_j] = True
                    per_class[int(pl[bi])]["tp"] += 1; img_tp += 1
                else:
                    per_class[int(pl[bi])]["fp"] += 1; img_fp += 1
            for j in np.where(~matched)[0]:
                per_class[int(gt_cls[j])]["fn"] += 1
            img_fn = int((~matched).sum())
            per_image.append({
                "idx": int(real_idx),
                "path": str(raw_dataset.img_paths[real_idx]),
                "tp": img_tp, "fp": img_fp, "fn": img_fn,
            })

        summary = {}
        for cid, c in per_class.items():
            tp, fp, fn = c["tp"], c["fp"], c["fn"]
            prec = tp/(tp+fp) if (tp+fp) else 0.0
            rec  = tp/(tp+fn) if (tp+fn) else 0.0
            summary[self.class_names.get(cid, str(cid))] = {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(prec, 4), "recall": round(rec, 4),
            }

        per_image.sort(key=lambda r: -(r["fn"] + r["fp"]))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import json as _json
        with open(out_path, "w") as f:
            _json.dump({
                "iou_threshold": iou_threshold,
                "conf_threshold": conf_threshold,
                "per_class": summary,
                "hardest_10": per_image[:10],
            }, f, indent=2)
        logger.info("HFValPredictionCallback: wrote %s", out_path)

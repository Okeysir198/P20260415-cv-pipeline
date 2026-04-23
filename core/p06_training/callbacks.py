"""Training callbacks for checkpointing, early stopping, and logging.

Provides:
- Callback: Base class with training lifecycle hooks.
- CheckpointSaver: Save best and periodic model checkpoints.
- EarlyStopping: Stop training when metric plateaus.
- WandBLogger: Log metrics, config, and artifacts to Weights & Biases.
- CallbackRunner: Manages and invokes a list of callbacks.
"""

import logging
import math
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import supervision as sv
import torch
import torch.nn as nn

import wandb
from core.p10_inference.supervision_bridge import annotate_gt_pred
from utils.viz import (
    VizStyle,
    annotate_detections,
    save_image_grid,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

logger = logging.getLogger(__name__)

_LABEL_PALETTE = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (0, 128, 255), (128, 0, 255),
    (0, 255, 128), (255, 128, 0), (64, 0, 255), (255, 64, 0),
    (0, 255, 64), (64, 255, 0), (255, 0, 64), (0, 64, 255),
    (128, 128, 0), (0, 128, 128), (128, 0, 128), (64, 64, 0),
]


def _subset_indices(dataset: Any) -> list[int] | None:
    """Return ``list(dataset.indices)`` if dataset is a torch Subset, else None."""
    return list(dataset.indices) if hasattr(dataset, "indices") else None


def _run_splits_and_subsets(trainer: Any) -> dict[str, list[int] | None]:
    """Return ``{split: subset_indices or None}`` for splits actually used in the run.

    The custom pytorch trainer sets ``trainer.train_loader`` and
    ``trainer.val_loader`` (never a test loader during training), so this
    returns train+val only in that path. The HF-Trainer viz bridge also
    supplies a ``test_loader`` stub so test stats/grids are included — the
    reference notebook treats CPPE-5 as 850/150/29 train/val/test, and we
    mirror that here when the third loader is present.
    """
    out: dict[str, list[int] | None] = {}
    for split in ("train", "val", "test"):
        loader = getattr(trainer, f"{split}_loader", None)
        if loader is not None:
            out[split] = _subset_indices(loader.dataset)
    return out


def _draw_gt_boxes(
    image: np.ndarray,
    targets: np.ndarray,
    class_names: dict,
    thickness: int = 2,
    text_scale: float = 0.5,
) -> np.ndarray:
    """Draw normalized CXCYWH GT boxes on a BGR image.

    Thin adapter over :func:`utils.viz.annotate_detections`. Input and output
    are BGR (callers originate from ``cv2.imread`` / ``YOLOXDataset.get_raw_item``
    which is BGR). We convert to RGB at the boundary, annotate, and convert
    back — supervision annotators work in whatever channel order they receive
    but ``class_palette()`` colors are defined as RGB.
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

    style = VizStyle(
        box_thickness=thickness,
        label_text_scale=text_scale,
    )
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    annotated_rgb = annotate_detections(image_rgb, dets, class_names=class_names, style=style)
    return cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)


def _save_image_grid(
    annotated: list[np.ndarray],
    grid_cols: int,
    title: str,
    out_path: Path,
    dpi: int,
) -> None:
    """Tile a list of BGR images into a grid and save as PNG.

    Thin adapter over :func:`utils.viz.save_image_grid`. The helper expects
    RGB, so we convert each cell at the boundary (preserves the previous
    visual output byte-for-byte modulo the matplotlib layout, which is
    also preserved because save_image_grid mirrors the original grid math).
    """
    if not annotated:
        return
    rgb_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in annotated]
    save_image_grid(
        rgb_imgs,
        out_path,
        cols=min(grid_cols, len(rgb_imgs)),
        header=title,
    )


class Callback:
    """Base callback with training lifecycle hooks.

    Override any of these methods to inject custom behavior at
    specific points in the training loop.
    """

    def on_train_start(self, trainer: Any) -> None:
        """Called once at the beginning of training.

        Args:
            trainer: The DetectionTrainer instance.
        """

    def on_train_end(self, trainer: Any) -> None:
        """Called once at the end of training.

        Args:
            trainer: The DetectionTrainer instance.
        """

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        """Called at the beginning of each epoch.

        Args:
            trainer: The DetectionTrainer instance.
            epoch: Current epoch number (0-indexed).
        """

    def on_epoch_end(
        self, trainer: Any, epoch: int, metrics: dict[str, float]
    ) -> None:
        """Called at the end of each epoch.

        Args:
            trainer: The DetectionTrainer instance.
            epoch: Current epoch number (0-indexed).
            metrics: Dictionary of epoch metrics (train + val).
        """

    def on_batch_end(
        self, trainer: Any, batch_idx: int, metrics: dict[str, float]
    ) -> None:
        """Called at the end of each training batch.

        Args:
            trainer: The DetectionTrainer instance.
            batch_idx: Current batch index within the epoch.
            metrics: Dictionary of batch-level metrics.
        """


class CheckpointSaver(Callback):
    """Save model checkpoints: best model and periodic saves.

    Checkpoints include model state dict, optimizer state, scheduler state,
    epoch number, and metrics for seamless training resumption.

    Args:
        save_dir: Directory to save checkpoints.
        metric: Metric key to track for best model (e.g. "val/mAP50").
        mode: "max" if higher metric is better, "min" if lower. Default: "max".
        save_interval: Save periodic checkpoint every N epochs. 0 to disable.
            Default: 10.
        save_best: Whether to save the best model checkpoint. Default: True.
    """

    def __init__(
        self,
        save_dir: str,
        metric: str = "val/mAP50",
        mode: str = "max",
        save_interval: int = 10,
        save_best: bool = True,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.metric = metric
        self.mode = mode
        self.save_interval = save_interval
        self.save_best = save_best

        self._best_value: float | None = None
        self._best_epoch: int = 0

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_train_start(self, trainer: Any) -> None:
        """Copy training + data configs to the run directory for reproducibility."""
        import shutil

        import yaml

        config_path = getattr(trainer, "config_path", None)

        # Copy the original training YAML (named to match releases/ convention)
        if config_path is not None and Path(config_path).exists():
            shutil.copy2(config_path, self.save_dir / "06_training.yaml")
            logger.info("Saved training config to %s", self.save_dir / "06_training.yaml")

        # Copy the data config YAML (resolved from dataset_config reference)
        data_cfg = getattr(trainer, "_data_cfg", None)
        if data_cfg and config_path is not None:
            dataset_config_ref = data_cfg.get("dataset_config")
            if dataset_config_ref:
                data_config_path = Path(dataset_config_ref)
                if not data_config_path.is_absolute():
                    data_config_path = (Path(config_path).parent / dataset_config_ref).resolve()
                if data_config_path.exists():
                    shutil.copy2(data_config_path, self.save_dir / "05_data.yaml")
                    logger.info("Saved data config to %s", self.save_dir / "05_data.yaml")

        # Dump the resolved config (with CLI overrides applied)
        config = getattr(trainer, "config", None)
        if config is not None:
            resolved_path = self.save_dir / "config_resolved.yaml"
            with open(resolved_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def on_epoch_end(
        self, trainer: Any, epoch: int, metrics: dict[str, float]
    ) -> None:
        """Save checkpoints at epoch end.

        Args:
            trainer: The DetectionTrainer instance.
            epoch: Current epoch number (0-indexed).
            metrics: Dictionary of epoch metrics.
        """
        checkpoint = self._build_checkpoint(trainer, epoch, metrics)

        # Save best model
        if self.save_best and self.metric in metrics:
            current_value = metrics[self.metric]
            is_best = False

            if self._best_value is None:
                is_best = True
            elif self.mode == "max" and current_value > self._best_value:
                is_best = True
            elif self.mode == "min" and current_value < self._best_value:
                is_best = True

            if is_best:
                self._best_value = current_value
                self._best_epoch = epoch
                best_path = self.save_dir / "best.pth"
                torch.save(checkpoint, best_path)
                logger.info(
                    "Saved best model: %s=%.4f at epoch %d -> %s",
                    self.metric, current_value, epoch + 1, best_path,
                )

        # Save periodic checkpoint
        if self.save_interval > 0 and (epoch + 1) % self.save_interval == 0:
            periodic_path = self.save_dir / f"epoch_{epoch + 1}.pth"
            torch.save(checkpoint, periodic_path)
            logger.info("Saved periodic checkpoint: epoch %d -> %s", epoch + 1, periodic_path)

        # Always save last checkpoint for resume
        last_path = self.save_dir / "last.pth"
        torch.save(checkpoint, last_path)

    @staticmethod
    def _build_checkpoint(
        trainer: Any, epoch: int, metrics: dict[str, float]
    ) -> dict:
        """Build a checkpoint dictionary from trainer state.

        Args:
            trainer: The DetectionTrainer instance.
            epoch: Current epoch number.
            metrics: Current epoch metrics.

        Returns:
            Checkpoint dictionary.
        """
        checkpoint: dict[str, Any] = {
            "epoch": epoch,
            "metrics": metrics,
            "config": trainer.config,
        }

        # Model architecture metadata for reproducibility
        if hasattr(trainer, "_model_cfg") and trainer._model_cfg is not None:
            checkpoint["model_arch"] = {
                "depth": trainer._model_cfg.get("depth"),
                "width": trainer._model_cfg.get("width"),
                "num_classes": trainer._model_cfg.get("num_classes"),
                "arch": trainer._model_cfg.get("arch"),
                "input_size": trainer._model_cfg.get("input_size"),
            }
        # Output format for postprocessor dispatch
        if hasattr(trainer, "model") and trainer.model is not None:
            base = trainer.model
            if isinstance(base, nn.DataParallel):
                base = base.module
            checkpoint["output_format"] = getattr(base, "output_format", "yolox")

        if hasattr(trainer, "_data_cfg") and trainer._data_cfg is not None:
            checkpoint["dataset_name"] = trainer._data_cfg.get(
                "dataset_config", trainer._data_cfg.get("dataset_name", "unknown")
            )

        # Model state
        if hasattr(trainer, "model") and trainer.model is not None:
            model = trainer.model
            if isinstance(model, nn.DataParallel):
                checkpoint["model_state_dict"] = model.module.state_dict()
            else:
                checkpoint["model_state_dict"] = model.state_dict()

        # Optimizer state
        if hasattr(trainer, "optimizer") and trainer.optimizer is not None:
            checkpoint["optimizer_state_dict"] = trainer.optimizer.state_dict()

        # Scheduler state
        if hasattr(trainer, "scheduler") and trainer.scheduler is not None:
            if hasattr(trainer.scheduler, "state_dict"):
                checkpoint["scheduler_state_dict"] = trainer.scheduler.state_dict()

        # Scaler state (AMP)
        if hasattr(trainer, "scaler") and trainer.scaler is not None:
            checkpoint["scaler_state_dict"] = trainer.scaler.state_dict()

        # EMA state
        if hasattr(trainer, "ema") and trainer.ema is not None:
            checkpoint["ema_state_dict"] = trainer.ema.state_dict()

        # RNG states for exact reproducibility on resume
        import random as _random

        import numpy as _np

        checkpoint["rng_states"] = {
            "python": _random.getstate(),
            "numpy": _np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
        }

        # Callback states for correct resume behavior
        callback_runner = getattr(trainer, "callback_runner", None)
        if callback_runner is not None:
            cb_states = {}
            ckpt_cb = callback_runner.get_callback(CheckpointSaver)
            if ckpt_cb is not None:
                cb_states["checkpoint_saver"] = {
                    "best_value": ckpt_cb._best_value,
                    "best_epoch": ckpt_cb._best_epoch,
                }
            es_cb = callback_runner.get_callback(EarlyStopping)
            if es_cb is not None:
                cb_states["early_stopping"] = {
                    "best_value": es_cb._best_value,
                    "counter": es_cb._counter,
                }
            if cb_states:
                checkpoint["callback_states"] = cb_states

        return checkpoint


class EarlyStopping(Callback):
    """Stop training when a monitored metric stops improving.

    Args:
        metric: Metric key to monitor (e.g. "val/mAP50").
        mode: "max" if higher is better, "min" if lower. Default: "max".
        patience: Number of epochs to wait for improvement. Default: 50.
        min_delta: Minimum change to qualify as improvement. Default: 0.0.
    """

    def __init__(
        self,
        metric: str = "val/mAP50",
        mode: str = "max",
        patience: int = 50,
        min_delta: float = 0.0,
    ) -> None:
        self.metric = metric
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta

        self._best_value: float | None = None
        self._counter: int = 0
        self._best_epoch: int = 0
        self.should_stop: bool = False

    def on_epoch_end(
        self, trainer: Any, epoch: int, metrics: dict[str, float]
    ) -> None:
        """Check if training should stop.

        Sets self.should_stop = True if patience is exhausted.

        Args:
            trainer: The DetectionTrainer instance.
            epoch: Current epoch number (0-indexed).
            metrics: Dictionary of epoch metrics.
        """
        if self.metric not in metrics:
            return

        current_value = metrics[self.metric]
        improved = False

        if self._best_value is None:
            improved = True
        elif self.mode == "max" and current_value > self._best_value + self.min_delta:
            improved = True
        elif self.mode == "min" and current_value < self._best_value - self.min_delta:
            improved = True

        if improved:
            self._best_value = current_value
            self._best_epoch = epoch
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "Early stopping triggered at epoch %d. "
                    "Best %s=%.4f at epoch %d. No improvement for %d epochs.",
                    epoch + 1, self.metric, self._best_value,
                    self._best_epoch + 1, self.patience,
                )


class WandBLogger(Callback):
    """Log training metrics, config, and artifacts to Weights & Biases.

    Handles wandb initialization, per-epoch metric logging, and cleanup.
    Gracefully degrades if wandb is not installed or login fails.

    Args:
        project: W&B project name. Default: "smart-camera".
        run_name: Name for this training run. Default: None (auto-generated).
        config: Full training config dict to log. Default: None.
        log_interval: Log batch metrics every N batches. 0 for epoch-only. Default: 0.
        tags: Optional list of tags for the run.
    """

    def __init__(
        self,
        project: str = "smart-camera",
        run_name: str | None = None,
        config: dict | None = None,
        log_interval: int = 0,
        tags: list[str] | None = None,
    ) -> None:
        self.project = project
        self.run_name = run_name
        self._config = config
        self.log_interval = log_interval
        self.tags = tags
        self._run = None
        self._wandb = None
        self._enabled = False

    def on_train_start(self, trainer: Any) -> None:
        """Initialize W&B run.

        Args:
            trainer: The DetectionTrainer instance.
        """
        try:
            self._wandb = wandb
            self._run = wandb.init(
                project=self.project,
                name=self.run_name,
                config=self._config or trainer.config,
                tags=self.tags,
                reinit="finish_previous",
            )
            self._enabled = True
            logger.info("W&B logging enabled: project=%s, run=%s", self.project, self.run_name)
        except Exception as e:
            logger.warning("Failed to initialize W&B: %s. Continuing without logging.", e)

    def on_epoch_end(
        self, trainer: Any, epoch: int, metrics: dict[str, float]
    ) -> None:
        """Log epoch-level metrics to W&B.

        Args:
            trainer: The DetectionTrainer instance.
            epoch: Current epoch number (0-indexed).
            metrics: Dictionary of epoch metrics.
        """
        if not self._enabled:
            return

        # Add epoch and learning rate info
        log_data = {"epoch": epoch + 1, **metrics}

        if hasattr(trainer, "optimizer") and trainer.optimizer is not None:
            log_data["lr"] = trainer.optimizer.param_groups[0]["lr"]

        self._wandb.log(log_data, step=epoch + 1)

    def on_batch_end(
        self, trainer: Any, batch_idx: int, metrics: dict[str, float]
    ) -> None:
        """Log batch-level metrics to W&B (if log_interval > 0).

        Args:
            trainer: The DetectionTrainer instance.
            batch_idx: Current batch index within the epoch.
            metrics: Dictionary of batch-level metrics.
        """
        if not self._enabled or self.log_interval <= 0:
            return

        if (batch_idx + 1) % self.log_interval == 0:
            self._wandb.log(metrics)

    def on_train_end(self, trainer: Any) -> None:
        """Finalize and close the W&B run.

        Args:
            trainer: The DetectionTrainer instance.
        """
        if self._enabled and self._run is not None:
            self._run.finish()
            logger.info("W&B run finished.")
            self._enabled = False


class ValPredictionLogger(Callback):
    """Save a grid of predictions each epoch for a given split (val or train).

    GT boxes: solid purple (thickness=2) drawn first.
    Pred boxes: solid green (thickness=1) drawn on top.
    Saved to ``<save_dir>/{split}_predictions/epoch_<N>.png``.
    """

    _DEFAULT_GT_COLOR = sv.Color(r=160, g=32, b=240)   # purple
    _DEFAULT_PRED_COLOR = sv.Color(r=0, g=200, b=0)    # green

    def __init__(
        self,
        save_dir: str,
        split: str = "val",
        num_samples: int = 12,
        conf_threshold: float = 0.05,
        grid_cols: int = 2,
        gt_thickness: int = 2,
        pred_thickness: int = 1,
        gt_color_rgb: tuple = (160, 32, 240),
        pred_color_rgb: tuple = (0, 200, 0),
        text_scale: float = 0.4,
        dpi: int = 150,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.split = split
        self.num_samples = num_samples
        self.conf_threshold = conf_threshold
        self.grid_cols = grid_cols
        self.gt_thickness = gt_thickness
        self.pred_thickness = pred_thickness
        self.gt_color = sv.Color(r=gt_color_rgb[0], g=gt_color_rgb[1], b=gt_color_rgb[2])
        self.pred_color = sv.Color(r=pred_color_rgb[0], g=pred_color_rgb[1], b=pred_color_rgb[2])
        self.text_scale = text_scale
        self.dpi = dpi
        self._sample_indices: list[int] | None = None

    @staticmethod
    def _unwrap_subset(dataset: Any):
        """Return (underlying_dataset, index_fn) unwrapping torch Subset if needed."""
        if hasattr(dataset, "indices"):  # torch.utils.data.Subset
            indices = dataset.indices
            return dataset.dataset, lambda i: indices[i]
        return dataset, lambda i: i

    def _get_loader(self, trainer: Any):
        return trainer.train_loader if self.split == "train" else trainer.val_loader

    def on_train_start(self, trainer: Any) -> None:
        dataset = self._get_loader(trainer).dataset
        n = len(dataset)
        k = min(self.num_samples, n)
        self._sample_indices = sorted(random.sample(range(n), k))
        logger.info("ValPredictionLogger(%s): sampled %d images for visualization", self.split, k)

    def on_epoch_end(
        self, trainer: Any, epoch: int, metrics: dict[str, float]
    ) -> None:
        if self._sample_indices is None:
            return

        raw_dataset, idx_map = self._unwrap_subset(self._get_loader(trainer).dataset)
        model = trainer.model
        device = trainer.device
        model.eval()
        input_h, input_w = trainer._model_cfg["input_size"]
        loaded_cfg = getattr(trainer, "_loaded_data_cfg", trainer._data_cfg)
        class_names = {int(k): str(v) for k, v in loaded_cfg.get("names", {}).items()}

        # --- Phase 1: load images + prepare input tensors (CPU) ---
        samples = []  # (subset_idx, orig_bgr, input_tensor_chw_float)
        for idx in self._sample_indices:
            real_idx = idx_map(idx)
            img_path = raw_dataset.img_paths[real_idx]
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            raw_img = raw_dataset.get_raw_item(real_idx)["image"]
            if isinstance(raw_img, np.ndarray):
                resized = cv2.resize(raw_img, (input_w, input_h))
                tensor = torch.from_numpy(
                    np.ascontiguousarray((resized.astype(np.float32) / 255.0).transpose(2, 0, 1))
                )
            else:
                tensor = raw_img
            samples.append((real_idx, image, tensor))

        if not samples:
            return

        # --- Phase 2: single batched GPU forward pass ---
        batch = torch.stack([t for _, _, t in samples]).to(device)
        with torch.no_grad():
            all_preds = model(batch)
        all_decoded = trainer._decode_predictions(all_preds, conf_threshold=self.conf_threshold)

        # --- Phase 3: draw GT + pred via shared supervision bridge (CPU) ---
        rows = []
        for i, (real_idx, image, _) in enumerate(samples):
            orig_h, orig_w = image.shape[:2]

            # Ground truth boxes (YOLO normalized → pixel xyxy)
            gt_xyxy, gt_class_ids = None, None
            gt_targets = raw_dataset._load_label(raw_dataset.img_paths[real_idx])
            if gt_targets is not None and len(gt_targets) > 0:
                gt_np = gt_targets if isinstance(gt_targets, np.ndarray) else np.array(gt_targets)
                cx, cy, w, h = gt_np[:, 1], gt_np[:, 2], gt_np[:, 3], gt_np[:, 4]
                gt_xyxy = np.stack([
                    (cx - w / 2) * orig_w, (cy - h / 2) * orig_h,
                    (cx + w / 2) * orig_w, (cy + h / 2) * orig_h,
                ], axis=1)
                gt_class_ids = gt_np[:, 0].astype(np.int64)

            # Predictions (scale from model input size to original image size)
            pred = all_decoded[i] if i < len(all_decoded) else {}
            pred_boxes = np.asarray(pred.get("boxes", []), dtype=np.float64).reshape(-1, 4)
            pred_labels = np.asarray(pred.get("labels", []), dtype=np.int64).ravel()
            pred_scores = np.asarray(pred.get("scores", []), dtype=np.float64).ravel()
            if pred_boxes.shape[0] > 0:
                pred_boxes[:, [0, 2]] *= orig_w / input_w
                pred_boxes[:, [1, 3]] *= orig_h / input_h
            pred_dets = sv.Detections(xyxy=pred_boxes, class_id=pred_labels, confidence=pred_scores)

            rows.append(annotate_gt_pred(
                image, gt_xyxy, gt_class_ids, pred_dets, class_names,
                gt_color=self.gt_color, pred_color=self.pred_color,
                gt_thickness=self.gt_thickness, pred_thickness=self.pred_thickness,
                text_scale=self.text_scale, draw_legend=True,
            ))

        if not rows:
            return

        ncols = self.grid_cols
        nrows = math.ceil(len(rows) / ncols)
        map_val = metrics.get("val/mAP50", 0.0)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5))
        axes = np.asarray(axes).ravel()
        for i in range(nrows * ncols):
            axes[i].axis("off")
            if i < len(rows):
                axes[i].imshow(cv2.cvtColor(rows[i], cv2.COLOR_BGR2RGB))
        fig.suptitle(f"Epoch {epoch + 1} — mAP50: {map_val:.4f}", fontsize=14)
        fig.tight_layout()

        # Per-epoch grids go under <split>_predictions/epochs/ so best.png
        # and error_analysis/ (post-train) stay at the split root alongside.
        out_dir = self.save_dir / f"{self.split}_predictions" / "epochs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"epoch_{epoch + 1:03d}.png"
        fig.savefig(str(out_path), dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s prediction grid: %s", self.split, out_path)

        # Log to wandb if available
        cb_runner = getattr(trainer, "callback_runner", None)
        if cb_runner is not None:
            wb_cb = cb_runner.get_callback(WandBLogger)
            if wb_cb is not None and wb_cb._enabled:
                wb_cb._wandb.log(
                    {f"{self.split}/predictions": wandb.Image(str(out_path))},
                    step=epoch + 1,
                )



class DatasetStatsLogger(Callback):
    """Save 00_dataset_info.{md,json} + 01_dataset_stats.{png,json} once before training.

    Always runs — not conditional on data_viz.enabled.
    """

    def __init__(
        self,
        save_dir: str,
        data_config: dict,
        base_dir: str,
        splits: list[str],
        dpi: int = 120,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.data_config = data_config
        self.base_dir = base_dir
        self.splits = splits
        self.dpi = dpi

    def on_train_start(self, trainer: Any, **kwargs: Any) -> None:
        class_names = {int(k): str(v) for k, v in self.data_config.get("names", {}).items()}
        out_dir = self.save_dir / "data_preview"
        try:
            from core.p05_data.run_viz import (
                _load_cached_stats,
                generate_dataset_stats,
                write_dataset_info,
            )
            # Restrict stats to splits the run actually uses (train+val, with
            # data.subset.* applied) so data_preview mirrors the run's data.
            run_splits = _run_splits_and_subsets(trainer)
            active_splits = [s for s in self.splits if s in run_splits]
            subset_map = {s: run_splits[s] for s in active_splits}

            # 00_dataset_info.{md,json} — always written (cheap, self-describing).
            try:
                split_sizes = {
                    s: (len(idxs) if idxs is not None else 0)
                    for s, idxs in subset_map.items()
                }
                write_dataset_info(
                    out_dir,
                    feature_name=getattr(trainer, "_feature_name", None),
                    data_config_path=getattr(trainer, "_data_config_path", None),
                    training_config_path=getattr(trainer, "config_path", None),
                    data_cfg=self.data_config,
                    training_cfg=getattr(trainer, "config", None),
                    class_names=class_names,
                    split_sizes=split_sizes,
                )
            except Exception as e:
                logger.warning("DatasetStatsLogger: write_dataset_info failed — %s", e)

            if _load_cached_stats(out_dir):
                logger.info("DatasetStatsLogger: cache hit — skipping recompute (%s)", out_dir)
                return
            generate_dataset_stats(
                self.data_config, self.base_dir, class_names,
                active_splits, out_dir, self.dpi,
                subset_indices=subset_map,
            )
            stats_path = out_dir / "01_dataset_stats.png"
            wb_cb = next((c for c in trainer.callback_runner.callbacks if isinstance(c, WandBLogger)), None)
            if wb_cb is not None and wb_cb._enabled:
                wb_cb._wandb.log({"viz/dataset_stats": wandb.Image(str(stats_path))})
        except Exception as e:
            logger.warning("DatasetStatsLogger: dataset stats failed — %s", e)


class DataLabelGridLogger(Callback):
    """Save a grid of raw images with GT labels for each configured split.

    Runs once in on_train_start — never per epoch.
    Output: <save_dir>/data_preview/02_data_labels_<split>.png
    """

    def __init__(
        self,
        save_dir: str,
        splits: list[str],
        data_config: dict,
        base_dir: str,
        num_samples: int = 16,
        grid_cols: int = 4,
        thickness: int = 2,
        text_scale: float = 0.4,
        dpi: int = 120,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.splits = splits
        self.data_config = data_config
        self.base_dir = base_dir
        self.num_samples = num_samples
        self.grid_cols = grid_cols
        self.thickness = thickness
        self.text_scale = text_scale
        self.dpi = dpi

    def on_train_start(self, trainer: Any) -> None:
        from core.p05_data.detection_dataset import YOLOXDataset
        class_names = {int(k): str(v) for k, v in self.data_config.get("names", {}).items()}

        # Restrict to splits the run actually uses so samples reflect the
        # data.subset.* filtering rather than the full split on disk.
        run_splits = _run_splits_and_subsets(trainer)

        for split in self.splits:
            if split not in run_splits:
                logger.info("DataLabelGridLogger: skip split %s (not used by this run)", split)
                continue
            try:
                ds = YOLOXDataset(
                    data_config=self.data_config,
                    split=split,
                    transforms=None,
                    base_dir=self.base_dir,
                )
            except Exception as e:
                logger.warning("DataLabelGridLogger: skip split %s — %s", split, e)
                continue

            subset = run_splits[split]
            pool = list(range(len(ds))) if subset is None else list(subset)
            n = len(pool)
            if n == 0:
                logger.warning("DataLabelGridLogger: split %s is empty, skipping", split)
                continue

            k = min(self.num_samples, n)
            indices = sorted(random.sample(pool, k))

            annotated = []
            for idx in indices:
                item = ds.get_raw_item(idx)
                img = item["image"]  # HWC BGR uint8
                targets = ds._load_label(ds.img_paths[idx])
                if targets is None or len(targets) == 0:
                    targets = np.zeros((0, 5), dtype=np.float32)
                annotated.append(_draw_gt_boxes(img, targets, class_names, self.thickness, self.text_scale))

            if not annotated:
                continue

            out_path = self.save_dir / "data_preview" / f"02_data_labels_{split}.png"
            _save_image_grid(
                annotated, self.grid_cols,
                f"Data + Labels [{split}] — {k} samples",
                out_path, self.dpi,
            )
            logger.info("Saved data label grid (%s): %s", split, out_path)

            # Log to WandB if available
            cb_runner = getattr(trainer, "callback_runner", None)
            if cb_runner is not None:
                wb_cb = cb_runner.get_callback(WandBLogger)
                if wb_cb is not None and wb_cb._enabled:
                    wb_cb._wandb.log({f"viz/data_labels_{split}": wandb.Image(str(out_path))})


class AugLabelGridLogger(Callback):
    """Save a grid of augmented training images with transformed GT labels.

    Mosaic/MixUp/CopyPaste are disabled so each cell shows a single identifiable
    image — makes flip/HSV/affine parameters visually verifiable.

    Runs once in on_train_start — never per epoch.
    Output: <save_dir>/data_preview/03_aug_labels_<split>.png
    """

    def __init__(
        self,
        save_dir: str,
        splits: list[str],
        data_config: dict,
        aug_config: dict,
        base_dir: str,
        num_samples: int = 16,
        grid_cols: int = 4,
        thickness: int = 2,
        text_scale: float = 0.4,
        dpi: int = 120,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.splits = splits
        self.data_config = data_config
        self.aug_config = aug_config
        self.base_dir = base_dir
        self.num_samples = num_samples
        self.grid_cols = grid_cols
        self.thickness = thickness
        self.text_scale = text_scale
        self.dpi = dpi

    def on_train_start(self, trainer: Any) -> None:
        from core.p05_data.detection_dataset import YOLOXDataset
        from core.p05_data.transforms import build_transforms

        class_names = {int(k): str(v) for k, v in self.data_config.get("names", {}).items()}
        mean = np.array(self.data_config.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32).reshape(1, 1, 3)
        std = np.array(self.data_config.get("std", [0.229, 0.224, 0.225]), dtype=np.float32).reshape(1, 1, 3)
        input_size = tuple(trainer._model_cfg["input_size"])

        # Two passes: simple (no mosaic/mixup) and mosaic (full augmentation).
        # Skip mosaic pass if mosaic is disabled in config — avoids slow CPU mosaic for DETR-family.
        aug_variants = [
            ("simple", {**self.aug_config, "mosaic": False, "mixup": False, "copypaste": False}),
        ]
        if self.aug_config.get("mosaic", False):
            aug_variants.append(("mosaic", self.aug_config))

        run_splits = _run_splits_and_subsets(trainer)

        for split in self.splits:
            if split != "train":
                logger.info("AugLabelGridLogger: skip split %s (no augmentation)", split)
                continue
            if split not in run_splits:
                logger.info("AugLabelGridLogger: skip split %s (not used by this run)", split)
                continue
            subset = run_splits[split]

            for label, aug_cfg in aug_variants:
                transforms = build_transforms(
                    config=aug_cfg, is_train=True, input_size=input_size,
                    mean=self.data_config.get("mean"), std=self.data_config.get("std"),
                )
                try:
                    ds = YOLOXDataset(
                        data_config=self.data_config, split=split,
                        transforms=transforms, base_dir=self.base_dir,
                    )
                except Exception as e:
                    logger.warning("AugLabelGridLogger: skip %s/%s — %s", split, label, e)
                    continue

                pool = list(range(len(ds))) if subset is None else list(subset)
                n = len(pool)
                if n == 0:
                    continue

                k = min(self.num_samples, n)
                indices = sorted(random.sample(pool, k))

                annotated = []
                for i in indices:
                    try:
                        result = ds[i]
                        aug_tensor, targets_tensor = result[0], result[1]
                    except Exception as e:
                        logger.warning("AugLabelGridLogger: failed idx %d — %s", i, e)
                        continue
                    aug_np = aug_tensor.numpy().transpose(1, 2, 0)
                    aug_np = np.clip(aug_np * std + mean, 0, 1)
                    aug_bgr = (aug_np[:, :, ::-1] * 255).astype(np.uint8)
                    targets_np = targets_tensor.numpy() if len(targets_tensor) > 0 else np.zeros((0, 5), dtype=np.float32)
                    annotated.append(_draw_gt_boxes(aug_bgr, targets_np, class_names, self.thickness, self.text_scale))

                if not annotated:
                    continue

                suffix = "" if label == "simple" else "_mosaic"
                out_path = self.save_dir / "data_preview" / f"03_aug_labels_{split}{suffix}.png"
                desc = "no mosaic/mixup" if label == "simple" else "with mosaic/mixup"
                _save_image_grid(
                    annotated, self.grid_cols,
                    f"Augmented + Labels [{split}] ({desc}) — {k} samples",
                    out_path, self.dpi,
                )
                logger.info("Saved aug label grid (%s/%s): %s", split, label, out_path)

                cb_runner = getattr(trainer, "callback_runner", None)
                if cb_runner is not None:
                    wb_cb = cb_runner.get_callback(WandBLogger)
                    if wb_cb is not None and wb_cb._enabled:
                        wb_cb._wandb.log({f"viz/aug_labels_{split}_{label}": wandb.Image(str(out_path))})


class CallbackRunner:
    """Manages and invokes a list of training callbacks.

    Iterates through registered callbacks and calls the appropriate
    hook method on each.

    Args:
        callbacks: List of Callback instances.
    """

    def __init__(self, callbacks: list[Callback] | None = None) -> None:
        self.callbacks: list[Callback] = callbacks or []

    def add(self, callback: Callback) -> None:
        """Register a new callback.

        Args:
            callback: Callback instance to add.
        """
        self.callbacks.append(callback)

    def on_train_start(self, trainer: Any) -> None:
        """Fire on_train_start for all callbacks.

        Args:
            trainer: The DetectionTrainer instance.
        """
        for cb in self.callbacks:
            cb.on_train_start(trainer)

    def on_train_end(self, trainer: Any) -> None:
        """Fire on_train_end for all callbacks.

        Args:
            trainer: The DetectionTrainer instance.
        """
        for cb in self.callbacks:
            cb.on_train_end(trainer)

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        """Fire on_epoch_start for all callbacks.

        Args:
            trainer: The DetectionTrainer instance.
            epoch: Current epoch number.
        """
        for cb in self.callbacks:
            cb.on_epoch_start(trainer, epoch)

    def on_epoch_end(
        self, trainer: Any, epoch: int, metrics: dict[str, float]
    ) -> None:
        """Fire on_epoch_end for all callbacks.

        Args:
            trainer: The DetectionTrainer instance.
            epoch: Current epoch number.
            metrics: Dictionary of epoch metrics.
        """
        for cb in self.callbacks:
            cb.on_epoch_end(trainer, epoch, metrics)

    def on_batch_end(
        self, trainer: Any, batch_idx: int, metrics: dict[str, float]
    ) -> None:
        """Fire on_batch_end for all callbacks.

        Args:
            trainer: The DetectionTrainer instance.
            batch_idx: Current batch index.
            metrics: Dictionary of batch-level metrics.
        """
        for cb in self.callbacks:
            cb.on_batch_end(trainer, batch_idx, metrics)

    def get_callback(self, callback_type: type) -> Callback | None:
        """Find a callback by type.

        Args:
            callback_type: The class type to search for.

        Returns:
            First matching callback instance, or None.
        """
        for cb in self.callbacks:
            if isinstance(cb, callback_type):
                return cb
        return None

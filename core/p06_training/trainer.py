"""Config-driven object detection training loop.

Handles end-to-end training: model construction, optimizer/scheduler setup,
data loading, train/val loops, checkpointing, logging, and early stopping.
All hyperparameters are read from YAML config — no hardcoded values.

Usage:
    trainer = DetectionTrainer(config_path="features/safety-fire_detection/configs/06_training.yaml")
    trainer.train()
"""

import copy
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from core.p05_data.detection_dataset import build_dataloader as _build_detection_dataloader
from core.p05_data.transforms import build_gpu_transforms
from core.p06_models import build_model
from core.p06_training.callbacks import (
    AugLabelGridLogger,
    CallbackRunner,
    CheckpointSaver,
    DataLabelGridLogger,
    DatasetStatsLogger,
    EarlyStopping,
    ValPredictionLogger,
    WandBLogger,
)
from core.p06_training.losses import build_loss
from core.p06_training.lr_scheduler import build_scheduler
from core.p06_training.metrics_registry import compute_metrics
from core.p06_training.postprocess import postprocess as _postprocess_registry
from core.p06_training._common import task_from_output_format as _task_from_output
from utils.config import feature_name_from_config_path, load_config, merge_configs, validate_config
from utils.device import get_device, set_seed
from utils.progress import TrainingProgress

logger = logging.getLogger(__name__)

_LOSS_ZERO = torch.tensor(0.0)


# ---------------------------------------------------------------------------
# Exponential Moving Average (EMA)
# ---------------------------------------------------------------------------


class ModelEMA:
    """Exponential Moving Average of model parameters for stable evaluation.

    Maintains a shadow copy of model weights updated with exponential decay
    after each optimizer step. The EMA model typically yields +1-2% mAP.

    Args:
        model: The model to track.
        decay: Base EMA decay rate. Default: 0.9998.
        warmup_steps: Number of steps for decay warmup. Default: 2000.
    """

    def __init__(
        self, model: nn.Module, decay: float = 0.9998, warmup_steps: int = 2000
    ) -> None:
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.updates = 0

    def update(self, model: nn.Module) -> None:
        """Update EMA parameters and copy BN buffers from the training model."""
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.warmup_steps))
        with torch.no_grad():
            model_params = dict(model.named_parameters())
            for name, ema_param in self.ema_model.named_parameters():
                if name in model_params:
                    ema_param.mul_(d).add_(model_params[name].data, alpha=1 - d)
            # Copy BN running stats (buffers) directly — they must not be EMA-averaged
            model_buffers = dict(model.named_buffers())
            for name, ema_buf in self.ema_model.named_buffers():
                if name in model_buffers:
                    ema_buf.copy_(model_buffers[name])

    def state_dict(self) -> dict:
        return {
            "ema_model": self.ema_model.state_dict(),
            "decay": self.decay,
            "updates": self.updates,
        }

    def load_state_dict(self, state: dict) -> None:
        self.ema_model.load_state_dict(state["ema_model"])
        self.decay = state.get("decay", self.decay)
        self.updates = state.get("updates", 0)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class DetectionTrainer:
    """Config-driven detection model training loop.

    Handles: model building, optimizer setup, data loading,
    train/val loops, checkpointing, logging, early stopping.

    All hyperparameters come from a YAML config file — no hardcoded training
    parameters.

    Args:
        config_path: Path to the training YAML config file.
        overrides: Optional dictionary of config overrides (key=value).
    """

    def __init__(self, config_path: str, overrides: dict | None = None) -> None:
        self.config_path = Path(config_path)
        self.config = load_config(config_path)

        if overrides:
            self.config = merge_configs(self.config, overrides)

        validate_config(self.config, "training")

        # Convenience accessors
        self._model_cfg = self.config["model"]
        self._train_cfg = self.config["training"]
        self._data_cfg = self.config["data"]
        self._aug_cfg = self.config.get("augmentation", {})
        self._log_cfg = self.config.get("logging", {})
        self._ckpt_cfg = self.config.get("checkpoint", {})

        # Seed
        seed = self.config.get("seed", 42)
        set_seed(seed)

        # Device
        self.device = get_device(self.config.get("device"))
        logger.info("Using device: %s", self.device)

        # Placeholders (built in train())
        self.model: nn.Module | None = None
        self.optimizer: optim.Optimizer | None = None
        self.scheduler: Any | None = None
        self.scaler: torch.amp.GradScaler | None = None
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self.gpu_transform: Any | None = None
        self.loss_fn: nn.Module | None = None
        self.callback_runner: CallbackRunner | None = None
        self.ema: ModelEMA | None = None
        self._start_epoch: int = 0

    @property
    def _base_model(self) -> nn.Module:
        """Unwrap DataParallel if present."""
        m = self.model
        return m.module if isinstance(m, nn.DataParallel) else m

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------

    def _build_model(self) -> nn.Module:
        """Build detection model from config via the model registry.

        Uses ``models.build_model`` to instantiate the architecture specified
        by ``config["model"]["arch"]``.  Loads pretrained weights if a path
        is provided in the config.

        Returns:
            Model moved to the configured device.
        """
        model = build_model(self.config)

        # Load pretrained weights if specified
        pretrained_path = self._model_cfg.get("pretrained")
        if pretrained_path:
            pretrained_path = Path(pretrained_path)
            if not pretrained_path.is_absolute():
                pretrained_path = (self.config_path.parent / pretrained_path).resolve()

            if pretrained_path.exists():
                state = torch.load(pretrained_path, map_location=self.device, weights_only=False)
                if "model" in state:
                    state = state["model"]
                elif "model_state_dict" in state:
                    state = state["model_state_dict"]

                # Load with strict=False to handle num_classes mismatch
                missing, unexpected = model.load_state_dict(state, strict=False)
                if missing:
                    logger.info("Pretrained weights: %d missing keys (expected for head).", len(missing))
                if unexpected:
                    logger.info("Pretrained weights: %d unexpected keys.", len(unexpected))
                logger.info("Loaded pretrained weights from %s", pretrained_path)
            else:
                logger.warning("Pretrained weights not found at %s. Training from scratch.", pretrained_path)

        model = model.to(self.device)

        # DataParallel if multiple GPUs
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            logger.info("Using DataParallel with %d GPUs.", torch.cuda.device_count())

        return model

    def _freeze_layers(self, freeze_list: list) -> None:
        """Freeze model components by name.

        Sets ``requires_grad_(False)`` on all parameters whose name starts
        with one of the component names in *freeze_list*.

        Args:
            freeze_list: Component names to freeze, e.g. ``["backbone"]``
                or ``["backbone", "neck"]``.
        """
        base_model = self._base_model

        frozen_count = 0
        for name, param in base_model.named_parameters():
            for component in freeze_list:
                if name.startswith(component + ".") or name.startswith(component + "["):
                    param.requires_grad_(False)
                    frozen_count += 1
                    break

        trainable = sum(1 for p in base_model.parameters() if p.requires_grad)
        logger.info(
            "Froze %d parameters in %s. Remaining trainable: %d.",
            frozen_count, freeze_list, trainable,
        )

    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer with optional per-component LR scaling.

        If the model provides ``get_param_groups()`` (e.g. YOLOX with
        backbone/neck/head split), uses per-component groups with
        ``backbone_lr_scale`` and ``neck_lr_scale`` from config.
        Otherwise falls back to a flat decay/no-decay split.

        Returns:
            Configured optimizer.

        Raises:
            ValueError: If unknown optimizer type is specified.
        """
        opt_type = self._train_cfg.get("optimizer", "sgd").lower()
        lr = self._train_cfg["lr"]
        weight_decay = self._train_cfg.get("weight_decay", 0.0005)
        momentum = self._train_cfg.get("momentum", 0.9)
        backbone_lr_scale = self._train_cfg.get("backbone_lr_scale", 1.0)
        neck_lr_scale = self._train_cfg.get("neck_lr_scale", 1.0)

        base_model = self._base_model

        if hasattr(base_model, "get_param_groups"):
            param_groups = base_model.get_param_groups(lr, weight_decay)
            # Apply per-component LR scaling
            for pg in param_groups:
                group_name = pg.get("group_name", "")
                if "backbone" in group_name:
                    pg["lr"] = lr * backbone_lr_scale
                elif "neck" in group_name:
                    pg["lr"] = lr * neck_lr_scale
                # head keeps base lr
        else:
            # Fallback: flat 2-group split (BN/bias no decay, rest decay)
            no_decay_ids: set = set()
            pg_no_decay = []
            pg_decay = []
            for module in self.model.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                    for p in module.parameters():
                        if id(p) not in no_decay_ids:
                            pg_no_decay.append(p)
                            no_decay_ids.add(id(p))
                else:
                    if hasattr(module, "bias") and isinstance(module.bias, nn.Parameter):
                        if id(module.bias) not in no_decay_ids:
                            pg_no_decay.append(module.bias)
                            no_decay_ids.add(id(module.bias))
                    if hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
                        if id(module.weight) not in no_decay_ids:
                            pg_decay.append(module.weight)
            param_groups = [
                {"params": pg_decay, "weight_decay": weight_decay},
                {"params": pg_no_decay, "weight_decay": 0.0},
            ]

        # Filter out frozen parameters
        for pg in param_groups:
            pg["params"] = [p for p in pg["params"] if p.requires_grad]
        param_groups = [pg for pg in param_groups if pg["params"]]

        if opt_type == "sgd":
            return optim.SGD(param_groups, lr=lr, momentum=momentum, nesterov=True)
        elif opt_type in ("adam", "adamw"):
            betas = (
                self._train_cfg.get("beta1", 0.9),
                self._train_cfg.get("beta2", 0.999),
            )
            return optim.AdamW(param_groups, lr=lr, betas=betas)
        else:
            raise ValueError(f"Unknown optimizer: '{opt_type}'. Supported: sgd, adamw.")

    def _build_scheduler(self) -> Any:
        """Build learning rate scheduler from config.

        Returns:
            Scheduler instance.
        """
        return build_scheduler(self.optimizer, self.config)

    def _get_data_components(self):
        """Return (data_config, base_dir, build_fn) shared by all dataloader builders."""
        dataset_config_path = self._data_cfg.get("dataset_config")
        if dataset_config_path:
            if not Path(dataset_config_path).is_absolute():
                dataset_config_path = str(
                    (self.config_path.parent / dataset_config_path).resolve()
                )
            data_config = load_config(dataset_config_path)
        else:
            data_config = self._data_cfg

        self._data_config_path = dataset_config_path
        self._feature_name = feature_name_from_config_path(self.config_path)
        base_dir = str(self.config_path.parent)
        output_format = getattr(self._base_model, "output_format", "yolox")

        if output_format == "classification":
            from core.p05_data.classification_dataset import build_classification_dataloader
            build_fn = build_classification_dataloader
        elif output_format == "segmentation":
            from core.p05_data.segmentation_dataset import build_segmentation_dataloader
            build_fn = build_segmentation_dataloader
        else:
            build_fn = _build_detection_dataloader

        return data_config, base_dir, build_fn

    def _build_dataloaders(self) -> tuple[DataLoader, DataLoader | None]:
        """Build training and validation data loaders from config.

        When ``training.val_full_interval > 0``, the quick val loader uses
        ``training.val_subset_fraction`` (default 0.2) so per-epoch validation
        is fast. A separate full val loader (no subset) is built in ``train()``
        for periodic full evaluation.

        Returns:
            Tuple of (train_loader, val_loader). val_loader may be None.
        """
        import copy
        data_config, base_dir, build_fn = self._get_data_components()

        train_loader = build_fn(
            data_config, split="train", training_config=self.config, base_dir=base_dir
        )

        # Quick val: apply val_subset_fraction when val_full_interval is active
        val_full_interval = self._train_cfg.get("val_full_interval", 0)
        val_subset_fraction = self._train_cfg.get("val_subset_fraction", 0.2)
        if val_full_interval > 0 and val_subset_fraction is not None:
            quick_config = copy.deepcopy(self.config)
            quick_config.setdefault("data", {}).setdefault("subset", {})["val"] = val_subset_fraction
            val_loader = build_fn(data_config, split="val", training_config=quick_config, base_dir=base_dir)
        else:
            val_loader = build_fn(data_config, split="val", training_config=self.config, base_dir=base_dir)

        self._loaded_data_cfg = data_config
        return train_loader, val_loader

    def _build_full_val_loader(self) -> Any | None:
        """Build val loader on the full val set (no subset) for periodic full evaluation."""
        import copy
        data_config, base_dir, build_fn = self._get_data_components()
        full_config = copy.deepcopy(self.config)
        full_config.setdefault("data", {}).setdefault("subset", {})["val"] = None
        return build_fn(data_config, split="val", training_config=full_config, base_dir=base_dir)

    def _build_callbacks(self) -> CallbackRunner:
        """Build training callbacks from config.

        Returns:
            CallbackRunner with configured callbacks.
        """
        callbacks = []

        # Checkpoint saver — auto-generate timestamped run dir
        from utils.config import generate_run_dir

        save_dir = self._log_cfg.get("save_dir")
        if save_dir:
            if not Path(save_dir).is_absolute():
                save_dir = str((self.config_path.parent / save_dir).resolve())
        else:
            from utils.config import feature_name_from_config_path
            save_dir = str(generate_run_dir(
                feature_name_from_config_path(self.config_path), "06_training"
            ))
        # Expose for _finalize_training to reach after the main loop
        # (post-train runner + test eval both need it).
        self.save_dir = save_dir

        callbacks.append(
            CheckpointSaver(
                save_dir=save_dir,
                metric=self._ckpt_cfg.get("metric", "val/mAP50"),
                mode=self._ckpt_cfg.get("mode", "max"),
                save_interval=self._ckpt_cfg.get("save_interval", 10),
                save_best=self._ckpt_cfg.get("save_best", True),
            )
        )

        # Dataset statistics — always generated, not conditional on data_viz config
        data_splits = self._train_cfg.get("data_viz", {}).get("splits", ["train", "val", "test"])
        loaded_data_cfg = getattr(self, "_loaded_data_cfg", self._data_cfg)
        callbacks.append(
            DatasetStatsLogger(
                save_dir=save_dir,
                data_config=loaded_data_cfg,
                base_dir=str(self.config_path.parent),
                splits=data_splits,
                dpi=self._train_cfg.get("data_viz", {}).get("dpi", 120),
            )
        )

        # Early stopping
        patience = self._train_cfg.get("patience", 0)
        if patience > 0:
            callbacks.append(
                EarlyStopping(
                    metric=self._ckpt_cfg.get("metric", "val/mAP50"),
                    mode=self._ckpt_cfg.get("mode", "max"),
                    patience=patience,
                    min_delta=self._train_cfg.get("early_stop_min_delta", 0.0),
                )
            )

        # W&B logger
        wandb_project = self._log_cfg.get("wandb_project")
        if wandb_project:
            callbacks.append(
                WandBLogger(
                    project=wandb_project,
                    run_name=self._log_cfg.get("run_name"),
                    config=self.config,
                    log_interval=self._log_cfg.get("log_interval", 0),
                )
            )

        # Prediction visualization (val and optionally train)
        for split, cfg_key in (("val", "val_viz"), ("train", "train_viz")):
            viz_cfg = self._train_cfg.get(cfg_key, {})
            if viz_cfg.get("enabled", False):
                callbacks.append(
                    ValPredictionLogger(
                        save_dir=save_dir,
                        split=split,
                        num_samples=viz_cfg.get("num_samples", 8),
                        conf_threshold=viz_cfg.get("conf_threshold", 0.05),
                        grid_cols=viz_cfg.get("grid_cols", 2),
                        gt_thickness=viz_cfg.get("gt_thickness", 2),
                        pred_thickness=viz_cfg.get("pred_thickness", 1),
                        gt_color_rgb=tuple(viz_cfg.get("gt_color_rgb", [160, 32, 240])),
                        pred_color_rgb=tuple(viz_cfg.get("pred_color_rgb", [0, 200, 0])),
                        text_scale=viz_cfg.get("text_scale", 0.4),
                        dpi=viz_cfg.get("dpi", 150),
                    )
                )

        # Data label visualization (raw images + GT, once at train start)
        data_viz_cfg = self._train_cfg.get("data_viz", {})
        if data_viz_cfg.get("enabled", False):
            callbacks.append(
                DataLabelGridLogger(
                    save_dir=save_dir,
                    splits=data_viz_cfg.get("splits", ["train"]),
                    data_config=loaded_data_cfg,
                    base_dir=str(self.config_path.parent),
                    num_samples=data_viz_cfg.get("num_samples", 16),
                    grid_cols=data_viz_cfg.get("grid_cols", 4),
                    thickness=data_viz_cfg.get("thickness", 2),
                    text_scale=data_viz_cfg.get("text_scale", 0.4),
                    dpi=data_viz_cfg.get("dpi", 120),
                )
            )

        # Augmentation label visualization (once at train start)
        aug_viz_cfg = self._train_cfg.get("aug_viz", {})
        if aug_viz_cfg.get("enabled", False):
            callbacks.append(
                AugLabelGridLogger(
                    save_dir=save_dir,
                    splits=aug_viz_cfg.get("splits", ["train"]),
                    data_config=loaded_data_cfg,
                    aug_config=self.config.get("augmentation", {}),
                    base_dir=str(self.config_path.parent),
                    num_samples=aug_viz_cfg.get("num_samples", 16),
                    grid_cols=aug_viz_cfg.get("grid_cols", 4),
                    thickness=aug_viz_cfg.get("thickness", 2),
                    text_scale=aug_viz_cfg.get("text_scale", 0.4),
                    dpi=aug_viz_cfg.get("dpi", 120),
                )
            )

        # Step-by-step transform pipeline viz (per-class row walk + final
        # Denormalize(Normalize) inverse-check cell).
        transform_viz_cfg = self._train_cfg.get("transform_viz", {})
        if transform_viz_cfg.get("enabled", True):
            from core.p06_training.callbacks_viz import TransformPipelineCallback
            class_names_nc = {int(k): str(v)
                              for k, v in (loaded_data_cfg.get("names", {}) or {}).items()}
            callbacks.append(
                TransformPipelineCallback(
                    save_dir=save_dir,
                    data_config=loaded_data_cfg,
                    training_config=self.config,
                    base_dir=str(self.config_path.parent),
                    class_names=class_names_nc,
                    max_samples=transform_viz_cfg.get("max_samples", 5),
                )
            )

        return CallbackRunner(callbacks)

    def _build_loss(self) -> nn.Module:
        """Build loss function from config via the loss registry.

        Uses ``build_loss`` which dispatches to the registered loss class
        based on ``config["loss"]["type"]`` (default ``"yolox"``).

        Returns:
            Configured loss module.
        """
        return build_loss(self.config)

    # ------------------------------------------------------------------
    # Target scaling
    # ------------------------------------------------------------------

    def _scale_targets_to_pixels(self, targets: list, img_h: int, img_w: int) -> list:
        """Scale normalised (0-1) YOLO targets to pixel coordinates.

        Each target tensor has columns [class_id, cx, cy, w, h].
        cx/w are scaled by img_w, cy/h by img_h.

        Args:
            targets: List of (M_i, 5) tensors with normalised coords.
            img_h: Image height in pixels.
            img_w: Image width in pixels.

        Returns:
            List of (M_i, 5) tensors with pixel-space coords.
        """
        scaled = []
        for t in targets:
            if t.shape[0] == 0:
                scaled.append(t)
                continue
            s = t.clone()
            s[:, 1] *= img_w   # cx
            s[:, 2] *= img_h   # cy
            s[:, 3] *= img_w   # w
            s[:, 4] *= img_h   # h
            scaled.append(s)
        return scaled

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> dict[str, Any]:
        """Execute the full training loop.

        Builds all components, runs epochs, handles callbacks, and
        returns a summary of the training run.

        Returns:
            Dictionary with training summary:
                - best_metric: Best validation metric achieved.
                - best_epoch: Epoch of best metric.
                - total_epochs: Number of epochs actually trained.
                - final_metrics: Metrics from the last epoch.
        """
        logger.info("Starting training with config: %s", self.config_path)

        # Build components — but skip if load_checkpoint already built them
        # during resume. Unconditional rebuild here would reload the
        # pretrained weights and overwrite the checkpoint's restored state.
        if self.model is None:
            self.model = self._build_model()

            # Apply layer freezing before optimizer (frozen params excluded)
            freeze_list = self._train_cfg.get("freeze", [])
            if freeze_list:
                self._freeze_layers(freeze_list)

        # Validate the tensor_prep contract now that the model (and any
        # HF processor on it) exists. Hard-errors on backend mismatch,
        # double/missing-normalize, or missing mean/std.
        try:
            from utils.config import _validate_tensor_prep as _vtp
            _processor = getattr(self._base_model, "processor", None)
            _vtp(self.config, backend="pytorch", processor=_processor)
        except Exception as _e:
            logger.error("tensor_prep validation failed: %s", _e)
            raise

        if self.optimizer is None:
            self.optimizer = self._build_optimizer()
        if self.scheduler is None:
            self.scheduler = self._build_scheduler()
        if self.loss_fn is None:
            self.loss_fn = self._build_loss()

        # Data loaders must come before callbacks so _loaded_data_cfg is available
        self.train_loader, self.val_loader = self._build_dataloaders()

        self.callback_runner = self._build_callbacks()

        # EMA
        use_ema = self._train_cfg.get("ema", False)
        if use_ema:
            ema_decay = self._train_cfg.get("ema_decay", 0.9998)
            base_model = self._base_model
            self.ema = ModelEMA(base_model, decay=ema_decay)
            logger.info("EMA enabled with decay=%.4f", ema_decay)

        # AMP scaler
        use_amp = self._train_cfg.get("amp", True) and self.device.type == "cuda"
        if use_amp:
            self.scaler = torch.amp.GradScaler("cuda")
            logger.info("AMP (automatic mixed precision) enabled.")
        else:
            self.scaler = None

        # GPU augmentation (stateless transforms: RandomAffine, ColorJitter, Flips, Normalize)
        if self._train_cfg.get("gpu_augment", False):
            base_model = self._base_model
            output_format = getattr(base_model, "output_format", "yolox")
            if output_format not in ("classification", "segmentation"):
                data_cfg = getattr(self, "_loaded_data_cfg", self._data_cfg)
                input_size = tuple(data_cfg.get("input_size", self._model_cfg["input_size"]))
                self.gpu_transform = build_gpu_transforms(
                    config=self.config.get("augmentation", {}),
                    input_size=input_size,
                    mean=data_cfg.get("mean"),
                    std=data_cfg.get("std"),
                )
                logger.info(
                    "GPU augmentation enabled: RandomAffine, ColorJitter, Flips on %s. "
                    "Label correctness guaranteed by TVTensor dispatch (same affine matrix "
                    "applied to image and boxes, then ClampBoundingBoxes + SanitizeBoundingBoxes).",
                    self.device,
                )

        total_epochs = self._train_cfg["epochs"]
        grad_clip = self._train_cfg.get("grad_clip", 35.0)

        self.callback_runner.on_train_start(self)

        best_metrics: dict[str, float] = {}
        final_metrics: dict[str, float] = {}
        epoch = self._start_epoch - 1  # in case total_epochs == 0

        # Build full val loader once if val_full_interval is configured
        val_full_interval = self._train_cfg.get("val_full_interval", 0)
        self._full_val_loader = self._build_full_val_loader() if val_full_interval > 0 else None

        with TrainingProgress(
            total_epochs=total_epochs,
            batches_per_epoch=len(self.train_loader) if self.train_loader else 0,
        ) as progress:
            for epoch in range(self._start_epoch, total_epochs):
                self.callback_runner.on_epoch_start(self, epoch)
                progress.start_epoch(epoch)

                # Train one epoch
                train_metrics = self._train_one_epoch(epoch, progress, grad_clip, use_amp)

                # Validate (use EMA model if available)
                # val_full_interval > 0: quick val (subset) every epoch for logging,
                # full val every N epochs for checkpoint/early-stop/scheduler.
                val_metrics = {}
                is_full_val_epoch = val_full_interval > 0 and (epoch + 1) % val_full_interval == 0
                if self.val_loader is not None:
                    def _run_validate(loader=None):
                        if self.ema is not None:
                            orig_model = self.model
                            self.model = self.ema.ema_model
                            m = self._validate(loader)
                            self.model = orig_model
                            return m
                        return self._validate(loader)

                    if val_full_interval > 0:
                        # Quick val every epoch (logged but not used for checkpoint/ES)
                        quick_metrics = _run_validate()
                        logger.debug("Quick val: %s", quick_metrics)
                        if is_full_val_epoch:
                            # Full val every N epochs — drives checkpoint/ES/scheduler
                            val_metrics = _run_validate(self._full_val_loader)
                            logger.info("Full val (epoch %d): %s", epoch + 1, val_metrics)
                        else:
                            # Non-full-val epoch: use quick metrics for logging only
                            val_metrics = quick_metrics
                    else:
                        val_metrics = _run_validate()

                # Full evaluation (periodic, optional)
                full_eval_interval = self._train_cfg.get("full_eval_interval", 0)
                if full_eval_interval > 0 and (epoch + 1) % full_eval_interval == 0:
                    full_metrics = self._full_evaluate()
                    val_metrics.update(full_metrics)

                # Combine metrics
                epoch_metrics = {**train_metrics, **val_metrics}
                final_metrics = epoch_metrics

                # Update progress tracker
                track_metric = self._ckpt_cfg.get("metric", "val/mAP50")
                track_mode = self._ckpt_cfg.get("mode", "max")
                is_best = progress.end_epoch(
                    metrics=epoch_metrics,
                    track_metric=track_metric,
                    mode=track_mode,
                )
                if is_best:
                    best_metrics = epoch_metrics.copy()

                # Callbacks
                self.callback_runner.on_epoch_end(self, epoch, epoch_metrics)

                # Scheduler step
                if self.scheduler is not None:
                    if hasattr(self.scheduler, "step"):
                        metric_val = epoch_metrics.get(track_metric)
                        self.scheduler.step(epoch=epoch + 1, metrics=metric_val)

                # Early stopping check
                early_stop_cb = self.callback_runner.get_callback(EarlyStopping)
                if isinstance(early_stop_cb, EarlyStopping) and early_stop_cb.should_stop:
                    logger.info("Training stopped early at epoch %d.", epoch + 1)
                    break

        self.callback_runner.on_train_end(self)

        # Backend-agnostic post-train finalize: reload best checkpoint, run
        # test-set eval if a `test` split exists, and dispatch post-train
        # artifacts (best.png for val+test, error_analysis/* with charts +
        # per-error/per-class galleries). Default on; opt out via
        # training.post_train.enabled: false.
        test_metrics = None
        try:
            test_metrics = self._finalize_training()
        except Exception as e:  # pragma: no cover — never block
            logger.warning("post-train finalize skipped: %s", e, exc_info=True)

        summary = {
            "best_metric": progress.best_metric,
            "best_epoch": progress.best_epoch + 1,
            "total_epochs": epoch + 1,
            "final_metrics": final_metrics,
            "best_metrics": best_metrics,
        }
        if test_metrics is not None:
            summary["test_metrics"] = test_metrics
        logger.info("Training summary: %s", summary)
        return summary

    def _finalize_training(self) -> dict | None:
        """Reload best.pth, run test-set eval, render post-train artifacts.

        Mirrors HF backend's ``load_best_model_at_end`` + auto-test behavior.
        Skippable via ``training.post_train.enabled: false``.
        """
        post_cfg = (self.config.get("training", {}) or {}).get("post_train", {}) or {}
        if not post_cfg.get("enabled", True):
            return None

        save_dir = Path(self.save_dir)
        best_path = save_dir / "best.pth"
        if best_path.exists():
            logger.info("Reloading best checkpoint from %s", best_path)
            try:
                self.load_checkpoint(str(best_path))
            except Exception as e:
                logger.warning("best.pth reload failed, using last-epoch weights: %s", e)

        # Build test loader if a test split is present; if not, skip test eval.
        data_cfg_resolved = getattr(self, "_loaded_data_cfg", None) or self.config.get("data", {})
        # base_dir must be the training-config's directory so YOLOXDataset can
        # resolve `data_cfg["path"]` (e.g. `"../../../dataset_store/..."`) the
        # same way it does for train/val during training — NOT CWD.
        base_dir = str(self.config_path.parent)
        test_loader = None
        try:
            test_loader = self._maybe_build_test_loader(data_cfg_resolved, base_dir)
        except Exception as e:
            logger.warning("failed to build test loader (skipping test eval): %s", e)

        test_metrics = None
        if test_loader is not None:
            try:
                logger.info("Running final test-set evaluation on best checkpoint...")
                test_metrics = self._validate(test_loader)
                with open(save_dir / "test_results.json", "w") as f:
                    import json as _json
                    _json.dump(test_metrics, f, indent=2, sort_keys=True, default=str)
                logger.info("Test metrics written to %s", save_dir / "test_results.json")
            except Exception as e:
                logger.warning("test-set eval skipped: %s", e, exc_info=True)

        # Post-train artifacts (best-checkpoint val+test grids + error analysis)
        try:
            from core.p06_training.post_train import run_post_train_artifacts
            from core.p10_inference.supervision_bridge import VizStyle

            class_names = {int(k): str(v)
                           for k, v in (data_cfg_resolved.get("names", {}) or {}).items()}
            input_size = tuple(data_cfg_resolved.get("input_size", (640, 640)))
            style = VizStyle.from_config(self.config)

            val_ds = getattr(self.val_loader, "dataset", None) if hasattr(self, "val_loader") else None
            test_ds = getattr(test_loader, "dataset", None) if test_loader is not None else None

            training_config = self._build_pytorch_training_config(
                data_cfg_resolved=data_cfg_resolved, test_metrics=test_metrics,
            )
            run_post_train_artifacts(
                model=self.model,
                save_dir=save_dir,
                val_dataset=val_ds,
                test_dataset=test_ds,
                task=_task_from_output(getattr(self.model, "output_format", None)),
                class_names=class_names,
                input_size=input_size,
                style=style,
                best_num_samples=int(post_cfg.get("num_samples", 16)),
                best_conf_threshold=float(post_cfg.get("conf_threshold", 0.1)),
                error_analysis_conf_threshold=float(post_cfg.get("error_conf_threshold", 0.05)),
                error_analysis_iou_threshold=float(post_cfg.get("error_iou_threshold", 0.5)),
                error_analysis_max_samples=post_cfg.get("error_max_samples", 500),
                error_analysis_hard_images_per_class=int(post_cfg.get("hard_images_per_class", 20)),
                training_config=training_config,
            )
        except Exception as e:
            logger.warning("post-train artifacts skipped: %s", e, exc_info=True)

        return test_metrics

    def _safe_best_metric(self):
        """Return the CheckpointSaver's best value if available, else None."""
        try:
            cb = self.callback_runner.get_callback(CheckpointSaver)
            return getattr(cb, "best_value", None)
        except Exception:
            return None

    def _build_pytorch_training_config(self, *, data_cfg_resolved: dict,
                                        test_metrics: dict | None) -> dict:
        """Compact training-config snapshot for the error-analysis report.

        Mirrors the HF backend helper (`_build_hf_training_config`) so both
        backends attach the same sections to `summary.json`.
        """
        train_cfg = self._train_cfg or {}
        model_cfg = self._model_cfg or {}
        aug_cfg = self.config.get("augmentation", {}) or {}
        try:
            params = int(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        except Exception:
            params = None
        return {
            "model": {
                "arch": model_cfg.get("arch"),
                "trainable_params": params,
                "input_size": tuple(data_cfg_resolved.get("input_size",
                                    model_cfg.get("input_size", (640, 640)))),
            },
            "training": {
                "backend": "pytorch",
                "epochs": train_cfg.get("epochs"),
                "batch_size": (self.config.get("data", {}) or {}).get("batch_size"),
                "lr": train_cfg.get("lr"),
                "optimizer": train_cfg.get("optimizer"),
                "scheduler": train_cfg.get("scheduler"),
                "warmup_epochs": train_cfg.get("warmup_epochs"),
                "weight_decay": train_cfg.get("weight_decay"),
                "amp": train_cfg.get("amp"),
                "ema": train_cfg.get("ema"),
                "seed": self.config.get("seed"),
                "max_grad_norm": train_cfg.get("grad_clip") or train_cfg.get("max_grad_norm"),
            },
            "augmentation": {
                k: aug_cfg.get(k) for k in (
                    "library", "normalize", "mosaic", "mixup", "copypaste",
                    "fliplr", "flipud", "perspective_p", "brightness_contrast_p",
                    "hsv_p", "scale", "degrees", "translate", "shear",
                )
            },
            "run": {
                "best_val_metric": self._safe_best_metric(),
                "total_epochs": getattr(self, "_last_epoch", None),
                "test_metrics": test_metrics,
            },
        }

    def _maybe_build_test_loader(self, data_cfg: dict, base_dir: str):
        """Construct a test DataLoader if the dataset has a test split on disk.

        Delegates split resolution to the dataset class itself — matches the
        HF-backend pattern in ``hf_trainer._try_build_test_dataset``:
        ``YOLOXDataset(split="test")`` raises ``FileNotFoundError`` cleanly
        when the split is absent, so no pre-existence check is needed.
        """
        from torch.utils.data import DataLoader

        output_format = getattr(self.model, "output_format", "yolox")
        try:
            if output_format in {"detr", "yolox"}:
                from core.p05_data.detection_dataset import YOLOXDataset
                from core.p05_data.transforms import build_transforms
                from utils.config import resolve_tensor_prep
                tp = resolve_tensor_prep(self.config, backend="pytorch") or None
                input_size = tuple(
                    (tp or {}).get("input_size") or data_cfg.get("input_size", (640, 640))
                )
                eval_transforms = build_transforms(
                    config=self.config.get("augmentation", {}),
                    is_train=False, input_size=input_size,
                    mean=data_cfg.get("mean"), std=data_cfg.get("std"),
                    tensor_prep=tp,
                )
                ds = YOLOXDataset(
                    data_cfg, split="test",
                    transforms=eval_transforms,
                    base_dir=base_dir,
                )
            elif output_format == "classification":
                from core.p05_data.classification_dataset import (
                    ClassificationDataset, build_classification_transforms,
                )
                transforms = build_classification_transforms(
                    self.config, data_cfg, is_train=False)
                ds = ClassificationDataset(
                    data_cfg, split="test", transforms=transforms, base_dir=base_dir,
                )
            elif output_format == "segmentation":
                from core.p05_data.segmentation_dataset import (
                    SegmentationDataset, build_segmentation_transforms,
                )
                transforms = build_segmentation_transforms(
                    self.config, data_cfg, is_train=False)
                ds = SegmentationDataset(
                    data_cfg, split="test", transform=transforms, base_dir=base_dir,
                )
            else:
                return None
        except FileNotFoundError:
            logger.info("No test split on disk — skipping test-set evaluation.")
            return None
        except Exception as e:  # pragma: no cover
            logger.warning("test dataset build failed: %s", e)
            return None

        batch_size = (self.config.get("data", {}) or {}).get("batch_size", 8)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=(self.config.get("data", {}) or {}).get("num_workers", 2),
            pin_memory=True,
            collate_fn=getattr(self.val_loader, "collate_fn", None) if hasattr(self, "val_loader") else None,
        )

    def _train_one_epoch(
        self,
        epoch: int,
        progress: TrainingProgress,
        grad_clip: float,
        use_amp: bool,
    ) -> dict[str, float]:
        """Run one training epoch.

        Args:
            epoch: Current epoch number (0-indexed).
            progress: Training progress tracker for batch updates.
            grad_clip: Maximum gradient norm. 0 to disable clipping.
            use_amp: Whether to use automatic mixed precision.

        Returns:
            Dictionary of training metrics for this epoch:
                {"train/loss", "train/cls_loss", "train/obj_loss", "train/reg_loss"}.
        """
        import math as _math

        self.model.train()
        if hasattr(self.loss_fn, 'set_epoch'):
            self.loss_fn.set_epoch(epoch)

        if self.train_loader is None:
            logger.warning("No training data loader. Skipping training epoch.")
            return {}

        running_loss = 0.0
        running_cls = 0.0
        running_obj = 0.0
        running_reg = 0.0
        num_batches = 0

        amp_device_type = "cuda" if self.device.type == "cuda" else "cpu"

        input_h, input_w = self._model_cfg["input_size"]

        base_model = self._base_model
        output_format = getattr(base_model, "output_format", "yolox")
        grad_accum_steps = self._train_cfg.get("gradient_accumulation_steps", 1)

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["images"].to(self.device, non_blocking=True)
            targets = [t.to(self.device, non_blocking=True) for t in batch["targets"]]

            with torch.amp.autocast(device_type=amp_device_type, enabled=use_amp):
                if self.gpu_transform is not None:
                    images, targets = self.gpu_transform(images, targets)

                # Scale normalised targets to pixel coordinates (detection only)
                if output_format not in ("classification", "segmentation"):
                    targets = self._scale_targets_to_pixels(targets, input_h, input_w)

                if hasattr(base_model, 'forward_with_loss'):
                    loss, loss_dict, predictions = base_model.forward_with_loss(images, targets)
                else:
                    predictions = self.model(images)
                    loss, loss_dict = self.loss_fn(predictions, targets)

                if grad_accum_steps > 1:
                    loss = loss / grad_accum_steps

            is_accum_step = (batch_idx + 1) % grad_accum_steps == 0

            loss_val = loss.item()
            if not _math.isfinite(loss_val):
                # Skip backward entirely — NaN gradients would corrupt weights even if
                # the optimizer step is later skipped by GradScaler
                logger.debug("Skipping batch %d: loss=%s", batch_idx, loss_val)
                self.optimizer.zero_grad()
            elif use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
                if is_accum_step:
                    if grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                loss.backward()
                if is_accum_step:
                    if grad_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # EMA update (only on finite optimizer step)
            if is_accum_step and _math.isfinite(loss_val) and self.ema is not None:
                self.ema.update(base_model)

            # Accumulate metrics — skip NaN/inf batches so epoch average remains meaningful
            cls_val = loss_dict.get("cls_loss", _LOSS_ZERO).item()
            obj_val = loss_dict.get("obj_loss", _LOSS_ZERO).item()
            reg_val = loss_dict.get("reg_loss", _LOSS_ZERO).item()

            if _math.isfinite(loss_val):
                running_loss += loss_val
                running_cls += cls_val if _math.isfinite(cls_val) else 0.0
                running_obj += obj_val if _math.isfinite(obj_val) else 0.0
                running_reg += reg_val if _math.isfinite(reg_val) else 0.0
                num_batches += 1

            # Progress + callback — flexible batch metrics
            batch_metrics = {
                "loss": loss_val,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            # Add standard component losses if non-zero (reuse already-computed values)
            for short_key, computed_val in (("cls", cls_val), ("obj", obj_val), ("reg", reg_val)):
                if computed_val > 0:
                    batch_metrics[short_key] = computed_val
            # Add any non-standard loss keys from the model
            for key, val in loss_dict.items():
                if key in ("cls_loss", "obj_loss", "reg_loss"):
                    continue
                short_key = key.replace("_loss", "")
                if short_key not in batch_metrics:
                    batch_metrics[short_key] = val.item() if hasattr(val, 'item') else float(val)

            progress.update_batch(metrics=batch_metrics)
            self.callback_runner.on_batch_end(self, batch_idx, batch_metrics)

        if num_batches == 0:
            return {}

        epoch_metrics = {"train/loss": running_loss / num_batches}
        # Add standard detection losses if accumulated
        if running_cls > 0:
            epoch_metrics["train/cls_loss"] = running_cls / num_batches
        if running_obj > 0:
            epoch_metrics["train/obj_loss"] = running_obj / num_batches
        if running_reg > 0:
            epoch_metrics["train/reg_loss"] = running_reg / num_batches
        return epoch_metrics

    @torch.no_grad()
    def _validate(self, loader: Any | None = None) -> dict[str, float]:
        """Run validation and compute task-appropriate metrics.

        Dispatches metrics computation based on the model's ``output_format``:
        - ``"classification"``: accuracy and top-5 accuracy.
        - ``"segmentation"``: mean IoU per class.
        - Detection formats (``"yolox"``, ``"detr"``, etc.): COCO-style mAP@0.5
          using ``compute_map()``.

        Returns:
            Dictionary of validation metrics. Keys depend on the task:
                - Detection: {"val/loss", "val/mAP50", "val/AP50_cls{id}", ...}
                - Classification: {"val/loss", "val/accuracy", "val/top5_accuracy"}
                - Segmentation: {"val/loss", "val/mIoU"}
        """
        self.model.eval()

        active_loader = loader if loader is not None else self.val_loader
        if active_loader is None:
            return {}

        base_model = self._base_model
        output_format = getattr(base_model, "output_format", "yolox")
        is_classification = output_format == "classification"
        is_segmentation = output_format == "segmentation"

        running_loss = 0.0
        running_cls = 0.0
        running_obj = 0.0
        running_reg = 0.0
        num_batches = 0

        all_predictions: list = []    # Detection: list of dicts; Classification/Segmentation: list of tensors
        all_ground_truths: list = []  # Detection: list of dicts; Classification/Segmentation: list of tensors

        amp_device_type = "cuda" if self.device.type == "cuda" else "cpu"
        use_amp = self._train_cfg.get("amp", True) and self.device.type == "cuda"
        input_h, input_w = self._model_cfg["input_size"]

        for batch in active_loader:
            images = batch["images"].to(self.device, non_blocking=True)
            targets = [t.to(self.device, non_blocking=True) for t in batch["targets"]]
            # Scale normalised targets to pixel coordinates (detection only)
            if not is_classification and not is_segmentation:
                targets = self._scale_targets_to_pixels(targets, input_h, input_w)

            with torch.amp.autocast(device_type=amp_device_type, enabled=use_amp):
                if hasattr(base_model, 'forward_with_loss'):
                    loss, loss_dict, predictions = base_model.forward_with_loss(images, targets)
                else:
                    predictions = self.model(images)
                    loss, loss_dict = self.loss_fn(predictions, targets)

            running_loss += loss.item()
            running_cls += loss_dict.get("cls_loss", _LOSS_ZERO).item()
            running_obj += loss_dict.get("obj_loss", _LOSS_ZERO).item()
            running_reg += loss_dict.get("reg_loss", _LOSS_ZERO).item()
            num_batches += 1

            if is_classification:
                # Collect raw logits and class labels for accuracy computation
                all_predictions.append(predictions.cpu())
                for t in targets:
                    all_ground_truths.append(t.cpu())
            elif is_segmentation:
                # Argmax (B, C, H, W) logits → per-image (H, W) class maps.
                # Some models (e.g. SegFormer) output at 1/4 resolution — upsample to input size.
                class_maps = predictions.argmax(dim=1)  # (B, H', W')
                if class_maps.shape[-2:] != (input_h, input_w):
                    import torch.nn.functional as F
                    class_maps = (
                        F.interpolate(
                            class_maps.unsqueeze(1).float(),
                            size=(input_h, input_w),
                            mode="nearest",
                        )
                        .squeeze(1)
                        .long()
                    )
                for i in range(class_maps.shape[0]):
                    all_predictions.append(class_maps[i].cpu())
                all_ground_truths.extend([t.cpu() for t in targets])
            else:
                # Detection: decode predictions into compute_map format
                batch_preds = self._decode_predictions(predictions, conf_threshold=0.01)
                all_predictions.extend(batch_preds)

                # Convert ground truths into compute_map format
                for gt in targets:
                    gt_np = gt.cpu().numpy()
                    if gt_np.shape[0] == 0:
                        all_ground_truths.append({
                            "boxes": np.zeros((0, 4)),
                            "labels": np.zeros(0, dtype=np.int64),
                        })
                    else:
                        # Convert cxcywh to xyxy
                        cx, cy, w, h = gt_np[:, 1], gt_np[:, 2], gt_np[:, 3], gt_np[:, 4]
                        x1 = cx - w / 2
                        y1 = cy - h / 2
                        x2 = cx + w / 2
                        y2 = cy + h / 2
                        boxes = np.stack([x1, y1, x2, y2], axis=1)
                        labels = gt_np[:, 0].astype(np.int64)
                        all_ground_truths.append({"boxes": boxes, "labels": labels})

        if num_batches == 0:
            return {}

        metrics: dict[str, float] = {"val/loss": running_loss / num_batches}
        # Add detection-specific loss components if non-zero
        if running_cls > 0:
            metrics["val/cls_loss"] = running_cls / num_batches
        if running_obj > 0:
            metrics["val/obj_loss"] = running_obj / num_batches
        if running_reg > 0:
            metrics["val/reg_loss"] = running_reg / num_batches

        num_classes = self._model_cfg["num_classes"]
        task_metrics = compute_metrics(
            output_format,
            all_predictions,
            all_ground_truths,
            num_classes=num_classes,
        )
        metrics.update(task_metrics)

        return metrics

    def _decode_predictions(
        self, predictions: torch.Tensor, conf_threshold: float = 0.01
    ) -> list:
        """Decode raw model predictions into task-appropriate format.

        Dispatches based on the model's ``output_format``:
        - ``"classification"``: returns per-sample class id, confidence, and
          probability vector.
        - Detection formats (``"yolox"``, ``"detr"``, etc.): delegates to
          :func:`~core.p06_training.postprocess.postprocess`, which first
          checks for ``model.postprocess()`` then falls back to the registry.

        Args:
            predictions: Model output tensor. Shape depends on task:
                - Classification: (B, C) logits.
                - Detection: (B, N, 5+C) tensor.
            conf_threshold: Minimum confidence for detection PR curve (use
                low value like 0.01 to capture the full precision-recall
                curve). Ignored for classification.

        Returns:
            List of B result dicts. Keys depend on task:
                - Detection: {"boxes", "scores", "labels"} (numpy arrays).
                - Classification: {"class_id", "confidence", "probabilities"}.
        """
        base_model = self._base_model
        output_format = getattr(base_model, "output_format", "yolox")

        if output_format == "classification":
            results = []
            probs = torch.softmax(predictions, dim=-1)
            pred_classes = probs.argmax(dim=-1)
            pred_scores = probs.max(dim=-1).values
            for b in range(predictions.shape[0]):
                results.append({
                    "class_id": pred_classes[b].item(),
                    "confidence": pred_scores[b].item(),
                    "probabilities": probs[b].cpu().numpy(),
                })
            return results

        if output_format == "segmentation":
            import torch.nn.functional as F_seg

            class_maps = predictions.argmax(dim=1)  # (B, H', W')
            input_h, input_w = self._model_cfg["input_size"]
            if class_maps.shape[-2:] != (input_h, input_w):
                class_maps = F_seg.interpolate(
                    class_maps.unsqueeze(1).float(),
                    size=(input_h, input_w),
                    mode="nearest",
                ).squeeze(1).long()
            return [{"class_map": class_maps[i].cpu().numpy()} for i in range(class_maps.shape[0])]

        # Detection: build target_sizes for HF models that need them
        input_h, input_w = self._model_cfg["input_size"]
        target_sizes = torch.tensor(
            [[input_h, input_w]] * predictions.shape[0],
            device=predictions.device,
        )

        return _postprocess_registry(
            output_format=output_format,
            model=base_model,
            predictions=predictions,
            conf_threshold=conf_threshold,
            nms_threshold=self._train_cfg.get("nms_threshold", 0.45),
            target_sizes=target_sizes,
        )

    def _full_evaluate(self) -> dict[str, float]:
        """Run full evaluation using ModelEvaluator (optional periodic eval).

        Provides confusion matrix, per-class AP, and failure case analysis
        on top of the standard inline mAP. Controlled by
        ``training.full_eval_interval`` in config.

        Returns:
            Dictionary of metrics with ``val/full_`` prefix.
        """
        from core.p08_evaluation.evaluator import (
            ModelEvaluator,  # lazy import to avoid circular dependency
        )

        base_model = self._base_model
        try:
            evaluator = ModelEvaluator(
                model=base_model,
                data_config=self._data_cfg,
                device=self.device,
                conf_threshold=0.25,
                iou_threshold=0.45,
            )
            results = evaluator.evaluate(split="val")
            # Prefix all keys with val/full_
            return {f"val/full_{k}": v for k, v in results.items() if isinstance(v, (int, float))}
        except Exception as e:
            logger.warning("Full evaluation failed: %s", e)
            return {}

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str, epoch: int, metrics: dict[str, float]) -> None:
        """Save a training checkpoint manually.

        Args:
            path: File path for the checkpoint.
            epoch: Current epoch number.
            metrics: Current metrics dictionary.
        """
        checkpoint = CheckpointSaver._build_checkpoint(self, epoch, metrics)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        logger.info("Saved checkpoint: epoch %d -> %s", epoch + 1, path)

    def load_checkpoint(self, path: str) -> int:
        """Load a checkpoint to resume training.

        Restores model, optimizer, scheduler, and AMP scaler states.

        Args:
            path: Path to the checkpoint file.

        Returns:
            The epoch number to resume from (next epoch after the saved one).

        Raises:
            FileNotFoundError: If checkpoint file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        logger.info("Loading checkpoint from %s (epoch %d)", path, checkpoint.get("epoch", -1) + 1)

        # Build components if not already built
        if self.model is None:
            self.model = self._build_model()
        if self.optimizer is None:
            self.optimizer = self._build_optimizer()
        if self.scheduler is None:
            self.scheduler = self._build_scheduler()

        # Restore states
        if "model_state_dict" in checkpoint:
            self._base_model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint and hasattr(self.scheduler, "load_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "scaler_state_dict" in checkpoint:
            use_amp = self._train_cfg.get("amp", True) and self.device.type == "cuda"
            if use_amp:
                if self.scaler is None:
                    self.scaler = torch.amp.GradScaler("cuda")
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if "ema_state_dict" in checkpoint and self.ema is not None:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])

        # Restore RNG states for exact reproducibility (best-effort — set_rng_state
        # requires ByteTensor on CPU, and load-time dtype can drift after
        # `map_location=device` round-trips through CUDA).
        import random as _random
        rng_states = checkpoint.get("rng_states")
        if rng_states:
            try:
                _random.setstate(rng_states["python"])
                np.random.set_state(rng_states["numpy"])
                torch_cpu = rng_states["torch_cpu"]
                if hasattr(torch_cpu, "to"):
                    torch_cpu = torch_cpu.to(dtype=torch.uint8, device="cpu")
                torch.random.set_rng_state(torch_cpu)
                if "torch_cuda" in rng_states:
                    cuda_states = [
                        s.to(dtype=torch.uint8, device="cpu") if hasattr(s, "to") else s
                        for s in rng_states["torch_cuda"]
                    ]
                    torch.cuda.set_rng_state_all(cuda_states)
                logger.info("Restored RNG states for reproducibility.")
            except (TypeError, RuntimeError) as e:
                logger.warning("Could not restore RNG states (%s) — continuing with current RNG.", e)

        # Restore callback states (best metric tracking, early stopping counter)
        cb_states = checkpoint.get("callback_states", {})
        if cb_states and self.callback_runner is not None:
            ckpt_state = cb_states.get("checkpoint_saver")
            if ckpt_state:
                ckpt_cb = self.callback_runner.get_callback(CheckpointSaver)
                if ckpt_cb is not None:
                    ckpt_cb._best_value = ckpt_state["best_value"]
                    ckpt_cb._best_epoch = ckpt_state["best_epoch"]
                    logger.info("Restored CheckpointSaver: best_value=%s at epoch %d",
                                ckpt_cb._best_value, ckpt_cb._best_epoch + 1)

            es_state = cb_states.get("early_stopping")
            if es_state:
                es_cb = self.callback_runner.get_callback(EarlyStopping)
                if es_cb is not None:
                    es_cb._best_value = es_state["best_value"]
                    es_cb._counter = es_state["counter"]
                    logger.info("Restored EarlyStopping: counter=%d, best_value=%s",
                                es_cb._counter, es_cb._best_value)

        resume_epoch = checkpoint.get("epoch", -1) + 1
        self._start_epoch = resume_epoch
        logger.info("Resuming training from epoch %d.", resume_epoch + 1)

        return resume_epoch

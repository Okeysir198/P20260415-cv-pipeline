"""Train models using HuggingFace Trainer.

Thin bridge between our YAML config and HF's TrainingArguments.
Use for HF models and timm models to get DDP, DeepSpeed, gradient
accumulation, and built-in checkpointing for free.

Our YAML config is the single source of truth — this module reads it
and maps relevant fields to HF TrainingArguments internally. Users
never touch TrainingArguments directly.

Usage:
    from core.p06_training.hf_trainer import train_with_hf
    summary = train_with_hf("features/ppe-shoes_detection/configs/06_training.yaml")

    # With overrides
    summary = train_with_hf("features/ppe-shoes_detection/configs/06_training.yaml",
                            overrides={"training": {"lr": 0.0005}})
"""

import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from core.p06_models import build_model
from core.p05_data.base_dataset import IMAGENET_MEAN, IMAGENET_STD
from utils.config import generate_run_dir, load_config, merge_configs

logger = logging.getLogger(__name__)

# Mapping: our optimizer names → HF optim names
_OPTIM_MAP = {
    "sgd": "sgd",
    "adam": "adamw_torch",
    "adamw": "adamw_torch",
}


def train_with_hf(
    config_path: str,
    overrides: Optional[dict] = None,
    resume_from: Optional[str] = None,
) -> Dict[str, Any]:
    """Train a model using HF Trainer with our YAML config.

    Args:
        config_path: Path to training YAML config.
        overrides: Optional config overrides (same as --override CLI).
        resume_from: Path to HF checkpoint directory to resume from.

    Returns:
        Training summary dict with metrics.
    """
    config_path = Path(config_path)
    config = load_config(str(config_path))
    if overrides:
        config = merge_configs(config, overrides)

    # Build model via our registry (same as native trainer)
    model = build_model(config)
    output_format = getattr(model, "output_format", "yolox")
    logger.info("Training with HF Trainer: output_format=%s", output_format)

    # Resolve data config
    data_cfg = config.get("data", {})
    dataset_config_path = data_cfg.get("dataset_config")
    if dataset_config_path:
        if not Path(dataset_config_path).is_absolute():
            dataset_config_path = str((config_path.parent / dataset_config_path).resolve())
        data_config = load_config(dataset_config_path)
    else:
        data_config = data_cfg

    base_dir = str(config_path.parent)

    # Build datasets based on task type
    train_dataset, eval_dataset, data_collator = _build_datasets(
        data_config, config, output_format, base_dir,
    )

    # Map our config → HF TrainingArguments
    training_args = _config_to_training_args(config, output_format, config_path)

    # Build compute_metrics based on task
    compute_metrics = _build_compute_metrics(output_format, config)

    # Build callbacks
    callbacks = _build_callbacks(config)

    # Create HF Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # Save our YAML configs to the output directory for lineage
    _save_configs(training_args.output_dir, config_path, data_config, dataset_config_path, config)

    # Train
    result = trainer.train(resume_from_checkpoint=resume_from)

    # Save final model
    trainer.save_model()

    summary = {
        "train_loss": result.training_loss,
        "total_epochs": int(result.metrics.get("epoch", 0)),
        "metrics": result.metrics,
    }
    logger.info("HF Trainer complete: %s", summary)
    return summary


def _build_datasets(
    data_config: dict,
    training_config: dict,
    output_format: str,
    base_dir: str,
) -> tuple:
    """Build train and eval datasets based on task type.

    Returns:
        (train_dataset, eval_dataset, data_collator)
    """
    if output_format == "classification":
        from core.p05_data.classification_dataset import (
            ClassificationDataset,
            build_classification_transforms,
            classification_collate_fn,
        )
        input_size = tuple(data_config["input_size"])
        mean = data_config.get("mean", IMAGENET_MEAN)
        std = data_config.get("std", IMAGENET_STD)

        train_transforms = build_classification_transforms(
            is_train=True, input_size=input_size, mean=mean, std=std,
        )
        eval_transforms = build_classification_transforms(
            is_train=False, input_size=input_size, mean=mean, std=std,
        )

        train_dataset = ClassificationDataset(
            data_config, split="train", transforms=train_transforms, base_dir=base_dir,
        )
        eval_dataset = ClassificationDataset(
            data_config, split="val", transforms=eval_transforms, base_dir=base_dir,
        )

        def hf_cls_collate(batch):
            """Collate for HF Trainer: uses 'images'+'targets' keys (model handles kwargs)."""
            result = classification_collate_fn(batch)
            # HF Trainer extracts labels from batch for compute_metrics
            labels = torch.stack(result["targets"])
            return {"images": result["images"], "targets": result["targets"], "labels": labels}

        return train_dataset, eval_dataset, hf_cls_collate

    elif output_format == "segmentation":
        from core.p05_data.segmentation_dataset import (
            SegmentationDataset,
            build_segmentation_transforms,
            segmentation_collate_fn,
        )
        input_size = tuple(data_config["input_size"])
        mean = data_config.get("mean", IMAGENET_MEAN)
        std = data_config.get("std", IMAGENET_STD)

        train_transforms = build_segmentation_transforms(
            is_train=True, input_size=input_size, mean=mean, std=std,
        )
        eval_transforms = build_segmentation_transforms(
            is_train=False, input_size=input_size, mean=mean, std=std,
        )

        train_dataset = SegmentationDataset(
            data_config, split="train", transforms=train_transforms, base_dir=base_dir,
        )
        eval_dataset = SegmentationDataset(
            data_config, split="val", transforms=eval_transforms, base_dir=base_dir,
        )
        return train_dataset, eval_dataset, segmentation_collate_fn

    else:
        from core.p05_data.detection_dataset import YOLOXDataset, collate_fn
        from core.p05_data.transforms import build_transforms

        input_size = tuple(data_config["input_size"])
        mean = data_config.get("mean", IMAGENET_MEAN)
        std = data_config.get("std", IMAGENET_STD)
        aug_config = training_config.get("augmentation", {})

        train_transforms = build_transforms(
            config=aug_config, is_train=True, input_size=input_size, mean=mean, std=std,
        )
        eval_transforms = build_transforms(
            config=aug_config, is_train=False, input_size=input_size, mean=mean, std=std,
        )

        train_dataset = YOLOXDataset(
            data_config, split="train", transforms=train_transforms, base_dir=base_dir,
        )
        eval_dataset = YOLOXDataset(
            data_config, split="val", transforms=eval_transforms, base_dir=base_dir,
        )
        return train_dataset, eval_dataset, collate_fn


def _config_to_training_args(
    config: dict,
    output_format: str,
    config_path: Path,
) -> TrainingArguments:
    """Map our YAML config keys to HF TrainingArguments."""
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    log_cfg = config.get("logging", {})
    ckpt_cfg = config.get("checkpoint", {})

    # Resolve output directory
    save_dir = log_cfg.get("save_dir")
    if save_dir:
        if not Path(save_dir).is_absolute():
            save_dir = str((config_path.parent / save_dir).resolve())
    else:
        # features/<name>/configs/06_training.yaml → parent.parent.name = <name>
        run_name = (
            log_cfg.get("run_name")
            or log_cfg.get("project")
            or config_path.parent.parent.name
        )
        save_dir = str(generate_run_dir(run_name, "06_training"))

    # Map optimizer name
    optim_name = _OPTIM_MAP.get(
        train_cfg.get("optimizer", "adamw").lower(), "adamw_torch"
    )

    # Map checkpoint metric — HF uses "eval_" prefix
    ckpt_metric = ckpt_cfg.get("metric", "val/mAP50")
    # Convert our metric names to HF's eval_ prefix format
    hf_metric = ckpt_metric.replace("val/", "eval_")

    # Warmup: convert epochs to steps
    warmup_epochs = train_cfg.get("warmup_epochs", 0)
    epochs = train_cfg.get("epochs", 100)
    warmup_ratio = warmup_epochs / epochs if epochs > 0 else 0.0

    return TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=epochs,
        learning_rate=train_cfg.get("lr", 0.001),
        weight_decay=train_cfg.get("weight_decay", 0.0005),
        per_device_train_batch_size=data_cfg.get("batch_size", 16),
        per_device_eval_batch_size=data_cfg.get("batch_size", 16),
        optim=optim_name,
        warmup_ratio=warmup_ratio,
        fp16=train_cfg.get("amp", True) and torch.cuda.is_available(),
        max_grad_norm=train_cfg.get("grad_clip", 35.0),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=ckpt_cfg.get("save_best", False),
        metric_for_best_model=hf_metric if ckpt_cfg.get("save_best", False) else None,
        greater_is_better=ckpt_cfg.get("mode", "max") == "max" if ckpt_cfg.get("save_best", False) else None,
        report_to="wandb" if log_cfg.get("wandb_project") else "none",
        run_name=log_cfg.get("run_name"),
        seed=config.get("seed", 42),
        dataloader_num_workers=data_cfg.get("num_workers", 4),
        dataloader_pin_memory=data_cfg.get("pin_memory", True),
        remove_unused_columns=False,  # Our datasets return custom dicts
        logging_steps=10,
    )


def _build_compute_metrics(output_format: str, config: dict):
    """Build a compute_metrics function for HF Trainer based on task type."""
    if output_format == "classification":
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            accuracy = (preds == labels).mean()
            result = {"accuracy": float(accuracy)}
            # Top-5 accuracy if enough classes
            num_classes = logits.shape[1]
            if num_classes >= 5:
                top5 = np.argsort(logits, axis=-1)[:, -5:]
                top5_correct = np.any(top5 == labels[:, None], axis=1).mean()
                result["top5_accuracy"] = float(top5_correct)
            return result
        return compute_metrics

    elif output_format == "segmentation":
        num_classes = config.get("model", {}).get("num_classes", 2)

        def compute_metrics(eval_pred):
            logits, masks = eval_pred
            preds = np.argmax(logits, axis=1)  # (N, H, W)
            intersection = np.zeros(num_classes)
            union = np.zeros(num_classes)
            for pred, gt in zip(preds, masks):
                for c in range(num_classes):
                    p = pred == c
                    g = gt == c
                    intersection[c] += np.logical_and(p, g).sum()
                    union[c] += np.logical_or(p, g).sum()
            iou = np.where(union > 0, intersection / (union + 1e-10), 0.0)
            return {"mIoU": float(np.mean(iou))}

        return compute_metrics

    else:
        logger.warning(
            "Detection metrics in HF Trainer are limited (mAP50 stub). "
            "For full mAP tracking, use backend=pytorch or run evaluate.py after training."
        )

        def compute_metrics(eval_pred):
            return {"mAP50": 0.0}

        return compute_metrics


def _build_callbacks(config: dict) -> list:
    """Build HF Trainer callbacks from our config."""
    callbacks = []

    patience = config.get("training", {}).get("patience", 0)
    if patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    return callbacks


def _save_configs(
    output_dir: str,
    config_path: Path,
    data_config: dict,
    dataset_config_path: Optional[str],
    resolved_config: dict,
) -> None:
    """Save our YAML configs to the HF output directory for lineage.

    Matches the naming convention used by native trainer and releases/.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy original training config
    if config_path.exists():
        shutil.copy2(config_path, output_dir / "06_training.yaml")

    # Copy data config
    if dataset_config_path and Path(dataset_config_path).exists():
        shutil.copy2(dataset_config_path, output_dir / "05_data.yaml")

    # Dump resolved config (with overrides applied)
    resolved_path = output_dir / "config_resolved.yaml"
    with open(resolved_path, "w") as f:
        yaml.dump(resolved_config, f, default_flow_style=False, sort_keys=False)

    logger.info("Saved configs to %s", output_dir)

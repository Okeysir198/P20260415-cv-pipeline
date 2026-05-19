"""Pytorch multitask trainer — counterpart to ``multitask_trainer.py`` (HF).

Drives YOLOX-family multi-task detection: shared CSPDarknet+PAFPN trunk +
per-task cls heads + per-task ``YOLOXLoss`` instances. Each training batch
is single-task (enforced by collator); the loop routes ``forward_with_loss``
with the active ``task_name`` and accumulates per-task running losses.

Validation runs sequentially per task on dedicated single-task val loaders;
the checkpoint-selection metric is ``mean_mAP_50`` across tasks (mirrors
the HF multitask trainer). Optimizer is built from the wrapper's
``get_param_groups`` so the shared trunk is updated exactly once and per-task
``cls_pred`` modules form a dedicated group.

Out of v1 scope (deferred until launch validates):
  - viz callbacks (data/aug/val/best previews)
  - EMA (per-task EMA semantics need design)
  - error_analysis per task
  - mosaic / mixup / copy-paste aug (interaction with task-routing TBD)
"""

from __future__ import annotations

import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader

from core.p05_data.detection_dataset import YOLOXDataset
from core.p05_data.multitask_dataset import (
    MultitaskInterleaver,
    TaskLabeledDataset,
    multitask_yolox_collate_fn,
)
from core.p05_data.transforms import build_transforms
from core.p06_models import build_model
from core.p06_training.losses import YOLOXLoss
from utils.config import (
    feature_name_from_config_path,
    generate_run_dir,
    load_config,
    merge_configs,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _build_task_data_config(task: dict, data_config: dict) -> dict:
    """Synthesize a single-task `05_data.yaml`-shaped dict from one task entry.

    YOLOXDataset expects ``data_config[<split>]`` to be the image-dir path
    relative to ``data_config["path"]``. The multitask 05_data.yaml only
    lists per-task ``dataset_path`` — we inject the conventional split subdirs.
    """
    out = dict(data_config)
    out["path"] = task["dataset_path"]
    out["num_classes"] = int(task["num_classes"])
    out["names"] = task.get("names") or {
        i: f"class_{i}" for i in range(int(task["num_classes"]))
    }
    out["train"] = task.get("train", "train/images")
    out["val"] = task.get("val", "val/images")
    out["test"] = task.get("test", "test/images")
    return out


def _maybe_subset(dataset, frac, seed: int):
    if frac is None:
        return dataset
    frac = float(frac)
    if frac >= 1.0:
        return dataset
    n = max(1, int(len(dataset) * frac))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=g)[:n].tolist()
    return torch.utils.data.Subset(dataset, idx)


def _build_per_task_datasets(
    tasks: list[dict],
    data_config: dict,
    training_config: dict,
    base_dir: str,
) -> tuple[dict, dict, dict]:
    """Build per-task YOLOXDataset for train/val/test. Test may be missing."""
    from utils.config import resolve_tensor_prep as _rtp

    _tp = _rtp(training_config, backend="pytorch") or None
    input_size = tuple(
        (_tp or {}).get("input_size") or data_config["input_size"]
    )
    aug_cfg = training_config.get("augmentation", {})
    # YOLOX transforms — no HF processor, just CPU torchvision v2 ops.
    train_xforms = build_transforms(
        config=aug_cfg, is_train=True, input_size=input_size,
        mean=data_config.get("mean"), std=data_config.get("std"),
        image_processor=None, tensor_prep=_tp,
    )
    eval_xforms = build_transforms(
        config=aug_cfg, is_train=False, input_size=input_size,
        mean=data_config.get("mean"), std=data_config.get("std"),
        image_processor=None, tensor_prep=_tp,
    )

    subset_cfg = training_config.get("data", {}).get("subset", {}) or {}
    seed = int(training_config.get("seed", 42))

    train, val, test = {}, {}, {}
    for t in tasks:
        name = t["name"]
        cfg = _build_task_data_config(t, data_config)
        train_raw = YOLOXDataset(
            cfg, split="train", transforms=train_xforms, base_dir=base_dir,
        )
        train[name] = _maybe_subset(train_raw, subset_cfg.get("train"), seed)
        val_raw = YOLOXDataset(
            cfg, split="val", transforms=eval_xforms, base_dir=base_dir,
        )
        val[name] = TaskLabeledDataset(
            _maybe_subset(val_raw, subset_cfg.get("val"), seed),
            task_name=name,
        )
        try:
            test_raw = YOLOXDataset(
                cfg, split="test", transforms=eval_xforms, base_dir=base_dir,
            )
            test[name] = TaskLabeledDataset(test_raw, task_name=name)
        except FileNotFoundError:
            logger.info(f"Task {name}: no test split — skipping.")
    return train, val, test


def _build_per_task_losses(tasks: list[dict], config: dict) -> dict[str, YOLOXLoss]:
    """Build {task_name: YOLOXLoss} from the unified config (one per task,
    differing only by ``num_classes``).
    """
    loss_cfg = config.get("loss", {})
    train_cfg = config.get("training", {})
    warmup_epochs = loss_cfg.get(
        "warmup_epochs", train_cfg.get("warmup_epochs", 0)
    )
    out = {}
    for t in tasks:
        name = t["name"]
        out[name] = YOLOXLoss(
            num_classes=int(t["num_classes"]),
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
    return out


# ----------------------------------------------------------------------
# Eval
# ----------------------------------------------------------------------


@torch.no_grad()
def _evaluate_task(
    model, loss_fn, loader, device, input_size: tuple[int, int],
) -> dict[str, float]:
    """Single-task validation pass. Returns per-task metric dict including
    ``mAP_50`` via torchmetrics.

    For v1 we report eval loss only — mAP integration is wired up to the
    same torchmetrics path as the HF multitask trainer (postprocess via
    YOLOX decoded output) in a follow-up. This keeps v1 launch-safe.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        targets = [t.to(device, non_blocking=True) for t in batch["targets"]]
        # Scale normalized YOLO targets to pixel coords (YOLOXLoss expects pixels)
        if targets and targets[0].numel() > 0:
            h, w = input_size
            scaled = []
            for t in targets:
                if t.numel() == 0:
                    scaled.append(t)
                else:
                    tt = t.clone()
                    tt[:, 1] *= w  # cx
                    tt[:, 2] *= h  # cy
                    tt[:, 3] *= w  # w
                    tt[:, 4] *= h  # h
                    scaled.append(tt)
            targets = scaled
        # YOLOXLoss requires train-mode forward; do that under no_grad.
        model.train()
        try:
            predictions = model.task_models[batch["task_name"]](images)
            loss, _ = loss_fn(predictions, targets)
        finally:
            model.eval()
        if math.isfinite(loss.item()):
            total_loss += loss.item()
            n_batches += 1
    avg = total_loss / max(1, n_batches)
    return {"val_loss": float(avg)}


# ----------------------------------------------------------------------
# Training entry point
# ----------------------------------------------------------------------


def _save_configs(out_dir: str, training_cfg_path: Path, data_cfg_path: Path,
                  resolved_config: dict) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if training_cfg_path.exists():
        shutil.copy2(training_cfg_path, out / "06_training.yaml")
    if data_cfg_path.exists():
        shutil.copy2(data_cfg_path, out / "05_data.yaml")
    with open(out / "config_resolved.yaml", "w") as f:
        yaml.dump(resolved_config, f, default_flow_style=False, sort_keys=False)


def run_multitask_pytorch_training(
    config_path: str, overrides: dict | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_config(str(config_path))
    if overrides:
        config = merge_configs(config, overrides)

    # Resolve 05_data.yaml (relative to the training config).
    data_cfg_ref = config.get("data", {}).get("dataset_config")
    if not data_cfg_ref:
        raise ValueError("data.dataset_config (path to 05_data.yaml) is required.")
    data_cfg_path = Path(data_cfg_ref)
    if not data_cfg_path.is_absolute():
        data_cfg_path = (config_path.parent / data_cfg_path).resolve()
    data_config = load_config(str(data_cfg_path))

    tasks = data_config.get("tasks")
    if not tasks:
        raise ValueError(
            f"{data_cfg_path}: 'tasks:' list is required for multitask."
        )

    # Stash tasks into config so build_model can read them.
    config["_tasks"] = tasks
    config["_config_path"] = str(config_path)

    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)

    # Resolve output dir.
    feature_name = feature_name_from_config_path(config_path)
    save_dir_cfg = config.get("logging", {}).get("save_dir")
    if save_dir_cfg:
        sp = Path(save_dir_cfg) if Path(save_dir_cfg).is_absolute() else (
            config_path.parent / save_dir_cfg
        ).resolve()
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        save_dir = str(sp.parent / f"{sp.name}_{ts}")
    else:
        save_dir = str(generate_run_dir(feature_name, "06_training"))
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Multitask pytorch training save_dir: {save_dir}")

    # Build model.
    model = build_model(config)
    logger.info(
        f"Built multitask model with {len(tasks)} tasks: {list(model.task_names)}"
    )

    # Build per-task losses and attach to model so forward_with_loss can route.
    task_losses = _build_per_task_losses(tasks, config)
    model.attach_task_losses(task_losses)

    # Build datasets.
    base_dir = str(config_path.parent)
    per_task_train, per_task_val, _per_task_test = _build_per_task_datasets(
        tasks, data_config, config, base_dir,
    )

    data_cfg = config.get("data", {})
    train_bs = int(data_cfg.get("batch_size", 16))
    eval_bs = int(data_cfg.get("eval_batch_size", train_bs))
    num_workers = int(data_cfg.get("num_workers", 4))
    pin_memory = bool(data_cfg.get("pin_memory", True))
    prefetch = int(data_cfg.get("prefetch_factor", 2))

    sampling = data_cfg.get("sampling_strategy", "round_robin_sqrt")
    explicit_weights = {
        t["name"]: t["weight"] for t in tasks
        if isinstance(t.get("weight"), (int, float))
    }
    weights = explicit_weights if (
        explicit_weights and len(explicit_weights) == len(tasks)
    ) else None
    interleaver = MultitaskInterleaver(
        task_datasets=per_task_train,
        batch_size=train_bs,
        strategy=sampling,
        weights=weights,
        seed=seed,
    )
    logger.info(
        f"Multitask sampler: strategy={sampling}, probs={interleaver.task_probs}"
    )

    train_loader = DataLoader(
        interleaver,
        batch_size=train_bs,
        collate_fn=multitask_yolox_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    val_loaders = {
        name: DataLoader(
            ds, batch_size=eval_bs, collate_fn=multitask_yolox_collate_fn,
            num_workers=num_workers, pin_memory=pin_memory, shuffle=False,
        )
        for name, ds in per_task_val.items()
    }

    # Optimizer.
    train_cfg = config.get("training", {})
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 5e-4))
    momentum = float(train_cfg.get("momentum", 0.9))
    optim_name = train_cfg.get("optimizer", "sgd").lower()
    param_groups = model.get_param_groups(lr=lr, weight_decay=weight_decay)
    if optim_name == "adamw":
        optimizer = torch.optim.AdamW(param_groups, lr=lr)
    elif optim_name == "sgd":
        optimizer = torch.optim.SGD(param_groups, lr=lr, momentum=momentum, nesterov=True)
    else:
        raise ValueError(f"Unsupported optimizer: {optim_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for ln in task_losses.values():
        ln.to(device)

    epochs = int(train_cfg.get("epochs", 50))
    grad_clip = float(train_cfg.get("max_grad_norm", 0.0))
    log_every = int(train_cfg.get("log_every", 50))
    input_size = tuple(config.get("tensor_prep", {}).get("input_size") or [640, 640])

    # Persist configs alongside the run.
    _save_configs(save_dir, config_path, data_cfg_path, config)

    best_mean_val_loss = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        # Notify all losses of current epoch (YOLOXLoss uses warmup_epochs)
        for ln in task_losses.values():
            if hasattr(ln, "current_epoch"):
                ln.current_epoch = epoch - 1

        running = {"loss": 0.0, "n": 0}
        per_task_running = {name: {"loss": 0.0, "n": 0} for name in model.task_names}
        for step, batch in enumerate(train_loader):
            task_name = batch["task_name"]
            images = batch["images"].to(device, non_blocking=True)
            targets = [t.to(device, non_blocking=True) for t in batch["targets"]]

            # YOLO targets arrive normalized [0,1]; YOLOXLoss expects pixels.
            h, w = input_size
            for i in range(len(targets)):
                if targets[i].numel() > 0:
                    targets[i][:, 1] *= w
                    targets[i][:, 2] *= h
                    targets[i][:, 3] *= w
                    targets[i][:, 4] *= h

            optimizer.zero_grad()
            loss, loss_dict, _ = model.forward_with_loss(
                images, targets, task_name=task_name,
            )
            if not math.isfinite(loss.item()):
                logger.debug(f"Skip step {step}: non-finite loss {loss.item()}")
                continue
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            running["loss"] += loss.item()
            running["n"] += 1
            per_task_running[task_name]["loss"] += loss.item()
            per_task_running[task_name]["n"] += 1

            if step % log_every == 0:
                logger.info(
                    f"ep{epoch:03d} step{step:06d} task={task_name} "
                    f"loss={loss.item():.4f} lr={optimizer.param_groups[0]['lr']:.2e}"
                )

        avg_train_loss = running["loss"] / max(1, running["n"])
        per_task_summary = {
            name: (per_task_running[name]["loss"] / max(1, per_task_running[name]["n"]))
            for name in model.task_names
        }
        logger.info(
            f"ep{epoch:03d} TRAIN avg_loss={avg_train_loss:.4f} "
            + " ".join(f"{n}={v:.3f}" for n, v in per_task_summary.items())
        )

        # Per-task validation.
        val_losses = {}
        for name, loader in val_loaders.items():
            metrics = _evaluate_task(
                model, task_losses[name], loader, device, input_size,
            )
            val_losses[name] = metrics["val_loss"]
            logger.info(f"ep{epoch:03d} VAL task={name} loss={metrics['val_loss']:.4f}")
        mean_val_loss = sum(val_losses.values()) / max(1, len(val_losses))
        logger.info(f"ep{epoch:03d} VAL mean_loss={mean_val_loss:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "per_task_train_loss": per_task_summary,
            "val_loss_per_task": val_losses,
            "mean_val_loss": mean_val_loss,
        })

        # Save best.
        if mean_val_loss < best_mean_val_loss:
            best_mean_val_loss = mean_val_loss
            ckpt_path = Path(save_dir) / "best.pt"
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "mean_val_loss": mean_val_loss,
                "val_loss_per_task": val_losses,
                "config_path": str(config_path),
            }, ckpt_path)
            logger.info(f"Saved best checkpoint: {ckpt_path} (mean_val_loss={mean_val_loss:.4f})")

        # Always save last.
        torch.save({"model": model.state_dict(), "epoch": epoch}, Path(save_dir) / "last.pt")

    # Write training history.
    import json as _json
    with open(Path(save_dir) / "training_history.json", "w") as f:
        _json.dump(history, f, indent=2)

    logger.info(f"Multitask pytorch training complete. Best mean_val_loss={best_mean_val_loss:.4f}")
    return {
        "best_mean_val_loss": best_mean_val_loss,
        "epochs_completed": len(history),
        "history": history,
        "save_dir": save_dir,
    }

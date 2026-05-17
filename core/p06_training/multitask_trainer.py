"""Multi-task HF Trainer entry point for D-FINE shared-trunk training.

Selected via ``training.backend: hf_multitask`` in the feature's
``06_training_*.yaml``. Tasks are listed in the referenced ``05_data.yaml``
as ``tasks: [{name, dataset_path, num_classes, names, weight}, ...]``.

Each batch is single-task (cls-head routing); per-task val datasets are
evaluated separately in ``evaluation_loop`` and combined into
``eval_mean_mAP_50`` (the checkpoint-selection metric).
"""

from __future__ import annotations

import math
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from loguru import logger
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.image_transforms import center_to_corners_format

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.p05_data.base_dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from core.p05_data.multitask_dataset import (  # noqa: E402
    MultitaskInterleaver,
    multitask_collate_fn,
)
from core.p06_models import build_model  # noqa: E402
from core.p06_training.hf_trainer import (  # noqa: E402
    EMACallback,
    _OPTIM_MAP,
    _maybe_subset,
    _wandb_credentials_available,
)
from utils.config import (  # noqa: E402
    feature_name_from_config_path,
    generate_run_dir,
    load_config,
    merge_configs,
)


# ----------------------------------------------------------------------
# Per-task dataset construction
# ----------------------------------------------------------------------


def _build_task_data_config(
    task_entry: dict, global_data_cfg: dict,
) -> dict:
    """Synthesize a single-task 05_data.yaml-style dict for one task.

    Standard layout assumed: ``<dataset_path>/{train,val,test}/images``.
    """
    return {
        "dataset_name": task_entry["name"],
        "path": task_entry["dataset_path"],
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": task_entry["names"],
        "num_classes": int(task_entry["num_classes"]),
        "input_size": list(global_data_cfg["input_size"]),
        "mean": list(global_data_cfg.get("mean", IMAGENET_MEAN)),
        "std": list(global_data_cfg.get("std", IMAGENET_STD)),
    }


def _build_per_task_datasets(
    tasks: list[dict],
    data_config: dict,
    training_config: dict,
    base_dir: str,
    processor,
) -> tuple[dict, dict, dict]:
    """Build {task_name: YOLOXDataset} for train/val/test splits.

    Test datasets may be missing on disk → silently dropped.
    """
    from core.p05_data.detection_dataset import YOLOXDataset
    from core.p05_data.multitask_dataset import TaskLabeledDataset
    from core.p05_data.transforms import build_transforms
    from utils.config import resolve_tensor_prep as _rtp

    _tp = _rtp(training_config, backend="hf") or None
    input_size = tuple(
        (_tp or {}).get("input_size") or data_config["input_size"]
    )
    aug_cfg = training_config.get("augmentation", {})
    train_xforms = build_transforms(
        config=aug_cfg, is_train=True, input_size=input_size,
        mean=data_config.get("mean"), std=data_config.get("std"),
        image_processor=processor, tensor_prep=_tp,
    )
    eval_xforms = build_transforms(
        config=aug_cfg, is_train=False, input_size=input_size,
        mean=data_config.get("mean"), std=data_config.get("std"),
        image_processor=processor, tensor_prep=_tp,
    )

    subset_cfg = training_config.get("data", {}).get("subset", {}) or {}
    seed = int(training_config.get("seed", 42))

    train, val, test = {}, {}, {}
    for t in tasks:
        name = t["name"]
        cfg = _build_task_data_config(t, data_config)
        train[name] = _maybe_subset(
            YOLOXDataset(cfg, split="train", transforms=train_xforms, base_dir=base_dir),
            subset_cfg.get("train"), seed,
        )
        # Val/test datasets are consumed by the multitask collator (which
        # requires per-sample task_name); wrap raw YOLOXDataset so its 3-tuple
        # samples become (image, target, task_name, path) 4-tuples.
        val_ds = _maybe_subset(
            YOLOXDataset(cfg, split="val", transforms=eval_xforms, base_dir=base_dir),
            subset_cfg.get("val"), seed,
        )
        val[name] = TaskLabeledDataset(val_ds, task_name=name)
        try:
            test_raw = YOLOXDataset(
                cfg, split="test", transforms=eval_xforms, base_dir=base_dir,
            )
            test[name] = TaskLabeledDataset(test_raw, task_name=name)
        except FileNotFoundError:
            logger.info("Task %s: no test split — skipping.", name)
    return train, val, test


# ----------------------------------------------------------------------
# Multitask HF Trainer
# ----------------------------------------------------------------------


class MultitaskHFTrainer(Trainer):
    """HF Trainer subclass for multi-task detection.

    Overrides:
      - ``compute_loss``: routes through ``model(task_name=...)``
      - ``evaluation_loop``: runs full per-task eval, returns ``mean_mAP_50``
      - ``_save``: torch.save (avoids safetensors shared-tensor footgun)
    """

    def __init__(
        self,
        *args,
        per_task_val_datasets: dict | None = None,
        per_task_test_datasets: dict | None = None,
        per_task_compute_metrics: dict | None = None,
        input_size: tuple[int, int] = (640, 640),
        score_threshold: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._per_task_val = per_task_val_datasets or {}
        self._per_task_test = per_task_test_datasets or {}
        self._per_task_compute_metrics = per_task_compute_metrics or {}
        self._input_size = input_size
        self._score_threshold = float(score_threshold)

    # -- routing through task-specific head --
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None,
    ):
        task_name = inputs.pop("task_name", None)
        outputs = model(
            pixel_values=inputs["pixel_values"],
            labels=inputs.get("labels"),
            task_name=task_name,
        )
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        return (loss, outputs) if return_outputs else loss

    # -- per-task evaluation --
    def evaluate(self, *args, **kwargs):
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Caller can pass eval_dataset directly; otherwise we run per-task.
        if kwargs.get("eval_dataset") is not None:
            return super().evaluate(*args, **kwargs)
        prefix = kwargs.get("metric_key_prefix", "eval")
        return self._evaluate_per_task(prefix=prefix)

    def _evaluate_per_task(self, prefix: str = "eval") -> dict[str, float]:
        """Run a separate evaluation pass per task, merge into a single
        metric dict. Adds ``eval_mean_mAP_50`` averaged across tasks for
        checkpoint selection.
        """
        all_metrics: dict[str, float] = {}
        per_task_map50: list[float] = []
        datasets = self._per_task_val if prefix == "eval" else self._per_task_test
        for task_name, ds in datasets.items():
            logger.info("[%s] task=%s — running eval (n=%d)", prefix, task_name, len(ds))
            self._active_eval_task = task_name
            # Swap compute_metrics to the task-specific one so per-class
            # mAP keys carry that task's class names.
            task_cm = self._per_task_compute_metrics.get(task_name)
            if task_cm is not None:
                self.compute_metrics = task_cm
            # Use a single-task eval_dataset; collator still emits task_name.
            metrics = super().evaluate(
                eval_dataset=ds,
                metric_key_prefix=f"{prefix}_{task_name}",
            )
            for k, v in metrics.items():
                all_metrics[k] = v
            map50 = metrics.get(f"{prefix}_{task_name}_map_50")
            if map50 is not None and not math.isnan(map50):
                per_task_map50.append(float(map50))
            self._active_eval_task = None
        if per_task_map50:
            mean_v = sum(per_task_map50) / len(per_task_map50)
            all_metrics[f"{prefix}_mean_mAP_50"] = mean_v
            all_metrics[f"{prefix}_mean_map_50"] = mean_v  # HF-style alias
            logger.info("[%s] mean_mAP_50 = %.4f (across %d tasks)",
                        prefix, mean_v, len(per_task_map50))
        # HF Trainer caches state.log_history with these; emit.
        self.log(all_metrics)
        return all_metrics

    def _save(self, output_dir=None, state_dict=None):
        import os as _os
        output_dir = output_dir or self.args.output_dir
        _os.makedirs(output_dir, exist_ok=True)
        model = self.model
        if state_dict is None:
            state_dict = model.state_dict()
        torch.save(state_dict, _os.path.join(output_dir, "pytorch_model.bin"))
        # Persist per-task config snapshots for standalone reload.
        primary = getattr(model, "primary_task", None)
        if primary and hasattr(model, "task_models"):
            inner = model.task_models[primary]
            if hasattr(inner, "config"):
                inner.config.save_pretrained(output_dir)
            proc = model.task_processors.get(primary)
            if proc is not None and hasattr(proc, "save_pretrained"):
                proc.save_pretrained(output_dir)
        torch.save(self.args, _os.path.join(output_dir, "training_args.bin"))


# ----------------------------------------------------------------------
# compute_metrics: per-batch mAP for the active task
# ----------------------------------------------------------------------


def _build_per_task_compute_metrics(
    processor,
    input_size: tuple[int, int],
    score_threshold: float = 0.0,
    id2label: dict[int, str] | None = None,
):
    """Returns a compute_metrics callable for HF Trainer.

    Emits per-class mAP keys (``map_per_class_<classname>``, ``mAP50_per_class_<classname>``)
    in addition to scalar metrics. Trainer prefixes everything with
    ``eval_<task_name>_`` downstream, so the final keys look like e.g.
    ``eval_fire_smoke_map_50_per_class_fire``.
    """
    from torchmetrics.detection import MeanAveragePrecision

    H_in, W_in = int(input_size[0]), int(input_size[1])

    def compute_metrics(eval_pred):
        from types import SimpleNamespace
        evaluator = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
        predictions, label_ids = eval_pred.predictions, eval_pred.label_ids
        for batch_pred, batch_labels in zip(predictions, label_ids, strict=True):
            batch_logits = torch.as_tensor(batch_pred[1])
            batch_boxes = torch.as_tensor(batch_pred[2])
            batch_size = batch_logits.shape[0]
            target_sizes = torch.tensor([[H_in, W_in]] * batch_size)
            hf_output = SimpleNamespace(logits=batch_logits, pred_boxes=batch_boxes)
            preds = processor.post_process_object_detection(
                hf_output, threshold=score_threshold, target_sizes=target_sizes,
            )
            preds = [{k: v.detach().cpu() for k, v in p.items()} for p in preds]
            targets = []
            scale = torch.tensor([W_in, H_in, W_in, H_in], dtype=torch.float32)
            for lbl in batch_labels:
                boxes_norm = torch.as_tensor(lbl["boxes"], dtype=torch.float32)
                cls = torch.as_tensor(lbl["class_labels"], dtype=torch.long)
                if boxes_norm.numel() == 0:
                    targets.append({
                        "boxes": torch.zeros(0, 4, dtype=torch.float32),
                        "labels": torch.zeros(0, dtype=torch.long),
                    })
                    continue
                boxes_xyxy = center_to_corners_format(boxes_norm) * scale
                targets.append({"boxes": boxes_xyxy, "labels": cls})
            evaluator.update(preds, targets)
        raw = evaluator.compute()
        classes = raw.get("classes")
        out: dict[str, float] = {}
        for k, v in raw.items():
            if k == "classes":
                continue
            if isinstance(v, torch.Tensor):
                if v.ndim == 0:
                    out[k] = float(v.item())
                elif v.ndim == 1:
                    ids = classes.tolist() if classes is not None else list(range(v.numel()))
                    for cid, val in zip(ids, v.tolist(), strict=True):
                        name = (id2label.get(int(cid), str(int(cid)))
                                if id2label else str(int(cid)))
                        out[f"{k}_per_class_{name}"] = float(val)
        if "map_50" in out:
            out["mAP50"] = out["map_50"]
        return out

    return compute_metrics


# ----------------------------------------------------------------------
# Config → TrainingArguments mapping
# ----------------------------------------------------------------------


def _config_to_training_args(
    config: dict, config_path: Path,
) -> TrainingArguments:
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    log_cfg = config.get("logging", {})
    ckpt_cfg = config.get("checkpoint", {})

    # Resolve output directory under the feature folder.
    feature_name = feature_name_from_config_path(config_path)
    save_dir_cfg = log_cfg.get("save_dir")
    if save_dir_cfg:
        sp = Path(save_dir_cfg) if Path(save_dir_cfg).is_absolute() else (
            config_path.parent / save_dir_cfg
        ).resolve()
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        save_dir = str(sp.parent / f"{sp.name}_{ts}")
    else:
        save_dir = str(generate_run_dir(feature_name, "06_training"))

    optim_name = _OPTIM_MAP.get(
        train_cfg.get("optimizer", "adamw").lower(), "adamw_torch"
    )

    warmup_steps = train_cfg.get("warmup_steps")
    warmup_epochs = train_cfg.get("warmup_epochs", 0)
    epochs = train_cfg.get("epochs", 100)
    warmup_ratio = 0.0 if warmup_steps else (warmup_epochs / epochs if epochs > 0 else 0.0)
    lr_scheduler_type = train_cfg.get("scheduler", "linear")
    lr_scheduler_kwargs = train_cfg.get("lr_scheduler_kwargs") or None

    ckpt_metric = ckpt_cfg.get("metric", "mean_mAP_50")
    hf_metric = ckpt_metric.replace("val/", "eval_")
    if not hf_metric.startswith("eval_"):
        hf_metric = f"eval_{hf_metric}"

    report_to = log_cfg.get("report_to") or "tensorboard"
    if isinstance(report_to, str):
        report_to_list = [report_to] if report_to != "none" else []
    else:
        report_to_list = list(report_to)
    if any(r == "wandb" for r in report_to_list) and not _wandb_credentials_available():
        logger.warning("wandb credentials missing — dropping from report_to.")
        report_to_list = [r for r in report_to_list if r != "wandb"]
    report_to_resolved = report_to_list if report_to_list else "none"

    return TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=epochs,
        learning_rate=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        per_device_train_batch_size=data_cfg.get("batch_size", 16),
        per_device_eval_batch_size=data_cfg.get(
            "eval_batch_size", data_cfg.get("batch_size", 16)
        ),
        optim=optim_name,
        warmup_steps=warmup_steps or 0,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        fp16=train_cfg.get("amp", False),
        bf16=train_cfg.get("bf16", False),
        max_grad_norm=train_cfg.get("max_grad_norm", 0.1),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=ckpt_cfg.get("save_total_limit", 3),
        load_best_model_at_end=ckpt_cfg.get("save_best", False),
        metric_for_best_model=hf_metric if ckpt_cfg.get("save_best", False) else None,
        greater_is_better=ckpt_cfg.get("mode", "max") == "max",
        report_to=report_to_resolved,
        seed=config.get("seed", 42),
        data_seed=config.get("seed", 42),
        dataloader_num_workers=data_cfg.get("num_workers", 4),
        dataloader_pin_memory=data_cfg.get("pin_memory", True),
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        eval_accumulation_steps=4,
        logging_steps=10,
    )


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def _save_configs(output_dir: str, training_cfg_path: Path,
                  data_cfg_path: Path, resolved_config: dict) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if training_cfg_path.exists():
        shutil.copy2(training_cfg_path, out / "06_training.yaml")
    if data_cfg_path.exists():
        shutil.copy2(data_cfg_path, out / "05_data.yaml")
    with open(out / "config_resolved.yaml", "w") as f:
        yaml.dump(resolved_config, f, default_flow_style=False, sort_keys=False)


def run_multitask_training(
    config_path: str, overrides: dict | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_config(str(config_path))
    if overrides:
        config = merge_configs(config, overrides)

    # Hard-enforce D-FINE invariant.
    arch = (config.get("model", {}).get("arch", "") or "").lower()
    if arch.startswith("dfine") and config.get("training", {}).get("bf16", False):
        raise ValueError(
            "D-FINE multitask diverges under bf16 (val mAP stalls). "
            "Set training.bf16=False."
        )

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
        raise ValueError(f"{data_cfg_path}: 'tasks:' list is required for multitask.")

    # Stash tasks into config so build_model can read them.
    config["_tasks"] = tasks
    config["_config_path"] = str(config_path)

    # Seed before build_model (HF reinit RNG sensitivity).
    from transformers import set_seed as _set_seed
    _set_seed(int(config.get("seed", 42)))

    model = build_model(config)
    logger.info("Built multitask model with %d tasks: %s",
                len(tasks), list(model.task_names))

    # Build datasets (using the primary task's processor for transform parity).
    base_dir = str(config_path.parent)
    primary_proc = model.processor
    per_task_train, per_task_val, per_task_test = _build_per_task_datasets(
        tasks, data_config, config, base_dir, primary_proc,
    )

    # Interleave train across tasks.
    data_cfg = config.get("data", {})
    batch_size = int(data_cfg.get("batch_size", 16))
    sampling = data_cfg.get("sampling_strategy", "round_robin_sqrt")
    weights = None
    explicit_weights = {
        t["name"]: t["weight"] for t in tasks
        if isinstance(t.get("weight"), (int, float))
    }
    if explicit_weights and len(explicit_weights) == len(tasks):
        weights = explicit_weights
    interleaver = MultitaskInterleaver(
        task_datasets=per_task_train,
        batch_size=batch_size,
        strategy=sampling,
        weights=weights,
        seed=int(config.get("seed", 42)),
    )
    logger.info("Multitask sampler: strategy=%s, probs=%s",
                sampling, interleaver.task_probs)

    # Pick any val dataset as a placeholder for HF Trainer's `eval_dataset`
    # (we override `evaluate` to run per-task instead).
    placeholder_val = next(iter(per_task_val.values()))

    training_args = _config_to_training_args(config, config_path)

    # Pull input_size for compute_metrics + per-task scoring.
    from utils.config import resolve_tensor_prep as _rtp
    tp_eval = _rtp(config, backend="hf") or {}
    input_size = tuple(
        tp_eval.get("input_size") or data_config.get("input_size") or (640, 640)
    )
    score_thr = float(config.get("evaluation", {}).get("score_threshold", 0.0))
    # Per-task compute_metrics — each carries its own id2label so per-class
    # keys land as eval_<task>_map_<metric>_per_class_<classname>.
    per_task_compute_metrics = {}
    for task_name, inner_model in model.task_models.items():
        per_task_compute_metrics[task_name] = _build_per_task_compute_metrics(
            primary_proc,
            input_size=input_size,
            score_threshold=score_thr,
            id2label=getattr(inner_model.config, "id2label", None),
        )
    # Placeholder for Trainer init — the MultitaskHFTrainer swaps in the
    # right per-task callable before each super().evaluate() call.
    compute_metrics = next(iter(per_task_compute_metrics.values()))

    # Callbacks: EarlyStopping + EMA (subset of single-task callback suite —
    # full viz callbacks expect single-task semantics, deferred).
    callbacks: list = []
    train_cfg = config.get("training", {})
    patience = train_cfg.get("patience", 0)
    if patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
    if train_cfg.get("ema", False):
        callbacks.append(EMACallback(
            decay=train_cfg.get("ema_decay", 0.9998),
            warmup_steps=train_cfg.get("ema_warmup_steps", 2000),
        ))

    trainer = MultitaskHFTrainer(
        model=model,
        args=training_args,
        train_dataset=interleaver,
        eval_dataset=placeholder_val,  # not used; evaluate() runs per-task
        compute_metrics=compute_metrics,
        data_collator=multitask_collate_fn,
        callbacks=callbacks,
        per_task_val_datasets=per_task_val,
        per_task_test_datasets=per_task_test,
        per_task_compute_metrics=per_task_compute_metrics,
        input_size=input_size,
        score_threshold=score_thr,
    )

    _save_configs(training_args.output_dir, config_path, data_cfg_path, config)

    result = trainer.train()
    trainer.save_model()

    summary = {
        "train_loss": result.training_loss,
        "total_epochs": int(result.metrics.get("epoch", 0)),
        "metrics": result.metrics,
    }

    # Final test eval per task.
    if per_task_test:
        # Strip ES callback to silence "metric not found" warning during test.
        trainer.callback_handler.callbacks = [
            cb for cb in trainer.callback_handler.callbacks
            if not isinstance(cb, EarlyStoppingCallback)
        ]
        try:
            test_metrics = trainer._evaluate_per_task(prefix="test")
            summary["test_metrics"] = test_metrics
            import json as _json
            with open(Path(training_args.output_dir) / "test_results.json", "w") as f:
                _json.dump(test_metrics, f, indent=2, sort_keys=True)
        except Exception as e:
            logger.warning("Per-task test eval failed (training still succeeded): %s", e)

    logger.info("Multitask training complete: %s", summary)
    return summary

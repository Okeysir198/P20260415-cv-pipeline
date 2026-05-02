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

import os as _os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.image_transforms import center_to_corners_format

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from loguru import logger  # noqa: E402

from core.p05_data.base_dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from core.p06_models import build_model  # noqa: E402
from core.p06_training.hf_callbacks import (  # noqa: E402
    HFAugLabelGridCallback,
    HFDataLabelGridCallback,
    HFDatasetStatsCallback,
    HFValPredictionCallback,
)
from utils.config import generate_run_dir, load_config, merge_configs  # noqa: E402


class FreezeBackboneCallback(TrainerCallback):
    """Freeze the backbone for the first N epochs, then unfreeze.

    Activated when ``training.freeze_backbone_epochs > 0``. Matches the
    convention documented in `features/CLAUDE.md` (head-only warm-up,
    then full fine-tune) which previously only worked on the pytorch
    backend via ``training.freeze: ["backbone"]``.

    Mechanics:
    - ``on_train_begin``: sets ``requires_grad = False`` on every param
      whose name contains ``"backbone"``.
    - ``on_epoch_begin``: at epoch ``freeze_backbone_epochs``, flips
      ``requires_grad`` back to True. Adam / AdamW pick the new params
      up immediately because HF Trainer already put them in the optimizer
      param-group — they just weren't receiving gradients while frozen.

    Optimizer-state caveat: when unfreezing, Adam's momentum stats for
    the backbone params start from zero. That's the intended behaviour
    — the first few post-unfreeze steps act like a mini-warmup.
    """

    def __init__(self, freeze_epochs: int) -> None:
        self.freeze_epochs = int(freeze_epochs)
        self._frozen = False
        self._frozen_param_count = 0

    def _flip_backbone(self, model, requires_grad: bool) -> int:
        n = 0
        for name, p in model.named_parameters():
            if "backbone" in name:
                p.requires_grad = requires_grad
                n += 1
        return n

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is None or self.freeze_epochs <= 0:
            return control
        self._frozen_param_count = self._flip_backbone(model, requires_grad=False)
        self._frozen = True
        logger.info(
            "FreezeBackboneCallback: froze %d backbone params for the first %d epochs",
            self._frozen_param_count, self.freeze_epochs,
        )
        return control

    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        if not self._frozen or model is None:
            return control
        if state.epoch is not None and state.epoch >= self.freeze_epochs:
            n = self._flip_backbone(model, requires_grad=True)
            self._frozen = False
            logger.info(
                "FreezeBackboneCallback: unfroze %d backbone params at epoch %d",
                n, int(state.epoch),
            )
        return control


class EMACallback(TrainerCallback):
    """HF TrainerCallback wrapping our `ModelEMA` for the HF backend.

    Maintains a shadow copy of model params, updated after each optimizer step
    with an exponential decay schedule. Before each evaluation, swaps the live
    model's params with the EMA's — Trainer runs eval against the averaged
    weights (typically +1-2% mAP on detection) — then swaps back so training
    continues from the raw params. Snapshots the EMA weights to
    `<output_dir>/ema_model.bin` at end of training.

    Activates when `training.ema: true` is set in the config. Uses our
    in-repo `ModelEMA` (`core/p06_training/trainer.py:ModelEMA`) so behaviour
    matches the pytorch backend exactly (same decay + warmup curve).
    """

    def __init__(self, decay: float = 0.9998, warmup_steps: int = 2000):
        self._decay = decay
        self._warmup_steps = warmup_steps
        self._ema = None
        self._backup_sd = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        from core.p06_training.trainer import ModelEMA
        if model is None:
            return control
        self._ema = ModelEMA(model, decay=self._decay, warmup_steps=self._warmup_steps)
        logger.info("EMA enabled (decay=%s, warmup_steps=%s)",
                    self._decay, self._warmup_steps)
        return control

    def on_optimizer_step(self, args, state, control, model=None, **kwargs):
        """Updated once per real gradient step (accounting for grad accumulation)."""
        if self._ema is not None and model is not None:
            self._ema.update(model)
        return control

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """No-op hook — swap happens in the pair of prediction hooks below.
        Kept for symmetry / future extension.
        """
        return control

    def on_prediction_step(self, args, state, control, **kwargs):
        return control

    # HF Trainer 5.x fires `on_pre_evaluate` before evaluation starts and
    # `on_evaluate` after. We swap weights into/out-of the live model around
    # evaluation so the test-set number reflects EMA.
    def on_pre_evaluate(self, args, state, control, model=None, **kwargs):
        if self._ema is not None and model is not None:
            self._backup_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(self._ema.ema_model.state_dict(), strict=False)
        return control

    def on_evaluate_end(self, args, state, control, model=None, **kwargs):
        if self._backup_sd is not None and model is not None:
            model.load_state_dict(self._backup_sd, strict=False)
            self._backup_sd = None
        return control

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if self._ema is not None:
            out_path = Path(args.output_dir) / "ema_model.bin"
            torch.save(self._ema.ema_model.state_dict(), out_path)
            logger.info("EMA weights saved to %s", out_path)
        return control


class _DetectionTrainer(Trainer):
    """HF Trainer subclass that saves via the wrapped HF model's own
    `save_pretrained(..., safe_serialization=False)`.

    RT-DETRv2 / D-FINE share the decoder `class_embed` + `bbox_embed`
    modules across decoder layers (same `nn.Module` instances referenced
    multiple times). When HF Trainer's default `_save` flattens
    `HFDetectionModel.state_dict()` and passes it to
    `safetensors.torch.save_file`, safetensors rejects the duplicate tensor
    aliases with a `shared tensors` `RuntimeError`. In transformers 5.x the
    `save_safetensors=False` TrainingArguments knob was removed, so we
    override `_save` directly instead.
    """

    def _save(self, output_dir=None, state_dict=None):
        import torch  # local import — trainer.py set up determinism already
        output_dir = output_dir or self.args.output_dir
        _os.makedirs(output_dir, exist_ok=True)
        model = self.model

        # Save the *wrapper's* state_dict (keys prefixed with `hf_model.`) via
        # plain torch.save so HF Trainer's `_load_best_model` can reload into
        # the same wrapper without a prefix mismatch. safetensors can't handle
        # RT-DETRv2's shared class_embed / bbox_embed tensors, and the newer
        # transformers removed the `save_safetensors=False` arg, so we go
        # directly to torch.save — which handles aliased tensors fine.
        if state_dict is None:
            state_dict = model.state_dict()
        torch.save(state_dict, _os.path.join(output_dir, "pytorch_model.bin"))

        # Also dump the inner model config + processor for standalone
        # inference (same directory layout as `save_pretrained`).
        inner = getattr(model, "hf_model", None)
        if inner is not None and hasattr(inner, "config"):
            inner.config.save_pretrained(output_dir)
        processor = getattr(model, "processor", None)
        if processor is not None and hasattr(processor, "save_pretrained"):
            processor.save_pretrained(output_dir)

        torch.save(self.args, _os.path.join(output_dir, "training_args.bin"))

    def create_optimizer(self):
        """Build AdamW with a backbone vs head param-group split.

        Activates when `training.lr_backbone` is set in the config (stashed
        onto `self.args.backbone_lr` by `_config_to_training_args`). Matches
        the official Peterande/D-FINE recipe (backbone 2.5e-5, head 2.5e-4)
        and the RT-DETR reference (backbone 1e-5, head 1e-4). When unset,
        falls back to HF Trainer's default single-LR optimizer.

        The decay/no-decay split (LayerNorm + bias → wd=0) is preserved
        inside each group via `self.get_decay_parameter_names`, so the
        official behaviour is a strict superset of the default.
        """
        if self.optimizer is not None:
            return self.optimizer
        lr_bb = getattr(self.args, "backbone_lr", None)
        if lr_bb is None:
            return super().create_optimizer()

        opt_model = self.model
        decay_names = set(self.get_decay_parameter_names(opt_model))
        lr_head = self.args.learning_rate
        wd = self.args.weight_decay
        bb_decay, bb_nodecay, hd_decay, hd_nodecay = [], [], [], []
        for n, p in opt_model.named_parameters():
            if not p.requires_grad:
                continue
            is_backbone = "backbone" in n
            in_decay = n in decay_names
            bucket = (
                bb_decay if is_backbone and in_decay else
                bb_nodecay if is_backbone else
                hd_decay if in_decay else
                hd_nodecay
            )
            bucket.append(p)
        param_groups = [
            {"params": bb_decay,   "lr": lr_bb,   "weight_decay": wd},
            {"params": bb_nodecay, "lr": lr_bb,   "weight_decay": 0.0},
            {"params": hd_decay,   "lr": lr_head, "weight_decay": wd},
            {"params": hd_nodecay, "lr": lr_head, "weight_decay": 0.0},
        ]
        param_groups = [g for g in param_groups if g["params"]]
        n_bb = sum(len(g["params"]) for g in param_groups if g["lr"] == lr_bb)
        n_hd = sum(len(g["params"]) for g in param_groups if g["lr"] == lr_head)
        logger.info(
            "Layered AdamW: backbone lr=%g (%d params), head lr=%g (%d params), wd=%g",
            lr_bb, n_bb, lr_head, n_hd, wd,
        )
        cls, kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        kwargs.pop("lr", None)  # per-group LRs win
        self.optimizer = cls(param_groups, **kwargs)
        return self.optimizer


# Mapping: our optimizer names → HF optim names
_OPTIM_MAP = {
    "sgd": "sgd",
    "adam": "adamw_torch",
    "adamw": "adamw_torch",
}


def _maybe_subset(dataset, fraction_or_count, seed: int):
    """Wrap `dataset` in `torch.utils.data.Subset` if caller requested a
    subset. `fraction_or_count` can be None (no subset), a float in (0, 1]
    (fraction), or an int (absolute sample count). Uses a deterministic
    shuffle so multi-seed runs on the same subset see the same images.
    """
    if fraction_or_count is None:
        return dataset
    n = len(dataset)
    if isinstance(fraction_or_count, float):
        k = max(1, int(round(n * fraction_or_count)))
    else:
        k = int(fraction_or_count)
    if k >= n:
        return dataset
    import numpy as _np
    import torch.utils.data as _td
    rng = _np.random.default_rng(seed)
    indices = sorted(rng.choice(n, size=k, replace=False).tolist())
    return _td.Subset(dataset, indices)


def _hf_detection_collate(batch):
    """Collate `YOLOXDataset.__getitem__` tuples into HF DETR batch dict.

    When image_processor is used, targets is already a dict with
    ``{"class_labels": LongTensor[N], "boxes": FloatTensor[N, 4]}`` — pass
    through directly.  Otherwise targets is an (N, 5) tensor with
    ``[class_id, cx_norm, cy_norm, w_norm, h_norm]``; split columns into the
    HF label format.
    """
    images = torch.stack([sample[0] for sample in batch])
    labels = []
    for _, targets, _ in batch:
        if hasattr(targets, "class_labels"):
            labels.append(targets)
        elif isinstance(targets, torch.Tensor) and targets.numel() == 0:
            labels.append({
                "class_labels": torch.zeros(0, dtype=torch.long),
                "boxes": torch.zeros(0, 4, dtype=torch.float32),
            })
        else:
            labels.append({
                "class_labels": targets[:, 0].long(),
                "boxes": targets[:, 1:5].float(),
            })
    return {"pixel_values": images, "labels": labels}


def _build_detection_compute_metrics(
    image_processor, input_size, id2label=None, score_threshold=0.0
):
    """Real `compute_metrics` for HF Trainer, detection task.

    Mirrors qubvel's reference `MAPEvaluator`
    (`notebooks/detr_finetune_reference/reference_rtdetr_v2/finetune.py`):
    post-process each batch via `image_processor.post_process_object_detection`
    to get xyxy-pixel predictions, convert normalized-cxcywh targets to
    xyxy-pixel, then accumulate in `torchmetrics.MeanAveragePrecision`.

    Returns both scalar metrics (`map`, `map_50`, `map_75`, `map_small`, ...)
    and per-class metrics (`map_<classname>`, `mar_100_<classname>`) when
    ``id2label`` is supplied — matches the reference MAPEvaluator output
    shape so ``trainer_state.json`` has the same fields as qubvel's.

    This path requires `eval_do_concat_batches=False` in `TrainingArguments`
    (detection labels are variable-length per image; concat would crash).
    """
    from torchmetrics.detection import MeanAveragePrecision

    H_in, W_in = int(input_size[0]), int(input_size[1])

    def compute_metrics(eval_pred):
        evaluator = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
        predictions, label_ids = eval_pred.predictions, eval_pred.label_ids

        for batch_pred, batch_labels in zip(predictions, label_ids, strict=True):
            # HF wraps `ModelOutput(loss, logits, pred_boxes, ...)` into a tuple
            # per-batch when `eval_do_concat_batches=False`. Index 0 is loss,
            # index 1 = logits (B, Q, C), index 2 = pred_boxes (B, Q, 4) in
            # normalized cxcywh — same convention as qubvel's notebook.
            batch_logits = torch.as_tensor(batch_pred[1])
            batch_boxes = torch.as_tensor(batch_pred[2])
            batch_size = batch_logits.shape[0]
            target_sizes = torch.tensor([[H_in, W_in]] * batch_size)

            hf_output = SimpleNamespace(logits=batch_logits, pred_boxes=batch_boxes)
            preds = image_processor.post_process_object_detection(
                hf_output, threshold=score_threshold, target_sizes=target_sizes,
            )
            # torchmetrics expects {"boxes","scores","labels"} cpu tensors
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
        classes = raw.get("classes")  # LongTensor of present class ids

        out: dict[str, float] = {}
        for metric_name, value in raw.items():
            if metric_name == "classes":
                continue
            if isinstance(value, torch.Tensor):
                if value.ndim == 0:
                    # Scalar metric — e.g. map, map_50, map_75, map_small,
                    # mar_100, etc. Keep under its original torchmetrics key.
                    out[metric_name] = float(value.item())
                elif value.ndim == 1:
                    # Per-class vector — unpack to `<metric>_<classname>`.
                    # The `classes` tensor says which class id each entry
                    # corresponds to (only classes seen in eval show up).
                    ids = classes.tolist() if classes is not None else list(range(value.numel()))
                    for cid, v in zip(ids, value.tolist(), strict=True):
                        class_name = (id2label.get(int(cid), str(int(cid)))
                                       if id2label else str(int(cid)))
                        out[f"{metric_name}_{class_name}"] = float(v)
            else:
                out[metric_name] = float(value)

        # Mirror `mAP50` under our in-repo convention key so configs using
        # `metric: val/mAP50` still resolve correctly (HF prefixes with eval_).
        if "map_50" in out:
            out["mAP50"] = out["map_50"]
        return out

    return compute_metrics


def train_with_hf(
    config_path: str,
    overrides: dict | None = None,
    resume_from: str | None = None,
) -> dict[str, Any]:
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

    # Resolve data config first so `names` can be forwarded into the model's
    # id2label/label2id via `build_hf_model` (which reads `config.data.names`).
    data_cfg = config.get("data", {})
    dataset_config_path = data_cfg.get("dataset_config")
    if dataset_config_path:
        if not Path(dataset_config_path).is_absolute():
            dataset_config_path = str((config_path.parent / dataset_config_path).resolve())
        data_config = load_config(dataset_config_path)
    else:
        data_config = data_cfg

    # Merge resolved data names into training config's data section so the model
    # sees them. The custom pytorch trainer does this implicitly via its own
    # resolution path (`_loaded_data_cfg`); the HF backend needs it done
    # explicitly here because `build_model` only sees the training config.
    if "names" in data_config and "names" not in config.get("data", {}):
        config.setdefault("data", {})["names"] = data_config["names"]

    base_dir = str(config_path.parent)

    # Seed *before* build_model. `from_pretrained(ignore_mismatched_sizes=True)`
    # reinits the class/bbox/denoising heads using whatever RNG state torch
    # booted with; HF Trainer's own seed call happens later inside
    # Trainer.__init__ (too late for this). Matches qubvel's recipe + the
    # RT-DETRv2 reproduction convention.
    from transformers import set_seed as _hf_set_seed
    _hf_set_seed(int(config.get("seed", 42)))

    # Build model via our registry (same as native trainer)
    model = build_model(config)
    output_format = getattr(model, "output_format", "yolox")
    logger.info("Training with HF Trainer: output_format=%s", output_format)

    _validate_hf_backend_config(config, output_format)

    # Contract validator — hard-errors on tensor_prep/backend/processor
    # misalignment. Run AFTER build_model so the processor's forced
    # attributes are observable.
    from utils.config import _validate_tensor_prep as _vtp
    _vtp(config, backend="hf", processor=getattr(model, "processor", None))

    # Build datasets based on task type
    _ip = (
        model.processor if output_format in {"detr", "keypoint"} else None
    )
    train_dataset, eval_dataset, data_collator = _build_datasets(
        data_config, config, output_format, base_dir,
        image_processor=_ip,
    )
    # Build test dataset up-front so the val-prediction callback can render
    # best-checkpoint predictions on it (on_train_end). `None` when no test
    # split exists — callback handles that case gracefully.
    test_dataset = _try_build_test_dataset(data_config, config, output_format, base_dir)

    # Map our config → HF TrainingArguments
    training_args = _config_to_training_args(config, output_format, config_path)

    # Build compute_metrics based on task
    if output_format == "detr":
        # Detection needs the image_processor for post_process_object_detection,
        # the input_size to know the target coord space, and id2label so
        # per-class mAP entries in trainer_state.json get human-readable keys
        # (e.g. `map_Coverall` instead of `map_0`) — matches the reference's
        # MAPEvaluator output shape.
        from utils.config import resolve_tensor_prep as _rtp
        _tp_for_eval = _rtp(config, backend="hf")
        input_size = tuple(
            (_tp_for_eval or {}).get("input_size") or data_config["input_size"]
        )
        id2label = None
        hf_inner = getattr(model, "hf_model", None)
        if hf_inner is not None and getattr(hf_inner, "config", None) is not None:
            id2label = getattr(hf_inner.config, "id2label", None)
        # Canonical mAP uses threshold=0.0 (torchmetrics ranks the full PR
        # curve). Raising it truncates predictions and mechanically lowers
        # reported mAP; only do so when matching an external baseline.
        eval_thr = float(config.get("evaluation", {}).get("score_threshold", 0.0))
        if eval_thr > 0.0:
            logger.warning(
                "evaluation.score_threshold=%g (non-canonical; 0.0 is standard "
                "for mAP — higher values lower reported mAP)", eval_thr,
            )
        compute_metrics = _build_detection_compute_metrics(
            image_processor=model.processor,
            input_size=input_size,
            id2label=id2label,
            score_threshold=eval_thr,
        )
    else:
        compute_metrics = _build_compute_metrics(output_format, config, data_config)

    # Compute subset indices to pass into data-preview callbacks so their
    # stats/grids reflect the 20%-subset run (not the full underlying dataset).
    # Test is always full (no subset_cfg.test applied) → None is correct.
    import torch.utils.data as _tud
    def _subset_indices(ds):
        return list(ds.indices) if isinstance(ds, _tud.Subset) else None
    subset_map = {
        "train": _subset_indices(train_dataset),
        "val":   _subset_indices(eval_dataset),
        "test":  None,
    }
    # Full (pre-subset) split sizes for 00_dataset_info provenance. When a
    # dataset is wrapped in Subset, reach for the underlying dataset's len so
    # the info reflects the dataset size, not the subset size.
    def _full_len(ds):
        if ds is None:
            return 0
        inner = getattr(ds, "dataset", ds)
        return len(inner)
    full_sizes = {
        "train": _full_len(train_dataset),
        "val":   _full_len(eval_dataset),
        "test":  _full_len(test_dataset),
    }

    # Build callbacks (incl. viz-bridge for detection)
    callbacks = _build_callbacks(
        config,
        output_format=output_format,
        model=model,
        data_config=data_config,
        base_dir=base_dir,
        save_dir=training_args.output_dir,
        subset_map=subset_map,
        test_dataset=test_dataset,
        train_dataset=train_dataset,
        config_path=config_path,
        dataset_config_path=dataset_config_path,
        full_sizes=full_sizes,
    )

    # Create HF Trainer. Detection uses a subclass that avoids safetensors'
    # shared-weights error; other tasks use the vanilla Trainer.
    TrainerClass = _DetectionTrainer if output_format == "detr" else Trainer
    trainer = TrainerClass(
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

    # Save final model (load_best_model_at_end=True means this is the best ckpt)
    trainer.save_model()

    summary = {
        "train_loss": result.training_loss,
        "total_epochs": int(result.metrics.get("epoch", 0)),
        "metrics": result.metrics,
    }

    # Final test-set eval, matching the reference notebook's
    # `trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="eval")`
    # call. `test_dataset` was built earlier (before _build_callbacks so
    # val-prediction callback could render best-checkpoint grids on it).
    if test_dataset is not None:
        logger.info("Running final test-set evaluation on best checkpoint...")
        try:
            test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
            summary["test_metrics"] = test_metrics
            # Also write test_results.json next to HF's all_results.json for lineage.
            import json as _json
            test_json = Path(training_args.output_dir) / "test_results.json"
            with open(test_json, "w") as f:
                _json.dump(test_metrics, f, indent=2, sort_keys=True)
            logger.info("Test metrics written to %s", test_json)
        except Exception as e:  # pragma: no cover — don't fail the whole run on test eval issues
            logger.warning("Test-set evaluation failed (training still succeeded): %s", e)
    else:
        logger.info("No test split available — skipping test-set evaluation.")

    logger.info("HF Trainer complete: %s", summary)
    return summary


_SUPPORTED_HF_TASKS = {"detr", "classification", "segmentation", "keypoint"}


def _wandb_credentials_available() -> bool:
    """Best-effort check for wandb auth without raising. Returns True if
    WANDB_API_KEY is set OR a logged-in `wandb` session exists in netrc/env.
    Used to drop wandb from `report_to` rather than letting HF Trainer's
    wandb callback hard-fail at trainer.train() setup.
    """
    if _os.environ.get("WANDB_API_KEY"):
        return True
    try:
        import wandb  # noqa: PLC0415
        # `wandb.api.api_key` reads from netrc/env without triggering a login.
        return bool(getattr(wandb.api, "api_key", None))
    except Exception:
        return False


def _validate_hf_backend_config(config: dict, output_format: str) -> None:
    """Fail fast on config combos the HF backend can't honour, rather than
    silently downgrading features and surprising the user at training time.

    Keeps this short on purpose — only hard incompatibilities go here, not
    style lint (different feature / different taste).
    """
    train_cfg = config.get("training", {})
    hard_errors: list = []
    soft_warnings: list = []

    # 1) Supported task types
    if output_format not in _SUPPORTED_HF_TASKS:
        hard_errors.append(
            f"training.backend='hf' does not yet support output_format="
            f"'{output_format}'. Supported: {sorted(_SUPPORTED_HF_TASKS)}. "
            f"Use training.backend='pytorch' for this task."
        )

    # 2) Features the HF backend doesn't (yet) implement
    if train_cfg.get("gpu_augment", False):
        soft_warnings.append(
            "training.gpu_augment=True is ignored on the HF backend "
            "(HF Trainer uses its own DataLoader; GPU augmentation runs "
            "only on the pytorch backend). Aug runs on CPU; torchvision "
            "v2 and albumentations are at parity on this codebase."
        )

    # 3) Detection-specific sanity
    if output_format == "detr":
        if train_cfg.get("amp", False) and not train_cfg.get("bf16", False):
            hard_errors.append(
                "Detection: training.amp=True (fp16) overflows the DETR "
                "decoder. Use training.bf16=True (RTX 5090/A100+) or "
                "training.amp=False."
            )
        # D-FINE's distribution-focused loss stalls val mAP ~0.15 under bf16;
        # RT-DETRv2 is bf16-safe. Block the foot-gun rather than letting the
        # run silently under-train.
        arch = (config.get("model", {}).get("arch", "") or "").lower()
        if arch.startswith("dfine") and train_cfg.get("bf16", False):
            hard_errors.append(
                "D-FINE diverges under bf16 (val mAP stalls ~0.15). Set "
                "training.bf16=False (use fp32) — fp16/amp also overflows. "
                "RT-DETRv2 is bf16-safe; only D-FINE is affected."
            )
        aug_cfg = config.get("augmentation", {})
        if aug_cfg.get("mosaic", False):
            soft_warnings.append(
                "augmentation.mosaic=True on a DETR-family model — mosaic "
                "is designed for anchor-based detectors; DETR-family models "
                "do not benefit from it and it often hurts."
            )

    for w in soft_warnings:
        logger.warning("[hf_trainer config] %s", w)
    if hard_errors:
        raise ValueError(
            "HF Trainer config validation failed:\n  - "
            + "\n  - ".join(hard_errors)
        )


def _try_build_test_dataset(data_config, config, output_format, base_dir):
    """Build the test-split dataset only if one exists on disk. Returns None
    if the split doesn't exist or the task doesn't yet support HF-Trainer eval.

    Delegates split resolution to `YOLOXDataset` itself (same codepath as the
    train/val loaders) rather than reinventing it — catches `FileNotFoundError`
    cleanly when a dataset has no test split.
    """
    if output_format != "detr":
        # Classification / segmentation test-dataset construction mirrors
        # `_build_datasets`; add parallel branches here when those backends
        # grow an end-of-training test eval need.
        return None

    from core.p05_data.detection_dataset import YOLOXDataset
    from core.p05_data.transforms import build_transforms
    from utils.config import resolve_tensor_prep as _rtp

    _tp = _rtp(config, backend="hf") or None
    input_size = tuple(
        (_tp or {}).get("input_size") or data_config["input_size"]
    )
    aug_config = config.get("augmentation", {})
    eval_transforms = build_transforms(
        config=aug_config, is_train=False, input_size=input_size,
        mean=data_config.get("mean"), std=data_config.get("std"),
        tensor_prep=_tp,
    )
    try:
        return YOLOXDataset(
            data_config, split="test", transforms=eval_transforms, base_dir=base_dir,
        )
    except FileNotFoundError:
        logger.info("No test split on disk — skipping test-set evaluation.")
        return None
    except Exception as e:  # pragma: no cover
        logger.info("Could not build test dataset (%s) — skipping.", e)
        return None


def _build_datasets(
    data_config: dict,
    training_config: dict,
    output_format: str,
    base_dir: str,
    image_processor=None,
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
            """Collate for HF Trainer: emits 'pixel_values'+'labels' so stock
            Trainer can call model(**batch) directly."""
            result = classification_collate_fn(batch)
            labels = torch.stack(result["targets"])
            return {"pixel_values": result["images"], "labels": labels}

        return train_dataset, eval_dataset, hf_cls_collate

    elif output_format == "keypoint":
        from core.p05_data.keypoint_dataset import (
            KeypointTopDownDataset,
            keypoint_topdown_collate_fn,
        )
        model_cfg = training_config.get("model", {})
        bbox_padding = float(model_cfg.get("bbox_padding", 1.25))
        heatmap_cfg = training_config.get("training", {}).get("heatmap", {}) or {}
        sigma = float(heatmap_cfg.get("sigma", 2.0))

        train_dataset = KeypointTopDownDataset(
            data_config=data_config, split="train",
            processor=image_processor,
            bbox_padding=bbox_padding, heatmap_sigma=sigma,
            is_train=True, base_dir=base_dir,
        )
        eval_dataset = KeypointTopDownDataset(
            data_config=data_config, split="val",
            processor=image_processor,
            bbox_padding=bbox_padding, heatmap_sigma=sigma,
            is_train=False, base_dir=base_dir,
        )

        subset_cfg = training_config.get("data", {}).get("subset", {}) or {}
        seed = int(training_config.get("seed", 42))
        train_dataset = _maybe_subset(train_dataset, subset_cfg.get("train"), seed)
        eval_dataset = _maybe_subset(eval_dataset, subset_cfg.get("val"), seed)
        return train_dataset, eval_dataset, keypoint_topdown_collate_fn

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

        def hf_seg_collate(batch):
            """HF-Trainer-compatible seg collate: {pixel_values, labels}."""
            result = segmentation_collate_fn(batch)
            labels = torch.stack(result["targets"]).long()
            return {"pixel_values": result["images"], "labels": labels}

        subset_cfg = training_config.get("data", {}).get("subset", {}) or {}
        seed = int(training_config.get("seed", 42))
        train_dataset = _maybe_subset(train_dataset, subset_cfg.get("train"), seed)
        eval_dataset = _maybe_subset(eval_dataset, subset_cfg.get("val"), seed)
        return train_dataset, eval_dataset, hf_seg_collate

    else:
        from core.p05_data.detection_dataset import YOLOXDataset
        from core.p05_data.transforms import build_transforms
        from utils.config import resolve_tensor_prep as _rtp

        _tp = _rtp(training_config, backend="hf") or None
        input_size = tuple(
            (_tp or {}).get("input_size") or data_config["input_size"]
        )
        mean = (_tp or {}).get("mean") or data_config.get("mean", IMAGENET_MEAN)
        std = (_tp or {}).get("std") or data_config.get("std", IMAGENET_STD)
        aug_config = training_config.get("augmentation", {})

        train_transforms = build_transforms(
            config=aug_config, is_train=True, input_size=input_size, mean=mean, std=std,
            image_processor=image_processor,
            tensor_prep=_tp,
        )
        eval_transforms = build_transforms(
            config=aug_config, is_train=False, input_size=input_size, mean=mean, std=std,
            image_processor=image_processor,
            tensor_prep=_tp,
        )

        train_dataset = YOLOXDataset(
            data_config, split="train", transforms=train_transforms, base_dir=base_dir,
        )
        eval_dataset = YOLOXDataset(
            data_config, split="val", transforms=eval_transforms, base_dir=base_dir,
        )
        # Respect `data.subset.{train,val}` the same way our custom pytorch
        # trainer does (via `torch.utils.data.Subset`). The HF path bypasses
        # `build_dataloader` where that wrapping normally happens, so we
        # replicate it here. Accepts int (N samples) or float in (0, 1].
        subset_cfg = training_config.get("data", {}).get("subset", {}) or {}
        seed = int(training_config.get("seed", 42))
        train_dataset = _maybe_subset(train_dataset, subset_cfg.get("train"), seed)
        eval_dataset = _maybe_subset(eval_dataset, subset_cfg.get("val"), seed)
        # HF Trainer calls `model(**batch)` expecting `pixel_values` + `labels`
        # where labels is a list[dict] per-image in HF DETR format. YOLOXDataset
        # returns (image, targets_normalized_cxcywh, path); our HF DETR labels
        # are also normalized cxcywh so we just split the class col off and
        # hand it through — no pixel-scale dance like the custom pytorch trainer.
        return train_dataset, eval_dataset, _hf_detection_collate


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
        save_path = Path(save_dir) if Path(save_dir).is_absolute() else (config_path.parent / save_dir).resolve()
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        save_dir = str(save_path.parent / f"{save_path.name}_{ts}")
    else:
        # features/<name>/configs/06_training.yaml → parent.parent.name = <name>
        feature_name = config_path.parent.parent.name
        explicit_run_name = log_cfg.get("run_name") or log_cfg.get("project")
        run_name = explicit_run_name or feature_name
        # Stale `logging.run_name` silently creates ghost folders under
        # features/<run_name>/runs/ instead of the actual feature folder.
        # Warn loudly so the user catches it before training finishes.
        if explicit_run_name and explicit_run_name != feature_name:
            logger.warning(
                f"logging.run_name={explicit_run_name!r} differs from feature folder "
                f"{feature_name!r} — runs will land in features/{run_name}/runs/ "
                f"(likely a stale config). Drop run_name or set it to {feature_name!r}."
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

    # Warmup: prefer explicit warmup_steps (what the reference notebook pins),
    # fall back to warmup_epochs for legacy feature configs.
    warmup_steps = train_cfg.get("warmup_steps")
    warmup_epochs = train_cfg.get("warmup_epochs", 0)
    epochs = train_cfg.get("epochs", 100)
    warmup_ratio = 0.0 if warmup_steps else (warmup_epochs / epochs if epochs > 0 else 0.0)

    # lr scheduler: pass through from config (cosine/linear/...). HF Trainer
    # defaults to "linear"; keep that behaviour when config doesn't specify.
    lr_scheduler_type = train_cfg.get("scheduler", "linear")
    # For non-detection tasks we keep the current concat-batch behaviour.
    eval_do_concat_batches = output_format != "detr"

    # Resolve report_to up-front so we can drop wandb when the host has no
    # credentials. HF Trainer's wandb callback hard-fails at trainer.train()
    # setup if neither WANDB_API_KEY nor a logged-in netrc is found, killing
    # the run before epoch 1. Detect early and fall back to tensorboard.
    report_to = log_cfg.get("report_to") or (
        ["wandb", "tensorboard"] if log_cfg.get("wandb_project") else "tensorboard"
    )
    if isinstance(report_to, str):
        report_to_list = [report_to]
    elif isinstance(report_to, (list, tuple)):
        report_to_list = list(report_to)
    else:
        report_to_list = []
    if any(r == "wandb" for r in report_to_list) and not _wandb_credentials_available():
        logger.warning(
            "wandb requested in logging.report_to but no credentials found "
            "(WANDB_API_KEY unset and no logged-in wandb session). Dropping "
            "wandb to avoid hard-failing trainer setup; tensorboard kept. "
            "Run `wandb login` to re-enable."
        )
        report_to_list = [r for r in report_to_list if r != "wandb"]
    report_to_resolved = report_to_list if report_to_list else "none"

    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=epochs,
        learning_rate=train_cfg.get("lr", 0.001),
        weight_decay=train_cfg.get("weight_decay", 0.0005),
        per_device_train_batch_size=data_cfg.get("batch_size", 16),
        per_device_eval_batch_size=data_cfg.get(
            "eval_batch_size", data_cfg.get("batch_size", 16)
        ),
        optim=optim_name,
        warmup_steps=warmup_steps or 0,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        fp16=train_cfg.get("amp", False),
        bf16=train_cfg.get("bf16", False),
        max_grad_norm=train_cfg.get("max_grad_norm", train_cfg.get("grad_clip", 35.0)),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=ckpt_cfg.get("save_best", False),
        metric_for_best_model=hf_metric if ckpt_cfg.get("save_best", False) else None,
        greater_is_better=(
            ckpt_cfg.get("mode", "max") == "max" if ckpt_cfg.get("save_best", False) else None
        ),
        report_to=report_to_resolved,
        run_name=log_cfg.get("run_name"),
        seed=config.get("seed", 42),
        data_seed=config.get("seed", 42),
        dataloader_num_workers=data_cfg.get("num_workers", 4),
        dataloader_pin_memory=data_cfg.get("pin_memory", True),
        remove_unused_columns=False,  # Our datasets return custom dicts
        eval_do_concat_batches=eval_do_concat_batches,
        logging_steps=10,
    )
    # Layered LR — stashed as a dynamic attribute (TrainingArguments is a
    # @dataclass but not frozen). `_DetectionTrainer.create_optimizer` reads
    # it to build a backbone vs head AdamW param-group split (official D-FINE
    # recipe: backbone 2.5e-5, head 2.5e-4). Unset → falls back to HF default.
    training_args.backbone_lr = train_cfg.get("lr_backbone")

    # Keypoint task: tell HF Trainer to treat target_heatmap/target_weight as
    # labels so prediction_step recognises them (without this, eval_loss is
    # never computed because has_labels=False).
    if output_format == "keypoint":
        training_args.label_names = ["target_heatmap", "target_weight"]

    return training_args


def _build_compute_metrics(
    output_format: str, config: dict, data_config: dict | None = None,
):
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

        ignore_index = int(config.get("training", {}).get("ignore_index", -1))

        def compute_metrics(eval_pred):
            logits, masks = eval_pred
            logits_t = torch.from_numpy(logits)
            # SegFormer emits logits at H/4 — upsample to mask resolution.
            logits_t = torch.nn.functional.interpolate(
                logits_t, size=masks.shape[-2:], mode="bilinear", align_corners=False,
            )
            preds = logits_t.argmax(dim=1).numpy()  # (N, H, W)
            intersection = np.zeros(num_classes)
            union = np.zeros(num_classes)
            for pred, gt in zip(preds, masks, strict=True):
                valid = gt != ignore_index if ignore_index >= 0 else np.ones_like(gt, dtype=bool)
                for c in range(num_classes):
                    if c == ignore_index:
                        continue
                    p = (pred == c) & valid
                    g = (gt == c) & valid
                    intersection[c] += np.logical_and(p, g).sum()
                    union[c] += np.logical_or(p, g).sum()
            iou = np.where(union > 0, intersection / (union + 1e-10), 0.0)
            return {"mean_iou": float(np.mean(iou[union > 0])) if (union > 0).any() else 0.0}

        return compute_metrics

    elif output_format == "keypoint":
        # Top-down keypoint compute_metrics — PCK + OKS-AP via numpy.
        # Sigmas come from 05_data.yaml::oks_sigmas (falls back to the
        # COCO 17-kpt defaults). Input HW from data config / tensor_prep.
        from core.p08_evaluation.keypoint_metrics import (
            build_compute_metrics_keypoint,
        )
        from utils.config import resolve_tensor_prep as _rtp

        sigmas = (data_config or {}).get("oks_sigmas")
        input_hw = None
        if data_config and data_config.get("input_size"):
            input_hw = tuple(data_config["input_size"])
        if input_hw is None:
            _tp = _rtp(config, backend="hf") or {}
            input_hw = tuple(_tp.get("input_size") or (256, 192))
        stride = int(
            (config.get("training", {}).get("heatmap", {}) or {}).get("stride", 4)
        )
        return build_compute_metrics_keypoint(
            sigmas=sigmas, input_hw=input_hw, stride=stride,
        )

    else:
        logger.warning(
            "Detection metrics in HF Trainer are limited (mAP50 stub). "
            "For full mAP tracking, use backend=pytorch or run evaluate.py after training."
        )

        def compute_metrics(eval_pred):
            return {"mAP50": 0.0}

        return compute_metrics


def _build_callbacks(
    config: dict,
    output_format: str = "yolox",
    model=None,
    data_config: dict | None = None,
    base_dir: str | None = None,
    save_dir: str | None = None,
    subset_map: dict[str, list[int] | None] | None = None,
    test_dataset=None,
    train_dataset=None,
    config_path: Path | None = None,
    dataset_config_path: str | None = None,
    full_sizes: dict[str, int] | None = None,
) -> list:
    """Build HF Trainer callbacks from our config.

    Always adds:
    - `EarlyStoppingCallback` if `training.patience > 0`.

    Adds for detection tasks via native HF TrainerCallback subclasses
    (`core/p06_training/hf_callbacks.py`):
    - `DatasetStatsLogger` (always on)
    - `DataLabelGridLogger` (if `training.data_viz.enabled`)
    - `AugLabelGridLogger`  (if `training.aug_viz.enabled`)
    - `ValPredictionLogger` for val (if `training.val_viz.enabled`)
    - `ValPredictionLogger` for train (if `training.train_viz.enabled`)
    """
    callbacks = []

    train_cfg = config.get("training", {})
    patience = train_cfg.get("patience", 0)
    if patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    # EMA — HF-native TrainerCallback that wraps our ModelEMA. Activates when
    # `training.ema: true`. Same decay defaults as the pytorch backend.
    if train_cfg.get("ema", False):
        callbacks.append(EMACallback(
            decay=train_cfg.get("ema_decay", 0.9998),
            warmup_steps=train_cfg.get("ema_warmup_steps", 2000),
        ))

    # Freeze-backbone warmup — `training.freeze_backbone_epochs: N` freezes
    # backbone params for the first N epochs, then unfreezes. Stabilises
    # small-data DETR fine-tunes where the matcher destabilises the
    # backbone during the warmup phase.
    freeze_epochs = train_cfg.get("freeze_backbone_epochs", 0)
    if freeze_epochs and int(freeze_epochs) > 0:
        callbacks.append(FreezeBackboneCallback(freeze_epochs=int(freeze_epochs)))

    # Viz callbacks — native HF TrainerCallback subclasses
    # (core/p06_training/hf_callbacks.py). Gated on canonical task membership:
    # detection/classification/segmentation/keypoint all supported. Unknown
    # tasks bail gracefully (no viz, training still runs).
    from core.p06_training._common import task_from_output_format
    if data_config is None or save_dir is None:
        return callbacks
    task = task_from_output_format(output_format)
    if task not in {"detection", "classification", "segmentation", "keypoint"}:
        return callbacks

    splits = train_cfg.get("data_viz", {}).get("splits", ["train", "val"])
    input_size = tuple(data_config.get("input_size", (640, 640)))

    # DatasetStatsLogger — always on, matching custom-trainer policy.
    # subset_map threads the active data.subset.{train,val,test} indices so
    # data_preview reflects what the run actually trained on, not the full
    # underlying dataset. Mirrors the pytorch-backend DatasetStatsLogger
    # (callbacks.py::_subset_indices).
    data_viz = train_cfg.get("data_viz", {})
    from utils.config import feature_name_from_config_path as _feature_name
    callbacks.append(HFDatasetStatsCallback(
        save_dir=save_dir, data_config=data_config, base_dir=base_dir or "",
        splits=splits,
        subsets=subset_map,
        dpi=data_viz.get("dpi", 120),
        training_config=config,
        training_config_path=str(config_path) if config_path else None,
        data_config_path=dataset_config_path,
        feature_name=_feature_name(str(config_path)) if config_path else None,
        full_sizes=full_sizes,
    ))

    if data_viz.get("enabled", True):
        callbacks.append(HFDataLabelGridCallback(
            save_dir=save_dir, splits=splits,
            data_config=data_config, base_dir=base_dir or "",
            task=task,
            subsets=subset_map,
            num_samples=data_viz.get("num_samples", 16),
            grid_cols=data_viz.get("grid_cols", 4),
            thickness=data_viz.get("thickness", 2),
            text_scale=data_viz.get("text_scale", 0.4),
            dpi=data_viz.get("dpi", 120),
        ))

    aug_viz = train_cfg.get("aug_viz", {})
    # Aug-label grid — task-aware via HFDataLabelGridCallback's _render_gt_panel
    # dispatch (detection=boxes, classification=banner, segmentation=mask,
    # keypoint=dots+skeleton). Runs for every supported task.
    if aug_viz.get("enabled", True):
        callbacks.append(HFAugLabelGridCallback(
            save_dir=save_dir,
            splits=aug_viz.get("splits", ["train"]),
            data_config=data_config,
            aug_config=config.get("augmentation", {}),
            base_dir=base_dir or "",
            input_size=input_size,
            task=task,
            subsets=subset_map,
            num_samples=aug_viz.get("num_samples", 16),
            grid_cols=aug_viz.get("grid_cols", 4),
            thickness=aug_viz.get("thickness", 2),
            text_scale=aug_viz.get("text_scale", 0.4),
            dpi=aug_viz.get("dpi", 120),
        ))

    val_viz = train_cfg.get("val_viz", {})
    best_viz = train_cfg.get("best_viz", {})
    class_names = {int(k): str(v) for k, v in data_config.get("names", {}).items()}
    # Register the callback whenever EITHER per-epoch val grids OR post-train
    # best/test artifacts are wanted. Internal gates below disable each hook
    # independently so HPO sweeps can opt into post-train-only.
    want_val_viz = val_viz.get("enabled", True)
    want_best_viz = best_viz.get("enabled", True)
    if want_val_viz or want_best_viz:
        callbacks.append(HFValPredictionCallback(
            save_dir=save_dir, class_names=class_names, input_size=input_size,
            num_samples=val_viz.get("num_samples", 12),
            conf_threshold=val_viz.get("conf_threshold", 0.05),
            grid_cols=val_viz.get("grid_cols", 2),
            test_dataset=test_dataset if want_best_viz else None,
            train_dataset=train_dataset if want_best_viz else None,
            best_num_samples=best_viz.get("num_samples", 16),
            best_conf_threshold=best_viz.get("conf_threshold", 0.1),
            enable_epoch_end=want_val_viz,
            enable_train_end=want_best_viz,
            data_config=data_config,
            base_dir=base_dir,
        ))

    # Step-by-step transform pipeline viz — fires once on train-begin. Catches
    # double/missing-normalize footguns, box-format drift, and broken
    # inverse-normalize algebra before the first GPU forward pass.
    # Task-aware: render_transform_pipeline dispatches by task (detection walks
    # paired boxes via tv_tensors; classification/segmentation/keypoint use a
    # simpler per-task walker with _render_gt_panel for GT overlay).
    transform_viz = train_cfg.get("transform_viz", {})
    if transform_viz.get("enabled", True):
        from core.p06_training.callbacks_viz import TransformPipelineCallback
        callbacks.append(TransformPipelineCallback(
            save_dir=save_dir,
            data_config=data_config,
            training_config=config,
            base_dir=base_dir or "",
            class_names=class_names,
            max_samples=transform_viz.get("max_samples", 5),
            task=task,
        ))

    # train_viz would run the same viz on the train_dataloader — not wired
    # in the native HF callback path (HFValPredictionCallback reads HF's
    # eval_dataloader specifically). Add a `split="train"` variant here if
    # per-epoch train-set predictions become important for a future feature.

    return callbacks


def _save_configs(
    output_dir: str,
    config_path: Path,
    data_config: dict,
    dataset_config_path: str | None,
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

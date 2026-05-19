"""YOLOX multi-task detector.

Shared CSPDarknet backbone + PAFPN neck + per-head reg/obj branches, with
N per-task classification branches. Each forward routes through one task's
cls predictions while reusing all shared modules.

Implementation strategy (parallel to ``dfine_multitask.py``):

1. Build N full ``YOLOXModel`` instances, one per task, each with that
   task's ``num_classes``. ``YOLOXLoss`` works out-of-the-box for each.
2. After construction, replace the SHARED submodules in models[1..N] with
   *references* to model[0]'s submodules. PyTorch deduplicates parameters
   by tensor-object identity in ``named_parameters()`` — shared modules
   counted exactly once.

Task-specific (per-model):
    - ``heads[i].cls_pred``   for i in {0, 1, 2}   (final 1x1 conv per FPN level)

Shared (referenced from task[0]):
    - ``backbone``                              (CSPDarknet)
    - ``neck``                                  (PAFPN)
    - ``heads[i].stem``           for each i    (1x1 channel projection)
    - ``heads[i].cls_convs``      for each i    (2x 3x3 conv refiner — task-agnostic features)
    - ``heads[i].reg_convs``                    (regression refiner)
    - ``heads[i].reg_pred``                     (box geometry)
    - ``heads[i].obj_pred``                     (objectness)

Forward: ``forward(x, task_name=str)`` routes to the task's model;
predictions have shape (B, N_anchors, 5 + num_classes_task).

The custom-impl YOLOX path is the only one supported by this wrapper.
Official Megvii adapter (``yolox.py::_OfficialYOLOXAdapter``) wraps the
upstream package's monolithic ``YOLOX`` class and would need a deeper
port. v1 covers custom impl only.
"""

from __future__ import annotations

from typing import Any

import torch
from loguru import logger
from torch import nn

from core.p06_models.base import DetectionModel
from core.p06_models.registry import register_model
from core.p06_models.yolox import YOLOXModel


# YOLOX size presets (matches yolox.py builder). Width and depth multipliers.
_YOLOX_PRESETS = {
    "nano": (0.33, 0.25),
    "tiny": (0.33, 0.375),
    "s":    (0.33, 0.50),
    "m":    (0.67, 0.75),
    "l":    (1.0,  1.0),
}


def _share_heads_except_cls_pred(target: YOLOXModel, source: YOLOXModel) -> None:
    """Replace shared head submodules in ``target`` with references to those
    in ``source``, EXCEPT each head's ``cls_pred`` (task-specific final conv).
    """
    # Whole-trunk sharing
    target.backbone = source.backbone
    target.neck = source.neck
    # Per-FPN-level: share everything except cls_pred
    for tgt_head, src_head in zip(target.heads, source.heads, strict=True):
        tgt_head.stem = src_head.stem
        tgt_head.cls_convs = src_head.cls_convs
        tgt_head.reg_convs = src_head.reg_convs
        tgt_head.reg_pred = src_head.reg_pred
        tgt_head.obj_pred = src_head.obj_pred
        # cls_pred stays per-task — already independently initialised by ctor


class YOLOXMultitaskModel(DetectionModel):
    """Multi-task YOLOX with shared trunk + per-task cls heads.

    Use ``forward(x, task_name=str)`` for both training and inference.
    Returns the decoded predictions tensor in the same format as single-task
    YOLOX: ``(B, N_anchors, 5 + num_classes_task)`` where the layout is
    ``[cx, cy, w, h, obj_logit, cls_0_logit, ..., cls_C_logit]`` (logits in
    training, sigmoid-activated in eval).
    """

    def __init__(
        self,
        task_models: dict[str, YOLOXModel],
        primary_task: str,
    ) -> None:
        super().__init__()
        self.task_models = nn.ModuleDict(task_models)
        self.primary_task = primary_task
        self.task_names = list(task_models.keys())
        self.num_classes = {
            name: m.num_classes for name, m in task_models.items()
        }
        # Strides are identical across tasks (shared neck → same FPN levels).
        primary = task_models[primary_task]
        self._strides = list(primary.strides)

    @property
    def output_format(self) -> str:
        return "yolox"

    @property
    def strides(self) -> list[int]:
        return list(self._strides)

    def forward(
        self,
        x: torch.Tensor,
        task_name: str | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if task_name is None:
            task_name = self.primary_task
        if task_name not in self.task_models:
            raise ValueError(
                f"Unknown task_name={task_name!r}; expected one of {self.task_names}"
            )
        return self.task_models[task_name](x)

    def attach_task_losses(
        self, task_losses: dict[str, "nn.Module"]
    ) -> None:
        """Attach per-task ``YOLOXLoss`` instances so ``forward_with_loss``
        can route by ``task_name``. Stored as a plain dict — losses are not
        ``nn.Module``-tracked (their parameters, if any, are negligible and
        the optimizer is built from the model wrapper, not the losses).
        """
        if not isinstance(task_losses, dict):
            raise TypeError("task_losses must be a {task_name: nn.Module} dict")
        missing = set(self.task_names) - set(task_losses)
        if missing:
            raise ValueError(f"Missing losses for tasks: {sorted(missing)}")
        self._task_losses = task_losses

    def forward_with_loss(
        self,
        images: torch.Tensor,
        targets,
        task_name: str | None = None,
    ):
        """Multitask forward + loss. Mirrors ``HFDetectionModel.forward_with_loss``
        contract: returns ``(loss, loss_dict, predictions_or_None)``.

        Predictions are returned only at eval time (so the metric path can
        consume them); ``None`` at training time matching the YOLOX adapter.
        ``task_name`` must be provided (no default — multitask is task-aware).
        """
        if not hasattr(self, "_task_losses") or self._task_losses is None:
            raise RuntimeError(
                "attach_task_losses() must be called before forward_with_loss()"
            )
        if task_name is None or task_name not in self._task_losses:
            raise ValueError(
                f"forward_with_loss requires task_name in {self.task_names}; got {task_name!r}"
            )
        task_model = self.task_models[task_name]
        task_loss = self._task_losses[task_name]
        predictions = task_model(images)
        loss, loss_dict = task_loss(predictions, targets)
        if self.training:
            return loss, loss_dict, None
        return loss, loss_dict, predictions

    def get_param_groups(self, lr: float, weight_decay: float) -> list[dict]:
        """Return YOLOX's six param groups for the SHARED trunk + cls_pred
        per task. Because shared modules are referenced (same tensor identity),
        delegating to the primary task's ``get_param_groups`` naturally
        captures all shared params exactly once. Task-specific ``cls_pred``
        params from tasks 1..N are NOT included — we collect them explicitly
        and append a separate group.
        """
        primary = self.task_models[self.primary_task]
        groups = primary.get_param_groups(lr, weight_decay)

        # Collect per-task cls_pred params that aren't the primary task's.
        seen = set()
        for g in groups:
            for p in g["params"]:
                seen.add(id(p))

        per_task_cls = []
        for name, m in self.task_models.items():
            for head in m.heads:
                for p in head.cls_pred.parameters():
                    if id(p) not in seen:
                        per_task_cls.append(p)
                        seen.add(id(p))

        if per_task_cls:
            groups.append({
                "params": per_task_cls,
                "lr": lr,
                "weight_decay": weight_decay,
                "name": "per_task_cls_pred",
                "group_name": "per_task_cls_pred",
            })
        return groups


def _resolve_size(arch: str) -> tuple[float, float]:
    """Extract size suffix from arch like 'yolox-m-multitask' → ('m')."""
    # arch format: yolox-<size>-multitask
    parts = arch.split("-")
    if len(parts) < 3 or parts[0] != "yolox" or parts[-1] != "multitask":
        raise ValueError(
            f"Expected arch like 'yolox-<size>-multitask', got {arch!r}"
        )
    size = parts[1]
    if size not in _YOLOX_PRESETS:
        raise ValueError(
            f"Unknown YOLOX size {size!r}; expected one of {list(_YOLOX_PRESETS)}"
        )
    return _YOLOX_PRESETS[size]


def _load_pretrained_yolox(model: YOLOXModel, pretrained: str | None) -> None:
    """Load Megvii pretrained weights into a single YOLOXModel (per-task).

    ``YOLOXModel.load_state_dict(strict=False)`` auto-remaps official keys
    and filters shape-mismatched entries (e.g. 80-class COCO ``cls_preds``
    loading into a 2-class head — the cls_pred stays at its prior_prob init).
    Skips load if ``pretrained`` is None or empty.
    """
    if not pretrained:
        return
    import os
    if not os.path.exists(pretrained):
        logger.warning(
            f"YOLOX multitask: pretrained path {pretrained!r} not found, "
            f"skipping weight load (initialising fresh)."
        )
        return
    state = torch.load(pretrained, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)


@register_model("yolox-nano-multitask")
@register_model("yolox-tiny-multitask")
@register_model("yolox-s-multitask")
@register_model("yolox-m-multitask")
@register_model("yolox-l-multitask")
def build_yolox_multitask(config: dict) -> YOLOXMultitaskModel:
    """Build a YOLOX multi-task wrapper from the unified config.

    The config provides:
      model.arch, model.pretrained (optional, Megvii .pth path)
      tasks: [{name, num_classes, names}, ...]   (loaded from 05_data.yaml)
    """
    model_cfg = config.get("model", {})
    arch = model_cfg["arch"]
    depth, width = _resolve_size(arch)
    pretrained = model_cfg.get("pretrained")
    act_type = model_cfg.get("act_type", "silu")
    depthwise = bool(model_cfg.get("depthwise", False))

    tasks = config.get("_tasks")
    if not tasks:
        raise ValueError(
            "yolox-*-multitask requires resolved tasks under config['_tasks']; "
            "the multitask trainer entry point populates this from 05_data.yaml."
        )

    task_models: dict[str, YOLOXModel] = {}
    primary = tasks[0]["name"]

    for t in tasks:
        name = t["name"]
        nc = int(t["num_classes"])
        logger.info(
            f"Multitask: building YOLOX-{arch.split('-')[1]} for task={name} "
            f"(num_classes={nc})"
        )
        m = YOLOXModel(
            num_classes=nc,
            depth=depth,
            width=width,
            act_type=act_type,
            depthwise=depthwise,
        )
        # Only load pretrained into the primary (shared trunk receives weights);
        # other tasks would re-init shared modules over the loaded ones if loaded
        # individually. cls_pred for each task stays at prior_prob init.
        if name == primary:
            _load_pretrained_yolox(m, pretrained)
        task_models[name] = m

    # Share trunk + per-head submodules across tasks (everything except cls_pred).
    primary_model = task_models[primary]
    for name, m in task_models.items():
        if name == primary:
            continue
        _share_heads_except_cls_pred(m, primary_model)

    wrapper = YOLOXMultitaskModel(task_models, primary_task=primary)
    n_total = sum(p.numel() for _, p in wrapper.named_parameters())
    logger.info(
        f"Multitask YOLOX-{arch.split('-')[1]} built: ~{n_total / 1e6:.2f} M "
        f"parameters across {len(task_models)} tasks (shared trunk + per-task cls_pred)"
    )
    return wrapper

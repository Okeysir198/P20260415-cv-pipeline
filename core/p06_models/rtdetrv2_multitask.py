"""RT-DETRv2 multi-task detector.

Mirror of ``dfine_multitask.py`` — shared backbone + encoder + decoder +
box head, N per-task classification heads. RT-DETRv2's submodule layout is
near-identical to D-FINE (no DFL/LQE), so only the model class and the
shared-attr lists differ.

Task-specific (per-model):
    - ``model.class_embed``                  (top-level cls heads list)
    - ``model.model.decoder.class_embed``    (per-decoder-layer cls)
    - ``model.model.denoising_class_embed``  (CDN cls embedding)
    - ``model.model.enc_score_head``         (encoder-side cls)
    - ``model.config``                       (holds num_labels, id2label, ...)

Shared (referenced from task[0]):
    - ``model.bbox_embed``                   (top-level box heads — geometry only)
    - ``model.model.backbone``
    - ``model.model.encoder_input_proj``
    - ``model.model.encoder``
    - ``model.model.enc_output``
    - ``model.model.enc_bbox_head``          (geometry only)
    - ``model.model.decoder_input_proj``
    - ``model.model.decoder.layers``
    - ``model.model.decoder.query_pos_head``
    - ``model.model.decoder.bbox_embed``     (per-layer box heads — geometry only)

bf16 is safe for RT-DETRv2 (unlike D-FINE which diverges in bf16).
"""

from __future__ import annotations

from typing import Any

import torch
from loguru import logger
from torch import nn

from core.p06_models.base import DetectionModel
from core.p06_models.hf_model import _resolve_pretrained_path
from core.p06_models.registry import register_model


_SHARED_TOP_ATTRS = ("bbox_embed",)
_SHARED_INNER_MODEL_ATTRS = (
    "backbone",
    "encoder_input_proj",
    "encoder",
    "enc_output",
    "enc_bbox_head",
    "decoder_input_proj",
)
_SHARED_DECODER_ATTRS = (
    "layers",
    "query_pos_head",
    "bbox_embed",
)


def _build_one_rtdetrv2(
    pretrained: str, num_classes: int, hf_kwargs: dict[str, Any],
) -> nn.Module:
    from transformers import RTDetrV2ForObjectDetection
    kwargs = dict(hf_kwargs)
    kwargs["num_labels"] = num_classes
    if "id2label" not in kwargs:
        kwargs["id2label"] = {i: f"class_{i}" for i in range(num_classes)}
    if "label2id" not in kwargs:
        kwargs["label2id"] = {v: k for k, v in kwargs["id2label"].items()}
    return RTDetrV2ForObjectDetection.from_pretrained(
        pretrained, **kwargs, ignore_mismatched_sizes=True,
    )


def _share_modules(target: nn.Module, source: nn.Module) -> None:
    for attr in _SHARED_TOP_ATTRS:
        setattr(target, attr, getattr(source, attr))
    for attr in _SHARED_INNER_MODEL_ATTRS:
        setattr(target.model, attr, getattr(source.model, attr))
    for attr in _SHARED_DECODER_ATTRS:
        setattr(target.model.decoder, attr, getattr(source.model.decoder, attr))


class RTDetrV2MultitaskModel(DetectionModel):
    """Multi-task RT-DETRv2 with shared trunk + per-task cls heads."""

    _keys_to_ignore_on_save = None

    def __init__(
        self,
        task_models: dict[str, nn.Module],
        task_processors: dict[str, Any],
        primary_task: str,
    ) -> None:
        super().__init__()
        self.task_models = nn.ModuleDict(task_models)
        self.task_processors = task_processors
        self.primary_task = primary_task
        self.task_names = list(task_models.keys())
        self.num_classes = {
            name: m.config.num_labels for name, m in task_models.items()
        }

    @property
    def output_format(self) -> str:
        return "detr"

    @property
    def processor(self):
        return self.task_processors[self.primary_task]

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: list[dict[str, torch.Tensor]] | None = None,
        task_name: str | None = None,
        **kwargs,
    ):
        if task_name is None:
            task_name = self.primary_task
        if task_name not in self.task_models:
            raise ValueError(
                f"Unknown task_name={task_name!r}; expected one of {self.task_names}"
            )
        model = self.task_models[task_name]
        outputs = model(pixel_values=pixel_values, labels=labels, **kwargs)

        # Eval-mode ModelOutput strip (parity with HFDetectionModel +
        # DFineMultitaskModel) — prevents per-eval CPU RAM growth.
        if labels is not None and not self.training:
            placeholder = torch.empty(0, device=pixel_values.device)
            for fld in (
                "encoder_last_hidden_state", "encoder_hidden_states",
                "encoder_attentions", "decoder_hidden_states",
                "decoder_attentions", "cross_attentions",
                "intermediate_hidden_states", "intermediate_reference_points",
                "intermediate_logits", "intermediate_predicted_corners",
                "initial_reference_points", "init_reference_points",
                "auxiliary_outputs", "enc_topk_logits", "enc_topk_bboxes",
                "enc_outputs_class", "enc_outputs_coord_logits",
                "denoising_meta_values",
            ):
                if hasattr(outputs, fld) and outputs.get(fld, None) is not None:
                    outputs[fld] = placeholder
        return outputs


@register_model("rtdetr-r18-multitask")
@register_model("rtdetr-r50-multitask")
def build_rtdetrv2_multitask(config: dict) -> RTDetrV2MultitaskModel:
    """Build the multi-task RT-DETRv2 wrapper from the unified config."""
    from transformers import AutoImageProcessor

    model_cfg = config.get("model", {})
    pretrained = model_cfg.get("pretrained")
    if not pretrained:
        raise ValueError("model.pretrained required for rtdetr-*-multitask")
    pretrained = _resolve_pretrained_path(pretrained, config)
    input_size = model_cfg.get("input_size", [640, 640])

    tasks = config.get("_tasks")
    if not tasks:
        raise ValueError(
            "rtdetr-*-multitask requires resolved tasks under config['_tasks']; "
            "the multitask trainer entry point populates this from 05_data.yaml."
        )

    _NON_HF = {
        "arch", "pretrained", "input_size", "num_classes", "depth", "width",
        "ignore_mismatched_sizes", "hf_model_id", "share_box_head",
        "share_decoder", "task_loss_weights", "resume_weights",
    }
    hf_kwargs = {k: v for k, v in model_cfg.items() if k not in _NON_HF}

    task_models: dict[str, nn.Module] = {}
    task_processors: dict[str, Any] = {}
    primary = tasks[0]["name"]

    for t in tasks:
        name = t["name"]
        nc = int(t["num_classes"])
        names = t.get("names")
        per_task_kwargs = dict(hf_kwargs)
        if isinstance(names, dict) and len(names) == nc:
            per_task_kwargs["id2label"] = {int(k): str(v) for k, v in names.items()}
            per_task_kwargs["label2id"] = {
                v: k for k, v in per_task_kwargs["id2label"].items()
            }
        logger.info(
            f"Multitask: building RT-DETRv2 for task={name} (num_classes={nc}) "
            f"from {pretrained}"
        )
        m = _build_one_rtdetrv2(pretrained, nc, per_task_kwargs)
        task_models[name] = m

        processor = AutoImageProcessor.from_pretrained(
            pretrained,
            do_resize=True,
            size={"height": int(input_size[0]), "width": int(input_size[1])},
        )
        task_processors[name] = processor

    primary_model = task_models[primary]
    for name, m in task_models.items():
        if name == primary:
            continue
        _share_modules(m, primary_model)

    from utils.config import resolve_tensor_prep
    tp = resolve_tensor_prep(config, backend="hf")
    for processor in task_processors.values():
        if tp.get("applied_by", "hf_processor") == "hf_processor":
            processor.do_rescale = bool(tp.get("rescale", True))
            processor.do_normalize = bool(tp.get("normalize", True))
            if processor.do_normalize:
                processor.image_mean = list(tp["mean"])
                processor.image_std = list(tp["std"])
            processor.do_resize = True
            processor.size = {
                "height": int(tp["input_size"][0]),
                "width": int(tp["input_size"][1]),
            }
        else:
            processor.do_rescale = False
            processor.do_normalize = False
            processor.do_resize = False

    wrapper = RTDetrV2MultitaskModel(task_models, task_processors, primary)
    n_total = sum(p.numel() for _, p in wrapper.named_parameters())
    logger.info(
        f"Multitask RT-DETRv2 built: ~{n_total / 1e6:.2f} M parameters across "
        f"{len(task_models)} tasks (shared trunk)"
    )
    return wrapper

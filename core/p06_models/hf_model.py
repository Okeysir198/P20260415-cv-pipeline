"""Generic adapters for HuggingFace models: detection, classification, segmentation.

All architecture config passes directly to HF's config system.
These modules only adapt I/O format between HF and our trainer.
"""

from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    DFineConfig,
    DFineForObjectDetection,
    RTDetrV2Config,
    RTDetrV2ForObjectDetection,
)

from core.p06_models.base import DetectionModel
from loguru import logger
from core.p06_models.registry import register_model

# arch name → (ModelClass, ConfigClass, default_pretrained)
HF_MODEL_REGISTRY: dict[str, tuple[Any, Any, str]] = {
    "dfine": (DFineForObjectDetection, DFineConfig, "ustc-community/dfine_s_coco"),
    "dfine-s": (DFineForObjectDetection, DFineConfig, "ustc-community/dfine_s_coco"),
    "dfine-n": (DFineForObjectDetection, DFineConfig, "ustc-community/dfine_n_coco"),
    "dfine-m": (DFineForObjectDetection, DFineConfig, "ustc-community/dfine_m_coco"),
    "dfine-l": (DFineForObjectDetection, DFineConfig, "ustc-community/dfine-large-coco"),
    "dfine-large": (DFineForObjectDetection, DFineConfig, "ustc-community/dfine-large-coco"),
    "rtdetr": (RTDetrV2ForObjectDetection, RTDetrV2Config, "PekingU/rtdetr_v2_r18vd"),
    "rtdetr-r18": (RTDetrV2ForObjectDetection, RTDetrV2Config, "PekingU/rtdetr_v2_r18vd"),
    "rtdetr-r50": (RTDetrV2ForObjectDetection, RTDetrV2Config, "PekingU/rtdetr_v2_r50vd"),
}

# Keys that belong to our pipeline, NOT to HF config
_NON_HF_KEYS = {
    "arch", "pretrained", "input_size", "num_classes",
    "depth", "width", "ignore_mismatched_sizes", "hf_model_id",
}


class HFDetectionModel(DetectionModel):
    """Thin adapter around any HuggingFace ForObjectDetection model.

    Handles I/O format conversion between HF and our trainer. All
    architecture config passes through to HF unchanged.
    """

    # `PreTrainedModel` marker attributes that HF Trainer's post-training
    # best-checkpoint reload code assumes exist on `self.model`. Our wrapper
    # is plain nn.Module so we supply them explicitly; otherwise
    # `_issue_warnings_after_load` raises `AttributeError` on checkpoint reload.
    _keys_to_ignore_on_save = None

    def __init__(self, hf_model: torch.nn.Module, processor: Any) -> None:
        super().__init__()
        self.hf_model = hf_model
        self.processor = processor
        self.num_classes = hf_model.config.num_labels

    @property
    def output_format(self) -> str:
        return "detr"

    @property
    def strides(self) -> list[int]:
        return [8, 16, 32]

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: list[dict[str, torch.Tensor]] | None = None,
        **kwargs,
    ):
        """Two callers:

        - **Our custom PyTorch trainer** (`labels is None`): returns the legacy
          concatenated tensor `(B, Q, 4+C)` with pixel cxcywh + logits — same as
          before; the trainer's postprocess chain depends on this shape.
        - **HuggingFace `Trainer`** (`labels` supplied as a list of HF-format
          dicts): delegates straight to `self.hf_model(pixel_values=..., labels=...)`
          and returns the raw HF `ModelOutput` so HF Trainer can read `.loss`
          and backprop.

        The collator in `core/p06_training/hf_trainer.py` is responsible for
        emitting `labels` already in HF format (list of
        `{"class_labels": LongTensor, "boxes": FloatTensor cxcywh-normalized}`).
        """
        if labels is not None:
            # HF Trainer path — pass through so Trainer.compute_loss finds .loss
            return self.hf_model(pixel_values=pixel_values, labels=labels, **kwargs)

        # Legacy inference path for the custom trainer.
        outputs = self.hf_model(pixel_values=pixel_values)
        logits = outputs.logits  # (B, Q, C)
        boxes_norm = outputs.pred_boxes  # (B, Q, 4) normalized cxcywh

        _, _, H, W = pixel_values.shape
        scale = boxes_norm.new_tensor([W, H, W, H]).reshape(1, 1, 4)
        boxes_px = boxes_norm * scale

        return torch.cat([boxes_px, logits], dim=-1)

    def forward_with_loss(
        self, images: torch.Tensor, targets: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """Forward pass with HF built-in loss computation."""
        _, _, H, W = images.shape
        hf_labels = self._format_targets(targets, H, W)

        outputs = self.hf_model(pixel_values=images, labels=hf_labels)

        loss = outputs.loss
        raw_loss_dict = getattr(outputs, "loss_dict", {})

        cls_loss = loss * 0
        reg_loss = loss * 0
        for key, val in raw_loss_dict.items():
            key_lower = key.lower()
            # VFL = Varifocal Loss (D-FINE, RT-DETRv2), also covers CE/focal.
            # DFL = Distribution Focal Loss (D-FINE reg component) goes under reg_loss.
            if "vfl" in key_lower or "ce" in key_lower or "class" in key_lower or "focal" in key_lower:
                cls_loss = cls_loss + val
            if "bbox" in key_lower or "giou" in key_lower or "dfl" in key_lower or "reg" in key_lower:
                reg_loss = reg_loss + val

        loss_dict = {
            "cls_loss": cls_loss,
            "obj_loss": loss * 0,
            "reg_loss": reg_loss,
        }

        # Build predictions tensor
        logits = outputs.logits
        boxes_norm = outputs.pred_boxes
        scale = boxes_norm.new_tensor([W, H, W, H]).reshape(1, 1, 4)
        predictions = torch.cat([boxes_norm * scale, logits], dim=-1)

        return loss, loss_dict, predictions

    def postprocess(
        self,
        predictions: torch.Tensor,
        conf_threshold: float,
        target_sizes: torch.Tensor,
    ) -> list[dict[str, np.ndarray]]:
        """Decode predictions using HF's built-in post_process_object_detection.

        Args:
            predictions: (B, Q, 4+C) tensor with pixel cxcywh + logits.
            conf_threshold: Minimum score to keep.
            target_sizes: (B, 2) tensor of [H, W] per image.

        Returns:
            List of dicts with "boxes" (xyxy numpy), "scores", "labels".
        """
        boxes_px = predictions[:, :, :4]
        logits = predictions[:, :, 4:]

        # Convert pixel cxcywh back to normalized for HF processor
        h = target_sizes[:, 0:1].unsqueeze(-1).float()  # (B, 1, 1)
        w = target_sizes[:, 1:2].unsqueeze(-1).float()  # (B, 1, 1)
        scale = torch.cat([w, h, w, h], dim=-1)  # (B, 1, 4)
        boxes_norm = boxes_px / scale

        hf_out = SimpleNamespace(logits=logits, pred_boxes=boxes_norm)
        hf_results = self.processor.post_process_object_detection(
            hf_out, threshold=conf_threshold, target_sizes=target_sizes,
        )

        results = []
        for r in hf_results:
            scores = r["scores"]
            if len(scores) == 0:
                results.append({
                    "boxes": np.zeros((0, 4), dtype=np.float32),
                    "scores": np.zeros(0, dtype=np.float32),
                    "labels": np.zeros(0, dtype=np.int64),
                })
            else:
                results.append({
                    "boxes": r["boxes"].detach().cpu().numpy().astype(np.float32),
                    "scores": scores.detach().cpu().numpy().astype(np.float32),
                    "labels": r["labels"].detach().cpu().numpy().astype(np.int64),
                })
        return results

    def _format_targets(
        self, targets: list[torch.Tensor], H: int, W: int,
    ) -> list[dict[str, torch.Tensor]]:
        """Convert YOLO pixel cxcywh targets to HF normalized cxcywh labels."""
        hf_labels = []
        for tgt in targets:
            if tgt.numel() == 0:
                hf_labels.append({
                    "class_labels": torch.zeros(0, dtype=torch.long, device=tgt.device),
                    "boxes": torch.zeros(0, 4, dtype=torch.float32, device=tgt.device),
                })
                continue

            class_ids = tgt[:, 0].long()
            boxes_px = tgt[:, 1:5].float()
            scale = boxes_px.new_tensor([W, H, W, H]).unsqueeze(0)
            hf_labels.append({
                "class_labels": class_ids,
                "boxes": boxes_px / scale,
            })
        return hf_labels


@register_model("hf_detection")
def build_hf_model(config: dict) -> HFDetectionModel:
    """Build any HF ForObjectDetection model from config.

    All config["model"] keys except pipeline-specific ones pass
    directly to HF's from_pretrained as kwargs.
    """
    model_cfg = config.get("model", {})
    arch = model_cfg.get("arch", "dfine-s").lower()
    num_classes = model_cfg["num_classes"]

    # Collect all non-pipeline keys as HF kwargs
    hf_kwargs = {k: v for k, v in model_cfg.items() if k not in _NON_HF_KEYS}
    hf_kwargs["num_labels"] = num_classes

    # HF cookbook passes id2label/label2id so reinitialised cls-head + matcher
    # have a coherent label space. Prefer names from the resolved data config
    # when available; fall back to generic "class_N" so the dicts are always
    # present with the correct num_labels shape.
    data_names = config.get("data", {}).get("names") or model_cfg.get("names")
    if isinstance(data_names, dict) and len(data_names) == num_classes:
        id2label = {int(k): str(v) for k, v in data_names.items()}
    else:
        id2label = {i: f"class_{i}" for i in range(num_classes)}
    hf_kwargs.setdefault("id2label", id2label)
    hf_kwargs.setdefault("label2id", {v: k for k, v in id2label.items()})

    if arch in HF_MODEL_REGISTRY:
        # Fast path: explicit registry lookup
        ModelClass, _ConfigClass, default_pretrained = HF_MODEL_REGISTRY[arch]
        pretrained = model_cfg.get("pretrained", default_pretrained)

        logger.info(
            "Building HF model: arch=%s, pretrained=%s, hf_kwargs=%s",
            arch, pretrained, hf_kwargs,
        )

        hf_model = ModelClass.from_pretrained(pretrained, **hf_kwargs, ignore_mismatched_sizes=True)
        input_size = model_cfg.get("input_size", [640, 640])
        processor = AutoImageProcessor.from_pretrained(
            pretrained,
            do_resize=True,
            size={"height": int(input_size[0]), "width": int(input_size[1])},
        )
    elif "hf_model_id" in model_cfg:
        # Dynamic fallback: load any HF ForObjectDetection model via Auto class
        hf_model_id = model_cfg["hf_model_id"]
        logger.info(
            "Dynamic HF model loading: hf_model_id=%s, hf_kwargs=%s",
            hf_model_id,
            hf_kwargs,
        )

        hf_model = AutoModelForObjectDetection.from_pretrained(
            hf_model_id, **hf_kwargs, ignore_mismatched_sizes=True,
        )
        input_size = model_cfg.get("input_size", [640, 640])
        processor = AutoImageProcessor.from_pretrained(
            hf_model_id,
            do_resize=True,
            size={"height": int(input_size[0]), "width": int(input_size[1])},
        )
    else:
        raise ValueError(
            f"Unknown HF arch '{arch}'. Either use a registered arch "
            f"({sorted(HF_MODEL_REGISTRY)}) or provide 'hf_model_id' in config "
            f"for dynamic AutoModelForObjectDetection loading."
        )

    # Authoritative tensor_prep contract: force processor attributes so there
    # is exactly one normalize site (the processor when applied_by=hf_processor,
    # the v2 pipeline otherwise). Removes the old "processor-default leakage"
    # footgun where a checkpoint with do_normalize=False would silently disable
    # normalize on both sites.
    from utils.config import resolve_tensor_prep
    tp = resolve_tensor_prep(config, backend="hf")
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
    else:  # v2_pipeline — our pipeline does rescale+normalize+resize
        processor.do_rescale = False
        processor.do_normalize = False
        processor.do_resize = False

    logger.info(
        "tensor_prep: applied_by=%s, processor.do_rescale=%s, do_normalize=%s, "
        "mean=%s, std=%s",
        tp.get("applied_by"),
        processor.do_rescale, processor.do_normalize,
        getattr(processor, "image_mean", None),
        getattr(processor, "image_std", None),
    )

    return HFDetectionModel(hf_model, processor)


class HFClassificationModel(DetectionModel):
    """Thin adapter around any HuggingFace ForImageClassification model.

    Uses HF's built-in cross-entropy loss computation.
    """

    def __init__(self, hf_model: torch.nn.Module) -> None:
        super().__init__()
        self.hf_model = hf_model
        self.num_classes = hf_model.config.num_labels

    @property
    def output_format(self) -> str:
        return "classification"

    def forward(self, x=None, pixel_values=None, labels=None, **kwargs):
        """HF-Trainer-compatible: accepts pixel_values+labels kwargs and
        returns the underlying HF ModelOutput (with .loss if labels given).
        Also supports positional inference call `model(tensor)` → logits.
        """
        px = pixel_values if pixel_values is not None else x
        if labels is not None:
            return self.hf_model(pixel_values=px, labels=labels)
        outputs = self.hf_model(pixel_values=px)
        return outputs.logits

    def forward_with_loss(
        self, images: torch.Tensor, targets: list,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """Forward pass with HF built-in loss."""
        labels = torch.stack(targets).to(images.device)

        outputs = self.hf_model(pixel_values=images, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss_dict = {"cls_loss": loss.detach()}
        return loss, loss_dict, logits


class HFSegmentationModel(DetectionModel):
    """Thin adapter around any HuggingFace ForSemanticSegmentation model.

    Uses HF's built-in loss computation (cross-entropy on pixel masks).
    """

    def __init__(self, hf_model: torch.nn.Module) -> None:
        super().__init__()
        self.hf_model = hf_model
        self.num_classes = hf_model.config.num_labels

    @property
    def output_format(self) -> str:
        return "segmentation"

    def forward(self, x=None, pixel_values=None, labels=None, **kwargs):
        """HF-Trainer-compatible: accepts pixel_values+labels kwargs and
        returns the underlying HF ModelOutput (with .loss if labels given).
        Also supports positional inference `model(tensor)` → logits.
        """
        px = pixel_values if pixel_values is not None else x
        if labels is not None:
            return self.hf_model(pixel_values=px, labels=labels)
        outputs = self.hf_model(pixel_values=px)
        return outputs.logits

    def forward_with_loss(
        self, images: torch.Tensor, targets: list,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """Forward pass with HF built-in segmentation loss."""
        labels = torch.stack(targets).to(images.device)

        outputs = self.hf_model(pixel_values=images, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss_dict = {"seg_loss": loss.detach()}
        return loss, loss_dict, logits


def _build_hf_generic_model(config: dict, auto_cls, wrapper_cls, task_name: str):
    """Build an HF model from config using the given AutoModel and wrapper classes.

    Args:
        config: Full training config with a ``"model"`` section.
        auto_cls: HF AutoModel class (e.g. AutoModelForImageClassification).
        wrapper_cls: Our wrapper class (e.g. HFClassificationModel).
        task_name: Human-readable task name for error/log messages.

    Returns:
        Wrapped model instance.
    """
    model_cfg = config.get("model", {})
    num_classes = model_cfg["num_classes"]
    pretrained = model_cfg.get("pretrained")

    if not pretrained:
        raise ValueError(
            f"config['model']['pretrained'] is required for HF {task_name} models"
        )

    hf_kwargs = {k: v for k, v in model_cfg.items() if k not in _NON_HF_KEYS}
    hf_kwargs["num_labels"] = num_classes

    logger.info("Building HF %s model: pretrained=%s, kwargs=%s", task_name, pretrained, hf_kwargs)
    hf_model = auto_cls.from_pretrained(
        pretrained, **hf_kwargs, ignore_mismatched_sizes=True
    )

    return wrapper_cls(hf_model)


@register_model("hf_classification")
def build_hf_classification_model(config: dict) -> HFClassificationModel:
    """Build any HF ForImageClassification model from config.

    Config example::

        model:
          arch: hf-vit-cls
          pretrained: google/vit-base-patch16-224
          num_classes: 2
          input_size: [224, 224]
    """
    from transformers import AutoModelForImageClassification

    return _build_hf_generic_model(
        config, AutoModelForImageClassification, HFClassificationModel, "classification",
    )


@register_model("hf_segmentation")
def build_hf_segmentation_model(config: dict) -> HFSegmentationModel:
    """Build any HF ForSemanticSegmentation model from config.

    Config example::

        model:
          arch: hf-segformer
          pretrained: nvidia/segformer-b0-finetuned-ade-512-512
          num_classes: 19
          input_size: [512, 512]
    """
    from transformers import AutoModelForSemanticSegmentation

    return _build_hf_generic_model(
        config, AutoModelForSemanticSegmentation, HFSegmentationModel, "segmentation",
    )


class HFKeypointModel(DetectionModel):
    """Thin adapter around HF top-down keypoint estimators (ViTPose family).

    HF's ``VitPoseForPoseEstimation.forward`` returns heatmaps with no
    built-in loss, so this wrapper computes weighted heatmap MSE when
    ``target_heatmap`` + ``target_weight`` are passed alongside
    ``pixel_values`` (HF Trainer convention for custom losses).
    """

    _keys_to_ignore_on_save = None

    def __init__(self, hf_model: torch.nn.Module, processor: Any) -> None:
        super().__init__()
        self.hf_model = hf_model
        self.processor = processor
        cfg = hf_model.config
        # ViTPose's `num_labels` == number of keypoints.
        self.num_keypoints = getattr(cfg, "num_labels", None) or getattr(
            cfg, "num_keypoints", 17
        )

    @property
    def output_format(self) -> str:
        return "keypoint"

    def forward(
        self,
        pixel_values: torch.Tensor,
        target_heatmap: torch.Tensor | None = None,
        target_weight: torch.Tensor | None = None,
        labels: Any = None,  # noqa: ARG002 — HF Trainer may inject this; ignored
        **kwargs,
    ):
        """Two callers:

        - **HF Trainer** with targets supplied (training/eval): compute
          weighted heatmap MSE and return ``SimpleNamespace(loss, heatmaps)``
          so ``Trainer.compute_loss`` finds ``.loss``.
        - **Inference** (no targets): return raw HF ``ModelOutput`` so
          callers can use ``processor.post_process_pose_estimation``.
        """
        outputs = self.hf_model(pixel_values=pixel_values, **kwargs)
        heatmaps = outputs.heatmaps if hasattr(outputs, "heatmaps") else outputs.logits

        if target_heatmap is None:
            return outputs

        weight = target_weight[:, :, None, None] if target_weight is not None else 1.0
        loss = ((heatmaps - target_heatmap) ** 2 * weight).mean()
        # HF Trainer expects either ModelOutput or a dict supporting
        # `outputs["loss"]`. SimpleNamespace fails that subscript.
        return {"loss": loss, "heatmaps": heatmaps}

    def forward_with_loss(
        self, images: torch.Tensor, targets: list,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """Pytorch-backend hook (kept for parity with HFDetectionModel).

        ``targets`` is a list of ``(target_heatmap, target_weight)`` tuples;
        the keypoint collator (HF path) stacks these instead.
        """
        target_hm = torch.stack([t[0] for t in targets]).to(images.device)
        target_wt = torch.stack([t[1] for t in targets]).to(images.device)
        out = self.forward(
            pixel_values=images, target_heatmap=target_hm, target_weight=target_wt,
        )
        loss = out["loss"]
        heatmaps = out["heatmaps"]
        return loss, {"kpt_loss": loss.detach()}, heatmaps


@register_model("hf_keypoint")
def build_hf_keypoint_model(config: dict) -> HFKeypointModel:
    """Build any HF top-down keypoint estimator (ViTPose family) from config.

    Config example::

        model:
          arch: hf_keypoint
          pretrained: usyd-community/vitpose-base-simple
          num_keypoints: 17
          input_size: [256, 192]   # (H, W)
          ignore_mismatched_sizes: true
    """
    from transformers import AutoImageProcessor, VitPoseForPoseEstimation

    model_cfg = config.get("model", {})
    pretrained = model_cfg.get("pretrained")
    if not pretrained:
        raise ValueError(
            "config['model']['pretrained'] is required for HF keypoint models"
        )

    num_keypoints = int(model_cfg.get("num_keypoints", 17))
    ignore_mm = bool(model_cfg.get("ignore_mismatched_sizes", True))

    logger.info(
        "Building HF keypoint model: pretrained=%s, num_keypoints=%d",
        pretrained, num_keypoints,
    )
    hf_model = VitPoseForPoseEstimation.from_pretrained(
        pretrained,
        num_labels=num_keypoints,
        ignore_mismatched_sizes=ignore_mm,
    )
    processor = AutoImageProcessor.from_pretrained(pretrained)
    return HFKeypointModel(hf_model, processor)

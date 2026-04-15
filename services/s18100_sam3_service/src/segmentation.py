"""Stateless image segmentation functions (box, text, auto_mask)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

from src.config import config, AUTO_MASK_PROMPTS
from src.helpers import mask_post_kwargs, mask_to_detection
from src.models import device, dtype, get_text, get_tracker


def segment_box(image: Image.Image, box: list[int]) -> dict:
    """Box-prompted segmentation via Sam3TrackerVideoModel. Returns single result."""
    model, processor = get_tracker()
    with torch.no_grad():
        session = processor.init_video_session(video=[image], inference_device=device(), dtype=dtype())
        processor.add_inputs_to_inference_session(
            inference_session=session, frame_idx=0, obj_ids=1, input_boxes=[[box]],
        )
        outputs = model(inference_session=session, frame_idx=0)
        masks = processor.post_process_masks(
            [outputs.pred_masks], original_sizes=[[session.video_height, session.video_width]],
            **mask_post_kwargs(),
        )[0]
    mask = masks[0].cpu().numpy().squeeze().astype(bool)
    iou_score = float(outputs.iou_scores.cpu().max()) if hasattr(outputs, "iou_scores") else 1.0
    det = mask_to_detection(mask, iou_score)
    det["iou_score"] = iou_score
    return det


def segment_text(
    image: Image.Image, text: str,
    detection_threshold: Optional[float] = None,
    mask_threshold: Optional[float] = None,
) -> list[dict]:
    """Text-prompted segmentation via Sam3Model."""
    model, processor = get_text()
    cfg = config["segmentation"]
    dt = detection_threshold if detection_threshold is not None else cfg["detection_threshold"]
    mt = mask_threshold if mask_threshold is not None else cfg["mask_threshold"]

    inputs = processor(images=image, text=text, return_tensors="pt").to(device())
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs, threshold=dt, mask_threshold=mt,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]
    return [
        mask_to_detection(np.asarray(m.cpu()).astype(bool), float(s))
        for m, s in zip(results.get("masks", []), results.get("scores", []))
    ]


def segment_auto(
    image: Image.Image,
    threshold: Optional[float] = None,
    prompts: Optional[list[str]] = None,
) -> list[dict]:
    """Auto-mask via Sam3Model with multi-prompt open-vocabulary detection."""
    model, processor = get_text()
    thr = threshold if threshold is not None else config["segmentation"].get("auto_mask_threshold", 0.2)
    prompt_list = prompts if prompts is not None else AUTO_MASK_PROMPTS
    img_w, img_h = image.size

    img_inputs = processor(images=image, return_tensors="pt").to(device())
    with torch.no_grad():
        vision_embeds = model.get_vision_features(pixel_values=img_inputs.pixel_values)

    detections = []
    for prompt in prompt_list:
        text_inputs = processor(text=prompt, return_tensors="pt").to(device())
        with torch.no_grad():
            outputs = model(vision_embeds=vision_embeds, **text_inputs)
        probs = torch.sigmoid(outputs.pred_logits[0]).cpu()
        pred_masks = outputs.pred_masks[0].cpu()
        for idx in (probs > thr).nonzero(as_tuple=True)[0]:
            mask = F.interpolate(
                pred_masks[idx].unsqueeze(0).unsqueeze(0).float(), size=(img_h, img_w),
                mode="bilinear", align_corners=False,
            )[0, 0]
            mask = (mask > 0).numpy()
            area = float(mask.sum()) / mask.size
            max_area = config["segmentation"].get("auto_mask_max_area", 0.95)
            if area > max_area:
                continue
            detections.append(mask_to_detection(mask, float(probs[idx]), area))

    if not detections:
        return detections

    # NMS to remove overlapping detections from different prompts
    nms_thr = config["segmentation"].get("auto_mask_nms_threshold", 0.5)
    boxes = torch.tensor([[d["bbox"]["x1"], d["bbox"]["y1"], d["bbox"]["x2"], d["bbox"]["y2"]] for d in detections], dtype=torch.float32)
    scores = torch.tensor([d["score"] for d in detections], dtype=torch.float32)
    keep = torchvision.ops.nms(boxes, scores, nms_thr)
    detections = [detections[i] for i in keep.tolist()]

    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections


# ---------------------------------------------------------------------------
# Batch processing functions
# ---------------------------------------------------------------------------
# Note: Sam3's DETR cross-attention layer does not support batch>1 forward
# (shape mismatch between batched vision features and single text embedding).
# Batching just the vision encoder was benchmarked but provides no speedup
# due to large FPN feature maps dominating memory bandwidth. These functions
# process images sequentially — the batch endpoints exist for API convenience
# (single request with multiple images) not for GPU-level parallelism.
# ---------------------------------------------------------------------------


def segment_text_batch(
    images: list[Image.Image],
    text: str,
    detection_threshold: float | None = None,
    mask_threshold: float | None = None,
) -> list[list[dict]]:
    """Batch text-prompted segmentation (sequential per image).

    Args:
        images: List of PIL images (can be different sizes)
        text: Text prompt for all images
        detection_threshold: Override config detection_threshold
        mask_threshold: Override config mask_threshold

    Returns:
        List of detection lists (one per image)
    """
    return [segment_text(img, text, detection_threshold, mask_threshold) for img in images]


def segment_auto_batch(
    images: list[Image.Image],
    threshold: float | None = None,
    prompts: list[str] | None = None,
) -> list[list[dict]]:
    """Batch auto-mask segmentation (sequential per image).

    Args:
        images: List of PIL images (can be different sizes)
        threshold: Override config auto_mask_threshold
        prompts: Override config auto_mask_prompts

    Returns:
        List of detection lists (one per image)
    """
    return [segment_auto(img, threshold, prompts) for img in images]

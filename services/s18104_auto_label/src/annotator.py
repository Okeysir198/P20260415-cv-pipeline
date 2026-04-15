"""SAM3 HTTP callers and annotation mode logic."""

from __future__ import annotations

import base64
import io

import httpx
import numpy as np
import requests
from fastapi import HTTPException

from src.config import BATCH_CONCURRENCY, REQUEST_TIMEOUT, SAM3_URL, USE_BATCH_ENDPOINTS, logger
from src.geometry import apply_nms, bbox_from_sam3, compute_bbox_norm, decode_image, mask_to_polygon
from src.schemas import Detection


# ---------------------------------------------------------------------------
# SAM3 HTTP callers
# ---------------------------------------------------------------------------


def call_sam3_text(
    image_b64: str,
    text: str,
    detection_threshold: float | None = None,
    mask_threshold: float | None = None,
) -> list[dict]:
    """Call SAM3 /segment_text endpoint. Returns list of detection dicts."""
    payload: dict = {"image": image_b64, "text": text}
    if detection_threshold is not None:
        payload["detection_threshold"] = detection_threshold
    if mask_threshold is not None:
        payload["mask_threshold"] = mask_threshold
    resp = requests.post(
        f"{SAM3_URL}/segment_text", json=payload, timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json().get("detections", [])


def call_sam3_auto(image_b64: str, threshold: float | None = None) -> list[dict]:
    """Call SAM3 /auto_mask endpoint. Returns list of detection dicts."""
    payload: dict = {"image": image_b64}
    if threshold is not None:
        payload["threshold"] = threshold
    resp = requests.post(
        f"{SAM3_URL}/auto_mask", json=payload, timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json().get("detections", [])


def call_sam3_box(image_b64: str, box: list[int]) -> dict:
    """Call SAM3 /segment_box endpoint. Returns single result dict."""
    resp = requests.post(
        f"{SAM3_URL}/segment_box", json={"image": image_b64, "box": box}, timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["result"]


# ---------------------------------------------------------------------------
# SAM3 Batch HTTP callers (async, true tensor batching)
# ---------------------------------------------------------------------------


async def call_sam3_text_batch(
    images: list[tuple[str, int, int]],  # (b64, w, h)
    text: str,
    detection_threshold: float | None = None,
    mask_threshold: float | None = None,
) -> list[list[dict]]:
    """Call SAM3 /segment_text_batch endpoint (true batch tensor processing).

    Args:
        images: List of (base64_image, width, height) tuples
        text: Text prompt for all images
        detection_threshold: Optional detection threshold override
        mask_threshold: Optional mask threshold override

    Returns:
        List of detection lists (one per image) in original image coordinates
    """
    payload = {
        "items": [{"image": b64} for b64, _, _ in images],
        "text": text,
    }
    if detection_threshold is not None:
        payload["detection_threshold"] = detection_threshold
    if mask_threshold is not None:
        payload["mask_threshold"] = mask_threshold

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.post(f"{SAM3_URL}/segment_text_batch", json=payload)
        resp.raise_for_status()
        return resp.json()


async def call_sam3_auto_batch(
    images: list[tuple[str, int, int]],  # (b64, w, h)
    threshold: float | None = None,
    prompts: list[str] | None = None,
) -> list[list[dict]]:
    """Call SAM3 /auto_mask_batch endpoint (true batch tensor processing).

    Args:
        images: List of (base64_image, width, height) tuples
        threshold: Optional threshold override
        prompts: Optional prompt list override

    Returns:
        List of detection lists (one per image) in original image coordinates
    """
    payload = {"items": [{"image": b64} for b64, _, _ in images]}
    if threshold is not None:
        payload["threshold"] = threshold
    if prompts is not None:
        payload["prompts"] = prompts

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.post(f"{SAM3_URL}/auto_mask_batch", json=payload)
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Detection conversion
# ---------------------------------------------------------------------------


def sam3_det_to_detection(
    sam3_det: dict,
    class_id: int,
    class_name: str,
    img_w: int,
    img_h: int,
    include_masks: bool,
) -> Detection:
    """Convert a single SAM3 detection dict to our Detection model."""
    bbox_dict = sam3_det["bbox"]
    bbox_xyxy = bbox_from_sam3(bbox_dict)
    bbox_norm = compute_bbox_norm(bbox_xyxy, img_w, img_h)
    mask_b64 = sam3_det.get("mask")
    polygon = mask_to_polygon(mask_b64, img_h, img_w) if mask_b64 else []
    score = float(sam3_det.get("score", 0.0))
    area = float(sam3_det.get("area", 0.0))

    return Detection(
        class_id=class_id,
        class_name=class_name,
        score=score,
        bbox_xyxy=bbox_xyxy,
        bbox_norm=bbox_norm,
        polygon=polygon,
        mask=mask_b64 if include_masks else None,
        area=area,
    )


# ---------------------------------------------------------------------------
# Annotation modes
# ---------------------------------------------------------------------------


def annotate_text_mode(
    image_b64: str,
    classes: dict[int, str],
    text_prompts: dict[str, str],
    confidence_threshold: float,
    img_w: int,
    img_h: int,
    include_masks: bool,
) -> list[Detection]:
    """Text mode: for each class, call SAM3 /segment_text with class prompt."""
    detections: list[Detection] = []
    for cls_id, cls_name in classes.items():
        prompt = text_prompts.get(cls_name, cls_name)
        try:
            sam3_dets = call_sam3_text(image_b64, prompt)
        except Exception as exc:
            logger.warning("SAM3 segment_text failed for class '%s': %s", cls_name, exc)
            continue

        for sd in sam3_dets:
            det = sam3_det_to_detection(sd, int(cls_id), cls_name, img_w, img_h, include_masks)
            if det.score >= confidence_threshold:
                detections.append(det)

    return detections


def annotate_auto_mode(
    image_b64: str,
    classes: dict[int, str],
    text_prompts: dict[str, str],
    confidence_threshold: float,
    img_w: int,
    img_h: int,
    include_masks: bool,
) -> list[Detection]:
    """Auto mode: call SAM3 /auto_mask, then classify each mask by running
    /segment_text for each class on the crop and picking the best match."""
    try:
        auto_dets = call_sam3_auto(image_b64)
    except Exception as exc:
        logger.warning("SAM3 auto_mask failed: %s", exc)
        return []

    if not auto_dets:
        return []

    # Decode image once for cropping all masks
    try:
        img = decode_image(image_b64)
    except Exception:
        return []

    # For each auto-detected mask, try to classify it
    detections: list[Detection] = []
    for ad in auto_dets:
        bbox_xyxy = bbox_from_sam3(ad.get("bbox", {}))
        x1, y1, x2, y2 = bbox_xyxy

        # Skip tiny detections
        if (x2 - x1) < 2 or (y2 - y1) < 2:
            continue

        # Crop the region and encode for classification
        try:
            crop = img.crop((x1, y1, x2, y2))
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            crop_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            continue

        # Classify crop by running text segmentation for each class
        best_cls_id = -1
        best_cls_name = ""
        best_score = 0.0
        for cls_id, cls_name in classes.items():
            prompt = text_prompts.get(cls_name, cls_name)
            try:
                cls_dets = call_sam3_text(crop_b64, prompt)
            except Exception:
                continue
            for cd in cls_dets:
                s = float(cd.get("score", 0.0))
                if s > best_score:
                    best_score = s
                    best_cls_id = int(cls_id)
                    best_cls_name = cls_name

        if best_cls_id < 0 or best_score < confidence_threshold:
            continue

        det = sam3_det_to_detection(
            ad, best_cls_id, best_cls_name, img_w, img_h, include_masks,
        )
        # Override score with classification score
        det = det.model_copy(update={"score": best_score})
        detections.append(det)

    return detections


def annotate_hybrid_mode(
    image_b64: str,
    classes: dict[int, str],
    text_prompts: dict[str, str],
    confidence_threshold: float,
    img_w: int,
    img_h: int,
    include_masks: bool,
) -> list[Detection]:
    """Hybrid mode: text first for high-confidence detections, then auto for
    regions not already covered by text detections."""
    # Run text mode first
    text_dets = annotate_text_mode(
        image_b64, classes, text_prompts, confidence_threshold, img_w, img_h, include_masks,
    )

    # Run auto mode
    auto_dets = annotate_auto_mode(
        image_b64, classes, text_prompts, confidence_threshold, img_w, img_h, include_masks,
    )

    if not auto_dets:
        return text_dets

    # Filter auto detections that overlap significantly with text detections
    if not text_dets:
        return auto_dets

    text_boxes = np.array([d.bbox_xyxy for d in text_dets], dtype=np.float64)
    filtered_auto: list[Detection] = []
    for ad in auto_dets:
        auto_box = np.array(ad.bbox_xyxy, dtype=np.float64)
        # Compute IoU with all text boxes
        xx1 = np.maximum(auto_box[0], text_boxes[:, 0])
        yy1 = np.maximum(auto_box[1], text_boxes[:, 1])
        xx2 = np.minimum(auto_box[2], text_boxes[:, 2])
        yy2 = np.minimum(auto_box[3], text_boxes[:, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_auto = (auto_box[2] - auto_box[0]) * (auto_box[3] - auto_box[1])
        area_text = (text_boxes[:, 2] - text_boxes[:, 0]) * (text_boxes[:, 3] - text_boxes[:, 1])
        iou = inter / (area_auto + area_text - inter + 1e-6)
        if iou.max() < 0.5:
            filtered_auto.append(ad)

    return text_dets + filtered_auto


def annotate_single(
    image_b64: str,
    classes: dict[int, str],
    text_prompts: dict[str, str],
    mode: str,
    confidence_threshold: float,
    nms_iou_threshold: float,
    include_masks: bool,
    img_w: int,
    img_h: int,
    detection_classes: dict[str, str] | None = None,
    class_rules: list[dict] | None = None,
    vlm_verify_config: dict | None = None,
) -> list[Detection]:
    """Main annotation entry point. Dispatches to the requested mode."""
    # If rule-based mode: detect intermediate classes, not final classes
    if detection_classes:
        detection_class_map = {name: 100 + i for i, name in enumerate(detection_classes.keys())}
        annotate_classes = {tid: name for name, tid in detection_class_map.items()}
        annotate_prompts = detection_classes  # prompts ARE the detection_classes values
    else:
        annotate_classes = classes
        annotate_prompts = text_prompts

    if mode == "text":
        detections = annotate_text_mode(
            image_b64, annotate_classes, annotate_prompts, confidence_threshold, img_w, img_h, include_masks,
        )
    elif mode == "auto":
        detections = annotate_auto_mode(
            image_b64, annotate_classes, annotate_prompts, confidence_threshold, img_w, img_h, include_masks,
        )
    elif mode == "hybrid":
        detections = annotate_hybrid_mode(
            image_b64, annotate_classes, annotate_prompts, confidence_threshold, img_w, img_h, include_masks,
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown mode '{mode}'. Supported: text, auto, hybrid",
        )

    # Apply NMS
    detections = apply_nms(detections, nms_iou_threshold)

    # Rule-based classification (if configured)
    if detection_classes and class_rules:
        from src.rule_classifier import apply_rule_classification

        detection_class_map = {name: 100 + i for i, name in enumerate(detection_classes.keys())}
        detections = apply_rule_classification(detections, class_rules, detection_class_map, classes)

    # VLM verification (if configured)
    if vlm_verify_config:
        from src.vlm_verifier import verify_detections_vlm

        derived_ids = {
            r["output_class_id"]
            for r in (class_rules or [])
            if r.get("condition") in ("overlap", "no_overlap")
        }
        detections = verify_detections_vlm(
            image_b64, detections, vlm_verify_config, classes, derived_ids, img_w, img_h,
        )

    return detections

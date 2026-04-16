"""SAM3 HTTP callers, mask helpers, and verification logic."""

from __future__ import annotations

import base64
import io
import time

import numpy as np
import requests
from PIL import Image, ImageDraw

from src.config import (
    REQUEST_TIMEOUT,
    SAM3_URL,
    logger,
    DEFAULT_TEXT_VERIFY_THRESHOLD,
    DEFAULT_AUTO_MASK_MIN_AREA,
    DEFAULT_AUTO_MASK_MAX_AREA,
    DEFAULT_MISSING_OVERLAP_THRESH,
)
from src.geometry import (
    compute_iou_matrix,
    compute_single_iou,
    norm_cxcywh_to_xyxy,
    norm_xyxy_to_pixel,
    pixel_xyxy_to_norm_cxcywh,
)
from src.schemas import ParsedAnnotation, SAM3Verification


# ---------------------------------------------------------------------------
# SAM3 HTTP callers
# ---------------------------------------------------------------------------


def call_sam3_box(image_b64: str, box: list[int]) -> dict:
    """Call SAM3 /segment_box endpoint. Returns single result dict."""
    resp = requests.post(
        f"{SAM3_URL}/segment_box",
        json={"image": image_b64, "box": box},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["result"]


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


def bbox_from_sam3(bbox_dict: dict) -> list[int]:
    """Extract [x1, y1, x2, y2] pixel coords from SAM3 bbox dict."""
    return [int(bbox_dict["x1"]), int(bbox_dict["y1"]), int(bbox_dict["x2"]), int(bbox_dict["y2"])]


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------


def rasterize_polygon_mask(
    polygon_norm: list[float],
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """Rasterize a normalized polygon into a boolean mask using PIL.

    Args:
        polygon_norm: Flat list [x1,y1,x2,y2,...] of normalized coordinates.
        img_w: Image width.
        img_h: Image height.

    Returns:
        Boolean mask of shape (img_h, img_w).
    """
    # Convert normalized coords to pixel coords
    pixel_coords: list[tuple[int, int]] = []
    for i in range(0, len(polygon_norm) - 1, 2):
        px = int(round(polygon_norm[i] * img_w))
        py = int(round(polygon_norm[i + 1] * img_h))
        pixel_coords.append((px, py))

    if len(pixel_coords) < 3:
        return np.zeros((img_h, img_w), dtype=bool)

    # Draw polygon on a PIL image
    mask_img = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask_img)
    draw.polygon(pixel_coords, fill=255)
    return np.array(mask_img) > 127


def decode_sam3_mask(mask_b64: str) -> np.ndarray:
    """Decode a base64 PNG mask from SAM3 into a boolean array."""
    raw = base64.b64decode(mask_b64)
    mask_arr = np.array(Image.open(io.BytesIO(raw)).convert("L"))
    return mask_arr > 127


def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute pixel-level IoU between two boolean masks."""
    if mask_a.shape != mask_b.shape:
        # Resize mask_b to match mask_a
        pil_b = Image.fromarray(mask_b.astype(np.uint8) * 255)
        pil_b = pil_b.resize((mask_a.shape[1], mask_a.shape[0]), Image.Resampling.NEAREST)
        mask_b = np.array(pil_b) > 127

    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


# ---------------------------------------------------------------------------
# SAM3 verification
# ---------------------------------------------------------------------------


def verify_with_sam3(
    image_b64: str,
    annotations: list[ParsedAnnotation],
    classes: dict[int, str],
    text_prompts: dict[str, str],
    include_missing: bool,
    cfg: dict,
    img_w: int,
    img_h: int,
) -> SAM3Verification:
    """Verify annotations against SAM3.

    Steps:
        (a) Box IoU: For each annotation, call SAM3 /segment_box, compute IoU.
        (b) Mask IoU: For seg formats with polygon data, rasterize and compute pixel-level IoU.
        (c) Text verification: For low-IoU annotations, call /segment_text to check class.
        (d) Missing detection: Call /auto_mask, filter, check overlap with existing.

    Args:
        image_b64: Base64 image string.
        annotations: Parsed annotations.
        classes: Class mapping.
        text_prompts: Optional refined prompts per class name.
        include_missing: Whether to run missing detection.
        cfg: Config overrides.
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        SAM3Verification with IoUs, misclassified indices, and missing detections.
    """
    text_verify_threshold: float = cfg.get("text_verify_threshold", DEFAULT_TEXT_VERIFY_THRESHOLD)
    auto_mask_min_area: float = cfg.get("auto_mask_min_area", DEFAULT_AUTO_MASK_MIN_AREA)
    auto_mask_max_area: float = cfg.get("auto_mask_max_area", DEFAULT_AUTO_MASK_MAX_AREA)
    missing_overlap_thresh: float = cfg.get("missing_overlap_threshold", DEFAULT_MISSING_OVERLAP_THRESH)
    max_retries: int = cfg.get("max_retries", 1)
    retry_delay_s: float = cfg.get("retry_delay_s", 2.0)

    valid_classes = {int(k): v for k, v in classes.items()}

    box_ious: list[float] = []
    mask_ious: list[float] = []
    misclassified: list[int] = []
    has_polygon_data = any(ann.polygon_norm for ann in annotations)
    sam3_failed = False

    # --- (a) Box IoU and (b) Mask IoU ---
    for ann_idx, ann in enumerate(annotations):
        cx, cy, w, h = ann.bbox_norm
        norm_xyxy = norm_cxcywh_to_xyxy(cx, cy, w, h)
        pixel_box = norm_xyxy_to_pixel(norm_xyxy, img_w, img_h)

        sam3_result = None
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                sam3_result = call_sam3_box(image_b64, pixel_box)
                break
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    logger.debug(
                        "SAM3 /segment_box failed for annotation %d (attempt %d/%d): %s — retrying",
                        ann_idx, attempt + 1, max_retries + 1, exc,
                    )
                    time.sleep(retry_delay_s)

        if sam3_result is None:
            logger.debug(
                "SAM3 /segment_box unavailable for annotation %d after %d attempt(s): %s",
                ann_idx, max_retries + 1, last_exc,
            )
            sam3_failed = True
            break

        # Compute box IoU
        sam3_bbox = sam3_result.get("bbox", {})
        if sam3_bbox:
            sam3_pixel_box = bbox_from_sam3(sam3_bbox)
            iou_val = compute_single_iou(
                [float(x) for x in pixel_box],
                [float(x) for x in sam3_pixel_box],
            )
        else:
            iou_val = 0.0
        box_ious.append(iou_val)

        # Compute mask IoU (only for formats with polygon data)
        if ann.polygon_norm and has_polygon_data:
            sam3_mask_b64 = sam3_result.get("mask")
            if sam3_mask_b64:
                try:
                    sam3_mask = decode_sam3_mask(sam3_mask_b64)
                    ann_mask = rasterize_polygon_mask(ann.polygon_norm, img_w, img_h)
                    m_iou = compute_mask_iou(ann_mask, sam3_mask)
                    mask_ious.append(m_iou)
                except Exception as exc:
                    logger.debug("Mask IoU computation failed for annotation %d: %s", ann_idx, exc)
                    mask_ious.append(0.0)
            else:
                mask_ious.append(0.0)
        elif has_polygon_data:
            mask_ious.append(0.0)

        # --- (c) Text verification for low-IoU annotations ---
        if iou_val < text_verify_threshold:
            class_name = valid_classes.get(ann.class_id, str(ann.class_id))
            prompt = text_prompts.get(class_name, class_name)
            try:
                text_dets = call_sam3_text(image_b64, prompt)
            except Exception:
                logger.debug("SAM3 /segment_text failed for annotation %d", ann_idx)
                continue

            # Check if any text detection overlaps with this annotation
            has_overlap = False
            for td in text_dets:
                td_bbox = td.get("bbox", {})
                if not td_bbox:
                    continue
                td_pixel_box = bbox_from_sam3(td_bbox)
                overlap = compute_single_iou(
                    [float(x) for x in pixel_box],
                    [float(x) for x in td_pixel_box],
                )
                if overlap > 0.1:
                    has_overlap = True
                    break

            if not has_overlap:
                misclassified.append(ann_idx)

    # SAM3 was unreachable — return sentinel so caller can grade as "unverified"
    if sam3_failed:
        return SAM3Verification(
            box_ious=None,
            mask_ious=[],
            mean_box_iou=0.0,
            mean_mask_iou=0.0,
            misclassified=[],
            missing_detections=[],
        )

    # --- (d) Missing detection ---
    missing_detections: list[dict] = []
    if include_missing:
        try:
            auto_dets = call_sam3_auto(image_b64)
        except Exception as exc:
            logger.debug("SAM3 /auto_mask failed: %s", exc)
            auto_dets = []

        if auto_dets:
            # Build existing annotation boxes in pixel coords
            if annotations:
                ann_pixel_boxes = np.array([
                    norm_xyxy_to_pixel(
                        norm_cxcywh_to_xyxy(*a.bbox_norm),
                        img_w, img_h,
                    ) for a in annotations
                ], dtype=np.float64)
            else:
                ann_pixel_boxes = np.zeros((0, 4), dtype=np.float64)

            for ad in auto_dets:
                ad_bbox = ad.get("bbox", {})
                if not ad_bbox:
                    continue
                ad_pixel = bbox_from_sam3(ad_bbox)

                # Filter by relative area
                ad_w = (ad_pixel[2] - ad_pixel[0]) / img_w if img_w > 0 else 0.0
                ad_h = (ad_pixel[3] - ad_pixel[1]) / img_h if img_h > 0 else 0.0
                area = ad_w * ad_h
                if area < auto_mask_min_area or area > auto_mask_max_area:
                    continue

                # Check overlap with existing annotations
                if ann_pixel_boxes.shape[0] > 0:
                    ad_arr = np.array([ad_pixel], dtype=np.float64)
                    ious = compute_iou_matrix(ad_arr, ann_pixel_boxes)
                    if float(ious.max()) > missing_overlap_thresh:
                        continue

                # Convert to normalized cxcywh
                norm_bbox = pixel_xyxy_to_norm_cxcywh(ad_pixel, img_w, img_h)
                missing_detections.append({
                    "bbox_norm": norm_bbox,
                    "area": round(area, 4),
                    "score": round(float(ad.get("score", 0.0)), 4),
                })

    mean_box_iou = float(np.mean(box_ious)) if box_ious else 0.0
    mean_mask_iou = float(np.mean(mask_ious)) if mask_ious else 0.0

    return SAM3Verification(
        box_ious=[round(v, 4) for v in box_ious],
        mask_ious=[round(v, 4) for v in mask_ious],
        mean_box_iou=round(mean_box_iou, 4),
        mean_mask_iou=round(mean_mask_iou, 4),
        misclassified=misclassified,
        missing_detections=missing_detections,
    )

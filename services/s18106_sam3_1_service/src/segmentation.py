"""Stateless image segmentation using SAM3.1 native multiplex predictor."""

from __future__ import annotations

import os
import tempfile
from typing import Optional

import numpy as np
import torch
import torchvision
from PIL import Image

from src.config import AUTO_MASK_PROMPTS, SHM_DIR, config
from src.helpers import mask_to_detection
from src.models import get_predictor, inference_lock


def _save_temp_jpg(image: Image.Image) -> str:
    """Save image to a temp JPEG in RAM (/dev/shm) if available, else /tmp."""
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir=SHM_DIR)
    image.save(tmp.name, format="JPEG")
    tmp.close()
    return tmp.name


def _start_session(pred, resource_path: str, session_id: str | None = None) -> str:
    """Start a native session. Returns the session_id."""
    req = dict(type="start_session", resource_path=resource_path)
    if session_id is not None:
        req["session_id"] = session_id
    resp = pred.handle_request(req)
    return resp["session_id"]


def _close_session(pred, session_id: str) -> None:
    """Close a native session, ignoring errors."""
    try:
        pred.handle_request(dict(type="close_session", session_id=session_id))
    except Exception:
        pass


def segment_box(image: Image.Image, box: list[int]) -> dict:
    """Box-prompted segmentation. Returns a single detection dict."""
    pred = get_predictor()
    W, H = image.size
    # Normalize pixel [x1, y1, x2, y2] → [xmin, ymin, w, h] normalized for native API
    norm_box = [box[0] / W, box[1] / H, (box[2] - box[0]) / W, (box[3] - box[1]) / H]

    tmp_path = _save_temp_jpg(image)
    try:
        with inference_lock, torch.autocast("cuda", dtype=torch.bfloat16):
            sid = _start_session(pred, tmp_path)
            resp = pred.handle_request(dict(
                type="add_prompt",
                session_id=sid,
                frame_index=0,
                bounding_boxes=torch.tensor([norm_box], dtype=torch.float32),
                bounding_box_labels=torch.tensor([1], dtype=torch.int32),
                obj_id=1,
            ))
            outputs = resp["outputs"]
            _close_session(pred, sid)
    finally:
        os.unlink(tmp_path)

    masks = outputs.get("out_binary_masks", np.array([]))
    probs = outputs.get("out_probs", np.array([]))

    if len(probs) == 0 or len(masks) == 0:
        blank = np.zeros((H, W), dtype=bool)
        det = mask_to_detection(blank, 0.0, 0.0)
        det["iou_score"] = 0.0
        return det

    mask = masks[0].astype(bool)
    score = float(probs[0])
    area = float(mask.sum()) / mask.size
    det = mask_to_detection(mask, score, area)
    det["iou_score"] = score
    return det


def segment_text(
    image: Image.Image,
    text: str,
    detection_threshold: Optional[float] = None,
    mask_threshold: Optional[float] = None,
) -> list[dict]:
    """Text-prompted open-vocabulary segmentation."""
    pred = get_predictor()
    cfg = config["segmentation"]
    dt = detection_threshold if detection_threshold is not None else cfg["detection_threshold"]

    tmp_path = _save_temp_jpg(image)
    try:
        with inference_lock, torch.autocast("cuda", dtype=torch.bfloat16):
            sid = _start_session(pred, tmp_path)
            resp = pred.handle_request(dict(
                type="add_prompt",
                session_id=sid,
                frame_index=0,
                text=text,
                output_prob_thresh=dt,
            ))
            outputs = resp["outputs"]
            _close_session(pred, sid)
    finally:
        os.unlink(tmp_path)

    masks = outputs.get("out_binary_masks", np.array([]))
    probs = outputs.get("out_probs", np.array([]))

    detections = []
    for i in range(len(probs)):
        if probs[i] >= dt:
            mask = masks[i].astype(bool)
            score = float(probs[i])
            area = float(mask.sum()) / mask.size
            detections.append(mask_to_detection(mask, score, area))
    return detections


def segment_auto(
    image: Image.Image,
    threshold: Optional[float] = None,
    prompts: Optional[list[str]] = None,
) -> list[dict]:
    """Auto-mask via single-pass multi-class text query (one add_prompt call, one backbone pass)."""
    pred = get_predictor()
    cfg = config["segmentation"]
    thr = threshold if threshold is not None else cfg.get("auto_mask_threshold", 0.35)
    prompt_list = prompts if prompts is not None else AUTO_MASK_PROMPTS
    max_area = cfg.get("auto_mask_max_area", 0.95)
    nms_thr = cfg.get("auto_mask_nms_threshold", 0.5)

    # Join all prompts into one combined text query so add_prompt runs once
    combined_text = " ".join(p.strip().rstrip(".").strip() for p in prompt_list)
    if not combined_text.endswith("."):
        combined_text += "."

    tmp_path = _save_temp_jpg(image)
    try:
        with inference_lock, torch.autocast("cuda", dtype=torch.bfloat16):
            sid = _start_session(pred, tmp_path)

            resp = pred.handle_request(dict(
                type="add_prompt",
                session_id=sid,
                frame_index=0,
                text=combined_text,
                output_prob_thresh=thr,
            ))
            outputs = resp["outputs"]
            masks = outputs.get("out_binary_masks", np.array([]))
            probs = outputs.get("out_probs", np.array([]))
            all_detections = []
            for i in range(len(probs)):
                if probs[i] >= thr:
                    mask = masks[i].astype(bool)
                    area = float(mask.sum()) / mask.size
                    if area < max_area:
                        all_detections.append(mask_to_detection(mask, float(probs[i]), area))

            _close_session(pred, sid)
    finally:
        os.unlink(tmp_path)

    if not all_detections:
        return all_detections

    # NMS to remove overlapping detections across prompts
    boxes = torch.tensor(
        [[d["bbox"]["x1"], d["bbox"]["y1"], d["bbox"]["x2"], d["bbox"]["y2"]]
         for d in all_detections],
        dtype=torch.float32,
    )
    scores = torch.tensor([d["score"] for d in all_detections], dtype=torch.float32)
    keep = torchvision.ops.nms(boxes, scores, nms_thr)
    all_detections = [all_detections[i] for i in keep.tolist()]
    all_detections.sort(key=lambda d: d["score"], reverse=True)
    return all_detections


def segment_text_batch(
    images: list[Image.Image],
    text: str,
    detection_threshold: float | None = None,
    mask_threshold: float | None = None,
) -> list[list[dict]]:
    return [segment_text(img, text, detection_threshold, mask_threshold) for img in images]


def segment_auto_batch(
    images: list[Image.Image],
    threshold: float | None = None,
    prompts: list[str] | None = None,
) -> list[list[dict]]:
    return [segment_auto(img, threshold, prompts) for img in images]

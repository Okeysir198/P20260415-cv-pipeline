"""Image decoding and mask conversion utilities."""

from __future__ import annotations

import base64
import io

import numpy as np
from fastapi import HTTPException
from PIL import Image

from src.config import config


def decode_image(data: str) -> Image.Image:
    """Decode a base64-encoded image string to PIL Image."""
    if "," in data and data.split(",")[0].startswith("data:"):
        data = data.split(",", 1)[1]
    try:
        raw = base64.b64decode(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64: {exc}") from exc
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (max 20MB)")
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc
    if img.width > 8192 or img.height > 8192:
        raise HTTPException(status_code=413, detail=f"Image dimensions too large: {img.width}x{img.height} (max 8192x8192)")
    return img


def decode_frames(frames_b64: list[str]) -> list[Image.Image]:
    """Decode a list of base64-encoded frames."""
    return [decode_image(f) for f in frames_b64]


def mask_to_detection(mask: np.ndarray, score: float, area: float = 0.0) -> dict:
    """Convert a boolean mask to detection dict with base64 PNG, bbox, score, area."""
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    if not rows.any():
        bbox = {"x1": 0, "y1": 0, "x2": 0, "y2": 0}
    else:
        y1, y2 = int(np.where(rows)[0][[0, -1]][0]), int(np.where(rows)[0][[0, -1]][1]) + 1
        x1, x2 = int(np.where(cols)[0][[0, -1]][0]), int(np.where(cols)[0][[0, -1]][1]) + 1
        bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    buf = io.BytesIO()
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(buf, format="PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    if area == 0.0 and mask.size > 0:
        area = float(mask.sum()) / mask.size
    return {"mask": mask_b64, "bbox": bbox, "score": score, "area": area}


def mask_post_kwargs() -> dict:
    """Return post_process_masks kwargs from config for clean mask output."""
    cfg = config.get("segmentation", {})
    return {
        "binarize": True,
        "mask_threshold": cfg.get("mask_threshold", 0.5),
        "max_hole_area": cfg.get("max_hole_area", 0.0),
        "max_sprinkle_area": cfg.get("max_sprinkle_area", 0.0),
    }

"""HTTP clients for downstream services (SAM3, Flux NIM)."""

from __future__ import annotations

import requests
from fastapi import HTTPException

from .config import FLUX_NIM_URL, REQUEST_TIMEOUT, SAM3_URL


def call_sam3_box(image_b64: str, box: list[float]) -> str:
    """Call SAM3 /segment_box endpoint."""
    resp = requests.post(
        f"{SAM3_URL}/segment_box",
        json={"image": image_b64, "box": box},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["result"]["mask"]


def call_sam3_text(image_b64: str, text: str) -> str:
    """Call SAM3 /segment_text endpoint, return highest-scoring mask."""
    resp = requests.post(
        f"{SAM3_URL}/segment_text",
        json={"image": image_b64, "text": text},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    detections = resp.json()["detections"]
    if not detections:
        raise HTTPException(
            status_code=422,
            detail=f"SAM3 found no segments for text: {text}",
        )
    return max(detections, key=lambda r: r.get("score", 0.0))["mask"]


def call_flux(image_b64: str, prompt: str, seed: int, steps: int) -> str:
    """Call Flux NIM /v1/infer endpoint."""
    payload = {
        "prompt": prompt,
        "image": [f"data:image/png;base64,{image_b64}"],
        "seed": seed,
        "steps": steps,
    }
    resp = requests.post(f"{FLUX_NIM_URL}/v1/infer", json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    artifacts = resp.json().get("artifacts", [])
    if not artifacts:
        raise HTTPException(status_code=502, detail="Flux NIM returned no artifacts")
    result = artifacts[0]
    if result.get("finishReason") == "CONTENT_FILTERED":
        raise HTTPException(
            status_code=422,
            detail="Flux NIM content filter triggered — try a different prompt or image",
        )
    if not result.get("base64"):
        raise HTTPException(status_code=502, detail="Flux NIM returned empty image")
    return result["base64"]

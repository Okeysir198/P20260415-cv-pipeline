"""FastAPI app and endpoint handlers for Image Editor service."""

from __future__ import annotations

import asyncio
from typing import Optional

import requests
from fastapi import FastAPI

from .clients import call_flux, call_sam3_box, call_sam3_text
from .compositing import mask_composite
from .config import FLUX_NIM_URL, REQUEST_TIMEOUT, SAM3_URL, config
from .helpers import decode_image, decode_mask, encode_image
from .schemas import HealthResponse, InpaintRequest, InpaintResponse

app = FastAPI(title="Image Editor Orchestrator", version="1.0.0")


def _resolve_mask(req: InpaintRequest) -> Optional[str]:
    """Resolve mask from request: provided, or generated via SAM3."""
    if req.mask:
        return req.mask
    if req.bbox is not None:
        return call_sam3_box(req.image, req.bbox)
    if req.text_prompt is not None:
        return call_sam3_text(req.image, req.text_prompt)
    return None


def _inpaint_sync(req: InpaintRequest) -> InpaintResponse:
    """Synchronous inpaint logic (runs in executor)."""
    seed = req.seed or 0

    # Step 1: Resolve mask
    mask_b64 = _resolve_mask(req)

    # Step 2: Generate variants
    images = []
    for i in range(req.num_variants):
        edited_b64 = call_flux(req.image, req.prompt, seed + i, req.steps)
        if mask_b64:
            original = decode_image(req.image)
            edited = decode_image(edited_b64)
            mask_arr = decode_mask(mask_b64)
            images.append(encode_image(mask_composite(original, edited, mask_arr)))
        else:
            images.append(edited_b64)

    return InpaintResponse(images=images, mask_used=mask_b64, seed=seed)


@app.post("/inpaint", response_model=InpaintResponse)
async def inpaint_endpoint(request: InpaintRequest) -> InpaintResponse:
    """Inpaint an image using SAM3 segmentation + Flux NIM generation.

    Four mask modes:
    - mask provided → composite directly
    - bbox provided → SAM3 /segment_box → composite
    - text_prompt provided → SAM3 /segment_text → composite
    - none → direct Flux edit (no compositing)
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _inpaint_sync, request)


@app.get("/health", response_model=HealthResponse)
async def health_endpoint() -> HealthResponse:
    """Check health of orchestrator and downstream services."""
    flux_status = "unknown"
    sam3_status = "unknown"

    try:
        resp = requests.get(f"{FLUX_NIM_URL}/v1/health/ready", timeout=5)
        flux_status = "ok" if resp.status_code == 200 else f"error ({resp.status_code})"
    except Exception as exc:
        flux_status = f"unreachable ({exc})"

    try:
        resp = requests.get(f"{SAM3_URL}/health", timeout=5)
        sam3_status = "ok" if resp.status_code == 200 else f"error ({resp.status_code})"
    except Exception as exc:
        sam3_status = f"unreachable ({exc})"

    overall = "ok" if flux_status == "ok" and sam3_status == "ok" else "degraded"
    return HealthResponse(status=overall, flux_nim=flux_status, sam3=sam3_status)

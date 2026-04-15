"""FastAPI app and endpoint handlers for SAM3.1 segmentation service."""

from __future__ import annotations

import asyncio
import time as _time
from typing import Union

from fastapi import FastAPI

from src.config import config, logger
from src.helpers import decode_image
from src.models import device, loaded_models
from src.schemas import (
    AutoMaskRequest,
    AutoMaskResponse,
    BatchAutoMaskRequest,
    BatchTextRequest,
    BoxRequest,
    BoxResponse,
    BoxResultData,
    Detection,
    FrameAddRequest,
    FrameResponse,
    HealthResponse,
    PromptRequest,
    PropagateRequest,
    PropagateResponse,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionDetection,
    TextRequest,
    TextResponse,
)
from src.segmentation import segment_auto, segment_auto_batch, segment_box, segment_text, segment_text_batch
from src.sessions import (
    add_frame_sync,
    add_prompts_sync,
    create_session_sync,
    delete_session_sync,
    get_sessions_info,
    propagate_sync,
)

app = FastAPI(title="SAM3.1 Segmentation Service", version="1.0.0")


@app.middleware("http")
async def log_request_timing(request, call_next):
    start = _time.perf_counter()
    response = await call_next(request)
    elapsed = _time.perf_counter() - start
    logger.info("%s %s %.3fs %d", request.method, request.url.path, elapsed, response.status_code)
    return response


# ---------------------------------------------------------------------------
# Image endpoints (stateless)
# ---------------------------------------------------------------------------


def _process_box(req: BoxRequest) -> BoxResponse:
    image = decode_image(req.image)
    result = segment_box(image, req.box)
    return BoxResponse(result=BoxResultData(**result))


def _process_text(req: TextRequest) -> TextResponse:
    image = decode_image(req.image)
    dets = segment_text(image, req.text, req.detection_threshold, req.mask_threshold)
    return TextResponse(detections=[Detection(**d) for d in dets])


def _process_auto(req: AutoMaskRequest) -> AutoMaskResponse:
    image = decode_image(req.image)
    dets = segment_auto(image, req.threshold, req.prompts)
    return AutoMaskResponse(detections=[Detection(**d) for d in dets])


@app.post("/segment_box", response_model=Union[BoxResponse, list[BoxResponse]])
async def segment_box_endpoint(request: Union[BoxRequest, list[BoxRequest]]):
    """Box-prompted segmentation. Accepts single request or list for batch."""
    loop = asyncio.get_running_loop()
    if isinstance(request, list):
        tasks = [loop.run_in_executor(None, _process_box, r) for r in request]
        return await asyncio.gather(*tasks)
    return await asyncio.to_thread(_process_box, request)


@app.post("/segment_text", response_model=Union[TextResponse, list[TextResponse]])
async def segment_text_endpoint(request: Union[TextRequest, list[TextRequest]]):
    """Text-prompted open-vocab segmentation. Accepts single request or list for batch."""
    loop = asyncio.get_running_loop()
    if isinstance(request, list):
        tasks = [loop.run_in_executor(None, _process_text, r) for r in request]
        return await asyncio.gather(*tasks)
    return await asyncio.to_thread(_process_text, request)


@app.post("/auto_mask", response_model=Union[AutoMaskResponse, list[AutoMaskResponse]])
async def auto_mask_endpoint(request: Union[AutoMaskRequest, list[AutoMaskRequest]]):
    """Segment everything via multi-prompt open-vocab detection. Accepts single or batch."""
    loop = asyncio.get_running_loop()
    if isinstance(request, list):
        tasks = [loop.run_in_executor(None, _process_auto, r) for r in request]
        return await asyncio.gather(*tasks)
    return await asyncio.to_thread(_process_auto, request)


# ---------------------------------------------------------------------------
# Batch endpoints (convenience API for multiple images in one request)
# ---------------------------------------------------------------------------


@app.post("/segment_text_batch", response_model=list[TextResponse])
async def segment_text_batch_endpoint(request: BatchTextRequest):
    """Batch text segmentation — process multiple images with one request."""
    images = [decode_image(item.image) for item in request.items]
    dets_list = await asyncio.to_thread(
        segment_text_batch,
        images,
        request.text,
        request.detection_threshold,
        request.mask_threshold,
    )
    return [
        TextResponse(detections=[Detection(**d) for d in dets])
        for dets in dets_list
    ]


@app.post("/auto_mask_batch", response_model=list[AutoMaskResponse])
async def auto_mask_batch_endpoint(request: BatchAutoMaskRequest):
    """Batch auto-mask segmentation — process multiple images with one request."""
    images = [decode_image(item.image) for item in request.items]
    dets_list = await asyncio.to_thread(
        segment_auto_batch,
        images,
        request.threshold,
        request.prompts,
    )
    return [
        AutoMaskResponse(detections=[Detection(**d) for d in dets])
        for dets in dets_list
    ]


# ---------------------------------------------------------------------------
# Session endpoints (stateful)
# ---------------------------------------------------------------------------


@app.post("/sessions", response_model=SessionCreateResponse)
async def create_session(request: SessionCreateRequest):
    """Create a video session. File mode: pass frames now. Streaming mode: omit frames."""
    result = await asyncio.to_thread(
        create_session_sync, request.mode, request.frames, request.text,
    )
    return SessionCreateResponse(**result)


@app.post("/sessions/{session_id}/frames", response_model=FrameResponse)
async def add_frame(session_id: str, request: FrameAddRequest):
    """Add a frame to a streaming tracker session. Returns masks for tracked objects."""
    result = await asyncio.to_thread(add_frame_sync, session_id, request.frame)
    return FrameResponse(
        frame_idx=result["frame_idx"],
        detections=[SessionDetection(**d) for d in result["detections"]],
    )


@app.post("/sessions/{session_id}/prompts", response_model=FrameResponse)
async def add_prompts(session_id: str, request: PromptRequest):
    """Add interactive prompts (points/boxes/masks) to a tracker session on a specific frame."""
    result = await asyncio.to_thread(
        add_prompts_sync, session_id,
        request.frame_idx, request.obj_ids,
        request.points, request.labels, request.boxes, request.masks,
    )
    return FrameResponse(
        frame_idx=result["frame_idx"],
        detections=[SessionDetection(**d) for d in result["detections"]],
    )


@app.post("/sessions/{session_id}/propagate", response_model=PropagateResponse)
async def propagate(session_id: str, request: PropagateRequest = PropagateRequest()):
    """Propagate tracked objects through all frames."""
    results = await asyncio.to_thread(propagate_sync, session_id, request.max_frames)
    return PropagateResponse(
        frames=[
            FrameResponse(
                frame_idx=r["frame_idx"],
                detections=[SessionDetection(**d) for d in r["detections"]],
            )
            for r in results
        ]
    )


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and free GPU memory."""
    return delete_session_sync(session_id)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        device=device(),
        model=config["model"]["name"],
        dtype=config.get("model", {}).get("dtype", "bfloat16"),
        loaded=loaded_models(),
        sessions=get_sessions_info(),
    )

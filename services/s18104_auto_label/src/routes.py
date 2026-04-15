"""FastAPI app, lifespan, and all endpoint handlers."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import requests
from fastapi import FastAPI, HTTPException, Query

from src.annotator import annotate_single
from src.config import MAX_CONCURRENT_JOBS, SAM3_URL, logger
from src.formatters import format_output
from src.geometry import decode_image, strip_data_uri
from src.schemas import (
    AnnotateRequest,
    AnnotateResponse,
    ConvertRequest,
    ConvertResponse,
    HealthResponse,
    JobCreateRequest,
    JobCreateResponse,
    JobListItem,
    JobState,
    VideoFrameDetection,
    VideoFrameRequest,
    VideoFrameResponse,
    VideoSessionRequest,
    VideoSessionResponse,
)
from src.state import (
    add_video_frame_sync,
    cancel_job,
    create_job,
    create_video_session_sync,
    delete_video_session_sync,
    get_active_job_count,
    get_job,
    get_video_session_count,
    init_job_state,
    list_jobs,
    process_job,
    propagate_video_session_sync,
    shutdown_jobs,
    shutdown_video_sessions,
    ttl_cleanup_loop,
)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Initialize semaphore and background cleanup on startup, cleanup on shutdown."""
    init_job_state()
    ttl_task = asyncio.create_task(ttl_cleanup_loop())
    logger.info(
        "Auto Label Service started — SAM3_URL=%s, max_concurrent_jobs=%d",
        SAM3_URL, MAX_CONCURRENT_JOBS,
    )
    yield
    # Shutdown
    shutdown_jobs()
    ttl_task.cancel()
    shutdown_video_sessions()
    logger.info("Auto Label Service shut down")


app = FastAPI(title="Auto Label Service", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# POST /annotate — Sync single-image annotation
# ---------------------------------------------------------------------------


@app.post("/annotate", response_model=AnnotateResponse)
async def annotate_endpoint(request: AnnotateRequest) -> AnnotateResponse:
    """Annotate a single image synchronously using SAM3."""
    t0 = time.time()
    loop = asyncio.get_event_loop()

    image_b64 = strip_data_uri(request.image)
    img = decode_image(image_b64)
    img_w, img_h = img.size

    detections = await loop.run_in_executor(
        None,
        annotate_single,
        image_b64,
        request.classes,
        request.text_prompts,
        request.mode,
        request.confidence_threshold,
        request.nms_iou_threshold,
        request.include_masks,
        img_w,
        img_h,
        request.detection_classes,
        request.class_rules,
        request.vlm_verify,
    )

    formatted = format_output(detections, request.output_format, img_w, img_h)

    return AnnotateResponse(
        detections=detections,
        image_width=img_w,
        image_height=img_h,
        num_detections=len(detections),
        processing_time_s=round(time.time() - t0, 3),
        formatted_output=formatted,
    )


# ---------------------------------------------------------------------------
# POST /jobs — Create async batch job
# ---------------------------------------------------------------------------


@app.post("/jobs", response_model=JobCreateResponse)
async def create_job_endpoint(request: JobCreateRequest) -> JobCreateResponse:
    """Create an async batch annotation job."""
    if not request.images:
        raise HTTPException(status_code=400, detail="No images provided")

    # Validate output format early
    format_output([], request.output_format, 1, 1)

    job_id, total_images = create_job(request)
    asyncio.create_task(process_job(job_id, request))

    return JobCreateResponse(
        job_id=job_id,
        total_images=total_images,
        status="queued",
    )


# ---------------------------------------------------------------------------
# GET /jobs/{job_id} — Get job status with partial results
# ---------------------------------------------------------------------------


@app.get("/jobs/{job_id}", response_model=JobState)
async def get_job_endpoint(job_id: str) -> JobState:
    """Get status and partial results for a batch job."""
    return JobState(**get_job(job_id))


# ---------------------------------------------------------------------------
# GET /jobs — List all jobs
# ---------------------------------------------------------------------------


@app.get("/jobs", response_model=list[JobListItem])
async def list_jobs_endpoint(status: str | None = Query(None, description="Filter by status")) -> list[JobListItem]:
    """List all jobs, optionally filtered by status."""
    return [JobListItem(**j) for j in list_jobs(status)]


# ---------------------------------------------------------------------------
# DELETE /jobs/{job_id} — Cancel a job
# ---------------------------------------------------------------------------


@app.delete("/jobs/{job_id}")
async def cancel_job_endpoint(job_id: str) -> dict:
    """Cancel a running or queued job."""
    return cancel_job(job_id)


# ---------------------------------------------------------------------------
# POST /video/sessions — Create video session
# ---------------------------------------------------------------------------


@app.post("/video/sessions", response_model=VideoSessionResponse)
async def create_video_session(request: VideoSessionRequest) -> VideoSessionResponse:
    """Create a video annotation session backed by SAM3."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, create_video_session_sync, request)


# ---------------------------------------------------------------------------
# POST /video/sessions/{id}/frames — Add frame
# ---------------------------------------------------------------------------


@app.post("/video/sessions/{session_id}/frames", response_model=VideoFrameResponse)
async def add_video_frame(session_id: str, request: VideoFrameRequest) -> VideoFrameResponse:
    """Add a frame to a video session. Returns detections with class info and polygons."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, add_video_frame_sync, session_id, request)
    return VideoFrameResponse(
        frame_idx=result["frame_idx"],
        detections=[VideoFrameDetection(**d) for d in result["detections"]],
    )


# ---------------------------------------------------------------------------
# POST /video/sessions/{id}/propagate — Get all frame results (video mode)
# ---------------------------------------------------------------------------


@app.post("/video/sessions/{session_id}/propagate")
async def propagate_video_session(session_id: str) -> dict:
    """Propagate video session and return all frame results (for video mode)."""
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, propagate_video_session_sync, session_id)
    return {"frames": results}


# ---------------------------------------------------------------------------
# DELETE /video/sessions/{id} — Close video session
# ---------------------------------------------------------------------------


@app.delete("/video/sessions/{session_id}")
async def delete_video_session(session_id: str) -> dict:
    """Delete a video session and its backing SAM3 session."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, delete_video_session_sync, session_id)


# ---------------------------------------------------------------------------
# POST /convert — Format conversion
# ---------------------------------------------------------------------------


@app.post("/convert", response_model=ConvertResponse)
async def convert_endpoint(request: ConvertRequest) -> ConvertResponse:
    """Convert detections to a different output format (no SAM3 call needed)."""
    formatted = format_output(
        request.detections, request.output_format, request.image_width, request.image_height,
    )
    return ConvertResponse(formatted_output=formatted)


# ---------------------------------------------------------------------------
# GET /health — Service health check
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health_endpoint() -> HealthResponse:
    """Check service health, SAM3 reachability, and active workloads."""
    def _check_sam3() -> str:
        try:
            resp = requests.get(f"{SAM3_URL}/health", timeout=5)
            return "ok" if resp.status_code == 200 else f"error ({resp.status_code})"
        except Exception as exc:
            return f"unreachable ({exc})"

    loop = asyncio.get_event_loop()
    sam3_status = await loop.run_in_executor(None, _check_sam3)

    active_jobs = get_active_job_count()
    active_sessions = get_video_session_count()

    overall = "ok" if sam3_status == "ok" else "degraded"
    return HealthResponse(
        status=overall,
        sam3=sam3_status,
        active_jobs=active_jobs,
        active_video_sessions=active_sessions,
    )

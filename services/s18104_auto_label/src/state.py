"""Mutable state: batch job engine and video session management."""

from __future__ import annotations

import asyncio
import threading
import time
import uuid

import httpx
import requests
from fastapi import HTTPException

from src.annotator import annotate_single
from src.config import (
    BATCH_CONCURRENCY,
    JOB_TTL_SECONDS,
    MAX_CONCURRENT_JOBS,
    MAX_VIDEO_SESSIONS,
    REQUEST_TIMEOUT,
    SAM3_URL,
    VIDEO_SESSION_TTL,
    logger,
)
from src.formatters import format_output
from src.geometry import bbox_from_sam3, compute_bbox_norm, decode_image, mask_to_polygon, strip_data_uri
from src.schemas import (
    JobCreateRequest,
    VideoFrameRequest,
    VideoSessionRequest,
    VideoSessionResponse,
)


# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_job_semaphore: asyncio.Semaphore | None = None
_shutdown_event = asyncio.Event()


def init_job_state() -> None:
    """Initialize the job semaphore. Call from lifespan startup."""
    global _job_semaphore
    _job_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)


# ---------------------------------------------------------------------------
# Job processing
# ---------------------------------------------------------------------------


async def process_job(job_id: str, request: JobCreateRequest) -> None:
    """Process a batch job asynchronously with parallel batch processing.

    Images are processed in parallel batches (default: 4 at a time) using asyncio.gather.
    Each image is decoded independently, supporting different image sizes.
    """
    assert _job_semaphore is not None
    async with _job_semaphore:
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is None:
                return
            job["status"] = "running"

        loop = asyncio.get_event_loop()

        # Process in parallel batches
        batch_size = BATCH_CONCURRENCY
        total_images = len(request.images)

        for batch_start in range(0, total_images, batch_size):
            batch_end = min(batch_start + batch_size, total_images)
            batch = request.images[batch_start:batch_end]

            # Check for cancellation before starting batch
            with _jobs_lock:
                job = _jobs.get(job_id)
                if job is None or job["status"] == "cancelled":
                    return

            # Prepare batch data
            batch_data = []
            for img_input in batch:
                try:
                    image_b64 = strip_data_uri(img_input.image)
                    img = decode_image(image_b64)
                    batch_data.append((image_b64, img.size[0], img.size[1], img_input.filename))
                except Exception as exc:
                    logger.error("Job %s: failed to decode image %s: %s", job_id, img_input.filename, exc)
                    batch_data.append((None, 0, 0, img_input.filename, exc))

            # Process batch in parallel
            async def process_single(img_data: tuple) -> dict:
                image_b64, img_w, img_h, filename = img_data[:4]
                if image_b64 is None:
                    return {"filename": filename, "error": str(img_data[4])}

                try:
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
                    )

                    formatted = format_output(detections, request.output_format, img_w, img_h)

                    return {
                        "filename": filename,
                        "num_detections": len(detections),
                        "detections": [d.model_dump() for d in detections],
                        "formatted_output": formatted,
                        "image_width": img_w,
                        "image_height": img_h,
                    }
                except Exception as exc:
                    logger.error("Job %s: failed on image %s: %s", job_id, filename, exc)
                    return {"filename": filename, "error": str(exc)}

            tasks = [process_single(data) for data in batch_data]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Store results
            with _jobs_lock:
                job = _jobs.get(job_id)
                if job is None or job["status"] == "cancelled":
                    return

                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        filename = batch_data[i][3]
                        job["results"].append({"filename": filename, "error": str(result)})
                    else:
                        job["results"].append(result)

                job["processed_images"] = batch_end

                # Clear image data from request to free memory
                for img_input in batch:
                    img_input.image = ""

        # Mark job as completed
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is not None and job["status"] == "running":
                job["status"] = "completed"

        # Webhook notification
        if request.webhook_url:
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    with _jobs_lock:
                        job_data = _jobs.get(job_id, {})
                    await client.post(request.webhook_url, json={
                        "job_id": job_id,
                        "status": job_data.get("status", "completed"),
                        "total_images": job_data.get("total_images", 0),
                        "processed_images": job_data.get("processed_images", 0),
                    })
            except Exception as exc:
                logger.warning("Job %s: webhook notification failed: %s", job_id, exc)


async def ttl_cleanup_loop() -> None:
    """Background task: periodically remove completed/failed jobs older than TTL."""
    while not _shutdown_event.is_set():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            return
        now = time.time()
        with _jobs_lock:
            expired = [
                jid for jid, j in _jobs.items()
                if j["status"] in ("completed", "failed", "cancelled")
                and now - j["created_at"] > JOB_TTL_SECONDS
            ]
            for jid in expired:
                del _jobs[jid]
                logger.info("Cleaned up expired job %s", jid)


def create_job(request: JobCreateRequest) -> tuple[str, int]:
    """Create a new job entry. Returns (job_id, total_images)."""
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "total_images": len(request.images),
            "processed_images": 0,
            "results": [],
            "error": None,
            "created_at": time.time(),
        }
    return job_id, len(request.images)


def get_job(job_id: str) -> dict:
    """Get a job dict or raise 404."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


def list_jobs(status: str | None = None) -> list[dict]:
    """List all jobs, optionally filtered by status."""
    with _jobs_lock:
        items = []
        for j in _jobs.values():
            if status and j["status"] != status:
                continue
            items.append({
                "job_id": j["job_id"],
                "status": j["status"],
                "total_images": j["total_images"],
                "processed_images": j["processed_images"],
                "created_at": j["created_at"],
            })
    return items


def cancel_job(job_id: str) -> dict:
    """Cancel a running or queued job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        if job["status"] in ("completed", "failed", "cancelled"):
            return {"job_id": job_id, "status": job["status"], "message": "Job already finished"}
        job["status"] = "cancelled"
    return {"job_id": job_id, "status": "cancelled"}


def shutdown_jobs() -> None:
    """Cancel all active jobs on shutdown."""
    _shutdown_event.set()
    with _jobs_lock:
        for job in _jobs.values():
            if job["status"] in ("queued", "running"):
                job["status"] = "cancelled"


def get_active_job_count() -> int:
    """Return number of active (queued/running) jobs."""
    with _jobs_lock:
        return sum(1 for j in _jobs.values() if j["status"] in ("queued", "running"))


# ---------------------------------------------------------------------------
# Video session state
# ---------------------------------------------------------------------------

_video_sessions: dict[str, dict] = {}
_video_sessions_lock = threading.Lock()


def _cleanup_expired_video_sessions() -> None:
    """Remove expired video sessions."""
    now = time.time()
    # Collect and remove expired sessions under lock, then close SAM3 sessions outside lock
    with _video_sessions_lock:
        expired = [
            (sid, _video_sessions.pop(sid))
            for sid, s in list(_video_sessions.items())
            if now - s.get("created_at", 0) > VIDEO_SESSION_TTL
        ]
    for sid, sess in expired:
        try:
            requests.delete(
                f"{SAM3_URL}/sessions/{sess['sam3_session_id']}",
                timeout=5,
            )
        except Exception:
            pass
        logger.info("Cleaned up expired video session %s", sid)


def create_video_session_sync(req: VideoSessionRequest) -> VideoSessionResponse:
    """Create a video session by proxying to SAM3."""
    _cleanup_expired_video_sessions()

    with _video_sessions_lock:
        if len(_video_sessions) >= MAX_VIDEO_SESSIONS:
            raise HTTPException(
                status_code=429,
                detail=f"Max {MAX_VIDEO_SESSIONS} active video sessions reached",
            )

    # Validate video mode requirements
    if req.mode == "video":
        if not req.text:
            raise HTTPException(status_code=400, detail="video mode requires 'text' prompt")
        if not req.frames:
            raise HTTPException(status_code=400, detail="video mode requires 'frames' (all frames upfront)")

    # Create SAM3 session
    payload: dict = {"mode": req.mode}
    if req.text:
        payload["text"] = req.text
    if req.frames:
        payload["frames"] = req.frames
    resp = requests.post(f"{SAM3_URL}/sessions", json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    sam3_resp = resp.json()
    sam3_session_id = sam3_resp["session_id"]

    local_id = uuid.uuid4().hex[:12]

    # Build obj_id → class mapping from classes
    obj_class_map: dict[int, tuple[int, str]] = {}
    for cls_id, cls_name in req.classes.items():
        obj_class_map[int(cls_id)] = (int(cls_id), cls_name)

    with _video_sessions_lock:
        _video_sessions[local_id] = {
            "sam3_session_id": sam3_session_id,
            "mode": req.mode,
            "classes": req.classes,
            "output_format": req.output_format,
            "obj_class_map": obj_class_map,
            "created_at": time.time(),
            "frame_count": len(req.frames) if req.frames else 0,
            "width": sam3_resp.get("width", 0),
            "height": sam3_resp.get("height", 0),
        }

    return VideoSessionResponse(
        session_id=local_id,
        sam3_session_id=sam3_session_id,
        mode=req.mode,
    )


def _get_video_session(session_id: str) -> dict:
    """Retrieve a video session or raise 404."""
    with _video_sessions_lock:
        session = _video_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Video session {session_id} not found")
    return session


def add_video_frame_sync(session_id: str, req: VideoFrameRequest) -> dict:
    """Add a frame to a video session, optionally with prompts. Returns detections."""
    session = _get_video_session(session_id)
    sam3_sid = session["sam3_session_id"]

    # Add frame to SAM3 session
    resp = requests.post(
        f"{SAM3_URL}/sessions/{sam3_sid}/frames",
        json={"frame": req.frame},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    frame_result = resp.json()
    frame_idx = frame_result.get("frame_idx", session["frame_count"])

    # Update frame count
    with _video_sessions_lock:
        session["frame_count"] = frame_idx + 1

    # Add prompts if provided
    if req.prompts:
        for prompt in req.prompts:
            resp = requests.post(
                f"{SAM3_URL}/sessions/{sam3_sid}/prompts",
                json=prompt,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            frame_result = resp.json()

    # Get image dimensions from frame for bbox normalization
    try:
        img = decode_image(req.frame)
        img_w, img_h = img.size
    except Exception:
        img_w, img_h = 1, 1

    # Convert SAM3 detections to our format
    sam3_dets = frame_result.get("detections", [])
    obj_class_map = session.get("obj_class_map", {})
    detections = []
    for sd in sam3_dets:
        obj_id = sd.get("obj_id", -1)
        cls_id, cls_name = obj_class_map.get(obj_id, (-1, ""))
        bbox_xyxy = bbox_from_sam3(sd.get("bbox", {}))
        bbox_norm = compute_bbox_norm(bbox_xyxy, img_w, img_h)
        mask_b64 = sd.get("mask")
        polygon = mask_to_polygon(mask_b64, img_h, img_w) if mask_b64 else []

        detections.append({
            "obj_id": obj_id,
            "class_id": cls_id,
            "class_name": cls_name,
            "score": float(sd.get("score", 0.0)),
            "bbox_xyxy": bbox_xyxy,
            "bbox_norm": bbox_norm,
            "polygon": polygon,
            "mask": mask_b64,
            "area": float(sd.get("area", 0.0)),
        })

    return {"frame_idx": frame_idx, "detections": detections}


def delete_video_session_sync(session_id: str) -> dict:
    """Delete a video session and its SAM3 session."""
    with _video_sessions_lock:
        session = _video_sessions.pop(session_id, None)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Video session {session_id} not found")

    # Delete SAM3 session
    try:
        requests.delete(
            f"{SAM3_URL}/sessions/{session['sam3_session_id']}",
            timeout=10,
        )
    except Exception as exc:
        logger.warning("Failed to delete SAM3 session %s: %s", session["sam3_session_id"], exc)

    return {"deleted": True, "session_id": session_id}


def propagate_video_session_sync(session_id: str) -> list[dict]:
    """Propagate a video session and return all frame results.

    For video mode (text-driven), this calls SAM3's propagate endpoint
    and returns detections for all frames with class mapping applied.
    """
    session = _get_video_session(session_id)
    sam3_sid = session["sam3_session_id"]
    obj_class_map = session.get("obj_class_map", {})
    img_w = session.get("width", 0) or 1
    img_h = session.get("height", 0) or 1

    # Call SAM3 propagate
    resp = requests.post(
        f"{SAM3_URL}/sessions/{sam3_sid}/propagate",
        json={},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    sam3_result = resp.json()

    # Convert SAM3 results to our format
    frames_output = []
    for frame_result in sam3_result.get("frames", []):
        frame_dets = []
        for sd in frame_result.get("detections", []):
            obj_id = sd.get("obj_id", -1)
            cls_id, cls_name = obj_class_map.get(obj_id, (-1, ""))
            bbox_xyxy = bbox_from_sam3(sd.get("bbox", {}))
            bbox_norm = compute_bbox_norm(bbox_xyxy, img_w, img_h)
            mask_b64 = sd.get("mask")
            polygon = mask_to_polygon(mask_b64, img_h, img_w) if mask_b64 else []

            frame_dets.append({
                "obj_id": obj_id,
                "class_id": cls_id,
                "class_name": cls_name,
                "score": float(sd.get("score", 0.0)),
                "bbox_xyxy": bbox_xyxy,
                "bbox_norm": bbox_norm,
                "polygon": polygon,
                "mask": mask_b64,
                "area": float(sd.get("area", 0.0)),
            })

        frames_output.append({
            "frame_idx": frame_result.get("frame_idx", len(frames_output)),
            "detections": frame_dets,
        })

    return frames_output


def shutdown_video_sessions() -> None:
    """Close all video sessions on shutdown."""
    with _video_sessions_lock:
        for session in _video_sessions.values():
            try:
                requests.delete(
                    f"{SAM3_URL}/sessions/{session['sam3_session_id']}",
                    timeout=5,
                )
            except Exception as exc:
                logger.warning("Failed to delete SAM3 session %s on shutdown: %s", session["sam3_session_id"], exc)
        _video_sessions.clear()


def get_video_session_count() -> int:
    """Return number of active video sessions."""
    with _video_sessions_lock:
        return len(_video_sessions)

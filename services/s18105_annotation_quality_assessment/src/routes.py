"""FastAPI app, lifespan, and all endpoint handlers."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, Query

from src.config import MAX_CONCURRENT_JOBS, OLLAMA_URL, SAM3_URL, logger
from src.geometry import decode_image, strip_data_uri
from src.scoring import aggregate_results, apply_fixes
from src.schemas import (
    FixRequest, FixResponse, HealthResponse,
    JobCreateResponse, JobListItem, JobState,
    QAJobRequest, ReportRequest, ReportResponse,
    SAM3Verification, SuggestedFix, ValidateRequest,
    ValidateResponse, ValidationIssue, VLMVerification, VerifyRequest, VerifyResponse,
)
from src.state import (
    cancel_job, create_job, get_active_job_count, get_job,
    init_job_state, list_jobs, process_job, process_single_image_validate,
    process_single_image_verify_async, shutdown_jobs, ttl_cleanup_loop,
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
        "Annotation QA Service started — SAM3_URL=%s, max_concurrent_jobs=%d",
        SAM3_URL, MAX_CONCURRENT_JOBS,
    )
    yield
    shutdown_jobs()
    ttl_task.cancel()
    logger.info("Annotation QA Service shut down")


app = FastAPI(title="Annotation QA Service", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoint: POST /validate — Structural validation only, no SAM3
# ---------------------------------------------------------------------------


@app.post("/validate", response_model=ValidateResponse)
async def validate_endpoint(request: ValidateRequest) -> ValidateResponse:
    """Validate annotations structurally (no SAM3 call)."""
    t0 = time.time()
    loop = asyncio.get_event_loop()

    image_b64 = strip_data_uri(request.image)
    img = decode_image(image_b64)
    img_w, img_h = img.size

    result = await loop.run_in_executor(
        None,
        process_single_image_validate,
        request.labels,
        request.label_format,
        request.classes,
        request.config,
        img_w,
        img_h,
    )

    return ValidateResponse(
        issues=[ValidationIssue(**i) for i in result["issues"]],
        num_annotations=result["num_annotations"],
        num_issues=result["num_issues"],
        quality_score=result["quality_score"],
        grade=result["grade"],
        suggested_fixes=[SuggestedFix(**f) for f in result["suggested_fixes"]],
        label_format=request.label_format,
        processing_time_s=round(time.time() - t0, 3),
    )


# ---------------------------------------------------------------------------
# Endpoint: POST /verify — Structural + SAM3 verification
# ---------------------------------------------------------------------------


@app.post("/verify", response_model=VerifyResponse)
async def verify_endpoint(request: VerifyRequest) -> VerifyResponse:
    """Verify annotations with structural validation, SAM3, and optional VLM."""
    t0 = time.time()

    image_b64 = strip_data_uri(request.image)
    img = decode_image(image_b64)
    img_w, img_h = img.size

    result = await process_single_image_verify_async(
        image_b64,
        request.labels,
        request.label_format,
        request.classes,
        request.text_prompts,
        request.include_missing_detection,
        request.config,
        img_w,
        img_h,
        enable_vlm=request.enable_vlm,
        vlm_trigger=request.vlm_trigger,
        class_rules=request.class_rules,
        vlm_budget=request.vlm_budget,
    )

    vlm_v = None
    if result.get("vlm_verification"):
        vlm_v = VLMVerification(**result["vlm_verification"])

    return VerifyResponse(
        issues=[ValidationIssue(**i) for i in result["issues"]],
        sam3_verification=SAM3Verification(**result["sam3_verification"]),
        num_annotations=result["num_annotations"],
        num_issues=result["num_issues"],
        quality_score=result["quality_score"],
        grade=result["grade"],
        suggested_fixes=[SuggestedFix(**f) for f in result["suggested_fixes"]],
        label_format=request.label_format,
        processing_time_s=round(time.time() - t0, 3),
        vlm_verification=vlm_v,
    )


# ---------------------------------------------------------------------------
# Endpoint: POST /fix — Apply corrections
# ---------------------------------------------------------------------------


@app.post("/fix", response_model=FixResponse)
async def fix_endpoint(request: FixRequest) -> FixResponse:
    """Apply suggested fixes to labels.

    Accepts the issues and suggested_fixes from a previous validate/verify call,
    applies auto-fixable corrections, and returns the corrected labels.
    """
    loop = asyncio.get_event_loop()

    # We need image dimensions for COCO format conversion.
    # For YOLO/YOLO-seg, dimensions don't matter for output, but we use 1x1 as placeholder.
    # If the caller needs COCO output, they should provide the dimensions in the config.
    img_w = 1
    img_h = 1

    # For COCO labels, try to infer dimensions from the labels themselves
    if request.label_format == "coco" and request.labels:
        first_label = request.labels[0]
        if isinstance(first_label, dict):
            # We can't reliably infer dimensions from COCO, use a large default
            img_w = 1000
            img_h = 1000

    result = await loop.run_in_executor(
        None,
        apply_fixes,
        request.labels,
        request.label_format,
        request.suggested_fixes,
        request.auto_apply,
        img_w,
        img_h,
    )

    return result


# ---------------------------------------------------------------------------
# Endpoint: POST /jobs — Create batch QA job
# ---------------------------------------------------------------------------


@app.post("/jobs", response_model=JobCreateResponse)
async def create_job_endpoint(request: QAJobRequest) -> JobCreateResponse:
    """Create an async batch QA job."""
    job_id, total_images = create_job(request)

    asyncio.create_task(process_job(job_id, request))

    return JobCreateResponse(
        job_id=job_id,
        total_images=total_images,
        status="queued",
    )


# ---------------------------------------------------------------------------
# Endpoint: GET /jobs/{job_id} — Get job status
# ---------------------------------------------------------------------------


@app.get("/jobs/{job_id}", response_model=JobState)
async def get_job_endpoint(job_id: str) -> JobState:
    """Get status and partial results for a batch QA job."""
    job = get_job(job_id)
    return JobState(**job)


# ---------------------------------------------------------------------------
# Endpoint: GET /jobs — List all jobs
# ---------------------------------------------------------------------------


@app.get("/jobs", response_model=list[JobListItem])
async def list_jobs_endpoint(status: Optional[str] = Query(None, description="Filter by status")) -> list[JobListItem]:
    """List all QA jobs, optionally filtered by status."""
    items = list_jobs(status)
    return [JobListItem(**item) for item in items]


# ---------------------------------------------------------------------------
# Endpoint: DELETE /jobs/{job_id} — Cancel a job
# ---------------------------------------------------------------------------


@app.delete("/jobs/{job_id}")
async def cancel_job_endpoint(job_id: str) -> dict:
    """Cancel a running or queued QA job."""
    return cancel_job(job_id)


# ---------------------------------------------------------------------------
# Endpoint: POST /report — Aggregate results into a report
# ---------------------------------------------------------------------------


@app.post("/report", response_model=ReportResponse)
async def report_endpoint(request: ReportRequest) -> ReportResponse:
    """Aggregate per-image QA results into a dataset-level report."""
    return aggregate_results(request.results, request.dataset_name, request.classes)


# ---------------------------------------------------------------------------
# Endpoint: GET /health — Service health check
# ---------------------------------------------------------------------------


async def _check_service(url: str) -> str:
    """Check if a service is reachable. Returns 'ok' or error string."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(url)
        return "ok" if resp.status_code == 200 else f"error ({resp.status_code})"
    except Exception as exc:
        return f"unreachable ({exc})"


@app.get("/health", response_model=HealthResponse)
async def health_endpoint() -> HealthResponse:
    """Check service health, SAM3, and Ollama reachability."""
    sam3_status, ollama_status = await asyncio.gather(
        _check_service(f"{SAM3_URL}/health"),
        _check_service(f"{OLLAMA_URL}/api/tags"),
    )

    active_jobs = get_active_job_count()

    # Ollama is optional — only SAM3 affects overall status
    overall = "ok" if sam3_status == "ok" else "degraded"
    return HealthResponse(
        status=overall,
        sam3=sam3_status,
        ollama=ollama_status,
        active_jobs=active_jobs,
    )

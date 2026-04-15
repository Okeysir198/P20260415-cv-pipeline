"""Job system — state management, per-image processing, and background cleanup."""

from __future__ import annotations

import asyncio
import threading
import time
import uuid

import httpx
from fastapi import HTTPException

from src.config import JOB_TTL_SECONDS, MAX_CONCURRENT_JOBS, logger
from src.geometry import decode_image, strip_data_uri
from src.parsers import parse_labels
from src.sam3 import verify_with_sam3
from src.scoring import generate_sam3_fixes, generate_vlm_fixes, score_image
from src.schemas import (
    QAJobRequest, SAM3Verification, SuggestedFix, ValidationIssue,
)
from src.validators import validate_annotations
from src.vlm import verify_with_vlm

# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_job_semaphore: asyncio.Semaphore | None = None
_shutdown_event = asyncio.Event()


def init_job_state() -> None:
    """Initialize the job semaphore. Must be called inside an async context."""
    global _job_semaphore
    _job_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------


def process_single_image_validate(
    labels: list[str] | list[dict],
    label_format: str,
    classes: dict[int, str],
    cfg: dict,
    img_w: int,
    img_h: int,
) -> dict:
    """Process a single image in validate mode (structural only)."""
    annotations = parse_labels(labels, label_format, img_w, img_h)

    if not labels:
        issues = [ValidationIssue(
            type="empty_label",
            severity="medium",
            annotation_idx=None,
            detail="No labels provided",
        )]
        fixes: list[SuggestedFix] = []
    else:
        issues, fixes = validate_annotations(annotations, classes, cfg)

    score, grade = score_image(len(annotations), issues, None, cfg, class_rules=[])

    annotation_classes = [ann.class_id for ann in annotations]

    return {
        "issues": [i.model_dump() for i in issues],
        "num_annotations": len(annotations),
        "num_issues": len(issues),
        "quality_score": score,
        "grade": grade,
        "suggested_fixes": [f.model_dump() for f in fixes],
        "annotation_classes": annotation_classes,
    }


def _derive_class_ids(class_rules: list[dict] | None) -> set[int]:
    """Extract class IDs produced by overlap/no_overlap rules."""
    if not class_rules:
        return set()
    derived: set[int] = set()
    for rule in class_rules:
        if rule.get("condition") in ("overlap", "no_overlap"):
            output_id = rule.get("output_class_id")
            if output_id is not None:
                derived.add(int(output_id))
    return derived


def process_single_image_verify(
    image_b64: str,
    labels: list[str] | list[dict],
    label_format: str,
    classes: dict[int, str],
    text_prompts: dict[str, str],
    include_missing: bool,
    cfg: dict,
    img_w: int,
    img_h: int,
    class_rules: list[dict],
) -> dict:
    """Process a single image in verify mode (structural + SAM3)."""
    annotations = parse_labels(labels, label_format, img_w, img_h)

    if not labels:
        issues = [ValidationIssue(
            type="empty_label",
            severity="medium",
            annotation_idx=None,
            detail="No labels provided",
        )]
        structural_fixes: list[SuggestedFix] = []
    else:
        issues, structural_fixes = validate_annotations(annotations, classes, cfg)

    # SAM3 verification
    sam3_verification = verify_with_sam3(
        image_b64, annotations, classes, text_prompts, include_missing, cfg, img_w, img_h,
    )

    # Add SAM3 issues
    sam3_issues: list[ValidationIssue] = []
    for idx in sam3_verification.misclassified:
        sam3_issues.append(ValidationIssue(
            type="misclassified",
            severity="high",
            annotation_idx=idx,
            detail=f"SAM3 text verification found no matching class for annotation {idx}",
        ))
    for det in sam3_verification.missing_detections:
        sam3_issues.append(ValidationIssue(
            type="missing_annotation",
            severity="medium",
            annotation_idx=None,
            detail=f"SAM3 auto-mask found unannotated object (area={det.get('area', 0):.4f})",
        ))

    all_issues = issues + sam3_issues

    # Generate SAM3-specific fixes (structural fixes already in structural_fixes)
    sam3_fixes = generate_sam3_fixes(annotations, structural_fixes, sam3_verification)
    all_fixes = structural_fixes + sam3_fixes

    score, grade = score_image(
        len(annotations), all_issues, sam3_verification, cfg, class_rules=class_rules,
    )

    annotation_classes = [ann.class_id for ann in annotations]

    return {
        "issues": [i.model_dump() for i in all_issues],
        "sam3_verification": sam3_verification.model_dump(),
        "num_annotations": len(annotations),
        "num_issues": len(all_issues),
        "quality_score": score,
        "grade": grade,
        "suggested_fixes": [f.model_dump() for f in all_fixes],
        "annotation_classes": annotation_classes,
    }


async def process_single_image_verify_async(
    image_b64: str,
    labels: list[str] | list[dict],
    label_format: str,
    classes: dict[int, str],
    text_prompts: dict[str, str],
    include_missing: bool,
    cfg: dict,
    img_w: int,
    img_h: int,
    class_rules: list[dict],
    vlm_budget: dict,
    enable_vlm: bool = False,
    vlm_trigger: str = "selective",
) -> dict:
    """Process verify with optional async VLM step.

    Runs structural + SAM3 synchronously in an executor, then
    optionally runs async VLM verification in the event loop.

    Args:
        class_rules: Rules that derived the labels (direct/overlap/no_overlap).
        vlm_budget: VLM priority sampling budget.
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        process_single_image_verify,
        image_b64, labels, label_format, classes, text_prompts,
        include_missing, cfg, img_w, img_h, class_rules,
    )

    vlm_verification = None
    if enable_vlm:
        grade = result["grade"]
        should_run = (
            vlm_trigger == "all"
            or (vlm_trigger == "selective" and grade in ("review", "bad"))
            or vlm_trigger == "standalone"
        )
        if should_run:
            parsed = parse_labels(labels, label_format, img_w, img_h)
            ann_dicts = [
                {"class_id": a.class_id, "bbox_norm": a.bbox_norm}
                for a in parsed
            ]
            derived_class_ids = _derive_class_ids(class_rules)
            vlm_verification = await verify_with_vlm(
                image_b64, ann_dicts, classes, cfg, img_w, img_h,
                vlm_budget=vlm_budget,
                derived_class_ids=derived_class_ids or None,
            )
            if vlm_verification and vlm_verification.available:
                sam3_v = SAM3Verification(**result["sam3_verification"])
                issues = [ValidationIssue(**i) for i in result["issues"]]
                score, grade = score_image(
                    len(ann_dicts), issues, sam3_v, cfg, class_rules, vlm_verification,
                )
                vlm_fixes = generate_vlm_fixes(parsed, vlm_verification)
                result["quality_score"] = score
                result["grade"] = grade
                result["suggested_fixes"].extend(
                    [f.model_dump() for f in vlm_fixes]
                )

    result["vlm_verification"] = (
        vlm_verification.model_dump() if vlm_verification else None
    )
    return result


# ---------------------------------------------------------------------------
# Job processing
# ---------------------------------------------------------------------------


async def process_job(job_id: str, request: QAJobRequest) -> None:
    """Process a batch QA job asynchronously."""
    assert _job_semaphore is not None
    async with _job_semaphore:
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is None:
                return
            job["status"] = "running"

        loop = asyncio.get_event_loop()

        for idx, img_input in enumerate(request.images):
            # Check for cancellation
            with _jobs_lock:
                job = _jobs.get(job_id)
                if job is None or job["status"] == "cancelled":
                    return

            try:
                image_b64 = strip_data_uri(img_input.image)
                img = decode_image(image_b64)
                img_w, img_h = img.size

                cfg = request.config

                if request.mode == "validate":
                    result = await loop.run_in_executor(
                        None,
                        process_single_image_validate,
                        img_input.labels,
                        request.label_format,
                        request.classes,
                        cfg,
                        img_w,
                        img_h,
                    )
                else:  # verify
                    result = await process_single_image_verify_async(
                        image_b64,
                        img_input.labels,
                        request.label_format,
                        request.classes,
                        request.text_prompts,
                        request.include_missing_detection,
                        cfg,
                        img_w,
                        img_h,
                        enable_vlm=request.enable_vlm,
                        vlm_trigger=request.vlm_trigger,
                        class_rules=request.class_rules,
                        vlm_budget=request.vlm_budget,
                    )

                result["filename"] = img_input.filename

                with _jobs_lock:
                    job = _jobs.get(job_id)
                    if job is not None:
                        job["results"].append(result)
                        job["processed_images"] = idx + 1

                # Free memory
                img_input.image = ""

            except Exception as exc:
                logger.error("Job %s: failed on image %s: %s", job_id, img_input.filename, exc)
                with _jobs_lock:
                    job = _jobs.get(job_id)
                    if job is not None:
                        job["results"].append({
                            "filename": img_input.filename,
                            "error": str(exc),
                        })
                        job["processed_images"] = idx + 1

        # Mark completed
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


# ---------------------------------------------------------------------------
# Job CRUD
# ---------------------------------------------------------------------------


def create_job(request: QAJobRequest) -> tuple[str, int]:
    """Create a new batch QA job.

    Returns:
        Tuple of (job_id, total_images).
    """
    if not request.images:
        raise HTTPException(status_code=400, detail="No images provided")

    if request.mode not in ("validate", "verify"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown mode '{request.mode}'. Supported: validate, verify",
        )

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
    """Get job state by ID. Raises 404 if not found."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


def list_jobs(status: str | None) -> list[dict]:
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
    """Cancel a running or queued job. Raises 404 if not found."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        if job["status"] in ("completed", "failed", "cancelled"):
            return {"job_id": job_id, "status": job["status"], "message": "Job already finished"}
        job["status"] = "cancelled"
    return {"job_id": job_id, "status": "cancelled"}


def shutdown_jobs() -> None:
    """Cancel all active jobs (called during shutdown)."""
    _shutdown_event.set()
    with _jobs_lock:
        for job in _jobs.values():
            if job["status"] in ("queued", "running"):
                job["status"] = "cancelled"


def get_active_job_count() -> int:
    """Return the number of active (queued or running) jobs."""
    with _jobs_lock:
        return sum(1 for j in _jobs.values() if j["status"] in ("queued", "running"))

"""Functional API annotation QA pipeline with LangGraph @entrypoint / @task.

Uses LangGraph's functional API for a clean, imperative pipeline that supports:

1. **Stratified sampling** -- select representative images from the dataset
2. **Structural validation** -- POST to /validate for per-image checks
3. **SAM3 verification** -- POST to /verify with optional auto_label config
4. **Aggregation** -- compute summary, generate QA report

When ``auto_label_config`` is present in state, the ``/verify`` payload
includes ``detection_classes``, ``class_rules``, and ``vlm_budget`` so the
QA service can use the auto-label pipeline for smarter verification.
"""

import logging
import time
from math import ceil
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from langgraph.func import entrypoint, task

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (same as nodes.py but standalone so tasks are self-contained)
# ---------------------------------------------------------------------------


def _image_to_b64(image_path: Path) -> str:
    """Encode an image file as a base64 PNG string."""
    from PIL import Image

    from utils.yolo_io import pil_to_b64

    image = Image.open(image_path).convert("RGB")
    return pil_to_b64(image)


def _read_label_lines(label_path: Path) -> List[str]:
    """Read YOLO label lines from a file."""
    if not label_path.exists():
        return []
    text = label_path.read_text().strip()
    if not text:
        return []
    return text.splitlines()


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@task
def sample_task(
    data_config: Dict[str, Any],
    qa_config: Dict[str, Any],
    config_dir: str,
    splits: List[str],
) -> Dict[str, List[str]]:
    """Sample images from the dataset for QA checking.

    Uses :class:`StratifiedSampler` to perform class-aware stratified
    sampling across the requested splits.

    Returns:
        Dict mapping split name to list of image path strings.
    """
    from core.p02_annotation_qa.sampler import StratifiedSampler

    sampler_config = dict(qa_config)
    sampling_section = dict(sampler_config.get("sampling", {}))
    sampling_section["splits"] = splits
    sampler_config["sampling"] = sampling_section

    sampler = StratifiedSampler(
        config=sampler_config,
        data_config=data_config,
        config_dir=Path(config_dir),
    )

    sampled = sampler.sample()
    return {split: [str(p) for p in paths] for split, paths in sampled.items()}


@task
def validate_image_task(
    image_path: str,
    split: str,
    class_names: Dict[int, str],
    qa_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Structural validation of a single image via POST /validate.

    Reads the image + YOLO labels from disk, encodes image as base64,
    and calls the QA service's ``/validate`` endpoint.

    Returns:
        Per-image result dict with validation issues, score, grade.
    """
    import httpx

    from utils.yolo_io import image_to_label_path, parse_yolo_label

    img_path = Path(image_path)
    service_cfg = qa_config.get("qa_service", {})
    service_url = service_cfg.get("url", "http://localhost:18105")
    timeout = float(service_cfg.get("timeout", 120))
    validation_config = qa_config.get("validation", {})

    label_path = image_to_label_path(img_path)
    annotations = parse_yolo_label(label_path)
    label_lines = _read_label_lines(label_path)

    # Encode image
    try:
        image_b64 = _image_to_b64(img_path)
    except Exception:
        logger.warning("Failed to read image: %s", img_path)
        return {
            "image_path": str(img_path),
            "split": split,
            "annotations": annotations,
            "validation_issues": [{
                "type": "unreadable_image",
                "severity": "high",
                "annotation_idx": None,
                "detail": f"Could not read image: {img_path.name}",
            }],
            "sam3_verification": {},
            "quality_score": 0.0,
            "grade": "bad",
            "suggested_fixes": [],
            "timing": {"validate_s": 0.0},
        }

    t0 = time.perf_counter()

    payload = {
        "image": image_b64,
        "labels": label_lines,
        "label_format": "yolo",
        "classes": {str(k): v for k, v in class_names.items()},
        "config": validation_config,
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(f"{service_url}/validate", json=payload)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("QA service /validate failed for %s: %s", img_path.name, exc)
        data = {
            "issues": [],
            "quality_score": 0.0,
            "grade": "review",
            "suggested_fixes": [],
        }

    validate_time = time.perf_counter() - t0

    issues = [
        {
            "type": issue.get("type", "unknown"),
            "severity": issue.get("severity", "medium"),
            "annotation_idx": issue.get("annotation_idx"),
            "detail": issue.get("detail", ""),
        }
        for issue in data.get("issues", [])
    ]

    suggested_fixes = [
        {
            "type": fix.get("type", ""),
            "annotation_idx": fix.get("annotation_idx", -1),
            "original": fix.get("original"),
            "suggested": fix.get("suggested"),
            "reason": fix.get("reason", ""),
        }
        for fix in data.get("suggested_fixes", [])
    ]

    return {
        "image_path": str(img_path),
        "split": split,
        "annotations": annotations,
        "validation_issues": issues,
        "sam3_verification": {},
        "quality_score": float(data.get("quality_score", 0.0)),
        "grade": data.get("grade", ""),
        "suggested_fixes": suggested_fixes,
        "timing": {"validate_s": round(validate_time, 4)},
    }


@task
def verify_image_task(
    result: Dict[str, Any],
    class_names: Dict[int, str],
    qa_config: Dict[str, Any],
    auto_label_config: Dict[str, Any],
    data_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """SAM3 verification of a single image via POST /verify.

    When ``auto_label_config`` is provided, includes ``detection_classes``,
    ``class_rules``, and ``vlm_budget`` in the request payload so the QA
    service can leverage the auto-label pipeline for smarter verification.

    Returns:
        Updated per-image result dict with SAM3 verification data.
    """
    import httpx

    img_path = Path(result["image_path"])
    service_cfg = qa_config.get("qa_service", {})
    service_url = service_cfg.get("url", "http://localhost:18105")
    timeout = float(service_cfg.get("timeout", 120))
    validation_config = qa_config.get("validation", {})
    # Prefer feature-local prompts in 05_data.yaml; fall back to shared qa_config (legacy).
    data_config = data_config or {}
    text_prompts: Dict[str, str] = (
        data_config.get("text_prompts")
        or qa_config.get("text_prompts", {})
    )

    if not img_path.exists():
        logger.warning("Image not found for SAM3: %s", img_path)
        return result

    label_path = Path(str(img_path).replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt")
    label_lines = _read_label_lines(label_path)

    try:
        image_b64 = _image_to_b64(img_path)
    except Exception:
        logger.warning("Failed to read image for verify: %s", img_path)
        return result

    t0 = time.perf_counter()

    # Auto-mask "missing detection" finds every object in the scene (not just
    # target classes). For class-restricted datasets (fire/smoke only, helmet-only,
    # etc.) this flags unrelated objects as "missing annotation" and produces
    # catastrophic false-positive rates. Disable via sam3.include_missing_detection=false.
    sam3_cfg = qa_config.get("sam3", {})
    include_missing = bool(sam3_cfg.get("include_missing_detection", True))

    payload: Dict[str, Any] = {
        "image": image_b64,
        "labels": label_lines,
        "label_format": "yolo",
        "classes": {str(k): v for k, v in class_names.items()},
        "text_prompts": text_prompts,
        "include_missing_detection": include_missing,
        "config": validation_config,
    }

    # Include class_rules and vlm_budget for rule-aware scoring and VLM priority sampling
    payload["class_rules"] = auto_label_config.get("class_rules", [])
    vlm_budget = auto_label_config.get("vlm_verify", {}).get("budget")
    if vlm_budget is not None:
        payload["vlm_budget"] = vlm_budget
    else:
        payload["vlm_budget"] = {"sample_rate": 1.0, "max_samples": 10}

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(f"{service_url}/verify", json=payload)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("QA service /verify failed for %s: %s", img_path.name, exc)
        timing = dict(result.get("timing", {}))
        timing["sam3_verify_s"] = round(time.perf_counter() - t0, 4)
        result_copy = dict(result)
        result_copy["timing"] = timing
        return result_copy

    sam3_time = time.perf_counter() - t0

    result_copy = dict(result)

    sam3_v = data.get("sam3_verification", {})
    result_copy["sam3_verification"] = {
        "box_ious": sam3_v.get("box_ious", []),
        "mean_iou": sam3_v.get("mean_box_iou", 0.0),
        "misclassified": sam3_v.get("misclassified", []),
        "missing_masks": sam3_v.get("missing_detections", []),
    }

    result_copy["validation_issues"] = [
        {
            "type": issue.get("type", "unknown"),
            "severity": issue.get("severity", "medium"),
            "annotation_idx": issue.get("annotation_idx"),
            "detail": issue.get("detail", ""),
        }
        for issue in data.get("issues", [])
    ]

    result_copy["quality_score"] = float(data.get("quality_score", 0.0))
    result_copy["grade"] = data.get("grade", "")
    result_copy["suggested_fixes"] = [
        {
            "type": fix.get("type", ""),
            "annotation_idx": fix.get("annotation_idx", -1),
            "original": fix.get("original"),
            "suggested": fix.get("suggested"),
            "reason": fix.get("reason", ""),
        }
        for fix in data.get("suggested_fixes", [])
    ]

    timing = dict(result.get("timing", {}))
    timing["sam3_verify_s"] = round(sam3_time, 4)
    result_copy["timing"] = timing

    return result_copy


@task
def score_results_task(
    results: List[Dict[str, Any]],
    qa_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Score any results that lack a grade (local fallback).

    When the QA service returns scores via /validate or /verify, this is
    a pass-through. Only invokes :class:`QualityScorer` for results where
    the service call failed and no grade was set.

    Returns:
        List of result dicts with scores filled in.
    """
    needs_scoring = any(not r.get("grade") for r in results)
    if not needs_scoring:
        return results

    from core.p02_annotation_qa.scorer import QualityScorer

    scorer = QualityScorer(config=qa_config)
    for r in results:
        if not r.get("grade"):
            scorer.score_image(r)

    return results


@task
def aggregate_results_task(
    image_results: List[Dict[str, Any]],
    dataset_name: str,
    qa_config: Dict[str, Any],
    class_names: Dict[int, str],
    config_dir: str = ".",
) -> Dict[str, Any]:
    """Aggregate per-image results into a dataset-level summary and report.

    Computes grade distribution, average quality score, issue breakdown,
    per-class stats, worst images, auto-fixable count, and timing stats.
    Uses :class:`QAReporter` to persist the report to disk.

    Returns:
        Dict with ``summary`` and ``report_path``.
    """
    from core.p02_annotation_qa.reporter import QAReporter
    from utils.config import generate_run_dir

    reporting_config = qa_config.get("reporting", {})
    worst_n: int = reporting_config.get("worst_images_count", 50)

    # Grade distribution
    grade_counts: Dict[str, int] = {"good": 0, "review": 0, "bad": 0}
    for r in image_results:
        grade = r.get("grade", "review")
        grade_counts[grade] = grade_counts.get(grade, 0) + 1

    # Average quality score
    scores = [r.get("quality_score", 0.0) for r in image_results]
    avg_score = float(np.mean(scores)) if scores else 0.0

    # Issue-type breakdown
    issue_counts: Dict[str, int] = {}
    for r in image_results:
        for issue in r.get("validation_issues", []):
            itype = issue["type"]
            issue_counts[itype] = issue_counts.get(itype, 0) + 1

    # Per-class statistics
    per_class: Dict[str, Dict[str, int]] = {}
    for r in image_results:
        for ann in r.get("annotations", []):
            cls_id = ann[0]
            cname = class_names.get(cls_id, str(cls_id))
            if cname not in per_class:
                per_class[cname] = {"count": 0, "issues": 0}
            per_class[cname]["count"] += 1
        for issue in r.get("validation_issues", []):
            ann_idx = issue.get("annotation_idx")
            if ann_idx is not None and ann_idx < len(r.get("annotations", [])):
                cls_id = r["annotations"][ann_idx][0]
                cname = class_names.get(cls_id, str(cls_id))
                if cname in per_class:
                    per_class[cname]["issues"] += 1

    # Worst N images
    sorted_results = sorted(image_results, key=lambda r: r.get("quality_score", 0.0))
    worst_images = [
        {
            "image_path": r["image_path"],
            "split": r["split"],
            "quality_score": r.get("quality_score", 0.0),
            "grade": r.get("grade", ""),
            "issue_count": len(r.get("validation_issues", [])),
        }
        for r in sorted_results[:worst_n]
    ]

    # Auto-fixable count
    auto_fixable = 0
    for r in image_results:
        for fix in r.get("suggested_fixes", []):
            if fix.get("type") in ("clip_bbox", "remove_duplicate", "remove_degenerate"):
                auto_fixable += 1

    # Timing statistics
    validate_times = [r.get("timing", {}).get("validate_s", 0.0) for r in image_results]
    sam3_times = [r.get("timing", {}).get("sam3_verify_s", 0.0) for r in image_results]
    total_times = [v + s for v, s in zip(validate_times, sam3_times)]
    timing_stats: Dict[str, Any] = {
        "avg_validate_s": round(float(np.mean(validate_times)), 4) if validate_times else 0.0,
        "avg_sam3_verify_s": round(float(np.mean(sam3_times)), 4) if sam3_times else 0.0,
        "avg_total_per_sample_s": round(float(np.mean(total_times)), 4) if total_times else 0.0,
        "max_total_per_sample_s": round(float(np.max(total_times)), 4) if total_times else 0.0,
        "min_total_per_sample_s": round(float(np.min(total_times)), 4) if total_times else 0.0,
    }

    summary: Dict[str, Any] = {
        "dataset": dataset_name,
        "total_checked": len(image_results),
        "grades": grade_counts,
        "avg_quality_score": round(avg_score, 4),
        "issue_breakdown": issue_counts,
        "per_class_stats": per_class,
        "worst_images": worst_images,
        "auto_fixable_count": auto_fixable,
        "timing": timing_stats,
    }

    # Generate report
    from utils.config import feature_name_from_config_path
    feature_name = feature_name_from_config_path(config_dir) if config_dir != "." else dataset_name
    output_dir = str(generate_run_dir(feature_name, "02_annotation_quality"))
    reporter = QAReporter(
        output_dir=output_dir,
        dataset_name=dataset_name,
        config=reporting_config,
    )
    report_path = reporter.generate_report(
        image_results=image_results,
        summary=summary,
    )

    logger.info("QA report saved to %s", report_path)

    return {
        "summary": summary,
        "report_path": str(report_path),
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


@entrypoint()
def qa_pipeline(state: Dict[str, Any]) -> Dict[str, Any]:
    """Run the annotation QA pipeline (functional API).

    When ``auto_label_config`` is present, the /verify payload is enriched
    with ``detection_classes``, ``class_rules``, and ``vlm_budget`` so the
    QA service can leverage the auto-label pipeline for smarter verification.

    Args:
        state: Pipeline configuration dict with keys matching ``QAState``.

    Returns:
        Dict with ``summary`` and ``report_path``.
    """
    data_config = state["data_config"]
    qa_config = state["qa_config"]
    config_dir = state.get("config_dir", ".")
    dataset_name = state.get("dataset_name", "unknown")
    class_names = {int(k): v for k, v in state["class_names"].items()}
    splits = state.get("splits", ["train", "val"])
    batch_size = state.get("batch_size", 32)
    use_sam3 = state.get("use_sam3", True)
    auto_label_config: Dict[str, Any] = state.get("auto_label_config", {})

    # --- Sample ---
    sampled_paths = sample_task(data_config, qa_config, config_dir, splits).result()

    # Flatten into (image_path, split) pairs
    all_pairs: List[tuple] = []
    for split, paths in sampled_paths.items():
        for p in paths:
            all_pairs.append((p, split))

    total_images = len(all_pairs)
    if total_images == 0:
        logger.warning("No images sampled for QA.")
        return {"summary": {"total_checked": 0}, "report_path": ""}

    total_batches = ceil(total_images / batch_size)
    logger.info("Sampled %d images in %d batches", total_images, total_batches)

    # --- Process batches ---
    all_results: List[Dict[str, Any]] = []

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total_images)
        batch_pairs = all_pairs[start:end]

        logger.info("Batch %d/%d (%d images)", batch_idx + 1, total_batches, len(batch_pairs))

        # Validate -- parallel per image
        validate_futures = [
            validate_image_task(img, split, class_names, qa_config)
            for img, split in batch_pairs
        ]
        batch_results = [f.result() for f in validate_futures]

        # SAM3 verify -- parallel per image (if enabled)
        if use_sam3:
            verify_futures = [
                verify_image_task(
                    r, class_names, qa_config, auto_label_config, data_config,
                )
                for r in batch_results
            ]
            batch_results = [f.result() for f in verify_futures]

        # Score (local fallback for any results missing grades)
        batch_results = score_results_task(batch_results, qa_config).result()

        all_results.extend(batch_results)

    # --- Aggregate ---
    result = aggregate_results_task(
        all_results, dataset_name, qa_config, class_names, config_dir,
    ).result()

    return result

"""LangGraph node functions for auto-annotation pipeline.

Six nodes that form the core processing stages:
1. scan_node — discover images needing annotation
2. annotate_batch — annotation via auto-label service (s18104)
3. validate_batch — structural validation of generated annotations
4. nms_filter_batch — NMS filtering of overlapping detections
5. write_batch — write YOLO label files to disk
6. aggregate_node — dataset-level summary and reporting
"""

import sys
import time
from math import ceil
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from loguru import logger
from typing_extensions import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from utils.langgraph_common import (
    get_batch_paths,
    get_batch_range,
    list_append_reducer,
    replace_reducer,
)

# ---------------------------------------------------------------------------
# LangGraph state schema
# ---------------------------------------------------------------------------

class AutoAnnotateState(TypedDict, total=False):
    """Auto-annotation pipeline state."""
    # Config
    data_config: Annotated[dict[str, Any], replace_reducer]
    annotate_config: Annotated[dict[str, Any], replace_reducer]
    dataset_name: Annotated[str, replace_reducer]
    class_names: Annotated[dict[int, str], replace_reducer]
    text_prompts: Annotated[dict[str, str], replace_reducer]
    config_dir: Annotated[str, replace_reducer]
    # Scan
    image_paths: Annotated[dict[str, list[str]], replace_reducer]
    total_images: Annotated[int, replace_reducer]
    current_batch_idx: Annotated[int, replace_reducer]
    total_batches: Annotated[int, replace_reducer]
    batch_size: Annotated[int, replace_reducer]
    # Processing
    image_results: Annotated[list[dict[str, Any]], list_append_reducer]
    mode: Annotated[str, replace_reducer]
    output_format: Annotated[str, replace_reducer]
    dry_run: Annotated[bool, replace_reducer]
    filter_mode: Annotated[str, replace_reducer]
    # Output
    summary: Annotated[dict[str, Any], replace_reducer]
    report_path: Annotated[str, replace_reducer]
    output_dir_override: Annotated[str, replace_reducer]
    # Backup
    backup_dir: Annotated[str, replace_reducer]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Node 1: scan_node
# ---------------------------------------------------------------------------

def scan_node(state: AutoAnnotateState) -> dict:
    """Discover images that need annotation.

    Uses ImageScanner to find images based on filter_mode:
    - "missing": images with no label file or empty label
    - "all": all images in the dataset

    Returns:
        Dict with ``image_paths``, ``total_images``,
        ``total_batches``, ``current_batch_idx``, and ``batch_size``.
    """
    from core.p01_auto_annotate.scanner import ImageScanner

    data_config = state.get("data_config")
    annotate_config = state["annotate_config"]
    config_dir = Path(state.get("config_dir", "."))
    batch_size = annotate_config.get("processing", {}).get("batch_size", 32)
    filter_mode = state.get("filter_mode", "missing")
    splits = annotate_config.get("auto_annotate", {}).get("splits", ["train", "val", "test"])
    image_dir = annotate_config.get("_image_dir")

    scanner = ImageScanner(
        data_config=data_config,
        config_dir=config_dir,
        image_dir=image_dir,
        filter_mode=filter_mode,
        splits=splits,
    )

    image_paths = scanner.scan()
    # Convert Path to str for serialisation
    image_paths_str: dict[str, list[str]] = {
        split: [str(p) for p in paths]
        for split, paths in image_paths.items()
    }
    total_images = sum(len(paths) for paths in image_paths_str.values())
    total_batches = ceil(total_images / batch_size) if total_images > 0 else 0

    logger.info(
        "Scanned %d images across %d splits (%d batches of %d)",
        total_images,
        len(image_paths_str),
        total_batches,
        batch_size,
    )

    return {
        "image_paths": image_paths_str,
        "total_images": total_images,
        "total_batches": total_batches,
        "current_batch_idx": 0,
        "batch_size": batch_size,
    }


# ---------------------------------------------------------------------------
# Node 2: annotate_batch
# ---------------------------------------------------------------------------

def annotate_batch(state: AutoAnnotateState) -> dict:
    """Generate annotations for the current batch via the auto-label service.

    Delegates to Annotator REST client which calls the s18104 auto-label
    service. The service handles SAM3 calls, NMS, and polygon extraction.

    Returns:
        Dict with ``image_results`` extended by this batch's annotations.
    """
    from core.p01_auto_annotate.annotator import Annotator

    annotate_config = state["annotate_config"]
    class_names = {int(k): v for k, v in state["class_names"].items()}
    text_prompts = state.get("text_prompts", {})
    mode = state.get("mode", "text")
    output_format = state.get("output_format", "bbox")

    processing_config = annotate_config.get("processing", {})
    confidence_threshold = processing_config.get("confidence_threshold", 0.5)
    nms_config = annotate_config.get("nms", {})
    nms_iou_threshold = nms_config.get("per_class_iou_threshold", 0.5)

    # Auto-label service connection settings
    service_config = annotate_config.get("auto_label_service", {})
    service_url = service_config.get("url", "http://localhost:18104")
    timeout = service_config.get("timeout", 120)

    annotator = Annotator(
        class_names=class_names,
        text_prompts=text_prompts,
        mode=mode,
        confidence_threshold=confidence_threshold,
        nms_iou_threshold=nms_iou_threshold,
        service_url=service_url,
        timeout=timeout,
    )

    batch_paths = get_batch_paths(state)
    new_results: list[dict[str, Any]] = []

    for split, img_path in batch_paths:
        t_start = time.perf_counter()

        detections = annotator.annotate_image(img_path, output_format=output_format)

        annotate_time = time.perf_counter() - t_start
        result: dict[str, Any] = {
            "image_path": str(img_path),
            "split": split,
            "detections": detections,
            "validation_issues": [],
            "written": False,
            "timing": {"annotate_s": round(annotate_time, 4)},
        }
        new_results.append(result)

    return {"image_results": state.get("image_results", []) + new_results}


# ---------------------------------------------------------------------------
# Node 3: validate_batch
# ---------------------------------------------------------------------------

def validate_batch(state: AutoAnnotateState) -> dict:
    """Structural validation of generated annotations for the current batch.

    Checks: out-of-bounds coordinates, degenerate boxes, extreme aspect ratios.

    Returns:
        Dict with ``image_results`` updated with validation issues.
    """
    start, end = get_batch_range(state)
    image_results = list(state.get("image_results", []))
    class_names = {int(k): v for k, v in state["class_names"].items()}

    for result_idx in range(start, min(end, len(image_results))):
        t_start = time.perf_counter()
        result = image_results[result_idx]
        detections = result.get("detections", [])
        issues: list[dict[str, Any]] = []

        for idx, det in enumerate(detections):
            cx, cy, w, h = det["cx"], det["cy"], det["w"], det["h"]

            # Out-of-bounds check
            for name, val in [("cx", cx), ("cy", cy), ("w", w), ("h", h)]:
                if val < 0.0 or val > 1.0:
                    issues.append({
                        "type": "out_of_bounds",
                        "severity": "high",
                        "detection_idx": idx,
                        "detail": f"{name}={val:.4f} outside [0, 1]",
                    })

            # Degenerate box check — high severity so tiny FPs are removed
            if w < 0.01 or h < 0.01:
                issues.append({
                    "type": "degenerate_box",
                    "severity": "high",
                    "detection_idx": idx,
                    "detail": f"Box too small: w={w:.4f}, h={h:.4f}",
                })

            # Invalid class ID
            if det["class_id"] not in class_names:
                issues.append({
                    "type": "invalid_class",
                    "severity": "high",
                    "detection_idx": idx,
                    "detail": f"Class ID {det['class_id']} not in valid classes",
                })

            # Extreme aspect ratio
            if w > 0 and h > 0:
                aspect = w / h
                if aspect < 0.05 or aspect > 20.0:
                    issues.append({
                        "type": "extreme_aspect_ratio",
                        "severity": "low",
                        "detection_idx": idx,
                        "detail": f"Aspect ratio {aspect:.2f}",
                    })

        validate_time = time.perf_counter() - t_start
        result["validation_issues"] = issues
        timing = result.get("timing", {})
        timing["validate_s"] = round(validate_time, 4)
        result["timing"] = timing

    return {"image_results": image_results}


# ---------------------------------------------------------------------------
# Node 4: nms_filter_batch
# ---------------------------------------------------------------------------

def nms_filter_batch(state: AutoAnnotateState) -> dict:
    """Apply NMS filtering to the current batch.

    Delegates to NMSFilter for per-class and optional cross-class NMS.

    Returns:
        Dict with ``image_results`` updated with filtered detections.
    """
    from core.p01_auto_annotate.nms_filter import NMSFilter

    annotate_config = state["annotate_config"]
    nms_config = annotate_config.get("nms", {})

    nms = NMSFilter(
        per_class_iou_threshold=nms_config.get("per_class_iou_threshold", 0.5),
        cross_class_enabled=nms_config.get("cross_class_enabled", False),
        cross_class_iou_threshold=nms_config.get("cross_class_iou_threshold", 0.8),
    )

    start, end = get_batch_range(state)
    image_results = list(state.get("image_results", []))

    for result_idx in range(start, min(end, len(image_results))):
        result = image_results[result_idx]
        t_start = time.perf_counter()

        detections = result.get("detections", [])
        # Filter out detections flagged with high-severity validation issues
        high_severity_indices = {
            issue["detection_idx"]
            for issue in result.get("validation_issues", [])
            if issue.get("severity") == "high"
        }
        valid_detections = [
            d for i, d in enumerate(detections)
            if i not in high_severity_indices
        ]

        filtered = nms.filter(valid_detections)

        nms_time = time.perf_counter() - t_start
        result["detections"] = filtered
        timing = result.get("timing", {})
        timing["nms_s"] = round(nms_time, 4)
        result["timing"] = timing

    return {"image_results": image_results}


# ---------------------------------------------------------------------------
# Node 5: write_batch
# ---------------------------------------------------------------------------

def write_batch(state: AutoAnnotateState) -> dict:
    """Write YOLO label files for the current batch.

    Delegates to LabelWriter, respects dry_run flag.
    Advances the batch counter.

    Returns:
        Dict with ``image_results`` (updated written flags) and
        ``current_batch_idx`` (incremented by one).
    """
    from core.p01_auto_annotate.writer import LabelWriter

    output_format = state.get("output_format", "bbox")
    dry_run = state.get("dry_run", False)

    batch_idx = state["current_batch_idx"]
    start, end = get_batch_range(state)
    image_results = list(state.get("image_results", []))

    # --- Backup existing labels on first batch ---
    backup_dir_str: str = state.get("backup_dir", "")
    if batch_idx == 0 and not backup_dir_str:
        # Collect all unique labels dirs and count existing label files
        existing_label_files: list[Path] = []
        for result in image_results:
            img_path = Path(result["image_path"])
            label_path = LabelWriter._get_label_path(img_path)
            if label_path.exists():
                existing_label_files.append(label_path)

        if existing_label_files:
            # Derive the labels dir from the first existing label file
            first_label = existing_label_files[0]
            parts = list(first_label.parts)
            labels_dir = None
            for i, part in enumerate(parts):
                if part == "labels":
                    labels_dir = Path(*parts[: i + 1])
                    break
            if labels_dir is None:
                labels_dir = first_label.parent

            backup_dir_path = LabelWriter.create_backup_dir(labels_dir)
            backup_dir_str = str(backup_dir_path)
            logger.info(
                "Backed up %d existing labels to %s",
                len(existing_label_files),
                backup_dir_path,
            )

    backup_dir = Path(backup_dir_str) if backup_dir_str else None

    writer = LabelWriter(
        output_format=output_format,
        dry_run=dry_run,
        backup_dir=backup_dir,
    )

    for result_idx in range(start, min(end, len(image_results))):
        result = image_results[result_idx]
        t_start = time.perf_counter()

        img_path = Path(result["image_path"])
        detections = result.get("detections", [])

        written = writer.write(img_path, detections)

        write_time = time.perf_counter() - t_start
        result["written"] = written
        timing = result.get("timing", {})
        timing["write_s"] = round(write_time, 4)
        result["timing"] = timing

    return {
        "image_results": image_results,
        "current_batch_idx": batch_idx + 1,
        "backup_dir": backup_dir_str,
    }


# ---------------------------------------------------------------------------
# Node 6: aggregate_node
# ---------------------------------------------------------------------------

def aggregate_node(state: AutoAnnotateState) -> dict:
    """Aggregate all per-image results into a dataset-level summary.

    Uses AutoAnnotateReporter to persist the report to disk.

    Returns:
        Dict with ``summary`` and ``report_path``.
    """
    from core.p01_auto_annotate.reporter import AutoAnnotateReporter

    annotate_config = state["annotate_config"]
    class_names = {int(k): v for k, v in state["class_names"].items()}
    image_results: list[dict[str, Any]] = state.get("image_results", [])

    # Compute summary statistics
    total_annotated = sum(1 for r in image_results if r.get("written", False))
    total_detections = sum(len(r.get("detections", [])) for r in image_results)

    # Per-class counts
    per_class: dict[str, int] = {}
    for r in image_results:
        for det in r.get("detections", []):
            cls_id = det.get("class_id", -1)
            cname = class_names.get(cls_id, str(cls_id))
            per_class[cname] = per_class.get(cname, 0) + 1

    # Per-split counts
    per_split: dict[str, int] = {}
    for r in image_results:
        split = r.get("split", "unknown")
        per_split[split] = per_split.get(split, 0) + 1

    # Avg detections per image
    avg_detections = total_detections / len(image_results) if image_results else 0.0

    # Timing stats
    annotate_times = [r.get("timing", {}).get("annotate_s", 0.0) for r in image_results]
    total_times = [
        sum(r.get("timing", {}).values())
        for r in image_results
    ]

    timing_stats: dict[str, Any] = {
        "avg_annotate_s": round(float(np.mean(annotate_times)), 4) if annotate_times else 0.0,
        "avg_total_per_sample_s": round(float(np.mean(total_times)), 4) if total_times else 0.0,
        "max_total_per_sample_s": round(float(np.max(total_times)), 4) if total_times else 0.0,
        "min_total_per_sample_s": round(float(np.min(total_times)), 4) if total_times else 0.0,
    }

    summary: dict[str, Any] = {
        "dataset": state.get("dataset_name", "unknown"),
        "total_images": len(image_results),
        "total_annotated": total_annotated,
        "total_detections": total_detections,
        "avg_detections_per_image": round(avg_detections, 2),
        "per_class_counts": per_class,
        "per_split_counts": per_split,
        "output_format": state.get("output_format", "bbox"),
        "mode": state.get("mode", "text"),
        "dry_run": state.get("dry_run", False),
        "timing": timing_stats,
    }

    # Generate report
    from utils.config import generate_run_dir

    dataset_name = state.get("dataset_name", "unknown")
    override = state.get("output_dir_override")
    if override:
        output_dir = str(override)
    else:
        from utils.config import feature_name_from_config_path
        config_dir = state.get("config_dir", ".")
        feature_name = feature_name_from_config_path(config_dir) if config_dir != "." else dataset_name
        output_dir = str(generate_run_dir(feature_name, "01_auto_annotate"))
    reporting_config = annotate_config.get("reporting", {})

    reporter = AutoAnnotateReporter(
        output_dir=output_dir,
        dataset_name=dataset_name,
        config=reporting_config,
    )
    report_path = reporter.generate_report(
        image_results=image_results,
        summary=summary,
    )

    logger.info("Auto-annotation report saved to %s", report_path)

    return {
        "summary": summary,
        "report_path": str(report_path),
    }

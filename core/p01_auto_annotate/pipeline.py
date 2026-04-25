"""Functional API auto-annotation pipeline with rule-based classification + VLM verification.

Uses LangGraph's ``@entrypoint`` / ``@task`` decorators for a clean,
imperative pipeline that supports:

1. **SAM3 text detection** — detect intermediate objects via auto-label service
2. **Rule-based classification** — IoU overlap rules derive final classes
3. **VLM verification** — optional Qwen3.5 crop check via Ollama

When ``auto_label`` is absent from the data config, this pipeline falls
back to direct text-prompt detection (identical to the StateGraph pipeline).
"""

import time
from math import ceil
from pathlib import Path
from typing import Any

from langgraph.func import entrypoint, task
from loguru import logger


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@task
def scan_task(
    data_config: dict[str, Any] | None,
    annotate_config: dict[str, Any],
    config_dir: str,
    filter_mode: str,
    auto_label_config: dict[str, Any] | None,
) -> dict[str, list[str]]:
    """Discover images needing annotation."""
    from core.p01_auto_annotate.scanner import ImageScanner

    splits = annotate_config.get("auto_annotate", {}).get("splits", ["train", "val", "test"])
    image_dir = annotate_config.get("_image_dir")

    scanner = ImageScanner(
        data_config=data_config,
        config_dir=Path(config_dir),
        image_dir=image_dir,
        filter_mode=filter_mode,
        splits=splits,
    )
    image_paths = scanner.scan()
    return {split: [str(p) for p in paths] for split, paths in image_paths.items()}


@task
def annotate_image_task(
    image_path: str,
    split: str,
    class_names: dict[int, str],
    text_prompts: dict[str, str],
    mode: str,
    output_format: str,
    annotate_config: dict[str, Any],
    auto_label_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Annotate a single image via the auto-label service.

    When ``auto_label_config`` is provided, the service handles rule-based
    classification and optional VLM verification internally.
    """
    from core.p01_auto_annotate.annotator import Annotator

    svc_cfg = annotate_config.get("auto_label_service", {})
    processing = annotate_config.get("processing", {})

    annotator = Annotator(
        class_names=class_names,
        text_prompts=text_prompts,
        mode=mode,
        confidence_threshold=processing.get("confidence_threshold", 0.3),
        nms_iou_threshold=annotate_config.get("nms", {}).get("per_class_iou_threshold", 0.5),
        service_url=svc_cfg.get("url", "http://localhost:18104"),
        timeout=svc_cfg.get("timeout", 120),
        detection_classes=auto_label_config.get("detection_classes") if auto_label_config else None,
        class_rules=auto_label_config.get("class_rules") if auto_label_config else None,
        vlm_verify=auto_label_config.get("vlm_verify") if auto_label_config else None,
    )

    t0 = time.perf_counter()
    detections = annotator.annotate_image(Path(image_path), output_format)
    elapsed = time.perf_counter() - t0

    return {
        "image_path": image_path,
        "split": split,
        "detections": detections,
        "timing": {"annotate_s": round(elapsed, 4)},
    }


@task
def validate_and_nms_task(
    result: dict[str, Any],
    class_names: dict[int, str],
    valid_class_ids: set,
    annotate_config: dict[str, Any],
) -> dict[str, Any]:
    """Structural validation + NMS filtering for one image result."""
    from core.p01_auto_annotate.nms_filter import NMSFilter

    detections = result.get("detections", [])
    nms_cfg = annotate_config.get("nms", {})
    timing = dict(result.get("timing", {}))

    # --- Validate ---
    t0 = time.perf_counter()
    issues: list[dict[str, Any]] = []
    for idx, det in enumerate(detections):
        cx, cy, w, h = det["cx"], det["cy"], det["w"], det["h"]
        for name, val in [("cx", cx), ("cy", cy), ("w", w), ("h", h)]:
            if val < 0.0 or val > 1.0:
                issues.append({"type": "out_of_bounds", "severity": "high", "detection_idx": idx})
        if w < 0.01 or h < 0.01:
            issues.append({"type": "degenerate_box", "severity": "high", "detection_idx": idx})
        if det["class_id"] not in valid_class_ids:
            issues.append({"type": "invalid_class", "severity": "high", "detection_idx": idx})
    timing["validate_s"] = round(time.perf_counter() - t0, 4)

    # --- NMS (remove high-severity issues first) ---
    t0 = time.perf_counter()
    high_severity = {i["detection_idx"] for i in issues if i.get("severity") == "high"}
    clean_dets = [d for idx, d in enumerate(detections) if idx not in high_severity]

    nms_filter = NMSFilter(
        per_class_iou_threshold=nms_cfg.get("per_class_iou_threshold", 0.5),
        cross_class_enabled=nms_cfg.get("cross_class_enabled", True),
        cross_class_iou_threshold=nms_cfg.get("cross_class_iou_threshold", 0.8),
    )
    filtered_dets = nms_filter.filter(clean_dets)
    timing["nms_s"] = round(time.perf_counter() - t0, 4)

    result_copy = dict(result)
    result_copy["detections"] = filtered_dets
    result_copy["validation_issues"] = issues
    result_copy["timing"] = timing
    return result_copy


@task
def write_task(
    results: list[dict[str, Any]],
    output_format: str,
    dry_run: bool,
    backup_dir: str | None,
) -> list[dict[str, Any]]:
    """Write YOLO labels for a batch of results."""
    from core.p01_auto_annotate.writer import LabelWriter

    writer = LabelWriter(
        output_format=output_format,
        dry_run=dry_run,
        backup_dir=Path(backup_dir) if backup_dir else None,
    )

    for result in results:
        t0 = time.perf_counter()
        written = writer.write(Path(result["image_path"]), result.get("detections", []))
        result["written"] = written
        timing = dict(result.get("timing", {}))
        timing["write_s"] = round(time.perf_counter() - t0, 4)
        result["timing"] = timing

    return results


@task
def aggregate_task(
    all_results: list[dict[str, Any]],
    dataset_name: str,
    annotate_config: dict[str, Any],
    class_names: dict[int, str],
    mode: str,
    output_format: str,
    dry_run: bool,
    config_dir: str = ".",
    output_dir_override: str | None = None,
) -> dict[str, Any]:
    """Generate summary report."""
    from core.p01_auto_annotate.reporter import AutoAnnotateReporter
    from utils.config import generate_run_dir

    total_annotated = sum(1 for r in all_results if r.get("detections"))
    total_detections = sum(len(r.get("detections", [])) for r in all_results)
    avg = total_detections / len(all_results) if all_results else 0

    # Per-class counts using final class names
    per_class: dict[str, int] = {}
    for r in all_results:
        for det in r.get("detections", []):
            name = class_names.get(det["class_id"], f"class_{det['class_id']}")
            per_class[name] = per_class.get(name, 0) + 1

    # Timing
    timing_keys = ["annotate_s", "validate_s", "nms_s", "rule_classify_s", "vlm_verify_s", "write_s"]
    timing_stats: dict[str, float] = {}
    for key in timing_keys:
        vals = [r.get("timing", {}).get(key, 0) for r in all_results if r.get("timing", {}).get(key)]
        if vals:
            timing_stats[f"avg_{key}"] = round(sum(vals) / len(vals), 4)
    total_per_sample = [sum(r.get("timing", {}).get(k, 0) for k in timing_keys) for r in all_results]
    if total_per_sample:
        timing_stats["avg_total_per_sample_s"] = round(sum(total_per_sample) / len(total_per_sample), 4)

    summary = {
        "dataset": dataset_name,
        "total_images": len(all_results),
        "total_annotated": total_annotated,
        "total_detections": total_detections,
        "avg_detections_per_image": round(avg, 2),
        "per_class_counts": per_class,
        "output_format": output_format,
        "mode": mode,
        "dry_run": dry_run,
        "timing": timing_stats,
    }

    # Explicit override takes precedence (e.g. --image-dir mode co-locates with data).
    if output_dir_override:
        output_dir = str(output_dir_override)
    else:
        from utils.config import feature_name_from_config_path
        feature_name = feature_name_from_config_path(config_dir) if config_dir != "." else dataset_name
        output_dir = str(generate_run_dir(feature_name, "01_auto_annotate"))
    reporting_config = annotate_config.get("reporting", {})
    reporter = AutoAnnotateReporter(output_dir=output_dir, dataset_name=dataset_name, config=reporting_config)
    report_path = reporter.generate_report(image_results=all_results, summary=summary)

    return {"summary": summary, "report_path": str(report_path)}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


@entrypoint()
def auto_annotate_pipeline(state: dict[str, Any]) -> dict[str, Any]:
    """Run the auto-annotation pipeline (functional API).

    When ``auto_label_config`` is present, the auto-label service (s18104)
    handles rule-based classification and VLM verification internally.
    The pipeline is a thin orchestrator: scan → annotate (service) → validate+NMS → write.

    Args:
        state: Pipeline configuration dict.

    Returns:
        Dict with ``summary`` and ``report_path``.
    """
    data_config = state.get("data_config")
    annotate_config = state["annotate_config"]
    config_dir = state.get("config_dir", ".")
    dataset_name = state.get("dataset_name", "unknown")
    final_class_names = state["class_names"]
    mode = state.get("mode", "text")
    output_format = state.get("output_format", "bbox")
    dry_run = state.get("dry_run", False)
    filter_mode = state.get("filter_mode", "missing")
    batch_size = state.get("batch_size", 32)
    auto_label_config = state.get("auto_label_config")

    # Final class IDs for validation
    final_class_ids = {int(k) for k in final_class_names}
    annotator_class_names = {int(k): v for k, v in final_class_names.items()}
    text_prompts = state.get("text_prompts", {})

    # --- Scan ---
    image_paths = scan_task(data_config, annotate_config, config_dir, filter_mode, auto_label_config).result()

    all_pairs: list[tuple] = []
    for split, paths in image_paths.items():
        for p in paths:
            all_pairs.append((p, split))

    total_images = len(all_pairs)
    if total_images == 0:
        logger.warning("No images found to annotate.")
        return {"summary": {"total_images": 0}, "report_path": ""}

    total_batches = ceil(total_images / batch_size)
    logger.info("Found %d images in %d batches", total_images, total_batches)

    # --- Backup existing labels ---
    backup_dir: str | None = None
    if not dry_run:
        from core.p01_auto_annotate.writer import LabelWriter
        first_img = Path(all_pairs[0][0])
        candidate_label = LabelWriter._get_label_path(first_img)
        if candidate_label.exists():
            parts = list(candidate_label.parts)
            labels_dir = candidate_label.parent
            for i, part in enumerate(parts):
                if part == "labels":
                    labels_dir = Path(*parts[: i + 1])
                    break
            backup_path = LabelWriter.create_backup_dir(labels_dir)
            backup_dir = str(backup_path)
            logger.info("Will back up existing labels to %s", backup_path)

    # --- Process batches ---
    all_results: list[dict[str, Any]] = []

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total_images)
        batch_pairs = all_pairs[start:end]

        logger.info("Batch %d/%d (%d images)", batch_idx + 1, total_batches, len(batch_pairs))

        # Annotate — service handles detection + rule classification + VLM verify
        annotate_futures = [
            annotate_image_task(
                img, split, annotator_class_names, text_prompts,
                mode, output_format, annotate_config, auto_label_config,
            )
            for img, split in batch_pairs
        ]
        batch_results = [f.result() for f in annotate_futures]

        # Validate + NMS (client-side, on final class IDs)
        vn_futures = [
            validate_and_nms_task(r, final_class_names, final_class_ids, annotate_config)
            for r in batch_results
        ]
        batch_results = [f.result() for f in vn_futures]

        # Write labels
        written = write_task(batch_results, output_format, dry_run, backup_dir).result()
        all_results.extend(written)

    # --- Aggregate ---
    result = aggregate_task(
        all_results, dataset_name, annotate_config,
        {int(k): v for k, v in final_class_names.items()},
        mode, output_format, dry_run,
        state.get("config_dir", "."),
        state.get("output_dir_override"),
    ).result()

    return result

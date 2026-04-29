"""LangGraph node functions for generative augmentation pipeline.

Five processing nodes plus an aggregate node:
1. scan_node — discover images containing the source class
2. segment_batch — SAM3 box-prompted segmentation of source objects
3. inpaint_batch — inpaint masked regions with replacement prompts
4. validate_batch — validate inpainted images for quality
5. write_batch — write augmented images and modified labels
6. aggregate_node — dataset-level summary and reporting
"""

import random
import sys
import time
from math import ceil
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from loguru import logger
from PIL import Image
from typing_extensions import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from core.p03_generative_aug.inpainter import Inpainter
from utils.langgraph_common import (
    get_batch_paths,
    get_batch_range,
    list_append_reducer,
    replace_reducer,
)
from utils.yolo_io import image_to_label_path, parse_yolo_label

# ---------------------------------------------------------------------------
# LangGraph state schema
# ---------------------------------------------------------------------------

class GenAugmentState(TypedDict, total=False):
    """Generative augmentation pipeline state."""
    # Config
    data_config: Annotated[dict[str, Any], replace_reducer]
    augment_config: Annotated[dict[str, Any], replace_reducer]
    dataset_name: Annotated[str, replace_reducer]
    class_names: Annotated[dict[int, str], replace_reducer]
    config_dir: Annotated[str, replace_reducer]
    # Source / target
    source_class_id: Annotated[int, replace_reducer]
    target_class_id: Annotated[int, replace_reducer]
    replacement_prompts: Annotated[list[str], replace_reducer]
    # Scan
    image_paths: Annotated[dict[str, list[str]], replace_reducer]
    total_images: Annotated[int, replace_reducer]
    current_batch_idx: Annotated[int, replace_reducer]
    total_batches: Annotated[int, replace_reducer]
    batch_size: Annotated[int, replace_reducer]
    # Processing
    image_results: Annotated[list[dict[str, Any]], list_append_reducer]
    dry_run: Annotated[bool, replace_reducer]
    # Output
    summary: Annotated[dict[str, Any], replace_reducer]
    report_path: Annotated[str, replace_reducer]
    output_dir: Annotated[str, replace_reducer]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _yolo_bbox_to_xyxy(
    cx: float, cy: float, w: float, h: float, img_w: int, img_h: int
) -> list[int]:
    """Convert normalised YOLO bbox to absolute ``[x1, y1, x2, y2]``.

    Args:
        cx, cy, w, h: Normalised YOLO bbox coordinates.
        img_w, img_h: Image dimensions in pixels.

    Returns:
        ``[x1, y1, x2, y2]`` in absolute pixels.
    """
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return [max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)]


# ---------------------------------------------------------------------------
# Node 1: scan_node
# ---------------------------------------------------------------------------

def scan_node(state: GenAugmentState) -> dict:
    """Discover images containing the source class.

    Uses ImageScanner from step00_auto_annotate with filter_mode="all",
    then filters to keep only images whose YOLO label contains the
    source_class_id.

    Returns:
        Dict with ``image_paths``, ``total_images``,
        ``total_batches``, ``current_batch_idx``, and ``batch_size``.
    """
    from core.p01_auto_annotate.scanner import ImageScanner

    data_config = state.get("data_config")
    augment_config = state["augment_config"]
    config_dir = Path(state.get("config_dir", "."))
    batch_size = augment_config.get("processing", {}).get("batch_size", 16)
    splits = augment_config.get("generative_augment", {}).get("splits", ["train", "val", "test"])
    source_class_id = state["source_class_id"]

    scanner = ImageScanner(
        data_config=data_config,
        config_dir=config_dir,
        filter_mode="all",
        splits=splits,
    )

    raw_image_paths = scanner.scan()

    # Filter: keep only images whose label contains the source class
    filtered_paths: dict[str, list[str]] = {}
    for split, paths in raw_image_paths.items():
        kept: list[str] = []
        for p in paths:
            label_path = image_to_label_path(p)
            annotations = parse_yolo_label(label_path)
            if any(cls_id == source_class_id for cls_id, *_ in annotations):
                kept.append(str(p))
        filtered_paths[split] = kept

    total_images = sum(len(paths) for paths in filtered_paths.values())
    total_batches = ceil(total_images / batch_size) if total_images > 0 else 0

    logger.info(
        "Scanned and filtered %d images with source class %d across %d splits (%d batches of %d)",
        total_images,
        source_class_id,
        len(filtered_paths),
        total_batches,
        batch_size,
    )

    return {
        "image_paths": filtered_paths,
        "total_images": total_images,
        "total_batches": total_batches,
        "current_batch_idx": 0,
        "batch_size": batch_size,
    }


# ---------------------------------------------------------------------------
# Node 2: segment_batch
# ---------------------------------------------------------------------------

def segment_batch(state: GenAugmentState) -> dict:
    """Segment source-class objects in the current batch using SAM3.

    For each image, parses its YOLO label to find bboxes of source_class_id,
    then uses SAM3Client.segment_with_box() to get a precise mask for each.

    Returns:
        Dict with ``image_results`` extended by this batch's segmentations.
    """
    from core.p02_annotation_qa.sam3_client import SAM3Client

    augment_config = state["augment_config"]
    sam3_config = augment_config.get("sam3", {})
    source_class_id = state["source_class_id"]

    sam3 = SAM3Client(
        service_url=sam3_config.get("service_url", "http://localhost:18100"),
        timeout=sam3_config.get("timeout", 120),
    )

    batch_paths = get_batch_paths(state)
    new_results: list[dict[str, Any]] = []

    for split, img_path in batch_paths:
        t_start = time.perf_counter()

        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        label_path = image_to_label_path(img_path)
        annotations = parse_yolo_label(label_path)

        objects: list[dict[str, Any]] = []
        for cls_id, cx, cy, w, h in annotations:
            if cls_id != source_class_id:
                continue

            box_xyxy = _yolo_bbox_to_xyxy(cx, cy, w, h, img_w, img_h)

            try:
                seg_result = sam3.segment_with_box(image, box_xyxy)
                mask = seg_result["mask"]
                iou_score = seg_result["iou_score"]
            except Exception as e:
                logger.warning(
                    "SAM3 segmentation failed for %s bbox [%s]: %s",
                    img_path.name, box_xyxy, e,
                )
                mask = None
                iou_score = 0.0

            objects.append({
                "bbox_yolo": (cx, cy, w, h),
                "bbox_xyxy": box_xyxy,
                "mask": mask,
                "iou_score": iou_score,
                "original_class_id": cls_id,
            })

        segment_time = time.perf_counter() - t_start
        result: dict[str, Any] = {
            "image_path": str(img_path),
            "split": split,
            "objects": objects,
            "all_annotations": annotations,
            "image_size": (img_w, img_h),
            "inpainted_images": [],
            "validation_status": [],
            "written": False,
            "timing": {"segment_s": round(segment_time, 4)},
        }
        new_results.append(result)

    return {"image_results": state.get("image_results", []) + new_results}


# ---------------------------------------------------------------------------
# Node 3: inpaint_batch
# ---------------------------------------------------------------------------

def inpaint_batch(state: GenAugmentState) -> dict:
    """Inpaint masked regions with replacement prompts.

    For each source-class object in the current batch, calls an inpainting
    pipeline with the SAM3 mask and a randomly chosen replacement prompt.
    If dry_run is True, skips actual inpainting but records what would happen.

    Returns:
        Dict with ``image_results`` updated with inpainted images.
    """
    dry_run = state.get("dry_run", False)
    replacement_prompts = state.get("replacement_prompts", ["object"])
    augment_config = state["augment_config"]

    start, end = get_batch_range(state)
    image_results = list(state.get("image_results", []))

    inpaint_pipeline = None
    if not dry_run:
        inpaint_pipeline = Inpainter(augment_config)

    for result_idx in range(start, min(end, len(image_results))):
        result = image_results[result_idx]
        t_start = time.perf_counter()

        img_path = Path(result["image_path"])
        objects = result.get("objects", [])

        inpainted_images: list[dict[str, Any]] = []

        if dry_run:
            # Record what would happen without actually inpainting
            for obj_idx, obj in enumerate(objects):
                if obj.get("mask") is None:
                    continue
                prompt = random.choice(replacement_prompts)
                inpainted_images.append({
                    "object_idx": obj_idx,
                    "prompt": prompt,
                    "image": None,
                    "dry_run": True,
                })
        else:
            image = Image.open(img_path).convert("RGB")
            for obj_idx, obj in enumerate(objects):
                mask = obj.get("mask")
                if mask is None:
                    continue

                prompt = random.choice(replacement_prompts)

                try:
                    variants = inpaint_pipeline.inpaint(
                        image=image,
                        mask=mask,
                        prompt=prompt,
                    )
                    for v_img in variants:
                        inpainted_images.append({
                            "object_idx": obj_idx,
                            "prompt": prompt,
                            "image": v_img,
                            "dry_run": False,
                        })
                except Exception as e:
                    logger.warning(
                        "Inpainting failed for %s object %d: %s",
                        img_path.name, obj_idx, e,
                    )

        inpaint_time = time.perf_counter() - t_start
        result["inpainted_images"] = inpainted_images
        timing = result.get("timing", {})
        timing["inpaint_s"] = round(inpaint_time, 4)
        result["timing"] = timing

    return {"image_results": image_results}


# ---------------------------------------------------------------------------
# Node 4: validate_batch
# ---------------------------------------------------------------------------

def validate_batch(state: GenAugmentState) -> dict:
    """Validate inpainted images for quality.

    Checks:
    - Image is not blank/corrupt (not None or empty)
    - Pixel distribution is reasonable (not all same colour)
    - Image dimensions match original

    Returns:
        Dict with ``image_results`` updated with validation status.
    """
    start, end = get_batch_range(state)
    image_results = list(state.get("image_results", []))

    for result_idx in range(start, min(end, len(image_results))):
        t_start = time.perf_counter()
        result = image_results[result_idx]
        inpainted_images = result.get("inpainted_images", [])
        original_size = result.get("image_size", (0, 0))
        validation_statuses: list[dict[str, Any]] = []

        for inpaint_entry in inpainted_images:
            status: dict[str, Any] = {
                "object_idx": inpaint_entry.get("object_idx"),
                "valid": True,
                "issues": [],
            }

            if inpaint_entry.get("dry_run", False):
                status["valid"] = True
                status["issues"].append("dry_run: validation skipped")
                validation_statuses.append(status)
                continue

            img = inpaint_entry.get("image")

            # Check not blank/corrupt
            if img is None:
                status["valid"] = False
                status["issues"].append("image is None")
                validation_statuses.append(status)
                continue

            try:
                img_array = np.asarray(img)
            except Exception:
                status["valid"] = False
                status["issues"].append("cannot convert image to array")
                validation_statuses.append(status)
                continue

            # Check not empty
            if img_array.size == 0:
                status["valid"] = False
                status["issues"].append("image array is empty")
                validation_statuses.append(status)
                continue

            # Check dimensions match original
            orig_w, orig_h = original_size
            if img_array.ndim >= 2:
                h, w = img_array.shape[:2]
                if orig_w > 0 and orig_h > 0 and (w != orig_w or h != orig_h):
                    status["valid"] = False
                    status["issues"].append(
                        f"dimension mismatch: got ({w}, {h}), expected ({orig_w}, {orig_h})"
                    )

            # Check pixel distribution (not all same colour)
            if img_array.ndim >= 2:
                std = float(np.std(img_array))
                if std < 1.0:
                    status["valid"] = False
                    status["issues"].append(
                        f"pixel std too low ({std:.2f}), likely uniform colour"
                    )

            validation_statuses.append(status)

        validate_time = time.perf_counter() - t_start
        result["validation_status"] = validation_statuses
        timing = result.get("timing", {})
        timing["validate_s"] = round(validate_time, 4)
        result["timing"] = timing

    return {"image_results": image_results}


# ---------------------------------------------------------------------------
# Node 5: write_batch
# ---------------------------------------------------------------------------

def write_batch(state: GenAugmentState) -> dict:
    """Write augmented images and modified YOLO labels.

    Saves augmented images with naming ``{original_stem}_aug{variant_idx}.{ext}``.
    Creates modified YOLO labels where the inpainted object's class_id is changed
    from source_class_id to target_class_id (same bbox).

    If dry_run is True, skips writing but logs what would be written.
    Advances the batch counter.

    Returns:
        Dict with ``image_results`` (updated written flags) and
        ``current_batch_idx`` (incremented by one).
    """
    dry_run = state.get("dry_run", False)
    source_class_id = state["source_class_id"]
    target_class_id = state["target_class_id"]
    output_dir_str = state["output_dir"]
    output_dir = Path(output_dir_str)

    batch_idx = state["current_batch_idx"]
    start, end = get_batch_range(state)
    image_results = list(state.get("image_results", []))

    for result_idx in range(start, min(end, len(image_results))):
        result = image_results[result_idx]
        t_start = time.perf_counter()

        img_path = Path(result["image_path"])
        split = result.get("split", "default")
        all_annotations = result.get("all_annotations", [])
        inpainted_images = result.get("inpainted_images", [])
        validation_status = result.get("validation_status", [])

        written_count = 0

        for variant_idx, inpaint_entry in enumerate(inpainted_images):
            obj_idx = inpaint_entry.get("object_idx", 0)

            # Check validation
            if variant_idx < len(validation_status):
                vstatus = validation_status[variant_idx]
                if not vstatus.get("valid", True) and not inpaint_entry.get("dry_run", False):
                    logger.info(
                        "Skipping invalid inpainted image for %s variant %d: %s",
                        img_path.name, variant_idx, vstatus.get("issues"),
                    )
                    continue

            # Build output paths
            split_dir = output_dir / split
            images_dir = split_dir / "images"
            labels_dir = split_dir / "labels"

            aug_stem = f"{img_path.stem}_aug{variant_idx}"
            aug_img_path = images_dir / f"{aug_stem}{img_path.suffix}"
            aug_label_path = labels_dir / f"{aug_stem}.txt"

            if dry_run:
                logger.info(
                    "[DRY RUN] Would write: %s (prompt: %s)",
                    aug_img_path, inpaint_entry.get("prompt", "N/A"),
                )
                written_count += 1
                continue

            # Create directories
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            # Save augmented image
            inpainted_img = inpaint_entry.get("image")
            if inpainted_img is not None:
                if hasattr(inpainted_img, "save"):
                    inpainted_img.save(str(aug_img_path))
                else:
                    Image.fromarray(np.asarray(inpainted_img)).save(str(aug_img_path))

            # Build modified label: copy all annotations, change source -> target
            # for the specific object that was inpainted
            label_lines: list[str] = []
            source_obj_count = 0
            for cls_id, cx, cy, w, h in all_annotations:
                if cls_id == source_class_id:
                    if source_obj_count == obj_idx:
                        # This is the inpainted object: change class
                        label_lines.append(
                            f"{target_class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                        )
                    else:
                        label_lines.append(
                            f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                        )
                    source_obj_count += 1
                else:
                    label_lines.append(
                        f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                    )

            aug_label_path.write_text("\n".join(label_lines) + "\n")
            written_count += 1

        write_time = time.perf_counter() - t_start
        result["written"] = written_count > 0
        result["written_count"] = written_count
        timing = result.get("timing", {})
        timing["write_s"] = round(write_time, 4)
        result["timing"] = timing

    return {
        "image_results": image_results,
        "current_batch_idx": batch_idx + 1,
    }


# ---------------------------------------------------------------------------
# Node 6: aggregate_node
# ---------------------------------------------------------------------------

def aggregate_node(state: GenAugmentState) -> dict:
    """Aggregate all per-image results into a dataset-level summary.

    Summarises: total processed, total written, per-prompt counts,
    timing statistics.

    Returns:
        Dict with ``summary`` and ``report_path``.
    """
    class_names = {int(k): v for k, v in state["class_names"].items()}
    image_results: list[dict[str, Any]] = state.get("image_results", [])
    source_class_id = state["source_class_id"]
    target_class_id = state["target_class_id"]

    total_processed = len(image_results)
    total_written = sum(r.get("written_count", 0) for r in image_results)
    total_objects_segmented = sum(
        len(r.get("objects", [])) for r in image_results
    )

    # Per-prompt counts
    per_prompt: dict[str, int] = {}
    for r in image_results:
        for inpaint_entry in r.get("inpainted_images", []):
            prompt = inpaint_entry.get("prompt", "unknown")
            per_prompt[prompt] = per_prompt.get(prompt, 0) + 1

    # Per-split counts
    per_split: dict[str, int] = {}
    for r in image_results:
        split = r.get("split", "unknown")
        per_split[split] = per_split.get(split, 0) + 1

    # Timing stats
    segment_times = [r.get("timing", {}).get("segment_s", 0.0) for r in image_results]
    inpaint_times = [r.get("timing", {}).get("inpaint_s", 0.0) for r in image_results]
    total_times = [sum(r.get("timing", {}).values()) for r in image_results]

    timing_stats: dict[str, Any] = {
        "avg_segment_s": round(float(np.mean(segment_times)), 4) if segment_times else 0.0,
        "avg_inpaint_s": round(float(np.mean(inpaint_times)), 4) if inpaint_times else 0.0,
        "avg_total_per_sample_s": round(float(np.mean(total_times)), 4) if total_times else 0.0,
        "max_total_per_sample_s": round(float(np.max(total_times)), 4) if total_times else 0.0,
        "min_total_per_sample_s": round(float(np.min(total_times)), 4) if total_times else 0.0,
    }

    source_name = class_names.get(source_class_id, str(source_class_id))
    target_name = class_names.get(target_class_id, str(target_class_id))

    summary: dict[str, Any] = {
        "dataset": state.get("dataset_name", "unknown"),
        "source_class": f"{source_class_id} ({source_name})",
        "target_class": f"{target_class_id} ({target_name})",
        "total_images_processed": total_processed,
        "total_objects_segmented": total_objects_segmented,
        "total_augmented_written": total_written,
        "per_prompt_counts": per_prompt,
        "per_split_counts": per_split,
        "dry_run": state.get("dry_run", False),
        "timing": timing_stats,
    }

    # Write summary report to disk
    output_dir_str = state["output_dir"]
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = state.get("dataset_name", "unknown")
    report_path = output_dir / f"{dataset_name}_augment_report.txt"

    lines = [
        "=" * 70,
        f"  Generative Augmentation Report: {dataset_name}",
        "=" * 70,
        f"  Source class     : {summary['source_class']}",
        f"  Target class     : {summary['target_class']}",
        f"  Images processed : {total_processed}",
        f"  Objects segmented: {total_objects_segmented}",
        f"  Augmented written: {total_written}",
        f"  Dry run          : {summary['dry_run']}",
        "",
        "  Per-split:",
    ]
    for split_name, count in sorted(per_split.items()):
        lines.append(f"    {split_name:<20}: {count}")

    lines.append("")
    lines.append("  Per-prompt:")
    for prompt, count in sorted(per_prompt.items(), key=lambda x: -x[1]):
        lines.append(f"    {prompt:<40}: {count}")

    lines.append("")
    lines.append("  Timing:")
    for tkey, tval in timing_stats.items():
        lines.append(f"    {tkey:<30}: {tval:.4f}s")

    lines.append("=" * 70)
    report_text = "\n".join(lines) + "\n"
    report_path.write_text(report_text)

    logger.info("Generative augmentation report saved to %s", report_path)

    return {
        "summary": summary,
        "report_path": str(report_path),
    }

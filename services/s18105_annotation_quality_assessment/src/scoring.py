"""Scoring, fix generation, fix application, label conversion, and report aggregation."""

from __future__ import annotations

import numpy as np

from src.config import DEFAULT_SCORING_WEIGHTS, DEFAULT_SCORING_THRESHOLDS, DEFAULT_WORST_IMAGES_COUNT
from src.parsers import parse_labels
from src.schemas import (
    FixResponse, ParsedAnnotation, ReportResponse,
    SAM3Verification, SuggestedFix, ValidationIssue,
    VLMVerification,
)


def _compute_derived_class_ids(class_rules: list[dict]) -> set[int]:
    """Extract class IDs that are derived via overlap/no_overlap rules.

    Args:
        class_rules: List of rule dicts, each with ``condition`` and ``output_class_id``.

    Returns:
        Set of class IDs produced by non-direct rules.
    """
    if not class_rules:
        return set()
    derived: set[int] = set()
    for rule in class_rules:
        rule_type = rule.get("condition", "direct")
        if rule_type in ("overlap", "no_overlap"):
            output_id = rule.get("output_class_id")
            if output_id is not None:
                derived.add(int(output_id))
    return derived


def score_image(
    num_annotations: int,
    issues: list[ValidationIssue],
    sam3_verification: SAM3Verification | None,
    cfg: dict,
    class_rules: list[dict],
    vlm_verification: VLMVerification | None = None,
) -> tuple[float, str]:
    """Compute quality score and grade for an image.

    Score formula:
        score = 1.0 - (w_structural * structural_penalty
                       + w_bbox * bbox_penalty
                       + w_classification * classification_penalty
                       + w_coverage * coverage_penalty
                       + w_vlm * vlm_penalty)

    When *class_rules* is provided, structural issues on derived classes
    (overlap/no_overlap) receive a 0.5x weight reduction since those labels
    are rule-inferred and expected to carry some uncertainty.

    Args:
        num_annotations: Number of annotations.
        issues: Validation issues found.
        sam3_verification: SAM3 verification results (or None if validate-only).
        cfg: Config overrides with scoring weights and thresholds.
        vlm_verification: VLM verification results (or None if not used).
        class_rules: Optional rules that derived the labels.

    Returns:
        Tuple of (score, grade) where score is in [0, 1] and grade is 'good'|'review'|'bad'.
    """
    weights = cfg.get("scoring", {}).get("weights", DEFAULT_SCORING_WEIGHTS)
    thresholds = cfg.get("scoring", {}).get("thresholds", DEFAULT_SCORING_THRESHOLDS)

    num_ann = max(num_annotations, 1)

    derived_ids = _compute_derived_class_ids(class_rules)

    # Structural penalty — apply 0.5x weight for issues on derived classes
    if derived_ids:
        effective_issue_count = 0.0
        for issue in issues:
            # Issues with annotation_idx carry class context through the idx
            # but we don't have class info directly; count all structural issues
            # at reduced weight when derived classes are present.
            # Only reduce weight for structural issue types (not SAM3/VLM issues).
            structural_types = {
                "out_of_bounds", "invalid_class", "degenerate_box", "large_box",
                "duplicate_annotation", "extreme_aspect_ratio", "polygon_too_few_vertices",
                "polygon_self_intersection", "polygon_bbox_mismatch", "polygon_out_of_bounds",
            }
            if issue.type in structural_types and issue.annotation_idx is not None:
                # Apply 0.5x weight for derived-class annotations
                effective_issue_count += 0.5
            else:
                effective_issue_count += 1.0
        structural_penalty = effective_issue_count / num_ann
    else:
        structural_penalty = len(issues) / num_ann

    # Bbox quality penalty (from SAM3)
    bbox_penalty = 0.0
    if sam3_verification:
        if sam3_verification.mask_ious:
            bbox_penalty = 1.0 - (float(np.mean(sam3_verification.mask_ious)) if sam3_verification.mask_ious else 0.0)
        elif sam3_verification.box_ious:
            bbox_penalty = 1.0 - (float(np.mean(sam3_verification.box_ious)) if sam3_verification.box_ious else 0.0)

    # Classification penalty (from SAM3)
    classification_penalty = 0.0
    if sam3_verification:
        classification_penalty = len(sam3_verification.misclassified) / num_ann

    # Coverage penalty (from SAM3)
    coverage_penalty = 0.0
    if sam3_verification:
        num_missing = len(sam3_verification.missing_detections)
        denominator = max(num_annotations + num_missing, 1)
        coverage_penalty = num_missing / denominator

    # VLM penalty
    vlm_penalty = 0.0
    if vlm_verification and vlm_verification.available:
        crop_v = vlm_verification.crop_verification
        scene_v = vlm_verification.scene_verification
        crop_penalty = crop_v.num_incorrect / max(crop_v.num_checked, 1)
        scene_penalty = 1.0 - scene_v.quality_score
        vlm_weights = cfg.get("vlm", {})
        crop_w = vlm_weights.get("crop_weight", 0.6)
        scene_w = vlm_weights.get("scene_weight", 0.4)
        vlm_penalty = crop_w * crop_penalty + scene_w * scene_penalty

    # Weighted score
    score = 1.0
    score -= weights.get("structural", 0.3) * structural_penalty
    score -= weights.get("bbox_quality", 0.3) * bbox_penalty
    score -= weights.get("classification", 0.2) * classification_penalty
    score -= weights.get("coverage", 0.2) * coverage_penalty
    score -= weights.get("vlm", 0.0) * vlm_penalty
    score = max(0.0, min(1.0, score))

    # Grade
    good_thresh = thresholds.get("good", 0.8)
    review_thresh = thresholds.get("review", 0.5)
    if score >= good_thresh:
        grade = "good"
    elif score >= review_thresh:
        grade = "review"
    else:
        grade = "bad"

    return round(score, 4), grade


def generate_sam3_fixes(
    annotations: list[ParsedAnnotation],
    structural_fixes: list[SuggestedFix],
    sam3_verification: SAM3Verification,
) -> list[SuggestedFix]:
    """Generate SAM3-based fix suggestions.

    Structural fixes (clip_bbox, remove_duplicate, remove_degenerate) are already
    produced by validate_annotations() — this function only adds SAM3-specific fixes.

    Fix types:
        - tighten_bbox: SAM3 suggests tighter bbox.
        - remove_annotation: Misclassified annotation removed.

    Args:
        annotations: Parsed annotations.
        structural_fixes: Fixes already produced by validate_annotations().
        sam3_verification: SAM3 verification data.

    Returns:
        List of SAM3-specific suggested fixes.
    """
    fixes: list[SuggestedFix] = []

    # Misclassified annotations
    for idx in sam3_verification.misclassified:
        if 0 <= idx < len(annotations):
            ann = annotations[idx]
            fixes.append(SuggestedFix(
                type="remove_annotation",
                annotation_idx=idx,
                original={"class_id": ann.class_id, "bbox_norm": ann.bbox_norm},
                suggested=None,
                reason="SAM3 verification: likely misclassified.",
            ))

    # Build set of indices already marked for removal by structural fixes
    removed_indices = {
        f.annotation_idx
        for f in structural_fixes
        if f.type in ("remove_annotation", "remove_degenerate", "remove_duplicate")
    } | {idx for idx in sam3_verification.misclassified}

    # Low box IoU — suggest tightening
    for idx, iou_val in enumerate(sam3_verification.box_ious):
        if iou_val < 0.6 and 0 <= idx < len(annotations) and idx not in removed_indices:
            ann = annotations[idx]
            fixes.append(SuggestedFix(
                type="tighten_bbox",
                annotation_idx=idx,
                original={"class_id": ann.class_id, "bbox_norm": ann.bbox_norm},
                suggested=None,
                reason=f"SAM3 suggests tighter bbox (IoU={iou_val:.2f}).",
            ))

    return fixes


def generate_vlm_fixes(
    annotations: list[ParsedAnnotation],
    vlm_verification: VLMVerification,
) -> list[SuggestedFix]:
    """Generate VLM-based fix suggestions.

    Fix types:
        - vlm_flagged: VLM disagrees with annotation class.
    """
    fixes: list[SuggestedFix] = []

    # Crop-level flags
    for result in vlm_verification.crop_verification.results:
        if not result.is_correct and 0 <= result.annotation_idx < len(annotations):
            ann = annotations[result.annotation_idx]
            fixes.append(SuggestedFix(
                type="vlm_flagged",
                annotation_idx=result.annotation_idx,
                original={"class_id": ann.class_id, "bbox_norm": ann.bbox_norm},
                suggested=None,
                reason=f"VLM: not a {result.class_name} (conf={result.confidence:.2f}, {result.reason})",
            ))

    # Scene-level flags
    flagged_by_crop = {f.annotation_idx for f in fixes}
    for idx in vlm_verification.scene_verification.incorrect_indices:
        if 0 <= idx < len(annotations) and idx not in flagged_by_crop:
            ann = annotations[idx]
            fixes.append(SuggestedFix(
                type="vlm_flagged",
                annotation_idx=idx,
                original={"class_id": ann.class_id, "bbox_norm": ann.bbox_norm},
                suggested=None,
                reason="VLM scene check: annotation likely incorrect.",
            ))

    return fixes


def annotations_to_labels(
    annotations: list[ParsedAnnotation],
    label_format: str,
    img_w: int,
    img_h: int,
) -> list[str] | list[dict]:
    """Convert ParsedAnnotation list back to the original label format.

    Args:
        annotations: List of corrected annotations.
        label_format: Target format.
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        Labels in the original format.
    """
    fmt = label_format.lower().strip()
    if fmt == "yolo":
        lines: list[str] = []
        for ann in annotations:
            cx, cy, w, h = ann.bbox_norm
            lines.append(f"{ann.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        return lines
    elif fmt == "yolo_seg":
        lines = []
        for ann in annotations:
            if ann.polygon_norm:
                coords = " ".join(f"{v:.6f}" for v in ann.polygon_norm)
                lines.append(f"{ann.class_id} {coords}")
            else:
                # Fall back to bbox corners
                cx, cy, w, h = ann.bbox_norm
                x1, y1 = cx - w / 2, cy - h / 2
                x2, y2 = cx + w / 2, cy + h / 2
                coords = f"{x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}"
                lines.append(f"{ann.class_id} {coords}")
        return lines
    elif fmt == "coco":
        results: list[dict] = []
        for ann in annotations:
            cx, cy, w, h = ann.bbox_norm
            # Convert normalized cxcywh to pixel xywh
            px_x = (cx - w / 2) * img_w
            px_y = (cy - h / 2) * img_h
            px_w = w * img_w
            px_h = h * img_h

            coco_ann: dict = {
                "category_id": ann.class_id,
                "bbox": [round(px_x, 2), round(px_y, 2), round(px_w, 2), round(px_h, 2)],
            }

            if ann.polygon_norm:
                # Convert normalized polygon back to pixel polygon
                pixel_poly: list[float] = []
                for i in range(0, len(ann.polygon_norm) - 1, 2):
                    pixel_poly.append(round(ann.polygon_norm[i] * img_w, 2))
                    pixel_poly.append(round(ann.polygon_norm[i + 1] * img_h, 2))
                coco_ann["segmentation"] = [pixel_poly]

            results.append(coco_ann)
        return results
    else:
        return []


def apply_fixes(
    labels: list[str] | list[dict],
    label_format: str,
    suggested_fixes: list[dict],
    auto_apply: list[str],
    img_w: int,
    img_h: int,
) -> FixResponse:
    """Apply fixes to labels and return corrected version.

    Args:
        labels: Original labels.
        label_format: Label format.
        suggested_fixes: Fix suggestions (dicts).
        auto_apply: List of fix types to auto-apply.
        img_w: Image width.
        img_h: Image height.

    Returns:
        FixResponse with corrected labels and applied/needs_review lists.
    """
    # Parse original labels
    annotations = parse_labels(labels, label_format, img_w, img_h)
    num_before = len(annotations)

    applied_fixes: list[dict] = []
    needs_review: list[dict] = []
    indices_to_remove: set[int] = set()
    clip_map: dict[int, list[float]] = {}  # idx -> clipped bbox_norm

    # Categorize fixes
    for fix in suggested_fixes:
        fix_type = fix.get("type", "")
        idx = fix.get("annotation_idx", -1)

        if fix_type in auto_apply:
            if fix_type == "clip_bbox" and fix.get("suggested"):
                clip_map[idx] = fix["suggested"].get("bbox_norm", [])
                applied_fixes.append(fix)
            elif fix_type in ("remove_duplicate", "remove_degenerate"):
                indices_to_remove.add(idx)
                applied_fixes.append(fix)
            else:
                applied_fixes.append(fix)
        else:
            needs_review.append(fix)

    # Apply clip fixes
    for idx, new_bbox in clip_map.items():
        if 0 <= idx < len(annotations) and len(new_bbox) == 4:
            annotations[idx] = ParsedAnnotation(
                class_id=annotations[idx].class_id,
                bbox_norm=new_bbox,
                polygon_norm=annotations[idx].polygon_norm,
                source_format=annotations[idx].source_format,
            )

    # Remove indices in reverse order to maintain correct indexing
    corrected = [
        ann for idx, ann in enumerate(annotations) if idx not in indices_to_remove
    ]

    # Convert back to original format
    corrected_labels = annotations_to_labels(corrected, label_format, img_w, img_h)

    return FixResponse(
        corrected_labels=corrected_labels,
        applied_fixes=applied_fixes,
        needs_review=needs_review,
        num_applied=len(applied_fixes),
        num_needs_review=len(needs_review),
        num_annotations_before=num_before,
        num_annotations_after=len(corrected),
    )


def aggregate_results(
    results: list[dict],
    dataset_name: str,
    classes: dict[int, str],
) -> ReportResponse:
    """Aggregate per-image QA results into a dataset-level report.

    Args:
        results: List of per-image result dicts.
        dataset_name: Name of the dataset.
        classes: Class mapping.

    Returns:
        ReportResponse with aggregated statistics.
    """
    valid_classes = {int(k): v for k, v in classes.items()}
    worst_n = DEFAULT_WORST_IMAGES_COUNT

    # Grade distribution
    grade_counts: dict[str, int] = {"good": 0, "review": 0, "bad": 0}
    for r in results:
        grade = r.get("grade", "review")
        grade_counts[grade] = grade_counts.get(grade, 0) + 1

    # Average quality score
    scores = [r.get("quality_score", 0.0) for r in results]
    avg_score = float(np.mean(scores)) if scores else 0.0

    # Issue-type breakdown
    issue_counts: dict[str, int] = {}
    for r in results:
        for issue in r.get("issues", []):
            if isinstance(issue, dict):
                itype = issue.get("type", "unknown")
            else:
                itype = getattr(issue, "type", "unknown")
            issue_counts[itype] = issue_counts.get(itype, 0) + 1

    # Per-class statistics
    per_class: dict[str, dict] = {}
    for r in results:
        # Count annotations per class
        ann_classes = r.get("annotation_classes", [])
        for cls_id in ann_classes:
            cname = valid_classes.get(int(cls_id), str(cls_id))
            if cname not in per_class:
                per_class[cname] = {"count": 0, "issues": 0}
            per_class[cname]["count"] += 1

        # Count issues per class from suggested_fixes
        for fix in r.get("suggested_fixes", []):
            if isinstance(fix, dict):
                original = fix.get("original", {})
                if original and isinstance(original, dict):
                    cls_id = original.get("class_id")
                    if cls_id is not None:
                        cname = valid_classes.get(int(cls_id), str(cls_id))
                        if cname not in per_class:
                            per_class[cname] = {"count": 0, "issues": 0}
                        per_class[cname]["issues"] += 1

    # If per_class is still empty, try to populate from raw issues
    if not per_class:
        for cname in valid_classes.values():
            per_class[cname] = {"count": 0, "issues": 0}

    # Worst N images
    sorted_results = sorted(results, key=lambda r: r.get("quality_score", 0.0))
    worst_images = [
        {
            "filename": r.get("filename", ""),
            "quality_score": r.get("quality_score", 0.0),
            "grade": r.get("grade", ""),
            "num_issues": r.get("num_issues", len(r.get("issues", []))),
        }
        for r in sorted_results[:worst_n]
    ]

    # Auto-fixable count
    auto_fixable = 0
    auto_fix_types = {"clip_bbox", "remove_duplicate", "remove_degenerate"}
    for r in results:
        for fix in r.get("suggested_fixes", []):
            if isinstance(fix, dict):
                if fix.get("type") in auto_fix_types:
                    auto_fixable += 1

    return ReportResponse(
        dataset=dataset_name,
        total_checked=len(results),
        grades=grade_counts,
        avg_quality_score=round(avg_score, 4),
        issue_breakdown=issue_counts,
        per_class_stats=per_class,
        worst_images=worst_images,
        auto_fixable_count=auto_fixable,
    )

"""Task-dispatched error analysis runner — detection / classification / segmentation / keypoint.

Entry point: :func:`run_error_analysis`. Called from
:mod:`core.p06_training.post_train` and (indirectly) from both backends'
on-train-end hooks.

Responsibilities:
  1. Run best-checkpoint inference over the whole dataset (capped at
     ``max_samples``).
  2. Compute per-task failure-mode summary (TP/FP/FN for detection, confusion
     matrix + confidence for classification, per-class IoU for segmentation,
     per-keypoint PCK for keypoint).
  3. Persist JSON + markdown summaries.
  4. Emit chart PNGs (per_class_pr_f1, confusion_matrix, confidence_calibration,
     size_recall / per_class_iou / per_keypoint_pck, hardest_images).
  5. Build per-error-type × per-class galleries of hard images, each rendered
     via the unified :func:`annotate_gt_pred` under
     ``error_analysis/hard_images/{false_positives,false_negatives,class_confusion}/<class>/``.

Every GT-vs-Pred rendering here goes through :func:`annotate_gt_pred` with a
shared :class:`VizStyle` — no local box-drawing.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import supervision as sv
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.p06_training._common import (
    unwrap_subset as _unwrap,
    yolo_targets_to_xyxy,
)
from core.p06_training.postprocess import postprocess as _registry_postprocess
from core.p08_evaluation.error_analysis import (
    _LARGE_AREA,
    _SMALL_AREA,
    _size_category,
)

# Human-readable labels used by charts + JSON so every consumer sees the same
# exact threshold definition (COCO convention, expressed in input-size pixels).
SIZE_TIER_LABELS = {
    "small":  f"small (<{int(_SMALL_AREA ** 0.5)}²px, i.e. <{_SMALL_AREA} px²)",
    "medium": f"medium ({int(_SMALL_AREA ** 0.5)}²–{int(_LARGE_AREA ** 0.5)}²px)",
    "large":  f"large (≥{int(_LARGE_AREA ** 0.5)}²px, i.e. ≥{_LARGE_AREA} px²)",
}
from core.p10_inference.supervision_bridge import VizStyle, annotate_gt_pred

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_name(name: str) -> str:
    """Filesystem-safe version of a class/image name."""
    return _SAFE_NAME.sub("_", str(name))[:80]


def _preprocess_for_model(
    raw_image: np.ndarray,
    input_size: tuple[int, int],
    model=None,
) -> torch.Tensor:
    """BGR HWC uint8 → CHW float32 tensor the model can forward on.

    Three cases:
      * Model has ``model.processor`` (HF wrappers e.g. ``HFDetectionModel``):
        delegate to ``AutoImageProcessor`` so inputs are rescaled + normalized
        with the exact ``(mean, std)`` the model was trained with. Without
        this, DETR-family decoders receive un-normalized pixels and produce
        garbage predictions — observed as zero TP/FP in the analyzer when
        HF Trainer's own eval reported mAP50=0.82.
      * Model has ``augmentation.normalize=False`` semantics (YOLOX raw-pixel
        recipe): the postprocess expects [0, 255] uint8 inputs; we feed the
        resized tensor straight through as float without any mean/std shift.
      * Fallback: ImageNet-norm manually (matches our default transforms path).
    """
    h, w = int(input_size[0]), int(input_size[1])
    processor = getattr(model, "processor", None)
    if processor is not None:
        # HF processor wants a list of PIL-style RGB HWC uint8 arrays
        rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (w, h))
        out = processor(images=[resized], return_tensors="pt", do_resize=False)
        return out["pixel_values"][0]

    resized = cv2.resize(raw_image, (w, h))
    arr = resized.astype(np.float32)
    # YOLOX expects raw [0, 255]; others (timm etc.) expect ImageNet-normalized.
    output_format = (getattr(model, "output_format", "") or "").lower()
    if output_format in {"yolox"}:
        tensor_np = arr.transpose(2, 0, 1)  # keep in [0, 255]
    else:
        mean = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255], dtype=np.float32).reshape(1, 1, 3)
        # Our transforms feed BGR to the model; keep channel order consistent.
        tensor_np = ((arr - mean) / std).transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(tensor_np))


def _dispatch_forward(model, tensor_batch: torch.Tensor):
    """Unified forward for the analyzer — never raises on wrapper type."""
    if hasattr(model, "hf_model"):
        return model(pixel_values=tensor_batch)
    return model(tensor_batch)


def _dispatch_postprocess(model, preds_raw, conf_threshold, target_sizes):
    """HF wrappers have ``.postprocess``; our YOLOX models go through the
    registry dispatcher which handles conf_threshold / nms_threshold /
    target_sizes keyword separation per arch.
    """
    if hasattr(model, "postprocess"):
        return model.postprocess(preds_raw, conf_threshold, target_sizes)
    output_format = getattr(model, "output_format", "yolox")
    return _registry_postprocess(
        output_format=output_format,
        model=model,
        predictions=preds_raw,
        conf_threshold=conf_threshold,
        target_sizes=target_sizes,
    )


# ---------------------------------------------------------------------------
# Public: dispatcher
# ---------------------------------------------------------------------------


def run_error_analysis(
    *,
    model,
    dataset,
    output_dir: Path | str,
    task: str,
    class_names: dict[int, str],
    input_size: tuple[int, int],
    style: VizStyle | None = None,
    conf_threshold: float = 0.3,
    iou_threshold: float = 0.5,
    max_samples: int | None = 500,
    hard_images_per_class: int = 20,
    training_config: dict | None = None,
) -> dict[str, Any]:
    """Dispatch error analysis to the right task-specific analyzer.

    Returns a dict describing what was written (paths to summary.json,
    summary.md, chart PNGs, gallery root).
    """
    task = (task or "detection").lower()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    style = style or VizStyle()

    if task == "detection":
        return _analyze_detection(
            model=model, dataset=dataset, output_dir=output_dir,
            class_names=class_names, input_size=input_size, style=style,
            conf_threshold=conf_threshold, iou_threshold=iou_threshold,
            max_samples=max_samples,
            hard_images_per_class=hard_images_per_class,
            training_config=training_config,
        )
    if task == "classification":
        return _analyze_classification(
            model=model, dataset=dataset, output_dir=output_dir,
            class_names=class_names, input_size=input_size, style=style,
            max_samples=max_samples,
            hard_images_per_class=hard_images_per_class,
        )
    if task == "segmentation":
        return _analyze_segmentation(
            model=model, dataset=dataset, output_dir=output_dir,
            class_names=class_names, input_size=input_size, style=style,
            max_samples=max_samples,
            hard_images_per_class=hard_images_per_class,
        )
    if task == "keypoint":
        return _analyze_keypoint(
            model=model, dataset=dataset, output_dir=output_dir,
            class_names=class_names, input_size=input_size, style=style,
            conf_threshold=conf_threshold,
            max_samples=max_samples,
            hard_images_per_class=hard_images_per_class,
        )
    raise ValueError(f"Unknown task for error analysis: {task!r}")


# ===========================================================================
# DETECTION
# ===========================================================================


def _analyze_detection(
    *, model, dataset, output_dir: Path,
    class_names: dict[int, str], input_size: tuple[int, int], style: VizStyle,
    conf_threshold: float, iou_threshold: float,
    max_samples: int | None, hard_images_per_class: int,
    training_config: dict | None = None,
) -> dict[str, Any]:
    raw_ds, idx_map = _unwrap(dataset)
    n = len(dataset)
    indices = list(range(n)) if max_samples is None or max_samples >= n else \
        sorted(np.random.default_rng(0).choice(n, size=max_samples, replace=False).tolist())

    device = next(model.parameters()).device
    input_h, input_w = int(input_size[0]), int(input_size[1])

    # Per-class accumulators
    per_class = {cid: {"tp": 0, "fp": 0, "fn": 0} for cid in class_names}
    confidence_tp: list[float] = []
    confidence_fp: list[float] = []
    confusion = np.zeros((len(class_names) + 1, len(class_names) + 1), dtype=np.int64)
    # rows = GT (last = no GT/background), cols = Pred (last = no Pred/missed)
    size_stats = {
        "small":  {"tp": 0, "fp": 0, "fn": 0},
        "medium": {"tp": 0, "fp": 0, "fn": 0},
        "large":  {"tp": 0, "fp": 0, "fn": 0},
    }
    per_image: list[dict] = []
    # Data-distribution accumulators
    gt_per_class: dict[int, int] = {c: 0 for c in class_names}
    gt_per_class_size: dict[int, dict[str, int]] = {
        c: {"small": 0, "medium": 0, "large": 0} for c in class_names
    }
    total_images_with_gt = 0
    boxes_per_image_counts: list[int] = []        # for crowdedness histogram
    gt_aspect_ratios: dict[int, list[float]] = {c: [] for c in class_names}
    # Per-prediction records — drives PR curves, F1-vs-threshold, mAP-vs-IoU
    # without re-running inference. Each entry: predicted class, confidence,
    # best IoU against any same-class GT (for AP at arbitrary IoU threshold),
    # best IoU against any GT (for the confusion branch).
    detections: list[dict] = []
    fp_gallery: dict[int, list[dict]] = {c: [] for c in class_names}
    fn_gallery: dict[int, list[dict]] = {c: [] for c in class_names}
    conf_gallery: dict[tuple[int, int], list[dict]] = {}
    # Cache raw image + predictions + GT per image so gallery renderers
    # below don't re-run inference (plan's efficiency acceptance criterion:
    # galleries must reuse analyzer predictions rather than forwarding again).
    pred_cache: dict[int, dict] = {}

    for local_i, ds_idx in enumerate(indices):
        real_idx = idx_map(ds_idx)
        try:
            raw = raw_ds.get_raw_item(real_idx)
        except Exception:
            continue
        image = raw["image"]
        if image is None:
            continue
        orig_h, orig_w = image.shape[:2]
        tensor = _preprocess_for_model(image, (input_h, input_w), model=model).unsqueeze(0).to(device)
        with torch.no_grad():
            preds_raw = _dispatch_forward(model, tensor)
        target_sizes = torch.tensor([[input_h, input_w]], device=device)
        decoded = _dispatch_postprocess(model, preds_raw, conf_threshold, target_sizes)[0]

        pb = np.asarray(decoded.get("boxes", []), dtype=np.float64).reshape(-1, 4)
        pl = np.asarray(decoded.get("labels", []), dtype=np.int64).ravel()
        ps = np.asarray(decoded.get("scores", []), dtype=np.float64).ravel()

        # Rescale pred boxes to original resolution for consistent match/draw
        if len(pb) > 0:
            pb[:, [0, 2]] *= orig_w / input_w
            pb[:, [1, 3]] *= orig_h / input_h

        gt_xyxy, gt_cls = yolo_targets_to_xyxy(raw.get("targets"), orig_w, orig_h)

        # Data-distribution counts — class + size-tier + aspect-ratio per GT box.
        if len(gt_xyxy) > 0:
            total_images_with_gt += 1
        boxes_per_image_counts.append(int(len(gt_xyxy)))
        for j in range(len(gt_xyxy)):
            cid = int(gt_cls[j])
            gt_per_class[cid] = gt_per_class.get(cid, 0) + 1
            bw = max(1.0, gt_xyxy[j, 2] - gt_xyxy[j, 0])
            bh = max(1.0, gt_xyxy[j, 3] - gt_xyxy[j, 1])
            gt_area = bw * bh
            tier = _size_category(gt_area)
            gt_per_class_size.setdefault(cid,
                {"small": 0, "medium": 0, "large": 0})[tier] += 1
            gt_aspect_ratios.setdefault(cid, []).append(float(bw / bh))

        # Cache everything gallery renderers need — image bytes stay on disk
        # (we only keep the path) so 500-image analysis doesn't balloon RAM.
        pred_cache[int(real_idx)] = {
            "path": raw.get("path", ""),
            "pb": pb, "pl": pl, "ps": ps,
            "gt_xyxy": gt_xyxy, "gt_cls": gt_cls,
            "orig_shape": (orig_h, orig_w),
        }

        matched_gt = np.zeros(len(gt_xyxy), dtype=bool)
        img_tp, img_fp, img_fn = 0, 0, 0

        # Greedy match: each pred → best IoU GT across classes; if matching
        # class present → TP; else classify as confusion / FP.
        for bi in range(len(pb)):
            best_iou, best_j = 0.0, -1
            best_same_class_iou, best_same_class_j = 0.0, -1
            for j in range(len(gt_xyxy)):
                if matched_gt[j]:
                    continue
                iou = _iou(pb[bi], gt_xyxy[j])
                if iou > best_iou:
                    best_iou, best_j = iou, j
                if gt_cls[j] == pl[bi] and iou > best_same_class_iou:
                    best_same_class_iou, best_same_class_j = iou, j

            area = max(0.0, (pb[bi, 2] - pb[bi, 0])) * max(0.0, (pb[bi, 3] - pb[bi, 1]))
            size = _size_category(area)

            # Per-pred record — drives threshold/IoU sweeps below.
            detections.append({
                "pred_cls": int(pl[bi]),
                "score": float(ps[bi]),
                "best_iou_same_class": float(best_same_class_iou),
                "best_iou_any": float(best_iou),
                "gt_cls_at_best_iou": int(gt_cls[best_j]) if best_j >= 0 else -1,
            })

            if best_same_class_iou >= iou_threshold:
                matched_gt[best_same_class_j] = True
                per_class[int(pl[bi])]["tp"] += 1
                size_stats[size]["tp"] += 1
                confusion[int(pl[bi]), int(pl[bi])] += 1
                confidence_tp.append(float(ps[bi]))
                img_tp += 1
            elif best_iou >= iou_threshold and best_j >= 0 and gt_cls[best_j] != pl[bi]:
                # Class confusion: localized on wrong class
                matched_gt[best_j] = True
                per_class[int(pl[bi])]["fp"] += 1
                size_stats[size]["fp"] += 1
                confusion[int(gt_cls[best_j]), int(pl[bi])] += 1
                confidence_fp.append(float(ps[bi]))
                img_fp += 1
                key = (int(gt_cls[best_j]), int(pl[bi]))
                conf_gallery.setdefault(key, []).append({
                    "image_idx": int(real_idx),
                    "path": raw.get("path", ""),
                    "pred_box": pb[bi].tolist(),
                    "score": float(ps[bi]),
                    "iou": float(best_iou),
                })
            else:
                # Background FP
                per_class[int(pl[bi])]["fp"] += 1
                size_stats[size]["fp"] += 1
                confusion[len(class_names), int(pl[bi])] += 1
                confidence_fp.append(float(ps[bi]))
                img_fp += 1
                fp_gallery.setdefault(int(pl[bi]), []).append({
                    "image_idx": int(real_idx),
                    "path": raw.get("path", ""),
                    "pred_box": pb[bi].tolist(),
                    "score": float(ps[bi]),
                })

        # Unmatched GT → FN (missed)
        for j in np.where(~matched_gt)[0]:
            cid = int(gt_cls[j])
            per_class[cid]["fn"] += 1
            area = max(0.0, (gt_xyxy[j, 2] - gt_xyxy[j, 0])) * max(0.0, (gt_xyxy[j, 3] - gt_xyxy[j, 1]))
            size_stats[_size_category(area)]["fn"] += 1
            confusion[cid, len(class_names)] += 1
            img_fn += 1
            fn_gallery.setdefault(cid, []).append({
                "image_idx": int(real_idx),
                "path": raw.get("path", ""),
                "gt_box": gt_xyxy[j].tolist(),
            })

        per_image.append({
            "idx": int(real_idx), "path": raw.get("path", ""),
            "tp": img_tp, "fp": img_fp, "fn": img_fn,
        })

    # ---- summary ----
    summary = _summarize_detection(per_class, size_stats, confidence_tp, confidence_fp, class_names)
    if training_config:
        summary["training_config"] = training_config
    # Attach data distribution + explicit size-tier definitions so readers
    # see the underlying counts behind precision/recall and the exact area
    # thresholds used for the size breakdown.
    summary["size_tier_definitions"] = {
        "small":  {"max_area_px2": _SMALL_AREA, "description": SIZE_TIER_LABELS["small"]},
        "medium": {"min_area_px2": _SMALL_AREA, "max_area_px2": _LARGE_AREA,
                   "description": SIZE_TIER_LABELS["medium"]},
        "large":  {"min_area_px2": _LARGE_AREA, "description": SIZE_TIER_LABELS["large"]},
    }
    summary["data_distribution"] = {
        "total_images_with_gt": total_images_with_gt,
        "total_gt_boxes": int(sum(gt_per_class.values())),
        "per_class": {class_names.get(cid, str(cid)): int(v)
                      for cid, v in gt_per_class.items()},
        "per_class_per_size": {
            class_names.get(cid, str(cid)): {
                k: int(v) for k, v in tiers.items()
            }
            for cid, tiers in gt_per_class_size.items()
        },
    }
    artifacts = {}

    # ---- charts ----
    # Dataset-shape (axis 1)
    artifacts["data_distribution"] = _plot_data_distribution(
        gt_per_class, gt_per_class_size, class_names,
        output_dir / "data_distribution.png",
    )
    artifacts["boxes_per_image"] = _plot_boxes_per_image(
        boxes_per_image_counts, output_dir / "boxes_per_image.png",
    )
    artifacts["bbox_aspect_ratio"] = _plot_bbox_aspect_ratio(
        gt_aspect_ratios, class_names, output_dir / "bbox_aspect_ratio.png",
    )
    # Model-performance (axis 3)
    artifacts["per_class_pr_f1"] = _plot_per_class_pr_f1(
        per_class, class_names, output_dir / "per_class_pr_f1.png",
    )
    artifacts["confusion_matrix"] = _plot_confusion_matrix(
        confusion, class_names, output_dir / "confusion_matrix.png",
    )
    artifacts["confidence_calibration"] = _plot_confidence_hist(
        confidence_tp, confidence_fp, output_dir / "confidence_calibration.png",
    )
    artifacts["size_recall"] = _plot_size_recall(
        size_stats, output_dir / "size_recall.png",
    )
    # Threshold-sweep charts (from per-pred records — no re-inference)
    artifacts["pr_curves"] = _plot_pr_curves(
        detections, gt_per_class, class_names, iou_threshold,
        output_dir / "pr_curves.png",
    )
    artifacts["f1_vs_threshold"] = _plot_f1_vs_threshold(
        detections, gt_per_class, class_names, iou_threshold,
        output_dir / "f1_vs_threshold.png",
    )
    artifacts["map_vs_iou"] = _plot_map_vs_iou(
        detections, gt_per_class, class_names,
        output_dir / "map_vs_iou.png",
    )

    # Expose the numbers the sweep charts are built from in summary.json
    summary["model_metrics"] = {
        "ap50_per_class": _per_class_ap(detections, gt_per_class, class_names, 0.5),
        "ap75_per_class": _per_class_ap(detections, gt_per_class, class_names, 0.75),
        "map_vs_iou": _map_at_iou_sweep(detections, gt_per_class,
                                         np.arange(0.5, 1.0, 0.05)),
        "best_f1_per_class": _best_f1_and_threshold(
            detections, gt_per_class, class_names, iou_threshold,
        ),
    }

    # Persist the full report now that every section is populated.
    _write_json_md(
        output_dir / "summary.json", output_dir / "summary.md",
        summary,
        title="Detection Error Analysis",
        header=[
            f"- Samples analyzed: **{len(per_image)}**",
            f"- Images with ≥1 GT box: **{total_images_with_gt}**",
            f"- Total GT boxes: **{int(sum(gt_per_class.values()))}**",
            f"- IoU threshold (base): {iou_threshold}",
            f"- Confidence threshold (base): {conf_threshold}",
            f"- Size tiers (COCO, in pixels² of box area):",
            f"    * {SIZE_TIER_LABELS['small']}",
            f"    * {SIZE_TIER_LABELS['medium']}",
            f"    * {SIZE_TIER_LABELS['large']}",
        ],
    )
    artifacts["summary_json"] = output_dir / "summary.json"
    artifacts["summary_md"] = output_dir / "summary.md"

    # ---- hardest images overview grid (top 12 by FP+FN) ----
    per_image.sort(key=lambda r: -(r["fn"] + r["fp"]))
    artifacts["hardest_images"] = _render_hardest_overview(
        raw_ds, per_image[:12], class_names, style, pred_cache,
        output_dir / "hardest_images.png",
    )

    # ---- per-class galleries — all three share the same rendering path,
    # differing only in which dict they iterate and the filename suffix. ----
    if hard_images_per_class > 0:
        gallery_root = output_dir / "hard_images"
        artifacts["hard_images_root"] = gallery_root

        def _fp_name(rec): return f"fp_score_{rec['score']:.2f}"
        def _fn_name(rec): return "fn"
        def _conf_name(rec): return f"iou_{rec['iou']:.2f}"

        _render_gallery(
            raw_ds, class_names, style, pred_cache, hard_images_per_class,
            grouped=fp_gallery, class_labeler=lambda cid: class_names.get(cid, str(cid)),
            root=gallery_root / "false_positives", suffix_fn=_fp_name,
            sort_key=lambda r: -r["score"],
        )
        _render_gallery(
            raw_ds, class_names, style, pred_cache, hard_images_per_class,
            grouped=fn_gallery, class_labeler=lambda cid: class_names.get(cid, str(cid)),
            root=gallery_root / "false_negatives", suffix_fn=_fn_name,
            sort_key=None,
        )
        _render_gallery(
            raw_ds, class_names, style, pred_cache, hard_images_per_class,
            grouped=conf_gallery,
            class_labeler=lambda pair: f"{_safe_name(class_names.get(pair[1], str(pair[1])))}__from__{_safe_name(class_names.get(pair[0], str(pair[0])))}",
            root=gallery_root / "class_confusion", suffix_fn=_conf_name,
            sort_key=lambda r: -r["iou"],
        )

    return artifacts


# --------------------------- detection helpers ---------------------------


def _iou(a, b) -> float:
    xa = max(a[0], b[0]); ya = max(a[1], b[1])
    xb = min(a[2], b[2]); yb = min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _summarize_detection(per_class, size_stats, tp_scores, fp_scores, class_names):
    out_classes = {}
    for cid, c in per_class.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out_classes[class_names.get(cid, str(cid))] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
        }
    return {
        "task": "detection",
        "per_class": out_classes,
        "size_breakdown": size_stats,
        "confidence": {
            "tp_count": len(tp_scores),
            "fp_count": len(fp_scores),
            "tp_mean_score": round(float(np.mean(tp_scores)), 4) if tp_scores else None,
            "fp_mean_score": round(float(np.mean(fp_scores)), 4) if fp_scores else None,
        },
    }


def _plot_per_class_pr_f1(per_class, class_names, path: Path) -> Path:
    names = [class_names.get(cid, str(cid)) for cid in per_class]
    tp = np.array([c["tp"] for c in per_class.values()], dtype=np.float32)
    fp = np.array([c["fp"] for c in per_class.values()], dtype=np.float32)
    fn = np.array([c["fn"] for c in per_class.values()], dtype=np.float32)
    prec = np.where(tp + fp > 0, tp / (tp + fp + 1e-9), 0)
    rec = np.where(tp + fn > 0, tp / (tp + fn + 1e-9), 0)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec + 1e-9), 0)

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.1), 5))
    x = np.arange(len(names))
    w = 0.28
    ax.bar(x - w, prec, w, label="Precision", color="#4c72b0")
    ax.bar(x,     rec,  w, label="Recall",    color="#55a868")
    ax.bar(x + w, f1,   w, label="F1",         color="#c44e52")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("score")
    ax.set_title("Per-class Precision / Recall / F1")
    ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_confusion_matrix(cm: np.ndarray, class_names: dict[int, str], path: Path) -> Path:
    labels = [class_names.get(cid, str(cid)) for cid in class_names] + ["(none)"]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), max(5, len(labels) * 0.7)))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground-truth")
    ax.set_title("Confusion matrix (rows: GT class; cols: pred class; last = none)")
    fig.colorbar(im, ax=ax)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            if v == 0:
                continue
            ax.text(j, i, str(int(v)), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_confidence_hist(tp_scores: list[float], fp_scores: list[float], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 21)
    if tp_scores:
        ax.hist(tp_scores, bins=bins, alpha=0.6, label=f"TP (n={len(tp_scores)})", color="#55a868")
    if fp_scores:
        ax.hist(fp_scores, bins=bins, alpha=0.6, label=f"FP (n={len(fp_scores)})", color="#c44e52")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Count")
    ax.set_title("Confidence calibration — TP vs FP score distribution")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_size_recall(size_stats: dict, path: Path) -> Path:
    tiers = ["small", "medium", "large"]
    rec = []
    prec = []
    for t in tiers:
        tp = size_stats[t]["tp"]; fp = size_stats[t]["fp"]; fn = size_stats[t]["fn"]
        rec.append(tp / (tp + fn) if (tp + fn) else 0.0)
        prec.append(tp / (tp + fp) if (tp + fp) else 0.0)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(3); w = 0.4
    ax.bar(x - w/2, prec, w, label="Precision", color="#4c72b0")
    ax.bar(x + w/2, rec,  w, label="Recall",    color="#55a868")
    for i, t in enumerate(tiers):
        c = size_stats[t]
        ax.text(i, 1.02, f"TP {c['tp']} FP {c['fp']} FN {c['fn']}", ha="center", fontsize=9)
    ax.set_xticks(x)
    # Explicit thresholds in the tick labels so readers know what "small" means
    ax.set_xticklabels([SIZE_TIER_LABELS[t] for t in tiers], fontsize=9)
    ax.set_ylim(0, 1.2); ax.set_ylabel("score")
    ax.set_title("Size-stratified recall / precision (COCO tiers, box area in pixels²)")
    ax.legend(loc="lower right"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_data_distribution(
    gt_per_class: dict, gt_per_class_size: dict,
    class_names: dict[int, str], path: Path,
) -> Path:
    """Two-panel chart: (left) total GT boxes per class, (right) stacked by
    size tier. Answers "what data does the model see?" at a glance — essential
    context for interpreting per-class recall + size-recall charts.
    """
    ordered_cids = sorted(gt_per_class.keys())
    names = [class_names.get(cid, str(cid)) for cid in ordered_cids]
    totals = [gt_per_class[cid] for cid in ordered_cids]
    small = [gt_per_class_size.get(cid, {}).get("small", 0) for cid in ordered_cids]
    medium = [gt_per_class_size.get(cid, {}).get("medium", 0) for cid in ordered_cids]
    large = [gt_per_class_size.get(cid, {}).get("large", 0) for cid in ordered_cids]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(11, len(names) * 1.3), 5))
    x = np.arange(len(names))

    # Left: bar per class (absolute counts + percentage)
    total = max(1, sum(totals))
    ax1.bar(x, totals, color="#4c72b0")
    for i, v in enumerate(totals):
        pct = 100.0 * v / total
        ax1.text(i, v + total * 0.01, f"{v} ({pct:.1f}%)",
                 ha="center", fontsize=9)
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=30, ha="right")
    ax1.set_ylabel("GT box count")
    ax1.set_title(f"Class distribution (total GT boxes = {total})")
    ax1.grid(axis="y", alpha=0.3)

    # Right: stacked by COCO size tier
    ax2.bar(x, small, label=SIZE_TIER_LABELS["small"], color="#dd8452")
    ax2.bar(x, medium, bottom=small, label=SIZE_TIER_LABELS["medium"], color="#8172b3")
    bottom_ml = [s + m for s, m in zip(small, medium)]
    ax2.bar(x, large, bottom=bottom_ml, label=SIZE_TIER_LABELS["large"], color="#64b5cd")
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=30, ha="right")
    ax2.set_ylabel("GT box count")
    ax2.set_title("Per-class × per-size-tier distribution")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Threshold-sweep + dataset-shape helpers
# ---------------------------------------------------------------------------


def _per_class_ap_curve(
    detections: list[dict], gt_count: int, target_cls: int, iou_thr: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute PR curve + AP for one class at a given IoU threshold.

    Inputs are the flat per-prediction records from the analyzer's matching
    loop. A detection counts as TP if its best-same-class IoU ≥ iou_thr,
    counted in score-descending order; duplicate matches on the same GT are
    handled by the earlier greedy matcher (detections already carry the
    matched-GT identity). Returns (recall, precision, ap) where ap is the
    all-points AUC (standard COCO definition).
    """
    # Filter to this class, sort by score desc.
    class_dets = [d for d in detections if d["pred_cls"] == target_cls]
    class_dets.sort(key=lambda r: -r["score"])
    if not class_dets or gt_count == 0:
        return np.zeros(1), np.zeros(1), 0.0
    tp = np.array([1 if d["best_iou_same_class"] >= iou_thr else 0
                    for d in class_dets], dtype=np.float32)
    fp = 1 - tp
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / max(1, gt_count)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    # All-points AP
    prec_interp = np.concatenate([[1.0], precision, [0.0]])
    rec_interp = np.concatenate([[0.0], recall, [1.0]])
    for i in range(len(prec_interp) - 1, 0, -1):
        prec_interp[i - 1] = max(prec_interp[i - 1], prec_interp[i])
    idx = np.where(rec_interp[1:] != rec_interp[:-1])[0]
    ap = float(np.sum((rec_interp[idx + 1] - rec_interp[idx]) * prec_interp[idx + 1]))
    return recall, precision, ap


def _per_class_ap(detections, gt_per_class, class_names, iou_thr) -> dict:
    out = {}
    for cid in class_names:
        _, _, ap = _per_class_ap_curve(detections, int(gt_per_class.get(cid, 0)), cid, iou_thr)
        out[class_names.get(cid, str(cid))] = round(float(ap), 4)
    return out


def _map_at_iou_sweep(detections, gt_per_class, iou_values) -> dict:
    """mAP computed at a grid of IoU thresholds (COCO 0.5..0.95 step 0.05)."""
    out = {}
    for iou in iou_values:
        aps = []
        for cid, gt_count in gt_per_class.items():
            if gt_count == 0:
                continue
            _, _, ap = _per_class_ap_curve(detections, int(gt_count), int(cid), float(iou))
            aps.append(ap)
        out[f"iou_{iou:.2f}"] = round(float(np.mean(aps)) if aps else 0.0, 4)
    return out


def _best_f1_and_threshold(detections, gt_per_class, class_names, iou_thr) -> dict:
    """Per-class best-F1 threshold via a 19-point conf sweep."""
    thresholds = np.linspace(0.05, 0.95, 19)
    out = {}
    for cid in class_names:
        class_dets = [d for d in detections if d["pred_cls"] == cid]
        gt_count = int(gt_per_class.get(cid, 0))
        if not class_dets or gt_count == 0:
            out[class_names.get(cid, str(cid))] = {"threshold": None, "f1": 0.0,
                                                    "precision": 0.0, "recall": 0.0}
            continue
        best = {"threshold": None, "f1": 0.0, "precision": 0.0, "recall": 0.0}
        for thr in thresholds:
            kept = [d for d in class_dets if d["score"] >= thr]
            tp = sum(1 for d in kept if d["best_iou_same_class"] >= iou_thr)
            fp = len(kept) - tp
            fn = gt_count - tp
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            if f1 > best["f1"]:
                best = {"threshold": round(float(thr), 2),
                        "f1": round(f1, 4),
                        "precision": round(prec, 4),
                        "recall": round(rec, 4)}
        out[class_names.get(cid, str(cid))] = best
    return out


def _plot_pr_curves(detections, gt_per_class, class_names, iou_thr, path: Path) -> Path:
    """One PR curve per class at the base IoU threshold. AP in legend."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(class_names), 10)))
    for i, cid in enumerate(sorted(class_names)):
        gt_count = int(gt_per_class.get(cid, 0))
        recall, precision, ap = _per_class_ap_curve(detections, gt_count, cid, iou_thr)
        ax.plot(recall, precision, color=colors[i % 10], lw=1.8,
                label=f"{class_names.get(cid, str(cid))}  AP={ap:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title(f"PR curves per class @ IoU={iou_thr}")
    ax.grid(alpha=0.3); ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


def _plot_f1_vs_threshold(detections, gt_per_class, class_names, iou_thr, path: Path) -> Path:
    """F1 vs conf_threshold per class — answers "what threshold to deploy at?"."""
    thresholds = np.linspace(0.05, 0.95, 19)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(class_names), 10)))
    for i, cid in enumerate(sorted(class_names)):
        class_dets = [d for d in detections if d["pred_cls"] == cid]
        gt_count = int(gt_per_class.get(cid, 0))
        if not class_dets or gt_count == 0:
            continue
        f1s = []
        for thr in thresholds:
            kept = [d for d in class_dets if d["score"] >= thr]
            tp = sum(1 for d in kept if d["best_iou_same_class"] >= iou_thr)
            fp = len(kept) - tp
            fn = gt_count - tp
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
        best_idx = int(np.argmax(f1s))
        ax.plot(thresholds, f1s, color=colors[i % 10], lw=1.8,
                label=f"{class_names.get(cid, str(cid))} (best F1={f1s[best_idx]:.2f} @ thr={thresholds[best_idx]:.2f})")
        ax.axvline(thresholds[best_idx], color=colors[i % 10], ls=":", alpha=0.4)
    ax.set_xlabel("Confidence threshold"); ax.set_ylabel("F1")
    ax.set_xlim(0.05, 0.95); ax.set_ylim(0, 1.02)
    ax.set_title(f"F1 vs confidence threshold per class @ IoU={iou_thr}")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


def _plot_map_vs_iou(detections, gt_per_class, class_names, path: Path) -> Path:
    """mAP vs IoU threshold (0.50 → 0.95). Shows localization vs classification bottleneck."""
    iou_values = np.arange(0.5, 1.0, 0.05)
    map_values = []
    for iou in iou_values:
        aps = []
        for cid, gt_count in gt_per_class.items():
            if gt_count == 0: continue
            _, _, ap = _per_class_ap_curve(detections, int(gt_count), int(cid), float(iou))
            aps.append(ap)
        map_values.append(float(np.mean(aps)) if aps else 0.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iou_values, map_values, marker="o", color="#4c72b0", lw=2)
    for x, y in zip(iou_values, map_values):
        ax.text(x, y + 0.02, f"{y:.3f}", ha="center", fontsize=8)
    ax.set_xlabel("IoU threshold")
    ax.set_ylabel("mAP (mean over classes with ≥1 GT)")
    ax.set_xlim(0.48, 0.97); ax.set_ylim(0, min(1.05, max(map_values) * 1.3 + 0.1))
    ax.set_title(
        f"mAP vs IoU  —  AP50={map_values[0]:.3f}  AP75={map_values[5]:.3f}  "
        f"AP@[.5:.95]={np.mean(map_values):.3f}"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


def _plot_boxes_per_image(counts: list[int], path: Path) -> Path:
    """Histogram of #GT boxes per image. Crowdedness diagnostic."""
    if not counts:
        return path
    counts_arr = np.asarray(counts, dtype=np.int32)
    max_n = max(1, int(counts_arr.max()))
    bins = np.arange(0, max_n + 2) - 0.5
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(counts_arr, bins=bins, color="#4c72b0", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("GT boxes per image")
    ax.set_ylabel("# images")
    mean = float(counts_arr.mean())
    median = float(np.median(counts_arr))
    p95 = float(np.percentile(counts_arr, 95))
    ax.set_title(
        f"Boxes per image — mean {mean:.1f} / median {median:.0f} / p95 {p95:.0f} / max {max_n}"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


def _plot_bbox_aspect_ratio(gt_aspect_ratios, class_names, path: Path) -> Path:
    """Per-class aspect-ratio (w/h) distribution. Reveals shape bias."""
    flat = [(cid, r) for cid, ratios in gt_aspect_ratios.items() for r in ratios]
    if not flat:
        return path
    # Use log scale for aspect ratio so 0.5 (tall) and 2.0 (wide) are symmetric.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bins = np.logspace(np.log10(0.1), np.log10(10.0), 26)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(class_names), 10)))
    for i, cid in enumerate(sorted(gt_aspect_ratios.keys())):
        ratios = gt_aspect_ratios.get(cid, [])
        if not ratios:
            continue
        ax.hist(ratios, bins=bins, alpha=0.55, color=colors[i % 10],
                label=f"{class_names.get(cid, str(cid))} (n={len(ratios)})")
    ax.axvline(1.0, color="black", ls="--", lw=1, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("bbox aspect ratio (w / h, log scale)")
    ax.set_ylabel("count")
    ax.set_title(
        "GT bbox aspect-ratio distribution per class  "
        "(dashed = square; <1 = taller-than-wide; >1 = wider-than-tall)"
    )
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


def _annotate_from_cache(image, entry, class_names, style) -> np.ndarray:
    """Render GT + Pred overlay using cached analyzer predictions.

    No forward pass — reuses pb/pl/ps/gt_xyxy/gt_cls stored by
    :func:`_analyze_detection`. This is the efficiency win: galleries that
    previously re-ran inference now read from the cache.
    """
    pb = entry.get("pb", np.zeros((0, 4)))
    pl = entry.get("pl", np.zeros(0, dtype=np.int64))
    ps = entry.get("ps", np.zeros(0, dtype=np.float64))
    pred_dets = sv.Detections(xyxy=pb, class_id=pl, confidence=ps)
    return annotate_gt_pred(
        image, entry.get("gt_xyxy"), entry.get("gt_cls"),
        pred_dets, class_names, style=style,
    )


def _render_hardest_overview(
    raw_ds, hardest: list[dict], class_names, style, pred_cache: dict, path: Path,
) -> Path | None:
    """Grid of the top-N hardest images, reusing cached predictions."""
    if not hardest:
        return None
    rows = []
    for rec in hardest:
        idx = rec["idx"]
        entry = pred_cache.get(idx)
        if entry is None:
            continue
        try:
            raw = raw_ds.get_raw_item(idx)
        except Exception:
            continue
        image = raw["image"]
        if image is None:
            continue
        rows.append(_annotate_from_cache(image, entry, class_names, style))

    if not rows:
        return None
    from core.p06_training.post_train import _save_grid
    _save_grid(rows, path, "Hardest images (sorted by FP+FN)", ncols=4, dpi=130)
    return path


def _render_gallery(
    raw_ds, class_names, style, pred_cache: dict, cap: int,
    *, grouped: dict, class_labeler, root: Path, suffix_fn, sort_key,
):
    """Unified per-class gallery renderer — replaces FP / FN / confusion
    copy-paste. ``grouped`` maps a key (int class id or (gt_cid, pred_cid) pair)
    to a list of records; ``class_labeler(key)`` returns the class-dir name;
    ``suffix_fn(rec)`` builds the trailing filename discriminator; ``sort_key``
    ranks records (None → preserve insertion order)."""
    for key, items in grouped.items():
        if sort_key is not None:
            items = sorted(items, key=sort_key)
        items = items[:cap]
        if not items:
            continue
        class_dir = root / _safe_name(class_labeler(key))
        class_dir.mkdir(parents=True, exist_ok=True)
        for rec in items:
            idx = rec["image_idx"]
            entry = pred_cache.get(idx)
            if entry is None:
                continue
            try:
                raw = raw_ds.get_raw_item(idx)
            except Exception:
                continue
            image = raw["image"]
            if image is None:
                continue
            annotated = _annotate_from_cache(image, entry, class_names, style)
            stem = Path(rec["path"]).stem or f"img_{idx}"
            out = class_dir / f"{_safe_name(stem)}__{suffix_fn(rec)}.png"
            cv2.imwrite(str(out), annotated)


# ===========================================================================
# CLASSIFICATION
# ===========================================================================


def _analyze_classification(
    *, model, dataset, output_dir: Path,
    class_names: dict[int, str], input_size: tuple[int, int], style: VizStyle,
    max_samples: int | None, hard_images_per_class: int,
) -> dict[str, Any]:
    raw_ds, idx_map = _unwrap(dataset)
    n = len(dataset)
    indices = list(range(n)) if max_samples is None or max_samples >= n else \
        sorted(np.random.default_rng(0).choice(n, size=max_samples, replace=False).tolist())
    device = next(model.parameters()).device
    input_h, input_w = int(input_size[0]), int(input_size[1])

    C = len(class_names)
    cm = np.zeros((C, C), dtype=np.int64)
    per_class = {cid: {"tp": 0, "fp": 0, "fn": 0} for cid in class_names}
    correct_scores: list[float] = []
    wrong_scores: list[float] = []
    per_image: list[dict] = []
    wrong_gallery: dict[int, list[dict]] = {c: [] for c in class_names}  # key = predicted class

    for ds_idx in indices:
        real_idx = idx_map(ds_idx)
        try:
            raw = raw_ds.get_raw_item(real_idx)
        except Exception:
            continue
        image = raw["image"]
        gt = raw.get("targets")
        if image is None or gt is None:
            continue
        tensor = _preprocess_for_model(image, (input_h, input_w), model=model).unsqueeze(0).to(device)
        with torch.no_grad():
            out = _dispatch_forward(model, tensor)
        logits = out.logits if hasattr(out, "logits") else out
        probs = torch.softmax(logits, dim=-1).cpu().numpy().ravel()
        pred = int(np.argmax(probs)); score = float(probs[pred])
        gt_cid = int(gt)
        if 0 <= gt_cid < C and 0 <= pred < C:
            cm[gt_cid, pred] += 1
        if pred == gt_cid:
            per_class[gt_cid]["tp"] += 1
            correct_scores.append(score)
        else:
            per_class[pred]["fp"] += 1
            per_class[gt_cid]["fn"] += 1
            wrong_scores.append(score)
            wrong_gallery.setdefault(pred, []).append({
                "image_idx": int(real_idx),
                "path": raw.get("path", ""),
                "gt_cid": gt_cid, "pred_cid": pred, "score": score,
            })
        per_image.append({
            "idx": int(real_idx), "path": raw.get("path", ""),
            "correct": bool(pred == gt_cid), "score": score,
            "gt": gt_cid, "pred": pred,
        })

    # summary
    out_classes = {}
    for cid, c in per_class.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out_classes[class_names.get(cid, str(cid))] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
        }
    overall_acc = (sum(1 for r in per_image if r["correct"]) / max(1, len(per_image)))
    summary = {
        "task": "classification",
        "overall_accuracy": round(overall_acc, 4),
        "per_class": out_classes,
        "confidence": {
            "correct_count": len(correct_scores),
            "wrong_count": len(wrong_scores),
            "correct_mean_score": round(float(np.mean(correct_scores)), 4) if correct_scores else None,
            "wrong_mean_score": round(float(np.mean(wrong_scores)), 4) if wrong_scores else None,
        },
    }
    _write_json_md(
        output_dir / "summary.json", output_dir / "summary.md",
        summary,
        title="Classification Error Analysis",
        header=[f"- Samples analyzed: **{len(per_image)}**",
                f"- Overall accuracy: **{overall_acc:.4f}**"],
    )

    artifacts = {"summary_json": output_dir / "summary.json",
                 "summary_md": output_dir / "summary.md"}
    artifacts["per_class_pr_f1"] = _plot_per_class_pr_f1(per_class, class_names,
                                                         output_dir / "per_class_pr_f1.png")
    # Add background column/row for consistency with detection cm
    cm_padded = np.zeros((C + 1, C + 1), dtype=np.int64)
    cm_padded[:C, :C] = cm
    artifacts["confusion_matrix"] = _plot_confusion_matrix(cm_padded, class_names,
                                                           output_dir / "confusion_matrix.png")
    artifacts["confidence_calibration"] = _plot_confidence_hist(
        correct_scores, wrong_scores, output_dir / "confidence_calibration.png",
    )

    # hard images gallery: grouped by predicted class, worst-confidence first
    if hard_images_per_class > 0:
        root = output_dir / "hard_images" / "misclassified"
        root.mkdir(parents=True, exist_ok=True)
        for pred_cid, items in wrong_gallery.items():
            items = sorted(items, key=lambda r: -r["score"])[:hard_images_per_class]
            if not items:
                continue
            class_dir = root / _safe_name(class_names.get(pred_cid, str(pred_cid)))
            class_dir.mkdir(parents=True, exist_ok=True)
            for rec in items:
                try:
                    raw = raw_ds.get_raw_item(rec["image_idx"])
                except Exception:
                    continue
                image = raw["image"]
                if image is None:
                    continue
                # Classification annotate: overlay GT + Pred strip via annotate_gt_pred
                # with task-agnostic fallback — we render a caption bar ourselves since
                # classification has no boxes.
                bar = np.full((28, image.shape[1], 3), 30, dtype=np.uint8)
                text = (f"GT: {class_names.get(rec['gt_cid'])}    "
                        f"Pred: {class_names.get(rec['pred_cid'])} ({rec['score']:.2f})")
                pred_bgr = (style.gt_color_rgb[2], style.gt_color_rgb[1], style.gt_color_rgb[0])
                cv2.putText(bar, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            pred_bgr, 1, cv2.LINE_AA)
                annotated = np.vstack([bar, image])
                stem = Path(rec["path"]).stem or f"img_{rec['image_idx']}"
                out = class_dir / f"{_safe_name(stem)}__pred_{_safe_name(class_names.get(rec['pred_cid']))}_conf_{rec['score']:.2f}.png"
                cv2.imwrite(str(out), annotated)

    return artifacts


# ===========================================================================
# SEGMENTATION
# ===========================================================================


def _analyze_segmentation(
    *, model, dataset, output_dir: Path,
    class_names: dict[int, str], input_size: tuple[int, int], style: VizStyle,
    max_samples: int | None, hard_images_per_class: int,
) -> dict[str, Any]:
    raw_ds, idx_map = _unwrap(dataset)
    n = len(dataset)
    indices = list(range(n)) if max_samples is None or max_samples >= n else \
        sorted(np.random.default_rng(0).choice(n, size=max_samples, replace=False).tolist())
    device = next(model.parameters()).device
    input_h, input_w = int(input_size[0]), int(input_size[1])

    C = len(class_names)
    intersect = np.zeros(C, dtype=np.float64)
    union = np.zeros(C, dtype=np.float64)
    per_image_miou: list[dict] = []

    for ds_idx in indices:
        real_idx = idx_map(ds_idx)
        try:
            raw = raw_ds.get_raw_item(real_idx)
        except Exception:
            continue
        image = raw["image"]
        gt_mask = raw.get("targets")
        if image is None or gt_mask is None:
            continue
        tensor = _preprocess_for_model(image, (input_h, input_w), model=model).unsqueeze(0).to(device)
        with torch.no_grad():
            out = _dispatch_forward(model, tensor)
        logits = out.logits if hasattr(out, "logits") else out
        pred_mask = logits.argmax(dim=1)[0].cpu().numpy()
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask.astype(np.int32),
                                    (gt_mask.shape[1], gt_mask.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
        img_inter = 0.0; img_union = 0.0
        for cid in range(C):
            gt_bin = (gt_mask == cid)
            pr_bin = (pred_mask == cid)
            inter = float(np.logical_and(gt_bin, pr_bin).sum())
            uni = float(np.logical_or(gt_bin, pr_bin).sum())
            intersect[cid] += inter
            union[cid] += uni
            img_inter += inter; img_union += uni
        per_image_miou.append({
            "idx": int(real_idx), "path": raw.get("path", ""),
            "miou": float(img_inter / img_union) if img_union > 0 else 0.0,
        })

    ious = np.where(union > 0, intersect / (union + 1e-9), 0.0)
    summary = {
        "task": "segmentation",
        "mean_iou": round(float(ious.mean()), 4),
        "per_class_iou": {class_names.get(i, str(i)): round(float(ious[i]), 4) for i in range(C)},
    }
    _write_json_md(
        output_dir / "summary.json", output_dir / "summary.md",
        summary,
        title="Segmentation Error Analysis",
        header=[f"- Samples analyzed: **{len(per_image_miou)}**",
                f"- mean IoU: **{summary['mean_iou']:.4f}**"],
    )

    # per_class_iou chart
    fig, ax = plt.subplots(figsize=(max(7, C * 0.9), 5))
    names = [class_names.get(i, str(i)) for i in range(C)]
    ax.bar(range(C), ious, color="#4c72b0")
    ax.set_xticks(range(C)); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1.0); ax.set_ylabel("IoU")
    ax.set_title("Per-class IoU")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    (output_dir / "per_class_iou.png").parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_dir / "per_class_iou.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # hard images gallery: lowest mIoU
    artifacts = {
        "summary_json": output_dir / "summary.json",
        "summary_md": output_dir / "summary.md",
        "per_class_iou": output_dir / "per_class_iou.png",
    }
    if hard_images_per_class > 0:
        per_image_miou.sort(key=lambda r: r["miou"])
        root = output_dir / "hard_images" / "worst_miou"
        root.mkdir(parents=True, exist_ok=True)
        cap = hard_images_per_class
        for rec in per_image_miou[:cap]:
            try:
                raw = raw_ds.get_raw_item(rec["idx"])
            except Exception:
                continue
            image = raw["image"]; gt_mask = raw.get("targets")
            if image is None:
                continue
            tensor = _preprocess_for_model(image, (input_h, input_w), model=model).unsqueeze(0).to(device)
            with torch.no_grad():
                out = _dispatch_forward(model, tensor)
            logits = out.logits if hasattr(out, "logits") else out
            pred_mask = logits.argmax(dim=1)[0].cpu().numpy()
            if gt_mask is not None and pred_mask.shape != gt_mask.shape:
                pred_mask = cv2.resize(pred_mask.astype(np.int32),
                                       (gt_mask.shape[1], gt_mask.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
            overlay = _blend_seg_for_analysis(image, gt_mask, pred_mask, style)
            stem = Path(rec["path"]).stem or f"img_{rec['idx']}"
            out_p = root / f"{_safe_name(stem)}__miou_{rec['miou']:.2f}.png"
            cv2.imwrite(str(out_p), overlay)
        artifacts["hard_images_root"] = output_dir / "hard_images"

    return artifacts


def _blend_seg_for_analysis(image, gt_mask, pred_mask, style: VizStyle) -> np.ndarray:
    overlay = image.copy()
    if gt_mask is not None and gt_mask.any():
        mask3 = (gt_mask > 0).astype(np.uint8)[..., None]
        tint = np.array(style.gt_color_rgb[::-1], dtype=np.uint8)
        overlay = np.where(mask3,
                           (overlay * (1 - style.mask_alpha) + tint * style.mask_alpha).astype(np.uint8),
                           overlay)
    if pred_mask is not None and pred_mask.any():
        mask3 = (pred_mask > 0).astype(np.uint8)[..., None]
        tint = np.array(style.pred_color_rgb[::-1], dtype=np.uint8)
        overlay = np.where(mask3,
                           (overlay * (1 - style.mask_alpha) + tint * style.mask_alpha).astype(np.uint8),
                           overlay)
    return overlay


# ===========================================================================
# KEYPOINT
# ===========================================================================


def _analyze_keypoint(
    *, model, dataset, output_dir: Path,
    class_names: dict[int, str], input_size: tuple[int, int], style: VizStyle,
    conf_threshold: float,
    max_samples: int | None, hard_images_per_class: int,
) -> dict[str, Any]:
    """PCK@0.2 per keypoint — visibility-gated."""
    raw_ds, idx_map = _unwrap(dataset)
    n = len(dataset)
    indices = list(range(n)) if max_samples is None or max_samples >= n else \
        sorted(np.random.default_rng(0).choice(n, size=max_samples, replace=False).tolist())
    device = next(model.parameters()).device
    input_h, input_w = int(input_size[0]), int(input_size[1])

    per_kp_correct: dict[int, int] = {}
    per_kp_total: dict[int, int] = {}
    per_image_pck: list[dict] = []

    for ds_idx in indices:
        real_idx = idx_map(ds_idx)
        try:
            raw = raw_ds.get_raw_item(real_idx)
        except Exception:
            continue
        image = raw["image"]; gt = raw.get("targets")
        if image is None or gt is None:
            continue
        gt_kp = np.asarray(gt.get("keypoints") if isinstance(gt, dict) else gt,
                           dtype=np.float32).reshape(-1, 3)
        if gt_kp.size == 0:
            continue
        tensor = _preprocess_for_model(image, (input_h, input_w), model=model).unsqueeze(0).to(device)
        with torch.no_grad():
            preds_raw = _dispatch_forward(model, tensor)
        target_sizes = torch.tensor([[input_h, input_w]], device=device)
        decoded = _dispatch_postprocess(model, preds_raw, conf_threshold, target_sizes)[0]
        pred_kp = np.asarray(decoded.get("keypoints", []), dtype=np.float32).reshape(-1, 3)

        # Normalize threshold by GT bbox diagonal (PCK@0.2)
        x_min, y_min = gt_kp[gt_kp[:, 2] > 0, :2].min(axis=0) if (gt_kp[:, 2] > 0).any() else (0, 0)
        x_max, y_max = gt_kp[gt_kp[:, 2] > 0, :2].max(axis=0) if (gt_kp[:, 2] > 0).any() else (1, 1)
        diag = max(1.0, float(np.hypot(x_max - x_min, y_max - y_min)))
        thr = 0.2 * diag

        correct = total = 0
        K = min(len(gt_kp), len(pred_kp)) if pred_kp.size else 0
        for k in range(len(gt_kp)):
            if gt_kp[k, 2] <= 0:
                continue
            per_kp_total[k] = per_kp_total.get(k, 0) + 1
            total += 1
            if k < K and pred_kp[k, 2] > 0:
                d = float(np.hypot(pred_kp[k, 0] - gt_kp[k, 0],
                                    pred_kp[k, 1] - gt_kp[k, 1]))
                if d <= thr:
                    per_kp_correct[k] = per_kp_correct.get(k, 0) + 1
                    correct += 1
        per_image_pck.append({"idx": int(real_idx), "path": raw.get("path", ""),
                               "pck": (correct / max(1, total))})

    # summary
    per_kp = {}
    for k in sorted(per_kp_total):
        c = per_kp_correct.get(k, 0); t = per_kp_total[k]
        per_kp[f"kp_{k}"] = {"pck": round(c / max(1, t), 4), "total": t}
    summary = {
        "task": "keypoint",
        "mean_pck": round(float(np.mean([v["pck"] for v in per_kp.values()])), 4) if per_kp else 0.0,
        "per_keypoint": per_kp,
    }
    _write_json_md(
        output_dir / "summary.json", output_dir / "summary.md",
        summary,
        title="Keypoint Error Analysis",
        header=[f"- Samples analyzed: **{len(per_image_pck)}**",
                f"- Mean PCK@0.2: **{summary['mean_pck']:.4f}**"],
    )

    # per_keypoint_pck chart
    fig, ax = plt.subplots(figsize=(max(7, len(per_kp) * 0.7), 5))
    names = list(per_kp.keys())
    vals = [per_kp[n]["pck"] for n in names]
    ax.bar(range(len(names)), vals, color="#4c72b0")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=60, ha="right")
    ax.set_ylim(0, 1.0); ax.set_ylabel("PCK@0.2")
    ax.set_title("Per-keypoint PCK@0.2")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_dir / "per_keypoint_pck.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    return {
        "summary_json": output_dir / "summary.json",
        "summary_md": output_dir / "summary.md",
        "per_keypoint_pck": output_dir / "per_keypoint_pck.png",
    }


# ===========================================================================
# SHARED I/O
# ===========================================================================


def _write_json_md(json_path: Path, md_path: Path, summary: dict, *, title: str, header: list[str]):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=str)

    lines = [f"# {title}", ""] + header + [""]
    for section, payload in summary.items():
        if section in ("task",):
            continue
        lines.append(f"## {section}")
        lines.append("")
        if isinstance(payload, dict):
            # flat one-level table
            for k, v in payload.items():
                if isinstance(v, dict):
                    subs = "  ".join(f"{kk}={vv}" for kk, vv in v.items())
                    lines.append(f"- **{k}** — {subs}")
                else:
                    lines.append(f"- **{k}**: {v}")
        else:
            lines.append(str(payload))
        lines.append("")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w") as f:
        f.write("\n".join(lines))

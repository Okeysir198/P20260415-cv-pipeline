"""Cleanlab-style label-quality analyzer.

Surfaces samples where the trained model is **confident** but disagrees
with the ground truth — the canonical signal for label noise. Per-class
estimated noise rates + a gallery of suspected mislabels + a CSV
exportable to Label Studio for human review.

Tasks supported:
- ``classification`` — flag samples where ``top1_score > 0.8`` and ``pred ≠ gt``.
- ``segmentation``   — flag samples where ``frac_pixels(pred_score > 0.9 AND
                       pred_class ≠ gt_class)`` is high.
- ``detection``      — match each GT box to its highest-IoU pred (IoU > 0.7);
                       flag pairs where ``pred.score > 0.7`` and ``pred.class
                       ≠ gt.class`` (probable class-label noise) plus pred
                       boxes with ``pred.score > 0.8`` that match no GT box
                       (probable missed-annotation).

Outputs (flat, under the error_analysis dir passed as ``output_dir``):
- ``04_label_quality.png`` — per-class estimated noise rate bars.
- ``04_label_quality_gallery.png`` — 16-cell GT|Pred grid of the most-suspect
  samples.
- ``04_label_quality.json`` — per-class noise + per-sample scores.
- ``04_suspected_mislabels.csv`` — Label Studio re-import CSV.

The signal is "**suspected** mislabels", never "definite" — humans must
confirm. Thresholds intentionally conservative to keep false-positive rate
low. See ``feedback_three_diagnostic_dimensions.md`` for the rationale.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from core.p10_inference.supervision_bridge import (
    VizStyle,
    render_gt_pred_side_by_side,
)
from utils.viz import save_image_grid

from loguru import logger

matplotlib.use("Agg")


# Filename map — keep parallel to error_analysis_runner.CHART_FILENAMES so
# downstream report wiring can index them the same way.
LQ_FILENAMES: dict[str, str] = {
    "label_noise_estimate":         "04_label_quality.png",
    "confident_disagreement_gallery": "04_label_quality_gallery.png",
    "json":                          "04_label_quality.json",
    "csv":                           "04_suspected_mislabels.csv",
}

# Conservative thresholds — keep FP rate of "suspected mislabel" flag low.
_CLS_CONFIDENCE_THRESH = 0.80
_DET_SCORE_THRESH = 0.70
_DET_IOU_THRESH = 0.70
_SEG_PIXEL_SCORE_THRESH = 0.90
# Per-image suspect threshold for segmentation: image flagged if more than
# this fraction of non-ignore pixels are confident-disagreement.
_SEG_PIXEL_FRAC_THRESH = 0.05
_GALLERY_MAX = 16


def analyze_label_quality(
    *,
    model,
    dataset,
    output_dir: Path | str,
    task: str,
    class_names: dict[int, str],
    input_size: tuple[int, int],
    style: VizStyle | None = None,
    max_samples: int | None = 500,
    ignore_index: int | None = None,
) -> dict[str, Any]:
    """Run task-aware label-quality analysis. Returns dict of artefact paths.

    The dispatcher calls this after the main ``run_error_analysis`` step;
    it can also be invoked standalone given a model + dataset.
    """
    output_dir = Path(output_dir)
    style = style or VizStyle()

    task = (task or "detection").lower()
    out_dir = output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if task == "segmentation":
        result = _analyze_segmentation(
            model, dataset, out_dir, class_names, input_size,
            style=style, max_samples=max_samples,
            ignore_index=ignore_index if ignore_index is not None else 0,
        )
    elif task == "classification":
        result = _analyze_classification(
            model, dataset, out_dir, class_names, input_size,
            style=style, max_samples=max_samples,
        )
    elif task == "detection":
        result = _analyze_detection(
            model, dataset, out_dir, class_names, input_size,
            style=style, max_samples=max_samples,
        )
    else:
        logger.info("analyze_label_quality: task=%s not supported; skipping", task)
        return {}

    return result


# ---------------------------------------------------------------------------
# Task-specific analyzers
# ---------------------------------------------------------------------------


def _analyze_segmentation(
    model,
    dataset,
    out_dir: Path,
    class_names: dict[int, str],
    input_size: tuple[int, int],
    *,
    style: VizStyle,
    max_samples: int | None,
    ignore_index: int,
) -> dict[str, Any]:
    """Confident-disagreement at the pixel level, ranked per-image."""
    n = len(dataset)
    if max_samples is not None and n > max_samples:
        idxs = np.random.RandomState(42).choice(n, max_samples, replace=False).tolist()
        idxs = sorted(idxs)
    else:
        idxs = list(range(n))

    device = next(model.parameters()).device

    per_image: list[dict] = []
    per_class_disagree_pixels: dict[int, int] = {}
    per_class_total_pixels: dict[int, int] = {}

    for idx in idxs:
        try:
            raw = _safe_get_raw_item(dataset, idx)
            if raw is None or raw.get("image") is None:
                continue
            image_bgr = raw["image"]
            gt_mask = raw.get("targets")
            path = raw.get("path", str(idx))
            pred_mask, pred_max_score = _seg_forward(
                model, image_bgr, input_size, device,
            )
            # Resize gt_mask to pred resolution for fair comparison.
            if gt_mask is None:
                continue
            gt_mask = np.asarray(gt_mask)
            if gt_mask.shape != pred_mask.shape:
                gt_mask = cv2.resize(
                    gt_mask.astype(np.uint8),
                    (pred_mask.shape[1], pred_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            valid = gt_mask != ignore_index
            confident = pred_max_score >= _SEG_PIXEL_SCORE_THRESH
            disagree = pred_mask != gt_mask
            mask_flag = valid & confident & disagree

            n_valid = int(valid.sum())
            n_flag = int(mask_flag.sum())
            frac = float(n_flag) / max(n_valid, 1)

            # Per-class accumulation: indexed by GT class (the class the
            # annotator wrote that the model disagrees with at high conf).
            for cid in np.unique(gt_mask[mask_flag]):
                per_class_disagree_pixels[int(cid)] = (
                    per_class_disagree_pixels.get(int(cid), 0)
                    + int(((gt_mask == cid) & mask_flag).sum())
                )
            for cid in np.unique(gt_mask[valid]):
                per_class_total_pixels[int(cid)] = (
                    per_class_total_pixels.get(int(cid), 0)
                    + int(((gt_mask == cid) & valid).sum())
                )

            # Best (most-disagreement) pred class for the gallery banner.
            top_pred_cid = -1
            if n_flag > 0:
                vals, counts = np.unique(pred_mask[mask_flag], return_counts=True)
                top_pred_cid = int(vals[counts.argmax()])
            per_image.append({
                "image_idx": idx,
                "image_bgr": image_bgr,
                "gt_mask": gt_mask,
                "pred_mask": pred_mask,
                "frac": frac,
                "n_flag": n_flag,
                "top_pred_cid": top_pred_cid,
                "stem": Path(path).stem or f"img_{idx}",
            })
        except Exception as e:  # pragma: no cover
            logger.warning("analyze_label_quality[seg]: idx %d failed — %s", idx, e)

    artifacts: dict[str, Any] = {}
    if not per_image:
        logger.info("analyze_label_quality[seg]: no analyzable samples")
        return artifacts

    # ---- per-class noise rate bars ----
    per_class_rate = {
        class_names.get(cid, str(cid)): (
            per_class_disagree_pixels.get(cid, 0) / max(per_class_total_pixels.get(cid, 1), 1)
        )
        for cid in sorted(per_class_total_pixels)
    }
    artifacts["label_noise_estimate"] = _plot_noise_bars(
        per_class_rate,
        out_dir / LQ_FILENAMES["label_noise_estimate"],
        title="Estimated label noise per class — pixel-level confident disagreement",
        ylabel="% pixels (pred conf > 0.9, pred ≠ gt)",
    )

    # ---- confident-disagreement gallery ----
    suspect = [r for r in per_image if r["frac"] >= _SEG_PIXEL_FRAC_THRESH]
    suspect.sort(key=lambda r: r["frac"], reverse=True)
    suspect = suspect[:_GALLERY_MAX]
    annotated: list[np.ndarray] = []
    for rec in suspect:
        try:
            image_rgb = cv2.cvtColor(rec["image_bgr"], cv2.COLOR_BGR2RGB)
            panel_rgb = render_gt_pred_side_by_side(
                image_rgb, rec["gt_mask"], rec["pred_mask"],
                task="segmentation", class_names=class_names, style=style,
                banner={
                    "title": f"suspected mislabel: {rec['frac']:.0%} pixels disagree",
                    "subtitle": f"top pred class: {class_names.get(rec['top_pred_cid'], '?')}",
                },
            )
            annotated.append(cv2.cvtColor(panel_rgb, cv2.COLOR_RGB2BGR))
        except Exception as e:  # pragma: no cover
            logger.warning("analyze_label_quality[seg]: gallery render failed — %s", e)
    if annotated:
        rgb_imgs = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in annotated]
        out_path = out_dir / LQ_FILENAMES["confident_disagreement_gallery"]
        save_image_grid(rgb_imgs, out_path, cols=min(2, len(rgb_imgs)),
                        header=f"Suspected mislabels — {len(annotated)} samples")
        artifacts["confident_disagreement_gallery"] = out_path

    # ---- exports ----
    json_payload = {
        "task": "segmentation",
        "n_samples_analyzed": len(per_image),
        "per_class_pixel_noise_rate": per_class_rate,
        "n_suspected": len(suspect),
        "suspect_threshold": _SEG_PIXEL_FRAC_THRESH,
    }
    (out_dir / LQ_FILENAMES["json"]).write_text(
        json.dumps(json_payload, indent=2, default=str)
    )
    _write_label_studio_csv(
        out_dir / LQ_FILENAMES["csv"],
        rows=[{
            "stem": rec["stem"],
            "frac_disagree": f"{rec['frac']:.4f}",
            "n_flagged_pixels": rec["n_flag"],
            "top_pred_class": class_names.get(rec["top_pred_cid"], str(rec["top_pred_cid"])),
        } for rec in suspect],
        fieldnames=["stem", "frac_disagree", "n_flagged_pixels", "top_pred_class"],
    )
    artifacts["json"] = out_dir / LQ_FILENAMES["json"]
    artifacts["csv"] = out_dir / LQ_FILENAMES["csv"]

    chart_metrics = _seg_chart_metrics(per_class_rate, suspect, len(per_image))
    return {"artifacts": artifacts, "chart_metrics": chart_metrics}


def _analyze_classification(
    model,
    dataset,
    out_dir: Path,
    class_names: dict[int, str],
    input_size: tuple[int, int],
    *,
    style: VizStyle,
    max_samples: int | None,
) -> dict[str, Any]:
    """Confident-disagreement at the sample level: top1_score >= 0.8 AND pred != gt."""
    n = len(dataset)
    if max_samples is not None and n > max_samples:
        idxs = np.random.RandomState(42).choice(n, max_samples, replace=False).tolist()
        idxs = sorted(idxs)
    else:
        idxs = list(range(n))

    device = next(model.parameters()).device
    per_class_total: dict[int, int] = {}
    per_class_suspect: dict[int, int] = {}
    suspect: list[dict] = []

    for idx in idxs:
        try:
            raw = _safe_get_raw_item(dataset, idx)
            if raw is None or raw.get("image") is None:
                continue
            image_bgr = raw["image"]
            gt = raw.get("targets")
            try:
                gt_cid = int(gt)
            except (TypeError, ValueError):
                continue
            path = raw.get("path", str(idx))
            pred_cid, pred_score = _cls_forward(
                model, image_bgr, input_size, device,
            )
            per_class_total[gt_cid] = per_class_total.get(gt_cid, 0) + 1
            if pred_cid != gt_cid and pred_score >= _CLS_CONFIDENCE_THRESH:
                per_class_suspect[gt_cid] = per_class_suspect.get(gt_cid, 0) + 1
                suspect.append({
                    "image_idx": idx,
                    "image_bgr": image_bgr,
                    "gt_cid": gt_cid,
                    "pred_cid": pred_cid,
                    "pred_score": float(pred_score),
                    "stem": Path(path).stem or f"img_{idx}",
                })
        except Exception as e:  # pragma: no cover
            logger.warning("analyze_label_quality[cls]: idx %d failed — %s", idx, e)

    artifacts: dict[str, Any] = {}
    if not per_class_total:
        return artifacts

    per_class_rate = {
        class_names.get(cid, str(cid)): per_class_suspect.get(cid, 0) / max(total, 1)
        for cid, total in per_class_total.items()
    }
    artifacts["label_noise_estimate"] = _plot_noise_bars(
        per_class_rate,
        out_dir / LQ_FILENAMES["label_noise_estimate"],
        title="Estimated label noise per class — high-confidence wrong predictions",
        ylabel="% samples (pred conf > 0.8, pred ≠ gt)",
    )

    suspect.sort(key=lambda r: r["pred_score"], reverse=True)
    suspect = suspect[:_GALLERY_MAX]
    annotated: list[np.ndarray] = []
    for rec in suspect:
        try:
            image_rgb = cv2.cvtColor(rec["image_bgr"], cv2.COLOR_BGR2RGB)
            panel_rgb = render_gt_pred_side_by_side(
                image_rgb, rec["gt_cid"], (rec["pred_cid"], rec["pred_score"]),
                task="classification", class_names=class_names, style=style,
                banner={
                    "title": "suspected mislabel",
                    "subtitle": f"model {class_names.get(rec['pred_cid'], '?')} @ {rec['pred_score']:.2f} vs label {class_names.get(rec['gt_cid'], '?')}",
                },
            )
            annotated.append(cv2.cvtColor(panel_rgb, cv2.COLOR_RGB2BGR))
        except Exception as e:  # pragma: no cover
            logger.warning("analyze_label_quality[cls]: gallery render failed — %s", e)
    if annotated:
        rgb_imgs = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in annotated]
        out_path = out_dir / LQ_FILENAMES["confident_disagreement_gallery"]
        save_image_grid(rgb_imgs, out_path, cols=min(2, len(rgb_imgs)),
                        header=f"Suspected mislabels — {len(annotated)} samples")
        artifacts["confident_disagreement_gallery"] = out_path

    json_payload = {
        "task": "classification",
        "n_samples_analyzed": sum(per_class_total.values()),
        "per_class_noise_rate": per_class_rate,
        "n_suspected": len(suspect),
        "suspect_threshold": _CLS_CONFIDENCE_THRESH,
    }
    (out_dir / LQ_FILENAMES["json"]).write_text(
        json.dumps(json_payload, indent=2, default=str)
    )
    _write_label_studio_csv(
        out_dir / LQ_FILENAMES["csv"],
        rows=[{
            "stem": rec["stem"],
            "gt_class": class_names.get(rec["gt_cid"], str(rec["gt_cid"])),
            "pred_class": class_names.get(rec["pred_cid"], str(rec["pred_cid"])),
            "pred_score": f"{rec['pred_score']:.4f}",
        } for rec in suspect],
        fieldnames=["stem", "gt_class", "pred_class", "pred_score"],
    )
    artifacts["json"] = out_dir / LQ_FILENAMES["json"]
    artifacts["csv"] = out_dir / LQ_FILENAMES["csv"]

    chart_metrics = _cls_chart_metrics(per_class_rate, suspect, sum(per_class_total.values()))
    return {"artifacts": artifacts, "chart_metrics": chart_metrics}


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
#
# MVP heuristic (deliberately simple — full Cleanlab-style per-box confident-
# disagreement matching needs the predictor+matcher chain we don't want to
# duplicate here). For each image we compute a per-image label-quality score:
#
#     score = mean(IoU(pred*, gt)) over all GT boxes
#
# where pred* is the highest-IoU prediction for that GT box (any class). Low
# score → model thinks the GT boxes are in the wrong place, suggesting bad
# annotation. Images below `_DET_LOW_SCORE` are flagged as suspected mislabels.
# This is conservative: false-positive rate dominates only on classes the model
# truly cannot detect (which is also a useful signal).
#
# Follow-up (not blocking): per-GT class-confusion + missed-annotation flags
# via the same matcher used in error_analysis_runner.

_DET_LOW_SCORE = 0.30
_DET_PRED_CONF = 0.30


def _analyze_detection(
    model,
    dataset,
    out_dir: Path,
    class_names: dict[int, str],
    input_size: tuple[int, int],
    *,
    style: VizStyle,
    max_samples: int | None,
) -> dict[str, Any]:
    """Per-image label-quality score = mean best-IoU(pred, gt) across GT boxes."""
    from core.p06_training._common import yolo_targets_to_xyxy as _gt_to_xyxy
    from core.p06_training.post_train import _forward_batch_detection
    from core.p08_evaluation.error_analysis_runner import _preprocess_for_model
    from core.p10_inference.supervision_bridge import render_gt_pred_side_by_side

    n = len(dataset)
    if max_samples is not None and n > max_samples:
        idxs = sorted(np.random.RandomState(42).choice(n, max_samples, replace=False).tolist())
    else:
        idxs = list(range(n))

    device = next(model.parameters()).device
    h_in, w_in = int(input_size[0]), int(input_size[1])
    per_image: list[dict] = []

    for idx in idxs:
        try:
            raw = _safe_get_raw_item(dataset, idx)
            if raw is None or raw.get("image") is None:
                continue
            img = raw["image"]
            gt = raw.get("targets")
            if not isinstance(gt, np.ndarray) or gt.size == 0:
                continue
            H, W = img.shape[:2]
            tensor = _preprocess_for_model(img, (h_in, w_in), model=model)
            target_size = torch.tensor([[h_in, w_in]], dtype=torch.int64)
            preds = _forward_batch_detection(
                model, [tensor], target_size, conf_threshold=_DET_PRED_CONF,
            )
            p = preds[0] if preds else {}
            pred_boxes = np.asarray(p.get("boxes", []), dtype=np.float64).reshape(-1, 4)
            if len(pred_boxes) > 0:
                pred_boxes[:, [0, 2]] *= W / w_in
                pred_boxes[:, [1, 3]] *= H / h_in
            gt_xyxy, gt_cls = _gt_to_xyxy(gt, W, H)
            gt_xyxy = np.asarray(gt_xyxy).reshape(-1, 4)
            if gt_xyxy.size == 0:
                continue
            best_ious = _per_gt_best_iou(gt_xyxy, pred_boxes)
            score = float(best_ious.mean()) if best_ious.size else 0.0
            per_image.append({
                "image_idx": idx,
                "image_bgr": img,
                "score": score,
                "n_gt": int(gt_xyxy.shape[0]),
                "n_pred": int(pred_boxes.shape[0]),
                "gt_xyxy": gt_xyxy,
                "gt_cls": np.asarray(gt_cls, dtype=np.int64).ravel(),
                "pred_boxes": pred_boxes,
                "pred_labels": np.asarray(p.get("labels", []), dtype=np.int64).ravel(),
                "pred_scores": np.asarray(p.get("scores", []), dtype=np.float64).ravel(),
                "stem": Path(raw.get("path", str(idx))).stem or f"img_{idx}",
            })
        except Exception as e:  # pragma: no cover
            logger.warning("analyze_label_quality[det]: idx %d failed — %s", idx, e)

    artifacts: dict[str, Any] = {}
    if not per_image:
        return artifacts

    # Histogram of per-image scores with threshold line.
    scores = np.array([r["score"] for r in per_image])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores, bins=20, range=(0, 1), color="#1f77b4", edgecolor="white")
    ax.axvline(_DET_LOW_SCORE, color="#d62728", linestyle="--",
               label=f"suspect threshold = {_DET_LOW_SCORE}")
    ax.set_xlabel("per-image label-quality score (mean best-IoU pred↔gt)")
    ax.set_ylabel("# images")
    ax.set_title("Detection label quality — per-image score histogram")
    ax.legend()
    plt.tight_layout()
    out_path = out_dir / LQ_FILENAMES["label_noise_estimate"]
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    artifacts["label_noise_estimate"] = out_path

    suspect = [r for r in per_image if r["score"] < _DET_LOW_SCORE]
    suspect.sort(key=lambda r: r["score"])
    suspect = suspect[:_GALLERY_MAX]

    import supervision as sv
    annotated: list[np.ndarray] = []
    for rec in suspect:
        try:
            img_rgb = cv2.cvtColor(rec["image_bgr"], cv2.COLOR_BGR2RGB)
            pred_dets = sv.Detections(
                xyxy=rec["pred_boxes"].astype(np.float32),
                class_id=rec["pred_labels"].astype(int),
                confidence=rec["pred_scores"].astype(np.float32),
            ) if rec["pred_boxes"].size > 0 else sv.Detections.empty()
            panel_rgb = render_gt_pred_side_by_side(
                img_rgb,
                (rec["gt_xyxy"], rec["gt_cls"]),
                pred_dets,
                task="detection", class_names=class_names, style=style,
                banner={
                    "title": f"suspected mislabel — score {rec['score']:.2f}",
                    "subtitle": f"{rec['n_gt']} GT vs {rec['n_pred']} pred",
                },
            )
            annotated.append(cv2.cvtColor(panel_rgb, cv2.COLOR_RGB2BGR))
        except Exception as e:  # pragma: no cover
            logger.warning("analyze_label_quality[det]: gallery render failed — %s", e)
    if annotated:
        rgb_imgs = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in annotated]
        gallery_path = out_dir / LQ_FILENAMES["confident_disagreement_gallery"]
        save_image_grid(rgb_imgs, gallery_path, cols=min(2, len(rgb_imgs)),
                        header=f"Suspected mislabels — {len(annotated)} samples")
        artifacts["confident_disagreement_gallery"] = gallery_path

    json_payload = {
        "task": "detection",
        "n_samples_analyzed": len(per_image),
        "score_mean": float(scores.mean()),
        "score_median": float(np.median(scores)),
        "n_suspected": len(suspect),
        "suspect_threshold": _DET_LOW_SCORE,
    }
    (out_dir / LQ_FILENAMES["json"]).write_text(
        json.dumps(json_payload, indent=2, default=str)
    )
    _write_label_studio_csv(
        out_dir / LQ_FILENAMES["csv"],
        rows=[{
            "stem": rec["stem"],
            "score": f"{rec['score']:.4f}",
            "n_gt": rec["n_gt"],
            "n_pred": rec["n_pred"],
        } for rec in suspect],
        fieldnames=["stem", "score", "n_gt", "n_pred"],
    )
    artifacts["json"] = out_dir / LQ_FILENAMES["json"]
    artifacts["csv"] = out_dir / LQ_FILENAMES["csv"]

    chart_metrics = {
        "04_label_quality": {
            "score_mean": float(scores.mean()),
            "score_median": float(np.median(scores)),
        },
        "04_label_quality_gallery": {
            "n_suspected": len(suspect),
            "n_total": len(per_image),
            "suspect_frac": len(suspect) / max(len(per_image), 1),
        },
    }
    return {"artifacts": artifacts, "chart_metrics": chart_metrics}


def _per_gt_best_iou(gt_xyxy: np.ndarray, pred_xyxy: np.ndarray) -> np.ndarray:
    """For each GT box, return the highest IoU over any pred box (0 if no pred)."""
    if pred_xyxy.size == 0 or gt_xyxy.size == 0:
        return np.zeros(gt_xyxy.shape[0], dtype=np.float64)
    gt = gt_xyxy.astype(np.float64)
    pr = pred_xyxy.astype(np.float64)
    # Pairwise IoU (G, P)
    g_x1, g_y1, g_x2, g_y2 = gt[:, 0:1], gt[:, 1:2], gt[:, 2:3], gt[:, 3:4]
    p_x1, p_y1, p_x2, p_y2 = pr[:, 0], pr[:, 1], pr[:, 2], pr[:, 3]
    ix1 = np.maximum(g_x1, p_x1)
    iy1 = np.maximum(g_y1, p_y1)
    ix2 = np.minimum(g_x2, p_x2)
    iy2 = np.minimum(g_y2, p_y2)
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    ga = (g_x2 - g_x1) * (g_y2 - g_y1)
    pa = (p_x2 - p_x1) * (p_y2 - p_y1)
    union = ga + pa - inter + 1e-9
    iou = inter / union  # (G, P)
    return iou.max(axis=1)


# ---------------------------------------------------------------------------
# Forward helpers (per task)
# ---------------------------------------------------------------------------


def _seg_forward(
    model, image_bgr: np.ndarray, input_size: tuple[int, int], device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a seg forward and return (pred_mask, pred_max_score) at native resolution."""
    from core.p08_evaluation.error_analysis_runner import _preprocess_for_model
    h, w = image_bgr.shape[:2]
    x = _preprocess_for_model(image_bgr, input_size, model=model).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
    logits = _extract_logits(out)
    if logits.ndim == 4:
        # (B, C, h, w) — bilinear upsample then softmax to native resolution.
        logits = torch.nn.functional.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False,
        )
        probs = torch.softmax(logits, dim=1)[0]
    elif logits.ndim == 3:
        probs = torch.softmax(logits, dim=0)
    else:
        raise ValueError(f"unsupported seg logits ndim {logits.ndim}")
    pred_mask = probs.argmax(dim=0).cpu().numpy().astype(np.int64)
    pred_max_score = probs.max(dim=0).values.cpu().numpy().astype(np.float32)
    return pred_mask, pred_max_score


def _cls_forward(
    model, image_bgr: np.ndarray, input_size: tuple[int, int], device,
) -> tuple[int, float]:
    from core.p08_evaluation.error_analysis_runner import _preprocess_for_model
    x = _preprocess_for_model(image_bgr, input_size, model=model).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
    logits = _extract_logits(out)
    if logits.ndim == 2:
        probs = torch.softmax(logits[0], dim=0)
    elif logits.ndim == 1:
        probs = torch.softmax(logits, dim=0)
    else:
        raise ValueError(f"unsupported cls logits ndim {logits.ndim}")
    pred_cid = int(probs.argmax().item())
    pred_score = float(probs.max().item())
    return pred_cid, pred_score


def _extract_logits(out) -> torch.Tensor:
    """Pull a logits tensor out of a model output (raw tensor / dict / HF
    ``ModelOutput`` with ``.logits``)."""
    if torch.is_tensor(out):
        return out
    logits = getattr(out, "logits", None)
    if logits is not None:
        return logits
    if isinstance(out, dict) and "logits" in out:
        return out["logits"]
    raise ValueError(f"cannot extract logits from {type(out)}")


def _safe_get_raw_item(dataset, idx: int) -> dict | None:
    """Reach through a torch Subset to call get_raw_item on the inner dataset."""
    inner = getattr(dataset, "dataset", dataset)
    indices = getattr(dataset, "indices", None)
    real_idx = indices[idx] if indices is not None else idx
    if hasattr(inner, "get_raw_item"):
        return inner.get_raw_item(real_idx)
    return None


# ---------------------------------------------------------------------------
# Render + IO helpers
# ---------------------------------------------------------------------------


def _plot_noise_bars(
    per_class_rate: dict[str, float], out_path: Path, *, title: str, ylabel: str,
) -> Path:
    """Horizontal-bar chart of per-class noise rates."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    items = sorted(per_class_rate.items(), key=lambda kv: kv[1], reverse=True)
    classes = [k for k, _ in items]
    rates = [v * 100.0 for _, v in items]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(classes) + 1)))
    bars = ax.barh(range(len(classes)), rates,
                   color=["#d62728" if r >= 10 else "#1f77b4" for r in rates])
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(ylabel)
    ax.set_title(title, fontsize=11)
    for bar, r in zip(bars, rates, strict=False):
        if r > 0:
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{r:.1f}%", va="center", fontsize=7)
    ax.set_xlim(0, max(max(rates), 1.0) * 1.12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _write_label_studio_csv(
    out_path: Path, *, rows: list[dict], fieldnames: list[str],
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_path


# ---------------------------------------------------------------------------
# Chart-meta payload builders
# ---------------------------------------------------------------------------


def _seg_chart_metrics(
    per_class_rate: dict[str, float], suspect: list[dict], n_total: int,
) -> dict[str, dict]:
    if per_class_rate:
        worst_class, worst_rate = max(per_class_rate.items(), key=lambda kv: kv[1])
    else:
        worst_class, worst_rate = "?", 0.0
    return {
        "04_label_quality": {
            "worst_class": worst_class,
            "worst_rate": worst_rate,
        },
        "04_label_quality_gallery": {
            "n_suspected": len(suspect),
            "n_total": n_total,
            "suspect_frac": len(suspect) / max(n_total, 1),
        },
    }


def _cls_chart_metrics(
    per_class_rate: dict[str, float], suspect: list[dict], n_total: int,
) -> dict[str, dict]:
    if per_class_rate:
        worst_class, worst_rate = max(per_class_rate.items(), key=lambda kv: kv[1])
    else:
        worst_class, worst_rate = "?", 0.0
    return {
        "04_label_quality": {
            "worst_class": worst_class,
            "worst_rate": worst_rate,
        },
        "04_label_quality_gallery": {
            "n_suspected": len(suspect),
            "n_total": n_total,
            "suspect_frac": len(suspect) / max(n_total, 1),
        },
    }

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

from core.p10_inference.supervision_bridge import VizStyle, annotate_gt_pred

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_name(name: str) -> str:
    """Filesystem-safe version of a class/image name."""
    return _SAFE_NAME.sub("_", str(name))[:80]


def _unwrap(ds):
    if hasattr(ds, "indices") and hasattr(ds, "dataset"):
        return ds.dataset, (lambda i: ds.indices[i])
    return ds, (lambda i: i)


def _preprocess_for_model(raw_image: np.ndarray, input_size: tuple[int, int]) -> torch.Tensor:
    """BGR HWC uint8 → normalized CHW float32 tensor. The HF processor or
    our transform pipeline normally handles this; for the error analysis path
    we replicate the minimum the model expects (resize + /255 + CHW)."""
    h, w = int(input_size[0]), int(input_size[1])
    resized = cv2.resize(raw_image, (w, h))
    tensor = torch.from_numpy(
        np.ascontiguousarray(
            (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)
        )
    )
    return tensor


def _dispatch_forward(model, tensor_batch: torch.Tensor):
    """Unified forward for the analyzer — never raises on wrapper type."""
    if hasattr(model, "hf_model"):
        return model(pixel_values=tensor_batch)
    return model(tensor_batch)


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
    fp_gallery: dict[int, list[dict]] = {c: [] for c in class_names}
    fn_gallery: dict[int, list[dict]] = {c: [] for c in class_names}
    conf_gallery: dict[tuple[int, int], list[dict]] = {}

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
        tensor = _preprocess_for_model(image, (input_h, input_w)).unsqueeze(0).to(device)
        with torch.no_grad():
            preds_raw = _dispatch_forward(model, tensor)
        target_sizes = torch.tensor([[input_h, input_w]], device=device)
        if not hasattr(model, "postprocess"):
            continue
        decoded = model.postprocess(preds_raw, conf_threshold, target_sizes)[0]

        pb = np.asarray(decoded.get("boxes", []), dtype=np.float64).reshape(-1, 4)
        pl = np.asarray(decoded.get("labels", []), dtype=np.int64).ravel()
        ps = np.asarray(decoded.get("scores", []), dtype=np.float64).ravel()

        # Rescale pred boxes to original resolution for consistent match/draw
        if len(pb) > 0:
            pb[:, [0, 2]] *= orig_w / input_w
            pb[:, [1, 3]] *= orig_h / input_h

        targets = raw.get("targets")
        gt_xyxy = np.zeros((0, 4), dtype=np.float32)
        gt_cls = np.zeros(0, dtype=np.int64)
        if isinstance(targets, np.ndarray) and targets.size > 0:
            tcx, tcy, tw, th = targets[:, 1], targets[:, 2], targets[:, 3], targets[:, 4]
            gt_xyxy = np.stack([
                (tcx - tw / 2) * orig_w, (tcy - th / 2) * orig_h,
                (tcx + tw / 2) * orig_w, (tcy + th / 2) * orig_h,
            ], axis=1).astype(np.float32)
            gt_cls = targets[:, 0].astype(np.int64)

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
    _write_json_md(
        output_dir / "summary.json", output_dir / "summary.md",
        summary,
        title="Detection Error Analysis",
        header=[
            f"- Samples analyzed: **{len(per_image)}**",
            f"- IoU threshold: {iou_threshold}",
            f"- Confidence threshold: {conf_threshold}",
        ],
    )

    artifacts = {"summary_json": output_dir / "summary.json",
                 "summary_md": output_dir / "summary.md"}

    # ---- charts ----
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

    # ---- hardest images overview grid (top 12 by FP+FN) ----
    per_image.sort(key=lambda r: -(r["fn"] + r["fp"]))
    artifacts["hardest_images"] = _render_hardest_overview(
        raw_ds, per_image[:12], model, class_names, input_size, style,
        output_dir / "hardest_images.png",
    )

    # ---- per-class galleries ----
    if hard_images_per_class > 0:
        gallery_root = output_dir / "hard_images"
        artifacts["hard_images_root"] = gallery_root
        _render_fp_gallery(
            raw_ds, model, fp_gallery, class_names, input_size, style,
            gallery_root / "false_positives", hard_images_per_class,
        )
        _render_fn_gallery(
            raw_ds, model, fn_gallery, class_names, input_size, style,
            gallery_root / "false_negatives", hard_images_per_class,
        )
        _render_confusion_gallery(
            raw_ds, model, conf_gallery, class_names, input_size, style,
            gallery_root / "class_confusion", hard_images_per_class,
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


def _size_category(area: float) -> str:
    if area < 32 * 32:
        return "small"
    if area < 96 * 96:
        return "medium"
    return "large"


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
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(3); w = 0.4
    ax.bar(x - w/2, prec, w, label="Precision", color="#4c72b0")
    ax.bar(x + w/2, rec,  w, label="Recall",    color="#55a868")
    for i, t in enumerate(tiers):
        c = size_stats[t]
        ax.text(i, 1.02, f"TP {c['tp']} FP {c['fp']} FN {c['fn']}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(tiers)
    ax.set_ylim(0, 1.2); ax.set_ylabel("score")
    ax.set_title("Size-stratified recall / precision (COCO tiers)")
    ax.legend(loc="lower right"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _render_hardest_overview(
    raw_ds, hardest: list[dict], model, class_names, input_size, style, path: Path,
) -> Path | None:
    """Grid of the top-N hardest images (full GT vs Pred on each)."""
    if not hardest:
        return None
    input_h, input_w = int(input_size[0]), int(input_size[1])
    device = next(model.parameters()).device
    rows = []
    for rec in hardest:
        idx = rec["idx"]
        try:
            raw = raw_ds.get_raw_item(idx)
        except Exception:
            continue
        image = raw["image"]
        if image is None:
            continue
        orig_h, orig_w = image.shape[:2]
        tensor = _preprocess_for_model(image, (input_h, input_w)).unsqueeze(0).to(device)
        with torch.no_grad():
            preds_raw = _dispatch_forward(model, tensor)
        target_sizes = torch.tensor([[input_h, input_w]], device=device)
        decoded = model.postprocess(preds_raw, 0.3, target_sizes)[0]
        pb = np.asarray(decoded.get("boxes", []), dtype=np.float64).reshape(-1, 4)
        pl = np.asarray(decoded.get("labels", []), dtype=np.int64).ravel()
        ps = np.asarray(decoded.get("scores", []), dtype=np.float64).ravel()
        if len(pb):
            pb[:, [0, 2]] *= orig_w / input_w
            pb[:, [1, 3]] *= orig_h / input_h
        pred_dets = sv.Detections(xyxy=pb, class_id=pl, confidence=ps)

        targets = raw.get("targets")
        gt_xyxy = gt_cls = None
        if isinstance(targets, np.ndarray) and targets.size > 0:
            tcx, tcy, tw, th = targets[:, 1], targets[:, 2], targets[:, 3], targets[:, 4]
            gt_xyxy = np.stack([
                (tcx - tw / 2) * orig_w, (tcy - th / 2) * orig_h,
                (tcx + tw / 2) * orig_w, (tcy + th / 2) * orig_h,
            ], axis=1).astype(np.float32)
            gt_cls = targets[:, 0].astype(np.int64)
        rows.append(annotate_gt_pred(image, gt_xyxy, gt_cls, pred_dets, class_names, style=style))

    from core.p06_training.post_train import _save_grid
    _save_grid(rows, path, "Hardest images (sorted by FP+FN)", ncols=4, dpi=130)
    return path


def _render_fp_gallery(raw_ds, model, fp_gallery, class_names, input_size, style, root: Path, cap: int):
    """Draw one PNG per (class, FP image): full image with all GT + all preds."""
    input_h, input_w = int(input_size[0]), int(input_size[1])
    device = next(model.parameters()).device
    for cid, items in fp_gallery.items():
        items = sorted(items, key=lambda r: -r["score"])[:cap]
        class_dir = root / _safe_name(class_names.get(cid, str(cid)))
        class_dir.mkdir(parents=True, exist_ok=True)
        for rec in items:
            stem = Path(rec["path"]).stem or f"img_{rec['image_idx']}"
            try:
                raw = raw_ds.get_raw_item(rec["image_idx"])
            except Exception:
                continue
            image = raw["image"]
            if image is None:
                continue
            annotated = _annotate_full_image(image, raw, model, class_names, style,
                                             (input_h, input_w), device)
            out = class_dir / f"{_safe_name(stem)}__fp_score_{rec['score']:.2f}.png"
            cv2.imwrite(str(out), annotated)


def _render_fn_gallery(raw_ds, model, fn_gallery, class_names, input_size, style, root: Path, cap: int):
    input_h, input_w = int(input_size[0]), int(input_size[1])
    device = next(model.parameters()).device
    for cid, items in fn_gallery.items():
        items = items[:cap]
        class_dir = root / _safe_name(class_names.get(cid, str(cid)))
        class_dir.mkdir(parents=True, exist_ok=True)
        for rec in items:
            stem = Path(rec["path"]).stem or f"img_{rec['image_idx']}"
            try:
                raw = raw_ds.get_raw_item(rec["image_idx"])
            except Exception:
                continue
            image = raw["image"]
            if image is None:
                continue
            annotated = _annotate_full_image(image, raw, model, class_names, style,
                                             (input_h, input_w), device)
            out = class_dir / f"{_safe_name(stem)}__fn.png"
            cv2.imwrite(str(out), annotated)


def _render_confusion_gallery(raw_ds, model, conf_gallery, class_names, input_size, style, root: Path, cap: int):
    input_h, input_w = int(input_size[0]), int(input_size[1])
    device = next(model.parameters()).device
    for (gt_cid, pred_cid), items in conf_gallery.items():
        items = sorted(items, key=lambda r: -r["iou"])[:cap]
        sub = root / f"{_safe_name(class_names.get(pred_cid, str(pred_cid)))}__from__{_safe_name(class_names.get(gt_cid, str(gt_cid)))}"
        sub.mkdir(parents=True, exist_ok=True)
        for rec in items:
            stem = Path(rec["path"]).stem or f"img_{rec['image_idx']}"
            try:
                raw = raw_ds.get_raw_item(rec["image_idx"])
            except Exception:
                continue
            image = raw["image"]
            if image is None:
                continue
            annotated = _annotate_full_image(image, raw, model, class_names, style,
                                             (input_h, input_w), device)
            out = sub / f"{_safe_name(stem)}__iou_{rec['iou']:.2f}.png"
            cv2.imwrite(str(out), annotated)


def _annotate_full_image(image, raw, model, class_names, style, input_size, device) -> np.ndarray:
    """Run model on image, overlay full GT+Pred via shared annotate_gt_pred."""
    input_h, input_w = int(input_size[0]), int(input_size[1])
    orig_h, orig_w = image.shape[:2]
    tensor = _preprocess_for_model(image, (input_h, input_w)).unsqueeze(0).to(device)
    with torch.no_grad():
        preds_raw = _dispatch_forward(model, tensor)
    target_sizes = torch.tensor([[input_h, input_w]], device=device)
    decoded = model.postprocess(preds_raw, 0.25, target_sizes)[0]
    pb = np.asarray(decoded.get("boxes", []), dtype=np.float64).reshape(-1, 4)
    pl = np.asarray(decoded.get("labels", []), dtype=np.int64).ravel()
    ps = np.asarray(decoded.get("scores", []), dtype=np.float64).ravel()
    if len(pb):
        pb[:, [0, 2]] *= orig_w / input_w
        pb[:, [1, 3]] *= orig_h / input_h
    pred_dets = sv.Detections(xyxy=pb, class_id=pl, confidence=ps)

    targets = raw.get("targets")
    gt_xyxy = gt_cls = None
    if isinstance(targets, np.ndarray) and targets.size > 0:
        tcx, tcy, tw, th = targets[:, 1], targets[:, 2], targets[:, 3], targets[:, 4]
        gt_xyxy = np.stack([
            (tcx - tw / 2) * orig_w, (tcy - th / 2) * orig_h,
            (tcx + tw / 2) * orig_w, (tcy + th / 2) * orig_h,
        ], axis=1).astype(np.float32)
        gt_cls = targets[:, 0].astype(np.int64)
    return annotate_gt_pred(image, gt_xyxy, gt_cls, pred_dets, class_names, style=style)


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
        tensor = _preprocess_for_model(image, (input_h, input_w)).unsqueeze(0).to(device)
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
        tensor = _preprocess_for_model(image, (input_h, input_w)).unsqueeze(0).to(device)
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
            tensor = _preprocess_for_model(image, (input_h, input_w)).unsqueeze(0).to(device)
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
        tensor = _preprocess_for_model(image, (input_h, input_w)).unsqueeze(0).to(device)
        with torch.no_grad():
            preds_raw = _dispatch_forward(model, tensor)
        target_sizes = torch.tensor([[input_h, input_w]], device=device)
        if not hasattr(model, "postprocess"):
            continue
        decoded = model.postprocess(preds_raw, conf_threshold, target_sizes)[0]
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

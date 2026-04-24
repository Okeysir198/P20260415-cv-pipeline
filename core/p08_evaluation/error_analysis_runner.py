"""Task-dispatched error analysis runner — detection / classification /
segmentation / keypoint.

Redesigned artifact set (all tasks, identical numbered prefixes):

    01_overview.png                — headline metrics card
    02_data_distribution.png       — class balance + sample/size breakdown
    03_per_class_performance.png   — per-class P/R/F1 | IoU | PCK
    04_confusion_matrix.png or 04_top_confused_pairs.png
    05_confidence_calibration.png  — score distributions (correct vs wrong)
    06_failure_mode_contribution.png
    07_failure_by_attribute.png    — detection only
    08_hardest_images.png          — top-12 worst-samples overview grid
    09_failure_mode_examples/      — per-mode × per-class GT | Pred PNGs
                                     (only failed samples, via
                                     ``render_gt_pred_side_by_side``)
    10_recoverable_map_vs_iou.png  — detection only
    11_confidence_attribution.png  — detection only
    12_boxes_per_image.png         — detection only
    13_bbox_aspect_ratio.png       — detection only
    14_size_recall.png             — detection only

Entry point: :func:`run_error_analysis`. Every GT-vs-Pred rendering goes
through :func:`render_gt_pred_side_by_side` with a shared :class:`VizStyle`.

``CHART_FILENAMES`` (module-level) maps logical → numbered filename so callers
and tests never hardcode paths.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import supervision as sv
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.p06_training._common import (
    unwrap_subset as _unwrap,
)
from core.p06_training._common import (
    yolo_targets_to_xyxy,
)
from core.p06_training.postprocess import postprocess as _registry_postprocess
from core.p08_evaluation.error_analysis import (
    _LARGE_AREA,
    _SMALL_AREA,
    _size_category,
)
from core.p10_inference.supervision_bridge import (
    VizStyle,
    render_gt_pred_side_by_side,
)
from utils.viz import apply_plot_style, fit_figsize, new_figure, shorten_label

apply_plot_style()

logger = logging.getLogger(__name__)

SIZE_TIER_LABELS = {
    "small":  f"small (<{int(_SMALL_AREA ** 0.5)}²px, i.e. <{_SMALL_AREA} px²)",
    "medium": f"medium ({int(_SMALL_AREA ** 0.5)}²–{int(_LARGE_AREA ** 0.5)}²px)",
    "large":  f"large (≥{int(_LARGE_AREA ** 0.5)}²px, i.e. ≥{_LARGE_AREA} px²)",
}

# Logical name → numbered filename. Callers look up paths here.
CHART_FILENAMES: dict[str, str] = {
    "overview":                  "01_overview.png",
    "data_distribution":         "02_data_distribution.png",
    "per_class_performance":     "03_per_class_performance.png",
    "confusion_matrix":          "04_confusion_matrix.png",
    "top_confused_pairs":        "04_top_confused_pairs.png",
    "confidence_calibration":    "05_confidence_calibration.png",
    "failure_mode_contribution": "06_failure_mode_contribution.png",
    "failure_by_attribute":      "07_failure_by_attribute.png",
    "hardest_images":            "08_hardest_images.png",
    "failure_mode_examples":     "09_failure_mode_examples",
    "recoverable_map_vs_iou":    "10_recoverable_map_vs_iou.png",
    "confidence_attribution":    "11_confidence_attribution.png",
    "boxes_per_image":           "12_boxes_per_image.png",
    "bbox_aspect_ratio":         "13_bbox_aspect_ratio.png",
    "size_recall":               "14_size_recall.png",
}

_LARGE_CLASS_THRESHOLD = 20  # confusion-matrix → top-K swap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_name(name: str) -> str:
    return _SAFE_NAME.sub("_", str(name))[:80]


def _chart_path(out_dir: Path, key: str) -> Path:
    return out_dir / CHART_FILENAMES[key]


def _preprocess_for_model(
    raw_image: np.ndarray,
    input_size: tuple[int, int],
    model=None,
) -> torch.Tensor:
    """BGR HWC uint8 → CHW float32 tensor the model can forward on."""
    h, w = int(input_size[0]), int(input_size[1])
    processor = getattr(model, "processor", None)
    if processor is not None:
        rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (w, h))
        out = processor(images=[resized], return_tensors="pt", do_resize=False)
        return out["pixel_values"][0]

    resized = cv2.resize(raw_image, (w, h))
    arr = resized.astype(np.float32)
    output_format = (getattr(model, "output_format", "") or "").lower()
    if output_format in {"yolox"}:
        tensor_np = arr.transpose(2, 0, 1)
    else:
        mean = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255], dtype=np.float32).reshape(1, 1, 3)
        tensor_np = ((arr - mean) / std).transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(tensor_np))


def _dispatch_forward(model, tensor_batch: torch.Tensor):
    if hasattr(model, "hf_model"):
        return model(pixel_values=tensor_batch)
    return model(tensor_batch)


def _dispatch_postprocess(model, preds_raw, conf_threshold, target_sizes):
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


def _sampling_indices(n: int, max_samples: int | None) -> list[int]:
    if max_samples is None or max_samples >= n:
        return list(range(n))
    return sorted(
        np.random.default_rng(0).choice(n, size=max_samples, replace=False).tolist()
    )


def _rotate_xticks(ax, labels: list[str], *, threshold: int = 5) -> None:
    """Anti-overlap: rotate+right-align when many categories."""
    if len(labels) > threshold:
        ax.set_xticklabels(labels, rotation=40, ha="right")
    else:
        ax.set_xticklabels(labels)


def _savefig(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path))
    plt.close(fig)
    return path


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR HWC uint8 → RGB. Dataset `get_raw_item` returns BGR
    (cv2-native); `render_gt_pred_side_by_side` expects RGB."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Shared chart plotters (used by multiple tasks)
# ---------------------------------------------------------------------------


def _plot_overview(
    title: str, lines: list[str], path: Path,
) -> Path:
    """01_overview.png — headline metrics card. Text-only panel."""
    fig, ax = new_figure(n_items=1, figsize=(11, 5.5))
    ax.set_axis_off()
    ax.text(
        0.01, 0.98, title, fontsize=18, fontweight="bold", va="top",
        transform=ax.transAxes,
    )
    ax.text(
        0.01, 0.88, "\n".join(lines), fontsize=11, family="monospace", va="top",
        transform=ax.transAxes,
    )
    return _savefig(fig, path)


def _plot_per_class_bars(
    counts: dict[str, float],
    path: Path,
    *,
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None = (0, 1.2),
    value_fmt: str = "{:.2f}",
) -> Path:
    """Generic per-class bar chart (single series) — used for IoU, PCK, etc."""
    names = [shorten_label(k) for k in counts]
    vals = list(counts.values())
    fig, ax = new_figure(n_items=len(names))
    x = np.arange(len(names))
    ax.bar(x, vals, color="#4c72b0")
    for i, v in enumerate(vals):
        ax.text(i, v + (0.015 if ylim else max(vals) * 0.02 if vals else 0),
                value_fmt.format(v), ha="center", fontsize=8)
    ax.set_xticks(x)
    _rotate_xticks(ax, names)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    return _savefig(fig, path)


def _plot_per_class_prf1(
    per_class: dict[int, dict], class_names: dict[int, str], path: Path,
    *, conf_threshold: float | None = None, title_prefix: str = "Per-class",
) -> Path:
    """03_per_class_performance.png for detection/classification."""
    names = [shorten_label(class_names.get(cid, str(cid))) for cid in per_class]
    tp = np.array([c["tp"] for c in per_class.values()], dtype=np.float32)
    fp = np.array([c["fp"] for c in per_class.values()], dtype=np.float32)
    fn = np.array([c["fn"] for c in per_class.values()], dtype=np.float32)
    prec = np.where(tp + fp > 0, tp / (tp + fp + 1e-9), 0)
    rec = np.where(tp + fn > 0, tp / (tp + fn + 1e-9), 0)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec + 1e-9), 0)

    fig, ax = new_figure(n_items=len(names))
    x = np.arange(len(names))
    w = 0.28
    ax.bar(x - w, prec, w, label="Precision", color="#4c72b0")
    ax.bar(x,     rec,  w, label="Recall",    color="#55a868")
    ax.bar(x + w, f1,   w, label="F1",        color="#c44e52")
    for i, (p_val, r_val, f_val) in enumerate(zip(prec, rec, f1)):
        ax.text(i - w, p_val + 0.02, f"{p_val:.2f}", ha="center", fontsize=7)
        ax.text(i,     r_val + 0.02, f"{r_val:.2f}", ha="center", fontsize=7)
        ax.text(i + w, f_val + 0.02, f"{f_val:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    _rotate_xticks(ax, names)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("score (0 = worst, 1 = perfect)")
    title = f"{title_prefix} Precision / Recall / F1"
    if conf_threshold is not None:
        title += f"\n(conf ≥ {conf_threshold})"
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    return _savefig(fig, path)


def _plot_confusion_matrix(
    cm: np.ndarray, class_names: dict[int, str], path: Path,
    *, conf_threshold: float | None = None,
) -> Path:
    """04_confusion_matrix.png. Cell text shrinks with class count."""
    labels = [shorten_label(class_names.get(cid, str(cid))) for cid in class_names] + ["(none)"]
    n = len(labels)
    fig, ax = new_figure(n_items=n, figsize=fit_figsize(n, base=7.5, per_item=0.35,
                                                        min_w=8, max_w=20, height=max(6, n * 0.45)))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(n))
    _rotate_xticks(ax, labels)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted (last col = missed / no prediction)")
    ax.set_ylabel("Ground truth (last row = background / no GT)")
    title = "Confusion matrix"
    if conf_threshold is not None:
        title += f"  (conf ≥ {conf_threshold}, IoU ≥ 0.5)"
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="count")
    font_size = max(6, int(11 - 0.2 * n))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            if v == 0:
                continue
            ax.text(j, i, str(int(v)), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black", fontsize=font_size)
    return _savefig(fig, path)


def _plot_top_confused_pairs(
    cm: np.ndarray, class_names: dict[int, str], path: Path, top_k: int = 20,
) -> Path:
    """04_top_confused_pairs.png — horizontal bars for top off-diagonal cells."""
    n_cls = len(class_names)
    cids = list(class_names.keys())
    pairs: list[tuple[str, int]] = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j or cm[i, j] == 0:
                continue
            gt_name = class_names.get(cids[i], "(none)") if i < n_cls else "(background)"
            pred_name = class_names.get(cids[j], "(none)") if j < n_cls else "(none)"
            pairs.append((f"{shorten_label(gt_name)} → {shorten_label(pred_name)}", int(cm[i, j])))
    pairs.sort(key=lambda r: -r[1])
    pairs = pairs[:top_k]
    fig, ax = new_figure(n_items=len(pairs), figsize=(10, max(4, 0.35 * len(pairs) + 2)))
    if not pairs:
        ax.text(0.5, 0.5, "no confusion — all diagonal", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="#999999")
        ax.set_axis_off()
        return _savefig(fig, path)
    labels = [p[0] for p in pairs]
    counts = [p[1] for p in pairs]
    ax.barh(labels, counts, color="#c44e52")
    for i, c in enumerate(counts):
        ax.text(c + max(counts) * 0.01, i, str(c), va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("count")
    ax.set_title(f"Top {len(pairs)} confused class pairs  (GT → Pred)")
    ax.grid(axis="x", alpha=0.3)
    return _savefig(fig, path)


def _plot_confidence_hist(
    correct_scores: list[float], wrong_scores: list[float], path: Path,
) -> Path:
    """05_confidence_calibration.png — TP/correct vs FP/wrong score histograms."""
    fig, ax = new_figure(n_items=10, figsize=(10, 5.5))
    bins = np.linspace(0, 1, 21)
    if correct_scores:
        m = float(np.mean(correct_scores))
        ax.hist(correct_scores, bins=bins, alpha=0.6,
                label=f"correct (n={len(correct_scores)}, mean={m:.2f})",
                color="#55a868")
        ax.axvline(m, color="#2d6a4f", ls="--", lw=1, alpha=0.7)
    if wrong_scores:
        m = float(np.mean(wrong_scores))
        ax.hist(wrong_scores, bins=bins, alpha=0.6,
                label=f"wrong  (n={len(wrong_scores)}, mean={m:.2f})",
                color="#c44e52")
        ax.axvline(m, color="#9d0208", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("Prediction confidence")
    ax.set_ylabel("# predictions")
    ax.set_title("Confidence calibration — correct vs wrong")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    ax.grid(alpha=0.3)
    return _savefig(fig, path)


def _plot_hardest_images_grid(
    images: list[np.ndarray], titles: list[str], path: Path, *, header: str,
) -> Path | None:
    """08_hardest_images.png — simple matplotlib grid."""
    if not images:
        return None
    ncols = 4
    nrows = max(1, int(np.ceil(len(images) / ncols)))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 4.2, nrows * 3.6),
        constrained_layout=True, squeeze=False,
    )
    for i in range(nrows * ncols):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        ax.set_axis_off()
        if i < len(images):
            img = images[i]
            if img.shape[2] == 3:
                # images are RGB here (we converted)
                ax.imshow(img)
            if i < len(titles):
                ax.set_title(titles[i], fontsize=9)
    fig.suptitle(header, fontsize=12)
    return _savefig(fig, path)


# ---------------------------------------------------------------------------
# Task-agnostic gallery renderer (failed samples only).
# ---------------------------------------------------------------------------


def _render_gallery_side_by_side(
    cases: list[dict],
    out_dir: Path,
    *,
    task: str,
    class_names: dict[int, str],
    style: VizStyle,
    cap: int,
    sort_key: Callable[[dict], Any] | None = None,
    suffix_fn: Callable[[dict], str] = lambda r: "",
) -> int:
    """Render each case as a GT | Pred side-by-side PNG.

    Each ``case`` dict must carry:
      * ``image``   — BGR HWC uint8 (raw pixels)
      * ``gt``      — task-specific GT payload (see render_gt_pred_side_by_side)
      * ``pred``    — task-specific Pred payload
      * ``stem``    — filename stem (e.g. image basename)
      * ``banner``  — dict/str forwarded to the header banner
    """
    if sort_key is not None:
        cases = sorted(cases, key=sort_key)
    cases = cases[:cap]
    if not cases:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for rec in cases:
        img = rec.get("image")
        if img is None:
            continue
        rgb = _bgr_to_rgb(img)
        try:
            panel = render_gt_pred_side_by_side(
                rgb, rec.get("gt"), rec.get("pred"),
                task=task, class_names=class_names, style=style,
                banner=rec.get("banner"),
            )
        except Exception as e:
            logger.warning("gallery render failed (%s): %s", task, e)
            continue
        # Back to BGR for cv2.imwrite
        panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        suffix = suffix_fn(rec)
        stem = _safe_name(rec.get("stem") or "img")
        name = f"{stem}__{suffix}.png" if suffix else f"{stem}.png"
        cv2.imwrite(str(out_dir / name), panel_bgr)
        written += 1
    return written


# ---------------------------------------------------------------------------
# Shared markdown / json writer
# ---------------------------------------------------------------------------


def _write_json_md(
    out_dir: Path, summary: dict, *,
    title: str, header: list[str], chart_refs: list[str],
) -> tuple[Path, Path]:
    json_path = out_dir / "summary.json"
    md_path = out_dir / "summary.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=str)

    lines = [f"# {title}", "", *header, ""]
    if chart_refs:
        lines.append("## Charts")
        lines.append("")
        for ref in chart_refs:
            lines.append(f"- `{ref}`")
            lines.append(f"  ![{ref}]({ref})")
        lines.append("")
    for section, payload in summary.items():
        if section == "task":
            continue
        lines.append(f"## {section}")
        lines.append("")
        if isinstance(payload, dict):
            for k, v in payload.items():
                if isinstance(v, dict):
                    subs = "  ".join(f"{kk}={vv}" for kk, vv in v.items())
                    lines.append(f"- **{k}** — {subs}")
                else:
                    lines.append(f"- **{k}**: {v}")
        else:
            lines.append(str(payload))
        lines.append("")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    return json_path, md_path


# ===========================================================================
# Public dispatcher
# ===========================================================================


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
    """Dispatch error analysis to the task-specific analyzer.

    Returns a dict of logical-name → artifact path (plus
    ``summary_json``, ``summary_md``, ``failure_mode_examples_root``).
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


def _iou(a, b) -> float:
    xa = max(a[0], b[0])
    ya = max(a[1], b[1])
    xb = min(a[2], b[2])
    yb = min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _analyze_detection(
    *, model, dataset, output_dir: Path,
    class_names: dict[int, str], input_size: tuple[int, int], style: VizStyle,
    conf_threshold: float, iou_threshold: float,
    max_samples: int | None, hard_images_per_class: int,
    training_config: dict | None = None,
) -> dict[str, Any]:
    raw_ds, idx_map = _unwrap(dataset)
    indices = _sampling_indices(len(dataset), max_samples)
    device = next(model.parameters()).device
    input_h, input_w = int(input_size[0]), int(input_size[1])

    # Accumulators
    per_class = {cid: {"tp": 0, "fp": 0, "fn": 0} for cid in class_names}
    confidence_tp: list[float] = []
    confidence_fp: list[float] = []
    confusion = np.zeros((len(class_names) + 1, len(class_names) + 1), dtype=np.int64)
    size_stats = {t: {"tp": 0, "fp": 0, "fn": 0} for t in ("small", "medium", "large")}
    per_image: list[dict] = []
    gt_per_class: dict[int, int] = {c: 0 for c in class_names}
    gt_per_class_size: dict[int, dict[str, int]] = {
        c: {"small": 0, "medium": 0, "large": 0} for c in class_names
    }
    total_images_with_gt = 0
    boxes_per_image_counts: list[int] = []
    gt_aspect_ratios: dict[int, list[float]] = {c: [] for c in class_names}
    detections: list[dict] = []
    # Per-mode galleries — each entry stores enough to render a side-by-side.
    mode_galleries: dict[str, dict] = {
        "missed":          {c: [] for c in class_names},
        "localization":    {c: [] for c in class_names},
        "class_confusion": {},  # keyed by (gt_cid, pred_cid)
        "duplicate":       {c: [] for c in class_names},
        "background_fp":   {c: [] for c in class_names},
    }
    mode_counts: dict[str, dict[int, int]] = {
        m: {c: 0 for c in class_names} for m in
        ("correct", "missed", "localization", "class_confusion", "duplicate", "background_fp")
    }
    LOCALIZATION_IOU_LOW = 0.3

    _AR_BUCKETS = ("tall", "square", "wide")
    _CROWD_BUCKETS = ("1-2", "3-5", "6-10", "11+")
    gt_ar_bucket: dict[int, dict[str, int]] = {
        c: {k: 0 for k in _AR_BUCKETS} for c in class_names
    }
    gt_crowd_bucket: dict[int, dict[str, int]] = {
        c: {k: 0 for k in _CROWD_BUCKETS} for c in class_names
    }
    missed_by_size: dict[int, dict[str, int]] = {
        c: {"small": 0, "medium": 0, "large": 0} for c in class_names
    }
    missed_by_ar: dict[int, dict[str, int]] = {
        c: {k: 0 for k in _AR_BUCKETS} for c in class_names
    }
    missed_by_crowd: dict[int, dict[str, int]] = {
        c: {k: 0 for k in _CROWD_BUCKETS} for c in class_names
    }
    fn_attribution: dict[int, dict[str, int]] = {
        c: {"true_miss": 0, "under_confidence": 0, "localization_fail": 0}
        for c in class_names
    }

    def _ar_bucket(ar: float) -> str:
        return "tall" if ar < 0.5 else ("wide" if ar > 2.0 else "square")

    def _crowd_bucket(n: int) -> str:
        if n <= 2:
            return "1-2"
        if n <= 5:
            return "3-5"
        if n <= 10:
            return "6-10"
        return "11+"

    pred_cache: dict[int, dict] = {}

    for ds_idx in indices:
        real_idx = idx_map(ds_idx)
        try:
            raw = raw_ds.get_raw_item(real_idx)
        except Exception:
            continue
        image = raw["image"]
        if image is None:
            continue
        orig_h, orig_w = image.shape[:2]
        tensor = (
            _preprocess_for_model(image, (input_h, input_w), model=model).unsqueeze(0).to(device)
        )
        with torch.no_grad():
            preds_raw = _dispatch_forward(model, tensor)
        target_sizes = torch.tensor([[input_h, input_w]], device=device)
        decoded = _dispatch_postprocess(model, preds_raw, conf_threshold, target_sizes)[0]

        pb = np.asarray(decoded.get("boxes", []), dtype=np.float64).reshape(-1, 4)
        pl = np.asarray(decoded.get("labels", []), dtype=np.int64).ravel()
        ps = np.asarray(decoded.get("scores", []), dtype=np.float64).ravel()
        if len(pb) > 0:
            pb[:, [0, 2]] *= orig_w / input_w
            pb[:, [1, 3]] *= orig_h / input_h

        gt_xyxy, gt_cls = yolo_targets_to_xyxy(raw.get("targets"), orig_w, orig_h)

        if len(gt_xyxy) > 0:
            total_images_with_gt += 1
        boxes_per_image_counts.append(int(len(gt_xyxy)))
        img_crowd = _crowd_bucket(int(len(gt_xyxy)))
        for j in range(len(gt_xyxy)):
            cid = int(gt_cls[j])
            gt_per_class[cid] = gt_per_class.get(cid, 0) + 1
            bw = max(1.0, gt_xyxy[j, 2] - gt_xyxy[j, 0])
            bh = max(1.0, gt_xyxy[j, 3] - gt_xyxy[j, 1])
            gt_area = bw * bh
            tier = _size_category(gt_area)
            gt_per_class_size.setdefault(cid,
                {"small": 0, "medium": 0, "large": 0})[tier] += 1
            ar = float(bw / bh)
            gt_aspect_ratios.setdefault(cid, []).append(ar)
            gt_ar_bucket.setdefault(cid,
                {k: 0 for k in _AR_BUCKETS})[_ar_bucket(ar)] += 1
            gt_crowd_bucket.setdefault(cid,
                {k: 0 for k in _CROWD_BUCKETS})[img_crowd] += 1

        pred_cache[int(real_idx)] = {
            "path": raw.get("path", ""),
            "pb": pb, "pl": pl, "ps": ps,
            "gt_xyxy": gt_xyxy, "gt_cls": gt_cls,
            "orig_shape": (orig_h, orig_w),
        }

        matched_gt = np.zeros(len(gt_xyxy), dtype=bool)
        img_tp, img_fp, img_fn = 0, 0, 0

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
            detections.append({
                "pred_cls": int(pl[bi]),
                "score": float(ps[bi]),
                "best_iou_same_class": float(best_same_class_iou),
                "best_iou_any": float(best_iou),
                "gt_cls_at_best_iou": int(gt_cls[best_j]) if best_j >= 0 else -1,
            })

            det_mode = "background_fp"
            pcid = int(pl[bi])

            if best_same_class_iou >= iou_threshold:
                if matched_gt[best_same_class_j]:
                    det_mode = "duplicate"
                    per_class[pcid]["fp"] += 1
                    size_stats[size]["fp"] += 1
                    confusion[len(class_names), pcid] += 1
                    confidence_fp.append(float(ps[bi]))
                    img_fp += 1
                    mode_galleries["duplicate"].setdefault(pcid, []).append({
                        "image_idx": int(real_idx), "path": raw.get("path", ""),
                        "pred_box": pb[bi].tolist(), "score": float(ps[bi]),
                        "iou": float(best_same_class_iou),
                        "matched_gt_box": gt_xyxy[best_same_class_j].tolist(),
                        "matched_gt_cls": int(gt_cls[best_same_class_j]),
                    })
                else:
                    det_mode = "correct"
                    matched_gt[best_same_class_j] = True
                    per_class[pcid]["tp"] += 1
                    size_stats[size]["tp"] += 1
                    confusion[pcid, pcid] += 1
                    confidence_tp.append(float(ps[bi]))
                    img_tp += 1
            elif best_iou >= iou_threshold and best_j >= 0 and gt_cls[best_j] != pl[bi]:
                det_mode = "class_confusion"
                matched_gt[best_j] = True
                per_class[pcid]["fp"] += 1
                size_stats[size]["fp"] += 1
                gt_true_cls = int(gt_cls[best_j])
                per_class[gt_true_cls]["fn"] += 1
                gt_area = max(0.0, (gt_xyxy[best_j, 2] - gt_xyxy[best_j, 0])) * \
                          max(0.0, (gt_xyxy[best_j, 3] - gt_xyxy[best_j, 1]))
                size_stats[_size_category(gt_area)]["fn"] += 1
                confusion[gt_true_cls, pcid] += 1
                confidence_fp.append(float(ps[bi]))
                img_fp += 1
                img_fn += 1
                key = (gt_true_cls, pcid)
                mode_galleries["class_confusion"].setdefault(key, []).append({
                    "image_idx": int(real_idx), "path": raw.get("path", ""),
                    "pred_box": pb[bi].tolist(), "score": float(ps[bi]),
                    "iou": float(best_iou),
                    "gt_box": gt_xyxy[best_j].tolist(),
                    "gt_cls": gt_true_cls, "pred_cls": pcid,
                })
            elif LOCALIZATION_IOU_LOW <= best_same_class_iou < iou_threshold:
                det_mode = "localization"
                per_class[pcid]["fp"] += 1
                size_stats[size]["fp"] += 1
                confusion[len(class_names), pcid] += 1
                confidence_fp.append(float(ps[bi]))
                img_fp += 1
                mode_galleries["localization"].setdefault(pcid, []).append({
                    "image_idx": int(real_idx), "path": raw.get("path", ""),
                    "pred_box": pb[bi].tolist(), "score": float(ps[bi]),
                    "iou": float(best_same_class_iou),
                    "matched_gt_box": gt_xyxy[best_same_class_j].tolist()
                        if best_same_class_j >= 0 else None,
                    "matched_gt_cls": int(gt_cls[best_same_class_j])
                        if best_same_class_j >= 0 else int(pcid),
                })
            else:
                det_mode = "background_fp"
                per_class[pcid]["fp"] += 1
                size_stats[size]["fp"] += 1
                confusion[len(class_names), pcid] += 1
                confidence_fp.append(float(ps[bi]))
                img_fp += 1
                mode_galleries["background_fp"].setdefault(pcid, []).append({
                    "image_idx": int(real_idx), "path": raw.get("path", ""),
                    "pred_box": pb[bi].tolist(), "score": float(ps[bi]),
                    "pred_cls": pcid,
                })

            mode_counts[det_mode][pcid] = mode_counts[det_mode].get(pcid, 0) + 1
            detections[-1]["mode"] = det_mode

        for j in np.where(~matched_gt)[0]:
            cid = int(gt_cls[j])
            per_class[cid]["fn"] += 1
            bw_gt = max(0.0, (gt_xyxy[j, 2] - gt_xyxy[j, 0]))
            bh_gt = max(0.0, (gt_xyxy[j, 3] - gt_xyxy[j, 1]))
            area = bw_gt * bh_gt
            size_tier = _size_category(area)
            size_stats[size_tier]["fn"] += 1
            confusion[cid, len(class_names)] += 1
            img_fn += 1
            mode_counts["missed"][cid] = mode_counts["missed"].get(cid, 0) + 1
            ar = float(max(1.0, bw_gt) / max(1.0, bh_gt))
            missed_by_size.setdefault(cid, {"small": 0, "medium": 0, "large": 0})[size_tier] += 1
            missed_by_ar.setdefault(cid,
                {k: 0 for k in _AR_BUCKETS})[_ar_bucket(ar)] += 1
            missed_by_crowd.setdefault(cid,
                {k: 0 for k in _CROWD_BUCKETS})[_crowd_bucket(int(len(gt_xyxy)))] += 1
            best_any, best_same = 0.0, 0.0
            best_any_pred = None
            for bi in range(len(pb)):
                iou_ = _iou(pb[bi], gt_xyxy[j])
                if iou_ > best_any:
                    best_any = iou_
                    best_any_pred = bi
                if int(pl[bi]) == cid and iou_ > best_same:
                    best_same = iou_
            if best_any < LOCALIZATION_IOU_LOW:
                fn_sub = "true_miss"
            elif LOCALIZATION_IOU_LOW <= best_same < iou_threshold:
                fn_sub = "localization_fail"
            else:
                fn_sub = "under_confidence"
            fn_attribution.setdefault(cid,
                {"true_miss": 0, "under_confidence": 0, "localization_fail": 0}
            )[fn_sub] += 1
            mode_galleries["missed"].setdefault(cid, []).append({
                "image_idx": int(real_idx),
                "path": raw.get("path", ""),
                "gt_box": gt_xyxy[j].tolist(),
                "gt_cls": cid,
                "size_tier": size_tier,
                "area": float(area),
                "fn_sub": fn_sub,
                "nearest_pred_box": (
                    pb[best_any_pred].tolist() if best_any_pred is not None else None
                ),
                "nearest_pred_cls": (
                    int(pl[best_any_pred]) if best_any_pred is not None else None
                ),
                "nearest_pred_score": (
                    float(ps[best_any_pred]) if best_any_pred is not None else None
                ),
            })

        per_image.append({
            "idx": int(real_idx), "path": raw.get("path", ""),
            "tp": img_tp, "fp": img_fp, "fn": img_fn,
        })

    # ---- summary (computational parts kept) ----
    summary = _summarize_detection_counts(per_class, size_stats,
                                          confidence_tp, confidence_fp, class_names)
    if training_config:
        summary["training_config"] = training_config
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
            class_names.get(cid, str(cid)): {k: int(v) for k, v in tiers.items()}
            for cid, tiers in gt_per_class_size.items()
        },
    }

    missed_per_class: dict[int, int] = {cid: int(mode_counts["missed"].get(cid, 0))
                                         for cid in class_names}
    contribution = _compute_recoverable_map(
        detections, gt_per_class, missed_per_class, class_names,
        iou_thr=iou_threshold,
    )
    mode_total = {m: int(sum(mode_counts[m].values()))
                  for m in mode_counts if m != "correct"}
    confusion_pairs_top = sorted(
        [
            {
                "gt": class_names.get(g, str(g)),
                "pred": class_names.get(p, str(p)),
                "count": len(v),
            }
            for (g, p), v in mode_galleries["class_confusion"].items()
        ],
        key=lambda r: -r["count"],
    )[:10]
    failure_mode = {
        "baseline_map50": contribution["baseline_map50"],
        "ceiling_map50": round(
            min(1.0, contribution["baseline_map50"]
                + sum(max(0.0, c["delta_map50"])
                      for c in contribution["modes"].values())),
            4,
        ),
        "error_types": mode_total,
        "per_class": {
            class_names.get(cid, str(cid)): {
                m: int(mode_counts[m].get(cid, 0)) for m in mode_counts
            }
            for cid in class_names
        },
        "confusion_pairs_top": confusion_pairs_top,
        "contribution": contribution["modes"],
        "fn_attribution": {
            class_names.get(cid, str(cid)): dict(v) for cid, v in fn_attribution.items()
        },
        "miss_by_attribute": {
            "by_size": {
                class_names.get(cid, str(cid)): dict(v) for cid, v in missed_by_size.items()
            },
            "by_aspect_ratio": {
                class_names.get(cid, str(cid)): dict(v) for cid, v in missed_by_ar.items()
            },
            "by_crowdedness": {
                class_names.get(cid, str(cid)): dict(v) for cid, v in missed_by_crowd.items()
            },
        },
    }
    summary["model_metrics"] = {
        "ap50_per_class": _per_class_ap(detections, gt_per_class, class_names, 0.5),
        "map_vs_iou": _map_at_iou_sweep(detections, gt_per_class,
                                         np.arange(0.5, 1.0, 0.05)),
        "failure_mode": failure_mode,
    }

    artifacts: dict[str, Any] = {}

    # --- 01 overview ---
    ap50 = float(np.mean(list(summary["model_metrics"]["ap50_per_class"].values())) or 0)
    artifacts["overview"] = _plot_overview(
        "Detection — Error Analysis Overview",
        [
            f"Samples analyzed      : {len(per_image)}",
            f"Images with ≥1 GT box : {total_images_with_gt}",
            f"Total GT boxes        : {int(sum(gt_per_class.values()))}",
            f"Classes               : {len(class_names)}",
            f"mAP50 (baseline)      : {contribution['baseline_map50']:.4f}",
            f"mAP50 (if all fixed)  : {failure_mode['ceiling_map50']:.4f}",
            f"Mean AP50 / class     : {ap50:.4f}",
            f"IoU threshold         : {iou_threshold}",
            f"Confidence threshold  : {conf_threshold}",
        ],
        _chart_path(output_dir, "overview"),
    )
    # --- 02 data distribution ---
    artifacts["data_distribution"] = _plot_data_distribution(
        gt_per_class, gt_per_class_size, class_names,
        _chart_path(output_dir, "data_distribution"),
    )
    # --- 03 per-class P/R/F1 ---
    artifacts["per_class_performance"] = _plot_per_class_prf1(
        per_class, class_names, _chart_path(output_dir, "per_class_performance"),
        conf_threshold=conf_threshold,
    )
    # --- 04 confusion or top pairs ---
    if len(class_names) <= _LARGE_CLASS_THRESHOLD:
        artifacts["confusion_matrix"] = _plot_confusion_matrix(
            confusion, class_names, _chart_path(output_dir, "confusion_matrix"),
            conf_threshold=conf_threshold,
        )
    else:
        artifacts["top_confused_pairs"] = _plot_top_confused_pairs(
            confusion, class_names, _chart_path(output_dir, "top_confused_pairs"),
        )
    # --- 05 confidence calibration ---
    artifacts["confidence_calibration"] = _plot_confidence_hist(
        confidence_tp, confidence_fp,
        _chart_path(output_dir, "confidence_calibration"),
    )
    # --- 06 failure-mode contribution ---
    p = _plot_failure_mode_contribution(
        mode_total, contribution,
        _chart_path(output_dir, "failure_mode_contribution"),
    )
    if p is not None:
        artifacts["failure_mode_contribution"] = p
    # --- 07 failure-by-attribute ---
    artifacts["failure_by_attribute"] = _plot_failure_by_attribute(
        missed_by_size, gt_per_class_size,
        missed_by_ar, gt_ar_bucket,
        missed_by_crowd, gt_crowd_bucket,
        confusion_pairs_top, class_names,
        _chart_path(output_dir, "failure_by_attribute"),
    )
    # --- 08 hardest images overview ---
    per_image.sort(key=lambda r: -(r["fn"] + r["fp"]))
    top_rows = []
    top_titles = []
    for rec in per_image[:12]:
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
        rgb = _bgr_to_rgb(image)
        pred_dets = sv.Detections(
            xyxy=entry["pb"] if len(entry["pb"]) > 0 else np.zeros((0, 4)),
            class_id=entry["pl"].astype(int),
            confidence=entry["ps"].astype(np.float32),
        )
        panel = render_gt_pred_side_by_side(
            rgb, (entry["gt_xyxy"], entry["gt_cls"]), pred_dets,
            task="detection", class_names=class_names, style=style,
        )
        top_rows.append(panel)
        top_titles.append(f"FP={rec['fp']}  FN={rec['fn']}")
    p = _plot_hardest_images_grid(
        top_rows, top_titles, _chart_path(output_dir, "hardest_images"),
        header="Hardest images — top 12 by (FP + FN).  GT (purple) | Pred (green)",
    )
    if p is not None:
        artifacts["hardest_images"] = p

    # --- 10–13: detection-specific extras ---
    artifacts["recoverable_map_vs_iou"] = _plot_recoverable_map_vs_iou(
        detections, gt_per_class, missed_per_class, class_names,
        _chart_path(output_dir, "recoverable_map_vs_iou"),
    )
    artifacts["confidence_attribution"] = _plot_confidence_attribution(
        fn_attribution, class_names,
        _chart_path(output_dir, "confidence_attribution"),
    )
    artifacts["boxes_per_image"] = _plot_boxes_per_image(
        boxes_per_image_counts, _chart_path(output_dir, "boxes_per_image"),
    )
    artifacts["bbox_aspect_ratio"] = _plot_bbox_aspect_ratio(
        gt_aspect_ratios, class_names,
        _chart_path(output_dir, "bbox_aspect_ratio"),
    )
    artifacts["size_recall"] = _plot_size_recall(
        size_stats, _chart_path(output_dir, "size_recall"),
    )

    # --- 09 failure-mode examples (side-by-side, only failed samples) ---
    if hard_images_per_class > 0:
        examples_root = output_dir / CHART_FILENAMES["failure_mode_examples"]
        artifacts["failure_mode_examples_root"] = examples_root

        def _cls_label(cid):
            return _safe_name(class_names.get(cid, str(cid)))
        def _confpair_label(pair):
            g, p = pair
            p_name = _safe_name(class_names.get(p, str(p)))
            g_name = _safe_name(class_names.get(g, str(g)))
            return f"{p_name}__from__{g_name}"

        # --- missed: GT panel = only the missed box; Pred panel = empty ---
        for cid, items in mode_galleries["missed"].items():
            cases = []
            for rec in items:
                entry = pred_cache.get(rec["image_idx"])
                if entry is None:
                    continue
                try:
                    raw = raw_ds.get_raw_item(rec["image_idx"])
                except Exception:
                    continue
                if raw["image"] is None:
                    continue
                gt_xyxy_filtered = np.asarray([rec["gt_box"]], dtype=np.float32)
                gt_cls_filtered = np.asarray([rec["gt_cls"]], dtype=int)
                pred_dets = sv.Detections(
                    xyxy=np.zeros((0, 4), dtype=np.float32),
                    class_id=np.zeros(0, dtype=int),
                    confidence=np.zeros(0, dtype=np.float32),
                )
                cases.append({
                    "image": raw["image"],
                    "gt": (gt_xyxy_filtered, gt_cls_filtered),
                    "pred": pred_dets,
                    "stem": Path(rec["path"]).stem or f"img_{rec['image_idx']}",
                    "banner": {
                        "title": f"missed: {class_names.get(cid, str(cid))}",
                        "subtitle": (
                            f"area={int(rec['area'])}px² ({rec['size_tier']}) sub={rec['fn_sub']}"
                        ),
                    },
                    "_sort": rec["area"],
                })
            _render_gallery_side_by_side(
                cases, examples_root / "missed" / _cls_label(cid),
                task="detection", class_names=class_names, style=style,
                cap=hard_images_per_class,
                sort_key=lambda r: r["_sort"],
                suffix_fn=lambda r: f"missed__area_{int(r['_sort'])}",
            )

        # --- localization: show matched GT + the drifted pred ---
        for cid, items in mode_galleries["localization"].items():
            cases = []
            for rec in items:
                try:
                    raw = raw_ds.get_raw_item(rec["image_idx"])
                except Exception:
                    continue
                if raw["image"] is None:
                    continue
                if rec.get("matched_gt_box") is None:
                    continue
                gt_xyxy_filtered = np.asarray([rec["matched_gt_box"]], dtype=np.float32)
                gt_cls_filtered = np.asarray([rec["matched_gt_cls"]], dtype=int)
                pred_dets = sv.Detections(
                    xyxy=np.asarray([rec["pred_box"]], dtype=np.float32),
                    class_id=np.asarray([cid], dtype=int),
                    confidence=np.asarray([rec["score"]], dtype=np.float32),
                )
                cases.append({
                    "image": raw["image"],
                    "gt": (gt_xyxy_filtered, gt_cls_filtered),
                    "pred": pred_dets,
                    "stem": Path(rec["path"]).stem or f"img_{rec['image_idx']}",
                    "banner": {
                        "title": f"localization: {class_names.get(cid, str(cid))}",
                        "subtitle": f"IoU={rec['iou']:.2f} score={rec['score']:.2f}",
                    },
                    "_sort": -rec["score"],
                })
            _render_gallery_side_by_side(
                cases, examples_root / "localization" / _cls_label(cid),
                task="detection", class_names=class_names, style=style,
                cap=hard_images_per_class,
                sort_key=lambda r: r["_sort"],
                suffix_fn=lambda r: f"loc__iou_{-r['_sort']:.2f}",
            )

        # --- class_confusion: only the confused GT and confused Pred ---
        for pair, items in mode_galleries["class_confusion"].items():
            cases = []
            gt_cid, pred_cid = pair
            for rec in items:
                try:
                    raw = raw_ds.get_raw_item(rec["image_idx"])
                except Exception:
                    continue
                if raw["image"] is None:
                    continue
                gt_xyxy_filtered = np.asarray([rec["gt_box"]], dtype=np.float32)
                gt_cls_filtered = np.asarray([rec["gt_cls"]], dtype=int)
                pred_dets = sv.Detections(
                    xyxy=np.asarray([rec["pred_box"]], dtype=np.float32),
                    class_id=np.asarray([rec["pred_cls"]], dtype=int),
                    confidence=np.asarray([rec["score"]], dtype=np.float32),
                )
                cases.append({
                    "image": raw["image"],
                    "gt": (gt_xyxy_filtered, gt_cls_filtered),
                    "pred": pred_dets,
                    "stem": Path(rec["path"]).stem or f"img_{rec['image_idx']}",
                    "banner": {
                        "title": f"class_confusion: gt={class_names.get(gt_cid, str(gt_cid))} "
                                 f"pred={class_names.get(pred_cid, str(pred_cid))}",
                        "subtitle": f"IoU={rec['iou']:.2f} score={rec['score']:.2f}",
                    },
                    "_sort": -rec["score"],
                })
            _render_gallery_side_by_side(
                cases, examples_root / "class_confusion" / _confpair_label(pair),
                task="detection", class_names=class_names, style=style,
                cap=hard_images_per_class,
                sort_key=lambda r: r["_sort"],
                suffix_fn=lambda r: f"conf__score_{-r['_sort']:.2f}",
            )

        # --- duplicate: GT panel = matched GT only; Pred panel = duplicate only ---
        for cid, items in mode_galleries["duplicate"].items():
            cases = []
            for rec in items:
                try:
                    raw = raw_ds.get_raw_item(rec["image_idx"])
                except Exception:
                    continue
                if raw["image"] is None:
                    continue
                gt_xyxy_filtered = np.asarray([rec["matched_gt_box"]], dtype=np.float32)
                gt_cls_filtered = np.asarray([rec["matched_gt_cls"]], dtype=int)
                pred_dets = sv.Detections(
                    xyxy=np.asarray([rec["pred_box"]], dtype=np.float32),
                    class_id=np.asarray([cid], dtype=int),
                    confidence=np.asarray([rec["score"]], dtype=np.float32),
                )
                cases.append({
                    "image": raw["image"],
                    "gt": (gt_xyxy_filtered, gt_cls_filtered),
                    "pred": pred_dets,
                    "stem": Path(rec["path"]).stem or f"img_{rec['image_idx']}",
                    "banner": {
                        "title": f"duplicate: {class_names.get(cid, str(cid))}",
                        "subtitle": f"IoU={rec['iou']:.2f} score={rec['score']:.2f}",
                    },
                    "_sort": -rec["iou"],
                })
            _render_gallery_side_by_side(
                cases, examples_root / "duplicate" / _cls_label(cid),
                task="detection", class_names=class_names, style=style,
                cap=hard_images_per_class,
                sort_key=lambda r: r["_sort"],
                suffix_fn=lambda r: f"dup__iou_{-r['_sort']:.2f}",
            )

        # --- background_fp: GT panel empty, Pred panel = spurious only ---
        for cid, items in mode_galleries["background_fp"].items():
            cases = []
            for rec in items:
                try:
                    raw = raw_ds.get_raw_item(rec["image_idx"])
                except Exception:
                    continue
                if raw["image"] is None:
                    continue
                pred_dets = sv.Detections(
                    xyxy=np.asarray([rec["pred_box"]], dtype=np.float32),
                    class_id=np.asarray([cid], dtype=int),
                    confidence=np.asarray([rec["score"]], dtype=np.float32),
                )
                cases.append({
                    "image": raw["image"],
                    "gt": (np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=int)),
                    "pred": pred_dets,
                    "stem": Path(rec["path"]).stem or f"img_{rec['image_idx']}",
                    "banner": {
                        "title": f"background_fp: {class_names.get(cid, str(cid))}",
                        "subtitle": f"score={rec['score']:.2f}",
                    },
                    "_sort": -rec["score"],
                })
            _render_gallery_side_by_side(
                cases, examples_root / "background_fp" / _cls_label(cid),
                task="detection", class_names=class_names, style=style,
                cap=hard_images_per_class,
                sort_key=lambda r: r["_sort"],
                suffix_fn=lambda r: f"bgfp__score_{-r['_sort']:.2f}",
            )

    # ---- summary.{json,md} ----
    ranking_lines = _render_failure_mode_table(
        mode_total, contribution, baseline=contribution["baseline_map50"],
    )
    chart_refs = [CHART_FILENAMES[k] for k in (
        "overview", "data_distribution", "per_class_performance",
        "confusion_matrix" if len(class_names) <= _LARGE_CLASS_THRESHOLD else "top_confused_pairs",
        "confidence_calibration", "failure_mode_contribution",
        "failure_by_attribute", "hardest_images",
        "recoverable_map_vs_iou", "confidence_attribution",
        "boxes_per_image", "bbox_aspect_ratio", "size_recall",
    )]
    json_path, md_path = _write_json_md(
        output_dir, summary,
        title="Detection Error Analysis",
        header=[
            f"- Samples analyzed: **{len(per_image)}**",
            f"- Images with ≥1 GT box: **{total_images_with_gt}**",
            f"- Total GT boxes: **{int(sum(gt_per_class.values()))}**",
            f"- IoU threshold (base): {iou_threshold}",
            f"- Confidence threshold (base): {conf_threshold}",
            "",
            *ranking_lines,
        ],
        chart_refs=chart_refs,
    )
    artifacts["summary_json"] = json_path
    artifacts["summary_md"] = md_path
    return artifacts


def _summarize_detection_counts(per_class, size_stats, tp_scores, fp_scores, class_names):
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


# ---- Detection-specific plotters (AP, counterfactual, etc.) ----


def _per_class_ap_curve(
    detections: list[dict], gt_count: int, target_cls: int, iou_thr: float,
) -> tuple[np.ndarray, np.ndarray, float]:
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


def _mutate_for_mode(detections, mode, missed_per_class, median_tp_score, iou_thr):
    out: list[dict] = []
    for d in detections:
        m = d.get("mode")
        if mode in ("duplicate", "background_fp") and m == mode:
            continue
        if mode == "class_confusion" and m == "class_confusion":
            fixed = dict(d)
            fixed["pred_cls"] = int(d.get("gt_cls_at_best_iou", d["pred_cls"]))
            fixed["best_iou_same_class"] = max(float(iou_thr),
                                                 float(d.get("best_iou_any", 0.0)))
            fixed["mode"] = "correct"
            out.append(fixed)
            continue
        if mode == "localization" and m == "localization":
            fixed = dict(d)
            fixed["best_iou_same_class"] = max(float(iou_thr),
                                                 float(d["best_iou_same_class"]))
            fixed["mode"] = "correct"
            out.append(fixed)
            continue
        out.append(d)

    if mode == "missed":
        for cid, n in missed_per_class.items():
            for _ in range(int(n)):
                out.append({
                    "pred_cls": int(cid),
                    "score": float(median_tp_score),
                    "best_iou_same_class": 1.0,
                    "best_iou_any": 1.0,
                    "gt_cls_at_best_iou": int(cid),
                    "mode": "correct",
                })
    return out


def _compute_recoverable_map(
    detections: list[dict],
    gt_per_class: dict[int, int],
    missed_per_class: dict[int, int],
    class_names: dict[int, str],
    iou_thr: float = 0.5,
) -> dict:
    tp_scores = [d["score"] for d in detections if d.get("mode") == "correct"]
    median_tp = float(np.median(tp_scores)) if tp_scores else 0.5

    def _per_class_ap_dict(dets):
        return {
            cid: _per_class_ap_curve(dets, int(gt_per_class.get(cid, 0)), cid, iou_thr)[2]
            for cid in class_names if gt_per_class.get(cid, 0) > 0
        }

    baseline_per_class = _per_class_ap_dict(detections)
    baseline = float(np.mean(list(baseline_per_class.values()))) if baseline_per_class else 0.0

    modes_out: dict[str, dict] = {}
    per_class_out: dict[str, dict[str, float]] = {}
    for mode in ("missed", "class_confusion", "localization", "duplicate", "background_fp"):
        mutated = _mutate_for_mode(detections, mode, missed_per_class, median_tp, iou_thr)
        new_per_class = _per_class_ap_dict(mutated)
        new_map = float(np.mean(list(new_per_class.values()))) if new_per_class else 0.0
        modes_out[mode] = {
            "delta_map50": round(new_map - baseline, 4),
            "new_map50": round(new_map, 4),
        }
        per_class_out[mode] = {
            class_names.get(cid, str(cid)):
                round(new_per_class.get(cid, 0.0) - baseline_per_class.get(cid, 0.0), 4)
            for cid in baseline_per_class
        }
    return {
        "baseline_map50": round(baseline, 4),
        "baseline_ap50_per_class": {
            class_names.get(cid, str(cid)): round(v, 4)
            for cid, v in baseline_per_class.items()
        },
        "modes": modes_out,
        "modes_per_class": per_class_out,
    }


def _render_failure_mode_table(mode_total, contribution, *, baseline: float) -> list[str]:
    modes = contribution["modes"]
    total_errors = max(1, sum(mode_total.values()))
    rows = []
    for m, info in modes.items():
        count = mode_total.get(m, 0)
        rows.append((m, count, info["delta_map50"], 100.0 * count / total_errors))
    rows.sort(key=lambda r: -r[2])

    lines = [
        "## Failure-mode ranking (biggest accuracy wins first)",
        "",
        "| Mode              | Count | Δ mAP50 if fixed | % of error volume |",
        "|-------------------|------:|-----------------:|------------------:|",
    ]
    total_delta = 0.0
    for m, count, delta, pct in rows:
        total_delta += max(0.0, delta)
        lines.append(f"| {m:<17} | {count:>5} | {delta:+.4f} | {pct:>5.1f}% |")
    ceiling = min(1.0, baseline + total_delta)
    lines.append("")
    lines.append(f"**Baseline mAP50 = {baseline:.4f}.  "
                  f"If every mode were fixed → {ceiling:.4f} (ceiling).**")
    lines.append("")
    return lines


def _plot_failure_mode_contribution(mode_total, contribution, path: Path) -> Path | None:
    modes_info = contribution["modes"]
    if not modes_info:
        return None
    rows = sorted(modes_info.items(), key=lambda kv: kv[1]["delta_map50"])
    labels = [m for m, _ in rows]
    deltas = [info["delta_map50"] for _, info in rows]
    counts = [mode_total.get(m, 0) for m, _ in rows]
    baseline = contribution["baseline_map50"]
    ceiling = min(1.0, baseline + sum(max(0.0, d) for d in deltas))

    modes_pc = contribution.get("modes_per_class", {})
    class_names_list = sorted({c for row in modes_pc.values() for c in row})
    heatmap_modes = ["missed", "localization", "class_confusion", "duplicate", "background_fp"]

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(16, max(5, 0.35 * max(1, len(class_names_list)) + 3)),
        constrained_layout=True,
    )
    colors_a = ["#c44e52" if d >= 0 else "#999999" for d in deltas]
    bars = ax_a.barh(labels, deltas, color=colors_a)
    for bar, d, n in zip(bars, deltas, counts):
        x = bar.get_width()
        ax_a.text(x + 0.002, bar.get_y() + bar.get_height() / 2,
                   f"n={n}  Δ={d:+.4f}", va="center", fontsize=9)
    ax_a.axvline(0, color="black", lw=0.8)
    ax_a.set_xlabel("Δ mAP50 if mode counter-factually fixed")
    ax_a.set_title(f"(a) Global — baseline {baseline:.3f} → ceiling {ceiling:.3f}")
    ax_a.grid(axis="x", alpha=0.3)

    if class_names_list and modes_pc:
        mat = np.zeros((len(class_names_list), len(heatmap_modes)), dtype=np.float32)
        for i, cname in enumerate(class_names_list):
            for j, m in enumerate(heatmap_modes):
                mat[i, j] = float(modes_pc.get(m, {}).get(cname, 0.0))
        vmax = max(0.01, float(np.abs(mat).max()))
        im = ax_b.imshow(mat, cmap="Reds", vmin=0, vmax=vmax, aspect="auto")
        ax_b.set_xticks(range(len(heatmap_modes)))
        ax_b.set_xticklabels(heatmap_modes, rotation=40, ha="right")
        ax_b.set_yticks(range(len(class_names_list)))
        ax_b.set_yticklabels([shorten_label(c) for c in class_names_list])
        font_size = max(6, int(10 - 0.2 * len(class_names_list)))
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if v > 0.001:
                    color = "white" if v > vmax * 0.55 else "black"
                    ax_b.text(j, i, f"{v:+.3f}", ha="center", va="center",
                              fontsize=font_size, color=color)
        fig.colorbar(im, ax=ax_b, label="Δ AP50 for this class")
        ax_b.set_title("(b) Per-class — which mode hurts each class most")
    else:
        ax_b.set_axis_off()
        ax_b.text(0.5, 0.5, "no per-class data", ha="center", va="center",
                   fontsize=11, color="#999999")
    return _savefig(fig, path)


def _plot_recoverable_map_vs_iou(
    detections, gt_per_class, missed_per_class, class_names, path: Path,
) -> Path | None:
    iou_values = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    modes = ["missed", "localization", "class_confusion", "duplicate", "background_fp"]
    colors = {
        "missed":          "#c44e52",
        "localization":    "#8172b2",
        "class_confusion": "#dd8452",
        "duplicate":       "#4c72b0",
        "background_fp":   "#55a868",
    }

    baselines: list[float] = []
    delta_by_mode: dict[str, list[float]] = {m: [] for m in modes}
    ceilings: list[float] = []
    for iou in iou_values:
        contrib = _compute_recoverable_map(
            detections, gt_per_class, missed_per_class, class_names, iou_thr=float(iou),
        )
        baselines.append(contrib["baseline_map50"])
        total = 0.0
        for m in modes:
            d = contrib["modes"][m]["delta_map50"]
            delta_by_mode[m].append(d)
            total += max(0.0, d)
        ceilings.append(min(1.0, contrib["baseline_map50"] + total))

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(16, 5.5), constrained_layout=True)
    ax_a.plot(iou_values, baselines, lw=2.5, color="black", marker="o",
               label="baseline mAP")
    ax_a.plot(iou_values, ceilings, lw=2.0, color="#999999", linestyle="--", marker="s",
               label="ceiling (all modes fixed)")
    for m in modes:
        curve = [baselines[i] + delta_by_mode[m][i] for i in range(len(iou_values))]
        ax_a.plot(iou_values, curve, lw=1.3, marker=".", color=colors[m],
                   label=f"+ fix {m}", alpha=0.85)
    ax_a.set_xlabel("IoU threshold")
    ax_a.set_ylabel("mAP")
    ax_a.set_ylim(0, 1.05)
    ax_a.set_xticks(iou_values)
    ax_a.set_title("(a) Baseline vs ceiling across IoU strictness")
    ax_a.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    ax_a.grid(alpha=0.3)

    x = np.arange(len(iou_values))
    bottom = np.zeros(len(iou_values))
    for m in modes:
        ys = np.array([max(0.0, d) for d in delta_by_mode[m]])
        ax_b.bar(x, ys, 0.6, bottom=bottom, color=colors[m], label=m)
        bottom += ys
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([f"{v:.2f}" for v in iou_values])
    ax_b.set_xlabel("IoU threshold")
    ax_b.set_ylabel("Δ mAP stacked")
    ax_b.set_title("(b) Stacked Δ per IoU")
    ax_b.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    ax_b.grid(axis="y", alpha=0.3)
    return _savefig(fig, path)


def _plot_failure_by_attribute(
    missed_by_size, gt_size_totals,
    missed_by_ar, gt_ar_totals,
    missed_by_crowd, gt_crowd_totals,
    confusion_pairs_top: list[dict],
    class_names: dict[int, str], path: Path,
) -> Path:
    def _rates(miss, total, buckets):
        out = {}
        for cid in class_names:
            m = miss.get(cid, {})
            t = total.get(cid, {})
            row = []
            for b in buckets:
                tot = t.get(b, 0)
                row.append((m.get(b, 0) / tot) if tot > 0 else 0.0)
            out[cid] = row
        return out

    size_buckets = ("small", "medium", "large")
    ar_buckets = ("tall", "square", "wide")
    crowd_buckets = ("1-2", "3-5", "6-10", "11+")
    size_rates = _rates(missed_by_size, gt_size_totals, size_buckets)
    ar_rates = _rates(missed_by_ar, gt_ar_totals, ar_buckets)
    crowd_rates = _rates(missed_by_crowd, gt_crowd_totals, crowd_buckets)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    (ax_size, ax_ar), (ax_crowd, ax_conf) = axes
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(class_names), 10)))

    def _grouped_bars(ax, buckets, rates_per_class, title, xlabel):
        n_cls = max(1, len(class_names))
        x = np.arange(len(buckets))
        w = 0.8 / n_cls
        for i, cid in enumerate(class_names):
            row = rates_per_class[cid]
            ax.bar(x + (i - n_cls / 2) * w + w / 2, row, w,
                   label=shorten_label(class_names.get(cid, str(cid))),
                   color=colors[i % 10])
            for bi, val in enumerate(row):
                if val > 0:
                    ax.text(x[bi] + (i - n_cls / 2) * w + w / 2, val + 0.01,
                            f"{val:.2f}", ha="center", fontsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels(buckets)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("miss-rate")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    _grouped_bars(ax_size, size_buckets, size_rates,
                   "(a) Miss-rate by size (COCO tiers)", "size tier")
    _grouped_bars(ax_ar, ar_buckets, ar_rates,
                   "(b) Miss-rate by aspect ratio", "w/h bucket")
    _grouped_bars(ax_crowd, crowd_buckets, crowd_rates,
                   "(c) Miss-rate by crowdedness", "boxes per image")
    handles, labels = ax_size.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
                bbox_to_anchor=(0.5, 1.02), ncol=min(6, len(labels)), fontsize=9)

    if confusion_pairs_top:
        pairs = confusion_pairs_top[:10]
        pair_labels = [f"{shorten_label(p['gt'])} → {shorten_label(p['pred'])}" for p in pairs]
        pair_counts = [p["count"] for p in pairs]
        ax_conf.barh(pair_labels, pair_counts, color="#c44e52")
        for i, cnt in enumerate(pair_counts):
            ax_conf.text(cnt + 0.05, i, f"n={cnt}", va="center", fontsize=9)
        ax_conf.invert_yaxis()
        ax_conf.set_xlabel("count")
        ax_conf.set_title("(d) Top confusion pairs  (GT → Pred)")
        ax_conf.grid(axis="x", alpha=0.3)
    else:
        ax_conf.set_axis_off()
        ax_conf.text(0.5, 0.5, "no class-confusion errors", ha="center", va="center",
                      fontsize=11, color="#999999")
        ax_conf.set_title("(d) Top confusion pairs")
    return _savefig(fig, path)


def _plot_confidence_attribution(
    fn_attribution: dict[int, dict[str, int]], class_names: dict[int, str], path: Path,
) -> Path:
    names = [shorten_label(class_names.get(cid, str(cid))) for cid in class_names]
    tm = np.array(
        [fn_attribution.get(cid, {}).get("true_miss", 0) for cid in class_names],
        dtype=np.float32,
    )
    uc = np.array(
        [fn_attribution.get(cid, {}).get("under_confidence", 0) for cid in class_names],
        dtype=np.float32,
    )
    lf = np.array(
        [fn_attribution.get(cid, {}).get("localization_fail", 0) for cid in class_names],
        dtype=np.float32,
    )
    total = tm + uc + lf
    if total.sum() == 0:
        fig, ax = new_figure(n_items=1, figsize=(9, 4.5))
        ax.text(0.5, 0.5, "no FN — nothing to attribute", ha="center", va="center",
                fontsize=12, color="#555555")
        ax.set_axis_off()
        return _savefig(fig, path)

    fig, (ax_abs, ax_pct) = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    x = np.arange(len(names))
    ax_abs.bar(x, tm, label="true_miss", color="#c44e52")
    ax_abs.bar(x, uc, bottom=tm, label="under_confidence", color="#dd8452")
    ax_abs.bar(x, lf, bottom=tm + uc, label="localization_fail", color="#8172b2")
    for i, t in enumerate(total):
        if t > 0:
            ax_abs.text(i, t + 0.5, f"n={int(t)}", ha="center", fontsize=8)
    ax_abs.set_xticks(x)
    _rotate_xticks(ax_abs, names)
    ax_abs.set_ylabel("FN count")
    ax_abs.set_title("(a) FN causality — absolute")
    ax_abs.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    ax_abs.grid(axis="y", alpha=0.3)

    tm_p = np.where(total > 0, tm / np.maximum(total, 1), 0)
    uc_p = np.where(total > 0, uc / np.maximum(total, 1), 0)
    lf_p = np.where(total > 0, lf / np.maximum(total, 1), 0)
    ax_pct.bar(x, tm_p, color="#c44e52")
    ax_pct.bar(x, uc_p, bottom=tm_p, color="#dd8452")
    ax_pct.bar(x, lf_p, bottom=tm_p + uc_p, color="#8172b2")
    for i in range(len(names)):
        if uc_p[i] > 0.05:
            ax_pct.text(i, tm_p[i] + uc_p[i] / 2, f"{uc_p[i]*100:.0f}%",
                         ha="center", va="center", fontsize=7, color="white")
    ax_pct.set_xticks(x)
    _rotate_xticks(ax_pct, names)
    ax_pct.set_ylim(0, 1.05)
    ax_pct.set_ylabel("share of FN")
    ax_pct.set_title("(b) FN causality — normalized")
    ax_pct.grid(axis="y", alpha=0.3)
    return _savefig(fig, path)


def _plot_boxes_per_image(counts: list[int], path: Path) -> Path:
    if not counts:
        fig, ax = new_figure(n_items=1)
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no GT", ha="center", va="center")
        return _savefig(fig, path)
    counts_arr = np.asarray(counts, dtype=np.int32)
    max_n = max(1, int(counts_arr.max()))
    bins = np.arange(0, max_n + 2) - 0.5
    fig, ax = new_figure(n_items=max_n, figsize=(10, 5))
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
    return _savefig(fig, path)


def _plot_bbox_aspect_ratio(gt_aspect_ratios, class_names, path: Path) -> Path:
    flat = [(cid, r) for cid, ratios in gt_aspect_ratios.items() for r in ratios]
    fig, ax = new_figure(n_items=len(class_names), figsize=(10, 5.5))
    if not flat:
        ax.text(0.5, 0.5, "no GT", ha="center", va="center")
        ax.set_axis_off()
        return _savefig(fig, path)
    bins = np.logspace(np.log10(0.1), np.log10(10.0), 26)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(class_names), 10)))
    for i, cid in enumerate(sorted(gt_aspect_ratios.keys())):
        ratios = gt_aspect_ratios.get(cid, [])
        if not ratios:
            continue
        ax.hist(ratios, bins=bins, alpha=0.55, color=colors[i % 10],
                label=f"{shorten_label(class_names.get(cid, str(cid)))} (n={len(ratios)})")
    ax.axvline(1.0, color="black", ls="--", lw=1, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("bbox aspect ratio (w/h, log scale)")
    ax.set_ylabel("count")
    ax.set_title("GT bbox aspect-ratio distribution per class")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    ax.grid(alpha=0.3, which="both")
    return _savefig(fig, path)


def _plot_size_recall(size_stats: dict, path: Path) -> Path:
    tiers = ["small", "medium", "large"]
    rec = []
    prec = []
    for t in tiers:
        tp = size_stats[t]["tp"]
        fp = size_stats[t]["fp"]
        fn = size_stats[t]["fn"]
        rec.append(tp / (tp + fn) if (tp + fn) else 0.0)
        prec.append(tp / (tp + fp) if (tp + fp) else 0.0)
    fig, ax = new_figure(n_items=3, figsize=(10, 5.5))
    x = np.arange(3)
    w = 0.4
    pbars = ax.bar(x - w/2, prec, w, label="Precision", color="#4c72b0")
    rbars = ax.bar(x + w/2, rec,  w, label="Recall",    color="#55a868")
    for bars, vals in [(pbars, prec), (rbars, rec)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.015, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)
    for i, t in enumerate(tiers):
        c = size_stats[t]
        ax.text(i, 1.08, f"TP {c['tp']}   FP {c['fp']}   FN {c['fn']}",
                ha="center", fontsize=9, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([SIZE_TIER_LABELS[t] for t in tiers], fontsize=9)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("score")
    ax.set_title("Size-stratified detection performance")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    return _savefig(fig, path)


def _plot_data_distribution(
    gt_per_class, gt_per_class_size, class_names, path: Path,
) -> Path:
    ordered_cids = sorted(gt_per_class.keys())
    names = [shorten_label(class_names.get(cid, str(cid))) for cid in ordered_cids]
    totals = [gt_per_class[cid] for cid in ordered_cids]
    small = [gt_per_class_size.get(cid, {}).get("small", 0) for cid in ordered_cids]
    medium = [gt_per_class_size.get(cid, {}).get("medium", 0) for cid in ordered_cids]
    large = [gt_per_class_size.get(cid, {}).get("large", 0) for cid in ordered_cids]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=fit_figsize(len(names), base=13, per_item=0.6,
                                   min_w=12, max_w=24, height=5.5),
        constrained_layout=True,
    )
    x = np.arange(len(names))
    total = max(1, sum(totals))
    y_max = max(totals) * 1.25 if totals else 1.0

    ax1.bar(x, totals, color="#4c72b0")
    for i, v in enumerate(totals):
        pct = 100.0 * v / total
        ax1.text(i, v + max(totals) * 0.02, f"{v} ({pct:.1f}%)",
                 ha="center", fontsize=9)
    ax1.set_xticks(x)
    _rotate_xticks(ax1, names)
    ax1.set_ylim(0, y_max)
    ax1.set_ylabel("GT box count")
    ax1.set_title(f"Class distribution (total = {total})")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, small, label=SIZE_TIER_LABELS["small"], color="#dd8452")
    ax2.bar(x, medium, bottom=small, label=SIZE_TIER_LABELS["medium"], color="#8172b3")
    bottom_ml = [s + m for s, m in zip(small, medium)]
    ax2.bar(x, large, bottom=bottom_ml, label=SIZE_TIER_LABELS["large"], color="#64b5cd")
    for i, (s_n, m_n, l_n) in enumerate(zip(small, medium, large)):
        tot = s_n + m_n + l_n
        if tot > 0:
            ax2.text(i, tot + max(totals) * 0.02, str(int(tot)), ha="center", fontsize=9)
    ax2.set_xticks(x)
    _rotate_xticks(ax2, names)
    ax2.set_ylim(0, y_max)
    ax2.set_ylabel("GT box count")
    ax2.set_title("Per-class × per-size-tier")
    ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    ax2.grid(axis="y", alpha=0.3)
    return _savefig(fig, path)


# ===========================================================================
# CLASSIFICATION
# ===========================================================================


def _analyze_classification(
    *, model, dataset, output_dir: Path,
    class_names: dict[int, str], input_size: tuple[int, int], style: VizStyle,
    max_samples: int | None, hard_images_per_class: int,
) -> dict[str, Any]:
    raw_ds, idx_map = _unwrap(dataset)
    indices = _sampling_indices(len(dataset), max_samples)
    device = next(model.parameters()).device
    input_h, input_w = int(input_size[0]), int(input_size[1])

    C = len(class_names)
    cm = np.zeros((C, C), dtype=np.int64)
    per_class = {cid: {"tp": 0, "fp": 0, "fn": 0} for cid in class_names}
    correct_scores: list[float] = []
    wrong_scores: list[float] = []
    per_image: list[dict] = []
    # keyed by (gt_cid, pred_cid) for class_confusion-style grouping
    wrong_gallery: dict[tuple[int, int], list[dict]] = {}
    # count per-class GT for data distribution
    gt_per_class: dict[int, int] = {c: 0 for c in class_names}

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
        tensor = (
            _preprocess_for_model(image, (input_h, input_w), model=model).unsqueeze(0).to(device)
        )
        with torch.no_grad():
            out = _dispatch_forward(model, tensor)
        logits = out.logits if hasattr(out, "logits") else out
        probs = torch.softmax(logits, dim=-1).cpu().numpy().ravel()
        pred = int(np.argmax(probs))
        score = float(probs[pred])
        gt_cid = int(gt)
        gt_per_class[gt_cid] = gt_per_class.get(gt_cid, 0) + 1
        if 0 <= gt_cid < C and 0 <= pred < C:
            cm[gt_cid, pred] += 1
        if pred == gt_cid:
            per_class[gt_cid]["tp"] += 1
            correct_scores.append(score)
        else:
            per_class[pred]["fp"] += 1
            per_class[gt_cid]["fn"] += 1
            wrong_scores.append(score)
            wrong_gallery.setdefault((gt_cid, pred), []).append({
                "image_idx": int(real_idx),
                "path": raw.get("path", ""),
                "gt_cid": gt_cid, "pred_cid": pred, "score": score,
            })
        per_image.append({
            "idx": int(real_idx), "path": raw.get("path", ""),
            "correct": bool(pred == gt_cid), "score": score,
            "gt": gt_cid, "pred": pred,
        })

    # ---- summary ----
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
    overall_acc = sum(1 for r in per_image if r["correct"]) / max(1, len(per_image))
    summary = {
        "task": "classification",
        "overall_accuracy": round(overall_acc, 4),
        "per_class": out_classes,
        "data_distribution": {
            class_names.get(cid, str(cid)): int(v) for cid, v in gt_per_class.items()
        },
        "confidence": {
            "correct_count": len(correct_scores),
            "wrong_count": len(wrong_scores),
            "correct_mean_score": (
                round(float(np.mean(correct_scores)), 4) if correct_scores else None
            ),
            "wrong_mean_score": round(float(np.mean(wrong_scores)), 4) if wrong_scores else None,
        },
    }

    artifacts: dict[str, Any] = {}

    # --- 01 overview ---
    artifacts["overview"] = _plot_overview(
        "Classification — Error Analysis Overview",
        [
            f"Samples analyzed    : {len(per_image)}",
            f"Classes             : {C}",
            f"Overall accuracy    : {overall_acc:.4f}",
            f"Correct predictions : {len(correct_scores)}",
            f"Wrong predictions   : {len(wrong_scores)}",
        ],
        _chart_path(output_dir, "overview"),
    )
    # --- 02 data distribution ---
    artifacts["data_distribution"] = _plot_per_class_bars(
        {class_names.get(cid, str(cid)): float(gt_per_class.get(cid, 0)) for cid in class_names},
        _chart_path(output_dir, "data_distribution"),
        title="Class distribution (samples per class)",
        ylabel="# samples", ylim=None,
        value_fmt="{:.0f}",
    )
    # --- 03 per-class P/R/F1 ---
    artifacts["per_class_performance"] = _plot_per_class_prf1(
        per_class, class_names, _chart_path(output_dir, "per_class_performance"),
        title_prefix="Per-class",
    )
    # --- 04 confusion or top pairs ---
    cm_padded = np.zeros((C + 1, C + 1), dtype=np.int64)
    cm_padded[:C, :C] = cm
    if C <= _LARGE_CLASS_THRESHOLD:
        artifacts["confusion_matrix"] = _plot_confusion_matrix(
            cm_padded, class_names, _chart_path(output_dir, "confusion_matrix"),
        )
    else:
        artifacts["top_confused_pairs"] = _plot_top_confused_pairs(
            cm_padded, class_names, _chart_path(output_dir, "top_confused_pairs"),
        )
    # --- 05 confidence calibration ---
    artifacts["confidence_calibration"] = _plot_confidence_hist(
        correct_scores, wrong_scores,
        _chart_path(output_dir, "confidence_calibration"),
    )
    # --- 06 failure-mode contribution (wrong-class bar) ---
    wrong_per_class = {
        class_names.get(cid, str(cid)): per_class[cid]["fn"] for cid in class_names
    }
    artifacts["failure_mode_contribution"] = _plot_per_class_bars(
        wrong_per_class,
        _chart_path(output_dir, "failure_mode_contribution"),
        title="FN count per class (true class mis-classified)",
        ylabel="# wrong (FN)", ylim=None, value_fmt="{:.0f}",
    )
    # --- 08 hardest images overview (lowest-conf correct + highest-conf wrong) ---
    per_image.sort(key=lambda r: (r["correct"], -r["score"] if not r["correct"] else r["score"]))
    hard_imgs: list[np.ndarray] = []
    hard_titles: list[str] = []
    for rec in per_image[:12]:
        try:
            raw = raw_ds.get_raw_item(rec["idx"])
        except Exception:
            continue
        if raw["image"] is None:
            continue
        rgb = _bgr_to_rgb(raw["image"])
        panel = render_gt_pred_side_by_side(
            rgb, rec["gt"], (rec["pred"], rec["score"]),
            task="classification", class_names=class_names, style=style,
        )
        hard_imgs.append(panel)
        gt_n = class_names.get(rec["gt"], str(rec["gt"]))
        pr_n = class_names.get(rec["pred"], str(rec["pred"]))
        hard_titles.append(
            f"{'✓' if rec['correct'] else '✗'}  gt={shorten_label(gt_n)} "
            f"pred={shorten_label(pr_n)} ({rec['score']:.2f})"
        )
    p = _plot_hardest_images_grid(
        hard_imgs, hard_titles, _chart_path(output_dir, "hardest_images"),
        header="Hardest / most confident-wrong predictions",
    )
    if p is not None:
        artifacts["hardest_images"] = p

    # --- 09 failure-mode examples: misclassified / true__as__pred ---
    if hard_images_per_class > 0:
        examples_root = output_dir / CHART_FILENAMES["failure_mode_examples"]
        artifacts["failure_mode_examples_root"] = examples_root
        for (gt_cid, pred_cid), items in wrong_gallery.items():
            cases = []
            for rec in items:
                try:
                    raw = raw_ds.get_raw_item(rec["image_idx"])
                except Exception:
                    continue
                if raw["image"] is None:
                    continue
                cases.append({
                    "image": raw["image"],
                    "gt": rec["gt_cid"],
                    "pred": (rec["pred_cid"], rec["score"]),
                    "stem": Path(rec["path"]).stem or f"img_{rec['image_idx']}",
                    "banner": {
                        "title": f"misclassified: gt={class_names.get(gt_cid, str(gt_cid))} "
                                 f"as={class_names.get(pred_cid, str(pred_cid))}",
                        "subtitle": f"score={rec['score']:.2f}",
                    },
                    "_sort": -rec["score"],
                })
            folder = f"{_safe_name(class_names.get(gt_cid, str(gt_cid)))}__as__" \
                     f"{_safe_name(class_names.get(pred_cid, str(pred_cid)))}"
            _render_gallery_side_by_side(
                cases, examples_root / "misclassified" / folder,
                task="classification", class_names=class_names, style=style,
                cap=hard_images_per_class,
                sort_key=lambda r: r["_sort"],
                suffix_fn=lambda r: f"score_{-r['_sort']:.2f}",
            )

    chart_refs = [CHART_FILENAMES[k] for k in (
        "overview", "data_distribution", "per_class_performance",
        "confusion_matrix" if C <= _LARGE_CLASS_THRESHOLD else "top_confused_pairs",
        "confidence_calibration", "failure_mode_contribution", "hardest_images",
    )]
    json_path, md_path = _write_json_md(
        output_dir, summary,
        title="Classification Error Analysis",
        header=[
            f"- Samples analyzed: **{len(per_image)}**",
            f"- Overall accuracy: **{overall_acc:.4f}**",
            f"- Classes: **{C}**",
        ],
        chart_refs=chart_refs,
    )
    artifacts["summary_json"] = json_path
    artifacts["summary_md"] = md_path
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
    indices = _sampling_indices(len(dataset), max_samples)
    device = next(model.parameters()).device
    input_h, input_w = int(input_size[0]), int(input_size[1])

    C = len(class_names)
    intersect = np.zeros(C, dtype=np.float64)
    union = np.zeros(C, dtype=np.float64)
    per_image_records: list[dict] = []
    gt_pixel_counts: dict[int, int] = {c: 0 for c in class_names}
    # per-class per-image IoUs → used for gallery bucketing + failure-mode bars
    per_class_iou_per_image: dict[int, list[dict]] = {c: [] for c in class_names}

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
        tensor = (
            _preprocess_for_model(image, (input_h, input_w), model=model).unsqueeze(0).to(device)
        )
        with torch.no_grad():
            out = _dispatch_forward(model, tensor)
        logits = out.logits if hasattr(out, "logits") else out
        pred_mask = logits.argmax(dim=1)[0].cpu().numpy()
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask.astype(np.int32),
                                    (gt_mask.shape[1], gt_mask.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
        img_inter = 0.0
        img_union = 0.0
        for cid in range(C):
            gt_bin = (gt_mask == cid)
            pr_bin = (pred_mask == cid)
            inter = float(np.logical_and(gt_bin, pr_bin).sum())
            uni = float(np.logical_or(gt_bin, pr_bin).sum())
            intersect[cid] += inter
            union[cid] += uni
            img_inter += inter
            img_union += uni
            gt_pixel_counts[cid] = gt_pixel_counts.get(cid, 0) + int(gt_bin.sum())
            if uni > 0:
                per_class_iou_per_image[cid].append({
                    "image_idx": int(real_idx), "path": raw.get("path", ""),
                    "iou": float(inter / uni),
                    "gt_mask": gt_bin, "pred_mask": pr_bin,
                })
        per_image_records.append({
            "idx": int(real_idx), "path": raw.get("path", ""),
            "miou": float(img_inter / img_union) if img_union > 0 else 0.0,
            "gt_mask": gt_mask, "pred_mask": pred_mask,
        })

    ious = np.where(union > 0, intersect / (union + 1e-9), 0.0)
    mean_iou = float(ious.mean()) if ious.size else 0.0
    summary = {
        "task": "segmentation",
        "mean_iou": round(mean_iou, 4),
        "per_class_iou": {class_names.get(i, str(i)): round(float(ious[i]), 4) for i in range(C)},
        "data_distribution": {
            class_names.get(cid, str(cid)): int(v) for cid, v in gt_pixel_counts.items()
        },
    }

    artifacts: dict[str, Any] = {}

    # --- 01 overview ---
    artifacts["overview"] = _plot_overview(
        "Segmentation — Error Analysis Overview",
        [
            f"Samples analyzed : {len(per_image_records)}",
            f"Classes          : {C}",
            f"Mean IoU         : {mean_iou:.4f}",
            f"Best class IoU   : {float(ious.max()):.4f}",
            f"Worst class IoU  : {float(ious.min()):.4f}",
        ],
        _chart_path(output_dir, "overview"),
    )
    # --- 02 data distribution (GT pixels per class) ---
    artifacts["data_distribution"] = _plot_per_class_bars(
        {class_names.get(cid, str(cid)): float(gt_pixel_counts.get(cid, 0)) for cid in class_names},
        _chart_path(output_dir, "data_distribution"),
        title="GT pixel count per class",
        ylabel="# pixels", ylim=None, value_fmt="{:.0f}",
    )
    # --- 03 per-class IoU ---
    artifacts["per_class_performance"] = _plot_per_class_bars(
        {class_names.get(cid, str(cid)): float(ious[cid]) for cid in range(C)},
        _chart_path(output_dir, "per_class_performance"),
        title="Per-class IoU",
        ylabel="IoU",
    )
    # --- 06 failure-mode contribution: per-class IoU gap vs mean ---
    gap = {class_names.get(cid, str(cid)): max(0.0, mean_iou - float(ious[cid]))
           for cid in range(C)}
    artifacts["failure_mode_contribution"] = _plot_per_class_bars(
        gap, _chart_path(output_dir, "failure_mode_contribution"),
        title=f"Per-class IoU gap vs mean ({mean_iou:.3f}) — larger bar = worse",
        ylabel="mean − class IoU", ylim=(0, 1),
    )
    # --- 08 hardest-images overview (lowest mIoU) ---
    per_image_records.sort(key=lambda r: r["miou"])
    hard_imgs: list[np.ndarray] = []
    hard_titles: list[str] = []
    for rec in per_image_records[:12]:
        try:
            raw = raw_ds.get_raw_item(rec["idx"])
        except Exception:
            continue
        if raw["image"] is None:
            continue
        rgb = _bgr_to_rgb(raw["image"])
        panel = render_gt_pred_side_by_side(
            rgb, rec["gt_mask"], rec["pred_mask"],
            task="segmentation", class_names=class_names, style=style,
            banner={"title": "low mIoU", "subtitle": f"mIoU={rec['miou']:.2f}"},
        )
        hard_imgs.append(panel)
        hard_titles.append(f"mIoU={rec['miou']:.2f}")
    p = _plot_hardest_images_grid(
        hard_imgs, hard_titles, _chart_path(output_dir, "hardest_images"),
        header="Hardest images — lowest mean IoU",
    )
    if p is not None:
        artifacts["hardest_images"] = p

    # --- 09 failure-mode examples: low_iou per class ---
    if hard_images_per_class > 0:
        examples_root = output_dir / CHART_FILENAMES["failure_mode_examples"]
        artifacts["failure_mode_examples_root"] = examples_root
        LOW_IOU_THR = 0.5
        for cid in class_names:
            items = [r for r in per_class_iou_per_image[cid] if r["iou"] < LOW_IOU_THR]
            if not items:
                continue
            cases = []
            for rec in items:
                try:
                    raw = raw_ds.get_raw_item(rec["image_idx"])
                except Exception:
                    continue
                if raw["image"] is None:
                    continue
                cases.append({
                    "image": raw["image"],
                    "gt": rec["gt_mask"],
                    "pred": rec["pred_mask"],
                    "stem": Path(rec["path"]).stem or f"img_{rec['image_idx']}",
                    "banner": {
                        "title": f"low_iou: {class_names.get(cid, str(cid))}",
                        "subtitle": f"IoU={rec['iou']:.2f}",
                    },
                    "_sort": rec["iou"],
                })
            _render_gallery_side_by_side(
                cases, examples_root / "low_iou" / _safe_name(class_names.get(cid, str(cid))),
                task="segmentation", class_names=class_names, style=style,
                cap=hard_images_per_class,
                sort_key=lambda r: r["_sort"],
                suffix_fn=lambda r: f"iou_{r['_sort']:.2f}",
            )

    chart_refs = [CHART_FILENAMES[k] for k in (
        "overview", "data_distribution", "per_class_performance",
        "failure_mode_contribution", "hardest_images",
    )]
    json_path, md_path = _write_json_md(
        output_dir, summary,
        title="Segmentation Error Analysis",
        header=[
            f"- Samples analyzed: **{len(per_image_records)}**",
            f"- Mean IoU: **{mean_iou:.4f}**",
        ],
        chart_refs=chart_refs,
    )
    artifacts["summary_json"] = json_path
    artifacts["summary_md"] = md_path
    return artifacts


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
    indices = _sampling_indices(len(dataset), max_samples)
    device = next(model.parameters()).device
    input_h, input_w = int(input_size[0]), int(input_size[1])

    per_kp_correct: dict[int, int] = {}
    per_kp_total: dict[int, int] = {}
    per_kp_errors: dict[int, list[dict]] = {}
    per_image_records: list[dict] = []

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
        gt_kp = np.asarray(gt.get("keypoints") if isinstance(gt, dict) else gt,
                           dtype=np.float32).reshape(-1, 3)
        if gt_kp.size == 0:
            continue
        tensor = (
            _preprocess_for_model(image, (input_h, input_w), model=model).unsqueeze(0).to(device)
        )
        with torch.no_grad():
            preds_raw = _dispatch_forward(model, tensor)
        target_sizes = torch.tensor([[input_h, input_w]], device=device)
        decoded = _dispatch_postprocess(model, preds_raw, conf_threshold, target_sizes)[0]
        pred_kp = np.asarray(decoded.get("keypoints", []), dtype=np.float32).reshape(-1, 3)

        vis = gt_kp[:, 2] > 0
        if vis.any():
            x_min, y_min = gt_kp[vis, :2].min(axis=0)
            x_max, y_max = gt_kp[vis, :2].max(axis=0)
        else:
            x_min = y_min = 0
            x_max = y_max = 1
        diag = max(1.0, float(np.hypot(x_max - x_min, y_max - y_min)))
        thr = 0.2 * diag

        correct = total = 0
        per_kp_err = {}
        K_pred = len(pred_kp)
        for k in range(len(gt_kp)):
            if gt_kp[k, 2] <= 0:
                continue
            per_kp_total[k] = per_kp_total.get(k, 0) + 1
            total += 1
            err_px = float("inf")
            ok = False
            if k < K_pred and pred_kp[k, 2] > 0:
                d = float(np.hypot(pred_kp[k, 0] - gt_kp[k, 0],
                                    pred_kp[k, 1] - gt_kp[k, 1]))
                err_px = d
                if d <= thr:
                    per_kp_correct[k] = per_kp_correct.get(k, 0) + 1
                    correct += 1
                    ok = True
            per_kp_err[k] = err_px
            if not ok:
                per_kp_errors.setdefault(k, []).append({
                    "image_idx": int(real_idx), "path": raw.get("path", ""),
                    "joint": k, "err_px": err_px,
                    "gt_kp": gt_kp.copy(), "pred_kp": pred_kp.copy(),
                })
        per_image_records.append({
            "idx": int(real_idx), "path": raw.get("path", ""),
            "pck": (correct / max(1, total)),
            "gt_kp": gt_kp.copy(), "pred_kp": pred_kp.copy(),
        })

    per_kp = {}
    for k in sorted(per_kp_total):
        c = per_kp_correct.get(k, 0)
        t = per_kp_total[k]
        per_kp[k] = {"pck": round(c / max(1, t), 4), "total": t}
    mean_pck = float(np.mean([v["pck"] for v in per_kp.values()])) if per_kp else 0.0
    summary = {
        "task": "keypoint",
        "mean_pck": round(mean_pck, 4),
        "per_keypoint": {f"kp_{k}": v for k, v in per_kp.items()},
        "data_distribution": {f"kp_{k}": v["total"] for k, v in per_kp.items()},
    }

    artifacts: dict[str, Any] = {}

    # --- 01 overview ---
    artifacts["overview"] = _plot_overview(
        "Keypoint — Error Analysis Overview",
        [
            f"Samples analyzed : {len(per_image_records)}",
            f"Keypoints        : {len(per_kp)}",
            f"Mean PCK@0.2     : {mean_pck:.4f}",
        ],
        _chart_path(output_dir, "overview"),
    )
    # --- 02 data distribution (visible count per joint) ---
    artifacts["data_distribution"] = _plot_per_class_bars(
        {f"kp_{k}": float(v["total"]) for k, v in per_kp.items()},
        _chart_path(output_dir, "data_distribution"),
        title="Visible keypoint count per joint",
        ylabel="# visible", ylim=None, value_fmt="{:.0f}",
    )
    # --- 03 per-joint PCK ---
    artifacts["per_class_performance"] = _plot_per_class_bars(
        {f"kp_{k}": float(v["pck"]) for k, v in per_kp.items()},
        _chart_path(output_dir, "per_class_performance"),
        title="Per-joint PCK@0.2",
        ylabel="PCK",
    )
    # --- 06 failure contribution: per-joint PCK gap vs mean ---
    gap = {f"kp_{k}": max(0.0, mean_pck - float(v["pck"])) for k, v in per_kp.items()}
    artifacts["failure_mode_contribution"] = _plot_per_class_bars(
        gap, _chart_path(output_dir, "failure_mode_contribution"),
        title=f"Per-joint PCK gap vs mean ({mean_pck:.3f})",
        ylabel="mean − joint PCK", ylim=(0, 1),
    )
    # --- 08 hardest images overview ---
    per_image_records.sort(key=lambda r: r["pck"])
    hard_imgs: list[np.ndarray] = []
    hard_titles: list[str] = []
    for rec in per_image_records[:12]:
        try:
            raw = raw_ds.get_raw_item(rec["idx"])
        except Exception:
            continue
        if raw["image"] is None:
            continue
        rgb = _bgr_to_rgb(raw["image"])
        panel = render_gt_pred_side_by_side(
            rgb, rec["gt_kp"], rec["pred_kp"],
            task="keypoint", class_names=class_names, style=style,
            banner={"title": "low PCK", "subtitle": f"PCK={rec['pck']:.2f}"},
        )
        hard_imgs.append(panel)
        hard_titles.append(f"PCK={rec['pck']:.2f}")
    p = _plot_hardest_images_grid(
        hard_imgs, hard_titles, _chart_path(output_dir, "hardest_images"),
        header="Hardest images — lowest per-image PCK",
    )
    if p is not None:
        artifacts["hardest_images"] = p

    # --- 09 high-error gallery per joint ---
    if hard_images_per_class > 0:
        examples_root = output_dir / CHART_FILENAMES["failure_mode_examples"]
        artifacts["failure_mode_examples_root"] = examples_root
        for k, items in per_kp_errors.items():
            cases = []
            for rec in items:
                try:
                    raw = raw_ds.get_raw_item(rec["image_idx"])
                except Exception:
                    continue
                if raw["image"] is None:
                    continue
                # Dim other joints: only the failing joint carries confidence.
                gt_kp = rec["gt_kp"].copy()
                pred_kp = rec["pred_kp"].copy()
                # Zero visibility on all but joint k so the renderer hides them.
                if gt_kp.shape[1] == 3:
                    mask = np.zeros(gt_kp.shape[0], dtype=bool)
                    if k < len(mask):
                        mask[k] = True
                    gt_kp[~mask, 2] = 0.0
                if pred_kp.size and pred_kp.shape[1] == 3:
                    mask_p = np.zeros(pred_kp.shape[0], dtype=bool)
                    if k < len(mask_p):
                        mask_p[k] = True
                    pred_kp[~mask_p, 2] = 0.0
                err = rec["err_px"]
                err_txt = f"err={err:.1f}px" if np.isfinite(err) else "err=missing"
                cases.append({
                    "image": raw["image"],
                    "gt": gt_kp, "pred": pred_kp,
                    "stem": Path(rec["path"]).stem or f"img_{rec['image_idx']}",
                    "banner": {
                        "title": f"high_error: joint kp_{k}",
                        "subtitle": err_txt,
                    },
                    "_sort": -err if np.isfinite(err) else -1e9,
                })
            _render_gallery_side_by_side(
                cases, examples_root / "high_error" / f"kp_{k}",
                task="keypoint", class_names=class_names, style=style,
                cap=hard_images_per_class,
                sort_key=lambda r: r["_sort"],
                suffix_fn=lambda r: f"err_{-r['_sort']:.1f}"
                            if r["_sort"] > -1e8 else "err_missing",
            )

    chart_refs = [CHART_FILENAMES[k] for k in (
        "overview", "data_distribution", "per_class_performance",
        "failure_mode_contribution", "hardest_images",
    )]
    json_path, md_path = _write_json_md(
        output_dir, summary,
        title="Keypoint Error Analysis",
        header=[
            f"- Samples analyzed: **{len(per_image_records)}**",
            f"- Mean PCK@0.2: **{mean_pck:.4f}**",
        ],
        chart_refs=chart_refs,
    )
    artifacts["summary_json"] = json_path
    artifacts["summary_md"] = md_path
    return artifacts

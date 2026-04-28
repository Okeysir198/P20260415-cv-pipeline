#!/usr/bin/env python3
"""Standalone CLI to visualize dataset samples and augmentation before training.

Generates (numbered by pipeline step):
  - 00_dataset_info.{md,json}    — dataset provenance (name, path, classes, splits)
  - 01_dataset_stats.{png,json}  — split sizes + class balance + bbox tiers
  - 02_data_labels_<split>.png   — raw images with GT bounding boxes
  - 03_aug_labels_<split>.png    — augmented images with transformed GT boxes
  - 04_transform_pipeline.png    — step-by-step transform walk (per class, max 5 samples); last col = Denormalize(Normalize) sanity check

Usage::

    uv run core/p05_data/run_viz.py \\
        --config features/safety-fire_detection/configs/06_training.yaml

    # Override splits:
    uv run core/p05_data/run_viz.py \\
        --config features/safety-fire_detection/configs/06_training.yaml \\
        --splits train val test

Output lands in a timestamped run dir:
    features/<feature>/runs/<ts>_05_data_viz/data_preview/
"""

import argparse
import datetime
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from core.p05_data.detection_dataset import YOLOXDataset
from core.p05_data.transforms import build_transforms
from utils.config import feature_name_from_config_path, generate_run_dir, load_config
from loguru import logger
from utils.viz import (
    VizStyle,
    annotate_detections,
    apply_plot_style,
    safe_colorbar,
    save_image_grid,
    truncate_label,
    wrap_suptitle,
)

# Unified matplotlib rcParams for all statistical panels in this module.
apply_plot_style()


# Palette used only by the matplotlib class-balance panels below (BGR tuples —
# the `[::-1]` at the call sites flips to RGB for matplotlib).
_LABEL_PALETTE = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (0, 128, 255), (128, 0, 255),
    (0, 255, 128), (255, 128, 0), (64, 0, 255), (255, 64, 0),
    (0, 255, 64), (64, 255, 0), (255, 0, 64), (0, 64, 255),
    (128, 128, 0), (0, 128, 128), (128, 0, 128), (64, 64, 0),
]


def _yolo_targets_to_sv(targets: np.ndarray, h: int, w: int) -> sv.Detections:
    """Convert YOLO-normalized (cls, cx, cy, w, h) rows to ``sv.Detections`` in pixel xyxy."""
    if targets is None or len(targets) == 0:
        return sv.Detections.empty()
    t = np.asarray(targets, dtype=np.float32).reshape(-1, 5)
    cls = t[:, 0].astype(int)
    cx, cy, bw, bh = t[:, 1], t[:, 2], t[:, 3], t[:, 4]
    x1 = (cx - bw / 2) * w
    y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w
    y2 = (cy + bh / 2) * h
    xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    return sv.Detections(xyxy=xyxy, class_id=cls)


def _annotate_gt(image_rgb: np.ndarray, targets: np.ndarray, class_names: dict,
                 thickness: int, text_scale: float) -> np.ndarray:
    """Draw GT boxes + labels. ``thickness`` is ignored — VizStyle auto-scales."""
    del thickness  # auto_box_thickness handles this in VizStyle
    h, w = image_rgb.shape[:2]
    dets = _yolo_targets_to_sv(targets, h, w)
    style = VizStyle(label_text_scale=text_scale)
    return annotate_detections(image_rgb, dets, class_names=class_names, style=style)


# ---------------------------------------------------------------------------
# Visualization functions
# ---------------------------------------------------------------------------

def viz_data_labels(data_cfg, base_dir, class_names, split, num_samples, grid_cols, thickness, text_scale, dpi, out_dir):
    try:
        ds = YOLOXDataset(data_config=data_cfg, split=split, transforms=None, base_dir=base_dir)
    except Exception as e:
        logger.warning("Skipping split %s: %s", split, e)
        return
    n = len(ds)
    if n == 0:
        logger.warning("Split %s is empty, skipping", split)
        return
    k = min(num_samples, n)
    indices = sorted(random.sample(range(n), k))
    annotated = []
    for idx in indices:
        item = ds.get_raw_item(idx)
        img_rgb = cv2.cvtColor(item["image"], cv2.COLOR_BGR2RGB)
        targets = ds._load_label(ds.img_paths[idx])
        if targets is None or len(targets) == 0:
            targets = np.zeros((0, 5), dtype=np.float32)
        annotated.append(_annotate_gt(img_rgb, targets, class_names, thickness, text_scale))
    out_path = out_dir / f"02_data_labels_{split}.png"
    save_image_grid(
        annotated, out_path, cols=grid_cols,
        header=f"Data + Labels [{split}] — {k}/{n} samples",
    )
    logger.info("Saved: %s", out_path)


def _render_aug_samples(ds, indices, mean, std, class_names, thickness, text_scale):
    """Denormalize and annotate augmented samples from a dataset."""
    annotated = []
    for i in indices:
        try:
            result = ds[i]
            aug_tensor, targets_tensor = result[0], result[1]
        except Exception as e:
            logger.warning("Failed sample idx %d: %s", i, e)
            continue
        aug_np = aug_tensor.numpy().transpose(1, 2, 0)
        aug_np = np.clip(aug_np * std + mean, 0, 1)
        aug_rgb = (aug_np * 255).astype(np.uint8)
        targets_np = targets_tensor.numpy() if len(targets_tensor) > 0 else np.zeros((0, 5), dtype=np.float32)
        annotated.append(_annotate_gt(aug_rgb, targets_np, class_names, thickness, text_scale))
    return annotated


def viz_aug_labels(data_cfg, train_cfg, base_dir, class_names, split, num_samples, grid_cols, thickness, text_scale, dpi, out_dir):
    """Generate two aug grids per split: one without Mosaic/MixUp (simple) and one with (mosaic)."""
    if split != "train":
        logger.info("Aug viz skipped for split %s (no augmentation applied)", split)
        return
    mean = np.array(data_cfg.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32).reshape(1,1,3)
    std = np.array(data_cfg.get("std", [0.229, 0.224, 0.225]), dtype=np.float32).reshape(1,1,3)
    input_size = tuple(data_cfg["input_size"])
    base_aug = train_cfg.get("augmentation", {})

    for label, aug_override in [
        ("simple", {"mosaic": False, "mixup": False, "copypaste": False}),
        ("mosaic", {}),
    ]:
        aug_config = {**base_aug, **aug_override}
        transforms = build_transforms(config=aug_config, is_train=True, input_size=input_size, mean=data_cfg.get("mean"), std=data_cfg.get("std"))
        try:
            ds = YOLOXDataset(data_config=data_cfg, split=split, transforms=transforms, base_dir=base_dir)
        except Exception as e:
            logger.warning("Skipping aug viz (%s) for split %s: %s", label, split, e)
            continue
        n = len(ds)
        if n == 0:
            continue
        k = min(num_samples, n)
        indices = sorted(random.sample(range(n), k))
        annotated = _render_aug_samples(ds, indices, mean, std, class_names, thickness, text_scale)
        if not annotated:
            continue
        suffix = "" if label == "simple" else "_mosaic"
        out_path = out_dir / f"03_aug_labels_{split}{suffix}.png"
        title = f"Augmented + Labels [{split}] ({'no mosaic/mixup' if label == 'simple' else 'with mosaic/mixup'}) — {k}/{n} samples"
        save_image_grid(annotated, out_path, cols=grid_cols, header=title)
        logger.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

_SPLIT_COLORS = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}


def _normalize_task(task: str | None) -> str:
    """Normalize data_cfg['task'] value to a canonical token.

    Accepts ``detection``, ``classification`` / ``cls``,
    ``semantic_segmentation`` / ``segmentation`` / ``seg``,
    ``keypoint`` / ``pose``. Defaults to ``detection``.
    """
    t = (task or "detection").lower().strip()
    if t in {"detection", "det", "object_detection"}:
        return "detection"
    if t in {"classification", "cls", "image_classification"}:
        return "classification"
    if t in {"segmentation", "seg", "semantic_segmentation", "semseg"}:
        return "segmentation"
    if t in {"keypoint", "pose", "keypoints", "pose_estimation"}:
        return "keypoint"
    return "detection"


def _save_stats(fig, json_out: dict, out_dir: Path, dpi: int) -> None:
    """Write 01_dataset_stats.{png,json}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "01_dataset_stats.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    logger.info("Saved: %s", json_path)
    out_path = out_dir / "01_dataset_stats.png"
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as _plt
    _plt.close(fig)
    logger.info("Saved: %s", out_path)


def generate_dataset_stats(
    data_cfg: dict,
    base_dir: str,
    class_names: dict,
    splits: list[str],
    out_dir: Path,
    dpi: int = 150,
    subset_indices: dict[str, list[int]] | None = None,
    subset_active: bool = False,
    subset_pct: float | None = None,
) -> None:
    """Task-aware dataset stats → ``01_dataset_stats.{png,json}``.

    Dispatches on ``data_cfg['task']``:
      - ``detection``     → bbox tiers, class balance, labels/image
      - ``classification``→ class balance + imbalance ratio + img-size distrib
      - ``segmentation``  → pixel-class %, mask coverage, component counts
      - ``keypoint``      → instances + visibility + spatial heatmap + limb lengths

    When ``subset_indices[split]`` is provided, stats reflect only those indices.
    """
    task = _normalize_task(data_cfg.get("task"))
    if task == "classification":
        return _stats_classification(data_cfg, base_dir, class_names, splits, out_dir, dpi, subset_indices, subset_active, subset_pct)
    if task == "segmentation":
        return _stats_segmentation(data_cfg, base_dir, class_names, splits, out_dir, dpi, subset_indices, subset_active, subset_pct)
    if task == "keypoint":
        return _stats_keypoint(data_cfg, base_dir, class_names, splits, out_dir, dpi, subset_indices, subset_active, subset_pct)
    return _stats_detection(data_cfg, base_dir, class_names, splits, out_dir, dpi, subset_indices, subset_active, subset_pct)


# ---------------------------------------------------------------------------
# Shared per-image samplers (resolution + pixel mean/std)
# ---------------------------------------------------------------------------


def _sample_image_meta(
    img_paths: list,
    sample_cap: int | None = 2048,
    pixel_downscale: int = 64,
    seed: int = 0,
    full_resolution: bool = True,
    wh_cache: dict[str, tuple[int, int]] | None = None,
) -> dict:
    """Per-image sample for resolution scatter + per-channel mean/std.

    Resolution is read from EVERY image by default (PIL header-only —
    cheap, ~0.1 ms/image). Pixel mean/std reads are heavier (full image
    decode + downscale to ``pixel_downscale``) so they are sampled —
    capped at ``sample_cap`` (set to ``None`` for full).

    Returns:
        widths/heights: full per-image lists if ``full_resolution`` else
            sampled. Always at least the pixel-sample subset.
        scatter: same list capped at 4096 entries for chart use.
        mean_rgb / std_rgb: in [0, 1].
        pixel_sample_n: actual #images used for mean/std.
        resolution_full: whether resolution covers full split.
    """
    try:
        from PIL import Image as _PILImage
    except Exception:
        _PILImage = None

    n = len(img_paths)
    if n == 0:
        return {
            "widths": [], "heights": [], "scatter": [],
            "mean_rgb": [0.0, 0.0, 0.0], "std_rgb": [0.0, 0.0, 0.0],
            "pixel_sample_n": 0,
            "resolution_full": True, "resolution_n": 0, "n_images": 0,
        }
    rng = random.Random(seed)
    cap = n if sample_cap is None else min(sample_cap, n)
    pixel_idx = set(rng.sample(range(n), cap))

    widths: list[int] = []
    heights: list[int] = []
    ch_sum = np.zeros(3, dtype=np.float64)
    ch_sumsq = np.zeros(3, dtype=np.float64)
    n_pix = 0

    for i, p in enumerate(img_paths):
        in_pixel_sample = i in pixel_idx
        # Resolution pass — PIL header for every image (full) when enabled,
        # else only the pixel-sample subset.
        if full_resolution or in_pixel_sample:
            W = H = 0
            cached = wh_cache.get(str(p)) if wh_cache else None
            if cached is not None:
                W, H = cached
            else:
                if _PILImage is not None:
                    try:
                        with _PILImage.open(str(p)) as im:
                            W, H = im.size
                    except Exception:
                        W = H = 0
                if W <= 0 or H <= 0:
                    try:
                        im = cv2.imread(str(p))
                        if im is not None:
                            H, W = im.shape[:2]
                    except Exception:
                        pass
            if W > 0 and H > 0:
                widths.append(int(W))
                heights.append(int(H))
        # Pixel stats — sampled subset only.
        if not in_pixel_sample:
            continue
        try:
            im = cv2.imread(str(p))
            if im is None:
                continue
            if pixel_downscale and (im.shape[0] > pixel_downscale or im.shape[1] > pixel_downscale):
                im = cv2.resize(im, (pixel_downscale, pixel_downscale),
                                interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            flat = rgb.reshape(-1, 3)
            ch_sum += flat.sum(axis=0)
            ch_sumsq += (flat ** 2).sum(axis=0)
            n_pix += flat.shape[0]
        except Exception:
            continue

    if n_pix > 0:
        mean = ch_sum / n_pix
        var = ch_sumsq / n_pix - mean ** 2
        std = np.sqrt(np.clip(var, 0, None))
    else:
        mean = np.zeros(3); std = np.zeros(3)

    return {
        "widths": widths,
        "heights": heights,
        "scatter": list(zip(widths, heights, strict=False))[:4096],
        "mean_rgb": [float(x) for x in mean],
        "std_rgb": [float(x) for x in std],
        "pixel_sample_n": cap,
        "resolution_full": full_resolution,
        "resolution_n": len(widths),
        "n_images": n,
    }


def _chart_caption(ax, text: str, *, fontsize: float = 7.5,
                   color: str = "#444", y_offset: float = -0.30) -> None:
    """One-line italic caption immediately below a subplot.

    Anchored in axes-fraction coords so it follows the subplot on resize.
    Uses a negative y-offset large enough to clear the x-axis label.
    """
    ax.text(
        0.5, y_offset, text,
        transform=ax.transAxes, ha="center", va="top",
        fontsize=fontsize, fontstyle="italic", color=color,
    )


def _panel_resolution_scatter(ax, stats_by_split: dict, split_list: list[str],
                              input_size=None) -> None:
    any_data = False
    for sp in split_list:
        meta = stats_by_split[sp].get("_img_meta", {})
        w = meta.get("widths", []); h = meta.get("heights", [])
        if w and h:
            any_data = True
            ax.scatter(w, h, s=8, alpha=0.4, label=sp,
                       color=_SPLIT_COLORS.get(sp, "#888"))
    if not any_data:
        ax.text(0.5, 0.5, "(no resolution data)", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="#888")
        ax.axis("off")
        return
    if input_size is not None:
        try:
            ih, iw = int(input_size[0]), int(input_size[1])
            ax.axvline(iw, color="#c00", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.axhline(ih, color="#c00", linestyle="--", linewidth=0.8, alpha=0.6)
        except Exception:
            pass
    # Title reflects whether we covered full split or only sampled, with N.
    metas = [stats_by_split[sp].get("_img_meta", {}) for sp in split_list]
    res_n = sum(int(m.get("resolution_n", 0)) for m in metas)
    img_n = sum(int(m.get("n_images", 0)) for m in metas) or sum(
        int(stats_by_split[sp].get("n_images", 0)) for sp in split_list
    )
    full = all(m.get("resolution_full", True) for m in metas)
    if full or (res_n and res_n >= img_n):
        title = f"Image Resolution (full, n={res_n:,})"
    else:
        title = f"Image Resolution (sampled, n={res_n:,}/{img_n:,})"
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("width (px)")
    ax.set_ylabel("height (px)")
    ax.legend(fontsize=8, framealpha=0.9)


def _panel_pixel_stats(ax, stats_by_split: dict, split_list: list[str]) -> None:
    """Per-channel R/G/B mean bars with std whiskers, grouped per split."""
    if not split_list:
        ax.axis("off")
        return
    channels = ["R", "G", "B"]
    chan_colors = ["#d62728", "#2ca02c", "#1f77b4"]
    n_split = len(split_list)
    width = 0.8 / max(n_split, 1)
    x = np.arange(3)
    for i, sp in enumerate(split_list):
        meta = stats_by_split[sp].get("_img_meta", {})
        m = meta.get("mean_rgb", [0, 0, 0])
        s = meta.get("std_rgb", [0, 0, 0])
        offset = (i - n_split / 2 + 0.5) * width
        ax.bar(x + offset, m, width, yerr=s, capsize=3,
               color=[c if n_split == 1 else _SPLIT_COLORS.get(sp, "#888")
                      for c in chan_colors] if n_split == 1 else None,
               label=sp if n_split > 1 else None,
               alpha=0.85,
               edgecolor="#222", linewidth=0.4,
               ecolor="#444")
    ax.set_xticks(x)
    ax.set_xticklabels(channels)
    ax.set_ylim(0, 1.05)
    metas = [stats_by_split[sp].get("_img_meta", {}) for sp in split_list]
    pix_n = sum(int(m.get("pixel_sample_n", 0)) for m in metas)
    img_n = sum(int(m.get("n_images", 0)) for m in metas) or sum(
        int(stats_by_split[sp].get("n_images", 0)) for sp in split_list
    )
    if pix_n and pix_n >= img_n:
        suffix = f"(full, n={pix_n:,})"
    else:
        suffix = f"(sampled, n={pix_n:,}/{img_n:,})"
    ax.set_title(f"Per-Channel Mean ± Std (RGB, [0,1]) {suffix}",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("normalized value")
    if n_split > 1:
        ax.legend(fontsize=8, framealpha=0.9)


def _panel_empty_labels(ax, empty_by_split: dict[str, int],
                        n_by_split: dict[str, int],
                        title: str = "Empty / Degenerate Labels") -> None:
    splits = list(empty_by_split.keys())
    if not splits:
        ax.axis("off")
        return
    n_empty = [empty_by_split[s] for s in splits]
    n_total = [n_by_split.get(s, 0) for s in splits]
    pct = [(e / max(t, 1)) * 100 for e, t in zip(n_empty, n_total, strict=True)]
    colors = [_SPLIT_COLORS.get(s, "#888") for s in splits]
    bars = ax.bar(splits, n_empty, color=colors, alpha=0.85)
    for b, e, p in zip(bars, n_empty, pct, strict=True):
        ax.text(b.get_x() + b.get_width() / 2,
                b.get_height() + max(n_empty + [1]) * 0.02,
                f"{e} ({p:.1f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel("# images")
    mx = max(n_empty + [1])
    ax.set_ylim(0, mx * 1.25)


def _format_suptitle(dataset_name: str, subset_active: bool, subset_pct: float | None) -> str:
    base = f"{dataset_name} — 01_dataset_stats"
    if subset_active and subset_pct is not None:
        base += f" (subset {subset_pct:.0f}%)"
    return wrap_suptitle(base, 80)


def _stats_detection(
    data_cfg: dict,
    base_dir: str,
    class_names: dict,
    splits: list[str],
    out_dir: Path,
    dpi: int = 120,
    subset_indices: dict[str, list[int]] | None = None,
    subset_active: bool = False,
    subset_pct: float | None = None,
) -> None:
    """Detection stats — split sizes, class distribution, bbox area tiers.

    Reads only label .txt files — no image loading — so it's fast even on large splits.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    # Collect stats per split (label-file pass only, no image I/O)
    stats = {}
    for split in splits:
        try:
            ds = YOLOXDataset(data_config=data_cfg, split=split, transforms=None, base_dir=base_dir)
        except Exception as e:
            logger.warning("Stats: skipping split %s — %s", split, e)
            continue

        idx_list = (subset_indices or {}).get(split)
        img_paths = ds.img_paths if idx_list is None else [ds.img_paths[i] for i in idx_list]

        class_counts: dict = {}
        bbox_areas_px: list[float] = []  # px² (w_norm*W * h_norm*H per box)
        bbox_aspects: list[float] = []   # w_px / h_px per box
        bbox_centers: list[tuple[float, float]] = []  # (cx_norm, cy_norm) per box
        labels_per_image: list[int] = []
        n_empty = 0

        # Fast header-only read for image dims via PIL.
        try:
            from PIL import Image as _PILImage
        except Exception:
            _PILImage = None

        for img_path in img_paths:
            labels = ds._load_label(img_path)
            n = len(labels)
            labels_per_image.append(n)
            if n == 0:
                n_empty += 1
            if n == 0 or _PILImage is None:
                W = H = 0
            else:
                try:
                    with _PILImage.open(str(img_path)) as _im:
                        W, H = _im.size
                except Exception:
                    W = H = 0
            for row in labels:
                cid = int(row[0])
                class_counts[cid] = class_counts.get(cid, 0) + 1
                cx_n = float(row[1]); cy_n = float(row[2])
                w_n = float(row[3]); h_n = float(row[4])
                bbox_centers.append((cx_n, cy_n))
                if W > 0 and H > 0:
                    bbox_areas_px.append(w_n * W * h_n * H)
                    if h_n > 0:
                        bbox_aspects.append((w_n * W) / (h_n * H))
                elif h_n > 0:
                    bbox_aspects.append(w_n / h_n)

        stats[split] = {
            "n_images": len(img_paths),
            "class_counts": class_counts,
            "bbox_areas_px": bbox_areas_px,
            "bbox_aspects": bbox_aspects,
            "bbox_centers": bbox_centers,
            "labels_per_image": labels_per_image,
            "n_empty": n_empty,
            "n_annotations": sum(class_counts.values()),
            "_img_meta": _sample_image_meta(img_paths),
        }

    if not stats:
        logger.warning("Stats: no splits could be loaded, skipping dataset_stats.png")
        return

    split_list = [s for s in splits if s in stats]
    all_class_ids = sorted({cid for s in stats.values() for cid in s["class_counts"]})
    split_colors = _SPLIT_COLORS

    # Figure
    total_images = sum(stats[s]["n_images"] for s in split_list)
    total_ann = sum(stats[s]["n_annotations"] for s in split_list)
    split_summary = "  |  ".join(
        f"{sp.capitalize()}: {stats[sp]['n_images']:,}" for sp in split_list
    )

    dataset_name = data_cfg.get("dataset_name") or data_cfg.get("name") or "dataset"
    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(
        _format_suptitle(dataset_name, subset_active, subset_pct)
        + f"\n{split_summary}  |  Total: {total_images:,} images  |  {total_ann:,} annotations",
        fontsize=12, fontweight="bold", y=0.995,
    )
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.85, wspace=0.40)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[1, 2])
    ax6 = fig.add_subplot(gs[2, 0])  # resolution scatter
    ax7 = fig.add_subplot(gs[2, 1])  # pixel mean/std
    ax8 = fig.add_subplot(gs[2, 2])  # empty-label panel
    ax9 = fig.add_subplot(gs[3, 0])  # bbox aspect-ratio histogram
    ax10 = fig.add_subplot(gs[3, 1])  # bbox center 2D heatmap

    # --- [0,0] Images per split ---
    n_imgs = [stats[s]["n_images"] for s in split_list]
    colors = [split_colors.get(s, "#888") for s in split_list]
    bars = ax0.barh(split_list, n_imgs, color=colors)
    ax0.set_title("Images per Split", fontsize=11, fontweight="bold")
    ax0.set_xlabel("# images")
    for bar, n in zip(bars, n_imgs, strict=True):
        ax0.text(
            bar.get_width() + max(n_imgs) * 0.02, bar.get_y() + bar.get_height() / 2,
            f"{n:,}", va="center", fontsize=9,
        )
    ax0.set_xlim(0, max(n_imgs) * 1.25)

    # --- [0,1] Class instances per split (grouped bar) ---
    x = np.arange(len(all_class_ids))
    width = 0.75 / max(len(split_list), 1)
    for i, sp in enumerate(split_list):
        counts = [stats[sp]["class_counts"].get(cid, 0) for cid in all_class_ids]
        offset = (i - len(split_list) / 2 + 0.5) * width
        ax1.bar(x + offset, counts, width, label=sp,
                color=split_colors.get(sp, "#888"), alpha=0.85)
    ax1.set_title("Class Instances per Split", fontsize=11, fontweight="bold")
    ax1.set_xlabel("class")
    ax1.set_ylabel("# instances")
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [truncate_label(class_names.get(cid, str(cid)), 14) for cid in all_class_ids],
        rotation=45, ha="right", fontsize=8,
    )
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.9)
    # Headroom so the legend at upper-right does not overlap the tallest bar.
    y_max = ax1.get_ylim()[1]
    ax1.set_ylim(top=y_max * 1.18)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))

    # --- [0,2] Class balance % (stacked horizontal bar per split) ---
    for i, sp in enumerate(split_list):
        total = sum(stats[sp]["class_counts"].values()) or 1
        left = 0.0
        for cid in all_class_ids:
            pct = stats[sp]["class_counts"].get(cid, 0) / total * 100
            rgb = tuple(c / 255 for c in _LABEL_PALETTE[cid % len(_LABEL_PALETTE)][::-1])
            ax2.barh(sp, pct, left=left, color=rgb,
                     label=class_names.get(cid, str(cid)) if i == 0 else "")
            if pct > 4:
                ax2.text(
                    left + pct / 2, i, f"{pct:.1f}%",
                    ha="center", va="center", fontsize=8,
                    color="white", fontweight="bold",
                )
            left += pct
    ax2.set_title("Class Balance %", fontsize=11, fontweight="bold")
    ax2.set_xlabel("% of annotations")
    ax2.set_xlim(0, 100)
    handles = [
        plt.Rectangle((0, 0), 1, 1,
                       color=tuple(c / 255 for c in _LABEL_PALETTE[cid % len(_LABEL_PALETTE)][::-1]))
        for cid in all_class_ids
    ]
    ax2.legend(handles, [class_names.get(cid, str(cid)) for cid in all_class_ids],
               fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0),
               framealpha=1.0)

    # --- [1,0] BBox area distribution in px² (adaptive log-scale bins) ---
    all_areas = [a for s in split_list for a in stats[s]["bbox_areas_px"]]
    if all_areas:
        lo = max(min(all_areas), 1.0)
        hi = max(max(all_areas), lo * 10.0)
        bins = np.logspace(np.log10(lo), np.log10(hi), 30)

        # COCO-style reference bands in px² (small <32², medium <96²) for context.
        ref_bands = [
            (lo, 32 * 32, "small\n(<32²)", "#ffcccc"),
            (32 * 32, 96 * 96, "medium\n(32²-96²)", "#fff0b3"),
            (96 * 96, hi, "large\n(>96²)", "#c8d8f8"),
        ]
        for x0, x1, lbl, color in ref_bands:
            x0c, x1c = max(x0, lo), min(x1, hi)
            if x0c >= x1c:
                continue
            ax3.axvspan(x0c, x1c, alpha=0.18, color=color, zorder=0)
            mid = 10 ** ((np.log10(x0c) + np.log10(x1c)) / 2)
            ax3.text(mid, 0.97, lbl, transform=ax3.get_xaxis_transform(),
                     fontsize=6.5, color="#555", ha="center", va="top")

        for sp in split_list:
            areas = stats[sp]["bbox_areas_px"]
            if not areas:
                continue
            color = split_colors.get(sp, "#888")
            ax3.hist(areas, bins=bins, alpha=0.55, color=color, label=sp, zorder=2)

        ax3.set_xscale("log")
        ax3.set_title("BBox Area (px², log scale)", fontsize=11, fontweight="bold")
        ax3.set_xlabel("area (px²)")
        ax3.set_ylabel("# boxes")
        ax3.legend(fontsize=8, loc="upper right", framealpha=0.9)
    else:
        ax3.text(0.5, 0.5, "(no bbox area data)", ha="center", va="center",
                 transform=ax3.transAxes, fontsize=10, color="#888")
        ax3.axis("off")

    # --- [1,1] Labels per image — box plot per split ---
    lpi_data = [stats[sp]["labels_per_image"] for sp in split_list if stats[sp].get("labels_per_image")]
    lpi_labels = [sp for sp in split_list if stats[sp].get("labels_per_image")]
    if lpi_data:
        bp = ax4.boxplot(
            lpi_data, labels=lpi_labels, patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker=".", markersize=3, alpha=0.4),
        )
        for patch, sp in zip(bp["boxes"], lpi_labels, strict=True):
            patch.set_facecolor(split_colors.get(sp, "#888"))
            patch.set_alpha(0.6)
        # Overlay individual mean markers
        for i, (sp, data) in enumerate(zip(lpi_labels, lpi_data, strict=True), start=1):
            ax4.scatter(i, float(np.mean(data)), marker="D", color=split_colors.get(sp, "#888"),
                        s=30, zorder=5, label=f"{sp} mean={np.mean(data):.1f}")
    ax4.set_title("Labels per Image", fontsize=11, fontweight="bold")
    ax4.set_ylabel("# annotations per image")
    ax4.set_ylim(bottom=0)
    ax4.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    # Headroom for the legend above the boxplot whiskers.
    _y_max_4 = ax4.get_ylim()[1]
    ax4.set_ylim(top=_y_max_4 * 1.18)

    # --- [1,2] Summary text panel ---
    # Keep this terse: per-class counts already appear in Class Instances
    # and Class Balance %. Repeating them here overflows the cell on
    # datasets with many classes / splits.
    ax5.axis("off")
    small_t = 32 * 32
    med_t = 96 * 96
    lines = ["Key Statistics", ""]
    lines.append(f"{'split':<6}{'imgs':>8}{'anns':>8}{'lbls/img':>10}{'empty':>8}")
    lines.append("-" * 40)
    for sp in split_list:
        s = stats[sp]
        lpi = s["labels_per_image"]
        avg_lpi = float(np.mean(lpi)) if lpi else 0.0
        empty_pct = s["n_empty"] / s["n_images"] * 100 if s["n_images"] else 0.0
        lines.append(
            f"{sp:<6}{s['n_images']:>8,}{s['n_annotations']:>8,}"
            f"{avg_lpi:>10.2f}{s['n_empty']:>5} ({empty_pct:.0f}%)"
        )
    lines.append("")
    lines.append("BBox size mix (across all splits):")
    all_areas = [a for sp in split_list for a in stats[sp]["bbox_areas_px"]]
    if all_areas:
        n_a = len(all_areas)
        sm = sum(1 for a in all_areas if a < small_t) / n_a * 100
        md = sum(1 for a in all_areas if small_t <= a < med_t) / n_a * 100
        lg = 100 - sm - md
        avg_a = float(np.mean(all_areas))
        lines.append(f"  small <32²    {sm:>5.1f}%")
        lines.append(f"  med 32²–96²   {md:>5.1f}%")
        lines.append(f"  large >96²    {lg:>5.1f}%")
        lines.append(f"  avg area      {avg_a:>7,.0f} px²")

    ax5.text(
        0.02, 0.98, "\n".join(lines),
        transform=ax5.transAxes, fontsize=9.0, va="top", ha="left",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", alpha=0.85),
    )

    # --- [2,0] Resolution scatter ---
    _panel_resolution_scatter(ax6, stats, split_list, input_size=data_cfg.get("input_size"))
    # --- [2,1] Per-channel pixel mean/std ---
    _panel_pixel_stats(ax7, stats, split_list)
    # --- [2,2] Empty-label panel ---
    _panel_empty_labels(
        ax8,
        {sp: stats[sp]["n_empty"] for sp in split_list},
        {sp: stats[sp]["n_images"] for sp in split_list},
        title="Empty Images (0 boxes)",
    )

    # --- [3,0] BBox aspect-ratio histogram (log-x, w/h) ---
    all_aspects = [a for s in split_list for a in stats[s]["bbox_aspects"] if a > 0]
    if all_aspects:
        bins = np.logspace(np.log10(0.1), np.log10(10.0), 30)
        for sp in split_list:
            asp = [a for a in stats[sp]["bbox_aspects"] if a > 0]
            if not asp:
                continue
            ax9.hist(np.clip(asp, 0.1, 10.0), bins=bins, alpha=0.55,
                     color=split_colors.get(sp, "#888"), label=sp)
        ax9.set_xscale("log")
        ax9.axvline(1.0, color="#555", linestyle="--", linewidth=0.8, alpha=0.7)
        ax9.set_title("BBox Aspect Ratio (w/h, log)", fontsize=11, fontweight="bold")
        ax9.set_xlabel("aspect (w/h)")
        ax9.set_ylabel("# boxes")
        ax9.legend(fontsize=8, framealpha=0.9)
    else:
        ax9.text(0.5, 0.5, "(no aspect data)", ha="center", va="center",
                 transform=ax9.transAxes, fontsize=10, color="#888")
        ax9.axis("off")

    # --- [3,1] BBox center-point 2D heatmap (cx_norm, cy_norm) ---
    all_centers = [c for s in split_list for c in stats[s]["bbox_centers"]]
    if all_centers:
        cx = np.array([c[0] for c in all_centers])
        cy = np.array([c[1] for c in all_centers])
        H_heat, _, _ = np.histogram2d(
            cx, cy, bins=50, range=[[0.0, 1.0], [0.0, 1.0]]
        )
        # Display with origin top-left (image convention): flip y.
        im = ax10.imshow(
            H_heat.T, origin="upper", extent=(0, 1, 1, 0),
            cmap="viridis", aspect="equal", interpolation="nearest",
        )
        ax10.set_title("BBox Centers (normalized)", fontsize=11, fontweight="bold")
        ax10.set_xlabel("cx (normalized)")
        ax10.set_ylabel("cy (normalized)")
        safe_colorbar(ax10, im, label="Box count")
    else:
        ax10.text(0.5, 0.5, "(no center data)", ha="center", va="center",
                  transform=ax10.transAxes, fontsize=10, color="#888")
        ax10.axis("off")

    # Per-chart captions (replace centralized guide).
    _chart_caption(ax0, "Total image count per split.")
    _chart_caption(ax1, "Annotation count per class. Big gaps across splits cause train/val mAP divergence.")
    _chart_caption(ax2, "Per-split class share. Compare bars across splits to spot stratification issues.")
    _chart_caption(ax3, "Object size distribution. Bands = COCO small/medium/large. Heavy-small tail → raise input_size or tune anchors.")
    _chart_caption(ax4, "Crowdedness. Long tails suggest mosaic / copy-paste augmentation.")
    _chart_caption(ax6, "W×H of source images. Crosshair = model input_size; far-out points get heavily resized.")
    _chart_caption(ax7, "Sampled RGB stats; should match mean/std in 05_data.yaml. Train↔val gap = lighting/sensor shift.")
    _chart_caption(ax8, "0-box images per split. Nonzero = unlabeled images; fix labels or enable empty-image training.")
    _chart_caption(ax9, "w/h distribution. Skew left = tall, right = wide objects. Use for anchor design.")
    _chart_caption(ax10, "Where objects sit in frame. Strong center bias → random crops help generalization.")

    # --- JSON export ---
    json_out = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "task": "detection",
        "total_images": total_images,
        "total_annotations": total_ann,
        "empty_labels": {sp: stats[sp]["n_empty"] for sp in split_list},
        "pixel_stats_rgb": {
            sp: {
                "mean": [round(v, 5) for v in stats[sp]["_img_meta"]["mean_rgb"]],
                "std": [round(v, 5) for v in stats[sp]["_img_meta"]["std_rgb"]],
                "range": "[0,1]",
                "sample_n": stats[sp]["_img_meta"]["pixel_sample_n"],
            } for sp in split_list
        },
        "resolution_scatter_sample": {
            sp: [list(map(int, p)) for p in stats[sp]["_img_meta"]["scatter"]]
            for sp in split_list
        },
        "splits": {},
    }
    for sp in split_list:
        s = stats[sp]
        lpi = s["labels_per_image"]
        areas = s["bbox_areas_px"]
        n_ann = s["n_annotations"]
        total_area = max(len(areas), 1)
        small_t = 32 * 32
        med_t = 96 * 96
        json_out["splits"][sp] = {
            "n_images": s["n_images"],
            "n_annotations": n_ann,
            "n_empty": s["n_empty"],
            "empty_pct": round(s["n_empty"] / s["n_images"] * 100, 2) if s["n_images"] else 0.0,
            "labels_per_image": {
                "min": int(np.min(lpi)) if lpi else 0,
                "max": int(np.max(lpi)) if lpi else 0,
                "mean": round(float(np.mean(lpi)), 3) if lpi else 0.0,
                "median": round(float(np.median(lpi)), 1) if lpi else 0.0,
                "p25": round(float(np.percentile(lpi, 25)), 1) if lpi else 0.0,
                "p75": round(float(np.percentile(lpi, 75)), 1) if lpi else 0.0,
            },
            "bbox_area_px2": {
                "mean": round(float(np.mean(areas)), 1) if areas else 0.0,
                "median": round(float(np.median(areas)), 1) if areas else 0.0,
                "p25": round(float(np.percentile(areas, 25)), 1) if areas else 0.0,
                "p75": round(float(np.percentile(areas, 75)), 1) if areas else 0.0,
            } if areas else {},
            "bbox_size_tiers_coco_px2": {
                "small_lt32sq": {
                    "count": sum(1 for a in areas if a < small_t),
                    "pct": round(sum(1 for a in areas if a < small_t) / total_area * 100, 1),
                },
                "medium_32_96sq": {
                    "count": sum(1 for a in areas if small_t <= a < med_t),
                    "pct": round(sum(1 for a in areas if small_t <= a < med_t) / total_area * 100, 1),
                },
                "large_gt96sq": {
                    "count": sum(1 for a in areas if a >= med_t),
                    "pct": round(sum(1 for a in areas if a >= med_t) / total_area * 100, 1),
                },
            },
            "class_counts": {
                class_names.get(cid, str(cid)): s["class_counts"].get(cid, 0)
                for cid in all_class_ids
            },
            "class_balance_pct": {
                class_names.get(cid, str(cid)): round(
                    s["class_counts"].get(cid, 0) / max(n_ann, 1) * 100, 2
                )
                for cid in all_class_ids
            },
        }
        # task-aware extras: aspect-ratio sample + center 2D heatmap
        asp_pos = [a for a in s["bbox_aspects"] if a > 0]
        asp_clipped = [round(float(a), 4) for a in np.clip(asp_pos[:500], 0.1, 10.0)] if asp_pos else []
        cx_arr = np.array([c[0] for c in s["bbox_centers"]]) if s["bbox_centers"] else np.zeros(0)
        cy_arr = np.array([c[1] for c in s["bbox_centers"]]) if s["bbox_centers"] else np.zeros(0)
        if cx_arr.size > 0:
            H_sp, _, _ = np.histogram2d(
                cx_arr, cy_arr, bins=50, range=[[0.0, 1.0], [0.0, 1.0]]
            )
            heatmap_list = H_sp.astype(int).tolist()
        else:
            heatmap_list = []
        json_out["splits"][sp]["task_specific"] = {
            "detection": {
                "aspect_ratios_sample": asp_clipped,
                "center_heatmap": heatmap_list,
            }
        }
    _save_stats(fig, json_out, out_dir, dpi)


# ---------------------------------------------------------------------------
# Classification / Segmentation / Keypoint sub-stats
# ---------------------------------------------------------------------------


def _palette_rgb(cid: int) -> tuple:
    return tuple(c / 255 for c in _LABEL_PALETTE[cid % len(_LABEL_PALETTE)][::-1])


def _bar_images_per_split(ax, split_list: list[str], n_imgs: list[int]) -> None:
    colors = [_SPLIT_COLORS.get(s, "#888") for s in split_list]
    bars = ax.barh(split_list, n_imgs, color=colors)
    ax.set_title("Images per Split", fontsize=11, fontweight="bold")
    ax.set_xlabel("# images")
    mx = max(n_imgs) if n_imgs else 1
    for bar, n in zip(bars, n_imgs, strict=True):
        ax.text(bar.get_width() + mx * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{n:,}", va="center", fontsize=9)
    ax.set_xlim(0, mx * 1.25)


def _stats_classification(
    data_cfg: dict, base_dir: str, class_names: dict,
    splits: list[str], out_dir: Path, dpi: int,
    subset_indices: dict[str, list[int]] | None,
    subset_active: bool = False,
    subset_pct: float | None = None,
) -> None:
    """Classification stats: split sizes + class histogram (+ imbalance ratio)
    + image-resolution (aspect) distribution + per-channel mean/std table.

    Image pixel stats are computed on a capped random subset of each split
    (≤ 256 imgs) to keep runtime bounded on large datasets.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    from core.p05_data.classification_dataset import ClassificationDataset

    stats: dict[str, dict] = {}
    PIXEL_SAMPLE_CAP = 2048

    for split in splits:
        try:
            ds = ClassificationDataset(
                data_config=data_cfg, split=split, transforms=None, base_dir=base_dir,
            )
        except Exception as e:
            logger.warning("Stats: skipping split %s — %s", split, e)
            continue
        idx_list = (subset_indices or {}).get(split)
        samples = ds.samples if idx_list is None else [ds.samples[i] for i in idx_list]
        class_counts: dict[int, int] = {}
        widths: list[int] = []
        heights: list[int] = []
        aspects: list[float] = []
        # Pixel-stats sampled subset; resolution collected for ALL images
        # via cheap PIL header read.
        try:
            from PIL import Image as _PILImage
        except Exception:
            _PILImage = None
        rng = random.Random(0)
        n = len(samples)
        pick_n = min(PIXEL_SAMPLE_CAP, n)
        pick_idx = set(rng.sample(range(n), pick_n)) if n else set()
        ch_means = np.zeros(3, dtype=np.float64)
        ch_stds = np.zeros(3, dtype=np.float64)
        n_pix = 0
        for i, (path, cid) in enumerate(samples):
            class_counts[int(cid)] = class_counts.get(int(cid), 0) + 1
            # Resolution — full-coverage PIL header read.
            W = H = 0
            if _PILImage is not None:
                try:
                    with _PILImage.open(str(path)) as im:
                        W, H = im.size
                except Exception:
                    pass
            if W and H:
                widths.append(int(W))
                heights.append(int(H))
                aspects.append(W / max(H, 1))
            # Pixel stats — sampled subset only.
            if i in pick_idx:
                img = cv2.imread(str(path))
                if img is None:
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                flat = rgb.reshape(-1, 3)
                ch_means += flat.sum(axis=0)
                ch_stds += (flat ** 2).sum(axis=0)
                n_pix += flat.shape[0]
        if n_pix > 0:
            mean = ch_means / n_pix
            var = ch_stds / n_pix - mean ** 2
            std = np.sqrt(np.clip(var, 0, None))
        else:
            mean = np.zeros(3); std = np.zeros(3)

        # Empty-label count for cls = images whose class id is not in class_names
        valid_cids = set(int(k) for k in class_names.keys())
        n_empty = sum(1 for _, cid in samples if int(cid) not in valid_cids) if valid_cids else 0

        stats[split] = {
            "n_images": n,
            "class_counts": class_counts,
            "widths": widths,
            "heights": heights,
            "aspects": aspects,
            "pix_mean_rgb": mean.tolist(),
            "pix_std_rgb": std.tolist(),
            "pixel_sample_n": pick_n,
            "n_empty": n_empty,
            "_img_meta": {
                "widths": widths,
                "heights": heights,
                "scatter": list(zip(widths, heights, strict=False))[:256],
                "mean_rgb": mean.tolist(),
                "std_rgb": std.tolist(),
                "pixel_sample_n": pick_n,
            },
        }

    if not stats:
        logger.warning("Stats (classification): no splits loaded, skipping")
        return

    split_list = [s for s in splits if s in stats]
    all_class_ids = sorted({cid for s in stats.values() for cid in s["class_counts"]})
    total_images = sum(stats[s]["n_images"] for s in split_list)
    split_summary = "  |  ".join(f"{sp.capitalize()}: {stats[sp]['n_images']:,}" for sp in split_list)

    dataset_name = data_cfg.get("dataset_name") or data_cfg.get("name") or "dataset"
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        _format_suptitle(dataset_name, subset_active, subset_pct)
        + f"\n{split_summary}  |  Total: {total_images:,} images  |  Classification",
        fontsize=12, fontweight="bold", y=0.995,
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.80, wspace=0.40)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[1, 2])
    ax6 = fig.add_subplot(gs[2, 0])  # pixel mean/std bar
    ax7 = fig.add_subplot(gs[2, 1])  # empty-label panel
    ax8 = fig.add_subplot(gs[2, 2])  # imbalance ratio bar

    # [0,0] Images per split
    _bar_images_per_split(ax0, split_list, [stats[s]["n_images"] for s in split_list])

    # [0,1] Class histogram grouped by split
    x = np.arange(len(all_class_ids))
    width = 0.75 / max(len(split_list), 1)
    for i, sp in enumerate(split_list):
        counts = [stats[sp]["class_counts"].get(cid, 0) for cid in all_class_ids]
        offset = (i - len(split_list) / 2 + 0.5) * width
        ax1.bar(x + offset, counts, width, label=sp,
                color=_SPLIT_COLORS.get(sp, "#888"), alpha=0.85)
    ax1.set_title("Images per Class", fontsize=11, fontweight="bold")
    ax1.set_xlabel("class")
    ax1.set_ylabel("# images")
    ax1.set_xticks(x)
    labels_x = [truncate_label(class_names.get(cid, str(cid)), 14) for cid in all_class_ids]
    ax1.set_xticklabels(labels_x, rotation=45, ha="right", fontsize=8)
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.9)
    y_max = ax1.get_ylim()[1] or 1
    ax1.set_ylim(top=y_max * 1.18)

    # Imbalance ratios (text annotation on class histogram)
    imbalance_by_split = {}
    for sp in split_list:
        cc = list(stats[sp]["class_counts"].values())
        if cc:
            imbalance_by_split[sp] = (max(cc) / max(min(cc), 1))
    if imbalance_by_split:
        ann = " | ".join(f"{sp}: {r:.1f}×" for sp, r in imbalance_by_split.items())
        ax1.text(0.02, 0.97, f"Imbalance (max/min): {ann}",
                 transform=ax1.transAxes, fontsize=8, va="top",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="whitesmoke", alpha=0.9))

    # [0,2] Class balance %
    for i, sp in enumerate(split_list):
        total = sum(stats[sp]["class_counts"].values()) or 1
        left = 0.0
        for cid in all_class_ids:
            pct = stats[sp]["class_counts"].get(cid, 0) / total * 100
            ax2.barh(sp, pct, left=left, color=_palette_rgb(cid),
                     label=class_names.get(cid, str(cid)) if i == 0 else "")
            if pct > 4:
                ax2.text(left + pct / 2, i, f"{pct:.1f}%",
                         ha="center", va="center", fontsize=8, color="white", fontweight="bold")
            left += pct
    ax2.set_title("Class Balance %", fontsize=11, fontweight="bold")
    ax2.set_xlabel("% of images")
    ax2.set_xlim(0, 100)
    handles = [plt.Rectangle((0, 0), 1, 1, color=_palette_rgb(cid)) for cid in all_class_ids]
    ax2.legend(handles, [class_names.get(cid, str(cid)) for cid in all_class_ids],
               fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), framealpha=1.0)

    # [1,0] Aspect-ratio histogram
    all_aspects = [a for sp in split_list for a in stats[sp]["aspects"]]
    if all_aspects:
        bins = np.linspace(max(min(all_aspects) * 0.95, 0.1),
                           max(all_aspects) * 1.05, 30)
        for sp in split_list:
            a = stats[sp]["aspects"]
            if a:
                ax3.hist(a, bins=bins, alpha=0.45, label=sp,
                         color=_SPLIT_COLORS.get(sp, "#888"))
        ax3.axvline(1.0, color="#555", linestyle="--", linewidth=0.8, alpha=0.7)
        n_full = sum(len(stats[sp]["aspects"]) for sp in split_list)
        n_total = sum(stats[sp]["n_images"] for sp in split_list)
        suffix = f"(full, n={n_full:,})" if n_full >= n_total else f"(sampled, n={n_full:,}/{n_total:,})"
        ax3.set_title(f"Image Aspect Ratio (W/H) {suffix}", fontsize=11, fontweight="bold")
        ax3.set_xlabel("aspect ratio")
        ax3.set_ylabel("# images")
        ax3.legend(fontsize=8, framealpha=0.9)

    # [1,1] Resolution scatter (W vs H), sampled
    for sp in split_list:
        w = stats[sp]["widths"]; h = stats[sp]["heights"]
        if w and h:
            ax4.scatter(w, h, s=6, alpha=0.35, label=sp,
                        color=_SPLIT_COLORS.get(sp, "#888"))
    n_full = sum(len(stats[sp]["widths"]) for sp in split_list)
    n_total = sum(stats[sp]["n_images"] for sp in split_list)
    suffix = f"(full, n={n_full:,})" if n_full >= n_total else f"(sampled, n={n_full:,}/{n_total:,})"
    ax4.set_title(f"Image Resolution {suffix}", fontsize=11, fontweight="bold")
    ax4.set_xlabel("width (px)")
    ax4.set_ylabel("height (px)")
    if any(stats[s]["widths"] for s in split_list):
        ax4.legend(fontsize=8, framealpha=0.9)

    # [1,2] Per-channel mean/std table — terse; class counts already
    # appear in Class Instances + Class Balance % charts above.
    ax5.axis("off")
    lines = ["Pixel statistics (RGB, [0,1])", ""]
    for sp in split_list:
        m = stats[sp]["pix_mean_rgb"]; s = stats[sp]["pix_std_rgb"]
        n_s = stats[sp]["pixel_sample_n"]
        n_total = stats[sp]["n_images"]
        tag = "full" if n_s >= n_total else "sampled"
        lines.append(f"[{sp}]  ({tag}, n={n_s:,}/{n_total:,})")
        lines.append(f"  mean R={m[0]:.3f} G={m[1]:.3f} B={m[2]:.3f}")
        lines.append(f"  std  R={s[0]:.3f} G={s[1]:.3f} B={s[2]:.3f}")
        lines.append("")
    ax5.text(0.02, 0.98, "\n".join(lines), transform=ax5.transAxes,
             fontsize=9.0, va="top", ha="left", family="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", alpha=0.85))

    # [2,0] Per-channel pixel mean/std
    _panel_pixel_stats(ax6, stats, split_list)
    # [2,1] Empty-label panel (class id not in class_names)
    _panel_empty_labels(
        ax7,
        {sp: stats[sp]["n_empty"] for sp in split_list},
        {sp: stats[sp]["n_images"] for sp in split_list},
        title="Unknown / Out-of-Vocabulary Class IDs",
    )
    # [2,2] Imbalance ratio bar
    if imbalance_by_split:
        sps = list(imbalance_by_split.keys())
        ratios = [imbalance_by_split[sp] for sp in sps]
        colors = [_SPLIT_COLORS.get(sp, "#888") for sp in sps]
        bars = ax8.bar(sps, ratios, color=colors)
        for b, r in zip(bars, ratios, strict=True):
            ax8.text(b.get_x() + b.get_width() / 2,
                     b.get_height() + max(ratios) * 0.02,
                     f"{r:.2f}×", ha="center", va="bottom", fontsize=9)
        ax8.set_title("Class Imbalance Ratio (max / min)",
                      fontsize=11, fontweight="bold")
        ax8.set_ylabel("ratio")
        ax8.set_ylim(0, max(ratios) * 1.25)
    else:
        ax8.axis("off")

    # Per-chart captions.
    _chart_caption(ax0, "Total image count per split.")
    _chart_caption(ax1, "Per-class image count per split. Sustained gaps across splits = class-weighted resampling.")
    _chart_caption(ax2, "Per-split class share. Tells you if any split is missing a class.")
    _chart_caption(ax3, "W/H of every image. Spike at 1.0 = uniform square images.")
    _chart_caption(ax4, "W×H of every image. Single point = uniform resolution.")
    _chart_caption(ax6, "Sampled RGB stats; should match mean/std in 05_data.yaml.")
    _chart_caption(ax7, "Images whose class id ∉ class_names. Nonzero = label/config mismatch.")
    _chart_caption(ax8, "Largest class count ÷ smallest. >2× → consider class weights or resampling.")

    json_out = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "task": "classification",
        "total_images": total_images,
        "empty_labels": {sp: stats[sp]["n_empty"] for sp in split_list},
        "pixel_stats_rgb": {
            sp: {
                "mean": [round(v, 5) for v in stats[sp]["_img_meta"]["mean_rgb"]],
                "std": [round(v, 5) for v in stats[sp]["_img_meta"]["std_rgb"]],
                "range": "[0,1]",
                "sample_n": stats[sp]["_img_meta"]["pixel_sample_n"],
            } for sp in split_list
        },
        "resolution_scatter_sample": {
            sp: [list(map(int, p)) for p in stats[sp]["_img_meta"]["scatter"]]
            for sp in split_list
        },
        "splits": {
            sp: {
                "n_images": stats[sp]["n_images"],
                "class_counts": {
                    class_names.get(cid, str(cid)): stats[sp]["class_counts"].get(cid, 0)
                    for cid in all_class_ids
                },
                "imbalance_ratio_max_over_min": round(imbalance_by_split.get(sp, 0.0), 3),
                "pixel_mean_rgb": [round(v, 5) for v in stats[sp]["pix_mean_rgb"]],
                "pixel_std_rgb": [round(v, 5) for v in stats[sp]["pix_std_rgb"]],
                "pixel_sample_n": stats[sp]["pixel_sample_n"],
                "resolution": {
                    "width": {
                        "min": int(min(stats[sp]["widths"])) if stats[sp]["widths"] else 0,
                        "max": int(max(stats[sp]["widths"])) if stats[sp]["widths"] else 0,
                        "mean": round(float(np.mean(stats[sp]["widths"])), 1) if stats[sp]["widths"] else 0.0,
                    },
                    "height": {
                        "min": int(min(stats[sp]["heights"])) if stats[sp]["heights"] else 0,
                        "max": int(max(stats[sp]["heights"])) if stats[sp]["heights"] else 0,
                        "mean": round(float(np.mean(stats[sp]["heights"])), 1) if stats[sp]["heights"] else 0.0,
                    },
                    "aspect": {
                        "mean": round(float(np.mean(stats[sp]["aspects"])), 3) if stats[sp]["aspects"] else 0.0,
                        "median": round(float(np.median(stats[sp]["aspects"])), 3) if stats[sp]["aspects"] else 0.0,
                    },
                },
            }
            for sp in split_list
        },
    }
    _save_stats(fig, json_out, out_dir, dpi)


def _stats_segmentation(
    data_cfg: dict, base_dir: str, class_names: dict,
    splits: list[str], out_dir: Path, dpi: int,
    subset_indices: dict[str, list[int]] | None,
    subset_active: bool = False,
    subset_pct: float | None = None,
) -> None:
    """Segmentation stats: pixel-class histogram + mask coverage + component counts.

    Operates on capped sample of masks per split to keep runtime bounded.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    from core.p05_data.segmentation_dataset import SegmentationDataset

    MASK_SAMPLE_CAP = 400
    num_classes = int(data_cfg.get("num_classes", 0))
    ignore_index = int(data_cfg.get("ignore_index", 255))

    stats: dict[str, dict] = {}
    for split in splits:
        try:
            ds = SegmentationDataset(
                data_config=data_cfg, split=split, transforms=None, base_dir=base_dir,
            )
        except Exception as e:
            logger.warning("Stats: skipping split %s — %s", split, e)
            continue
        idx_list = (subset_indices or {}).get(split)
        img_paths = ds.img_paths if idx_list is None else [ds.img_paths[i] for i in idx_list]
        n = len(img_paths)
        rng = random.Random(0)
        pick_n = min(MASK_SAMPLE_CAP, n)
        pick_idx = rng.sample(range(n), pick_n) if n else []

        pixel_counts = np.zeros(max(num_classes, 1), dtype=np.int64)
        coverage: list[float] = []  # non-bg % per image (excl. ignore)
        comp_counts_per_class: dict[int, list[int]] = {}
        comp_areas_per_class: dict[int, list[int]] = {}
        n_degenerate = 0  # foreground coverage <0.1%

        for i in pick_idx:
            img_path = img_paths[i]
            mask_path = ds.mask_dir / f"{img_path.stem}.png"
            if not mask_path.exists():
                continue
            m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            # Mask out ignore_index in counts
            m_keep = m != ignore_index
            bc = np.bincount(m[m_keep].ravel(), minlength=pixel_counts.shape[0]) if m_keep.any() \
                else np.zeros(pixel_counts.shape[0], dtype=np.int64)
            if bc.shape[0] > pixel_counts.shape[0]:
                extra = np.zeros(bc.shape[0] - pixel_counts.shape[0], dtype=np.int64)
                pixel_counts = np.concatenate([pixel_counts, extra])
            pixel_counts[: bc.shape[0]] += bc
            total_keep = int(m_keep.sum())
            non_bg = int(((m != 0) & m_keep).sum())
            cov_pct = (non_bg / max(total_keep, 1)) * 100.0
            coverage.append(cov_pct)
            if cov_pct < 0.1:
                n_degenerate += 1

            # Connected components per class (cheap loop over unique non-bg ids)
            uniq = np.unique(m)
            for cid in uniq:
                cid = int(cid)
                if cid == 0:
                    continue
                binm = (m == cid).astype(np.uint8)
                nlab, labels, stats_cc, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
                # stats_cc[0] is background of the binary map
                n_comp = max(nlab - 1, 0)
                comp_counts_per_class.setdefault(cid, []).append(n_comp)
                if n_comp > 0:
                    areas = stats_cc[1:, cv2.CC_STAT_AREA].tolist()
                    comp_areas_per_class.setdefault(cid, []).extend(areas)

        stats[split] = {
            "n_images": n,
            "sample_n": pick_n,
            "pixel_counts": pixel_counts.tolist(),
            "coverage_pct": coverage,
            "comp_counts_per_class": comp_counts_per_class,
            "comp_areas_per_class": comp_areas_per_class,
            "n_empty": n_degenerate,
            "_img_meta": _sample_image_meta(img_paths),
        }

    if not stats:
        logger.warning("Stats (segmentation): no splits loaded, skipping")
        return

    split_list = [s for s in splits if s in stats]
    # Build authoritative class-id list from both declared names and observed pixels
    observed_cids = set()
    for s in stats.values():
        for i, c in enumerate(s["pixel_counts"]):
            if c > 0:
                observed_cids.add(i)
    all_class_ids = sorted(set(class_names.keys()) | observed_cids)
    total_images = sum(stats[s]["n_images"] for s in split_list)
    split_summary = "  |  ".join(f"{sp.capitalize()}: {stats[sp]['n_images']:,}" for sp in split_list)

    dataset_name = data_cfg.get("dataset_name") or data_cfg.get("name") or "dataset"
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        _format_suptitle(dataset_name, subset_active, subset_pct)
        + f"\n{split_summary}  |  Total: {total_images:,} images  |  Semantic Segmentation",
        fontsize=12, fontweight="bold", y=0.995,
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.85, wspace=0.40)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1:])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 2])
    ax5 = fig.add_subplot(gs[2, 0])  # resolution scatter
    ax6 = fig.add_subplot(gs[2, 1])  # pixel mean/std
    ax7 = fig.add_subplot(gs[2, 2])  # empty/degenerate-label panel

    # [0,0] Images per split
    _bar_images_per_split(ax0, split_list, [stats[s]["n_images"] for s in split_list])

    # [0,1:] Pixel-class histogram per split (log y, includes bg class 0)
    x = np.arange(len(all_class_ids))
    width = 0.75 / max(len(split_list), 1)
    for i, sp in enumerate(split_list):
        pc = stats[sp]["pixel_counts"]
        vals = [pc[cid] if cid < len(pc) else 0 for cid in all_class_ids]
        offset = (i - len(split_list) / 2 + 0.5) * width
        ax1.bar(x + offset, vals, width, label=sp,
                color=_SPLIT_COLORS.get(sp, "#888"), alpha=0.85)
    ax1.set_yscale("log")
    ax1.set_title("Pixel Count per Class (log y)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("class")
    ax1.set_ylabel("# pixels (across all masks)")
    ax1.set_xticks(x)
    labels_x = [class_names.get(cid, str(cid)) for cid in all_class_ids]
    ax1.set_xticklabels(labels_x, rotation=60, ha="right", fontsize=7)
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.9)

    # [1,0] Mask coverage (non-bg %) distribution per split
    for sp in split_list:
        cov = stats[sp]["coverage_pct"]
        if cov:
            ax2.hist(cov, bins=30, alpha=0.45, label=sp,
                     color=_SPLIT_COLORS.get(sp, "#888"))
    ax2.set_title("Non-Background Coverage %", fontsize=11, fontweight="bold")
    ax2.set_xlabel("% pixels labeled (non-bg)")
    ax2.set_ylabel("# images")
    ax2.legend(fontsize=8, framealpha=0.9)

    # [1,1] Median components/image per class (bar)
    # Aggregate across splits (weighted by split size)
    agg_counts: dict[int, list[int]] = {}
    for sp in split_list:
        for cid, lst in stats[sp]["comp_counts_per_class"].items():
            agg_counts.setdefault(cid, []).extend(lst)
    if agg_counts:
        cids_sorted = sorted(agg_counts)[:30]  # cap for readability
        medians = [float(np.median(agg_counts[c])) for c in cids_sorted]
        colors = [_palette_rgb(c) for c in cids_sorted]
        ax3.bar(range(len(cids_sorted)), medians, color=colors)
        ax3.set_xticks(range(len(cids_sorted)))
        ax3.set_xticklabels([class_names.get(c, str(c)) for c in cids_sorted],
                            rotation=60, ha="right", fontsize=7)
        ax3.set_title("Median # Components / Image", fontsize=11, fontweight="bold")
        ax3.set_ylabel("median components")

    # [1,2] Median component area per class (log y)
    agg_areas: dict[int, list[int]] = {}
    for sp in split_list:
        for cid, lst in stats[sp]["comp_areas_per_class"].items():
            agg_areas.setdefault(cid, []).extend(lst)
    if agg_areas:
        cids_sorted = sorted(agg_areas)[:30]
        medians = [float(np.median(agg_areas[c])) for c in cids_sorted]
        colors = [_palette_rgb(c) for c in cids_sorted]
        ax4.bar(range(len(cids_sorted)), medians, color=colors)
        ax4.set_yscale("log")
        ax4.set_xticks(range(len(cids_sorted)))
        ax4.set_xticklabels([class_names.get(c, str(c)) for c in cids_sorted],
                            rotation=60, ha="right", fontsize=7)
        ax4.set_title("Median Component Area (px, log)", fontsize=11, fontweight="bold")
        ax4.set_ylabel("pixels")

    # [2,0] Resolution scatter
    _panel_resolution_scatter(ax5, stats, split_list, input_size=data_cfg.get("input_size"))
    # [2,1] Per-channel pixel mean/std
    _panel_pixel_stats(ax6, stats, split_list)
    # [2,2] Empty/degenerate-label panel (coverage <0.1%)
    _panel_empty_labels(
        ax7,
        {sp: stats[sp]["n_empty"] for sp in split_list},
        {sp: stats[sp]["sample_n"] for sp in split_list},
        title="Degenerate Masks (FG <0.1%)",
    )

    # Per-chart captions.
    _chart_caption(ax0, "Total image count per split.")
    _chart_caption(ax1, "Total labeled pixels per class per split. Orders-of-magnitude gaps drive class-weighted loss decisions.")
    _chart_caption(ax2, "% of pixels labeled per image (excluding ignore_index). Low % = sparse annotations.")
    _chart_caption(ax3, "Connected-component count per class. High = fragmented; low = contiguous regions.")
    _chart_caption(ax4, "Median component size per class. Tiny components may be lost at output stride.")
    _chart_caption(ax5, "W×H of every image. Crosshair = model input_size.")
    _chart_caption(ax6, "Sampled RGB stats; should match mean/std in 05_data.yaml.")
    _chart_caption(ax7, "Masks with <0.1% foreground (excluding ignore_index). Usually annotation errors.")

    # JSON export
    json_out = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "task": "segmentation",
        "total_images": total_images,
        "empty_labels": {sp: stats[sp]["n_empty"] for sp in split_list},
        "pixel_stats_rgb": {
            sp: {
                "mean": [round(v, 5) for v in stats[sp]["_img_meta"]["mean_rgb"]],
                "std": [round(v, 5) for v in stats[sp]["_img_meta"]["std_rgb"]],
                "range": "[0,1]",
                "sample_n": stats[sp]["_img_meta"]["pixel_sample_n"],
            } for sp in split_list
        },
        "resolution_scatter_sample": {
            sp: [list(map(int, p)) for p in stats[sp]["_img_meta"]["scatter"]]
            for sp in split_list
        },
        "ignore_index": ignore_index,
        "splits": {
            sp: {
                "n_images": stats[sp]["n_images"],
                "sample_n": stats[sp]["sample_n"],
                "pixel_counts_by_class": {
                    class_names.get(cid, str(cid)):
                        int(stats[sp]["pixel_counts"][cid])
                        if cid < len(stats[sp]["pixel_counts"]) else 0
                    for cid in all_class_ids
                },
                "coverage_pct": {
                    "mean": round(float(np.mean(stats[sp]["coverage_pct"])), 3)
                        if stats[sp]["coverage_pct"] else 0.0,
                    "median": round(float(np.median(stats[sp]["coverage_pct"])), 3)
                        if stats[sp]["coverage_pct"] else 0.0,
                    "p25": round(float(np.percentile(stats[sp]["coverage_pct"], 25)), 3)
                        if stats[sp]["coverage_pct"] else 0.0,
                    "p75": round(float(np.percentile(stats[sp]["coverage_pct"], 75)), 3)
                        if stats[sp]["coverage_pct"] else 0.0,
                },
                "components_per_class": {
                    class_names.get(cid, str(cid)): {
                        "median_count": round(
                            float(np.median(stats[sp]["comp_counts_per_class"][cid])), 2
                        ) if cid in stats[sp]["comp_counts_per_class"] else 0.0,
                        "median_area_px": round(
                            float(np.median(stats[sp]["comp_areas_per_class"][cid])), 2
                        ) if cid in stats[sp]["comp_areas_per_class"] else 0.0,
                    }
                    for cid in all_class_ids
                },
            }
            for sp in split_list
        },
    }
    _save_stats(fig, json_out, out_dir, dpi)


def _stats_keypoint(
    data_cfg: dict, base_dir: str, class_names: dict,
    splits: list[str], out_dir: Path, dpi: int,
    subset_indices: dict[str, list[int]] | None,
    subset_active: bool = False,
    subset_pct: float | None = None,
) -> None:
    """Keypoint stats: images/split, instances/split, per-joint visibility rate,
    spatial heatmap (normalized to bbox), inter-joint distance histogram.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    from core.p05_data.keypoint_dataset import KeypointDataset

    # `num_keypoints` may be set explicitly or derived from `kpt_shape: [K, 3]`
    # (the convention used by every keypoint feature in this repo).
    num_keypoints = int(data_cfg.get("num_keypoints", 0))
    if num_keypoints <= 0:
        kpt_shape = data_cfg.get("kpt_shape") or []
        if isinstance(kpt_shape, (list, tuple)) and len(kpt_shape) >= 1:
            try:
                num_keypoints = int(kpt_shape[0])
            except (TypeError, ValueError):
                num_keypoints = 0
    if num_keypoints <= 0:
        logger.warning("Stats (keypoint): num_keypoints/kpt_shape missing in data_cfg; "
                       "no panels will render.")
        return
    skeleton = data_cfg.get("skeleton") or []

    # PIL header read for instance-size scatter — imported once.
    try:
        from PIL import Image as _PILImage
    except Exception:
        _PILImage = None

    HEAT = 64
    skeleton_pairs = np.asarray(
        [(a, b) for a, b in skeleton if 0 <= a < num_keypoints and 0 <= b < num_keypoints],
        dtype=np.int64,
    ) if skeleton else np.zeros((0, 2), dtype=np.int64)

    stats: dict[str, dict] = {}
    for split in splits:
        try:
            ds = KeypointDataset(
                data_config=data_cfg, split=split, transforms=None, base_dir=base_dir,
            )
        except Exception as e:
            logger.warning("Stats: skipping split %s — %s", split, e)
            continue
        idx_list = (subset_indices or {}).get(split)
        img_paths = ds.img_paths if idx_list is None else [ds.img_paths[i] for i in idx_list]

        n_instances = 0
        n_empty = 0
        vis_counts = np.zeros(num_keypoints, dtype=np.int64)
        labeled_counts = np.zeros(num_keypoints, dtype=np.int64)
        v_breakdown = np.zeros((num_keypoints, 3), dtype=np.int64)
        heat = np.zeros((HEAT, HEAT), dtype=np.float64)
        edge_lengths: dict[tuple[int, int], list[float]] = {}
        inst_sizes_vs_vis: list[tuple[float, int]] = []
        # Resolution cache so we don't header-read images twice
        # (here + inside _sample_image_meta below).
        wh_cache: dict[str, tuple[int, int]] = {}

        for img_path in img_paths:
            lbl = ds._load_label(img_path)
            boxes = lbl["boxes"]           # (N, 5) normalized: [cls, cx, cy, w, h]
            kpts = lbl["keypoints"]        # (N, K, 3) normalized image coords
            if len(boxes) == 0:
                n_empty += 1
                continue
            vis_per_inst = (kpts[:, :, 2] > 0).sum(axis=1)  # (N,)
            if int(vis_per_inst.sum()) == 0:
                n_empty += 1
            n_instances += len(boxes)
            labeled_counts += len(boxes)
            vis_counts += (kpts[:, :, 2] > 0).sum(axis=0).astype(np.int64)
            v_levels = kpts[:, :, 2].astype(np.int64)
            for v_lvl in (0, 1, 2):
                v_breakdown[:, v_lvl] += (v_levels == v_lvl).sum(axis=0)

            # Image dims for instance-size scatter (cached for `_sample_image_meta`).
            W = H = 0
            if _PILImage is not None:
                try:
                    with _PILImage.open(str(img_path)) as im:
                        W, H = im.size
                except Exception:
                    W = H = 0
            if W > 0 and H > 0:
                wh_cache[str(img_path)] = (W, H)
                bw_px = boxes[:, 3] * W
                bh_px = boxes[:, 4] * H
                valid = (bw_px > 0) & (bh_px > 0)
                if valid.any():
                    sizes = np.sqrt(bw_px[valid] * bh_px[valid])
                    counts = vis_per_inst[valid]
                    inst_sizes_vs_vis.extend(
                        zip(sizes.astype(float).tolist(),
                            counts.astype(int).tolist(), strict=True)
                    )

            # --- Vectorized bbox-normalized heatmap accumulation ---
            cx = boxes[:, 1]; cy = boxes[:, 2]
            bw = boxes[:, 3]; bh = boxes[:, 4]
            valid_box = (bw > 0) & (bh > 0)
            if valid_box.any():
                x0 = (cx - bw / 2)[valid_box, None]   # (M,1)
                y0 = (cy - bh / 2)[valid_box, None]
                bwv = bw[valid_box, None]
                bhv = bh[valid_box, None]
                kp = kpts[valid_box]                   # (M, K, 3)
                kv = kp[:, :, 2] > 0
                # Normalize joint positions to bbox; clip to [0, 0.999].
                kx = np.clip((kp[:, :, 0] - x0) / bwv, 0, 0.999)
                ky = np.clip((kp[:, :, 1] - y0) / bhv, 0, 0.999)
                hx = (kx * HEAT).astype(np.int64)
                hy = (ky * HEAT).astype(np.int64)
                # Mask out unlabeled joints, then scatter-add.
                hx_v = hx[kv]; hy_v = hy[kv]
                if hx_v.size:
                    np.add.at(heat, (hy_v, hx_v), 1.0)

                # --- Vectorized skeleton edge lengths (image-normalized) ---
                if skeleton_pairs.size:
                    a_idx = skeleton_pairs[:, 0]
                    b_idx = skeleton_pairs[:, 1]
                    # (M, E)
                    both_vis = kv[:, a_idx] & kv[:, b_idx]
                    dx = kp[:, a_idx, 0] - kp[:, b_idx, 0]
                    dy = kp[:, a_idx, 1] - kp[:, b_idx, 1]
                    L = np.sqrt(dx * dx + dy * dy)
                    for ei, (a, b) in enumerate(skeleton_pairs.tolist()):
                        sel = both_vis[:, ei]
                        if sel.any():
                            edge_lengths.setdefault((int(a), int(b)), []).extend(
                                L[sel, ei].astype(float).tolist()
                            )

        # _sample_image_meta does its own PIL reads — pass cache to skip duplicates.
        meta = _sample_image_meta(img_paths, wh_cache=wh_cache or None)
        stats[split] = {
            "n_images": len(img_paths),
            "n_instances": n_instances,
            "vis_counts": vis_counts.tolist(),
            "labeled_counts": labeled_counts.tolist(),
            "v_breakdown": v_breakdown.tolist(),
            "heat": heat,
            "edge_lengths": edge_lengths,
            "inst_sizes_vs_vis": inst_sizes_vs_vis,
            "n_empty": n_empty,
            "_img_meta": meta,
        }

    if not stats:
        logger.warning("Stats (keypoint): no splits loaded, skipping")
        return

    split_list = [s for s in splits if s in stats]
    total_images = sum(stats[s]["n_images"] for s in split_list)
    total_inst = sum(stats[s]["n_instances"] for s in split_list)
    split_summary = "  |  ".join(
        f"{sp.capitalize()}: {stats[sp]['n_images']:,} imgs / {stats[sp]['n_instances']:,} inst"
        for sp in split_list
    )

    dataset_name = data_cfg.get("dataset_name") or data_cfg.get("name") or "dataset"
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        _format_suptitle(dataset_name, subset_active, subset_pct)
        + f"\n{split_summary}  |  Total: {total_images:,} images / {total_inst:,} instances",
        fontsize=12, fontweight="bold", y=0.995,
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.85, wspace=0.40)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[1, 2])
    ax6 = fig.add_subplot(gs[2, 0])  # resolution scatter
    ax7 = fig.add_subplot(gs[2, 1])  # pixel mean/std
    ax8 = fig.add_subplot(gs[2, 2])  # empty-label panel

    # [0,0] Images per split
    _bar_images_per_split(ax0, split_list, [stats[s]["n_images"] for s in split_list])

    # [0,1] Instances per split
    _bar_images_per_split(ax1, split_list, [stats[s]["n_instances"] for s in split_list])
    ax1.set_title("Instances per Split", fontsize=11, fontweight="bold")
    ax1.set_xlabel("# instances")

    # [0,2] Per-joint v=0/v=1/v=2 stacked bar (% of instances).
    total_vb = np.zeros((num_keypoints, 3), dtype=np.int64)
    total_lab = np.zeros(num_keypoints, dtype=np.int64)
    for sp in split_list:
        total_vb += np.asarray(stats[sp]["v_breakdown"], dtype=np.int64)
        total_lab += np.asarray(stats[sp]["labeled_counts"], dtype=np.int64)
    denom = np.maximum(total_lab, 1)
    pct_v0 = total_vb[:, 0] / denom * 100.0
    pct_v1 = total_vb[:, 1] / denom * 100.0
    pct_v2 = total_vb[:, 2] / denom * 100.0
    x = np.arange(num_keypoints)
    ax2.bar(x, pct_v0, color="#cccccc", label="v=0 (unlabeled)")
    ax2.bar(x, pct_v1, bottom=pct_v0, color="#f0ad4e", label="v=1 (occluded)")
    ax2.bar(x, pct_v2, bottom=pct_v0 + pct_v1, color="#4C72B0", label="v=2 (visible)")
    ax2.set_ylim(0, 105)
    ax2.set_title("Per-Joint Visibility (v=0/1/2 stacked)",
                  fontsize=11, fontweight="bold")
    ax2.set_xlabel("joint index")
    ax2.set_ylabel("% of instances")
    ax2.legend(fontsize=7.5, framealpha=0.9, loc="lower right")

    # [1,0] Spatial heatmap (bbox-normalized, aggregated across splits)
    heat = np.zeros_like(next(iter(stats.values()))["heat"])
    for sp in split_list:
        heat += stats[sp]["heat"]
    im = ax3.imshow(heat, origin="upper", cmap="magma", aspect="equal")
    ax3.set_title("Keypoint Spatial Heatmap\n(normalized to bbox)",
                  fontsize=11, fontweight="bold")
    ax3.set_xticks([]); ax3.set_yticks([])
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    # [1,1] Skeleton-edge length histogram (normalized image units)
    agg_edges: dict[tuple[int, int], list[float]] = {}
    for sp in split_list:
        for k, v in stats[sp]["edge_lengths"].items():
            agg_edges.setdefault(k, []).extend(v)
    if agg_edges:
        # Violin distribution of length per skeleton edge — all edges, not top-8.
        edge_keys = sorted(agg_edges.keys())
        data = [agg_edges[k] for k in edge_keys]
        labels_e = [f"{a}-{b}" for (a, b) in edge_keys]
        try:
            parts = ax4.violinplot(data, showmeans=False, showmedians=True, widths=0.85)
            for pc in parts.get("bodies", []):
                pc.set_facecolor("#4C72B0")
                pc.set_alpha(0.55)
        except Exception:
            # Fallback: boxplot on degenerate input
            ax4.boxplot(data, labels=labels_e)
        ax4.set_xticks(range(1, len(labels_e) + 1))
        ax4.set_xticklabels(labels_e, rotation=60, ha="right", fontsize=7)
        ax4.set_title("Skeleton Edge Length Distribution (all edges)",
                      fontsize=11, fontweight="bold")
        ax4.set_ylabel("normalized length")
    else:
        ax4.axis("off")
        ax4.text(0.5, 0.5, "(no skeleton: set data.skeleton)",
                 ha="center", va="center", transform=ax4.transAxes, fontsize=10, color="#888")

    # [1,2] Summary text panel — table form to fit one cell.
    ax5.axis("off")
    lines = ["Key Statistics (keypoint)", ""]
    lines.append(f"{'split':<6}{'imgs':>8}{'inst':>8}{'vis%':>9}")
    lines.append("-" * 32)
    for sp in split_list:
        s = stats[sp]
        if s["n_instances"]:
            vis = np.mean(np.asarray(s["vis_counts"]) /
                          max(s["n_instances"], 1)) * 100
            vis_str = f"{vis:>8.1f}%"
        else:
            vis_str = f"{'-':>9}"
        lines.append(f"{sp:<6}{s['n_images']:>8,}{s['n_instances']:>8,}{vis_str}")
    lines.append("")
    ax5.text(0.02, 0.98, "\n".join(lines), transform=ax5.transAxes,
             fontsize=9, va="top", ha="left", family="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", alpha=0.85))

    # [2,0] Resolution scatter
    _panel_resolution_scatter(ax6, stats, split_list, input_size=data_cfg.get("input_size"))
    # [2,1] Per-channel pixel mean/std
    _panel_pixel_stats(ax7, stats, split_list)
    # [2,2] Empty-label panel (0 visible joints / no instances)
    _panel_empty_labels(
        ax8,
        {sp: stats[sp]["n_empty"] for sp in split_list},
        {sp: stats[sp]["n_images"] for sp in split_list},
        title="Empty Images (0 visible joints)",
    )

    # Per-chart captions.
    _chart_caption(ax0, "Total image count per split.")
    _chart_caption(ax1, "Annotated person/instance count per split.")
    _chart_caption(ax2, "Fraction of instances where each joint is marked visible. Rare joints → per-joint loss weights or drop.")
    _chart_caption(ax3, "Joint locations normalized to bbox. Strong central peak = canonical pose; spread = pose diversity.")
    _chart_caption(ax4, "Distribution of skeleton edge lengths. Wide spread = pose variety; narrow = canonical pose only.")
    _chart_caption(ax6, "W×H of every image. Crosshair = model input_size.")
    _chart_caption(ax7, "Sampled RGB stats; should match mean/std in 05_data.yaml.")
    _chart_caption(ax8, "Instances with no visible joints — usually fully occluded; exclude from loss.")

    json_out = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "task": "keypoint",
        "num_keypoints": num_keypoints,
        "total_images": total_images,
        "total_instances": total_inst,
        "empty_labels": {sp: stats[sp]["n_empty"] for sp in split_list},
        "pixel_stats_rgb": {
            sp: {
                "mean": [round(v, 5) for v in stats[sp]["_img_meta"]["mean_rgb"]],
                "std": [round(v, 5) for v in stats[sp]["_img_meta"]["std_rgb"]],
                "range": "[0,1]",
                "sample_n": stats[sp]["_img_meta"]["pixel_sample_n"],
            } for sp in split_list
        },
        "resolution_scatter_sample": {
            sp: [list(map(int, p)) for p in stats[sp]["_img_meta"]["scatter"]]
            for sp in split_list
        },
        "task_specific": {
            "keypoint": {
                "v_breakdown_per_joint": {
                    sp: stats[sp]["v_breakdown"] for sp in split_list
                },
                "instance_size_vs_visible_joints": {
                    sp: [list(t) for t in stats[sp]["inst_sizes_vs_vis"][:512]]
                    for sp in split_list
                },
            },
        },
        "splits": {
            sp: {
                "n_images": stats[sp]["n_images"],
                "n_instances": stats[sp]["n_instances"],
                "vis_counts": stats[sp]["vis_counts"],
                "labeled_counts": stats[sp]["labeled_counts"],
                "visibility_rate_per_joint": [
                    round(v / max(l, 1), 4)
                    for v, l in zip(stats[sp]["vis_counts"],
                                    stats[sp]["labeled_counts"], strict=True)
                ],
                "skeleton_edge_lengths": {
                    f"{a}-{b}": {
                        "mean": round(float(np.mean(v)), 4),
                        "median": round(float(np.median(v)), 4),
                        "n": len(v),
                    }
                    for (a, b), v in stats[sp]["edge_lengths"].items()
                },
            }
            for sp in split_list
        },
    }
    _save_stats(fig, json_out, out_dir, dpi)


_STATS_PNG = "01_dataset_stats.png"
_STATS_JSON = "01_dataset_stats.json"
_LEGACY_STATS_PNG = "dataset_stats.png"
_LEGACY_STATS_JSON = "dataset_stats.json"


def _load_cached_stats(out_dir: Path) -> bool:
    """Return True if cached stats (numbered or legacy) already exist."""
    if (out_dir / _STATS_JSON).exists() and (out_dir / _STATS_PNG).exists():
        return True
    return (out_dir / _LEGACY_STATS_JSON).exists() and (out_dir / _LEGACY_STATS_PNG).exists()


# ---------------------------------------------------------------------------
# Dataset provenance (00_dataset_info.{md,json})
# ---------------------------------------------------------------------------


def _git_sha() -> str | None:
    """Return short git SHA of the repo, or None if unavailable."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parent.parent.parent),
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode().strip() or None
    except Exception:
        return None


def _describe_augmentation(training_cfg: dict) -> dict:
    """Extract a human-readable summary of the augmentation pipeline."""
    aug = (training_cfg or {}).get("augmentation", {}) or {}
    # Only surface toggles the reader actually needs to reason about.
    # Normalize moved to the tensor_prep section — exclude it here.
    keys = (
        "library", "mosaic", "mixup", "copypaste", "ir_simulation",
        "hflip", "vflip", "affine", "perspective", "color_jitter",
        "brightness", "contrast", "saturation", "hue",
    )
    return {k: aug[k] for k in keys if k in aug}


def _resolve_dataset_root(data_cfg: dict, data_config_path: str | Path | None) -> Path | None:
    """Resolve ``data_cfg['path']`` to an absolute Path, anchored at the data
    config's directory when the value is relative. Returns None if unset.
    """
    raw = data_cfg.get("path") or data_cfg.get("root")
    if not raw:
        return None
    p = Path(str(raw))
    if p.is_absolute():
        return p
    if data_config_path:
        anchor = Path(str(data_config_path)).resolve().parent
        return (anchor / p).resolve()
    return p.resolve()


def _split_size_block(
    split_sizes: dict[str, int],
    full_sizes: dict[str, int] | None,
) -> dict[str, dict]:
    """Per-split block: always emits ``n_images``; when subset is active and
    full size is known, also emits ``n_images_full`` + ``n_images_used``.
    """
    out: dict[str, dict] = {}
    full_sizes = full_sizes or {}
    for sp, n in (split_sizes or {}).items():
        block: dict = {"n_images": int(n)}
        full = full_sizes.get(sp)
        if full is not None and int(full) > 0 and int(full) != int(n):
            block["n_images_full"] = int(full)
            block["n_images_used"] = int(n)
        out[str(sp)] = block
    return out


def write_dataset_info(
    out_dir: Path,
    *,
    feature_name: str | None,
    data_config_path: str | Path | None,
    training_config_path: str | Path | None,
    data_cfg: dict,
    training_cfg: dict | None,
    class_names: dict[int, str],
    split_sizes: dict[str, int],
    full_sizes: dict[str, int] | None = None,
) -> None:
    """Emit ``00_dataset_info.{md,json}`` with dataset provenance.

    Outputs are portable: only the dataset basename, a relative path from
    ``out_dir`` to the dataset root, and config **basenames** are written
    — no absolute paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    training_cfg = training_cfg or {}
    data_section = training_cfg.get("data", {}) or {}
    from utils.config import resolve_tensor_prep
    backend = (training_cfg.get("training", {}) or {}).get("backend", "pytorch")
    tp = resolve_tensor_prep(training_cfg, backend=backend) or {}
    input_size = (
        tp.get("input_size") or data_cfg.get("input_size") or data_section.get("input_size")
    )
    mean = tp.get("mean") or data_cfg.get("mean") or data_section.get("mean")
    std = tp.get("std") or data_cfg.get("std") or data_section.get("std")
    gpu_augment = bool((training_cfg.get("training", {}) or {}).get("gpu_augment", False))

    # Relative-path provenance (no absolutes leak into outputs).
    abs_root = _resolve_dataset_root(data_cfg, data_config_path)
    if abs_root is not None:
        dataset_name = abs_root.name.rstrip("/") or str(abs_root)
        try:
            dataset_relpath = os.path.relpath(str(abs_root), start=str(out_dir))
        except ValueError:
            # Different drives on Windows etc. — fall back to basename.
            dataset_relpath = abs_root.name
    else:
        dataset_name = data_cfg.get("dataset_name") or data_cfg.get("name") or "unknown"
        dataset_relpath = None

    data_config_basename = (
        os.path.basename(str(data_config_path)) if data_config_path else None
    )
    training_config_basename = (
        os.path.basename(str(training_config_path)) if training_config_path else None
    )

    info = {
        "feature_name": feature_name,
        "dataset_name": dataset_name,
        "dataset_relpath": dataset_relpath,
        "data_config": data_config_basename,
        "training_config": training_config_basename,
        "task": _normalize_task(data_cfg.get("task")),
        "num_classes": len(class_names) if class_names else None,
        "class_names": {int(k): str(v) for k, v in (class_names or {}).items()},
        "splits": _split_size_block(split_sizes, full_sizes),
        "input_size": list(input_size) if input_size is not None else None,
        "mean": list(mean) if mean is not None else None,
        "std": list(std) if std is not None else None,
        "backend": backend,
        "tensor_prep": {
            "applied_by": tp.get("applied_by"),
            "rescale": tp.get("rescale"),
            "normalize": tp.get("normalize"),
            "input_size": list(tp["input_size"]) if tp.get("input_size") is not None else None,
            "mean": list(tp["mean"]) if tp.get("mean") is not None else None,
            "std": list(tp["std"]) if tp.get("std") is not None else None,
        } if tp else None,
        "augmentation": _describe_augmentation(training_cfg),
        "gpu_augment": gpu_augment,
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
    }

    json_path = out_dir / "00_dataset_info.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    def _fmt(v):
        if v is None:
            return "_n/a_"
        if isinstance(v, dict):
            return ", ".join(f"{k}={v[k]}" for k in v)
        if isinstance(v, list):
            return str(v)
        return str(v)

    splits_str = ", ".join(
        f"{k}={v['n_images']}"
        + (f" (full {v['n_images_full']})" if "n_images_full" in v else "")
        for k, v in info["splits"].items()
    )
    dataset_line = f"`{info['dataset_name']}`"
    if dataset_relpath:
        dataset_line += f" (rel: `{dataset_relpath}`)"

    md_lines = [
        f"# Dataset info — {info['feature_name'] or 'unknown'}",
        "",
        f"- **Feature folder**: `{info['feature_name'] or 'unknown'}`",
        f"- **Dataset**: {dataset_line}",
        f"- **Data config**: `{info['data_config'] or 'n/a'}`",
        f"- **Training config**: `{info['training_config'] or 'n/a'}`",
        f"- **Task**: `{info['task']}`",
        f"- **Num classes**: {info['num_classes']}",
        f"- **Class names**: {_fmt(info['class_names'])}",
        f"- **Splits**: {splits_str}",
        f"- **Input size**: {_fmt(info['input_size'])}",
        f"- **Normalization**: mean={_fmt(info['mean'])}, std={_fmt(info['std'])}",
        f"- **tensor_prep**: {_fmt(info['tensor_prep'])}",
        f"- **Training backend**: `{info['backend']}`",
        f"- **GPU augmentation**: `{info['gpu_augment']}`",
        f"- **Augmentation**: {_fmt(info['augmentation'])}",
        f"- **Created (UTC)**: {info['created_utc']}",
        f"- **Git SHA**: `{info['git_sha'] or 'n/a'}`",
        "",
    ]
    md_path = out_dir / "00_dataset_info.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("Saved: %s", md_path)
    logger.info("Saved: %s", json_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    parser = argparse.ArgumentParser(description="Visualize dataset samples and augmentation before training.")
    parser.add_argument("--config", required=True, help="Path to 06_training.yaml config")
    parser.add_argument("--splits", nargs="+", default=None,
                        help="Splits to visualize (overrides config). e.g. --splits train val test")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples per split (overrides config).")
    parser.add_argument("--save-dir", default=None,
                        help="Output directory (default: auto-generated run dir)")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    train_cfg = load_config(str(config_path))

    # Resolve data config
    data_cfg_path = config_path.parent / train_cfg["data"]["dataset_config"]
    data_cfg = load_config(str(data_cfg_path))
    base_dir = str(config_path.parent)
    class_names = {int(k): str(v) for k, v in data_cfg.get("names", {}).items()}

    # Resolve output dir
    if args.save_dir:
        out_dir = Path(args.save_dir) / "data_preview"
    else:
        feature_name = feature_name_from_config_path(str(config_path))
        run_dir = generate_run_dir(feature_name, "05_data_viz")
        out_dir = Path(run_dir) / "data_preview"

    # Data viz config
    data_viz_cfg = train_cfg.get("training", {}).get("data_viz", {})
    aug_viz_cfg = train_cfg.get("training", {}).get("aug_viz", {})

    data_splits = args.splits or data_viz_cfg.get("splits", ["train", "val", "test"])
    aug_splits = args.splits or aug_viz_cfg.get("splits", ["train"])
    num_samples = args.num_samples if args.num_samples is not None else data_viz_cfg.get("num_samples", 16)
    grid_cols = data_viz_cfg.get("grid_cols", 4)
    thickness = data_viz_cfg.get("thickness", 2)
    text_scale = data_viz_cfg.get("text_scale", 0.4)
    dpi = data_viz_cfg.get("dpi", 150)

    logger.info("Output: %s", out_dir)

    generate_dataset_stats(data_cfg, base_dir, class_names, data_splits, out_dir, dpi)

    for split in data_splits:
        viz_data_labels(data_cfg, base_dir, class_names, split, num_samples, grid_cols, thickness, text_scale, dpi, out_dir)

    aug_num_samples = args.num_samples if args.num_samples is not None else aug_viz_cfg.get("num_samples", 16)
    aug_grid_cols = aug_viz_cfg.get("grid_cols", grid_cols)
    aug_thickness = aug_viz_cfg.get("thickness", thickness)
    aug_text_scale = aug_viz_cfg.get("text_scale", text_scale)
    aug_dpi = aug_viz_cfg.get("dpi", dpi)

    for split in aug_splits:
        viz_aug_labels(data_cfg, train_cfg, base_dir, class_names, split, aug_num_samples, aug_grid_cols, aug_thickness, aug_text_scale, aug_dpi, out_dir)

    logger.info("Done. Files in: %s", out_dir)


if __name__ == "__main__":
    main()

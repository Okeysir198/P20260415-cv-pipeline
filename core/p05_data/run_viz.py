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
from utils.viz import VizStyle, annotate_detections, apply_plot_style, save_image_grid

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
    dpi: int = 120,
    subset_indices: dict[str, list[int]] | None = None,
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
        return _stats_classification(data_cfg, base_dir, class_names, splits, out_dir, dpi, subset_indices)
    if task == "segmentation":
        return _stats_segmentation(data_cfg, base_dir, class_names, splits, out_dir, dpi, subset_indices)
    if task == "keypoint":
        return _stats_keypoint(data_cfg, base_dir, class_names, splits, out_dir, dpi, subset_indices)
    return _stats_detection(data_cfg, base_dir, class_names, splits, out_dir, dpi, subset_indices)


def _stats_detection(
    data_cfg: dict,
    base_dir: str,
    class_names: dict,
    splits: list[str],
    out_dir: Path,
    dpi: int = 120,
    subset_indices: dict[str, list[int]] | None = None,
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
        bbox_areas: list[float] = []
        labels_per_image: list[int] = []
        n_empty = 0

        for img_path in img_paths:
            labels = ds._load_label(img_path)
            n = len(labels)
            labels_per_image.append(n)
            if n == 0:
                n_empty += 1
            for row in labels:
                cid = int(row[0])
                class_counts[cid] = class_counts.get(cid, 0) + 1
                bbox_areas.append(float(row[3]) * float(row[4]) * 100.0)  # % of image area

        stats[split] = {
            "n_images": len(img_paths),
            "class_counts": class_counts,
            "bbox_areas": bbox_areas,
            "labels_per_image": labels_per_image,
            "n_empty": n_empty,
            "n_annotations": sum(class_counts.values()),
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

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"{split_summary}  |  Total: {total_images:,} images  |  {total_ann:,} annotations",
        fontsize=12, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[1, 2])

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
    ax1.set_xticklabels([class_names.get(cid, str(cid)) for cid in all_class_ids], rotation=15)
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

    # --- [1,0] BBox area distribution (histogram + KDE, log x-axis) ---
    from scipy.stats import gaussian_kde
    all_areas = [a for s in split_list for a in stats[s]["bbox_areas"]]
    if all_areas:
        lo = max(min(all_areas), 1e-4)
        hi = max(all_areas)
        bins = np.logspace(np.log10(lo), np.log10(hi), 45)
        x_log = np.linspace(np.log10(lo), np.log10(hi), 300)
        x_lin = 10 ** x_log

        # Shaded tier regions (drawn first, behind everything)
        tier_regions = [
            (lo,   1.0,  "tiny\n(<1%)",    "#ffcccc"),
            (1.0,  5.0,  "small\n(1–5%)",  "#fff0b3"),
            (5.0,  15.0, "medium\n(5–15%)","#c8f0c8"),
            (15.0, hi,   "large\n(>15%)",  "#c8d8f8"),
        ]
        for x0, x1, lbl, color in tier_regions:
            x0c, x1c = max(x0, lo), min(x1, hi)
            if x0c >= x1c:
                continue
            ax3.axvspan(x0c, x1c, alpha=0.18, color=color, zorder=0)
            # Label centred at geometric midpoint of the region
            mid = 10 ** ((np.log10(x0c) + np.log10(x1c)) / 2)
            ax3.text(mid, 0.97, lbl, transform=ax3.get_xaxis_transform(),
                     fontsize=6.5, color="#555", ha="center", va="top")

        # Thin boundary lines between tiers
        for x_val in (1.0, 5.0, 15.0):
            ax3.axvline(x_val, color="#999", linestyle="--", linewidth=0.7, alpha=0.7, zorder=1)

        # Histogram + KDE per split
        for sp in split_list:
            areas = stats[sp]["bbox_areas"]
            if not areas:
                continue
            color = split_colors.get(sp, "#888")
            ax3.hist(areas, bins=bins, alpha=0.15, color=color, density=True, zorder=2)
            try:
                kde = gaussian_kde(np.log10(areas), bw_method="scott")
                kde_vals = kde(x_log) / (x_lin * np.log(10))
                ax3.plot(x_lin, kde_vals, color=color, linewidth=1.8, label=sp, zorder=3)
            except Exception:
                pass

        ax3.set_xscale("log")
        ax3.set_title("BBox Area (% of image)", fontsize=11, fontweight="bold")
        ax3.set_xlabel("area %  [log scale]")
        ax3.set_ylabel("density")
        ax3.legend(fontsize=8, loc="upper right", framealpha=0.9)

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
    ax5.axis("off")
    lines = ["Key Statistics\n"]
    for sp in split_list:
        s = stats[sp]
        lpi = s["labels_per_image"]
        areas = s["bbox_areas"]
        avg_lpi = float(np.mean(lpi)) if lpi else 0.0
        empty_pct = s["n_empty"] / s["n_images"] * 100 if s["n_images"] else 0.0
        avg_area = float(np.mean(areas)) if areas else 0.0
        tiny_pct = sum(1 for a in areas if a < 1.0) / max(len(areas), 1) * 100
        small_pct = sum(1 for a in areas if 1.0 <= a < 5.0) / max(len(areas), 1) * 100
        med_pct = sum(1 for a in areas if 5.0 <= a < 15.0) / max(len(areas), 1) * 100
        large_pct = sum(1 for a in areas if a >= 15.0) / max(len(areas), 1) * 100
        lines.append(f"[{sp}]")
        lines.append(f"  images         {s['n_images']:>7,}")
        lines.append(f"  annotations    {s['n_annotations']:>7,}")
        lines.append(f"  avg labels/img {avg_lpi:>7.2f}")
        lines.append(f"  empty images   {s['n_empty']:>5} ({empty_pct:.1f}%)")
        lines.append("  bbox size tiers:")
        lines.append(f"    tiny  (<1%)  {tiny_pct:>6.1f}%")
        lines.append(f"    small (1-5%) {small_pct:>6.1f}%")
        lines.append(f"    med  (5-15%) {med_pct:>6.1f}%")
        lines.append(f"    large (>15%) {large_pct:>6.1f}%")
        lines.append(f"  avg bbox area  {avg_area:.3f}%")
        lines.append("  class counts:")
        for cid in all_class_ids:
            cnt = s["class_counts"].get(cid, 0)
            pct = cnt / max(s["n_annotations"], 1) * 100
            name = class_names.get(cid, str(cid))
            lines.append(f"    {name:<12} {cnt:>6,} ({pct:.1f}%)")
        lines.append("")

    ax5.text(
        0.04, 0.97, "\n".join(lines),
        transform=ax5.transAxes, fontsize=8.5, va="top", ha="left",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", alpha=0.85),
    )

    # --- JSON export ---
    json_out = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "task": "detection",
        "total_images": total_images,
        "total_annotations": total_ann,
        "splits": {},
    }
    for sp in split_list:
        s = stats[sp]
        lpi = s["labels_per_image"]
        areas = s["bbox_areas"]
        n_ann = s["n_annotations"]
        total_area = max(len(areas), 1)
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
            "bbox_area_pct": {
                "mean": round(float(np.mean(areas)), 4) if areas else 0.0,
                "median": round(float(np.median(areas)), 4) if areas else 0.0,
                "p25": round(float(np.percentile(areas, 25)), 4) if areas else 0.0,
                "p75": round(float(np.percentile(areas, 75)), 4) if areas else 0.0,
            } if areas else {},
            "bbox_size_tiers": {
                "tiny_lt1pct": {
                    "count": sum(1 for a in areas if a < 1.0),
                    "pct": round(sum(1 for a in areas if a < 1.0) / total_area * 100, 1),
                },
                "small_1to5pct": {
                    "count": sum(1 for a in areas if 1.0 <= a < 5.0),
                    "pct": round(sum(1 for a in areas if 1.0 <= a < 5.0) / total_area * 100, 1),
                },
                "medium_5to15pct": {
                    "count": sum(1 for a in areas if 5.0 <= a < 15.0),
                    "pct": round(sum(1 for a in areas if 5.0 <= a < 15.0) / total_area * 100, 1),
                },
                "large_gt15pct": {
                    "count": sum(1 for a in areas if a >= 15.0),
                    "pct": round(sum(1 for a in areas if a >= 15.0) / total_area * 100, 1),
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
    PIXEL_SAMPLE_CAP = 256

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
        # Random subset for pixel stats
        rng = random.Random(0)
        n = len(samples)
        pick_n = min(PIXEL_SAMPLE_CAP, n)
        pick_idx = rng.sample(range(n), pick_n) if n else []
        ch_means = np.zeros(3, dtype=np.float64)
        ch_stds = np.zeros(3, dtype=np.float64)
        n_pix = 0
        for i, (path, cid) in enumerate(samples):
            class_counts[int(cid)] = class_counts.get(int(cid), 0) + 1
            # Cheap header-only read avoided; fall back to cv2 imread on sampled set
            if i in pick_idx:
                img = cv2.imread(str(path))
                if img is None:
                    continue
                h, w = img.shape[:2]
                widths.append(w)
                heights.append(h)
                aspects.append(w / max(h, 1))
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

        stats[split] = {
            "n_images": n,
            "class_counts": class_counts,
            "widths": widths,
            "heights": heights,
            "aspects": aspects,
            "pix_mean_rgb": mean.tolist(),
            "pix_std_rgb": std.tolist(),
            "pixel_sample_n": pick_n,
        }

    if not stats:
        logger.warning("Stats (classification): no splits loaded, skipping")
        return

    split_list = [s for s in splits if s in stats]
    all_class_ids = sorted({cid for s in stats.values() for cid in s["class_counts"]})
    total_images = sum(stats[s]["n_images"] for s in split_list)
    split_summary = "  |  ".join(f"{sp.capitalize()}: {stats[sp]['n_images']:,}" for sp in split_list)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"{split_summary}  |  Total: {total_images:,} images  |  Classification",
        fontsize=12, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[1, 2])

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
    ax1.set_xticklabels([class_names.get(cid, str(cid)) for cid in all_class_ids], rotation=20)
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
        ax3.set_title("Image Aspect Ratio (W/H)", fontsize=11, fontweight="bold")
        ax3.set_xlabel("aspect ratio  [sampled]")
        ax3.set_ylabel("# images")
        ax3.legend(fontsize=8, framealpha=0.9)

    # [1,1] Resolution scatter (W vs H), sampled
    for sp in split_list:
        w = stats[sp]["widths"]; h = stats[sp]["heights"]
        if w and h:
            ax4.scatter(w, h, s=6, alpha=0.35, label=sp,
                        color=_SPLIT_COLORS.get(sp, "#888"))
    ax4.set_title("Resolution (sampled)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("width (px)")
    ax4.set_ylabel("height (px)")
    if any(stats[s]["widths"] for s in split_list):
        ax4.legend(fontsize=8, framealpha=0.9)

    # [1,2] Per-channel mean/std table
    ax5.axis("off")
    lines = ["Pixel statistics (RGB, [0,1])\n"]
    for sp in split_list:
        m = stats[sp]["pix_mean_rgb"]; s = stats[sp]["pix_std_rgb"]
        n_s = stats[sp]["pixel_sample_n"]
        lines.append(f"[{sp}]  (n={n_s} imgs sampled)")
        lines.append(f"  mean  R={m[0]:.3f}  G={m[1]:.3f}  B={m[2]:.3f}")
        lines.append(f"  std   R={s[0]:.3f}  G={s[1]:.3f}  B={s[2]:.3f}")
        lines.append("")
    lines.append("Class counts")
    for sp in split_list:
        lines.append(f"[{sp}]")
        total = sum(stats[sp]["class_counts"].values()) or 1
        for cid in all_class_ids:
            cnt = stats[sp]["class_counts"].get(cid, 0)
            name = class_names.get(cid, str(cid))
            lines.append(f"  {name:<14} {cnt:>6,} ({cnt / total * 100:5.1f}%)")
        lines.append("")
    ax5.text(0.02, 0.97, "\n".join(lines), transform=ax5.transAxes,
             fontsize=8.5, va="top", ha="left", family="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", alpha=0.85))

    json_out = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "task": "classification",
        "total_images": total_images,
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
        coverage: list[float] = []  # non-bg % per image
        comp_counts_per_class: dict[int, list[int]] = {}
        comp_areas_per_class: dict[int, list[int]] = {}

        for i in pick_idx:
            img_path = img_paths[i]
            mask_path = ds.mask_dir / f"{img_path.stem}.png"
            if not mask_path.exists():
                continue
            m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            bc = np.bincount(m.ravel(), minlength=pixel_counts.shape[0])
            # Extend if mask has id beyond declared num_classes
            if bc.shape[0] > pixel_counts.shape[0]:
                extra = np.zeros(bc.shape[0] - pixel_counts.shape[0], dtype=np.int64)
                pixel_counts = np.concatenate([pixel_counts, extra])
            pixel_counts[: bc.shape[0]] += bc
            total_px = m.size
            non_bg = int((m != 0).sum())
            coverage.append(non_bg / total_px * 100.0)

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

    fig = plt.figure(figsize=(20, 11))
    fig.suptitle(
        f"{split_summary}  |  Total: {total_images:,} images  |  Semantic Segmentation",
        fontsize=12, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.38)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1:])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 2])

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
    ax1.set_ylabel("# pixels (sampled)")
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

    # JSON export
    json_out = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "task": "segmentation",
        "total_images": total_images,
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
) -> None:
    """Keypoint stats: images/split, instances/split, per-joint visibility rate,
    spatial heatmap (normalized to bbox), inter-joint distance histogram.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    from core.p05_data.keypoint_dataset import KeypointDataset

    num_keypoints = int(data_cfg.get("num_keypoints", 0))
    skeleton = data_cfg.get("skeleton") or []  # list of [a, b] joint-index pairs

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
        vis_counts = np.zeros(num_keypoints, dtype=np.int64)   # visibility > 0
        labeled_counts = np.zeros(num_keypoints, dtype=np.int64)  # total instances
        # Spatial heatmap buffer (bbox-normalized keypoint locations)
        HEAT = 64
        heat = np.zeros((HEAT, HEAT), dtype=np.float64)
        edge_lengths: dict[tuple[int, int], list[float]] = {}

        for img_path in img_paths:
            lbl = ds._load_label(img_path)
            boxes = lbl["boxes"]           # (N, 5) normalized
            kpts = lbl["keypoints"]        # (N, K, 3) normalized image coords
            if len(boxes) == 0:
                continue
            n_instances += len(boxes)
            labeled_counts += len(boxes)
            vis = (kpts[:, :, 2] > 0)
            vis_counts += vis.sum(axis=0).astype(np.int64)

            # Bbox-normalize keypoint positions (0..1 inside bbox)
            for i in range(len(boxes)):
                cx, cy, bw, bh = boxes[i, 1], boxes[i, 2], boxes[i, 3], boxes[i, 4]
                if bw <= 0 or bh <= 0:
                    continue
                x0, y0 = cx - bw / 2, cy - bh / 2
                labeled_mask = kpts[i, :, 2] > 0
                if labeled_mask.any():
                    kx = (kpts[i, labeled_mask, 0] - x0) / bw
                    ky = (kpts[i, labeled_mask, 1] - y0) / bh
                    kx = np.clip(kx, 0, 0.999)
                    ky = np.clip(ky, 0, 0.999)
                    hx = (kx * HEAT).astype(int)
                    hy = (ky * HEAT).astype(int)
                    np.add.at(heat, (hy, hx), 1.0)
                # Skeleton edge lengths (in normalized image coords)
                for (a, b) in skeleton:
                    if a < num_keypoints and b < num_keypoints:
                        if kpts[i, a, 2] > 0 and kpts[i, b, 2] > 0:
                            dx = kpts[i, a, 0] - kpts[i, b, 0]
                            dy = kpts[i, a, 1] - kpts[i, b, 1]
                            edge_lengths.setdefault((int(a), int(b)), []).append(
                                float(np.sqrt(dx * dx + dy * dy))
                            )

        stats[split] = {
            "n_images": len(img_paths),
            "n_instances": n_instances,
            "vis_counts": vis_counts.tolist(),
            "labeled_counts": labeled_counts.tolist(),
            "heat": heat,
            "edge_lengths": edge_lengths,
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

    fig = plt.figure(figsize=(20, 11))
    fig.suptitle(
        f"{split_summary}  |  Total: {total_images:,} images / {total_inst:,} instances  |  Keypoint",
        fontsize=12, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.38)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[1, 2])

    # [0,0] Images per split
    _bar_images_per_split(ax0, split_list, [stats[s]["n_images"] for s in split_list])

    # [0,1] Instances per split
    _bar_images_per_split(ax1, split_list, [stats[s]["n_instances"] for s in split_list])
    ax1.set_title("Instances per Split", fontsize=11, fontweight="bold")
    ax1.set_xlabel("# instances")

    # [0,2] Per-joint visibility rate
    # Aggregate across splits
    total_vis = np.zeros(num_keypoints, dtype=np.int64)
    total_lab = np.zeros(num_keypoints, dtype=np.int64)
    for sp in split_list:
        total_vis += np.asarray(stats[sp]["vis_counts"], dtype=np.int64)
        total_lab += np.asarray(stats[sp]["labeled_counts"], dtype=np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(total_lab > 0, total_vis / np.maximum(total_lab, 1), 0.0)
    ax2.bar(range(num_keypoints), rate * 100.0, color="#4C72B0")
    ax2.set_ylim(0, 105)
    ax2.set_title("Per-Joint Visibility Rate", fontsize=11, fontweight="bold")
    ax2.set_xlabel("joint index")
    ax2.set_ylabel("% instances w/ visible")

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
        # Flat histogram of all edges (color-coded first 8 edges separately)
        top_edges = list(agg_edges.items())[:8]
        for (a, b), lens in top_edges:
            ax4.hist(lens, bins=30, alpha=0.35, label=f"{a}-{b}")
        ax4.set_title("Skeleton Edge Lengths (top 8)", fontsize=11, fontweight="bold")
        ax4.set_xlabel("normalized length")
        ax4.set_ylabel("count")
        ax4.legend(fontsize=7, framealpha=0.9)
    else:
        ax4.axis("off")
        ax4.text(0.5, 0.5, "(no skeleton: set data.skeleton)",
                 ha="center", va="center", transform=ax4.transAxes, fontsize=10, color="#888")

    # [1,2] Summary text panel
    ax5.axis("off")
    lines = ["Key Statistics (keypoint)\n"]
    for sp in split_list:
        s = stats[sp]
        lines.append(f"[{sp}]")
        lines.append(f"  images        {s['n_images']:>7,}")
        lines.append(f"  instances     {s['n_instances']:>7,}")
        if s["n_instances"]:
            mean_vis = np.mean(np.asarray(s["vis_counts"]) /
                               max(s["n_instances"], 1)) * 100
            lines.append(f"  mean vis rate  {mean_vis:>6.1f}%")
        lines.append("")
    ax5.text(0.02, 0.97, "\n".join(lines), transform=ax5.transAxes,
             fontsize=9, va="top", ha="left", family="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", alpha=0.85))

    json_out = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "task": "keypoint",
        "num_keypoints": num_keypoints,
        "total_images": total_images,
        "total_instances": total_inst,
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
) -> None:
    """Emit ``00_dataset_info.{md,json}`` with dataset provenance.

    Self-describing run folders: any reader can tell which dataset a run
    trained on without needing the original training config in hand.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    training_cfg = training_cfg or {}
    data_section = training_cfg.get("data", {}) or {}
    # tensor_prep is authoritative for input_size / mean / std / normalize.
    from utils.config import resolve_tensor_prep
    backend = (training_cfg.get("training", {}) or {}).get("backend", "pytorch")
    tp = resolve_tensor_prep(training_cfg, backend=backend) or {}
    input_size = (
        tp.get("input_size") or data_cfg.get("input_size") or data_section.get("input_size")
    )
    mean = tp.get("mean") or data_cfg.get("mean") or data_section.get("mean")
    std = tp.get("std") or data_cfg.get("std") or data_section.get("std")
    dataset_root = data_cfg.get("path") or data_cfg.get("root")
    gpu_augment = bool((training_cfg.get("training", {}) or {}).get("gpu_augment", False))

    info = {
        "feature_name": feature_name,
        "data_config": str(data_config_path) if data_config_path else None,
        "training_config": str(training_config_path) if training_config_path else None,
        "dataset_name": data_cfg.get("dataset_name") or data_cfg.get("name"),
        "dataset_root": str(dataset_root) if dataset_root else None,
        "num_classes": len(class_names) if class_names else None,
        "class_names": {int(k): str(v) for k, v in (class_names or {}).items()},
        "split_sizes": {str(k): int(v) for k, v in (split_sizes or {}).items()},
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
        "run_started": datetime.datetime.now().isoformat(timespec="seconds"),
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

    md_lines = [
        f"# Dataset info — {info['feature_name'] or 'unknown'}",
        "",
        f"- **Feature folder**: `{info['feature_name']}`" if info["feature_name"] else "- **Feature folder**: _unknown_",
        f"- **Data config**: `{info['data_config']}`",
        f"- **Training config**: `{info['training_config']}`",
        f"- **Dataset name**: `{info['dataset_name']}`",
        f"- **Dataset root**: `{info['dataset_root']}`",
        f"- **Num classes**: {info['num_classes']}",
        f"- **Class names**: {_fmt(info['class_names'])}",
        f"- **Split sizes**: {_fmt(info['split_sizes'])}",
        f"- **Input size**: {_fmt(info['input_size'])}",
        f"- **Normalization**: mean={_fmt(info['mean'])}, std={_fmt(info['std'])}",
        f"- **tensor_prep**: {_fmt(info['tensor_prep'])}",
        f"- **Training backend**: `{info['backend']}`",
        f"- **GPU augmentation**: `{info['gpu_augment']}`",
        f"- **Augmentation**: {_fmt(info['augmentation'])}",
        f"- **Run started**: {info['run_started']}",
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
    dpi = data_viz_cfg.get("dpi", 120)

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

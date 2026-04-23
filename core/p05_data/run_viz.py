#!/usr/bin/env python3
"""Standalone CLI to visualize dataset samples and augmentation before training.

Generates (numbered by pipeline step):
  - 00_dataset_info.{md,json}    — dataset provenance (name, path, classes, splits)
  - 01_dataset_stats.{png,json}  — split sizes + class balance + bbox tiers
  - 02_data_labels_<split>.png   — raw images with GT bounding boxes
  - 03_aug_labels_<split>.png    — augmented images with transformed GT boxes
  - 04_normalize_check.png       — normalize check: raw | normalized (false-color) | denormalized

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
import logging
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
from utils.viz import VizStyle, annotate_detections, apply_plot_style, save_image_grid

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

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

def generate_dataset_stats(
    data_cfg: dict,
    base_dir: str,
    class_names: dict,
    splits: list[str],
    out_dir: Path,
    dpi: int = 120,
    subset_indices: dict[str, list[int]] | None = None,
) -> None:
    """Generate dataset_stats.png: split sizes, class distribution, bbox stats.

    Reads only label .txt files — no image loading — so it's fast even on large splits.
    Output: out_dir/dataset_stats.png

    When ``subset_indices[split]`` is provided, stats reflect only those indices
    (mirrors the run's active ``data.subset.*`` filtering). Splits without an
    entry (or ``None``) use the full split.
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
    split_colors = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}

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
    ax1.legend(fontsize=8)
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
               fontsize=8, loc="lower right")

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
        ax3.legend(fontsize=8)

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
    ax4.legend(fontsize=7.5)

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

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- JSON export ---
    json_out = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
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
    json_path = out_dir / "01_dataset_stats.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    logger.info("Saved: %s", json_path)

    out_path = out_dir / "01_dataset_stats.png"
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


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
    keys = (
        "library", "mosaic", "mixup", "copypaste", "ir_simulation",
        "hflip", "vflip", "affine", "perspective", "color_jitter",
        "brightness", "contrast", "saturation", "hue", "normalize",
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
    input_size = data_cfg.get("input_size") or data_section.get("input_size")
    mean = data_cfg.get("mean") or data_section.get("mean")
    std = data_cfg.get("std") or data_section.get("std")
    dataset_root = data_cfg.get("path") or data_cfg.get("root")
    backend = (training_cfg.get("training", {}) or {}).get("backend", "pytorch")
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

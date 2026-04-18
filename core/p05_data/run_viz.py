#!/usr/bin/env python3
"""Standalone CLI to visualize dataset samples and augmentation before training.

Generates:
  - data_labels_<split>.png  — raw images with GT bounding boxes
  - aug_labels_<split>.png   — augmented images with transformed GT boxes

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
from typing import List

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from utils.config import load_config, feature_name_from_config_path, generate_run_dir
from core.p05_data.detection_dataset import YOLOXDataset
from core.p05_data.transforms import build_transforms

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Drawing helpers (same palette as callbacks.py)
# ---------------------------------------------------------------------------

_LABEL_PALETTE = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (0, 128, 255), (128, 0, 255),
    (0, 255, 128), (255, 128, 0), (64, 0, 255), (255, 64, 0),
    (0, 255, 64), (64, 255, 0), (255, 0, 64), (0, 64, 255),
    (128, 128, 0), (0, 128, 128), (128, 0, 128), (64, 64, 0),
]


def _draw_gt_boxes(image, targets, class_names, thickness=2, text_scale=0.5):
    vis = image.copy()
    h, w = vis.shape[:2]
    for row in targets:
        cls_id = int(row[0])
        cx, cy, bw, bh = row[1], row[2], row[3], row[4]
        x1, y1 = int((cx - bw/2)*w), int((cy - bh/2)*h)
        x2, y2 = int((cx + bw/2)*w), int((cy + bh/2)*h)
        color = _LABEL_PALETTE[cls_id % len(_LABEL_PALETTE)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        label = class_names.get(cls_id, str(cls_id))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 1)
        cv2.rectangle(vis, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
        cv2.putText(vis, label, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255,255,255), 1)
    return vis


def _save_image_grid(annotated, grid_cols, title, out_path, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncols = min(grid_cols, len(annotated))
    nrows = -(-len(annotated) // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*3.5))
    axes = np.asarray(axes).ravel()
    for i in range(nrows * ncols):
        axes[i].axis("off")
        if i < len(annotated):
            axes[i].imshow(cv2.cvtColor(annotated[i], cv2.COLOR_BGR2RGB))
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


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
        img = item["image"]
        targets = ds._load_label(ds.img_paths[idx])
        if targets is None or len(targets) == 0:
            targets = np.zeros((0, 5), dtype=np.float32)
        annotated.append(_draw_gt_boxes(img, targets, class_names, thickness, text_scale))
    out_path = out_dir / f"data_labels_{split}.png"
    _save_image_grid(annotated, grid_cols, f"Data + Labels [{split}] — {k}/{n} samples", out_path, dpi)


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
        aug_bgr = (aug_np[:, :, ::-1] * 255).astype(np.uint8)
        targets_np = targets_tensor.numpy() if len(targets_tensor) > 0 else np.zeros((0, 5), dtype=np.float32)
        annotated.append(_draw_gt_boxes(aug_bgr, targets_np, class_names, thickness, text_scale))
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
        out_path = out_dir / f"aug_labels_{split}{suffix}.png"
        title = f"Augmented + Labels [{split}] ({'no mosaic/mixup' if label == 'simple' else 'with mosaic/mixup'}) — {k}/{n} samples"
        _save_image_grid(annotated, grid_cols, title, out_path, dpi)


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

def generate_dataset_stats(
    data_cfg: dict,
    base_dir: str,
    class_names: dict,
    splits: List[str],
    out_dir: Path,
    dpi: int = 120,
) -> None:
    """Generate dataset_stats.png: split sizes, class distribution, bbox stats.

    Reads only label .txt files — no image loading — so it's fast even on large splits.
    Output: out_dir/dataset_stats.png
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # Collect stats per split (label-file pass only, no image I/O)
    stats = {}
    for split in splits:
        try:
            ds = YOLOXDataset(data_config=data_cfg, split=split, transforms=None, base_dir=base_dir)
        except Exception as e:
            logger.warning("Stats: skipping split %s — %s", split, e)
            continue

        class_counts: dict = {}
        bbox_areas: List[float] = []
        labels_per_image: List[int] = []
        n_empty = 0

        for img_path in ds.img_paths:
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
            "n_images": len(ds),
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
    for bar, n in zip(bars, n_imgs):
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
        for patch, sp in zip(bp["boxes"], lpi_labels):
            patch.set_facecolor(split_colors.get(sp, "#888"))
            patch.set_alpha(0.6)
        # Overlay individual mean markers
        for i, (sp, data) in enumerate(zip(lpi_labels, lpi_data), start=1):
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
        lines.append(f"  bbox size tiers:")
        lines.append(f"    tiny  (<1%)  {tiny_pct:>6.1f}%")
        lines.append(f"    small (1-5%) {small_pct:>6.1f}%")
        lines.append(f"    med  (5-15%) {med_pct:>6.1f}%")
        lines.append(f"    large (>15%) {large_pct:>6.1f}%")
        lines.append(f"  avg bbox area  {avg_area:.3f}%")
        lines.append(f"  class counts:")
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
    json_path = out_dir / "dataset_stats.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    logger.info("Saved: %s", json_path)

    out_path = out_dir / "dataset_stats.png"
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


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

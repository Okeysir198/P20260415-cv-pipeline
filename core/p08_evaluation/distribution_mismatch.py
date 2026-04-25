"""Train ↔ Val distribution-mismatch analyzer.

Detects hidden domain gap between training and validation splits by
comparing two cheap-to-compute distributions:

1. **Class distribution by split** — per-class prevalence in train vs val.
   Quantified with Jensen-Shannon divergence (symmetric, bounded [0, 1]).
2. **Image statistics drift** — per-image brightness / contrast / aspect
   ratio / area (px²). Quantified per-stat with the Kolmogorov-Smirnov
   two-sample test p-value.

Outputs (flat, under the error_analysis dir passed as ``output_dir``):

- ``03_distribution_mismatch.png`` — 2-panel combined figure (class bars
  + image-stats histograms).
- ``03_distribution_mismatch.json`` — all numeric divergences.

Both panels use random-subset sampling capped at ``max_samples_per_split``
(default 500) to keep wall-time bounded on large datasets.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


DM_FILENAMES: dict[str, str] = {
    "distribution_mismatch": "03_distribution_mismatch.png",
    "json":                  "03_distribution_mismatch.json",
}

_MAX_SAMPLES = 500
_RNG_SEED = 42


def analyze_distribution_mismatch(
    *,
    train_dataset,
    val_dataset,
    test_dataset=None,
    output_dir: Path | str,
    task: str,
    class_names: dict[int, str],
    max_samples_per_split: int = _MAX_SAMPLES,
) -> dict[str, Any]:
    """Render distribution-drift charts + emit json. Returns artefacts dict."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {"train": train_dataset, "val": val_dataset}
    if test_dataset is not None and len(test_dataset) > 0:
        splits["test"] = test_dataset

    artifacts: dict[str, Path] = {}

    class_dist = {
        s: _per_class_counts(ds, task=task, max_samples=max_samples_per_split)
        for s, ds in splits.items() if ds is not None and len(ds) > 0
    }
    js_train_val = _js_divergence(
        class_dist.get("train", {}), class_dist.get("val", {}),
    )
    stats = {
        s: _image_stats(ds, max_samples=max_samples_per_split)
        for s, ds in splits.items() if ds is not None and len(ds) > 0
    }
    ks_results = _ks_per_stat(stats)

    artifacts["distribution_mismatch"] = _plot_combined(
        class_dist=class_dist, class_names=class_names, js=js_train_val,
        stats=stats, ks_results=ks_results,
        out_path=out_dir / DM_FILENAMES["distribution_mismatch"],
    )

    payload = {
        "task": task,
        "class_distribution_js_train_val": round(js_train_val, 4),
        "image_stats_ks": {
            stat: {pair: round(p, 4) for pair, p in pairs.items()}
            for stat, pairs in ks_results.items()
        },
        "n_samples_per_split": {s: len(d) for s, d in stats.items()},
    }
    json_path = out_dir / DM_FILENAMES["json"]
    json_path.write_text(json.dumps(payload, indent=2))
    artifacts["json"] = json_path

    chart_metrics = {
        "03_distribution_mismatch": {
            "js_divergence": js_train_val,
            "ks_min_pvalue": _min_ks_p(ks_results),
        },
    }
    return {"artifacts": artifacts, "chart_metrics": chart_metrics, "payload": payload}


def _plot_combined(
    *, class_dist, class_names, js, stats, ks_results, out_path,
) -> Path:
    """Render class-distribution bars + image-stats histograms in one figure."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    palette = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1.5, 1.5], hspace=0.45, wspace=0.3)

    # -------- Top row: class distribution (full width) --------
    ax_top = fig.add_subplot(gs[0, :])
    all_classes = sorted({c for d in class_dist.values() for c in d})
    if all_classes:
        labels = [class_names.get(c, str(c)) for c in all_classes]
        splits = list(class_dist.keys())
        bar_w = 0.8 / max(len(splits), 1)
        x = np.arange(len(all_classes))
        for i, s in enumerate(splits):
            d = class_dist[s]
            total = sum(d.values()) or 1
            rates = [d.get(c, 0) / total * 100.0 for c in all_classes]
            ax_top.bar(
                x + (i - (len(splits) - 1) / 2) * bar_w, rates, bar_w,
                label=s, color=palette.get(s, "#999999"),
                edgecolor="black", linewidth=0.5,
            )
        ax_top.set_xticks(x)
        ax_top.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax_top.set_ylabel("% of GT instances in split")
        ax_top.legend(loc="upper right")
    else:
        ax_top.text(0.5, 0.5, "No class data available", ha="center", va="center")
        ax_top.axis("off")
    ax_top.set_title(
        f"Class distribution by split — JS divergence (train vs val): {js:.3f}",
        fontsize=11,
    )

    # -------- Bottom 2x2: image stats histograms --------
    axes = [
        fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]),
    ]
    for ax, key in zip(axes, _STAT_KEYS, strict=False):
        for s, d in stats.items():
            xs = d.get(key, [])
            if not xs:
                continue
            ax.hist(
                xs, bins=25, alpha=0.55, label=s,
                color=palette.get(s, "#999999"),
                edgecolor="black", linewidth=0.3,
            )
        ax.set_title(_pretty_stat_label(key, ks_results.get(key, {})), fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle("Distribution mismatch — train vs val", fontsize=12, fontweight="bold")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Class distribution
# ---------------------------------------------------------------------------


def _per_class_counts(ds, *, task: str, max_samples: int) -> dict[int, int]:
    """Count GT instances per class across a sampled subset."""
    counts: dict[int, int] = {}
    n = len(ds)
    if n == 0:
        return counts
    rng = np.random.RandomState(_RNG_SEED)
    n_use = min(n, max_samples)
    indices = sorted(rng.choice(n, n_use, replace=False).tolist())
    inner = getattr(ds, "dataset", ds)
    real_indices = [getattr(ds, "indices", list(range(n)))[i] for i in indices]
    if not hasattr(inner, "get_raw_item"):
        return counts
    for idx in real_indices:
        try:
            raw = inner.get_raw_item(idx)
            target = raw.get("targets") if raw else None
            if target is None:
                continue
            if task == "detection":
                if isinstance(target, np.ndarray) and target.ndim == 2:
                    for cid in target[:, 0].astype(int):
                        counts[int(cid)] = counts.get(int(cid), 0) + 1
            elif task == "segmentation":
                arr = np.asarray(target)
                vals, cs = np.unique(arr, return_counts=True)
                for v, c in zip(vals, cs, strict=False):
                    counts[int(v)] = counts.get(int(v), 0) + int(c)
            elif task == "classification":
                try:
                    counts[int(target)] = counts.get(int(target), 0) + 1
                except (TypeError, ValueError):
                    pass
            elif task == "keypoint":
                if isinstance(target, dict) and "boxes" in target:
                    boxes = np.asarray(target["boxes"])
                    if boxes.ndim == 2 and boxes.shape[1] >= 5:
                        for cid in boxes[:, 0].astype(int):
                            counts[int(cid)] = counts.get(int(cid), 0) + 1
        except Exception as e:  # pragma: no cover
            logger.warning("dist_mismatch[%s]: idx %d failed — %s", task, idx, e)
    return counts


def _js_divergence(a: dict[int, int], b: dict[int, int]) -> float:
    """Jensen-Shannon divergence between two count dicts (base 2, bounded [0,1])."""
    if not a or not b:
        return 0.0
    keys = sorted(set(a) | set(b))
    pa = np.array([a.get(k, 0) for k in keys], dtype=np.float64)
    pb = np.array([b.get(k, 0) for k in keys], dtype=np.float64)
    if pa.sum() == 0 or pb.sum() == 0:
        return 0.0
    pa /= pa.sum()
    pb /= pb.sum()
    m = 0.5 * (pa + pb)
    eps = 1e-12

    def _kl(p, q):
        return float(np.sum(np.where(p > 0, p * np.log2((p + eps) / (q + eps)), 0.0)))

    return 0.5 * _kl(pa, m) + 0.5 * _kl(pb, m)


def _plot_class_distribution(
    class_dist: dict[str, dict[int, int]],
    class_names: dict[int, str],
    js: float,
    out_path: Path,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_classes = sorted({c for d in class_dist.values() for c in d})
    if not all_classes:
        # Empty — write a placeholder so downstream linkage doesn't break.
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No class data available", ha="center", va="center")
        ax.axis("off")
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out_path

    labels = [class_names.get(c, str(c)) for c in all_classes]
    splits = list(class_dist.keys())
    bar_w = 0.8 / max(len(splits), 1)
    x = np.arange(len(all_classes))
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(all_classes) + 2), 5))
    palette = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}
    for i, s in enumerate(splits):
        d = class_dist[s]
        total = sum(d.values()) or 1
        rates = [d.get(c, 0) / total * 100.0 for c in all_classes]
        ax.bar(x + (i - (len(splits) - 1) / 2) * bar_w, rates, bar_w,
               label=s, color=palette.get(s, "#999999"), edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("% of GT instances in split")
    ax.set_title(f"Class distribution by split — JS divergence (train vs val): {js:.3f}",
                 fontsize=11)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Image statistics
# ---------------------------------------------------------------------------


_STAT_KEYS = ["brightness", "contrast", "aspect_ratio", "area_px2"]


def _image_stats(ds, *, max_samples: int) -> dict[str, list[float]]:
    """Per-image brightness / contrast / aspect_ratio / area_px²."""
    out: dict[str, list[float]] = {k: [] for k in _STAT_KEYS}
    n = len(ds)
    if n == 0:
        return out
    rng = np.random.RandomState(_RNG_SEED)
    n_use = min(n, max_samples)
    indices = sorted(rng.choice(n, n_use, replace=False).tolist())
    inner = getattr(ds, "dataset", ds)
    real_indices = [getattr(ds, "indices", list(range(n)))[i] for i in indices]
    if not hasattr(inner, "get_raw_item"):
        return out
    for idx in real_indices:
        try:
            raw = inner.get_raw_item(idx)
            img = raw.get("image") if raw else None
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            h, w = img.shape[:2]
            out["brightness"].append(float(gray.mean()))
            out["contrast"].append(float(gray.std()))
            out["aspect_ratio"].append(float(w) / max(h, 1))
            out["area_px2"].append(float(h * w))
        except Exception as e:  # pragma: no cover
            logger.warning("dist_mismatch[image_stats]: idx %d failed — %s", idx, e)
    return out


def _ks_per_stat(stats: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, float]]:
    """Pairwise KS test per stat. Returns {stat_key: {"train_vs_val": p_value, …}}."""
    try:
        from scipy.stats import ks_2samp
    except Exception:
        logger.info("scipy not available — skipping KS tests")
        return {k: {} for k in _STAT_KEYS}
    res: dict[str, dict[str, float]] = {k: {} for k in _STAT_KEYS}
    splits = list(stats.keys())
    for k in _STAT_KEYS:
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                a, b = splits[i], splits[j]
                xs, ys = stats[a].get(k, []), stats[b].get(k, [])
                if len(xs) < 2 or len(ys) < 2:
                    continue
                p = float(ks_2samp(xs, ys).pvalue)
                res[k][f"{a}_vs_{b}"] = p
    return res


def _min_ks_p(ks_results: dict[str, dict[str, float]]) -> float:
    """Smallest p-value across all (stat, pair) — most-significant drift signal."""
    vals = [v for d in ks_results.values() for v in d.values()]
    return float(min(vals)) if vals else 1.0


def _plot_image_stats(
    stats: dict[str, dict[str, list[float]]],
    ks_results: dict[str, dict[str, float]],
    out_path: Path,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    palette = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()
    for ax, key in zip(axes, _STAT_KEYS, strict=False):
        for s, d in stats.items():
            xs = d.get(key, [])
            if not xs:
                continue
            ax.hist(xs, bins=25, alpha=0.55, label=s,
                    color=palette.get(s, "#999999"), edgecolor="black", linewidth=0.3)
        ax.set_title(_pretty_stat_label(key, ks_results.get(key, {})), fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("Image-statistics drift across splits", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _pretty_stat_label(key: str, pvals: dict[str, float]) -> str:
    base = {
        "brightness": "Mean luma",
        "contrast": "Luma std-dev",
        "aspect_ratio": "Image aspect ratio (W/H)",
        "area_px2": "Image area (px²)",
    }.get(key, key)
    if pvals:
        # Show smallest p-value (most extreme drift)
        worst_pair, worst_p = min(pvals.items(), key=lambda kv: kv[1])
        return f"{base} — KS({worst_pair}) p={worst_p:.3f}"
    return base

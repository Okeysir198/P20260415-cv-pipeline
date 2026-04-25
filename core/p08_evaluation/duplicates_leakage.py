"""Near-duplicate + cross-split leakage analyzer.

Detects two closely-related data hygiene issues that silently inflate val
metrics:

1. **Near-duplicates within a split** — repeated / near-copy images in train
   (redundant training signal) or val (metric over-counts the same content).
2. **Cross-split leakage** — the same (or near-same) image appearing in both
   train and val/test, which directly leaks test labels into training and
   produces optimistically-biased held-out metrics.

Signal is **perceptual hash (pHash)** with Hamming-distance threshold ≤ 6.
Cheap (one 32×32 DCT per image), robust to recompression / mild resize /
minor crops, and sensitive enough for the real-world duplication modes we
see in scraped datasets. For crops + heavy recolor a CLIP-embedding path
would be stronger — left as a follow-up if pHash under-detects in practice.

Outputs (flat, under the error_analysis dir passed as ``output_dir``):

- ``05_duplicates_leakage.png`` — two panels:
    1. Near-duplicate pair count per split.
    2. Cross-split leakage pair counts (train↔val, train↔test, val↔test).
- ``05_duplicates_leakage.json`` — all pair listings (paths + Hamming
  distance) + summary counts.

Module shape mirrors ``label_quality.py`` / ``distribution_mismatch.py`` /
``learning_ability.py``: single public ``run()`` entry point that writes
PNG + JSON and returns an ``{artifacts, chart_metrics, payload}`` dict.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils.viz import apply_plot_style

from loguru import logger

matplotlib.use("Agg")


DL_FILENAMES: dict[str, str] = {
    "duplicates_leakage": "05_duplicates_leakage.png",
}

# pHash Hamming distance threshold. 0 = identical hash; 6/64 ≈ 9% bits differ,
# the commonly-cited "near-duplicate" cutoff in the imagehash literature.
_HAMMING_THRESH = 6
_HASH_SIZE = 8           # final hash side (→ 64 bits)
_DCT_SIZE = 32           # pre-DCT downsize
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run(
    data_config: dict[str, Any],
    output_dir: Path | str,
    *,
    max_samples_per_split: int | None = None,
    hamming_thresh: int = _HAMMING_THRESH,
    task: str | None = None,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Hash every image in every split, report self + cross-split near-duplicates.

    Args:
        data_config: resolved ``05_data.yaml`` dict. Reads ``path`` (root) and
            ``train`` / ``val`` / ``test`` (split subpaths) keys.
        output_dir: error-analysis dir; artefacts land directly inside it.
        max_samples_per_split: optional cap. ``None`` = hash everything.
        hamming_thresh: pair is a near-duplicate if Hamming distance ≤ this.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_paths = _enumerate_split_images(
        data_config, max_samples_per_split, task=task, base_dir=base_dir,
    )
    hashes_by_split = {
        split: _hash_paths(paths) for split, paths in split_paths.items()
    }

    within = {
        split: _find_within_pairs(pairs, hamming_thresh)
        for split, pairs in hashes_by_split.items()
    }
    cross = _find_cross_pairs(hashes_by_split, hamming_thresh)

    png_path = _plot(
        within=within, cross=cross,
        out_path=out_dir / DL_FILENAMES["duplicates_leakage"],
        hamming_thresh=hamming_thresh,
    )

    payload = {
        "hamming_thresh": hamming_thresh,
        "n_images_per_split": {s: len(h) for s, h in hashes_by_split.items()},
        "within_split_pairs": {
            s: {"count": len(p), "pairs": p} for s, p in within.items()
        },
        "cross_split_pairs": {
            pair: {"count": len(p), "pairs": p} for pair, p in cross.items()
        },
    }
    json_path = out_dir / "05_duplicates_leakage.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str))

    chart_metrics = {
        "05_duplicates_leakage": {
            "n_within_duplicates": sum(len(p) for p in within.values()),
            "n_cross_leakage": sum(len(p) for p in cross.values()),
            "worst_cross_pair": _worst_cross_pair(cross),
        }
    }
    return {
        "artifacts": {"duplicates_leakage": png_path, "json": json_path},
        "chart_metrics": chart_metrics,
        "payload": payload,
    }


# ---------------------------------------------------------------------------
# Image enumeration
# ---------------------------------------------------------------------------


def _enumerate_split_images(
    data_config: dict[str, Any], max_per_split: int | None,
    *, task: str | None = None, base_dir: str | Path | None = None,
) -> dict[str, list[Path]]:
    """Resolve each split's image paths via the same p05 loader the trainer
    uses, falling back to a raw-filesystem walk only if the loader can't be
    constructed.

    Loader-based path: builds ``YOLOXDataset`` / ``ClassificationDataset`` /
    etc. via :func:`core.p06_training._common.build_dataset_for_viz` and
    reads ``dataset.image_paths``. This handles split-aliasing (e.g.
    ``test → val/images``) and dataset-class-specific path resolution that
    the raw walk misses.

    Filesystem fallback: ``<path>/<split>`` (or ``<path>/<split>/images``
    when present). Logs a warning per skipped split rather than failing.
    """
    # ---- Loader-based enumeration (preferred) --------------------------------
    loader_paths = _enumerate_via_loader(data_config, task, base_dir)
    if loader_paths:
        return _cap_per_split(loader_paths, max_per_split)

    # ---- Filesystem fallback -------------------------------------------------
    root = Path(data_config.get("path", "."))
    out: dict[str, list[Path]] = {}
    for split in ("train", "val", "test"):
        sub = data_config.get(split)
        if not sub:
            continue
        split_dir = (root / sub).resolve()
        if not split_dir.exists():
            # Fallback: maybe `sub` already points at a `<split>/` dir whose
            # images live under `images/`.
            alt = (root / sub / "images").resolve()
            if alt.exists():
                split_dir = alt
            else:
                logger.info("duplicates_leakage: split dir not found — %s", split_dir)
                continue
        # If split_dir is e.g. ".../train" with an images/ subdir, prefer that.
        if (split_dir / "images").is_dir():
            split_dir = split_dir / "images"
        paths = sorted(
            p for p in split_dir.rglob("*") if p.suffix.lower() in _IMG_EXTS
        )
        out[split] = paths
    return _cap_per_split(out, max_per_split)


def _cap_per_split(
    split_paths: dict[str, list[Path]], max_per_split: int | None,
) -> dict[str, list[Path]]:
    if max_per_split is None:
        return split_paths
    rng = np.random.RandomState(42)
    capped: dict[str, list[Path]] = {}
    for split, paths in split_paths.items():
        if len(paths) > max_per_split:
            idx = sorted(rng.choice(len(paths), max_per_split, replace=False).tolist())
            capped[split] = [paths[i] for i in idx]
        else:
            capped[split] = paths
    return capped


def _enumerate_via_loader(
    data_config: dict[str, Any], task: str | None, base_dir: str | Path | None,
) -> dict[str, list[Path]]:
    """Try building each split via ``build_dataset_for_viz`` and reading
    ``dataset.image_paths``. Returns ``{}`` if the import / construction fails
    for every split — caller falls back to the filesystem walk.
    """
    try:
        from core.p06_training._common import build_dataset_for_viz
    except Exception as e:  # pragma: no cover
        logger.info("duplicates_leakage: loader import failed — %s", e)
        return {}
    task_low = (task or "detection").lower()
    if task_low not in {"detection", "classification", "segmentation", "keypoint"}:
        return {}
    bd = str(base_dir) if base_dir is not None else "."
    out: dict[str, list[Path]] = {}
    for split in ("train", "val", "test"):
        if not data_config.get(split):
            continue
        try:
            ds = build_dataset_for_viz(
                task=task_low, split=split, data_config=data_config,
                base_dir=bd, transforms=None,
            )
        except Exception as e:
            logger.info("duplicates_leakage: loader build failed (%s) — %s", split, e)
            continue
        # `image_paths` (BaseDataset / cls / seg / kpt) or `img_paths` (YOLOXDataset).
        paths = list(
            getattr(ds, "image_paths", None)
            or getattr(ds, "img_paths", None)
            or []
        )
        if paths:
            out[split] = [Path(p) for p in paths]
    return out


# ---------------------------------------------------------------------------
# pHash (PIL + numpy fallback — no `imagehash` dependency)
# ---------------------------------------------------------------------------


def _phash(path: Path) -> int | None:
    """64-bit perceptual hash via DCT on a 32×32 greyscale thumbnail.

    Returns an int whose popcount-xor-distance against another phash is the
    Hamming distance in bits.
    """
    try:
        with Image.open(path) as im:
            im = im.convert("L").resize(
                (_DCT_SIZE, _DCT_SIZE), Image.Resampling.LANCZOS,
            )
            arr = np.asarray(im, dtype=np.float32)
    except Exception as e:  # pragma: no cover
        logger.warning("duplicates_leakage: hash failed %s — %s", path, e)
        return None
    # 2D DCT-II via separable 1D DCTs (numpy only — scipy avoided).
    dct = _dct2(arr)
    top = dct[:_HASH_SIZE, :_HASH_SIZE]
    # Exclude the DC coefficient when computing the median (imagehash
    # convention — DC dominates and would bias the threshold).
    med = float(np.median(top.flatten()[1:]))
    bits = (top > med).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def _dct2(x: np.ndarray) -> np.ndarray:
    """Type-II 2D DCT via the FFT trick — no scipy dependency."""
    return _dct1(_dct1(x, axis=0), axis=1)


def _dct1(x: np.ndarray, *, axis: int) -> np.ndarray:
    n = x.shape[axis]
    # Mirror-extend: [x, reverse(x)] → FFT → take real part of first n bins,
    # multiplied by the standard DCT-II phase factor.
    y = np.concatenate([x, np.flip(x, axis=axis)], axis=axis)
    Y = np.fft.fft(y, axis=axis)
    k = np.arange(n)
    shape = [1] * x.ndim
    shape[axis] = n
    phase = np.exp(-1j * np.pi * k / (2 * n)).reshape(shape)
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(0, n)
    return (Y[tuple(sl)] * phase).real


def _hash_paths(paths: list[Path]) -> list[tuple[Path, int]]:
    out: list[tuple[Path, int]] = []
    for p in paths:
        h = _phash(p)
        if h is not None:
            out.append((p, h))
    return out


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# ---------------------------------------------------------------------------
# Pair finders
# ---------------------------------------------------------------------------


def _find_within_pairs(
    hashes: list[tuple[Path, int]], thresh: int,
) -> list[dict[str, Any]]:
    """O(N²) self-compare — fine for N ≤ a few thousand. Returns sorted pairs."""
    pairs: list[dict[str, Any]] = []
    for (pa, ha), (pb, hb) in combinations(hashes, 2):
        d = _hamming(ha, hb)
        if d <= thresh:
            pairs.append({"a": str(pa), "b": str(pb), "hamming": d})
    pairs.sort(key=lambda r: r["hamming"])
    return pairs


def _find_cross_pairs(
    hashes_by_split: dict[str, list[tuple[Path, int]]], thresh: int,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    splits = list(hashes_by_split.keys())
    for i, j in combinations(range(len(splits)), 2):
        sa, sb = splits[i], splits[j]
        key = f"{sa}_vs_{sb}"
        pairs: list[dict[str, Any]] = []
        for pa, ha in hashes_by_split[sa]:
            for pb, hb in hashes_by_split[sb]:
                d = _hamming(ha, hb)
                if d <= thresh:
                    pairs.append({
                        "split_a": sa, "a": str(pa),
                        "split_b": sb, "b": str(pb),
                        "hamming": d,
                    })
        pairs.sort(key=lambda r: r["hamming"])
        out[key] = pairs
    return out


def _worst_cross_pair(cross: dict[str, list]) -> str | None:
    worst = max(cross.items(), key=lambda kv: len(kv[1]), default=(None, []))
    return worst[0] if worst[1] else None


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _plot(
    *,
    within: dict[str, list],
    cross: dict[str, list],
    out_path: Path,
    hamming_thresh: int,
) -> Path:
    apply_plot_style()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1 — within-split near-duplicates
    ax = axes[0]
    splits = list(within.keys()) or ["(none)"]
    counts = [len(within.get(s, [])) for s in splits]
    palette = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}
    colors = [palette.get(s, "#999999") for s in splits]
    bars = ax.bar(splits, counts, color=colors, edgecolor="black", linewidth=0.6)
    for b, c in zip(bars, counts, strict=False):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05,
                str(c), ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("# near-duplicate pairs")
    ax.set_title(
        f"Within-split near-duplicates (Hamming ≤ {hamming_thresh})",
        fontsize=11,
    )
    ax.set_ylim(0, max(max(counts, default=0) * 1.2, 1))

    # Panel 2 — cross-split leakage
    ax = axes[1]
    pair_keys = list(cross.keys()) or ["(none)"]
    pair_counts = [len(cross.get(k, [])) for k in pair_keys]
    bar_colors = [
        "#d62728" if c > 0 else "#bbbbbb" for c in pair_counts
    ]
    bars = ax.bar(pair_keys, pair_counts, color=bar_colors,
                  edgecolor="black", linewidth=0.6)
    for b, c in zip(bars, pair_counts, strict=False):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05,
                str(c), ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("# leakage pairs")
    ax.set_title(
        f"Cross-split leakage (Hamming ≤ {hamming_thresh})",
        fontsize=11,
    )
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylim(0, max(max(pair_counts, default=0) * 1.2, 1))

    fig.suptitle(
        "Near-duplicate & cross-split leakage detection (pHash)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI — standalone smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover
    import argparse

    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-per-split", type=int, default=None)
    args = parser.parse_args()

    data_cfg = yaml.safe_load(args.data_config.read_text())
    # Resolve relative path against the config's directory.
    if "path" in data_cfg and not Path(data_cfg["path"]).is_absolute():
        data_cfg["path"] = str((args.data_config.parent / data_cfg["path"]).resolve())
    res = run(data_cfg, args.output_dir, max_samples_per_split=args.max_per_split)
    print(json.dumps({
        k: v for k, v in res["chart_metrics"]["05_duplicates_leakage"].items()
    }, indent=2))

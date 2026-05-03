"""Perceptual-hash deduplication + group-aware stratified splitting.

Pure-function module used by p00 data prep and the legacy `scripts/dedup_split.py`
wrapper. No filesystem I/O beyond `compute_phashes` (which only reads images).

Pipeline:
    1. compute_phashes  — pHash every image
    2. build_groups     — connected components at hamming ≤ thresh
    3. stratified_group_split  — assign whole groups to splits while balancing
       per-class box counts AND per-source image counts toward target ratios
    4. apply_max_per_group_eval (optional)  — cap eval-split groups
    5. verify_no_leakage — sanity-check zero cross-split near-dupes

The splitter never breaks a group across splits, eliminating near-dup leakage
by construction. Joint stratification on `class` + `source` prevents the
known failure mode where one source family lands entirely in train (e.g.
the `industrial_hazards` source in `safety-fire_detection`).
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from core.p08_evaluation.duplicates_leakage import _phash

SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# pHash + group construction
# ---------------------------------------------------------------------------


def _hash_one(p: Path) -> tuple[Path, int | None]:
    return p, _phash(p)


def compute_phashes(image_paths: list[Path], n_workers: int = 8) -> dict[Path, int]:
    """Compute 64-bit perceptual hash for every image path. Skips unreadable files."""
    if not image_paths:
        return {}
    t0 = datetime.now()
    print(f"  [dedup] hashing {len(image_paths)} images on {n_workers} workers...", flush=True)
    with Pool(n_workers) as pool:
        results = pool.map(_hash_one, image_paths, chunksize=64)
    out = {p: h for p, h in results if h is not None}
    print(
        f"  [dedup] done in {(datetime.now() - t0).total_seconds():.1f}s — {len(out)} hashes",
        flush=True,
    )
    return out


def build_groups(hashes: dict[Path, int], hamming_thresh: int) -> dict[Path, int]:
    """Path → group_id via connected components at hamming distance ≤ thresh.

    Vectorized: stack hashes into uint64 array, block-wise pairwise XOR + popcount,
    then union-find. Same algorithm as the standalone scripts/dedup_split.py.
    """
    paths = list(hashes.keys())
    if not paths:
        return {}
    arr = np.array([hashes[p] for p in paths], dtype=np.uint64)
    n = len(arr)

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    print(
        f"  [dedup] building groups (hamming ≤ {hamming_thresh}, vectorized)...",
        flush=True,
    )
    t0 = datetime.now()
    block = 512
    for i in range(0, n, block):
        chunk = arr[i:i + block]
        xor = arr[None, :] ^ chunk[:, None]
        pc = np.unpackbits(xor.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
        pc = pc.reshape(xor.shape)
        rows, cols = np.where(pc <= hamming_thresh)
        for r, c in zip(rows, cols):
            gi = i + int(r)
            gj = int(c)
            if gi < gj:
                union(gi, gj)
    root_to_gid: dict[int, int] = {}
    out: dict[Path, int] = {}
    for i, p in enumerate(paths):
        r = find(i)
        if r not in root_to_gid:
            root_to_gid[r] = len(root_to_gid)
        out[p] = root_to_gid[r]
    print(
        f"  [dedup] {len(root_to_gid)} groups from {n} images "
        f"(reduction {100 * (1 - len(root_to_gid) / n):.1f}%, "
        f"{(datetime.now() - t0).total_seconds():.1f}s)",
        flush=True,
    )
    return out


# ---------------------------------------------------------------------------
# Joint stratified group → split assignment
# ---------------------------------------------------------------------------


def stratified_group_split(
    group_to_classes: dict[int, list[int]],
    group_to_images: dict[int, int],
    group_to_source: dict[int, str] | None = None,
    target_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    stratify_by: list[str] | None = None,
    seed: int = 42,
) -> dict[int, str]:
    """Assign each group_id to {'train','val','test'} balancing the configured axes.

    When `source` is in stratify_by AND group_to_source is provided, groups are
    partitioned by source and each source is split independently — guarantees
    every source gets its proportional share of every split (the failure mode
    that motivated this code: a small source family being entirely absorbed
    into train because its image budget barely dents the overall target).

    Within each per-source pass, groups are processed largest-first; for each
    group, pick the split that minimises a weighted L1 deficit:

        cost = 10·image_overshoot_dev + 1·image_undershoot_dev + 0.3·class_deficit

    Setting `stratify_by=['class']` reproduces the legacy class-only behavior
    (one global pass, no per-source partitioning).
    """
    if stratify_by is None:
        stratify_by = ["class", "source"]
    use_class = "class" in stratify_by
    use_source = "source" in stratify_by and group_to_source is not None

    rng = random.Random(seed)

    group_box_counts: dict[int, Counter] = {
        gid: Counter(classes) for gid, classes in group_to_classes.items()
    }

    # Partition groups by source (one bucket if not using source).
    buckets: dict[str | None, list[int]] = defaultdict(list)
    if use_source:
        for gid in group_to_classes:
            buckets[group_to_source.get(gid, "unknown")].append(gid)
    else:
        buckets[None] = list(group_to_classes.keys())

    out: dict[int, str] = {}
    for _bucket_key, gids in buckets.items():
        bucket_total = sum(group_to_images[g] for g in gids)
        target_imgs = {s: r * bucket_total for s, r in zip(SPLITS, target_ratios)}

        bucket_per_class: Counter = Counter()
        for g in gids:
            bucket_per_class.update(group_box_counts[g])
        target_class = {
            s: {cls: r * cnt for cls, cnt in bucket_per_class.items()}
            for s, r in zip(SPLITS, target_ratios)
        }

        actual_imgs = {s: 0 for s in SPLITS}
        actual_class: dict[str, Counter] = {s: Counter() for s in SPLITS}

        rng.shuffle(gids)
        gids.sort(key=lambda g: group_to_images[g], reverse=True)

        for gid in gids:
            g_imgs = group_to_images[gid]
            g_box = group_box_counts[gid]

            def cost(split: str) -> float:
                new_imgs = actual_imgs[split] + g_imgs
                img_dev = (new_imgs - target_imgs[split]) / max(1.0, target_imgs[split])
                img_pen = (img_dev * 10.0) if img_dev > 0 else (-img_dev)

                class_dev = 0.0
                if use_class:
                    for cls, cnt in g_box.items():
                        tgt = target_class[split].get(cls, 0.0) or 1.0
                        cur = actual_class[split].get(cls, 0)
                        class_dev += abs((cur + cnt) - tgt) / tgt

                return img_pen + class_dev * 0.3

            chosen = min(SPLITS, key=cost)
            out[gid] = chosen
            actual_imgs[chosen] += g_imgs
            actual_class[chosen].update(g_box)

    return out


# ---------------------------------------------------------------------------
# Optional eval-split capping
# ---------------------------------------------------------------------------


def apply_max_per_group_eval(
    img_to_group: dict[Path, int],
    group_to_split: dict[int, str],
    max_per_group_eval: int,
    seed: int = 42,  # noqa: ARG001 - kept for API symmetry with stratified_group_split
) -> dict[Path, str | None]:
    """Cap eval-split groups to ≤N evenly-strided members. Train splits never capped.

    Excess images map to None (caller drops them entirely — no new leakage).
    """
    eval_splits = {"val", "test"}
    group_eval_imgs: dict[int, list[Path]] = defaultdict(list)
    for img, gid in img_to_group.items():
        if group_to_split[gid] in eval_splits:
            group_eval_imgs[gid].append(img)

    keep: set[Path] = set()
    for _gid, imgs in group_eval_imgs.items():
        imgs_sorted = sorted(imgs)
        n = len(imgs_sorted)
        if n <= max_per_group_eval:
            keep.update(imgs_sorted)
        else:
            stride = max(1, n // max_per_group_eval)
            selected = sorted(imgs_sorted[i] for i in range(0, n, stride))[:max_per_group_eval]
            keep.update(selected)

    out: dict[Path, str | None] = {}
    for img, gid in img_to_group.items():
        assigned = group_to_split[gid]
        if assigned in eval_splits:
            out[img] = assigned if img in keep else None
        else:
            out[img] = assigned

    dropped = sum(1 for s in out.values() if s is None)
    print(
        f"  [dedup] max_per_group_eval={max_per_group_eval}: "
        f"dropped {dropped} eval near-duplicates (train unchanged)",
        flush=True,
    )
    return out


# ---------------------------------------------------------------------------
# Leakage verification
# ---------------------------------------------------------------------------


def verify_no_leakage(
    img_to_hash: dict[Path, int],
    img_to_split: dict[Path, str],
    hamming_thresh: int,
) -> int:
    """Return count of cross-split pairs at hamming ≤ thresh. 0 = clean."""
    paths = [p for p in img_to_hash if p in img_to_split]
    if not paths:
        return 0
    arr = np.array([img_to_hash[p] for p in paths], dtype=np.uint64)
    splits = np.array([img_to_split[p] for p in paths])
    leaks = 0
    block = 512
    n = len(arr)
    for i in range(0, n, block):
        chunk = arr[i:i + block]
        xor = arr[None, :] ^ chunk[:, None]
        pc = np.unpackbits(xor.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
        pc = pc.reshape(xor.shape)
        for r in range(chunk.shape[0]):
            gi = i + r
            row = pc[r]
            for j in range(gi + 1, n):
                if row[j] <= hamming_thresh and splits[gi] != splits[j]:
                    leaks += 1
    return leaks


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


_DEDUP_DEFAULTS = {
    "enabled": True,
    "hamming_thresh": 3,
    "max_per_group_eval": None,
    "stratify_by": ["class", "source"],
    "source_from": "adapter",
    "verify_no_leakage": True,
}

_VALID_SOURCE_FROM = {"adapter", "filename_prefix"}
_VALID_STRATIFY = {"class", "source"}


def validate_dedup_config(config: dict) -> dict:
    """Fill defaults + hard-error on unknown keys / invalid values. Returns merged dict."""
    if not isinstance(config, dict):
        raise ValueError(f"dedup config must be a dict, got {type(config).__name__}")

    unknown = set(config) - set(_DEDUP_DEFAULTS)
    if unknown:
        raise ValueError(
            f"dedup config: unknown keys {sorted(unknown)}; "
            f"valid keys are {sorted(_DEDUP_DEFAULTS)}"
        )

    merged = {**_DEDUP_DEFAULTS, **config}

    if not isinstance(merged["enabled"], bool):
        raise ValueError("dedup.enabled must be bool")

    if not isinstance(merged["hamming_thresh"], int) or not 0 <= merged["hamming_thresh"] <= 16:
        raise ValueError("dedup.hamming_thresh must be int in [0, 16]")

    mpge = merged["max_per_group_eval"]
    if mpge is not None and (not isinstance(mpge, int) or mpge <= 0):
        raise ValueError("dedup.max_per_group_eval must be null or positive int")

    if not isinstance(merged["stratify_by"], list) or not all(
        s in _VALID_STRATIFY for s in merged["stratify_by"]
    ):
        raise ValueError(
            f"dedup.stratify_by must be a list with values in {sorted(_VALID_STRATIFY)}"
        )

    if merged["source_from"] not in _VALID_SOURCE_FROM:
        raise ValueError(
            f"dedup.source_from must be one of {sorted(_VALID_SOURCE_FROM)}, "
            f"got '{merged['source_from']}'"
        )

    if not isinstance(merged["verify_no_leakage"], bool):
        raise ValueError("dedup.verify_no_leakage must be bool")

    return merged

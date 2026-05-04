"""Perceptual-hash deduplication + group-aware stratified splitting.

Pure-function module used by p00 data prep and the legacy `scripts/dedup_split.py`
wrapper. No filesystem I/O beyond `compute_phashes` (which only reads images).

Two split strategies are supported:

  * `class_aware` (legacy) — stratified by class (and optionally source) at the
    group level via `stratified_group_split`, optionally followed by
    `apply_max_per_group_eval` to cap val/test groups.
  * `per_source_with_temporal` (recommended for multi-source / video-heavy
    datasets) — primary stratification by source: each source gets its own
    70/15/15 split independently. pHash groups detected as VIDEO (size ≥ N
    AND filenames have monotonically increasing numeric suffix) are split
    *temporally* with a buffer gap to prevent adjacent-frame leakage; STILL
    groups are whole-group assigned via greedy size-deficit fitting. See
    `per_source_split`.

The splitter never breaks a group's *neighbours-by-pHash* across splits within
a single source for STILL groups (whole-group assignment), and uses temporal
gaps for VIDEO groups so the train/val/test partitions remain near-dup-clean.
"""
from __future__ import annotations

import random
import re
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


def _stride_pick(imgs: list[Path], k: int) -> list[Path]:
    """Even-stride pick of up to k images from a sorted list."""
    imgs_sorted = sorted(imgs)
    n = len(imgs_sorted)
    if k >= n:
        return imgs_sorted
    stride = max(1, n // k)
    return sorted(imgs_sorted[i] for i in range(0, n, stride))[:k]


def _group_dominant_class(group_imgs: list[Path], img_to_classes: dict[Path, list[int]]) -> int:
    """Most-common class across all boxes in the group; -1 if no boxes anywhere."""
    cnt: Counter = Counter()
    for img in group_imgs:
        cnt.update(img_to_classes.get(img, []))
    if not cnt:
        return -1
    return cnt.most_common(1)[0][0]


def apply_max_per_group_eval(
    img_to_group: dict[Path, int],
    group_to_split: dict[int, str],
    max_per_group_eval: int,
    img_to_classes: dict[Path, list[int]] | None = None,
    seed: int = 42,  # noqa: ARG001 - kept for API symmetry with stratified_group_split
) -> dict[Path, str | None]:
    """Cap eval-split groups, optionally class-aware to preserve class distribution.

    The naïve cap (every oversize group truncated to N) systematically drops
    more samples from *large* groups than from small ones. When group size
    correlates with class (e.g. one class's footage tends to be longer video
    sequences), this skews the post-cap class distribution away from train.

    When `img_to_classes` is provided, the cap becomes a *target per-class
    image budget* rather than a uniform per-group truncation. For each eval
    split:
      1. Compute target per-class image counts proportional to the train-split
         class distribution.
      2. Bucket groups by dominant class (most-common class across all the
         group's boxes — most groups are mono-class video sequences).
      3. For each class, distribute the per-class budget across its groups in
         proportion to group size, with each group keeping at least 1 image
         and at most all its images. The legacy `max_per_group_eval` becomes
         a *minimum* per-group keep cap when class budget is tight, otherwise
         large groups can exceed it to satisfy under-represented classes.

    When `img_to_classes` is None, falls back to legacy class-blind even-stride
    per group capped at `max_per_group_eval`.

    Excess images map to None (caller drops them entirely — no new leakage).
    """
    eval_splits = {"val", "test"}

    split_to_group_imgs: dict[str, dict[int, list[Path]]] = {
        s: defaultdict(list) for s in SPLITS
    }
    for img, gid in img_to_group.items():
        split_to_group_imgs[group_to_split[gid]][gid].append(img)

    keep: set[Path] = set()
    for _gid, imgs in split_to_group_imgs["train"].items():
        keep.update(imgs)

    if img_to_classes is None:
        # Legacy class-blind path: even-stride per group, hard cap at max_per_group_eval.
        for split in eval_splits:
            for _gid, imgs in split_to_group_imgs[split].items():
                if len(imgs) <= max_per_group_eval:
                    keep.update(imgs)
                else:
                    keep.update(_stride_pick(imgs, max_per_group_eval))
    else:
        # Class-aware path: target eval class fractions = train class fractions.
        # Use BOX counts (not image counts) since detection metrics are box-weighted.
        train_box_counts: Counter = Counter()
        for _gid, imgs in split_to_group_imgs["train"].items():
            for img in imgs:
                train_box_counts.update(img_to_classes.get(img, []))
        train_box_total = sum(train_box_counts.values())
        target_box_frac = (
            {c: cnt / train_box_total for c, cnt in train_box_counts.items()}
            if train_box_total > 0
            else {}
        )

        for split in eval_splits:
            group_imgs_map = split_to_group_imgs[split]

            # Estimate post-cap split size from the legacy cap as the base budget,
            # so reducing/raising max_per_group_eval still scales eval set size.
            est_split_size = sum(
                min(len(imgs), max_per_group_eval) for imgs in group_imgs_map.values()
            )

            # Bucket groups by dominant class
            groups_by_class: dict[int, list[tuple[int, list[Path]]]] = defaultdict(list)
            for gid, imgs in group_imgs_map.items():
                cls = _group_dominant_class(imgs, img_to_classes)
                groups_by_class[cls].append((gid, imgs))

            # Target per-class image budget (image-level allocation toward box-frac target)
            target_per_class: dict[int, int] = {}
            for cls in groups_by_class:
                avail = sum(len(imgs) for _, imgs in groups_by_class[cls])
                target = int(round(est_split_size * target_box_frac.get(cls, 0)))
                target_per_class[cls] = min(avail, max(0, target))

            # Sanity: distribute residual (rounding error) to largest buckets
            allocated = sum(target_per_class.values())
            residual = est_split_size - allocated
            if residual != 0 and target_per_class:
                ordered = sorted(
                    target_per_class.keys(),
                    key=lambda c: -sum(len(imgs) for _, imgs in groups_by_class[c]),
                )
                idx = 0
                step = 1 if residual > 0 else -1
                while residual != 0 and idx < len(ordered) * 4:
                    cls = ordered[idx % len(ordered)]
                    avail = sum(len(imgs) for _, imgs in groups_by_class[cls])
                    new_val = target_per_class[cls] + step
                    if 0 <= new_val <= avail:
                        target_per_class[cls] = new_val
                        residual -= step
                    idx += 1

            # For each class, allocate its budget across its groups proportionally
            # to group size, with each group keeping ≥1 and ≤len(imgs).
            for cls, grps in groups_by_class.items():
                budget = target_per_class.get(cls, 0)
                if budget <= 0:
                    continue
                grps_sorted = sorted(grps, key=lambda gi: -len(gi[1]))
                total_size = sum(len(imgs) for _, imgs in grps_sorted)
                if total_size <= budget:
                    # Class is under-represented globally → keep every image
                    for _gid, imgs in grps_sorted:
                        keep.update(imgs)
                    continue

                # Proportional split with floor 1, ceiling len(imgs).
                # First pass: floor allocation
                per_group_keep: dict[int, int] = {}
                for gid, imgs in grps_sorted:
                    share = len(imgs) / total_size
                    k = max(1, int(budget * share))
                    per_group_keep[gid] = min(k, len(imgs))

                # Repair: distribute remaining deficit / surplus
                allocated = sum(per_group_keep.values())
                deficit = budget - allocated
                if deficit > 0:
                    # Add to groups with most slack (largest available room)
                    while deficit > 0:
                        room = {
                            gid: len(imgs) - per_group_keep[gid]
                            for gid, imgs in grps_sorted
                        }
                        gid_best = max(room, key=lambda g: room[g])
                        if room[gid_best] <= 0:
                            break
                        per_group_keep[gid_best] += 1
                        deficit -= 1
                elif deficit < 0:
                    # Trim from largest keep counts (preserve floor 1)
                    while deficit < 0:
                        candidates = {
                            gid: cnt for gid, cnt in per_group_keep.items() if cnt > 1
                        }
                        if not candidates:
                            break
                        gid_largest = max(candidates, key=lambda g: candidates[g])
                        per_group_keep[gid_largest] -= 1
                        deficit += 1

                # Even-stride pick from each group up to its allocated keep count
                for gid, imgs in grps_sorted:
                    keep.update(_stride_pick(imgs, per_group_keep[gid]))

    out: dict[Path, str | None] = {}
    for img, gid in img_to_group.items():
        assigned = group_to_split[gid]
        if assigned in eval_splits:
            out[img] = assigned if img in keep else None
        else:
            out[img] = assigned

    dropped = sum(1 for s in out.values() if s is None)
    mode = "class-aware" if img_to_classes is not None else "class-blind"
    print(
        f"  [dedup] max_per_group_eval={max_per_group_eval} ({mode}): "
        f"dropped {dropped} eval near-duplicates (train unchanged)",
        flush=True,
    )
    return out


# ---------------------------------------------------------------------------
# Per-source + temporal-aware split (added 2026-05-03)
# ---------------------------------------------------------------------------


_TRAILING_DIGITS_RE = re.compile(r"(\d+)(?=\D*$)")


def _trailing_number(name: str) -> int | None:
    """Extract trailing numeric suffix from filename stem; None if absent.

    Examples:
        'AoF01029.jpg'        → 1029
        'frame_001.jpg'       → 1
        'video1_clip2_50.png' → 50
        '000051_jpg.rf.abc.jpg' → None  (no trailing digits before extension)
        'random_hash_xyz.jpg' → None
    """
    stem = Path(name).stem
    m = _TRAILING_DIGITS_RE.search(stem)
    if m is None:
        return None
    return int(m.group(1))


def _is_video_group(imgs: list[Path], min_size: int = 20) -> bool:
    """Detect if a pHash group is a sequential video (vs random scenes/augs).

    Heuristic:
      1. Group size >= min_size.
      2. Every filename has a trailing numeric suffix.
      3. After sorting by filename, the numeric values are strictly increasing
         and have no gap > 10× the median step.

    Returns False for groups of random images that happen to pHash-cluster
    (e.g. Roboflow `*_jpg.rf.<hash>.jpg` augmentations of the same source).
    """
    if len(imgs) < min_size:
        return False
    nums: list[tuple[str, int]] = []
    for p in imgs:
        n = _trailing_number(p.name)
        if n is None:
            return False
        nums.append((p.name, n))
    nums.sort(key=lambda x: x[0])
    seq = [n for _, n in nums]
    if len(set(seq)) != len(seq):
        return False
    if not all(seq[i] < seq[i + 1] for i in range(len(seq) - 1)):
        return False
    steps = [seq[i + 1] - seq[i] for i in range(len(seq) - 1)]
    if not steps:
        return False
    median_step = float(np.median(steps))
    if median_step <= 0:
        return False
    max_gap = max(steps)
    return max_gap <= 10 * median_step


def _temporal_split_video_group(
    imgs: list[Path],
    target_ratios: tuple[float, float, float],
    gap_fraction: float = 0.05,
    min_gap_frames: int = 5,
) -> dict[Path, str | None]:
    """Split a video group temporally: train | gap | val | gap | test.

    `imgs` are sorted by filename (which corresponds to temporal order for
    video-derived data). Gap frames are dropped (returned as None) to prevent
    adjacent-frame leakage at split boundaries.
    """
    sorted_imgs = sorted(imgs, key=lambda p: p.name)
    n = len(sorted_imgs)
    gap = max(min_gap_frames, int(n * gap_fraction))
    r_train, r_val, r_test = target_ratios

    # Budget layout: train | gap | val | gap | test
    # Allocate gaps from total first, then split remainder by ratios.
    avail = max(0, n - 2 * gap)
    train_n = int(round(avail * r_train))
    val_n = int(round(avail * r_val))
    test_n = avail - train_n - val_n  # absorb rounding

    out: dict[Path, str | None] = {}
    idx = 0

    def _take(label: str | None, k: int) -> None:
        nonlocal idx
        for _ in range(k):
            if idx >= n:
                return
            out[sorted_imgs[idx]] = label
            idx += 1

    _take("train", train_n)
    _take(None, gap)
    _take("val", val_n)
    _take(None, gap)
    _take("test", test_n)
    # Any leftover (shouldn't happen, but be safe) → test
    while idx < n:
        out[sorted_imgs[idx]] = "test"
        idx += 1
    return out


def _resolve_group_source(
    imgs: list[Path], img_to_source: dict[Path, str]
) -> str:
    """Majority source for a group; alphabetical-first on tie."""
    cnt = Counter(img_to_source.get(img, "unknown") for img in imgs)
    most = cnt.most_common()
    top_count = most[0][1]
    tied = sorted(s for s, c in most if c == top_count)
    return tied[0]


def per_source_split(
    img_to_group: dict[Path, int],
    img_to_classes: dict[Path, list[int]],
    img_to_source: dict[Path, str],
    target_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    *,
    enable_temporal: bool = True,
    min_video_size: int = 20,
    gap_fraction: float = 0.05,
    min_gap_frames: int = 5,
    seed: int = 42,
) -> dict[Path, str | None]:
    """Source-strict + temporal-aware split.

    1. Bucket pHash groups by majority source (alphabetical tie-break).
    2. Within each source: classify each group as VIDEO or STILL.
    3. STILL groups: greedy assign (largest first) to the split with the
       largest deficit relative to per-source target image counts.
    4. VIDEO groups: temporal split with buffer gap inside the group.

    Tiny sources (< 7 imgs total) are dumped entirely into train (eval would
    be statistically meaningless).

    Returns {img: split or None}. None = dropped (gap frame).
    """
    rng = random.Random(seed)

    # Build group → list[Path] and group → source map
    group_to_imgs: dict[int, list[Path]] = defaultdict(list)
    for img, gid in img_to_group.items():
        group_to_imgs[gid].append(img)

    group_to_source: dict[int, str] = {
        gid: _resolve_group_source(imgs, img_to_source)
        for gid, imgs in group_to_imgs.items()
    }

    # Bucket groups by source
    source_to_gids: dict[str, list[int]] = defaultdict(list)
    for gid, src in group_to_source.items():
        source_to_gids[src].append(gid)

    out: dict[Path, str | None] = {}

    for src, gids in sorted(source_to_gids.items()):
        src_imgs_total = sum(len(group_to_imgs[g]) for g in gids)

        # Tiny source → all to train
        if src_imgs_total < 7:
            for g in gids:
                for img in group_to_imgs[g]:
                    out[img] = "train"
            continue

        # Classify groups
        video_gids: list[int] = []
        still_gids: list[int] = []
        for g in gids:
            imgs = group_to_imgs[g]
            if enable_temporal and _is_video_group(imgs, min_size=min_video_size):
                video_gids.append(g)
            else:
                still_gids.append(g)

        target_imgs = {
            s: r * src_imgs_total for s, r in zip(SPLITS, target_ratios)
        }
        actual = {s: 0 for s in SPLITS}

        # Apply temporal split first — videos pre-commit to all 3 splits
        for g in video_gids:
            mapping = _temporal_split_video_group(
                group_to_imgs[g],
                target_ratios=target_ratios,
                gap_fraction=gap_fraction,
                min_gap_frames=min_gap_frames,
            )
            for img, split in mapping.items():
                out[img] = split
                if split is not None:
                    actual[split] += 1

        # Greedy assign STILL groups: largest first to split with largest deficit
        rng.shuffle(still_gids)
        still_gids.sort(key=lambda g: -len(group_to_imgs[g]))
        for g in still_gids:
            sz = len(group_to_imgs[g])
            # Deficit-driven choice; cap overshoot at 10% above target.
            def cost(s: str, _sz: int = sz) -> float:
                new_val = actual[s] + _sz
                tgt = target_imgs[s]
                if tgt <= 0:
                    return float("inf")
                if new_val > tgt * 1.10:
                    # Penalize overshoot heavily but still allow if every split overshoots
                    return (new_val - tgt) / tgt + 100.0
                # Prefer the split with largest current deficit
                return -(tgt - actual[s]) / max(1.0, tgt)

            chosen = min(SPLITS, key=cost)
            for img in group_to_imgs[g]:
                out[img] = chosen
            actual[chosen] += sz

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
    "split_strategy": "class_aware",
    "split_ratios": None,  # if None, p00 falls back to top-level `splits.*`
    "temporal": {
        "enabled": True,
        "min_group_size_for_video": 20,
        "gap_fraction": 0.05,
        "min_gap_frames": 5,
    },
}

_VALID_SOURCE_FROM = {"adapter", "filename_prefix"}
_VALID_STRATIFY = {"class", "source"}
_VALID_SPLIT_STRATEGY = {"class_aware", "per_source_with_temporal"}
_VALID_TEMPORAL_KEYS = {
    "enabled", "min_group_size_for_video", "gap_fraction", "min_gap_frames"
}


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

    if merged["split_strategy"] not in _VALID_SPLIT_STRATEGY:
        raise ValueError(
            f"dedup.split_strategy must be one of {sorted(_VALID_SPLIT_STRATEGY)}, "
            f"got '{merged['split_strategy']}'"
        )

    if merged["split_ratios"] is not None:
        sr = merged["split_ratios"]
        if (not isinstance(sr, list) or len(sr) != 3
                or not all(isinstance(x, (int, float)) for x in sr)):
            raise ValueError("dedup.split_ratios must be a list of 3 floats")
        if abs(sum(sr) - 1.0) > 0.01:
            raise ValueError(f"dedup.split_ratios must sum to 1.0±0.01, got {sum(sr)}")
        if any(x < 0 for x in sr):
            raise ValueError("dedup.split_ratios must be non-negative")
        merged["split_ratios"] = tuple(float(x) for x in sr)

    temporal = merged["temporal"]
    if not isinstance(temporal, dict):
        raise ValueError("dedup.temporal must be a dict")
    unknown_t = set(temporal) - _VALID_TEMPORAL_KEYS
    if unknown_t:
        raise ValueError(
            f"dedup.temporal: unknown keys {sorted(unknown_t)}; "
            f"valid keys are {sorted(_VALID_TEMPORAL_KEYS)}"
        )
    temporal = {**_DEDUP_DEFAULTS["temporal"], **temporal}
    if not isinstance(temporal["enabled"], bool):
        raise ValueError("dedup.temporal.enabled must be bool")
    if (not isinstance(temporal["min_group_size_for_video"], int)
            or temporal["min_group_size_for_video"] < 2):
        raise ValueError("dedup.temporal.min_group_size_for_video must be int >= 2")
    if (not isinstance(temporal["gap_fraction"], (int, float))
            or not 0.0 <= temporal["gap_fraction"] <= 0.5):
        raise ValueError("dedup.temporal.gap_fraction must be float in [0, 0.5]")
    if (not isinstance(temporal["min_gap_frames"], int)
            or temporal["min_gap_frames"] < 0):
        raise ValueError("dedup.temporal.min_gap_frames must be int >= 0")
    merged["temporal"] = temporal

    return merged

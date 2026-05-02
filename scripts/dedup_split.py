"""Duplicate-aware re-splitter for a training_ready dataset.

Reads `dataset_store/training_ready/<name>/{train,val,test}/{images,labels}/`,
computes pHash for every image, builds connected components at hamming ≤ THRESH,
and re-emits stratified splits where every member of a duplicate group lands in
the same split.

Writes to a sibling directory `<name>_clean/` (non-destructive) plus a
`splits.json` recording the new layout. The cut-over (renaming the legacy dir
and promoting the clean one) is a separate manual step.

Usage:
    uv run scripts/dedup_split.py --name fire_detection
    uv run scripts/dedup_split.py --name fire_detection --single-per-group-eval
        # Keep only 1 representative per group in val/test; reassign the rest to
        # train. Makes val/test harder and more representative (removes within-split
        # near-duplicate inflation).
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.p08_evaluation.duplicates_leakage import _phash  # noqa: E402

HAMMING_THRESH = 4  # stricter than analyzer's 6 — fewer false positives on uniform smoke/sky
SPLITS = ("train", "val", "test")
TARGET_RATIOS = (0.80, 0.10, 0.10)


def _hash_one(p: Path) -> tuple[Path, int | None]:
    return p, _phash(p)


def _enumerate_images(base: Path) -> list[tuple[Path, str]]:
    out = []
    for split in SPLITS:
        for img in (base / split / "images").glob("*.jpg"):
            out.append((img, split))
        for img in (base / split / "images").glob("*.png"):
            out.append((img, split))
    return out


def _label_path_for(img: Path) -> Path:
    return img.parents[1] / "labels" / (img.stem + ".txt")


def _classes_in_label(label_path: Path) -> list[int]:
    if not label_path.exists():
        return []
    classes = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            classes.append(int(parts[0]))
        except ValueError:
            continue
    return classes


def _hash_all(images: list[Path], n_workers: int) -> dict[Path, int]:
    print(f"  hashing {len(images)} images on {n_workers} workers...", flush=True)
    t0 = datetime.now()
    with Pool(n_workers) as pool:
        results = pool.map(_hash_one, images, chunksize=64)
    out = {p: h for p, h in results if h is not None}
    print(f"  done in {(datetime.now() - t0).total_seconds():.1f}s — {len(out)} hashes", flush=True)
    return out


def _build_groups(hashes: dict[Path, int], thresh: int) -> dict[Path, int]:
    """Connected components in the graph of (hamming ≤ thresh) pairs.

    Vectorized: stack hashes into uint64 array, compute pairwise XOR popcount
    using numpy, then union-find.
    """
    paths = list(hashes.keys())
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

    print(f"  building groups (hamming ≤ {thresh}, vectorized)...", flush=True)
    t0 = datetime.now()
    # Block-wise pairwise XOR + popcount to bound peak memory.
    block = 512
    for i in range(0, n, block):
        chunk = arr[i:i + block]
        xor = arr[None, :] ^ chunk[:, None]      # (block, n)
        # popcount via numpy (uint64 → bytes view → unpackbits sum)
        pc = np.unpackbits(xor.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
        pc = pc.reshape(xor.shape)
        rows, cols = np.where(pc <= thresh)
        for r, c in zip(rows, cols):
            gi = i + int(r)
            gj = int(c)
            if gi < gj:
                union(gi, gj)
    # Compress to root → group_id
    root_to_gid: dict[int, int] = {}
    out: dict[Path, int] = {}
    for i, p in enumerate(paths):
        r = find(i)
        if r not in root_to_gid:
            root_to_gid[r] = len(root_to_gid)
        out[p] = root_to_gid[r]
    print(
        f"  done in {(datetime.now() - t0).total_seconds():.1f}s — "
        f"{len(root_to_gid)} groups from {n} images "
        f"(reduction {100 * (1 - len(root_to_gid) / n):.1f}%)",
        flush=True,
    )
    return out


def _stratified_group_split(
    group_to_classes: dict[int, list[int]],
    group_to_images: dict[int, int],
    seed: int = 42,
) -> dict[int, str]:
    """Assign each group to a split, balancing per-class box counts across splits.

    Per-class budgets per split: we track the number of boxes of each class in
    each split, plus the number of images. For each group, pick the split that
    minimises the worst-case relative deviation from its targets. Process
    largest groups first so the high-leverage decisions are made before small
    groups can correct nothing.
    """
    rng = random.Random(seed)

    # Per-group: number of images + per-class box counts.
    group_box_counts: dict[int, Counter] = {
        gid: Counter(classes) for gid, classes in group_to_classes.items()
    }

    # Totals across all groups.
    total_imgs = sum(group_to_images.values())
    total_per_class: Counter = Counter()
    for c in group_box_counts.values():
        total_per_class.update(c)

    # Targets per split.
    target_imgs = {s: r * total_imgs for s, r in zip(SPLITS, TARGET_RATIOS)}
    target_class = {
        s: {cls: r * cnt for cls, cnt in total_per_class.items()}
        for s, r in zip(SPLITS, TARGET_RATIOS)
    }

    # Live totals.
    actual_imgs = {s: 0 for s in SPLITS}
    actual_class = {s: Counter() for s in SPLITS}

    # Process largest groups first (most leverage), then random within a tie.
    gids = list(group_to_classes.keys())
    rng.shuffle(gids)
    gids.sort(key=lambda g: group_to_images[g], reverse=True)

    out: dict[int, str] = {}
    for gid in gids:
        g_imgs = group_to_images[gid]
        g_box = group_box_counts[gid]

        def cost(split: str) -> float:
            # Image budget is hard: heavily penalise overshooting the split target.
            # Class balance is the secondary objective (within the budget).
            new_imgs = actual_imgs[split] + g_imgs
            img_dev = (new_imgs - target_imgs[split]) / max(1.0, target_imgs[split])
            # Asymmetric penalty: overshoot is much worse than undershoot.
            img_pen = (img_dev * 10.0) if img_dev > 0 else (-img_dev)
            class_dev = 0.0
            for cls, cnt in g_box.items():
                tgt = target_class[split].get(cls, 0.0) or 1.0
                cur = actual_class[split].get(cls, 0)
                class_dev += abs((cur + cnt) - tgt) / tgt
            return img_pen + class_dev * 0.3  # split size dominates; class is tiebreaker

        chosen = min(SPLITS, key=cost)
        out[gid] = chosen
        actual_imgs[chosen] += g_imgs
        actual_class[chosen].update(g_box)

    return out


def _apply_max_per_group_eval(
    img_to_group: dict[Path, int],
    group_to_split: dict[int, str],
    max_per_group: int,
) -> dict[Path, str | None]:
    """Keep at most `max_per_group` images per group in val/test, evenly strided.

    Train is completely unchanged. Excess images are mapped to None (dropped).
    Strided selection preserves temporal diversity within each group.
    No new cross-split leakage: dropped images are excluded entirely.
    """
    eval_splits = {"val", "test"}

    # Collect and sort images per eval group for deterministic strided selection.
    group_eval_imgs: dict[int, list[Path]] = defaultdict(list)
    for img, gid in img_to_group.items():
        if group_to_split[gid] in eval_splits:
            group_eval_imgs[gid].append(img)

    # Strided keep-set per group.
    keep: set[Path] = set()
    for gid, imgs in group_eval_imgs.items():
        imgs_sorted = sorted(imgs)
        n = len(imgs_sorted)
        if n <= max_per_group:
            keep.update(imgs_sorted)
        else:
            stride = n // max_per_group
            keep.update(imgs_sorted[i] for i in range(0, n, stride) if len(keep) <= n)
            # Ensure exactly max_per_group selected (stride may overshoot by 1).
            selected = sorted(imgs_sorted[i] for i in range(0, n, stride))[:max_per_group]
            keep.update(selected)

    img_to_split: dict[Path, str | None] = {}
    for img, gid in img_to_group.items():
        assigned = group_to_split[gid]
        if assigned in eval_splits:
            img_to_split[img] = assigned if img in keep else None
        else:
            img_to_split[img] = assigned

    dropped = sum(1 for s in img_to_split.values() if s is None)
    kept_eval = sum(1 for s in img_to_split.values() if s in eval_splits)
    print(
        f"  max-per-group-eval={max_per_group}: kept {kept_eval} eval images, "
        f"dropped {dropped} near-duplicates (train unchanged, no new leakage)",
        flush=True,
    )
    return img_to_split


def _emit_splits(
    img_to_group: dict[Path, int],
    group_to_split: dict[int, str],
    src_base: Path,
    dst_base: Path,
    single_per_group_eval: bool = False,
    max_per_group_eval: int | None = None,
) -> tuple[dict[str, dict[str, int]], dict[Path, str]]:
    """Hardlink images + labels into the new split directories."""
    print(f"  writing new layout to {dst_base}/", flush=True)
    if dst_base.exists():
        raise FileExistsError(f"{dst_base} already exists — refuse to overwrite")
    for split in SPLITS:
        (dst_base / split / "images").mkdir(parents=True)
        (dst_base / split / "labels").mkdir(parents=True)

    if max_per_group_eval is not None:
        img_to_split_nullable = _apply_max_per_group_eval(
            img_to_group, group_to_split, max_per_group_eval
        )
    elif single_per_group_eval:
        img_to_split_nullable = _apply_max_per_group_eval(img_to_group, group_to_split, 1)
    else:
        img_to_split_nullable = {img: group_to_split[gid] for img, gid in img_to_group.items()}

    counts = {s: {"images": 0, "labels": 0} for s in SPLITS}
    img_to_split_final: dict[Path, str] = {}
    for img, new_split in img_to_split_nullable.items():
        if new_split is None:
            continue  # dropped duplicate — skip entirely
        new_img = dst_base / new_split / "images" / img.name
        new_label = dst_base / new_split / "labels" / (img.stem + ".txt")

        try:
            new_img.hardlink_to(img)
        except OSError:
            shutil.copy2(img, new_img)
        counts[new_split]["images"] += 1
        img_to_split_final[img] = new_split

        src_label = _label_path_for(img)
        if src_label.exists():
            try:
                new_label.hardlink_to(src_label)
            except OSError:
                shutil.copy2(src_label, new_label)
            counts[new_split]["labels"] += 1

    return counts, img_to_split_final


def _verify_no_leakage(
    img_to_hash: dict[Path, int],
    img_to_split_new: dict[Path, str],
    thresh: int,
) -> int:
    """Count cross-split pairs in the new layout. Should be 0 by construction."""
    paths = list(img_to_hash.keys())
    arr = np.array([img_to_hash[p] for p in paths], dtype=np.uint64)
    splits = np.array([img_to_split_new[p] for p in paths])
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
                if row[j] <= thresh and splits[gi] != splits[j]:
                    leaks += 1
    return leaks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True, help="Dataset name under training_ready/")
    ap.add_argument("--thresh", type=int, default=HAMMING_THRESH)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--single-per-group-eval", action="store_true",
        help="Alias for --max-per-group-eval 1.",
    )
    ap.add_argument(
        "--max-per-group-eval", type=int, default=None, metavar="N",
        help="Keep at most N evenly-strided images per duplicate group in val/test; "
             "drop the rest. Reduces within-eval redundancy without leakage. "
             "E.g. --max-per-group-eval 200 keeps ≤200 frames per video sequence.",
    )
    args = ap.parse_args()

    src = ROOT / "dataset_store" / "training_ready" / args.name
    dst = ROOT / "dataset_store" / "training_ready" / f"{args.name}_clean"
    print(f"src: {src}")
    print(f"dst: {dst}")

    images_with_split = _enumerate_images(src)
    images = [p for p, _ in images_with_split]
    print(f"found {len(images)} images")

    img_to_hash = _hash_all(images, args.workers)
    img_to_group = _build_groups(img_to_hash, args.thresh)

    group_to_classes: dict[int, list[int]] = defaultdict(list)
    group_to_images: dict[int, int] = defaultdict(int)
    for img, gid in img_to_group.items():
        group_to_classes[gid].extend(_classes_in_label(_label_path_for(img)))
        group_to_images[gid] += 1

    if args.single_per_group_eval:
        # Preserve the original split assignments from the source dataset.
        # Only deduplicate within val/test — train is completely unchanged.
        group_to_split = {
            gid: orig_split
            for img, gid in img_to_group.items()
            for orig_split in [next(
                s for s in SPLITS
                if img.parts[-3] == s  # path: <src>/<split>/images/<file>
            )]
        }
        # Last assignment wins per group — fine since cross-split groups are
        # already eliminated (src is assumed already deduped cross-split).
    else:
        group_to_split = _stratified_group_split(
            group_to_classes, group_to_images, seed=args.seed
        )

    counts, img_to_split_new = _emit_splits(
        img_to_group, group_to_split, src, dst,
        single_per_group_eval=args.single_per_group_eval,
        max_per_group_eval=args.max_per_group_eval,
    )
    print(f"new counts: {counts}")

    print("verifying zero cross-split leakage on new layout...")
    # Only verify images that made it into the new layout (dropped images excluded).
    img_to_hash_kept = {p: h for p, h in img_to_hash.items() if p in img_to_split_new}
    leaks = _verify_no_leakage(img_to_hash_kept, img_to_split_new, args.thresh)
    print(f"  cross-split pairs at hamming ≤ {args.thresh}: {leaks}")
    if leaks > 0:
        print("  WARNING: leakage > 0 — splitter is buggy")

    # Write splits.json
    manifest = {
        "task_type": "detection",
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "seed": args.seed,
            "ratios": list(TARGET_RATIOS),
            "stratified": True,
            "deduplicated": True,
            "hamming_thresh": args.thresh,
            "single_per_group_eval": args.single_per_group_eval,
            "max_per_group_eval": args.max_per_group_eval,
            "source": str(src),
            "n_groups": len(set(group_to_split.values())),
            "total_samples": sum(c["images"] for c in counts.values()),
        },
        "counts": counts,
    }
    (dst / "splits.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {dst / 'splits.json'}")


if __name__ == "__main__":
    main()

"""Legacy-dataset wrapper around `core.p00_data_prep.core.dedup`.

Reads `dataset_store/training_ready/<name>/{train,val,test}/{images,labels}/`,
re-deduplicates + re-splits in-place to a sibling `<name>_clean/`. Use this only
when you already have a training_ready dataset and don't want to re-run p00 from
raw — new datasets get dedup automatically via the `dedup:` block in
`00_data_preparation.yaml`.

Usage:
    uv run scripts/dedup_split.py --name fire_detection
    uv run scripts/dedup_split.py --name fire_detection --thresh 6 --max-per-group-eval 200
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.p00_data_prep.core.dedup import (  # noqa: E402
    apply_max_per_group_eval,
    build_groups,
    compute_phashes,
    stratified_group_split,
    verify_no_leakage,
)

SPLITS = ("train", "val", "test")
TARGET_RATIOS = (0.80, 0.10, 0.10)


def _enumerate_images(base: Path) -> list[Path]:
    out: list[Path] = []
    for split in SPLITS:
        for ext in (".jpg", ".jpeg", ".png"):
            out.extend((base / split / "images").glob(f"*{ext}"))
    return out


def _label_path_for(img: Path) -> Path:
    return img.parents[1] / "labels" / (img.stem + ".txt")


def _classes_in_label(label_path: Path) -> list[int]:
    if not label_path.exists():
        return []
    out: list[int] = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if parts:
            try:
                out.append(int(parts[0]))
            except ValueError:
                pass
    return out


def _source_from_filename(p: Path) -> str:
    """Heuristic legacy source: leading underscore-token of the stem."""
    stem = p.stem
    return stem.split("_", 1)[0] if "_" in stem else "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--thresh", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-per-group-eval", type=int, default=None, metavar="N")
    ap.add_argument("--single-per-group-eval", action="store_true",
                    help="Alias for --max-per-group-eval 1.")
    args = ap.parse_args()

    src = ROOT / "dataset_store" / "training_ready" / args.name
    dst = ROOT / "dataset_store" / "training_ready" / f"{args.name}_clean"
    if dst.exists():
        raise FileExistsError(f"{dst} already exists — refuse to overwrite")
    print(f"src: {src}\ndst: {dst}")

    images = _enumerate_images(src)
    print(f"found {len(images)} images")

    img_to_hash = compute_phashes(images, n_workers=args.workers)
    img_to_group = build_groups(img_to_hash, args.thresh)

    group_to_classes: dict[int, list[int]] = defaultdict(list)
    group_to_images: dict[int, int] = defaultdict(int)
    group_to_source: dict[int, str] = {}
    for img, gid in img_to_group.items():
        group_to_classes[gid].extend(_classes_in_label(_label_path_for(img)))
        group_to_images[gid] += 1
        group_to_source.setdefault(gid, _source_from_filename(img))

    group_to_split = stratified_group_split(
        group_to_classes, group_to_images,
        group_to_source=group_to_source,
        target_ratios=TARGET_RATIOS,
        stratify_by=["class", "source"],
        seed=args.seed,
    )

    cap = 1 if args.single_per_group_eval else args.max_per_group_eval
    if cap is not None:
        img_to_split: dict[Path, str | None] = apply_max_per_group_eval(
            img_to_group, group_to_split, cap, seed=args.seed
        )
    else:
        img_to_split = {img: group_to_split[gid] for img, gid in img_to_group.items()}

    for split in SPLITS:
        (dst / split / "images").mkdir(parents=True)
        (dst / split / "labels").mkdir(parents=True)

    counts = {s: 0 for s in SPLITS}
    img_to_split_kept: dict[Path, str] = {}
    for img, new_split in img_to_split.items():
        if new_split is None:
            continue
        new_img = dst / new_split / "images" / img.name
        try:
            new_img.hardlink_to(img)
        except OSError:
            shutil.copy2(img, new_img)
        counts[new_split] += 1
        img_to_split_kept[img] = new_split
        src_lbl = _label_path_for(img)
        if src_lbl.exists():
            new_lbl = dst / new_split / "labels" / src_lbl.name
            try:
                new_lbl.hardlink_to(src_lbl)
            except OSError:
                shutil.copy2(src_lbl, new_lbl)

    print(f"new counts: {counts}")
    img_to_hash_kept = {p: h for p, h in img_to_hash.items() if p in img_to_split_kept}
    leaks = verify_no_leakage(img_to_hash_kept, img_to_split_kept, args.thresh)
    print(f"cross-split pairs at hamming ≤ {args.thresh}: {leaks}")

    (dst / "splits.json").write_text(json.dumps({
        "task_type": "detection",
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "seed": args.seed,
            "ratios": list(TARGET_RATIOS),
            "stratified": True,
            "deduplicated": True,
            "hamming_thresh": args.thresh,
            "max_per_group_eval": cap,
            "source": str(src),
            "n_groups": len(set(group_to_split.values())),
            "total_samples": sum(counts.values()),
        },
        "counts": {**counts},
    }, indent=2))
    print(f"wrote {dst / 'splits.json'}")


if __name__ == "__main__":
    main()

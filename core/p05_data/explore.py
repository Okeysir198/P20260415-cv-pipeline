#!/usr/bin/env python3
"""Dataset exploration and normalization computation utilities.

Provides dataset statistics (split sizes, class distribution, image dimensions,
annotation statistics) and per-channel RGB normalization computation.

Usage (exploration)::

    uv run core/p05_data/explore.py explore --config features/safety-fire_detection/configs/05_data.yaml

Usage (normalization)::

    uv run core/p05_data/explore.py normalize --config features/safety-fire_detection/configs/05_data.yaml
    uv run core/p05_data/explore.py normalize --config features/safety-fire_detection/configs/05_data.yaml --sample-size 2000
"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from utils.yolo_io import IMAGE_EXTENSIONS, image_to_label_path, parse_yolo_label
from utils.config import load_config, resolve_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_image_paths(images_dir: Path) -> List[Path]:
    """Collect all image files from a directory.

    Args:
        images_dir: Directory containing image files.

    Returns:
        Sorted list of image file paths.
    """
    if not images_dir.exists():
        return []
    return sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def count_images(images_dir: Path) -> int:
    """Count image files in a directory.

    Args:
        images_dir: Directory to scan.

    Returns:
        Number of image files found.
    """
    return len(get_image_paths(images_dir))


def resolve_split_dir(config: dict, split: str, config_dir: Path) -> Optional[Path]:
    """Resolve the images directory for a given split.

    Args:
        config: Loaded data config dictionary.
        split: Split name ('train', 'val', or 'test').
        config_dir: Directory containing the config file (for relative path resolution).

    Returns:
        Resolved Path to the images directory, or None if the split is not defined.
    """
    if split not in config:
        return None
    base_path = resolve_path(config["path"], config_dir)
    return base_path / config[split]


def print_separator(char: str = "=", width: int = 70) -> None:
    """Print a separator line."""
    print(char * width)


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------


def compute_class_distribution(
    images_dir: Path, class_names: Dict[int, str]
) -> Dict[str, int]:
    """Count object instances per class across all labels in a split.

    Args:
        images_dir: Directory containing the image files.
        class_names: Mapping from class ID to class name.

    Returns:
        Dictionary mapping class name to instance count.
    """
    counts: Counter = Counter()
    image_paths = get_image_paths(images_dir)

    for img_path in image_paths:
        label_path = image_to_label_path(img_path)
        annotations = parse_yolo_label(label_path)
        for class_id, *_ in annotations:
            name = class_names.get(class_id, f"unknown_{class_id}")
            counts[name] += 1

    return dict(counts)


def compute_image_stats(images_dir: Path) -> Dict[str, Any]:
    """Compute image dimension statistics for a split.

    Args:
        images_dir: Directory containing image files.

    Returns:
        Dictionary with min/max/mean width and height.
    """
    widths: List[int] = []
    heights: List[int] = []
    image_paths = get_image_paths(images_dir)

    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
        except Exception:
            continue

    if not widths:
        return {"count": 0}

    return {
        "count": len(widths),
        "width_min": min(widths),
        "width_max": max(widths),
        "width_mean": round(sum(widths) / len(widths), 1),
        "height_min": min(heights),
        "height_max": max(heights),
        "height_mean": round(sum(heights) / len(heights), 1),
    }


def compute_annotation_stats(
    images_dir: Path, class_names: Dict[int, str]
) -> Dict[str, Any]:
    """Compute annotation statistics for a split.

    Args:
        images_dir: Directory containing image files.
        class_names: Mapping from class ID to class name.

    Returns:
        Dictionary with avg objects per image, images with no annotations,
        and class co-occurrence matrix.
    """
    image_paths = get_image_paths(images_dir)
    total_objects = 0
    images_with_labels = 0
    images_without_labels = 0
    objects_per_image: List[int] = []
    cooccurrence: Counter = Counter()

    for img_path in image_paths:
        label_path = image_to_label_path(img_path)
        annotations = parse_yolo_label(label_path)

        n = len(annotations)
        objects_per_image.append(n)
        total_objects += n

        if n > 0:
            images_with_labels += 1
        else:
            images_without_labels += 1

        # Class co-occurrence: which classes appear together in the same image
        classes_in_image = set()
        for class_id, *_ in annotations:
            name = class_names.get(class_id, f"unknown_{class_id}")
            classes_in_image.add(name)

        for cls in classes_in_image:
            cooccurrence[(cls, cls)] += 1

        classes_list = sorted(classes_in_image)
        for i in range(len(classes_list)):
            for j in range(i + 1, len(classes_list)):
                cooccurrence[(classes_list[i], classes_list[j])] += 1

    n_images = len(image_paths)
    avg_objects = round(total_objects / n_images, 2) if n_images > 0 else 0.0

    return {
        "total_images": n_images,
        "total_objects": total_objects,
        "avg_objects_per_image": avg_objects,
        "max_objects_per_image": max(objects_per_image) if objects_per_image else 0,
        "images_with_labels": images_with_labels,
        "images_without_labels": images_without_labels,
        "class_cooccurrence": dict(cooccurrence),
    }


# ---------------------------------------------------------------------------
# Exploration (high-level)
# ---------------------------------------------------------------------------


def explore(config_path: str) -> None:
    """Run full dataset exploration and print results.

    Args:
        config_path: Path to the data YAML config file.
    """
    config_file = Path(config_path).resolve()
    config_dir = config_file.parent
    config = load_config(config_file)

    class_names: Dict[int, str] = {int(k): v for k, v in config["names"].items()}

    # --- Header ---
    print_separator()
    print(f"  Dataset Exploration: {config['dataset_name']}")
    print_separator()
    print(f"  Config : {config_file}")
    print(f"  Path   : {resolve_path(config['path'], config_dir)}")
    print(f"  Classes: {config['num_classes']} — {list(class_names.values())}")
    print(f"  Input  : {config['input_size']}")
    print()

    splits = ["train", "val", "test"]

    # --- Split image counts ---
    print_separator("-")
    print("  Split Summary")
    print_separator("-")
    split_dirs: Dict[str, Optional[Path]] = {}
    total_images = 0
    for split in splits:
        split_dir = resolve_split_dir(config, split, config_dir)
        split_dirs[split] = split_dir
        if split_dir and split_dir.exists():
            n = count_images(split_dir)
            total_images += n
            print(f"  {split:>5s}: {n:>8,d} images  ({split_dir})")
        else:
            print(f"  {split:>5s}:        - (not found)")
    print(f"  {'total':>5s}: {total_images:>8,d} images")
    print()

    # --- Class distribution per split ---
    print_separator("-")
    print("  Class Distribution (object instances)")
    print_separator("-")
    for split in splits:
        split_dir = split_dirs[split]
        if split_dir is None or not split_dir.exists():
            continue
        dist = compute_class_distribution(split_dir, class_names)
        total = sum(dist.values())
        print(f"\n  [{split}] — {total:,d} total objects")
        for name in sorted(class_names.values()):
            count = dist.get(name, 0)
            pct = (count / total * 100) if total > 0 else 0.0
            bar = "#" * int(pct / 2)
            print(f"    {name:<20s} {count:>8,d}  ({pct:5.1f}%)  {bar}")
    print()

    # --- Image size statistics ---
    print_separator("-")
    print("  Image Size Statistics")
    print_separator("-")
    for split in splits:
        split_dir = split_dirs[split]
        if split_dir is None or not split_dir.exists():
            continue
        stats = compute_image_stats(split_dir)
        if stats["count"] == 0:
            print(f"\n  [{split}] — no readable images")
            continue
        print(f"\n  [{split}] — {stats['count']:,d} images")
        print(f"    Width  : min={stats['width_min']}, max={stats['width_max']}, mean={stats['width_mean']}")
        print(f"    Height : min={stats['height_min']}, max={stats['height_max']}, mean={stats['height_mean']}")
    print()

    # --- Annotation statistics ---
    print_separator("-")
    print("  Annotation Statistics")
    print_separator("-")
    for split in splits:
        split_dir = split_dirs[split]
        if split_dir is None or not split_dir.exists():
            continue
        ann_stats = compute_annotation_stats(split_dir, class_names)
        print(f"\n  [{split}]")
        print(f"    Total objects          : {ann_stats['total_objects']:,d}")
        print(f"    Avg objects/image      : {ann_stats['avg_objects_per_image']}")
        print(f"    Max objects/image      : {ann_stats['max_objects_per_image']}")
        print(f"    Images with labels     : {ann_stats['images_with_labels']:,d}")
        print(f"    Images without labels  : {ann_stats['images_without_labels']:,d}")

        # Co-occurrence
        cooc = ann_stats["class_cooccurrence"]
        if cooc:
            print(f"    Class co-occurrence (images where both appear):")
            for (cls_a, cls_b), count in sorted(cooc.items()):
                if cls_a == cls_b:
                    print(f"      {cls_a:<20s} alone/with-others: {count:>7,d} images")
                else:
                    print(f"      {cls_a} + {cls_b}: {count:>7,d} images")
    print()
    print_separator()
    print("  Exploration complete.")
    print_separator()


# ---------------------------------------------------------------------------
# Normalization computation
# ---------------------------------------------------------------------------


def compute_channel_stats(
    image_paths: List[Path],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and std using Welford's online algorithm.

    Processes images one at a time to avoid loading the full dataset into
    memory. Each image is converted to float32 in [0, 1] range before
    accumulating statistics.

    Args:
        image_paths: List of image file paths to process.

    Returns:
        Tuple of (mean, std) arrays, each of shape (3,), in [0, 1] range.
    """
    n_pixels = 0
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sum_sq = np.zeros(3, dtype=np.float64)

    for i, img_path in enumerate(image_paths):
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                pixels = np.asarray(img, dtype=np.float64) / 255.0
                n = pixels.shape[0] * pixels.shape[1]
                n_pixels += n
                channel_sum += pixels.sum(axis=(0, 1))
                channel_sum_sq += (pixels ** 2).sum(axis=(0, 1))
        except Exception as e:
            print(f"  Warning: skipping {img_path.name} — {e}", file=sys.stderr)
            continue

        if (i + 1) % 100 == 0 or (i + 1) == len(image_paths):
            print(f"  Processed {i + 1}/{len(image_paths)} images...", file=sys.stderr)

    if n_pixels == 0:
        raise RuntimeError("No valid images found to compute statistics.")

    mean = channel_sum / n_pixels
    std = np.sqrt(channel_sum_sq / n_pixels - mean ** 2)

    return mean, std


def normalize(config_path: str, sample_size: int = 1000, split: str = "train", seed: int = 42) -> None:
    """Compute and print normalization statistics for a dataset split.

    Args:
        config_path: Path to the data YAML config file.
        sample_size: Number of images to sample (0 for all).
        split: Which split to sample from.
        seed: Random seed for reproducible sampling.
    """
    config_file = Path(config_path).resolve()
    config_dir = config_file.parent
    config = load_config(config_file)

    if split not in config:
        print(f"Error: split '{split}' not defined in config.", file=sys.stderr)
        sys.exit(1)

    base_path = resolve_path(config["path"], config_dir)
    images_dir = base_path / config[split]

    if not images_dir.exists():
        print(f"Error: images directory not found: {images_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Dataset    : {config['dataset_name']}", file=sys.stderr)
    print(f"Split      : {split}", file=sys.stderr)
    print(f"Images dir : {images_dir}", file=sys.stderr)

    all_paths = get_image_paths(images_dir)
    if not all_paths:
        print(f"Error: no images found in {images_dir}", file=sys.stderr)
        sys.exit(1)

    if sample_size > 0 and sample_size < len(all_paths):
        random.seed(seed)
        sampled_paths = random.sample(all_paths, sample_size)
        print(
            f"Sampling   : {sample_size} / {len(all_paths)} images (seed={seed})",
            file=sys.stderr,
        )
    else:
        sampled_paths = all_paths
        print(f"Using all  : {len(all_paths)} images", file=sys.stderr)

    print(file=sys.stderr)

    mean, std = compute_channel_stats(sampled_paths)

    print(file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    print("  Normalization Statistics (RGB, [0-1] range)", file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    print(f"  R: mean={mean[0]:.6f}, std={std[0]:.6f}", file=sys.stderr)
    print(f"  G: mean={mean[1]:.6f}, std={std[1]:.6f}", file=sys.stderr)
    print(f"  B: mean={mean[2]:.6f}, std={std[2]:.6f}", file=sys.stderr)
    print(file=sys.stderr)

    mean_str = f"[{mean[0]:.3f}, {mean[1]:.3f}, {mean[2]:.3f}]"
    std_str = f"[{std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f}]"

    print("# Computed normalization — paste into your data config YAML")
    print(f"mean: {mean_str}")
    print(f"std: {std_str}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point with subcommands."""
    parser = argparse.ArgumentParser(
        description="Dataset exploration and normalization utilities.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- explore ---
    explore_parser = subparsers.add_parser(
        "explore",
        help="Explore a YOLO dataset (split sizes, class distribution, image stats).",
    )
    explore_parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the data YAML config file.",
    )

    # --- normalize ---
    norm_parser = subparsers.add_parser(
        "normalize",
        help="Compute per-channel RGB mean and std for normalization.",
    )
    norm_parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the data YAML config file.",
    )
    norm_parser.add_argument(
        "--sample-size", type=int, default=1000,
        help="Number of images to sample (default: 1000, 0 for all).",
    )
    norm_parser.add_argument(
        "--split", type=str, default="train", choices=["train", "val", "test"],
        help="Which split to sample from (default: train).",
    )
    norm_parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )

    args = parser.parse_args()

    if args.command == "explore":
        explore(args.config)
    elif args.command == "normalize":
        normalize(args.config, args.sample_size, args.split, args.seed)


if __name__ == "__main__":
    main()

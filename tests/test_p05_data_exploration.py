"""Test p05: Data Exploration — run explore_dataset and compute_normalization on real data."""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import passed, failed, errors, run_test, run_all
from core.p05_data.explore import (
    explore,
    compute_class_distribution,
    compute_image_stats,
    compute_annotation_stats,
    get_image_paths,
    resolve_split_dir,
    compute_channel_stats,
)
from utils.yolo_io import parse_yolo_label, image_to_label_path
from utils.config import load_config, resolve_path

OUTPUTS = Path(__file__).resolve().parent / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

DATA_CONFIG = str(ROOT / "configs" / "_test" / "05_data.yaml")

_normalization_stats = None


def test_explore_dataset():
    """Run explore() on test dataset."""
    # explore() prints stats — just verify it runs without error
    explore(DATA_CONFIG)


def test_compute_normalization():
    """Compute channel mean/std on test images."""
    config = load_config(DATA_CONFIG)
    base_dir = Path(DATA_CONFIG).parent
    dataset_path = resolve_path(config["path"], base_dir)
    train_images_dir = dataset_path / config["train"]

    image_paths = sorted(train_images_dir.glob("*"))
    image_paths = [p for p in image_paths if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    assert len(image_paths) > 0, "No images found for normalization"

    mean, std = compute_channel_stats(image_paths)
    assert mean.shape == (3,), f"Mean shape: {mean.shape}"
    assert std.shape == (3,), f"Std shape: {std.shape}"
    assert all(0 <= m <= 1 for m in mean), f"Mean out of range: {mean}"
    assert all(0 < s <= 1 for s in std), f"Std out of range: {std}"
    print(f"    Mean: {mean.tolist()}, Std: {std.tolist()}")

    # Save stats to global for output saving
    global _normalization_stats
    _normalization_stats = {"mean": mean.tolist(), "std": std.tolist(), "num_images": len(image_paths)}


def test_parse_yolo_label():
    """Parse a real YOLO label file and verify structure."""
    config = load_config(DATA_CONFIG)
    base_dir = Path(DATA_CONFIG).parent
    dataset_path = resolve_path(config["path"], base_dir)
    train_images_dir = dataset_path / config["train"]
    train_labels_dir = train_images_dir.parent / "labels"

    label_files = sorted(train_labels_dir.glob("*.txt"))
    assert len(label_files) > 0, "No label files found"

    annotations = parse_yolo_label(label_files[0])
    assert isinstance(annotations, list), f"Expected list, got {type(annotations)}"
    assert len(annotations) > 0, "Label file is empty"

    for ann in annotations:
        assert isinstance(ann, tuple), f"Expected tuple, got {type(ann)}"
        assert len(ann) == 5, f"Expected 5 elements, got {len(ann)}"
        class_id, cx, cy, w, h = ann
        assert isinstance(class_id, int), f"class_id should be int, got {type(class_id)}"
        assert isinstance(cx, float), f"cx should be float, got {type(cx)}"
        assert 0 <= cx <= 1, f"cx out of range: {cx}"
        assert 0 <= cy <= 1, f"cy out of range: {cy}"
        assert 0 <= w <= 1, f"w out of range: {w}"
        assert 0 <= h <= 1, f"h out of range: {h}"

    print(f"    Parsed {len(annotations)} annotations from {label_files[0].name}")


def test_compute_class_distribution():
    """Compute class distribution on train split and verify counts."""
    config = load_config(DATA_CONFIG)
    base_dir = Path(DATA_CONFIG).parent
    dataset_path = resolve_path(config["path"], base_dir)
    train_images_dir = dataset_path / config["train"]

    class_names = {int(k): v for k, v in config["names"].items()}
    dist = compute_class_distribution(train_images_dir, class_names)
    assert isinstance(dist, dict), f"Expected dict, got {type(dist)}"
    assert any(count > 0 for count in dist.values()), "All class counts are zero"
    print(f"    Class distribution: {dist}")


def test_compute_image_stats():
    """Compute image dimension stats on train split."""
    config = load_config(DATA_CONFIG)
    base_dir = Path(DATA_CONFIG).parent
    dataset_path = resolve_path(config["path"], base_dir)
    train_images_dir = dataset_path / config["train"]

    stats = compute_image_stats(train_images_dir)
    assert isinstance(stats, dict), f"Expected dict, got {type(stats)}"
    assert stats["count"] > 0, "No images found"
    for key in ("width_min", "width_max", "width_mean", "height_min", "height_max", "height_mean"):
        assert key in stats, f"Missing key: {key}"
    print(f"    Image stats: {stats}")


def test_compute_annotation_stats():
    """Compute annotation stats on train split."""
    config = load_config(DATA_CONFIG)
    base_dir = Path(DATA_CONFIG).parent
    dataset_path = resolve_path(config["path"], base_dir)
    train_images_dir = dataset_path / config["train"]

    class_names = {int(k): v for k, v in config["names"].items()}
    ann_stats = compute_annotation_stats(train_images_dir, class_names)
    assert isinstance(ann_stats, dict), f"Expected dict, got {type(ann_stats)}"
    assert ann_stats["avg_objects_per_image"] > 0, "avg_objects_per_image should be > 0"
    print(f"    Annotation stats: avg_objects_per_image={ann_stats['avg_objects_per_image']}")


def test_get_image_paths():
    """Call get_image_paths() on test dataset train images dir."""
    config = load_config(DATA_CONFIG)
    base_dir = Path(DATA_CONFIG).parent
    dataset_path = resolve_path(config["path"], base_dir)
    train_images_dir = dataset_path / config["train"]

    paths = get_image_paths(train_images_dir)
    assert isinstance(paths, list), f"Expected list, got {type(paths)}"
    assert len(paths) > 0, "No image paths returned"
    assert all(isinstance(p, Path) for p in paths), "Not all elements are Path objects"

    # Test dataset has ~80 train images
    assert len(paths) >= 50, f"Expected ~80 train images, got {len(paths)}"
    assert len(paths) <= 120, f"Expected ~80 train images, got {len(paths)}"

    # Verify all paths are actual image files
    for p in paths:
        assert p.exists(), f"Image path does not exist: {p}"
        assert p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}, (
            f"Unexpected extension: {p.suffix}"
        )

    print(f"    Found {len(paths)} image paths in {train_images_dir}")


def test_image_to_label_path():
    """Call image_to_label_path() on a real image path, verify correct label path."""
    config = load_config(DATA_CONFIG)
    base_dir = Path(DATA_CONFIG).parent
    dataset_path = resolve_path(config["path"], base_dir)
    train_images_dir = dataset_path / config["train"]

    image_paths = get_image_paths(train_images_dir)
    assert len(image_paths) > 0, "No images found"

    img_path = image_paths[0]
    label_path = image_to_label_path(img_path)

    # Verify images/ -> labels/ substitution
    assert "labels" in str(label_path), f"Label path should contain 'labels': {label_path}"
    assert "images" not in str(label_path.parent.name), (
        f"Label path parent should not be 'images': {label_path}"
    )

    # Verify extension changed to .txt
    assert label_path.suffix == ".txt", f"Expected .txt suffix, got {label_path.suffix}"

    # Verify stem matches the image stem
    assert label_path.stem == img_path.stem, (
        f"Stem mismatch: image={img_path.stem}, label={label_path.stem}"
    )

    # Verify the label file actually exists for the test dataset
    assert label_path.exists(), f"Label file does not exist: {label_path}"

    print(f"    Image: {img_path.name} -> Label: {label_path.name} (exists={label_path.exists()})")


def test_resolve_split_dir():
    """Test resolve_split_dir() with the test data config for train/val splits."""
    config = load_config(DATA_CONFIG)
    config_dir = Path(DATA_CONFIG).parent

    # Train split
    train_dir = resolve_split_dir(config, "train", config_dir)
    assert train_dir is not None, "resolve_split_dir returned None for 'train'"
    assert train_dir.exists(), f"Train split dir does not exist: {train_dir}"
    print(f"    train dir: {train_dir} (exists)")

    # Val split
    val_dir = resolve_split_dir(config, "val", config_dir)
    assert val_dir is not None, "resolve_split_dir returned None for 'val'"
    assert val_dir.exists(), f"Val split dir does not exist: {val_dir}"
    print(f"    val dir:   {val_dir} (exists)")

    # Non-existent split should return None
    missing_dir = resolve_split_dir(config, "nonexistent_split", config_dir)
    assert missing_dir is None, f"Expected None for missing split, got {missing_dir}"
    print(f"    nonexistent_split: None (correct)")


def save_outputs(stats):
    """Save exploration stats."""
    out_file = OUTPUTS / "01_exploration_stats.json"
    with open(out_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"    Saved: {out_file}")


if __name__ == "__main__":
    run_all([
        ("explore_dataset", test_explore_dataset),
        ("compute_normalization", test_compute_normalization),
        ("parse_yolo_label", test_parse_yolo_label),
        ("compute_class_distribution", test_compute_class_distribution),
        ("compute_image_stats", test_compute_image_stats),
        ("compute_annotation_stats", test_compute_annotation_stats),
        ("get_image_paths", test_get_image_paths),
        ("image_to_label_path", test_image_to_label_path),
        ("resolve_split_dir", test_resolve_split_dir),
    ], title="Test 02: Data Exploration", exit_on_fail=False)

    try:
        if _normalization_stats:
            save_outputs(_normalization_stats)
    except Exception as e:
        print(f"  WARNING: Could not save outputs — {e}")

    if failed > 0:
        sys.exit(1)

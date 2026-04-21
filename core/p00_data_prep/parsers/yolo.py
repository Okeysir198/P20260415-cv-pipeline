"""
YOLO format parser.

Parses YOLO-format datasets with .txt annotation files.
Supports both pre-split (train/valid/test) and unsplit (flat) structures.
"""

from pathlib import Path

import yaml

from core.p00_data_prep.utils.file_ops import IMAGE_EXTENSIONS, resolve_data_root


def parse_yolo(source_config: dict, base_dir: Path) -> list[dict]:
    """
    Parse YOLO format dataset.

    Args:
        source_config: Dict with:
            - path: Path to dataset (relative or absolute)
            - has_splits: Whether dataset has train/valid/test subdirs
            - splits_to_use: Which splits to include (if has_splits=True)
            - source_classes: Optional list of class names by ID order
        base_dir: Base directory for resolving relative paths

    Returns:
        List of sample dicts with 'filename', 'image_path', 'labels', 'bboxes', 'source'
    """
    data_root = resolve_data_root(source_config, base_dir)
    source_name = source_config.get("name", "yolo_source")
    has_splits = source_config.get("has_splits", False)
    splits_to_use = source_config.get("splits_to_use", ["train", "valid", "test"])

    # Resolve class ID → class name mapping
    id_to_name = _resolve_class_names(source_config, data_root)

    samples = []

    if has_splits:
        for split in splits_to_use:
            img_dir = data_root / split / "images"
            label_dir = data_root / split / "labels"

            if not img_dir.exists():
                continue

            samples.extend(_parse_yolo_split(img_dir, label_dir, source_name, id_to_name, original_split=split))
    else:
        img_dir = data_root / "images"
        label_dir = data_root / "labels"

        if not img_dir.exists():
            img_dir = data_root
            label_dir = data_root

        samples.extend(_parse_yolo_split(img_dir, label_dir, source_name, id_to_name))

    return samples


def _resolve_class_names(source_config: dict, data_root: Path) -> dict[str, str]:
    """
    Build class ID → class name mapping from source_classes config or data.yaml.

    Args:
        source_config: Source dataset config
        data_root: Root directory of the dataset

    Returns:
        Dict mapping string class IDs ("0", "1", ...) to class names.
        Empty dict if no mapping available (labels stay as raw IDs).
    """
    # Priority 1: explicit source_classes in config
    source_classes = source_config.get("source_classes")
    if source_classes:
        return {str(i): name for i, name in enumerate(source_classes)}

    # Priority 2: data.yaml in dataset root
    classes = get_yolo_classes(data_root / "data.yaml")
    if classes:
        return {str(i): name for i, name in enumerate(classes)}

    return {}


def _parse_yolo_split(
    img_dir: Path,
    label_dir: Path,
    source_name: str,
    id_to_name: dict[str, str],
    original_split: str | None = None,
) -> list[dict]:
    """
    Parse a single YOLO split.

    Args:
        img_dir: Directory containing images
        label_dir: Directory containing label .txt files
        source_name: Name of source dataset
        id_to_name: Mapping from string class IDs to class names

    Returns:
        List of sample dicts
    """
    samples = []

    image_files = [
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    for img_path in image_files:
        label_path = label_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            continue

        labels, bboxes = _parse_yolo_label(label_path)

        # Resolve class IDs to names
        if id_to_name:
            labels = [id_to_name.get(label, label) for label in labels]

        sample: dict = {
            "filename": img_path.name,
            "image_path": img_path,
            "labels": labels,
            "bboxes": bboxes,
            "source": source_name,
        }
        if original_split is not None:
            sample["original_split"] = original_split
        samples.append(sample)

    return samples


def _parse_yolo_label(label_path: Path) -> tuple[list[str], list[list[float]]]:
    """
    Parse a YOLO format label file.

    Format: class_id cx cy w h (one line per object)

    Returns:
        Tuple of (class_id strings, bboxes as [[cx, cy, w, h], ...])
    """
    labels = []
    bboxes = []

    try:
        with open(label_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) >= 5:
                    labels.append(parts[0])
                    bboxes.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    except OSError:
        pass

    return labels, bboxes


def get_yolo_classes(data_yaml: Path | None = None) -> list[str]:
    """
    Get class names from YOLO data.yaml file.

    Args:
        data_yaml: Path to data.yaml file

    Returns:
        List of class names
    """
    if data_yaml is None or not data_yaml.exists():
        return []

    with open(data_yaml) as f:
        data = yaml.safe_load(f)

    return data.get("names", [])

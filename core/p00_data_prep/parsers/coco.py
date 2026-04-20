"""
COCO format parser.

Parses COCO JSON format annotations.
"""

import json
from pathlib import Path
from typing import Dict, List

from core.p00_data_prep.utils.file_ops import resolve_data_root


def parse_coco(source_config: Dict, base_dir: Path) -> List[Dict]:
    """
    Parse COCO format dataset.

    Args:
        source_config: Dict with:
            - path: Path to dataset (relative or absolute)
            - has_splits: Whether dataset has train/val/test subdirs
            - splits_to_use: Which splits to include (if has_splits=True)
        base_dir: Base directory for resolving relative paths

    Returns:
        List of sample dicts with 'filename', 'image_path', 'labels', 'bboxes', 'source'
    """
    data_root = resolve_data_root(source_config, base_dir)
    source_name = source_config.get("name", "coco_source")
    has_splits = source_config.get("has_splits", True)
    splits_to_use = source_config.get("splits_to_use", ["train", "val", "test"])

    samples = []

    if has_splits:
        for split in splits_to_use:
            ann_file = None
            img_dir = None

            test_ann = data_root / "data" / split / "_annotations.coco.json"
            if test_ann.exists():
                ann_file = test_ann
                img_dir = data_root / "data" / split

            if ann_file is None:
                test_ann = data_root / "annotations" / f"{split}.json"
                if test_ann.exists():
                    ann_file = test_ann
                    img_dir = data_root / "images" / split

            if ann_file is None or img_dir is None:
                continue

            samples.extend(_parse_coco_split(ann_file, img_dir, source_name))
    else:
        ann_file = None
        img_dir = None

        test_ann = data_root / "annotations.json"
        if test_ann.exists():
            ann_file = test_ann
            img_dir = data_root / "images"

        if ann_file is None:
            test_ann = data_root / "_annotations.coco.json"
            if test_ann.exists():
                ann_file = test_ann
                img_dir = data_root

        if ann_file is None:
            test_ann = data_root / "data" / "_annotations.coco.json"
            if test_ann.exists():
                ann_file = test_ann
                img_dir = data_root / "data"

        if ann_file is not None and img_dir is not None:
            samples.extend(_parse_coco_split(ann_file, img_dir, source_name))

    return samples


def _parse_coco_split(
    ann_file: Path,
    img_dir: Path,
    source_name: str
) -> List[Dict]:
    """
    Parse a single COCO annotation file.

    Args:
        ann_file: Path to COCO annotations JSON
        img_dir: Directory containing images
        source_name: Name of source dataset

    Returns:
        List of sample dicts
    """
    with open(ann_file) as f:
        data = json.load(f)

    image_id_to_anns: Dict[int, list] = {}
    for ann in data.get("annotations", []):
        image_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    image_id_to_info = {img["id"]: img for img in data.get("images", [])}
    cat_id_to_name = {cat["id"]: cat["name"] for cat in data.get("categories", [])}

    samples = []

    from ._image_dims import actual_image_dims

    for img_id, img_info in image_id_to_info.items():
        anns = image_id_to_anns.get(img_id, [])

        if not anns:
            continue

        img_path = img_dir / img_info["file_name"]
        if not img_path.exists():
            continue

        # Prefer actual image dims (reads header via PIL, ~O(1) syscall) over
        # COCO metadata — the latter is wrong in a small fraction of public
        # datasets and silently produces out-of-range YOLO coords when used
        # blindly. Matches the "never trust metadata" principle from qubvel's
        # reference notebook pipeline.
        img_w, img_h = actual_image_dims(
            img_path,
            fallback_w=int(img_info.get("width") or 1),
            fallback_h=int(img_info.get("height") or 1),
        )

        labels = []
        bboxes = []
        for ann in anns:
            cat_name = cat_id_to_name.get(ann["category_id"], f"class_{ann['category_id']}")
            labels.append(cat_name)

            # COCO bbox: [x, y, w, h] pixel absolute → YOLO [cx, cy, w, h] normalized
            x, y, w, h = ann["bbox"]
            bboxes.append([
                max(0.0, min(1.0, (x + w / 2) / img_w)),
                max(0.0, min(1.0, (y + h / 2) / img_h)),
                max(0.0, min(1.0, w / img_w)),
                max(0.0, min(1.0, h / img_h)),
            ])

        samples.append({
            "filename": img_info["file_name"],
            "image_path": img_path,
            "labels": labels,
            "bboxes": bboxes,
            "source": source_name
        })

    return samples

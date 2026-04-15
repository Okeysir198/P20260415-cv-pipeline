#!/usr/bin/env python3
"""Create a small test dataset from real fire images.

Converts VOC XML annotations to YOLO format for 100 real fire images.
"""

import sys
import random
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import xml.etree.ElementTree as ET
import cv2
import numpy as np
from tqdm import tqdm


def parse_voc_xml(xml_path: Path) -> list:
    """Parse Pascal VOC XML annotation file.

    Returns:
        List of dicts: [{'class': name, 'bbox': [xmin, ymin, xmax, ymax]}, ...]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        # Normalize class names
        if name.lower() in ['fire', 'smoke']:
            class_id = 0 if name.lower() == 'fire' else 1

            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            objects.append({'class_id': class_id, 'bbox': [xmin, ymin, xmax, ymax]})

    return objects


def voc_to_yolo(voc_bbox, img_w, img_h):
    """Convert VOC bbox [xmin, ymin, xmax, ymax] to YOLO [cx, cy, w, h] (normalized)."""
    xmin, ymin, xmax, ymax = voc_bbox
    cx = (xmin + xmax) / 2.0 / img_w
    cy = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return cx, cy, w, h


def create_test_dataset(
    source_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
):
    """Create YOLO-format test dataset from VOC source.

    Args:
        source_dir: Path to fire_smoke_datacluster folder
        output_dir: Path to output YOLO dataset
        train_ratio: Ratio of training images (rest go to val)
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    # Find source paths (handle nested folders)
    img_dir = None
    xml_dir = None

    # Search for images and annotations
    for root in source_dir.rglob('*'):
        if root.is_file():
            if root.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                if img_dir is None:
                    img_dir = root.parent
            elif root.suffix.lower() == '.xml':
                if xml_dir is None:
                    xml_dir = root.parent

    if img_dir is None or xml_dir is None:
        raise FileNotFoundError(f"Could not find images or annotations in {source_dir}")

    print(f"Found images in: {img_dir}")
    print(f"Found annotations in: {xml_dir}")

    # Get all images
    img_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
    print(f"Found {len(img_files)} images")

    # Create output directories
    for split in ['train', 'val']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Shuffle and split
    random.shuffle(img_files)
    split_idx = int(len(img_files) * train_ratio)
    train_files = img_files[:split_idx]
    val_files = img_files[split_idx:]

    print(f"\nSplit: {len(train_files)} train, {len(val_files)} val")

    # Process each split
    for split, files in [('train', train_files), ('val', val_files)]:
        print(f"\nProcessing {split}...")
        out_img_dir = output_dir / split / 'images'
        out_lbl_dir = output_dir / split / 'labels'

        for img_path in tqdm(files):
            # Find corresponding XML
            xml_path = xml_dir / f"{img_path.stem}.xml"
            if not xml_path.exists():
                print(f"Warning: No annotation for {img_path.name}")
                continue

            # Read image to get dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not read {img_path.name}")
                continue
            img_h, img_w = img.shape[:2]

            # Parse annotations
            objects = parse_voc_xml(xml_path)
            if not objects:
                print(f"Warning: No valid objects in {xml_path.name}")
                continue

            # Copy image
            out_img_path = out_img_dir / img_path.name
            cv2.imwrite(str(out_img_path), img)

            # Write YOLO labels
            out_lbl_path = out_lbl_dir / f"{img_path.stem}.txt"
            with open(out_lbl_path, 'w') as f:
                for obj in objects:
                    cx, cy, w, h = voc_to_yolo(obj['bbox'], img_w, img_h)
                    # Clip to [0, 1]
                    cx = max(0, min(1, cx))
                    cy = max(0, min(1, cy))
                    w = max(0, min(1, w))
                    h = max(0, min(1, h))
                    f.write(f"{obj['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    print(f"\n✅ Created test dataset at: {output_dir}")
    print(f"   Classes: 0=fire, 1=smoke")


if __name__ == "__main__":
    source = PROJECT_ROOT / "dataset_store/raw/fire_detection/fire_smoke_datacluster"
    output = PROJECT_ROOT / "dataset_store/test_fire_100"

    create_test_dataset(source, output, train_ratio=0.8, seed=42)

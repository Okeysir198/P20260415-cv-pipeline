#!/usr/bin/env python3
"""Convert YOLO-format dataset to COCO JSON format for PaddleDetection.

Reads a pipeline 05_data.yaml config and converts YOLO .txt labels to COCO JSON
annotations that PaddleDetection (PicoDet, PP-YOLOE, etc.) can consume directly.

Usage:
    uv run utils/paddle_bridge.py \
        --data-config features/ppe-shoes_detection/configs/05_data.yaml

    uv run utils/paddle_bridge.py \
        --data-config features/safety-fire_detection/configs/05_data.yaml \
        --output-dir dataset_store/fire_detection/annotations

    uv run utils/paddle_bridge.py \
        --data-config features/ppe-shoes_detection/configs/05_data.yaml \
        --splits train val
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # ai/ root

from utils.config import load_config

try:
    from PIL import Image
except ImportError:
    Image = None

DEFAULT_SPLITS = ["train", "val", "test"]


def get_image_size(image_path: Path) -> tuple[int, int]:
    """Get image (width, height). Uses PIL if available, falls back to OpenCV."""
    if Image is not None:
        with Image.open(image_path) as img:
            return img.size  # (width, height)

    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]
    return w, h


def yolo_to_coco(
    data_config_path: str,
    output_dir: str | None = None,
    splits: list[str] | None = None,
) -> dict[str, Path]:
    """Convert YOLO-format dataset to COCO JSON format.

    Args:
        data_config_path: Path to pipeline 05_data.yaml config.
        output_dir: Output directory for JSON files. Defaults to
            <dataset_path>/annotations/.
        splits: Which splits to convert. Defaults to ["train", "val", "test"].

    Returns:
        Dict mapping split name to output JSON path.
    """
    config = load_config(data_config_path)
    config_dir = Path(data_config_path).resolve().parent

    dataset_path = (config_dir / config["path"]).resolve()
    class_names = config["names"]  # {0: "fire", 1: "smoke"}
    splits = splits or DEFAULT_SPLITS

    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = dataset_path / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)

    # COCO categories (1-indexed, PaddleDetection convention)
    categories = [
        {"id": class_id + 1, "name": name, "supercategory": "none"}
        for class_id, name in sorted(class_names.items())
    ]
    # Map YOLO 0-indexed class IDs to COCO 1-indexed
    yolo_to_coco_id = {class_id: class_id + 1 for class_id in class_names}

    split_key_map = {
        "train": config.get("train", "train/images"),
        "val": config.get("val", "val/images"),
        "test": config.get("test", "test/images"),
    }

    output_paths = {}
    for split in splits:
        images_rel = split_key_map.get(split)
        if images_rel is None:
            print(f"  Skipping split '{split}': not defined in config")
            continue

        images_dir = dataset_path / images_rel
        if not images_dir.exists():
            print(f"  Skipping split '{split}': {images_dir} does not exist")
            continue

        # Derive labels dir: train/images → train/labels
        labels_dir = images_dir.parent / "labels"
        if not labels_dir.exists():
            print(f"  Skipping split '{split}': {labels_dir} does not exist")
            continue

        coco = {
            "images": [],
            "annotations": [],
            "categories": categories,
        }

        image_id = 0
        ann_id = 0
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_files = sorted(
            f for f in images_dir.iterdir() if f.suffix.lower() in image_extensions
        )

        skipped = 0
        for img_path in image_files:
            label_path = labels_dir / (img_path.stem + ".txt")

            try:
                w, h = get_image_size(img_path)
            except (ValueError, OSError) as e:
                print(f"  Warning: {e}, skipping")
                skipped += 1
                continue

            image_id += 1
            coco["images"].append(
                {
                    "id": image_id,
                    "file_name": str(img_path.relative_to(dataset_path)),
                    "width": w,
                    "height": h,
                }
            )

            if not label_path.exists():
                continue

            for line in label_path.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                if class_id not in yolo_to_coco_id:
                    continue

                # YOLO normalized cx, cy, bw, bh → COCO pixel x, y, w, h
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                box_w = bw * w
                box_h = bh * h
                box_x = (cx * w) - (box_w / 2)
                box_y = (cy * h) - (box_h / 2)

                ann_id += 1
                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": yolo_to_coco_id[class_id],
                        "bbox": [round(box_x, 2), round(box_y, 2), round(box_w, 2), round(box_h, 2)],
                        "area": round(box_w * box_h, 2),
                        "iscrowd": 0,
                    }
                )

        out_path = out_dir / f"{split}.json"
        with open(out_path, "w") as f:
            json.dump(coco, f, indent=2)

        output_paths[split] = out_path
        print(
            f"  {split}: {len(coco['images'])} images, "
            f"{len(coco['annotations'])} annotations → {out_path}"
            + (f" ({skipped} skipped)" if skipped else "")
        )

    return output_paths


def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLO dataset to COCO JSON for PaddleDetection"
    )
    parser.add_argument(
        "--data-config",
        required=True,
        help="Path to pipeline 05_data.yaml config",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for JSON files (default: <dataset_path>/annotations/)",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=DEFAULT_SPLITS,
        help="Splits to convert (default: train val test)",
    )
    args = parser.parse_args()

    print(f"Converting YOLO → COCO from config: {args.data_config}")
    yolo_to_coco(args.data_config, args.output_dir, args.splits)
    print("Done.")


if __name__ == "__main__":
    main()

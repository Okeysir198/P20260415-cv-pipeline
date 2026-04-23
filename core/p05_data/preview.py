#!/usr/bin/env python3
"""CLI tool to preview augmentation transforms on real dataset samples.

Generates side-by-side comparison images (original vs augmented) with
bounding boxes drawn, and saves them as PNGs.

Usage::

    python augmentation_preview.py \
        --data-config ../features/safety-fire_detection/configs/05_data.yaml \
        --train-config ../features/safety-fire_detection/configs/06_training.yaml \
        --num-samples 8 \
        --save-dir outputs/augmentation_preview
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from core.p05_data.detection_dataset import YOLOXDataset
from core.p05_data.transforms import build_transforms
from utils.config import load_config
from utils.viz import VizStyle, annotate_detections

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def draw_bboxes(
    image: np.ndarray,
    targets: np.ndarray,
    class_names: dict,
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes and class labels on an image.

    Args:
        image: HWC uint8 RGB image (will be copied).
        targets: (N, 5) array [class_id, cx, cy, w, h] normalised.
        class_names: Mapping ``{int: str}`` of class ids to names.
        thickness: Unused — :class:`VizStyle` auto-scales box thickness.

    Returns:
        Annotated RGB image copy.
    """
    del thickness  # VizStyle.auto_box_thickness handles this
    h, w = image.shape[:2]
    if targets is None or len(targets) == 0:
        dets = sv.Detections.empty()
    else:
        t = np.asarray(targets, dtype=np.float32).reshape(-1, 5)
        cls = t[:, 0].astype(int)
        cx, cy, bw, bh = t[:, 1], t[:, 2], t[:, 3], t[:, 4]
        xyxy = np.stack([
            (cx - bw / 2) * w, (cy - bh / 2) * h,
            (cx + bw / 2) * w, (cy + bh / 2) * h,
        ], axis=1).astype(np.float32)
        dets = sv.Detections(xyxy=xyxy, class_id=cls)
    style = VizStyle(label_text_scale=0.5)
    return annotate_detections(image, dets, class_names=class_names, style=style)


def make_side_by_side(
    orig: np.ndarray,
    augmented: np.ndarray,
    target_height: int = 480,
) -> np.ndarray:
    """Create a side-by-side image (original | augmented).

    Both images are resized to the same height while keeping aspect ratio,
    separated by a thin white line.

    Args:
        orig: Original annotated image.
        augmented: Augmented annotated image.
        target_height: Height to resize both images to.

    Returns:
        Concatenated image.
    """
    def _resize_keep_ar(img: np.ndarray, h: int) -> np.ndarray:
        ratio = h / img.shape[0]
        new_w = int(img.shape[1] * ratio)
        return cv2.resize(img, (new_w, h))

    left = _resize_keep_ar(orig, target_height)
    right = _resize_keep_ar(augmented, target_height)
    separator = np.full((target_height, 4, 3), 255, dtype=np.uint8)
    return np.concatenate([left, separator, right], axis=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview augmentation transforms on dataset samples."
    )
    parser.add_argument(
        "--data-config", type=str, required=True,
        help="Path to data YAML config (e.g. features/safety-fire_detection/configs/05_data.yaml).",
    )
    parser.add_argument(
        "--train-config", type=str, required=True,
        help="Path to training YAML config (e.g. features/safety-fire_detection/configs/06_training.yaml).",
    )
    parser.add_argument(
        "--num-samples", type=int, default=8,
        help="Number of samples to preview (default: 8).",
    )
    parser.add_argument(
        "--save-dir", type=str, default="outputs/augmentation_preview",
        help="Directory to save preview images.",
    )
    parser.add_argument(
        "--split", type=str, default="train",
        help="Dataset split to sample from (default: train).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Resolve config paths relative to this script's parent (pipeline/)
    pipeline_dir = Path(__file__).resolve().parent.parent
    data_config = load_config(args.data_config)
    train_config = load_config(args.train_config)

    input_size = tuple(data_config["input_size"])
    mean = data_config.get("mean", [0.485, 0.456, 0.406])
    std = data_config.get("std", [0.229, 0.224, 0.225])
    aug_config = train_config.get("augmentation", {})
    class_names = data_config["names"]

    # Build augmented transforms (training mode)
    aug_transforms = build_transforms(
        config=aug_config, is_train=True,
        input_size=input_size, mean=mean, std=std,
    )

    # Build the dataset without transforms (raw) for originals
    raw_dataset = YOLOXDataset(
        data_config=data_config,
        split=args.split,
        transforms=None,
        base_dir=pipeline_dir,
    )

    # Build the dataset with transforms for augmented views
    aug_dataset = YOLOXDataset(
        data_config=data_config,
        split=args.split,
        transforms=aug_transforms,
        base_dir=pipeline_dir,
    )

    # Sample random indices
    n = min(args.num_samples, len(raw_dataset))
    indices = np.random.choice(len(raw_dataset), size=n, replace=False)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {n} augmentation previews...")
    print(f"  Data config:  {args.data_config}")
    print(f"  Train config: {args.train_config}")
    print(f"  Save dir:     {save_dir}")
    print(f"  Input size:   {input_size}")
    print()

    for i, idx in enumerate(indices):
        # Original (raw)
        raw_item = raw_dataset.get_raw_item(idx)
        orig_img, orig_tgt = raw_item["image"], raw_item["targets"]
        orig_resized = cv2.resize(orig_img, (input_size[1], input_size[0]))
        orig_rgb = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2RGB)
        orig_vis = draw_bboxes(orig_rgb, orig_tgt, class_names)

        # Augmented — get the tensor output and reverse for visualisation
        aug_tensor, aug_tgt_tensor, path = aug_dataset[idx]

        # Reverse normalisation for display
        aug_np = aug_tensor.numpy().transpose(1, 2, 0)  # CHW -> HWC
        mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        aug_np = aug_np * std_arr + mean_arr
        aug_np = np.clip(aug_np * 255, 0, 255).astype(np.uint8)

        aug_tgt_np = aug_tgt_tensor.numpy() if len(aug_tgt_tensor) else np.zeros((0, 5))
        aug_vis = draw_bboxes(aug_np, aug_tgt_np, class_names)

        # Side by side
        combined = make_side_by_side(orig_vis, aug_vis)

        out_path = save_dir / f"sample_{i:03d}.png"
        # `combined` is RGB; convert to BGR for cv2.imwrite.
        cv2.imwrite(str(out_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

        n_orig = len(orig_tgt)
        n_aug = len(aug_tgt_np)
        print(f"  [{i+1}/{n}] idx={idx}  objects: {n_orig} -> {n_aug}  saved: {out_path.name}")

    print(f"\nDone. Previews saved to {save_dir}/")


if __name__ == "__main__":
    main()

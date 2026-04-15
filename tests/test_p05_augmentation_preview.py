"""Test 02: Augmentation Preview — preview class and helpers."""

import sys
import traceback
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import passed, failed, errors, run_test, run_all
from core.p05_data.preview import draw_bboxes, make_side_by_side
from core.p05_data.transforms import build_transforms
from core.p05_data.detection_dataset import YOLOXDataset
from utils.config import load_config

OUTPUTS = Path(__file__).resolve().parent / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

DATA_CONFIG_PATH = str(ROOT / "configs" / "_test" / "05_data.yaml")
TRAIN_CONFIG_PATH = str(ROOT / "configs" / "_test" / "06_training.yaml")


def test_draw_bboxes_preview():
    """Draw bboxes on a real image, verify output shape and modification."""
    from fixtures import real_image_with_targets
    real_img, real_tgt = real_image_with_targets()
    image = cv2.resize(real_img, (640, 480))
    # Use real targets if available, otherwise synthetic
    targets = real_tgt if len(real_tgt) > 0 else np.array([[0, 0.5, 0.5, 0.3, 0.2]])
    result = draw_bboxes(image, targets, {0: "fire", 1: "smoke"})

    assert result.shape == image.shape, (
        f"Expected shape {image.shape}, got {result.shape}"
    )
    assert result.dtype == np.uint8, (
        f"Expected dtype uint8, got {result.dtype}"
    )
    assert not np.array_equal(result, image), (
        "Result should differ from input (boxes should be drawn)"
    )
    print(f"    draw_bboxes output shape: {result.shape}, dtype: {result.dtype}")


def test_make_side_by_side():
    """Create side-by-side from two different-sized real images."""
    from fixtures import real_image
    real = real_image()
    img1 = cv2.resize(real, (400, 300))
    img2 = cv2.resize(real, (600, 500))
    result = make_side_by_side(img1, img2, target_height=480)

    assert result.shape[0] == 480, (
        f"Expected height 480, got {result.shape[0]}"
    )
    # After scaling to height 480: img1 width = 400*(480/300)=640, img2 width = 600*(480/500)=576
    # Combined = 640 + 4 (separator) + 576 = 1220
    scaled_w1 = int(400 * (480 / 300))
    scaled_w2 = int(600 * (480 / 500))
    assert result.shape[1] > scaled_w1, (
        f"Combined width {result.shape[1]} should be wider than scaled img1 width {scaled_w1}"
    )
    assert result.shape[1] > scaled_w2, (
        f"Combined width {result.shape[1]} should be wider than scaled img2 width {scaled_w2}"
    )
    assert result.dtype == np.uint8, (
        f"Expected dtype uint8, got {result.dtype}"
    )
    print(f"    Side-by-side shape: {result.shape}")


def test_preview_with_real_data():
    """Generate a preview with real fire dataset images."""
    data_config = load_config(DATA_CONFIG_PATH)
    train_config = load_config(TRAIN_CONFIG_PATH)
    input_size = tuple(data_config["input_size"])
    aug_config = train_config.get("augmentation", {})
    class_names = data_config["names"]
    mean = data_config.get("mean", [0.485, 0.456, 0.406])
    std = data_config.get("std", [0.229, 0.224, 0.225])

    # Raw dataset
    raw_ds = YOLOXDataset(data_config, split="train", transforms=None, base_dir=Path(DATA_CONFIG_PATH).parent)

    # Augmented dataset
    aug_tfm = build_transforms(aug_config, is_train=True, input_size=input_size, mean=mean, std=std)
    aug_ds = YOLOXDataset(data_config, split="train", transforms=aug_tfm, base_dir=Path(DATA_CONFIG_PATH).parent)

    # Get raw image
    raw = raw_ds.get_raw_item(0)
    orig_img, orig_tgt = raw["image"], raw["targets"]
    orig_resized = cv2.resize(orig_img, (input_size[1], input_size[0]))
    orig_vis = draw_bboxes(orig_resized, orig_tgt, class_names)

    # Get augmented image
    aug_tensor, aug_tgt, _ = aug_ds[0]
    aug_np = aug_tensor.numpy().transpose(1, 2, 0)
    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    aug_np = aug_np * std_arr + mean_arr
    aug_np = np.clip(aug_np * 255, 0, 255).astype(np.uint8)
    aug_tgt_np = aug_tgt.numpy() if len(aug_tgt) else np.zeros((0, 5))
    aug_vis = draw_bboxes(aug_np, aug_tgt_np, class_names)

    # Combine
    combined = make_side_by_side(orig_vis, aug_vis)
    assert combined.shape[1] > orig_vis.shape[1], "Combined should be wider than original"
    assert combined.shape[0] == 480, f"Expected height 480, got {combined.shape[0]}"

    # Save
    cv2.imwrite(str(OUTPUTS / "augmentation_preview_test.png"), combined)
    print(f"    Preview saved, combined shape: {combined.shape}")


if __name__ == "__main__":
    run_all([
        ("draw_bboxes_preview", test_draw_bboxes_preview),
        ("make_side_by_side", test_make_side_by_side),
        ("preview_with_real_data", test_preview_with_real_data),
    ], title="Test 15: Augmentation Preview")

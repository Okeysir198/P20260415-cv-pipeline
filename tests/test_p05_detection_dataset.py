"""Test 01: Detection Dataset — test transforms, dataset, and dataloader with real images."""

import sys
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import passed, failed, errors, run_test, run_all
from core.p05_data.transforms import (
    build_transforms, Mosaic, MixUp, IRSimulation, DetectionTransform,
    _to_v2_sample, _from_v2_sample,
)
from torchvision.transforms import v2
from torchvision import tv_tensors
import torch
from core.p05_data.detection_dataset import YOLOXDataset, build_dataloader
from utils.config import load_config, resolve_path

OUTPUTS = Path(__file__).resolve().parent / "outputs" / "01_detection_dataset"
OUTPUTS.mkdir(parents=True, exist_ok=True)

DATA_CONFIG_PATH = str(ROOT / "configs" / "_test" / "05_data.yaml")
TRAIN_CONFIG_PATH = str(ROOT / "configs" / "_test" / "06_training.yaml")


def test_build_transforms_train():
    """Build training transform pipeline with real config."""
    config = load_config(TRAIN_CONFIG_PATH)
    aug_config = config.get("augmentation", {})
    tfm = build_transforms(aug_config, is_train=True, input_size=(640, 640))
    assert tfm is not None
    assert hasattr(tfm, "__call__")
    print(f"    Train transforms: {len(tfm.transforms)} stages")


def test_build_transforms_eval():
    """Build eval transform pipeline (no augmentation)."""
    config = load_config(TRAIN_CONFIG_PATH)
    aug_config = config.get("augmentation", {})
    tfm = build_transforms(aug_config, is_train=False, input_size=(640, 640))
    assert tfm is not None
    print(f"    Eval transforms: {len(tfm.transforms)} stages")


def test_dataset_load():
    """Load YOLOXDataset with real images."""
    data_config = load_config(DATA_CONFIG_PATH)
    train_config = load_config(TRAIN_CONFIG_PATH)
    aug_config = train_config.get("augmentation", {})
    tfm = build_transforms(aug_config, is_train=True, input_size=(640, 640))

    ds = YOLOXDataset(
        data_config, split="train", transforms=tfm,
        base_dir=Path(DATA_CONFIG_PATH).parent,
    )
    assert len(ds) > 0, f"Dataset is empty"
    print(f"    Dataset length: {len(ds)}")

    # Get one sample
    img, targets, path = ds[0]
    assert img.ndim == 3, f"Image ndim: {img.ndim}"
    assert img.shape[0] == 3, f"Image channels: {img.shape[0]}"
    print(f"    Sample shape: {img.shape}, targets: {targets.shape}, path: {Path(path).name}")


def test_dataloader():
    """Build dataloader and iterate one batch."""
    data_config = load_config(DATA_CONFIG_PATH)
    train_config = load_config(TRAIN_CONFIG_PATH)

    loader = build_dataloader(
        data_config, "train", train_config,
        base_dir=Path(DATA_CONFIG_PATH).parent,
    )
    batch = next(iter(loader))
    assert isinstance(batch, dict), f"Batch type: {type(batch)}"
    assert "images" in batch, f"Missing 'images' key"
    assert "targets" in batch, f"Missing 'targets' key"
    assert "paths" in batch, f"Missing 'paths' key"
    print(f"    Batch images: {batch['images'].shape}, targets: {len(batch['targets'])} items")


def test_save_augmented_samples():
    """Apply transforms to real images and save 8 samples."""
    data_config = load_config(DATA_CONFIG_PATH)
    train_config = load_config(TRAIN_CONFIG_PATH)
    aug_config = train_config.get("augmentation", {})

    # Build training transforms (with augmentation)
    tfm = build_transforms(aug_config, is_train=True, input_size=(640, 640))
    ds = YOLOXDataset(
        data_config, split="train", transforms=tfm,
        base_dir=Path(DATA_CONFIG_PATH).parent,
    )

    saved = 0
    for i in range(min(8, len(ds))):
        img_tensor, targets, path = ds[i]
        # Convert CHW float tensor to HWC uint8 for saving
        img_np = img_tensor.numpy().transpose(1, 2, 0)  # CHW -> HWC
        # Denormalize (approximate — just for visualization)
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        out_path = OUTPUTS / f"sample_{i:02d}.png"
        cv2.imwrite(str(out_path), img_np)
        saved += 1

    assert saved > 0, "No samples saved"
    print(f"    Saved {saved} augmented samples to {OUTPUTS}")


def _get_first_train_image_path():
    """Helper: return path to first training image."""
    data_config = load_config(DATA_CONFIG_PATH)
    base_dir = Path(DATA_CONFIG_PATH).parent
    dataset_path = resolve_path(data_config["path"], base_dir)
    train_images_dir = dataset_path / data_config["train"]
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(p for p in train_images_dir.iterdir() if p.suffix.lower() in exts)
    assert len(images) > 0, "No training images found"
    return images[0]


def test_resize_transform():
    """Apply v2.Resize to a real image tensor and verify output shape."""
    img_path = _get_first_train_image_path()
    image = cv2.imread(str(img_path))
    assert image is not None, f"Failed to read image: {img_path}"
    original_h, original_w = image.shape[:2]

    # Convert BGR -> RGB, HWC uint8 -> CHW tv_tensors.Image
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    chw = torch.from_numpy(rgb.transpose(2, 0, 1))  # HWC -> CHW
    img_tv = tv_tensors.Image(chw)

    resize = v2.Resize((416, 416), antialias=True)
    resized = resize(img_tv)

    assert resized.shape == (3, 416, 416), f"Expected (3, 416, 416), got {resized.shape}"
    print(f"    Resized ({original_h}, {original_w}) -> {resized.shape[1:]}")


def test_color_jitter():
    """Apply v2.ColorJitter (replaces HSVAugment) and verify shape/dtype."""
    img_path = _get_first_train_image_path()
    image = cv2.imread(str(img_path))
    assert image is not None, f"Failed to read image: {img_path}"

    # Convert BGR -> RGB, HWC uint8 -> CHW tv_tensors.Image
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    chw = torch.from_numpy(rgb.transpose(2, 0, 1))
    img_tv = tv_tensors.Image(chw)
    original_shape = img_tv.shape

    jitter = v2.ColorJitter(hue=0.015, saturation=0.7, brightness=0.4)
    result = jitter(img_tv)

    assert result.shape == original_shape, f"Shape changed: {original_shape} -> {result.shape}"
    assert result.dtype == torch.uint8, f"Expected uint8, got {result.dtype}"
    print(f"    ColorJitter output shape: {result.shape}, dtype: {result.dtype}")


def test_sanitize_targets():
    """Test v2.ClampBoundingBoxes + v2.SanitizeBoundingBoxes clips and removes degenerate boxes."""
    canvas_size = (100, 100)  # H, W

    # Box 0: extends beyond canvas (x1=-10, y1=20, x2=110, y2=60) -> should be clamped
    # Box 1: zero-area box (x1=50, y1=50, x2=50, y2=50) -> should be removed
    boxes = tv_tensors.BoundingBoxes(
        torch.tensor([[-10.0, 20.0, 110.0, 60.0],
                       [50.0, 50.0, 50.0, 50.0]]),
        format=tv_tensors.BoundingBoxFormat.XYXY,
        canvas_size=canvas_size,
    )
    labels = torch.tensor([0, 1])

    sample = {"boxes": boxes, "labels": labels}

    # Apply clamp then sanitize
    clamp = v2.ClampBoundingBoxes()
    sanitize = v2.SanitizeBoundingBoxes(min_area=1.0)

    sample = clamp(sample)
    sample = sanitize(sample)

    result_boxes = sample["boxes"]
    result_labels = sample["labels"]

    # Zero-area box should be removed
    assert len(result_boxes) == 1, f"Expected 1 box after sanitize, got {len(result_boxes)}"

    # Remaining box should be clamped to canvas [0, 100]
    x1, y1, x2, y2 = result_boxes[0].tolist()
    assert x1 >= 0, f"x1 should be >= 0, got {x1}"
    assert y1 >= 0, f"y1 should be >= 0, got {y1}"
    assert x2 <= canvas_size[1], f"x2 should be <= {canvas_size[1]}, got {x2}"
    assert y2 <= canvas_size[0], f"y2 should be <= {canvas_size[0]}, got {y2}"
    assert x2 > x1 and y2 > y1, f"Box should have positive area: ({x1}, {y1}, {x2}, {y2})"
    print(f"    Sanitized: 2 input boxes -> {len(result_boxes)} valid box, clamped to ({x1}, {y1}, {x2}, {y2})")


def test_dataset_label_validation():
    """Verify YOLOXDataset returns valid tensors for a real sample."""
    data_config = load_config(DATA_CONFIG_PATH)
    train_config = load_config(TRAIN_CONFIG_PATH)
    aug_config = train_config.get("augmentation", {})
    tfm = build_transforms(aug_config, is_train=False, input_size=(640, 640))

    ds = YOLOXDataset(
        data_config, split="train", transforms=tfm,
        base_dir=Path(DATA_CONFIG_PATH).parent,
    )
    assert len(ds) > 0, "Dataset is empty"

    img, targets, path = ds[0]
    assert img.ndim == 3, f"Image ndim: {img.ndim}"
    assert img.shape[0] == 3, f"Expected CHW format with 3 channels, got {img.shape}"
    assert hasattr(targets, 'shape'), f"Targets has no shape attribute, type: {type(targets)}"
    print(f"    Sample: image={img.shape}, targets shape={targets.shape}")


def test_random_horizontal_flip():
    """Apply v2.RandomHorizontalFlip with p=1.0 via _to_v2_sample/_from_v2_sample and verify cx is mirrored."""
    img_path = _get_first_train_image_path()
    image = cv2.imread(str(img_path))
    assert image is not None, f"Failed to read image: {img_path}"

    # Targets in YOLO format: [class_id, cx, cy, w, h] normalised
    targets = np.array([[0, 0.3, 0.5, 0.2, 0.3]])

    # Convert to v2 sample dict
    sample = _to_v2_sample(image, targets)

    # Apply horizontal flip (p=1.0 guarantees flip)
    flip = v2.RandomHorizontalFlip(p=1.0)
    flipped_sample = flip(sample)

    # Convert back — returns (CHW tensor, (N,5) tensor)
    flipped_img, flipped_targets = _from_v2_sample(flipped_sample)

    # CHW tensor shape should match original image dimensions
    assert flipped_img.shape[1] == image.shape[0] and flipped_img.shape[2] == image.shape[1], (
        f"Shape mismatch: expected H={image.shape[0]}, W={image.shape[1]}, got {flipped_img.shape}"
    )
    flipped_cx = flipped_targets[0, 1].item()
    assert abs(flipped_cx - 0.7) < 0.05, (
        f"Expected flipped cx ~ 0.7, got {flipped_cx}"
    )
    print(f"    Flipped cx: 0.3 -> {flipped_cx:.4f}")


def test_random_affine_preserves_shape():
    """Apply v2.RandomAffine via _to_v2_sample/_from_v2_sample and verify output shape and target validity."""
    img_path = _get_first_train_image_path()
    image = cv2.imread(str(img_path))
    assert image is not None, f"Failed to read image: {img_path}"
    image = cv2.resize(image, (640, 640))
    targets = np.array([[0, 0.5, 0.5, 0.3, 0.3]])

    # Convert to v2 sample dict
    sample = _to_v2_sample(image, targets)

    # Apply affine + clamp + sanitize
    transform = v2.Compose([
        v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        v2.ClampBoundingBoxes(),
        v2.SanitizeBoundingBoxes(min_area=1.0),
    ])
    result_sample = transform(sample)

    # Convert back — returns (CHW tensor, (N,5) tensor)
    result_img, result_targets = _from_v2_sample(result_sample)

    assert result_img.shape == (3, 640, 640), (
        f"Expected (3, 640, 640), got {result_img.shape}"
    )
    if len(result_targets) > 0:
        coords = result_targets[:, 1:].numpy()
        assert np.all(coords >= 0) and np.all(coords <= 1), (
            f"Target coords out of [0, 1]: {coords}"
        )
    print(f"    Affine output shape: {result_img.shape}, targets: {len(result_targets)}")


def test_mosaic_output():
    """Apply Mosaic transform using dataset and verify output through build_transforms."""
    data_config = load_config(DATA_CONFIG_PATH)
    train_config = load_config(TRAIN_CONFIG_PATH)
    aug_config = train_config.get("augmentation", {})

    # Build training transforms with mosaic enabled
    mosaic_config = {**aug_config, "mosaic": True}
    tfm = build_transforms(mosaic_config, is_train=True, input_size=(640, 640))

    ds = YOLOXDataset(
        data_config, split="train", transforms=tfm,
        base_dir=Path(DATA_CONFIG_PATH).parent,
    )
    assert len(ds) > 0, "Dataset is empty"

    # Get a sample — mosaic is applied inside the pipeline
    img_tensor, targets, path = ds[0]

    assert img_tensor.shape == (3, 640, 640), (
        f"Expected (3, 640, 640), got {img_tensor.shape}"
    )
    if len(targets) > 0:
        # Targets should have valid normalised coords
        if hasattr(targets, 'numpy'):
            tgt_np = targets.numpy()
        else:
            tgt_np = targets
        coords = tgt_np[:, 1:]
        assert np.all(coords >= 0) and np.all(coords <= 1), (
            f"Mosaic target coords out of [0, 1]: {coords}"
        )
    print(f"    Mosaic output: {img_tensor.shape}, targets: {len(targets)}")


def test_ir_simulation():
    """Apply IRSimulation with ir_prob=1.0 and verify grayscale output via _to_v2_sample."""
    img_path = _get_first_train_image_path()
    image = cv2.imread(str(img_path))
    assert image is not None, f"Failed to read image: {img_path}"
    targets = np.zeros((0, 5))

    # Convert to v2 sample, apply IRSimulation, convert back
    sample = _to_v2_sample(image, targets)
    ir = IRSimulation(ir_prob=1.0, low_light_prob=0.0)
    result_sample = ir(sample)
    result_img, _ = _from_v2_sample(result_sample)

    # result_img is CHW tensor — check channels are equal (grayscale)
    ch0 = result_img[0].numpy()
    ch1 = result_img[1].numpy()
    ch2 = result_img[2].numpy()
    assert np.allclose(ch0, ch1), (
        "IR simulation: channels 0 and 1 should be equal (grayscale)"
    )
    assert np.allclose(ch0, ch2), (
        "IR simulation: channels 0 and 2 should be equal (grayscale)"
    )
    print(f"    IR simulation: all 3 channels are equal (grayscale)")


def test_eval_pipeline_chain():
    """Test full eval pipeline (resize + normalize + to_tensor) via build_transforms."""
    img_path = _get_first_train_image_path()
    image = cv2.imread(str(img_path))
    assert image is not None, f"Failed to read image: {img_path}"
    image = cv2.resize(image, (640, 640))
    targets = np.zeros((0, 5))

    # Build eval transforms — handles BGR->RGB, resize, normalize, to_tensor internally
    tfm = build_transforms(config={}, is_train=False, input_size=(640, 640))
    result, result_targets = tfm(image, targets)

    assert isinstance(result, torch.Tensor), f"Expected torch.Tensor, got {type(result)}"
    assert result.shape == (3, 640, 640), f"Expected (3, 640, 640), got {result.shape}"
    assert result.dtype == torch.float32, f"Expected float32, got {result.dtype}"
    print(f"    Eval pipeline output: {result.shape}, dtype: {result.dtype}")


if __name__ == "__main__":
    run_all([
        ("build_transforms_train", test_build_transforms_train),
        ("build_transforms_eval", test_build_transforms_eval),
        ("dataset_load", test_dataset_load),
        ("dataloader", test_dataloader),
        ("save_augmented_samples", test_save_augmented_samples),
        ("resize_transform", test_resize_transform),
        ("color_jitter", test_color_jitter),
        ("sanitize_targets", test_sanitize_targets),
        ("dataset_label_validation", test_dataset_label_validation),
        ("random_horizontal_flip", test_random_horizontal_flip),
        ("random_affine_preserves_shape", test_random_affine_preserves_shape),
        ("mosaic_output", test_mosaic_output),
        ("ir_simulation", test_ir_simulation),
        ("eval_pipeline_chain", test_eval_pipeline_chain),
    ], title="Test 03: Data Transforms")

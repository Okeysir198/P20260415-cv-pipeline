"""Test 03: Segmentation Dataset — test transforms, collate, and dataloader."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.p05_data.segmentation_dataset import (
    SegmentationDataset,
    build_segmentation_dataloader,
    build_segmentation_transforms,
    segmentation_collate_fn,
)


# ---------------------------------------------------------------------------
# Helper to create a tiny segmentation dataset
# ---------------------------------------------------------------------------


def _create_segmentation_dataset(
    tmp_path: Path, num_classes: int = 3, num_images: int = 6
) -> dict:
    """Create a minimal segmentation dataset with images and masks.

    Creates train/ and val/ splits with images/*.jpg and masks/*.png.
    Masks are grayscale with pixel values as class IDs (0 to num_classes-1).

    Returns:
        data_config dict ready for SegmentationDataset.
    """
    class_names = {i: f"class_{i}" for i in range(num_classes)}
    for split in ("train", "val"):
        img_dir = tmp_path / split / "images"
        mask_dir = tmp_path / split / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        n = num_images if split == "train" else max(2, num_images // 2)
        for j in range(n):
            # Random BGR image
            img = np.random.randint(0, 255, (64, 80, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / f"img_{j:03d}.jpg"), img)
            # Mask with class IDs
            mask = np.random.randint(0, num_classes, (64, 80), dtype=np.uint8)
            cv2.imwrite(str(mask_dir / f"img_{j:03d}.png"), mask)

    data_config = {
        "path": str(tmp_path),
        "train": "train",
        "val": "val",
        "names": class_names,
        "num_classes": num_classes,
        "input_size": [128, 128],
    }
    return data_config


# ---------------------------------------------------------------------------
# Tests: SegmentationDataset
# ---------------------------------------------------------------------------


class TestSegmentationDataset:
    def test_load_dataset(self, tmp_path):
        config = _create_segmentation_dataset(tmp_path)
        ds = SegmentationDataset(config, split="train", base_dir=tmp_path)
        assert len(ds) == 6
        assert ds.num_classes == 3

    def test_getitem_types(self, tmp_path):
        config = _create_segmentation_dataset(tmp_path)
        ds = SegmentationDataset(config, split="train", base_dir=tmp_path)
        img, mask, path_str = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.dtype == torch.float32
        assert img.shape == (3, 128, 128)
        assert isinstance(mask, torch.Tensor)
        assert mask.dtype == torch.int64
        assert mask.ndim == 2
        assert isinstance(path_str, str)

    def test_mask_range(self, tmp_path):
        config = _create_segmentation_dataset(tmp_path, num_classes=3)
        ds = SegmentationDataset(config, split="train", base_dir=tmp_path)
        for i in range(len(ds)):
            _, mask, _ = ds[i]
            assert mask.min() >= 0
            assert mask.max() < 3

    def test_val_split(self, tmp_path):
        config = _create_segmentation_dataset(tmp_path, num_images=6)
        ds = SegmentationDataset(config, split="val", base_dir=tmp_path)
        assert len(ds) == 3  # max(2, 6 // 2)

    def test_with_transforms(self, tmp_path):
        config = _create_segmentation_dataset(tmp_path)
        tfm = build_segmentation_transforms(is_train=False, input_size=(128, 128))
        ds = SegmentationDataset(config, split="train", transforms=tfm, base_dir=tmp_path)
        img, mask, _ = ds[0]
        assert img.shape == (3, 128, 128)
        assert img.dtype == torch.float32
        assert mask.dtype == torch.int64

    def test_missing_mask_returns_zeros(self, tmp_path):
        """When a mask file is absent, dataset should return a zero mask."""
        config = _create_segmentation_dataset(tmp_path, num_images=2)
        # Delete one mask file
        mask_dir = tmp_path / "train" / "masks"
        mask_files = sorted(mask_dir.glob("*.png"))
        assert len(mask_files) > 0
        mask_files[0].unlink()

        ds = SegmentationDataset(config, split="train", base_dir=tmp_path)
        _, mask, _ = ds[0]
        assert mask.sum() == 0  # all zeros since mask was missing


# ---------------------------------------------------------------------------
# Tests: transforms
# ---------------------------------------------------------------------------


class TestSegmentationTransforms:
    def test_train_transform_shape(self):
        tfm = build_segmentation_transforms(is_train=True, input_size=(128, 128))
        img_bgr = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        mask_np = np.random.randint(0, 3, (200, 300), dtype=np.uint8)
        img_out, mask_out = tfm(img_bgr, mask_np)
        assert img_out.shape == (3, 128, 128)
        assert img_out.dtype == torch.float32

    def test_val_transform_shape(self):
        tfm = build_segmentation_transforms(is_train=False, input_size=(256, 256))
        img_bgr = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        mask_np = np.random.randint(0, 2, (100, 150), dtype=np.uint8)
        img_out, mask_out = tfm(img_bgr, mask_np)
        assert img_out.shape == (3, 256, 256)
        assert mask_out.dtype == torch.int64

    def test_matching_spatial_dims(self):
        """Image and mask should have the same spatial dimensions after transform."""
        tfm = build_segmentation_transforms(is_train=True, input_size=(128, 192))
        img_bgr = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        mask_np = np.random.randint(0, 5, (300, 400), dtype=np.uint8)
        img_out, mask_out = tfm(img_bgr, mask_np)
        assert img_out.shape[1:] == mask_out.shape  # (H, W) matches


# ---------------------------------------------------------------------------
# Tests: collate and dataloader
# ---------------------------------------------------------------------------


class TestCollateAndDataloader:
    def test_collate_fn(self):
        batch = [
            (torch.randn(3, 128, 128), torch.randint(0, 3, (128, 128), dtype=torch.long), "/a.jpg"),
            (torch.randn(3, 128, 128), torch.randint(0, 3, (128, 128), dtype=torch.long), "/b.jpg"),
        ]
        result = segmentation_collate_fn(batch)
        assert result["images"].shape == (2, 3, 128, 128)
        assert len(result["targets"]) == 2
        assert result["targets"][0].shape == (128, 128)
        assert len(result["paths"]) == 2
        assert result["paths"][0] == "/a.jpg"

    def test_build_dataloader(self, tmp_path):
        data_config = _create_segmentation_dataset(tmp_path)
        training_config = {"data": {"batch_size": 2, "num_workers": 0, "pin_memory": False}}
        loader = build_segmentation_dataloader(
            data_config, "train", training_config, base_dir=tmp_path
        )
        batch = next(iter(loader))
        assert batch["images"].shape == (2, 3, 128, 128)
        assert len(batch["targets"]) == 2
        assert batch["targets"][0].dtype == torch.int64
        assert len(batch["paths"]) == 2


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_invalid_split_raises(self, tmp_path):
        config = _create_segmentation_dataset(tmp_path)
        with pytest.raises(ValueError, match="split must be"):
            SegmentationDataset(config, split="invalid", base_dir=tmp_path)

    def test_missing_split_key_raises(self, tmp_path):
        config = _create_segmentation_dataset(tmp_path)
        del config["val"]
        with pytest.raises(ValueError, match="data config missing"):
            SegmentationDataset(config, split="val", base_dir=tmp_path)

    def test_empty_image_dir_raises(self, tmp_path):
        # Create structure with empty images dir
        (tmp_path / "empty" / "train" / "images").mkdir(parents=True)
        (tmp_path / "empty" / "train" / "masks").mkdir(parents=True)
        config = {
            "path": str(tmp_path / "empty"),
            "train": "train",
            "names": {0: "bg"},
            "num_classes": 1,
            "input_size": [128, 128],
        }
        with pytest.raises(FileNotFoundError, match="No images found"):
            SegmentationDataset(config, split="train", base_dir=tmp_path)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


def run_test():
    """Run all tests in this file (standalone mode)."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v"],
        cwd=str(ROOT),
    )
    return result.returncode == 0


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)

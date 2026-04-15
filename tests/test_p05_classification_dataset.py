"""Test 02: Classification Dataset — test with both folder-based and label-file layouts."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.p05_data.classification_dataset import (
    ClassificationDataset,
    build_classification_dataloader,
    build_classification_transforms,
    classification_collate_fn,
)


# ---------------------------------------------------------------------------
# Helpers to create tiny temp datasets
# ---------------------------------------------------------------------------

def _create_folder_dataset(tmp_path: Path, num_classes: int = 3, imgs_per_class: int = 4):
    """Create a folder-based classification dataset in tmp_path."""
    class_names = {i: f"class_{i}" for i in range(num_classes)}
    for split in ("train", "val"):
        for idx, name in class_names.items():
            class_dir = tmp_path / split / name
            class_dir.mkdir(parents=True, exist_ok=True)
            n = imgs_per_class if split == "train" else max(1, imgs_per_class // 2)
            for j in range(n):
                img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                cv2.imwrite(str(class_dir / f"img_{j:03d}.jpg"), img)

    data_config = {
        "path": str(tmp_path),
        "train": "train",
        "val": "val",
        "names": class_names,
        "num_classes": num_classes,
        "input_size": [224, 224],
        "layout": "folder",
    }
    return data_config


def _create_label_file_dataset(tmp_path: Path, num_classes: int = 2, num_images: int = 6):
    """Create a label-file (YOLO-style) classification dataset in tmp_path."""
    class_names = {i: f"cls_{i}" for i in range(num_classes)}
    for split in ("train", "val"):
        img_dir = tmp_path / split / "images"
        lbl_dir = tmp_path / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        n = num_images if split == "train" else max(2, num_images // 2)
        for j in range(n):
            img = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / f"img_{j:03d}.jpg"), img)
            cls_id = j % num_classes
            (lbl_dir / f"img_{j:03d}.txt").write_text(str(cls_id))

    data_config = {
        "path": str(tmp_path),
        "train": "train",
        "val": "val",
        "names": class_names,
        "num_classes": num_classes,
        "input_size": [224, 224],
        "layout": "yolo",
    }
    return data_config


# ---------------------------------------------------------------------------
# Tests: folder layout
# ---------------------------------------------------------------------------

class TestFolderLayout:
    def test_load_folder_dataset(self, tmp_path):
        config = _create_folder_dataset(tmp_path)
        ds = ClassificationDataset(config, split="train")
        assert len(ds) == 12  # 3 classes * 4 images
        assert ds.layout == "folder"

    def test_getitem_returns_correct_types(self, tmp_path):
        config = _create_folder_dataset(tmp_path)
        ds = ClassificationDataset(config, split="train")
        img, label, path = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 224, 224)
        assert img.dtype == torch.float32
        assert isinstance(label, torch.Tensor)
        assert label.dtype == torch.long
        assert 0 <= label.item() < 3
        assert isinstance(path, str)

    def test_val_split(self, tmp_path):
        config = _create_folder_dataset(tmp_path)
        ds = ClassificationDataset(config, split="val")
        # 3 classes * 2 images each
        assert len(ds) == 6

    def test_with_transforms(self, tmp_path):
        config = _create_folder_dataset(tmp_path)
        tfm = build_classification_transforms(is_train=True, input_size=(224, 224))
        ds = ClassificationDataset(config, split="train", transforms=tfm)
        img, label, path = ds[0]
        assert img.shape == (3, 224, 224)
        assert img.dtype == torch.float32


# ---------------------------------------------------------------------------
# Tests: label-file layout
# ---------------------------------------------------------------------------

class TestLabelFileLayout:
    def test_load_label_file_dataset(self, tmp_path):
        config = _create_label_file_dataset(tmp_path)
        ds = ClassificationDataset(config, split="train")
        assert len(ds) == 6
        assert ds.layout == "yolo"

    def test_getitem_returns_correct_label(self, tmp_path):
        config = _create_label_file_dataset(tmp_path, num_classes=2, num_images=4)
        ds = ClassificationDataset(config, split="train")
        for i in range(len(ds)):
            _, label, _ = ds[i]
            assert 0 <= label.item() < 2


# ---------------------------------------------------------------------------
# Tests: auto-detection
# ---------------------------------------------------------------------------

class TestAutoDetect:
    def test_auto_detects_folder(self, tmp_path):
        config = _create_folder_dataset(tmp_path)
        config["layout"] = "auto"
        ds = ClassificationDataset(config, split="train")
        assert ds.layout == "folder"

    def test_auto_detects_yolo(self, tmp_path):
        config = _create_label_file_dataset(tmp_path)
        config["layout"] = "auto"
        ds = ClassificationDataset(config, split="train")
        assert ds.layout == "yolo"


# ---------------------------------------------------------------------------
# Tests: transforms
# ---------------------------------------------------------------------------

class TestTransforms:
    def test_train_transforms_output_shape(self):
        tfm = build_classification_transforms(is_train=True, input_size=(128, 128))
        img_bgr = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        result = tfm(img_bgr)
        assert result.shape == (3, 128, 128)
        assert result.dtype == torch.float32

    def test_val_transforms_output_shape(self):
        tfm = build_classification_transforms(is_train=False, input_size=(128, 128))
        img_bgr = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        result = tfm(img_bgr)
        assert result.shape == (3, 128, 128)
        assert result.dtype == torch.float32


# ---------------------------------------------------------------------------
# Tests: collate and dataloader
# ---------------------------------------------------------------------------

class TestCollateAndDataloader:
    def test_collate_fn(self):
        batch = [
            (torch.randn(3, 224, 224), torch.tensor(0, dtype=torch.long), "/a.jpg"),
            (torch.randn(3, 224, 224), torch.tensor(1, dtype=torch.long), "/b.jpg"),
        ]
        result = classification_collate_fn(batch)
        assert result["images"].shape == (2, 3, 224, 224)
        assert len(result["targets"]) == 2
        assert len(result["paths"]) == 2

    def test_build_dataloader(self, tmp_path):
        data_config = _create_folder_dataset(tmp_path)
        training_config = {"data": {"batch_size": 4, "num_workers": 0, "pin_memory": False}}
        loader = build_classification_dataloader(
            data_config, "train", training_config, base_dir=tmp_path
        )
        batch = next(iter(loader))
        assert batch["images"].shape[0] == 4
        assert batch["images"].shape[1:] == (3, 224, 224)
        assert len(batch["targets"]) == 4
        assert len(batch["paths"]) == 4


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_invalid_split_raises(self, tmp_path):
        config = _create_folder_dataset(tmp_path)
        with pytest.raises(ValueError, match="split must be"):
            ClassificationDataset(config, split="invalid")

    def test_missing_split_key_raises(self, tmp_path):
        config = _create_folder_dataset(tmp_path)
        del config["val"]
        with pytest.raises(ValueError, match="data config missing"):
            ClassificationDataset(config, split="val")

    def test_empty_dataset_raises(self, tmp_path):
        empty_dir = tmp_path / "empty" / "train"
        empty_dir.mkdir(parents=True)
        config = {
            "path": str(tmp_path / "empty"),
            "train": "train",
            "names": {0: "a"},
            "num_classes": 1,
            "input_size": [224, 224],
            "layout": "folder",
        }
        with pytest.raises(FileNotFoundError, match="No samples found"):
            ClassificationDataset(config, split="train")

    def test_skips_out_of_range_class_in_label_file(self, tmp_path):
        """Label files with class_id >= num_classes should be skipped."""
        img_dir = tmp_path / "train" / "images"
        lbl_dir = tmp_path / "train" / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        # Valid image with class 0
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "good.jpg"), img)
        (lbl_dir / "good.txt").write_text("0")

        # Image with out-of-range class
        cv2.imwrite(str(img_dir / "bad.jpg"), img)
        (lbl_dir / "bad.txt").write_text("99")

        config = {
            "path": str(tmp_path),
            "train": "train",
            "names": {0: "a"},
            "num_classes": 1,
            "input_size": [224, 224],
            "layout": "yolo",
        }
        ds = ClassificationDataset(config, split="train")
        assert len(ds) == 1  # only the valid sample

"""Test 05: COCO Detection Dataset — test loading, getitem, collate, and dataloader."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.p05_data.coco_dataset import (
    COCODetectionDataset,
    build_coco_dataloader,
    _build_coco_category_map,
)
from core.p05_data.detection_dataset import collate_fn


# ---------------------------------------------------------------------------
# Helpers to create a tiny COCO-format dataset in a temp directory
# ---------------------------------------------------------------------------


def _create_coco_dataset(
    tmp_path: Path,
    num_images: int = 6,
    num_classes: int = 2,
    objs_per_image: int = 3,
    img_size: int = 128,
) -> dict:
    """Create a minimal COCO-format dataset with synthetic images.

    Returns the data_config dict ready for ``COCODetectionDataset``.
    """
    class_names = {i: f"class_{i}" for i in range(num_classes)}

    for split, n_imgs in [("train", num_images), ("val", max(2, num_images // 2))]:
        img_dir = tmp_path / split
        ann_dir = tmp_path / "annotations"
        img_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)

        images_list = []
        annotations_list = []
        ann_id = 1

        for i in range(n_imgs):
            # Create a random image
            img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
            fname = f"img_{i:04d}.jpg"
            cv2.imwrite(str(img_dir / fname), img)

            img_entry = {
                "id": i + 1,
                "file_name": fname,
                "width": img_size,
                "height": img_size,
            }
            images_list.append(img_entry)

            # Create annotations for this image
            for j in range(objs_per_image):
                cls_id = j % num_classes
                # Random box in absolute pixels — keep inside image
                bw = np.random.randint(10, img_size // 2)
                bh = np.random.randint(10, img_size // 2)
                x = np.random.randint(0, img_size - bw)
                y = np.random.randint(0, img_size - bh)
                ann = {
                    "id": ann_id,
                    "image_id": i + 1,
                    "category_id": cls_id,
                    "bbox": [int(x), int(y), int(bw), int(bh)],
                    "area": int(bw * bh),
                    "iscrowd": 0,
                }
                annotations_list.append(ann)
                ann_id += 1

        categories = [
            {"id": cid, "name": cname}
            for cid, cname in class_names.items()
        ]

        coco_json = {
            "images": images_list,
            "annotations": annotations_list,
            "categories": categories,
        }

        ann_file = ann_dir / f"instances_{split}.json"
        with open(ann_file, "w") as f:
            json.dump(coco_json, f)

    data_config = {
        "path": str(tmp_path),
        "annotation_format": "coco",
        "train": "train",
        "val": "val",
        "train_ann": "annotations/instances_train.json",
        "val_ann": "annotations/instances_val.json",
        "names": class_names,
        "num_classes": num_classes,
        "input_size": [64, 64],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    return data_config


# ---------------------------------------------------------------------------
# Tests: basic loading
# ---------------------------------------------------------------------------


class TestCOCODatasetLoading:
    def test_load_train_split(self, tmp_path):
        config = _create_coco_dataset(tmp_path)
        ds = COCODetectionDataset(config, split="train", base_dir=tmp_path)
        assert len(ds) == 6

    def test_load_val_split(self, tmp_path):
        config = _create_coco_dataset(tmp_path)
        ds = COCODetectionDataset(config, split="val", base_dir=tmp_path)
        assert len(ds) == 3

    def test_class_attributes(self, tmp_path):
        config = _create_coco_dataset(tmp_path, num_classes=3)
        ds = COCODetectionDataset(config, split="train", base_dir=tmp_path)
        assert ds.num_classes == 3
        assert ds.input_size == (64, 64)
        assert len(ds.class_names) == 3


# ---------------------------------------------------------------------------
# Tests: get_raw_item and __getitem__
# ---------------------------------------------------------------------------


class TestGetItem:
    def test_get_raw_item_returns_dict(self, tmp_path):
        config = _create_coco_dataset(tmp_path)
        ds = COCODetectionDataset(config, split="train", base_dir=tmp_path)
        raw = ds.get_raw_item(0)
        assert isinstance(raw, dict)
        assert "image" in raw
        assert "targets" in raw
        assert isinstance(raw["image"], np.ndarray)
        assert raw["image"].ndim == 3
        assert raw["image"].dtype == np.uint8

    def test_get_raw_item_targets_shape(self, tmp_path):
        config = _create_coco_dataset(tmp_path, objs_per_image=2)
        ds = COCODetectionDataset(config, split="train", base_dir=tmp_path)
        raw = ds.get_raw_item(0)
        targets = raw["targets"]
        assert isinstance(targets, np.ndarray)
        assert targets.dtype == np.float32
        assert targets.ndim == 2
        assert targets.shape[1] == 5
        # Should have 2 objects per image
        assert targets.shape[0] == 2

    def test_targets_normalised_range(self, tmp_path):
        config = _create_coco_dataset(tmp_path)
        ds = COCODetectionDataset(config, split="train", base_dir=tmp_path)
        for i in range(len(ds)):
            raw = ds.get_raw_item(i)
            targets = raw["targets"]
            if len(targets) > 0:
                # class_id should be valid
                assert np.all(targets[:, 0] >= 0)
                assert np.all(targets[:, 0] < config["num_classes"])
                # coords should be in [0, 1]
                assert np.all(targets[:, 1:] >= 0.0)
                assert np.all(targets[:, 1:] <= 1.0)

    def test_getitem_no_transforms(self, tmp_path):
        config = _create_coco_dataset(tmp_path)
        ds = COCODetectionDataset(config, split="train", base_dir=tmp_path)
        img, targets, path = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.dtype == torch.float32
        assert img.ndim == 3  # (C, H, W)
        assert isinstance(targets, torch.Tensor)
        assert targets.dtype == torch.float32
        assert targets.ndim == 2
        assert targets.shape[1] == 5
        assert isinstance(path, str)

    def test_getitem_with_transforms(self, tmp_path):
        from core.p05_data.transforms import build_transforms

        config = _create_coco_dataset(tmp_path)
        tfm = build_transforms(
            config={},
            is_train=False,
            input_size=(64, 64),
        )
        ds = COCODetectionDataset(
            config, split="train", transforms=tfm, base_dir=tmp_path,
        )
        img, targets, path = ds[0]
        assert img.shape == (3, 64, 64)
        assert img.dtype == torch.float32
        assert targets.ndim == 2
        assert targets.shape[1] == 5


# ---------------------------------------------------------------------------
# Tests: collate and dataloader
# ---------------------------------------------------------------------------


class TestCollateAndDataloader:
    def test_collate_fn(self):
        batch = [
            (
                torch.randn(3, 64, 64),
                torch.tensor([[0, 0.5, 0.5, 0.2, 0.3]], dtype=torch.float32),
                "/a.jpg",
            ),
            (
                torch.randn(3, 64, 64),
                torch.zeros((0, 5), dtype=torch.float32),
                "/b.jpg",
            ),
        ]
        result = collate_fn(batch)
        assert result["images"].shape == (2, 3, 64, 64)
        assert len(result["targets"]) == 2
        assert result["targets"][0].shape == (1, 5)
        assert result["targets"][1].shape == (0, 5)
        assert len(result["paths"]) == 2

    def test_build_dataloader(self, tmp_path):
        config = _create_coco_dataset(tmp_path, num_images=8)
        training_config = {
            "augmentation": {},
            "data": {
                "batch_size": 4,
                "num_workers": 0,
                "pin_memory": False,
            },
        }
        loader = build_coco_dataloader(
            config, "train", training_config, base_dir=tmp_path,
        )
        batch = next(iter(loader))
        assert batch["images"].shape[0] == 4
        assert batch["images"].shape[1:] == (3, 64, 64)
        assert len(batch["targets"]) == 4
        assert len(batch["paths"]) == 4

    def test_val_dataloader_no_shuffle(self, tmp_path):
        config = _create_coco_dataset(tmp_path)
        training_config = {
            "augmentation": {},
            "data": {
                "batch_size": 2,
                "num_workers": 0,
                "pin_memory": False,
            },
        }
        loader = build_coco_dataloader(
            config, "val", training_config, base_dir=tmp_path,
        )
        # Just verify it iterates without error
        batch = next(iter(loader))
        assert batch["images"].ndim == 4


# ---------------------------------------------------------------------------
# Tests: category mapping
# ---------------------------------------------------------------------------


class TestCategoryMapping:
    def test_identity_mapping(self):
        cats = [
            {"id": 0, "name": "person"},
            {"id": 1, "name": "car"},
        ]
        names = {0: "person", 1: "car"}
        mapping = _build_coco_category_map(cats, names)
        assert mapping == {0: 0, 1: 1}

    def test_remapped_ids(self):
        cats = [
            {"id": 1, "name": "person"},
            {"id": 3, "name": "car"},
        ]
        names = {0: "person", 1: "car"}
        mapping = _build_coco_category_map(cats, names)
        assert mapping == {1: 0, 3: 1}

    def test_case_insensitive(self):
        cats = [{"id": 5, "name": "Person"}]
        names = {0: "person"}
        mapping = _build_coco_category_map(cats, names)
        assert mapping == {5: 0}

    def test_unmatched_categories_skipped(self):
        cats = [
            {"id": 1, "name": "person"},
            {"id": 2, "name": "bicycle"},
        ]
        names = {0: "person"}
        mapping = _build_coco_category_map(cats, names)
        assert mapping == {1: 0}
        assert 2 not in mapping


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_invalid_split_raises(self, tmp_path):
        config = _create_coco_dataset(tmp_path)
        with pytest.raises(ValueError, match="split must be"):
            COCODetectionDataset(config, split="invalid")

    def test_missing_split_key_raises(self, tmp_path):
        config = _create_coco_dataset(tmp_path)
        del config["val"]
        with pytest.raises(ValueError, match="data config missing"):
            COCODetectionDataset(config, split="val", base_dir=tmp_path)

    def test_missing_ann_key_raises(self, tmp_path):
        config = _create_coco_dataset(tmp_path)
        del config["val_ann"]
        with pytest.raises(ValueError, match="data config missing"):
            COCODetectionDataset(config, split="val", base_dir=tmp_path)

    def test_crowd_annotations_skipped(self, tmp_path):
        """Annotations with iscrowd=1 should be excluded."""
        config = _create_coco_dataset(
            tmp_path, num_images=1, objs_per_image=0,
        )
        # Manually add a crowd annotation to the JSON
        ann_path = tmp_path / "annotations" / "instances_train.json"
        with open(ann_path) as f:
            coco_data = json.load(f)
        coco_data["annotations"] = [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 0,
                "bbox": [10, 10, 30, 30],
                "area": 900,
                "iscrowd": 1,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 0,
                "bbox": [50, 50, 20, 20],
                "area": 400,
                "iscrowd": 0,
            },
        ]
        with open(ann_path, "w") as f:
            json.dump(coco_data, f)

        ds = COCODetectionDataset(config, split="train", base_dir=tmp_path)
        raw = ds.get_raw_item(0)
        # Only the non-crowd annotation should remain
        assert raw["targets"].shape[0] == 1

    def test_zero_size_bbox_skipped(self, tmp_path):
        """Annotations with zero-width or zero-height bbox are dropped."""
        config = _create_coco_dataset(
            tmp_path, num_images=1, objs_per_image=0,
        )
        ann_path = tmp_path / "annotations" / "instances_train.json"
        with open(ann_path) as f:
            coco_data = json.load(f)
        coco_data["annotations"] = [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 0,
                "bbox": [10, 10, 0, 30],
                "area": 0,
                "iscrowd": 0,
            },
        ]
        with open(ann_path, "w") as f:
            json.dump(coco_data, f)

        ds = COCODetectionDataset(config, split="train", base_dir=tmp_path)
        raw = ds.get_raw_item(0)
        assert raw["targets"].shape == (0, 5)

    def test_image_with_no_annotations(self, tmp_path):
        """Images with no annotations should return empty targets."""
        config = _create_coco_dataset(
            tmp_path, num_images=1, objs_per_image=0,
        )
        ds = COCODetectionDataset(config, split="train", base_dir=tmp_path)
        raw = ds.get_raw_item(0)
        assert raw["targets"].shape == (0, 5)
        # __getitem__ should also work
        img, targets, path = ds[0]
        assert targets.shape == (0, 5)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


def run_test():
    """Run tests without pytest (standalone mode)."""
    import tempfile

    passed = 0
    failed = 0
    total = 0

    def check(name, fn):
        nonlocal passed, failed, total
        total += 1
        try:
            with tempfile.TemporaryDirectory() as td:
                fn(Path(td))
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1

    print("=== COCO Detection Dataset Tests ===\n")

    # Loading
    check("load_train_split", lambda p: (
        TestCOCODatasetLoading().test_load_train_split(p)
    ))
    check("load_val_split", lambda p: (
        TestCOCODatasetLoading().test_load_val_split(p)
    ))
    check("class_attributes", lambda p: (
        TestCOCODatasetLoading().test_class_attributes(p)
    ))

    # GetItem
    check("get_raw_item_returns_dict", lambda p: (
        TestGetItem().test_get_raw_item_returns_dict(p)
    ))
    check("get_raw_item_targets_shape", lambda p: (
        TestGetItem().test_get_raw_item_targets_shape(p)
    ))
    check("targets_normalised_range", lambda p: (
        TestGetItem().test_targets_normalised_range(p)
    ))
    check("getitem_no_transforms", lambda p: (
        TestGetItem().test_getitem_no_transforms(p)
    ))
    check("getitem_with_transforms", lambda p: (
        TestGetItem().test_getitem_with_transforms(p)
    ))

    # Collate
    check("collate_fn", lambda _: (
        TestCollateAndDataloader().test_collate_fn()
    ))
    check("build_dataloader", lambda p: (
        TestCollateAndDataloader().test_build_dataloader(p)
    ))

    # Category mapping (no tmp_path needed)
    check("identity_mapping", lambda _: (
        TestCategoryMapping().test_identity_mapping()
    ))
    check("remapped_ids", lambda _: (
        TestCategoryMapping().test_remapped_ids()
    ))
    check("case_insensitive", lambda _: (
        TestCategoryMapping().test_case_insensitive()
    ))
    check("unmatched_categories_skipped", lambda _: (
        TestCategoryMapping().test_unmatched_categories_skipped()
    ))

    # Edge cases
    check("crowd_annotations_skipped", lambda p: (
        TestEdgeCases().test_crowd_annotations_skipped(p)
    ))
    check("zero_size_bbox_skipped", lambda p: (
        TestEdgeCases().test_zero_size_bbox_skipped(p)
    ))
    check("image_with_no_annotations", lambda p: (
        TestEdgeCases().test_image_with_no_annotations(p)
    ))

    print(f"\n--- {passed}/{total} passed, {failed} failed ---")
    return failed == 0


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)

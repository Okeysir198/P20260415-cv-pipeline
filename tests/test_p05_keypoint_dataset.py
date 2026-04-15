"""Test 04: Keypoint Dataset — test with YOLO-pose format labels."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.p05_data.keypoint_dataset import (
    KeypointDataset,
    build_keypoint_dataloader,
    build_keypoint_transforms,
    keypoint_collate_fn,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _create_keypoint_dataset(
    tmp_path: Path,
    num_classes: int = 1,
    num_keypoints: int = 5,
    num_images: int = 6,
):
    """Create a tiny YOLO-pose dataset under tmp_path with train/val splits.

    Returns the data_config dict.
    """
    class_names = {i: f"cls_{i}" for i in range(num_classes)}

    for split in ("train", "val"):
        img_dir = tmp_path / split / "images"
        lbl_dir = tmp_path / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        n = num_images if split == "train" else max(2, num_images // 2)
        for j in range(n):
            # Random BGR image
            img = np.random.randint(0, 255, (80, 120, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / f"img_{j:03d}.jpg"), img)

            # YOLO-pose label: class_id cx cy w h kx1 ky1 v1 ... kxK kyK vK
            lines = []
            num_objects = np.random.randint(1, 4)
            for _ in range(num_objects):
                cls_id = np.random.randint(0, num_classes)
                cx, cy = np.random.uniform(0.2, 0.8, size=2)
                w, h = np.random.uniform(0.05, 0.4, size=2)
                kpts = []
                for _ in range(num_keypoints):
                    kx = np.random.uniform(0.0, 1.0)
                    ky = np.random.uniform(0.0, 1.0)
                    vis = np.random.choice([0, 1, 2])
                    kpts.extend([kx, ky, vis])
                vals = [cls_id, cx, cy, w, h] + kpts
                lines.append(" ".join(f"{v:.6f}" for v in vals))

            lbl_path = lbl_dir / f"img_{j:03d}.txt"
            lbl_path.write_text("\n".join(lines) + "\n")

    data_config = {
        "path": str(tmp_path),
        "train": "train/images",
        "val": "val/images",
        "names": class_names,
        "num_classes": num_classes,
        "input_size": [64, 64],
        "num_keypoints": num_keypoints,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    return data_config


# ---------------------------------------------------------------------------
# TestKeypointDataset
# ---------------------------------------------------------------------------

class TestKeypointDataset:
    """Core dataset loading and indexing."""

    def test_load_dataset(self, tmp_path):
        cfg = _create_keypoint_dataset(tmp_path, num_images=6)
        ds = KeypointDataset(cfg, split="train", base_dir=tmp_path)
        assert len(ds) == 6

    def test_getitem_types(self, tmp_path):
        cfg = _create_keypoint_dataset(tmp_path)
        tfm = build_keypoint_transforms(
            is_train=False, input_size=(64, 64),
            mean=cfg["mean"], std=cfg["std"],
        )
        ds = KeypointDataset(cfg, split="train", transforms=tfm, base_dir=tmp_path)
        img, targets, path = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.dtype == torch.float32
        assert isinstance(targets, dict)
        assert "boxes" in targets and "keypoints" in targets
        assert isinstance(path, str)

    def test_keypoint_shape(self, tmp_path):
        num_kpts = 7
        cfg = _create_keypoint_dataset(tmp_path, num_keypoints=num_kpts)
        tfm = build_keypoint_transforms(
            is_train=False, input_size=(64, 64),
            mean=cfg["mean"], std=cfg["std"],
        )
        ds = KeypointDataset(cfg, split="train", transforms=tfm, base_dir=tmp_path)
        _, targets, _ = ds[0]
        kpts = targets["keypoints"]
        assert kpts.ndim == 3
        assert kpts.shape[1] == num_kpts
        assert kpts.shape[2] == 3

    def test_val_split(self, tmp_path):
        cfg = _create_keypoint_dataset(tmp_path, num_images=6)
        ds = KeypointDataset(cfg, split="val", base_dir=tmp_path)
        # val gets max(2, 6//2) = 3 images
        assert len(ds) == 3

    def test_transforms_applied(self, tmp_path):
        cfg = _create_keypoint_dataset(tmp_path)
        tfm = build_keypoint_transforms(
            is_train=False, input_size=(64, 64),
            mean=cfg["mean"], std=cfg["std"],
        )
        ds = KeypointDataset(cfg, split="train", transforms=tfm, base_dir=tmp_path)
        img, _, _ = ds[0]
        assert img.shape == (3, 64, 64)

    def test_missing_label_empty(self, tmp_path):
        cfg = _create_keypoint_dataset(tmp_path, num_images=4)
        # Remove one label file to simulate missing annotation
        lbl_dir = tmp_path / "train" / "labels"
        label_files = sorted(lbl_dir.glob("*.txt"))
        assert len(label_files) > 0
        label_files[0].unlink()

        tfm = build_keypoint_transforms(
            is_train=False, input_size=(64, 64),
            mean=cfg["mean"], std=cfg["std"],
        )
        ds = KeypointDataset(cfg, split="train", transforms=tfm, base_dir=tmp_path)
        img, targets, _ = ds[0]
        # Should return empty boxes/keypoints without error
        assert targets["boxes"].shape[0] == 0
        assert targets["keypoints"].shape[0] == 0


# ---------------------------------------------------------------------------
# TestKeypointTransforms
# ---------------------------------------------------------------------------

class TestKeypointTransforms:
    """Transform pipeline shape and coordinate behavior."""

    def test_train_output_shape(self, tmp_path):
        cfg = _create_keypoint_dataset(tmp_path)
        tfm = build_keypoint_transforms(
            is_train=True, input_size=(64, 64),
            mean=cfg["mean"], std=cfg["std"],
            aug_config={"hsv_h": 0.01, "hsv_s": 0.3, "hsv_v": 0.3},
        )
        img_bgr = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        boxes = np.array([[0, 0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
        kpts = np.random.uniform(0, 1, (1, 5, 3)).astype(np.float32)
        kpts[:, :, 2] = 2.0  # visible

        img_t, tgt = tfm(img_bgr, {"boxes": boxes, "keypoints": kpts})
        assert img_t.shape == (3, 64, 64)
        assert tgt["boxes"].shape == (1, 5)
        assert tgt["keypoints"].shape == (1, 5, 3)

    def test_val_preserves_coords(self, tmp_path):
        cfg = _create_keypoint_dataset(tmp_path)
        tfm = build_keypoint_transforms(
            is_train=False, input_size=(64, 64),
            mean=cfg["mean"], std=cfg["std"],
        )
        boxes = np.array([[0, 0.3, 0.4, 0.1, 0.2]], dtype=np.float32)
        kpts = np.array([[[0.25, 0.35, 2.0], [0.45, 0.55, 1.0]]], dtype=np.float32)
        img_bgr = np.random.randint(0, 255, (80, 120, 3), dtype=np.uint8)

        _, tgt = tfm(img_bgr, {"boxes": boxes, "keypoints": kpts})
        # Val transform has no flip/jitter so normalised coords stay the same
        np.testing.assert_allclose(tgt["boxes"].numpy(), boxes, atol=1e-5)
        np.testing.assert_allclose(tgt["keypoints"].numpy(), kpts, atol=1e-5)


# ---------------------------------------------------------------------------
# TestCollateAndDataloader
# ---------------------------------------------------------------------------

class TestCollateAndDataloader:
    """Collate function and dataloader builder."""

    def test_collate_fn(self, tmp_path):
        cfg = _create_keypoint_dataset(tmp_path)
        tfm = build_keypoint_transforms(
            is_train=False, input_size=(64, 64),
            mean=cfg["mean"], std=cfg["std"],
        )
        ds = KeypointDataset(cfg, split="train", transforms=tfm, base_dir=tmp_path)

        batch = [ds[i] for i in range(min(3, len(ds)))]
        result = keypoint_collate_fn(batch)

        assert "images" in result
        assert "targets" in result
        assert "paths" in result
        assert result["images"].shape[0] == len(batch)
        assert result["images"].shape[1:] == (3, 64, 64)
        assert len(result["targets"]) == len(batch)
        assert len(result["paths"]) == len(batch)

    def test_build_dataloader(self, tmp_path):
        cfg = _create_keypoint_dataset(tmp_path, num_images=8)
        training_config = {
            "augmentation": {},
            "data": {
                "batch_size": 2,
                "num_workers": 0,
                "pin_memory": False,
            },
        }
        loader = build_keypoint_dataloader(
            cfg, split="train", training_config=training_config, base_dir=tmp_path,
        )
        batch = next(iter(loader))
        assert batch["images"].shape == (2, 3, 64, 64)
        assert len(batch["targets"]) == 2


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Error handling and edge cases."""

    def test_invalid_split_raises(self, tmp_path):
        cfg = _create_keypoint_dataset(tmp_path)
        with pytest.raises(ValueError, match="split must be"):
            KeypointDataset(cfg, split="invalid", base_dir=tmp_path)

    def test_wrong_column_count_skipped(self, tmp_path):
        """Lines with wrong number of columns are silently skipped.

        np.loadtxt raises ValueError when rows have different column counts,
        so the dataset returns empty targets for that file. We verify no crash.
        """
        cfg = _create_keypoint_dataset(tmp_path, num_keypoints=5)
        lbl_dir = tmp_path / "train" / "labels"
        lbl_file = sorted(lbl_dir.glob("*.txt"))[0]

        # Write only an invalid line (too few columns).
        # Expected columns: 5 + 5*3 = 20, but we write 7.
        invalid = "0 0.5 0.5 0.2 0.2 0.1 0.2"
        lbl_file.write_text(f"{invalid}\n")

        tfm = build_keypoint_transforms(
            is_train=False, input_size=(64, 64),
            mean=cfg["mean"], std=cfg["std"],
        )
        ds = KeypointDataset(cfg, split="train", transforms=tfm, base_dir=tmp_path)
        _, targets, _ = ds[0]
        # Invalid line should be dropped, resulting in empty targets
        assert targets["boxes"].shape[0] == 0
        assert targets["keypoints"].shape[0] == 0


# ---------------------------------------------------------------------------
# TestGetRawItem
# ---------------------------------------------------------------------------

class TestGetRawItem:
    """get_raw_item returns raw numpy data."""

    def test_returns_dict(self, tmp_path):
        cfg = _create_keypoint_dataset(tmp_path)
        ds = KeypointDataset(cfg, split="train", base_dir=tmp_path)
        raw = ds.get_raw_item(0)
        assert isinstance(raw, dict)
        assert "image" in raw
        assert "targets" in raw
        assert "keypoints" in raw

    def test_image_is_bgr_numpy(self, tmp_path):
        cfg = _create_keypoint_dataset(tmp_path)
        ds = KeypointDataset(cfg, split="train", base_dir=tmp_path)
        raw = ds.get_raw_item(0)
        img = raw["image"]
        assert isinstance(img, np.ndarray)
        assert img.dtype == np.uint8
        assert img.ndim == 3
        assert img.shape[2] == 3  # BGR channels


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def run_test():
    """Run all tests in standalone mode."""
    import traceback

    test_classes = [
        TestKeypointDataset,
        TestKeypointTransforms,
        TestCollateAndDataloader,
        TestEdgeCases,
        TestGetRawItem,
    ]

    import tempfile
    passed = 0
    failed = 0

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        for method_name in sorted(methods):
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                try:
                    getattr(instance, method_name)(tmp_path)
                    print(f"  PASS  {cls.__name__}.{method_name}")
                    passed += 1
                except Exception:
                    print(f"  FAIL  {cls.__name__}.{method_name}")
                    traceback.print_exc()
                    failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    print("=" * 60)
    print("Test: Keypoint Dataset (YOLO-pose format)")
    print("=" * 60)
    run_test()

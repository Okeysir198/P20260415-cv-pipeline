"""Test: Segmentation Training — real SegmentationDataset, no loader patching.

Unlike TestHFSegmentationTraining in test_p06_classification_training.py which
patches _build_dataloaders with synthetic data, this test exercises the REAL
pipeline: SegmentationDataset → build_segmentation_dataloader → trainer.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_segmentation_dataset(
    root: Path, num_classes: int = 3, num_train: int = 16, num_val: int = 8
) -> dict:
    """Create a real segmentation dataset on disk.

    Images have strong class signal: horizontal bands of distinct colors
    per class to encourage loss decrease within a few epochs.

    Returns:
        data_config dict compatible with SegmentationDataset.
    """
    rng = np.random.RandomState(42)
    class_names = {i: f"class_{i}" for i in range(num_classes)}

    for split, n_images in [("train", num_train), ("val", num_val)]:
        img_dir = root / split / "images"
        mask_dir = root / split / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        for j in range(n_images):
            h, w = 64, 64
            img = np.zeros((h, w, 3), dtype=np.uint8)
            mask = np.zeros((h, w), dtype=np.uint8)
            band_h = h // num_classes

            for cls_id in range(num_classes):
                y0 = cls_id * band_h
                y1 = (cls_id + 1) * band_h if cls_id < num_classes - 1 else h
                # Strong color signal per class
                base_color = [(cls_id * 80 + 30) % 256,
                              ((cls_id + 1) * 60 + 50) % 256,
                              ((cls_id + 2) * 100 + 20) % 256]
                img[y0:y1] = base_color
                img[y0:y1] += rng.randint(0, 15, (y1 - y0, w, 3), dtype=np.uint8)
                mask[y0:y1] = cls_id

            cv2.imwrite(str(img_dir / f"img_{j:03d}.jpg"), img)
            cv2.imwrite(str(mask_dir / f"img_{j:03d}.png"), mask)

    return {
        "dataset_name": "test_segmentation",
        "path": str(root),
        "train": "train",
        "val": "val",
        "names": class_names,
        "num_classes": num_classes,
        "input_size": [64, 64],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }


@pytest.fixture
def seg_dataset(tmp_path):
    """Create a tiny segmentation dataset on disk."""
    data_root = tmp_path / "data"
    return _create_segmentation_dataset(data_root, num_classes=3)


@pytest.fixture
def seg_training_config(seg_dataset, tmp_path):
    """Write training + data YAML configs and return the training config path."""
    # Write data config
    data_config_path = tmp_path / "05_data.yaml"
    with open(data_config_path, "w") as f:
        yaml.dump(seg_dataset, f, default_flow_style=False)

    # Write training config
    config = {
        "model": {
            "arch": "hf-segformer",
            "pretrained": "nvidia/segformer-b0-finetuned-ade-512-512",
            "num_classes": 3,
            "input_size": [64, 64],
            "ignore_mismatched_sizes": True,
        },
        "data": {
            "dataset_config": str(data_config_path),
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
        },
        "training": {
            "epochs": 3,
            "optimizer": "adamw",
            "lr": 0.0001,
            "weight_decay": 0.01,
            "warmup_epochs": 0,
            "scheduler": "cosine",
            "patience": 0,
            "amp": False,
            "grad_clip": 1.0,
            "ema": False,
        },
        "logging": {
            "save_dir": str(tmp_path / "runs"),
            "wandb_project": None,
        },
        "checkpoint": {
            "save_best": True,
            "metric": "val/mIoU",
            "mode": "max",
            "save_interval": 0,
        },
        "seed": 42,
    }

    config_path = tmp_path / "06_training.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return str(config_path)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _collect_epoch_losses(trainer) -> list:
    """Patch trainer to collect per-epoch train losses."""
    epoch_losses: list[float] = []
    original_fn = trainer._train_one_epoch

    def _patched(*args, **kwargs):
        metrics = original_fn(*args, **kwargs)
        epoch_losses.append(metrics.get("train/loss", float("inf")))
        return metrics

    trainer._train_one_epoch = _patched
    return epoch_losses


try:
    from transformers import SegformerForSemanticSegmentation as _  # noqa: F401
    _HAS_HF_SEG = True
except ImportError:
    _HAS_HF_SEG = False

_requires_hf_seg = pytest.mark.skipif(
    not _HAS_HF_SEG, reason="transformers with SegFormer not installed"
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@_requires_hf_seg
class TestSegmentationTrainingReal:
    """Full 3-epoch training using real SegmentationDataset (no loader patching)."""

    def test_train_3_epochs_loss_decreases(self, seg_training_config, tmp_path):
        """Train SegFormer-B0 for 3 epochs on real disk dataset."""
        from core.p06_training.trainer import DetectionTrainer

        trainer = DetectionTrainer(config_path=seg_training_config)
        epoch_losses = _collect_epoch_losses(trainer)
        summary = trainer.train()

        assert len(epoch_losses) == 3, f"Expected 3 epochs, got {len(epoch_losses)}"
        for i, loss in enumerate(epoch_losses):
            assert 0 < loss < 100, f"Epoch {i+1} loss={loss:.4f} out of range"
        assert epoch_losses[-1] < epoch_losses[0], (
            f"Loss did not decrease: {epoch_losses[0]:.4f} → {epoch_losses[-1]:.4f}"
        )

    def test_val_miou_and_per_class_iou(self, seg_training_config):
        """Verify val/mIoU and per-class IoU keys present in final metrics."""
        from core.p06_training.trainer import DetectionTrainer

        trainer = DetectionTrainer(config_path=seg_training_config)
        summary = trainer.train()

        final_metrics = summary.get("final_metrics", {})
        assert "val/mIoU" in final_metrics, (
            f"val/mIoU missing. Keys: {list(final_metrics.keys())}"
        )
        for cls_id in range(3):
            key = f"val/IoU_cls{cls_id}"
            assert key in final_metrics, (
                f"{key} missing. Keys: {list(final_metrics.keys())}"
            )

    def test_checkpoint_saved(self, seg_training_config, tmp_path):
        """Verify that a checkpoint file is saved after training."""
        from core.p06_training.trainer import DetectionTrainer

        trainer = DetectionTrainer(config_path=seg_training_config)
        trainer.train()

        runs_dir = tmp_path / "runs"
        checkpoints = list(runs_dir.rglob("*.pth")) + list(runs_dir.rglob("*.pt"))
        assert len(checkpoints) > 0, f"No checkpoint found in {runs_dir}"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


def run_test():
    """Run all tests in this file (standalone mode)."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    return result.returncode == 0


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)

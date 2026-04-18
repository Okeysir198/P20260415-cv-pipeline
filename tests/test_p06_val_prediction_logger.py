"""Test p06: ValPredictionLogger — grid visualization of GT vs predictions."""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import run_all
from core.p06_training.callbacks import ValPredictionLogger
from core.p05_data.detection_dataset import YOLOXDataset
from utils.config import load_config

DATA_CONFIG = ROOT / "features/safety-fire_detection/configs/05_data.yaml"
OUT_DIR = ROOT / "tests/outputs/p06_val_prediction_logger"


class _MockModel(nn.Module):
    """Returns YOLOX-format predictions (B, 8400, 7) with two hard-coded detections."""
    output_format = "yolox"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        out = torch.zeros(B, 8400, 7, device=x.device)
        out[:, 0, :4] = torch.tensor([160.0, 120.0, 80.0, 60.0])
        out[:, 0, 4] = 0.9
        out[:, 0, 5] = 0.9   # class 0
        out[:, 1, :4] = torch.tensor([300.0, 200.0, 60.0, 50.0])
        out[:, 1, 4] = 0.85
        out[:, 1, 6] = 0.85  # class 1
        return out


class _MockTrainer:
    def __init__(self, dataset, model, device):
        self.val_loader = type("L", (), {"dataset": dataset})()
        self.model = model
        self.device = device
        self._model_cfg = {"input_size": [416, 416], "num_classes": 2}
        self._data_cfg = {"names": {0: "fire", 1: "smoke"}}
        self.callback_runner = None

    def _decode_predictions(self, preds, conf_threshold=0.25):
        results = []
        for b in range(preds.shape[0]):
            raw = preds[b]
            obj = raw[:, 4]
            cls_scores, cls_ids = raw[:, 5:].max(dim=-1)
            scores = obj * cls_scores
            mask = scores > conf_threshold
            cxcywh = raw[mask, :4].cpu().numpy()
            if cxcywh.shape[0] > 0:
                cx, cy, w, h = cxcywh[:, 0], cxcywh[:, 1], cxcywh[:, 2], cxcywh[:, 3]
                boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
            else:
                boxes = np.zeros((0, 4))
            results.append({
                "boxes": boxes,
                "labels": cls_ids[mask].cpu().numpy(),
                "scores": scores[mask].cpu().numpy(),
            })
        return results


def _make_trainer(num_samples=12):
    assert DATA_CONFIG.exists(), f"Missing config: {DATA_CONFIG}"
    data_cfg = load_config(str(DATA_CONFIG))
    dataset = YOLOXDataset(data_cfg, split="val", base_dir=str(DATA_CONFIG.parent))
    assert len(dataset) > 0, "Empty val dataset"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _MockTrainer(dataset, _MockModel().to(device), device)


def test_grid_created():
    """Grid image is created at the expected output path."""
    trainer = _make_trainer()
    cb = ValPredictionLogger(save_dir=str(OUT_DIR), num_samples=12, conf_threshold=0.25)
    cb.on_train_start(trainer)
    cb.on_epoch_end(trainer, epoch=0, metrics={"val/mAP50": 0.42})

    out_path = OUT_DIR / "val_predictions" / "epoch_001.png"
    assert out_path.exists(), f"Output not created: {out_path}"
    img = cv2.imread(str(out_path))
    assert img is not None and img.shape[0] > 100
    print(f"    {img.shape[1]}x{img.shape[0]} → {out_path.relative_to(ROOT)}")


def test_samples_fixed_across_epochs():
    """Sample indices are drawn once at on_train_start and never change."""
    trainer = _make_trainer()
    cb = ValPredictionLogger(save_dir=str(OUT_DIR), num_samples=8)
    cb.on_train_start(trainer)
    indices = list(cb._sample_indices)
    cb.on_epoch_end(trainer, epoch=1, metrics={})
    assert cb._sample_indices == indices
    print(f"    indices stable: {indices[:4]}...")


def test_dynamic_grid_rows():
    """Grid rows = ceil(loaded / grid_cols), not a hardcoded value."""
    trainer = _make_trainer()
    # 6 samples, 4 cols → ceil(6/4) = 2 rows
    cb = ValPredictionLogger(save_dir=str(OUT_DIR), num_samples=6, grid_cols=4)
    cb.on_train_start(trainer)
    cb.on_epoch_end(trainer, epoch=0, metrics={})
    img = cv2.imread(str(OUT_DIR / "val_predictions" / "epoch_001.png"))
    assert img is not None
    aspect = img.shape[1] / img.shape[0]
    assert aspect > 1.5, f"Expected wide 2-row grid, got aspect={aspect:.2f}"
    print(f"    6 samples / 4 cols → {img.shape[1]}x{img.shape[0]} (aspect={aspect:.1f})")


if __name__ == "__main__":
    run_all([
        ("grid_created", test_grid_created),
        ("samples_fixed_across_epochs", test_samples_fixed_across_epochs),
        ("dynamic_grid_rows", test_dynamic_grid_rows),
    ], title="\n=== Test p06: ValPredictionLogger ===\n", exit_on_fail=False)

"""Test 07: Model Variants — train 3 epochs per model, verify loss decreases."""

import json
import sys
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import passed, failed, errors, run_test, run_all
from core.p06_training.trainer import DetectionTrainer

OUTPUTS = Path(__file__).resolve().parent / "outputs" / "07_model_variants"
OUTPUTS.mkdir(parents=True, exist_ok=True)

TRAIN_CONFIG_PATH = str(ROOT / "configs" / "_test" / "06_training.yaml")


def _train_and_check(variant_name, overrides):
    """Train 3 epochs and assert loss decreases from epoch 1 to epoch 3."""
    trainer = DetectionTrainer(
        config_path=TRAIN_CONFIG_PATH,
        overrides={
            **overrides,
            "training": {
                **overrides.get("training", {}),
                "epochs": 3,
                "patience": 0,
                "amp": False,
                "ema": False,
                "warmup_epochs": 0,
            },
            "logging": {
                "save_dir": str(OUTPUTS / variant_name),
                "wandb_project": None,
            },
        },
    )

    # Collect per-epoch losses by monkey-patching callback
    epoch_losses = []
    original_train_one_epoch = trainer._train_one_epoch

    def patched_train_one_epoch(*args, **kwargs):
        metrics = original_train_one_epoch(*args, **kwargs)
        epoch_losses.append(metrics.get("train/loss", float("inf")))
        return metrics

    trainer._train_one_epoch = patched_train_one_epoch

    summary = trainer.train()
    assert len(epoch_losses) == 3, f"Expected 3 epoch losses, got {len(epoch_losses)}"

    # Save losses
    loss_path = OUTPUTS / f"{variant_name}_losses.json"
    with open(loss_path, "w") as f:
        json.dump({"epoch_losses": epoch_losses, "summary": str(summary)}, f, indent=2)
    print(f"    Epoch losses: {[f'{l:.4f}' for l in epoch_losses]}")

    # Assert loss is finite and reasonable
    for i, loss in enumerate(epoch_losses):
        assert 0 < loss < 200, (
            f"Epoch {i+1} loss={loss:.4f} is not in valid range (0, 200)"
        )

    # Verify the pipeline actually trains — last epoch loss must be lower than first
    assert epoch_losses[-1] < epoch_losses[0], (
        f"Loss did not decrease: epoch 1={epoch_losses[0]:.4f} → "
        f"epoch {len(epoch_losses)}={epoch_losses[-1]:.4f}. "
        f"Training pipeline may not be working."
    )

    # Cleanup GPU memory
    del trainer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# --- YOLOX variants ---

def test_yolox_tiny():
    _train_and_check("yolox_tiny", {
        "model": {"arch": "yolox-tiny", "depth": 0.33, "width": 0.375,
                  "num_classes": 2, "input_size": [640, 640], "pretrained": None},
        "data": {"batch_size": 4},
    })


def test_yolox_s():
    _train_and_check("yolox_s", {
        "model": {"arch": "yolox-s", "depth": 0.33, "width": 0.50,
                  "num_classes": 2, "input_size": [640, 640], "pretrained": None},
        "data": {"batch_size": 4},
    })


def test_yolox_m():
    _train_and_check("yolox_m", {
        "model": {"arch": "yolox-m", "depth": 0.67, "width": 0.75,
                  "num_classes": 2, "input_size": [640, 640], "pretrained": None},
        "data": {"batch_size": 4},
    })


def test_yolox_l():
    _train_and_check("yolox_l", {
        "model": {"arch": "yolox-l", "depth": 1.0, "width": 1.0,
                  "num_classes": 2, "input_size": [640, 640], "pretrained": None},
        "data": {"batch_size": 2},
    })


def test_yolox_x():
    _train_and_check("yolox_x", {
        "model": {"arch": "yolox-x", "depth": 1.33, "width": 1.25,
                  "num_classes": 2, "input_size": [640, 640], "pretrained": None},
        "data": {"batch_size": 2},
    })


# --- HF Transformer variants ---

def test_dfine():
    _train_and_check("dfine_s", {
        "model": {"arch": "dfine-s", "pretrained": "ustc-community/dfine_s_coco",
                  "num_classes": 2, "input_size": [640, 640]},
        "data": {"batch_size": 2},
        "training": {"optimizer": "adamw", "lr": 0.0001},
    })


def test_rtdetrv2():
    _train_and_check("rtdetr_r18", {
        "model": {"arch": "rtdetr-r18", "pretrained": "PekingU/rtdetr_v2_r18vd",
                  "num_classes": 2, "input_size": [640, 640]},
        "data": {"batch_size": 2},
        "training": {"optimizer": "adamw", "lr": 0.0001},
    })


# --- Custom arch tests (HF config passthrough + YOLOX depthwise) ---

def test_dfine_custom_arch():
    _train_and_check("dfine_custom", {
        "model": {"arch": "dfine-s", "pretrained": "ustc-community/dfine_s_coco",
                  "num_classes": 2, "input_size": [640, 640],
                  "decoder_layers": 2, "num_queries": 50},
        "data": {"batch_size": 2},
        "training": {"optimizer": "adamw", "lr": 0.0001},
    })


def test_rtdetr_custom_arch():
    _train_and_check("rtdetr_custom", {
        "model": {"arch": "rtdetr-r18", "pretrained": "PekingU/rtdetr_v2_r18vd",
                  "num_classes": 2, "input_size": [640, 640],
                  "decoder_layers": 3, "num_queries": 100},
        "data": {"batch_size": 2},
        "training": {"optimizer": "adamw", "lr": 0.0001},
    })


def test_yolox_depthwise():
    _train_and_check("yolox_depthwise", {
        "model": {"arch": "yolox-tiny", "depth": 0.33, "width": 0.375,
                  "num_classes": 2, "input_size": [640, 640], "pretrained": None,
                  "depthwise": True, "act_type": "relu"},
        "data": {"batch_size": 4},
    })


if __name__ == "__main__":
    run_all([
        ("yolox_tiny", test_yolox_tiny),
        ("yolox_s", test_yolox_s),
        ("yolox_m", test_yolox_m),
        ("yolox_l", test_yolox_l),
        ("yolox_x", test_yolox_x),
        ("dfine_s", test_dfine),
        ("rtdetrv2_r18", test_rtdetrv2),
        ("dfine_custom_arch", test_dfine_custom_arch),
        ("rtdetr_custom_arch", test_rtdetr_custom_arch),
        ("yolox_depthwise", test_yolox_depthwise),
    ], title="Test 09: Model Variants (3 epochs each)")

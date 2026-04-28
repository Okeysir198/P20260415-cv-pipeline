"""Test 08: Training — train 2 epochs on real 100 images, save checkpoint."""

import sys
import json
import tempfile
import traceback
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import passed, failed, errors, run_test, run_all
from core.p06_training.trainer import DetectionTrainer, ModelEMA
from core.p06_training.losses import FocalLoss, IoULoss, build_loss, YOLOXLoss, LOSS_REGISTRY
from core.p06_training.lr_scheduler import CosineScheduler, build_scheduler, PlateauScheduler
from core.p06_training.callbacks import EarlyStopping, CheckpointSaver, CallbackRunner, Callback
from core.p06_models.registry import build_model
from core.p06_models import base  # triggers model registration
import core.p06_models.yolox  # triggers YOLOX registration in MODEL_REGISTRY
from utils.device import get_device

OUTPUTS = Path(__file__).resolve().parent / "outputs" / "08_training"
OUTPUTS.mkdir(parents=True, exist_ok=True)

TRAIN_CONFIG_PATH = str(ROOT / "configs" / "_test" / "06_training.yaml")


def test_train_2_epochs():
    """Train DetectionTrainer for 2 epochs and verify outputs."""
    trainer = DetectionTrainer(
        config_path=TRAIN_CONFIG_PATH,
        overrides={
            "training": {"epochs": 2, "patience": 10},
            "logging": {"save_dir": str(OUTPUTS)},
        },
    )

    metrics = trainer.train()
    assert isinstance(metrics, dict), f"train() returned {type(metrics)}, expected dict"

    # Check expected keys exist
    print(f"    Returned metrics keys: {list(metrics.keys())}")

    # Check checkpoint files (.pt or .pth)
    ckpt_files = list(OUTPUTS.glob("best.*")) + list(OUTPUTS.glob("last.*"))
    ckpt_files = [f for f in ckpt_files if f.suffix in (".pt", ".pth")]
    assert len(ckpt_files) > 0, f"No checkpoint found in {OUTPUTS}"
    for f in ckpt_files:
        print(f"    {f.name}: exists ({f.stat().st_size / 1e6:.1f} MB)")

    # Save training summary
    summary = {k: v if not hasattr(v, 'item') else v.item() for k, v in metrics.items()}
    summary_path = OUTPUTS / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"    Saved: {summary_path}")


def test_checkpoint_loadable():
    """Verify saved checkpoint can be loaded."""
    # Try best.pth, best.pt, last.pth, last.pt
    ckpt_path = None
    for name in ["best.pth", "best.pt", "last.pth", "last.pt"]:
        p = OUTPUTS / name
        if p.exists():
            ckpt_path = p
            break
    assert ckpt_path is not None, f"No checkpoint to load in {OUTPUTS}"

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    assert isinstance(ckpt, dict), f"Checkpoint type: {type(ckpt)}"
    print(f"    Checkpoint keys: {list(ckpt.keys())}")

    # Should have model state dict
    has_model = "model_state_dict" in ckpt or "model" in ckpt or "state_dict" in ckpt
    assert has_model, f"No model weights in checkpoint"


def test_focal_loss():
    """Verify FocalLoss produces a positive scalar loss."""
    device = get_device()
    pred = torch.randn(10, 2, device=device)
    target = torch.zeros(10, 2, device=device)
    target[:3, 0] = 1.0
    target[5:8, 1] = 1.0

    loss = FocalLoss()(pred, target)
    assert loss.dim() == 0, f"Expected scalar, got shape {loss.shape}"
    assert loss.item() > 0, f"Expected loss > 0, got {loss.item()}"
    print(f"    FocalLoss value: {loss.item():.6f}")


def test_iou_loss():
    """Verify IoULoss: near-zero for identical boxes, high for disjoint boxes."""
    device = get_device()
    iou_loss_fn = IoULoss(variant="giou")

    # Identical boxes — loss should be near 0
    pred = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32, device=device)
    target = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32, device=device)
    loss_identical = iou_loss_fn(pred, target)
    assert loss_identical.item() < 0.1, f"Identical boxes loss={loss_identical.item():.4f}, expected < 0.1"
    print(f"    Identical boxes loss: {loss_identical.item():.6f}")

    # Disjoint boxes — loss should be > 0.5
    pred = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32, device=device)
    target = torch.tensor([[50, 50, 60, 60]], dtype=torch.float32, device=device)
    loss_disjoint = iou_loss_fn(pred, target)
    assert loss_disjoint.item() > 0.5, f"Disjoint boxes loss={loss_disjoint.item():.4f}, expected > 0.5"
    print(f"    Disjoint boxes loss: {loss_disjoint.item():.6f}")


def test_cosine_scheduler():
    """Verify CosineScheduler warmup and decay behavior."""
    optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.01)
    scheduler = CosineScheduler(optimizer, total_epochs=100, warmup_epochs=5)

    lrs = []
    for epoch in range(100):
        scheduler.step(epoch)
        lrs.append(optimizer.param_groups[0]["lr"])

    lr_epoch0 = lrs[0]
    lr_epoch5 = lrs[5]
    lr_epoch10 = lrs[10]
    lr_epoch99 = lrs[99]

    assert lr_epoch0 < lr_epoch5, (
        f"Warmup failed: lr@0={lr_epoch0:.6f} should be < lr@5={lr_epoch5:.6f}"
    )
    assert lr_epoch99 < lr_epoch10, (
        f"Decay failed: lr@99={lr_epoch99:.6f} should be < lr@10={lr_epoch10:.6f}"
    )
    print(f"    LR@0={lr_epoch0:.6f}, LR@5={lr_epoch5:.6f}, LR@10={lr_epoch10:.6f}, LR@99={lr_epoch99:.6f}")


def test_model_ema():
    """Verify ModelEMA params diverge from model params after update."""
    device = get_device()
    model = torch.nn.Linear(10, 2).to(device)
    ema = ModelEMA(model, decay=0.999, warmup_steps=10)

    # Fake optimizer step: modify model params
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.5)

    ema.update(model)

    # EMA params should differ from model params
    differs = False
    for (name, ema_p), (_, model_p) in zip(
        ema.ema_model.named_parameters(), model.named_parameters()
    ):
        if not torch.allclose(ema_p, model_p, atol=1e-8):
            differs = True
            break

    assert differs, "EMA params should differ from model params after update"
    print(f"    EMA params diverge from model params as expected (updates={ema.updates})")


def test_early_stopping():
    """Verify EarlyStopping triggers after patience is exhausted."""
    es = EarlyStopping(metric="val/mAP50", patience=3)

    # Improving metrics for 2 epochs
    es.on_epoch_end(None, 0, {"val/mAP50": 0.5})
    assert not es.should_stop, "Should not stop after epoch 0"
    es.on_epoch_end(None, 1, {"val/mAP50": 0.6})
    assert not es.should_stop, "Should not stop after epoch 1"

    # Non-improving metrics for 4 epochs
    es.on_epoch_end(None, 2, {"val/mAP50": 0.55})
    assert not es.should_stop, "Should not stop after 1 non-improving epoch"
    es.on_epoch_end(None, 3, {"val/mAP50": 0.55})
    assert not es.should_stop, "Should not stop after 2 non-improving epochs"
    es.on_epoch_end(None, 4, {"val/mAP50": 0.55})
    assert es.should_stop, "Should stop after 3 non-improving epochs (patience=3)"
    print(f"    EarlyStopping triggered at epoch 4 as expected (patience=3)")


def test_build_loss_yolox():
    """Verify build_loss returns YOLOXLoss for yolox-m config."""
    config = {"model": {"arch": "yolox-m", "num_classes": 2, "input_size": [640, 640], "depth": 0.67, "width": 0.75}, "loss": {}}
    loss = build_loss(config)
    assert isinstance(loss, YOLOXLoss), f"Expected YOLOXLoss, got {type(loss)}"
    print(f"    build_loss returned {type(loss).__name__} as expected")


def test_build_loss_unknown():
    """Verify build_loss raises ValueError for unknown loss type."""
    config = {"loss": {"type": "totally_invalid_loss"}}
    try:
        build_loss(config)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass
    print("    build_loss raised ValueError for unknown loss type as expected")


def test_checkpoint_saver_best():
    """Verify CheckpointSaver tracks best metric correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        saver = CheckpointSaver(save_dir=tmpdir, metric="val/mAP50", mode="max", save_interval=0)

        # Create a minimal trainer-like object
        trainer = types.SimpleNamespace(
            model=torch.nn.Linear(10, 2),
            optimizer=None,
            scheduler=None,
            scaler=None,
            ema=None,
            config={"test": True},
            _model_cfg=None,
            _data_cfg=None,
        )

        # Epoch 0: mAP=0.5
        saver.on_epoch_end(trainer, 0, {"val/mAP50": 0.5})
        best_path = Path(tmpdir) / "best.pth"
        assert best_path.exists(), "best.pth should exist after first epoch"

        # Epoch 1: mAP=0.7 (improvement)
        saver.on_epoch_end(trainer, 1, {"val/mAP50": 0.7})
        ckpt = torch.load(str(best_path), map_location="cpu", weights_only=False)
        assert ckpt["metrics"]["val/mAP50"] == 0.7, f"best should be 0.7, got {ckpt['metrics']['val/mAP50']}"

        # Epoch 2: mAP=0.6 (regression)
        saver.on_epoch_end(trainer, 2, {"val/mAP50": 0.6})
        ckpt = torch.load(str(best_path), map_location="cpu", weights_only=False)
        assert ckpt["metrics"]["val/mAP50"] == 0.7, "best should still be 0.7 after regression"
    print("    CheckpointSaver correctly tracks best metric")


def test_callback_runner_lifecycle():
    """Verify CallbackRunner calls all lifecycle hooks in order."""
    class TrackingCallback(Callback):
        def __init__(self):
            self.calls = []
        def on_train_start(self, trainer):
            self.calls.append("train_start")
        def on_epoch_start(self, trainer, epoch):
            self.calls.append(f"epoch_start_{epoch}")
        def on_batch_end(self, trainer, batch_idx, metrics):
            self.calls.append(f"batch_end_{batch_idx}")
        def on_epoch_end(self, trainer, epoch, metrics):
            self.calls.append(f"epoch_end_{epoch}")
        def on_train_end(self, trainer):
            self.calls.append("train_end")

    tracker = TrackingCallback()
    runner = CallbackRunner([tracker])

    runner.on_train_start(None)
    runner.on_epoch_start(None, 0)
    runner.on_batch_end(None, 0, {"loss": 1.0})
    runner.on_epoch_end(None, 0, {"val/mAP50": 0.5})
    runner.on_train_end(None)

    expected = ["train_start", "epoch_start_0", "batch_end_0", "epoch_end_0", "train_end"]
    assert tracker.calls == expected, f"Expected {expected}, got {tracker.calls}"
    print(f"    CallbackRunner lifecycle hooks called in correct order")


def test_build_scheduler_cosine():
    """Verify build_scheduler returns CosineScheduler for cosine config."""
    optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.01)
    config = {"training": {"scheduler": "cosine", "epochs": 100, "warmup_epochs": 5}}
    scheduler = build_scheduler(optimizer, config)
    assert isinstance(scheduler, CosineScheduler), f"Expected CosineScheduler, got {type(scheduler)}"
    print(f"    build_scheduler returned {type(scheduler).__name__} as expected")


def test_build_scheduler_plateau():
    """Verify build_scheduler returns PlateauScheduler for plateau config."""
    optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.01)
    config = {"training": {"scheduler": "plateau", "epochs": 100, "warmup_epochs": 5}}
    scheduler = build_scheduler(optimizer, config)
    assert isinstance(scheduler, PlateauScheduler), f"Expected PlateauScheduler, got {type(scheduler)}"
    print(f"    build_scheduler returned {type(scheduler).__name__} as expected")


def test_build_model_yolox():
    """Verify build_model creates a YOLOX model with correct attributes."""
    import core.p06_models.yolox  # ensure registration
    config = {"model": {"arch": "yolox-m", "num_classes": 2, "input_size": [640, 640], "depth": 0.67, "width": 0.75}, "data": {"num_classes": 2}}
    model = build_model(config)
    assert isinstance(model, torch.nn.Module)
    assert hasattr(model, 'output_format')
    print(f"    build_model created {type(model).__name__} with output_format={model.output_format}")


def test_build_model_unknown():
    """Verify build_model raises ValueError for unknown architecture."""
    config = {"model": {"arch": "totally_nonexistent_model"}}
    try:
        build_model(config)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass
    print("    build_model raised ValueError for unknown arch as expected")


def test_get_param_groups():
    """Verify model.get_param_groups returns valid parameter groups."""
    import core.p06_models.yolox
    config = {"model": {"arch": "yolox-m", "num_classes": 2, "input_size": [640, 640], "depth": 0.67, "width": 0.75}, "data": {"num_classes": 2}}
    model = build_model(config)
    groups = model.get_param_groups(lr=0.01, weight_decay=5e-4)
    assert isinstance(groups, list) and len(groups) >= 2, f"Expected >=2 groups, got {len(groups)}"
    total_params = sum(p.numel() for g in groups for p in g["params"])
    model_params = sum(p.numel() for p in model.parameters())
    assert total_params > 0, "Param groups should cover some params"
    print(f"    get_param_groups: {len(groups)} groups, {total_params} params (model has {model_params})")


def test_yolox_no_double_sigmoid():
    """Regression: YOLOX `_DecoupledHead` (yolox.py:553-555) and the official
    Megvii adapter (decode_in_inference=True) BOTH sigmoid obj+cls inside the
    model's eval-mode forward. `_postprocess_yolox` MUST NOT re-sigmoid —
    doing so squashes scores into [0.25, 0.55] and makes conf_threshold
    meaningless. Drive the postprocess with known *probability* inputs
    (already-sigmoided) and assert the output score equals obj * cls
    exactly. A re-sigmoid would replace 0.6 * 0.95 = 0.570 with
    sigmoid(0.6) * sigmoid(0.95) = 0.646 * 0.721 = 0.466.
    """
    from core.p06_training.postprocess import _postprocess_yolox  # noqa: PLC0415

    # Single anchor, single class. xywh in pixel space, obj=0.6, cls=0.95.
    # Box: cx=100, cy=100, w=20, h=20 → xyxy = [90, 90, 110, 110].
    pred = torch.tensor([[[100.0, 100.0, 20.0, 20.0, 0.6, 0.95]]])
    results = _postprocess_yolox(pred, conf_threshold=0.0, nms_threshold=0.5)
    assert len(results) == 1 and results[0]["scores"].size == 1, \
        f"expected 1 detection, got {results[0]}"
    score = float(results[0]["scores"][0])
    expected = 0.6 * 0.95  # 0.570
    double_sigmoid = float(torch.sigmoid(torch.tensor(0.6)) *
                           torch.sigmoid(torch.tensor(0.95)))  # ~0.466
    assert abs(score - expected) < 1e-5, (
        f"YOLOX postprocess returned score={score:.4f}, expected {expected:.4f}. "
        f"If a sigmoid was re-applied the score would be ~{double_sigmoid:.4f}."
    )
    # And conf_threshold must filter using the un-sigmoided product.
    filtered = _postprocess_yolox(pred, conf_threshold=0.55, nms_threshold=0.5)
    assert filtered[0]["scores"].size == 1, (
        "conf_threshold=0.55 should KEEP score=0.57 — got 0 detections, "
        "indicating the product was incorrectly re-sigmoided to ~0.47."
    )
    print(f"    yolox postprocess score={score:.3f} (expected {expected:.3f}; "
          f"double-sigmoid would be {double_sigmoid:.3f})")


if __name__ == "__main__":
    run_all([
        ("train_2_epochs", test_train_2_epochs),
        ("checkpoint_loadable", test_checkpoint_loadable),
        ("focal_loss", test_focal_loss),
        ("iou_loss", test_iou_loss),
        ("cosine_scheduler", test_cosine_scheduler),
        ("model_ema", test_model_ema),
        ("early_stopping", test_early_stopping),
        ("build_loss_yolox", test_build_loss_yolox),
        ("build_loss_unknown", test_build_loss_unknown),
        ("checkpoint_saver_best", test_checkpoint_saver_best),
        ("callback_runner_lifecycle", test_callback_runner_lifecycle),
        ("build_scheduler_cosine", test_build_scheduler_cosine),
        ("build_scheduler_plateau", test_build_scheduler_plateau),
        ("build_model_yolox", test_build_model_yolox),
        ("build_model_unknown", test_build_model_unknown),
        ("get_param_groups", test_get_param_groups),
        ("yolox_no_double_sigmoid", test_yolox_no_double_sigmoid),
    ], title="Test 04: Training (2 epochs)")

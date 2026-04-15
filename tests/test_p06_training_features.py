"""Test 09: Training Features — schedulers, gradient accumulation, segmentation postprocess/metrics."""

import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import passed, failed, errors, run_test, run_all
from core.p06_training.lr_scheduler import (
    StepScheduler,
    OneCycleScheduler,
    build_scheduler,
)
from core.p06_training.postprocess import POSTPROCESSOR_REGISTRY
from core.p06_training.metrics_registry import compute_metrics


# ---------------------------------------------------------------------------
# Schedulers
# ---------------------------------------------------------------------------


def test_step_scheduler():
    """StepScheduler: warmup increases LR, state_dict roundtrips."""
    model = nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = StepScheduler(
        optimizer, step_size=3, gamma=0.1, warmup_epochs=2, warmup_start_factor=0.01,
    )
    lrs = []
    for epoch in range(10):
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
    # Warmup: first epoch should have LR < base LR
    assert lrs[0] < 0.1, f"Expected LR < 0.1 during warmup, got {lrs[0]}"
    # LR should increase during warmup (epoch 1 > epoch 0)
    assert lrs[1] >= lrs[0], f"Warmup should increase LR: {lrs[1]} >= {lrs[0]}"
    # After warmup + step_size, LR should drop
    assert lrs[-1] < lrs[2], f"StepLR should decay: {lrs[-1]} < {lrs[2]}"
    # Check state_dict/load_state_dict roundtrip
    state = scheduler.state_dict()
    assert "current_epoch" in state, f"state_dict keys: {list(state.keys())}"
    print(f"    LRs (first 5): {[f'{lr:.6f}' for lr in lrs[:5]]}")


def test_onecycle_scheduler():
    """OneCycleScheduler: LR rises then falls, state_dict roundtrips."""
    model = nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    scheduler = OneCycleScheduler(optimizer, max_lr=0.01, total_epochs=10)
    lrs = []
    for epoch in range(10):
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
    # LR should rise then fall
    max_lr_idx = int(np.argmax(lrs))
    assert max_lr_idx > 0, f"Peak LR should not be at start, peak idx={max_lr_idx}"
    assert max_lr_idx < 9, f"Peak LR should not be at end, peak idx={max_lr_idx}"
    state = scheduler.state_dict()
    assert "current_epoch" in state, f"state_dict keys: {list(state.keys())}"
    print(f"    Peak LR at epoch {max_lr_idx}: {lrs[max_lr_idx]:.6f}")


def test_build_scheduler_step():
    """build_scheduler with type='step' returns StepScheduler."""
    model = nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    config = {
        "training": {
            "scheduler": "step",
            "step_size": 5,
            "gamma": 0.5,
            "epochs": 20,
            "warmup_epochs": 2,
        },
    }
    s = build_scheduler(optimizer, config)
    assert isinstance(s, StepScheduler), f"Expected StepScheduler, got {type(s).__name__}"
    print(f"    Built: {type(s).__name__}")


def test_build_scheduler_onecycle():
    """build_scheduler with type='onecycle' returns OneCycleScheduler."""
    model = nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    config = {"training": {"scheduler": "onecycle", "lr": 0.01, "epochs": 20}}
    s = build_scheduler(optimizer, config)
    assert isinstance(s, OneCycleScheduler), f"Expected OneCycleScheduler, got {type(s).__name__}"
    print(f"    Built: {type(s).__name__}")


def test_build_scheduler_unknown_raises():
    """build_scheduler with unknown type raises ValueError."""
    model = nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    config = {"training": {"scheduler": "invalid_name", "epochs": 10}}
    try:
        build_scheduler(optimizer, config)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "invalid_name" in str(e), f"Error message should mention 'invalid_name': {e}"
        print(f"    Raised ValueError as expected: {e}")


# ---------------------------------------------------------------------------
# Gradient Accumulation
# ---------------------------------------------------------------------------


def test_gradient_accumulation():
    """Verify trainer runs without error with gradient_accumulation_steps=2."""
    from core.p06_training.trainer import DetectionTrainer

    TRAIN_CONFIG_PATH = str(ROOT / "configs" / "_test" / "06_training.yaml")
    trainer = DetectionTrainer(
        config_path=TRAIN_CONFIG_PATH,
        overrides={
            "training": {"epochs": 1, "gradient_accumulation_steps": 2, "patience": 0},
        },
    )
    summary = trainer.train()
    assert "final_metrics" in summary, f"Expected 'final_metrics' in summary, got keys: {list(summary.keys())}"
    print(f"    Training completed with grad_accum=2, keys: {list(summary.keys())}")


# ---------------------------------------------------------------------------
# Segmentation Postprocessor
# ---------------------------------------------------------------------------


def test_segmentation_postprocessor():
    """Segmentation postprocessor produces correct class_map shape and dtype."""
    assert "segmentation" in POSTPROCESSOR_REGISTRY, (
        f"'segmentation' not in POSTPROCESSOR_REGISTRY: {sorted(POSTPROCESSOR_REGISTRY.keys())}"
    )
    logits = torch.randn(2, 5, 32, 32)
    results = POSTPROCESSOR_REGISTRY["segmentation"](logits)
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    for i, r in enumerate(results):
        assert "class_map" in r, f"Result {i} missing 'class_map' key"
        assert r["class_map"].shape == (32, 32), f"Expected (32, 32), got {r['class_map'].shape}"
        assert r["class_map"].dtype == np.int64, f"Expected int64, got {r['class_map'].dtype}"
        assert r["class_map"].min() >= 0, f"class_map min < 0: {r['class_map'].min()}"
        assert r["class_map"].max() <= 4, f"class_map max > 4: {r['class_map'].max()}"
    print(f"    Postprocessed {len(results)} images, class_map shape: {results[0]['class_map'].shape}")


# ---------------------------------------------------------------------------
# Segmentation Metrics
# ---------------------------------------------------------------------------


def test_segmentation_metrics_registry():
    """Segmentation metrics: identical pred/gt for one class yields mIoU > 0."""
    preds = [np.zeros((10, 10), dtype=np.int64)]
    gts = [np.zeros((10, 10), dtype=np.int64)]
    result = compute_metrics("segmentation", preds, gts, num_classes=2)
    assert "val/mIoU" in result, f"Expected 'val/mIoU' in result, got keys: {list(result.keys())}"
    assert result["val/mIoU"] > 0, f"Expected mIoU > 0, got {result['val/mIoU']}"
    print(f"    mIoU (all-zeros): {result['val/mIoU']:.4f}")


def test_segmentation_metrics_perfect():
    """Perfect segmentation: pred == gt gives mIoU == 1.0."""
    mask = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]], dtype=np.int64)
    preds = [mask.copy()]
    gts = [mask.copy()]
    result = compute_metrics("segmentation", preds, gts, num_classes=3)
    assert abs(result["val/mIoU"] - 1.0) < 1e-6, f"Expected mIoU ~1.0, got {result['val/mIoU']}"
    print(f"    mIoU (perfect): {result['val/mIoU']:.6f}")


def test_segmentation_metrics_no_overlap():
    """Zero overlap: pred all-0, gt all-1 gives mIoU near 0."""
    preds = [np.zeros((10, 10), dtype=np.int64)]  # all class 0
    gts = [np.ones((10, 10), dtype=np.int64)]  # all class 1
    result = compute_metrics("segmentation", preds, gts, num_classes=2)
    assert result["val/mIoU"] < 0.01, f"Expected mIoU < 0.01, got {result['val/mIoU']}"
    print(f"    mIoU (no overlap): {result['val/mIoU']:.6f}")


# ---------------------------------------------------------------------------
# Configurable NMS threshold
# ---------------------------------------------------------------------------


def test_configurable_nms_threshold():
    """Trainer respects nms_threshold override in config."""
    from core.p06_training.trainer import DetectionTrainer

    TRAIN_CONFIG_PATH = str(ROOT / "configs" / "_test" / "06_training.yaml")
    trainer = DetectionTrainer(
        config_path=TRAIN_CONFIG_PATH,
        overrides={"training": {"nms_threshold": 0.6, "epochs": 1, "patience": 0}},
    )
    assert trainer._train_cfg.get("nms_threshold", 0.45) == 0.6, (
        f"Expected nms_threshold=0.6, got {trainer._train_cfg.get('nms_threshold')}"
    )
    print(f"    nms_threshold: {trainer._train_cfg['nms_threshold']}")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_all([
        ("step_scheduler", test_step_scheduler),
        ("onecycle_scheduler", test_onecycle_scheduler),
        ("build_scheduler_step", test_build_scheduler_step),
        ("build_scheduler_onecycle", test_build_scheduler_onecycle),
        ("build_scheduler_unknown_raises", test_build_scheduler_unknown_raises),
        ("gradient_accumulation", test_gradient_accumulation),
        ("segmentation_postprocessor", test_segmentation_postprocessor),
        ("segmentation_metrics_registry", test_segmentation_metrics_registry),
        ("segmentation_metrics_perfect", test_segmentation_metrics_perfect),
        ("segmentation_metrics_no_overlap", test_segmentation_metrics_no_overlap),
        ("configurable_nms_threshold", test_configurable_nms_threshold),
    ], title="\n=== Test 10: New Training Features ===\n", exit_on_fail=False)

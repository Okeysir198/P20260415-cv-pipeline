"""Test: HF Trainer backend for detection — 1 epoch, small subset.

Validates the new `backend: hf` detection path end-to-end:
- Config validator accepts the setup (no false-positive hard errors).
- Collator, model wrapper, compute_metrics, _DetectionTrainer._save all work.
- Viz bridge emits at least `dataset_stats.json` under the run dir.
- `trainer_state.json` has a finite `best_metric`.
- Test-set eval writes `test_results.json` with real mAP keys.

Doesn't measure quality — just that the pipeline completes without crashing.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import run_all  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "08_training_hf_detection"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CFG = str(ROOT / "configs" / "_test" / "06_training.yaml")


def test_hf_detection_one_epoch():
    """Run 1 epoch of HF Trainer detection on the fire_100 fixture."""
    import os
    os.environ.setdefault("WANDB_MODE", "disabled")

    from core.p06_training.hf_trainer import train_with_hf

    summary = train_with_hf(
        config_path=TRAIN_CFG,
        overrides={
            "model": {
                # Swap YOLOX-M (not supported by HF backend) for RT-DETRv2-R18
                # — the only HF detection arch our registry currently serves.
                "arch": "rtdetr-r18",
                "pretrained": "PekingU/rtdetr_v2_r18vd",
                "input_size": [480, 480],
            },
            "data": {"batch_size": 4, "num_workers": 2,
                      "subset": {"train": 0.5, "val": 0.5}},
            "augmentation": {
                "normalize": False,
                "mosaic": False,       # DETR doesn't use mosaic
                "library": "albumentations",
                "fliplr": 0.5,
            },
            "training": {
                "backend": "hf",
                "epochs": 1,
                "bf16": True,
                "amp": False,
                "optimizer": "adamw",
                "lr": 1.0e-4,
                "weight_decay": 1.0e-4,
                "scheduler": "cosine",
                "warmup_steps": 10,
                "max_grad_norm": 0.1,
                "patience": 0,
                "val_viz": {"enabled": False},
                "train_viz": {"enabled": False},
                "aug_viz": {"enabled": False},
                "data_viz": {"enabled": True, "splits": ["train", "val", "test"]},
            },
            "checkpoint": {"save_best": True, "metric": "val/mAP50", "mode": "max"},
            "logging": {"save_dir": str(OUT_DIR), "wandb_project": None},
            "seed": 42,
        },
    )

    # 1) train_with_hf returned a summary dict
    assert isinstance(summary, dict), f"summary type {type(summary)}"
    assert "metrics" in summary
    assert "train_loss" in summary
    print(f"    train_loss={summary['train_loss']:.3f}")

    # 2) HF-Trainer-standard files in the output dir
    expect = ["pytorch_model.bin", "config.json", "preprocessor_config.json"]
    for f in expect:
        assert (OUT_DIR / f).exists(), f"missing {f} in {OUT_DIR}"

    # 3) At least one checkpoint-* subdir, each with trainer_state.json
    ckpts = sorted(p for p in OUT_DIR.iterdir() if p.is_dir() and p.name.startswith("checkpoint-"))
    assert len(ckpts) >= 1, f"no checkpoint-* dirs in {OUT_DIR}"
    ts_path = ckpts[-1] / "trainer_state.json"
    ts = json.loads(ts_path.read_text())
    assert ts.get("best_metric") is not None, "best_metric missing from trainer_state"
    assert ts["best_metric"] >= 0.0, f"best_metric={ts['best_metric']} below zero"
    print(f"    checkpoint-*={len(ckpts)}  best_metric={ts['best_metric']:.4f}")

    # 4) dataset_stats.json got all three splits via the viz bridge
    stats_path = OUT_DIR / "data_preview" / "dataset_stats.json"
    assert stats_path.exists(), f"data_preview/dataset_stats.json missing"
    stats = json.loads(stats_path.read_text())
    assert set(stats["splits"]) >= {"train", "val", "test"}, \
        f"splits={list(stats['splits'])} missing one of train/val/test"
    print(f"    viz-bridge splits={list(stats['splits'])}")

    # 5) test_results.json written with real mAP keys
    test_json = OUT_DIR / "test_results.json"
    assert test_json.exists(), "test_results.json missing — end-of-training test eval didn't run"
    test_metrics = json.loads(test_json.read_text())
    assert any(k.startswith("test_") and "map" in k for k in test_metrics), \
        f"no mAP keys in test_metrics: {list(test_metrics)}"
    print(f"    test_map={test_metrics.get('test_map', 'n/a')}")


def test_hf_ema_enabled_one_epoch():
    """EMA callback runs: shadow weights updated + saved at end of training."""
    import os
    os.environ.setdefault("WANDB_MODE", "disabled")

    from core.p06_training.hf_trainer import train_with_hf

    ema_out = OUT_DIR.parent / "08_training_hf_ema"
    ema_out.mkdir(parents=True, exist_ok=True)

    train_with_hf(
        config_path=TRAIN_CFG,
        overrides={
            "model": {"arch": "rtdetr-r18", "pretrained": "PekingU/rtdetr_v2_r18vd",
                       "input_size": [480, 480]},
            "data": {"batch_size": 4, "num_workers": 2, "subset": {"train": 0.5, "val": 0.5}},
            "augmentation": {"normalize": False, "mosaic": False, "library": "albumentations", "fliplr": 0.5},
            "training": {
                "backend": "hf", "epochs": 1, "bf16": True, "amp": False,
                "optimizer": "adamw", "lr": 1e-4, "weight_decay": 1e-4,
                "scheduler": "cosine", "warmup_steps": 10, "max_grad_norm": 0.1,
                "patience": 0,
                "ema": True, "ema_warmup_steps": 5,   # short warmup so 10 steps hit decay
                "val_viz": {"enabled": False}, "train_viz": {"enabled": False},
                "aug_viz": {"enabled": False}, "data_viz": {"enabled": False},
            },
            "checkpoint": {"save_best": True, "metric": "val/mAP50", "mode": "max"},
            "logging": {"save_dir": str(ema_out), "wandb_project": None},
            "seed": 42,
        },
    )

    ema_bin = ema_out / "ema_model.bin"
    assert ema_bin.exists(), f"ema_model.bin not written to {ema_out}"
    import torch
    sd = torch.load(str(ema_bin), map_location="cpu", weights_only=True)
    assert isinstance(sd, dict) and len(sd) > 0
    print(f"    EMA ckpt keys={len(sd)}  size={ema_bin.stat().st_size/1e6:.1f}MB")


def test_hf_backend_rejects_unsupported_task():
    """Config validator hard-fails when HF backend is asked to train a task
    it doesn't support yet (e.g. a pose model)."""
    from core.p06_training.hf_trainer import _validate_hf_backend_config
    try:
        _validate_hf_backend_config(
            {"training": {"backend": "hf"}}, output_format="pose",
        )
    except ValueError as e:
        assert "does not yet support" in str(e), f"unexpected: {e}"
        print(f"    correctly rejected pose: {str(e).splitlines()[0]}")
        return
    raise AssertionError("validator should have raised on output_format='pose'")


def test_hf_backend_rejects_fp16_on_detection():
    """Config validator hard-fails on amp=True for DETR (decoder overflows in fp16)."""
    from core.p06_training.hf_trainer import _validate_hf_backend_config
    try:
        _validate_hf_backend_config(
            {"training": {"backend": "hf", "amp": True, "bf16": False}},
            output_format="detr",
        )
    except ValueError as e:
        assert "overflow" in str(e).lower() or "fp16" in str(e), f"unexpected: {e}"
        print(f"    correctly rejected amp=True: {str(e).splitlines()[0]}")
        return
    raise AssertionError("validator should have raised on detection + amp=True")


if __name__ == "__main__":
    run_all(
        [
            ("reject_unsupported_task", test_hf_backend_rejects_unsupported_task),
            ("reject_fp16_on_detection", test_hf_backend_rejects_fp16_on_detection),
            ("hf_detection_one_epoch", test_hf_detection_one_epoch),
            ("hf_ema_enabled_one_epoch", test_hf_ema_enabled_one_epoch),
        ],
        title="HF Trainer Detection Backend",
    )

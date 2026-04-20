"""Compute mAP on the TRAIN subset that was actually used to train a checkpoint.

Uses val-style (no-augmentation) transforms so we measure pure memorization —
how well the model predicts on the exact pixels it was trained on.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.device import auto_select_gpu  # noqa: E402
auto_select_gpu()

import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from core.p05_data.detection_dataset import YOLOXDataset, collate_fn  # noqa: E402
from core.p05_data.transforms import build_transforms  # noqa: E402
from core.p06_training.trainer import DetectionTrainer  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="06_training.yaml used for the run")
    p.add_argument("--ckpt", required=True, help="Checkpoint .pth")
    p.add_argument("--subset", type=float, default=0.05, help="Train subset fraction (same as training)")
    p.add_argument("--extra-override", nargs="*", default=[], help="Extra key=value overrides, e.g. model.impl=official augmentation.normalize=false")
    args = p.parse_args()

    # Parse extra overrides into nested dict
    extra_overrides: dict = {}
    for kv in args.extra_override:
        k, v = kv.split("=", 1)
        # cast common types
        if v.lower() in ("true", "false"):
            v = v.lower() == "true"
        else:
            try:
                v = float(v) if "." in v or "e" in v.lower() else int(v)
            except ValueError:
                pass
        parts = k.split(".")
        d = extra_overrides
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = v

    # Must match training-time seed + subset so the Subset draws the same indices
    overrides = {
        "data": {"subset": {"train": args.subset, "val": args.subset}},
        "training": {
            "data_viz": {"enabled": False},
            "aug_viz": {"enabled": False},
            "val_viz": {"enabled": False},
            "val_full_interval": 0,
        },
        "logging": {"wandb_project": None},
    }
    # merge extra overrides
    def _merge(a, b):
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                _merge(a[k], v)
            else:
                a[k] = v
    _merge(overrides, extra_overrides)

    trainer = DetectionTrainer(args.config, overrides=overrides)
    trainer.train_loader, trainer.val_loader = trainer._build_dataloaders()
    trainer.model = trainer._build_model()

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=trainer.device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    trainer._base_model.load_state_dict(state)
    print(f"Loaded {args.ckpt} (epoch {ckpt.get('epoch', -1) + 1})")

    # Reconstruct the same train subset, but with val-style transforms (no aug)
    train_ds_with_aug = trainer.train_loader.dataset
    subset_indices = list(train_ds_with_aug.indices) if hasattr(train_ds_with_aug, "indices") else None
    if subset_indices is None:
        raise SystemExit("train_loader.dataset has no .indices — subset not active?")

    data_cfg = trainer._loaded_data_cfg
    aug_cfg = trainer.config.get("augmentation", {})
    input_size = tuple(trainer._model_cfg["input_size"])
    # is_train=False → no augmentation; same resize + normalize as val
    eval_transforms = build_transforms(
        config=aug_cfg, is_train=False, input_size=input_size,
        mean=data_cfg.get("mean"), std=data_cfg.get("std"),
    )
    base_dir = str(trainer.config_path.parent)
    eval_train_ds = YOLOXDataset(
        data_config=data_cfg, split="train", transforms=eval_transforms, base_dir=base_dir,
    )
    eval_train_subset = Subset(eval_train_ds, subset_indices)
    eval_loader = DataLoader(
        eval_train_subset,
        batch_size=trainer._data_cfg.get("batch_size", 16),
        shuffle=False,
        num_workers=trainer._data_cfg.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
    )
    print(f"Evaluating on {len(eval_train_subset)} train images (val-style transforms — no augmentation)")

    metrics = trainer._validate(eval_loader)
    print("\n=== TRAIN-set mAP (subset, no augmentation) ===")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k.replace('val/', 'train/')}: {v:.4f}")


if __name__ == "__main__":
    main()

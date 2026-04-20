"""One-shot validation prediction grid for a YOLOX checkpoint.

Reuses DetectionTrainer + ValPredictionLogger machinery so the output matches
what training saves to ``<run>/val_predictions/epoch_NNN.png``.

Usage (official impl, uses .venv-yolox-official):
    .venv-yolox-official/bin/python scripts/yolox_val_viz_oneshot.py \\
        --run-dir features/safety-fire_detection/runs/2026-04-19_184025_06_training \\
        --ckpt best.pth --impl official --no-normalize
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.device import auto_select_gpu  # noqa: E402
auto_select_gpu()

from core.p06_training.callbacks import ValPredictionLogger  # noqa: E402
from core.p06_training.trainer import DetectionTrainer  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="Run directory containing 06_training.yaml + checkpoint")
    p.add_argument("--ckpt", default="best.pth", help="Checkpoint filename in run-dir")
    p.add_argument("--impl", choices=["custom", "official"], default="official")
    p.add_argument("--no-normalize", action="store_true", help="augmentation.normalize=false (match official training)")
    p.add_argument("--conf", type=float, default=0.25, help="Prediction confidence threshold for viz")
    p.add_argument("--num-samples", type=int, default=12)
    p.add_argument("--out-name", default=None, help="Output filename (default: val_predictions/oneshot_<ckpt_stem>.png)")
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    ckpt_path = run_dir / args.ckpt
    if not ckpt_path.exists():
        raise SystemExit(f"Missing {ckpt_path}")

    # Use feature's original 06_training.yaml — run-dir copies have ../../../ paths
    # that break because the copies live one level deeper than configs/.
    config_path = ROOT / "features" / "safety-fire_detection" / "configs" / "06_training_yolox.yaml"
    if not config_path.exists():
        config_path = ROOT / "features" / "safety-fire_detection" / "configs" / "06_training.yaml"
    if not config_path.exists():
        raise SystemExit(f"Missing feature config in {config_path.parent}")
    overrides = {
        "model": {"impl": args.impl, "pretrained": None},
        "training": {
            "val_viz": {"enabled": True, "num_samples": args.num_samples, "conf_threshold": args.conf},
            "data_viz": {"enabled": False},
            "aug_viz": {"enabled": False},
        },
        "logging": {"wandb_project": None},
    }
    if args.no_normalize:
        overrides["augmentation"] = {"normalize": False}

    trainer = DetectionTrainer(str(config_path), overrides=overrides)
    trainer.train_loader, trainer.val_loader = trainer._build_dataloaders()
    trainer.model = trainer._build_model()

    import torch
    ckpt = torch.load(ckpt_path, map_location=trainer.device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    trainer._base_model.load_state_dict(state)
    epoch = ckpt.get("epoch", -1)
    print(f"Loaded {ckpt_path} (epoch {epoch + 1})")

    viz_cfg = trainer._train_cfg["val_viz"]
    viz = ValPredictionLogger(
        save_dir=str(run_dir),
        split="val",
        num_samples=viz_cfg.get("num_samples", args.num_samples),
        conf_threshold=viz_cfg.get("conf_threshold", args.conf),
        grid_cols=viz_cfg.get("grid_cols", 2),
        gt_thickness=viz_cfg.get("gt_thickness", 2),
        pred_thickness=viz_cfg.get("pred_thickness", 1),
        gt_color_rgb=tuple(viz_cfg.get("gt_color_rgb", [160, 32, 240])),
        pred_color_rgb=tuple(viz_cfg.get("pred_color_rgb", [0, 200, 0])),
        text_scale=viz_cfg.get("text_scale", 0.4),
        dpi=viz_cfg.get("dpi", 150),
    )
    viz.on_train_start(trainer)
    viz.on_epoch_end(trainer, epoch=max(epoch, 0), metrics={"val/mAP50": 0.0})

    default_out = run_dir / "val_predictions" / f"epoch_{max(epoch, 0) + 1:03d}.png"
    if args.out_name and default_out.exists():
        default_out.rename(run_dir / "val_predictions" / args.out_name)
        print(f"Saved: {run_dir / 'val_predictions' / args.out_name}")
    else:
        print(f"Saved: {default_out}")


if __name__ == "__main__":
    main()

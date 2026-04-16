"""CLI entrypoint for model training.

Supports three backends selected via ``training.backend`` in the config YAML:

- ``pytorch`` (default): Custom PyTorch trainer with full control (EMA, per-component LR,
  custom loss). Alias ``native`` accepted for backward compatibility.
- ``hf``: HuggingFace Trainer with DDP, DeepSpeed, gradient accumulation.
- ``custom``: Dynamically import a trainer class specified by
  ``training.custom_trainer_class`` (e.g. ``my_pkg.trainers.MyTrainer``).

Usage:
    python train.py --config features/safety-fire_detection/configs/06_training.yaml
    python train.py --config features/safety-fire_detection/configs/06_training.yaml --resume runs/fire_detection/last.pth
    python train.py --config features/safety-fire_detection/configs/06_training.yaml --override training.lr=0.005 training.epochs=100
"""

import argparse
import importlib
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

# Pick the idle GPU before any torch cuda init. Respects an explicit
# CUDA_VISIBLE_DEVICES if the user set one. Safe to call before `import
# torch` (auto_select_gpu uses only subprocess + env vars).
from utils.device import auto_select_gpu  # noqa: E402
auto_select_gpu()

from utils.config import load_config, parse_overrides  # noqa: E402


def main() -> None:
    """Parse CLI arguments and launch training."""
    parser = argparse.ArgumentParser(
        description="Config-driven object detection training pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training YAML config file (e.g. features/safety-fire_detection/configs/06_training.yaml).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint file to resume training from.",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Config overrides as key=value pairs (e.g. training.lr=0.005 training.epochs=100).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force a specific device (e.g. 'cuda:0', 'cpu'). Auto-detects if not set.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level. Default: INFO.",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("train")

    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        log.error("Config file not found: %s", config_path)
        sys.exit(1)

    # Parse overrides
    overrides = {}
    if args.override:
        try:
            overrides = parse_overrides(args.override)
            log.info("Config overrides: %s", overrides)
        except ValueError as e:
            log.error("Invalid override: %s", e)
            sys.exit(1)

    if args.device:
        overrides["device"] = args.device

    overrides_or_none = overrides or None

    config = load_config(str(config_path))
    training_cfg = config.get("training", {})
    backend = training_cfg.get("backend", "pytorch")
    log.info("Training backend: %s", backend)

    def _maybe_resume(trainer_obj: Any) -> None:
        """Load checkpoint into *trainer_obj* when --resume is set."""
        if not args.resume:
            return
        resume_path = Path(args.resume)
        if not resume_path.exists():
            log.error("Resume checkpoint not found: %s", resume_path)
            sys.exit(1)
        trainer_obj.load_checkpoint(str(resume_path))
        log.info("Resuming from checkpoint: %s", resume_path)

    try:
        if backend == "hf":
            from core.p06_training.hf_trainer import train_with_hf

            summary = train_with_hf(
                config_path=str(config_path),
                overrides=overrides_or_none,
                resume_from=args.resume,
            )
            log.info("Training complete (HF Trainer).")
            log.info("  Total epochs: %d", summary.get("total_epochs", 0))
        elif backend == "custom":
            custom_class_path = training_cfg.get("custom_trainer_class")
            if not custom_class_path:
                log.error(
                    "backend='custom' requires training.custom_trainer_class in config. "
                    "Example: training.custom_trainer_class: my_pkg.trainers.MyTrainer"
                )
                sys.exit(1)
            module_path, class_name = custom_class_path.rsplit(".", 1)
            TrainerClass = getattr(importlib.import_module(module_path), class_name)
            trainer = TrainerClass(config_path=str(config_path), overrides=overrides_or_none)
            _maybe_resume(trainer)
            summary = trainer.train()
            log.info("Training complete (custom: %s).", custom_class_path)
            log.info("  Total epochs: %d", summary.get("total_epochs", 0))
        elif backend in ("pytorch", "native"):
            from core.p06_training.trainer import DetectionTrainer

            trainer = DetectionTrainer(config_path=str(config_path), overrides=overrides_or_none)
            _maybe_resume(trainer)
            summary = trainer.train()
            log.info("Training complete (pytorch).")
            log.info("  Best metric: %.4f at epoch %d", summary["best_metric"] or 0.0, summary["best_epoch"])
            log.info("  Total epochs: %d", summary["total_epochs"])
        else:
            log.error(
                "Unknown training backend: '%s'. Valid values: pytorch, hf, custom", backend
            )
            sys.exit(1)

    except KeyboardInterrupt:
        log.info("Training interrupted by user.")
        sys.exit(0)
    except Exception:
        log.exception("Training failed with error:")
        sys.exit(1)


if __name__ == "__main__":
    main()

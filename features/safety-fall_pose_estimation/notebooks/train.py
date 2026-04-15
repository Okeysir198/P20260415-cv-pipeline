#!/usr/bin/env python3
"""Train Fall Pose Estimation model.

Usage:
    python features/safety-fall_pose_estimation/experiments/train.py
    python features/safety-fall_pose_estimation/experiments/train.py --resume runs/fall_pose_estimation/last.pth
    python features/safety-fall_pose_estimation/experiments/train.py --override training.lr=0.005
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from core.p06_training.trainer import DetectionTrainer
from core.p06_training.train import parse_overrides
from utils.config import load_config, merge_configs

DEFAULT_CONFIG = "features/safety-fall_pose_estimation/configs/06_training.yaml"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train Fall Pose Estimation model")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Training config path")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--override", nargs="*", default=[], help="Config overrides (key=value)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.override:
        overrides = parse_overrides(args.override)
        config = merge_configs(config, overrides)

    trainer = DetectionTrainer(config, resume_path=args.resume)
    trainer.train()


if __name__ == "__main__":
    main()

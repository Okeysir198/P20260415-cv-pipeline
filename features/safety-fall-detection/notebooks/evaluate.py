#!/usr/bin/env python3
"""Evaluate Fall Classification model.

Usage:
    python features/safety-fall-detection/experiments/evaluate.py
    python features/safety-fall-detection/experiments/evaluate.py --model runs/fall_detection/best.pt --split test
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from core.p08_evaluation.evaluator import ModelEvaluator
from core.p02_models import build_model
from utils.config import load_config
from utils.device import get_device

DEFAULT_MODEL = "runs/fall_detection/best.pt"
DEFAULT_DATA_CONFIG = "features/safety-fall-detection/configs/05_data.yaml"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Fall Classification model")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model checkpoint path")
    parser.add_argument("--config", default=DEFAULT_DATA_CONFIG, help="Data config path")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Dataset split")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    data_config = load_config(args.config)
    from core.p08_evaluation.evaluate import load_model

    device = get_device()
    model = load_model(args.model, data_config, device)

    evaluator = ModelEvaluator(model=model, data_config=data_config, device=device, conf_threshold=args.conf)
    results = evaluator.evaluate(split=args.split)
    evaluator.print_results(results)


if __name__ == "__main__":
    main()

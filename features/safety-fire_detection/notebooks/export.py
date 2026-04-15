#!/usr/bin/env python3
"""Export Fire Detection model to ONNX.

Usage:
    python features/safety-fire_detection/experiments/export.py
    python features/safety-fire_detection/experiments/export.py --model runs/fire_detection/best.pt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from core.p09_export.exporter import ModelExporter
from core.p09_export.quantize import ModelQuantizer
from core.p02_models import build_model
from utils.config import load_config
from utils.device import get_device

DEFAULT_MODEL = "runs/fire_detection/best.pt"
DEFAULT_TRAINING_CONFIG = "features/safety-fire_detection/configs/06_training.yaml"
DEFAULT_EXPORT_CONFIG = "configs/_shared/09_export.yaml"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Export Fire Detection model to ONNX")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model checkpoint path")
    parser.add_argument("--training-config", default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument("--export-config", default=DEFAULT_EXPORT_CONFIG)
    parser.add_argument("--optimize", choices=["O1", "O2", "O3", "O4"], default=None,
                        help="Graph optimization level")
    parser.add_argument("--quantize", choices=["dynamic", "static"], default=None,
                        help="Quantization mode")
    args = parser.parse_args()

    training_config = load_config(args.training_config)
    export_config = load_config(args.export_config)

    device = get_device()
    model = build_model(training_config)

    import torch

    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    exporter = ModelExporter(model, export_config, model_name="fire")
    onnx_path = exporter.export_onnx()
    print(f"Exported: {onnx_path}")

    if args.optimize:
        quantizer = ModelQuantizer(onnx_path)
        onnx_path = quantizer.optimize(level=args.optimize)
        print(f"Optimized: {onnx_path}")

    if args.quantize:
        quantizer = ModelQuantizer(onnx_path)
        if args.quantize == "dynamic":
            quant_path = quantizer.quantize_dynamic()
        else:
            quant_path = quantizer.quantize_dynamic()  # fallback, static needs data
        print(f"Quantized: {quant_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CLI entrypoint for exporting trained detection models to ONNX.

Usage:
    python export.py \
        --model runs/fire_detection/best.pt \
        --training-config features/safety-fire_detection/configs/06_training.yaml \
        --export-config configs/_shared/09_export.yaml \
        [--version 1] \
        [--output-dir ../../models/]

Loads a trained PyTorch checkpoint, exports to ONNX, validates the
exported model, and prints model info (size, params).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

import torch

from core.p05_data.detection_dataset import build_dataloader
from core.p06_models import build_model
from core.p09_export.exporter import ModelExporter
from core.p09_export.quantize import ModelQuantizer
from loguru import logger
from utils.config import load_config
from utils.device import get_device

logger.remove()
logger.add(sys.stderr, level="INFO")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export trained detection model to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt / .pth)",
    )
    parser.add_argument(
        "--training-config",
        type=str,
        required=True,
        help="Path to training config YAML (for model arch info)",
    )
    parser.add_argument(
        "--export-config",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent.parent
            / "configs"
            / "_shared"
            / "09_export.yaml"
        ),
        help="Path to export config YAML (default: configs/_shared/09_export.yaml)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="1",
        help="Model version string (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from export config",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for export (default: auto-detect)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip ONNX validation step",
    )
    parser.add_argument(
        "--skip-simplify",
        action="store_true",
        help="Skip ONNX simplification even if configured",
    )
    parser.add_argument(
        "--optimize",
        type=str,
        default=None,
        choices=["O1", "O2", "O3", "O4"],
        help="Graph optimization level (default: from config or skip)",
    )
    parser.add_argument(
        "--skip-optimize",
        action="store_true",
        help="Skip graph optimization even if configured",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default=None,
        choices=["dynamic", "static"],
        help="Quantization mode (default: from config or skip)",
    )
    parser.add_argument(
        "--quant-preset",
        type=str,
        default=None,
        choices=["avx512_vnni", "arm64", "avx2"],
        help="Quantization hardware preset (default: from config or avx512_vnni)",
    )
    return parser.parse_args()


def _load_model_from_checkpoint(
    checkpoint_path: str,
    training_config: dict,
    device: torch.device,
) -> torch.nn.Module:
    """Load a detection model from a checkpoint file using the model registry.

    Supports two checkpoint formats:
    1. Full checkpoint dict with 'model' key (from training pipeline).
    2. Raw state_dict (direct torch.save of model.state_dict()).

    The model architecture is resolved via the model registry
    (:func:`models.build_model`), which supports YOLOX, D-FINE, RT-DETRv2,
    and any future registered architecture.

    Args:
        checkpoint_path: Path to the .pt/.pth file.
        training_config: Training config dict (for model arch, num_classes, etc.).
        device: Device to load the model onto.

    Returns:
        Loaded model in eval mode on the specified device.
    """
    logger.info("Loading checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_cfg = training_config.get("model", {})
    arch = model_cfg.get("arch", "yolox-m")
    num_classes = model_cfg.get("num_classes", 80)

    # Extract state_dict from checkpoint (support multiple key conventions)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            # Could be a state_dict or a full model object
            if isinstance(checkpoint["model"], dict):
                state_dict = checkpoint["model"]
            elif isinstance(checkpoint["model"], torch.nn.Module):
                # Full model object saved directly
                model = checkpoint["model"]
                model = model.to(device)
                model.eval()
                logger.info("Model loaded from checkpoint object: %s (num_classes=%d)", arch, num_classes)
                return model
            else:
                state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    elif isinstance(checkpoint, torch.nn.Module):
        model = checkpoint.to(device)
        model.eval()
        return model
    else:
        state_dict = checkpoint

    # Handle DataParallel/DistributedDataParallel prefixes
    if isinstance(state_dict, dict) and any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    # Build model from checkpoint config or training config
    build_config = checkpoint["config"] if isinstance(checkpoint, dict) and "config" in checkpoint else training_config
    model = build_model(build_config)
    if isinstance(state_dict, dict):
        model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    logger.info("Model loaded via registry: %s (num_classes=%d)", arch, num_classes)
    return model


def _format_params(n: int) -> str:
    """Format parameter count for display."""
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def _format_flops(flops) -> str:
    """Format FLOPs for display."""
    if flops is None:
        return "N/A"
    if flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPs"
    return f"{flops:.0f} FLOPs"


def main():
    """Main export workflow."""
    args = parse_args()

    # Load configs
    training_config = load_config(args.training_config)
    export_config = load_config(args.export_config)

    # Override from CLI args or auto-generate timestamped dir
    if args.output_dir:
        export_config["output_dir"] = args.output_dir
    elif "output_dir" not in export_config:
        from utils.config import generate_run_dir

        config_path = Path(args.training_config)
        # features/<name>/configs/06_training.yaml → parent.parent.name = <name>
        log_cfg = training_config.get("logging", {})
        run_name = (
            log_cfg.get("run_name")
            or log_cfg.get("project")
            or config_path.parent.parent.name
        )
        export_config["output_dir"] = str(generate_run_dir(run_name, "09_export"))
    if args.skip_simplify:
        export_config["simplify"] = False

    # Determine device
    device = get_device(args.device)
    logger.info("Using device: %s", device)

    # Load model
    model = _load_model_from_checkpoint(args.model, training_config, device)

    # Determine model name from training config
    model_cfg = training_config.get("model", {})
    arch = model_cfg.get("arch", "yolox-m")
    # Derive model name from training config run_name or logging section
    logging_cfg = training_config.get("logging", {})
    run_name = logging_cfg.get("run_name", "model")
    # Extract model name: "fire_yoloxm_v1" -> "fire"
    model_name = run_name.split("_")[0] if "_" in run_name else run_name

    # Create exporter
    exporter = ModelExporter(
        model=model,
        config=export_config,
        model_name=model_name,
        version=args.version,
    )

    # Export
    onnx_path = exporter.export_onnx()
    logger.info("Exported ONNX model: %s", onnx_path)

    # Validate
    if not args.skip_validation:
        try:
            valid = exporter.validate_onnx(onnx_path)
            if valid:
                logger.info("ONNX validation: PASSED")
            else:
                logger.warning(
                    "ONNX validation: FAILED (outputs differ beyond tolerance). "
                    "Model may still be usable — check accuracy on test data."
                )
        except Exception as e:
            logger.warning("Skipping validation: %s", e)
    else:
        logger.info("Skipping ONNX validation (--skip-validation)")

    # Graph optimization (if requested)
    optimized_path = None
    optimize_level = args.optimize or export_config.get("optimization_level")
    if optimize_level and not args.skip_optimize:
        quantizer = ModelQuantizer(onnx_path)
        optimized_path = quantizer.optimize(level=optimize_level)
        logger.info("Optimized ONNX model: %s", optimized_path)
        # Use optimized model as input for quantization
        onnx_path = optimized_path

    # Quantization (if requested)
    quant_path = None
    quant_mode = args.quantize
    quant_config = export_config.get("quantization", {})
    if quant_mode is None and quant_config.get("enabled", False):
        quant_mode = quant_config.get("mode", "dynamic")

    if quant_mode:
        quantizer = ModelQuantizer(onnx_path)

        if quant_mode == "dynamic":
            quant_path = quantizer.quantize_dynamic()
        elif quant_mode == "static":
            data_config_path = training_config.get("data", {}).get("config")
            if data_config_path:
                data_config = load_config(data_config_path)
                cal_loader = build_dataloader(data_config, split="val", batch_size=1)
                quant_path = quantizer.quantize_static(cal_loader)
            else:
                logger.warning("Static quantization requires data config. Falling back to dynamic.")
                quant_path = quantizer.quantize_dynamic()

        logger.info("Quantized model: %s", quant_path)

    # Model info
    info = exporter.get_model_info()
    onnx_size_mb = Path(onnx_path).stat().st_size / (1024 * 1024)

    print("\n" + "=" * 60)
    print("  Export Summary")
    print("=" * 60)
    print(f"  Model name     : {model_name}")
    print(f"  Architecture   : {arch}")
    print(f"  Input size     : {info['input_size']}")
    print(f"  Total params   : {_format_params(info['total_params'])}")
    print(f"  Trainable      : {_format_params(info['trainable_params'])}")
    print(f"  FLOPs          : {_format_flops(info['flops_estimate'])}")
    print(f"  PT size (est)  : {info['model_size_mb']:.2f} MB")
    print(f"  ONNX size      : {onnx_size_mb:.2f} MB")
    print(f"  ONNX path      : {onnx_path}")
    if optimized_path:
        print(f"  Optimized      : {optimized_path} ({onnx_size_mb:.2f} MB, level={optimize_level})")
    if quant_path:
        quant_size_mb = Path(quant_path).stat().st_size / (1024 * 1024)
        print(f"  Quantized      : {quant_path} ({quant_size_mb:.2f} MB, mode={quant_mode})")
    print(f"  Version        : v{args.version}")
    print("=" * 60)

    return onnx_path


if __name__ == "__main__":
    main()

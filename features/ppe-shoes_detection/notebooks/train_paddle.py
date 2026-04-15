#!/usr/bin/env python3
"""Train PicoDet using PaddleDetection for Safety Shoes Detection.

This is a Level 3 (fully standalone) experiment script. PaddleDetection runs in
its own environment — this script handles data conversion, training invocation,
and ONNX export. The resulting ONNX model can be evaluated and deployed using
the standard pipeline (evaluate.py, inference.py).

Prerequisites:
    1. Clone PaddleDetection:
       git clone https://github.com/PaddlePaddle/PaddleDetection.git ~/repos/PaddleDetection

    2. Install PaddlePaddle + PaddleDetection + paddle2onnx:
       pip install paddlepaddle-gpu paddle2onnx
       cd ~/repos/PaddleDetection && pip install -e .

    3. Convert dataset (one-time):
       uv run python utils/paddle_bridge.py \
           --data-config features/ppe-shoes_detection/configs/05_data.yaml

Usage:
    python features/ppe-shoes_detection/experiments/train_paddle.py
    python features/ppe-shoes_detection/experiments/train_paddle.py --paddle-dir ~/repos/PaddleDetection
    python features/ppe-shoes_detection/experiments/train_paddle.py --export-only --model-dir output/picodet_s
    python features/ppe-shoes_detection/experiments/train_paddle.py --epochs 120 --batch-size 32
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from utils.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_CONFIG = "features/ppe-shoes_detection/configs/05_data.yaml"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "shoes_detection"
DEFAULT_PADDLE_DIR = Path.home() / "repos" / "PaddleDetection"


def convert_dataset(data_config: str) -> Path:
    """Convert YOLO labels to COCO JSON if not already done."""
    config = load_config(data_config)
    config_dir = Path(data_config).resolve().parent
    dataset_path = (config_dir / config["path"]).resolve()
    annotations_dir = dataset_path / "annotations"

    if (annotations_dir / "train.json").exists():
        print(f"COCO annotations already exist at {annotations_dir}, skipping conversion.")
        return annotations_dir

    print("Converting YOLO labels to COCO JSON...")
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "paddle_bridge" / "yolo_to_coco.py"),
            "--data-config",
            data_config,
        ],
        check=True,
    )
    return annotations_dir


def generate_paddle_config(
    data_config: str,
    paddle_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> Path:
    """Generate a PaddleDetection YAML config for PicoDet-S.

    Writes a config file that inherits from PaddleDetection's base PicoDet config
    and overrides dataset paths, class count, and training hyperparameters.
    """
    config = load_config(data_config)
    config_dir = Path(data_config).resolve().parent
    dataset_path = (config_dir / config["path"]).resolve()
    annotations_dir = dataset_path / "annotations"
    num_classes = config["num_classes"]
    input_h, input_w = config.get("input_size", [640, 640])

    # PicoDet-S base config path in PaddleDetection
    base_config = "configs/picodet/picodet_s_320_coco_lcnet.yml"

    paddle_config = f"""# Auto-generated PicoDet-S config for {config['dataset_name']}
# Base: PaddleDetection/{base_config}

_BASE_: [
  '{paddle_dir / base_config}',
]

metric: COCO
num_classes: {num_classes}

TrainDataset:
  !COCODataSet
    image_dir: {dataset_path / 'train' / 'images'}
    anno_path: {annotations_dir / 'train.json'}
    dataset_dir: .
    data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']

EvalDataset:
  !COCODataSet
    image_dir: {dataset_path / 'val' / 'images'}
    anno_path: {annotations_dir / 'val.json'}
    dataset_dir: .

TestDataset:
  !ImageFolder
    anno_path: {annotations_dir / 'val.json'}

epoch: {epochs}
LearningRate:
  base_lr: {learning_rate}
  schedulers:
  - !CosineDecay
    max_epochs: {epochs}
  - !LinearWarmup
    start_factor: 0.1
    steps: 300

TrainReader:
  batch_size: {batch_size}

EvalReader:
  batch_size: {batch_size}

# Input size override (PicoDet default is 320, adjust if needed)
# Uncomment to change:
# PicoDet:
#   head:
#     input_size: [{input_h}, {input_w}]
"""

    out_path = PROJECT_ROOT / "experiments" / "shoes_detection" / "picodet_s_shoes.yml"
    out_path.write_text(paddle_config)
    print(f"PaddleDetection config written to: {out_path}")
    return out_path


def train(paddle_dir: Path, config_path: Path) -> Path:
    """Run PaddleDetection training."""
    print(f"\nStarting PicoDet training with PaddleDetection...")
    print(f"  Config: {config_path}")
    print(f"  PaddleDetection: {paddle_dir}")

    output_dir = PROJECT_ROOT / "runs" / "shoes_detection" / "paddle"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(paddle_dir / "tools" / "train.py"),
        "-c",
        str(config_path),
        "-o",
        f"save_dir={output_dir}",
    ]

    print(f"  Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    return output_dir


def export_onnx(paddle_dir: Path, config_path: Path, model_dir: Path) -> Path:
    """Export trained PaddleDetection model to ONNX via paddle2onnx."""
    print(f"\nExporting to Paddle inference model...")

    infer_dir = model_dir / "inference"
    infer_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Export to Paddle inference format
    subprocess.run(
        [
            sys.executable,
            str(paddle_dir / "tools" / "export_model.py"),
            "-c",
            str(config_path),
            "-o",
            f"weights={model_dir / 'best_model.pdparams'}",
            f"export_dir={infer_dir}",
        ],
        check=True,
    )

    # Step 2: Convert Paddle inference model to ONNX
    onnx_output = OUTPUT_DIR / "picodet_s.onnx"
    onnx_output.parent.mkdir(parents=True, exist_ok=True)

    # Find the .pdmodel file
    pdmodel_files = list(infer_dir.rglob("*.pdmodel"))
    if not pdmodel_files:
        raise FileNotFoundError(f"No .pdmodel found in {infer_dir}")

    model_file = pdmodel_files[0]
    params_file = model_file.with_suffix(".pdiparams")

    print(f"Converting to ONNX: {model_file} → {onnx_output}")
    subprocess.run(
        [
            "paddle2onnx",
            "--model_dir",
            str(model_file.parent),
            "--model_filename",
            model_file.name,
            "--params_filename",
            params_file.name,
            "--save_file",
            str(onnx_output),
            "--opset_version",
            "13",
        ],
        check=True,
    )

    print(f"\nONNX model saved to: {onnx_output}")
    print("You can now evaluate with the standard pipeline:")
    print(f"  uv run features/ppe-shoes_detection/experiments/evaluate.py "
          f"--model {onnx_output.relative_to(PROJECT_ROOT)}")
    return onnx_output


def main():
    parser = argparse.ArgumentParser(description="Train PicoDet-S with PaddleDetection")
    parser.add_argument(
        "--paddle-dir",
        type=Path,
        default=DEFAULT_PADDLE_DIR,
        help=f"Path to PaddleDetection clone (default: {DEFAULT_PADDLE_DIR})",
    )
    parser.add_argument(
        "--data-config",
        default=DATA_CONFIG,
        help="Pipeline data config path",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.06, help="Base learning rate")
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Skip training, only export existing model to ONNX",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Model directory for --export-only",
    )
    args = parser.parse_args()

    if not args.paddle_dir.exists():
        print(f"Error: PaddleDetection not found at {args.paddle_dir}")
        print("Clone it first:")
        print(f"  git clone https://github.com/PaddlePaddle/PaddleDetection.git {args.paddle_dir}")
        sys.exit(1)

    # Step 1: Convert dataset
    convert_dataset(args.data_config)

    # Step 2: Generate PaddleDetection config
    config_path = generate_paddle_config(
        args.data_config,
        args.paddle_dir,
        args.epochs,
        args.batch_size,
        args.lr,
    )

    if args.export_only:
        # Export only
        model_dir = args.model_dir or (OUTPUT_DIR / "paddle")
        export_onnx(args.paddle_dir, config_path, model_dir)
    else:
        # Step 3: Train
        model_dir = train(args.paddle_dir, config_path)
        # Step 4: Export to ONNX
        export_onnx(args.paddle_dir, config_path, model_dir)


if __name__ == "__main__":
    main()

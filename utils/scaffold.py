#!/usr/bin/env python3
"""Scaffold a complete use-case workspace from a single command.

Usage:
    uv run utils/scaffold.py <usecase_name> --model yolox-m --classes "0:car,1:truck,2:bus"
    uv run utils/scaffold.py vehicle_detection --model dfine-s --classes "0:car,1:truck" --dry-run
    uv run utils/scaffold.py vehicle_detection --model yolox-m --classes "0:car,1:truck" --force

Generates:
    configs/<usecase>/05_data.yaml
    configs/<usecase>/06_training.yaml
    experiments/<usecase>/train.py
    experiments/<usecase>/evaluate.py
    experiments/<usecase>/export.py
    experiments/<usecase>/inference.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # ai/ root

from utils.yolo_io import parse_classes

# Pretrained weight files per model arch
PRETRAINED_WEIGHTS = {
    "yolox-m": "../../pretrained/yolox_m.pth",
    "yolox-tiny": "../../pretrained/yolox_tiny.pth",
    "dfine-s": "true",
    "dfine-n": "true",
    "dfine-m": "true",
    "rtdetrv2-r18": "true",
}

# YOLOX depth/width per arch
YOLOX_PARAMS = {
    "yolox-m": {"depth": 0.67, "width": 0.75},
    "yolox-tiny": {"depth": 0.33, "width": 0.375},
}


def _title(usecase: str) -> str:
    return usecase.replace("_", " ").title()


def build_05_data_yaml(usecase: str, classes: dict[int, str]) -> str:
    num_classes = len(classes)
    names_lines = "\n".join(f"  {k}: {v}" for k, v in sorted(classes.items()))
    title = _title(usecase)
    return (
        f"# Dataset config — {title}\n"
        f'dataset_name: "{usecase}"\n'
        f'path: "../../dataset_store/{usecase}"\n'
        f'train: "train/images"\n'
        f'val: "val/images"\n'
        f'test: "test/images"\n'
        f"\n"
        f"names:\n"
        f"{names_lines}\n"
        f"num_classes: {num_classes}\n"
        f"input_size: [640, 640]\n"
        f"\n"
        f"# Normalization (ImageNet defaults, updated by compute_normalization.py)\n"
        f"mean: [0.485, 0.456, 0.406]\n"
        f"std: [0.229, 0.224, 0.225]\n"
    )


def build_06_training_yaml(usecase: str, model: str, classes: dict[int, str]) -> str:
    num_classes = len(classes)
    pretrained = PRETRAINED_WEIGHTS.get(model, "true")
    run_name = f"{usecase}_{model.replace('-', '_')}_v1"
    title = _title(usecase)
    loss_type = "yolox" if model.startswith("yolox") else "detr-passthrough"

    lines = [
        f"# Training config — {title} ({model.upper()})",
        "model:",
        f"  arch: {model}",
        f"  pretrained: {pretrained}",
        f"  num_classes: {num_classes}",
        "  input_size: [640, 640]",
    ]
    if model in YOLOX_PARAMS:
        p = YOLOX_PARAMS[model]
        lines += [f"  depth: {p['depth']}", f"  width: {p['width']}"]

    lines += [
        "",
        "data:",
        "  dataset_config: 05_data.yaml",
        "  batch_size: 16",
        "  num_workers: 4",
        "  pin_memory: true",
        "",
        "augmentation:",
        "  mosaic: true",
        "  mixup: true",
        "  hsv_h: 0.015",
        "  hsv_s: 0.7",
        "  hsv_v: 0.4",
        "  fliplr: 0.5",
        "  flipud: 0.0",
        "  scale: [0.1, 2.0]",
        "  degrees: 10.0",
        "  translate: 0.1",
        "  shear: 2.0",
        "",
        "training:",
        "  backend: pytorch",
        "  epochs: 200",
        "  optimizer: sgd",
        "  lr: 0.01",
        "  momentum: 0.9",
        "  weight_decay: 0.0005",
        "  warmup_epochs: 5",
        "  scheduler: cosine",
        "  patience: 50",
        "  amp: true",
        "  grad_clip: 35.0",
        "  ema: true",
        "  ema_decay: 0.9998",
        "",
        "loss:",
        f"  type: {loss_type}",
        "",
        "logging:",
        "  wandb_project: smart-camera",
        f"  run_name: {run_name}",
        "",
        "checkpoint:",
        "  save_best: true",
        "  metric: val/mAP50",
        "  mode: max",
        "  save_interval: 10",
        "",
        "seed: 42",
        "",
    ]
    return "\n".join(lines)


def build_train_py(usecase: str) -> str:
    title = _title(usecase)
    return '\n'.join([
        "#!/usr/bin/env python3",
        f'"""Train {title} model.',
        "",
        "Usage:",
        f"    python experiments/{usecase}/train.py",
        f"    python experiments/{usecase}/train.py --resume runs/{usecase}/last.pth",
        f"    python experiments/{usecase}/train.py --override training.lr=0.005",
        '"""',
        "",
        "import sys",
        "from pathlib import Path",
        "",
        "sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root",
        "",
        "from core.p06_training.trainer import DetectionTrainer",
        "from core.p06_training.train import parse_overrides",
        "from utils.config import load_config, merge_configs",
        "",
        f'DEFAULT_CONFIG = "configs/{usecase}/06_training.yaml"',
        "",
        "",
        "def main():",
        "    import argparse",
        "",
        f'    parser = argparse.ArgumentParser(description="Train {title} model")',
        '    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Training config path")',
        '    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")',
        '    parser.add_argument("--override", nargs="*", default=[], help="Config overrides (key=value)")',
        "    args = parser.parse_args()",
        "",
        "    config = load_config(args.config)",
        "    if args.override:",
        "        overrides = parse_overrides(args.override)",
        "        config = merge_configs(config, overrides)",
        "",
        "    trainer = DetectionTrainer(config, resume_path=args.resume)",
        "    trainer.train()",
        "",
        "",
        'if __name__ == "__main__":',
        "    main()",
        "",
    ])


def build_evaluate_py(usecase: str) -> str:
    title = _title(usecase)
    return '\n'.join([
        "#!/usr/bin/env python3",
        f'"""Evaluate {title} model.',
        "",
        "Usage:",
        f"    python experiments/{usecase}/evaluate.py",
        f"    python experiments/{usecase}/evaluate.py --model runs/{usecase}/best.pt --split test",
        '"""',
        "",
        "import sys",
        "from pathlib import Path",
        "",
        "sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root",
        "",
        "from core.p08_evaluation.evaluator import ModelEvaluator",
        "from core.p06_models import build_model",
        "from utils.config import load_config",
        "from utils.device import get_device",
        "",
        f'DEFAULT_MODEL = "runs/{usecase}/best.pt"',
        f'DEFAULT_DATA_CONFIG = "configs/{usecase}/05_data.yaml"',
        "",
        "",
        "def main():",
        "    import argparse",
        "",
        f'    parser = argparse.ArgumentParser(description="Evaluate {title} model")',
        '    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model checkpoint path")',
        '    parser.add_argument("--config", default=DEFAULT_DATA_CONFIG, help="Data config path")',
        '    parser.add_argument("--split", default="val", choices=["val", "test"], help="Dataset split")',
        '    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")',
        "    args = parser.parse_args()",
        "",
        "    data_config = load_config(args.config)",
        "    from core.p08_evaluation.evaluate import load_model",
        "",
        "    device = get_device()",
        "    model = load_model(args.model, data_config, device)",
        "",
        "    evaluator = ModelEvaluator(model=model, data_config=data_config, device=device, conf_threshold=args.conf)",
        "    results = evaluator.evaluate(split=args.split)",
        "    evaluator.print_results(results)",
        "",
        "",
        'if __name__ == "__main__":',
        "    main()",
        "",
    ])


def build_export_py(usecase: str) -> str:
    title = _title(usecase)
    short = usecase.replace("_detection", "").replace("_", "")
    return '\n'.join([
        "#!/usr/bin/env python3",
        f'"""Export {title} model to ONNX.',
        "",
        "Usage:",
        f"    python experiments/{usecase}/export.py",
        f"    python experiments/{usecase}/export.py --model runs/{usecase}/best.pt",
        '"""',
        "",
        "import sys",
        "from pathlib import Path",
        "",
        "sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root",
        "",
        "from core.p09_export.exporter import ModelExporter",
        "from core.p09_export.quantize import ModelQuantizer",
        "from core.p06_models import build_model",
        "from utils.config import load_config",
        "from utils.device import get_device",
        "",
        f'DEFAULT_MODEL = "runs/{usecase}/best.pt"',
        f'DEFAULT_TRAINING_CONFIG = "configs/{usecase}/06_training.yaml"',
        f'DEFAULT_EXPORT_CONFIG = "configs/_shared/09_export.yaml"',
        "",
        "",
        "def main():",
        "    import argparse",
        "",
        f'    parser = argparse.ArgumentParser(description="Export {title} model to ONNX")',
        '    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model checkpoint path")',
        "    parser.add_argument(\"--training-config\", default=DEFAULT_TRAINING_CONFIG)",
        "    parser.add_argument(\"--export-config\", default=DEFAULT_EXPORT_CONFIG)",
        '    parser.add_argument("--optimize", choices=["O1", "O2", "O3", "O4"], default=None,',
        '                        help="Graph optimization level")',
        '    parser.add_argument("--quantize", action="store_true",',
        '                        help="Apply dynamic INT8 quantization")',
        "    args = parser.parse_args()",
        "",
        "    training_config = load_config(args.training_config)",
        "    export_config = load_config(args.export_config)",
        "",
        "    device = get_device()",
        "    model = build_model(training_config)",
        "",
        "    import torch",
        "",
        "    checkpoint = torch.load(args.model, map_location=device, weights_only=False)",
        '    if isinstance(checkpoint, dict) and "model" in checkpoint:',
        '        model.load_state_dict(checkpoint["model"])',
        "    else:",
        "        model.load_state_dict(checkpoint)",
        "    model.eval()",
        "",
        f'    exporter = ModelExporter(model, export_config, model_name="{short}")',
        "    onnx_path = exporter.export_onnx()",
        '    print(f"Exported: {onnx_path}")',
        "",
        "    if args.optimize:",
        "        quantizer = ModelQuantizer(onnx_path)",
        "        onnx_path = quantizer.optimize(level=args.optimize)",
        '        print(f"Optimized: {onnx_path}")',
        "",
        "    if args.quantize:",
        "        quantizer = ModelQuantizer(onnx_path)",
        "        quant_path = quantizer.quantize_dynamic()",
        '        print(f"Quantized: {quant_path}")',
        "",
        "",
        'if __name__ == "__main__":',
        "    main()",
        "",
    ])


def build_inference_py(usecase: str) -> str:
    title = _title(usecase)
    return '\n'.join([
        "#!/usr/bin/env python3",
        f'"""Run inference with {title} model.',
        "",
        "Usage:",
        f"    python experiments/{usecase}/inference.py --image path/to/image.jpg",
        f"    python experiments/{usecase}/inference.py --video path/to/video.mp4",
        '"""',
        "",
        "import sys",
        "from pathlib import Path",
        "",
        "sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root",
        "",
        "from core.p10_inference.predictor import DetectionPredictor",
        "from utils.config import load_config",
        "",
        f'DEFAULT_MODEL = "runs/{usecase}/best.pt"',
        f'DEFAULT_DATA_CONFIG = "configs/{usecase}/05_data.yaml"',
        "",
        "",
        "def main():",
        "    import argparse",
        "",
        f'    parser = argparse.ArgumentParser(description="Run {title} inference")',
        '    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model path (.pt or .onnx)")',
        '    parser.add_argument("--config", default=DEFAULT_DATA_CONFIG, help="Data config path")',
        '    parser.add_argument("--image", type=str, help="Image path for single-image inference")',
        '    parser.add_argument("--video", type=str, help="Video path for video inference")',
        '    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")',
        '    parser.add_argument("--save-dir", type=str, default=None, help="Save directory")',
        "    args = parser.parse_args()",
        "",
        "    data_config = load_config(args.config)",
        "    predictor = DetectionPredictor(model_path=args.model, data_config=data_config, conf_threshold=args.conf)",
        "",
        "    if args.image:",
        "        import cv2",
        "",
        "        image = cv2.imread(args.image)",
        "        results = predictor.predict(image)",
        "        print(f\"Detections: {len(results.get('boxes', []))}\")",
        "        if args.save_dir:",
        "            vis = predictor.visualize(image, results)",
        "            save_path = Path(args.save_dir) / Path(args.image).name",
        "            save_path.parent.mkdir(parents=True, exist_ok=True)",
        "            cv2.imwrite(str(save_path), vis)",
        '            print(f"Saved to {save_path}")',
        "    elif args.video:",
        "        from core.p10_inference.video_inference import VideoProcessor",
        "",
        "        processor = VideoProcessor(predictor=predictor)",
        "        processor.process_video(args.video, save_dir=args.save_dir)",
        "    else:",
        '        parser.error("Provide --image or --video")',
        "",
        "",
        'if __name__ == "__main__":',
        "    main()",
        "",
    ])


def write_file(path: Path, content: str, dry_run: bool, force: bool) -> bool:
    """Write file, returning True if written (or would be written in dry-run)."""
    if path.exists() and not force:
        print(f"  SKIP   {path}  (already exists, use --force to overwrite)")
        return False
    if dry_run:
        print(f"  DRY    {path}")
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"  CREATE {path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a complete use-case workspace (configs + experiment scripts).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("usecase", help="Use-case name, e.g. vehicle_detection")
    parser.add_argument(
        "--model",
        default="yolox-m",
        choices=list(PRETRAINED_WEIGHTS.keys()),
        help="Model architecture (default: yolox-m)",
    )
    parser.add_argument(
        "--classes",
        required=True,
        help="Comma-separated class mappings, e.g. '0:car,1:truck,2:bus'",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Project root to write into (default: current directory)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would be created without writing files")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    classes = parse_classes(args.classes)
    root = Path(args.output_dir).resolve()

    files: dict[Path, str] = {
        root / "configs" / args.usecase / "05_data.yaml": build_05_data_yaml(args.usecase, classes),
        root / "configs" / args.usecase / "06_training.yaml": build_06_training_yaml(args.usecase, args.model, classes),
        root / "experiments" / args.usecase / "train.py": build_train_py(args.usecase),
        root / "experiments" / args.usecase / "evaluate.py": build_evaluate_py(args.usecase),
        root / "experiments" / args.usecase / "export.py": build_export_py(args.usecase),
        root / "experiments" / args.usecase / "inference.py": build_inference_py(args.usecase),
    }

    suffix = " [DRY RUN]" if args.dry_run else ""
    print(f"\nScaffolding '{args.usecase}' ({args.model}, {len(classes)} classes){suffix}")
    created = sum(write_file(path, content, args.dry_run, args.force) for path, content in files.items())
    print(f"\n{'Would create' if args.dry_run else 'Created'} {created}/{len(files)} files.")

    if not args.dry_run and created:
        print(f"\nNext steps:")
        print(f"  1. Add your dataset to dataset_store/{args.usecase}/{{train,val,test}}/{{images,labels}}/")
        print(f"  2. uv run experiments/{args.usecase}/train.py")
        print(f"  3. uv run experiments/{args.usecase}/evaluate.py --split test")


if __name__ == "__main__":
    main()

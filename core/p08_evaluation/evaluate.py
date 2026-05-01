#!/usr/bin/env python3
"""CLI entrypoint for model evaluation.

Loads a trained detection model, evaluates on a dataset split, and produces:
- Printed metrics table (mAP, per-class AP, precision, recall)
- Confusion matrix image
- PR curve images (one per class)
- metrics.json summary

Usage:
    python evaluate.py --model runs/fire_detection/best.pt --config features/safety-fire_detection/configs/05_data.yaml
    python evaluate.py --model runs/fire_detection/best.pt --config features/safety-fire_detection/configs/05_data.yaml --split test --conf 0.5 --iou 0.5 --save-dir outputs/eval
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

# Auto-pick the idle GPU BEFORE torch gets imported so the env var
# reaches torch's first cuda init.
from utils.device import auto_select_gpu  # noqa: E402

auto_select_gpu()

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from core.p06_training.trainer import DetectionTrainer  # noqa: E402
from core.p08_evaluation.error_analysis import ErrorAnalyzer  # noqa: E402
from core.p08_evaluation.evaluator import ModelEvaluator  # noqa: E402
from core.p08_evaluation.visualization import (
    plot_confidence_histogram,
    plot_confusion_matrix,
    plot_error_breakdown,
    plot_pr_curve,
    plot_size_recall,
    plot_threshold_curves,
)
from utils.config import load_config  # noqa: E402
from utils.device import get_device


def load_model(model_path: str, data_config: dict, device: torch.device) -> torch.nn.Module:
    """Load a trained YOLOX model from a checkpoint.

    Supports two checkpoint formats:
    1. Full checkpoint dict with "model" key (from DetectionTrainer)
    2. Raw state dict

    If the checkpoint contains model architecture info, it is used to
    reconstruct the model. Otherwise, a default YOLOX-M is built.

    Args:
        model_path: Path to the .pt checkpoint file.
        data_config: Data config dict (for num_classes, input_size).
        device: Device to load the model onto.

    Returns:
        Model in eval mode on the specified device.

    Raises:
        FileNotFoundError: If model_path does not exist.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)

    # Build model from config in checkpoint
    model = None
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        trainer_config = checkpoint["config"]
    else:
        # Construct minimal config for model building
        trainer_config = {
            "model": {
                "arch": "yolox-m",
                "num_classes": data_config["num_classes"],
                "input_size": data_config["input_size"],
                "depth": 0.67,
                "width": 0.75,
            },
        }
    trainer = DetectionTrainer.__new__(DetectionTrainer)
    trainer.config = trainer_config
    trainer.device = device
    trainer._model_cfg = dict(trainer_config["model"])
    trainer._train_cfg = trainer_config.get("training", {})
    trainer._data_cfg = trainer_config.get("data", {})
    # Skip the pretrained-weight load path — the checkpoint's weights
    # will be loaded below and overwrite anything the trainer loaded.
    # Avoids needing trainer.config_path (not set on this trainer stub).
    trainer._model_cfg.pop("pretrained", None)
    model = trainer._build_model()

    # Load weights
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict",
                     checkpoint.get("model",
                     checkpoint.get("state_dict", checkpoint)))
        if isinstance(state_dict, dict):
            from utils.checkpoint import strip_hf_prefix  # noqa: PLC0415
            state_dict = strip_hf_prefix(state_dict)
            _, unexpected = model.load_state_dict(state_dict, strict=False)
            if unexpected:
                import logging  # noqa: PLC0415
                logging.getLogger(__name__).warning(
                    "load_state_dict: %d unexpected keys (first 3: %s)",
                    len(unexpected), list(unexpected)[:3],
                )
    else:
        if hasattr(checkpoint, "state_dict"):
            model.load_state_dict(checkpoint.state_dict(), strict=False)

    model.to(device)
    model.eval()
    return model


def print_metrics_table(metrics: dict, class_names: dict[int, str]) -> None:
    """Print evaluation metrics as a formatted table.

    Args:
        metrics: Result dict from ModelEvaluator.evaluate().
        class_names: Mapping class_id -> name.
    """
    print()
    print("=" * 65)
    print("  Evaluation Results")
    print("=" * 65)
    print(f"  Images evaluated : {metrics['num_images']}")
    print(f"  Conf threshold   : {metrics['conf_threshold']:.2f}")
    print(f"  IoU threshold    : {metrics['iou_threshold']:.2f}")
    print(f"  mAP@{metrics['iou_threshold']:.1f}          : {metrics['mAP']:.4f}")
    print("-" * 65)
    print(f"  {'Class':<20s} {'AP':>8s} {'Precision':>10s} {'Recall':>8s}")
    print("-" * 65)

    for cls_id in sorted(metrics["per_class_ap"].keys()):
        name = class_names.get(cls_id, f"class_{cls_id}")
        ap = metrics["per_class_ap"][cls_id]
        prec = metrics["precision"][cls_id]
        rec = metrics["recall"][cls_id]
        print(f"  {name:<20s} {ap:>8.4f} {prec:>10.4f} {rec:>8.4f}")

    print("=" * 65)
    print()


def save_metrics_json(metrics: dict, class_names: dict[int, str],
                      save_path: Path) -> None:
    """Save metrics to a JSON file.

    Args:
        metrics: Result dict from ModelEvaluator.evaluate().
        class_names: Mapping class_id -> name.
        save_path: Output JSON file path.
    """
    # Convert numpy types for JSON serialization
    serializable = {
        "mAP": float(metrics["mAP"]),
        "conf_threshold": float(metrics["conf_threshold"]),
        "iou_threshold": float(metrics["iou_threshold"]),
        "num_images": int(metrics["num_images"]),
        "per_class": {},
    }

    for cls_id in sorted(metrics["per_class_ap"].keys()):
        name = class_names.get(cls_id, f"class_{cls_id}")
        serializable["per_class"][name] = {
            "ap": float(metrics["per_class_ap"][cls_id]),
            "precision": float(metrics["precision"][cls_id]),
            "recall": float(metrics["recall"][cls_id]),
        }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Metrics saved to: {save_path}")


def _evaluate_classification(
    model: torch.nn.Module,
    data_config: dict,
    device: torch.device,
    args: argparse.Namespace,
    save_dir: Path,
) -> None:
    """Run classification evaluation: accuracy, top-5 accuracy, per-class metrics."""
    from core.p05_data.classification_dataset import (
        build_classification_dataloader,
    )

    class_names = {int(k): v for k, v in data_config["names"].items()}
    num_classes = data_config["num_classes"]

    # Build dataloader (use a minimal training config for eval)
    eval_train_cfg = {
        "augmentation": {},
        "data": {"batch_size": args.batch_size, "num_workers": 4, "pin_memory": True},
    }
    loader = build_classification_dataloader(
        data_config, split=args.split, training_config=eval_train_cfg,
    )

    model.eval()
    correct = 0
    top5_correct = 0
    total = 0
    per_class_correct: dict[int, int] = {i: 0 for i in range(num_classes)}
    per_class_total: dict[int, int] = {i: 0 for i in range(num_classes)}

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            labels = batch["targets"]
            if isinstance(labels, list):
                labels = torch.stack(labels)
            labels = labels.to(device)

            logits = model(images)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Top-5
            if num_classes >= 5:
                _, top5_pred = logits.topk(5, dim=1)
                top5_correct += sum(
                    labels[i].item() in top5_pred[i].tolist()
                    for i in range(labels.size(0))
                )

            for i in range(labels.size(0)):
                cls = int(labels[i].item())
                per_class_total[cls] += 1
                if predicted[i] == labels[i]:
                    per_class_correct[cls] += 1

    accuracy = correct / total if total > 0 else 0.0
    top5_acc = top5_correct / total if total > 0 and num_classes >= 5 else 0.0

    print()
    print("=" * 65)
    print("  Classification Evaluation Results")
    print("=" * 65)
    print(f"  Images evaluated : {total}")
    print(f"  Accuracy         : {accuracy:.4f}")
    if num_classes >= 5:
        print(f"  Top-5 Accuracy   : {top5_acc:.4f}")
    print("-" * 65)
    print(f"  {'Class':<20s} {'Accuracy':>10s} {'Samples':>8s}")
    print("-" * 65)
    for cls_id in sorted(per_class_total.keys()):
        name = class_names.get(cls_id, f"class_{cls_id}")
        n = per_class_total[cls_id]
        acc = per_class_correct[cls_id] / n if n > 0 else 0.0
        print(f"  {name:<20s} {acc:>10.4f} {n:>8d}")
    print("=" * 65)

    # Save JSON
    metrics_json = {
        "task": "classification",
        "accuracy": accuracy,
        "top5_accuracy": top5_acc,
        "total_samples": total,
        "per_class": {
            class_names.get(i, f"class_{i}"): {
                "accuracy": per_class_correct[i] / max(per_class_total[i], 1),
                "samples": per_class_total[i],
            }
            for i in range(num_classes)
        },
    }
    save_path = save_dir / "metrics.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"  Metrics saved to: {save_path}")


def _evaluate_segmentation(
    model: torch.nn.Module,
    data_config: dict,
    device: torch.device,
    args: argparse.Namespace,
    save_dir: Path,
) -> None:
    """Run segmentation evaluation: mIoU, per-class IoU."""
    from core.p05_data.segmentation_dataset import (
        build_segmentation_dataloader,
    )

    class_names = {int(k): v for k, v in data_config["names"].items()}
    num_classes = data_config["num_classes"]

    eval_train_cfg = {
        "augmentation": {},
        "data": {"batch_size": args.batch_size, "num_workers": 4, "pin_memory": True},
    }
    loader = build_segmentation_dataloader(
        data_config, split=args.split, training_config=eval_train_cfg,
    )

    model.eval()
    intersection = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)
    total_pixels = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            masks = batch["targets"]
            if isinstance(masks, list):
                masks = torch.stack(masks)
            masks = masks.to(device).long()

            logits = model(images)
            # Upsample logits to mask size if needed
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = torch.nn.functional.interpolate(
                    logits, size=masks.shape[-2:], mode="bilinear",
                    align_corners=False,
                )
            preds = logits.argmax(dim=1)
            total_pixels += masks.numel()

            for cls_id in range(num_classes):
                pred_mask = preds == cls_id
                gt_mask = masks == cls_id
                intersection[cls_id] += (pred_mask & gt_mask).sum()
                union[cls_id] += (pred_mask | gt_mask).sum()

    per_class_iou = {}
    for cls_id in range(num_classes):
        u = union[cls_id].item()
        iou = intersection[cls_id].item() / u if u > 0 else 0.0
        per_class_iou[cls_id] = iou

    miou = float(sum(per_class_iou.values()) / max(len(per_class_iou), 1))

    print()
    print("=" * 65)
    print("  Segmentation Evaluation Results")
    print("=" * 65)
    print(f"  Total pixels     : {total_pixels}")
    print(f"  mIoU             : {miou:.4f}")
    print("-" * 65)
    print(f"  {'Class':<20s} {'IoU':>10s}")
    print("-" * 65)
    for cls_id in sorted(per_class_iou.keys()):
        name = class_names.get(cls_id, f"class_{cls_id}")
        print(f"  {name:<20s} {per_class_iou[cls_id]:>10.4f}")
    print("=" * 65)

    metrics_json = {
        "task": "segmentation",
        "mIoU": miou,
        "total_pixels": total_pixels,
        "per_class_iou": {
            class_names.get(i, f"class_{i}"): per_class_iou[i]
            for i in range(num_classes)
        },
    }
    save_path = save_dir / "metrics.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"  Metrics saved to: {save_path}")


def main() -> None:
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained detection model on a dataset split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python evaluate.py --model runs/fire_detection/best.pt --config features/safety-fire_detection/configs/05_data.yaml\n"
            "  python evaluate.py --model runs/fire_detection/best.pt --config features/safety-fire_detection/configs/05_data.yaml "
            "--split test --conf 0.5 --iou 0.5 --save-dir outputs/eval/fire"
        ),
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the model checkpoint (.pt file).")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the data YAML config file.")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Dataset split to evaluate (default: test).")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Confidence threshold (default: 0.5).")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="IoU threshold for NMS and matching (default: 0.5).")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for inference (default: 4 — kept low "
                             "to avoid OOM on DETR-family eval; raise for "
                             "lighter models if VRAM allows).")
    parser.add_argument("--device", type=str, default=None,
                        help="Device string, e.g. 'cuda:0' (default: auto).")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save outputs (plots, JSON). "
                             "Default: outputs/eval/<dataset_name>")
    parser.add_argument("--task", type=str, default="detection",
                        choices=["detection", "classification", "segmentation"],
                        help="Task type for evaluation (default: detection). "
                             "Auto-detected from model when possible.")
    parser.add_argument("--error-analysis", action="store_true",
                        help="Run error analysis: classify errors, compute "
                             "optimal thresholds, generate error plots.")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config).resolve()
    data_config = load_config(config_path)
    data_config["_config_dir"] = config_path.parent

    class_names: dict[int, str] = {
        int(k): v for k, v in data_config["names"].items()
    }
    class_name_list = [class_names[i] for i in range(data_config["num_classes"])]

    # Determine save directory
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        dataset_name = data_config.get("dataset_name", "unknown")
        save_dir = Path("outputs/eval") / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = get_device(args.device)
    print(f"  Device: {device}")

    # Load model
    print(f"  Loading model: {args.model}")
    model = load_model(args.model, data_config, device)

    # Auto-detect task type from model output_format
    task = args.task
    output_format = getattr(model, "output_format", None)
    if output_format == "classification":
        task = "classification"
    elif output_format == "segmentation":
        task = "segmentation"
    elif output_format in ("yolox", "detr"):
        task = "detection"
    print(f"  Task: {task}")

    if task == "classification":
        _evaluate_classification(model, data_config, device, args, save_dir)
        return
    elif task == "segmentation":
        _evaluate_segmentation(model, data_config, device, args, save_dir)
        return

    # --- Detection evaluation (default) ---
    evaluator = ModelEvaluator(
        model=model,
        data_config=data_config,
        device=device,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        batch_size=args.batch_size,
    )

    # Single inference pass — reuse predictions for all downstream analysis
    predictions, ground_truths = evaluator.get_predictions(split=args.split)

    metrics = evaluator._evaluate_detection(predictions, ground_truths)
    metrics["num_images"] = len(predictions)
    metrics["conf_threshold"] = args.conf
    metrics["iou_threshold"] = args.iou

    # Print results
    print_metrics_table(metrics, class_names)

    # Save metrics JSON
    save_metrics_json(metrics, class_names, save_dir / "metrics.json")

    # Generate and save plots
    # Confusion matrix
    cm = metrics["confusion_matrix"]
    fig = plot_confusion_matrix(cm, class_name_list, save_path=str(save_dir / "confusion_matrix.png"))
    plt.close(fig)
    print(f"  Confusion matrix saved to: {save_dir / 'confusion_matrix.png'}")

    # Per-class PR curves
    per_class = evaluator._evaluate_per_class_detection(predictions, ground_truths)
    for cls_name, cls_metrics in per_class.items():
        prec, rec, _ = cls_metrics["pr_curve"]
        if prec.size > 0:
            fig = plot_pr_curve(
                prec, rec, cls_metrics["ap"], cls_name,
                save_path=str(save_dir / f"pr_curve_{cls_name}.png"),
            )
            plt.close(fig)
            print(f"  PR curve saved: {save_dir / f'pr_curve_{cls_name}.png'}")

    # --- Error analysis (optional) ---
    if args.error_analysis:
        print("\n  Running error analysis...")
        analyzer = ErrorAnalyzer(
            class_names=class_names,
            iou_threshold=args.iou,
        )
        report = analyzer.analyze(predictions, ground_truths)

        # Save error analysis JSON
        ea_json = {
            "summary": report.summary,
            "optimal_thresholds": {
                class_names.get(k, str(k)): v
                for k, v in report.optimal_thresholds.items()
            },
            "per_image_error_count": {
                str(k): v for k, v in report.per_image_error_count.items()
            },
        }
        ea_path = save_dir / "error_analysis.json"
        with open(ea_path, "w") as f:
            json.dump(ea_json, f, indent=2)
        print(f"  Error analysis saved to: {ea_path}")

        # Save optimal thresholds separately
        thresh_path = save_dir / "optimal_thresholds.json"
        thresh_json = {}
        for cls_id, result in report.optimal_thresholds.items():
            name = class_names.get(cls_id, f"class_{cls_id}")
            thresh_json[name] = {
                k: v for k, v in result.items() if k != "f1_curve"
            }
        with open(thresh_path, "w") as f:
            json.dump(thresh_json, f, indent=2)
        print(f"  Optimal thresholds saved to: {thresh_path}")

        # Generate error analysis plots
        fig = plot_error_breakdown(report.summary, save_path=str(save_dir / "error_breakdown.png"))
        plt.close(fig)
        print(f"  Error breakdown saved: {save_dir / 'error_breakdown.png'}")

        fig = plot_confidence_histogram(report.errors, save_path=str(save_dir / "confidence_histogram.png"))
        plt.close(fig)
        print(f"  Confidence histogram saved: {save_dir / 'confidence_histogram.png'}")

        fig = plot_size_recall(report.summary, save_path=str(save_dir / "size_recall.png"))
        plt.close(fig)
        print(f"  Size recall saved: {save_dir / 'size_recall.png'}")

        fig = plot_threshold_curves(
            report.optimal_thresholds, class_names,
            save_path=str(save_dir / "threshold_curves.png"),
        )
        plt.close(fig)
        print(f"  Threshold curves saved: {save_dir / 'threshold_curves.png'}")

    print(f"\n  All outputs saved to: {save_dir}")


if __name__ == "__main__":
    main()

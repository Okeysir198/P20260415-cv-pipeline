"""Reusable error analysis CLI for any trained detection model.

Runs inference on a dataset split, classifies every detection into an
error type (background FP / missed / localization / duplicate /
class confusion), and emits a structured breakdown that answers
"why isn't the model learning well?" — not just "what's the mAP?".

Works for any arch the model registry supports (YOLOX, RT-DETRv2,
D-FINE) because it builds the model from a training config rather
than defaulting to yolox-m like `evaluate.py` does for raw HF
`pytorch_model.bin` files.

Usage (from repo root):

    uv run core/p08_evaluation/analyze_errors.py \\
      --training-config features/safety-fire_detection/configs/06_training_rtdetr.yaml \\
      --checkpoint features/safety-fire_detection/runs/<run>/pytorch_model.bin \\
      --split train --subset 0.1 --conf 0.05 \\
      --save-dir features/safety-fire_detection/runs/<run>/error_analysis_train

Outputs (under ``--save-dir``):
- ``error_report.json``: full numeric report + hardest-images list.
- ``error_report.md``: human-readable summary with recommendations.
- ``error_breakdown.png``: TP/FP/FN bar chart per class.
- ``confidence_histogram.png``: FP confidence distribution.
- ``size_recall.png``: FP/FN counts by COCO size tier.
- ``hardest_images.png``: grid of top-8 worst images with error boxes drawn.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Pin GPU before torch import (matches other p0x CLIs).
from utils.device import auto_select_gpu  # noqa: E402

auto_select_gpu()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from core.p05_data.detection_dataset import YOLOXDataset  # noqa: E402
from core.p06_models import build_model  # noqa: E402
from core.p06_training.postprocess import postprocess as _dispatch_postprocess  # noqa: E402
from core.p08_evaluation.error_analysis import ErrorAnalyzer, ErrorCase, ErrorReport  # noqa: E402
from core.p08_evaluation.visualization import (  # noqa: E402
    plot_confidence_histogram,
    plot_error_breakdown,
    plot_hardest_images_grid,
    plot_size_recall,
)
from utils.config import load_config  # noqa: E402
from utils.device import get_device  # noqa: E402
from utils.progress import ProgressBar  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint loading — handles YOLOX .pt and HF pytorch_model.bin formats
# ---------------------------------------------------------------------------


def _load_model_from_training_config(
    training_config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[torch.nn.Module, dict]:
    """Build model from a training config + load any checkpoint format.

    Uses ``build_model(train_cfg)`` so arch is taken from the config —
    avoids ``evaluate.py``'s yolox-m default when the checkpoint dict
    lacks a "config" key (the common case for HF ``pytorch_model.bin``).

    Returns (model, train_cfg).
    """
    train_cfg = load_config(str(training_config_path))

    # Build model with the configured arch. Pretrained-weights path gets
    # popped because the checkpoint we're about to load supersedes it.
    model_cfg = dict(train_cfg.get("model", {}))
    model_cfg.pop("pretrained", None)
    build_cfg = dict(train_cfg)
    build_cfg["model"] = model_cfg
    model = build_model(build_cfg)

    raw = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    # Normalize to a flat state dict — try the formats we emit:
    #   1. DetectionTrainer save: {"model": state_dict, "config": {...}, ...}
    #   2. HF Trainer save:       {hf_model.xxx: ...}  (flat)
    #   3. Legacy raw state dict: {xxx: ...}
    if isinstance(raw, dict):
        if "model" in raw and isinstance(raw["model"], dict):
            state_dict = raw["model"]
        elif "state_dict" in raw and isinstance(raw["state_dict"], dict):
            state_dict = raw["state_dict"]
        elif "model_state_dict" in raw and isinstance(raw["model_state_dict"], dict):
            state_dict = raw["model_state_dict"]
        else:
            state_dict = raw
    elif hasattr(raw, "state_dict"):
        state_dict = raw.state_dict()
    else:
        raise RuntimeError(
            f"Unrecognized checkpoint format at {checkpoint_path}; "
            f"expected dict or nn.Module-like, got {type(raw)}."
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # `strict=False` is OK: HF checkpoints have `hf_model.*` prefix that
    # matches our wrapper; YOLOX checkpoints match directly. But emit a
    # warning if either list is non-trivial — that's usually a misconfig
    # (wrong arch for the checkpoint).
    if missing:
        logger.warning(
            "Missing %d keys when loading %s into %s (first 3: %s). "
            "This usually means the training config's model.arch doesn't "
            "match the checkpoint's architecture.",
            len(missing), checkpoint_path.name,
            model_cfg.get("arch", "unknown"), missing[:3],
        )
    if unexpected:
        logger.warning(
            "%d unexpected keys ignored (first 3: %s).",
            len(unexpected), unexpected[:3],
        )

    model.to(device).eval()
    return model, train_cfg


# ---------------------------------------------------------------------------
# Inference loop — arch-agnostic via model.output_format / model.postprocess
# ---------------------------------------------------------------------------


def _run_inference(
    model: torch.nn.Module,
    dataset: YOLOXDataset,
    indices: List[int],
    device: torch.device,
    input_size: Tuple[int, int],
    batch_size: int,
    conf_threshold: float,
    iou_threshold: float,
) -> Tuple[List[Dict], List[Dict], List[np.ndarray]]:
    """Batched inference on a dataset subset.

    Returns
    -------
    predictions : list of dict  (one per image, keys = boxes/scores/labels, pixel coords on ORIGINAL image)
    ground_truths : list of dict  (same keys, original pixel coords)
    raw_images : list of np.ndarray  (BGR HWC uint8, original res — for hardest-images grid)
    """
    input_h, input_w = input_size
    output_format = getattr(model, "output_format", "yolox")

    predictions: List[Dict] = []
    ground_truths: List[Dict] = []
    raw_images: List[np.ndarray] = []

    def _gt_from_yolo(gt_np: np.ndarray, w: int, h: int) -> Dict:
        if gt_np is None or len(gt_np) == 0:
            return {"boxes": np.zeros((0, 4), dtype=np.float32),
                    "labels": np.zeros(0, dtype=np.int64)}
        cx, cy, bw, bh = gt_np[:, 1], gt_np[:, 2], gt_np[:, 3], gt_np[:, 4]
        boxes = np.stack([
            (cx - bw / 2) * w, (cy - bh / 2) * h,
            (cx + bw / 2) * w, (cy + bh / 2) * h,
        ], axis=1).astype(np.float32)
        return {"boxes": boxes, "labels": gt_np[:, 0].astype(np.int64)}

    with ProgressBar(total=len(indices) // batch_size + 1, desc="Inferring") as pbar:
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]

            # Prepare batch. We match training-time preprocessing: resize +
            # convert to float [0, 1] and feed as (B, 3, H, W). YOLOX Megvii
            # weights expect raw [0, 255], but the pipeline normalises via
            # ToDtype(scale=True) unless explicitly disabled — we keep the
            # same semantics as training here for consistency.
            tensors, orig_dims = [], []
            for idx in batch_idx:
                raw = dataset.get_raw_item(idx)["image"]  # BGR HWC uint8
                raw_images.append(raw)
                resized = cv2.resize(raw, (input_w, input_h))
                t = torch.from_numpy(
                    np.ascontiguousarray(
                        (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)
                    )
                )
                tensors.append(t)
                orig_dims.append(raw.shape[:2])

            batch = torch.stack(tensors).to(device)
            with torch.no_grad():
                outputs = model(pixel_values=batch) if output_format != "yolox" else model(batch)

            # Postprocess: HF models expose ``postprocess`` directly; YOLOX
            # goes through the registry dispatcher.
            if output_format != "yolox" and hasattr(model, "postprocess"):
                target_sizes = torch.tensor(
                    [[input_h, input_w]] * batch.shape[0], device=device,
                )
                batch_preds = model.postprocess(outputs, conf_threshold, target_sizes)
            else:
                batch_preds = _dispatch_postprocess(
                    output_format, model,
                    predictions=outputs,
                    conf_threshold=conf_threshold,
                    nms_threshold=iou_threshold,
                )

            # Rescale predictions from input_size back to each image's
            # original dims so the error breakdown matches human inspection.
            for j, idx in enumerate(batch_idx):
                orig_h, orig_w = orig_dims[j]
                pred = batch_preds[j] if j < len(batch_preds) else {}
                pb = np.asarray(pred.get("boxes", []), dtype=np.float32).reshape(-1, 4)
                pl = np.asarray(pred.get("labels", []), dtype=np.int64).ravel()
                ps = np.asarray(pred.get("scores", []), dtype=np.float32).ravel()
                if pb.shape[0] > 0:
                    pb[:, [0, 2]] *= orig_w / input_w
                    pb[:, [1, 3]] *= orig_h / input_h
                predictions.append({"boxes": pb, "scores": ps, "labels": pl})

                gt_np = dataset._load_label(dataset.img_paths[idx])
                ground_truths.append(_gt_from_yolo(gt_np, orig_w, orig_h))

            pbar.update()

    return predictions, ground_truths, raw_images


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def _error_type_counts(errors: List[ErrorCase]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for e in errors:
        counts[e.error_type] = counts.get(e.error_type, 0) + 1
    return counts


def _size_category_counts(errors: List[ErrorCase]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for e in errors:
        counts[e.size_category] = counts.get(e.size_category, 0) + 1
    return counts


def _format_optimal_thresholds(
    report: ErrorReport, class_names: Dict[int, str]
) -> List[Dict]:
    out = []
    for cid, info in sorted(report.optimal_thresholds.items()):
        if not isinstance(info, dict):
            continue
        out.append({
            "class_id": int(cid),
            "class_name": class_names.get(int(cid), f"class_{cid}"),
            "best_threshold": float(info.get("best_threshold", info.get("threshold", 0.0))),
            "best_f1": float(info.get("best_f1", info.get("f1", 0.0))),
            "precision_at_best": float(info.get("precision", 0.0)),
            "recall_at_best": float(info.get("recall", 0.0)),
        })
    return out


def _format_hardest_images(
    report: ErrorReport,
    dataset: YOLOXDataset,
    indices: List[int],
    predictions: List[Dict],
    ground_truths: List[Dict],
    top_n: int = 10,
) -> List[Dict]:
    ranked = sorted(
        report.per_image_error_count.items(), key=lambda item: -item[1]
    )[:top_n]
    out = []
    for img_idx, err_count in ranked:
        real_idx = indices[img_idx]
        out.append({
            "image_idx": int(img_idx),
            "dataset_idx": int(real_idx),
            "path": str(dataset.img_paths[real_idx]),
            "n_errors": int(err_count),
            "n_gt": int(len(ground_truths[img_idx]["boxes"])),
            "n_pred": int(len(predictions[img_idx]["boxes"])),
        })
    return out


def _render_markdown_report(
    *,
    split: str,
    n_images: int,
    subset_fraction: Optional[float],
    conf_threshold: float,
    iou_threshold: float,
    class_names: Dict[int, str],
    errors: List[ErrorCase],
    type_counts: Dict[str, int],
    size_counts: Dict[str, int],
    thresholds: List[Dict],
    hardest: List[Dict],
    run_context: Dict,
) -> str:
    total_errors = len(errors)
    total_pct = lambda v: f"{100 * v / total_errors:5.1f} %" if total_errors else "  —  "

    lines: List[str] = []
    lines.append(f"# Error Analysis — `{split}` split\n")
    lines.append(f"- Arch: `{run_context.get('arch', '?')}`  |  "
                 f"checkpoint: `{run_context.get('checkpoint', '?')}`")
    lines.append(f"- Images evaluated: **{n_images}**" +
                 (f"  (subset fraction: {subset_fraction:.0%})"
                  if subset_fraction is not None else ""))
    lines.append(f"- Total errors (FP + FN): **{total_errors}**")
    lines.append(f"- Thresholds: conf ≥ {conf_threshold},  matching IoU ≥ {iou_threshold}\n")

    # --- Error type breakdown -----------------------------------------
    lines.append("## Error type breakdown")
    lines.append("| type | count | share |")
    lines.append("|---|---:|---:|")
    for k, v in sorted(type_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v} | {total_pct(v)} |")
    lines.append("")

    # --- Size breakdown -----------------------------------------------
    lines.append("## Size breakdown (within all errors)")
    lines.append("| size | count | share |")
    lines.append("|---|---:|---:|")
    for k in ("small", "medium", "large"):
        v = size_counts.get(k, 0)
        lines.append(f"| {k} | {v} | {total_pct(v)} |")
    lines.append("")

    # --- Per-class optimal thresholds ---------------------------------
    if thresholds:
        lines.append("## Per-class optimal thresholds (max-F1 on this split)")
        lines.append("| class | threshold | F1 | precision | recall |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in thresholds:
            lines.append(
                f"| {row['class_name']} | {row['best_threshold']:.3f} | "
                f"{row['best_f1']:.3f} | {row['precision_at_best']:.3f} | "
                f"{row['recall_at_best']:.3f} |"
            )
        lines.append("")

    # --- Hardest images -----------------------------------------------
    if hardest:
        lines.append(f"## Hardest {len(hardest)} images (by error count)")
        lines.append("| err | gt | pred | path |")
        lines.append("|---:|---:|---:|---|")
        for h in hardest:
            rel = Path(h["path"]).name
            lines.append(f"| {h['n_errors']} | {h['n_gt']} | {h['n_pred']} | "
                         f"{rel} |")
        lines.append("")

    # --- Quick diagnosis ----------------------------------------------
    lines.append("## Diagnosis")
    missed_pct = 100 * type_counts.get("missed", 0) / total_errors if total_errors else 0
    bgfp_pct = 100 * type_counts.get("background_fp", 0) / total_errors if total_errors else 0
    loc_pct = 100 * type_counts.get("localization", 0) / total_errors if total_errors else 0
    cls_pct = 100 * type_counts.get("class_confusion", 0) / total_errors if total_errors else 0

    if missed_pct > 60:
        lines.append(
            f"- **Under-proposal**: {missed_pct:.1f}% of errors are missed detections. "
            "The model isn't outputting enough high-confidence boxes for visible objects. "
            "Classic small-data DETR failure mode — try `matcher_class_cost: 5.0` "
            "(default 2), more `num_denoising`, or more training data."
        )
    if bgfp_pct > 40:
        lines.append(
            f"- **Hallucination**: {bgfp_pct:.1f}% of errors are background false positives. "
            "Lower conf threshold, or tighten class head with stronger WD / longer training."
        )
    if loc_pct > 15:
        lines.append(
            f"- **Poor localization**: {loc_pct:.1f}% of errors are boxes at IoU ∈ [0.3, 0.5). "
            "Box regression is undertrained — more epochs, or bigger `weight_loss_bbox`."
        )
    if cls_pct > 10:
        lines.append(
            f"- **Class confusion**: {cls_pct:.1f}% of errors are wrong-label matches. "
            "Add per-class augmentation, check for label noise, or upweight rare classes."
        )
    if not any([missed_pct > 60, bgfp_pct > 40, loc_pct > 15, cls_pct > 10]):
        lines.append("- No single error type dominates — model is learning broadly; "
                     "gains will likely require more data or longer training.")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--training-config", required=True,
                        help="Path to 06_training.yaml (used for arch + input_size).")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to the checkpoint file (.pt / .pth / pytorch_model.bin).")
    parser.add_argument("--data-config",
                        help="Optional 05_data.yaml override. Default: resolves from the "
                             "training config's `data.dataset_config` key.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--subset", type=float, default=None,
                        help="Subset fraction (0-1) or absolute image count. "
                             "Useful for quick iteration on large splits.")
    parser.add_argument("--conf", type=float, default=0.05,
                        help="Confidence threshold. 0.05 is the DETR-family default "
                             "for torchmetrics MAP; 0.25 matches YOLOX production.")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="IoU threshold for TP/FP matching.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-dir", required=True,
                        help="Output directory for report + plots.")
    parser.add_argument("--top-n-hardest", type=int, default=10,
                        help="Top-N hardest images to list + plot.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    train_cfg_path = Path(args.training_config).resolve()
    ckpt_path = Path(args.checkpoint).resolve()
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    # Resolve data config (allow explicit override, else read from training config).
    train_cfg_peek = load_config(str(train_cfg_path))
    if args.data_config:
        data_cfg_path = Path(args.data_config).resolve()
    else:
        data_ref = train_cfg_peek.get("data", {}).get("dataset_config", "05_data.yaml")
        data_cfg_path = (train_cfg_path.parent / data_ref).resolve()
    if not data_cfg_path.exists():
        raise FileNotFoundError(
            f"Could not find data config at {data_cfg_path}. "
            "Pass --data-config explicitly."
        )

    data_cfg = load_config(str(data_cfg_path))
    class_names = {int(k): str(v) for k, v in data_cfg["names"].items()}
    input_size = tuple(
        train_cfg_peek.get("model", {}).get("input_size")
        or data_cfg.get("input_size", [640, 640])
    )

    device = get_device()
    logger.info("Device: %s", device)
    logger.info("Classes: %s", class_names)
    logger.info("Input size: %s", input_size)

    # Build model + load checkpoint.
    model, _train_cfg = _load_model_from_training_config(train_cfg_path, ckpt_path, device)

    # Load dataset split (no transforms — we handle resize manually).
    dataset = YOLOXDataset(
        data_config=data_cfg, split=args.split, transforms=None,
        base_dir=data_cfg_path.parent,
    )
    n_full = len(dataset)

    # Resolve subset. `None` → full split; float ∈ (0, 1] → fraction;
    # int/float ≥ 1 → absolute image count.
    if args.subset is None:
        indices = list(range(n_full))
        subset_display = None
    elif 0 < args.subset <= 1:
        n_sub = max(1, int(args.subset * n_full))
        indices = list(range(n_sub))
        subset_display = args.subset
    else:
        n_sub = min(n_full, int(args.subset))
        indices = list(range(n_sub))
        subset_display = n_sub / n_full

    logger.info(
        "Split %s: %d / %d images (%s)", args.split, len(indices), n_full,
        f"{subset_display:.0%}" if subset_display is not None else "full",
    )

    # Run inference.
    predictions, ground_truths, raw_images = _run_inference(
        model, dataset, indices, device, input_size,
        args.batch_size, args.conf, args.iou,
    )

    # Classify errors.
    analyzer = ErrorAnalyzer(class_names=class_names, iou_threshold=args.iou)
    report = analyzer.analyze(
        predictions, ground_truths,
        image_paths=[str(dataset.img_paths[i]) for i in indices],
    )

    type_counts = _error_type_counts(report.errors)
    size_counts = _size_category_counts(report.errors)
    thresholds = _format_optimal_thresholds(report, class_names)
    hardest = _format_hardest_images(
        report, dataset, indices, predictions, ground_truths,
        top_n=args.top_n_hardest,
    )

    # --- Persist JSON ---------------------------------------------------
    run_context = {
        "arch": train_cfg_peek.get("model", {}).get("arch", "?"),
        "checkpoint": str(ckpt_path),
        "training_config": str(train_cfg_path),
        "data_config": str(data_cfg_path),
    }
    json_blob = {
        "split": args.split,
        "n_images": len(indices),
        "n_images_full_split": n_full,
        "subset_fraction": subset_display,
        "n_errors": len(report.errors),
        "error_type_counts": type_counts,
        "size_category_counts": size_counts,
        "summary": report.summary,
        "optimal_thresholds": thresholds,
        "hardest_images": hardest,
        "run_context": run_context,
        "conf_threshold": args.conf,
        "iou_threshold": args.iou,
    }
    (save_dir / "error_report.json").write_text(
        json.dumps(json_blob, indent=2, default=str),
    )

    # --- Plots ----------------------------------------------------------
    summary_dict = report.summary
    try:
        plot_error_breakdown(summary_dict, str(save_dir / "error_breakdown.png"))
    except Exception as exc:
        logger.warning("plot_error_breakdown failed: %s", exc)
    try:
        plot_confidence_histogram(report.errors, str(save_dir / "confidence_histogram.png"))
    except Exception as exc:
        logger.warning("plot_confidence_histogram failed: %s", exc)
    try:
        plot_size_recall(summary_dict, str(save_dir / "size_recall.png"))
    except Exception as exc:
        logger.warning("plot_size_recall failed: %s", exc)
    try:
        plot_hardest_images_grid(
            report.errors, raw_images, class_names,
            top_n=min(args.top_n_hardest, 8),
            save_path=str(save_dir / "hardest_images.png"),
        )
    except Exception as exc:
        logger.warning("plot_hardest_images_grid failed: %s", exc)

    # --- Markdown -------------------------------------------------------
    md = _render_markdown_report(
        split=args.split,
        n_images=len(indices),
        subset_fraction=subset_display,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        class_names=class_names,
        errors=report.errors,
        type_counts=type_counts,
        size_counts=size_counts,
        thresholds=thresholds,
        hardest=hardest,
        run_context=run_context,
    )
    (save_dir / "error_report.md").write_text(md)

    # --- Stdout summary -------------------------------------------------
    print()
    print(md)
    print(f"\nSaved: {save_dir}/")
    for name in ("error_report.json", "error_report.md",
                 "error_breakdown.png", "confidence_histogram.png",
                 "size_recall.png", "hardest_images.png"):
        if (save_dir / name).exists():
            print(f"  ✓ {name}")


if __name__ == "__main__":
    main()

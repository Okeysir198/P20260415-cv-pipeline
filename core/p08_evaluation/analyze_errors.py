"""Reusable error analysis CLI for any trained detection model.

Thin wrapper around
:func:`core.p08_evaluation.error_analysis_runner.run_error_analysis` — builds
the model from a training config, loads any supported checkpoint format,
opens the requested split, and delegates the full analysis pass (inference,
error classification, chart + gallery generation, JSON/MD summaries) to the
runner.

Works for any arch the model registry supports (YOLOX, RT-DETRv2, D-FINE).

Usage (from repo root):

    uv run core/p08_evaluation/analyze_errors.py \\
      --training-config features/safety-fire_detection/configs/06_training_rtdetr.yaml \\
      --checkpoint features/safety-fire_detection/runs/<run>/pytorch_model.bin \\
      --split train --subset 0.1 --conf 0.05 \\
      --save-dir features/safety-fire_detection/runs/<run>/error_analysis_train

Outputs (under ``--save-dir`` — numbered-prefix scheme from
``error_analysis_runner.CHART_FILENAMES``):

- ``summary.json`` / ``summary.md``                — full numeric report.
- ``01_overview.png``                              — headline metrics card.
- ``02_data_distribution.png``                     — class balance + size.
- ``03_per_class_performance.png``                 — per-class P/R/F1.
- ``04_confusion_matrix.png`` / ``04_top_confused_pairs.png``
- ``05_confidence_calibration.png``                — correct vs wrong scores.
- ``06_failure_mode_contribution.png``
- ``07_failure_by_attribute.png``                  — detection only.
- ``08_hardest_images.png``                        — top worst-samples grid.
- ``09_failure_mode_examples/``                    — per-mode × per-class
                                                     GT | Pred galleries.
- ``10_recoverable_map_vs_iou.png``
- ``11_confidence_attribution.png``
- ``12_boxes_per_image.png``
- ``13_bbox_aspect_ratio.png``
- ``14_size_recall.png``
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Pin GPU before torch import (matches other p0x CLIs).
from utils.device import auto_select_gpu  # noqa: E402

auto_select_gpu()

import torch  # noqa: E402
from torch.utils.data import Subset  # noqa: E402

from core.p05_data.detection_dataset import YOLOXDataset  # noqa: E402
from core.p06_models import build_model  # noqa: E402
from core.p08_evaluation.error_analysis_runner import (  # noqa: E402
    CHART_FILENAMES,
    run_error_analysis,
)
from utils.config import load_config  # noqa: E402
from utils.device import get_device  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint loading — handles YOLOX .pt and HF pytorch_model.bin formats
# ---------------------------------------------------------------------------


def _load_model_from_training_config(
    training_config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    """Build model from a training config + load any checkpoint format.

    Returns (model, train_cfg).
    """
    train_cfg = load_config(str(training_config_path))

    model_cfg = dict(train_cfg.get("model", {}))
    model_cfg.pop("pretrained", None)
    build_cfg = dict(train_cfg)
    build_cfg["model"] = model_cfg
    model = build_model(build_cfg)

    raw = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

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
    parser.add_argument("--batch-size", type=int, default=16,
                        help="(kept for CLI compatibility; runner batches internally)")
    parser.add_argument("--save-dir", required=True,
                        help="Output directory for report + plots.")
    parser.add_argument("--top-n-hardest", type=int, default=20,
                        help="Per-class cap on hardest-image examples in the "
                             "09_failure_mode_examples/ gallery.")
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

    model, train_cfg = _load_model_from_training_config(train_cfg_path, ckpt_path, device)

    dataset = YOLOXDataset(
        data_config=data_cfg, split=args.split, transforms=None,
        base_dir=data_cfg_path.parent,
    )
    n_full = len(dataset)

    # Resolve subset → a Subset wrapper so the runner sees a dataset-like object.
    if args.subset is None:
        ds = dataset
        n_used = n_full
        subset_display: float | None = None
    elif 0 < args.subset <= 1:
        n_used = max(1, int(args.subset * n_full))
        ds = Subset(dataset, list(range(n_used)))
        subset_display = args.subset
    else:
        n_used = min(n_full, int(args.subset))
        ds = Subset(dataset, list(range(n_used)))
        subset_display = n_used / n_full

    logger.info(
        "Split %s: %d / %d images (%s)", args.split, n_used, n_full,
        f"{subset_display:.0%}" if subset_display is not None else "full",
    )

    # Delegate the full analysis + chart/gallery emission to the shared runner.
    artifacts = run_error_analysis(
        model=model,
        dataset=ds,
        output_dir=save_dir,
        task="detection",
        class_names=class_names,
        input_size=input_size,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        max_samples=n_used,
        hard_images_per_class=args.top_n_hardest,
        training_config=train_cfg,
    )

    # --- Stdout summary -------------------------------------------------
    print(f"\nSaved: {save_dir}/")
    for key, fname in CHART_FILENAMES.items():
        target = save_dir / fname
        if target.exists():
            print(f"  ✓ {fname}")
    for extra in ("summary.json", "summary.md"):
        if (save_dir / extra).exists():
            print(f"  ✓ {extra}")

    # Runner may also expose a summary_md path — echo it if present.
    md_path = artifacts.get("summary_md") if isinstance(artifacts, dict) else None
    if md_path and Path(md_path).exists():
        print("\n--- summary.md ---\n")
        print(Path(md_path).read_text())


if __name__ == "__main__":
    main()

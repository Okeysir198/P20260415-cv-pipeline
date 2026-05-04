#!/usr/bin/env python3
"""Standalone CLI: run error analysis on any trained model + dataset.

Decoupled from the training loop — point it at a checkpoint and a data
config and it produces the full numbered error-analysis artifact tree.

Usage examples::

    # PyTorch / ONNX checkpoint
    uv run core/p08_evaluation/run_error_analysis.py \\
        --model features/safety-fire_detection/runs/20240101_120000/best.pth \\
        --data-config features/safety-fire_detection/configs/05_data.yaml \\
        --split test

    # HuggingFace Hub model
    uv run core/p08_evaluation/run_error_analysis.py \\
        --model PekingU/rtdetr_v2_r18vd \\
        --data-config features/safety-fire_detection/configs/05_data.yaml \\
        --training-config features/safety-fire_detection/configs/06_training_rtdetr.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.device import auto_select_gpu  # noqa: E402

auto_select_gpu()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from utils.config import load_config  # noqa: E402

# ---------------------------------------------------------------------------
# Shim: bridges ModelAdapter → error_analysis_runner's 3-step pipeline
# ---------------------------------------------------------------------------


class _CaptureProcessor:
    """Pretends to be an HF image processor so _preprocess_for_model stores
    the raw image for us instead of applying its own normalization.

    ``_preprocess_for_model`` calls::

        out = processor(images=[resized_rgb], return_tensors="pt", do_resize=False)
        return out["pixel_values"][0]

    We capture ``resized_rgb``, convert back to BGR, append to a queue, and
    return a dummy zero tensor so the downstream stack+forward receives
    something of the right shape. ``_AdapterShim.forward`` flushes the queue
    on each call so micro-batched analyzers (which preprocess N samples,
    stack, single forward) get all N captured images back in order.
    """

    def __init__(self, input_h: int, input_w: int) -> None:
        self._input_h = input_h
        self._input_w = input_w
        self.last_bgr: np.ndarray | None = None
        self._queue: list[np.ndarray] = []

    def __call__(
        self,
        images: list[np.ndarray],
        *,
        return_tensors: str = "pt",
        do_resize: bool = False,
    ) -> dict[str, torch.Tensor]:
        # images[0] is already RGB uint8 HWC (resized by _preprocess_for_model)
        rgb = images[0]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self.last_bgr = bgr  # back-compat for legacy single-image callers
        self._queue.append(bgr)
        dummy = torch.zeros(1, 3, self._input_h, self._input_w)
        return {"pixel_values": dummy}

    def flush(self) -> list[np.ndarray]:
        out, self._queue = self._queue, []
        return out


class _AdapterShim(nn.Module):
    """Thin nn.Module wrapper so run_error_analysis() can call the shim the
    same way it calls a native torch model.

    The shim owns a :class:`_CaptureProcessor` that intercepts
    ``_preprocess_for_model``.  On ``__call__`` the dummy tensor arrives and
    we store the pending prediction.  On ``postprocess()`` we run the adapter
    against the captured BGR image and return the standard dict list.
    """

    def __init__(
        self,
        adapter,
        input_h: int,
        input_w: int,
    ) -> None:
        super().__init__()
        self._adapter = adapter
        self._input_h = input_h
        self._input_w = input_w
        # Setting .processor triggers the HF-processor branch in
        # _preprocess_for_model and bypasses its internal normalization.
        self.processor = _CaptureProcessor(input_h, input_w)
        self._pending: list[dict[str, np.ndarray]] | None = None
        # error_analysis_runner calls next(model.parameters()) to get device.
        self._device_probe = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values is the dummy stacked tensor from _CaptureProcessor.
        # Pull every captured BGR (one per pre-stack preprocess call) and
        # run a single batched inference. ``predict_batch`` returns a list
        # of decoded dicts in the same order as the inputs — that becomes
        # ``_pending``, which ``postprocess()`` returns verbatim.
        bgrs = self.processor.flush()
        if not bgrs:
            # Legacy single-image callers may set last_bgr without queueing.
            if self.processor.last_bgr is None:
                raise RuntimeError("_CaptureProcessor did not capture an image.")
            bgrs = [self.processor.last_bgr]
        # Sanity: stacked dummy batch size should match captured count.
        n = int(pixel_values.shape[0]) if pixel_values.ndim >= 1 else len(bgrs)
        if n != len(bgrs):
            # Trust the captured queue — the dummy tensor's batch dim is
            # only there to keep _dispatch_forward happy.
            pass
        self._pending = self._adapter.predict_batch(bgrs)
        return torch.zeros(len(bgrs))

    def postprocess(
        self,
        preds_raw,  # ignored — we use self._pending
        conf_threshold: float,
        target_sizes: torch.Tensor,
    ) -> list[dict[str, np.ndarray]]:
        if self._pending is None:
            raise RuntimeError("postprocess() called before forward().")
        result = self._pending
        self._pending = None
        return result


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def _build_dataset(task: str, data_config: dict, split: str, project_root: Path):
    """Return the appropriate Dataset object for *task*."""
    base_dir = project_root

    if task == "detection":
        from core.p05_data.detection_dataset import YOLOXDataset  # noqa: PLC0415

        return YOLOXDataset(data_config, split=split, base_dir=base_dir)

    if task == "classification":
        from core.p05_data.classification_dataset import ClassificationDataset  # noqa: PLC0415

        return ClassificationDataset(data_config, split=split, base_dir=base_dir)

    if task == "segmentation":
        from core.p05_data.segmentation_dataset import SegmentationDataset  # noqa: PLC0415

        return SegmentationDataset(data_config, split=split, base_dir=base_dir)

    if task == "keypoint":
        from core.p05_data.keypoint_dataset import KeypointDataset  # noqa: PLC0415

        return KeypointDataset(data_config, split=split, base_dir=base_dir)

    raise ValueError(f"Unsupported task: {task!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run error analysis on any trained model + dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", required=True,
                        help="Path to .pth/.pt/.onnx checkpoint, HF Hub repo id, "
                             "or local HF model directory.")
    parser.add_argument("--data-config", required=True,
                        help="Path to 05_data.yaml (task, class names, dataset paths).")
    parser.add_argument("--training-config", default=None,
                        help="Path to 06_training.yaml (optional; provides input_size, "
                             "arch hints).")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"],
                        help="Dataset split to evaluate (default: val).")
    parser.add_argument("--out", default=None,
                        help="Output directory for error-analysis artifacts. "
                             "Defaults to <model_dir>/error_analysis/ or "
                             "./error_analysis/ for Hub models.")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="Confidence threshold. With "
                             "--threshold-policy=fixed this is the deploy "
                             "cutoff; with the f1_optimal[_per_class] policies "
                             "it is the eps floor for raw prediction collection "
                             "(default: 0.3).")
    parser.add_argument("--threshold-policy",
                        choices=["fixed", "f1_optimal", "f1_optimal_per_class"],
                        default="f1_optimal_per_class",
                        help="How to pick the operating point for the "
                             "threshold-sensitive charts (08/10/11/12/13). "
                             "'fixed' uses --conf for all classes; "
                             "'f1_optimal' uses a single global F1-max "
                             "threshold; 'f1_optimal_per_class' (default) "
                             "uses one threshold per class.")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="IoU threshold for matching (default: 0.5).")
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Cap dataset samples (default: 500). Use 0 for all.")
    parser.add_argument("--forward-batch-size", type=int, default=8,
                        help="Micro-batch size for the model forward inside "
                             "the analyzer (default: 8). The env var "
                             "EA_FORWARD_BATCH_SIZE overrides this when set.")
    parser.add_argument("--task", default=None,
                        help="Override task type (detection|classification|"
                             "segmentation|keypoint). Default: read from data config.")
    return parser.parse_args()


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    args = _parse_args()

    data_config: dict = load_config(args.data_config)
    training_config: dict | None = load_config(args.training_config) if args.training_config else None

    # Resolve task
    task: str = (
        args.task
        or data_config.get("task")
        or "detection"
    ).lower()

    # Resolve input_size
    input_size: list[int] = (
        (training_config or {}).get("model", {}).get("input_size")
        or data_config.get("input_size")
        or [640, 640]
    )
    input_h, input_w = int(input_size[0]), int(input_size[1])

    # Resolve class names
    raw_names = data_config.get("names", {})
    class_names: dict[int, str] = {int(k): str(v) for k, v in raw_names.items()}

    # Resolve output directory
    if args.out:
        out_dir = Path(args.out)
    else:
        model_p = Path(args.model)
        if model_p.suffix:
            out_dir = model_p.parent / "error_analysis"
        else:
            out_dir = Path.cwd() / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    max_samples: int | None = args.max_samples if args.max_samples > 0 else None

    logger.info("Task: {} | split: {} | input: {}x{}", task, args.split, input_h, input_w)
    logger.info("Output: {}", out_dir)

    # Build dataset — paths in 05_data.yaml are relative to the config file's directory
    data_config_dir = Path(args.data_config).resolve().parent
    dataset = _build_dataset(task, data_config, args.split, data_config_dir)
    logger.info("Dataset: {} samples", len(dataset))

    # Build adapter + shim. In policy modes, the adapter must surface
    # raw predictions down to the eps floor (run_error_analysis re-filters
    # per class). With `fixed` policy, the adapter applies the user cutoff.
    from core.p08_evaluation.threshold_policy import EPS_FLOOR  # noqa: PLC0415
    from core.p10_inference.model_adapter import resolve_adapter  # noqa: PLC0415

    adapter_conf = (
        args.conf if args.threshold_policy == "fixed" else EPS_FLOOR
    )
    adapter = resolve_adapter(
        model_path=args.model,
        data_config=data_config,
        training_config=training_config,
        conf_threshold=adapter_conf,
        iou_threshold=args.iou,
    )
    model = _AdapterShim(adapter, input_h=input_h, input_w=input_w)
    model.eval()

    # Run error analysis
    from core.p08_evaluation.error_analysis_runner import run_error_analysis  # noqa: PLC0415

    artifacts = run_error_analysis(
        model=model,
        dataset=dataset,
        output_dir=out_dir,
        task=task,
        class_names=class_names,
        input_size=(input_h, input_w),
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        max_samples=max_samples,
        training_config=training_config,
        threshold_policy=args.threshold_policy,
        split=args.split,
        forward_batch_size=args.forward_batch_size,
    )

    logger.info("Error analysis complete. Artifacts:")
    for key, path in artifacts.items():
        if path and Path(str(path)).exists():
            logger.info("  {:<40} {}", key, path)


if __name__ == "__main__":
    main()

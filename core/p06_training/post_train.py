"""Backend-agnostic post-training artifact runner.

Both the pytorch `DetectionTrainer` and the HF `train_with_hf` entry point
funnel through :func:`run_post_train_artifacts` once training finishes. The
runner owns:

1. **`val_predictions/best.png` + `test_predictions/best.png`** — a prediction
   grid rendered from the best-checkpoint weights on a fixed sample of each
   split (the same images as per-epoch grids for val, a fresh pool for test).

2. **`val_predictions/error_analysis/` + `test_predictions/error_analysis/`** —
   task-dispatched summary + charts + per-error-type × per-class hard-image
   galleries produced by `core/p08_evaluation/error_analysis_runner.run_error_analysis`.

Every GT-vs-Pred drawing call threads through :func:`render_prediction_grid`
here, which delegates to :func:`core.p10_inference.supervision_bridge.annotate_gt_pred`
with a shared :class:`VizStyle`. No call site in this file or in the
error-analysis runner draws boxes directly — the acceptance contract from the
plan is that ``cv2.rectangle`` / ``cv2.putText`` do not appear in
``core/p06_training/`` or ``core/p08_evaluation/``.
"""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from core.p06_training._common import (
    task_from_output_format as _task_from_output_format,
    unwrap_subset as _unwrap_subset,
    yolo_targets_to_xyxy as _gt_xyxy_from_yolo,
)
from core.p10_inference.supervision_bridge import (
    VizStyle,
    annotate_gt_pred,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared forward + GT/Pred extraction per task.
# ---------------------------------------------------------------------------


def _forward_batch_detection(model, tensors, target_sizes, conf_threshold):
    """Run a detection model forward and decode to xyxy-pixel preds.

    Returns a list of dicts ``{boxes, scores, labels}`` aligned to ``tensors``.

    Two dispatch paths:
    * Model has ``.postprocess(predictions, conf_threshold, target_sizes)`` —
      HF detection wrappers (`HFDetectionModel`). Call directly.
    * Model has no ``.postprocess`` — YOLOX / custom. Use the shared
      :func:`core.p06_training.postprocess.postprocess` dispatcher which knows
      each arch's nms_threshold vs target_sizes parameter order via the
      :data:`POSTPROCESSOR_REGISTRY`.
    """
    device = next(model.parameters()).device
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        preds_raw = model(pixel_values=batch) if hasattr(model, "hf_model") else model(batch)
    if hasattr(model, "postprocess"):
        return model.postprocess(preds_raw, conf_threshold, target_sizes.to(device))
    from core.p06_training.postprocess import postprocess as _registry_postprocess
    output_format = getattr(model, "output_format", "yolox")
    try:
        return _registry_postprocess(
            output_format=output_format,
            model=model,
            predictions=preds_raw,
            conf_threshold=conf_threshold,
            target_sizes=target_sizes.to(device) if hasattr(target_sizes, "to") else target_sizes,
        )
    except Exception as e:
        logger.warning("post-train decode failed for output_format=%s: %s",
                       output_format, e)
        return [{"boxes": np.zeros((0, 4)), "scores": np.zeros(0),
                 "labels": np.zeros(0, dtype=np.int64)}] * len(tensors)


# ---------------------------------------------------------------------------
# Public: render a prediction grid for ANY task.
# ---------------------------------------------------------------------------


def render_prediction_grid(
    model,
    dataset,
    indices: list[int],
    out_path: Path,
    *,
    title: str,
    class_names: dict[int, str],
    input_size: tuple[int, int],
    style: VizStyle,
    task: str = "detection",
    conf_threshold: float = 0.3,
    grid_cols: int = 4,
    dpi: int = 150,
) -> Path | None:
    """Render a task-aware GT-vs-Pred grid and save as PNG.

    Supports detection (boxes), classification (GT/Pred class names in title),
    segmentation (mask overlays via alpha-blend), and keypoint (skeleton).
    All GT-vs-Pred drawing delegates to :func:`annotate_gt_pred` so every
    grid in the repo — per-epoch val, best.png, hardest gallery, hard-image
    per-class galleries — is byte-consistent in color, thickness, and layout.
    """
    if not indices:
        logger.warning("render_prediction_grid(%s): no indices", out_path.name)
        return None

    raw_dataset, idx_map = _unwrap_subset(dataset)
    input_h, input_w = int(input_size[0]), int(input_size[1])

    samples = []
    skipped = 0
    for idx in indices:
        real_idx = idx_map(idx)
        try:
            raw = raw_dataset.get_raw_item(real_idx)
        except Exception as e:
            skipped += 1
            logger.debug("get_raw_item(%s) failed: %s", real_idx, e)
            continue
        image = raw["image"]
        if image is None:
            skipped += 1
            continue
        resized = cv2.resize(image, (input_w, input_h))
        tensor = torch.from_numpy(
            np.ascontiguousarray(
                (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)
            )
        )
        samples.append((real_idx, image, resized, tensor, raw.get("targets")))

    if not samples:
        logger.warning(
            "render_prediction_grid(%s): 0 samples from %d indices (%d skipped) — grid not saved",
            out_path.name, len(indices), skipped,
        )
        return None

    rows: list[np.ndarray] = []

    if task == "detection":
        target_sizes = torch.tensor(
            [[input_h, input_w]] * len(samples), dtype=torch.int64
        )
        decoded = _forward_batch_detection(
            model, [s[3] for s in samples], target_sizes, conf_threshold,
        )
        for i, (real_idx, orig_image, _resized, _tensor, gt_targets) in enumerate(samples):
            pred = decoded[i] if i < len(decoded) else {}
            pred_boxes = np.asarray(pred.get("boxes", []), dtype=np.float64).reshape(-1, 4)
            pred_labels = np.asarray(pred.get("labels", []), dtype=np.int64).ravel()
            pred_scores = np.asarray(pred.get("scores", []), dtype=np.float64).ravel()

            orig_h, orig_w = orig_image.shape[:2]
            if len(pred_boxes) > 0:
                pred_boxes[:, [0, 2]] *= orig_w / input_w
                pred_boxes[:, [1, 3]] *= orig_h / input_h

            import supervision as sv
            pred_dets = sv.Detections(
                xyxy=pred_boxes,
                class_id=pred_labels,
                confidence=pred_scores,
            )

            gt_xyxy, gt_cls = (None, None)
            if isinstance(gt_targets, np.ndarray) and gt_targets.size > 0:
                gt_xyxy, gt_cls = _gt_xyxy_from_yolo(gt_targets, orig_w, orig_h)

            rows.append(annotate_gt_pred(
                orig_image, gt_xyxy, gt_cls, pred_dets, class_names,
                style=style,
            ))

    elif task == "classification":
        device = next(model.parameters()).device
        batch = torch.stack([s[3] for s in samples]).to(device)
        with torch.no_grad():
            logits = model(pixel_values=batch) if hasattr(model, "hf_model") else model(batch)
        if hasattr(logits, "logits"):
            logits = logits.logits
        preds = logits.argmax(dim=-1).cpu().numpy()
        scores = torch.softmax(logits, dim=-1).max(dim=-1).values.cpu().numpy()
        for i, (real_idx, orig_image, _resized, _tensor, gt_cls) in enumerate(samples):
            gt_name = class_names.get(int(gt_cls) if gt_cls is not None else -1, "-")
            pred_name = class_names.get(int(preds[i]), str(int(preds[i])))
            ok = (gt_cls is not None and int(gt_cls) == int(preds[i]))
            # Draw a title bar over a copy of the image: GT | Pred (score)
            annotated = orig_image.copy()
            bar = np.full((28, annotated.shape[1], 3), 30, dtype=np.uint8)
            text_color = style.pred_color_rgb if ok else style.gt_color_rgb
            cv2.putText(
                bar,
                f"GT: {gt_name}    Pred: {pred_name} ({scores[i]:.2f})",
                (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (int(text_color[2]), int(text_color[1]), int(text_color[0])), 1, cv2.LINE_AA,
            )
            rows.append(np.vstack([bar, annotated]))

    elif task == "segmentation":
        device = next(model.parameters()).device
        batch = torch.stack([s[3] for s in samples]).to(device)
        with torch.no_grad():
            out = model(pixel_values=batch) if hasattr(model, "hf_model") else model(batch)
        seg_logits = out.logits if hasattr(out, "logits") else out
        pred_masks = seg_logits.argmax(dim=1).cpu().numpy()  # (B, H', W')
        for i, (real_idx, orig_image, _resized, _tensor, gt_mask) in enumerate(samples):
            orig_h, orig_w = orig_image.shape[:2]
            pm = pred_masks[i]
            if pm.shape != (orig_h, orig_w):
                pm = cv2.resize(pm.astype(np.int32), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            gm = gt_mask if isinstance(gt_mask, np.ndarray) else None
            if gm is not None and gm.shape != (orig_h, orig_w):
                gm = cv2.resize(gm.astype(np.int32), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            overlay = _blend_seg_masks(orig_image, gm, pm, class_names, style)
            rows.append(overlay)

    elif task == "keypoint":
        # Predict + draw keypoints using the detection adapter's postprocess
        # shape. Keypoint heads emit (N, K, 3) — skeleton drawn in style.
        target_sizes = torch.tensor(
            [[input_h, input_w]] * len(samples), dtype=torch.int64
        )
        decoded = _forward_batch_detection(
            model, [s[3] for s in samples], target_sizes, conf_threshold,
        )
        for i, (real_idx, orig_image, _resized, _tensor, gt_raw) in enumerate(samples):
            annotated = orig_image.copy()
            pred = decoded[i] if i < len(decoded) else {}
            pred_kp = np.asarray(pred.get("keypoints", []), dtype=np.float32).reshape(-1, 3)
            gt_kp = None
            if isinstance(gt_raw, dict) and "keypoints" in gt_raw:
                gt_kp = np.asarray(gt_raw["keypoints"], dtype=np.float32).reshape(-1, 3)
            annotated = _draw_keypoints(annotated, gt_kp, style.gt_color_rgb, style.gt_thickness)
            annotated = _draw_keypoints(annotated, pred_kp, style.pred_color_rgb, style.pred_thickness)
            rows.append(annotated)

    if not rows:
        return None

    _save_grid(rows, out_path, title, grid_cols, dpi)
    return out_path


def _blend_seg_masks(image, gt_mask, pred_mask, class_names, style: VizStyle) -> np.ndarray:
    """Overlay GT (purple-tinted) + pred (green-tinted) segmentation masks."""
    overlay = image.copy()
    if gt_mask is not None and gt_mask.any():
        mask3 = (gt_mask > 0).astype(np.uint8)[..., None]
        tint = np.array(style.gt_color_rgb[::-1], dtype=np.uint8)  # to BGR
        overlay = np.where(
            mask3,
            (overlay * (1 - style.mask_alpha) + tint * style.mask_alpha).astype(np.uint8),
            overlay,
        )
    if pred_mask is not None and pred_mask.any():
        mask3 = (pred_mask > 0).astype(np.uint8)[..., None]
        tint = np.array(style.pred_color_rgb[::-1], dtype=np.uint8)
        overlay = np.where(
            mask3,
            (overlay * (1 - style.mask_alpha) + tint * style.mask_alpha).astype(np.uint8),
            overlay,
        )
    return overlay


def _draw_keypoints(image, kp, color_rgb, thickness) -> np.ndarray:
    """Dot-draw keypoints; visibility gate at kp[:, 2] > 0."""
    if kp is None or kp.size == 0:
        return image
    img = image.copy()
    color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
    for x, y, v in kp:
        if v <= 0:
            continue
        cv2.circle(img, (int(x), int(y)), max(2, thickness + 1), color_bgr, -1, cv2.LINE_AA)
    return img


def _save_grid(rows: list[np.ndarray], out_path: Path, title: str, ncols: int, dpi: int):
    """Save a list of BGR images as a matplotlib grid under out_path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrows = max(1, math.ceil(len(rows) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5))
    axes = np.asarray(axes).ravel()
    for i in range(nrows * ncols):
        axes[i].axis("off")
        if i < len(rows):
            axes[i].imshow(cv2.cvtColor(rows[i], cv2.COLOR_BGR2RGB))
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public: the single post-train artifact runner.
# ---------------------------------------------------------------------------


def run_post_train_artifacts(
    model,
    save_dir: str | Path,
    *,
    val_dataset,
    test_dataset,
    task: str | None = None,
    class_names: dict[int, str],
    input_size: tuple[int, int],
    style: VizStyle | None = None,
    best_num_samples: int = 16,
    best_conf_threshold: float = 0.3,
    error_analysis_conf_threshold: float = 0.3,
    error_analysis_iou_threshold: float = 0.5,
    error_analysis_max_samples: int | None = 500,
    error_analysis_hard_images_per_class: int = 20,
    log_history_best_map: float | None = None,
    log_history_test_map: float | None = None,
    training_config: dict | None = None,
) -> dict[str, Any]:
    """Render best-checkpoint val/test grids + full error analysis.

    Called from HF `HFValPredictionCallback.on_train_end` and pytorch
    `DetectionTrainer._finalize_training` after best-checkpoint reload.

    Args:
        model: best-checkpoint model in eval mode.
        save_dir: root run directory (e.g. ``runs/<ts>/``).
        val_dataset, test_dataset: torch datasets (or None).
        task: one of ``detection / classification / segmentation / keypoint``;
            auto-detected from ``model.output_format`` if None.
        class_names: id → display name.
        input_size: model input HxW.
        style: VizStyle; defaults are loaded if None.
        best_num_samples: how many images per split to render in best.png.
        best_conf_threshold: confidence for detection best.png display.
        error_analysis_*: analyzer knobs; see error_analysis_runner.
        log_history_best_map / log_history_test_map: optional cached numbers
            for titling best.png without re-running eval.

    Returns:
        Dict of artifact name → path for every file produced.
    """
    save_dir = Path(save_dir)
    style = style or VizStyle()
    if task is None:
        task = _task_from_output_format(getattr(model, "output_format", None))

    was_training = bool(getattr(model, "training", False))
    model.eval()
    artifacts: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Import the error analyzer lazily — avoids a hard import dependency #
    # in callbacks that only need render_prediction_grid.                #
    # ------------------------------------------------------------------ #
    try:
        from core.p08_evaluation.error_analysis_runner import run_error_analysis
    except Exception as e:  # pragma: no cover
        logger.warning("error_analysis_runner unavailable: %s", e)
        run_error_analysis = None  # type: ignore[assignment]

    # -------- val best grid + error analysis --------
    if val_dataset is not None and len(val_dataset) > 0:
        n = len(val_dataset)
        k = min(best_num_samples, n)
        val_indices = sorted(random.sample(range(n), k)) if k > 0 else []
        val_title = (
            f"Best checkpoint (val) — mAP50: {log_history_best_map:.4f}"
            if log_history_best_map is not None
            else "Best checkpoint (val)"
        )
        p = render_prediction_grid(
            model, val_dataset, val_indices,
            save_dir / "val_predictions" / "best.png",
            title=val_title, class_names=class_names,
            input_size=input_size, style=style, task=task,
            conf_threshold=best_conf_threshold,
        )
        if p is not None:
            artifacts["val_best_png"] = p

        if run_error_analysis is not None:
            try:
                val_report = run_error_analysis(
                    model=model,
                    dataset=val_dataset,
                    output_dir=save_dir / "val_predictions" / "error_analysis",
                    task=task,
                    class_names=class_names,
                    input_size=input_size,
                    style=style,
                    conf_threshold=error_analysis_conf_threshold,
                    iou_threshold=error_analysis_iou_threshold,
                    max_samples=error_analysis_max_samples,
                    hard_images_per_class=error_analysis_hard_images_per_class,
                    training_config=training_config,
                )
                artifacts["val_error_analysis"] = val_report
            except Exception as e:
                logger.warning("val error analysis skipped: %s", e, exc_info=True)

    # -------- test best grid + error analysis --------
    if test_dataset is not None and len(test_dataset) > 0:
        n = len(test_dataset)
        k = min(best_num_samples, n)
        test_indices = sorted(random.sample(range(n), k)) if k > 0 else []
        test_title = (
            f"Best checkpoint (test) — mAP50: {log_history_test_map:.4f}"
            if log_history_test_map is not None
            else "Best checkpoint (test)"
        )
        p = render_prediction_grid(
            model, test_dataset, test_indices,
            save_dir / "test_predictions" / "best.png",
            title=test_title, class_names=class_names,
            input_size=input_size, style=style, task=task,
            conf_threshold=best_conf_threshold,
        )
        if p is not None:
            artifacts["test_best_png"] = p

        if run_error_analysis is not None:
            try:
                test_report = run_error_analysis(
                    model=model,
                    dataset=test_dataset,
                    output_dir=save_dir / "test_predictions" / "error_analysis",
                    task=task,
                    class_names=class_names,
                    input_size=input_size,
                    style=style,
                    conf_threshold=error_analysis_conf_threshold,
                    iou_threshold=error_analysis_iou_threshold,
                    max_samples=error_analysis_max_samples,
                    hard_images_per_class=error_analysis_hard_images_per_class,
                    training_config=training_config,
                )
                artifacts["test_error_analysis"] = test_report
            except Exception as e:
                logger.warning("test error analysis skipped: %s", e, exc_info=True)

    if was_training:
        model.train()
    return artifacts

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
import supervision as sv
import torch

from core.p06_training._common import (
    task_from_output_format as _task_from_output_format,
)
from core.p06_training._common import (
    unwrap_subset as _unwrap_subset,
)
from core.p06_training._common import (
    yolo_targets_to_xyxy as _gt_xyxy_from_yolo,
)
from core.p10_inference.supervision_bridge import (
    VizStyle,
    annotate_gt_pred,
)
from utils.viz import annotate_keypoints, classification_banner

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
        # Use the same preprocessing path as the analyzer so predictions
        # here match the error-analysis numbers. Critical for HF wrappers:
        # without processor-driven ImageNet normalization the DETR decoder
        # produces zero usable predictions.
        from core.p08_evaluation.error_analysis_runner import _preprocess_for_model
        tensor = _preprocess_for_model(image, (input_h, input_w), model=model)
        resized = cv2.resize(image, (input_w, input_h))
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
            text_color = style.pred_color_rgb if ok else style.gt_color_rgb
            # Preserve previous look: 28-px bar, RGB(30,30,30) bg, text at 0.5 scale.
            # Image is BGR here; classification_banner is channel-opaque, so
            # convert to RGB at boundary to keep the color semantics consistent
            # with the RGB-defined text_color, then convert the stacked result back.
            rgb_img = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            banner_style = VizStyle(
                banner_height=28,
                banner_bg_rgb=(30, 30, 30),
                banner_text_rgb=tuple(int(c) for c in text_color),
                banner_text_scale=0.5,
            )
            stacked_rgb = classification_banner(
                rgb_img,
                f"GT: {gt_name}    Pred: {pred_name} ({scores[i]:.2f})",
                style=banner_style,
                position="top",
            )
            rows.append(cv2.cvtColor(stacked_rgb, cv2.COLOR_RGB2BGR))

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
    """Dot-draw keypoints (no skeleton edges); visibility gate at kp[:, 2] > 0.

    Thin adapter over :func:`utils.viz.annotate_keypoints`. Image is BGR here
    (from the dataset), so we convert to RGB at the boundary — the helper
    works in RGB because ``sv.Color`` values from VizStyle/arg are RGB.
    """
    if kp is None or kp.size == 0:
        return image
    kp_arr = np.asarray(kp, dtype=np.float32)
    xy = kp_arr[:, :2]
    vis = kp_arr[:, 2]
    # Hide invisible keypoints by zeroing their coords (same effect as the
    # v<=0 continue in the previous impl).
    xy = xy.copy()
    xy[vis <= 0] = 0.0
    style = VizStyle(keypoint_radius=max(2, thickness + 1))
    color = sv.Color(r=int(color_rgb[0]), g=int(color_rgb[1]), b=int(color_rgb[2]))
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    annotated_rgb = annotate_keypoints(img_rgb, xy, skeleton_edges=None, style=style, color=color)
    return cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)


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
    best_conf_threshold: float = 0.1,
    error_analysis_conf_threshold: float = 0.05,
    error_analysis_iou_threshold: float = 0.5,
    error_analysis_max_samples: int | None = 500,
    error_analysis_hard_images_per_class: int = 20,
    log_history_best_map: float | None = None,
    log_history_test_map: float | None = None,
    training_config: dict | None = None,
    train_dataset=None,
    val_loader=None,
    learning_ability_subset_max: int = 500,
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

    val_ea_dir = save_dir / "val_predictions" / "error_analysis"
    test_ea_dir = save_dir / "test_predictions" / "error_analysis"

    # -------- Distribution Mismatch (train vs val drift) ----
    if train_dataset is not None and val_dataset is not None:
        try:
            from core.p08_evaluation.distribution_mismatch import (
                analyze_distribution_mismatch,
            )
            dm_result = analyze_distribution_mismatch(
                train_dataset=train_dataset, val_dataset=val_dataset,
                test_dataset=test_dataset, output_dir=val_ea_dir,
                task=task, class_names=class_names,
            )
            if dm_result:
                artifacts["distribution_mismatch"] = dm_result.get("artifacts", {})
                _append_section(
                    summary_md=val_ea_dir / "summary.md",
                    heading="## Distribution Mismatch — Train vs Val",
                    stems=("03_distribution_mismatch",),
                    chart_dir=val_ea_dir,
                    chart_metrics=dm_result.get("chart_metrics", {}),
                )
        except Exception as e:  # pragma: no cover
            logger.warning("distribution mismatch analyzer skipped: %s", e, exc_info=True)

    # -------- Learning Ability (bias / variance) -------- #
    if train_dataset is not None and len(train_dataset) > 0:
        try:
            from core.p08_evaluation.learning_ability import analyze_learning_ability
            primary_name, val_metric = _resolve_val_metric_for_la(
                task, log_history_best_map, val_dataset, model, input_size,
            )
            la_result = analyze_learning_ability(
                model=model, train_dataset=train_dataset,
                val_metric=val_metric, primary_metric_name=primary_name,
                task=task, save_dir=val_ea_dir, input_size=input_size,
                train_subset_max=learning_ability_subset_max,
                trainer_state_dir=None,
            )
            if la_result:
                artifacts["learning_ability"] = la_result.get("artifacts", {})
                _append_section(
                    summary_md=val_ea_dir / "summary.md",
                    heading="## Learning Ability — Bias vs Variance",
                    stems=("06_learning_ability",),
                    chart_dir=val_ea_dir,
                    chart_metrics=la_result.get("chart_metrics", {}),
                )
        except Exception as e:  # pragma: no cover
            logger.warning("learning ability analyzer skipped: %s", e, exc_info=True)

    # -------- Robustness sweep (blur/jpeg/brightness/rotation) ----
    rb_cfg = {}
    if isinstance(training_config, dict):
        rb_cfg = ((training_config.get("evaluation") or {}).get("robustness_sweep") or {})
    if rb_cfg.get("enabled", True) and val_dataset is not None and len(val_dataset) > 0:
        try:
            rb_result = _robustness_dispatch(
                model=model, val_dataset=val_dataset, val_loader=val_loader,
                task=task, save_dir=val_ea_dir,
                cached_val_metric=log_history_best_map,
            )
            if rb_result:
                artifacts["robustness_sweep"] = rb_result.get("artifacts", {})
                _append_section(
                    summary_md=val_ea_dir / "summary.md",
                    heading="## Robustness Sweep — Corruption Severity",
                    stems=("14_robustness_sweep",),
                    chart_dir=val_ea_dir,
                    chart_metrics=rb_result.get("chart_metrics", {}),
                )
        except Exception as e:  # pragma: no cover
            logger.warning("robustness sweep skipped: %s", e, exc_info=True)

    # -------- Duplicates / Leakage (pHash cross-split) ----
    # Prefer the resolved 05_data.yaml (`_loaded_data_cfg`) which carries
    # `path`/`train`/`val`/`test` keys; the raw `data:` block under
    # `06_training.yaml` only has `dataset_config`/`batch_size` and would
    # produce empty enumeration.
    data_cfg = None
    base_dir_for_dl: str | None = None
    if isinstance(training_config, dict):
        data_cfg = (
            training_config.get("_loaded_data_cfg")
            or training_config.get("data")
        )
        base_dir_for_dl = training_config.get("_config_dir") or None
    if data_cfg and val_ea_dir.exists():
        try:
            from core.p08_evaluation import duplicates_leakage
            dl_result = duplicates_leakage.run(
                data_cfg, output_dir=val_ea_dir,
                task=task, base_dir=base_dir_for_dl,
            )
            if dl_result:
                artifacts["duplicates_leakage"] = dl_result.get("artifacts", {})
                _append_section(
                    summary_md=val_ea_dir / "summary.md",
                    heading="## Near-duplicates & Cross-split Leakage",
                    stems=("05_duplicates_leakage",),
                    chart_dir=val_ea_dir,
                    chart_metrics=dl_result.get("chart_metrics", {}),
                )
        except Exception as e:  # pragma: no cover
            logger.warning("duplicates_leakage analyzer skipped: %s", e, exc_info=True)

    if was_training:
        model.train()
    return artifacts


def _robustness_dispatch(
    *, model, val_dataset, val_loader, task: str, save_dir: Path,
    cached_val_metric: float | None,
) -> dict[str, Any]:
    """Build a task-aware (loader, metric_fn) pair and run the robustness sweep.

    MVP coverage: detection / classification / segmentation / keypoint all wired.
    """
    from core.p08_evaluation import robustness_sweep
    task_low = (task or "").lower()
    if task_low not in {"detection", "classification", "segmentation", "keypoint"}:
        logger.info(
            "robustness_sweep: task=%s not wired for loader-based metric; skipping",
            task_low,
        )
        return {}

    # Build a loader if the trainer didn't pass one (HF eval_dataloader path
    # always does; pytorch trainer also passes self.val_loader).
    loader = val_loader
    if loader is None:
        loader = _build_minimal_val_loader(val_dataset, task_low)
    if loader is None:
        logger.info("robustness_sweep: could not build val loader; skipping")
        return {}

    if task_low == "detection":
        metric_fn = _make_detection_metric_fn()
        primary = "mAP@0.5"
    elif task_low == "classification":
        metric_fn = _make_classification_metric_fn()
        primary = "accuracy"
    elif task_low == "keypoint":
        metric_fn = _make_keypoint_metric_fn()
        primary = "PCK@0.05"
    else:  # segmentation
        metric_fn = _make_segmentation_metric_fn(model)
        primary = "mIoU"

    # If the caller already has the val metric cached, optionally short-circuit
    # the clean baseline. robustness_sweep recomputes it; the redundant pass is
    # cheap and keeps signatures uniform.
    _ = cached_val_metric  # reserved for future fast-path

    return robustness_sweep.run(
        model=model, val_loader=loader, task=task_low,
        metric_fn=metric_fn, primary_metric_name=primary,
        output_dir=save_dir,
    )


def _build_minimal_val_loader(dataset, task: str):
    """Best-effort DataLoader over a raw Dataset for robustness sweeps.

    Detection datasets need their custom collate; without a passed-in loader
    we fall back to a no-op collate that yields per-sample dicts. The metric
    functions below tolerate either batched tensors or list-of-samples.
    """
    if dataset is None or len(dataset) == 0:
        return None
    from torch.utils.data import DataLoader
    try:
        return DataLoader(
            dataset, batch_size=4, shuffle=False, num_workers=0,
            collate_fn=lambda batch: batch,  # list-of-samples; metric_fn handles
        )
    except Exception:
        return None


def _make_detection_metric_fn():
    """Return metric_fn(model, loader) -> mAP@0.5 over the (possibly
    corrupted) loader. Reuses ``compute_map`` from p08."""
    from core.p08_evaluation.sv_metrics import compute_map

    def metric_fn(model, loader) -> float:
        device = next(model.parameters()).device
        all_preds: list[dict] = []
        all_gts: list[dict] = []
        for batch in loader:
            tensors, gts = _normalize_det_batch(batch, device)
            if not tensors:
                continue
            target_sizes = torch.tensor(
                [t.shape[-2:] for t in tensors], dtype=torch.int64,
            )
            preds = _forward_batch_detection(
                model, tensors, target_sizes, conf_threshold=0.05,
            )
            for p in preds:
                all_preds.append({
                    "boxes": np.asarray(p.get("boxes", [])).reshape(-1, 4),
                    "scores": np.asarray(p.get("scores", [])).ravel(),
                    "labels": np.asarray(p.get("labels", []), dtype=np.int64).ravel(),
                })
            all_gts.extend(gts)
        if not all_preds:
            return 0.0
        try:
            res = compute_map(all_preds, all_gts, iou_threshold=0.5)
            return float(res.get("mAP50", res.get("mAP", 0.0)))
        except Exception as e:  # pragma: no cover
            logger.warning("compute_map failed: %s", e)
            return 0.0
    return metric_fn


def _normalize_det_batch(batch, device):
    """Coerce a batch (list-of-samples OR (images, targets) tuple OR dict)
    into (tensors, gt_dicts)."""
    tensors: list[torch.Tensor] = []
    gts: list[dict] = []
    if isinstance(batch, list) and batch and isinstance(batch[0], (tuple, list)):
        # list-of-samples from our minimal collate: each item is (img, targets)
        for img, t in batch:
            if torch.is_tensor(img):
                tensors.append(img.to(device))
                gts.append(_yolo_targets_to_gt_dict(t, img.shape[-2:]))
    elif isinstance(batch, (tuple, list)) and len(batch) >= 2 and torch.is_tensor(batch[0]):
        imgs = batch[0].to(device)
        for i in range(imgs.shape[0]):
            tensors.append(imgs[i])
            ti = batch[1][i] if i < len(batch[1]) else None
            gts.append(_yolo_targets_to_gt_dict(ti, imgs.shape[-2:]))
    return tensors, gts


def _yolo_targets_to_gt_dict(t, hw):
    h, w = int(hw[0]), int(hw[1])
    if t is None:
        return {"boxes": np.zeros((0, 4)), "labels": np.zeros(0, dtype=np.int64)}
    arr = t.cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
    if arr.size == 0 or arr.ndim < 2:
        return {"boxes": np.zeros((0, 4)), "labels": np.zeros(0, dtype=np.int64)}
    xyxy, cls = _gt_xyxy_from_yolo(arr, w, h)
    return {"boxes": np.asarray(xyxy).reshape(-1, 4),
            "labels": np.asarray(cls, dtype=np.int64).ravel()}


def _make_classification_metric_fn():
    def metric_fn(model, loader) -> float:
        device = next(model.parameters()).device
        correct = 0
        total = 0
        for batch in loader:
            imgs, labels = _normalize_cls_batch(batch, device)
            if imgs is None:
                continue
            with torch.no_grad():
                out = model(pixel_values=imgs) if hasattr(model, "hf_model") else model(imgs)
            logits = getattr(out, "logits", out)
            preds = logits.argmax(dim=-1).cpu().numpy()
            labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
            correct += int((preds == labels_np).sum())
            total += int(preds.shape[0])
        return correct / max(total, 1)
    return metric_fn


def _normalize_cls_batch(batch, device):
    if isinstance(batch, list) and batch and isinstance(batch[0], (tuple, list)):
        imgs = torch.stack([b[0] for b in batch if torch.is_tensor(b[0])]).to(device)
        labels = torch.tensor([int(b[1]) for b in batch], dtype=torch.long)
        return imgs, labels
    if isinstance(batch, (tuple, list)) and len(batch) >= 2 and torch.is_tensor(batch[0]):
        return batch[0].to(device), batch[1]
    if isinstance(batch, dict):
        x = batch.get("pixel_values") or batch.get("images")
        y = batch.get("labels")
        if torch.is_tensor(x):
            return x.to(device), y
    return None, None


def _make_keypoint_metric_fn():
    """Return ``metric_fn(model, loader) -> PCK@0.05`` (mean over visible joints).

    PCK@0.05: a predicted joint is correct when its L2 distance to the GT joint
    is within 5% of the GT-keypoint bounding-box diagonal. Visibility is
    enforced via ``gt_kp[:, 2] > 0`` — invisible/occluded joints don't count
    in the denominator. Reuses the same postprocess chain
    (:func:`_forward_batch_detection`) the trainer's keypoint preview path
    uses, so per-batch decode matches end-of-training.
    """
    def metric_fn(model, loader) -> float:
        device = next(model.parameters()).device
        correct = 0
        total = 0
        for batch in loader:
            tensors, gt_kp_list = _normalize_kpt_batch(batch, device)
            if not tensors:
                continue
            target_sizes = torch.tensor(
                [t.shape[-2:] for t in tensors], dtype=torch.int64,
            )
            decoded = _forward_batch_detection(
                model, tensors, target_sizes, conf_threshold=0.0,
            )
            for i, gt_kp in enumerate(gt_kp_list):
                if gt_kp is None or gt_kp.size == 0:
                    continue
                gt_kp = np.asarray(gt_kp, dtype=np.float32).reshape(-1, 3)
                dec = decoded[i] if i < len(decoded) else {}
                pred_kp = np.asarray(
                    dec.get("keypoints", []), dtype=np.float32,
                ).reshape(-1, 3)
                vis = gt_kp[:, 2] > 0
                if not vis.any():
                    continue
                xy_min = gt_kp[vis, :2].min(axis=0)
                xy_max = gt_kp[vis, :2].max(axis=0)
                diag = float(max(1.0, np.hypot(*(xy_max - xy_min))))
                thr = 0.05 * diag
                K_pred = len(pred_kp)
                for k in range(len(gt_kp)):
                    if gt_kp[k, 2] <= 0:
                        continue
                    total += 1
                    if k < K_pred and pred_kp[k, 2] > 0:
                        d = float(np.hypot(
                            pred_kp[k, 0] - gt_kp[k, 0],
                            pred_kp[k, 1] - gt_kp[k, 1],
                        ))
                        if d <= thr:
                            correct += 1
        return correct / max(total, 1)
    return metric_fn


def _normalize_kpt_batch(batch, device):
    """Coerce a keypoint batch into (image-tensors, gt-keypoints-list).

    Mirrors :func:`_normalize_det_batch` but expects per-sample targets shaped
    either as a dict ``{"keypoints": (K,3)}`` or a raw ``(K,3)`` array.
    """
    tensors: list[torch.Tensor] = []
    gts: list[np.ndarray | None] = []

    def _extract_kp(t):
        if t is None:
            return None
        if isinstance(t, dict):
            t = t.get("keypoints")
        if t is None:
            return None
        if torch.is_tensor(t):
            return t.cpu().numpy()
        return np.asarray(t)

    if isinstance(batch, list) and batch and isinstance(batch[0], (tuple, list)):
        for img, t in batch:
            if torch.is_tensor(img):
                tensors.append(img.to(device))
                gts.append(_extract_kp(t))
    elif isinstance(batch, (tuple, list)) and len(batch) >= 2 and torch.is_tensor(batch[0]):
        imgs = batch[0].to(device)
        for i in range(imgs.shape[0]):
            tensors.append(imgs[i])
            ti = batch[1][i] if i < len(batch[1]) else None
            gts.append(_extract_kp(ti))
    return tensors, gts


def _make_segmentation_metric_fn(model):
    """Pixel mIoU averaged across non-empty classes."""
    from core.p08_evaluation.learning_ability import _resolve_num_classes
    n_classes = _resolve_num_classes(model)

    def metric_fn(model, loader) -> float:
        device = next(model.parameters()).device
        intersect = np.zeros(n_classes, dtype=np.int64)
        union = np.zeros(n_classes, dtype=np.int64)
        for batch in loader:
            imgs, masks = _normalize_seg_batch(batch, device)
            if imgs is None:
                continue
            with torch.no_grad():
                out = model(pixel_values=imgs) if hasattr(model, "hf_model") else model(imgs)
            logits = getattr(out, "logits", out)
            if logits.ndim != 4:
                continue
            logits = torch.nn.functional.interpolate(
                logits, size=masks.shape[-2:], mode="bilinear", align_corners=False,
            )
            preds = logits.argmax(dim=1).cpu().numpy()
            gt = masks.cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
            valid = gt != 0
            for cid in range(n_classes):
                gc = (gt == cid) & valid
                pc = (preds == cid) & valid
                intersect[cid] += int((gc & pc).sum())
                union[cid] += int((gc | pc).sum())
        ious = np.where(union > 0, intersect / (union + 1e-9), 0.0)
        return float(ious[union > 0].mean()) if (union > 0).any() else 0.0
    return metric_fn


def _normalize_seg_batch(batch, device):
    if isinstance(batch, list) and batch and isinstance(batch[0], (tuple, list)):
        imgs = torch.stack([b[0] for b in batch if torch.is_tensor(b[0])]).to(device)
        masks = torch.stack([
            torch.as_tensor(b[1]) for b in batch
        ])
        return imgs, masks
    if isinstance(batch, (tuple, list)) and len(batch) >= 2 and torch.is_tensor(batch[0]):
        return batch[0].to(device), batch[1]
    if isinstance(batch, dict):
        x = batch.get("pixel_values")
        y = batch.get("labels")
        if torch.is_tensor(x):
            return x.to(device), y
    return None, None


def _resolve_val_metric_for_la(
    task: str, cached_val: float | None, val_dataset, model, input_size,
) -> tuple[str, float]:
    """Get the (name, value) of the primary val metric for LA comparison.

    Detection: uses HF Trainer's cached `eval_map_50` from log history.
    Cls / seg: re-evaluates on val with the LA evaluator (cheap; same
    primitives as the train eval).
    """
    from core.p08_evaluation.learning_ability import _eval_dataset_metric
    if task == "detection":
        return ("mAP50", float(cached_val) if cached_val is not None else 0.0)
    # For seg/cls: recompute on the (already small) val set.
    if val_dataset is not None and len(val_dataset) > 0:
        n = len(val_dataset)
        idxs = list(range(min(n, 500)))
        v = _eval_dataset_metric(model, val_dataset, idxs, task=task, input_size=input_size)
        if v is not None:
            return (
                "mIoU" if task == "segmentation" else "accuracy",
                float(v),
            )
    return ("metric", 0.0)


def _append_section(
    *, summary_md: Path, heading: str, stems: tuple[str, ...],
    chart_dir: Path, chart_metrics: dict,
) -> None:
    """Append a named section (one or more stems) to an existing summary.md."""
    if not summary_md.exists():
        return
    import os
    from core.p08_evaluation.chart_annotations import (
        CHART_META,
        evaluate_next_step,
        render_signal,
    )
    lines: list[str] = ["", heading, ""]
    wrote_any = False
    for stem in stems:
        png = chart_dir / f"{stem}.png"
        if not png.exists():
            continue
        wrote_any = True
        meta = CHART_META.get(stem)
        rel = os.path.relpath(png, start=summary_md.parent)
        sub_heading = meta.title if meta else stem
        lines.append(f"### {sub_heading}")
        lines.append("")
        lines.append(f"![{stem}]({rel})")
        lines.append("")
        if meta:
            lines.append(f"**What this shows.** {meta.description}")
            lines.append("")
            signal = render_signal(meta, chart_metrics.get(stem))
            if signal:
                lines.append(f"**Current signal.** {signal}")
                lines.append("")
            advice = evaluate_next_step(chart_metrics.get(stem), meta)
            lines.append(f"**Suggested next step.** {advice}")
            lines.append("")
    if not wrote_any:
        return
    with open(summary_md, "a") as f:
        f.write("\n".join(lines))

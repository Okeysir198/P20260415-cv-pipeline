"""Learning-ability / bias-vs-variance analyzer.

Evaluates the trained model on a stratified subset of the **training set**
in addition to the val set, then classifies the model's regime:

- ``train ≈ val and both low``   → high-bias (underfit)
- ``train high, val much lower`` → high-variance (overfit)
- ``train high, val high``       → healthy

Also parses ``trainer_state.json`` (HF backend) for per-epoch curves and
renders an overlay of train_loss + eval_metric across epochs.

Outputs (flat, under the error_analysis dir passed as ``save_dir``):

- ``06_learning_ability.png`` — 2-panel figure (train vs val bars
  + per-epoch learning curves).
- ``06_learning_ability.json`` — regime + numeric deltas.

Hooks into :func:`core.p06_training.post_train.run_post_train_artifacts`.
The analyzer is lightweight: it runs a single forward pass over a 500-image
stratified sample of train (default), so post-train wall-time grows by
~10-30 seconds depending on backbone.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from loguru import logger

matplotlib.use("Agg")


LA_FILENAMES: dict[str, str] = {
    "learning_ability": "06_learning_ability.png",
    "json":             "06_learning_ability.json",
}

# Regime thresholds — kept simple + interpretable.
_BIAS_VAL_CEIL = 0.40         # both < this → "model couldn't fit even train"
_HEALTHY_GAP_FLOOR = 0.05     # train − val ≤ this AND val ≥ ceil → healthy
_HIGH_VAR_GAP = 0.15          # train − val ≥ this → overfitting


def analyze_learning_ability(
    *,
    model,
    train_dataset,
    val_metric: float,
    primary_metric_name: str,
    task: str,
    save_dir: Path | str,
    input_size: tuple[int, int],
    train_subset_max: int = 500,
    trainer_state_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Run train-set eval + render train-vs-val regime + learning curves.

    Args:
        model: best-checkpoint model in eval mode.
        train_dataset: full train Dataset (will be subset-sampled).
        val_metric: already-computed val metric (mIoU / accuracy / mAP50).
        primary_metric_name: human label for the metric (e.g. ``"mIoU"``).
        task: canonical task string.
        save_dir: run dir (``LA_*.png`` go under ``save_dir / "learning_ability"``).
        input_size: model input HxW.
        train_subset_max: max number of train samples to evaluate (stratified
            sample is just random sampling here for simplicity).
        trainer_state_dir: dir containing HF Trainer's ``trainer_state.json``
            for the learning-curves plot. If None, looks under each child
            ``checkpoint-*`` dir of ``save_dir`` for the latest one.
    """
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- 1. train-set evaluation (subset) --------
    n = len(train_dataset)
    if n == 0:
        logger.info("analyze_learning_ability: empty train_dataset; skipping")
        return {}
    n_eval = min(train_subset_max, n)
    rng = np.random.RandomState(42)
    indices = sorted(rng.choice(n, n_eval, replace=False).tolist())

    train_metric = _eval_dataset_metric(
        model, train_dataset, indices, task=task, input_size=input_size,
    )
    if train_metric is None:
        logger.info("analyze_learning_ability: train eval not supported for task=%s", task)
        return {}

    # -------- 2. regime classification --------
    gap = float(train_metric - val_metric)
    if val_metric < _BIAS_VAL_CEIL and gap < _HEALTHY_GAP_FLOOR:
        regime = "high_bias"
        advice = (
            "Both train and val are low — model is too weak (or labels / "
            "features are bad). Try a larger model, better backbone, "
            "or audit label quality."
        )
    elif gap >= _HIGH_VAR_GAP:
        regime = "high_variance"
        advice = (
            "Train metric is much higher than val — overfitting. Add "
            "regularization (dropout, weight decay), more / stronger "
            "augmentation, or collect more training data."
        )
    elif val_metric >= _BIAS_VAL_CEIL and gap < _HIGH_VAR_GAP:
        regime = "healthy"
        advice = (
            "Train and val are both high with a small gap — healthy. "
            "Marginal gains from longer training, EMA, or ensembling."
        )
    else:
        regime = "ambiguous"
        advice = (
            "Mixed signal — neither clearly underfit nor overfit. "
            "Inspect per-class metrics and the learning-curves plot before "
            "deciding what to change."
        )

    # -------- 3. render combined 2-panel figure --------
    # Look up learning-curves data from trainer_state.json if available;
    # when absent the curves sub-plot collapses to a "not available" note.
    state_path = _resolve_trainer_state(
        Path(trainer_state_dir).parent if trainer_state_dir else out_dir.parent.parent,
        trainer_state_dir,
    )
    artifacts: dict[str, Path] = {}
    artifacts["learning_ability"] = _plot_combined(
        train_metric=train_metric, val_metric=val_metric,
        primary_metric_name=primary_metric_name, regime=regime, advice=advice,
        state_path=state_path,
        out_path=out_dir / LA_FILENAMES["learning_ability"],
    )

    # -------- 4. exports --------
    payload = {
        "task": task,
        "primary_metric": primary_metric_name,
        "train_metric": round(train_metric, 4),
        "val_metric": round(val_metric, 4),
        "gap": round(gap, 4),
        "regime": regime,
        "advice": advice,
        "n_train_samples_evaluated": n_eval,
    }
    json_path = out_dir / LA_FILENAMES["json"]
    json_path.write_text(json.dumps(payload, indent=2))
    artifacts["json"] = json_path

    chart_metrics = {
        "06_learning_ability": {
            "train_metric": train_metric,
            "val_metric": val_metric,
            "gap": gap,
            "regime": regime,
            "advice": advice,
            "primary_metric_name": primary_metric_name,
        },
    }
    return {"artifacts": artifacts, "chart_metrics": chart_metrics, "payload": payload}


def _plot_combined(
    *, train_metric, val_metric, primary_metric_name, regime, advice,
    state_path, out_path,
) -> Path:
    """Render bars + learning curves in a single 2-panel figure."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    color_map = {
        "high_bias":     "#d62728",
        "high_variance": "#ff7f0e",
        "healthy":       "#2ca02c",
        "ambiguous":     "#7f7f7f",
    }
    fig, (ax_bars, ax_curves) = plt.subplots(1, 2, figsize=(13, 5))

    bars = ax_bars.bar(
        ["train", "val"], [train_metric, val_metric],
        color=color_map.get(regime, "#1f77b4"),
        edgecolor="black", linewidth=1,
    )
    for b, v in zip(bars, [train_metric, val_metric], strict=False):
        ax_bars.text(
            b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
            f"{v:.3f}", ha="center", fontsize=11, fontweight="bold",
        )
    ax_bars.set_ylabel(primary_metric_name)
    ax_bars.set_ylim(0, max(1.0, max(train_metric, val_metric) * 1.2))
    ax_bars.set_title(
        f"Train vs Val — regime: **{regime.upper()}**\n"
        f"gap = {train_metric - val_metric:+.3f}",
        fontsize=11,
    )

    # -------- Curves sub-plot --------
    plotted = False
    if state_path is not None:
        try:
            _draw_learning_curves(state_path, primary_metric_name, ax_curves)
            plotted = True
        except Exception as e:  # pragma: no cover
            logger.warning("learning curves subplot skipped: %s", e)
    if not plotted:
        ax_curves.text(
            0.5, 0.5, "learning curves unavailable\n(no trainer_state.json found)",
            ha="center", va="center", transform=ax_curves.transAxes,
            fontsize=10, style="italic", color="#666666",
        )
        ax_curves.set_xticks([])
        ax_curves.set_yticks([])
    ax_curves.set_title("Learning curves — train_loss + eval metric")

    fig.text(0.5, 0.01, advice, ha="center", fontsize=9, style="italic", wrap=True)
    plt.subplots_adjust(bottom=0.16)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _draw_learning_curves(state_path: Path, primary_metric_name: str, ax1) -> None:
    """Populate an axes with train_loss (left) + eval_metric (right) overlay."""
    state = json.loads(Path(state_path).read_text())
    history = state.get("log_history", [])
    train_epochs, train_losses = [], []
    eval_epochs, eval_metrics = [], []
    metric_key = _pick_eval_metric_key(history, primary_metric_name)
    for entry in history:
        e = entry.get("epoch")
        if e is None:
            continue
        if "loss" in entry and "eval_loss" not in entry:
            train_epochs.append(e); train_losses.append(float(entry["loss"]))
        if metric_key in entry:
            eval_epochs.append(e); eval_metrics.append(float(entry[metric_key]))

    if train_losses:
        ax1.plot(train_epochs, train_losses, color="#1f77b4",
                 label="train loss", linewidth=1.5, alpha=0.9)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    if eval_metrics:
        ax2 = ax1.twinx()
        ax2.plot(eval_epochs, eval_metrics, color="#d62728",
                 marker="o", markersize=4, label=f"eval {primary_metric_name}",
                 linewidth=1.5)
        ax2.set_ylabel(f"eval {primary_metric_name}", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")


# ---------------------------------------------------------------------------
# Per-task metric evaluation
# ---------------------------------------------------------------------------


def _eval_dataset_metric(
    model, dataset, indices: list[int], *, task: str, input_size: tuple[int, int],
) -> float | None:
    """Compute the canonical metric (mIoU / accuracy / mAP50) over indices.

    Returns None for tasks not yet wired (detection — needs the predictor +
    matcher chain we don't want to duplicate here).
    """
    task = task.lower()
    if task == "segmentation":
        return _eval_segmentation_miou(model, dataset, indices, input_size)
    if task == "classification":
        return _eval_classification_accuracy(model, dataset, indices, input_size)
    if task == "detection":
        return _eval_detection_map50(model, dataset, indices, input_size)
    return None


def _eval_detection_map50(
    model, dataset, indices: list[int], input_size: tuple[int, int],
) -> float:
    """mAP@0.5 over a fixed sample of indices.

    Reuses the same predictor + matcher chain as end-of-training:
    `_forward_batch_detection` (which dispatches to either the model's own
    `.postprocess` for HF wrappers or the registry postprocess for YOLOX/
    custom heads) feeds detections into `compute_map(..., iou_thresholds=[0.5])`.
    No NMS or matching reimplementation here.
    """
    from core.p06_training._common import yolo_targets_to_xyxy as _gt_to_xyxy
    from core.p06_training.post_train import _forward_batch_detection
    from core.p08_evaluation.error_analysis_runner import _preprocess_for_model
    from core.p08_evaluation.label_quality import _safe_get_raw_item
    from core.p08_evaluation.sv_metrics import compute_map

    device = next(model.parameters()).device
    h_in, w_in = int(input_size[0]), int(input_size[1])
    all_preds: list[dict] = []
    all_gts: list[dict] = []

    for idx in indices:
        try:
            raw = _safe_get_raw_item(dataset, idx)
            if raw is None or raw.get("image") is None:
                continue
            img = raw["image"]
            gt = raw.get("targets")
            H, W = img.shape[:2]
            tensor = _preprocess_for_model(img, (h_in, w_in), model=model)
            target_size = torch.tensor([[h_in, w_in]], dtype=torch.int64)
            preds = _forward_batch_detection(
                model, [tensor.to(device)], target_size, conf_threshold=0.05,
            )
            p = preds[0] if preds else {}
            pred_boxes = np.asarray(p.get("boxes", []), dtype=np.float64).reshape(-1, 4)
            if len(pred_boxes) > 0:
                pred_boxes[:, [0, 2]] *= W / w_in
                pred_boxes[:, [1, 3]] *= H / h_in
            all_preds.append({
                "boxes": pred_boxes,
                "scores": np.asarray(p.get("scores", []), dtype=np.float64).ravel(),
                "labels": np.asarray(p.get("labels", []), dtype=np.int64).ravel(),
            })
            if isinstance(gt, np.ndarray) and gt.size > 0:
                gt_xyxy, gt_cls = _gt_to_xyxy(gt, W, H)
                all_gts.append({
                    "boxes": np.asarray(gt_xyxy).reshape(-1, 4),
                    "labels": np.asarray(gt_cls, dtype=np.int64).ravel(),
                })
            else:
                all_gts.append({
                    "boxes": np.zeros((0, 4)), "labels": np.zeros(0, dtype=np.int64),
                })
        except Exception as e:  # pragma: no cover
            logger.warning("eval_detection_map50: idx %d failed — %s", idx, e)

    if not all_preds:
        return 0.0
    try:
        res = compute_map(all_preds, all_gts, iou_threshold=0.5)
        return float(res.get("mAP50", res.get("mAP", 0.0)))
    except Exception as e:  # pragma: no cover
        logger.warning("eval_detection_map50: compute_map failed — %s", e)
        return 0.0


def _eval_segmentation_miou(
    model, dataset, indices: list[int], input_size: tuple[int, int],
) -> float:
    """Per-pixel mIoU averaged across classes (excluding empty classes)."""
    from core.p08_evaluation.error_analysis_runner import _preprocess_for_model
    from core.p08_evaluation.label_quality import _safe_get_raw_item

    device = next(model.parameters()).device
    # Discover class count from id2label / config.
    n_classes = _resolve_num_classes(model)
    intersect = np.zeros(n_classes, dtype=np.int64)
    union = np.zeros(n_classes, dtype=np.int64)

    for idx in indices:
        try:
            raw = _safe_get_raw_item(dataset, idx)
            if raw is None or raw.get("image") is None or raw.get("targets") is None:
                continue
            image_bgr = raw["image"]
            gt_mask = np.asarray(raw["targets"])
            x = _preprocess_for_model(image_bgr, input_size, model=model).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(x)
            from core.p08_evaluation.label_quality import _extract_logits
            logits = _extract_logits(out)
            if logits.ndim == 4:
                h, w = gt_mask.shape
                logits = torch.nn.functional.interpolate(
                    logits, size=(h, w), mode="bilinear", align_corners=False,
                )
                pred = logits.argmax(dim=1)[0].cpu().numpy()
            else:
                continue
            valid = gt_mask != 0  # ignore class 0 by convention
            for cid in range(n_classes):
                gt_c = (gt_mask == cid) & valid
                pred_c = (pred == cid) & valid
                intersect[cid] += int((gt_c & pred_c).sum())
                union[cid] += int((gt_c | pred_c).sum())
        except Exception as e:  # pragma: no cover
            logger.warning("eval_segmentation_miou: idx %d failed — %s", idx, e)

    ious = np.where(union > 0, intersect / (union + 1e-9), 0.0)
    valid_classes = union > 0
    return float(ious[valid_classes].mean()) if valid_classes.any() else 0.0


def _eval_classification_accuracy(
    model, dataset, indices: list[int], input_size: tuple[int, int],
) -> float:
    from core.p08_evaluation.label_quality import _cls_forward, _safe_get_raw_item
    correct = 0
    total = 0
    device = next(model.parameters()).device
    for idx in indices:
        try:
            raw = _safe_get_raw_item(dataset, idx)
            if raw is None or raw.get("image") is None:
                continue
            try:
                gt_cid = int(raw.get("targets"))
            except (TypeError, ValueError):
                continue
            pred_cid, _ = _cls_forward(model, raw["image"], input_size, device)
            total += 1
            if pred_cid == gt_cid:
                correct += 1
        except Exception as e:  # pragma: no cover
            logger.warning("eval_classification_accuracy: idx %d failed — %s", idx, e)
    return correct / max(total, 1)


def _resolve_num_classes(model) -> int:
    """Best-effort lookup of class count from the model wrapper."""
    inner = getattr(model, "hf_model", model)
    cfg = getattr(inner, "config", None)
    if cfg is not None and getattr(cfg, "num_labels", None):
        return int(cfg.num_labels)
    if cfg is not None and getattr(cfg, "id2label", None):
        return len(cfg.id2label)
    if getattr(model, "num_classes", None):
        return int(model.num_classes)
    return 35  # safe default


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


def _pick_eval_metric_key(history: list[dict], primary: str) -> str:
    """Best-effort match: prefers eval_<primary>, falls back to first eval_* key."""
    primary_low = primary.lower().replace(" ", "_")
    candidates = [f"eval_{primary_low}", f"eval_{primary_low.replace('-', '_')}"]
    for c in candidates:
        if any(c in e for e in history):
            return c
    for entry in history:
        for k in entry:
            if k.startswith("eval_") and k != "eval_loss":
                return k
    return ""


def _resolve_trainer_state(save_dir: Path, override: Path | str | None) -> Path | None:
    """Find the most-recent ``trainer_state.json`` available."""
    if override:
        p = Path(override) / "trainer_state.json"
        return p if p.exists() else None
    candidates = sorted(save_dir.glob("checkpoint-*/trainer_state.json"))
    if not candidates:
        return None
    # Pick the highest checkpoint number (most recent).
    return max(candidates, key=lambda p: int(p.parent.name.split("-")[-1]))

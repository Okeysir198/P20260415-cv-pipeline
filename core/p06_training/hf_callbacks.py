"""Native `transformers.TrainerCallback` subclasses for the HF detection backend.

Replaces the earlier `_HFVizBridge` attribute-proxy adapter — these are
first-class `TrainerCallback`s that read everything they need from HF's
documented callback kwargs (model, train_dataloader, eval_dataloader,
state.log_history) instead of synthesising a fake trainer object. Safer
against future HF Trainer API changes.

Four callbacks, one per viz we emit:

- :class:`HFDatasetStatsCallback`   — on_train_begin: `00_dataset_info.{md,json}` +
  `01_dataset_stats.{png,json}`
- :class:`HFDataLabelGridCallback`  — on_train_begin: `02_data_labels_<split>.png` per split
- :class:`HFAugLabelGridCallback`   — on_train_begin: `03_aug_labels_train.png`
- :class:`HFValPredictionCallback`  — on_epoch_end: `val_predictions/epoch_<N>.png`

Each takes all the data/config it needs at `__init__` so no trainer-proxy
attribute fetching is needed at hook time. Rendering helpers are imported
directly from the internal `callbacks` module — the module-level functions
there are pure (no trainer dependency).
"""
from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import supervision as sv
from loguru import logger
from transformers import TrainerCallback

from core.p05_data.transforms import build_transforms
from core.p06_training._common import (
    build_dataset_for_viz,
    task_from_output_format,
    yolo_targets_to_xyxy,
)
from utils.viz import (
    VizStyle,
    annotate_detections,
    classification_banner,
    save_image_grid,
)


def _is_topdown_keypoint_dataset(ds: Any) -> bool:
    """Return True for ``KeypointTopDownDataset`` (or a Subset wrapping one)."""
    inner = getattr(ds, "dataset", ds)
    return type(inner).__name__ == "KeypointTopDownDataset"


def _collect_source_bboxes(ds: Any, indices: np.ndarray) -> np.ndarray:
    """Return ``(N, 4)`` source-image bboxes in pixel xywh for each sample idx.

    For ``KeypointTopDownDataset`` we read the YOLO row + source image dims
    out of ``_index`` directly without re-decoding pixels. The returned
    bbox lives in **source-image** pixel coords (not crop coords) so
    pycocotools' area-based bucketing reflects the real person size in
    the original image. Falls back to a unit-area placeholder if the
    underlying dataset can't supply it.
    """
    inner = getattr(ds, "dataset", ds)
    n = len(indices)
    out = np.zeros((n, 4), dtype=np.float32)
    if type(inner).__name__ != "KeypointTopDownDataset":
        return out
    import cv2
    img_dims_cache: dict[Path, tuple[int, int]] = {}
    for i, idx in enumerate(indices):
        try:
            img_path, row = inner._index[int(idx)]
        except (AttributeError, IndexError):
            continue
        if img_path in img_dims_cache:
            ih, iw = img_dims_cache[img_path]
        else:
            im = cv2.imread(str(img_path))
            if im is None:
                continue
            ih, iw = im.shape[:2]
            img_dims_cache[img_path] = (ih, iw)
        cx, cy, bw, bh = float(row[1]), float(row[2]), float(row[3]), float(row[4])
        x = (cx - bw / 2.0) * iw
        y = (cy - bh / 2.0) * ih
        w = bw * iw
        h = bh * ih
        out[i] = [max(0.0, x), max(0.0, y), max(1.0, w), max(1.0, h)]
    return out


def _resolve_skeleton_edges(
    inner_ds: Any,
) -> list[tuple[int, int]] | None:
    """Pull skeleton edges from a dataset's `data_config["skeleton"]`.

    Returns ``None`` when no skeleton is defined — callers pass that to
    ``annotate_keypoints`` which then renders dots only.
    """
    cfg = getattr(inner_ds, "data_config", None) or {}
    raw = cfg.get("skeleton") or []
    edges: list[tuple[int, int]] = []
    for pair in raw:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            try:
                a, b = int(pair[0]), int(pair[1])
                edges.append((a, b))
            except (TypeError, ValueError):
                continue
    return edges or None


def _denormalize_pixel_values(
    pixel_values: np.ndarray,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
) -> np.ndarray:
    """Invert ImageNet normalization on a (3, H, W) tensor → uint8 RGB (H, W, 3)."""
    px = pixel_values.astype(np.float32)
    if px.ndim == 3 and px.shape[0] == 3:
        px = px.transpose(1, 2, 0)
    px = px * np.asarray(std, dtype=np.float32) + np.asarray(mean, dtype=np.float32)
    px = np.clip(px * 255.0, 0, 255).astype(np.uint8)
    return px


def _run_topdown_keypoint_post_train(
    model: Any,
    val_ds: Any,
    test_ds: Any,
    save_dir: Path,
    *,
    best_num_samples: int = 16,
    grid_cols: int = 4,
    dpi: int = 150,
) -> None:
    """End-of-training observability for top-down keypoint runs.

    Renders ``val_predictions/best.png`` (and ``test_predictions/best.png``
    if a test split is present), then a compact error analysis tree:

    - 01_overview: PCK@{0.05,0.1,0.2}, OKS-AP, OKS-AP50/75, mean pixel err.
    - 07_per_joint_performance: per-joint PCK@0.1, sorted ascending.
    - 08_confusion_left_right: 8 L↔R pairs × {correct, swapped, both wrong}
      heatmap (the workhorse diagnostic for symmetric-joint failures).
    - 09_confidence_vs_error: per-joint score-vs-pixel-error scatter
      (calibration check).
    - 10_failure_mode_contribution: per-joint × {ok, near_miss, far_off,
      low_score, ghost} heatmap (fractions of GT-visible joints).
    - 12_hardest_crops: top-12 worst val crops (highest mean pixel err)
      with GT|Pred skeletons side-by-side.
    - 13_failure_mode_examples/: galleries grouped by failure type —
      ``high_error/kp_<k>/``, ``ghost/kp_<k>/``, ``swapped_pair/<L>__<R>/``.
    - summary.{json,md}: compact metrics dump.

    No detection-shaped postprocess; everything goes through the same
    heatmap decoder used by the per-epoch grid + compute_metrics.
    """
    import json

    import torch

    from core.p08_evaluation.keypoint_metrics import (
        COCO_KP_SIGMAS,
        compute_oks,
        compute_oks_ap,
        compute_pck,
        decode_heatmaps_to_xy,
    )

    inner = getattr(val_ds, "dataset", val_ds)
    input_h, input_w = inner.input_hw
    num_kp = inner.num_keypoints
    sigmas = np.asarray(
        getattr(inner, "data_config", {}).get("oks_sigmas") or COCO_KP_SIGMAS[:num_kp],
        dtype=np.float32,
    )

    # ---- 1) Run model over the entire val split. ----
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    all_pred_xy: list[np.ndarray] = []
    all_pred_score: list[np.ndarray] = []
    all_gt_xy: list[np.ndarray] = []
    all_weight: list[np.ndarray] = []
    all_pixel_err: list[np.ndarray] = []
    all_per_idx: list[int] = []
    batch_size = 32

    n = len(val_ds)
    indices = list(range(n))
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            chunk = indices[start:start + batch_size]
            items = [val_ds[i] for i in chunk]
            px = torch.stack([it["pixel_values"] for it in items]).to(device)
            tg = torch.stack([it["target_heatmap"] for it in items]).numpy()
            tw = torch.stack([it["target_weight"] for it in items]).numpy()
            out = model(pixel_values=px)
            pred_h = out["heatmaps"] if isinstance(out, dict) else out.heatmaps
            pred_h = pred_h.detach().float().cpu().numpy()
            stride = int(input_h // pred_h.shape[2])
            pred_xy, pred_sc = decode_heatmaps_to_xy(pred_h, stride=stride)
            gt_xy, _ = decode_heatmaps_to_xy(tg, stride=stride)
            err = np.linalg.norm(pred_xy - gt_xy, axis=-1)
            all_pred_xy.append(pred_xy)
            all_pred_score.append(pred_sc)
            all_gt_xy.append(gt_xy)
            all_weight.append(tw)
            all_pixel_err.append(err)
            all_per_idx.extend(chunk)

    pred_xy = np.concatenate(all_pred_xy, axis=0)
    pred_score = np.concatenate(all_pred_score, axis=0)
    gt_xy = np.concatenate(all_gt_xy, axis=0)
    weight = np.concatenate(all_weight, axis=0)
    pixel_err = np.concatenate(all_pixel_err, axis=0)
    sample_idx = np.asarray(all_per_idx, dtype=np.int64)
    if was_training:
        model.train()

    ref_size = float(np.sqrt(input_h * input_h + input_w * input_w))
    pck = compute_pck(pred_xy, gt_xy, weight, ref_size=ref_size)
    oks = compute_oks(pred_xy, gt_xy, weight, sigmas=sigmas, area=float(input_h * input_w))
    ap_dict = compute_oks_ap(oks)

    # ---- 2) best.png — top-N samples ranked by OKS (best first). ----
    best_indices_topn = np.argsort(-oks).tolist()[:best_num_samples]
    pick = [sample_idx[i] for i in best_indices_topn]
    best_path = save_dir / "val_predictions" / "best.png"
    _render_topdown_keypoint_grid(
        model, val_ds, pick, best_path,
        title=f"Best-checkpoint val — AP {ap_dict['AP']:.4f} | PCK@0.1 {pck['pck_10']:.4f}",
        grid_cols=grid_cols, dpi=dpi,
    )

    if test_ds is not None and len(test_ds) > 0:
        test_path = save_dir / "test_predictions" / "best.png"
        test_pick = list(range(min(best_num_samples, len(test_ds))))
        try:
            _render_topdown_keypoint_grid(
                model, test_ds, test_pick, test_path,
                title=f"Best-checkpoint test — n={len(test_pick)}",
                grid_cols=grid_cols, dpi=dpi,
            )
        except Exception as e:
            logger.warning("test-set best grid skipped: %s", e)

    # ---- 3) Error analysis (compact). ----
    ea_dir = save_dir / "val_predictions" / "error_analysis"
    ea_dir.mkdir(parents=True, exist_ok=True)
    _render_topdown_keypoint_error_analysis(
        ea_dir=ea_dir, pred_xy=pred_xy, gt_xy=gt_xy, weight=weight,
        pred_score=pred_score, pixel_err=pixel_err, sample_idx=sample_idx,
        ref_size=ref_size, sigmas=sigmas, input_hw=(input_h, input_w),
        kp_names=getattr(inner, "data_config", {}).get("keypoint_names"),
        ds=val_ds, dpi=dpi,
    )

    # ---- 4) Sweeps that re-run the model. ----
    try:
        _chart_14_robustness_sweep(
            model=model, val_ds=val_ds, ea_dir=ea_dir,
            input_hw=(input_h, input_w), sigmas=sigmas, dpi=dpi,
        )
    except Exception as e:
        logger.warning("robustness_sweep skipped: %s", e, exc_info=True)
    try:
        _chart_20_bbox_padding_sweep(
            model=model, val_ds=val_ds, ea_dir=ea_dir,
            input_hw=(input_h, input_w), sigmas=sigmas, dpi=dpi,
        )
    except Exception as e:
        logger.warning("bbox_padding_sweep skipped: %s", e, exc_info=True)
        import traceback as _tb
        logger.warning(_tb.format_exc())

    # Real COCO-shape OKS-AP via pycocotools (12-stat dict). Skips silently
    # when pycocotools is missing.
    coco_eval: dict[str, float] = {}
    try:
        from core.p08_evaluation.keypoint_metrics import (
            compute_oks_ap_pycocotools,
        )
        # Collect per-person source-image bboxes (in source pixel coords) so
        # pycocotools can bucket persons into S / M / L by area. Without this
        # every crop has identical area and APm/ARm collapse to -1.
        bbox_xywh = _collect_source_bboxes(val_ds, sample_idx)
        coco_eval = compute_oks_ap_pycocotools(
            pred_xy=pred_xy, pred_scores=pred_score,
            gt_xy=gt_xy, weight=weight,
            image_ids=sample_idx, sigmas=sigmas,
            image_size=(input_h, input_w),
            bbox_xywh=bbox_xywh,
        )
    except Exception as e:
        logger.warning("pycocotools OKS-AP skipped: %s", e, exc_info=True)

    summary = {
        **pck, **ap_dict,
        "n_persons": int(n),
        "coco_eval": coco_eval,
    }
    (ea_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    md_lines = [
        "# Keypoint top-down error analysis",
        "",
        f"- N persons: **{n}**",
        f"- AP (mean OKS≥{{0.5..0.95}}): **{ap_dict['AP']:.4f}**",
        f"- AP50 / AP75: **{ap_dict['AP50']:.4f}** / **{ap_dict['AP75']:.4f}**",
        f"- OKS mean: **{ap_dict['OKS_mean']:.4f}**",
        f"- PCK@0.05 / 0.10 / 0.20: **{pck['pck_05']:.4f}** / "
        f"**{pck['pck_10']:.4f}** / **{pck['pck_20']:.4f}**",
        f"- Mean pixel error: **{pck['mean_pixel_err']:.2f} px**"
        f" (ref scale = sqrt(H²+W²) = {ref_size:.0f} px)",
        "",
        "Charts:",
        "- `01_overview.png` — headline metrics + per-joint err bar.",
        "- `07_per_joint_performance.png` — PCK@0.1 per joint sorted ascending.",
        "- `08_confusion_left_right.png` — symmetric-pair confusion heatmap.",
        "- `09_confidence_vs_error.png` — heatmap-peak score vs pixel error scatter.",
        "- `10_failure_mode_contribution.png` — per-joint × failure-mode fractions.",
        "- `12_hardest_crops.png` — top-12 worst predictions, GT vs Pred.",
        "- `13_failure_mode_examples/` — galleries: high_error, ghost, swapped_pair.",
        "- `14_robustness_sweep.png` — AP/PCK vs corruption (gaussian_blur,",
        "  jpeg, brightness) at 3 severities.",
        "- `20_bbox_padding_sweep.png` — AP/PCK vs bbox_padding ∈ {1.0..2.0}.",
    ]
    (ea_dir / "summary.md").write_text("\n".join(md_lines))
    logger.info("post-train (top-down keypoint): saved %s", ea_dir)


def _render_topdown_keypoint_error_analysis(
    *,
    ea_dir: Path,
    pred_xy: np.ndarray, gt_xy: np.ndarray, weight: np.ndarray,
    pred_score: np.ndarray, pixel_err: np.ndarray, sample_idx: np.ndarray,
    ref_size: float, sigmas: np.ndarray, input_hw: tuple[int, int],
    kp_names: list[str] | None, ds: Any, dpi: int,
) -> None:
    """Render charts 01, 07, 08, 12 to ``ea_dir``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = pred_xy.shape[1]
    if not kp_names:
        kp_names = [f"kp{i:02d}" for i in range(K)]

    vis = (weight > 0).astype(np.float32)

    # ----- 01 overview -----
    pck_thrs = [0.05, 0.1, 0.2]
    pck_vals = []
    for t in pck_thrs:
        thr = t * ref_size
        pck_vals.append(((pixel_err < thr) & (vis > 0)).sum() / max(vis.sum(), 1.0))

    n_vis_per_joint = np.maximum(vis.sum(axis=0), 1.0)
    err_per_joint = (pixel_err * vis).sum(axis=0) / n_vis_per_joint

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].bar(["PCK@0.05", "PCK@0.10", "PCK@0.20"], pck_vals,
              color=["#7e22ce", "#9333ea", "#a855f7"])
    for i, v in enumerate(pck_vals):
        ax[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax[0].set_ylim(0, 1.05)
    ax[0].set_ylabel("Recall")
    ax[0].set_title("Headline accuracy")
    order = np.argsort(err_per_joint)
    ax[1].barh([kp_names[i] for i in order], err_per_joint[order], color="#9333ea")
    ax[1].set_xlabel("Mean pixel error (px)")
    ax[1].set_title("Per-joint mean pixel error")
    fig.tight_layout()
    fig.savefig(ea_dir / "01_overview.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # ----- 07 per_joint_performance (PCK@0.1 per joint, sorted) -----
    thr10 = 0.1 * ref_size
    correct_per_joint = ((pixel_err < thr10) & (vis > 0)).sum(axis=0)
    pck10_per_joint = correct_per_joint / n_vis_per_joint
    order10 = np.argsort(pck10_per_joint)
    fig, ax = plt.subplots(figsize=(8, max(4, K * 0.3)))
    ax.barh([kp_names[i] for i in order10], pck10_per_joint[order10], color="#9333ea")
    for i, v in enumerate(pck10_per_joint[order10]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("PCK@0.1")
    ax.set_title("Per-joint PCK@0.1 (sorted ascending — worst at top)")
    fig.tight_layout()
    fig.savefig(ea_dir / "07_per_joint_performance.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # ----- 08 confusion_left_right -----
    # COCO 17-kpt L↔R index pairs. Skipped silently for non-COCO schemas.
    coco_lr_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
    pair_names = ["eyes", "ears", "shoulders", "elbows", "wrists", "hips", "knees", "ankles"]
    if K >= 17:
        thr = 0.1 * ref_size
        # cols: both_ok, swapped, L_only, R_only
        cnt = np.zeros((len(coco_lr_pairs), 4), dtype=np.float32)
        for pi, (li, ri) in enumerate(coco_lr_pairs):
            both_vis = (vis[:, li] > 0) & (vis[:, ri] > 0)
            if not both_vis.any():
                continue
            d_ll = np.linalg.norm(pred_xy[both_vis, li] - gt_xy[both_vis, li], axis=-1)
            d_rr = np.linalg.norm(pred_xy[both_vis, ri] - gt_xy[both_vis, ri], axis=-1)
            d_lr = np.linalg.norm(pred_xy[both_vis, li] - gt_xy[both_vis, ri], axis=-1)
            d_rl = np.linalg.norm(pred_xy[both_vis, ri] - gt_xy[both_vis, li], axis=-1)
            ll_ok = d_ll < thr
            rr_ok = d_rr < thr
            swap = (d_lr < thr) & (d_rl < thr) & ~ll_ok & ~rr_ok
            cnt[pi, 0] = (ll_ok & rr_ok).sum()
            cnt[pi, 1] = swap.sum()
            cnt[pi, 2] = (ll_ok & ~rr_ok).sum()
            cnt[pi, 3] = (~ll_ok & rr_ok).sum()
            total = both_vis.sum()
            if total > 0:
                cnt[pi] = cnt[pi] / total
        fig, ax = plt.subplots(figsize=(7, 3.6))
        im = ax.imshow(cnt, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["both\nok", "L↔R\nswapped", "L only", "R only"])
        ax.set_yticks(range(len(coco_lr_pairs)))
        ax.set_yticklabels(pair_names)
        for i in range(cnt.shape[0]):
            for j in range(cnt.shape[1]):
                ax.text(j, i, f"{cnt[i, j]:.2f}",
                        ha="center", va="center",
                        color="white" if cnt[i, j] < 0.6 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.04)
        ax.set_title("L↔R confusion (PCK@0.1 gate)")
        fig.tight_layout()
        fig.savefig(ea_dir / "08_confusion_left_right.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    # ----- 12 hardest_crops — top-12 worst by mean per-image pixel err -----
    n_vis_per_img = np.maximum(vis.sum(axis=1), 1.0)
    img_err = (pixel_err * vis).sum(axis=1) / n_vis_per_img
    hardest = np.argsort(-img_err)[:12].tolist()
    pick_idx = [int(sample_idx[i]) for i in hardest]
    from core.p10_inference.supervision_bridge import _draw_keypoints_panel
    from utils.viz import classification_banner as _banner

    inner = getattr(ds, "dataset", ds)
    style = VizStyle()
    skeleton_edges = _resolve_skeleton_edges(inner)

    def _annotated_crop(h_idx: int, caption: str) -> np.ndarray | None:
        real_idx = int(sample_idx[h_idx])
        try:
            item = ds[real_idx]
        except Exception:
            return None
        crop = _denormalize_pixel_values(item["pixel_values"].cpu().numpy())
        gt_kp = np.concatenate(
            [gt_xy[h_idx], (weight[h_idx] > 0).astype(np.float32)[:, None] * 2.0],
            axis=1,
        )
        pred_kp = np.concatenate([pred_xy[h_idx], pred_score[h_idx, :, None]], axis=1)
        panel = _draw_keypoints_panel(
            crop, gt_kp, style.gt_color_rgb, style, skeleton_edges=skeleton_edges,
        )
        pred_style = VizStyle(kpt_visibility_threshold=0.3)
        panel = _draw_keypoints_panel(
            panel, pred_kp, style.pred_color_rgb, pred_style,
            skeleton_edges=skeleton_edges,
        )
        panel = _banner(
            panel, caption, style=style, position="top",
            bg_color_rgb=(0, 0, 0), text_color_rgb=(255, 255, 255),
        )
        return panel

    if pick_idx:
        panels = []
        for h in hardest:
            real_idx = int(sample_idx[h])
            ann = _annotated_crop(h, f"#{real_idx}  err={img_err[h]:.1f}px")
            if ann is not None:
                panels.append(ann)
        if panels:
            save_image_grid(
                panels, ea_dir / "12_hardest_crops.png",
                cols=min(4, len(panels)),
                header="Hardest crops — top-12 by mean per-image pixel error",
            )

    # ----- 09 confidence_vs_error — per-joint score vs pixel-error scatter -----
    # Calibration check: a well-calibrated heatmap head produces high scores
    # at correct predictions and low scores at wrong ones.
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    cmap = plt.get_cmap("tab20", K)
    for k in range(K):
        m = vis[:, k] > 0
        if not m.any():
            continue
        ax.scatter(
            pred_score[m, k], pixel_err[m, k],
            s=8, alpha=0.4, color=cmap(k), label=kp_names[k] if K <= 17 else None,
        )
    ax.axhline(
        0.1 * ref_size, ls="--", color="0.4", lw=0.8,
        label=f"PCK@0.1 = {0.1 * ref_size:.0f}px",
    )
    ax.set_xlabel("Heatmap-peak score (pred)")
    ax.set_ylabel("Pixel error vs GT (px)")
    ax.set_title("Per-joint confidence vs error")
    if K <= 17:
        ax.legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(ea_dir / "09_confidence_vs_error.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # ----- 10 failure_mode_contribution — per-joint × mode heatmap -----
    # Modes (counts per joint, fractions of all GT-visible joints):
    #   ok          : visible joint, pred error within PCK@0.1
    #   far_off     : visible joint, pred error >= 2x PCK@0.1 threshold
    #   near_miss   : visible joint, PCK@0.1 < err < 2x PCK@0.1 (recoverable)
    #   low_score   : visible joint, pred score < 0.3 (likely missed)
    #   ghost       : non-visible joint (GT v=0), pred score >= 0.5 (false alarm)
    thr10 = 0.1 * ref_size
    modes = ["ok", "near_miss", "far_off", "low_score", "ghost"]
    counts = np.zeros((K, len(modes)), dtype=np.float32)
    for k in range(K):
        v = vis[:, k] > 0
        nv = ~v
        err_k = pixel_err[:, k]
        sc_k = pred_score[:, k]
        denom = max(v.sum(), 1)
        counts[k, 0] = ((err_k < thr10) & v).sum() / denom
        counts[k, 1] = ((err_k >= thr10) & (err_k < 2 * thr10) & v).sum() / denom
        counts[k, 2] = ((err_k >= 2 * thr10) & v).sum() / denom
        counts[k, 3] = ((sc_k < 0.3) & v).sum() / denom
        counts[k, 4] = ((sc_k >= 0.5) & nv).sum() / max(nv.sum(), 1)
    fig, ax = plt.subplots(figsize=(7, max(4, K * 0.3)))
    im = ax.imshow(counts, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(modes, rotation=20)
    ax.set_yticks(range(K))
    ax.set_yticklabels(kp_names)
    for i in range(K):
        for j in range(len(modes)):
            v = counts[i, j]
            if v > 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.55 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title("Failure-mode contribution per joint (fractions)")
    fig.tight_layout()
    fig.savefig(ea_dir / "10_failure_mode_contribution.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # ----- 13 failure_mode_examples/ — top-N crops per failure category -----
    # Three galleries that surface systematic issues, organised so a reviewer
    # can scroll through one folder per failure type:
    #   high_error/kp_<k>/   joints with pixel error in the top-5% per joint
    #   ghost/kp_<k>/        non-visible GT joints that pred fires hot on
    #   swapped_pair/<a>__<b>/  L↔R swaps (PCK@0.1 satisfied if labels flipped)
    gallery_dir = ea_dir / "13_failure_mode_examples"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    n_gallery_per_class = 6

    # high_error: per-joint hardest non-correct samples
    he_dir = gallery_dir / "high_error"
    for k in range(K):
        v = vis[:, k] > 0
        if not v.any():
            continue
        err_k = np.where(v, pixel_err[:, k], -np.inf)
        idx_sorted = np.argsort(-err_k)[:n_gallery_per_class]
        idx_sorted = [int(i) for i in idx_sorted if err_k[i] > thr10]
        if not idx_sorted:
            continue
        kdir = he_dir / f"kp_{k:02d}_{kp_names[k]}"
        kdir.mkdir(parents=True, exist_ok=True)
        panels = []
        for i in idx_sorted:
            ann = _annotated_crop(i, f"{kp_names[k]} err={pixel_err[i, k]:.0f}px")
            if ann is not None:
                panels.append(ann)
        if panels:
            save_image_grid(
                panels, kdir / "gallery.png",
                cols=min(3, len(panels)),
                header=f"high_error — {kp_names[k]} (top {len(panels)})",
            )

    # ghost: GT v=0 but pred score high
    gh_dir = gallery_dir / "ghost"
    for k in range(K):
        nv = ~(vis[:, k] > 0)
        if not nv.any():
            continue
        sc_k = np.where(nv, pred_score[:, k], -np.inf)
        idx_sorted = np.argsort(-sc_k)[:n_gallery_per_class]
        idx_sorted = [int(i) for i in idx_sorted if sc_k[i] >= 0.5]
        if not idx_sorted:
            continue
        kdir = gh_dir / f"kp_{k:02d}_{kp_names[k]}"
        kdir.mkdir(parents=True, exist_ok=True)
        panels = []
        for i in idx_sorted:
            ann = _annotated_crop(i, f"ghost {kp_names[k]} score={pred_score[i, k]:.2f}")
            if ann is not None:
                panels.append(ann)
        if panels:
            save_image_grid(
                panels, kdir / "gallery.png",
                cols=min(3, len(panels)),
                header=f"ghost — {kp_names[k]} (top {len(panels)})",
            )

    # swapped_pair: COCO 17-kpt L↔R pairs only
    if K >= 17:
        sp_dir = gallery_dir / "swapped_pair"
        coco_lr_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
        for li, ri in coco_lr_pairs:
            both = (vis[:, li] > 0) & (vis[:, ri] > 0)
            if not both.any():
                continue
            d_lr = np.linalg.norm(pred_xy[:, li] - gt_xy[:, ri], axis=-1)
            d_rl = np.linalg.norm(pred_xy[:, ri] - gt_xy[:, li], axis=-1)
            d_ll = np.linalg.norm(pred_xy[:, li] - gt_xy[:, li], axis=-1)
            d_rr = np.linalg.norm(pred_xy[:, ri] - gt_xy[:, ri], axis=-1)
            swap_mask = both & (d_lr < thr10) & (d_rl < thr10) & ((d_ll >= thr10) | (d_rr >= thr10))
            if not swap_mask.any():
                continue
            sample_indices = np.where(swap_mask)[0][:n_gallery_per_class]
            pdir = sp_dir / f"{kp_names[li]}__{kp_names[ri]}"
            pdir.mkdir(parents=True, exist_ok=True)
            panels = []
            for i in sample_indices:
                ann = _annotated_crop(int(i), f"swap {kp_names[li]}↔{kp_names[ri]}")
                if ann is not None:
                    panels.append(ann)
            if panels:
                save_image_grid(
                    panels, pdir / "gallery.png",
                    cols=min(3, len(panels)),
                    header=f"L↔R swap — {kp_names[li]}↔{kp_names[ri]}",
                )


def _topdown_eval_pass(
    model: Any, ds: Any, *, input_hw: tuple[int, int], sigmas: np.ndarray,
    pixel_transform=None, batch_size: int = 32,
) -> dict[str, float]:
    """Run a single eval pass over `ds` and return AP/PCK metrics.

    `pixel_transform`, if given, is called on the already-normalised
    ``pixel_values`` tensor (shape B,3,H,W) per batch and must return a
    same-shape tensor — used by the robustness sweep to inject
    corruptions in input space.
    """
    import torch

    from core.p08_evaluation.keypoint_metrics import (
        compute_oks,
        compute_oks_ap,
        compute_pck,
        decode_heatmaps_to_xy,
    )

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    pred_xy_buf, gt_xy_buf, weight_buf, pixel_err_buf = [], [], [], []
    n = len(ds)
    input_h, input_w = input_hw
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            chunk = list(range(start, min(start + batch_size, n)))
            items = [ds[i] for i in chunk]
            px = torch.stack([it["pixel_values"] for it in items]).to(device)
            if pixel_transform is not None:
                px = pixel_transform(px)
            tg = torch.stack([it["target_heatmap"] for it in items]).numpy()
            tw = torch.stack([it["target_weight"] for it in items]).numpy()
            out = model(pixel_values=px)
            ph = out["heatmaps"] if isinstance(out, dict) else out.heatmaps
            ph = ph.detach().float().cpu().numpy()
            stride = int(input_h // ph.shape[2])
            pxy, _ = decode_heatmaps_to_xy(ph, stride=stride)
            gxy, _ = decode_heatmaps_to_xy(tg, stride=stride)
            err = np.linalg.norm(pxy - gxy, axis=-1)
            pred_xy_buf.append(pxy)
            gt_xy_buf.append(gxy)
            weight_buf.append(tw)
            pixel_err_buf.append(err)

    if was_training:
        model.train()
    pxy = np.concatenate(pred_xy_buf, axis=0)
    gxy = np.concatenate(gt_xy_buf, axis=0)
    w = np.concatenate(weight_buf, axis=0)
    err = np.concatenate(pixel_err_buf, axis=0)
    ref_size = float(np.sqrt(input_h * input_h + input_w * input_w))
    pck = compute_pck(pxy, gxy, w, ref_size=ref_size)
    oks = compute_oks(pxy, gxy, w, sigmas=sigmas, area=float(input_h * input_w))
    ap = compute_oks_ap(oks)
    return {**pck, **ap}


def _chart_14_robustness_sweep(
    model: Any, val_ds: Any, ea_dir: Path,
    *, input_hw: tuple[int, int], sigmas: np.ndarray, dpi: int,
) -> None:
    """AP/PCK vs corruption (gaussian_blur, jpeg, brightness) at 3 severities."""
    import json

    import torch
    import torch.nn.functional as F

    _IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    device = next(model.parameters()).device
    mean_d = _IMAGENET_MEAN.to(device)
    std_d = _IMAGENET_STD.to(device)

    def _denorm_renorm(px: torch.Tensor, transform_rgb01) -> torch.Tensor:
        rgb = px * std_d + mean_d                # → [0, 1]
        rgb = rgb.clamp(0, 1)
        rgb = transform_rgb01(rgb)
        rgb = rgb.clamp(0, 1)
        return (rgb - mean_d) / std_d

    def _gaussian_kernel(sigma: float) -> torch.Tensor:
        radius = max(1, int(round(3 * sigma)))
        x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
        k1d = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        k1d = k1d / k1d.sum()
        k2d = (k1d[:, None] * k1d[None, :])
        return k2d.view(1, 1, k2d.shape[0], k2d.shape[1])

    def _gaussian_blur(sigma: float):
        k = _gaussian_kernel(sigma)
        kk = k.expand(3, 1, k.shape[2], k.shape[3])
        pad = kk.shape[2] // 2
        def fn(rgb: torch.Tensor) -> torch.Tensor:
            return F.conv2d(rgb, kk, padding=pad, groups=3)
        return fn

    def _brightness(delta: float):
        def fn(rgb: torch.Tensor) -> torch.Tensor:
            return rgb + delta
        return fn

    def _jpeg(quality: int):
        # Per-image cv2 JPEG round-trip; slow but accurate.
        import cv2 as _cv2
        def fn(rgb: torch.Tensor) -> torch.Tensor:
            out = []
            for i in range(rgb.shape[0]):
                arr = (rgb[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                bgr = _cv2.cvtColor(arr, _cv2.COLOR_RGB2BGR)
                ok, buf = _cv2.imencode(".jpg", bgr, [_cv2.IMWRITE_JPEG_QUALITY, quality])
                if not ok:
                    out.append(rgb[i])
                    continue
                dec = _cv2.imdecode(buf, _cv2.IMREAD_COLOR)
                rgb_dec = _cv2.cvtColor(dec, _cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                out.append(torch.from_numpy(rgb_dec).permute(2, 0, 1).to(rgb.device))
            return torch.stack(out, dim=0)
        return fn

    sweeps: dict[str, list[tuple[str, Any]]] = {
        "gaussian_blur": [("σ=1", _gaussian_blur(1.0)),
                          ("σ=2", _gaussian_blur(2.0)),
                          ("σ=4", _gaussian_blur(4.0))],
        "brightness":    [("Δ=-0.4", _brightness(-0.4)),
                          ("Δ=-0.2", _brightness(-0.2)),
                          ("Δ=+0.2", _brightness(+0.2))],
        "jpeg":          [("q=50", _jpeg(50)),
                          ("q=30", _jpeg(30)),
                          ("q=10", _jpeg(10))],
    }

    # Baseline (no corruption).
    baseline = _topdown_eval_pass(model, val_ds, input_hw=input_hw, sigmas=sigmas)

    results: dict[str, dict[str, dict]] = {"_baseline": baseline}
    for corruption, severities in sweeps.items():
        results[corruption] = {}
        for label, fn in severities:
            results[corruption][label] = _topdown_eval_pass(
                model, val_ds, input_hw=input_hw, sigmas=sigmas,
                pixel_transform=lambda px, fn=fn: _denorm_renorm(px, fn),
            )

    # Plot: AP across corruptions/severities.
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.25
    corr_names = list(sweeps.keys())
    severity_labels = ["mild", "moderate", "severe"]
    x = np.arange(len(corr_names))
    for i, sev in enumerate(severity_labels):
        ap_vals = [results[c][list(sweeps[c])[i][0]]["AP"] for c in corr_names]
        ax.bar(x + (i - 1) * width, ap_vals, width=width, label=sev,
               color=["#a78bfa", "#7c3aed", "#4c1d95"][i])
    ax.axhline(baseline["AP"], color="0.4", ls="--", lw=0.8,
               label=f"baseline AP={baseline['AP']:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(corr_names)
    ax.set_ylabel("OKS-AP")
    ax.set_title("Robustness sweep — AP under input corruption")
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(ea_dir / "14_robustness_sweep.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    (ea_dir / "14_robustness_sweep.json").write_text(json.dumps(results, indent=2))


def _chart_20_bbox_padding_sweep(
    model: Any, val_ds: Any, ea_dir: Path,
    *, input_hw: tuple[int, int], sigmas: np.ndarray, dpi: int,
) -> None:
    """AP/PCK vs bbox_padding ∈ {1.0, 1.1, 1.25, 1.5, 2.0}.

    Critical for deployment: the production detector's bbox is rarely as
    tight as the GT used in training, and the model's PCK degrades fast
    when the crop becomes too loose or tight.
    """
    import json

    inner = getattr(val_ds, "dataset", val_ds)
    if type(inner).__name__ != "KeypointTopDownDataset":
        return  # sweep only meaningful for this dataset

    paddings = [1.0, 1.1, 1.25, 1.5, 2.0]
    results: dict[float, dict[str, float]] = {}

    # Resolve the absolute dataset root once so re-instantiated datasets
    # don't depend on an inherited base_dir attribute (BaseDataset doesn't
    # keep one). `inner.img_dir` is `<root>/<split>[/images]` — peel back.
    abs_root = inner.img_dir
    if abs_root.name == "images":
        abs_root = abs_root.parent.parent
    else:
        abs_root = abs_root.parent
    abs_data_config = dict(inner.data_config)
    abs_data_config["path"] = str(abs_root)

    for pad in paddings:
        # Re-instantiate the underlying dataset with the new padding.
        from core.p05_data.keypoint_dataset import KeypointTopDownDataset
        new_inner = KeypointTopDownDataset(
            data_config=abs_data_config, split=inner.split,
            processor=inner.processor, bbox_padding=pad,
            heatmap_sigma=inner.heatmap_sigma, is_train=False,
            base_dir=None,
        )
        # Wrap with the same Subset (if any) by index reuse so we evaluate the
        # same person crops, just at different paddings.
        if hasattr(val_ds, "indices"):
            from torch.utils.data import Subset
            ds_at_pad = Subset(new_inner, list(val_ds.indices))
        else:
            ds_at_pad = new_inner
        results[pad] = _topdown_eval_pass(
            model, ds_at_pad, input_hw=input_hw, sigmas=sigmas,
        )

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pads = list(results.keys())
    ax.plot(pads, [results[p]["AP"] for p in pads], marker="o", lw=2,
            color="#7c3aed", label="OKS-AP")
    ax.plot(pads, [results[p]["pck_10"] for p in pads], marker="s", lw=1.5,
            color="#3b82f6", label="PCK@0.1")
    ax.plot(pads, [results[p]["pck_20"] for p in pads], marker="^", lw=1.5,
            color="#10b981", label="PCK@0.2")
    ax.set_xlabel("bbox_padding")
    ax.set_ylabel("metric")
    ax.set_xticks(pads)
    ax.set_title("BBox-padding sweep — robustness to detector-bbox tightness")
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ea_dir / "20_bbox_padding_sweep.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    (ea_dir / "20_bbox_padding_sweep.json").write_text(
        json.dumps({str(p): r for p, r in results.items()}, indent=2)
    )


def _render_topdown_keypoint_grid(
    model: Any,
    ds: Any,
    indices: list[int],
    out_path: Path,
    *,
    title: str,
    grid_cols: int = 4,
    dpi: int = 150,
    pred_score_threshold: float = 0.3,
) -> None:
    """Render a GT+Pred skeleton grid for a top-down keypoint dataset.

    For each sampled index we pull one (pixel_values, target_heatmap,
    target_weight) triple from the dataset, run the model on the stacked
    pixel_values, decode peaks for both pred + GT heatmaps, and draw both
    skeletons on the denormalized crop (GT in purple, pred in green).
    """
    import torch

    from core.p08_evaluation.keypoint_metrics import decode_heatmaps_to_xy

    inner = getattr(ds, "dataset", ds)
    index_map = (
        (lambda i: ds.indices[i]) if hasattr(ds, "indices") else (lambda i: i)
    )

    samples = []
    for idx in indices:
        try:
            real_idx = index_map(indices.index(idx))  # subset → underlying idx
        except (TypeError, ValueError):
            real_idx = idx
        try:
            item = ds[idx]  # respects Subset transparently
        except Exception:
            continue
        samples.append((real_idx, item))

    if not samples:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return

    device = next(model.parameters()).device
    pixel_batch = torch.stack([s[1]["pixel_values"] for s in samples]).to(device)
    target_hms = np.stack([s[1]["target_heatmap"].cpu().numpy() for s in samples])
    target_ws = np.stack([s[1]["target_weight"].cpu().numpy() for s in samples])

    with torch.inference_mode():
        out = model(pixel_values=pixel_batch)
    pred_hms_t = out["heatmaps"] if isinstance(out, dict) and "heatmaps" in out \
        else getattr(out, "heatmaps", None)
    if pred_hms_t is None:
        pred_hms_t = getattr(out, "logits", None)
    pred_hms = pred_hms_t.detach().float().cpu().numpy()

    stride = int(inner.input_hw[0] // pred_hms.shape[2])
    pred_xy, pred_scores = decode_heatmaps_to_xy(pred_hms, stride=stride)
    gt_xy, _ = decode_heatmaps_to_xy(target_hms, stride=stride)

    style = VizStyle()
    skeleton_edges = _resolve_skeleton_edges(inner)
    panels: list[np.ndarray] = []
    for i, (real_idx, _) in enumerate(samples):
        crop = _denormalize_pixel_values(samples[i][1]["pixel_values"].cpu().numpy())
        # GT in purple — gate by visibility weight
        gt_kp = np.concatenate(
            [gt_xy[i], (target_ws[i] > 0).astype(np.float32)[:, None] * 2.0],
            axis=1,
        )
        pred_kp = np.concatenate(
            [pred_xy[i], pred_scores[i, :, None]], axis=1,
        )
        from core.p10_inference.supervision_bridge import _draw_keypoints_panel
        panel = _draw_keypoints_panel(
            crop, gt_kp, style.gt_color_rgb, style, skeleton_edges=skeleton_edges,
        )
        # Override threshold for pred dots: model heatmap-peak scores aren't
        # in the same range as visibility {0,1,2}; use a configured cutoff.
        pred_style = VizStyle(kpt_visibility_threshold=pred_score_threshold)
        panel = _draw_keypoints_panel(
            panel, pred_kp, style.pred_color_rgb, pred_style,
            skeleton_edges=skeleton_edges,
        )
        # Caption: idx
        from utils.viz import classification_banner as _banner
        panel = _banner(panel, f"#{real_idx}", style=style, position="top",
                       bg_color_rgb=(0, 0, 0), text_color_rgb=(255, 255, 255))
        panels.append(panel)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image_grid(
        panels, out_path, cols=min(grid_cols, len(panels)), header=title,
    )


def _draw_gt_boxes(
    image: np.ndarray,
    targets: np.ndarray,
    class_names: dict,
    thickness: int = 2,
    text_scale: float = 0.5,
) -> np.ndarray:
    """BGR-in / BGR-out wrapper around ``utils.viz.annotate_detections``.

    Targets are YOLO normalized cxcywh in rows ``[cls, cx, cy, w, h]``.
    Mirrors the helper in ``callbacks.py`` so HF-backend viz output is
    byte-identical to pytorch-backend output.
    """
    if len(targets) == 0:
        return image.copy()
    h, w = image.shape[:2]
    xyxy, class_ids = yolo_targets_to_xyxy(targets, w, h)
    dets = sv.Detections(xyxy=xyxy.astype(np.float64), class_id=class_ids)
    style = VizStyle(box_thickness=thickness, label_text_scale=text_scale)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    annotated_rgb = annotate_detections(image_rgb, dets, class_names=class_names, style=style)
    return cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)


def _render_gt_panel(
    image_bgr: np.ndarray,
    target: Any,
    task: str,
    class_names: dict[int, str],
    thickness: int = 2,
    text_scale: float = 0.4,
    skeleton_edges: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Task-aware single-panel GT rendering on a BGR image.

    Dispatches to the correct primitive per task and returns a BGR uint8
    image. Used by ``HFDataLabelGridCallback`` and ``HFAugLabelGridCallback``
    so the ``02_data_labels_*.png`` / ``03_aug_labels_*.png`` grids reflect
    each task's actual GT structure (boxes / mask / class label / keypoints).

    Target contract per task:
    - detection:     YOLO-normalized targets ``(N, 5)`` = ``[cls, cx, cy, w, h]``
    - segmentation:  ``(H, W)`` uint8 mask (pixel values = class id)
    - classification:scalar ``int`` class id
    - keypoint:      ``(K, 2)`` or ``(K, 3)`` keypoint array
    """
    if task == "detection":
        targets_np = (
            target if isinstance(target, np.ndarray)
            else np.zeros((0, 5), dtype=np.float32)
        )
        return _draw_gt_boxes(image_bgr, targets_np, class_names, thickness, text_scale)

    from core.p10_inference.supervision_bridge import _draw_keypoints_panel, _mask_overlay

    style = VizStyle(box_thickness=thickness, label_text_scale=text_scale)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if task == "segmentation":
        mask = target if isinstance(target, np.ndarray) else np.zeros(image_bgr.shape[:2], np.uint8)
        annotated_rgb = _mask_overlay(image_rgb, mask, style.gt_color_rgb, style.mask_alpha)
    elif task == "classification":
        try:
            cid = int(target)
        except (TypeError, ValueError):
            cid = -1
        label = class_names.get(cid, str(cid)) if cid >= 0 else "?"
        annotated_rgb = classification_banner(
            image_rgb, f"Label: {label}", style=style, position="top",
            bg_color_rgb=style.gt_color_rgb, text_color_rgb=(255, 255, 255),
        )
    elif task == "keypoint":
        kpts = target if isinstance(target, np.ndarray) else None
        # KeypointDataset (multi-instance YOLO-pose) emits (N, K, 3); top-down
        # emits (1, K, 3). For now we render the first instance per sample —
        # top-down has exactly one anyway, and multi-instance datasets get a
        # representative skeleton. TODO: per-instance overlay loop.
        if isinstance(kpts, np.ndarray) and kpts.ndim == 3 and kpts.shape[0] >= 1:
            kpts = kpts[0]
        annotated_rgb = _draw_keypoints_panel(
            image_rgb, kpts, style.gt_color_rgb, style,
            skeleton_edges=skeleton_edges,
        )
    else:
        annotated_rgb = image_rgb

    return cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)


def _extract_raw_target(ds, idx: int, task: str):
    """Pull the per-task GT target for sample ``idx`` from a p05 Dataset.

    Returns the target in the shape expected by :func:`_render_gt_panel`.
    """
    if task == "detection":
        # YOLOXDataset: load YOLO normalized (N, 5) from the label .txt.
        labels = ds._load_label(ds.img_paths[idx])
        if labels is None or len(labels) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        return labels
    item = ds.get_raw_item(idx)
    if task == "keypoint":
        # `targets` here holds bboxes (compat with detection-style schema);
        # the GT skeleton lives under `keypoints`. Multi-instance YOLO-pose
        # `KeypointDataset` returns normalized [0,1] coords — denormalize to
        # pixel space for the renderer (which expects pixel coords).
        # Top-down `KeypointTopDownDataset` already returns pixel coords; we
        # detect the convention by max-coord magnitude rather than dataset
        # type so both work without coupling.
        kpts = item.get("keypoints")
        if isinstance(kpts, np.ndarray) and kpts.size and kpts[..., :2].max() <= 1.5:
            img = item.get("image")
            if img is not None:
                ih, iw = img.shape[:2]
                kpts = kpts.copy()
                kpts[..., 0] *= iw
                kpts[..., 1] *= ih
        return kpts
    # Segmentation / classification: get_raw_item returns the target directly.
    return item.get("targets")


def _save_image_grid(
    annotated: list[np.ndarray],
    grid_cols: int,
    title: str,
    out_path,
    dpi: int,
) -> None:
    """BGR-in wrapper around ``utils.viz.save_image_grid``."""
    if not annotated:
        return
    rgb_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in annotated]
    save_image_grid(
        rgb_imgs, out_path,
        cols=min(grid_cols, len(rgb_imgs)),
        header=title,
    )


def _build_class_names(data_config: dict) -> dict[int, str]:
    return {int(k): str(v) for k, v in data_config.get("names", {}).items()}


def _build_task_transforms(
    task: str,
    is_train: bool,
    aug_config: dict,
    input_size: tuple[int, int],
    mean: list | None,
    std: list | None,
):
    """Return a task-aware transform callable compatible with the p05 Dataset.

    - detection: :class:`DetectionTransform` (via core.p05_data.transforms.build_transforms)
    - classification: :func:`build_classification_transforms`
    - segmentation: :func:`build_segmentation_transforms`
    - keypoint: :func:`build_keypoint_transforms`

    Returns ``None`` if the task has no dedicated builder (lets the dataset
    fall back to its internal default).
    """
    if task == "detection":
        return build_transforms(
            config=aug_config, is_train=is_train, input_size=input_size,
            mean=mean, std=std,
        )
    if task == "classification":
        from core.p05_data.classification_dataset import build_classification_transforms
        return build_classification_transforms(
            is_train=is_train, input_size=input_size, mean=mean, std=std,
        )
    if task == "segmentation":
        try:
            from core.p05_data.segmentation_dataset import build_segmentation_transforms
            return build_segmentation_transforms(
                is_train=is_train, input_size=input_size, mean=mean, std=std,
            )
        except Exception:
            return None
    if task == "keypoint":
        try:
            from core.p05_data.keypoint_dataset import build_keypoint_transforms
            return build_keypoint_transforms(
                is_train=is_train, input_size=input_size, mean=mean, std=std,
            )
        except Exception:
            return None
    return None


def _extract_target_for_panel(targets_tensor, task: str):
    """Coerce a post-transform target into the shape :func:`_render_gt_panel` expects."""
    if task == "detection":
        if hasattr(targets_tensor, "numpy"):
            arr = targets_tensor.numpy()
        else:
            arr = np.asarray(targets_tensor)
        return arr if len(arr) > 0 else np.zeros((0, 5), dtype=np.float32)
    if task == "segmentation":
        if hasattr(targets_tensor, "numpy"):
            return targets_tensor.numpy().astype(np.uint8)
        return np.asarray(targets_tensor, dtype=np.uint8)
    if task == "classification":
        if hasattr(targets_tensor, "item"):
            return int(targets_tensor.item())
        return int(targets_tensor)
    # keypoint
    if hasattr(targets_tensor, "numpy"):
        return targets_tensor.numpy()
    if isinstance(targets_tensor, dict):
        # KeypointDataset may return {"keypoints": tensor, ...}
        kpts = targets_tensor.get("keypoints")
        if hasattr(kpts, "numpy"):
            return kpts.numpy()
        return np.asarray(kpts) if kpts is not None else None
    return np.asarray(targets_tensor)


def _tensor_to_denorm_bgr(
    tensor,
    mean: np.ndarray,
    std: np.ndarray,
    normalize_applied: bool,
) -> np.ndarray:
    """Convert a post-transform CHW float tensor → HWC uint8 BGR.

    If ``normalize_applied`` the tensor is denormalized via ``* std + mean``
    before clamp+scale. Otherwise it is already in ``[0, 1]``.
    """
    arr = tensor.detach().cpu().float().numpy().transpose(1, 2, 0)
    if normalize_applied:
        arr = np.clip(arr * std + mean, 0, 1)
    else:
        arr = np.clip(arr, 0, 1)
    return (arr[:, :, ::-1] * 255).astype(np.uint8)


class HFDatasetStatsCallback(TrainerCallback):
    """Emits `data_preview/00_dataset_info.{md,json}` + `01_dataset_stats.{png,json}`.

    Fires once at train-begin. Takes all inputs at init — doesn't need
    model/dataloader/trainer access.
    """

    def __init__(
        self,
        save_dir: str,
        data_config: dict,
        base_dir: str,
        splits: list[str],
        subsets: dict[str, list[int] | None] | None = None,
        dpi: int = 120,
        training_config: dict | None = None,
        training_config_path: str | None = None,
        data_config_path: str | None = None,
        feature_name: str | None = None,
        full_sizes: dict[str, int] | None = None,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.data_config = data_config
        self.base_dir = base_dir or ""
        self.splits = splits
        self.subsets = subsets or {s: None for s in splits}
        self.dpi = dpi
        self.training_config = training_config
        self.training_config_path = training_config_path
        self.data_config_path = data_config_path
        self.feature_name = feature_name
        self.full_sizes = full_sizes or {}

    def on_train_begin(self, args, state, control, **kwargs):
        from core.p05_data.run_viz import (
            _load_cached_stats,
            generate_dataset_stats,
            write_dataset_info,
        )

        out_dir = self.save_dir / "data_preview"
        class_names = _build_class_names(self.data_config)

        try:
            split_sizes = {
                s: (len(idxs) if idxs is not None else int(self.full_sizes.get(s, 0)))
                for s, idxs in self.subsets.items()
            }
            full_sizes = (
                {s: int(v) for s, v in self.full_sizes.items()} if self.full_sizes else None
            )
            write_dataset_info(
                out_dir,
                feature_name=self.feature_name,
                data_config_path=self.data_config_path,
                training_config_path=self.training_config_path,
                data_cfg=self.data_config,
                training_cfg=self.training_config,
                class_names=class_names,
                split_sizes=split_sizes,
                full_sizes=full_sizes,
            )
            if full_sizes and any(
                idxs is not None and len(idxs) < int(full_sizes.get(s, 0))
                for s, idxs in self.subsets.items()
            ):
                logger.warning(
                    "HFDatasetStatsCallback: data.subset.* active — dataset stats reflect subset, "
                    "not full splits. full=%s used=%s",
                    full_sizes, split_sizes,
                )
        except Exception as e:  # pragma: no cover
            logger.warning("HFDatasetStatsCallback: write_dataset_info failed — %s", e)

        if _load_cached_stats(out_dir):
            logger.info("HFDatasetStatsCallback: cache hit — skipping recompute (%s)", out_dir)
            return control

        try:
            # Compute subset_active / subset_pct from full vs used split sizes.
            subset_active = False
            subset_pct: float | None = None
            if self.full_sizes:
                ratios = []
                for s, idxs in self.subsets.items():
                    full_n = int(self.full_sizes.get(s, 0))
                    if full_n <= 0:
                        continue
                    used_n = len(idxs) if idxs is not None else full_n
                    if used_n < full_n:
                        subset_active = True
                    ratios.append(used_n / full_n)
                if subset_active and ratios:
                    subset_pct = round(float(np.mean(ratios)) * 100)
            generate_dataset_stats(
                self.data_config, self.base_dir, class_names,
                self.splits, out_dir, self.dpi,
                subset_indices=self.subsets,
                subset_active=subset_active,
                subset_pct=subset_pct,
            )
        except Exception as e:  # pragma: no cover
            logger.warning("HFDatasetStatsCallback failed: %s", e)
        return control


class HFDataLabelGridCallback(TrainerCallback):
    """Emits `data_preview/02_data_labels_<split>.png` once at training start.

    Task-aware: dispatches to the right p05 Dataset via
    :func:`build_dataset_for_viz` and renders GT per task via
    :func:`_render_gt_panel` — bbox overlays for detection, mask overlays for
    segmentation, class banners for classification, keypoint dots for
    keypoint.
    """

    def __init__(
        self,
        save_dir: str,
        splits: list[str],
        data_config: dict,
        base_dir: str,
        task: str = "detection",
        subsets: dict[str, list[int] | None] | None = None,
        num_samples: int = 16,
        grid_cols: int = 4,
        thickness: int = 2,
        text_scale: float = 0.4,
        dpi: int = 120,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.splits = splits
        self.data_config = data_config
        self.base_dir = base_dir or ""
        self.task = task
        self.subsets = subsets or {s: None for s in splits}
        self.num_samples = num_samples
        self.grid_cols = grid_cols
        self.thickness = thickness
        self.text_scale = text_scale
        self.dpi = dpi

    def on_train_begin(self, args, state, control, **kwargs):
        class_names = _build_class_names(self.data_config)
        skeleton_edges = (
            _resolve_skeleton_edges(SimpleNamespace(data_config=self.data_config))
            if self.task == "keypoint" else None
        )
        for split in self.splits:
            try:
                ds = build_dataset_for_viz(
                    self.task, split, self.data_config, self.base_dir, transforms=None,
                )
            except Exception as e:
                logger.info("HFDataLabelGridCallback: skip split %s — %s", split, e)
                continue

            subset = self.subsets.get(split)
            pool = list(range(len(ds))) if subset is None else list(subset)
            n = min(self.num_samples, len(pool))
            if n == 0:
                continue
            indices = sorted(random.sample(pool, n))

            annotated: list[np.ndarray] = []
            for idx in indices:
                try:
                    item = ds.get_raw_item(idx)
                    target = _extract_raw_target(ds, idx, self.task)
                except Exception as e:
                    logger.warning("HFDataLabelGridCallback: failed idx %d — %s", idx, e)
                    continue
                annotated.append(_render_gt_panel(
                    item["image"], target, self.task, class_names,
                    self.thickness, self.text_scale,
                    skeleton_edges=skeleton_edges,
                ))
            if not annotated:
                continue

            out_path = self.save_dir / "data_preview" / f"02_data_labels_{split}.png"
            _save_image_grid(
                annotated, self.grid_cols,
                f"Data + Labels [{split}] — {n} samples ({self.task})",
                out_path, self.dpi,
            )
            logger.info("HFDataLabelGridCallback: saved %s", out_path)
        return control


class HFAugLabelGridCallback(TrainerCallback):
    """Emits `data_preview/03_aug_labels_train.png` (augmented GT grid) at start.

    Applies `is_train=True` transforms with mosaic/mixup/copypaste disabled so
    each cell shows a single identifiable image — makes the HSV/affine/flip
    parameters visually verifiable. Mirrors the pytorch-backend
    :class:`AugLabelGridLogger`.
    """

    def __init__(
        self,
        save_dir: str,
        splits: list[str],
        data_config: dict,
        aug_config: dict,
        base_dir: str,
        input_size: tuple[int, int],
        task: str = "detection",
        subsets: dict[str, list[int] | None] | None = None,
        num_samples: int = 16,
        grid_cols: int = 4,
        thickness: int = 2,
        text_scale: float = 0.4,
        dpi: int = 120,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.splits = splits
        self.data_config = data_config
        self.aug_config = aug_config or {}
        self.base_dir = base_dir or ""
        self.input_size = tuple(input_size)
        self.task = task
        self.subsets = subsets or {s: None for s in splits}
        self.num_samples = num_samples
        self.grid_cols = grid_cols
        self.thickness = thickness
        self.text_scale = text_scale
        self.dpi = dpi

    def on_train_begin(self, args, state, control, **kwargs):
        class_names = _build_class_names(self.data_config)
        skeleton_edges = (
            _resolve_skeleton_edges(SimpleNamespace(data_config=self.data_config))
            if self.task == "keypoint" else None
        )
        mean = np.asarray(
            self.data_config.get("mean", [0.485, 0.456, 0.406]),
            dtype=np.float32,
        ).reshape(1, 1, 3)
        std = np.asarray(
            self.data_config.get("std", [0.229, 0.224, 0.225]),
            dtype=np.float32,
        ).reshape(1, 1, 3)

        # Drop batch-level ops so each rendered cell is one clear augmented image.
        simple_cfg = {
            **self.aug_config, "mosaic": False, "mixup": False, "copypaste": False,
        }
        try:
            transforms = _build_task_transforms(
                task=self.task, is_train=True, aug_config=simple_cfg,
                input_size=self.input_size,
                mean=self.data_config.get("mean"), std=self.data_config.get("std"),
            )
        except Exception as e:
            logger.info(
                "HFAugLabelGridCallback: transform-build failed for task=%s "
                "(%s) — falling back to dataset default.", self.task, e,
            )
            transforms = None

        for split in self.splits:
            if split != "train":
                continue
            try:
                ds = build_dataset_for_viz(
                    self.task, split, self.data_config, self.base_dir,
                    transforms=transforms,
                )
            except Exception as e:
                logger.info("HFAugLabelGridCallback: skip %s — %s", split, e)
                continue

            subset = self.subsets.get(split)
            pool = list(range(len(ds))) if subset is None else list(subset)
            n = min(self.num_samples, len(pool))
            if n == 0:
                continue
            indices = sorted(random.sample(pool, n))

            annotated: list[np.ndarray] = []
            for i in indices:
                try:
                    result = ds[i]
                    aug_tensor, targets_tensor = result[0], result[1]
                except Exception as e:
                    logger.warning("HFAugLabelGridCallback: failed idx %d — %s", i, e)
                    continue
                aug_bgr = _tensor_to_denorm_bgr(
                    aug_tensor, mean, std,
                    normalize_applied=self.aug_config.get("normalize", True),
                )
                target = _extract_target_for_panel(targets_tensor, self.task)
                annotated.append(_render_gt_panel(
                    aug_bgr, target, self.task, class_names,
                    self.thickness, self.text_scale,
                    skeleton_edges=skeleton_edges,
                ))
            if not annotated:
                continue

            out_path = self.save_dir / "data_preview" / f"03_aug_labels_{split}.png"
            _save_image_grid(
                annotated, self.grid_cols,
                f"Augmented + Labels [{split}] — {n} samples ({self.task})",
                out_path, self.dpi,
            )
            logger.info("HFAugLabelGridCallback: saved %s", out_path)
        return control


class HFValPredictionCallback(TrainerCallback):
    """Per-epoch val grids + (on_train_end) best-checkpoint val/test grids.

    Uses the HF `eval_dataloader` (passed via hook kwargs by HF Trainer) for
    per-epoch grids. Samples a fixed pool of indices on the first epoch so the
    same images appear across every epoch's grid for easy before/after
    comparison.

    On `on_train_end` HF has just reloaded the best checkpoint (via
    ``load_best_model_at_end=True``). We use that moment to render one final
    grid from the best weights — the same weights that produced the reported
    ``test_map`` — and save to ``{val,test}_predictions/best.png``. Test-set
    rendering fires only when a ``test_dataset`` is passed at init.
    """

    def __init__(
        self,
        save_dir: str,
        class_names: dict[int, str],
        input_size: tuple[int, int],
        num_samples: int = 12,
        conf_threshold: float = 0.05,
        grid_cols: int = 2,
        gt_thickness: int = 2,
        pred_thickness: int = 1,
        text_scale: float = 0.4,
        dpi: int = 150,
        test_dataset: Any = None,
        train_dataset: Any = None,
        best_num_samples: int = 16,
        best_conf_threshold: float = 0.1,
        enable_epoch_end: bool = True,
        enable_train_end: bool = True,
        data_config: dict | None = None,
        base_dir: str | None = None,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.class_names = class_names
        self.input_size = tuple(input_size)
        self.num_samples = num_samples
        self.conf_threshold = conf_threshold
        self.grid_cols = grid_cols
        self.gt_thickness = gt_thickness
        self.pred_thickness = pred_thickness
        self.text_scale = text_scale
        self.dpi = dpi
        self.test_dataset = test_dataset
        self.train_dataset = train_dataset
        self.best_num_samples = best_num_samples
        self.best_conf_threshold = best_conf_threshold
        self.enable_epoch_end = enable_epoch_end
        self.enable_train_end = enable_train_end
        self._loaded_data_cfg = data_config
        self._config_dir = base_dir
        self._sample_indices: list[int] | None = None

    def on_epoch_end(self, args, state, control, **kwargs):
        """Per-epoch val grid. Delegates rendering to the shared
        :func:`core.p06_training.post_train.render_prediction_grid` so the
        per-epoch grid is byte-consistent with best.png and the error-analysis
        galleries."""
        if not self.enable_epoch_end:
            return control
        eval_loader = kwargs.get("eval_dataloader")
        model = kwargs.get("model")
        if eval_loader is None or model is None:
            return control

        ds = eval_loader.dataset
        if self._sample_indices is None:
            n = len(ds)
            if n == 0:
                return control
            self._sample_indices = sorted(random.sample(range(n), min(self.num_samples, n)))

        was_training = model.training
        model.eval()

        epoch_idx = int(round(state.epoch or 0.0))

        # Top-down keypoint short-circuit — the shared grid path uses the
        # detection postprocess and doesn't fit our (pixel_values,
        # target_heatmap, target_weight) sample shape.
        task = _infer_task_from_model(model)
        if task == "keypoint" and _is_topdown_keypoint_dataset(ds):
            ap_val = 0.0
            if state.log_history:
                for entry in reversed(state.log_history):
                    if "eval_AP" in entry:
                        ap_val = float(entry["eval_AP"])
                        break
            out_path = (
                self.save_dir / "val_predictions" / "epochs"
                / f"epoch_{epoch_idx:03d}.png"
            )
            try:
                _render_topdown_keypoint_grid(
                    model, ds, self._sample_indices, out_path,
                    title=f"Epoch {epoch_idx} — AP: {ap_val:.4f}",
                    grid_cols=self.grid_cols, dpi=self.dpi,
                )
                logger.info(
                    "HFValPredictionCallback: saved epochs/epoch_%03d.png",
                    epoch_idx,
                )
            except Exception as e:
                logger.warning("per-epoch val grid skipped: %s", e)
            if was_training:
                model.train()
            return control

        map_val = 0.0
        if state.log_history:
            for entry in reversed(state.log_history):
                if "eval_map_50" in entry:
                    map_val = float(entry["eval_map_50"])
                    break

        from core.p06_training.post_train import render_prediction_grid
        from core.p10_inference.supervision_bridge import VizStyle
        out_path = self.save_dir / "val_predictions" / "epochs" / f"epoch_{epoch_idx:03d}.png"
        try:
            render_prediction_grid(
                model, ds, self._sample_indices, out_path,
                title=f"Epoch {epoch_idx} — mAP50: {map_val:.4f}",
                class_names=self.class_names, input_size=self.input_size,
                style=VizStyle(), task=task,
                conf_threshold=self.conf_threshold, grid_cols=self.grid_cols,
                dpi=self.dpi,
            )
            logger.info("HFValPredictionCallback: saved epochs/epoch_%03d.png", epoch_idx)
        except Exception as e:
            logger.warning("per-epoch val grid skipped: %s", e)

        if was_training:
            model.train()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        """Render best-checkpoint val+test grids + full error analysis.

        HF's ``load_best_model_at_end=True`` has already reloaded best weights
        by the time this hook fires — so the artifacts reflect the same
        checkpoint that produced the reported ``test_map``.

        Entirely delegated to :func:`core.p06_training.post_train.run_post_train_artifacts`
        so the pytorch backend and HF backend produce byte-identical artifact
        trees.
        """
        if not self.enable_train_end:
            return control
        model = kwargs.get("model")
        if model is None:
            return control

        from core.p06_training.post_train import run_post_train_artifacts
        from core.p10_inference.supervision_bridge import VizStyle

        best_map = 0.0
        for entry in state.log_history:
            if "eval_map_50" in entry:
                best_map = max(best_map, float(entry["eval_map_50"]))
        test_map = None
        for entry in reversed(state.log_history):
            if "test_map_50" in entry:
                test_map = float(entry["test_map_50"])
                break

        val_loader = kwargs.get("eval_dataloader")
        val_ds = val_loader.dataset if val_loader is not None else None

        # Top-down keypoint short-circuit — run_post_train_artifacts assumes
        # detection-shaped postprocess. We render best.png + a compact
        # error_analysis directly from the heatmap-decoded predictions.
        task_now = _infer_task_from_model(model)
        if task_now == "keypoint" and _is_topdown_keypoint_dataset(val_ds):
            try:
                _run_topdown_keypoint_post_train(
                    model=model, val_ds=val_ds, test_ds=self.test_dataset,
                    save_dir=self.save_dir,
                    best_num_samples=self.best_num_samples,
                    grid_cols=self.grid_cols,
                    dpi=self.dpi,
                )
            except Exception as e:
                logger.warning(
                    "post-train artifacts (top-down keypoint) skipped: %s",
                    e, exc_info=True,
                )
            return control

        try:
            training_config = _build_hf_training_config(args, state, model, best_map, test_map)
            # Plumb resolved data config + base_dir so post_train analyzers
            # (duplicates_leakage, etc.) can resolve split paths.
            if self._loaded_data_cfg is not None:
                training_config["_loaded_data_cfg"] = self._loaded_data_cfg
            if self._config_dir is not None:
                training_config["_config_dir"] = self._config_dir
            run_post_train_artifacts(
                model=model,
                save_dir=self.save_dir,
                val_dataset=val_ds,
                test_dataset=self.test_dataset,
                train_dataset=self.train_dataset,
                val_loader=val_loader,
                task=_infer_task_from_model(model),
                class_names=self.class_names,
                input_size=self.input_size,
                style=VizStyle(),
                best_num_samples=self.best_num_samples,
                best_conf_threshold=self.best_conf_threshold,
                log_history_best_map=best_map if best_map > 0 else None,
                log_history_test_map=test_map,
                training_config=training_config,
            )
        except Exception as e:
            logger.warning("post-train artifacts skipped: %s", e, exc_info=True)
        return control


def _build_hf_training_config(args, state, model, best_map: float, test_map: float | None) -> dict:
    """Extract a compact training-config snapshot from the HF trainer state.

    Shape matches the contract in the plan: model / training / augmentation / run.
    Missing fields are set to None — summary.md prints what's available.
    """
    inner = getattr(model, "hf_model", None)
    arch = None
    params = None
    if inner is not None:
        arch = getattr(getattr(inner, "config", None), "model_type", None) or \
               type(inner).__name__
        try:
            params = int(sum(p.numel() for p in inner.parameters() if p.requires_grad))
        except Exception:
            params = None
    best_epoch = None
    for e in state.log_history:
        if "eval_map_50" in e and float(e["eval_map_50"]) >= best_map - 1e-9:
            best_epoch = e.get("epoch")
            break
    return {
        "model": {"arch": arch, "trainable_params": params,
                  "input_size": getattr(args, "_input_size", None)},
        "training": {
            "backend": "hf",
            "epochs": getattr(args, "num_train_epochs", None),
            "batch_size": getattr(args, "per_device_train_batch_size", None),
            "lr": getattr(args, "learning_rate", None),
            "optimizer": getattr(args, "optim", None),
            "scheduler": getattr(args, "lr_scheduler_type", None),
            "warmup_steps": getattr(args, "warmup_steps", None),
            "weight_decay": getattr(args, "weight_decay", None),
            "bf16": getattr(args, "bf16", None),
            "fp16": getattr(args, "fp16", None),
            "seed": getattr(args, "seed", None),
            "max_grad_norm": getattr(args, "max_grad_norm", None),
        },
        "run": {
            "best_val_map_50": round(float(best_map), 4) if best_map else None,
            "best_epoch": best_epoch,
            "total_epochs": state.epoch,
            "test_map_50": round(float(test_map), 4) if test_map is not None else None,
        },
    }


def _infer_task_from_model(model) -> str:
    """Map ``model.output_format`` → canonical task for the post-train runner."""
    return task_from_output_format(getattr(model, "output_format", None))



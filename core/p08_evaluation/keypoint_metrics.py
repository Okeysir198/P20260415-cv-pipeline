"""Top-down keypoint metrics — PCK + OKS-AP + heatmap decode.

Used by the HF Trainer keypoint task and the `error_analysis_runner`'s
top-down `_analyze_keypoint` path. Designed for `eval_pred` shaped:

    predictions  = heatmaps,       (N, K, H/4, W/4)
    label_ids    = (target_heatmap, target_weight)
                   target_heatmap  (N, K, H/4, W/4)
                   target_weight   (N, K)              {0, 1}

All math is pure numpy — no pycocotools dependency. For full
COCO-shape OKS-AP with per-image scoring + crowd handling, swap in
pycocotools post-training; that wrapper lives in
`reference_vitpose_base/inference.py` (TODO).

Conventions:
- Heatmap stride 4 (input/4). Override via `stride=` if model differs.
- Reference scale for PCK is the input-crop diagonal in pixels:
  ``ref = sqrt(input_h^2 + input_w^2)``. With (256, 192) → 320 px.
- OKS uses object area = input_h * input_w (constant for fixed-size
  top-down crops). Per-joint sigmas from 05_data.yaml::oks_sigmas.
"""
from __future__ import annotations

import numpy as np


def decode_heatmaps_to_xy(
    heatmaps: np.ndarray, stride: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Argmax decode of heatmaps to (x, y) in input-pixel coords + scores.

    Args:
        heatmaps: ``(N, K, H, W)`` float array (any dtype).
        stride: Spatial stride from input pixels to heatmap pixels (4 for ViTPose).

    Returns:
        ``(xy, scores)`` where ``xy`` is ``(N, K, 2)`` float32 in input-pixel
        coords and ``scores`` is ``(N, K)`` float32 — the heatmap-peak value
        (post-softmax magnitude isn't used; raw activation suffices for ranking).
    """
    if heatmaps.ndim != 4:
        raise ValueError(f"heatmaps must be (N, K, H, W); got shape {heatmaps.shape}")
    N, K, H, W = heatmaps.shape
    flat = heatmaps.reshape(N, K, -1)
    idx = flat.argmax(axis=-1)                       # (N, K)
    scores = np.take_along_axis(flat, idx[..., None], axis=-1).squeeze(-1)
    yx = np.stack(np.unravel_index(idx, (H, W)), axis=-1)  # (N, K, 2) in (y, x)
    xy = yx[..., ::-1].astype(np.float32) * stride         # (N, K, 2) in (x, y)
    return xy, scores.astype(np.float32)


def compute_pck(
    pred_xy: np.ndarray, gt_xy: np.ndarray, weight: np.ndarray,
    ref_size: float, thresholds: tuple[float, ...] = (0.05, 0.1, 0.2),
) -> dict[str, float]:
    """Probability of Correct Keypoint at multiple thresholds.

    A joint is "correct" when its predicted-vs-GT pixel distance is below
    ``threshold * ref_size``. Visibility-gated: joints with ``weight == 0``
    are excluded.

    Args:
        pred_xy: ``(N, K, 2)`` predicted x, y in input-pixel coords.
        gt_xy:   ``(N, K, 2)`` ground-truth x, y, same coord space.
        weight:  ``(N, K)`` visibility mask {0, 1}.
        ref_size: Reference scale (e.g. crop diagonal in pixels).
        thresholds: Fractions of ``ref_size`` to evaluate.

    Returns:
        Dict with ``pck_<thr*100>`` keys (e.g. ``pck_05``, ``pck_10``,
        ``pck_20``) and ``mean_pixel_err`` for visible joints.
    """
    err = np.linalg.norm(pred_xy - gt_xy, axis=-1)   # (N, K)
    vis = (weight > 0).astype(np.float32)
    n_vis = max(vis.sum(), 1.0)
    out: dict[str, float] = {
        "mean_pixel_err": float((err * vis).sum() / n_vis),
    }
    for t in thresholds:
        thr = t * ref_size
        correct = ((err < thr) & (vis > 0)).sum()
        out[f"pck_{int(t * 100):02d}"] = float(correct / n_vis)
    return out


def compute_oks(
    pred_xy: np.ndarray, gt_xy: np.ndarray, weight: np.ndarray,
    sigmas: np.ndarray, area: float,
) -> np.ndarray:
    """Per-sample OKS (Object Keypoint Similarity) over all visible joints.

    OKS = sum_i [exp(-d_i^2 / (2 * s^2 * sigma_i^2 * 4)) * I(v_i > 0)]
          / sum_i I(v_i > 0)

    where d_i is the pred-GT distance for joint i, s = sqrt(area),
    sigma_i is the per-joint COCO sigma. Returns ``(N,)`` float32 OKS
    in [0, 1]; samples with no visible joints get OKS = 0.

    Args:
        pred_xy: ``(N, K, 2)`` predicted x, y.
        gt_xy:   ``(N, K, 2)`` ground-truth x, y.
        weight:  ``(N, K)`` visibility mask.
        sigmas:  ``(K,)`` per-joint COCO keypoint sigmas.
        area:    Object area in pixels (constant for fixed-size top-down crops).
    """
    sigmas = np.asarray(sigmas, dtype=np.float32)
    s_sq = float(area)
    var = (sigmas[None, :] ** 2) * 2 * s_sq * 4 + 1e-12   # (1, K)
    err_sq = ((pred_xy - gt_xy) ** 2).sum(axis=-1)         # (N, K)
    e = np.exp(-err_sq / var)                              # (N, K)
    vis = (weight > 0).astype(np.float32)
    n_vis = vis.sum(axis=1)                                # (N,)
    oks = np.where(n_vis > 0, (e * vis).sum(axis=1) / np.maximum(n_vis, 1), 0.0)
    return oks.astype(np.float32)


def compute_oks_ap(
    oks: np.ndarray,
    thresholds: tuple[float, ...] = tuple(np.arange(0.5, 1.0, 0.05)),
) -> dict[str, float]:
    """Top-down OKS-AP — % of samples passing each OKS threshold, averaged.

    Top-down evaluation has no detection-score sweep (every prediction is
    for a known GT bbox), so AP collapses to mean recall across thresholds.
    Reports the IoU-style averaged value plus the canonical 0.5 / 0.75
    thresholds for reference.

    Args:
        oks: ``(N,)`` per-sample OKS values.
        thresholds: OKS thresholds to evaluate.

    Returns:
        Dict with ``AP``, ``AP50``, ``AP75``, and ``OKS_mean``.
    """
    if oks.size == 0:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "OKS_mean": 0.0}
    aps = [(oks >= float(t)).mean() for t in thresholds]
    return {
        "AP": float(np.mean(aps)),
        "AP50": float((oks >= 0.50).mean()),
        "AP75": float((oks >= 0.75).mean()),
        "OKS_mean": float(oks.mean()),
    }


# COCO 17-keypoint default sigmas (used when 05_data.yaml omits oks_sigmas).
COCO_KP_SIGMAS = np.array(
    [.026, .025, .025, .035, .035, .079, .079, .072, .072,
     .062, .062, .107, .107, .087, .087, .089, .089],
    dtype=np.float32,
)


def build_compute_metrics_keypoint(
    sigmas: np.ndarray | list[float] | None,
    input_hw: tuple[int, int],
    stride: int = 4,
):
    """Return an HF-Trainer-compatible ``compute_metrics`` for top-down keypoint.

    Heatmap targets are decoded the same way as predictions, giving GT
    locations at heatmap resolution — sufficient for in-loop selection.
    For pixel-perfect GT, swap in raw kpt coords from a richer label
    pipeline (offline pycocotools eval).

    The returned dict's ``AP`` key is what ``metric_for_best_model: AP``
    will land on (HF prepends ``eval_`` automatically).

    Args:
        sigmas: Per-joint sigmas (length K). Falls back to COCO 17-kpt
            defaults when None.
        input_hw: Crop ``(H, W)`` (e.g. ``(256, 192)`` for ViTPose).
        stride: Heatmap stride (4 for ViTPose's H/4 head).
    """
    if sigmas is None:
        sigmas_arr = COCO_KP_SIGMAS
    else:
        sigmas_arr = np.asarray(sigmas, dtype=np.float32)

    H, W = int(input_hw[0]), int(input_hw[1])
    ref_size = float(np.sqrt(H * H + W * W))
    area = float(H * W)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        # HF concatenates label tensors into a tuple matching args.label_names
        # = ["target_heatmap", "target_weight"].
        if isinstance(labels, (list, tuple)) and len(labels) == 2:
            target_heatmap, target_weight = labels
        else:
            # Defensive fallback — single label tensor.
            target_heatmap = labels
            target_weight = np.ones(target_heatmap.shape[:2], dtype=np.float32)

        preds = np.asarray(preds, dtype=np.float32)
        target_heatmap = np.asarray(target_heatmap, dtype=np.float32)
        target_weight = np.asarray(target_weight, dtype=np.float32)

        pred_xy, _ = decode_heatmaps_to_xy(preds, stride=stride)
        gt_xy, _ = decode_heatmaps_to_xy(target_heatmap, stride=stride)

        pck = compute_pck(pred_xy, gt_xy, target_weight, ref_size=ref_size)
        oks = compute_oks(pred_xy, gt_xy, target_weight, sigmas=sigmas_arr, area=area)
        ap = compute_oks_ap(oks)

        # Per-joint mean pixel error — logged for diagnostics, not used in
        # `metric_for_best_model`.
        err = np.linalg.norm(pred_xy - gt_xy, axis=-1)
        vis = (target_weight > 0).astype(np.float32)
        n_vis_per_joint = np.maximum(vis.sum(axis=0), 1.0)
        per_joint_err = (err * vis).sum(axis=0) / n_vis_per_joint

        out = {**pck, **ap}
        for k, v in enumerate(per_joint_err):
            out[f"err_kp{k:02d}"] = float(v)
        return out

    return compute_metrics

"""Normalize-verification viz for the CPU transform pipeline.

Renders ``<save_dir>/data_preview/04_normalize_check.png`` — a 3-column grid
per sample:

    [raw (disk + GT)]  [normalized tensor (jet ±3σ)]  [denormalized (inverse + GT)]

Col 3 is the critical check: it's the algebraic inverse of the
``v2.Normalize + ToDtype(scale=True)`` step. If it doesn't look like a valid
augmented image (colour cast, box drift, clamping artefacts) the training
pipeline is broken before a single GPU forward pass.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from core.p05_data.base_dataset import IMAGENET_MEAN, IMAGENET_STD
from core.p05_data.transforms import build_transforms
from core.p10_inference.supervision_bridge import VizStyle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def denormalize_chw(
    tensor: torch.Tensor,
    mean: list[float],
    std: list[float],
) -> np.ndarray:
    """Invert ``v2.Normalize + ToDtype(scale=True)`` → HWC uint8 RGB.

    ``t_hwc = (tensor_chw * std + mean).clamp(0, 1) * 255 → uint8``.
    """
    t = tensor.detach().float().cpu()
    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    restored = (t * std_t + mean_t).clamp(0.0, 1.0) * 255.0
    return restored.byte().permute(1, 2, 0).contiguous().numpy()


def tensor_to_uint8_rgb(image: Any) -> np.ndarray:
    """Coerce any pipeline image representation to HWC uint8 RGB."""
    if isinstance(image, torch.Tensor):
        t = image.detach().cpu()
        if t.dtype == torch.uint8:
            return t.permute(1, 2, 0).contiguous().numpy()
        t = t.float()
        max_abs = float(t.abs().max()) if t.numel() else 0.0
        if max_abs > 1.5:
            arr = t.clamp(0, 255).byte().permute(1, 2, 0).contiguous().numpy()
        else:
            arr = (t.clamp(0, 1) * 255).byte().permute(1, 2, 0).contiguous().numpy()
        return arr
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.dtype != np.uint8:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def false_color_normalize(tensor: torch.Tensor) -> np.ndarray:
    """Render a normalized CHW tensor as a jet false-color HWC uint8 image."""
    import matplotlib

    t = tensor.detach().float().cpu().mean(dim=0)
    clipped = t.clamp(-3.0, 3.0)
    norm01 = ((clipped + 3.0) / 6.0).numpy()
    rgba = matplotlib.colormaps["jet"](norm01)
    return (rgba[..., :3] * 255).astype(np.uint8)


def _targets_to_xyxy(
    targets: np.ndarray | torch.Tensor | None, H: int, W: int
) -> tuple[np.ndarray, np.ndarray]:
    """YOLO-normalized (N,5) cxcywh → pixel xyxy + class_id."""
    if targets is None or len(targets) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    arr = (targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor)
           else np.asarray(targets))
    if arr.ndim != 2 or arr.shape[1] < 5:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    cls = arr[:, 0].astype(np.int64)
    cx, cy, bw, bh = arr[:, 1] * W, arr[:, 2] * H, arr[:, 3] * W, arr[:, 4] * H
    xyxy = np.stack(
        [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1
    ).astype(np.float32)
    return xyxy, cls


def _draw_boxes_rgb(
    rgb: np.ndarray,
    targets_n5: np.ndarray,
    class_names: dict[int, str],
    style: VizStyle,
) -> np.ndarray:
    """Draw YOLO-normalized (N,5) GT boxes on an HWC uint8 RGB image."""
    H, W = rgb.shape[:2]
    xyxy, cls = _targets_to_xyxy(targets_n5, H, W)
    if len(xyxy) == 0:
        return rgb
    import supervision as sv

    from utils.viz import annotate_detections
    dets = sv.Detections(
        xyxy=xyxy.astype(np.float64),
        class_id=cls.astype(np.int64),
    )
    return annotate_detections(rgb, dets, class_names=class_names, style=style)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_normalize_check(
    out_path: Path,
    *,
    dataset: Any = None,  # noqa: ARG001 — API symmetry; we build our own
    data_config: dict,
    training_config: dict,
    base_dir: str,
    class_names: dict[int, str],
    num_samples: int = 4,
    style: VizStyle | None = None,
) -> Path | None:
    """Render ``04_normalize_check.png`` — 3-col normalize verification.

    Columns per row:
      1. Raw (from disk + GT)
      2. Normalized tensor (model input, jet false-color ±3σ)
      3. Denormalized (inverse Normalize + GT — should look like a valid
         augmented image)

    ``dataset`` is ignored — we build two fresh ``YOLOXDataset`` instances
    internally (one raw, one train-transformed) to guarantee we're looking at
    the real pipeline.
    """
    import matplotlib.pyplot as plt

    from core.p05_data.detection_dataset import YOLOXDataset

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    style = style or VizStyle()
    aug_cfg = training_config.get("augmentation", {}) or {}
    input_size = tuple(data_config.get("input_size") or (640, 640))
    mean = list(data_config.get("mean") or IMAGENET_MEAN)
    std = list(data_config.get("std") or IMAGENET_STD)

    transform = build_transforms(
        config=aug_cfg, is_train=True, input_size=input_size,
        mean=mean, std=std,
    )

    try:
        raw_ds = YOLOXDataset(data_config, split="train", base_dir=base_dir,
                              transforms=None)
        train_ds = YOLOXDataset(data_config, split="train", base_dir=base_dir,
                                transforms=transform)
    except Exception as e:
        logger.warning("render_normalize_check: dataset build failed — %s", e)
        return None

    if len(raw_ds) == 0:
        logger.warning("render_normalize_check: empty train dataset")
        return None

    num_rows = min(num_samples, len(raw_ds))
    rng = random.Random(42)
    indices = rng.sample(range(len(raw_ds)), num_rows)

    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for idx in indices:
        try:
            # --- Col 1: raw from disk, BGR→RGB, with raw GT overlay.
            raw_bgr = raw_ds.get_raw_item(idx)["image"]
            raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
            raw_targets = raw_ds._load_label(raw_ds.img_paths[idx])
            if raw_targets is None:
                raw_targets = np.zeros((0, 5), dtype=np.float32)
            raw_annotated = _draw_boxes_rgb(raw_rgb, raw_targets, class_names, style)

            # --- Cols 2 & 3: run the real train transform.
            tensor_chw, targets_tensor, _ = train_ds[idx]
            if not isinstance(tensor_chw, torch.Tensor):
                logger.warning("render_normalize_check: sample %d did not yield tensor", idx)
                continue
            norm_viz = false_color_normalize(tensor_chw)
            denorm = denormalize_chw(tensor_chw, mean, std)
            denorm_annotated = _draw_boxes_rgb(
                denorm,
                targets_tensor.detach().cpu().numpy()
                if isinstance(targets_tensor, torch.Tensor) else np.asarray(targets_tensor),
                class_names, style,
            )
            rows.append((raw_annotated, norm_viz, denorm_annotated))
        except Exception as e:
            logger.warning("render_normalize_check: sample idx=%d failed — %s", idx, e)
            continue

    if not rows:
        logger.warning("render_normalize_check: no samples rendered")
        return None

    n = len(rows)
    fig, axes = plt.subplots(
        n, 3,
        figsize=(15, 4.2 * n + 1.2),
        dpi=130,
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0.18, wspace=0.06)

    col_titles = [
        "Raw (from disk + GT)",
        "Normalized tensor (model input, jet ±3σ)",
        "Denormalized (inverse check + GT)",
    ]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, pad=10)

    for r, (col1, col2, col3) in enumerate(rows):
        for ax, img in zip(axes[r], (col1, col2, col3), strict=True):
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)

    fig.suptitle(
        f"Normalize verification — mean={[round(m, 3) for m in mean]}  "
        f"std={[round(s, 3) for s in std]}  ({n} samples)",
        y=0.995, fontsize=12, fontweight="bold",
    )
    fig.text(
        0.5, 0.005,
        "denorm = inverse(Normalize + ToDtype). "
        "Col 3 should look like a valid augmented image; "
        "color casts or box drift indicate a bug.",
        ha="center", fontsize=9, style="italic", color="#444",
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("render_normalize_check: saved %s (%d samples)", out_path, n)
    return out_path

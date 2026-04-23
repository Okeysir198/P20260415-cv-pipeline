"""Step-by-step transform pipeline visualization.

Renders ``<save_dir>/data_preview/04_transform_pipeline.png`` — a ``K × N`` grid
where each row walks one sample through every stage of the CPU transform
pipeline. One representative sample per class (up to 5 classes) is picked from
the train split. The final column is a ``Denormalize(Normalize)`` cell — a
visual sanity check on the algebraic inverse of ``v2.Normalize``.

All text lives outside the image area (column titles on row 0, row labels on
the left margin, figure title, bottom caption). No per-cell banners.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as v2
from torchvision import tv_tensors

from core.p05_data.base_dataset import IMAGENET_MEAN, IMAGENET_STD
from core.p05_data.transforms import (
    DetectionTransform,
    _to_v2_sample,
    build_transforms,
)
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
    """Invert ``v2.Normalize + ToDtype(scale=True)`` → HWC uint8 RGB."""
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


def _boxes_to_xyxy(sample: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract (xyxy pixel boxes, class ids) from a v2 sample dict."""
    boxes = sample.get("boxes")
    labels = sample.get("labels")
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    xyxy = boxes.detach().float().cpu().numpy()
    cls = labels.detach().cpu().numpy().astype(np.int64)
    return xyxy, cls


def _draw_boxes_rgb(
    rgb: np.ndarray,
    xyxy: np.ndarray,
    class_ids: np.ndarray,
    class_names: dict[int, str],
    style: VizStyle,
) -> np.ndarray:
    """Draw xyxy pixel GT boxes on an RGB uint8 image."""
    if len(xyxy) == 0:
        return rgb
    import supervision as sv

    from utils.viz import annotate_detections
    dets = sv.Detections(
        xyxy=xyxy.astype(np.float64),
        class_id=class_ids.astype(np.int64),
    )
    return annotate_detections(rgb, dets, class_names=class_names, style=style)


def _apply_v2_transform(sample: dict, transform: Any) -> dict:
    """Apply a v2 transform to a deep-copied sample dict."""
    sample_copy = {
        "image": tv_tensors.Image(sample["image"].clone()),
        "boxes": tv_tensors.BoundingBoxes(
            sample["boxes"].clone(),
            format=sample["boxes"].format,
            canvas_size=sample["boxes"].canvas_size,
        ),
        "labels": sample["labels"].clone(),
    }
    return transform(sample_copy)


def _transform_name(t: Any) -> str:
    """Readable short name for a v2 / custom transform."""
    return type(t).__name__


def _transform_params(t: Any) -> str:
    """Compact parameter summary."""
    if isinstance(t, v2.Resize):
        size = list(t.size) if hasattr(t, "size") else t.size
        return f"size={size}"
    if isinstance(t, v2.Normalize):
        m = [round(float(x), 2) for x in t.mean]
        s = [round(float(x), 2) for x in t.std]
        return f"μ={m}\nσ={s}"
    if isinstance(t, v2.RandomAffine):
        return f"deg={t.degrees}"
    if isinstance(t, v2.RandomHorizontalFlip | v2.RandomVerticalFlip):
        return f"p={t.p}"
    if isinstance(t, v2.ColorJitter):
        return "ColorJitter"
    if isinstance(t, v2.ToDtype):
        dt = str(t.dtype).replace("torch.", "") if hasattr(t, "dtype") else ""
        scale = getattr(t, "scale", False)
        return f"{dt} scale={scale}"
    return ""


def _cell_title(step_idx: int, name: str, params: str) -> str:
    body = f"[{step_idx:02d}] {name}"
    return f"{body}\n{params}" if params else body


# ---------------------------------------------------------------------------
# Pipeline walk
# ---------------------------------------------------------------------------


def _walk_pipeline(
    raw_image_bgr: np.ndarray,
    raw_targets: np.ndarray,
    detection_transform: DetectionTransform,
    mean: list[float],
    std: list[float],
    class_names: dict[int, str],
    style: VizStyle,
) -> list[dict]:
    """Walk the transform pipeline cumulatively, snapshot each step.

    Returns a list of cell dicts; the last cell is always the denormalize
    round-trip (if Normalize was present). Each cell is:
    ``{"image": HWC uint8 rgb, "title": str, "is_normalize": bool}``.
    """
    orig_hw = (raw_image_bgr.shape[0], raw_image_bgr.shape[1])
    sample = _to_v2_sample(raw_image_bgr, raw_targets, orig_hw)

    cells: list[dict] = []

    # [00] Raw
    raw_rgb = cv2.cvtColor(raw_image_bgr, cv2.COLOR_BGR2RGB)
    xyxy0, cls0 = _boxes_to_xyxy(sample)
    cells.append({
        "image": _draw_boxes_rgb(raw_rgb, xyxy0, cls0, class_names, style),
        "title": _cell_title(0, "Raw", f"{orig_hw[0]}x{orig_hw[1]}"),
        "is_normalize": False,
    })

    post_norm_tensor: torch.Tensor | None = None

    for i, t in enumerate(detection_transform.transforms, start=1):
        try:
            sample = _apply_v2_transform(sample, t)
        except Exception as e:  # pragma: no cover
            logger.warning("transform_pipeline_viz: step %d (%s) failed: %s",
                           i, _transform_name(t), e)
            continue

        img = sample["image"]
        is_norm = isinstance(t, v2.Normalize)
        name = _transform_name(t)
        params = _transform_params(t)

        if is_norm:
            post_norm_tensor = img.clone() if isinstance(img, torch.Tensor) else None
            rgb = false_color_normalize(img)
            cells.append({
                "image": rgb,
                "title": _cell_title(i, name + " (jet ±3σ)", params),
                "is_normalize": True,
            })
        else:
            rgb = tensor_to_uint8_rgb(img)
            xyxy, cls = _boxes_to_xyxy(sample)
            cells.append({
                "image": _draw_boxes_rgb(rgb, xyxy, cls, class_names, style),
                "title": _cell_title(i, name, params),
                "is_normalize": False,
            })

    # Final cell: Denormalize(Normalize) sanity check.
    if post_norm_tensor is not None:
        denorm_rgb = denormalize_chw(post_norm_tensor, mean, std)
        xyxy_final, cls_final = _boxes_to_xyxy(sample)
        cells.append({
            "image": _draw_boxes_rgb(denorm_rgb, xyxy_final, cls_final,
                                     class_names, style),
            "title": _cell_title(len(cells), "Denorm[Normalize]", "inverse"),
            "is_normalize": False,
        })

    return cells


# ---------------------------------------------------------------------------
# Per-class sample picker
# ---------------------------------------------------------------------------


def _pick_one_per_class(
    raw_ds: Any, max_samples: int
) -> list[tuple[int, int]]:
    """Scan dataset in order; keep first idx per class_id.

    Returns ordered ``[(class_id, idx), ...]``. Stops once ``max_samples``
    classes are covered or the dataset is exhausted.
    """
    picked: dict[int, int] = {}
    for idx, _ in enumerate(raw_ds.img_paths):
        try:
            labels = raw_ds._load_label(raw_ds.img_paths[idx])
        except Exception:
            continue
        if labels is None or len(labels) == 0:
            continue
        for cls_id in np.unique(labels[:, 0].astype(np.int64)).tolist():
            if cls_id not in picked:
                picked[cls_id] = idx
            if len(picked) >= max_samples:
                break
        if len(picked) >= max_samples:
            break
    return sorted(picked.items(), key=lambda kv: kv[0])[:max_samples]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_transform_pipeline(
    out_path: Path,
    *,
    data_config: dict,
    training_config: dict,
    base_dir: str,
    class_names: dict[int, str],
    max_samples: int = 5,
    style: VizStyle | None = None,
) -> Path | None:
    """Render ``04_transform_pipeline.png`` — K rows × N columns.

    Each row walks one representative sample (first occurrence per class,
    up to ``max_samples`` classes) through every CPU transform step. Final
    column is ``Denormalize(Normalize)``, a visual inverse-algebra check.
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

    # Walker pipeline: force torchvision, strip dataset-level ops (mosaic/mixup
    # /copypaste need other items). Train-mode to exercise full aug chain.
    walker_cfg = dict(aug_cfg)
    walker_cfg["library"] = "torchvision"
    walker_cfg["mosaic"] = False
    walker_cfg["mixup"] = False
    walker_cfg["copypaste"] = False
    transform = build_transforms(
        config=walker_cfg, is_train=True, input_size=input_size,
        mean=mean, std=std,
    )
    if not isinstance(transform, DetectionTransform):
        logger.warning("render_transform_pipeline: expected DetectionTransform; got %s",
                       type(transform).__name__)
        return None

    try:
        raw_ds = YOLOXDataset(data_config, split="train", base_dir=base_dir,
                              transforms=None)
    except Exception as e:
        logger.warning("render_transform_pipeline: dataset build failed — %s", e)
        return None

    if len(raw_ds) == 0:
        logger.warning("render_transform_pipeline: empty train dataset")
        return None

    picks = _pick_one_per_class(raw_ds, max_samples)
    if not picks:
        logger.warning("render_transform_pipeline: no labeled samples found")
        return None

    rows: list[tuple[int, list[dict]]] = []
    for cls_id, idx in picks:
        try:
            raw_bgr = raw_ds.get_raw_item(idx)["image"]
            raw_targets = raw_ds._load_label(raw_ds.img_paths[idx])
            if raw_targets is None:
                raw_targets = np.zeros((0, 5), dtype=np.float32)
            cells = _walk_pipeline(
                raw_bgr, raw_targets, transform,
                mean=mean, std=std, class_names=class_names, style=style,
            )
            rows.append((cls_id, cells))
        except Exception as e:
            logger.warning("render_transform_pipeline: class=%d idx=%d failed — %s",
                           cls_id, idx, e)
            continue

    if not rows:
        logger.warning("render_transform_pipeline: no rows rendered")
        return None

    # Uniform column count — pipeline is identical per sample.
    n_cols = max(len(cells) for _, cells in rows)
    k_rows = len(rows)

    fig, axes = plt.subplots(
        k_rows, n_cols,
        figsize=(3 * n_cols, 3.2 * k_rows + 1.5),
        dpi=130,
        squeeze=False,
    )
    fig.subplots_adjust(wspace=0.05, hspace=0.15)

    # Column titles on row 0 only.
    ref_titles = rows[0][1]
    for c in range(n_cols):
        if c < len(ref_titles):
            axes[0, c].set_title(ref_titles[c]["title"], fontsize=9, pad=6)

    for r, (cls_id, cells) in enumerate(rows):
        cls_label = class_names.get(int(cls_id), f"class_{cls_id}")
        for c in range(n_cols):
            ax = axes[r, c]
            if c < len(cells):
                ax.imshow(cells[c]["image"])
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        axes[r, 0].set_ylabel(
            f"class: {cls_label}",
            rotation=0, ha="right", va="center", labelpad=40, fontsize=10,
        )

    feature = data_config.get("dataset_name") or "unknown"
    fig.suptitle(
        f"Transform pipeline — {feature} · input {input_size[0]}×{input_size[1]}  "
        f"mean={[round(m, 3) for m in mean]} std={[round(s, 3) for s in std]}",
        y=0.995, fontsize=11, fontweight="bold",
    )
    fig.text(
        0.5, 0.005,
        "Last column = Denormalize(Normalize). "
        "Should visually match the step before Normalize. Color cast = bug.",
        ha="center", fontsize=9, style="italic", color="#444",
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("render_transform_pipeline: saved %s (%d rows × %d cols)",
                out_path, k_rows, n_cols)
    return out_path

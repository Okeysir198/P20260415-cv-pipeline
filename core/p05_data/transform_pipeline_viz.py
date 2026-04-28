"""Step-by-step transform pipeline visualization.

Renders ``<save_dir>/data_preview/04_transform_pipeline.png`` — an ``N × K``
grid where **rows are pipeline steps** (in execution order, enabled only) and
**columns are one representative sample per class** (up to 5 classes). Each
cell carries a 2-line metadata caption (``dtype · shape`` / ``[min, max, μ]``)
computed on the *raw tensor* before display-resizing. Row 0 also surfaces the
class header; the final row is ``Denormalize(Normalize)``, a visual inverse-
algebra sanity check on ``v2.Normalize``.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable

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
from loguru import logger
from core.p10_inference.supervision_bridge import VizStyle


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


def _cell_meta(x: Any) -> str:
    """Return 2-line ``'<dtype> · <shape>\\n[min, max, μ=mean]'`` caption.

    Computed on the raw tensor/array *before* any display resizing so the
    caption reflects what the model actually sees.
    """
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
        dtype = str(x.dtype).replace("torch.", "")
        shape = "×".join(str(s) for s in tuple(x.shape))
    else:
        arr = np.asarray(x)
        dtype = str(arr.dtype)
        shape = "×".join(str(s) for s in arr.shape)
    if arr.size == 0:
        return f"{dtype} · {shape}\n[empty]"
    mn, mx, mu = float(arr.min()), float(arr.max()), float(arr.mean())
    if arr.dtype == np.uint8:
        return f"{dtype} · {shape}\n[{int(mn)}, {int(mx)}, μ={int(mu)}]"
    return f"{dtype} · {shape}\n[{mn:.2f}, {mx:.2f}, μ={mu:.2f}]"


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

    Returns an ordered list of cells, one per pipeline step. The last cell is
    always the ``Denormalize(Normalize)`` round-trip (if Normalize was
    present). Each cell is:

        {"image": HWC uint8 rgb, "step_name": str, "meta": str,
         "is_normalize": bool}
    """
    orig_hw = (raw_image_bgr.shape[0], raw_image_bgr.shape[1])
    sample = _to_v2_sample(raw_image_bgr, raw_targets, orig_hw)

    cells: list[dict] = []

    # [00] Raw — HWC uint8 BGR array pre-conversion.
    raw_rgb = cv2.cvtColor(raw_image_bgr, cv2.COLOR_BGR2RGB)
    xyxy0, cls0 = _boxes_to_xyxy(sample)
    cells.append({
        "image": _draw_boxes_rgb(raw_rgb, xyxy0, cls0, class_names, style),
        "step_name": "Raw",
        "meta": _cell_meta(raw_rgb),
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
        meta = _cell_meta(img)

        if is_norm:
            post_norm_tensor = img.clone() if isinstance(img, torch.Tensor) else None
            cells.append({
                "image": false_color_normalize(img),
                "step_name": f"{name} (jet ±3σ)",
                "meta": meta,
                "is_normalize": True,
            })
        else:
            rgb = tensor_to_uint8_rgb(img)
            xyxy, cls = _boxes_to_xyxy(sample)
            cells.append({
                "image": _draw_boxes_rgb(rgb, xyxy, cls, class_names, style),
                "step_name": name,
                "meta": meta,
                "is_normalize": False,
            })

    # Final cell: Denormalize(Normalize) sanity check.
    if post_norm_tensor is not None:
        denorm_rgb = denormalize_chw(post_norm_tensor, mean, std)
        xyxy_final, cls_final = _boxes_to_xyxy(sample)
        cells.append({
            "image": _draw_boxes_rgb(denorm_rgb, xyxy_final, cls_final,
                                     class_names, style),
            "step_name": "Denormalize(Normalize)",
            "meta": _cell_meta(denorm_rgb),
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
# Task-overlay registry (paired walker — non-detection tasks)
# ---------------------------------------------------------------------------


def _pick_first_n(samples: Any, n: int) -> list[int]:
    """Return ordered indices ``[0, 1, ..., min(len, n)-1]``."""
    total = len(samples) if hasattr(samples, "__len__") else 0
    return list(range(min(int(n), total)))


def _overlay_detection(
    ax, image_rgb: np.ndarray, target: Any, *,
    class_names: dict[int, str], style: VizStyle, **_kw,
) -> None:
    if image_rgb is None:
        return
    if target is None:
        ax.imshow(image_rgb)
        return
    if isinstance(target, dict) and "boxes" in target:
        xyxy, cls = _boxes_to_xyxy(target)
    elif isinstance(target, tuple) and len(target) == 2:
        xyxy, cls = target
    else:
        ax.imshow(image_rgb)
        return
    if len(xyxy) == 0:
        ax.imshow(image_rgb)
        return
    annotated = _draw_boxes_rgb(image_rgb, xyxy, cls, class_names, style)
    ax.imshow(annotated)


def _overlay_classification(
    ax, image_rgb: np.ndarray, target: Any, *,
    class_names: dict[int, str], style: VizStyle, **_kw,
) -> None:
    import matplotlib.patches as mpatches

    if image_rgb is None:
        return
    ax.imshow(image_rgb)
    if target is None:
        return
    try:
        cid = int(target)
    except (TypeError, ValueError):
        return
    label = class_names.get(cid, f"class_{cid}")
    h, w = image_rgb.shape[:2]
    color_rgb = getattr(style, "gt_color_rgb", (0, 200, 0))
    color = tuple(c / 255.0 for c in color_rgb)
    border = mpatches.Rectangle(
        (0, 0), w - 1, h - 1, linewidth=4, edgecolor=color, facecolor="none",
    )
    ax.add_patch(border)
    ax.text(
        w / 2, 8, f"Label: {label}",
        ha="center", va="top", fontsize=10, fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, edgecolor="none"),
    )


def _overlay_segmentation(
    ax, image_rgb: np.ndarray, target: Any, *,
    class_names: dict[int, str], style: VizStyle, **_kw,
) -> None:
    if image_rgb is None:
        return
    ax.imshow(image_rgb)
    if target is None:
        return
    mask = np.asarray(target)
    if mask.ndim != 2 or mask.size == 0:
        return
    h, w = image_rgb.shape[:2]
    if mask.shape != (h, w):
        try:
            mask = cv2.resize(mask.astype(np.int32), (w, h),
                              interpolation=cv2.INTER_NEAREST)
        except Exception:
            return
    mask = mask.astype(np.int64)
    # Class-id 0 = background (not rendered). 255 (and other ignore values)
    # also skipped. Color each remaining class via the shared label palette.
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    unique = [int(c) for c in np.unique(mask) if 0 < int(c) < 255]
    if not unique:
        return
    try:
        from core.p05_data.run_viz import _LABEL_PALETTE  # type: ignore[attr-defined]
    except Exception:
        _LABEL_PALETTE = None  # noqa: N806
    for cid in unique:
        sel = mask == cid
        if _LABEL_PALETTE is not None:
            bgr = _LABEL_PALETTE[cid % len(_LABEL_PALETTE)]
            color = (bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0)
        else:
            base = getattr(style, "gt_color_rgb", (0, 200, 0))
            color = tuple(c / 255.0 for c in base)
        rgba[sel, 0] = color[0]
        rgba[sel, 1] = color[1]
        rgba[sel, 2] = color[2]
        rgba[sel, 3] = 0.45
    ax.imshow(rgba)


def _default_skeleton() -> list[tuple[int, int]]:
    try:
        from utils.viz import COCO_17_SKELETON  # type: ignore[attr-defined]
        return list(COCO_17_SKELETON)
    except Exception:
        return []


def _overlay_keypoint(
    ax, image_rgb: np.ndarray, target: Any, *,
    class_names: dict[int, str], style: VizStyle, **kw,
) -> None:
    if image_rgb is None:
        return
    ax.imshow(image_rgb)
    if target is None:
        return
    kpts = np.asarray(target, dtype=np.float32)
    if kpts.ndim != 2 or kpts.shape[0] == 0:
        return
    has_vis = kpts.shape[1] >= 3
    xs, ys = kpts[:, 0], kpts[:, 1]
    vis = kpts[:, 2] if has_vis else np.ones(len(kpts), dtype=np.float32)
    skeleton = kw.get("skeleton") or _default_skeleton()
    color_rgb = getattr(style, "gt_color_rgb", (0, 200, 0))
    main_color = tuple(c / 255.0 for c in color_rgb)
    for a, b in skeleton:
        if a >= len(kpts) or b >= len(kpts):
            continue
        va, vb = float(vis[a]), float(vis[b])
        if va <= 0 and vb <= 0:
            continue
        edge_color = main_color if (va > 0 and vb > 0) else (0.6, 0.6, 0.6)
        ax.plot([xs[a], xs[b]], [ys[a], ys[b]],
                color=edge_color, linewidth=1.8, alpha=0.85)
    visible = vis > 0
    if visible.any():
        ax.scatter(xs[visible], ys[visible], s=18, c=[main_color],
                   edgecolors="white", linewidths=0.6, zorder=3)
    if (~visible).any():
        ax.scatter(xs[~visible], ys[~visible], s=12, c=[(0.6, 0.6, 0.6)],
                   edgecolors="white", linewidths=0.4, zorder=3)


_OVERLAYS: dict[str, Callable] = {
    "detection": _overlay_detection,
    "classification": _overlay_classification,
    "segmentation": _overlay_segmentation,
    "keypoint": _overlay_keypoint,
}


# ---------------------------------------------------------------------------
# Unified paired walker — public entry points are thin wrappers below
# ---------------------------------------------------------------------------


def _render_paired_walker(
    out_path: Path,
    *,
    task: str,
    data_config: dict,
    training_config: dict,
    base_dir: str,
    class_names: dict[int, str],
    max_samples: int = 5,
    style: VizStyle | None = None,
) -> Path | None:
    """Single shared paired-walker entry point.

    Detection routes through ``_render_detection_walker`` (the original
    detection-specific renderer) to preserve bit-for-bit parity. Cls /
    seg / kpt route through ``_render_task_walker`` which uses the
    overlay registry above.
    """
    if task == "detection":
        return _render_detection_walker(
            out_path,
            data_config=data_config,
            training_config=training_config,
            base_dir=base_dir,
            class_names=class_names,
            max_samples=max_samples,
            style=style,
        )
    return _render_task_walker(
        out_path,
        task=task,
        data_config=data_config,
        training_config=training_config,
        base_dir=base_dir,
        class_names=class_names,
        max_samples=max_samples,
        style=style,
    )


def _short_args(t: Any) -> str:
    """Compact one-liner of the transform's most-relevant kwargs."""
    cn = type(t).__name__
    bits: list[str] = []
    for attr in ("size", "p", "scale", "ratio", "degrees", "brightness",
                 "contrast", "saturation", "hue", "mean", "std",
                 "interpolation"):
        if hasattr(t, attr):
            v = getattr(t, attr)
            try:
                if isinstance(v, (list, tuple)) and len(v) > 4:
                    continue
                if isinstance(v, float):
                    bits.append(f"{attr}={v:.2g}")
                else:
                    bits.append(f"{attr}={v}")
            except Exception:
                continue
        if len(bits) >= 3:
            break
    return f"{cn}({', '.join(bits)})" if bits else cn


def _is_normalize(t: Any) -> bool:
    return isinstance(t, v2.Normalize) or type(t).__name__ == "Normalize"


def _tensor_chw_to_uint8_rgb(t: torch.Tensor) -> np.ndarray:
    """CHW float/uint8 tensor (assumed RGB) → HWC uint8 RGB."""
    x = t.detach().cpu()
    if x.dtype == torch.uint8:
        return x.permute(1, 2, 0).contiguous().numpy()
    x = x.float()
    mx = float(x.abs().max()) if x.numel() else 0.0
    if mx > 1.5:
        x = x.clamp(0, 255) / 255.0
    return (x.clamp(0, 1) * 255).byte().permute(1, 2, 0).contiguous().numpy()


def _pil_or_tensor_to_rgb(x: Any) -> np.ndarray:
    """Coerce a classification-pipeline intermediate (PIL or CHW tensor) to HWC uint8 RGB."""
    if isinstance(x, torch.Tensor):
        if x.ndim == 3 and x.shape[0] in (1, 3):
            return _tensor_chw_to_uint8_rgb(x)
        return tensor_to_uint8_rgb(x)
    try:
        from PIL import Image as _PILImage
        if isinstance(x, _PILImage.Image):
            return np.asarray(x.convert("RGB"))
    except Exception:
        pass
    return tensor_to_uint8_rgb(x)


def _walk_classification(
    raw_bgr: np.ndarray,
    target_int: int,
    transform_obj: Any,
    mean: list[float],
    std: list[float],
) -> list[dict]:
    """Step through a classification pipeline (legacy torchvision Compose).

    Target (int class id) is invariant — same on every step.
    """
    from PIL import Image as _PILImage

    rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    cells: list[dict] = [{
        "image": rgb, "target": target_int,
        "step_name": "Raw", "meta": _cell_meta(rgb), "is_normalize": False,
    }]
    if transform_obj is None:
        return cells

    inner = getattr(transform_obj, "transform", None)
    steps = getattr(inner, "transforms", None) if inner is not None else None
    if not steps:
        return cells

    x: Any = _PILImage.fromarray(rgb)
    post_norm: torch.Tensor | None = None
    for t in steps:
        try:
            x = t(x)
        except Exception as e:  # pragma: no cover
            logger.warning("classification walker: %s failed: %s",
                           type(t).__name__, e)
            continue
        is_norm = _is_normalize(t)
        meta = _cell_meta(x)
        if is_norm and isinstance(x, torch.Tensor):
            post_norm = x.clone()
            disp = false_color_normalize(x)
        else:
            disp = _pil_or_tensor_to_rgb(x)
        cells.append({
            "image": disp, "target": target_int,
            "step_name": _short_args(t), "meta": meta, "is_normalize": is_norm,
        })

    if post_norm is not None:
        denorm = denormalize_chw(post_norm, mean, std)
        cells.append({
            "image": denorm, "target": target_int,
            "step_name": "Denormalize(Normalize)",
            "meta": _cell_meta(denorm), "is_normalize": False,
        })
    return cells


def _seg_sample_to_rgb_and_mask(sample: dict) -> tuple[np.ndarray, np.ndarray]:
    img = sample["image"]
    mask = sample["mask"]
    if isinstance(img, torch.Tensor):
        rgb = _tensor_chw_to_uint8_rgb(img)
    else:
        rgb = np.asarray(img)
    if isinstance(mask, torch.Tensor):
        m = mask.detach().cpu().numpy()
    else:
        m = np.asarray(mask)
    return rgb, m.astype(np.uint8)


def _walk_segmentation(
    raw_bgr: np.ndarray,
    raw_mask: np.ndarray,
    transform_obj: Any,
    mean: list[float],
    std: list[float],
) -> list[dict]:
    """Step through a v2-Compose seg pipeline using a tv_tensors sample dict."""
    rgb_raw = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    cells: list[dict] = [{
        "image": rgb_raw, "target": raw_mask.astype(np.uint8),
        "step_name": "Raw", "meta": _cell_meta(rgb_raw), "is_normalize": False,
    }]
    if transform_obj is None:
        return cells

    inner = getattr(transform_obj, "transform", None)
    steps = getattr(inner, "transforms", None) if inner is not None else None
    if not steps:
        return cells

    image_chw = np.ascontiguousarray(rgb_raw.transpose(2, 0, 1))
    sample = {
        "image": tv_tensors.Image(torch.from_numpy(image_chw)),
        "mask": tv_tensors.Mask(torch.from_numpy(raw_mask.astype(np.int64))),
    }
    post_norm: torch.Tensor | None = None
    for t in steps:
        try:
            sample = t(sample)
        except Exception as e:  # pragma: no cover
            logger.warning("segmentation walker: %s failed: %s",
                           type(t).__name__, e)
            continue
        rgb_step, mask_step = _seg_sample_to_rgb_and_mask(sample)
        is_norm = _is_normalize(t)
        meta = _cell_meta(sample["image"])
        if is_norm and isinstance(sample["image"], torch.Tensor):
            post_norm = sample["image"].clone()
            disp = false_color_normalize(sample["image"])
        else:
            disp = rgb_step
        cells.append({
            "image": disp, "target": mask_step,
            "step_name": _short_args(t), "meta": meta, "is_normalize": is_norm,
        })

    if post_norm is not None:
        denorm = denormalize_chw(post_norm, mean, std)
        _, final_mask = _seg_sample_to_rgb_and_mask(sample)
        cells.append({
            "image": denorm, "target": final_mask,
            "step_name": "Denormalize(Normalize)",
            "meta": _cell_meta(denorm), "is_normalize": False,
        })
    return cells


def _walk_keypoint(
    raw_bgr: np.ndarray,
    raw_target_dict: Any,
    post_tensor: torch.Tensor | None,
    post_target: Any,
    transform_obj: Any,
    mean: list[float],
    std: list[float],
    *,
    is_topdown: bool = False,
) -> list[dict]:
    """Render the keypoint transform pipeline.

    Bottom-up (``KeypointDataset`` + ``KeypointTransform``) is monolithic
    so we render Raw → opaque-transform → Denormalize.

    Top-down (``KeypointTopDownDataset`` + HF ``VitPoseImageProcessor``)
    decomposes into 4 visible stages: Raw → BBoxCrop → ResizeToInput →
    Normalize → Denormalize. Each is rendered with the appropriate
    keypoints overlay so the user can see crop, scale, and color shift.
    """
    rgb_raw = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    h_raw, w_raw = rgb_raw.shape[:2]
    raw_kpts_arr = None
    if isinstance(raw_target_dict, dict):
        kp = raw_target_dict.get("keypoints")
        if kp is not None:
            arr = np.asarray(kp, dtype=np.float32)
            if arr.ndim == 3 and arr.shape[0] > 0:
                arr = arr.reshape(-1, arr.shape[-1]).copy()
                # Top-down `get_raw_item` already returns pixel coords;
                # bottom-up returns normalized 0–1.
                if not is_topdown:
                    arr[:, 0] *= w_raw
                    arr[:, 1] *= h_raw
                raw_kpts_arr = arr

    cells: list[dict] = [{
        "image": rgb_raw, "target": raw_kpts_arr,
        "step_name": "Raw", "meta": _cell_meta(rgb_raw), "is_normalize": False,
    }]
    if post_tensor is None:
        return cells

    # Top-down decomposition: bbox-crop → resize → normalize → denorm.
    if is_topdown and isinstance(raw_target_dict, dict):
        try:
            from core.p05_data.keypoint_dataset import _expand_bbox_topdown
        except Exception:
            _expand_bbox_topdown = None  # noqa: N806

        targets = raw_target_dict.get("targets")
        if (
            _expand_bbox_topdown is not None
            and targets is not None
            and len(targets) >= 1
        ):
            row = np.asarray(targets, dtype=np.float32).reshape(-1, 5)[0]
            cx, cy, w_box, h_box = float(row[1]), float(row[2]), float(row[3]), float(row[4])
            input_h, input_w = int(post_tensor.shape[-2]), int(post_tensor.shape[-1])
            bx, by, bw, bh = _expand_bbox_topdown(
                cx, cy, w_box, h_box, w_raw, h_raw,
                aspect_hw=(input_h, input_w), padding=1.25,
            )
            # Stage 1: crop rectangle drawn on the source image.
            x0 = int(max(0, round(bx)))
            y0 = int(max(0, round(by)))
            x1 = int(min(w_raw, round(bx + bw)))
            y1 = int(min(h_raw, round(by + bh)))
            crop_rgb = rgb_raw[y0:y1, x0:x1].copy() if (x1 > x0 and y1 > y0) else rgb_raw
            # Map raw kpts into crop pixel coords.
            crop_kpts = None
            if raw_kpts_arr is not None and crop_rgb.size:
                k = raw_kpts_arr.copy()
                k[:, 0] -= x0
                k[:, 1] -= y0
                crop_kpts = k
            cells.append({
                "image": crop_rgb, "target": crop_kpts,
                "step_name": f"BBoxCrop (pad=1.25 → {crop_rgb.shape[1]}×{crop_rgb.shape[0]})",
                "meta": _cell_meta(crop_rgb), "is_normalize": False,
            })

            # Stage 2: resize crop to input HxW (no normalize yet).
            if crop_rgb.size:
                resized = cv2.resize(crop_rgb, (input_w, input_h),
                                     interpolation=cv2.INTER_LINEAR)
            else:
                resized = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
            sx = input_w / max(crop_rgb.shape[1], 1)
            sy = input_h / max(crop_rgb.shape[0], 1)
            resized_kpts = None
            if crop_kpts is not None:
                k = crop_kpts.copy()
                k[:, 0] *= sx
                k[:, 1] *= sy
                resized_kpts = k
            cells.append({
                "image": resized, "target": resized_kpts,
                "step_name": f"ResizeToInput ({input_w}×{input_h})",
                "meta": _cell_meta(resized), "is_normalize": False,
            })

            # Stage 3: normalize (false-color jet of the post-Normalize tensor).
            cells.append({
                "image": false_color_normalize(post_tensor),
                "target": None,
                "step_name": "Normalize (jet ±3σ)",
                "meta": _cell_meta(post_tensor), "is_normalize": True,
            })

            # Stage 4: Denormalize sanity check — keypoints should match.
            denorm = denormalize_chw(post_tensor, mean, std)
            cells.append({
                "image": denorm, "target": resized_kpts,
                "step_name": "Denormalize(Normalize)",
                "meta": _cell_meta(denorm), "is_normalize": False,
            })
            return cells

    # Bottom-up (monolithic transform): raw → opaque-transform → denorm.
    name = type(transform_obj).__name__ if transform_obj is not None else "(transform)"
    is_norm = bool(getattr(transform_obj, "mean", None) is not None)
    disp = false_color_normalize(post_tensor) if is_norm else _tensor_chw_to_uint8_rgb(post_tensor)

    post_kpts_arr = None
    h_p, w_p = post_tensor.shape[-2], post_tensor.shape[-1]
    if isinstance(post_target, np.ndarray):
        kp = post_target
    elif isinstance(post_target, dict):
        kp = post_target.get("keypoints")
    else:
        kp = None
    if kp is not None:
        arr = np.asarray(kp, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] > 0:
            arr = arr.reshape(-1, arr.shape[-1]).copy()
            arr[:, 0] *= w_p
            arr[:, 1] *= h_p
            post_kpts_arr = arr

    cells.append({
        "image": disp, "target": post_kpts_arr,
        "step_name": f"{name} (opaque: flip+jitter+resize+normalize)",
        "meta": _cell_meta(post_tensor), "is_normalize": is_norm,
    })

    if is_norm:
        denorm = denormalize_chw(post_tensor, mean, std)
        cells.append({
            "image": denorm, "target": post_kpts_arr,
            "step_name": "Denormalize(Normalize)",
            "meta": _cell_meta(denorm), "is_normalize": False,
        })
    return cells


def _render_task_walker(
    out_path: Path,
    *,
    task: str,
    data_config: dict,
    training_config: dict,
    base_dir: str,
    class_names: dict[int, str],
    max_samples: int = 5,
    style: VizStyle | None = None,
) -> Path | None:
    """Per-step paired walker for classification / segmentation / keypoint.

    For each task-specific transform pipeline we step through it one
    transform at a time when the pipeline is v2-/Compose-decomposable
    (cls + seg). Keypoint's ``KeypointTransform`` is monolithic — it gets
    rendered as a single opaque step. Final row is always
    ``Denormalize(Normalize)`` when a normalize op was present. Each cell
    carries the same overlay routing as the post-only fallback used to.
    """
    import matplotlib.pyplot as plt

    from core.p06_training._common import build_dataset_for_viz
    from core.p06_training.hf_callbacks import (
        _build_task_transforms,
        _extract_target_for_panel,
    )
    from utils.viz import wrap_suptitle

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    style = style or VizStyle()
    try:
        VizStyle.apply_plot_style()
    except Exception:
        pass

    aug_cfg = dict(training_config.get("augmentation", {}) or {})
    aug_cfg["mosaic"] = False
    aug_cfg["mixup"] = False
    aug_cfg["copypaste"] = False
    input_size = tuple(data_config.get("input_size") or (640, 640))
    mean_list = list(data_config.get("mean") or IMAGENET_MEAN)
    std_list = list(data_config.get("std") or IMAGENET_STD)

    overlay = _OVERLAYS.get(task)
    if overlay is None:
        logger.warning("_render_task_walker: no overlay for task=%s", task)
        return None

    # Detect top-down keypoint via training-config arch — bottom-up
    # `KeypointDataset` doesn't fit ViTPose-family models.
    is_topdown_kpt = task == "keypoint" and (
        (training_config.get("model") or {}).get("arch") == "hf_keypoint"
        or (training_config.get("model") or {}).get("top_down") is True
    )

    try:
        if is_topdown_kpt:
            from core.p05_data.keypoint_dataset import KeypointTopDownDataset
            from transformers import AutoImageProcessor
            model_cfg = training_config.get("model") or {}
            pretrained = model_cfg.get("pretrained") or model_cfg.get("hf_model_id")
            processor = AutoImageProcessor.from_pretrained(pretrained) if pretrained else None
            raw_ds = KeypointTopDownDataset(
                data_config=data_config, split="train", processor=processor,
                bbox_padding=float(model_cfg.get("bbox_padding", 1.25)),
                is_train=False, base_dir=base_dir,
            )
            transforms = None  # opaque — we drive it through __getitem__
        else:
            raw_ds = build_dataset_for_viz(
                task, "train", data_config, base_dir, transforms=None,
            )
    except Exception as e:
        logger.warning("_render_task_walker: raw ds build failed — %s", e)
        return None

    if not is_topdown_kpt:
        try:
            transforms = _build_task_transforms(
                task=task, is_train=True, aug_config=aug_cfg,
                input_size=input_size, mean=mean_list, std=std_list,
            )
        except Exception as e:
            logger.warning("_render_task_walker: transforms build failed — %s", e)
            transforms = None

    n = min(max_samples, len(raw_ds))
    if n == 0:
        return None
    indices = _pick_first_n(raw_ds, n)

    # Per-column step-walks. Each column = one sample, list of step cells.
    columns: list[list[dict]] = []
    for idx in indices:
        try:
            raw_item = raw_ds.get_raw_item(idx)
            raw_bgr = raw_item["image"]
            raw_target = raw_item.get("targets")

            if task == "classification":
                cells = _walk_classification(
                    raw_bgr, int(raw_target) if raw_target is not None else 0,
                    transforms, mean_list, std_list,
                )
            elif task == "segmentation":
                mask = np.asarray(raw_target).astype(np.uint8) if raw_target is not None \
                    else np.zeros(raw_bgr.shape[:2], dtype=np.uint8)
                cells = _walk_segmentation(
                    raw_bgr, mask, transforms, mean_list, std_list,
                )
            elif task == "keypoint":
                post_tensor: torch.Tensor | None = None
                post_target: Any = None
                if is_topdown_kpt:
                    # Drive the dataset's processor + crop pipeline directly.
                    try:
                        item = raw_ds[idx]
                        pv = item.get("pixel_values") if isinstance(item, dict) else None
                        if pv is not None:
                            post_tensor = pv if isinstance(pv, torch.Tensor) \
                                else torch.from_numpy(np.asarray(pv))
                    except Exception as e:
                        logger.warning(
                            "_render_task_walker: top-down kpt __getitem__ failed at idx %d — %s",
                            idx, e,
                        )
                elif transforms is not None:
                    try:
                        # Bottom-up: KeypointTransform expects {"boxes","keypoints"}.
                        td = {
                            "boxes": np.asarray(raw_item.get("targets",
                                np.zeros((0, 5), dtype=np.float32))),
                            "keypoints": np.asarray(raw_item.get("keypoints",
                                np.zeros((0, 0, 3), dtype=np.float32))),
                        }
                        out = transforms(raw_bgr, td)
                        if isinstance(out, tuple) and len(out) == 2:
                            post_tensor, post_target = out
                    except Exception as e:
                        logger.warning(
                            "_render_task_walker: kpt transform failed at idx %d — %s",
                            idx, e,
                        )
                cells = _walk_keypoint(
                    raw_bgr, raw_item, post_tensor,
                    _extract_target_for_panel(post_target, "keypoint")
                    if post_target is not None else None,
                    transforms, mean_list, std_list,
                    is_topdown=is_topdown_kpt,
                )
            else:
                continue
            if cells:
                columns.append(cells)
        except Exception as e:
            logger.warning("_render_task_walker: idx %d failed — %s", idx, e)
            continue

    if not columns:
        return None

    n_rows = max(len(col) for col in columns)
    n_cols = len(columns)
    ref = columns[0]

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols + 1.2, 2.6 * n_rows + 1.0),
        dpi=150,
        squeeze=False,
    )
    fig.subplots_adjust(wspace=0.04, hspace=0.30)

    for r in range(n_rows):
        step_name = ref[r]["step_name"] if r < len(ref) else f"step_{r}"
        for c, cells in enumerate(columns):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r < len(cells):
                cell = cells[r]
                # Use overlay only on non-normalize cells (false-colour
                # normalize image is not in pixel space, overlays would
                # be misleading); just imshow it.
                if cell["is_normalize"]:
                    ax.imshow(cell["image"])
                else:
                    overlay(ax, cell["image"], cell["target"],
                            class_names=class_names, style=style)
                ax.set_title(cell["meta"], fontsize=10, pad=4,
                             family="monospace")
        axes[r, 0].set_ylabel(
            f"[{r + 1:02d}] {step_name}",
            rotation=0, ha="right", va="center", labelpad=12,
            fontsize=11, fontweight="bold",
        )

    feature = data_config.get("dataset_name") or "unknown"
    fig.suptitle(
        wrap_suptitle(
            f"Transform pipeline (task={task}) — {feature} · "
            f"input {input_size[0]}×{input_size[1]}"
        ),
        y=0.998, fontsize=14, fontweight="bold",
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("_render_task_walker: saved %s (%d rows × %d cols)",
                out_path, n_rows, n_cols)
    return out_path


# ---------------------------------------------------------------------------
# Public entry points (thin wrappers — preserve names + signatures)
# ---------------------------------------------------------------------------


def render_transform_pipeline_task(
    out_path: Path,
    *,
    task: str,
    data_config: dict,
    training_config: dict,
    base_dir: str,
    class_names: dict[int, str],
    max_samples: int = 5,
    style: VizStyle | None = None,
) -> Path | None:
    """Classification / segmentation / keypoint variant of the pipeline viz.

    Thin wrapper over :func:`_render_paired_walker` — the unified shared
    paired-walker dispatcher routes non-detection tasks to
    :func:`_render_task_walker` which uses the :data:`_OVERLAYS` registry.
    """
    return _render_paired_walker(
        out_path,
        task=task,
        data_config=data_config,
        training_config=training_config,
        base_dir=base_dir,
        class_names=class_names,
        max_samples=max_samples,
        style=style,
    )


def _render_detection_walker(
    out_path: Path,
    *,
    data_config: dict,
    training_config: dict,
    base_dir: str,
    class_names: dict[int, str],
    max_samples: int = 5,
    style: VizStyle | None = None,
) -> Path | None:
    """Render ``04_transform_pipeline.png`` — N steps (rows) × K classes (cols).

    Each row is a pipeline step applied cumulatively; each column is a
    representative sample per class (first occurrence per class, up to
    ``max_samples``). Final row is ``Denormalize(Normalize)``.
    """
    import matplotlib.pyplot as plt

    from core.p05_data.detection_dataset import YOLOXDataset

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    style = style or VizStyle()
    aug_cfg = training_config.get("augmentation", {}) or {}
    # Prefer tensor_prep when present so viz reflects the authoritative contract.
    from utils.config import resolve_tensor_prep as _rtp
    _backend = (training_config.get("training") or {}).get("backend", "pytorch")
    _tp = _rtp(training_config, backend=_backend) or {}
    input_size = tuple(_tp.get("input_size") or data_config.get("input_size") or (640, 640))
    mean = list(_tp.get("mean") or data_config.get("mean") or IMAGENET_MEAN)
    std = list(_tp.get("std") or data_config.get("std") or IMAGENET_STD)
    applied_by = _tp.get("applied_by", "v2_pipeline")
    rescale_flag = bool(_tp.get("rescale", True))
    normalize_flag = bool(_tp.get("normalize", True))

    # Walker pipeline: force torchvision, strip dataset-level ops (mosaic/mixup
    # /copypaste need other items). Train-mode to exercise full aug chain.
    # The walker always runs through the v2 Normalize step (regardless of
    # applied_by) so the viz surfaces a post-Normalize snapshot + denormalize
    # sanity check. When applied_by=hf_processor, we ALSO snapshot the HF
    # processor output as an extra final cell below.
    walker_cfg = dict(aug_cfg)
    walker_cfg["library"] = "torchvision"
    walker_cfg["mosaic"] = False
    walker_cfg["mixup"] = False
    walker_cfg["copypaste"] = False
    _walker_tp = {
        "input_size": list(input_size),
        "rescale": rescale_flag,
        "normalize": normalize_flag,
        "mean": list(mean),
        "std": list(std),
        "applied_by": "v2_pipeline",
    }
    transform = build_transforms(
        config=walker_cfg, is_train=True, input_size=input_size,
        mean=mean, std=std, tensor_prep=_walker_tp,
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

    # When applied_by=hf_processor, also build the live HF processor so we can
    # snapshot its output as an extra step. Best-effort — viz skips the step
    # silently on any processor build failure.
    _hf_processor = None
    if applied_by == "hf_processor":
        try:
            from transformers import AutoImageProcessor
            model_cfg = training_config.get("model", {}) or {}
            _pretrained = model_cfg.get("pretrained") or model_cfg.get("hf_model_id")
            if _pretrained:
                _hf_processor = AutoImageProcessor.from_pretrained(_pretrained)
                _hf_processor.do_rescale = rescale_flag
                _hf_processor.do_normalize = normalize_flag
                if normalize_flag:
                    _hf_processor.image_mean = list(mean)
                    _hf_processor.image_std = list(std)
                _hf_processor.do_resize = True
                _hf_processor.size = {"height": int(input_size[0]), "width": int(input_size[1])}
        except Exception as _e:
            logger.info("transform_pipeline_viz: skipping HF-processor step (%s)", _e)
            _hf_processor = None

    # Per-column walk: one column per (class_id, idx).
    columns: list[tuple[int, list[dict]]] = []
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
            # Extra step: feed the pre-Normalize final RGB through the HF
            # processor and snapshot its pixel_values. Catches silent
            # "processor didn't actually normalize" bugs.
            if _hf_processor is not None and cells:
                try:
                    # Use the last non-normalize cell's RGB as the processor input —
                    # it matches what the model sees when applied_by=hf_processor
                    # (our v2 pipeline skips rescale+normalize in that mode).
                    pre_norm_rgb = None
                    for c in reversed(cells):
                        if not c["is_normalize"]:
                            pre_norm_rgb = c["image"]
                            break
                    if pre_norm_rgb is not None:
                        _proc_out = _hf_processor(images=pre_norm_rgb, return_tensors="pt")
                        pv = _proc_out["pixel_values"][0]
                        label = (
                            f"HF Processor (do_rescale={rescale_flag} "
                            f"do_normalize={normalize_flag} μ≈0)"
                        )
                        if normalize_flag:
                            disp = false_color_normalize(pv)
                        else:
                            disp = tensor_to_uint8_rgb(pv)
                        cells.append({
                            "image": disp,
                            "step_name": label,
                            "meta": _cell_meta(pv),
                            "is_normalize": normalize_flag,
                        })
                except Exception as _e:
                    logger.info("transform_pipeline_viz: HF-processor step failed (%s)", _e)
            columns.append((cls_id, cells))
        except Exception as e:
            logger.warning("render_transform_pipeline: class=%d idx=%d failed — %s",
                           cls_id, idx, e)
            continue

    if not columns:
        logger.warning("render_transform_pipeline: no columns rendered")
        return None

    # Rows = pipeline steps (uniform across columns since the pipeline is
    # deterministic in structure; length varies only if a transform raised).
    n_rows = max(len(cells) for _, cells in columns)
    n_cols = len(columns)

    # Use the first column's step_name sequence for row labels.
    ref_steps = columns[0][1]

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols + 1.2, 2.6 * n_rows + 0.8),
        dpi=110,
        squeeze=False,
    )
    fig.subplots_adjust(wspace=0.04, hspace=0.35)

    for r in range(n_rows):
        step_name = ref_steps[r]["step_name"] if r < len(ref_steps) else f"step_{r}"
        for c, (cls_id, cells) in enumerate(columns):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r < len(cells):
                ax.imshow(cells[r]["image"])
                # 2-line metadata caption sits directly above the image.
                title = cells[r]["meta"]
                title_pad = 6
                # Row 0: leave headroom so the separate class header
                # (placed via fig.text) does not collide with metadata.
                if r == 0:
                    title_pad = 55
                ax.set_title(title, fontsize=11, pad=title_pad,
                             family="monospace")
        # Row label on col 0 only.
        axes[r, 0].set_ylabel(
            f"[{r + 1:02d}] {step_name}",
            rotation=0, ha="right", va="center", labelpad=14,
            fontsize=13, fontweight="bold",
        )

    # Class headers above row 0 — fig.text avoids collision with per-cell
    # metadata titles inside axes[0, c].
    # Compute approximate x-center of each column from the axis bbox.
    fig.canvas.draw()  # ensure layout is realized for bbox queries
    for c, (cls_id, _cells) in enumerate(columns):
        cls_label = class_names.get(int(cls_id), f"class_{cls_id}")
        bbox = axes[0, c].get_position()
        x_center = (bbox.x0 + bbox.x1) / 2.0
        y_top = bbox.y1 + 0.012
        fig.text(
            x_center, y_top,
            f"class: {cls_label}",
            ha="center", va="bottom", fontsize=14, fontweight="bold",
        )

    feature = data_config.get("dataset_name") or "unknown"
    fig.suptitle(
        f"Transform pipeline — {feature} · input {input_size[0]}×{input_size[1]}  "
        f"mean={[round(m, 3) for m in mean]} std={[round(s, 3) for s in std]}\n"
        f"Normalize: applied by {applied_by}  (rescale={rescale_flag} "
        f"normalize={normalize_flag})",
        y=0.998, fontsize=16, fontweight="bold",
    )
    fig.text(
        0.5, 0.003,
        f"Row [{n_rows:02d}] Denormalize(Normalize) should visually match the "
        "pre-Normalize row. Color cast = bug in mean/std or processor rescale "
        "collision.",
        ha="center", fontsize=13, style="italic", color="#444",
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("render_transform_pipeline: saved %s (%d rows × %d cols)",
                out_path, n_rows, n_cols)
    return out_path


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
    """Render ``04_transform_pipeline.png`` for detection runs.

    Thin wrapper over :func:`_render_paired_walker`. Detection routes to
    :func:`_render_detection_walker` (the original per-step paired-box
    walker) — output is bit-for-bit identical to the pre-refactor
    implementation.
    """
    return _render_paired_walker(
        out_path,
        task="detection",
        data_config=data_config,
        training_config=training_config,
        base_dir=base_dir,
        class_names=class_names,
        max_samples=max_samples,
        style=style,
    )

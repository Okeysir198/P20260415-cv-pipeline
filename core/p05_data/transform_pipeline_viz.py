"""Step-by-step visualization of the CPU transform pipeline.

Renders ``<save_dir>/data_preview/05_transform_pipeline.png`` — one PNG that
walks a single sample through every stage of :class:`DetectionTransform`
(built via :func:`core.p05_data.transforms.build_transforms`) and an
augmentation gallery showing each enabled augmentation in isolation.

Primary use: verify that normalize/denormalize is an exact inverse. The
pre-normalize and post-denormalize cells must look pixel-identical.
"""

from __future__ import annotations

import copy
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
    """Invert ``v2.Normalize + ToDtype(scale=True)`` → HWC uint8 RGB.

    ``t_hwc = (tensor_chw * std + mean).clamp(0, 1) * 255 → uint8``.
    """
    t = tensor.detach().float().cpu()
    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    restored = (t * std_t + mean_t).clamp(0.0, 1.0) * 255.0
    return restored.byte().permute(1, 2, 0).contiguous().numpy()


def tensor_to_uint8_rgb(image: Any) -> np.ndarray:
    """Coerce any pipeline image representation to HWC uint8 RGB.

    Handles tv_tensors.Image / torch.Tensor (CHW, any float range) and raw
    numpy HWC arrays. Float tensors in [0, 1] are rescaled; uint8 is passed
    through. Data outside [0, 1] (post-Normalize) is clamped.
    """
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


def boxes_to_xyxy(sample: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract (xyxy pixel boxes, class ids) from a v2 sample dict."""
    boxes = sample.get("boxes")
    labels = sample.get("labels")
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    xyxy = boxes.detach().float().cpu().numpy()
    cls = labels.detach().cpu().numpy().astype(np.int64)
    return xyxy, cls


def false_color_normalize(tensor: torch.Tensor) -> np.ndarray:
    """Render a normalized CHW tensor as a jet false-color HWC uint8 image.

    Collapses channels by per-pixel mean, clips to ±3σ, maps to [0, 1], and
    applies matplotlib's ``jet`` cmap.
    """
    import matplotlib

    t = tensor.detach().float().cpu().mean(dim=0)
    clipped = t.clamp(-3.0, 3.0)
    norm01 = ((clipped + 3.0) / 6.0).numpy()
    rgba = matplotlib.colormaps["jet"](norm01)
    return (rgba[..., :3] * 255).astype(np.uint8)


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
    name = type(t).__name__
    return name


def _transform_params(t: Any) -> str:
    """Short parameter summary for the title bar."""
    if isinstance(t, v2.Resize):
        return f"size={list(t.size) if hasattr(t, 'size') else t.size}"
    if isinstance(t, v2.Normalize):
        m = [round(float(x), 3) for x in t.mean]
        s = [round(float(x), 3) for x in t.std]
        return f"mean={m} std={s}"
    if isinstance(t, v2.RandomAffine):
        return f"deg={t.degrees} scale={t.scale}"
    if isinstance(t, v2.RandomHorizontalFlip):
        return f"p={t.p}"
    if isinstance(t, v2.RandomVerticalFlip):
        return f"p={t.p}"
    if isinstance(t, v2.ColorJitter):
        return f"b={t.brightness} c={t.contrast} s={t.saturation} h={t.hue}"
    if isinstance(t, v2.ToDtype):
        return f"dtype={t.dtype} scale={t.scale}"
    return ""


def _draw_boxes_rgb(
    rgb: np.ndarray,
    xyxy: np.ndarray,
    class_ids: np.ndarray,
    class_names: dict[int, str],
    style: VizStyle,
) -> np.ndarray:
    """Draw xyxy GT boxes on an RGB uint8 image."""
    if len(xyxy) == 0:
        return rgb
    import supervision as sv

    from utils.viz import annotate_detections
    dets = sv.Detections(
        xyxy=xyxy.astype(np.float64),
        class_id=class_ids.astype(np.int64),
    )
    return annotate_detections(rgb, dets, class_names=class_names, style=style)


def _cell_title(step_idx: int, name: str, params: str) -> str:
    body = f"[{step_idx:02d}] {name}"
    return f"{body}  {params}" if params else body


def _cell_subtitle(img_repr: Any) -> str:
    """`dtype · shape · [min, max, mean]`."""
    if isinstance(img_repr, torch.Tensor):
        t = img_repr.detach().float().cpu()
        dtype = str(img_repr.dtype).replace("torch.", "")
        shape = tuple(img_repr.shape)
        mn, mx, mu = float(t.min()), float(t.max()), float(t.mean())
    else:
        arr = np.asarray(img_repr)
        dtype = str(arr.dtype)
        shape = tuple(arr.shape)
        mn, mx, mu = float(arr.min()), float(arr.max()), float(arr.mean())
    return f"{dtype} {shape} [{mn:.3f}, {mx:.3f}] μ={mu:.3f}"


# ---------------------------------------------------------------------------
# Step-walk
# ---------------------------------------------------------------------------


def _walk_pipeline(
    raw_image_bgr: np.ndarray,
    raw_targets: np.ndarray,
    detection_transform: DetectionTransform,
    mean: list[float],
    std: list[float],
    class_names: dict[int, str],
    style: VizStyle,
) -> tuple[list[dict], torch.Tensor | None, torch.Tensor | None]:
    """Walk the pipeline cumulatively, snapshot each step.

    Returns ``(cells, pre_norm_tensor, post_norm_tensor)`` — last two used
    by callers who want to verify the denorm-inverse numerically.
    ``cells`` is a list of ``{image, title, subtitle, skip_boxes}`` dicts.
    """
    _ = detection_transform.canvas_size  # reference — used by the pipeline itself
    orig_hw = (raw_image_bgr.shape[0], raw_image_bgr.shape[1])
    sample = _to_v2_sample(raw_image_bgr, raw_targets, orig_hw)

    cells: list[dict] = []

    # [00] Raw — pre-pipeline state.
    raw_rgb = cv2.cvtColor(raw_image_bgr, cv2.COLOR_BGR2RGB)
    xyxy0, cls0 = boxes_to_xyxy(sample)
    cells.append({
        "image": _draw_boxes_rgb(raw_rgb, xyxy0, cls0, class_names, style),
        "title": _cell_title(0, "Raw", f"{orig_hw[0]}x{orig_hw[1]} BGR→RGB"),
        "subtitle": f"uint8 ({orig_hw[0]}, {orig_hw[1]}, 3) [0, 255]",
        "is_false_color": False,
    })

    pre_norm_tensor: torch.Tensor | None = None
    post_norm_tensor: torch.Tensor | None = None

    for i, t in enumerate(detection_transform.transforms, start=1):
        try:
            sample = _apply_v2_transform(sample, t)
        except Exception as e:  # pragma: no cover — surfaces buggy transforms
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
                "subtitle": _cell_subtitle(img),
                "is_false_color": True,
            })
        else:
            # Snapshot the pre-normalize state — the reference for the
            # denorm round-trip check.
            if isinstance(img, torch.Tensor) and img.is_floating_point():
                pre_norm_tensor = img.clone()
            rgb = tensor_to_uint8_rgb(img)
            xyxy, cls = boxes_to_xyxy(sample)
            cells.append({
                "image": _draw_boxes_rgb(rgb, xyxy, cls, class_names, style),
                "title": _cell_title(i, name, params),
                "subtitle": _cell_subtitle(img),
                "is_false_color": False,
            })

    # [N+1] Denormalize — the inverse-round-trip cell. Only if Normalize ran.
    if post_norm_tensor is not None:
        denorm_rgb = denormalize_chw(post_norm_tensor, mean, std)
        xyxy_final, cls_final = boxes_to_xyxy(sample)
        cells.append({
            "image": _draw_boxes_rgb(denorm_rgb, xyxy_final, cls_final,
                                     class_names, style),
            "title": _cell_title(len(cells), "Denormalize (inverse)",
                                 f"mean={[round(x, 3) for x in mean]}"),
            "subtitle": _cell_subtitle(denorm_rgb),
            "is_false_color": False,
        })

    return cells, pre_norm_tensor, post_norm_tensor


# ---------------------------------------------------------------------------
# Augmentation gallery (Block B)
# ---------------------------------------------------------------------------


_GALLERY_AUGS: list[tuple[str, dict]] = [
    ("Baseline (no aug)", {
        "mosaic": False, "mixup": False, "copypaste": False, "ir_simulation": False,
        "fliplr": 0.0, "flipud": 0.0, "degrees": 0.0, "translate": 0.0,
        "shear": 0.0, "scale": [1.0, 1.0], "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
        "brightness_contrast_p": 0.0, "hsv_p": 0.0, "perspective_p": 0.0,
    }),
    ("+RandomAffine", {"degrees": 10.0, "scale": [0.8, 1.2]}),
    ("+ColorJitter", {"hsv_h": 0.015, "hsv_s": 0.4, "hsv_v": 0.3}),
    ("+HFlip (p=1)", {"fliplr": 1.0}),
    ("+VFlip (p=1)", {"flipud": 1.0}),
    ("+Perspective", {"perspective_p": 1.0, "perspective_distortion": 0.2}),
]


def _build_gallery_cells(
    raw_image_bgr: np.ndarray,
    raw_targets: np.ndarray,
    data_config: dict,
    training_config: dict,
    class_names: dict[int, str],
    style: VizStyle,
    gallery_samples: int,
    dataset,
) -> list[dict]:
    """Block B: each enabled aug in isolation + baseline + full-pipeline row."""
    aug_cfg = training_config.get("augmentation", {}) or {}
    input_size = tuple(data_config.get("input_size") or (640, 640))
    mean = data_config.get("mean") or IMAGENET_MEAN
    std = data_config.get("std") or IMAGENET_STD

    cells: list[dict] = []

    # Only include augs the user has enabled in the real config.
    enabled_overrides: list[tuple[str, dict]] = [_GALLERY_AUGS[0]]
    mapping = {
        "+RandomAffine": (aug_cfg.get("degrees", 0) > 0
                          or aug_cfg.get("scale", [1, 1]) != [1, 1]),
        "+ColorJitter": (aug_cfg.get("hsv_h", 0) > 0
                         or aug_cfg.get("hsv_s", 0) > 0
                         or aug_cfg.get("hsv_v", 0) > 0
                         or aug_cfg.get("brightness_contrast_p", 0) > 0
                         or aug_cfg.get("hsv_p", 0) > 0),
        "+HFlip (p=1)": aug_cfg.get("fliplr", 0) > 0,
        "+VFlip (p=1)": aug_cfg.get("flipud", 0) > 0,
        "+Perspective": aug_cfg.get("perspective_p", 0) > 0,
    }
    for label, override in _GALLERY_AUGS[1:]:
        if mapping.get(label, False):
            enabled_overrides.append((label, override))

    for label, override in enabled_overrides:
        # Minimal single-aug config: disable everything then set this aug.
        cfg = {
            "library": "torchvision",
            "mosaic": False, "mixup": False, "copypaste": False,
            "ir_simulation": False,
            "fliplr": 0.0, "flipud": 0.0,
            "degrees": 0.0, "translate": 0.0, "shear": 0.0,
            "scale": [1.0, 1.0],
            "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
            "brightness_contrast_p": 0.0, "hsv_p": 0.0,
            "perspective_p": 0.0,
            "normalize": False,
        }
        cfg.update(override)
        try:
            tfm = build_transforms(
                config=cfg, is_train=True, input_size=input_size,
                mean=mean, std=std,
            )
            img_t, targets_t = tfm(raw_image_bgr, raw_targets)
            rgb = tensor_to_uint8_rgb(img_t)
            H, W = rgb.shape[:2]
            xyxy, cls = _targets_to_xyxy(targets_t, H, W)
            cells.append({
                "image": _draw_boxes_rgb(rgb, xyxy, cls, class_names, style),
                "title": label,
                "subtitle": _cell_subtitle(img_t),
                "is_false_color": False,
            })
        except Exception as e:  # pragma: no cover
            logger.warning("gallery aug %s failed: %s", label, e)

    # Full-pipeline samples — N different dataset indices through the REAL
    # user pipeline (honouring whatever augmentation.library is).
    for k in range(gallery_samples):
        try:
            idx = random.randrange(len(dataset))
            raw = dataset.get_raw_item(idx)
            tfm = build_transforms(
                config=aug_cfg, is_train=True, input_size=input_size,
                mean=mean, std=std,
            )
            img_t, targets_t = tfm(raw["image"], raw["targets"])
            # Always render denormalized so the full pipeline is human-readable.
            if aug_cfg.get("normalize", True):
                rgb = denormalize_chw(img_t, mean, std)
            else:
                rgb = tensor_to_uint8_rgb(img_t)
            H, W = rgb.shape[:2]
            xyxy, cls = _targets_to_xyxy(targets_t, H, W)
            cells.append({
                "image": _draw_boxes_rgb(rgb, xyxy, cls, class_names, style),
                "title": f"Full pipeline — sample {k + 1}",
                "subtitle": _cell_subtitle(img_t),
                "is_false_color": False,
            })
        except Exception as e:  # pragma: no cover
            logger.warning("gallery full-pipeline sample %d failed: %s", k, e)

    return cells


def _targets_to_xyxy(
    targets: torch.Tensor, H: int, W: int
) -> tuple[np.ndarray, np.ndarray]:
    """YOLO-normalized (N,5) cxcywh → pixel xyxy + class_id."""
    if targets is None or len(targets) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    arr = (targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor)
           else np.asarray(targets))
    cls = arr[:, 0].astype(np.int64)
    cx, cy, bw, bh = arr[:, 1] * W, arr[:, 2] * H, arr[:, 3] * W, arr[:, 4] * H
    xyxy = np.stack(
        [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1
    ).astype(np.float32)
    return xyxy, cls


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def _compose_png(
    block_a: list[dict],
    block_b: list[dict],
    out_path: Path,
    *,
    title: str,
    cols_a: int = 5,
    cols_b: int = 4,
    cell_px: int = 360,
) -> None:
    """Stack Block A + Block B as two grids in one PNG via matplotlib."""
    import matplotlib.pyplot as plt

    def _render_block(axes_block, cells):
        for ax, cell in zip(axes_block.flat, cells, strict=False):
            ax.imshow(cell["image"])
            ax.set_title(cell["title"], fontsize=8)
            ax.set_xlabel(cell["subtitle"], fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
        # blank any leftover axes
        for ax in list(axes_block.flat)[len(cells):]:
            ax.axis("off")

    rows_a = (len(block_a) + cols_a - 1) // cols_a
    rows_b = (len(block_b) + cols_b - 1) // cols_b if block_b else 0

    fig = plt.figure(
        figsize=(cols_a * 3.5, (rows_a + rows_b) * 3.5 + 1.2),
        constrained_layout=True,
    )
    spec = fig.add_gridspec(
        rows_a + rows_b, max(cols_a, cols_b), wspace=0.05, hspace=0.3,
    )

    # Block A
    axes_a = np.empty((rows_a, cols_a), dtype=object)
    for r in range(rows_a):
        for c in range(cols_a):
            axes_a[r, c] = fig.add_subplot(spec[r, c])
    _render_block(axes_a, block_a)

    # Block B
    if block_b:
        axes_b = np.empty((rows_b, cols_b), dtype=object)
        for r in range(rows_b):
            for c in range(cols_b):
                axes_b[r, c] = fig.add_subplot(spec[rows_a + r, c])
        _render_block(axes_b, block_b)

    fig.suptitle(title, fontsize=12, y=1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_transform_pipeline(
    out_path: Path,
    *,
    dataset,
    data_config: dict,
    training_config: dict,
    base_dir: str,
    class_names: dict[int, str],
    style: VizStyle | None = None,
    gallery_samples: int = 4,
    seed: int = 0,
    _return_snapshots: bool = False,
) -> dict | None:
    """Render ``05_transform_pipeline.png`` at ``out_path``.

    Walks a single dataset sample through every CPU transform step and
    composes a step-by-step grid plus an augmentation gallery.

    Args:
        out_path: Target PNG path (parents auto-created).
        dataset: Object exposing ``__len__`` and ``get_raw_item(idx)``.
        data_config: Resolved 05_data.yaml dict.
        training_config: 06_training.yaml dict (augmentation block read).
        base_dir: Directory used when rebuilding the pipeline (unused here
            but kept for symmetry with other callbacks).
        class_names: ``{int: str}`` for GT box labels.
        style: Optional :class:`VizStyle`; defaults to the library default.
        gallery_samples: N independent samples for the "Full pipeline" row.
        seed: Deterministic RNG seed (Python/NumPy/Torch).
        _return_snapshots: When True, returns a dict with pre/post-normalize
            tensors + the denorm-round-trip L∞ (for tests).
    """
    _ = base_dir  # kept for API symmetry with other callbacks

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    style = style or VizStyle()
    aug_cfg = training_config.get("augmentation", {}) or {}
    input_size = tuple(data_config.get("input_size") or (640, 640))
    mean = list(data_config.get("mean") or IMAGENET_MEAN)
    std = list(data_config.get("std") or IMAGENET_STD)

    # Build the real transform pipeline, then walk its `.transforms` list.
    # Force torchvision backend for the walker (albumentations' pipeline is
    # opaque to step-by-step introspection).
    walker_cfg = copy.deepcopy(aug_cfg)
    walker_cfg["library"] = "torchvision"
    # Drop dataset-level ops — they need other dataset items, not a single
    # sample. Block B will show the full pipeline including them.
    walker_cfg["mosaic"] = False
    walker_cfg["mixup"] = False
    walker_cfg["copypaste"] = False
    transform = build_transforms(
        config=walker_cfg, is_train=True, input_size=input_size,
        mean=mean, std=std,
    )
    if not isinstance(transform, DetectionTransform):
        logger.warning("transform_pipeline_viz: expected DetectionTransform; got %s",
                       type(transform).__name__)
        return None

    idx = random.randrange(len(dataset))
    raw = dataset.get_raw_item(idx)

    block_a, pre_norm, post_norm = _walk_pipeline(
        raw["image"], raw["targets"], transform,
        mean=mean, std=std, class_names=class_names, style=style,
    )
    block_b = _build_gallery_cells(
        raw["image"], raw["targets"], data_config, training_config,
        class_names, style, gallery_samples, dataset,
    )

    feature = data_config.get("dataset_name") or "unknown"
    title = (f"Transform pipeline — {feature} · input {input_size[0]}x{input_size[1]} "
             f"· aug={aug_cfg.get('library', 'torchvision')}")

    _compose_png(block_a, block_b, Path(out_path), title=title)
    logger.info("TransformPipelineCallback: saved %s (A=%d cells, B=%d cells)",
                out_path, len(block_a), len(block_b))

    if _return_snapshots:
        l_inf = None
        if pre_norm is not None and post_norm is not None:
            denorm = denormalize_chw(post_norm, mean, std)
            pre_rgb = tensor_to_uint8_rgb(pre_norm)
            if pre_rgb.shape == denorm.shape:
                l_inf = int(np.abs(pre_rgb.astype(np.int32)
                                   - denorm.astype(np.int32)).max())
        return {
            "pre_norm_tensor": pre_norm,
            "post_norm_tensor": post_norm,
            "block_a_len": len(block_a),
            "block_b_len": len(block_b),
            "denorm_l_inf": l_inf,
        }
    return None

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
        figsize=(3.2 * n_cols + 1.2, 3.8 * n_rows + 0.8),
        dpi=130,
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
            rotation=0, ha="right", va="center", labelpad=60,
            fontsize=15, fontweight="bold",
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

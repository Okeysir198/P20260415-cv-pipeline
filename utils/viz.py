"""Single source of truth for all image annotation (supervision-based) and
matplotlib plot styling.

Importing cv2 drawing primitives (``cv2.rectangle / putText / circle / line /
polylines / fillPoly / addWeighted``) elsewhere is a regression — use the
helpers in this module instead.

The helpers wrap ``supervision`` v0.27.x annotators and expose them with
pipeline-friendly signatures (numpy RGB in, numpy RGB out). Style is
controlled by :class:`VizStyle`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import supervision as sv

from core.p10_inference.supervision_bridge import (
    VizStyle,
    annotate_frame,
    annotate_gt_pred,
    build_annotators,
    build_labels,
    create_tracker,
    from_sv_detections,
    to_sv_detections,
    update_tracker,
)

__all__ = [
    # Re-exports
    "VizStyle",
    "to_sv_detections",
    "from_sv_detections",
    "build_annotators",
    "build_labels",
    "annotate_frame",
    "annotate_gt_pred",
    "create_tracker",
    "update_tracker",
    # New helpers
    "annotate_detections",
    "annotate_keypoints",
    "annotate_polygons",
    "classification_banner",
    "save_image_grid",
    "apply_plot_style",
    # Constants
    "COCO_SKELETON_EDGES",
    "PLOT_COLOR_CYCLE",
]


# Standard COCO-17 keypoint skeleton.
# Indices: 0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
# 5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
# 9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
# 13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle.
COCO_SKELETON_EDGES: list[tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),     # head
    (5, 6),                              # shoulders
    (5, 7), (7, 9),                      # left arm
    (6, 8), (8, 10),                     # right arm
    (5, 11), (6, 12), (11, 12),          # torso
    (11, 13), (13, 15),                  # left leg
    (12, 14), (14, 16),                  # right leg
    (0, 5), (0, 6),                      # nose-to-shoulders
]


# Hex colors derived from sv.ColorPalette.DEFAULT, so matplotlib plots and
# supervision image overlays share colors per class_id.
PLOT_COLOR_CYCLE: list[str] = [
    f"#{c.r:02x}{c.g:02x}{c.b:02x}" for c in sv.ColorPalette.DEFAULT.colors
]


def _default_style(style: VizStyle | None) -> VizStyle:
    return style if style is not None else VizStyle()


# ---------------------------------------------------------------------------
# Detections
# ---------------------------------------------------------------------------


def annotate_detections(
    image: np.ndarray,
    detections: sv.Detections,
    class_names: dict[int, str] | None = None,
    labels: list[str] | None = None,
    style: VizStyle | None = None,
    color: sv.Color | None = None,
) -> np.ndarray:
    """Draw boxes + labels on an RGB image.

    Replaces ad-hoc ``cv2.rectangle + cv2.putText`` blocks. Uses supervision's
    :class:`BoxAnnotator` and :class:`LabelAnnotator`; falls back to
    class-based color lookup when ``color`` is None.
    """
    style = _default_style(style)
    h, w = image.shape[:2]
    thickness = style.auto_box_thickness(h, w)

    box_kwargs: dict[str, Any] = {"thickness": thickness}
    label_kwargs: dict[str, Any] = {
        "text_scale": style.label_text_scale,
        "text_padding": style.label_text_padding,
        "text_position": style.sv_label_position(),
    }
    if color is not None:
        box_kwargs["color"] = color
        label_kwargs["color"] = color
    else:
        box_kwargs["color"] = style.class_palette()
        label_kwargs["color"] = style.class_palette()

    box_ann = sv.BoxAnnotator(**box_kwargs)
    lbl_ann = sv.LabelAnnotator(**label_kwargs)

    scene = image.copy()
    scene = box_ann.annotate(scene=scene, detections=detections)

    if labels is None and class_names is not None:
        labels = build_labels(detections, class_names)
    if labels is None:
        labels = [""] * len(detections)

    if len(detections) > 0:
        scene = lbl_ann.annotate(scene=scene, detections=detections, labels=labels)
    return scene


# ---------------------------------------------------------------------------
# Keypoints
# ---------------------------------------------------------------------------


def annotate_keypoints(
    image: np.ndarray,
    keypoints_xy: np.ndarray,
    skeleton_edges: list[tuple[int, int]] | None = None,
    confidence: np.ndarray | None = None,
    style: VizStyle | None = None,
    color: sv.Color | None = None,
) -> np.ndarray:
    """Draw keypoint vertices + skeleton edges using supervision annotators.

    Low-confidence points (below ``style.kpt_visibility_threshold``) are
    hidden by zeroing their coordinates before construction of
    :class:`sv.KeyPoints`.
    """
    style = _default_style(style)
    h, w = image.shape[:2]

    xy = np.asarray(keypoints_xy, dtype=np.float32)
    if xy.ndim == 2:
        xy = xy[None, ...]  # → (1, K, 2)
    if xy.size == 0:
        return image.copy()

    conf = None
    if confidence is not None:
        conf = np.asarray(confidence, dtype=np.float32)
        if conf.ndim == 1:
            conf = conf[None, ...]
        # hide low-conf points
        mask = conf < style.kpt_visibility_threshold
        xy = xy.copy()
        xy[mask] = 0.0

    keypoints = sv.KeyPoints(xy=xy, confidence=conf)

    draw_color = color if color is not None else sv.Color(
        r=style.skeleton_color_rgb[0],
        g=style.skeleton_color_rgb[1],
        b=style.skeleton_color_rgb[2],
    )

    vertex_ann = sv.VertexAnnotator(
        color=draw_color,
        radius=style.auto_keypoint_radius(h, w),
    )
    edge_ann = sv.EdgeAnnotator(
        color=draw_color,
        thickness=style.auto_skeleton_thickness(h, w),
        edges=skeleton_edges,
    )

    scene = image.copy()
    if skeleton_edges:
        scene = edge_ann.annotate(scene=scene, key_points=keypoints)
    scene = vertex_ann.annotate(scene=scene, key_points=keypoints)
    return scene


# ---------------------------------------------------------------------------
# Polygons
# ---------------------------------------------------------------------------


def annotate_polygons(
    image: np.ndarray,
    polygons: list[np.ndarray],
    labels: list[str] | None = None,
    style: VizStyle | None = None,
    color: sv.Color | None = None,
) -> np.ndarray:
    """Draw polygon outlines and translucent fill.

    Replaces ``cv2.polylines + cv2.fillPoly + cv2.addWeighted`` combos.
    Each polygon is rasterized to a mask and merged into a single
    :class:`sv.Detections` for annotation.
    """
    style = _default_style(style)
    h, w = image.shape[:2]
    scene = image.copy()

    if not polygons:
        return scene

    masks = np.zeros((len(polygons), h, w), dtype=bool)
    xyxy = np.zeros((len(polygons), 4), dtype=np.float32)
    for i, poly in enumerate(polygons):
        pts = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
        if pts.size == 0:
            continue
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        masks[i] = mask.astype(bool)
        xyxy[i] = [pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()]

    class_id = np.zeros(len(polygons), dtype=int)
    dets = sv.Detections(xyxy=xyxy, mask=masks, class_id=class_id)

    draw_color = color if color is not None else style.class_palette()
    outline = sv.PolygonAnnotator(color=draw_color, thickness=style.zone_outline_thickness)
    fill = sv.ColorAnnotator(color=draw_color, opacity=style.zone_fill_alpha)

    scene = fill.annotate(scene=scene, detections=dets)
    scene = outline.annotate(scene=scene, detections=dets)

    if labels:
        lbl_ann = sv.LabelAnnotator(
            color=draw_color if isinstance(draw_color, sv.Color) else style.class_palette(),
            text_scale=style.label_text_scale,
            text_padding=style.label_text_padding,
            text_position=style.sv_label_position(),
        )
        scene = lbl_ann.annotate(scene=scene, detections=dets, labels=labels)

    return scene


# ---------------------------------------------------------------------------
# Classification banner
# ---------------------------------------------------------------------------


def classification_banner(
    image: np.ndarray,
    text: str,
    style: VizStyle | None = None,
    position: str = "top",
    bg_color_rgb: tuple[int, int, int] | None = None,
    text_color_rgb: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Return a copy of ``image`` with a horizontal banner stacked above or below.

    Replaces ad-hoc ``cv2.putText``-on-bar blocks. ``position='top'`` or
    ``'bottom'`` stacks a new banner strip; ``'overlay_top'`` draws the banner
    on top of the image in-place (no height change).
    """
    style = _default_style(style)
    h, w = image.shape[:2]
    bh = int(style.banner_height)
    bg = bg_color_rgb if bg_color_rgb is not None else style.banner_bg_rgb
    fg = text_color_rgb if text_color_rgb is not None else style.banner_text_rgb

    banner = np.full((bh, w, 3), bg, dtype=np.uint8)
    # Text is drawn in RGB; cv2.putText treats triples opaquely, so passing
    # RGB in and reading RGB out is consistent (no cvtColor needed).
    cv2.putText(
        banner,
        text,
        (6, int(bh * 0.7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        style.banner_text_scale,
        tuple(int(x) for x in fg),
        1,
        cv2.LINE_AA,
    )

    if position == "top":
        return np.vstack([banner, image])
    if position == "bottom":
        return np.vstack([image, banner])
    if position == "overlay_top":
        out = image.copy()
        out[:bh, :, :] = banner
        return out
    raise ValueError(f"Unknown banner position: {position!r}")


# ---------------------------------------------------------------------------
# Image grid
# ---------------------------------------------------------------------------


def _letterbox(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def save_image_grid(
    images: list[np.ndarray],
    out_path: str | Path,
    cols: int | None = None,
    titles: list[str] | None = None,
    header: str | None = None,
    style: VizStyle | None = None,
) -> None:
    """Save an N-image grid as PNG using matplotlib for layout.

    Each cell is letterbox-resized to ``style.grid_cell_size``. Consolidates
    the two ``_save_image_grid`` helpers previously duplicated in
    ``callbacks.py`` and ``run_viz.py``.
    """
    import matplotlib.pyplot as plt

    style = _default_style(style)
    cols = int(cols or style.grid_cols)
    n = len(images)
    if n == 0:
        return
    rows = (n + cols - 1) // cols

    cell = style.grid_cell_size
    resized = [_letterbox(img, cell) for img in images]

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * 3, rows * 3 + (0.4 if header else 0)),
        squeeze=False,
    )
    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        ax.set_xticks([])
        ax.set_yticks([])
        if idx < n:
            ax.imshow(resized[idx])
            if titles is not None and idx < len(titles):
                ax.set_title(titles[idx], fontsize=9)
        else:
            ax.axis("off")

    if header:
        fig.suptitle(header, fontsize=12)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------


_PLOT_STYLE_APPLIED = False


def apply_plot_style() -> None:
    """Configure matplotlib rcParams once. Idempotent.

    Does NOT set ``legend.loc`` — call sites decide (outside-plot is a
    per-figure choice).
    """
    global _PLOT_STYLE_APPLIED
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "font.family": "DejaVu Sans",
        "axes.prop_cycle": mpl.cycler(color=PLOT_COLOR_CYCLE),
    })
    _PLOT_STYLE_APPLIED = True

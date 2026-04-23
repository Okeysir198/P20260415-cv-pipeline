"""Bridge between detection pipeline predictions and supervision/trackers ecosystem.

Converts predictor output (boxes, scores, labels) to/from sv.Detections,
builds annotator instances from config, and provides a single-call annotate function.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import supervision as sv

if TYPE_CHECKING:
    from trackers import ByteTrackTracker

# `trackers` is only used by create_tracker/update_tracker (video inference).
# Imported lazily so training workflows that only need annotate_gt_pred
# don't require the trackers package.

# Allow imports from pipeline root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Conversion: pipeline ↔ sv.Detections
# ---------------------------------------------------------------------------


def to_sv_detections(predictions: dict[str, Any]) -> sv.Detections:
    """Convert predictor output to sv.Detections.

    Args:
        predictions: Dict with ``boxes`` (N,4), ``scores`` (N,),
            ``labels`` (N,) as returned by predictor.predict().

    Returns:
        ``sv.Detections`` instance.
    """
    boxes = np.asarray(predictions["boxes"], dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(predictions["scores"], dtype=np.float32).ravel()
    labels = np.asarray(predictions["labels"], dtype=np.int64).ravel()

    return sv.Detections(
        xyxy=boxes,
        confidence=scores,
        class_id=labels.astype(int),
    )


def from_sv_detections(
    detections: sv.Detections,
    class_names: dict[int, str],
) -> dict[str, Any]:
    """Convert sv.Detections back to predictor output format.

    Args:
        detections: ``sv.Detections`` instance.
        class_names: Mapping from class ID to display name.

    Returns:
        Dict matching predictor output contract:
        ``boxes``, ``scores``, ``labels``, ``class_names``.
    """
    boxes = detections.xyxy.astype(np.float32)
    scores = (
        detections.confidence.astype(np.float32)
        if detections.confidence is not None
        else np.ones(len(detections), dtype=np.float32)
    )
    labels = (
        detections.class_id.astype(np.int64)
        if detections.class_id is not None
        else np.zeros(len(detections), dtype=np.int64)
    )
    names = [class_names.get(int(lbl), str(int(lbl))) for lbl in labels]

    return {
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
        "class_names": names,
    }


# ---------------------------------------------------------------------------
# Annotator builders
# ---------------------------------------------------------------------------


def build_annotators(config: dict | None = None) -> dict[str, Any]:
    """Create supervision annotator instances from config.

    Args:
        config: Optional dict with ``supervision`` key containing
            ``bbox``, ``label``, ``trace`` sub-dicts. Uses defaults if None.

    Returns:
        Dict with keys ``box``, ``label``, ``trace``, ``heatmap``
        mapping to annotator instances.
    """
    sv_config = (config or {}).get("supervision", {})
    bbox_cfg = sv_config.get("bbox", {})
    label_cfg = sv_config.get("label", {})
    trace_cfg = sv_config.get("trace", {})

    color_lookup_str = bbox_cfg.get("color_lookup", "CLASS")
    color_lookup = getattr(sv.ColorLookup, color_lookup_str, sv.ColorLookup.CLASS)

    return {
        "box": sv.BoxAnnotator(
            thickness=bbox_cfg.get("thickness", 2),
            color_lookup=color_lookup,
        ),
        "label": sv.LabelAnnotator(
            text_scale=label_cfg.get("text_scale", 0.5),
            text_thickness=label_cfg.get("text_thickness", 1),
            text_padding=label_cfg.get("text_padding", 5),
            color_lookup=color_lookup,
        ),
        "trace": sv.TraceAnnotator(
            trace_length=trace_cfg.get("trace_length", 60),
            thickness=trace_cfg.get("thickness", 2),
            color_lookup=color_lookup,
        ),
        "heatmap": sv.HeatMapAnnotator(
            opacity=0.5,
            kernel_size=25,
        ),
    }


def build_labels(
    detections: sv.Detections,
    class_names: dict[int, str],
) -> list[str]:
    """Build label strings for each detection.

    Format: ``"class_name 0.85"`` or ``"class_name #5 0.85"`` if tracker_id present.

    Args:
        detections: ``sv.Detections`` instance.
        class_names: Mapping from class ID to display name.

    Returns:
        List of label strings, one per detection.
    """
    labels = []
    for i in range(len(detections)):
        cls_id = int(detections.class_id[i]) if detections.class_id is not None else 0
        name = class_names.get(cls_id, str(cls_id))
        conf = (
            f" {detections.confidence[i]:.2f}"
            if detections.confidence is not None
            else ""
        )
        track = ""
        if detections.tracker_id is not None and detections.tracker_id[i] is not None:
            track = f" #{int(detections.tracker_id[i])}"
        labels.append(f"{name}{track}{conf}")
    return labels


# ---------------------------------------------------------------------------
# Single-call annotation
# ---------------------------------------------------------------------------


def annotate_frame(
    frame: np.ndarray,
    detections: sv.Detections,
    class_names: dict[int, str],
    annotators: dict[str, Any],
    draw_traces: bool = False,
    draw_heatmap: bool = False,
) -> np.ndarray:
    """Apply supervision annotators to a frame.

    Args:
        frame: BGR image (H, W, 3). A copy is made internally.
        detections: ``sv.Detections`` instance.
        class_names: Mapping from class ID to display name.
        annotators: Dict from :func:`build_annotators`.
        draw_traces: If True, draw movement traces (requires tracker_id).
        draw_heatmap: If True, overlay heatmap.

    Returns:
        Annotated BGR image (copy of original).
    """
    annotated = frame.copy()
    labels = build_labels(detections, class_names)

    annotated = annotators["box"].annotate(scene=annotated, detections=detections)
    annotated = annotators["label"].annotate(
        scene=annotated, detections=detections, labels=labels
    )
    if draw_traces and detections.tracker_id is not None:
        annotated = annotators["trace"].annotate(scene=annotated, detections=detections)
    if draw_heatmap:
        annotated = annotators["heatmap"].annotate(scene=annotated, detections=detections)

    return annotated


# ---------------------------------------------------------------------------
# GT + Prediction overlay (shared by training callbacks and inference)
# ---------------------------------------------------------------------------

_DEFAULT_GT_COLOR = sv.Color(r=160, g=32, b=240)    # purple
_DEFAULT_PRED_COLOR = sv.Color(r=0, g=200, b=0)      # green


# ---------------------------------------------------------------------------
# VizStyle — single source of truth for GT-vs-Pred rendering across callbacks,
# error-analysis, and post-train artifacts. Consumed by annotate_gt_pred.
# ---------------------------------------------------------------------------


_DEFAULT_ERROR_COLORS: dict[str, tuple[int, int, int]] = {
    "tp":        (46, 204, 113),    # #2ECC71 green
    "fp":        (231, 76, 60),     # #E74C3C red
    "fn":        (243, 156, 18),    # #F39C12 orange
    "duplicate": (155, 89, 182),    # #9B59B6 purple
    "bg_fp":     (52, 73, 94),      # #34495E dark
}


_POSITION_MAP: dict[str, Any] = {
    "top_left": sv.Position.TOP_LEFT,
    "top_right": sv.Position.TOP_RIGHT,
    "bottom_left": sv.Position.BOTTOM_LEFT,
    "bottom_right": sv.Position.BOTTOM_RIGHT,
    "top_center": sv.Position.TOP_CENTER,
    "bottom_center": sv.Position.BOTTOM_CENTER,
    "center": sv.Position.CENTER,
}


@dataclass
class VizStyle:
    """Shared visual language for every GT-vs-Pred rendering site.

    Built once per run from ``config["visualization"]`` via
    :meth:`VizStyle.from_config` and threaded through the post-train runner,
    callbacks, and error-analysis galleries so artifacts are byte-consistent.

    Colors are RGB tuples; supervision internals convert to BGR at draw time.
    """

    gt_color_rgb: tuple[int, int, int] = (160, 32, 240)    # purple
    pred_color_rgb: tuple[int, int, int] = (0, 200, 0)     # green
    gt_thickness: int = 2
    pred_thickness: int = 1
    text_scale: float = 0.4
    mask_alpha: float = 0.5             # segmentation overlay opacity
    skeleton_color_rgb: tuple[int, int, int] = (0, 140, 255)  # keypoint
    draw_legend: bool = True

    # --- Image annotation -------------------------------------------------
    palette: str = "default"
    box_thickness: int | None = None
    label_text_scale: float = 0.5
    label_text_padding: int = 4
    label_position: str = "top_left"
    keypoint_radius: int | None = None
    kpt_visibility_threshold: float = 0.3
    skeleton_edge_thickness: int | None = None
    zone_fill_alpha: float = 0.20
    zone_outline_thickness: int = 2

    # --- Error palette ----------------------------------------------------
    error_colors_rgb: dict[str, tuple[int, int, int]] = field(
        default_factory=lambda: dict(_DEFAULT_ERROR_COLORS)
    )

    # --- Grid layout ------------------------------------------------------
    grid_cell_size: int = 512
    grid_cols: int = 4
    grid_gutter_px: int = 4
    grid_title_template: str = "{feature} · epoch {epoch}/{total} · mAP50={map50:.3f}"

    # --- Banner -----------------------------------------------------------
    banner_height: int = 24
    banner_bg_rgb: tuple[int, int, int] = (34, 34, 34)
    banner_text_rgb: tuple[int, int, int] = (255, 255, 255)
    banner_text_scale: float = 0.45

    @property
    def gt_color(self) -> sv.Color:
        r, g, b = self.gt_color_rgb
        return sv.Color(r=r, g=g, b=b)

    @property
    def pred_color(self) -> sv.Color:
        r, g, b = self.pred_color_rgb
        return sv.Color(r=r, g=g, b=b)

    def auto_box_thickness(self, h: int, w: int) -> int:
        if self.box_thickness is not None:
            return int(self.box_thickness)
        return max(2, round(min(h, w) / 400))

    def auto_keypoint_radius(self, h: int, w: int) -> int:
        if self.keypoint_radius is not None:
            return int(self.keypoint_radius)
        return max(3, round(min(h, w) / 250))

    def auto_skeleton_thickness(self, h: int, w: int) -> int:
        if self.skeleton_edge_thickness is not None:
            return int(self.skeleton_edge_thickness)
        return max(2, round(min(h, w) / 500))

    def sv_label_position(self) -> Any:
        return _POSITION_MAP.get(self.label_position, sv.Position.TOP_LEFT)

    def error_color(self, kind: str) -> sv.Color:
        r, g, b = self.error_colors_rgb.get(kind, (128, 128, 128))
        return sv.Color(r=int(r), g=int(g), b=int(b))

    def class_palette(self) -> sv.ColorPalette:
        return sv.ColorPalette.DEFAULT

    @classmethod
    def from_config(cls, config: dict | None) -> "VizStyle":
        """Build a VizStyle from the top-level ``visualization:`` block.

        Accepts None or missing block → returns defaults. Unknown keys are
        ignored so forward-compat additions don't break older configs.
        ``viz:`` is accepted as an alias for ``visualization:``.
        """
        if not config:
            return cls()
        if isinstance(config, dict):
            viz = config.get("visualization") or config.get("viz") or {}
        else:
            viz = {}
        kwargs: dict = {}
        # RGB color tuples
        for key in (
            "gt_color_rgb", "pred_color_rgb", "skeleton_color_rgb",
            "banner_bg_rgb", "banner_text_rgb",
        ):
            if key in viz:
                kwargs[key] = tuple(int(x) for x in viz[key])
        # Integer fields
        for key in (
            "gt_thickness", "pred_thickness", "label_text_padding",
            "zone_outline_thickness", "grid_cell_size", "grid_cols",
            "grid_gutter_px", "banner_height",
        ):
            if key in viz:
                kwargs[key] = int(viz[key])
        # Optional integer fields (allow None)
        for key in ("box_thickness", "keypoint_radius", "skeleton_edge_thickness"):
            if key in viz:
                kwargs[key] = None if viz[key] is None else int(viz[key])
        # Float fields
        for key in (
            "text_scale", "mask_alpha", "label_text_scale",
            "kpt_visibility_threshold", "zone_fill_alpha", "banner_text_scale",
        ):
            if key in viz:
                kwargs[key] = float(viz[key])
        # String fields
        for key in ("palette", "label_position", "grid_title_template"):
            if key in viz:
                kwargs[key] = str(viz[key])
        if "draw_legend" in viz:
            kwargs["draw_legend"] = bool(viz["draw_legend"])
        if "error_colors_rgb" in viz and isinstance(viz["error_colors_rgb"], dict):
            kwargs["error_colors_rgb"] = {
                str(k): tuple(int(x) for x in v)
                for k, v in viz["error_colors_rgb"].items()
            }
        return cls(**kwargs)


def annotate_gt_pred(
    image: np.ndarray,
    gt_xyxy: np.ndarray | None,
    gt_class_ids: np.ndarray | None,
    pred_dets: sv.Detections,
    class_names: dict[int, str],
    gt_color: sv.Color = _DEFAULT_GT_COLOR,
    pred_color: sv.Color = _DEFAULT_PRED_COLOR,
    gt_thickness: int = 2,
    pred_thickness: int = 1,
    text_scale: float = 0.4,
    draw_legend: bool = True,
    style: VizStyle | None = None,
) -> np.ndarray:
    """Annotate a BGR image with GT boxes (solid, thick) and pred boxes (solid, thin).

    GT boxes are drawn first; predictions are drawn on top.
    GT labels appear at BOTTOM_LEFT; pred labels appear at TOP_LEFT.

    Args:
        image: BGR image (H, W, 3).
        gt_xyxy: Ground-truth boxes in pixel xyxy format, shape (N, 4), or None.
        gt_class_ids: Ground-truth class IDs, shape (N,), or None.
        pred_dets: Predicted detections as ``sv.Detections`` (pixel coords).
        class_names: Mapping from class ID to display name.
        gt_color: Supervision Color for GT boxes and labels.
        pred_color: Supervision Color for prediction boxes and labels.
        gt_thickness: Line thickness for GT boxes.
        pred_thickness: Line thickness for prediction boxes.
        text_scale: Font scale for all labels.
        draw_legend: Whether to draw a small legend in the top-left corner.

    Returns:
        Annotated BGR image (copy).
    """
    # VizStyle, when provided, overrides the legacy per-arg defaults.
    # Kept arg compat so existing call sites don't need immediate migration.
    if style is not None:
        gt_color = style.gt_color
        pred_color = style.pred_color
        gt_thickness = style.gt_thickness
        pred_thickness = style.pred_thickness
        text_scale = style.text_scale
        draw_legend = style.draw_legend

    gt_box_ann = sv.BoxAnnotator(color=gt_color, thickness=gt_thickness)
    gt_lbl_ann = sv.LabelAnnotator(
        color=gt_color, text_scale=text_scale, text_thickness=1,
        text_position=sv.Position.BOTTOM_LEFT,
    )
    pred_box_ann = sv.BoxAnnotator(color=pred_color, thickness=pred_thickness)
    pred_lbl_ann = sv.LabelAnnotator(
        color=pred_color, text_scale=text_scale, text_thickness=1,
        text_position=sv.Position.TOP_LEFT,
    )

    annotated = image.copy()

    # GT
    if gt_xyxy is not None and len(gt_xyxy) > 0 and gt_class_ids is not None:
        gt_dets = sv.Detections(xyxy=gt_xyxy.astype(np.float64), class_id=gt_class_ids)
        gt_labels = [
            f"GT:{int(c)} {class_names.get(int(c), str(int(c)))}" for c in gt_class_ids
        ]
        annotated = gt_box_ann.annotate(scene=annotated, detections=gt_dets)
        annotated = gt_lbl_ann.annotate(scene=annotated, detections=gt_dets, labels=gt_labels)

    # Predictions
    if len(pred_dets) > 0:
        pred_labels = [
            f"P:{int(pred_dets.class_id[i])} {class_names.get(int(pred_dets.class_id[i]), str(int(pred_dets.class_id[i])))} "
            f"{pred_dets.confidence[i]:.2f}" if pred_dets.confidence is not None
            else f"P:{int(pred_dets.class_id[i])} {class_names.get(int(pred_dets.class_id[i]), str(int(pred_dets.class_id[i])))}"
            for i in range(len(pred_dets))
        ]
        annotated = pred_box_ann.annotate(scene=annotated, detections=pred_dets)
        annotated = pred_lbl_ann.annotate(scene=annotated, detections=pred_dets, labels=pred_labels)

    if draw_legend:
        annotated = draw_gt_pred_legend(annotated, gt_color.as_bgr(), pred_color.as_bgr())

    return annotated


def draw_gt_pred_legend(image: np.ndarray, gt_bgr: tuple, pred_bgr: tuple) -> np.ndarray:
    """Overlay a legend: GT (thick solid), Pred (thin solid) with black backing
    so the swatches are readable on both light and dark images.
    """
    import cv2
    img = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    pad, line_len, w_bar = 6, 34, 80
    y0 = 10
    # Semi-transparent black backing for readability
    overlay = img.copy()
    cv2.rectangle(overlay, (pad - 3, y0 - 4), (pad + line_len + w_bar, y0 + 32),
                  (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
    # GT — thick line
    cv2.line(img, (pad, y0 + 6), (pad + line_len, y0 + 6), gt_bgr, 3, cv2.LINE_AA)
    cv2.putText(img, "GT (thick)", (pad + line_len + 4, y0 + 10), font, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    # Pred — thin line
    cv2.line(img, (pad, y0 + 22), (pad + line_len, y0 + 22), pred_bgr, 1, cv2.LINE_AA)
    cv2.putText(img, "Pred (thin)", (pad + line_len + 4, y0 + 26), font, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# Tracker helpers
# ---------------------------------------------------------------------------


def create_tracker(config: dict | None = None) -> "ByteTrackTracker":
    """Create a ByteTrackTracker from config.

    Args:
        config: Optional dict containing ByteTrack params (track_activation_threshold, etc.).

    Returns:
        ``ByteTrackTracker`` instance.
    """
    from trackers import ByteTrackTracker
    tracker_cfg = config or {}

    return ByteTrackTracker(
        track_activation_threshold=tracker_cfg.get("track_activation_threshold", 0.25),
        lost_track_buffer=tracker_cfg.get("lost_track_buffer", 30),
        minimum_iou_threshold=tracker_cfg.get("minimum_iou_threshold", 0.1),
        frame_rate=tracker_cfg.get("frame_rate", 30),
    )


def update_tracker(
    tracker: "ByteTrackTracker",
    detections: sv.Detections,
) -> sv.Detections:
    """Run tracker update and return detections with tracker_id populated.

    Args:
        tracker: Tracker instance from :func:`create_tracker`.
        detections: ``sv.Detections`` from current frame.

    Returns:
        Updated ``sv.Detections`` with ``tracker_id`` field set.
    """
    return tracker.update(detections)

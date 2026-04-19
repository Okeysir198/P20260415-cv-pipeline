"""Bridge between detection pipeline predictions and supervision/trackers ecosystem.

Converts predictor output (boxes, scores, labels) to/from sv.Detections,
builds annotator instances from config, and provides a single-call annotate function.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import supervision as sv

# `trackers` is only used by create_tracker/update_tracker (video inference).
# Imported lazily so training workflows that only need annotate_gt_pred
# don't require the trackers package.

# Allow imports from pipeline root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Conversion: pipeline ↔ sv.Detections
# ---------------------------------------------------------------------------


def to_sv_detections(predictions: Dict[str, Any]) -> sv.Detections:
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
    class_names: Dict[int, str],
) -> Dict[str, Any]:
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


def build_annotators(config: Optional[Dict] = None) -> Dict[str, Any]:
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
    class_names: Dict[int, str],
) -> List[str]:
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
    class_names: Dict[int, str],
    annotators: Dict[str, Any],
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


def annotate_gt_pred(
    image: np.ndarray,
    gt_xyxy: Optional[np.ndarray],
    gt_class_ids: Optional[np.ndarray],
    pred_dets: sv.Detections,
    class_names: Dict[int, str],
    gt_color: sv.Color = _DEFAULT_GT_COLOR,
    pred_color: sv.Color = _DEFAULT_PRED_COLOR,
    gt_thickness: int = 2,
    pred_thickness: int = 1,
    text_scale: float = 0.4,
    draw_legend: bool = True,
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
    """Overlay a small legend: GT color solid thick, Pred color solid thin.

    Args:
        image: BGR image to draw on (modified in-place on a copy).
        gt_bgr: GT box color as BGR tuple.
        pred_bgr: Pred box color as BGR tuple.

    Returns:
        Image with legend drawn.
    """
    import cv2
    img = image.copy()
    font, margin, line_len, y = cv2.FONT_HERSHEY_SIMPLEX, 6, 30, 16
    cv2.line(img, (margin, y), (margin + line_len, y), gt_bgr, 2, cv2.LINE_AA)
    cv2.putText(img, "GT", (margin + line_len + 4, y + 4), font, 0.4, gt_bgr, 1)
    cv2.line(img, (margin, y + 18), (margin + line_len, y + 18), pred_bgr, 1, cv2.LINE_AA)
    cv2.putText(img, "Pred", (margin + line_len + 4, y + 22), font, 0.4, pred_bgr, 1)
    return img


# ---------------------------------------------------------------------------
# Tracker helpers
# ---------------------------------------------------------------------------


def create_tracker(config: Optional[Dict] = None) -> "ByteTrackTracker":
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

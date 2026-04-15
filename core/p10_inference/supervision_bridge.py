"""Bridge between detection pipeline predictions and supervision/trackers ecosystem.

Converts predictor output (boxes, scores, labels) to/from sv.Detections,
builds annotator instances from config, and provides a single-call annotate function.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

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
# Tracker helpers
# ---------------------------------------------------------------------------


def create_tracker(config: Optional[Dict] = None) -> ByteTrackTracker:
    """Create a ByteTrackTracker from config.

    Args:
        config: Optional dict containing ByteTrack params (track_activation_threshold, etc.).

    Returns:
        ``ByteTrackTracker`` instance.
    """
    tracker_cfg = config or {}

    return ByteTrackTracker(
        track_activation_threshold=tracker_cfg.get("track_activation_threshold", 0.25),
        lost_track_buffer=tracker_cfg.get("lost_track_buffer", 30),
        minimum_iou_threshold=tracker_cfg.get("minimum_iou_threshold", 0.1),
        frame_rate=tracker_cfg.get("frame_rate", 30),
    )


def update_tracker(
    tracker: ByteTrackTracker,
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

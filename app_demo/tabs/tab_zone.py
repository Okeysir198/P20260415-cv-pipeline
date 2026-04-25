"""Zone Intrusion Detection tab -- define restricted zones, detect intrusions.

Works fully with COCO pretrained model (person = class 0). Uses
``sv.PolygonZone`` for in-zone checks and ``sv.PolygonZoneAnnotator`` for
drawing zones. Provides Image + Video sub-tabs with optional ByteTrack
tracking.
"""

import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import supervision as sv
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from app_demo.utils import bgr_to_rgb, rgb_to_bgr
from core.p10_inference.supervision_bridge import (
    create_tracker,
    update_tracker,
)

_COCO_PERSON_ID = 0

# Module-level zone storage (simple state for demo)
_zones: List[np.ndarray] = []


# ---------------------------------------------------------------------------
# Zone helpers
# ---------------------------------------------------------------------------


def _parse_zone(text: str) -> np.ndarray:
    """Parse ``'x1,y1 x2,y2 x3,y3 ...'`` into a numpy polygon array.

    Args:
        text: Whitespace-separated ``x,y`` pairs.

    Returns:
        Integer polygon array of shape ``(N, 2)``.

    Raises:
        ValueError: If the text cannot be parsed into valid coordinates.
    """
    points: List[List[int]] = []
    for pair in text.strip().split():
        parts = pair.split(",")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid coordinate pair: '{pair}'. Expected 'x,y'."
            )
        x, y = int(parts[0].strip()), int(parts[1].strip())
        points.append([x, y])
    if len(points) < 3:
        raise ValueError("A zone polygon requires at least 3 points.")
    return np.array(points, dtype=np.int32)


def _format_zones_html() -> str:
    """Render the current zone list as HTML."""
    if not _zones:
        return "<p>No zones defined. Add a zone above.</p>"
    items = []
    for i, polygon in enumerate(_zones):
        pts = ", ".join(f"({p[0]},{p[1]})" for p in polygon)
        items.append(f"<li>Zone {i + 1}: {pts}</li>")
    return f"<ul>{''.join(items)}</ul>"


def _annotate_zones(
    image: np.ndarray,
    detections: sv.Detections,
    intrusion_flags: List[bool],
    triggered_zones: List[sv.PolygonZone],
) -> np.ndarray:
    """Draw zone polygons using sv.PolygonZoneAnnotator and detection boxes.

    Args:
        image: BGR image to annotate.
        detections: Person detections from predictor.
        intrusion_flags: Per-zone boolean indicating intrusion.
        triggered_zones: Pre-triggered PolygonZone objects from
            ``_check_zone_intrusions``, reused to avoid duplicate work.

    Returns:
        Annotated BGR image.
    """
    annotated = image.copy()

    for i, zone in enumerate(triggered_zones):
        is_intruded = intrusion_flags[i] if i < len(intrusion_flags) else False
        zone_color = sv.Color(0, 0, 255) if is_intruded else sv.Color(0, 255, 255)

        zone_annotator = sv.PolygonZoneAnnotator(
            zone=zone,
            color=zone_color,
            thickness=3 if is_intruded else 2,
            text_scale=0.6,
            text_thickness=2,
        )
        annotated = zone_annotator.annotate(scene=annotated)

    # Draw detection boxes
    if len(detections) > 0:
        box_ann = sv.BoxAnnotator(thickness=2)
        label_ann = sv.LabelAnnotator(text_scale=0.5, text_padding=5)

        labels = []
        for i in range(len(detections)):
            conf = detections.confidence[i] if detections.confidence is not None else 0.0
            track = ""
            if detections.tracker_id is not None and detections.tracker_id[i] is not None:
                track = f" #{int(detections.tracker_id[i])}"
            labels.append(f"person{track} {conf:.2f}")

        annotated = box_ann.annotate(scene=annotated, detections=detections)
        annotated = label_ann.annotate(
            scene=annotated, detections=detections, labels=labels,
        )

    return annotated


def _make_alert_html(intrusions: List[Dict[str, Any]]) -> str:
    """Build alert HTML based on intrusion status."""
    if intrusions:
        details = ", ".join(
            f"Zone {e['zone']}: {e['count']} person(s)" for e in intrusions
        )
        return (
            f"<div style='background:#dc3545;color:white;padding:12px;"
            f"border-radius:6px;text-align:center;font-size:16px;'>"
            f"<b>INTRUSION DETECTED</b><br/>{details}</div>"
        )
    return (
        "<div style='background:#28a745;color:white;padding:12px;"
        "border-radius:6px;text-align:center;font-size:16px;'>"
        "<b>ZONE CLEAR</b></div>"
    )


# ---------------------------------------------------------------------------
# Detection helpers (shared by image and video)
# ---------------------------------------------------------------------------


def _run_detection(
    bgr: np.ndarray,
    predictor: Any,
    person_only: bool,
) -> sv.Detections:
    """Run predictor and return filtered sv.Detections.

    Args:
        bgr: BGR image.
        predictor: DetectionPredictor instance.
        person_only: If True, filter to person class only.

    Returns:
        sv.Detections with person-only filtering applied.
    """
    results = predictor.predict(bgr)

    boxes = np.asarray(
        results.get("boxes", np.empty((0, 4))), dtype=np.float32
    )
    scores = np.asarray(
        results.get("scores", np.empty(0)), dtype=np.float32
    )
    labels = np.asarray(
        results.get("labels", np.empty(0)), dtype=np.int64
    )

    if boxes.ndim == 1:
        boxes = boxes.reshape(-1, 4)

    # Filter to person class if requested
    if person_only and len(labels) > 0:
        mask = labels == _COCO_PERSON_ID
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

    return sv.Detections(
        xyxy=boxes,
        confidence=scores,
        class_id=labels.astype(int) if len(labels) > 0 else np.empty(0, dtype=int),
    )


def _check_zone_intrusions(
    detections: sv.Detections,
) -> Tuple[List[Dict[str, Any]], List[bool], List[sv.PolygonZone]]:
    """Check all defined zones for intrusions.

    Args:
        detections: Person detections.

    Returns:
        Tuple of (intrusion list, per-zone boolean flags,
        list of triggered PolygonZone objects for reuse in annotation).
    """
    intrusions: List[Dict[str, Any]] = []
    intrusion_flags: List[bool] = []
    triggered_zones: List[sv.PolygonZone] = []

    for i, polygon in enumerate(_zones):
        zone = sv.PolygonZone(polygon=polygon)
        zone_mask = zone.trigger(detections)
        n_in_zone = int(zone_mask.sum()) if zone_mask is not None else 0
        intrusion_flags.append(n_in_zone > 0)
        triggered_zones.append(zone)
        if n_in_zone > 0:
            intrusions.append({"zone": i + 1, "count": n_in_zone})

    return intrusions, intrusion_flags, triggered_zones


# ---------------------------------------------------------------------------
# Image detection
# ---------------------------------------------------------------------------


def _detect_intrusions_image(
    image: Optional[np.ndarray],
    conf_threshold: float,
    person_only: bool,
    manager: Any,
) -> Tuple[Optional[np.ndarray], str, dict]:
    """Run person detection and check zone intrusions on a single image.

    Args:
        image: RGB numpy array from Gradio.
        conf_threshold: Detection confidence threshold.
        person_only: If True, only consider person-class detections.
        manager: ModelManager instance.

    Returns:
        Tuple of (annotated RGB image, alert HTML, results dict).
    """
    if image is None:
        return None, _make_alert_html([]), {}

    bgr = rgb_to_bgr(image)
    predictor = manager.get_coco_predictor(conf_threshold)
    detections = _run_detection(bgr, predictor, person_only)
    intrusions, intrusion_flags, triggered_zones = _check_zone_intrusions(detections)

    # Annotate with supervision
    annotated = _annotate_zones(bgr, detections, intrusion_flags, triggered_zones)

    result_info = {
        "intrusions": intrusions,
        "total_persons": len(detections),
        "zones_defined": len(_zones),
        "confidence_threshold": conf_threshold,
        "person_only": person_only,
    }

    rgb_out = bgr_to_rgb(annotated)
    return rgb_out, _make_alert_html(intrusions), result_info


# ---------------------------------------------------------------------------
# Video detection
# ---------------------------------------------------------------------------


def _detect_intrusions_video(
    video_path: Optional[str],
    conf_threshold: float,
    person_only: bool,
    enable_tracking: bool,
    manager: Any,
    config: dict,
) -> Tuple[Optional[str], str]:
    """Process a video file for zone intrusion detection.

    Uses ``sv.get_video_frames_generator()`` and ``sv.VideoSink`` for
    frame-by-frame processing with zone checks and optional tracking.

    Args:
        video_path: Path to input video.
        conf_threshold: Confidence threshold.
        person_only: If True, filter to person class only.
        enable_tracking: Whether to enable ByteTrack tracking.
        manager: ModelManager instance.
        config: App configuration dict.

    Returns:
        Tuple of (output_video_path, summary text).
    """
    if not video_path:
        return None, "No video provided."

    if not _zones:
        return None, "No zones defined. Please add at least one zone before processing video."

    predictor = manager.get_coco_predictor(conf_threshold)

    # Setup tracker if enabled
    tracker = None
    if enable_tracking:
        tracker_config = config.get("tracker")
        tracker = create_tracker(tracker_config)

    video_info = sv.VideoInfo.from_video_path(video_path)
    generator = sv.get_video_frames_generator(source_path=video_path)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        output_path = tmp.name

    # Per-zone counters across frames
    zone_intrusion_frames: Dict[int, int] = defaultdict(int)
    total_intrusion_events = 0
    processed_frames = 0
    total_persons = 0

    with sv.VideoSink(target_path=output_path, video_info=video_info) as sink:
        for frame in generator:
            detections = _run_detection(frame, predictor, person_only)

            # Apply tracking if enabled
            if tracker is not None and len(detections) > 0:
                detections = update_tracker(tracker, detections)

            intrusions, intrusion_flags, triggered_zones = _check_zone_intrusions(detections)

            for entry in intrusions:
                zone_idx = entry["zone"]
                zone_intrusion_frames[zone_idx] += 1
            if intrusions:
                total_intrusion_events += 1

            total_persons += len(detections)

            annotated = _annotate_zones(frame, detections, intrusion_flags, triggered_zones)
            sink.write_frame(annotated)
            processed_frames += 1

    # Build summary
    lines = [
        f"Processed frames: {processed_frames}",
        f"Total person detections: {total_persons}",
        f"Zones defined: {len(_zones)}",
        f"Tracking: {'enabled' if enable_tracking else 'disabled'}",
        "",
        "--- Intrusion Summary ---",
        f"Frames with intrusions: {total_intrusion_events} / {processed_frames}",
    ]

    if zone_intrusion_frames:
        lines.append("")
        lines.append("--- Per-Zone Intrusion Frames ---")
        for zone_idx in sorted(zone_intrusion_frames.keys()):
            count = zone_intrusion_frames[zone_idx]
            pct = (count / processed_frames * 100) if processed_frames > 0 else 0
            lines.append(f"  Zone {zone_idx}: {count} frames ({pct:.1f}%)")
    else:
        lines.append("No intrusions detected in any zone.")

    return output_path, "\n".join(lines)


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------


def build_tab_zone(manager: Any, config: dict) -> None:
    """Build the Zone Intrusion Detection tab with Image and Video sub-tabs.

    Args:
        manager: ModelManager instance with ``get_coco_predictor`` method.
        config: Application config dict.
    """
    zone_config = config.get("zone", {})
    default_conf = zone_config.get("confidence_threshold", 0.5)

    with gr.Tab("Zone Intrusion"):
        gr.Markdown("## Zone Intrusion Detection")
        gr.Markdown(
            "Define restricted zones and detect person intrusions. "
            "Works with COCO pretrained model."
        )

        with gr.Row():
            conf_slider = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=default_conf,
                step=0.05,
                label="Confidence Threshold",
            )
            person_only_checkbox = gr.Checkbox(
                value=True, label="Person Only"
            )

        # Zone definition section
        gr.Markdown("### Define Restricted Zone")
        gr.Markdown(
            "Enter polygon vertices as comma-separated x,y pairs "
            "(e.g., '100,100 500,100 500,400 100,400')"
        )

        with gr.Row():
            zone_input = gr.Textbox(
                placeholder="100,100 500,100 500,400 100,400",
                label="Zone Polygon",
                scale=3,
            )
            add_zone_btn = gr.Button("Add Zone", scale=1)

        zones_display = gr.HTML(value=_format_zones_html())
        clear_zones_btn = gr.Button("Clear All Zones", variant="stop")

        # -- Callbacks for zone management --
        def _add_zone(zone_text: str) -> str:
            """Parse zone text and add to the zone list."""
            if not zone_text or not zone_text.strip():
                return _format_zones_html()
            try:
                polygon = _parse_zone(zone_text)
                _zones.append(polygon)
            except ValueError as exc:
                return f"<p style='color:red;'>Error: {exc}</p>" + _format_zones_html()
            return _format_zones_html()

        def _clear_zones() -> str:
            """Remove all defined zones."""
            _zones.clear()
            return _format_zones_html()

        add_zone_btn.click(
            fn=_add_zone, inputs=[zone_input], outputs=[zones_display]
        )
        clear_zones_btn.click(
            fn=_clear_zones, inputs=[], outputs=[zones_display]
        )

        # -- Image + Video sub-tabs --
        with gr.Tabs():
            # ---- Image sub-tab ----
            with gr.Tab("Image"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(type="numpy", label="Input Image")
                        detect_btn = gr.Button("Detect Intrusions", variant="primary")
                    with gr.Column():
                        image_output = gr.Image(
                            type="numpy", label="Detection Result"
                        )
                        intrusion_alert_html = gr.HTML(value=_make_alert_html([]))
                        results_json = gr.JSON(label="Detection Details")

                detect_btn.click(
                    fn=lambda img, conf, po: _detect_intrusions_image(
                        img, conf, po, manager,
                    ),
                    inputs=[image_input, conf_slider, person_only_checkbox],
                    outputs=[image_output, intrusion_alert_html, results_json],
                )

            # ---- Video sub-tab ----
            with gr.Tab("Video"):
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(label="Upload Video")
                        tracking_checkbox = gr.Checkbox(
                            label="Enable Tracking (ByteTrack)",
                            value=False,
                        )
                        process_btn = gr.Button(
                            "Process Video", variant="primary"
                        )
                    with gr.Column():
                        video_output = gr.Video(label="Processed Video")
                        video_summary = gr.Textbox(
                            label="Processing Summary",
                            lines=14,
                            interactive=False,
                        )

                process_btn.click(
                    fn=lambda vid, conf, po, track: _detect_intrusions_video(
                        vid, conf, po, track, manager, config,
                    ),
                    inputs=[
                        video_input,
                        conf_slider,
                        person_only_checkbox,
                        tracking_checkbox,
                    ],
                    outputs=[video_output, video_summary],
                )

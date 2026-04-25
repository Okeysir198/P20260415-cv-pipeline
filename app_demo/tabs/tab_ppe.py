"""PPE compliance tab — Helmet + Safety Shoes with Image and Video sub-tabs.

Each PPE sub-tab (Helmet, Shoes) contains Image and Video modes.
Uses supervision for all annotation with violation/compliance color coding.
Video mode uses VideoProcessor with optional ByteTrack tracking.
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gradio as gr
import numpy as np
import supervision as sv
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from app_demo.utils import bgr_to_rgb, rgb_to_bgr
from core.p10_inference.predictor import DetectionPredictor
from core.p10_inference.supervision_bridge import build_labels, to_sv_detections
from core.p10_inference.video_inference import VideoProcessor

# Color palette: green for compliance, red for violations, blue for person
_COLOR_COMPLIANCE = sv.Color(r=0, g=200, b=0)
_COLOR_VIOLATION = sv.Color(r=220, g=30, b=30)
_COLOR_PERSON = sv.Color(r=60, g=120, b=220)


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------


def _annotate_ppe_frame(
    frame_bgr: np.ndarray,
    detections: sv.Detections,
    class_names: Dict[int, str],
    violation_ids: set[int],
    compliance_ids: set[int],
    draw_traces: bool = False,
) -> np.ndarray:
    """Annotate a frame with PPE-specific color coding using supervision.

    Args:
        frame_bgr: BGR image.
        detections: sv.Detections from predictor output.
        class_names: Class ID to name mapping.
        violation_ids: Class IDs for violations (red).
        compliance_ids: Class IDs for compliance (green).
        draw_traces: Whether to draw tracking traces.

    Returns:
        Annotated BGR image.
    """
    annotated = frame_bgr.copy()
    if len(detections) == 0:
        return annotated

    # Build per-detection color list
    colors = []
    for cls_id in detections.class_id:
        cid = int(cls_id)
        if cid in violation_ids:
            colors.append(_COLOR_VIOLATION)
        elif cid in compliance_ids:
            colors.append(_COLOR_COMPLIANCE)
        else:
            colors.append(_COLOR_PERSON)

    palette = sv.ColorPalette(colors)
    box_ann = sv.BoxAnnotator(thickness=2, color=palette, color_lookup=sv.ColorLookup.INDEX)
    label_ann = sv.LabelAnnotator(
        text_scale=0.5, text_thickness=1, text_padding=5,
        color=palette, color_lookup=sv.ColorLookup.INDEX,
    )

    labels = build_labels(detections, class_names)
    annotated = box_ann.annotate(scene=annotated, detections=detections)
    annotated = label_ann.annotate(scene=annotated, detections=detections, labels=labels)

    if draw_traces and detections.tracker_id is not None:
        trace_ann = sv.TraceAnnotator(
            trace_length=60, thickness=2, color=palette, color_lookup=sv.ColorLookup.INDEX,
        )
        annotated = trace_ann.annotate(scene=annotated, detections=detections)

    return annotated


def _compliance_stats_html(
    detections: sv.Detections,
    violation_ids: set[int],
    compliance_ids: set[int],
    ppe_type: str,
) -> str:
    """Generate HTML compliance statistics from detections.

    Args:
        detections: sv.Detections from inference.
        violation_ids: Class IDs for violations.
        compliance_ids: Class IDs for compliance.
        ppe_type: "Helmet" or "Safety Shoes" for display.

    Returns:
        HTML string with compliance statistics.
    """
    if len(detections) == 0:
        return f"<p>No {ppe_type.lower()} detections found.</p>"

    violations = int(np.isin(detections.class_id, list(violation_ids)).sum())
    compliant = int(np.isin(detections.class_id, list(compliance_ids)).sum())
    total_ppe = violations + compliant
    compliance_rate = (compliant / total_ppe * 100) if total_ppe > 0 else 0.0

    status_color = "#28a745" if violations == 0 else "#dc3545"
    status_text = "COMPLIANT" if violations == 0 else "VIOLATIONS DETECTED"

    return f"""
    <div style="font-family: sans-serif; padding: 10px;">
        <h3 style="color: {status_color};">{status_text}</h3>
        <table style="border-collapse: collapse; width: 100%;">
            <tr><td style="padding: 4px 8px;"><b>{ppe_type} Compliance Rate</b></td>
                <td style="padding: 4px 8px;">{compliance_rate:.1f}%</td></tr>
            <tr><td style="padding: 4px 8px;">Compliant</td>
                <td style="padding: 4px 8px; color: #28a745;"><b>{compliant}</b></td></tr>
            <tr><td style="padding: 4px 8px;">Violations</td>
                <td style="padding: 4px 8px; color: #dc3545;"><b>{violations}</b></td></tr>
            <tr><td style="padding: 4px 8px;">Total PPE Detections</td>
                <td style="padding: 4px 8px;">{total_ppe}</td></tr>
            <tr><td style="padding: 4px 8px;">Persons Detected</td>
                <td style="padding: 4px 8px;">{len(detections) - total_ppe}</td></tr>
        </table>
    </div>
    """


# ---------------------------------------------------------------------------
# Image inference
# ---------------------------------------------------------------------------


def _ppe_image_inference(
    image: np.ndarray,
    predictor: DetectionPredictor,
    class_names: Dict[int, str],
    violation_ids: set[int],
    compliance_ids: set[int],
    ppe_type: str,
    conf_threshold: float,
) -> Tuple[Optional[np.ndarray], str]:
    """Run PPE detection on a single image.

    Args:
        image: RGB image from Gradio.
        predictor: Loaded DetectionPredictor.
        class_names: Class ID to name mapping.
        violation_ids: Violation class IDs.
        compliance_ids: Compliance class IDs.
        ppe_type: "Helmet" or "Safety Shoes".
        conf_threshold: Confidence threshold.

    Returns:
        Tuple of (annotated_rgb_image, compliance_stats_html).
    """
    if image is None:
        return None, "<p>No image provided.</p>"

    predictor.conf_threshold = conf_threshold
    bgr = rgb_to_bgr(image)
    results = predictor.predict(bgr)
    sv_dets = to_sv_detections(results)

    annotated_bgr = _annotate_ppe_frame(
        bgr, sv_dets, class_names, violation_ids, compliance_ids,
    )
    annotated_rgb = bgr_to_rgb(annotated_bgr)
    stats = _compliance_stats_html(sv_dets, violation_ids, compliance_ids, ppe_type)

    return annotated_rgb, stats


# ---------------------------------------------------------------------------
# Video inference
# ---------------------------------------------------------------------------


def _ppe_video_inference(
    video_path: str,
    predictor: DetectionPredictor,
    conf_threshold: float,
    enable_tracking: bool,
    tracker_config: Optional[Dict] = None,
) -> Tuple[Optional[str], str]:
    """Process a video for PPE detection using VideoProcessor.

    Args:
        video_path: Path to uploaded video.
        predictor: Loaded DetectionPredictor.
        conf_threshold: Confidence threshold.
        enable_tracking: Whether to enable ByteTrack.
        tracker_config: Optional tracker config dict.

    Returns:
        Tuple of (output_video_path, summary_json).
    """
    if not video_path:
        return None, json.dumps({"error": "No video provided"}, indent=2)

    predictor.conf_threshold = conf_threshold
    processor = VideoProcessor(
        predictor=predictor,
        enable_tracking=enable_tracking,
        tracker_config=tracker_config,
    )

    output_dir = Path(tempfile.mkdtemp(prefix="ppe_demo_"))
    output_path = output_dir / "output.mp4"

    summary = processor.process_video(
        video_path=video_path,
        output_path=str(output_path),
    )

    return str(output_path), json.dumps(summary, indent=2, default=str)


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------


def build_tab_ppe(manager: Any, config: Dict) -> None:
    """Build the PPE tab with Helmet and Safety Shoes sub-tabs.

    Each sub-tab has Image and Video modes.

    Args:
        manager: ModelManager instance.
        config: Demo config dict.
    """
    tracker_config = config.get("tracker")

    with gr.Tab("PPE Compliance"):
        gr.Markdown("## PPE Compliance Monitoring")

        # Build both sub-tabs with shared parameterized builder
        for use_case, label in [
            ("helmet", "Helmet"),
            ("shoes_det", "Safety Shoes"),
        ]:
          with gr.Tab(label):
            with gr.Tabs():
                with gr.Tab("Image"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            img_conf = gr.Slider(
                                minimum=0.05, maximum=0.95, value=0.25, step=0.05,
                                label="Confidence Threshold",
                            )
                            img_input = gr.Image(type="numpy", label="Upload Image")
                            img_btn = gr.Button(f"Detect {label}", variant="primary")
                        with gr.Column(scale=1):
                            img_output = gr.Image(type="numpy", label="Annotated Image")
                            img_stats = gr.HTML(label="Compliance Stats")

                    # Capture loop vars with default args
                    def _make_img_fn(uc=use_case, lb=label, cfg=config):
                        def fn(img, conf):
                            pred, _ = manager.get_use_case_predictor(uc, conf)
                            return _ppe_image_inference(
                                img, pred, pred.class_names,
                                set(cfg.get("use_cases", {}).get(uc, {}).get("violation_classes", [])),
                                set(cfg.get("use_cases", {}).get(uc, {}).get("compliance_classes", [])),
                                lb, conf,
                            )
                        return fn

                    img_btn.click(
                        fn=_make_img_fn(),
                        inputs=[img_input, img_conf],
                        outputs=[img_output, img_stats],
                    )

                with gr.Tab("Video"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            vid_conf = gr.Slider(
                                minimum=0.05, maximum=0.95, value=0.25, step=0.05,
                                label="Confidence Threshold",
                            )
                            vid_tracking = gr.Checkbox(
                                value=False, label="Enable Tracking (ByteTrack)",
                            )
                            vid_input = gr.Video(label="Upload Video")
                            vid_btn = gr.Button("Process Video", variant="primary")
                        with gr.Column(scale=1):
                            vid_output = gr.Video(label="Processed Video")
                            vid_json = gr.Textbox(
                                label="Summary (JSON)", lines=10, max_lines=25,
                            )

                    vid_btn.click(
                        fn=lambda v, conf, track, uc=use_case: _ppe_video_inference(
                            v, manager.get_use_case_predictor(uc, conf)[0],
                            conf, track, tracker_config,
                        ),
                        inputs=[vid_input, vid_conf, vid_tracking],
                        outputs=[vid_output, vid_json],
                    )

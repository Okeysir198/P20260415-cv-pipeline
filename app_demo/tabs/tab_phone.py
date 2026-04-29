"""Phone detection tab with Image and Video sub-tabs.

Detects phone usage (class 1 = "phone_usage" from features/safety-poketenashi_phone_usage/configs/05_data.yaml,
or COCO class 67 = "cell phone" when using a pretrained model).
Uses supervision for all annotation. Video mode uses VideoProcessor
with optional ByteTrack tracking.
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gradio as gr
import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from app_demo.utils import bgr_to_rgb, rgb_to_bgr
from core.p10_inference.predictor import DetectionPredictor
from core.p10_inference.supervision_bridge import (
    annotate_frame,
    build_annotators,
    to_sv_detections,
)
from core.p10_inference.video_inference import VideoProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_phone_class_names(predictor: DetectionPredictor) -> Dict[int, str]:
    """Resolve phone class names from predictor.

    Args:
        predictor: The loaded DetectionPredictor.

    Returns:
        Class ID to name mapping.
    """
    if hasattr(predictor, "class_names") and predictor.class_names:
        return predictor.class_names
    return {}


def _phone_stats_json(
    num_phones: int,
    total_detections: int,
    conf_threshold: float,
) -> str:
    """Build a JSON summary string for phone detections.

    Args:
        num_phones: Number of phone detections.
        total_detections: Total number of all detections.
        conf_threshold: Confidence threshold used.

    Returns:
        JSON string with detection summary.
    """
    status = "PHONE USAGE DETECTED" if num_phones > 0 else "No phone detected"
    return json.dumps(
        {
            "status": status,
            "phone_count": num_phones,
            "total_detections": total_detections,
            "confidence_threshold": conf_threshold,
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# Image inference
# ---------------------------------------------------------------------------


def _phone_image_inference(
    image: np.ndarray,
    predictor: DetectionPredictor,
    conf_threshold: float,
    annotators: Dict[str, Any],
    config: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], str]:
    """Run phone detection on a single image.

    Args:
        image: RGB image from Gradio.
        predictor: Loaded DetectionPredictor.
        conf_threshold: Confidence threshold.
        annotators: Supervision annotators from build_annotators().
        config: App configuration dict.

    Returns:
        Tuple of (annotated_rgb_image, summary_json).
    """
    if image is None:
        return None, json.dumps({"error": "No image provided"}, indent=2)

    predictor.conf_threshold = conf_threshold
    bgr = rgb_to_bgr(image)
    results = predictor.predict(bgr)
    sv_dets = to_sv_detections(results)
    class_names = _get_phone_class_names(predictor)

    annotated_bgr = annotate_frame(
        bgr, sv_dets, class_names, annotators,
    )
    annotated_rgb = bgr_to_rgb(annotated_bgr)

    # Get alert classes from config (phone classes that trigger alerts)
    alert_classes = set(config.get("use_cases", {}).get("phone", {}).get("alert_classes", ["phone"]))

    # Count phones by checking class names against alert classes
    phone_ids = {
        cid for cid, name in class_names.items()
        if name in alert_classes
    }
    phone_mask = np.isin(sv_dets.class_id, list(phone_ids)) if len(sv_dets) > 0 else np.array([])
    num_phones = int(phone_mask.sum()) if len(phone_mask) > 0 else 0

    summary = _phone_stats_json(num_phones, len(sv_dets), conf_threshold)
    return annotated_rgb, summary


# ---------------------------------------------------------------------------
# Video inference
# ---------------------------------------------------------------------------


def _phone_video_inference(
    video_path: str,
    predictor: DetectionPredictor,
    conf_threshold: float,
    enable_tracking: bool,
    tracker_config: Optional[Dict] = None,
) -> Tuple[Optional[str], str]:
    """Process a video for phone detection using VideoProcessor.

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

    output_dir = Path(tempfile.mkdtemp(prefix="phone_demo_"))
    output_path = output_dir / "output.mp4"

    summary = processor.process_video(
        video_path=video_path,
        output_path=str(output_path),
    )

    return str(output_path), json.dumps(summary, indent=2, default=str)


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------


def build_tab_phone(manager: Any, config: Dict) -> None:
    """Build the Phone Detection tab with Image and Video sub-tabs.

    Args:
        manager: ModelManager instance.
        config: Demo config dict.
    """
    tracker_config = config.get("tracker")
    annotators = build_annotators(config)

    with gr.Tab("Phone Detection"):
      with gr.Tabs():
        # -- Image --
        with gr.Tab("Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    p_img_conf = gr.Slider(
                        minimum=0.05, maximum=0.95, value=0.5, step=0.05,
                        label="Confidence Threshold",
                    )
                    p_img_input = gr.Image(type="numpy", label="Upload Image")
                    p_img_btn = gr.Button("Detect", variant="primary")
                with gr.Column(scale=1):
                    p_img_output = gr.Image(type="numpy", label="Annotated Image")
                    p_img_json = gr.Textbox(
                        label="Detection Summary", lines=8, max_lines=20,
                    )

            p_img_btn.click(
                fn=lambda img, conf: _phone_image_inference(
                    img, manager.get_use_case_predictor("phone", conf)[0], conf, annotators, config,
                ),
                inputs=[p_img_input, p_img_conf],
                outputs=[p_img_output, p_img_json],
            )

        # -- Video --
        with gr.Tab("Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    p_vid_conf = gr.Slider(
                        minimum=0.05, maximum=0.95, value=0.5, step=0.05,
                        label="Confidence Threshold",
                    )
                    p_vid_tracking = gr.Checkbox(
                        value=False, label="Enable Tracking (ByteTrack)",
                    )
                    p_vid_input = gr.Video(label="Upload Video")
                    p_vid_btn = gr.Button("Process Video", variant="primary")
                with gr.Column(scale=1):
                    p_vid_output = gr.Video(label="Processed Video")
                    p_vid_json = gr.Textbox(
                        label="Summary (JSON)", lines=10, max_lines=25,
                    )

            p_vid_btn.click(
                fn=lambda v, conf, track: _phone_video_inference(
                    v, manager.get_use_case_predictor("phone", conf)[0], conf, track, tracker_config,
                ),
                inputs=[p_vid_input, p_vid_conf, p_vid_tracking],
                outputs=[p_vid_output, p_vid_json],
            )

"""Generic Object Detection tab — image and video inference with any available model.

Provides a model-agnostic detection interface with sub-tabs for single-image
and video processing. Annotated results use the supervision library for
consistent, high-quality visualizations.
"""

import json
import logging
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gradio as gr
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from app_demo.utils import bgr_to_rgb, rgb_to_bgr
from core.p10_inference.supervision_bridge import (
    annotate_frame,
    build_annotators,
    to_sv_detections,
)
from core.p10_inference.video_inference import VideoProcessor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared video summary state
# ---------------------------------------------------------------------------

_last_video_summary: Dict[str, Any] = {}


def set_video_summary(summary: dict) -> None:
    """Store the most recent video processing summary for cross-tab access."""
    global _last_video_summary
    _last_video_summary = summary


def get_video_summary() -> dict:
    """Retrieve the most recent video processing summary."""
    return _last_video_summary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _detect_image(
    image: Optional[np.ndarray],
    model_choice: str,
    conf_threshold: float,
    manager: Any,
    config: dict,
) -> Tuple[Optional[np.ndarray], str, str]:
    """Run detection on a single image and return annotated result + JSON + class summary."""
    if image is None:
        return None, json.dumps({"error": "No image provided."}), ""

    try:
        predictor, model_info = manager.get_predictor_by_choice(
            model_choice, conf_threshold=conf_threshold
        )
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        return None, json.dumps({"error": f"Model load failed: {exc}"}), ""

    # Gradio delivers RGB; predictor expects BGR
    bgr_image = rgb_to_bgr(image)

    try:
        predictions = predictor.predict(bgr_image)
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        return None, json.dumps({"error": f"Prediction failed: {exc}"}), ""

    detections = to_sv_detections(predictions)
    class_names = predictor.class_names

    annotators = build_annotators(config)
    annotated_bgr = annotate_frame(bgr_image, detections, class_names, annotators)

    # Convert back to RGB for Gradio display
    annotated_rgb = bgr_to_rgb(annotated_bgr)

    # Build JSON summary
    boxes = np.asarray(predictions["boxes"]).tolist()
    scores = np.asarray(predictions["scores"]).tolist()
    labels = np.asarray(predictions["labels"]).tolist()

    det_list = []
    for i in range(len(detections)):
        det_list.append(
            {
                "class": class_names.get(int(labels[i]), str(int(labels[i]))),
                "confidence": round(scores[i], 4),
                "bbox": [round(v, 1) for v in boxes[i]],
            }
        )

    result = {
        "num_detections": len(detections),
        "detections": det_list,
    }

    # Class distribution summary
    class_counts = Counter(
        class_names.get(int(lid), str(int(lid))) for lid in labels
    )
    if class_counts:
        lines = [f"  {name}: {count}" for name, count in class_counts.most_common()]
        summary_text = f"Total: {len(detections)} detections\n" + "\n".join(lines)
    else:
        summary_text = "No detections."

    return annotated_rgb, json.dumps(result, indent=2), summary_text


def _process_video(
    video_path: Optional[str],
    model_choice: str,
    conf_threshold: float,
    enable_tracking: bool,
    manager: Any,
    config: dict,
) -> Tuple[Optional[str], str]:
    """Process a video file and return the annotated video path + JSON summary."""
    if video_path is None:
        return None, json.dumps({"error": "No video provided."})

    try:
        predictor, model_info = manager.get_predictor_by_choice(
            model_choice, conf_threshold=conf_threshold
        )
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        return None, json.dumps({"error": f"Model load failed: {exc}"})

    tracker_config = config.get("tracker") if enable_tracking else None

    processor = VideoProcessor(
        predictor,
        alert_config=None,
        enable_tracking=enable_tracking,
        tracker_config=tracker_config,
    )

    output_dir = tempfile.mkdtemp(prefix="detection_video_")
    output_path = str(Path(output_dir) / "output.mp4")

    try:
        summary = processor.process_video(video_path, output_path)
    except Exception as exc:
        logger.error("Video processing failed: %s", exc)
        return None, json.dumps({"error": f"Video processing failed: {exc}"})

    # Store summary for analytics tab
    set_video_summary(summary)

    # Serialize summary — convert non-JSON-safe types
    safe_summary = {}
    for key, value in summary.items():
        if isinstance(value, (int, float, str, bool)):
            safe_summary[key] = value
        elif isinstance(value, dict):
            safe_summary[key] = {
                str(k): int(v) if isinstance(v, (int, np.integer)) else v
                for k, v in value.items()
            }
        elif isinstance(value, list):
            safe_summary[key] = [
                {str(kk): vv for kk, vv in item.items()}
                if isinstance(item, dict)
                else item
                for item in value
            ]
        else:
            safe_summary[key] = str(value)

    return output_path, json.dumps(safe_summary, indent=2)


def _get_model_info(model_choice: str, manager: Any) -> str:
    """Return HTML snippet with model metadata including class names."""
    if not model_choice:
        return "<i>Select a model to view details.</i>"

    try:
        predictor, model_info = manager.get_predictor_by_choice(model_choice)
        class_names = predictor.class_names
        num_classes = len(class_names)
        classes_str = ", ".join(class_names.values())
        return (
            f"<b>Model:</b> {model_info}<br>"
            f"<b>Classes ({num_classes}):</b> {classes_str}"
        )
    except Exception as exc:
        return f"<b>Error:</b> {exc}"


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------


def build_tab_detection(manager: Any, config: dict) -> None:
    """Build the Generic Object Detection tab inside a ``gr.Tab`` context.

    Args:
        manager: ``ModelManager`` instance with model discovery and caching.
        config: Demo config dict (contains ``supervision``, ``tracker``, etc.).
    """
    with gr.Tab("Object Detection"):
        # -- Controls row --
        available_models = manager.list_available_models()
        default_model = available_models[0] if available_models else ""

        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=available_models,
                value=default_model,
                label="Model",
                interactive=True,
            )
            conf_slider = gr.Slider(
                minimum=0.05,
                maximum=0.95,
                value=0.25,
                step=0.05,
                label="Confidence Threshold",
            )

        model_info_html = gr.HTML(value="<i>Select a model to view details.</i>")

        # Update model info when dropdown changes
        model_dropdown.change(
            fn=lambda choice: _get_model_info(choice, manager),
            inputs=[model_dropdown],
            outputs=[model_info_html],
        )

        # -- Sub-tabs --
        with gr.Tabs():
            # ---- Image sub-tab ----
            with gr.Tab("Image"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(
                            type="numpy", label="Upload Image"
                        )
                        detect_btn = gr.Button("Detect", variant="primary")
                    with gr.Column():
                        image_output = gr.Image(
                            type="numpy", label="Annotated Result"
                        )
                        class_summary = gr.Textbox(
                            label="Class Distribution",
                            lines=4,
                            interactive=False,
                        )
                        json_output = gr.Textbox(
                            label="Detection Results (JSON)",
                            lines=10,
                            interactive=False,
                        )

                detect_btn.click(
                    fn=lambda img, model, conf: _detect_image(
                        img, model, conf, manager, config
                    ),
                    inputs=[image_input, model_dropdown, conf_slider],
                    outputs=[image_output, json_output, class_summary],
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
                        video_json = gr.Textbox(
                            label="Processing Summary (JSON)",
                            lines=10,
                            interactive=False,
                        )

                process_btn.click(
                    fn=lambda vid, model, conf, track: _process_video(
                        vid, model, conf, track, manager, config
                    ),
                    inputs=[
                        video_input,
                        model_dropdown,
                        conf_slider,
                        tracking_checkbox,
                    ],
                    outputs=[video_output, video_json],
                )

"""Fire & Smoke Detection tab for the comprehensive safety demo.

Provides image and video sub-tabs for detecting fire and smoke hazards.
Uses a fine-tuned model when available, falling back to COCO pretrained.
"""

import logging
import tempfile
from typing import Any, Dict, Optional, Tuple

import gradio as gr
import numpy as np
import supervision as sv

from app_demo.utils import bgr_to_rgb, rgb_to_bgr
from core.p10_inference.supervision_bridge import to_sv_detections
from core.p10_inference.video_inference import VideoProcessor

logger = logging.getLogger(__name__)


def _detect_fire_image(
    image: Optional[np.ndarray],
    conf_threshold: float,
    manager: Any,
    config: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], str, str]:
    """Run fire detection on a single image.

    Args:
        image: RGB image from Gradio.
        conf_threshold: Confidence threshold for detections.
        manager: ModelManager instance.
        config: App configuration dict.

    Returns:
        Tuple of (annotated_image, alert_html, results_json).
    """
    if image is None:
        return None, "", "{}"

    # Gradio provides RGB; predictors expect BGR
    bgr = rgb_to_bgr(image)

    predictor, model_type = manager.get_use_case_predictor("fire", conf_threshold)
    predictions = predictor.predict(bgr)
    class_names = predictor.class_names

    boxes = np.asarray(predictions["boxes"]).reshape(-1, 4)
    labels = np.asarray(predictions["labels"]).ravel()

    if len(boxes) == 0:
        alert_html = _build_alert_html(safe=True, model_type=model_type)
        return image, alert_html, "{\"detections\": 0}"

    # Get alert classes from config
    alert_classes = set(config.get("use_cases", {}).get("fire", {}).get("alert_classes", ["fire", "smoke"]))

    # Build supervision detections and annotate
    fire_color_map = {name: sv.Color.RED for name in alert_classes}
    detections = to_sv_detections(predictions)
    annotated = annotate_detections(
        bgr.copy(), detections, class_names, fire_color_map,
    )
    annotated_rgb = bgr_to_rgb(annotated)

    # Determine alert status
    detected_names = {class_names.get(int(cid), "") for cid in labels}
    fire_detected = bool(detected_names & alert_classes)
    alert_html = _build_alert_html(
        safe=not fire_detected,
        model_type=model_type,
        detected_classes=detected_names & alert_classes,
    )

    # Build results summary
    results = {
        "detections": len(boxes),
        "model_type": model_type,
        "classes": {},
    }
    for cid in np.unique(labels):
        name = class_names.get(int(cid), f"class_{cid}")
        count = int(np.sum(labels == cid))
        results["classes"][name] = count

    return annotated_rgb, alert_html, str(results)


def _process_fire_video(
    video_path: Optional[str],
    conf_threshold: float,
    enable_tracking: bool,
    manager: Any,
    config: Dict[str, Any],
) -> Tuple[Optional[str], str]:
    """Process a video file for fire/smoke detection.

    Args:
        video_path: Path to input video.
        conf_threshold: Confidence threshold.
        enable_tracking: Whether to enable ByteTrack tracking.
        manager: ModelManager instance.
        config: App configuration dict.

    Returns:
        Tuple of (output_video_path, alert_log_text).
    """
    if not video_path:
        return None, "No video provided."

    predictor, model_type = manager.get_use_case_predictor("fire", conf_threshold)

    alert_config = config.get("alerts", {}).get("fire", {})
    processor = VideoProcessor(
        predictor=predictor,
        alert_config=alert_config if alert_config else None,
        enable_tracking=enable_tracking,
    )

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        output_path = tmp.name

    summary = processor.process_video(
        video_path=video_path,
        output_path=output_path,
    )

    # Build alert log
    lines = [
        f"Model: {model_type}",
        f"Total frames: {summary.get('total_frames', 0)}",
        f"Processed frames: {summary.get('processed_frames', 0)}",
        f"Total detections: {summary.get('total_detections', 0)}",
        f"FPS: {summary.get('fps', 0):.1f}",
        "",
        "--- Alerts ---",
    ]
    alerts = summary.get("alerts", [])
    if alerts:
        for alert in alerts:
            lines.append(
                f"Frame {alert.get('frame', '?')}: "
                f"{alert.get('class_name', 'unknown')} "
                f"(conf={alert.get('confidence', 0):.2f})"
            )
    else:
        lines.append("No alerts triggered.")

    return output_path, "\n".join(lines)


def annotate_detections(
    bgr_frame: np.ndarray,
    detections: sv.Detections,
    class_names: Dict[int, str],
    color_map: Dict[str, sv.Color],
    default_color: sv.Color = sv.Color.BLUE,
) -> np.ndarray:
    """Annotate a BGR frame with per-class coloring.

    Shared helper used by fire, PPE, and other tabs to draw colored
    bounding boxes based on class-name-to-color mappings.

    Args:
        bgr_frame: BGR image to annotate.
        detections: Supervision detections.
        class_names: Mapping from class ID to name.
        color_map: Mapping from class name to ``sv.Color``.
        default_color: Color for classes not in *color_map*.

    Returns:
        Annotated BGR image.
    """
    if len(detections) == 0:
        return bgr_frame

    colors = [
        color_map.get(class_names.get(int(cid), ""), default_color)
        for cid in detections.class_id
    ]

    labels = []
    for i, cid in enumerate(detections.class_id):
        name = class_names.get(int(cid), f"class_{cid}")
        conf = detections.confidence[i] if detections.confidence is not None else 0.0
        labels.append(f"{name} {conf:.2f}")

    palette = sv.ColorPalette(colors)
    box_annotator = sv.BoxAnnotator(thickness=2, color=palette)
    label_annotator = sv.LabelAnnotator(
        text_scale=0.5, text_thickness=1, text_padding=5, color=palette,
    )

    annotated = box_annotator.annotate(scene=bgr_frame, detections=detections)
    annotated = label_annotator.annotate(
        scene=annotated, detections=detections, labels=labels,
    )
    return annotated


def _build_alert_html(
    safe: bool,
    model_type: str,
    detected_classes: Optional[set] = None,
) -> str:
    """Build alert status HTML.

    Args:
        safe: True if no fire/smoke detected.
        model_type: "fine-tuned" or "coco-pretrained".
        detected_classes: Set of detected fire/smoke class names.

    Returns:
        HTML string for the alert status display.
    """
    parts = []

    if model_type == "coco-pretrained":
        parts.append(
            '<div style="padding:8px;background:#fff3cd;border:1px solid #ffc107;'
            'border-radius:4px;margin-bottom:8px;">'
            "Using COCO pretrained model. Fire/smoke-specific detection "
            "requires a fine-tuned model."
            "</div>"
        )

    if safe:
        parts.append(
            '<div style="padding:12px;background:#d4edda;border:1px solid #28a745;'
            'border-radius:4px;text-align:center;font-weight:bold;font-size:16px;">'
            "SAFE - No fire or smoke detected"
            "</div>"
        )
    else:
        detected = detected_classes or set()
        alert_items = []
        if "fire" in detected:
            alert_items.append("FIRE DETECTED")
        if "smoke" in detected:
            alert_items.append("SMOKE DETECTED")
        alert_text = " | ".join(alert_items) if alert_items else "HAZARD DETECTED"
        parts.append(
            f'<div style="padding:12px;background:#f8d7da;border:1px solid #dc3545;'
            f'border-radius:4px;text-align:center;font-weight:bold;font-size:16px;'
            f'color:#721c24;">'
            f"{alert_text}"
            f"</div>"
        )

    return "\n".join(parts)


def build_model_status_html(model_type: str, label: str = "") -> str:
    """Build model status indicator HTML.

    Shared helper for all tabs to display model provenance.

    Args:
        model_type: "fine-tuned" or "coco-pretrained".
        label: Display label for the use case (e.g. "fire/smoke", "Helmet").

    Returns:
        HTML string showing model status.
    """
    suffix = f" ({label})" if label else ""
    if model_type == "fine-tuned":
        return (
            '<div style="padding:6px 12px;background:#d4edda;border:1px solid #28a745;'
            'border-radius:4px;display:inline-block;">'
            f"Model: <strong>Fine-tuned</strong>{suffix}"
            "</div>"
        )
    return (
        '<div style="padding:6px 12px;background:#fff3cd;border:1px solid #ffc107;'
        'border-radius:4px;display:inline-block;">'
        "Model: <strong>COCO Pretrained</strong> (limited)"
        "</div>"
    )


def build_tab_fire(manager: Any, config: Dict[str, Any]) -> None:
    """Build the Fire & Smoke Detection tab.

    Args:
        manager: ModelManager instance for obtaining predictors.
        config: Application configuration dict with alert settings.
    """
    # Check model availability without loading (avoid eager GPU allocation)
    fine_tuned = manager.discover_fine_tuned()
    initial_model_type = "fine-tuned" if "fire" in fine_tuned else "coco-pretrained"

    with gr.Tab("Fire & Smoke"):
        gr.Markdown("## Fire & Smoke Detection")
        gr.Markdown(
            "Detect fire and smoke for early warning. Uses fine-tuned model "
            "if available, otherwise COCO pretrained."
        )

        with gr.Row():
            model_status = gr.HTML(
                value=build_model_status_html(initial_model_type, "fire/smoke"),
            )
            conf_slider = gr.Slider(
                minimum=0.1,
                maximum=0.95,
                value=config.get("default_confidence", 0.25),
                step=0.05,
                label="Confidence Threshold",
            )

        with gr.Tabs():
            # --- Image sub-tab ---
            with gr.Tab("Image"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(
                            label="Input Image",
                            type="numpy",
                        )
                        detect_btn = gr.Button(
                            "Detect Fire & Smoke",
                            variant="primary",
                        )
                    with gr.Column():
                        image_output = gr.Image(label="Detection Result")
                        alert_status_html = gr.HTML(label="Alert Status")
                        results_json = gr.Textbox(
                            label="Results",
                            lines=6,
                            interactive=False,
                        )

                detect_btn.click(
                    fn=lambda img, conf: _detect_fire_image(img, conf, manager, config),
                    inputs=[image_input, conf_slider],
                    outputs=[image_output, alert_status_html, results_json],
                )

            # --- Video sub-tab ---
            with gr.Tab("Video"):
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(label="Input Video")
                        tracking_checkbox = gr.Checkbox(
                            label="Enable Tracking (ByteTrack)",
                            value=False,
                        )
                        process_btn = gr.Button(
                            "Process Video",
                            variant="primary",
                        )
                    with gr.Column():
                        video_output = gr.Video(label="Processed Video")
                        alert_log_textbox = gr.Textbox(
                            label="Alert Log",
                            lines=12,
                            interactive=False,
                        )

                process_btn.click(
                    fn=lambda vid, conf, track: _process_fire_video(
                        vid, conf, track, manager, config,
                    ),
                    inputs=[video_input, conf_slider, tracking_checkbox],
                    outputs=[video_output, alert_log_textbox],
                )

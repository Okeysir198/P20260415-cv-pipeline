"""Live Camera Streaming tab — real-time webcam detection using Gradio 6 streaming.

Uses ``gr.Image(sources="webcam", streaming=True)`` with the ``.stream()`` event
for frame-by-frame processing. All annotation uses the supervision library.
No FastRTC dependency — pure Gradio 6 HTTP streaming.
"""

import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Tuple

import gradio as gr
import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from app_demo.utils import create_status_html, rgb_to_bgr, bgr_to_rgb
from core.p10_inference.supervision_bridge import (
    annotate_frame,
    build_annotators,
    to_sv_detections,
)


# ---------------------------------------------------------------------------
# Frame processing
# ---------------------------------------------------------------------------


def _process_stream_frame(
    frame: Optional[np.ndarray],
    model_choice: str,
    conf_threshold: float,
    manager: Any,
    config: dict,
) -> Tuple[Optional[np.ndarray], str, str]:
    """Process a single webcam frame and return annotated result + stats.

    Args:
        frame: RGB numpy array (H, W, 3) from Gradio webcam, or None.
        model_choice: Selected model from dropdown.
        conf_threshold: Detection confidence threshold.
        manager: ModelManager instance.
        config: Demo config dict.

    Returns:
        Tuple of (annotated_rgb, status_html, stats_text).
    """
    if frame is None:
        return None, "", "Waiting for camera..."

    try:
        predictor, _ = manager.get_predictor_by_choice(
            model_choice, conf_threshold=conf_threshold
        )
    except Exception as exc:
        logger.error("Model load failed: %s", exc)
        return frame, create_status_html("alert", f"Model error: {exc}"), ""

    bgr = rgb_to_bgr(frame)

    t0 = time.perf_counter()
    predictions = predictor.predict(bgr)
    dt_ms = (time.perf_counter() - t0) * 1000

    detections = to_sv_detections(predictions)
    class_names = predictor.class_names

    annotators = build_annotators(config)
    annotated_bgr = annotate_frame(bgr, detections, class_names, annotators)
    annotated_rgb = bgr_to_rgb(annotated_bgr)

    # Build stats
    n = len(detections)
    fps_est = 1000 / dt_ms if dt_ms > 0 else 0
    labels = np.asarray(predictions["labels"]).tolist()
    class_counts = Counter(class_names.get(int(lid), str(int(lid))) for lid in labels)

    if class_counts:
        lines = [f"  {name}: {count}" for name, count in class_counts.most_common()]
        stats = f"{n} detections | {dt_ms:.0f}ms | ~{fps_est:.1f} FPS\n" + "\n".join(lines)
    else:
        stats = f"No detections | {dt_ms:.0f}ms | ~{fps_est:.1f} FPS"

    # Alert status
    alert_cfg = config.get("alerts", {})
    alert_thresholds = alert_cfg.get("confidence_thresholds", {})
    alert_classes = set()
    for lid, score in zip(labels, np.asarray(predictions["scores"]).tolist()):
        cls_name = class_names.get(int(lid), "")
        if cls_name in alert_thresholds and score >= alert_thresholds[cls_name]:
            alert_classes.add(cls_name)

    if alert_classes:
        status_html = create_status_html(
            "alert", f"ALERT: {', '.join(sorted(alert_classes))} detected"
        )
    elif n > 0:
        status_html = create_status_html("safe", f"{n} objects detected")
    else:
        status_html = create_status_html("safe", "No detections")

    return annotated_rgb, status_html, stats


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------


def build_tab_stream(manager: Any, config: dict) -> None:
    """Build the Live Camera Streaming tab.

    Args:
        manager: ModelManager instance.
        config: Demo config dict.
    """
    stream_cfg = config.get("streaming", {})
    default_conf = config.get("default_confidence", 0.25)
    stream_every = stream_cfg.get("stream_every", 0.1)
    time_limit = stream_cfg.get("time_limit", 300)

    with gr.Tab("Live Camera"):
        gr.Markdown(
            "Real-time object detection from your webcam. "
            "Select a model, adjust confidence, and start the camera."
        )

        # Controls
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
                value=default_conf,
                step=0.05,
                label="Confidence Threshold",
            )

        # Webcam + output
        with gr.Row():
            with gr.Column():
                webcam = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    type="numpy",
                    label="Camera Input",
                )
            with gr.Column():
                output_image = gr.Image(
                    type="numpy",
                    label="Detection Output",
                    streaming=True,
                )
                status_html = gr.HTML(value=create_status_html("safe", "Waiting for camera..."))
                stats_text = gr.Textbox(
                    label="Detection Stats",
                    lines=4,
                    interactive=False,
                    value="Waiting for camera...",
                )

        # Stream event — Gradio 6 native webcam streaming
        webcam.stream(
            fn=lambda frame, model, conf: _process_stream_frame(
                frame, model, conf, manager, config
            ),
            inputs=[webcam, model_dropdown, conf_slider],
            outputs=[output_image, status_html, stats_text],
            stream_every=stream_every,
            time_limit=time_limit,
            concurrency_limit=2,
        )

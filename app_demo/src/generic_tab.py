"""Generic detection tab builder for demo app.

Provides a reusable tab template that combines model loading, class filtering,
inference, metrics display, and export functionality. Can be used as a starting
point for building custom detection tabs.
"""

import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app_demo.model_manager import load_coco_data_config
from app_demo.utils import bgr_to_rgb, rgb_to_bgr
from app_demo.src.class_filter import ClassFilterComponent
from app_demo.src.metrics_display import MetricsDisplay
from app_demo.src.model_loader import ModelLoader
from core.p10_inference.supervision_bridge import (
    annotate_frame,
    build_annotators,
    to_sv_detections,
)

logger = logging.getLogger(__name__)


def build_tab_generic(manager: Any, config: Dict) -> None:
    """Build a generic object detection tab inside a ``gr.Tab`` context.

    This tab provides:
    - Model loading (file browser + predefined dropdown)
    - Class filtering checkboxes
    - Ground truth upload for metrics
    - Image/video inference
    - Metrics display (table + chart)
    - Results export (JSON)

    Args:
        manager: ModelManager instance with model discovery and caching.
        config: Demo config dict (contains supervision, tracker, etc.).
    """
    with gr.Tab("Generic Detection"):
        # Initialize components
        model_loader = ModelLoader(config)

        # Load COCO class names from external YAML
        _coco_cfg = load_coco_data_config(config)
        coco_names = {int(k): v for k, v in _coco_cfg.get("names", {}).items()}
        class_filter = ClassFilterComponent(coco_names, default_select_all=True)

        # ------------------------------------------------------------------------
        # Model Loading Section
        # ------------------------------------------------------------------------
        gr.Markdown("### Model Loading")
        with gr.Row():
            file_input, dropdown_input, source_radio = model_loader.build_ui(
                label="Detection Model",
                show_file_upload=True,
                show_dropdown=True,
            )

        conf_slider = gr.Slider(
            minimum=0.05,
            maximum=0.95,
            value=0.25,
            step=0.05,
            label="Confidence Threshold",
        )

        # Model info display
        model_info = gr.HTML(value="<i>Load a model to view details.</i>")

        # ------------------------------------------------------------------------
        # Class Filter Section
        # ------------------------------------------------------------------------
        gr.Markdown("### Class Filtering")
        class_checkboxes = class_filter.build_ui()

        # ------------------------------------------------------------------------
        # Ground Truth Upload (for metrics)
        # ------------------------------------------------------------------------
        gr.Markdown("### Ground Truth (Optional)")
        gt_json = gr.Textbox(
            label="Ground Truth JSON",
            placeholder='[{"boxes": [[x1,y1,x2,y2], ...], "labels": [0, 1, ...]}, ...]',
            lines=3,
            interactive=True,
        )

        # ------------------------------------------------------------------------
        # Inference Section
        # ------------------------------------------------------------------------
        gr.Markdown("### Inference")

        with gr.Tabs():
            # ---- Image sub-tab ----
            with gr.Tab("Image"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(type="numpy", label="Input Image")
                        detect_image_btn = gr.Button("Detect", variant="primary")
                    with gr.Column():
                        image_output = gr.Image(type="numpy", label="Annotated Result")
                        image_json = gr.Textbox(
                            label="Detection Results (JSON)",
                            lines=10,
                            interactive=False,
                        )

            # ---- Video sub-tab ----
            with gr.Tab("Video"):
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(label="Input Video")
                        process_video_btn = gr.Button("Process Video", variant="primary")
                    with gr.Column():
                        video_output = gr.Video(label="Annotated Video")
                        video_json = gr.Textbox(
                            label="Processing Summary (JSON)",
                            lines=10,
                            interactive=False,
                        )

        # ------------------------------------------------------------------------
        # Metrics Display Section
        # ------------------------------------------------------------------------
        gr.Markdown("### Metrics Display")
        with gr.Row():
            compute_metrics_btn = gr.Button("Compute Metrics", variant="secondary")
            export_results_btn = gr.Button("Export Results", variant="secondary")

        with gr.Row():
            metrics_table = gr.Textbox(
                label="Metrics Table",
                lines=12,
                interactive=False,
            )
            metrics_chart = gr.Plot(label="Metrics Chart")

        # ------------------------------------------------------------------------
        # Event Handlers
        # ------------------------------------------------------------------------

        def _load_model_info(file_path, model_name, source):
            """Load and display model information."""
            predictor = model_loader.load_from_ui(file_path, model_name, source)
            if predictor is None:
                return "<i>Failed to load model.</i>"

            class_names = predictor.class_names
            num_classes = len(class_names)
            classes_str = ", ".join(class_names.values())

            model_path = file_path if source == "Upload File" else model_name

            return (
                f"<b>Model:</b> {Path(model_path).name}<br>"
                f"<b>Backend:</b> {predictor.backend}<br>"
                f"<b>Classes ({num_classes}):</b> {classes_str}"
            )

        # Update model info when model selection changes
        source_radio.change(
            fn=_load_model_info,
            inputs=[file_input, dropdown_input, source_radio],
            outputs=[model_info],
        )
        dropdown_input.change(
            fn=_load_model_info,
            inputs=[file_input, dropdown_input, source_radio],
            outputs=[model_info],
        )
        file_input.change(
            fn=_load_model_info,
            inputs=[file_input, dropdown_input, source_radio],
            outputs=[model_info],
        )

        def _detect_image(
            image, conf, file_path, model_name, source, selected_classes
        ):
            """Run detection on a single image."""
            if image is None:
                return None, json.dumps({"error": "No image provided."})

            predictor = model_loader.load_from_ui(file_path, model_name, source, conf)
            if predictor is None:
                return None, json.dumps({"error": "Failed to load model."})

            # Gradio delivers RGB; predictor expects BGR
            bgr_image = rgb_to_bgr(image)

            try:
                predictions = predictor.predict(bgr_image)
            except Exception as exc:
                logger.error("Prediction failed: %s", exc)
                return None, json.dumps({"error": f"Prediction failed: {exc}"})

            # Convert to supervision and filter
            detections = to_sv_detections(predictions)
            name_to_id = {name: cid for cid, name in predictor.class_names.items()}
            detections = ClassFilterComponent.filter_detections(
                detections, selected_classes, name_to_id
            )

            # Annotate
            annotators = build_annotators(config)
            annotated_bgr = annotate_frame(
                bgr_image, detections, predictor.class_names, annotators
            )

            # Convert back to RGB for Gradio
            annotated_rgb = bgr_to_rgb(annotated_bgr)

            # Build JSON result
            boxes = detections.xyxy.tolist()
            scores = detections.confidence.tolist()
            labels = detections.class_id.tolist()

            det_list = []
            for i in range(len(detections)):
                det_list.append({
                    "class": predictor.class_names.get(int(labels[i]), str(int(labels[i]))),
                    "confidence": round(scores[i], 4),
                    "bbox": [round(v, 1) for v in boxes[i]],
                })

            result = {
                "num_detections": len(detections),
                "detections": det_list,
            }

            return annotated_rgb, json.dumps(result, indent=2)

        detect_image_btn.click(
            fn=_detect_image,
            inputs=[
                image_input,
                conf_slider,
                file_input,
                dropdown_input,
                source_radio,
                class_checkboxes,
            ],
            outputs=[image_output, image_json],
        )

        def _process_video(
            video_path, conf, file_path, model_name, source, selected_classes
        ):
            """Process a video file (frame-by-frame, no tracking for generic tab)."""
            if video_path is None:
                return None, json.dumps({"error": "No video provided."})

            predictor = model_loader.load_from_ui(file_path, model_name, source, conf)
            if predictor is None:
                return None, json.dumps({"error": "Failed to load model."})

            import cv2

            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # Create temp output file
            output_dir = tempfile.mkdtemp(prefix="generic_video_")
            output_path = str(Path(output_dir) / "output.mp4")
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )

            annotators = build_annotators(config)
            name_to_id = {name: cid for cid, name in predictor.class_names.items()}

            frame_count = 0
            all_detections = []

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Detect
                    predictions = predictor.predict(frame)
                    detections = to_sv_detections(predictions)

                    # Filter
                    detections = ClassFilterComponent.filter_detections(
                        detections, selected_classes, name_to_id
                    )

                    all_detections.append(predictions)

                    # Annotate
                    annotated = annotate_frame(
                        frame, detections, predictor.class_names, annotators
                    )
                    writer.write(annotated)

                    frame_count += 1

            finally:
                cap.release()
                writer.release()

            # Build summary
            total_dets = sum(len(d.get("labels", [])) for d in all_detections)

            summary = {
                "input_video": str(video_path),
                "output_video": output_path,
                "frames_processed": frame_count,
                "total_detections": total_dets,
                "fps": fps,
                "resolution": [width, height],
            }

            return output_path, json.dumps(summary, indent=2)

        process_video_btn.click(
            fn=_process_video,
            inputs=[
                video_input,
                conf_slider,
                file_input,
                dropdown_input,
                source_radio,
                class_checkboxes,
            ],
            outputs=[video_output, video_json],
        )

        def _compute_metrics(image_json_str, gt_json_str, file_path, model_name, source):
            """Compute metrics from detection results vs ground truth."""
            if not gt_json_str:
                return "No ground truth provided.", None

            try:
                predictions = [json.loads(image_json_str)] if image_json_str else []
                ground_truths = [json.loads(gt_json_str)]
            except json.JSONDecodeError as exc:
                return f"Invalid JSON: {exc}", None

            predictor = model_loader.load_from_ui(file_path, model_name, source)
            if predictor is None:
                return "Model not loaded.", None

            # Compute metrics
            metrics = MetricsDisplay.compute_from_detections(
                predictions=predictions,
                ground_truths=ground_truths,
                class_names=predictor.class_names,
            )

            # Create table and chart
            table = MetricsDisplay.create_metrics_table(metrics)
            chart = MetricsDisplay.create_metrics_chart(metrics)

            return table, chart

        compute_metrics_btn.click(
            fn=_compute_metrics,
            inputs=[
                image_json,
                gt_json,
                file_input,
                dropdown_input,
                source_radio,
            ],
            outputs=[metrics_table, metrics_chart],
        )

        def _export_results(image_json_str, video_json_str):
            """Export detection results as downloadable JSON file."""
            results = {}

            if image_json_str:
                try:
                    results["image"] = json.loads(image_json_str)
                except json.JSONDecodeError:
                    pass

            if video_json_str:
                try:
                    results["video"] = json.loads(video_json_str)
                except json.JSONDecodeError:
                    pass

            if not results:
                return None

            # Write to temp file
            output_dir = tempfile.mkdtemp(prefix="generic_export_")
            output_path = str(Path(output_dir) / "detection_results.json")

            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

            return output_path

        export_results_btn.click(
            fn=_export_results,
            inputs=[image_json, video_json],
            outputs=gr.File(label="Download Results"),
        )

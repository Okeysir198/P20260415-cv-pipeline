"""Fall Detection tab -- classification-based and pose-based approaches.

Provides two sub-tabs: one using a fine-tuned classifier that detects
``person`` vs ``fallen_person`` (with Image + Video sub-tabs), and another
using pose estimation to compute torso angle and infer fall state from
keypoints (image-only).

All annotation uses the supervision library via ``supervision_bridge``.
"""

import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from app_demo.utils import COCO_SKELETON_EDGES, bgr_to_rgb, create_status_html, rgb_to_bgr
from core.p10_inference.supervision_bridge import (
    build_labels,
    to_sv_detections,
)
from core.p10_inference.video_inference import VideoProcessor


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pose model dropdown choices -> config paths
# ---------------------------------------------------------------------------

_POSE_MODEL_CONFIGS: Dict[str, str] = {
    "RTMPose-S": "features/safety-fall_pose_estimation/configs/rtmpose_s.yaml",
    "MediaPipe Full": "features/safety-fall_pose_estimation/configs/mediapipe_full.yaml",
    "MediaPipe Lite": "configs/pose/mediapipe_lite.yaml",
}

# COCO keypoint indices used for fall analysis
_IDX_LEFT_SHOULDER = 5
_IDX_RIGHT_SHOULDER = 6
_IDX_LEFT_HIP = 11
_IDX_RIGHT_HIP = 12


# ---------------------------------------------------------------------------
# Fall analysis from keypoints
# ---------------------------------------------------------------------------


def _analyze_fall(
    keypoints: np.ndarray,
    keypoint_names: List[str],
    config: dict,
) -> List[Dict[str, Any]]:
    """Analyze keypoints to detect fall using body orientation.

    Uses the torso angle (shoulder midpoint to hip midpoint) relative to
    vertical. An angle exceeding the configured threshold indicates a fall.

    Args:
        keypoints: Array of shape (N_persons, K, 3) with [x, y, conf].
        keypoint_names: K names for each keypoint.
        config: Demo config dict with optional
            ``use_cases.fall_pose.torso_angle_threshold``.

    Returns:
        List of per-person analysis dicts with ``torso_angle``, ``is_fallen``,
        and midpoint coordinates.
    """
    torso_threshold = (
        config.get("use_cases", {})
        .get("fall_pose", {})
        .get("torso_angle_threshold", 60)
    )
    kpt_conf_threshold = (
        config.get("use_cases", {})
        .get("fall_pose", {})
        .get("keypoint_conf_threshold", 0.3)
    )

    results: List[Dict[str, Any]] = []

    for person_kpts in keypoints:
        num_kpts = len(person_kpts)

        # Ensure we have enough keypoints for torso analysis
        if num_kpts <= max(_IDX_LEFT_SHOULDER, _IDX_RIGHT_SHOULDER,
                          _IDX_LEFT_HIP, _IDX_RIGHT_HIP):
            results.append({
                "torso_angle": None,
                "is_fallen": False,
                "status": "insufficient keypoints",
            })
            continue

        l_shoulder = person_kpts[_IDX_LEFT_SHOULDER]
        r_shoulder = person_kpts[_IDX_RIGHT_SHOULDER]
        l_hip = person_kpts[_IDX_LEFT_HIP]
        r_hip = person_kpts[_IDX_RIGHT_HIP]

        # Check if all required keypoints are visible
        if all(kpt[2] > kpt_conf_threshold for kpt in [l_shoulder, r_shoulder, l_hip, r_hip]):
            shoulder_mid = [
                (l_shoulder[0] + r_shoulder[0]) / 2,
                (l_shoulder[1] + r_shoulder[1]) / 2,
            ]
            hip_mid = [
                (l_hip[0] + r_hip[0]) / 2,
                (l_hip[1] + r_hip[1]) / 2,
            ]

            # Torso angle from vertical: 0 = upright, 90 = horizontal
            dx = shoulder_mid[0] - hip_mid[0]
            dy = shoulder_mid[1] - hip_mid[1]
            angle = abs(float(np.degrees(np.arctan2(dx, -dy))))

            is_fallen = angle > torso_threshold
            results.append({
                "torso_angle": round(angle, 1),
                "is_fallen": is_fallen,
                "shoulder_mid": [round(v, 1) for v in shoulder_mid],
                "hip_mid": [round(v, 1) for v in hip_mid],
            })
        else:
            results.append({
                "torso_angle": None,
                "is_fallen": False,
                "status": "keypoints not visible",
            })

    return results


# ---------------------------------------------------------------------------
# Alert HTML generation
# ---------------------------------------------------------------------------


def _status_badge(message: str, style: str = "safe") -> str:
    """Thin wrapper around ``create_status_html`` for fall-specific badges."""
    return create_status_html(status=style, message=message)


def _make_alert_html(has_fall: bool) -> str:
    """Generate alert HTML badge for fall detection status."""
    if has_fall:
        return _status_badge("FALL DETECTED", style="alert")
    return _status_badge("SAFE", style="safe")


def _make_model_status_html(model_type: str, class_names: Dict[int, str]) -> str:
    """Generate model status HTML showing loaded model info."""
    classes_str = ", ".join(class_names.values())
    if model_type == "fine-tuned":
        return _status_badge(f"Fine-tuned model loaded | Classes: {classes_str}", style="safe")
    return _status_badge(
        f"COCO pretrained (no fallen_person class) | Classes: {classes_str}",
        style="warning",
    )


def _make_pose_status_html(pose_predictor: Any, model_name: str) -> str:
    """Generate pose model status HTML."""
    if pose_predictor is not None:
        return _status_badge(f"{model_name} loaded", style="safe")
    return _status_badge(
        f"{model_name} -- weights not found. "
        "Download model weights to use pose-based detection.",
        style="alert",
    )


# ---------------------------------------------------------------------------
# Classification sub-tab logic
# ---------------------------------------------------------------------------


def _classify_detect_image(
    image: Optional[np.ndarray],
    conf_threshold: float,
    manager: Any,
    config: dict,
) -> Tuple[Optional[np.ndarray], str, str]:
    """Run classification-based fall detection on a single image.

    Args:
        image: RGB image from Gradio, or None.
        conf_threshold: Detection confidence threshold.
        manager: ModelManager instance.
        config: Demo config dict.

    Returns:
        Tuple of (annotated RGB image, alert HTML, results JSON string).
    """
    if image is None:
        return None, _make_alert_html(False), json.dumps({"error": "No image provided."})

    predictor, model_type = manager.get_use_case_predictor(
        "fall_cls", conf_threshold
    )

    # Gradio delivers RGB; predictor expects BGR
    bgr_image = rgb_to_bgr(image)

    predictions = predictor.predict(bgr_image)
    class_names = predictor.class_names

    # Use supervision for annotation
    detections = to_sv_detections(predictions)

    # Get alert classes from config
    alert_classes = set(config.get("use_cases", {}).get("fall_cls", {}).get("alert_classes", ["fallen_person"]))

    # Determine fall status from class names
    has_fall = False
    if detections.class_id is not None:
        for cid in detections.class_id:
            name = class_names.get(int(cid), "")
            if name in alert_classes:
                has_fall = True
                break

    # Build per-class color map: red for fallen, blue for standing
    fall_colors: Dict[str, sv.Color] = {}
    for cid, name in class_names.items():
        if name in alert_classes:
            fall_colors[name] = sv.Color(0, 0, 255)  # red in BGR
        else:
            fall_colors[name] = sv.Color(255, 0, 0)  # blue in BGR

    # Assign per-detection colors
    colors = [
        fall_colors.get(class_names.get(int(cid), ""), sv.Color(255, 0, 0))
        for cid in (detections.class_id if detections.class_id is not None else [])
    ]
    palette = sv.ColorPalette(colors) if colors else sv.ColorPalette.DEFAULT

    box_ann = sv.BoxAnnotator(thickness=2, color=palette)
    label_ann = sv.LabelAnnotator(
        text_scale=0.5, text_thickness=1, text_padding=5, color=palette,
    )

    annotated = box_ann.annotate(scene=bgr_image.copy(), detections=detections)
    labels = build_labels(detections, class_names)
    annotated = label_ann.annotate(scene=annotated, detections=detections, labels=labels)

    # Convert back to RGB for Gradio
    annotated_rgb = bgr_to_rgb(annotated)

    # Build results JSON
    detections_list = []
    boxes = np.asarray(predictions.get("boxes", [])).reshape(-1, 4)
    scores = np.asarray(predictions.get("scores", []))
    pred_labels = np.asarray(predictions.get("labels", []))
    for i in range(len(scores)):
        label_id = int(pred_labels[i])
        name = class_names.get(label_id, str(label_id))
        detections_list.append({
            "class": name,
            "confidence": round(float(scores[i]), 4),
            "bbox": [round(float(v), 1) for v in boxes[i]],
        })

    result: Dict[str, Any] = {
        "model_type": model_type,
        "num_detections": len(detections_list),
        "fall_detected": has_fall,
        "detections": detections_list,
    }

    if model_type == "coco-pretrained":
        result["notice"] = "Using COCO pretrained -- only 'person' class available, no fallen_person detection"

    return annotated_rgb, _make_alert_html(has_fall), json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Classification video logic
# ---------------------------------------------------------------------------


def _classify_detect_video(
    video_path: Optional[str],
    conf_threshold: float,
    enable_tracking: bool,
    manager: Any,
    config: dict,
) -> Tuple[Optional[str], str]:
    """Process a video file for classification-based fall detection.

    Args:
        video_path: Path to input video.
        conf_threshold: Confidence threshold.
        enable_tracking: Whether to enable ByteTrack tracking.
        manager: ModelManager instance.
        config: App configuration dict.

    Returns:
        Tuple of (output_video_path, summary text).
    """
    if not video_path:
        return None, "No video provided."

    predictor, model_type = manager.get_use_case_predictor(
        "fall_cls", conf_threshold
    )

    alert_config = {
        "confidence_thresholds": {"fallen_person": 0.7},
        "frame_windows": {"fallen_person": 15},
        "window_ratio": 0.8,
        "cooldown_frames": 90,
    }
    user_alerts = config.get("alerts", {}).get("fall_cls", {})
    if user_alerts:
        alert_config.update(user_alerts)

    tracker_config = config.get("tracker") if enable_tracking else None

    processor = VideoProcessor(
        predictor=predictor,
        alert_config=alert_config,
        enable_tracking=enable_tracking,
        tracker_config=tracker_config,
    )

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        output_path = tmp.name

    summary = processor.process_video(
        video_path=video_path,
        output_path=output_path,
    )

    # Build summary text
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
                f"Frame {alert.get('frame_idx', '?')}: "
                f"{alert.get('type', 'unknown')} "
                f"(conf={alert.get('confidence', 0):.2f})"
            )
    else:
        lines.append("No fall alerts triggered.")

    class_counts = summary.get("class_counts", {})
    if class_counts:
        lines.append("")
        lines.append("--- Class Counts ---")
        for cls_name, count in class_counts.items():
            lines.append(f"  {cls_name}: {count}")

    return output_path, "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared pose annotation (supervision-based)
# ---------------------------------------------------------------------------


def _annotate_pose_frame(
    bgr_frame: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    keypoints: np.ndarray,
    skeleton: List[Tuple],
    fall_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """Annotate a frame with person boxes and skeleton using supervision.

    Uses ``sv.BoxAnnotator`` + ``sv.LabelAnnotator`` for person boxes and
    ``sv.EdgeAnnotator`` + ``sv.VertexAnnotator`` for skeleton keypoints.

    Args:
        bgr_frame: BGR image to annotate.
        boxes: (N, 4) person bounding boxes.
        scores: (N,) detection confidence scores.
        keypoints: (N, K, 3) keypoints [x, y, conf].
        skeleton: List of (start, end) edge pairs.
        fall_indices: Indices of fallen persons (drawn in red).

    Returns:
        Annotated BGR image.
    """
    if fall_indices is None:
        fall_indices = []

    annotated = bgr_frame.copy()
    num_persons = len(keypoints)

    # --- Person bounding boxes ---
    if len(boxes) > 0:
        colors = []
        det_labels = []
        for i in range(len(boxes)):
            is_fallen = i in fall_indices
            colors.append(sv.Color(0, 0, 255) if is_fallen else sv.Color(0, 255, 0))
            status_text = "FALLEN" if is_fallen else "standing"
            conf = float(scores[i]) if i < len(scores) else 0.0
            det_labels.append(f"person {conf:.2f} [{status_text}]")

        palette = sv.ColorPalette(colors)
        person_dets = sv.Detections(
            xyxy=boxes.reshape(-1, 4).astype(np.float32),
            confidence=scores.astype(np.float32) if len(scores) > 0 else None,
            class_id=np.zeros(len(boxes), dtype=int),
        )

        box_ann = sv.BoxAnnotator(thickness=2, color=palette)
        label_ann = sv.LabelAnnotator(
            text_scale=0.5, text_thickness=1, text_padding=5, color=palette,
        )
        annotated = box_ann.annotate(scene=annotated, detections=person_dets)
        annotated = label_ann.annotate(
            scene=annotated, detections=person_dets, labels=det_labels,
        )

    # --- Skeleton keypoints using supervision ---
    if num_persons > 0:
        # Build sv.KeyPoints from pose results
        # xy shape: (N, K, 2), confidence shape: (N, K)
        xy = keypoints[:, :, :2].astype(np.float32)
        kpt_conf = keypoints[:, :, 2].astype(np.float32)

        sv_kpts = sv.KeyPoints(
            xy=xy,
            confidence=kpt_conf,
        )

        # Use skeleton edges for EdgeAnnotator
        edges = skeleton if skeleton else COCO_SKELETON_EDGES
        edge_ann = sv.EdgeAnnotator(
            color=sv.Color(0, 255, 255),  # cyan
            thickness=2,
            edges=edges,
        )
        vertex_ann = sv.VertexAnnotator(
            color=sv.Color(0, 255, 0),  # green
            radius=4,
        )
        annotated = edge_ann.annotate(scene=annotated, key_points=sv_kpts)
        annotated = vertex_ann.annotate(scene=annotated, key_points=sv_kpts)

    return annotated


# ---------------------------------------------------------------------------
# Pose video processing
# ---------------------------------------------------------------------------


def _pose_detect_video(
    video_path: Optional[str],
    pose_model_name: str,
    conf_threshold: float,
    manager: Any,
    config: dict,
) -> Tuple[Optional[str], str]:
    """Process a video with pose-based fall detection.

    Runs person detection + pose estimation on each frame, draws skeleton
    overlay and fall alerts using supervision annotators.
    """
    if not video_path:
        return None, "No video provided."

    pose_config_path = _POSE_MODEL_CONFIGS.get(pose_model_name)
    if pose_config_path is None:
        return None, f"Unknown pose model: {pose_model_name}"

    pose_predictor = manager.get_pose_predictor(
        pose_config_path, person_class_ids=[0]
    )
    if pose_predictor is None:
        return None, f"Pose model '{pose_model_name}' weights not found."

    video_info = sv.VideoInfo.from_video_path(video_path)
    generator = sv.get_video_frames_generator(video_path)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        output_path = tmp.name

    total_falls = 0
    total_persons = 0
    frame_idx = 0

    with sv.VideoSink(target_path=output_path, video_info=video_info) as sink:
        for frame in generator:
            result = pose_predictor.predict(frame)

            kpts = np.asarray(result.get("keypoints", []))
            skeleton = result.get("skeleton", [])
            boxes = np.asarray(result.get("boxes", []))
            scores_arr = np.asarray(result.get("scores", []))
            num = len(kpts)
            total_persons += num

            analyses = _analyze_fall(kpts, result.get("keypoint_names", []), config) if num > 0 else []
            fall_idx = [i for i, a in enumerate(analyses) if a.get("is_fallen", False)]
            total_falls += len(fall_idx)

            annotated = _annotate_pose_frame(
                frame, boxes, scores_arr, kpts, skeleton, fall_idx
            )

            # Draw fall alert banner
            if fall_idx:
                cv2.putText(
                    annotated, "FALL DETECTED", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3,
                )

            sink.write_frame(annotated)
            frame_idx += 1

    summary_lines = [
        f"Pose model: {pose_model_name}",
        f"Total frames: {frame_idx}",
        f"Total persons detected: {total_persons}",
        f"Total fall detections: {total_falls}",
        f"Resolution: {video_info.width}x{video_info.height}",
        f"FPS: {video_info.fps}",
    ]

    return output_path, "\n".join(summary_lines)


# ---------------------------------------------------------------------------
# Pose sub-tab logic (image)
# ---------------------------------------------------------------------------


def _pose_detect(
    image: Optional[np.ndarray],
    pose_model_name: str,
    conf_threshold: float,
    manager: Any,
    config: dict,
) -> Tuple[Optional[np.ndarray], str, str]:
    """Run pose-based fall detection on a single image.

    Args:
        image: RGB image from Gradio, or None.
        pose_model_name: Name of pose model from dropdown.
        conf_threshold: Detection confidence threshold.
        manager: ModelManager instance.
        config: Demo config dict.

    Returns:
        Tuple of (annotated RGB image, pose metrics JSON string, fall alert HTML).
    """
    if image is None:
        return (
            None,
            json.dumps({"error": "No image provided."}),
            _make_alert_html(False),
        )

    pose_config_path = _POSE_MODEL_CONFIGS.get(pose_model_name)
    if pose_config_path is None:
        return (
            None,
            json.dumps({"error": f"Unknown pose model: {pose_model_name}"}),
            _make_alert_html(False),
        )

    pose_predictor = manager.get_pose_predictor(
        pose_config_path, person_class_ids=[0]
    )
    if pose_predictor is None:
        return (
            None,
            json.dumps({"error": f"Pose model '{pose_model_name}' weights not found."}),
            _make_pose_status_html(None, pose_model_name),
        )

    # Gradio delivers RGB; predictor expects BGR
    bgr_image = rgb_to_bgr(image)

    result = pose_predictor.predict(bgr_image)

    keypoints = np.asarray(result.get("keypoints", []))
    keypoint_names = result.get("keypoint_names", [])
    skeleton = result.get("skeleton", [])
    boxes = np.asarray(result.get("boxes", []))
    scores = np.asarray(result.get("scores", []))

    num_persons = len(keypoints)

    # Analyze fall from keypoints
    analyses = _analyze_fall(keypoints, keypoint_names, config) if num_persons > 0 else []

    # Determine which persons are fallen
    fall_indices = [i for i, a in enumerate(analyses) if a.get("is_fallen", False)]
    has_fall = len(fall_indices) > 0

    # Annotate with supervision
    annotated = _annotate_pose_frame(
        bgr_image, boxes, scores, keypoints, skeleton, fall_indices
    )

    # Convert back to RGB for Gradio
    annotated_rgb = bgr_to_rgb(annotated)

    # Build pose metrics JSON
    pose_analysis = []
    for i, analysis in enumerate(analyses):
        entry: Dict[str, Any] = {"person": i}
        if analysis.get("torso_angle") is not None:
            entry["torso_angle"] = analysis["torso_angle"]
            entry["status"] = "FALLEN" if analysis["is_fallen"] else "standing"
        else:
            entry["status"] = analysis.get("status", "unknown")
        pose_analysis.append(entry)

    metrics = {
        "num_persons": num_persons,
        "pose_model": pose_model_name,
        "fall_detected": has_fall,
        "pose_analysis": pose_analysis,
    }

    return annotated_rgb, json.dumps(metrics, indent=2), _make_alert_html(has_fall)


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------


def build_tab_fall(manager: Any, config: dict) -> None:
    """Build the Fall Detection tab with classification and pose sub-tabs.

    Classification sub-tab has Image + Video modes. Pose sub-tab is image-only.

    Args:
        manager: ``ModelManager`` instance for model loading and caching.
        config: Demo config dict with ``use_cases.fall_pose`` settings.
    """
    with gr.Tab("Fall Detection"):
        gr.Markdown("## Fall Detection")
        gr.Markdown("Two approaches: classification-based and pose-based.")

        with gr.Tabs():
            # ---- Classification sub-tab ----
            with gr.Tab("Classification"):
                # Check model availability without loading
                fine_tuned = manager.discover_fine_tuned()
                model_type = "fine-tuned" if "fall_cls" in fine_tuned else "coco-pretrained"
                cls_names = {0: "person", 1: "fallen_person"} if model_type == "fine-tuned" else {0: "person"}
                initial_status = _make_model_status_html(model_type, cls_names)

                with gr.Row():
                    cls_model_status = gr.HTML(value=initial_status)
                    cls_conf_slider = gr.Slider(
                        minimum=0.05,
                        maximum=0.95,
                        value=0.25,
                        step=0.05,
                        label="Confidence Threshold",
                    )

                with gr.Tabs():
                    # -- Image sub-tab --
                    with gr.Tab("Image"):
                        with gr.Row():
                            with gr.Column():
                                cls_image_input = gr.Image(
                                    type="numpy", label="Upload Image"
                                )
                                cls_detect_btn = gr.Button("Detect", variant="primary")
                            with gr.Column():
                                cls_image_output = gr.Image(
                                    type="numpy", label="Annotated Result"
                                )
                                cls_alert_html = gr.HTML(value=_make_alert_html(False))
                                cls_results_json = gr.Textbox(
                                    label="Detection Results (JSON)",
                                    lines=10,
                                    interactive=False,
                                )

                        cls_detect_btn.click(
                            fn=lambda img, conf: _classify_detect_image(
                                img, conf, manager, config
                            ),
                            inputs=[cls_image_input, cls_conf_slider],
                            outputs=[cls_image_output, cls_alert_html, cls_results_json],
                        )

                    # -- Video sub-tab --
                    with gr.Tab("Video"):
                        with gr.Row():
                            with gr.Column():
                                cls_video_input = gr.Video(label="Upload Video")
                                cls_tracking_checkbox = gr.Checkbox(
                                    label="Enable Tracking (ByteTrack)",
                                    value=False,
                                )
                                cls_video_btn = gr.Button(
                                    "Process Video", variant="primary"
                                )
                            with gr.Column():
                                cls_video_output = gr.Video(label="Processed Video")
                                cls_video_log = gr.Textbox(
                                    label="Processing Summary",
                                    lines=12,
                                    interactive=False,
                                )

                        cls_video_btn.click(
                            fn=lambda vid, conf, track: _classify_detect_video(
                                vid, conf, track, manager, config,
                            ),
                            inputs=[
                                cls_video_input,
                                cls_conf_slider,
                                cls_tracking_checkbox,
                            ],
                            outputs=[cls_video_output, cls_video_log],
                        )

            # ---- Pose Estimation sub-tab (Image + Video) ----
            with gr.Tab("Pose Estimation"):
                pose_model_choices = list(_POSE_MODEL_CONFIGS.keys())
                default_pose = pose_model_choices[0]

                initial_pose_status = _status_badge(
                    "Pose model will load on first use", "warning"
                )

                with gr.Row():
                    pose_model_dropdown = gr.Dropdown(
                        choices=pose_model_choices,
                        value=default_pose,
                        label="Pose Model",
                        interactive=True,
                    )
                    pose_conf_slider = gr.Slider(
                        minimum=0.05,
                        maximum=0.95,
                        value=0.25,
                        step=0.05,
                        label="Confidence Threshold",
                    )

                with gr.Row():
                    pose_status_html = gr.HTML(value=initial_pose_status)

                # Update pose status when model changes
                def _update_pose_status(model_name: str) -> str:
                    cfg_path = _POSE_MODEL_CONFIGS.get(model_name)
                    if cfg_path is None:
                        return _make_pose_status_html(None, model_name)
                    try:
                        pred = manager.get_pose_predictor(
                            cfg_path, person_class_ids=[0]
                        )
                    except Exception:
                        pred = None
                    return _make_pose_status_html(pred, model_name)

                pose_model_dropdown.change(
                    fn=_update_pose_status,
                    inputs=[pose_model_dropdown],
                    outputs=[pose_status_html],
                )

                with gr.Tabs():
                    # -- Image sub-tab --
                    with gr.Tab("Image"):
                        with gr.Row():
                            with gr.Column():
                                pose_image_input = gr.Image(
                                    type="numpy", label="Upload Image"
                                )
                                pose_detect_btn = gr.Button("Detect", variant="primary")
                            with gr.Column():
                                pose_image_output = gr.Image(
                                    type="numpy", label="Skeleton Overlay"
                                )
                                pose_metrics_json = gr.Textbox(
                                    label="Pose Metrics (JSON)",
                                    lines=10,
                                    interactive=False,
                                )
                                pose_fall_alert = gr.HTML(
                                    value=_make_alert_html(False)
                                )

                        pose_detect_btn.click(
                            fn=lambda img, model, conf: _pose_detect(
                                img, model, conf, manager, config
                            ),
                            inputs=[
                                pose_image_input,
                                pose_model_dropdown,
                                pose_conf_slider,
                            ],
                            outputs=[
                                pose_image_output,
                                pose_metrics_json,
                                pose_fall_alert,
                            ],
                        )

                    # -- Video sub-tab --
                    with gr.Tab("Video"):
                        with gr.Row():
                            with gr.Column():
                                pose_video_input = gr.Video(label="Upload Video")
                                pose_video_btn = gr.Button(
                                    "Process Video", variant="primary"
                                )
                            with gr.Column():
                                pose_video_output = gr.Video(label="Processed Video")
                                pose_video_log = gr.Textbox(
                                    label="Processing Summary",
                                    lines=12,
                                    interactive=False,
                                )

                        pose_video_btn.click(
                            fn=lambda vid, model, conf: _pose_detect_video(
                                vid, model, conf, manager, config,
                            ),
                            inputs=[
                                pose_video_input,
                                pose_model_dropdown,
                                pose_conf_slider,
                            ],
                            outputs=[pose_video_output, pose_video_log],
                        )

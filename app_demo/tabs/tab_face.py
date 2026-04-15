"""Face Recognition tab — enroll faces, identify persons, manage gallery.

Supports image and video identification with supervision-based annotation.
"""

import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional, Tuple

import gradio as gr
import numpy as np
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from app_demo.utils import bgr_to_rgb, rgb_to_bgr

logger = logging.getLogger(__name__)


def _check_models_available() -> bool:
    """Check whether SCRFD and MobileFaceNet ONNX models exist in pretrained/."""
    pretrained = Path(__file__).resolve().parent.parent.parent / "pretrained"
    return (pretrained / "scrfd_500m.onnx").exists() and (
        pretrained / "mobilefacenet_arcface.onnx"
    ).exists()


def _status_html() -> str:
    """Return an HTML status banner for model availability."""
    if _check_models_available():
        return (
            "<p style='color:green'>SCRFD-500M and MobileFaceNet ONNX models found "
            "in <code>pretrained/</code>.</p>"
        )
    return (
        "<p style='color:red'>Face ONNX models not found in <code>pretrained/</code>. "
        "Download <code>buffalo_sc.zip</code> from "
        "<a href='https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip'>"
        "InsightFace releases</a>, then rename "
        "<code>det_500m.onnx</code> &rarr; <code>pretrained/scrfd_500m.onnx</code> and "
        "<code>w600k_mbf.onnx</code> &rarr; <code>pretrained/mobilefacenet_arcface.onnx</code>.</p>"
    )


# ------------------------------------------------------------------
# Supervision annotators for face boxes
# ------------------------------------------------------------------

_box_annotator = sv.BoxAnnotator(thickness=2)
_label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=5)


def _annotate_faces(
    bgr_frame: np.ndarray,
    boxes: np.ndarray,
    identities: list[str | None],
    similarities: list[float],
) -> np.ndarray:
    """Draw face boxes and identity labels using supervision annotators.

    Args:
        bgr_frame: BGR image to annotate (modified in place).
        boxes: (N, 4) xyxy face boxes.
        identities: Identity name per face (None for unknown).
        similarities: Similarity score per face.

    Returns:
        Annotated BGR image.
    """
    if len(boxes) == 0:
        return bgr_frame

    detections = sv.Detections(xyxy=np.asarray(boxes, dtype=np.float32))

    labels = []
    for identity, sim in zip(identities, similarities):
        if identity:
            labels.append(f"{identity} ({sim:.2f})")
        else:
            labels.append(f"Unknown ({sim:.2f})")

    annotated = _box_annotator.annotate(scene=bgr_frame, detections=detections)
    annotated = _label_annotator.annotate(
        scene=annotated, detections=detections, labels=labels
    )
    return annotated


# ------------------------------------------------------------------
# Shared face detection + matching logic
# ------------------------------------------------------------------

def _detect_and_match_faces(
    bgr: np.ndarray,
    face_predictor: Any,
) -> Tuple[np.ndarray, list[dict]]:
    """Detect faces, embed each, match against gallery, and annotate.

    Args:
        bgr: BGR image.
        face_predictor: FacePredictor instance.

    Returns:
        Tuple of (annotated_bgr, results_list).
    """
    face_results = face_predictor.face_detector.detect(bgr)
    n_faces = face_results["boxes"].shape[0]

    if n_faces == 0:
        return bgr, []

    boxes = face_results["boxes"]
    scores = face_results["scores"]
    has_landmarks = "landmarks" in face_results

    identities: list[str | None] = []
    similarities: list[float] = []
    results: list[dict] = []

    for i in range(n_faces):
        box = boxes[i]
        landmarks = face_results["landmarks"][i] if has_landmarks else None
        embedding = face_predictor.face_embedder.extract(bgr, box, landmarks)
        identity, similarity = face_predictor.gallery.match(embedding)

        identities.append(identity)
        similarities.append(similarity)

        x1, y1, x2, y2 = box.astype(int)
        results.append({
            "face_idx": i,
            "identity": identity,
            "similarity": round(float(similarity), 4),
            "confidence": round(float(scores[i]), 4),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
        })

    annotated = _annotate_faces(bgr.copy(), boxes, identities, similarities)
    return annotated, results


# ------------------------------------------------------------------
# Enrollment
# ------------------------------------------------------------------

def _enroll_face(
    identity_name: str | None,
    face_image: np.ndarray | None,
    manager: object,
) -> tuple[str, str]:
    """Detect the most-confident face in *face_image*, embed it, and enroll."""
    if not identity_name or face_image is None:
        return (
            "<p style='color:red'>Please provide both identity name and face photo.</p>",
            "",
        )

    face_predictor = manager.get_face_predictor()
    if face_predictor is None:
        return (
            "<p style='color:red'>Face ONNX models not found in pretrained/</p>",
            "",
        )

    bgr = rgb_to_bgr(face_image)

    face_results = face_predictor.face_detector.detect(bgr)
    if face_results["boxes"].shape[0] == 0:
        return "<p style='color:red'>No face detected in the image.</p>", ""

    best_idx = int(face_results["scores"].argmax())
    face_box = face_results["boxes"][best_idx]
    landmarks = (
        face_results["landmarks"][best_idx]
        if "landmarks" in face_results
        else None
    )

    embedding = face_predictor.face_embedder.extract(bgr, face_box, landmarks)

    face_predictor.gallery.enroll(identity_name, embedding)
    face_predictor.gallery.save()

    return (
        f"<p style='color:green'>Successfully enrolled <b>{identity_name}</b></p>",
        f"Gallery size: {face_predictor.gallery.size()}",
    )


# ------------------------------------------------------------------
# Image identification
# ------------------------------------------------------------------

def _identify_faces(
    test_image: np.ndarray | None,
    manager: object,
) -> tuple[np.ndarray | None, dict]:
    """Detect all faces, embed each, and match against the gallery."""
    if test_image is None:
        return None, {}

    face_predictor = manager.get_face_predictor()
    if face_predictor is None:
        return None, {"error": "Face ONNX models not found"}

    bgr = rgb_to_bgr(test_image)
    annotated_bgr, results = _detect_and_match_faces(bgr, face_predictor)

    if not results:
        return test_image, {"message": "No faces detected"}

    annotated_rgb = bgr_to_rgb(annotated_bgr)
    return annotated_rgb, {"faces": results, "total": len(results)}


# ------------------------------------------------------------------
# Video identification
# ------------------------------------------------------------------

def _identify_faces_video(
    video_path: str | None,
    manager: object,
) -> tuple[str | None, str]:
    """Process a video for face identification using supervision video utilities.

    Args:
        video_path: Path to input video file.
        manager: ModelManager instance.

    Returns:
        Tuple of (output_video_path, results_json).
    """
    if not video_path:
        return None, json.dumps({"error": "No video provided."})

    face_predictor = manager.get_face_predictor()
    if face_predictor is None:
        return None, json.dumps({"error": "Face ONNX models not found"})

    video_info = sv.VideoInfo.from_video_path(video_path)
    generator = sv.get_video_frames_generator(video_path)

    output_dir = tempfile.mkdtemp(prefix="face_video_")
    output_path = str(Path(output_dir) / "output.mp4")

    total_faces = 0
    identified_count = 0
    identity_counts: dict[str, int] = {}

    with sv.VideoSink(output_path, video_info) as sink:
        for frame in generator:
            # frame is BGR from supervision
            annotated, results = _detect_and_match_faces(frame, face_predictor)
            total_faces += len(results)

            for r in results:
                name = r["identity"]
                if name:
                    identified_count += 1
                    identity_counts[name] = identity_counts.get(name, 0) + 1

            sink.write_frame(annotated)

    summary = {
        "total_frames": video_info.total_frames,
        "resolution": f"{video_info.width}x{video_info.height}",
        "fps": video_info.fps,
        "total_faces_detected": total_faces,
        "identified": identified_count,
        "unknown": total_faces - identified_count,
        "identity_counts": identity_counts,
    }

    return output_path, json.dumps(summary, indent=2)


# ------------------------------------------------------------------
# Gallery management
# ------------------------------------------------------------------

def _get_gallery_html(manager: object) -> str:
    """Render the gallery contents as an HTML table."""
    gallery = manager.get_face_gallery()
    if gallery is None:
        return "<p>Face models not available.</p>"

    identities = gallery.unique_identities
    if not identities:
        return "<p>Gallery is empty. Enroll faces in the Enrollment tab.</p>"

    rows = "".join(
        f"<tr><td>{i}</td><td>{name}</td></tr>"
        for i, name in enumerate(identities, 1)
    )
    return (
        "<table style='width:100%'><tr><th>#</th><th>Identity</th></tr>"
        f"{rows}</table>"
        f"<p>Total: {len(identities)} identities</p>"
    )


def _delete_identity(
    name: str | None,
    manager: object,
) -> tuple[str, str]:
    """Remove a single identity from the gallery."""
    if not name:
        return "<p style='color:red'>Please enter an identity name.</p>", _get_gallery_html(manager)

    gallery = manager.get_face_gallery()
    if gallery and gallery.remove(name):
        gallery.save()
        return (
            f"<p style='color:green'>Deleted {name}</p>",
            _get_gallery_html(manager),
        )
    return (
        f"<p style='color:red'>Identity '{name}' not found</p>",
        _get_gallery_html(manager),
    )


def _clear_gallery(manager: object) -> tuple[str, str]:
    """Remove every identity from the gallery."""
    gallery = manager.get_face_gallery()
    if gallery:
        for identity in gallery.unique_identities:
            gallery.remove(identity)
        gallery.save()
    return "<p style='color:green'>Gallery cleared</p>", _get_gallery_html(manager)


# ------------------------------------------------------------------
# Public builder
# ------------------------------------------------------------------

def build_tab_face(manager: object, config: dict) -> None:
    """Construct the Face Recognition Gradio tab."""
    with gr.Tab("Face Recognition"):
        gr.Markdown("## Face Recognition")
        gr.Markdown("Enroll faces into gallery and identify persons in images or videos.")

        status = gr.HTML(value=_status_html())

        with gr.Tabs():
            # ---------- Enrollment ----------
            with gr.Tab("Enrollment"):
                with gr.Row():
                    with gr.Column():
                        identity_name = gr.Textbox(
                            label="Identity Name",
                            placeholder="e.g., worker_A",
                        )
                        face_image = gr.Image(
                            type="numpy", label="Face Photo",
                        )
                        enroll_btn = gr.Button(
                            "Enroll Face", variant="primary",
                        )
                    with gr.Column():
                        enrollment_result_html = gr.HTML()
                        gallery_count_text = gr.Textbox(
                            label="Gallery Size", interactive=False,
                        )

                enroll_btn.click(
                    fn=lambda name, img: _enroll_face(name, img, manager),
                    inputs=[identity_name, face_image],
                    outputs=[enrollment_result_html, gallery_count_text],
                )

            # ---------- Identification ----------
            with gr.Tab("Identification"):
                with gr.Tabs():
                    # ---- Image sub-tab ----
                    with gr.Tab("Image"):
                        with gr.Row():
                            with gr.Column():
                                test_image = gr.Image(
                                    type="numpy", label="Test Image",
                                )
                                identify_btn = gr.Button(
                                    "Identify Faces", variant="primary",
                                )
                            with gr.Column():
                                annotated_output = gr.Image(label="Identified Faces")
                                results_json = gr.JSON(label="Recognition Results")

                        identify_btn.click(
                            fn=lambda img: _identify_faces(img, manager),
                            inputs=[test_image],
                            outputs=[annotated_output, results_json],
                        )

                    # ---- Video sub-tab ----
                    with gr.Tab("Video"):
                        with gr.Row():
                            with gr.Column():
                                face_video_input = gr.Video(label="Input Video")
                                face_video_btn = gr.Button(
                                    "Process Video", variant="primary",
                                )
                            with gr.Column():
                                face_video_output = gr.Video(label="Processed Video")
                                face_video_json = gr.Textbox(
                                    label="Recognition Summary (JSON)",
                                    lines=12,
                                    interactive=False,
                                )

                        face_video_btn.click(
                            fn=lambda vid: _identify_faces_video(vid, manager),
                            inputs=[face_video_input],
                            outputs=[face_video_output, face_video_json],
                        )

            # ---------- Gallery ----------
            with gr.Tab("Gallery"):
                gallery_list_html = gr.HTML(
                    value="<p>Click Refresh to load gallery.</p>",
                )
                with gr.Row():
                    delete_name = gr.Textbox(label="Identity to Delete")
                    delete_btn = gr.Button("Delete", variant="stop")
                clear_all_btn = gr.Button("Clear Gallery", variant="stop")
                refresh_btn = gr.Button("Refresh List")

                delete_btn.click(
                    fn=lambda name: _delete_identity(name, manager),
                    inputs=[delete_name],
                    outputs=[status, gallery_list_html],
                )
                clear_all_btn.click(
                    fn=lambda: _clear_gallery(manager),
                    inputs=[],
                    outputs=[status, gallery_list_html],
                )
                refresh_btn.click(
                    fn=lambda: _get_gallery_html(manager),
                    inputs=[],
                    outputs=[gallery_list_html],
                )

"""Shared helper utilities for the demo application.

Provides color conversion, result formatting, image annotation, keypoint
drawing, and HTML status badge generation used across all demo tabs.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root

from core.p10_inference.supervision_bridge import (
    annotate_frame,
    build_annotators,
    to_sv_detections,
)
from utils.viz import COCO_SKELETON_EDGES, annotate_keypoints


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to BGR for OpenCV/predictor.

    Args:
        image: RGB image array (H, W, 3).

    Returns:
        BGR image array (H, W, 3).
    """
    return image[:, :, ::-1].copy()


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to RGB for Gradio display.

    Args:
        image: BGR image array (H, W, 3).

    Returns:
        RGB image array (H, W, 3).
    """
    return image[:, :, ::-1].copy()


def format_results_json(
    predictions: Dict[str, Any],
    model_name: str = "",
    conf_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Format detection results as a structured dict for Gradio JSON display.

    Args:
        predictions: Dict with ``boxes`` (N,4), ``scores`` (N,), ``labels`` (N,),
            and ``class_names`` (N,) as returned by predictor.predict().
        model_name: Display name of the model used.
        conf_threshold: Confidence threshold applied during detection.

    Returns:
        Structured dict with model info, threshold, detection count, and
        per-detection details (class, confidence, bbox).
    """
    detections = []
    boxes = predictions.get("boxes", [])
    scores = predictions.get("scores", [])
    class_names = predictions.get("class_names", [])

    for i in range(len(scores)):
        detections.append({
            "class": class_names[i] if i < len(class_names) else "unknown",
            "confidence": round(float(scores[i]), 4),
            "bbox": [round(float(v), 1) for v in boxes[i].tolist()],
        })

    return {
        "model": model_name,
        "confidence_threshold": conf_threshold,
        "num_detections": len(detections),
        "detections": detections,
    }


def annotate_image(
    image_bgr: np.ndarray,
    predictions: Dict[str, Any],
    class_names: Dict[int, str],
    config: Dict,
) -> np.ndarray:
    """Annotate image with detection boxes using supervision bridge.

    Args:
        image_bgr: BGR image (H, W, 3).
        predictions: Dict with ``boxes``, ``scores``, ``labels`` from predictor.
        class_names: Mapping from class ID to display name.
        config: Demo config dict containing ``supervision`` annotator settings.

    Returns:
        Annotated BGR image.
    """
    detections = to_sv_detections(predictions)
    annotators = build_annotators(config)
    return annotate_frame(
        frame=image_bgr,
        detections=detections,
        class_names=class_names,
        annotators=annotators,
    )


def draw_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    skeleton: Optional[List[Tuple[int, int]]] = None,
    keypoint_names: Optional[List[str]] = None,
    conf_threshold: float = 0.3,
) -> np.ndarray:
    """Draw pose keypoints and skeleton on image.

    Thin wrapper around :func:`utils.viz.annotate_keypoints` preserved for
    backwards-compatible callers (e.g. ``tabs/tab_fall.py``). New code should
    call ``utils.viz.annotate_keypoints`` directly.

    Args:
        image: Image array (H, W, 3). A copy is made internally.
        keypoints: (K, 3) array where each row is (x, y, confidence).
        skeleton: (start, end) edge pairs. Defaults to COCO 17-keypoint.
        keypoint_names: Optional keypoint names (unused, kept for API compat).
        conf_threshold: Minimum confidence to draw a keypoint.

    Returns:
        Annotated image.
    """
    del keypoint_names  # API compatibility only
    edges = skeleton if skeleton else COCO_SKELETON_EDGES
    kpts = np.asarray(keypoints, dtype=np.float32)
    xy = kpts[:, :2]
    conf = kpts[:, 2] if kpts.shape[1] >= 3 else None

    from utils.viz import VizStyle

    style = VizStyle()
    style.kpt_visibility_threshold = conf_threshold
    return annotate_keypoints(
        image,
        keypoints_xy=xy,
        skeleton_edges=edges,
        confidence=conf,
        style=style,
    )


def create_status_html(status: str, message: str) -> str:
    """Create HTML status badge for display in Gradio.

    Args:
        status: One of ``"safe"``, ``"warning"``, or ``"alert"``.
        message: Text message to display in the badge.

    Returns:
        HTML string with a colored status badge.
    """
    colors = {
        "safe": ("#d4edda", "#155724", "#c3e6cb"),
        "warning": ("#fff3cd", "#856404", "#ffeeba"),
        "alert": ("#f8d7da", "#721c24", "#f5c6cb"),
    }
    bg, text_color, border_color = colors.get(status, colors["warning"])
    return (
        f'<div style="padding: 12px 16px; border-radius: 8px; '
        f"background-color: {bg}; color: {text_color}; "
        f"border: 1px solid {border_color}; "
        f'font-weight: 600; font-size: 14px;">'
        f"{message}</div>"
    )


def create_model_info_html(
    model_name: str, model_type: str, num_classes: int
) -> str:
    """Create HTML badge showing model info.

    Args:
        model_name: Display name of the model.
        model_type: Architecture type (e.g., ``"YOLOX-M"``, ``"D-FINE-S"``).
        num_classes: Number of detection classes.

    Returns:
        HTML string with model information badge.
    """
    return (
        f'<div style="padding: 8px 12px; border-radius: 6px; '
        f"background-color: #e8f4fd; color: #0c5460; "
        f"border: 1px solid #bee5eb; font-size: 13px; "
        f'display: inline-block; margin: 4px 0;">'
        f"<b>{model_name}</b> | {model_type} | {num_classes} classes"
        f"</div>"
    )

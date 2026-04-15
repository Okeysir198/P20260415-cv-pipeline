"""Shared fixtures and helpers for annotation_quality_assessment service tests.

All tests require the annotation_quality_assessment service running at localhost:18105.
SAM3 is optional — /verify endpoint returns 503 if SAM3 is unavailable.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import cv2
import numpy as np
import pytest
import requests
import supervision as sv

SERVICE_URL = "http://localhost:18105"
SAM3_URL = "http://localhost:18100"
DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def _service_available() -> bool:
    """Check if the annotation_quality_assessment service is reachable."""
    try:
        resp = requests.get(f"{SERVICE_URL}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def _sam3_available() -> bool:
    """Check if SAM3 service is reachable (for /verify endpoint)."""
    try:
        resp = requests.get(f"{SAM3_URL}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


skip_no_service = pytest.mark.skipif(
    not _service_available(),
    reason="annotation_quality_assessment service not running at :18105",
)

skip_no_sam3 = pytest.mark.skipif(
    not _sam3_available(),
    reason="SAM3 service not running at :18100 (required for /verify)",
)

OLLAMA_URL = "http://localhost:11434"


def _ollama_available() -> bool:
    """Check if Ollama is reachable (for VLM tests)."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


skip_no_ollama = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama not running at :11434 (required for VLM tests)",
)


@pytest.fixture(scope="session", autouse=True)
def ensure_output_dir():
    """Create outputs directory once per test session."""
    OUTPUT_DIR.mkdir(exist_ok=True)


def load_image_b64(filename: str) -> str:
    """Load an image from data/ and return base64-encoded string."""
    path = DATA_DIR / filename
    if not path.exists():
        pytest.skip(f"Test image not found: {path}")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def detections_to_sv(detections: list[dict], img_w: int, img_h: int) -> sv.Detections:
    """Convert annotation_quality_assessment API detections to sv.Detections.

    Handles both bbox_xyxy and polygon formats.
    """
    if not detections:
        return sv.Detections.empty()

    xyxy = []
    scores = []
    class_ids = []
    masks_list = []

    for det in detections:
        # Try bbox_xyxy first (pixel coords)
        bbox = det.get("bbox_xyxy", [])
        if len(bbox) == 4:
            xyxy.append(bbox)
        else:
            # Fall back to polygon (normalized coords)
            polygon = det.get("polygon", [])
            if polygon:
                pts = np.array(polygon) * [img_w, img_h]
                x1, y1 = pts.min(axis=0)
                x2, y2 = pts.max(axis=0)
                xyxy.append([x1, y1, x2, y2])
            else:
                continue

        scores.append(det.get("score", 0.0))
        class_ids.append(det.get("class_id", 0))

        # Convert polygon to mask if available
        polygon = det.get("polygon", [])
        if polygon and len(polygon) >= 3:
            pts = (np.array(polygon) * [img_w, img_h]).astype(np.int32)
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 1)
            masks_list.append(mask.astype(bool))
        else:
            masks_list.append(np.zeros((img_h, img_w), dtype=bool))

    if not xyxy:
        return sv.Detections.empty()

    result = sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        confidence=np.array(scores, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
    )
    if masks_list:
        result.mask = np.array(masks_list, dtype=bool)
    return result


def annotate_image(
    img: np.ndarray,
    detections: sv.Detections,
    class_names: dict[int, str],
) -> np.ndarray:
    """Annotate image with supervision bboxes, masks, and labels.

    Args:
        img: Input image as numpy array (BGR or RGB)
        detections: sv.Detections object
        class_names: Mapping from class_id to class name

    Returns:
        Annotated image as numpy array
    """
    labels = [
        f"{class_names.get(cid, str(cid))} {conf:.2f}"
        for cid, conf in zip(detections.class_id, detections.confidence)
    ]

    annotated = img.copy()

    # Draw masks first (if available)
    if detections.mask is not None and detections.mask.any():
        mask_annotator = sv.MaskAnnotator(opacity=0.3, color_lookup=sv.ColorLookup.CLASS)
        annotated = mask_annotator.annotate(scene=annotated, detections=detections)

    # Draw bboxes and labels
    box_annotator = sv.BoxAnnotator(thickness=3, color_lookup=sv.ColorLookup.CLASS)
    label_annotator = sv.LabelAnnotator(
        text_scale=0.6,
        text_thickness=2,
        text_padding=6,
        color_lookup=sv.ColorLookup.CLASS,
    )
    annotated = box_annotator.annotate(scene=annotated, detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

    return annotated


# Existing fixtures for backward compatibility

@pytest.fixture
def test_image_b64() -> str:
    """Generate a simple test image as base64."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = [120, 120, 120]  # Gray background
    buf = io.BytesIO()
    from PIL import Image
    Image.fromarray(img).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.fixture
def sample_yolo_labels() -> list[str]:
    """Simple YOLO labels for testing."""
    return ["0 0.5 0.5 0.3 0.2", "1 0.2 0.3 0.1 0.15"]


@pytest.fixture
def sample_classes() -> dict:
    """Class mapping for testing."""
    return {"0": "fire", "1": "smoke"}

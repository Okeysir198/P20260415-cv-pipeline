"""Shared fixtures and helpers for auto_label service tests.

All tests require the auto_label service running at localhost:18104
and SAM3 running at localhost:18100.
"""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
import pytest
import requests
import supervision as sv

SERVICE_URL = "http://localhost:18104"
DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def _service_available() -> bool:
    """Check if the auto_label service is reachable."""
    try:
        resp = requests.get(f"{SERVICE_URL}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


skip_no_service = pytest.mark.skipif(
    not _service_available(),
    reason="auto_label service not running at localhost:18104",
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
    """Convert auto_label API detections to sv.Detections."""
    if not detections:
        return sv.Detections.empty()

    xyxy = []
    scores = []
    class_ids = []
    masks_list = []

    for det in detections:
        bbox = det.get("bbox_xyxy", [])
        if len(bbox) == 4:
            xyxy.append(bbox)
        else:
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
    """Annotate image with supervision bboxes, masks, and labels."""
    labels = [
        f"{class_names.get(cid, str(cid))} {conf:.2f}"
        for cid, conf in zip(detections.class_id, detections.confidence)
    ]

    annotated = img.copy()

    if detections.mask is not None and detections.mask.any():
        mask_annotator = sv.MaskAnnotator(opacity=0.3, color_lookup=sv.ColorLookup.CLASS)
        annotated = mask_annotator.annotate(scene=annotated, detections=detections)

    box_annotator = sv.BoxAnnotator(thickness=3, color_lookup=sv.ColorLookup.CLASS)
    label_annotator = sv.LabelAnnotator(
        text_scale=0.6, text_thickness=2, text_padding=6, color_lookup=sv.ColorLookup.CLASS,
    )
    annotated = box_annotator.annotate(scene=annotated, detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

    return annotated


def write_vscode_video(
    frames: list[np.ndarray],
    output_path: str | Path,
    fps: float = 30.0,
) -> None:
    """Write frames as H.264 MP4 video that plays in VS Code.

    Uses PyAV to create a proper H.264 encoded MP4 file. VS Code's
    built-in video player only supports H.264 video codec.

    Args:
        frames: List of BGR numpy arrays (cv2 format)
        output_path: Path to output MP4 file
        fps: Frames per second for output video
    """
    import av
    from fractions import Fraction

    output_path = str(output_path)
    if not frames:
        raise ValueError("No frames to write")

    height, width = frames[0].shape[:2]

    # Create output container with H.264 codec
    container = av.open(output_path, mode="w")
    # PyAV requires Fraction or int for rate, not float
    rate = Fraction(int(fps), 1) if fps == int(fps) else Fraction(fps).limit_denominator(1000)
    stream = container.add_stream("libx264", rate=rate)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    # Set encoding options for compatibility
    stream.options = {
        "crf": "23",  # Quality (lower = better, 18-28 is typical range)
        "preset": "fast",  # Encoding speed
    }

    for frame_bgr in frames:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Create av.VideoFrame
        frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush remaining packets
    for packet in stream.encode():
        container.mux(packet)

    container.close()

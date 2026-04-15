"""Shared fixtures and helpers for SAM3.1 service tests.

All tests require the SAM3.1 service running at localhost:18106.

Test data (truck.jpg, cat.jpg, bedroom.mp4) must be copied from
../s18100_sam3_service/tests/data/ before running tests.
"""

from __future__ import annotations

import base64
import io
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import pytest
import requests
import supervision as sv
from PIL import Image

SERVICE_URL = "http://localhost:18106"
DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
REQUEST_TIMEOUT = 120


def _service_available() -> bool:
    """Check if the SAM3.1 service is reachable."""
    try:
        resp = requests.get(f"{SERVICE_URL}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


skip_no_service = pytest.mark.skipif(
    not _service_available(),
    reason="SAM3.1 service not running at localhost:18106",
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


def load_image_cv2(filename: str) -> np.ndarray:
    """Load an image from data/ as BGR numpy array (OpenCV format)."""
    path = DATA_DIR / filename
    if not path.exists():
        pytest.skip(f"Test image not found: {path}")
    return cv2.imread(str(path))


def load_video_frames_b64(filename: str, num_frames: int = 5) -> list[str]:
    """Extract frames from a video file and return as base64 strings."""
    path = DATA_DIR / filename
    if not path.exists():
        pytest.skip(f"Test video not found: {path}")
    container = av.open(str(path))
    stream = container.streams.video[0]
    total = stream.frames
    indices = set(int(i * (total - 1) / (num_frames - 1)) for i in range(num_frames))
    frames = []
    for i, frame in enumerate(container.decode(stream)):
        if i in indices:
            frames.append(frame.to_image())
        if len(frames) == num_frames:
            break
    container.close()
    result = []
    for img in frames:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return result


def load_video_frames(filename: str, num_frames: int = 5) -> list[np.ndarray]:
    """Extract evenly-spaced frames from a video file as BGR numpy arrays."""
    path = DATA_DIR / filename
    if not path.exists():
        pytest.skip(f"Test video not found: {path}")
    container = av.open(str(path))
    stream = container.streams.video[0]
    total = stream.frames
    indices = set(int(i * (total - 1) / (num_frames - 1)) for i in range(num_frames))
    frames = []
    for i, frame in enumerate(container.decode(stream)):
        if i in indices:
            frames.append(frame.to_ndarray(format="bgr24"))
        if len(frames) == num_frames:
            break
    container.close()
    return frames


def load_all_video_frames(
    filename: str, max_frames: int = 20,
) -> tuple[list[np.ndarray], list[str], sv.VideoInfo]:
    """Load frames from a video as (BGR arrays, base64 strings, video_info).

    Extracts up to max_frames evenly-spaced frames (default 20).
    """
    path = DATA_DIR / filename
    if not path.exists():
        pytest.skip(f"Test video not found: {path}")
    video_info = sv.VideoInfo.from_video_path(str(path))
    total = video_info.total_frames
    stride = max(1, total // max_frames)
    frames_bgr = []
    frames_b64 = []
    for i, frame in enumerate(sv.get_video_frames_generator(str(path))):
        if i % stride != 0:
            continue
        frames_bgr.append(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        frames_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        if len(frames_bgr) >= max_frames:
            break
    # Adjust video_info fps to match the subsampled output
    video_info = sv.VideoInfo(
        width=video_info.width, height=video_info.height,
        fps=max(1, video_info.fps // stride),
        total_frames=len(frames_bgr),
    )
    return frames_bgr, frames_b64, video_info


def base64_to_mask(b64: str) -> np.ndarray:
    """Decode a base64 grayscale PNG mask to a boolean numpy array."""
    raw = base64.b64decode(b64)
    return np.array(Image.open(io.BytesIO(raw)).convert("L")).astype(bool)


def detections_from_masks(
    detections_raw: list[dict],
) -> sv.Detections:
    """Convert SAM3.1 API detections (with base64 masks + bbox) to sv.Detections."""
    if not detections_raw:
        return sv.Detections.empty()

    xyxy = []
    scores = []
    masks_list = []

    for det in detections_raw:
        bbox = det["bbox"]
        xyxy.append([bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]])
        scores.append(det.get("score", 0.0))
        masks_list.append(base64_to_mask(det["mask"]))

    return sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        confidence=np.array(scores, dtype=np.float32),
        mask=np.array(masks_list, dtype=bool),
    )


def annotate_image(
    img: np.ndarray,
    detections: sv.Detections,
    labels: list[str] | None = None,
) -> np.ndarray:
    """Annotate image with supervision masks, bboxes, and labels."""
    annotated = img.copy()

    if detections.mask is not None and detections.mask.any():
        mask_annotator = sv.MaskAnnotator(opacity=0.4, color_lookup=sv.ColorLookup.INDEX)
        annotated = mask_annotator.annotate(scene=annotated, detections=detections)

    box_annotator = sv.BoxAnnotator(thickness=2, color_lookup=sv.ColorLookup.INDEX)
    annotated = box_annotator.annotate(scene=annotated, detections=detections)

    if labels:
        label_annotator = sv.LabelAnnotator(
            text_scale=0.5, text_thickness=1, text_padding=4,
            color_lookup=sv.ColorLookup.INDEX,
        )
        annotated = label_annotator.annotate(
            scene=annotated, detections=detections, labels=labels,
        )

    return annotated


def write_vscode_video(
    frames: list[np.ndarray],
    output_path: str | Path,
    fps: float = 30.0,
) -> None:
    """Write frames as H.264 MP4 video that plays in VS Code.

    Args:
        frames: List of BGR numpy arrays (cv2 format)
        output_path: Path to output MP4 file
        fps: Frames per second for output video
    """
    output_path = str(output_path)
    if not frames:
        raise ValueError("No frames to write")

    height, width = frames[0].shape[:2]

    container = av.open(output_path, mode="w")
    # PyAV requires Fraction or int for rate, not float
    rate = Fraction(int(fps), 1) if fps == int(fps) else Fraction(fps).limit_denominator(1000)
    stream = container.add_stream("libx264", rate=rate)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {
        "crf": "23",  # Quality (lower = better, 18-28 is typical range)
        "preset": "fast",  # Encoding speed
    }

    for frame_bgr in frames:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()

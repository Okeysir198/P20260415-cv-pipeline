"""Shared fixtures for Flux NIM service tests."""

import base64
import io
from pathlib import Path

import pytest
import requests
from PIL import Image

SERVICE_URL = "http://localhost:18101"
REQUEST_TIMEOUT = 120
DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def _service_available() -> bool:
    """Check if the Flux NIM service is reachable."""
    try:
        resp = requests.get(f"{SERVICE_URL}/v1/health/ready", timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


skip_no_service = pytest.mark.skipif(
    not _service_available(),
    reason="Flux NIM service not running at localhost:18101",
)


@pytest.fixture(scope="session", autouse=True)
def ensure_output_dir():
    """Create outputs directory once per session."""
    OUTPUT_DIR.mkdir(exist_ok=True)


def load_image_b64(filename: str) -> str:
    """Load image from data/ and return as base64 string."""
    path = DATA_DIR / filename
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def base64_to_image(b64: str) -> Image.Image:
    """Decode a base64 string to a PIL Image."""
    if "," in b64 and b64.split(",")[0].startswith("data:"):
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def image_to_base64(image: Image.Image) -> str:
    """Encode a PIL Image as base64 PNG string."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

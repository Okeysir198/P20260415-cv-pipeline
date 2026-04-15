"""Shared fixtures for Image Editor service tests."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest
import requests
from PIL import Image

SERVICE_URL = "http://localhost:18102"
FLUX_URL = "http://localhost:18101"
SAM3_URL = "http://localhost:18100"
REQUEST_TIMEOUT = 120
DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def _service_available() -> bool:
    """Check if the Image Editor service is reachable."""
    try:
        resp = requests.get(f"{SERVICE_URL}/health", timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


skip_no_service = pytest.mark.skipif(
    not _service_available(),
    reason="Image Editor service not running at localhost:18102",
)


@pytest.fixture(scope="session", autouse=True)
def ensure_output_dir():
    """Create outputs directory once per session."""
    OUTPUT_DIR.mkdir(exist_ok=True)


def load_image_b64(filename: str) -> str:
    """Load image from data/ and return as base64 string."""
    path = DATA_DIR / filename
    if not path.exists():
        pytest.skip(f"Test image not found: {path}")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_to_base64(image: Image.Image) -> str:
    """Encode a PIL Image as base64 PNG string."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_to_image(b64: str) -> Image.Image:
    """Decode a base64 string to a PIL Image."""
    if "," in b64 and b64.split(",")[0].startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def generate_source_image() -> Image.Image:
    """Generate a safe source image via Flux text2img for inpaint tests."""
    resp = requests.post(
        f"{FLUX_URL}/v1/infer",
        json={
            "prompt": "a cozy living room with a red sofa and bookshelf, photorealistic",
            "seed": 10,
            "steps": 4,
        },
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return base64_to_image(resp.json()["artifacts"][0]["base64"])

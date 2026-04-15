"""Image decoding utilities shared across services."""

from __future__ import annotations

import base64
import io

from PIL import Image


def strip_data_uri(b64: str) -> str:
    """Remove data URI prefix if present."""
    if "," in b64 and b64.split(",")[0].startswith("data:"):
        return b64.split(",", 1)[1]
    return b64


def decode_image(b64: str) -> Image.Image:
    """Decode base64 string to PIL Image (RGB)."""
    return Image.open(io.BytesIO(base64.b64decode(strip_data_uri(b64)))).convert("RGB")

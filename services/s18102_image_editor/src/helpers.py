"""Image encoding/decoding utilities."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from _shared.image_utils import strip_data_uri


def encode_image(image: Image.Image) -> str:
    """Encode PIL Image to base64 PNG string."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def decode_mask(b64: str) -> np.ndarray:
    """Decode base64 mask to numpy array (grayscale)."""
    return np.array(Image.open(io.BytesIO(base64.b64decode(strip_data_uri(b64)))).convert("L"))

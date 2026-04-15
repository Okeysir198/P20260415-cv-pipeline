"""Mask compositing logic for combining original + edited images."""

from __future__ import annotations

import numpy as np
from PIL import Image


def mask_composite(original: Image.Image, edited: Image.Image, mask: np.ndarray) -> Image.Image:
    """Composite edited image onto original using mask alpha blending.

    Args:
        original: Original PIL Image
        edited: Edited PIL Image from Flux NIM
        mask: Grayscale numpy mask (0-255)

    Returns:
        Composited PIL Image
    """
    # Resize edited to match original if needed
    if edited.size != original.size:
        edited = edited.resize(original.size, Image.Resampling.LANCZOS)

    # Resize mask to match original if needed
    if mask.shape[:2] != (original.size[1], original.size[0]):
        mask = np.array(
            Image.fromarray(mask, mode="L").resize(original.size, Image.Resampling.NEAREST)
        )

    # Alpha blend: mask=255 → fully edited, mask=0 → fully original
    alpha = (mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
    blended = (
        np.array(original, dtype=np.float32) * (1.0 - alpha)
        + np.array(edited, dtype=np.float32) * alpha
    )
    return Image.fromarray(blended.astype(np.uint8))

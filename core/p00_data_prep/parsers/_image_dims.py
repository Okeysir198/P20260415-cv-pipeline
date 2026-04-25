"""Shared helper for reading actual image dimensions during annotation parsing.

Centralised because annotation metadata (COCO JSON `images[].width`, VOC XML
`<size><width>`, HF Dataset `row["width"]`) is known to disagree with the
actual image file for a small but real fraction of public datasets
(confirmed: ~6% of CPPE-5 validation rows). Normalising bboxes by the
metadata field when it's wrong produces silently-out-of-range YOLO labels
that get dropped downstream — the canonical reference notebooks avoid this
entirely by reading image dimensions from the image itself at runtime.

This helper mirrors that approach: try to read actual dims via PIL (header
only — no pixel decode, so fast), fall back to metadata with a warning if
the image can't be opened.
"""
from __future__ import annotations

from pathlib import Path

from loguru import logger


def actual_image_dims(
    image_path: Path | None,
    fallback_w: int = 0,
    fallback_h: int = 0,
    warn_on_mismatch: bool = True,
) -> tuple[int, int]:
    """Return `(width, height)` from the image file on disk.

    Args:
        image_path: Path to the image. If None or missing, returns
            ``(fallback_w, fallback_h)``.
        fallback_w: Metadata width (used only if the image can't be read).
        fallback_h: Metadata height.
        warn_on_mismatch: If True and both actual and metadata dims are
            known and they disagree, emit a single warning line.

    Returns:
        ``(width, height)`` — preferring the actual image dimensions over
        metadata whenever available.
    """
    if image_path is None or not Path(image_path).exists():
        return fallback_w, fallback_h

    try:
        from PIL import Image  # local import — parsers don't need PIL unless called
    except Exception:
        return fallback_w, fallback_h

    try:
        with Image.open(str(image_path)) as img:
            actual_w, actual_h = int(img.size[0]), int(img.size[1])
    except Exception as e:
        logger.warning(
            "Could not read dims from image %s (%s) — using metadata (%d, %d)",
            image_path, e, fallback_w, fallback_h,
        )
        return fallback_w, fallback_h

    if (
        warn_on_mismatch
        and fallback_w > 0 and fallback_h > 0
        and (actual_w, actual_h) != (fallback_w, fallback_h)
    ):
        logger.warning(
            "Image dim mismatch for %s: metadata=(%d, %d) actual=(%d, %d) — using actual",
            image_path.name, fallback_w, fallback_h, actual_w, actual_h,
        )

    return actual_w, actual_h

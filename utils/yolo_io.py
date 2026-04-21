"""Shared YOLO I/O utilities used across tools."""

import base64
import io
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_yolo_label(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Parse a YOLO-format label file.

    Args:
        label_path: Path to the ``.txt`` label file.

    Returns:
        List of ``(class_id, cx, cy, w, h)`` tuples.  Returns an empty
        list when the file is missing or empty.
    """
    if not label_path.exists():
        return []
    text = label_path.read_text().strip()
    if not text:
        return []
    annotations: list[tuple[int, float, float, float, float]] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        annotations.append((
            int(parts[0]),
            float(parts[1]),
            float(parts[2]),
            float(parts[3]),
            float(parts[4]),
        ))
    return annotations


def image_to_label_path(image_path: Path) -> Path:
    """Derive the label path from an image path.

    Expects the standard YOLO layout where ``images/`` and ``labels/``
    are sibling directories.
    """
    return image_path.parent.parent / "labels" / (image_path.stem + ".txt")


def pil_to_b64(img: Image.Image) -> str:
    """Encode a PIL Image as a base64 PNG string.

    Args:
        img: PIL Image to encode.

    Returns:
        Base64-encoded PNG string.
    """
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def parse_classes(classes_str: str) -> dict[int, str]:
    """Parse CLI classes string like '0:person,1:car' into dict.

    Args:
        classes_str: Comma-separated "id:name" pairs.

    Returns:
        Mapping of class_id to class name.

    Raises:
        ValueError: If any item lacks the required "id:name" format.
    """
    class_names: dict[int, str] = {}
    for item in classes_str.split(","):
        item = item.strip()
        if ":" not in item:
            raise ValueError(f"Invalid class format '{item}' — expected 'id:name'")
        class_id, class_name = item.split(":", 1)
        class_names[int(class_id.strip())] = class_name.strip()
    return class_names

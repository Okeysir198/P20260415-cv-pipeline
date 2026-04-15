"""Test 09: Utils — yolo_io I/O helpers (base64 encode, label path derivation)."""

import base64
import io
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from _runner import run_all  # noqa: E402
from utils.yolo_io import (  # noqa: E402
    IMAGE_EXTENSIONS,
    image_to_label_path,
    parse_yolo_label,
    pil_to_b64,
)


def test_image_extensions_set_covers_common():
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"):
        assert ext in IMAGE_EXTENSIONS


def test_image_to_label_path_sibling_dirs():
    img = Path("/data/train/images/frame_001.jpg")
    label = image_to_label_path(img)
    assert label == Path("/data/train/labels/frame_001.txt")


def test_parse_yolo_label_missing_and_empty():
    with tempfile.TemporaryDirectory() as td:
        missing = Path(td) / "nope.txt"
        assert parse_yolo_label(missing) == []

        empty = Path(td) / "empty.txt"
        empty.write_text("")
        assert parse_yolo_label(empty) == []

        whitespace = Path(td) / "ws.txt"
        whitespace.write_text("   \n\n")
        assert parse_yolo_label(whitespace) == []


def test_parse_yolo_label_skips_malformed_and_parses_valid():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "lbl.txt"
        p.write_text(
            "0 0.5 0.5 0.25 0.25\n"
            "bad line\n"         # too few fields
            "1 0.1 0.2 0.3 0.4 0.9\n"  # extra field → still first 5 parsed
        )
        rows = parse_yolo_label(p)
        assert rows == [
            (0, 0.5, 0.5, 0.25, 0.25),
            (1, 0.1, 0.2, 0.3, 0.4),
        ]


def test_pil_to_b64_roundtrip():
    img = Image.new("RGB", (32, 16), color=(10, 20, 30))
    encoded = pil_to_b64(img)

    # Must be valid base64
    raw = base64.b64decode(encoded)
    decoded = Image.open(io.BytesIO(raw))
    assert decoded.format == "PNG"
    assert decoded.size == (32, 16)

    # Pixel round-trip (PNG is lossless)
    assert decoded.convert("RGB").getpixel((0, 0)) == (10, 20, 30)


if __name__ == "__main__":
    run_all([
        ("image_extensions", test_image_extensions_set_covers_common),
        ("image_to_label_path", test_image_to_label_path_sibling_dirs),
        ("parse_yolo_label_empty", test_parse_yolo_label_missing_and_empty),
        ("parse_yolo_label_malformed", test_parse_yolo_label_skips_malformed_and_parses_valid),
        ("pil_to_b64_roundtrip", test_pil_to_b64_roundtrip),
    ], title="Test utils09: yolo_io")

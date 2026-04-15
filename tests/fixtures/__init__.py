"""Shared test fixtures providing real images and labels.

Data lives in ``tests/fixtures/data/`` — a small subset of the test_fire_100
dataset (10 train + 5 val images with annotations) checked into the repo so
tests are self-contained and don't depend on external dataset paths.

Usage from any test file::

    from fixtures import real_image, real_image_bgr_640, real_image_with_targets, real_image_b64
"""

import base64
import io
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.config import load_config

_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
_FIXTURES_DIR = Path(__file__).resolve().parent
_DATA_DIR = _FIXTURES_DIR / "data"
_TRAIN_IMAGES_DIR = _DATA_DIR / "train" / "images"
_TRAIN_LABELS_DIR = _DATA_DIR / "train" / "labels"
_VAL_IMAGES_DIR = _DATA_DIR / "val" / "images"
_VAL_LABELS_DIR = _DATA_DIR / "val" / "labels"

_DATA_CONFIG_PATH = ROOT / "configs" / "_test" / "05_data.yaml"
_TRAIN_CONFIG_PATH = ROOT / "configs" / "_test" / "06_training.yaml"

_cache: dict = {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _list_images(directory: Path, n: int = 0) -> list[Path]:
    if not directory.exists():
        return []
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in _IMG_EXTENSIONS)
    return paths[:n] if n > 0 else paths


def _load_yolo_labels(label_path: Path) -> np.ndarray:
    if not label_path.exists():
        return np.zeros((0, 5), dtype=np.float32)
    try:
        data = np.loadtxt(label_path, dtype=np.float32, ndmin=2)
    except ValueError:
        return np.zeros((0, 5), dtype=np.float32)
    if data.size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, :5]


# ---------------------------------------------------------------------------
# Public API: image paths
# ---------------------------------------------------------------------------

def train_image_paths(n: int = 0) -> list[Path]:
    """All (or first *n*) training image paths from fixtures/data/."""
    return _list_images(_TRAIN_IMAGES_DIR, n)


def val_image_paths(n: int = 0) -> list[Path]:
    """All (or first *n*) validation image paths from fixtures/data/."""
    return _list_images(_VAL_IMAGES_DIR, n)


# ---------------------------------------------------------------------------
# Public API: loaded images
# ---------------------------------------------------------------------------

def real_image(idx: int = 0, split: str = "train") -> np.ndarray:
    """Load a real BGR uint8 image from fixtures/data/."""
    paths = train_image_paths() if split == "train" else val_image_paths()
    assert len(paths) > idx, f"Not enough {split} images (need idx={idx}, have {len(paths)})"
    img = cv2.imread(str(paths[idx]))
    assert img is not None, f"Failed to read {paths[idx]}"
    return img


def real_image_bgr_640(idx: int = 0, split: str = "train") -> np.ndarray:
    """Load a real image resized to 640x640 (standard model input size)."""
    return cv2.resize(real_image(idx, split), (640, 640))


def real_image_with_targets(idx: int = 0, split: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """Load a real image + its YOLO labels as (BGR image, (N,5) targets).

    Searches for an image WITH annotations starting from *idx*.
    """
    paths = train_image_paths() if split == "train" else val_image_paths()
    labels_dir = _TRAIN_LABELS_DIR if split == "train" else _VAL_LABELS_DIR
    assert len(paths) > 0, f"No {split} images in fixtures/data/"

    for i in range(len(paths)):
        actual_idx = (idx + i) % len(paths)
        img_path = paths[actual_idx]
        label_path = labels_dir / (img_path.stem + ".txt")
        targets = _load_yolo_labels(label_path)
        if len(targets) > 0:
            img = cv2.imread(str(img_path))
            assert img is not None, f"Failed to read {img_path}"
            return img, targets

    # Fallback: return requested idx even with empty targets
    img = cv2.imread(str(paths[idx]))
    assert img is not None, f"Failed to read {paths[idx]}"
    label_path = labels_dir / (paths[idx].stem + ".txt")
    return img, _load_yolo_labels(label_path)


def real_image_b64(idx: int = 0, split: str = "train", size: tuple[int, int] = (256, 256)) -> str:
    """Load a real image, resize, and return as base64 PNG string."""
    img_bgr = real_image(idx, split)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = PILImage.fromarray(img_rgb).resize(size)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Public API: configs
# ---------------------------------------------------------------------------

def data_config() -> dict:
    """Loaded test_fire_100 data config."""
    if "data_config" not in _cache:
        _cache["data_config"] = load_config(str(_DATA_CONFIG_PATH))
    return _cache["data_config"]


def train_config() -> dict:
    """Loaded test_fire_100 training config."""
    if "train_config" not in _cache:
        _cache["train_config"] = load_config(str(_TRAIN_CONFIG_PATH))
    return _cache["train_config"]


def class_names() -> dict[int, str]:
    """Class name mapping {0: "fire", 1: "smoke"}."""
    return {int(k): v for k, v in data_config()["names"].items()}


# ---------------------------------------------------------------------------
# Public API: pretrained model paths
# ---------------------------------------------------------------------------

_PRETRAINED_DIR = ROOT / "pretrained"

YOLOX_M_PRETRAINED = _PRETRAINED_DIR / "yolox_m.pth"

DFINE_S_PRETRAINED = "ustc-community/dfine_s_coco"
DFINE_N_PRETRAINED = "ustc-community/dfine_n_coco"
DFINE_M_PRETRAINED = "ustc-community/dfine_m_coco"
RTDETR_R18_PRETRAINED = "PekingU/rtdetr_v2_r18vd"
RTDETR_R50_PRETRAINED = "PekingU/rtdetr_v2_r50vd"


def has_yolox_pretrained() -> bool:
    """Check if YOLOX-M pretrained weights are available."""
    return YOLOX_M_PRETRAINED.exists()


TRAINED_CHECKPOINT = ROOT / "tests" / "outputs" / "07_training" / "best.pth"


def has_trained_checkpoint() -> bool:
    """Check if the test training checkpoint exists (from test_core13_training)."""
    return TRAINED_CHECKPOINT.exists()

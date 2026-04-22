"""Task-agnostic base dataset for the camera_edge pipeline."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from torch.utils.data import Dataset

# Standard image file extensions
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ImageNet normalization constants (RGB channel order)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def denormalize_tensor(
    tensor,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    to_bgr: bool = True,
) -> np.ndarray:
    """Invert the normalize-or-rescale step back to HWC uint8 for visualization.

    Auto-detects the input space:
    * If values look ImageNet-normalized (roughly in ``[-3, 3]``) → apply
      ``x * std + mean`` then ``* 255``.
    * If values look already-rescaled to ``[0, 1]`` (max ≤ ~1.5) → skip
      mean/std and just ``* 255``.
    * If values look like raw uint8-range ``[0, 255]`` (max > ~2.0) → cast
      directly; inputs arrived without normalization (e.g. YOLOX pipelines
      with ``augmentation.normalize: false``).

    This way the callback renders correctly across HF (ImageNet-norm),
    torchvision (rescale-only), and YOLOX (raw-pixel) training recipes with
    no per-arch branching at the call site.

    Accepts torch.Tensor or numpy array, ``(3, H, W)`` or ``(B, 3, H, W)``.
    Returns HWC uint8 BGR (default) or RGB.
    """
    try:
        import torch  # deferred
    except Exception:  # pragma: no cover — torch is a hard dep in this project
        torch = None

    if torch is not None and hasattr(tensor, "detach") and hasattr(tensor, "cpu"):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor)

    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(
            f"denormalize_tensor expects (3, H, W) CHW input, got shape {arr.shape}"
        )

    arr = arr.astype(np.float32)
    max_abs = float(np.max(np.abs(arr))) if arr.size else 0.0
    if max_abs > 3.0:
        # Raw pixel space [0, 255] — no mean/std applied; cast directly.
        out = np.clip(arr, 0, 255).astype(np.uint8)
    elif max_abs <= 1.5 and float(arr.min()) >= -0.05:
        # Rescale-only space [0, 1] — the ImageNet mean/std step was skipped.
        out = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    else:
        # ImageNet-normalized — apply the inverse.
        mean_arr = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        std_arr = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
        out = arr * std_arr + mean_arr
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)

    out = np.ascontiguousarray(out.transpose(1, 2, 0))
    if to_bgr:
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return out


class BaseDataset(Dataset, ABC):
    """Base dataset that handles image discovery and loading.

    Subclasses implement label loading and target formatting for
    specific CV tasks (detection, classification, segmentation, pose).
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        input_size: tuple[int, int] = (640, 640),
        transform=None,
    ):
        self.split = split
        self.input_size = input_size
        self.transform = transform
        self.image_dir = data_dir / split / "images"
        self.label_dir = data_dir / split / "labels"

        # Discover images
        self.image_paths = self._discover_images()

    def _discover_images(self) -> list:
        """Find all images in the image directory."""
        if not self.image_dir.exists():
            return []
        paths = sorted(
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in IMG_EXTENSIONS
        )
        return paths

    def _label_path_for(self, image_path: Path) -> Path:
        """Get the label file path corresponding to an image."""
        return self.label_dir / f"{image_path.stem}.txt"

    def _attach_dataset_to_transforms(self) -> None:
        """Give Mosaic and MixUp transforms a reference to this dataset."""
        transforms = getattr(self, "transforms", None)
        if transforms is None:
            return
        for t in getattr(transforms, "transforms", []):
            if hasattr(t, "set_dataset"):
                t.set_dataset(self)

    @abstractmethod
    def load_target(self, label_path: Path) -> Any:
        """Load and validate a single label file.

        Returns raw target data in task-specific format.
        """

    @abstractmethod
    def format_target(
        self, raw_target: Any, image_size: tuple[int, int]
    ) -> Any:
        """Convert raw label data to model-ready target format.

        Args:
            raw_target: Output from load_target().
            image_size: (height, width) of the loaded image.

        Returns:
            Target in the format expected by the model/loss.
        """

    def load_image(self, image_path: Path) -> np.ndarray:
        """Load an image as BGR numpy array."""
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        return img

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        img = self.load_image(image_path)
        h, w = img.shape[:2]

        label_path = self._label_path_for(image_path)
        raw_target = self.load_target(label_path)
        target = self.format_target(raw_target, (h, w))

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

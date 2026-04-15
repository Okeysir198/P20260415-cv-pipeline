"""Task-agnostic base dataset for the camera_edge pipeline."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset

# Standard image file extensions
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ImageNet normalization constants (RGB channel order)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class BaseDataset(Dataset, ABC):
    """Base dataset that handles image discovery and loading.

    Subclasses implement label loading and target formatting for
    specific CV tasks (detection, classification, segmentation, pose).
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        input_size: Tuple[int, int] = (640, 640),
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
        self, raw_target: Any, image_size: Tuple[int, int]
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

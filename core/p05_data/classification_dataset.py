"""Image classification dataset and dataloader utilities.

Supports folder-based (ImageFolder) and label-file-based (YOLO-style) layouts.
"""

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from loguru import logger

from core.p05_data.base_dataset import IMAGENET_MEAN, IMAGENET_STD, IMG_EXTENSIONS
from utils.config import resolve_path


class ClassificationDataset(Dataset):
    """Dataset for image classification tasks.

    Supports two layouts:
    - Folder-based: class_name/image.jpg (set layout: "folder" in data config)
    - Label-file: images/ + labels/ with one class_id per txt (set layout: "yolo")

    Auto-detects layout when ``layout`` is ``"auto"`` (default).

    Args:
        data_config: Data config dict with path, names, num_classes, input_size.
        split: ``"train"``, ``"val"``, or ``"test"``.
        transforms: Transform callable. If None, images are resized and
            normalised with default ImageNet stats.
        base_dir: Base directory for resolving relative paths.
    """

    def __init__(
        self,
        data_config: dict,
        split: str = "train",
        transforms: Any | None = None,
        base_dir: str | Path | None = None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

        if base_dir is None:
            base_dir = Path.cwd()
        self.base_dir = Path(base_dir)

        dataset_root = resolve_path(data_config["path"], self.base_dir)
        split_subdir = data_config.get(split)
        if split_subdir is None:
            raise ValueError(f"data config missing '{split}' key")
        self.split_dir = dataset_root / split_subdir

        self.transforms = transforms
        self.data_config = data_config
        self.num_classes = data_config["num_classes"]
        self.class_names = data_config["names"]
        self.input_size = tuple(data_config["input_size"])

        self.layout = data_config.get("layout", "auto")
        if self.layout == "auto":
            self.layout = self._detect_layout()

        if self.layout == "folder":
            self.samples = self._load_folder_samples()
        else:
            self.samples = self._load_label_file_samples()

        if len(self.samples) == 0:
            raise FileNotFoundError(f"No samples found in {self.split_dir}")

        logger.info(
            "ClassificationDataset: %d samples, %d classes, layout=%s, split=%s",
            len(self.samples),
            self.num_classes,
            self.layout,
            split,
        )

    def _detect_layout(self) -> str:
        """Auto-detect whether layout is folder-based or label-file-based."""
        images_dir = self.split_dir / "images"
        if images_dir.exists():
            return "yolo"
        if self.split_dir.exists() and any(d.is_dir() for d in self.split_dir.iterdir()):
            return "folder"
        return "yolo"

    def _load_folder_samples(self) -> list[tuple[Path, int]]:
        """Load samples from folder-based layout (class_name/image.jpg)."""
        samples: list[tuple[Path, int]] = []
        name_to_idx: dict[str, int] = {}
        if isinstance(self.class_names, dict):
            # {0: "fire", 1: "smoke"} -> {"fire": 0, "smoke": 1}
            name_to_idx = {v: int(k) for k, v in self.class_names.items()}

        for subdir in sorted(self.split_dir.iterdir()):
            if not subdir.is_dir():
                continue
            class_name = subdir.name
            if class_name in name_to_idx:
                class_idx = name_to_idx[class_name]
            else:
                # Try parsing as integer
                try:
                    class_idx = int(class_name)
                except ValueError:
                    logger.warning("Skipping unknown class directory: %s", class_name)
                    continue

            if class_idx < 0 or class_idx >= self.num_classes:
                logger.warning(
                    "Class index %d out of range [0, %d)", class_idx, self.num_classes
                )
                continue

            for img_path in sorted(subdir.iterdir()):
                if img_path.suffix.lower() in IMG_EXTENSIONS:
                    samples.append((img_path, class_idx))

        return samples

    def _load_label_file_samples(self) -> list[tuple[Path, int]]:
        """Load samples from label-file layout (images/ + labels/)."""
        images_subdir = self.split_dir / "images"
        img_dir = images_subdir if images_subdir.exists() else self.split_dir

        labels_subdir = self.split_dir / "labels"
        label_dir = labels_subdir if labels_subdir.exists() else self.split_dir.parent / "labels"

        samples: list[tuple[Path, int]] = []
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTENSIONS:
                continue
            label_path = label_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            try:
                class_idx = int(label_path.read_text().strip().split()[0])
            except (ValueError, IndexError):
                logger.warning("Invalid label file: %s", label_path)
                continue
            if 0 <= class_idx < self.num_classes:
                samples.append((img_path, class_idx))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------ #
    # Uniform dataset surface used by viz callbacks + error analysis.    #
    # Mirrors YOLOXDataset / KeypointDataset so every task-aware render  #
    # site can speak the same API.                                       #
    # ------------------------------------------------------------------ #

    @property
    def img_paths(self) -> list[Path]:
        """List of image paths in dataset order — same surface as detection."""
        return [p for p, _ in self.samples]

    def get_raw_item(self, idx: int) -> dict:
        """Return an un-transformed BGR uint8 HWC image + class-id target.

        Used by viz callbacks to render raw / augmented / best-checkpoint
        grids without re-applying the training transform chain.
        """
        img_path, class_idx = self.samples[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            image = np.full(
                (self.input_size[0], self.input_size[1], 3), 114, dtype=np.uint8
            )
        return {"image": image, "targets": int(class_idx), "path": str(img_path)}

    def _load_label(self, img_path) -> int:
        """Class-id lookup by path (detection/keypoint parity helper)."""
        img_path = Path(img_path)
        for p, cls in self.samples:
            if p == img_path:
                return int(cls)
        return -1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Get a sample: (image_tensor, class_label, image_path).

        Returns:
            image: float32 tensor (C, H, W).
            label: scalar long tensor with class index.
            path: image file path string.
        """
        img_path, class_idx = self.samples[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            image = np.full(
                (self.input_size[0], self.input_size[1], 3), 114, dtype=np.uint8
            )

        label = torch.tensor(class_idx, dtype=torch.long)

        if self.transforms is not None:
            image = self.transforms(image)
            if isinstance(image, torch.Tensor):
                return image, label, str(img_path)

        # Fallback: manual conversion
        if isinstance(image, np.ndarray):
            h, w = self.input_size
            image = cv2.resize(image, (w, h))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))

        return image, label, str(img_path)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


class _ClassificationTransform:
    """Wraps torchvision transforms to handle BGR numpy input from cv2."""

    def __init__(self, transforms_list: list) -> None:
        self.transform = T.Compose(transforms_list)

    def __call__(self, image_bgr_np: np.ndarray) -> torch.Tensor:
        image_rgb = cv2.cvtColor(image_bgr_np, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        return self.transform(pil_img)


def build_classification_transforms(
    is_train: bool,
    input_size: tuple[int, int] = (224, 224),
    mean: list | None = None,
    std: list | None = None,
) -> _ClassificationTransform:
    """Build standard classification transforms.

    Train: RandomResizedCrop, RandomHorizontalFlip, ColorJitter, Normalize.
    Val: Resize (slightly larger), CenterCrop, Normalize.

    Args:
        is_train: Whether to use training augmentation.
        input_size: Target (height, width).
        mean: Normalisation mean (default ImageNet).
        std: Normalisation std (default ImageNet).

    Returns:
        A callable that accepts a BGR numpy image and returns a float32 tensor.
    """
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD

    h, w = input_size

    if is_train:
        transforms_list = [
            T.RandomResizedCrop((h, w), scale=(0.08, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    else:
        resize_h = int(h * 256 / 224)
        resize_w = int(w * 256 / 224)
        transforms_list = [
            T.Resize((resize_h, resize_w)),
            T.CenterCrop((h, w)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]

    return _ClassificationTransform(transforms_list)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def classification_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, str]],
) -> dict:
    """Collate for classification: stack images and labels.

    Returns:
        Dict with ``"images"`` (B, C, H, W), ``"targets"`` list of scalar
        tensors, ``"paths"`` list of strings.
    """
    images, labels, paths = zip(*batch, strict=True)
    images = torch.stack(images, dim=0)
    targets = list(labels)
    return {"images": images, "targets": targets, "paths": list(paths)}


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------


def build_classification_dataloader(
    data_config: dict,
    split: str,
    training_config: dict,
    base_dir: str | Path | None = None,
) -> DataLoader:
    """Build a classification DataLoader.

    Args:
        data_config: Data config dict.
        split: ``"train"``, ``"val"``, or ``"test"``.
        training_config: Training config dict (uses ``data.batch_size``, etc.).
        base_dir: Base directory for resolving paths.

    Returns:
        DataLoader for classification.
    """
    is_train = split == "train"
    input_size = tuple(data_config["input_size"])
    mean = data_config.get("mean", IMAGENET_MEAN)
    std = data_config.get("std", IMAGENET_STD)

    transforms = build_classification_transforms(
        is_train=is_train,
        input_size=input_size,
        mean=mean,
        std=std,
    )

    dataset = ClassificationDataset(
        data_config=data_config,
        split=split,
        transforms=transforms,
        base_dir=base_dir,
    )

    data_section = training_config.get("data", {})
    batch_size = data_section.get("batch_size", 32)
    num_workers = data_section.get("num_workers", 4)
    pin_memory = data_section.get("pin_memory", True)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=classification_collate_fn,
        drop_last=is_train,
    )

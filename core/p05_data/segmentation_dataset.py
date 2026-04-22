"""Semantic segmentation dataset and dataloader utilities.

Reads image files and corresponding mask PNGs where pixel values encode
class IDs (0-indexed). Uses torchvision v2 transforms with tv_tensors
for joint image+mask augmentation.
"""

import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as v2
import torchvision.tv_tensors as tv_tensors
from torch.utils.data import DataLoader, Dataset

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from core.p05_data.base_dataset import IMAGENET_MEAN, IMAGENET_STD, IMG_EXTENSIONS
from utils.config import resolve_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SegmentationDataset(Dataset):
    """Dataset for semantic segmentation with mask PNGs.

    Reads images and corresponding grayscale mask PNGs where each pixel
    value is the class ID (0-indexed).

    Directory layout expected::

        <split_root>/
            images/
                img001.jpg
                ...
            masks/
                img001.png   (pixel values = class_id, 0-indexed)
                ...

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

        self.img_dir = self.split_dir / "images"
        self.mask_dir = self.split_dir / "masks"

        # Collect image paths
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        self.img_paths: list[Path] = sorted(
            p for p in self.img_dir.iterdir()
            if p.suffix.lower() in IMG_EXTENSIONS
        )
        if len(self.img_paths) == 0:
            raise FileNotFoundError(
                f"No images found in {self.img_dir} "
                f"(extensions: {IMG_EXTENSIONS})"
            )

        self.transforms = transforms
        self.data_config = data_config
        self.num_classes = data_config["num_classes"]
        self.class_names = data_config["names"]
        self.input_size = tuple(data_config["input_size"])  # (h, w)

        logger.info(
            "SegmentationDataset: %d images, %d classes, split=%s",
            len(self.img_paths),
            self.num_classes,
            split,
        )

    def __len__(self) -> int:
        return len(self.img_paths)

    # ------------------------------------------------------------------ #
    # Uniform dataset surface used by viz callbacks + error analysis.    #
    # Mirrors YOLOXDataset / KeypointDataset / ClassificationDataset.    #
    # ------------------------------------------------------------------ #

    def get_raw_item(self, idx: int) -> dict:
        """Return un-transformed BGR uint8 HWC image + mask target."""
        img_path = self.img_paths[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            image = np.full(
                (self.input_size[0], self.input_size[1], 3), 114, dtype=np.uint8
            )
        mask_path = self.mask_dir / f"{img_path.stem}.png"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        return {"image": image, "targets": mask, "path": str(img_path)}

    def _load_label(self, img_path):
        """Mask array lookup by image path (used by error analysis)."""
        img_path = Path(img_path)
        mask_path = self.mask_dir / f"{img_path.stem}.png"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                return mask
        return np.zeros((1, 1), dtype=np.uint8)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Get a single sample.

        Returns:
            ``(image_tensor, mask_tensor, img_path_str)``

            - ``image_tensor``: float32 tensor (C, H, W).
            - ``mask_tensor``: long tensor (H, W) with class IDs.
            - ``img_path_str``: Absolute path string of the source image.
        """
        img_path = self.img_paths[idx]
        img_path_str = str(img_path)

        # Load image (BGR)
        image = cv2.imread(img_path_str)
        if image is None:
            image = np.full(
                (self.input_size[0], self.input_size[1], 3), 114, dtype=np.uint8
            )

        # Load mask (grayscale, pixel values = class IDs)
        mask_path = self.mask_dir / f"{img_path.stem}.png"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros(
                    (image.shape[0], image.shape[1]), dtype=np.uint8
                )
        else:
            mask = np.zeros(
                (image.shape[0], image.shape[1]), dtype=np.uint8
            )

        if self.transforms is not None:
            image_tensor, mask_tensor = self.transforms(image, mask)
            return image_tensor, mask_tensor, img_path_str

        # Fallback: manual conversion
        h, w = self.input_size
        image = cv2.resize(image, (w, h))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy(mask.astype(np.int64))

        return image_tensor, mask_tensor, img_path_str


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


class _SegmentationTransform:
    """Wraps torchvision v2 transforms for joint image+mask augmentation.

    Handles BGR numpy input from cv2, wraps image and mask as tv_tensors,
    applies v2 transforms jointly, and extracts the output tensors.
    """

    def __init__(self, transform: v2.Compose) -> None:
        self.transform = transform

    def __call__(
        self, image_bgr_np: np.ndarray, mask_np: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply transforms to image and mask jointly.

        Args:
            image_bgr_np: BGR uint8 numpy array (H, W, 3).
            mask_np: Grayscale uint8 numpy array (H, W) with class IDs.

        Returns:
            ``(image_tensor, mask_tensor)`` where image is float32 (C, H, W)
            and mask is long (H, W).
        """
        # BGR -> RGB, HWC -> CHW for tv_tensors.Image
        image_rgb = cv2.cvtColor(image_bgr_np, cv2.COLOR_BGR2RGB)
        image_chw = np.ascontiguousarray(image_rgb.transpose(2, 0, 1))
        image_tv = tv_tensors.Image(torch.from_numpy(image_chw))

        # Mask as tv_tensors.Mask (H, W)
        mask_tv = tv_tensors.Mask(torch.from_numpy(mask_np.astype(np.int64)))

        # v2 transforms dispatch on tv_tensor types, applying spatial
        # transforms jointly to both Image and Mask
        sample = {"image": image_tv, "mask": mask_tv}
        result = self.transform(sample)

        image_out = result["image"]
        mask_out = result["mask"]

        # Ensure correct dtypes
        if not image_out.is_floating_point():
            image_out = image_out.float() / 255.0
        mask_out = mask_out.long()

        return image_out, mask_out


def build_segmentation_transforms(
    is_train: bool,
    input_size: tuple[int, int] = (512, 512),
    mean: list | None = None,
    std: list | None = None,
) -> _SegmentationTransform:
    """Build segmentation transforms with joint image+mask augmentation.

    Train: RandomResizedCrop, RandomHorizontalFlip, ColorJitter, ToDtype, Normalize.
    Val: Resize, ToDtype, Normalize.

    Mask always uses NEAREST interpolation to preserve class IDs.

    Args:
        is_train: Whether to use training augmentation.
        input_size: Target (height, width).
        mean: Normalisation mean (default ImageNet).
        std: Normalisation std (default ImageNet).

    Returns:
        A callable that accepts BGR numpy image and grayscale numpy mask,
        returning ``(image_tensor, mask_tensor)``.
    """
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD

    h, w = input_size

    if is_train:
        transforms_list = [
            v2.RandomResizedCrop(
                size=(h, w),
                scale=(0.5, 2.0),
                interpolation=v2.InterpolationMode.BILINEAR,
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            v2.ToDtype({tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64}, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    else:
        transforms_list = [
            v2.Resize(
                size=(h, w),
                interpolation=v2.InterpolationMode.BILINEAR,
            ),
            v2.ToDtype({tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64}, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]

    return _SegmentationTransform(v2.Compose(transforms_list))


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def segmentation_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, str]],
) -> dict:
    """Collate for segmentation: stack images, list masks.

    Returns:
        Dict with ``"images"`` (B, C, H, W), ``"targets"`` list of
        (H, W) long tensors, ``"paths"`` list of strings.
    """
    images, masks, paths = zip(*batch, strict=True)
    images = torch.stack(images, dim=0)
    return {"images": images, "targets": list(masks), "paths": list(paths)}


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------


def build_segmentation_dataloader(
    data_config: dict,
    split: str,
    training_config: dict,
    base_dir: str | Path | None = None,
) -> DataLoader:
    """Build a segmentation DataLoader.

    Args:
        data_config: Data config dict.
        split: ``"train"``, ``"val"``, or ``"test"``.
        training_config: Training config dict (uses ``data.batch_size``, etc.).
        base_dir: Base directory for resolving paths.

    Returns:
        DataLoader for semantic segmentation.
    """
    is_train = split == "train"
    input_size = tuple(data_config["input_size"])
    mean = data_config.get("mean", IMAGENET_MEAN)
    std = data_config.get("std", IMAGENET_STD)

    transforms = build_segmentation_transforms(
        is_train=is_train,
        input_size=input_size,
        mean=mean,
        std=std,
    )

    dataset = SegmentationDataset(
        data_config=data_config,
        split=split,
        transforms=transforms,
        base_dir=base_dir,
    )

    data_section = training_config.get("data", {})
    batch_size = data_section.get("batch_size", 8)
    num_workers = data_section.get("num_workers", 4)
    pin_memory = data_section.get("pin_memory", True)

    # Use forkserver to avoid deadlocks with torchvision v2 transforms (nn.Module)
    # when the main process has already run v2 transforms before forking workers.
    mp_context = "forkserver" if num_workers > 0 else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=segmentation_collate_fn,
        drop_last=is_train,
        multiprocessing_context=mp_context,
    )

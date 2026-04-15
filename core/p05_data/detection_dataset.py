"""YOLO-format dataset and dataloader utilities for object detection.

Reads image files and corresponding YOLO-format label ``.txt`` files
(one row per object: ``class_id cx cy w h``, normalised 0-1).
"""

import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import logging

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from core.p05_data.base_dataset import BaseDataset, IMG_EXTENSIONS, IMAGENET_MEAN, IMAGENET_STD  # noqa: F401
from core.p05_data.transforms import build_transforms
from utils.config import resolve_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class YOLOXDataset(BaseDataset):
    """Dataset for YOLO-format object detection data.

    Reads images and YOLO-format label files, applies transforms.

    Directory layout expected::

        <split_root>/
            images/   # (or the split path itself)
                img001.jpg
                ...
            labels/   # sibling of images dir
                img001.txt
                ...

    Each label file has one row per object::

        class_id cx cy w h

    All values normalised 0-1.

    Args:
        data_config: Data config dict (loaded from ``configs/<usecase>/05_data.yaml``).
            Must contain ``path``, ``train``/``val``/``test``, ``names``,
            ``num_classes``, ``input_size``, ``mean``, ``std``.
        split: ``"train"``, ``"val"``, or ``"test"``.
        transforms: A :class:`Compose` transform pipeline. If ``None``,
            images are returned raw (numpy HWC uint8) with numpy targets.
        base_dir: Base directory for resolving relative paths in the config.
            Defaults to the current working directory.
    """

    def __init__(
        self,
        data_config: dict,
        split: str = "train",
        transforms: Optional[Any] = None,
        base_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

        if base_dir is None:
            base_dir = Path.cwd()
        self.base_dir = Path(base_dir)

        # Resolve image directory
        dataset_root = resolve_path(data_config["path"], self.base_dir)
        split_subdir = data_config.get(split)
        if split_subdir is None:
            raise ValueError(f"data config missing '{split}' key")
        self.img_dir = dataset_root / split_subdir
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        # Derive label directory: sibling "labels" directory
        self.label_dir = self.img_dir.parent / "labels"

        # Collect image paths
        self.img_paths: List[Path] = sorted(
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

        # Wire up Mosaic/MixUp dataset references
        self._attach_dataset_to_transforms()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_label(self, img_path: Path) -> np.ndarray:
        """Load a YOLO label file corresponding to an image path.

        Returns:
            numpy array (N, 5) float32 with columns
            [class_id, cx, cy, w, h]. Returns empty (0, 5) if the label
            file is missing or empty.
        """
        label_path = self.label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            return np.zeros((0, 5), dtype=np.float32)

        try:
            data = np.loadtxt(label_path, dtype=np.float32, ndmin=2)
        except ValueError:
            return np.zeros((0, 5), dtype=np.float32)

        if data.size == 0:
            return np.zeros((0, 5), dtype=np.float32)

        # Expect 5 columns: class_id cx cy w h
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] < 5:
            return np.zeros((0, 5), dtype=np.float32)

        data = data[:, :5]

        # Validate labels
        valid_mask = np.ones(len(data), dtype=bool)

        # Class IDs must be in [0, num_classes)
        class_ids = data[:, 0].astype(int)
        bad_cls = (class_ids < 0) | (class_ids >= self.num_classes)
        if bad_cls.any():
            logger.warning(
                "%s: %d labels with invalid class IDs (not in [0, %d))",
                label_path.name, bad_cls.sum(), self.num_classes,
            )
            valid_mask &= ~bad_cls

        # Coordinates must be in [0, 1] and have positive w/h
        coords = data[:, 1:5]
        bad_range = (coords < 0).any(axis=1) | (coords > 1).any(axis=1)
        bad_size = (data[:, 3] <= 0) | (data[:, 4] <= 0)
        bad_coords = bad_range | bad_size
        if bad_coords.any():
            logger.warning(
                "%s: %d labels with invalid coordinates or zero-size boxes",
                label_path.name, bad_coords.sum(),
            )
            valid_mask &= ~bad_coords

        return data[valid_mask]

    # ------------------------------------------------------------------
    # BaseDataset abstract method implementations
    # ------------------------------------------------------------------

    def load_target(self, label_path: Path) -> np.ndarray:
        """Load and validate a YOLO-format label file.

        Delegates to :meth:`_load_label` using the image path derived
        from the label path so that existing validation logic is reused.

        Returns:
            numpy array (N, 5) float32 — [class_id, cx, cy, w, h].
        """
        # _load_label expects an image path and derives the label path itself,
        # so we reconstruct a compatible image path from the label path.
        img_stem = label_path.stem
        # Find the matching image path (fallback: create a synthetic path)
        for img_path in self.img_paths:
            if img_path.stem == img_stem:
                return self._load_label(img_path)
        # If no matching image found, load the label file directly
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
        return data[:, :5] if data.shape[1] >= 5 else np.zeros((0, 5), dtype=np.float32)

    def format_target(
        self, raw_target: np.ndarray, image_size: Tuple[int, int]
    ) -> np.ndarray:
        """Return raw YOLO targets as-is (already normalised 0-1).

        Args:
            raw_target: (N, 5) array from :meth:`load_target`.
            image_size: (height, width) — unused for YOLO normalised format.

        Returns:
            (N, 5) float32 array [class_id, cx, cy, w, h].
        """
        return raw_target

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.img_paths)

    def get_raw_item(self, idx: int) -> dict:
        """Return raw image and targets without any transforms.

        Used by :class:`Mosaic`, :class:`MixUp`, and :class:`CopyPaste`
        to sample additional images.

        Args:
            idx: Index into the dataset.

        Returns:
            Dict with ``"image"`` (numpy HWC uint8 BGR) and ``"targets"``
            (numpy (N, 5) float32).
        """
        img_path = self.img_paths[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            # Return a grey placeholder
            image = np.full(
                (self.input_size[0], self.input_size[1], 3), 114, dtype=np.uint8
            )
        targets = self._load_label(img_path)
        return {"image": image, "targets": targets}

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Get a single sample.

        Args:
            idx: Index into the dataset.

        Returns:
            ``(image_tensor, targets_tensor, img_path_str)``

            - ``image_tensor``: float32 tensor (C, H, W).
            - ``targets_tensor``: float32 tensor (N, 5) — may be (0, 5).
            - ``img_path_str``: Absolute path string of the source image.
        """
        raw = self.get_raw_item(idx)
        image, targets = raw["image"], raw["targets"]
        img_path_str = str(self.img_paths[idx])

        if self.transforms is not None:
            image, targets = self.transforms(image, targets)
            # transforms.ToTensor returns torch tensors
            if isinstance(image, torch.Tensor):
                return image, targets, img_path_str

        # Fallback: convert manually if no transforms
        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(
                np.ascontiguousarray(image.transpose(2, 0, 1))
            )
        if isinstance(targets, np.ndarray):
            targets = (
                torch.from_numpy(targets.astype(np.float32))
                if len(targets)
                else torch.zeros((0, 5), dtype=torch.float32)
            )
        return image, targets, img_path_str


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, str]],
) -> dict:
    """Custom collate for variable-length targets.

    Args:
        batch: List of ``(image, targets, path)`` tuples.

    Returns:
        Dictionary with keys:

        - ``images``: float32 tensor (B, C, H, W).
        - ``targets``: list of B tensors, each (N_i, 5).
        - ``paths``: list of B image path strings.
    """
    images, targets_list, paths = zip(*batch)
    images = torch.stack(images, dim=0)
    return {"images": images, "targets": list(targets_list), "paths": list(paths)}


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def build_dataloader(
    data_config: dict,
    split: str,
    training_config: dict,
    base_dir: Optional[Union[str, Path]] = None,
) -> DataLoader:
    """Build a DataLoader from data and training config dicts.

    Args:
        data_config: Loaded data YAML config dict.
        split: ``"train"``, ``"val"``, or ``"test"``.
        training_config: Loaded training YAML config dict (needs
            ``augmentation``, ``data.batch_size``, ``data.num_workers``,
            ``data.pin_memory``).
        base_dir: Base directory for resolving relative paths.

    Returns:
        A :class:`torch.utils.data.DataLoader` ready to iterate.
    """
    is_train = split == "train"
    input_size = tuple(data_config["input_size"])
    mean = data_config.get("mean", IMAGENET_MEAN)
    std = data_config.get("std", IMAGENET_STD)

    aug_config = training_config.get("augmentation", {})
    transforms = build_transforms(
        config=aug_config,
        is_train=is_train,
        input_size=input_size,
        mean=mean,
        std=std,
    )

    dataset = YOLOXDataset(
        data_config=data_config,
        split=split,
        transforms=transforms,
        base_dir=base_dir,
    )

    data_section = training_config.get("data", {})
    batch_size = data_section.get("batch_size", 16)
    num_workers = data_section.get("num_workers", 4)
    pin_memory = data_section.get("pin_memory", True)
    sampler_mode = data_section.get("sampler", "none")

    # Use forkserver to avoid deadlocks with torchvision v2 transforms (nn.Module)
    # when the main process has already run v2 transforms before forking workers.
    mp_context = "forkserver" if num_workers > 0 else None

    # Build optional weighted sampler for imbalanced datasets
    sampler = None
    shuffle = is_train
    if is_train and sampler_mode in ("balanced", "sqrt"):
        sampler = _build_weighted_sampler(dataset, mode=sampler_mode)
        shuffle = False  # sampler and shuffle are mutually exclusive

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=is_train,
        multiprocessing_context=mp_context,
    )

    return loader


def _build_weighted_sampler(
    dataset: YOLOXDataset,
    mode: str = "balanced",
) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler for class-imbalanced datasets.

    Args:
        dataset: The detection dataset to sample from.
        mode: ``"balanced"`` for inverse-frequency weighting,
            ``"sqrt"`` for square-root inverse-frequency (softer).

    Returns:
        A :class:`WeightedRandomSampler` for the dataset.
    """
    # Count dominant class per image
    class_counts = np.zeros(dataset.num_classes, dtype=np.float64)
    image_classes: List[int] = []

    for img_path in dataset.img_paths:
        labels = dataset._load_label(img_path)
        if len(labels) > 0:
            cls_ids = labels[:, 0].astype(int)
            # Assign image to most frequent class
            dominant = int(np.bincount(cls_ids, minlength=dataset.num_classes).argmax())
            image_classes.append(dominant)
            for c in cls_ids:
                class_counts[c] += 1
        else:
            # Background image — assign to a pseudo-class
            image_classes.append(-1)

    # Compute per-class weight
    class_counts = np.maximum(class_counts, 1.0)
    if mode == "sqrt":
        class_weights = 1.0 / np.sqrt(class_counts)
    else:
        class_weights = 1.0 / class_counts
    class_weights /= class_weights.sum()

    # Background images get minimum weight
    bg_weight = float(class_weights.min())

    # Assign weight per sample
    sample_weights = []
    for cls_id in image_classes:
        if cls_id < 0:
            sample_weights.append(bg_weight)
        else:
            sample_weights.append(float(class_weights[cls_id]))

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

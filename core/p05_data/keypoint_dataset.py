"""YOLO-pose format dataset and dataloader utilities for keypoint/pose estimation.

Reads image files and corresponding YOLO-pose format label ``.txt`` files
(one row per object: ``class_id cx cy w h kx1 ky1 v1 kx2 ky2 v2 ... kxK kyK vK``,
normalised 0-1, visibility: 0=not labeled, 1=occluded, 2=visible).
"""

import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from core.p05_data.base_dataset import IMAGENET_MEAN, IMAGENET_STD, BaseDataset  # noqa: F401
from loguru import logger
from utils.config import resolve_path

# Image file extensions to search for
_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

class KeypointTransform:
    """Transform wrapper for keypoint data: resize + color jitter + normalize.

    Accepts ``(image_BGR_np, targets_dict)`` where ``targets_dict`` has
    ``"boxes"`` (N, 5) and ``"keypoints"`` (N, K, 3). Returns
    ``(image_CHW_float32, {"boxes": Tensor(N,5), "keypoints": Tensor(N,K,3)})``.

    Keypoint coordinates are scaled proportionally on resize. Color jitter
    does not affect coordinates.

    Args:
        input_size: Target ``(height, width)``.
        mean: Normalisation mean (RGB order).
        std: Normalisation std (RGB order).
        is_train: If True, apply color jitter augmentation.
        hsv_h: Hue jitter gain (train only).
        hsv_s: Saturation jitter gain (train only).
        hsv_v: Brightness jitter gain (train only).
        fliplr: Probability of horizontal flip (train only).
        flip_indices: Keypoint index permutation for horizontal flip
            (e.g., swap left/right wrist). If None, indices are kept as-is.
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        mean: Sequence[float],
        std: Sequence[float],
        is_train: bool = True,
        hsv_h: float = 0.015,
        hsv_s: float = 0.7,
        hsv_v: float = 0.4,
        fliplr: float = 0.0,
        flip_indices: list[int] | None = None,
    ) -> None:
        self.target_h, self.target_w = input_size
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        self.is_train = is_train
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v
        self.fliplr = fliplr
        self.flip_indices = flip_indices

        # Build v2.ColorJitter for consistent augmentation with detection pipeline
        if is_train and (hsv_h > 0 or hsv_s > 0 or hsv_v > 0):
            self._color_jitter_fn = v2.ColorJitter(
                brightness=hsv_v if hsv_v > 0 else 0,
                contrast=0,
                saturation=hsv_s if hsv_s > 0 else 0,
                hue=hsv_h if hsv_h > 0 else 0,
            )
        else:
            self._color_jitter_fn = None

    def __call__(
        self,
        image: np.ndarray,
        targets_dict: dict[str, np.ndarray],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Apply transforms to image and keypoint targets.

        Args:
            image: HWC uint8 BGR numpy array.
            targets_dict: Dict with ``"boxes"`` (N, 5) float32
                ``[class_id, cx, cy, w, h]`` normalised 0-1, and
                ``"keypoints"`` (N, K, 3) float32 ``[kx, ky, vis]``
                normalised 0-1.

        Returns:
            ``(image_tensor, targets_dict)`` where image_tensor is CHW
            float32 and targets_dict values are torch tensors.
        """
        boxes = targets_dict["boxes"].copy()
        keypoints = targets_dict["keypoints"].copy()
        _orig_h, _orig_w = image.shape[:2]

        # --- Horizontal flip (train only) ---
        if self.is_train and self.fliplr > 0 and random.random() < self.fliplr:
            image = image[:, ::-1, :].copy()
            if len(boxes) > 0:
                # Flip box cx
                boxes[:, 1] = 1.0 - boxes[:, 1]
                # Flip keypoint x
                labeled = keypoints[:, :, 2] > 0  # (N, K) bool
                keypoints[:, :, 0] = np.where(
                    labeled, 1.0 - keypoints[:, :, 0], keypoints[:, :, 0]
                )
                # Swap left/right keypoint indices
                if self.flip_indices is not None:
                    keypoints = keypoints[:, self.flip_indices, :]

        # --- Color jitter (train only, does not affect coordinates) ---
        if self.is_train and self._color_jitter_fn is not None:
            image = self._apply_color_jitter(image)

        # --- Resize ---
        image = cv2.resize(image, (self.target_w, self.target_h))
        # Keypoints and boxes are normalised 0-1 so no coordinate rescaling
        # needed — they remain valid after resize.

        # --- BGR -> RGB, normalize to float32 ---
        image_rgb = image[:, :, ::-1].astype(np.float32) / 255.0
        image_rgb = (image_rgb - self.mean) / self.std

        # HWC -> CHW
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image_rgb.transpose(2, 0, 1))
        )

        # Convert targets to tensors
        if len(boxes) > 0:
            boxes_tensor = torch.from_numpy(boxes.astype(np.float32))
            kpts_tensor = torch.from_numpy(keypoints.astype(np.float32))
        else:
            boxes_tensor = torch.zeros((0, 5), dtype=torch.float32)
            num_kpts = keypoints.shape[1] if keypoints.ndim == 3 else 0
            kpts_tensor = torch.zeros((0, num_kpts, 3), dtype=torch.float32)

        return image_tensor, {"boxes": boxes_tensor, "keypoints": kpts_tensor}

    def _apply_color_jitter(self, image: np.ndarray) -> np.ndarray:
        """Apply color jitter using torchvision v2.ColorJitter.

        Converts BGR→RGB→tensor, applies v2.ColorJitter, converts back
        to BGR numpy for consistency with the rest of the pipeline.
        """
        # BGR -> RGB -> CHW tensor for v2
        rgb = image[:, :, ::-1].copy()
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1))
        jittered = self._color_jitter_fn(tensor)
        # CHW tensor -> HWC numpy -> RGB -> BGR
        result = jittered.numpy().transpose(1, 2, 0)
        return result[:, :, ::-1].copy()

    def __repr__(self) -> str:
        return (
            f"KeypointTransform("
            f"size=({self.target_h}, {self.target_w}), "
            f"train={self.is_train}, "
            f"hsv=({self.hsv_h}, {self.hsv_s}, {self.hsv_v}), "
            f"fliplr={self.fliplr})"
        )


def build_keypoint_transforms(
    is_train: bool,
    input_size: tuple[int, int],
    mean: Sequence[float],
    std: Sequence[float],
    aug_config: dict | None = None,
) -> KeypointTransform:
    """Build a keypoint transform pipeline.

    Args:
        is_train: If True, include colour jitter and flip augmentation.
        input_size: ``(height, width)`` target size.
        mean: Normalisation mean (RGB order).
        std: Normalisation std (RGB order).
        aug_config: Optional augmentation config dict. Supports keys:
            ``hsv_h``, ``hsv_s``, ``hsv_v``, ``fliplr``, ``flip_indices``.

    Returns:
        A :class:`KeypointTransform` callable.
    """
    if aug_config is None:
        aug_config = {}

    return KeypointTransform(
        input_size=input_size,
        mean=mean,
        std=std,
        is_train=is_train,
        hsv_h=aug_config.get("hsv_h", 0.015) if is_train else 0.0,
        hsv_s=aug_config.get("hsv_s", 0.7) if is_train else 0.0,
        hsv_v=aug_config.get("hsv_v", 0.4) if is_train else 0.0,
        fliplr=aug_config.get("fliplr", 0.0) if is_train else 0.0,
        flip_indices=aug_config.get("flip_indices"),
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class KeypointDataset(BaseDataset):
    """Dataset for YOLO-pose format keypoint/pose estimation data.

    Reads images and YOLO-pose label files, applies transforms.

    Directory layout expected::

        <split_root>/
            images/
                img001.jpg
                ...
            labels/   # sibling of images dir
                img001.txt
                ...

    Each label file has one row per object::

        class_id cx cy w h kx1 ky1 v1 kx2 ky2 v2 ... kxK kyK vK

    All coordinate values normalised 0-1. Visibility: 0=not labeled,
    1=occluded, 2=visible.

    Args:
        data_config: Data config dict (loaded from ``configs/<usecase>/05_data.yaml``).
            Must contain ``path``, ``train``/``val``/``test``, ``names``,
            ``num_classes``, ``num_keypoints``, ``input_size``, ``mean``, ``std``.
        split: ``"train"``, ``"val"``, or ``"test"``.
        transforms: A :class:`KeypointTransform` callable. If ``None``,
            images are returned raw (numpy HWC uint8) with numpy targets.
        base_dir: Base directory for resolving relative paths in the config.
            Defaults to the current working directory.
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
        self.img_paths: list[Path] = sorted(
            p for p in self.img_dir.iterdir()
            if p.suffix.lower() in _IMG_EXTENSIONS
        )
        if len(self.img_paths) == 0:
            raise FileNotFoundError(
                f"No images found in {self.img_dir} "
                f"(extensions: {_IMG_EXTENSIONS})"
            )

        self.transforms = transforms
        self.data_config = data_config
        self.num_classes = data_config["num_classes"]
        self.class_names = data_config["names"]
        self.num_keypoints = data_config["num_keypoints"]
        self.input_size = tuple(data_config["input_size"])  # (h, w)

        # Expected columns per label line: 5 (box) + num_keypoints * 3
        self._expected_cols = 5 + self.num_keypoints * 3

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_label(
        self, img_path: Path
    ) -> dict[str, np.ndarray]:
        """Load a YOLO-pose label file corresponding to an image path.

        Returns:
            Dict with:
            - ``"boxes"``: (N, 5) float32 [class_id, cx, cy, w, h]
            - ``"keypoints"``: (N, K, 3) float32 [kx, ky, visibility]
            Returns empty arrays if the label file is missing or empty.
        """
        label_path = self.label_dir / (img_path.stem + ".txt")
        empty = {
            "boxes": np.zeros((0, 5), dtype=np.float32),
            "keypoints": np.zeros((0, self.num_keypoints, 3), dtype=np.float32),
        }

        if not label_path.exists():
            return empty

        try:
            raw_lines = np.loadtxt(label_path, dtype=np.float32, ndmin=2)
        except ValueError:
            return empty

        if raw_lines.size == 0:
            return empty

        if raw_lines.ndim == 1:
            raw_lines = raw_lines.reshape(1, -1)

        # Filter lines with wrong column count
        valid_rows = []
        for i, row in enumerate(raw_lines):
            if len(row) != self._expected_cols:
                logger.warning(
                    "%s line %d: expected %d columns, got %d — skipping",
                    label_path.name, i + 1, self._expected_cols, len(row),
                )
                continue
            valid_rows.append(row)

        if not valid_rows:
            return empty

        data = np.array(valid_rows, dtype=np.float32)

        # Validate class IDs
        valid_mask = np.ones(len(data), dtype=bool)
        class_ids = data[:, 0].astype(int)
        bad_cls = (class_ids < 0) | (class_ids >= self.num_classes)
        if bad_cls.any():
            logger.warning(
                "%s: %d labels with invalid class IDs (not in [0, %d))",
                label_path.name, bad_cls.sum(), self.num_classes,
            )
            valid_mask &= ~bad_cls

        # Validate box coordinates: in [0, 1] with positive w/h
        coords = data[:, 1:5]
        bad_range = (coords < 0).any(axis=1) | (coords > 1).any(axis=1)
        bad_size = (data[:, 3] <= 0) | (data[:, 4] <= 0)
        bad_coords = bad_range | bad_size
        if bad_coords.any():
            logger.warning(
                "%s: %d labels with invalid box coordinates or zero-size boxes",
                label_path.name, bad_coords.sum(),
            )
            valid_mask &= ~bad_coords

        data = data[valid_mask]
        if len(data) == 0:
            return empty

        # Split boxes and keypoints
        boxes = data[:, :5]
        kpt_flat = data[:, 5:]  # (N, K*3)
        keypoints = kpt_flat.reshape(-1, self.num_keypoints, 3)

        # Clamp keypoint coordinates to [0, 1]
        keypoints[:, :, 0] = np.clip(keypoints[:, :, 0], 0.0, 1.0)
        keypoints[:, :, 1] = np.clip(keypoints[:, :, 1], 0.0, 1.0)

        return {"boxes": boxes, "keypoints": keypoints}

    # ------------------------------------------------------------------
    # BaseDataset abstract method implementations
    # ------------------------------------------------------------------

    def load_target(self, label_path: Path) -> dict[str, np.ndarray]:
        """Load and validate a YOLO-pose format label file.

        Delegates to :meth:`_load_label` using the image path derived
        from the label path so that existing validation logic is reused.

        Returns:
            Dict with ``"boxes"`` (N, 5) and ``"keypoints"`` (N, K, 3).
        """
        img_stem = label_path.stem
        for img_path in self.img_paths:
            if img_path.stem == img_stem:
                return self._load_label(img_path)

        # Fallback: no matching image found
        return {
            "boxes": np.zeros((0, 5), dtype=np.float32),
            "keypoints": np.zeros((0, self.num_keypoints, 3), dtype=np.float32),
        }

    def format_target(
        self,
        raw_target: dict[str, np.ndarray],
        image_size: tuple[int, int],  # noqa: ARG002
    ) -> dict[str, np.ndarray]:
        """Return raw YOLO-pose targets as-is (already normalised 0-1).

        Args:
            raw_target: Dict from :meth:`load_target`.
            image_size: (height, width) — unused for YOLO normalised format.

        Returns:
            Dict with ``"boxes"`` (N, 5) and ``"keypoints"`` (N, K, 3).
        """
        return raw_target

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.img_paths)

    def get_raw_item(self, idx: int) -> dict[str, Any]:
        """Return raw image and targets without any transforms.

        Args:
            idx: Index into the dataset.

        Returns:
            Dict with:
            - ``"image"``: numpy HWC uint8 (BGR)
            - ``"targets"``: numpy (N, 5) float32 [class_id, cx, cy, w, h]
            - ``"keypoints"``: numpy (N, K, 3) float32 [kx, ky, visibility]
        """
        img_path = self.img_paths[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            # Return a grey placeholder
            image = np.full(
                (self.input_size[0], self.input_size[1], 3), 114, dtype=np.uint8
            )
        label = self._load_label(img_path)
        return {
            "image": image,
            "targets": label["boxes"],
            "keypoints": label["keypoints"],
        }

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], str]:
        """Get a single sample.

        Args:
            idx: Index into the dataset.

        Returns:
            ``(image_tensor, targets_dict, img_path_str)``

            - ``image_tensor``: float32 tensor (C, H, W).
            - ``targets_dict``: dict with ``"boxes"`` Tensor (N, 5) and
              ``"keypoints"`` Tensor (N, K, 3).
            - ``img_path_str``: Absolute path string of the source image.
        """
        raw = self.get_raw_item(idx)
        image = raw["image"]
        targets_dict = {"boxes": raw["targets"], "keypoints": raw["keypoints"]}
        img_path_str = str(self.img_paths[idx])

        if self.transforms is not None:
            image, targets_dict = self.transforms(image, targets_dict)
            # transforms return torch tensors
            if isinstance(image, torch.Tensor):
                return image, targets_dict, img_path_str

        # Fallback: convert manually if no transforms
        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(
                np.ascontiguousarray(image.transpose(2, 0, 1))
            )

        boxes = targets_dict["boxes"]
        keypoints = targets_dict["keypoints"]

        if isinstance(boxes, np.ndarray):
            boxes = (
                torch.from_numpy(boxes.astype(np.float32))
                if len(boxes)
                else torch.zeros((0, 5), dtype=torch.float32)
            )
        if isinstance(keypoints, np.ndarray):
            keypoints = (
                torch.from_numpy(keypoints.astype(np.float32))
                if len(keypoints)
                else torch.zeros(
                    (0, self.num_keypoints, 3), dtype=torch.float32
                )
            )

        return image, {"boxes": boxes, "keypoints": keypoints}, img_path_str


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def keypoint_collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor], str]],
) -> dict[str, Any]:
    """Custom collate for variable-length keypoint targets.

    Args:
        batch: List of ``(image, targets_dict, path)`` tuples.

    Returns:
        Dictionary with keys:

        - ``images``: float32 tensor (B, C, H, W).
        - ``targets``: list of B dicts, each with ``"boxes"`` (N_i, 5)
          and ``"keypoints"`` (N_i, K, 3).
        - ``paths``: list of B image path strings.
    """
    images, targets_list, paths = zip(*batch, strict=True)
    images = torch.stack(images, dim=0)
    return {
        "images": images,
        "targets": list(targets_list),
        "paths": list(paths),
    }


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def build_keypoint_dataloader(
    data_config: dict,
    split: str,
    training_config: dict,
    base_dir: str | Path | None = None,
) -> DataLoader:
    """Build a DataLoader for keypoint/pose data from config dicts.

    Args:
        data_config: Loaded data YAML config dict. Must contain
            ``num_keypoints`` in addition to standard detection keys.
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
    transforms = build_keypoint_transforms(
        is_train=is_train,
        input_size=input_size,
        mean=mean,
        std=std,
        aug_config=aug_config,
    )

    dataset = KeypointDataset(
        data_config=data_config,
        split=split,
        transforms=transforms,
        base_dir=base_dir,
    )

    data_section = training_config.get("data", {})
    batch_size = data_section.get("batch_size", 16)
    num_workers = data_section.get("num_workers", 4)
    pin_memory = data_section.get("pin_memory", True)

    # Use forkserver to avoid deadlocks with v2 Transform subclasses (nn.Module)
    mp_context = "forkserver" if num_workers > 0 else None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=keypoint_collate_fn,
        drop_last=is_train,
        multiprocessing_context=mp_context,
    )

    return loader

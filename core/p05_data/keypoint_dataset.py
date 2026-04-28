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

        # Resolve image directory. Two supported layouts:
        #   A) <root>/<split>/images + <root>/<split>/labels       (preferred)
        #   B) <root>/<split>        + <root>/labels               (legacy)
        dataset_root = resolve_path(data_config["path"], self.base_dir)
        split_subdir = data_config.get(split)
        if split_subdir is None:
            raise ValueError(f"data config missing '{split}' key")
        candidate_a_img = dataset_root / split_subdir / "images"
        candidate_a_lbl = dataset_root / split_subdir / "labels"
        if candidate_a_img.exists():
            self.img_dir = candidate_a_img
            self.label_dir = candidate_a_lbl
        else:
            self.img_dir = dataset_root / split_subdir
            self.label_dir = self.img_dir.parent / "labels"
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

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
        # Accept either explicit `num_keypoints` or derive from `kpt_shape: [K, 3]`
        # — both conventions appear in real configs.
        nk = data_config.get("num_keypoints")
        if nk is None:
            kpt_shape = data_config.get("kpt_shape") or []
            if isinstance(kpt_shape, (list, tuple)) and len(kpt_shape) >= 1:
                nk = int(kpt_shape[0])
        if not nk:
            raise KeyError(
                "data_config must define `num_keypoints` or `kpt_shape: [K, 3]`"
            )
        self.num_keypoints = int(nk)
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


# ---------------------------------------------------------------------------
# Top-down (per-person crop) dataset for ViTPose-style HF keypoint models
# ---------------------------------------------------------------------------

def _expand_bbox_topdown(
    cx: float, cy: float, w: float, h: float,
    img_w: int, img_h: int,
    aspect_hw: tuple[int, int] = (256, 192),
    padding: float = 1.25,
) -> list[float]:
    """Expand a normalized YOLO cxcywh bbox to a model's H:W aspect + padding.

    Returns ``[x, y, w, h]`` in absolute pixels, clamped to image extents.
    """
    bx_w = w * img_w
    bx_h = h * img_h
    aspect_wh = aspect_hw[1] / aspect_hw[0]  # W/H
    if bx_w > aspect_wh * bx_h:
        bx_h = bx_w / aspect_wh
    else:
        bx_w = aspect_wh * bx_h
    bx_w *= padding
    bx_h *= padding
    pcx, pcy = cx * img_w, cy * img_h
    x = max(0.0, pcx - bx_w / 2.0)
    y = max(0.0, pcy - bx_h / 2.0)
    bx_w = min(img_w - x, bx_w)
    bx_h = min(img_h - y, bx_h)
    return [x, y, bx_w, bx_h]


def _encode_heatmap(
    kpts_in_crop: np.ndarray,  # (K, 2) input-pixel coords
    vis: np.ndarray,           # (K,) {0, 1, 2}
    out_hw: tuple[int, int],   # (H/4, W/4)
    sigma: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-joint 2D Gaussian heatmap target + visibility weight."""
    H, W = out_hw
    K = kpts_in_crop.shape[0]
    target = np.zeros((K, H, W), dtype=np.float32)
    weight = (vis > 0).astype(np.float32)
    stride = 4.0
    for j in range(K):
        if weight[j] == 0:
            continue
        mu_x = kpts_in_crop[j, 0] / stride
        mu_y = kpts_in_crop[j, 1] / stride
        if not (0 <= mu_x < W and 0 <= mu_y < H):
            weight[j] = 0
            continue
        x = np.arange(W, dtype=np.float32)
        y = np.arange(H, dtype=np.float32)[:, None]
        target[j] = np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * sigma ** 2))
    return target, weight


class KeypointTopDownDataset(BaseDataset):
    """Top-down keypoint dataset — one sample per labeled person.

    Reads YOLO-pose ``.txt`` labels (same disk format as ``KeypointDataset``)
    but expands the dataset so each ``__getitem__`` yields a single
    ``(pixel_values, target_heatmap, target_weight)`` triple ready for
    HF top-down models like ViTPose.

    Args:
        data_config: Loaded ``05_data.yaml`` (must include ``num_keypoints``,
            ``input_size``, ``mean``, ``std``).
        split: ``"train"``, ``"val"``, or ``"test"``.
        processor: HF ``VitPoseImageProcessor`` (or any processor with the
            same ``__call__(images, boxes=, return_tensors=)`` API).
        bbox_padding: Aspect-corrected bbox expansion factor.
        heatmap_sigma: Gaussian sigma in heatmap (output-grid) pixels.
        is_train: When True, applies a horizontal-flip augmentation that
            swaps left/right keypoint indices via ``flip_indices``.
        flip_prob: Horizontal flip probability when ``is_train``.
        base_dir: Base directory for resolving ``data_config['path']``.
    """

    def __init__(
        self,
        data_config: dict,
        split: str,
        processor: Any,
        bbox_padding: float = 1.25,
        heatmap_sigma: float = 2.0,
        is_train: bool = False,
        flip_prob: float = 0.5,
        base_dir: str | Path | None = None,
    ) -> None:
        # Resolve dataset root early so we can satisfy BaseDataset's required
        # `data_dir`. BaseDataset will populate image_dir/label_dir from
        # `<data_dir>/<split>/{images,labels}`; we still set our own
        # `img_dir`/`label_dir` further down to support the legacy layout
        # (split_subdir IS the images dir).
        if data_config.get("path") and base_dir is not None:
            _dataset_root = resolve_path(data_config["path"], base_dir)
        else:
            _dataset_root = Path(data_config.get("path", "."))
        super().__init__(
            data_dir=_dataset_root,
            split=split,
            input_size=tuple(data_config["input_size"]),
            transform=None,
        )
        self.data_config = data_config
        self.split = split
        self.processor = processor
        self.bbox_padding = float(bbox_padding)
        self.heatmap_sigma = float(heatmap_sigma)
        self.is_train = bool(is_train)
        self.flip_prob = float(flip_prob)

        self.input_hw = tuple(data_config["input_size"])  # (H, W)
        self.heatmap_hw = (self.input_hw[0] // 4, self.input_hw[1] // 4)
        nk = data_config.get("num_keypoints")
        if nk is None:
            kpt_shape = data_config.get("kpt_shape") or []
            if isinstance(kpt_shape, (list, tuple)) and len(kpt_shape) >= 1:
                nk = int(kpt_shape[0])
        if not nk:
            raise KeyError("data_config must define `num_keypoints` or `kpt_shape`")
        self.num_keypoints = int(nk)

        flip_idx = data_config.get("flip_indices")
        if flip_idx is not None and len(flip_idx) != self.num_keypoints:
            raise ValueError(
                f"flip_indices length {len(flip_idx)} != num_keypoints {self.num_keypoints}"
            )
        self.flip_indices = (
            np.asarray(flip_idx, dtype=np.int64) if flip_idx is not None else None
        )

        # Resolve image dir + label dir. Two supported layouts:
        #   A) <root>/<split>/images + <root>/<split>/labels       (preferred)
        #   B) <root>/<split_subdir>  + <root>/labels              (legacy)
        dataset_root = _dataset_root
        split_subdir = data_config.get(split, split)
        self.img_dir = dataset_root / split_subdir / "images"
        self.label_dir = dataset_root / split_subdir / "labels"
        if not self.img_dir.exists():
            self.img_dir = dataset_root / split_subdir
            self.label_dir = self.img_dir.parent / "labels"
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        # Build per-annotation index: list of (img_path, ann_row).
        self._index: list[tuple[Path, np.ndarray]] = []
        expected_cols = 5 + self.num_keypoints * 3
        for img_path in sorted(
            p for p in self.img_dir.iterdir() if p.suffix.lower() in _IMG_EXTENSIONS
        ):
            label_path = self.label_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                continue
            try:
                rows = np.loadtxt(label_path, dtype=np.float32, ndmin=2)
            except ValueError:
                continue
            if rows.size == 0:
                continue
            for row in rows:
                if len(row) != expected_cols:
                    continue
                kpt_flat = row[5:]
                kpts = kpt_flat.reshape(self.num_keypoints, 3)
                if (kpts[:, 2] > 0).sum() == 0:
                    continue  # no visible keypoints — skip
                self._index.append((img_path, row.astype(np.float32)))

        if not self._index:
            raise FileNotFoundError(
                f"No labeled person crops found under {self.img_dir} / "
                f"{self.label_dir}. Did you run the data dumper?"
            )

        # BaseDataset abstract methods need img_paths (used by some viz hooks).
        self.img_paths = [p for p, _ in self._index]

    # BaseDataset abstract method stubs (top-down doesn't fit YOLO targets).
    def load_target(self, label_path: Path) -> dict[str, np.ndarray]:  # noqa: ARG002
        return {"boxes": np.zeros((0, 5), dtype=np.float32)}

    def format_target(self, raw_target, image_size):  # noqa: ARG002
        return raw_target

    def __len__(self) -> int:
        return len(self._index)

    def get_raw_item(self, idx: int) -> dict[str, Any]:
        """Return the raw image + the single annotation row for `idx`.

        Used by viz callbacks that need GT to draw on top of the original
        image (skeleton on full frame, not on the cropped tensor).
        """
        img_path, row = self._index[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            image = np.full(
                (self.input_hw[0], self.input_hw[1], 3), 114, dtype=np.uint8
            )
        ih, iw = image.shape[:2]
        cls_id = int(row[0])
        cx, cy, w, h = float(row[1]), float(row[2]), float(row[3]), float(row[4])
        kpts = row[5:].reshape(self.num_keypoints, 3).copy()
        # Convert normalized kpts to pixel coords for viz.
        kpts_px = kpts.copy()
        kpts_px[:, 0] *= iw
        kpts_px[:, 1] *= ih
        return {
            "image": image,
            "targets": np.array([[cls_id, cx, cy, w, h]], dtype=np.float32),
            "keypoints": kpts_px[None],  # (1, K, 3) for viz parity
        }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_path, row = self._index[idx]
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        ih, iw = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        cx, cy, w, h = float(row[1]), float(row[2]), float(row[3]), float(row[4])
        kpts = row[5:].reshape(self.num_keypoints, 3).copy()  # normalized

        # Random horizontal flip (train only).
        flipped = False
        if self.is_train and self.flip_indices is not None and random.random() < self.flip_prob:
            image_rgb = image_rgb[:, ::-1].copy()
            cx = 1.0 - cx
            kpts[:, 0] = 1.0 - kpts[:, 0]
            kpts = kpts[self.flip_indices]
            flipped = True
        del flipped  # silence unused-warning; useful as a debug breakpoint

        # Expand bbox to model's aspect ratio + padding.
        bbox = _expand_bbox_topdown(
            cx, cy, w, h, iw, ih, aspect_hw=self.input_hw, padding=self.bbox_padding,
        )

        from PIL import Image as _PIL_Image
        pil_img = _PIL_Image.fromarray(image_rgb)
        proc_out = self.processor(images=pil_img, boxes=[[bbox]], return_tensors="np")
        pixel_values = proc_out["pixel_values"][0].astype(np.float32)  # (3, H, W)

        # Map keypoints into crop space (input-pixel coords).
        bx, by, bw, bh = bbox
        sx = self.input_hw[1] / max(bw, 1e-6)
        sy = self.input_hw[0] / max(bh, 1e-6)
        kpts_px_full = kpts.copy()
        kpts_px_full[:, 0] *= iw
        kpts_px_full[:, 1] *= ih
        kpts_in_crop = np.zeros((self.num_keypoints, 2), dtype=np.float32)
        kpts_in_crop[:, 0] = (kpts_px_full[:, 0] - bx) * sx
        kpts_in_crop[:, 1] = (kpts_px_full[:, 1] - by) * sy
        vis = kpts[:, 2].astype(np.int64)

        target_hm, target_w = _encode_heatmap(
            kpts_in_crop, vis, out_hw=self.heatmap_hw, sigma=self.heatmap_sigma,
        )

        return {
            "pixel_values": torch.from_numpy(pixel_values),
            "target_heatmap": torch.from_numpy(target_hm),
            "target_weight": torch.from_numpy(target_w),
        }


def keypoint_topdown_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack top-down samples into batched tensors for HF Trainer."""
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "target_heatmap": torch.stack([b["target_heatmap"] for b in batch]),
        "target_weight": torch.stack([b["target_weight"] for b in batch]),
    }

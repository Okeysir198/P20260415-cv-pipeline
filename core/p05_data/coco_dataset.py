"""COCO JSON dataset and dataloader utilities for object detection.

Reads images and COCO-format annotation JSON files
(``instances_train2017.json``, etc.) with bounding boxes in absolute
``[x, y, width, height]`` pixel format, converting them to the internal
``(N, 5)`` float32 ``[class_id, cx, cy, w, h]`` normalised 0-1 representation.
"""

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

# Allow imports from project root
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent)
)  # project root

from loguru import logger

from core.p05_data.base_dataset import IMAGENET_MEAN, IMAGENET_STD, BaseDataset  # noqa: E402
from core.p05_data.detection_dataset import collate_fn  # noqa: E402
from core.p05_data.transforms import build_transforms  # noqa: E402
from utils.config import resolve_path  # noqa: E402

# Image file extensions to search for
_IMG_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
}


def _build_coco_category_map(
    coco_categories: list[dict[str, Any]],
    names: dict[int, str],
) -> dict[int, int]:
    """Build a mapping from COCO category_id to internal class_id.

    The mapping is derived by matching COCO category names to the
    ``names`` dict values (case-insensitive).  If a name-based match
    is not found the COCO category is skipped (unmapped annotations
    will be dropped with a warning).

    As a fast-path, if the COCO category IDs already match the keys
    in ``names`` exactly, the identity mapping is returned.

    Args:
        coco_categories: The ``"categories"`` list from the COCO JSON.
        names: Config ``names`` dict mapping ``{class_id: class_name}``.

    Returns:
        Dict mapping ``coco_category_id -> internal_class_id``.
    """
    # Fast-path: check if COCO IDs already match config IDs exactly
    coco_id_set = {cat["id"] for cat in coco_categories}
    if coco_id_set == set(names.keys()):
        return {cid: cid for cid in coco_id_set}

    # Build name -> internal_class_id lookup (lowercased)
    name_to_id: dict[str, int] = {
        v.lower(): k for k, v in names.items()
    }

    mapping: dict[int, int] = {}
    for cat in coco_categories:
        cat_name = cat["name"].lower()
        if cat_name in name_to_id:
            mapping[cat["id"]] = name_to_id[cat_name]
        else:
            logger.debug(
                "COCO category '%s' (id=%d) has no match in config names",
                cat["name"],
                cat["id"],
            )
    return mapping


# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------


class COCODetectionDataset(BaseDataset):
    """Dataset for COCO JSON-format object detection data.

    Reads images and a COCO-format annotation JSON file, applies
    transforms, and returns samples in the same format as
    :class:`YOLOXDataset`.

    Expected directory layout::

        <dataset_root>/
            <split>/            # e.g. train2017/
                image1.jpg
                ...
            annotations/
                instances_<split>.json

    Each annotation has ``bbox`` in ``[x, y, width, height]`` absolute
    pixels (COCO standard).

    Args:
        data_config: Data config dict (loaded from YAML). Must contain
            ``path``, ``train``/``val``/``test``,
            ``train_ann``/``val_ann``/``test_ann``,
            ``names``, ``num_classes``, ``input_size``.
        split: ``"train"``, ``"val"``, or ``"test"``.
        transforms: A transform pipeline. If ``None``, images are
            returned raw (numpy HWC uint8) with numpy targets.
        base_dir: Base directory for resolving relative paths in the
            config.  Defaults to the current working directory.
    """

    def __init__(
        self,
        data_config: dict,
        split: str = "train",
        transforms: Any | None = None,
        base_dir: str | Path | None = None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(
                f"split must be 'train', 'val', or 'test', got '{split}'"
            )

        if base_dir is None:
            base_dir = Path.cwd()
        self.base_dir = Path(base_dir)

        # Resolve dataset root
        dataset_root = resolve_path(data_config["path"], self.base_dir)

        # Resolve image directory
        split_subdir = data_config.get(split)
        if split_subdir is None:
            raise ValueError(f"data config missing '{split}' key")
        self.img_dir = dataset_root / split_subdir
        if not self.img_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.img_dir}"
            )

        # Resolve annotation file
        ann_key = f"{split}_ann"
        ann_subpath = data_config.get(ann_key)
        if ann_subpath is None:
            raise ValueError(
                f"data config missing '{ann_key}' key for split '{split}'"
            )
        ann_path = dataset_root / ann_subpath
        if not ann_path.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {ann_path}"
            )

        self.transforms = transforms
        self.data_config = data_config
        self.num_classes = data_config["num_classes"]
        self.class_names = data_config["names"]
        self.input_size = tuple(data_config["input_size"])  # (h, w)

        # Parse COCO JSON via pycocotools
        from pycocotools.coco import COCO

        self.coco = COCO(str(ann_path))

        # Build category mapping
        coco_cats = self.coco.loadCats(self.coco.getCatIds())
        self._cat_map = _build_coco_category_map(
            coco_cats, self.class_names
        )
        if not self._cat_map:
            logger.warning(
                "No COCO categories matched config names — "
                "all annotations will be dropped"
            )

        # Collect image entries that actually exist on disk
        all_img_ids = sorted(self.coco.getImgIds())
        self._img_entries: list[dict[str, Any]] = []
        self.img_paths: list[Path] = []

        for img_id in all_img_ids:
            info = self.coco.loadImgs(img_id)[0]
            img_path = self.img_dir / info["file_name"]
            if img_path.exists():
                self._img_entries.append(info)
                self.img_paths.append(img_path)

        if len(self.img_paths) == 0:
            raise FileNotFoundError(
                f"No images found in {self.img_dir} matching COCO JSON"
            )

        # Pre-build per-image annotation lists for fast access
        self._img_annotations: list[list[dict[str, Any]]] = []
        for entry in self._img_entries:
            ann_ids = self.coco.getAnnIds(imgIds=entry["id"])
            anns = self.coco.loadAnns(ann_ids)
            self._img_annotations.append(anns)

        # Wire up Mosaic/MixUp dataset references
        self._attach_dataset_to_transforms()

    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------

    def _load_annotations(self, idx: int) -> np.ndarray:
        """Load and validate annotations for a single image.

        Converts COCO ``[x, y, w, h]`` absolute pixels to normalised
        ``[class_id, cx, cy, w, h]``.

        Args:
            idx: Index into the dataset.

        Returns:
            numpy array ``(N, 5)`` float32.  Returns ``(0, 5)`` if no
            valid annotations exist.
        """
        entry = self._img_entries[idx]
        img_w = entry["width"]
        img_h = entry["height"]
        anns = self._img_annotations[idx]

        if not anns:
            return np.zeros((0, 5), dtype=np.float32)

        rows: list[np.ndarray] = []
        n_unmapped = 0
        n_bad_coords = 0
        n_crowd = 0

        for ann in anns:
            # Skip crowd annotations
            if ann.get("iscrowd", 0):
                n_crowd += 1
                continue

            coco_cat_id = ann["category_id"]
            if coco_cat_id not in self._cat_map:
                n_unmapped += 1
                continue

            class_id = self._cat_map[coco_cat_id]

            # Validate class_id
            if class_id < 0 or class_id >= self.num_classes:
                n_unmapped += 1
                continue

            # COCO bbox: [x, y, w, h] absolute pixels
            x, y, bw, bh = ann["bbox"]

            # Skip zero/negative size
            if bw <= 0 or bh <= 0:
                n_bad_coords += 1
                continue

            # Convert to normalised cx, cy, w, h
            cx = (x + bw / 2.0) / img_w
            cy = (y + bh / 2.0) / img_h
            nw = bw / img_w
            nh = bh / img_h

            # Validate range
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
                    and 0.0 < nw <= 1.0 and 0.0 < nh <= 1.0):
                n_bad_coords += 1
                continue

            rows.append(
                np.array(
                    [class_id, cx, cy, nw, nh], dtype=np.float32,
                )
            )

        file_name = entry.get("file_name", f"img_id={entry['id']}")
        if n_unmapped > 0:
            logger.warning(
                "%s: %d annotations with unmapped category IDs",
                file_name,
                n_unmapped,
            )
        if n_bad_coords > 0:
            logger.warning(
                "%s: %d annotations with invalid bbox coordinates",
                file_name,
                n_bad_coords,
            )

        if not rows:
            return np.zeros((0, 5), dtype=np.float32)

        return np.stack(rows, axis=0)

    # ---------------------------------------------------------------
    # BaseDataset abstract method implementations
    # ---------------------------------------------------------------

    def load_target(self, label_path: Path) -> np.ndarray:
        """Load target from a label path (unused for COCO — delegates).

        For COCO datasets the annotations come from the JSON, not from
        individual label files.  This method is required by
        :class:`BaseDataset` but is not the primary loading path.

        Args:
            label_path: Ignored (present for ABC compatibility).

        Returns:
            Empty ``(0, 5)`` array.
        """
        return np.zeros((0, 5), dtype=np.float32)

    def format_target(
        self,
        raw_target: np.ndarray,
        image_size: tuple[int, int],
    ) -> np.ndarray:
        """Return targets as-is (already normalised 0-1).

        Args:
            raw_target: ``(N, 5)`` array from annotation loading.
            image_size: ``(height, width)`` — unused.

        Returns:
            ``(N, 5)`` float32 array ``[class_id, cx, cy, w, h]``.
        """
        return raw_target

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.img_paths)

    def get_raw_item(self, idx: int) -> dict:
        """Return raw image and targets without any transforms.

        Used by :class:`Mosaic`, :class:`MixUp`, and :class:`CopyPaste`
        to sample additional images.

        Args:
            idx: Index into the dataset.

        Returns:
            Dict with ``"image"`` (numpy HWC uint8 BGR) and
            ``"targets"`` (numpy ``(N, 5)`` float32).
        """
        img_path = self.img_paths[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            # Return a grey placeholder
            image = np.full(
                (self.input_size[0], self.input_size[1], 3),
                114,
                dtype=np.uint8,
            )
        targets = self._load_annotations(idx)
        return {"image": image, "targets": targets}

    def __getitem__(
        self, idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Get a single sample.

        Args:
            idx: Index into the dataset.

        Returns:
            ``(image_tensor, targets_tensor, img_path_str)``

            - ``image_tensor``: float32 tensor ``(C, H, W)``.
            - ``targets_tensor``: float32 tensor ``(N, 5)`` — may be
              ``(0, 5)``.
            - ``img_path_str``: Absolute path string of the source
              image.
        """
        raw = self.get_raw_item(idx)
        image, targets = raw["image"], raw["targets"]
        img_path_str = str(self.img_paths[idx])

        if self.transforms is not None:
            image, targets = self.transforms(image, targets)
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


# -------------------------------------------------------------------
# DataLoader builder
# -------------------------------------------------------------------


def build_coco_dataloader(
    data_config: dict,
    split: str,
    training_config: dict,
    base_dir: str | Path | None = None,
) -> DataLoader:
    """Build a DataLoader for a COCO JSON dataset.

    Signature matches :func:`build_dataloader` from
    ``detection_dataset.py`` so callers can swap between YOLO and COCO
    datasets transparently.

    Args:
        data_config: Loaded data YAML config dict.  Must include
            ``train_ann`` / ``val_ann`` / ``test_ann`` keys pointing to
            COCO JSON files.
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

    dataset = COCODetectionDataset(
        data_config=data_config,
        split=split,
        transforms=transforms,
        base_dir=base_dir,
    )

    data_section = training_config.get("data", {})
    batch_size = data_section.get("batch_size", 16)
    num_workers = data_section.get("num_workers", 4)
    pin_memory = data_section.get("pin_memory", True)

    # Use forkserver to avoid deadlocks with torchvision v2 transforms
    mp_context = "forkserver" if num_workers > 0 else None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=is_train,
        multiprocessing_context=mp_context,
    )

    return loader

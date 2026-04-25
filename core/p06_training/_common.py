"""Shared helpers for training-backend callbacks + post-train runner.

Kept tiny and dependency-free so every module in ``core/p06_training/`` and
``core/p08_evaluation/`` can import it without creating a cycle.
"""

from __future__ import annotations

import numpy as np


def unwrap_subset(dataset):
    """Return ``(underlying_dataset, idx_map_fn)`` for a torch Subset, or the
    identity for a regular dataset. Used by every viz/analysis callsite that
    needs to reach through a :class:`torch.utils.data.Subset` to call
    ``get_raw_item`` / ``_load_label`` on the real dataset.
    """
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        indices = dataset.indices
        return dataset.dataset, (lambda i: indices[i])
    return dataset, (lambda i: i)


def task_from_output_format(output_format: str | None) -> str:
    """Normalize ``model.output_format`` → canonical task string.

    Accepts the aliases our model registry emits (``detr``, ``yolox``, …)
    and returns one of ``detection / classification / segmentation / keypoint``.
    """
    of = (output_format or "detection").lower()
    if of in {"detr", "yolox", "detection"}:
        return "detection"
    if of in {"classification", "cls"}:
        return "classification"
    if of in {"segmentation", "seg"}:
        return "segmentation"
    if of in {"keypoint", "pose"}:
        return "keypoint"
    return "detection"


def build_dataset_for_viz(
    task: str,
    split: str,
    data_config: dict,
    base_dir: str,
    transforms=None,
):
    """Return the right p05 Dataset class for a canonical task.

    Mirrors the existing dispatch in ``trainer.py::_maybe_build_test_loader``
    so HF-backend viz callbacks can load the same per-task dataset the
    training loop uses. Raises if ``task`` is not in the supported set
    ({detection, classification, segmentation, keypoint}) — callers should
    gate on that upstream.
    """
    if task == "detection":
        from core.p05_data.detection_dataset import YOLOXDataset

        return YOLOXDataset(
            data_config=data_config, split=split,
            transforms=transforms, base_dir=base_dir,
        )
    if task == "classification":
        from core.p05_data.classification_dataset import ClassificationDataset

        return ClassificationDataset(
            data_config=data_config, split=split,
            transforms=transforms, base_dir=base_dir,
        )
    if task == "segmentation":
        from core.p05_data.segmentation_dataset import SegmentationDataset

        return SegmentationDataset(
            data_config=data_config, split=split,
            transforms=transforms, base_dir=base_dir,
        )
    if task == "keypoint":
        from core.p05_data.keypoint_dataset import KeypointDataset

        return KeypointDataset(
            data_config=data_config, split=split,
            transforms=transforms, base_dir=base_dir,
        )
    raise ValueError(f"build_dataset_for_viz: unsupported task {task!r}")


def yolo_targets_to_xyxy(targets: np.ndarray, w: int, h: int):
    """Denormalize YOLO ``(cls, cx, cy, w, h)`` rows → pixel xyxy + class ids.

    Returns ``(xyxy_float32, class_ids_int64)``. Returns empty arrays if
    ``targets`` is None or empty.
    """
    if targets is None or len(targets) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64)
    cx, cy, bw, bh = targets[:, 1], targets[:, 2], targets[:, 3], targets[:, 4]
    xyxy = np.stack([
        (cx - bw / 2) * w, (cy - bh / 2) * h,
        (cx + bw / 2) * w, (cy + bh / 2) * h,
    ], axis=1).astype(np.float32)
    return xyxy, targets[:, 0].astype(np.int64)

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


class VizSamplingMixin:
    """Shared sampling + live-reload behavior for per-epoch viz callbacks.

    Provides three methods consumed by both ``HFValPredictionCallback`` /
    ``HFTrainPredictionCallback`` (HF backend) and ``ValPredictionLogger``
    (pytorch backend):

    * :meth:`_build_class_buckets` — bucket dataset indices by dominant class
      for balanced sampling. Returns ``None`` when labels can't be cheaply
      read; caller falls back to random.
    * :meth:`_ensure_sample_indices` — append-only sample-pool maintenance.
      Pinned indices are never evicted; growing ``num_samples`` adds new
      indices preferentially from under-represented buckets.
    * :meth:`_refresh_from_config` — re-read the live YAML and refresh the
      mutable viz knobs (``enabled`` / ``num_samples`` / ``conf_threshold``
      / ``grid_cols`` / ``dpi`` / ``balanced``). Silent no-op on missing or
      malformed file.

    Subclass contract: instance must expose mutable attrs ``num_samples``,
    ``conf_threshold``, ``grid_cols``, ``dpi``, ``balanced``,
    ``_sample_indices`` (initially ``None``), and the live-reload locator
    pair ``_config_path`` (``Path | None``) + ``_viz_key`` (str). For pytorch
    backend the gate attribute is ``enabled``; for HF backend it's
    ``enable_epoch_end``. The mixin probes both names so each backend keeps
    its native attr.
    """

    def _build_class_buckets(self, ds) -> dict[int, list[int]] | None:
        if not getattr(self, "balanced", False) or ds is None:
            return None
        raw_ds, idx_map = unwrap_subset(ds)
        if not (hasattr(raw_ds, "img_paths") and hasattr(raw_ds, "_load_label")):
            return None
        from collections import Counter, defaultdict
        buckets: dict[int, list[int]] = defaultdict(list)
        n = len(ds)
        for i in range(n):
            try:
                labels = raw_ds._load_label(raw_ds.img_paths[idx_map(i)])
            except Exception:
                continue
            if labels is None or len(labels) == 0:
                continue
            cls_ids = labels[:, 0].astype(int).tolist()
            counts = Counter(cls_ids)
            dominant = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            buckets[dominant].append(i)
        return dict(buckets) if buckets else None

    def _ensure_sample_indices(self, n: int, ds=None) -> None:
        import random
        from loguru import logger
        target = max(1, min(self.num_samples, n))
        existing = set(self._sample_indices or [])
        buckets = self._build_class_buckets(ds)

        def _pick_balanced(needed: int) -> list[int]:
            assert buckets is not None
            cls_of: dict[int, int] = {
                i: cls for cls, idxs in buckets.items() for i in idxs
            }
            cur_count = {cls: 0 for cls in buckets}
            for i in existing:
                if i in cls_of:
                    cur_count[cls_of[i]] += 1
            avail = {
                cls: [i for i in idxs if i not in existing]
                for cls, idxs in buckets.items()
            }
            for cls in avail:
                random.shuffle(avail[cls])
            picked: list[int] = []
            while len(picked) < needed:
                eligible = [c for c in cur_count if avail.get(c)]
                if not eligible:
                    break
                cls = min(eligible, key=lambda c: cur_count[c])
                picked.append(avail[cls].pop())
                cur_count[cls] += 1
            return picked

        def _pick_random(needed: int) -> list[int]:
            available = [i for i in range(n) if i not in existing]
            if not available:
                return []
            return random.sample(available, min(needed, len(available)))

        if self._sample_indices is None:
            new = _pick_balanced(target) if buckets else _pick_random(target)
            self._sample_indices = sorted(new)
            logger.info(
                "viz sample pool initialized: {} samples ({})",
                len(self._sample_indices),
                "balanced" if buckets else "random",
            )
            return

        if target <= len(self._sample_indices):
            return

        needed = target - len(self._sample_indices)
        new = _pick_balanced(needed) if buckets else _pick_random(needed)
        if not new:
            return
        before = len(self._sample_indices)
        self._sample_indices = sorted(self._sample_indices + new)
        logger.info(
            "viz sample pool expanded: {} → {} (added {}, {})",
            before, len(self._sample_indices), len(new),
            "balanced" if buckets else "random",
        )

    def _refresh_from_config(self) -> None:
        from loguru import logger
        cfg_path = getattr(self, "_config_path", None)
        if cfg_path is None or not cfg_path.exists():
            return
        try:
            import yaml
            with open(cfg_path) as f:
                live = yaml.safe_load(f) or {}
            block = (live.get("training", {}) or {}).get(
                getattr(self, "_viz_key", "val_viz"), {}
            ) or {}
            if "enabled" in block:
                # HF callback uses enable_epoch_end, pytorch uses enabled.
                if hasattr(self, "enable_epoch_end"):
                    self.enable_epoch_end = bool(block["enabled"])
                else:
                    self.enabled = bool(block["enabled"])
            self.num_samples = int(block.get("num_samples", self.num_samples))
            self.conf_threshold = float(block.get("conf_threshold", self.conf_threshold))
            self.grid_cols = int(block.get("grid_cols", self.grid_cols))
            self.dpi = int(block.get("dpi", self.dpi))
            self.balanced = bool(block.get("balanced", self.balanced))
        except Exception as e:
            logger.debug("viz config live-reload skipped: {}", e)


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

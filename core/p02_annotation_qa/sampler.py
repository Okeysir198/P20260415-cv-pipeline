#!/usr/bin/env python3
"""Stratified sampling of dataset images for annotation QA.

Selects a representative subset of images from each dataset split,
ensuring coverage of all classes while respecting a configurable
sample size budget.

Usage (as library):
    from sampler import StratifiedSampler
    sampler = StratifiedSampler(qa_config, data_config, config_dir)
    samples = sampler.sample()  # {"train": [Path, ...], "val": [...]}
"""

import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from utils.config import resolve_path
from utils.exploration import get_image_paths
from utils.yolo_io import image_to_label_path, parse_yolo_label


class StratifiedSampler:
    """Stratified sampling of dataset images for annotation QA.

    Ensures that every class in the dataset is represented by at least
    ``min_per_class`` images in the sampled subset, with the remaining
    quota filled by random sampling from the rest of the dataset.
    """

    def __init__(self, config: dict, data_config: dict, config_dir: Path) -> None:
        """Initialize the sampler.

        Args:
            config: QA config dictionary (must contain a ``sampling`` section
                with keys ``sample_size``, ``strategy``, ``min_per_class``,
                ``seed``, and ``splits``).
            data_config: Data config dictionary (from ``configs/<usecase>/05_data.yaml``).
            config_dir: Directory of the config file, used for resolving
                relative dataset paths.
        """
        sampling = config["sampling"]
        self.sample_size: int = int(sampling["sample_size"])
        self.strategy: str = str(sampling.get("strategy", "stratified"))
        self.min_per_class: int = int(sampling.get("min_per_class", 10))
        self.seed: int = int(sampling.get("seed", 42))
        self.splits: List[str] = list(sampling.get("splits", ["train", "val"]))

        self.data_config = data_config
        self.config_dir = Path(config_dir)

        # Build class-id -> name mapping
        self.class_names: Dict[int, str] = {
            int(k): v for k, v in data_config["names"].items()
        }
        self.num_classes: int = int(data_config["num_classes"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(self) -> Dict[str, List[Path]]:
        """Sample image paths per split.

        Returns:
            Dict mapping split name to list of sampled image paths.

        Logic:
            1. For each split in ``self.splits``:
               a. Get all image paths.
               b. If strategy is ``"all"`` or dataset size <= sample_size,
                  return all images.
               c. Group images by which classes appear in their labels.
               d. Ensure ``min_per_class`` images per class.
               e. Fill remaining quota with random sampling from the
                  remaining (unselected) images.
               f. Seed the RNG for reproducibility.
        """
        results: Dict[str, List[Path]] = {}

        for split in self.splits:
            images_dir = self._resolve_split_dir(split)
            if images_dir is None or not images_dir.exists():
                results[split] = []
                continue

            all_images = get_image_paths(images_dir)
            if not all_images:
                results[split] = []
                continue

            # Return everything when appropriate
            if self.strategy == "all" or len(all_images) <= self.sample_size:
                results[split] = list(all_images)
                continue

            results[split] = self._stratified_sample(all_images)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_split_dir(self, split: str) -> Optional[Path]:
        """Resolve the images directory for a split.

        Args:
            split: Split name (``"train"``, ``"val"``, or ``"test"``).

        Returns:
            Resolved path to the images directory, or ``None`` if the
            split is not defined in the data config.
        """
        if split not in self.data_config:
            return None
        base_path = resolve_path(self.data_config["path"], self.config_dir)
        return base_path / self.data_config[split]

    def _group_by_class(
        self, image_paths: List[Path]
    ) -> Dict[int, List[Path]]:
        """Group images by which classes appear in their labels.

        Args:
            image_paths: List of image file paths.

        Returns:
            Mapping of class_id to list of image paths containing
            that class.
        """
        class_to_images: Dict[int, List[Path]] = defaultdict(list)

        for img_path in image_paths:
            label_path = image_to_label_path(img_path)
            annotations = parse_yolo_label(label_path)
            classes_present: Set[int] = {ann[0] for ann in annotations}
            for cls_id in classes_present:
                class_to_images[cls_id].append(img_path)

        return class_to_images

    def _stratified_sample(self, all_images: List[Path]) -> List[Path]:
        """Perform stratified sampling to build the subset.

        Ensures at least ``min_per_class`` images per class, then fills
        the remaining budget with random draws from unselected images.

        Args:
            all_images: Complete list of image paths for this split.

        Returns:
            Sampled subset of image paths.
        """
        rng = random.Random(self.seed)

        class_to_images = self._group_by_class(all_images)

        selected: Set[Path] = set()

        # Phase 1: guarantee min_per_class coverage
        for cls_id in range(self.num_classes):
            candidates = class_to_images.get(cls_id, [])
            already = [p for p in candidates if p in selected]
            needed = self.min_per_class - len(already)
            if needed <= 0:
                continue

            available = [p for p in candidates if p not in selected]
            pick_count = min(needed, len(available))
            picked = rng.sample(available, pick_count)
            selected.update(picked)

        # Phase 2: fill remaining quota randomly
        remaining_budget = self.sample_size - len(selected)
        if remaining_budget > 0:
            unselected = [p for p in all_images if p not in selected]
            pick_count = min(remaining_budget, len(unselected))
            picked = rng.sample(unselected, pick_count)
            selected.update(picked)

        # Return in a stable order (sorted by path)
        return sorted(selected)

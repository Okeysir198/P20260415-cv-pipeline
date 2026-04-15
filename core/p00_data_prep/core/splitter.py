"""
Split catalog generation with stratified sampling.

Generates splits.json files for flexible train/val/test splitting.
Uses stratified sampling by default to maintain class distribution.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


class SplitGenerator:
    """
    Generate train/val/test splits using stratified sampling by default.

    Stratified splitting ensures each class is proportionally represented
    across all splits, which is crucial for balanced training.
    """

    def __init__(
        self,
        ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        stratified: bool = True
    ):
        """
        Initialize split generator.

        Args:
            ratios: (train, val, test) ratios, must sum to 1.0
            seed: Random seed for reproducibility
            stratified: If True, use stratified sampling (default)
        """
        total = sum(ratios)
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

        self.ratios = ratios
        self.seed = seed
        self.stratified = stratified
        self.rng = np.random.default_rng(seed)

    def generate_splits(
        self,
        samples: List[Dict],
        output_file: Path,
        task_type: str = "detection"
    ) -> Dict[str, List[str]]:
        """
        Generate splits and write to JSON file.

        Args:
            samples: List of sample dicts with 'filename' and 'labels'
            output_file: Path to write splits.json
            task_type: Type of task (detection, classification, etc.)

        Returns:
            Dict with 'train', 'val', 'test' keys containing filename lists
        """
        if not samples:
            raise ValueError("Cannot split empty sample list")

        # Use stratified split by default
        if self.stratified:
            splits = self._stratified_split(samples)
        else:
            # Fallback to simple random split
            splits = self._random_split(samples)

        # Build split catalog
        catalog = {
            "task_type": task_type,
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "seed": self.seed,
                "ratios": list(self.ratios),
                "stratified": self.stratified,
                "total_samples": len(samples)
            },
            "splits": {
                "train": [s["filename"] for s in splits["train"]],
                "val": [s["filename"] for s in splits["val"]],
                "test": [s["filename"] for s in splits["test"]]
            }
        }

        # Write to file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(catalog, f, indent=2)

        return catalog["splits"]

    def _random_split(self, samples: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Simple random split without stratification.

        Args:
            samples: List of sample dicts

        Returns:
            Dict with 'train', 'val', 'test' keys containing sample lists
        """
        n = len(samples)
        indices = self.rng.permutation(n)

        # Use max(1, ...) for minority splits when n >= 3 to avoid empty val/test
        # for rare classes (e.g. n=5, ratio=0.1 → int() gives 0 samples)
        if n >= 3:
            n_val = max(1, round(n * self.ratios[1]))
            n_test = max(1, round(n * self.ratios[2]))
            n_train = n - n_val - n_test
        else:
            n_train = n
            n_val = 0
            n_test = 0

        return {
            "train": [samples[i] for i in indices[:n_train]],
            "val": [samples[i] for i in indices[n_train:n_train + n_val]],
            "test": [samples[i] for i in indices[n_train + n_val:]]
        }

    def _stratified_split(self, samples: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Stratified split maintaining class distribution.

        Groups samples by primary label, then splits each group
        according to ratios. This ensures rare classes are
        proportionally represented in all splits.

        Args:
            samples: List of sample dicts with 'labels' field

        Returns:
            Dict with 'train', 'val', 'test' keys containing sample lists
        """
        # Group samples by primary label
        by_class = {}
        for sample in samples:
            # Get primary label (first label or default)
            if sample.get("labels") and len(sample["labels"]) > 0:
                primary = sample["labels"][0]
            else:
                primary = "__unknown__"

            by_class.setdefault(primary, []).append(sample)

        # Split each class group separately
        splits = {"train": [], "val": [], "test": []}

        for class_samples in by_class.values():
            class_splits = self._random_split(class_samples)
            splits["train"].extend(class_splits["train"])
            splits["val"].extend(class_splits["val"])
            splits["test"].extend(class_splits["test"])

        # Shuffle each split (to mix classes)
        for split_name in splits:
            self.rng.shuffle(splits[split_name])

        return splits

    def load_existing_splits(self, splits_file: Path) -> Dict[str, List[str]]:
        """
        Load existing splits.json file.

        Args:
            splits_file: Path to splits.json

        Returns:
            Dict with 'train', 'val', 'test' keys
        """
        with open(splits_file) as f:
            data = json.load(f)
        return data["splits"]

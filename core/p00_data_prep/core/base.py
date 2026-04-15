"""
Abstract base classes for data preparation.

Defines the interface that all task-specific adapters must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List


class TaskAdapter(ABC):
    """
    Abstract base class for task-specific adapters.

    Each CV task (detection, classification, segmentation, pose) implements
    this interface to provide task-specific parsing, conversion, and validation.
    """

    @abstractmethod
    def parse_source(self, source_config: Dict, base_dir: Path) -> List[Dict]:
        """
        Parse source dataset into unified format.

        Args:
            source_config: Dict with 'name', 'path', 'format', 'class_map', etc.
            base_dir: Base directory for resolving relative paths

        Returns:
            List of sample dicts, each with:
                - 'filename': str (image filename)
                - 'image_path': Path
                - 'labels': List[Any] (task-specific labels)
                - 'source': str (source dataset name)
        """
        pass

    @abstractmethod
    def convert_annotations(
        self,
        samples: List[Dict],
        target_classes: List[str],
        class_mapper: Any
    ) -> List[Dict]:
        """
        Convert annotations to target format with class mapping.

        Args:
            samples: List of sample dicts from parse_source()
            target_classes: List of canonical class names
            class_mapper: ClassMapper instance for name->ID mapping

        Returns:
            List of sample dicts with converted annotations
        """
        pass

    @abstractmethod
    def validate_sample(self, image_path: Path, annotation: Dict) -> bool:
        """
        Validate a single sample.

        Args:
            image_path: Path to image file
            annotation: Annotation dict for this sample

        Returns:
            True if valid, False otherwise
        """
        pass

    @abstractmethod
    def get_class_statistics(self, samples: List[Dict]) -> Dict[str, int]:
        """
        Get per-class statistics.

        Args:
            samples: List of sample dicts

        Returns:
            Dict mapping class name to count
        """
        pass

    @abstractmethod
    def get_primary_label(self, sample: Dict) -> str:
        """
        Get primary label for stratified splitting.

        For detection: first object's class
        For classification: the class label
        For segmentation: dominant class
        For pose: person class (if present)

        Args:
            sample: Sample dict

        Returns:
            Primary class name as string
        """
        pass

    @abstractmethod
    def merge_sources(self, base_dir: Path) -> List[Dict]:
        """
        Parse all sources and merge into a single sample list.

        Args:
            base_dir: Base directory for resolving relative paths

        Returns:
            List of sample dicts with converted annotations
        """
        pass


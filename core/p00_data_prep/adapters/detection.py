"""
Detection task adapter.

Handles object detection datasets with bounding box annotations.
"""

import logging
from pathlib import Path
from typing import Dict, List

from core.p00_data_prep.core.base import TaskAdapter
from core.p00_data_prep.utils.class_mapper import ClassMapper
from core.p00_data_prep.parsers import yolo, voc, coco

logger = logging.getLogger(__name__)


class DetectionAdapter(TaskAdapter):
    """
    Adapter for object detection tasks.

    Supports YOLO, COCO, and Pascal VOC formats as input.
    Outputs YOLO format (class_id cx cy w h normalized).
    """

    def __init__(self, config: Dict):
        self.config = config
        self.target_classes = config.get("classes", [])
        self.class_mapper = ClassMapper(
            self.target_classes,
            self._build_class_map()
        )

    def _build_class_map(self) -> Dict[str, str]:
        class_map: Dict[str, str] = {}
        for source in self.config.get("sources", []):
            for src_name, tgt_name in source.get("class_map", {}).items():
                if src_name in class_map and class_map[src_name] != tgt_name:
                    logger.warning(
                        "Conflicting class map for '%s': '%s' (previous) vs '%s' (source '%s'). "
                        "Using '%s'.",
                        src_name, class_map[src_name], tgt_name,
                        source.get("name", "?"), tgt_name,
                    )
                class_map[src_name] = tgt_name
        return class_map

    def parse_source(self, source_config: Dict, base_dir: Path) -> List[Dict]:
        format_type = source_config.get("format", "yolo")

        if format_type == "yolo":
            return yolo.parse_yolo(source_config, base_dir)
        elif format_type == "voc":
            return voc.parse_voc(source_config, base_dir)
        elif format_type == "coco":
            return coco.parse_coco(source_config, base_dir)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def convert_annotations(
        self,
        samples: List[Dict],
        target_classes: List[str],
        class_mapper: ClassMapper
    ) -> List[Dict]:
        converted = []

        for sample in samples:
            objects = []
            bboxes = sample.get("bboxes", [])

            for i, label in enumerate(sample.get("labels", [])):
                target_id = class_mapper.get_target_id(label)

                if target_id is not None:
                    bbox = bboxes[i] if i < len(bboxes) else [0.5, 0.5, 1.0, 1.0]
                    objects.append({
                        "class_name": class_mapper.get_target_name(label),
                        "class_id": target_id,
                        "cx": bbox[0], "cy": bbox[1], "w": bbox[2], "h": bbox[3],
                    })

            if objects:
                entry = {
                    "filename": sample["filename"],
                    "image_path": sample["image_path"],
                    "labels": [obj["class_name"] for obj in objects],
                    "objects": objects,
                    "source": sample["source"],
                }
                if "original_split" in sample:
                    entry["original_split"] = sample["original_split"]
                converted.append(entry)

        return converted

    def validate_sample(self, image_path: Path, annotation: Dict) -> bool:
        return image_path.exists() and bool(annotation.get("objects"))

    def get_class_statistics(self, samples: List[Dict]) -> Dict[str, int]:
        stats = {name: 0 for name in self.target_classes}

        for sample in samples:
            for obj in sample.get("objects", []):
                class_id = obj.get("class_id")
                if class_id is not None and 0 <= class_id < len(self.target_classes):
                    stats[self.target_classes[class_id]] += 1

        return stats

    def get_primary_label(self, sample: Dict) -> str:
        objects = sample.get("objects", [])
        if objects:
            class_id = objects[0].get("class_id")
            if class_id is not None and 0 <= class_id < len(self.target_classes):
                return self.target_classes[class_id]
        return "__unknown__"

    def merge_sources(self, base_dir: Path) -> List[Dict]:
        all_samples = []

        for source_config in self.config.get("sources", []):
            samples = self.parse_source(source_config, base_dir)

            filters = source_config.get("filters", [])
            if filters:
                samples = self._apply_filters(samples, filters)

            all_samples.extend(samples)

        return self.convert_annotations(all_samples, self.target_classes, self.class_mapper)

    def _apply_filters(self, samples: List[Dict], filters: List[Dict]) -> List[Dict]:
        filtered = []

        for sample in samples:
            labels = sample.get("labels", [])
            for filter_config in filters:
                include_classes = filter_config.get("include_classes", [])
                if any(label in include_classes for label in labels):
                    filtered.append(sample)
                    break

        return filtered

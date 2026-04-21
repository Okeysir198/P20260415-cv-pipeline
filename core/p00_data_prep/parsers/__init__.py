"""Format parsers."""

from .coco import parse_coco
from .voc import parse_voc
from .yolo import parse_yolo

__all__ = ["parse_yolo", "parse_voc", "parse_coco"]

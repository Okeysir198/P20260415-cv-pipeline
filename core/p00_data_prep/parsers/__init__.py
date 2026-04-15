"""Format parsers."""

from .yolo import parse_yolo
from .voc import parse_voc
from .coco import parse_coco

__all__ = ["parse_yolo", "parse_voc", "parse_coco"]

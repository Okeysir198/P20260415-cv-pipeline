"""
Pascal VOC format parser.

Parses Pascal VOC XML annotation files.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

from core.p00_data_prep.utils.file_ops import resolve_data_root


def parse_voc(source_config: Dict, base_dir: Path) -> List[Dict]:
    """
    Parse Pascal VOC format dataset.

    Args:
        source_config: Dict with:
            - path: Path to dataset (relative or absolute)
        base_dir: Base directory for resolving relative paths

    Returns:
        List of sample dicts with 'filename', 'image_path', 'labels', 'bboxes', 'source'
    """
    data_root = resolve_data_root(source_config, base_dir)
    source_name = source_config.get("name", "voc_source")

    samples = []

    img_dir = None
    annotation_dir = None

    for possible_img in ["images", "Image", "JPEGImages", "data"]:
        test_dir = data_root / possible_img
        if test_dir.is_dir():
            img_dir = test_dir
            break

    for possible_ann in ["annotations", "Annotation", "Annotations", "labels"]:
        test_dir = data_root / possible_ann
        if test_dir.is_dir():
            annotation_dir = test_dir
            break

    if img_dir is None:
        data_dir = data_root / "data"
        if data_dir.is_dir():
            img_dir = data_dir / "images"
            annotation_dir = data_dir / "annotations"

    if img_dir is None:
        raise FileNotFoundError(f"Could not find image directory in {data_root}")

    if annotation_dir is None:
        raise FileNotFoundError(f"Could not find annotation directory in {data_root}")

    for xml_path in annotation_dir.glob("*.xml"):
        # Resolve the matching image before parsing so we can use its actual
        # dimensions for bbox normalization (VOC XML `<size>` is not always
        # in sync with the image file).
        img_name = xml_path.stem
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            test_path = img_dir / f"{img_name}{ext}"
            if test_path.exists():
                img_path = test_path
                break

        if img_path is None:
            continue

        objects = _parse_voc_xml(xml_path, img_path=img_path)

        if not objects:
            continue

        samples.append({
            "filename": img_path.name,
            "image_path": img_path,
            "labels": [obj["name"] for obj in objects],
            "bboxes": [obj["bbox"] for obj in objects],
            "source": source_name
        })

    return samples


def _parse_voc_xml(xml_path: Path, img_path: Path = None) -> List[Dict]:
    """
    Parse Pascal VOC XML annotation file.

    Prefers the actual image file's dimensions over the XML `<size>` metadata
    when ``img_path`` is provided — matches the robustness principle from the
    reference DETR notebook pipeline (never trust annotation metadata alone).

    Returns:
        List of dicts with 'name' and 'bbox' ([cx, cy, w, h] normalized)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_elem = root.find("size")
    meta_w, meta_h = 0, 0
    if size_elem is not None:
        w_elem = size_elem.find("width")
        h_elem = size_elem.find("height")
        meta_w = int(w_elem.text) if w_elem is not None and w_elem.text else 0
        meta_h = int(h_elem.text) if h_elem is not None and h_elem.text else 0

    if img_path is not None:
        from ._image_dims import actual_image_dims
        img_w, img_h = actual_image_dims(img_path, fallback_w=meta_w, fallback_h=meta_h)
    else:
        img_w, img_h = meta_w, meta_h

    objects = []
    for obj in root.findall("object"):
        name_elem = obj.find("name")
        if name_elem is None or not name_elem.text:
            continue

        bndbox = obj.find("bndbox")
        if bndbox is not None and img_w > 0 and img_h > 0:
            xmin_e = bndbox.find("xmin")
            ymin_e = bndbox.find("ymin")
            xmax_e = bndbox.find("xmax")
            ymax_e = bndbox.find("ymax")
            if all(e is not None and e.text for e in [xmin_e, ymin_e, xmax_e, ymax_e]):
                bbox = voc_to_yolo_bbox(
                    [float(xmin_e.text), float(ymin_e.text), float(xmax_e.text), float(ymax_e.text)],  # type: ignore[arg-type]
                    img_w, img_h
                )
            else:
                bbox = [0.5, 0.5, 1.0, 1.0]
        else:
            bbox = [0.5, 0.5, 1.0, 1.0]

        objects.append({"name": name_elem.text, "bbox": bbox})

    return objects


def voc_to_yolo_bbox(voc_bbox: List[float], img_w: int, img_h: int) -> List[float]:
    """
    Convert VOC bbox [xmin, ymin, xmax, ymax] to YOLO [cx, cy, w, h] (normalized).
    """
    xmin, ymin, xmax, ymax = voc_bbox

    cx = (xmin + xmax) / 2.0 / img_w
    cy = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h

    return [
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(0.0, min(1.0, w)),
        max(0.0, min(1.0, h)),
    ]

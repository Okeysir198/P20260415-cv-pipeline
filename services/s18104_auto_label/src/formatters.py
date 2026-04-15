"""Output format converters (COCO, YOLO, YOLO-seg, Label Studio)."""

from __future__ import annotations

from fastapi import HTTPException

from src.schemas import Detection


def to_coco(detections: list[Detection], img_w: int, img_h: int) -> list[dict]:
    """Convert detections to COCO-style annotation dicts."""
    results = []
    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        coco_bbox = [x1, y1, x2 - x1, y2 - y1]  # [x, y, w, h] in pixels

        # Segmentation: flatten polygon to [x1,y1,x2,y2,...] in pixels
        segmentation = []
        if det.polygon:
            seg_flat = []
            for pt in det.polygon:
                seg_flat.append(round(pt[0] * img_w, 2))
                seg_flat.append(round(pt[1] * img_h, 2))
            segmentation = [seg_flat]

        results.append({
            "bbox": coco_bbox,
            "segmentation": segmentation,
            "category_id": det.class_id,
            "category_name": det.class_name,
            "score": round(det.score, 4),
            "area": round(det.area * img_w * img_h, 2),
        })
    return results


def to_yolo(detections: list[Detection], _img_w: int, _img_h: int) -> list[str]:
    """Convert detections to YOLO format lines: 'class_id cx cy w h'."""
    lines = []
    for det in detections:
        cx, cy, w, h = det.bbox_norm
        lines.append(f"{det.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def to_yolo_seg(detections: list[Detection], _img_w: int, _img_h: int) -> list[str]:
    """Convert detections to YOLO-seg format lines: 'class_id x1 y1 x2 y2 ... xN yN'."""
    lines = []
    for det in detections:
        if not det.polygon:
            # Fall back to bbox corners if no polygon
            cx, cy, w, h = det.bbox_norm
            x1, y1 = cx - w / 2, cy - h / 2
            x2, y2 = cx + w / 2, cy + h / 2
            coords = f"{x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}"
        else:
            coords = " ".join(f"{pt[0]:.6f} {pt[1]:.6f}" for pt in det.polygon)
        lines.append(f"{det.class_id} {coords}")
    return lines


def to_label_studio(detections: list[Detection], img_w: int, img_h: int) -> list[dict]:
    """Convert detections to Label Studio prediction format."""
    results = []
    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        # Label Studio uses percentages (0-100)
        x_pct = (x1 / img_w) * 100
        y_pct = (y1 / img_h) * 100
        w_pct = ((x2 - x1) / img_w) * 100
        h_pct = ((y2 - y1) / img_h) * 100
        results.append({
            "type": "rectanglelabels",
            "value": {
                "x": round(x_pct, 4),
                "y": round(y_pct, 4),
                "width": round(w_pct, 4),
                "height": round(h_pct, 4),
                "rectanglelabels": [det.class_name],
            },
            "score": round(det.score, 4),
        })
    return results


def format_output(
    detections: list[Detection], output_format: str, img_w: int, img_h: int,
) -> list:
    """Route to the appropriate format converter."""
    fmt = output_format.lower().strip()
    if fmt == "coco":
        return to_coco(detections, img_w, img_h)
    elif fmt == "yolo":
        return to_yolo(detections, img_w, img_h)
    elif fmt == "yolo_seg":
        return to_yolo_seg(detections, img_w, img_h)
    elif fmt == "label_studio":
        return to_label_studio(detections, img_w, img_h)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown output_format '{output_format}'. Supported: coco, yolo, yolo_seg, label_studio",
        )

"""Tests for POST /convert endpoint."""

from __future__ import annotations

import requests

from conftest import SERVICE_URL, load_image_b64, skip_no_service

# First annotate to get real detections, then convert them
_SAMPLE_DETECTION = {
    "class_id": 0,
    "class_name": "fire",
    "score": 0.92,
    "bbox_xyxy": [222, 135, 319, 326],
    "bbox_norm": [0.422266, 0.656534, 0.151172, 0.542614],
    "polygon": [[0.35, 0.38], [0.50, 0.38], [0.50, 0.93], [0.35, 0.93]],
    "mask": None,
    "area": 0.025,
}


@skip_no_service
class TestConvert:
    def test_convert_to_yolo(self):
        """Convert detections from internal format to YOLO."""
        payload = {
            "detections": [_SAMPLE_DETECTION],
            "output_format": "yolo",
            "image_width": 640,
            "image_height": 352,
        }
        resp = requests.post(f"{SERVICE_URL}/convert", json=payload, timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert "formatted_output" in data
        assert len(data["formatted_output"]) == 1
        line = data["formatted_output"][0]
        parts = line.split()
        assert parts[0] == "0"  # class_id
        assert len(parts) == 5  # class_id cx cy w h

    def test_convert_to_coco(self):
        """Convert detections to COCO format."""
        payload = {
            "detections": [_SAMPLE_DETECTION],
            "output_format": "coco",
            "image_width": 640,
            "image_height": 352,
        }
        resp = requests.post(f"{SERVICE_URL}/convert", json=payload, timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["formatted_output"]) == 1
        entry = data["formatted_output"][0]
        assert "category_id" in entry
        assert "bbox" in entry
        assert entry["category_id"] == 0
        assert len(entry["bbox"]) == 4

    def test_convert_to_yolo_seg(self):
        """Convert detections to YOLO-seg format."""
        payload = {
            "detections": [_SAMPLE_DETECTION],
            "output_format": "yolo_seg",
            "image_width": 640,
            "image_height": 352,
        }
        resp = requests.post(f"{SERVICE_URL}/convert", json=payload, timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["formatted_output"]) == 1
        line = data["formatted_output"][0]
        parts = line.split()
        assert parts[0] == "0"  # class_id
        assert len(parts) >= 9  # class_id + 4 xy pairs

    def test_convert_to_label_studio(self):
        """Convert detections to Label Studio format."""
        payload = {
            "detections": [_SAMPLE_DETECTION],
            "output_format": "label_studio",
            "image_width": 640,
            "image_height": 352,
        }
        resp = requests.post(f"{SERVICE_URL}/convert", json=payload, timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["formatted_output"]) == 1
        entry = data["formatted_output"][0]
        assert entry["type"] == "rectanglelabels"
        assert "value" in entry
        assert "score" in entry

    def test_convert_invalid_format(self):
        """Unknown output format returns 400."""
        payload = {
            "detections": [_SAMPLE_DETECTION],
            "output_format": "unknown_format",
            "image_width": 640,
            "image_height": 352,
        }
        resp = requests.post(f"{SERVICE_URL}/convert", json=payload, timeout=10)
        assert resp.status_code == 400

    def test_convert_empty_detections(self):
        """Convert with empty detections returns empty list."""
        payload = {
            "detections": [],
            "output_format": "yolo",
            "image_width": 640,
            "image_height": 352,
        }
        resp = requests.post(f"{SERVICE_URL}/convert", json=payload, timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert data["formatted_output"] == []

    def test_convert_roundtrip_from_annotate(self):
        """Annotate an image, then convert its detections to another format."""
        image_b64 = load_image_b64("fire_sample_1.jpg")
        annotate_resp = requests.post(
            f"{SERVICE_URL}/annotate",
            json={
                "image": image_b64,
                "classes": {"0": "fire"},
                "mode": "text",
                "output_format": "coco",
            },
            timeout=120,
        )
        assert annotate_resp.status_code == 200
        annotate_data = annotate_resp.json()

        if not annotate_data["detections"]:
            return  # No detections to convert

        # Convert same detections to YOLO
        convert_resp = requests.post(
            f"{SERVICE_URL}/convert",
            json={
                "detections": annotate_data["detections"],
                "output_format": "yolo",
                "image_width": annotate_data["image_width"],
                "image_height": annotate_data["image_height"],
            },
            timeout=10,
        )
        assert convert_resp.status_code == 200
        convert_data = convert_resp.json()
        assert len(convert_data["formatted_output"]) == len(annotate_data["detections"])

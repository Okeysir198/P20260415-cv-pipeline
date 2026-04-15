"""Tests for POST /annotate endpoint."""

from __future__ import annotations

import json

import cv2
import requests

from conftest import (
    OUTPUT_DIR,
    SERVICE_URL,
    annotate_image,
    detections_to_sv,
    load_image_b64,
    skip_no_service,
)


@skip_no_service
class TestAnnotate:
    def test_annotate_text_mode(self):
        """Text mode returns detections with expected fields."""
        image_b64 = load_image_b64("fire_sample_1.jpg")
        payload = {
            "image": image_b64,
            "classes": {"0": "fire", "1": "smoke"},
            "mode": "text",
            "confidence_threshold": 0.3,
            "output_format": "coco",
            "include_masks": True,
        }
        resp = requests.post(f"{SERVICE_URL}/annotate", json=payload, timeout=120)
        assert resp.status_code == 200
        data = resp.json()

        assert "detections" in data
        assert "image_width" in data
        assert "image_height" in data
        assert "num_detections" in data
        assert "processing_time_s" in data
        assert "formatted_output" in data
        assert data["num_detections"] == len(data["detections"])

        # Check detection structure
        if data["detections"]:
            det = data["detections"][0]
            assert "class_id" in det
            assert "class_name" in det
            assert "score" in det
            assert "bbox_xyxy" in det
            assert "bbox_norm" in det
            assert len(det["bbox_xyxy"]) == 4
            assert len(det["bbox_norm"]) == 4

    def test_annotate_coco_format(self):
        """COCO output format has correct structure in formatted_output."""
        image_b64 = load_image_b64("fire_sample_1.jpg")
        payload = {
            "image": image_b64,
            "classes": {"0": "fire", "1": "smoke"},
            "mode": "text",
            "confidence_threshold": 0.3,
            "output_format": "coco",
        }
        resp = requests.post(f"{SERVICE_URL}/annotate", json=payload, timeout=120)
        assert resp.status_code == 200
        data = resp.json()
        for entry in data.get("formatted_output", []):
            assert "category_id" in entry
            assert "bbox" in entry
            assert "score" in entry
            assert len(entry["bbox"]) == 4

    def test_annotate_yolo_format(self):
        """YOLO output format returns normalized strings."""
        image_b64 = load_image_b64("fire_sample_1.jpg")
        payload = {
            "image": image_b64,
            "classes": {"0": "fire", "1": "smoke"},
            "mode": "text",
            "confidence_threshold": 0.3,
            "output_format": "yolo",
        }
        resp = requests.post(f"{SERVICE_URL}/annotate", json=payload, timeout=120)
        assert resp.status_code == 200
        data = resp.json()
        for line in data.get("formatted_output", []):
            assert isinstance(line, str)
            parts = line.split()
            assert len(parts) == 5  # class_id cx cy w h
            # All coords should be in [0, 1]
            for val in parts[1:]:
                assert 0.0 <= float(val) <= 1.0

    def test_annotate_yolo_seg_format(self):
        """YOLO-seg output format returns polygon strings."""
        image_b64 = load_image_b64("fire_sample_1.jpg")
        payload = {
            "image": image_b64,
            "classes": {"0": "fire"},
            "mode": "text",
            "confidence_threshold": 0.3,
            "output_format": "yolo_seg",
            "include_masks": True,
        }
        resp = requests.post(f"{SERVICE_URL}/annotate", json=payload, timeout=120)
        assert resp.status_code == 200
        data = resp.json()
        for line in data.get("formatted_output", []):
            assert isinstance(line, str)
            parts = line.split()
            assert len(parts) >= 5  # class_id + at least 2 xy pairs

    def test_annotate_label_studio_format(self):
        """Label Studio output format returns percentage-based dicts."""
        image_b64 = load_image_b64("fire_sample_1.jpg")
        payload = {
            "image": image_b64,
            "classes": {"0": "fire"},
            "mode": "text",
            "confidence_threshold": 0.3,
            "output_format": "label_studio",
        }
        resp = requests.post(f"{SERVICE_URL}/annotate", json=payload, timeout=120)
        assert resp.status_code == 200
        data = resp.json()
        for entry in data.get("formatted_output", []):
            assert entry["type"] == "rectanglelabels"
            assert "value" in entry
            val = entry["value"]
            assert 0 <= val["x"] <= 100
            assert 0 <= val["y"] <= 100
            assert 0 < val["width"] <= 100
            assert 0 < val["height"] <= 100

    def test_annotate_invalid_format(self):
        """Unknown output format returns 400."""
        image_b64 = load_image_b64("fire_sample_1.jpg")
        payload = {
            "image": image_b64,
            "classes": {"0": "fire"},
            "output_format": "invalid_format",
        }
        resp = requests.post(f"{SERVICE_URL}/annotate", json=payload, timeout=120)
        assert resp.status_code == 400

    def test_annotate_invalid_mode(self):
        """Unknown annotation mode returns 400."""
        image_b64 = load_image_b64("fire_sample_1.jpg")
        payload = {
            "image": image_b64,
            "classes": {"0": "fire"},
            "mode": "invalid_mode",
        }
        resp = requests.post(f"{SERVICE_URL}/annotate", json=payload, timeout=120)
        assert resp.status_code == 400

    def test_annotate_saves_overlay(self):
        """Annotate and save overlay visualization."""
        image_b64 = load_image_b64("fire_sample_1.jpg")
        payload = {
            "image": image_b64,
            "classes": {"0": "fire", "1": "smoke"},
            "mode": "text",
            "confidence_threshold": 0.5,
            "output_format": "coco",
            "include_masks": True,
        }
        resp = requests.post(f"{SERVICE_URL}/annotate", json=payload, timeout=120)
        assert resp.status_code == 200
        data = resp.json()

        # Save JSON
        json_path = OUTPUT_DIR / "test01_annotate_response.json"
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        # Save overlay
        detections_raw = data.get("detections", [])
        if detections_raw:
            from conftest import DATA_DIR

            img = cv2.imread(str(DATA_DIR / "fire_sample_1.jpg"))
            img_w, img_h = data["image_width"], data["image_height"]
            sv_dets = detections_to_sv(detections_raw, img_w, img_h)
            annotated = annotate_image(img, sv_dets, {0: "fire", 1: "smoke"})
            overlay_path = OUTPUT_DIR / "test01_annotate_overlay.png"
            cv2.imwrite(str(overlay_path), annotated)
            assert overlay_path.exists()

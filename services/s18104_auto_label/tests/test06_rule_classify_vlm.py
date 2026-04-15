"""Tests for rule-based classification and VLM verification features."""

from __future__ import annotations

import requests

from conftest import (
    OUTPUT_DIR,
    SERVICE_URL,
    load_image_b64,
    skip_no_service,
)


@skip_no_service
class TestRuleClassifyVLM:
    def test_rule_direct_passthrough(self):
        """Direct rules map intermediate detection classes to final output class IDs."""
        image_b64 = load_image_b64("fire_sample_1.jpg")
        payload = {
            "image": image_b64,
            "classes": {"0": "fire", "1": "smoke"},
            "detection_classes": {
                "fire_obj": "flames, burning fire",
                "smoke_obj": "smoke, haze",
            },
            "class_rules": [
                {"output_class_id": 0, "source": "fire_obj", "condition": "direct"},
                {"output_class_id": 1, "source": "smoke_obj", "condition": "direct"},
            ],
            "mode": "text",
            "confidence_threshold": 0.3,
            "output_format": "yolo",
        }
        resp = requests.post(f"{SERVICE_URL}/annotate", json=payload, timeout=120)
        assert resp.status_code == 200
        data = resp.json()

        assert "detections" in data
        assert "num_detections" in data

        # All detections must have final class IDs (0 or 1), not temp IDs like 100/101
        for det in data["detections"]:
            assert det["class_id"] in {0, 1}, (
                f"Expected class_id in {{0, 1}}, got {det['class_id']}"
            )

    def test_rule_overlap_and_no_overlap(self):
        """Overlap/no_overlap rules derive final classes from intermediate detections."""
        image_b64 = load_image_b64("fire_sample_1.jpg")
        payload = {
            "image": image_b64,
            "classes": {"0": "person", "1": "head_with_helmet", "2": "head_without_helmet"},
            "detection_classes": {
                "person": "a person",
                "head": "a person's head",
                "helmet": "a hard hat",
            },
            "class_rules": [
                {"output_class_id": 0, "source": "person", "condition": "direct"},
                {
                    "output_class_id": 1,
                    "source": "head",
                    "condition": "overlap",
                    "target": "helmet",
                    "min_iou": 0.3,
                },
                {
                    "output_class_id": 2,
                    "source": "head",
                    "condition": "no_overlap",
                    "target": "helmet",
                    "min_iou": 0.3,
                },
            ],
            "mode": "text",
            "confidence_threshold": 0.3,
            "output_format": "yolo",
        }
        resp = requests.post(f"{SERVICE_URL}/annotate", json=payload, timeout=120)
        assert resp.status_code == 200
        data = resp.json()

        assert "detections" in data
        # All detections must have final class IDs, not temp IDs (100, 101, 102)
        for det in data["detections"]:
            assert det["class_id"] in {0, 1, 2}, (
                f"Expected class_id in {{0, 1, 2}}, got {det['class_id']}"
            )

    def test_no_rules_backward_compatible(self):
        """Standard annotate call without detection_classes/class_rules still works."""
        image_b64 = load_image_b64("fire_sample_1.jpg")
        payload = {
            "image": image_b64,
            "classes": {"0": "fire", "1": "smoke"},
            "mode": "text",
            "confidence_threshold": 0.3,
        }
        resp = requests.post(f"{SERVICE_URL}/annotate", json=payload, timeout=120)
        assert resp.status_code == 200
        data = resp.json()

        assert "detections" in data
        assert "num_detections" in data
        assert data["num_detections"] == len(data["detections"])

        for det in data["detections"]:
            assert det["class_id"] in {0, 1}, (
                f"Expected class_id in {{0, 1}}, got {det['class_id']}"
            )

    def test_vlm_verify_config_accepted(self):
        """VLM verify config is accepted without crashing (fail-open if Ollama unavailable)."""
        image_b64 = load_image_b64("fire_sample_1.jpg")
        payload = {
            "image": image_b64,
            "classes": {"0": "fire"},
            "mode": "text",
            "vlm_verify": {
                "model": "qwen3.5:9b",
                "ollama_url": "http://localhost:11434",
                "verify_classes": [0],
                "budget": {"sample_rate": 0.1, "max_samples": 5},
            },
        }
        resp = requests.post(f"{SERVICE_URL}/annotate", json=payload, timeout=120)
        # VLM errors should be fail-open, not crash the request
        assert resp.status_code == 200
        data = resp.json()

        assert "detections" in data
        assert "num_detections" in data

    def test_rule_output_format_yolo(self):
        """Rule-based detections produce correct YOLO-formatted output with final class IDs."""
        image_b64 = load_image_b64("fire_sample_1.jpg")
        payload = {
            "image": image_b64,
            "classes": {"0": "fire", "1": "smoke"},
            "detection_classes": {
                "fire_obj": "flames, burning fire",
                "smoke_obj": "smoke, haze",
            },
            "class_rules": [
                {"output_class_id": 0, "source": "fire_obj", "condition": "direct"},
                {"output_class_id": 1, "source": "smoke_obj", "condition": "direct"},
            ],
            "mode": "text",
            "confidence_threshold": 0.3,
            "output_format": "yolo",
        }
        resp = requests.post(f"{SERVICE_URL}/annotate", json=payload, timeout=120)
        assert resp.status_code == 200
        data = resp.json()

        assert "formatted_output" in data
        for line in data.get("formatted_output", []):
            assert isinstance(line, str)
            parts = line.split()
            assert len(parts) == 5, f"Expected 5 parts (class_id cx cy w h), got {len(parts)}"
            # Class ID must be 0 or 1 (final IDs), not temp IDs
            class_id = int(parts[0])
            assert class_id in {0, 1}, f"Expected class_id in {{0, 1}}, got {class_id}"
            # All coords should be in [0, 1]
            for val in parts[1:]:
                assert 0.0 <= float(val) <= 1.0, f"Coord {val} not in [0, 1]"

    def test_detection_structure_with_rules(self):
        """Rule-based detections have all expected fields in each detection dict."""
        image_b64 = load_image_b64("fire_sample_1.jpg")
        payload = {
            "image": image_b64,
            "classes": {"0": "fire", "1": "smoke"},
            "detection_classes": {
                "fire_obj": "flames, burning fire",
                "smoke_obj": "smoke, haze",
            },
            "class_rules": [
                {"output_class_id": 0, "source": "fire_obj", "condition": "direct"},
                {"output_class_id": 1, "source": "smoke_obj", "condition": "direct"},
            ],
            "mode": "text",
            "confidence_threshold": 0.3,
            "output_format": "coco",
        }
        resp = requests.post(f"{SERVICE_URL}/annotate", json=payload, timeout=120)
        assert resp.status_code == 200
        data = resp.json()

        assert "detections" in data
        assert "image_width" in data
        assert "image_height" in data
        assert "num_detections" in data
        assert "processing_time_s" in data
        assert data["num_detections"] == len(data["detections"])

        # Each detection must have all expected fields
        for det in data["detections"]:
            assert "class_id" in det, "Missing 'class_id' field"
            assert "class_name" in det, "Missing 'class_name' field"
            assert "score" in det, "Missing 'score' field"
            assert "bbox_xyxy" in det, "Missing 'bbox_xyxy' field"
            assert "bbox_norm" in det, "Missing 'bbox_norm' field"
            assert len(det["bbox_xyxy"]) == 4
            assert len(det["bbox_norm"]) == 4
            assert isinstance(det["score"], float)
            assert 0.0 <= det["score"] <= 1.0
            assert det["class_id"] in {0, 1}
            assert det["class_name"] in {"fire", "smoke"}

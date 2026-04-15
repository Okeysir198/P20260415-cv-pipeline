"""Tests for POST /segment_text endpoint."""

from __future__ import annotations

import cv2
import requests

from conftest import (
    OUTPUT_DIR,
    REQUEST_TIMEOUT,
    SERVICE_URL,
    annotate_image,
    detections_from_masks,
    load_image_b64,
    load_image_cv2,
    skip_no_service,
)


@skip_no_service
class TestSegmentText:
    def test_single_text(self):
        """Text request returns detections list."""
        image_b64 = load_image_b64("cat.jpg")
        resp = requests.post(
            f"{SERVICE_URL}/segment_text",
            json={"image": image_b64, "text": "cat"},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "detections" in data
        assert isinstance(data["detections"], list)

    def test_detection_fields(self):
        """Each detection has mask, bbox, score, area."""
        image_b64 = load_image_b64("cat.jpg")
        resp = requests.post(
            f"{SERVICE_URL}/segment_text",
            json={"image": image_b64, "text": "cat"},
            timeout=REQUEST_TIMEOUT,
        )
        detections = resp.json()["detections"]
        if detections:
            det = detections[0]
            assert "mask" in det
            assert "bbox" in det
            assert "score" in det
            assert "area" in det
            assert det["score"] > 0

    def test_custom_thresholds(self):
        """Custom detection and mask thresholds are accepted."""
        image_b64 = load_image_b64("cat.jpg")
        resp = requests.post(
            f"{SERVICE_URL}/segment_text",
            json={
                "image": image_b64,
                "text": "cat",
                "detection_threshold": 0.3,
                "mask_threshold": 0.3,
            },
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200

    def test_batch_text(self):
        """Batch request returns a list of responses."""
        image_b64 = load_image_b64("cat.jpg")
        batch = [
            {"image": image_b64, "text": "cat"},
            {"image": image_b64, "text": "ear"},
        ]
        resp = requests.post(
            f"{SERVICE_URL}/segment_text",
            json=batch,
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 2

    def test_saves_overlay(self):
        """Segment text and save supervision overlay."""
        image_b64 = load_image_b64("cat.jpg")
        resp = requests.post(
            f"{SERVICE_URL}/segment_text",
            json={"image": image_b64, "text": "cat"},
            timeout=REQUEST_TIMEOUT,
        )
        detections_raw = resp.json()["detections"]
        img = load_image_cv2("cat.jpg")
        dets = detections_from_masks(detections_raw)
        labels = [f"score: {d['score']:.3f}" for d in detections_raw]
        annotated = annotate_image(img, dets, labels)

        out_path = OUTPUT_DIR / "test02_segment_text.png"
        cv2.imwrite(str(out_path), annotated)
        assert out_path.exists()

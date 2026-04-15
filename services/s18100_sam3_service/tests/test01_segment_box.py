"""Tests for POST /segment_box endpoint."""

from __future__ import annotations

import cv2
import requests

from conftest import (
    OUTPUT_DIR,
    REQUEST_TIMEOUT,
    SERVICE_URL,
    annotate_image,
    base64_to_mask,
    detections_from_masks,
    load_image_b64,
    load_image_cv2,
    skip_no_service,
)


@skip_no_service
class TestSegmentBox:
    def test_single_box(self):
        """Single box request returns result with expected fields."""
        image_b64 = load_image_b64("truck.jpg")
        resp = requests.post(
            f"{SERVICE_URL}/segment_box",
            json={"image": image_b64, "box": [200, 200, 600, 500]},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "result" in data
        result = data["result"]
        assert "mask" in result
        assert "bbox" in result
        assert "score" in result
        assert "iou_score" in result
        assert result["score"] > 0
        assert result["iou_score"] > 0

    def test_mask_is_valid_image(self):
        """Returned mask can be decoded to a boolean array."""
        image_b64 = load_image_b64("truck.jpg")
        resp = requests.post(
            f"{SERVICE_URL}/segment_box",
            json={"image": image_b64, "box": [200, 200, 600, 500]},
            timeout=REQUEST_TIMEOUT,
        )
        result = resp.json()["result"]
        mask = base64_to_mask(result["mask"])
        assert mask.ndim == 2
        assert mask.dtype == bool
        assert mask.any()

    def test_bbox_has_coordinates(self):
        """Returned bbox has x1, y1, x2, y2 fields."""
        image_b64 = load_image_b64("truck.jpg")
        resp = requests.post(
            f"{SERVICE_URL}/segment_box",
            json={"image": image_b64, "box": [200, 200, 600, 500]},
            timeout=REQUEST_TIMEOUT,
        )
        bbox = resp.json()["result"]["bbox"]
        assert all(k in bbox for k in ("x1", "y1", "x2", "y2"))
        assert bbox["x2"] > bbox["x1"]
        assert bbox["y2"] > bbox["y1"]

    def test_batch_box(self):
        """Batch request returns a list of results."""
        image_b64 = load_image_b64("truck.jpg")
        batch = [
            {"image": image_b64, "box": [200, 200, 600, 500]},
            {"image": image_b64, "box": [0, 0, 300, 300]},
        ]
        resp = requests.post(
            f"{SERVICE_URL}/segment_box",
            json=batch,
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 2
        for item in data:
            assert "result" in item

    def test_saves_overlay(self):
        """Segment box and save supervision overlay."""
        image_b64 = load_image_b64("truck.jpg")
        box = [200, 200, 600, 500]
        resp = requests.post(
            f"{SERVICE_URL}/segment_box",
            json={"image": image_b64, "box": box},
            timeout=REQUEST_TIMEOUT,
        )
        result = resp.json()["result"]
        img = load_image_cv2("truck.jpg")
        dets = detections_from_masks([result])
        labels = [f"IoU: {result['iou_score']:.3f}"]
        annotated = annotate_image(img, dets, labels)

        out_path = OUTPUT_DIR / "test01_segment_box.png"
        cv2.imwrite(str(out_path), annotated)
        assert out_path.exists()

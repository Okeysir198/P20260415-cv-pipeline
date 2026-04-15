"""Tests for POST /auto_mask endpoint."""

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
class TestAutoMask:
    def test_auto_mask_default(self):
        """Auto-mask with default prompts returns detections."""
        image_b64 = load_image_b64("cat.jpg")
        resp = requests.post(
            f"{SERVICE_URL}/auto_mask",
            json={"image": image_b64},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "detections" in data
        assert isinstance(data["detections"], list)

    def test_auto_mask_custom_prompts(self):
        """Custom prompts are accepted and return detections."""
        image_b64 = load_image_b64("cat.jpg")
        resp = requests.post(
            f"{SERVICE_URL}/auto_mask",
            json={"image": image_b64, "prompts": ["cat. ear. eye."]},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "detections" in data

    def test_auto_mask_custom_threshold(self):
        """Custom threshold is accepted."""
        image_b64 = load_image_b64("cat.jpg")
        resp = requests.post(
            f"{SERVICE_URL}/auto_mask",
            json={"image": image_b64, "threshold": 0.5},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200

    def test_saves_overlay(self):
        """Auto-mask and save supervision overlay."""
        image_b64 = load_image_b64("cat.jpg")
        resp = requests.post(
            f"{SERVICE_URL}/auto_mask",
            json={"image": image_b64},
            timeout=REQUEST_TIMEOUT,
        )
        detections_raw = resp.json()["detections"]
        img = load_image_cv2("cat.jpg")
        dets = detections_from_masks(detections_raw)
        labels = [f"score: {d['score']:.3f}" for d in detections_raw]
        annotated = annotate_image(img, dets, labels)

        out_path = OUTPUT_DIR / "test03_auto_mask.png"
        cv2.imwrite(str(out_path), annotated)
        assert out_path.exists()

"""Tests for batch job endpoints: POST/GET/DELETE /jobs."""

from __future__ import annotations

import json
import time

import cv2
import requests

from conftest import (
    DATA_DIR,
    OUTPUT_DIR,
    SERVICE_URL,
    annotate_image,
    detections_to_sv,
    load_image_b64,
    skip_no_service,
)

POLL_INTERVAL = 2


@skip_no_service
class TestJobs:
    def test_create_and_poll_job(self):
        """Create a batch job, poll until complete, verify results."""
        images = sorted(DATA_DIR.glob("*.jpg"))[:3]
        assert len(images) > 0, "No test images in data/"

        image_payloads = []
        for img_path in images:
            image_payloads.append({
                "image": load_image_b64(img_path.name),
                "filename": img_path.name,
            })

        payload = {
            "images": image_payloads,
            "classes": {"0": "fire", "1": "smoke"},
            "mode": "text",
            "confidence_threshold": 0.5,
            "output_format": "coco",
            "include_masks": False,
        }

        # POST /jobs — create
        resp = requests.post(f"{SERVICE_URL}/jobs", json=payload, timeout=30)
        assert resp.status_code == 200
        job = resp.json()
        assert "job_id" in job
        assert job["status"] == "queued"
        assert job["total_images"] == len(images)
        job_id = job["job_id"]

        # GET /jobs/{id} — poll until complete
        for _ in range(60):
            time.sleep(POLL_INTERVAL)
            resp = requests.get(f"{SERVICE_URL}/jobs/{job_id}", timeout=30)
            assert resp.status_code == 200
            status = resp.json()
            if status["status"] in ("completed", "failed"):
                break

        assert status["status"] == "completed"
        assert status["processed_images"] == len(images)
        assert len(status["results"]) == len(images)

        # Verify result structure
        for result in status["results"]:
            assert "filename" in result
            assert "num_detections" in result
            assert "detections" in result
            assert "formatted_output" in result

        # Save outputs
        json_path = OUTPUT_DIR / "test02_jobs_response.json"
        with open(json_path, "w") as f:
            json.dump(status, f, indent=2)

        # Save overlays
        class_names = {0: "fire", 1: "smoke"}
        image_map = {img.name: img for img in images}
        for res in status["results"]:
            filename = res.get("filename", "")
            detections = res.get("detections", [])
            if not detections or filename not in image_map:
                continue
            img = cv2.imread(str(image_map[filename]))
            img_h, img_w = img.shape[:2]
            sv_dets = detections_to_sv(detections, img_w, img_h)
            annotated = annotate_image(img, sv_dets, class_names)
            stem = image_map[filename].stem
            cv2.imwrite(str(OUTPUT_DIR / f"test02_jobs_overlay_{stem}.png"), annotated)

    def test_list_jobs(self):
        """GET /jobs returns a list."""
        resp = requests.get(f"{SERVICE_URL}/jobs", timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        if data:
            item = data[0]
            assert "job_id" in item
            assert "status" in item
            assert "total_images" in item
            assert "processed_images" in item
            assert "created_at" in item

    def test_list_jobs_filter_by_status(self):
        """GET /jobs?status=completed filters results."""
        resp = requests.get(f"{SERVICE_URL}/jobs?status=completed", timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        for item in data:
            assert item["status"] == "completed"

    def test_cancel_nonexistent_job(self):
        """DELETE /jobs/{id} returns 404 for unknown job."""
        resp = requests.delete(f"{SERVICE_URL}/jobs/nonexistent_id_123", timeout=10)
        assert resp.status_code == 404

    def test_cancel_job(self):
        """Create a job and cancel it immediately."""
        image_b64 = load_image_b64("fire_sample_1.jpg")
        payload = {
            "images": [{"image": image_b64, "filename": "cancel_test.jpg"}],
            "classes": {"0": "fire"},
            "mode": "text",
        }

        # Create
        resp = requests.post(f"{SERVICE_URL}/jobs", json=payload, timeout=30)
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        # Cancel
        resp = requests.delete(f"{SERVICE_URL}/jobs/{job_id}", timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == job_id
        assert data["status"] in ("cancelled", "completed", "failed")

    def test_create_job_no_images(self):
        """POST /jobs with empty images returns 400."""
        payload = {
            "images": [],
            "classes": {"0": "fire"},
        }
        resp = requests.post(f"{SERVICE_URL}/jobs", json=payload, timeout=10)
        assert resp.status_code == 400

    def test_get_nonexistent_job(self):
        """GET /jobs/{id} returns 404 for unknown job."""
        resp = requests.get(f"{SERVICE_URL}/jobs/nonexistent_id_456", timeout=10)
        assert resp.status_code == 404

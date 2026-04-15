"""Full endpoint coverage — real HTTP requests to running service."""

import time

import requests

from conftest import SERVICE_URL, skip_no_service, skip_no_ollama


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealth:

    @skip_no_service
    def test_health_returns_200(self):
        resp = requests.get(f"{SERVICE_URL}/health", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "sam3" in data
        assert "ollama" in data
        assert "active_jobs" in data

    @skip_no_service
    def test_health_ollama_field(self):
        resp = requests.get(f"{SERVICE_URL}/health", timeout=5)
        data = resp.json()
        assert "ollama" in data
        assert isinstance(data["ollama"], str)


# ---------------------------------------------------------------------------
# POST /validate
# ---------------------------------------------------------------------------


class TestValidate:

    @skip_no_service
    def test_validate_yolo(self, test_image_b64, sample_yolo_labels, sample_classes):
        resp = requests.post(f"{SERVICE_URL}/validate", json={
            "image": test_image_b64,
            "labels": sample_yolo_labels,
            "label_format": "yolo",
            "classes": sample_classes,
        }, timeout=30)
        assert resp.status_code == 200
        data = resp.json()
        assert "issues" in data
        assert "quality_score" in data
        assert "grade" in data

    @skip_no_service
    def test_validate_coco(self, test_image_b64, sample_classes):
        resp = requests.post(f"{SERVICE_URL}/validate", json={
            "image": test_image_b64,
            "labels": [
                {"category_id": 0, "bbox": [100, 50, 200, 150]},
            ],
            "label_format": "coco",
            "classes": sample_classes,
        }, timeout=30)
        assert resp.status_code == 200

    @skip_no_service
    def test_validate_empty_labels(self, test_image_b64, sample_classes):
        resp = requests.post(f"{SERVICE_URL}/validate", json={
            "image": test_image_b64,
            "labels": [],
            "label_format": "yolo",
            "classes": sample_classes,
        }, timeout=30)
        assert resp.status_code == 200

    @skip_no_service
    def test_validate_invalid_class(self, test_image_b64, sample_classes):
        resp = requests.post(f"{SERVICE_URL}/validate", json={
            "image": test_image_b64,
            "labels": ["99 0.5 0.5 0.2 0.2"],
            "label_format": "yolo",
            "classes": sample_classes,
        }, timeout=30)
        assert resp.status_code == 200
        data = resp.json()
        issue_types = [i["type"] for i in data["issues"]]
        assert "invalid_class" in issue_types

    @skip_no_service
    def test_validate_out_of_bounds(self, test_image_b64, sample_classes):
        resp = requests.post(f"{SERVICE_URL}/validate", json={
            "image": test_image_b64,
            "labels": ["0 1.1 0.5 0.2 0.2"],
            "label_format": "yolo",
            "classes": sample_classes,
        }, timeout=30)
        assert resp.status_code == 200
        data = resp.json()
        issue_types = [i["type"] for i in data["issues"]]
        assert "out_of_bounds" in issue_types


# ---------------------------------------------------------------------------
# POST /verify
# ---------------------------------------------------------------------------


class TestVerify:

    @skip_no_service
    def test_verify_basic(self, test_image_b64, sample_classes):
        resp = requests.post(f"{SERVICE_URL}/verify", json={
            "image": test_image_b64,
            "labels": ["0 0.5 0.5 0.3 0.2"],
            "classes": sample_classes,
        }, timeout=120)
        assert resp.status_code in (200, 503)

    @skip_no_service
    def test_verify_without_vlm(self, test_image_b64, sample_classes):
        resp = requests.post(f"{SERVICE_URL}/verify", json={
            "image": test_image_b64,
            "labels": ["0 0.5 0.5 0.3 0.2"],
            "classes": sample_classes,
        }, timeout=120)
        if resp.status_code == 200:
            data = resp.json()
            assert data.get("vlm_verification") is None

    @skip_no_service
    @skip_no_ollama
    def test_verify_with_vlm(self, test_image_b64, sample_classes):
        resp = requests.post(f"{SERVICE_URL}/verify", json={
            "image": test_image_b64,
            "labels": ["0 0.5 0.5 0.3 0.2"],
            "classes": sample_classes,
            "enable_vlm": True,
            "vlm_trigger": "all",
        }, timeout=180)
        assert resp.status_code in (200, 503)


# ---------------------------------------------------------------------------
# POST /fix
# ---------------------------------------------------------------------------


class TestFix:

    @skip_no_service
    def test_fix_clip(self, sample_classes):
        resp = requests.post(f"{SERVICE_URL}/fix", json={
            "labels": ["0 1.05 0.5 0.2 0.3"],
            "label_format": "yolo",
            "classes": sample_classes,
            "suggested_fixes": [{
                "type": "clip_bbox",
                "annotation_idx": 0,
                "suggested": {"bbox_norm": [0.95, 0.5, 0.1, 0.3]},
            }],
        }, timeout=30)
        assert resp.status_code == 200
        data = resp.json()
        assert "corrected_labels" in data

    @skip_no_service
    def test_fix_remove_duplicate(self, sample_classes):
        resp = requests.post(f"{SERVICE_URL}/fix", json={
            "labels": ["0 0.5 0.5 0.2 0.2", "0 0.501 0.501 0.2 0.2"],
            "label_format": "yolo",
            "classes": sample_classes,
            "suggested_fixes": [{
                "type": "remove_duplicate",
                "annotation_idx": 1,
            }],
        }, timeout=30)
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_annotations_after"] < data["num_annotations_before"]

    @skip_no_service
    def test_fix_roundtrip_coco(self, sample_classes):
        resp = requests.post(f"{SERVICE_URL}/fix", json={
            "labels": [{"category_id": 0, "bbox": [100, 50, 200, 150]}],
            "label_format": "coco",
            "classes": sample_classes,
            "suggested_fixes": [],
        }, timeout=30)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /jobs + GET /jobs + GET /jobs/{id} + DELETE /jobs/{id}
# ---------------------------------------------------------------------------


class TestJobs:

    @skip_no_service
    def test_create_and_poll_job(self, test_image_b64, sample_classes):
        resp = requests.post(f"{SERVICE_URL}/jobs", json={
            "images": [{
                "image": test_image_b64,
                "filename": "test.jpg",
                "labels": ["0 0.5 0.5 0.3 0.2"],
            }],
            "label_format": "yolo",
            "classes": sample_classes,
            "mode": "validate",
        }, timeout=10)
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        # Poll until complete
        for _ in range(30):
            status_resp = requests.get(f"{SERVICE_URL}/jobs/{job_id}", timeout=5)
            assert status_resp.status_code == 200
            if status_resp.json()["status"] == "completed":
                break
            time.sleep(1)

    @skip_no_service
    def test_list_jobs(self):
        resp = requests.get(f"{SERVICE_URL}/jobs", timeout=5)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    @skip_no_service
    def test_list_jobs_filter(self):
        resp = requests.get(f"{SERVICE_URL}/jobs?status=completed", timeout=5)
        assert resp.status_code == 200

    @skip_no_service
    def test_get_job_404(self):
        resp = requests.get(f"{SERVICE_URL}/jobs/nonexistent123", timeout=5)
        assert resp.status_code == 404

    @skip_no_service
    def test_delete_job_404(self):
        resp = requests.delete(f"{SERVICE_URL}/jobs/nonexistent123", timeout=5)
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /report
# ---------------------------------------------------------------------------


class TestReport:

    @skip_no_service
    def test_report_basic(self, sample_classes):
        resp = requests.post(f"{SERVICE_URL}/report", json={
            "results": [
                {"filename": "a.jpg", "quality_score": 0.9, "grade": "good",
                 "issues": [], "suggested_fixes": [], "num_issues": 0, "num_annotations": 3},
                {"filename": "b.jpg", "quality_score": 0.4, "grade": "bad",
                 "issues": [{"type": "out_of_bounds"}], "suggested_fixes": [], "num_issues": 1, "num_annotations": 2},
            ],
            "classes": sample_classes,
        }, timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_checked"] == 2
        assert "grades" in data

    @skip_no_service
    def test_report_empty(self, sample_classes):
        resp = requests.post(f"{SERVICE_URL}/report", json={
            "results": [],
            "classes": sample_classes,
        }, timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_checked"] == 0

    @skip_no_service
    def test_report_grade_distribution(self, sample_classes):
        resp = requests.post(f"{SERVICE_URL}/report", json={
            "results": [
                {"filename": "a.jpg", "quality_score": 0.9, "grade": "good",
                 "issues": [], "suggested_fixes": [], "num_issues": 0, "num_annotations": 1},
                {"filename": "b.jpg", "quality_score": 0.6, "grade": "review",
                 "issues": [], "suggested_fixes": [], "num_issues": 0, "num_annotations": 1},
            ],
            "classes": sample_classes,
        }, timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert data["grades"]["good"] == 1
        assert data["grades"]["review"] == 1

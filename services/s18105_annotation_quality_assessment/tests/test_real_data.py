"""End-to-end tests against 30 real fire/smoke images from fasdd_cv/val.

Each image has ground-truth YOLO labels. Asserts the service produces
semantically meaningful SAM3 verification (non-zero IoU, correct structure).
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
import requests

from conftest import SERVICE_URL, skip_no_sam3, skip_no_service

DATA_DIR = Path(__file__).resolve().parent / "data"
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"
MANIFEST = DATA_DIR / "manifest.json"

CLASSES = {"0": "fire", "1": "smoke"}
TEXT_PROMPTS = {"fire": "fire flames", "smoke": "smoke"}


def _load_manifest() -> list[dict]:
    if not MANIFEST.exists():
        pytest.skip(f"manifest missing: {MANIFEST}")
    return json.loads(MANIFEST.read_text())["items"]


def _load(item: dict) -> tuple[str, list[str]]:
    img_b64 = base64.b64encode((IMAGES_DIR / item["image"]).read_bytes()).decode()
    labels = (LABELS_DIR / (Path(item["image"]).stem + ".txt")).read_text().strip().splitlines()
    return img_b64, labels


ITEMS = _load_manifest() if MANIFEST.exists() else []


class TestRealData:

    @skip_no_service
    def test_dataset_size(self):
        assert len(ITEMS) >= 30, f"expected >= 30 real samples, got {len(ITEMS)}"

    @skip_no_service
    @pytest.mark.parametrize("item", ITEMS, ids=lambda it: it["image"])
    def test_validate_real(self, item):
        img, labels = _load(item)
        resp = requests.post(
            f"{SERVICE_URL}/validate",
            json={"image": img, "labels": labels, "label_format": "yolo", "classes": CLASSES},
            timeout=30,
        )
        assert resp.status_code == 200, resp.text
        d = resp.json()
        assert d["num_annotations"] == len(labels)
        assert 0.0 <= d["quality_score"] <= 1.0
        assert d["grade"] in ("good", "review", "bad")

    @skip_no_service
    @skip_no_sam3
    @pytest.mark.parametrize("item", ITEMS, ids=lambda it: it["image"])
    def test_verify_real(self, item):
        img, labels = _load(item)
        resp = requests.post(
            f"{SERVICE_URL}/verify",
            json={
                "image": img,
                "labels": labels,
                "label_format": "yolo",
                "classes": CLASSES,
                "text_prompts": TEXT_PROMPTS,
            },
            timeout=180,
        )
        assert resp.status_code == 200, resp.text
        d = resp.json()
        sam3 = d["sam3_verification"]
        assert len(sam3["box_ious"]) == len(labels)
        assert all(0.0 <= x <= 1.0 for x in sam3["box_ious"])
        assert d["num_annotations"] == len(labels)
        assert 0.0 <= d["quality_score"] <= 1.0

    @skip_no_service
    @skip_no_sam3
    def test_verify_aggregate_iou(self):
        """Across all 30 samples, mean box IoU must clearly exceed 0 — proves
        SAM3 is actually finding the annotated objects, not returning noise."""
        total_iou = 0.0
        total_anns = 0
        for item in ITEMS:
            img, labels = _load(item)
            resp = requests.post(
                f"{SERVICE_URL}/verify",
                json={
                    "image": img,
                    "labels": labels,
                    "label_format": "yolo",
                    "classes": CLASSES,
                    "text_prompts": TEXT_PROMPTS,
                },
                timeout=180,
            )
            assert resp.status_code == 200
            ious = resp.json()["sam3_verification"]["box_ious"]
            total_iou += sum(ious)
            total_anns += len(ious)
        mean_iou = total_iou / max(total_anns, 1)
        print(f"\naggregate mean box IoU over {total_anns} anns = {mean_iou:.3f}")
        assert mean_iou > 0.3, f"mean IoU {mean_iou:.3f} too low — SAM3 not matching GT"

    @skip_no_service
    def test_fix_real(self):
        """Validate → fix flow on a real sample with an injected out-of-bounds label."""
        img, labels = _load(ITEMS[0])
        labels = labels + ["0 1.05 0.5 0.3 0.3"]
        val = requests.post(
            f"{SERVICE_URL}/validate",
            json={"image": img, "labels": labels, "label_format": "yolo", "classes": CLASSES},
            timeout=30,
        ).json()
        assert val["num_issues"] >= 1
        fix = requests.post(
            f"{SERVICE_URL}/fix",
            json={
                "labels": labels,
                "label_format": "yolo",
                "classes": CLASSES,
                "issues": val["issues"],
                "suggested_fixes": val["suggested_fixes"],
            },
            timeout=30,
        )
        assert fix.status_code == 200
        d = fix.json()
        assert d["num_annotations_before"] == len(labels)
        assert d["num_applied"] >= 1

    @skip_no_service
    @skip_no_sam3
    def test_batch_job_real(self):
        """Submit a batch job with 5 real images, poll to completion."""
        import time
        batch = []
        for item in ITEMS[:5]:
            img, labels = _load(item)
            batch.append({"image": img, "labels": labels, "filename": item["image"]})
        resp = requests.post(
            f"{SERVICE_URL}/jobs",
            json={
                "images": batch,
                "label_format": "yolo",
                "classes": CLASSES,
                "text_prompts": TEXT_PROMPTS,
                "mode": "verify",
            },
            timeout=30,
        )
        assert resp.status_code == 200, resp.text
        job_id = resp.json()["job_id"]

        deadline = time.time() + 240
        while time.time() < deadline:
            status = requests.get(f"{SERVICE_URL}/jobs/{job_id}", timeout=10).json()
            if status["status"] in ("completed", "failed", "cancelled"):
                break
            time.sleep(2)
        assert status["status"] == "completed", status
        assert len(status.get("results", [])) == 5

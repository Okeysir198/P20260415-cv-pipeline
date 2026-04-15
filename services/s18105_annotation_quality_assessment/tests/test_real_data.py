"""End-to-end tests on real images from multiple use cases.

Each use case lives under tests/data/<use_case>/ with images/, labels/ (YOLO),
and manifest.json (classes + text_prompts + items). Asserts structural
validation and SAM3-backed verification produce semantically meaningful
results across diverse domains (fire, helmet, mask, glove, phone).
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
import requests

from conftest import SERVICE_URL, skip_no_sam3, skip_no_service

DATA_DIR = Path(__file__).resolve().parent / "data"


def _discover() -> list[dict]:
    cases = []
    if not DATA_DIR.exists():
        return cases
    for sub in sorted(DATA_DIR.iterdir()):
        m = sub / "manifest.json"
        if not m.exists():
            continue
        info = json.loads(m.read_text())
        info["base"] = sub
        cases.append(info)
    return cases


USE_CASES = _discover()
ALL_ITEMS = [(uc, it) for uc in USE_CASES for it in uc["items"]]


def _load(uc: dict, item: dict) -> tuple[str, list[str]]:
    base = uc["base"]
    img_b64 = base64.b64encode((base / "images" / item["image"]).read_bytes()).decode()
    labels = (base / "labels" / (Path(item["image"]).stem + ".txt")).read_text().strip().splitlines()
    return img_b64, labels


def _id(uc_item):
    uc, it = uc_item
    return f"{uc['use_case']}/{it['image']}"


class TestRealData:

    @skip_no_service
    def test_usecase_coverage(self):
        names = {uc["use_case"] for uc in USE_CASES}
        assert {"fire", "helmet", "mask", "glove", "phone", "safety_shoe", "zone"}.issubset(names)
        for uc in USE_CASES:
            assert len(uc["items"]) >= 10, f"{uc['use_case']} has {len(uc['items'])} < 10"

    @skip_no_service
    @pytest.mark.parametrize("uc_item", ALL_ITEMS, ids=_id)
    def test_validate_real(self, uc_item):
        uc, item = uc_item
        img, labels = _load(uc, item)
        resp = requests.post(
            f"{SERVICE_URL}/validate",
            json={"image": img, "labels": labels, "label_format": "yolo", "classes": uc["classes"]},
            timeout=30,
        )
        assert resp.status_code == 200, resp.text
        d = resp.json()
        assert d["num_annotations"] == len(labels)
        assert 0.0 <= d["quality_score"] <= 1.0

    @skip_no_service
    @skip_no_sam3
    @pytest.mark.parametrize("uc_item", ALL_ITEMS, ids=_id)
    def test_verify_real(self, uc_item):
        uc, item = uc_item
        img, labels = _load(uc, item)
        resp = requests.post(
            f"{SERVICE_URL}/verify",
            json={
                "image": img,
                "labels": labels,
                "label_format": "yolo",
                "classes": uc["classes"],
                "text_prompts": uc.get("text_prompts", {}),
            },
            timeout=180,
        )
        assert resp.status_code == 200, resp.text
        d = resp.json()
        assert len(d["sam3_verification"]["box_ious"]) == len(labels)

    @skip_no_service
    @skip_no_sam3
    @pytest.mark.parametrize("uc", USE_CASES, ids=lambda u: u["use_case"])
    def test_verify_aggregate_iou_per_usecase(self, uc):
        """Per use case, mean box IoU across its 10+ samples should clearly exceed 0."""
        total, count = 0.0, 0
        for item in uc["items"]:
            img, labels = _load(uc, item)
            resp = requests.post(
                f"{SERVICE_URL}/verify",
                json={
                    "image": img,
                    "labels": labels,
                    "label_format": "yolo",
                    "classes": uc["classes"],
                    "text_prompts": uc.get("text_prompts", {}),
                },
                timeout=180,
            )
            assert resp.status_code == 200
            ious = resp.json()["sam3_verification"]["box_ious"]
            total += sum(ious)
            count += len(ious)
        mean = total / max(count, 1)
        print(f"\n[{uc['use_case']}] mean box IoU over {count} anns = {mean:.3f}")
        assert mean > 0.2, f"{uc['use_case']} mean IoU {mean:.3f} too low"

    @skip_no_service
    def test_fix_real(self):
        """Validate → fix flow on a real sample with an injected out-of-bounds label."""
        uc = USE_CASES[0]
        item = uc["items"][0]
        img, labels = _load(uc, item)
        labels = labels + ["0 1.05 0.5 0.3 0.3"]
        val = requests.post(
            f"{SERVICE_URL}/validate",
            json={"image": img, "labels": labels, "label_format": "yolo", "classes": uc["classes"]},
            timeout=30,
        ).json()
        assert val["num_issues"] >= 1
        fix = requests.post(
            f"{SERVICE_URL}/fix",
            json={
                "labels": labels,
                "label_format": "yolo",
                "classes": uc["classes"],
                "issues": val["issues"],
                "suggested_fixes": val["suggested_fixes"],
            },
            timeout=30,
        )
        assert fix.status_code == 200
        assert fix.json()["num_applied"] >= 1

    @skip_no_service
    @skip_no_sam3
    def test_batch_job_real(self):
        """Mixed-use-case batch job: one image per use case."""
        import time
        uc = USE_CASES[0]  # use one use case's classes for a coherent batch
        batch = []
        for item in uc["items"][:5]:
            img, labels = _load(uc, item)
            batch.append({"image": img, "labels": labels, "filename": item["image"]})
        resp = requests.post(
            f"{SERVICE_URL}/jobs",
            json={
                "images": batch,
                "label_format": "yolo",
                "classes": uc["classes"],
                "text_prompts": uc.get("text_prompts", {}),
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

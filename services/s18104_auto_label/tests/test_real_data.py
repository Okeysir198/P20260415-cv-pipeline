"""End-to-end auto_label tests on 30 real fire/smoke images (fasdd_cv/val).

Asserts auto_label's /annotate endpoint (text mode) detects the GT classes
with reasonable IoU against ground-truth YOLO labels.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
import requests

from conftest import SERVICE_URL, skip_no_service

REAL_DIR = Path(__file__).resolve().parent / "data" / "real"
MANIFEST = REAL_DIR / "manifest.json"
CLASSES = {"0": "fire", "1": "smoke"}


def _load_manifest() -> list[dict]:
    if not MANIFEST.exists():
        pytest.skip(f"missing {MANIFEST}")
    return json.loads(MANIFEST.read_text())["items"]


def _load(item: dict):
    img_path = REAL_DIR / "images" / item["image"]
    lbl_path = REAL_DIR / "labels" / (Path(item["image"]).stem + ".txt")
    img_b64 = base64.b64encode(img_path.read_bytes()).decode()
    lines = lbl_path.read_text().strip().splitlines()
    w, h = item["width"], item["height"]
    anns = []
    for line in lines:
        c, cx, cy, bw, bh = line.split()
        c, cx, cy, bw, bh = int(c), float(cx), float(cy), float(bw), float(bh)
        anns.append({
            "class_id": c,
            "bbox_xyxy": [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h],
        })
    return img_b64, anns


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


ITEMS = _load_manifest() if MANIFEST.exists() else []


class TestRealData:

    @skip_no_service
    def test_dataset_size(self):
        assert len(ITEMS) >= 30

    @skip_no_service
    @pytest.mark.parametrize("item", ITEMS, ids=lambda it: it["image"])
    def test_annotate_text_real(self, item):
        img, anns = _load(item)
        resp = requests.post(
            f"{SERVICE_URL}/annotate",
            json={
                "image": img,
                "classes": CLASSES,
                "mode": "text",
                "confidence_threshold": 0.3,
                "output_format": "yolo",
            },
            timeout=120,
        )
        assert resp.status_code == 200, resp.text
        d = resp.json()
        assert d["num_detections"] == len(d["detections"])
        assert d["image_width"] == item["width"]
        assert d["image_height"] == item["height"]

    @skip_no_service
    def test_annotate_aggregate_iou(self):
        """Aggregate IoU of auto_label detections vs GT across 30 images."""
        total_iou = 0.0
        matched = 0
        total_anns = 0
        no_det = 0
        for item in ITEMS:
            img, anns = _load(item)
            resp = requests.post(
                f"{SERVICE_URL}/annotate",
                json={
                    "image": img,
                    "classes": CLASSES,
                    "mode": "text",
                    "confidence_threshold": 0.3,
                    "output_format": "yolo",
                },
                timeout=120,
            )
            assert resp.status_code == 200
            dets = resp.json().get("detections", [])
            if not dets:
                no_det += 1
            for ann in anns:
                total_anns += 1
                best = 0.0
                for det in dets:
                    if det.get("class_id") != ann["class_id"]:
                        continue
                    best = max(best, _iou(ann["bbox_xyxy"], det["bbox_xyxy"]))
                if best > 0.2:
                    matched += 1
                total_iou += best
        mean_iou = total_iou / max(total_anns, 1)
        print(
            f"\nauto_label: {matched}/{total_anns} GT matched IoU>0.2 "
            f"(class-aware), mean IoU={mean_iou:.3f}, images_with_no_det={no_det}/{len(ITEMS)}"
        )
        assert mean_iou > 0.2, f"mean IoU {mean_iou:.3f} too low"
        assert matched / total_anns > 0.4

    @skip_no_service
    @pytest.mark.parametrize("fmt", ["coco", "yolo", "yolo_seg"])
    def test_annotate_formats_real(self, fmt):
        item = ITEMS[0]
        img, _ = _load(item)
        resp = requests.post(
            f"{SERVICE_URL}/annotate",
            json={
                "image": img,
                "classes": CLASSES,
                "mode": "text",
                "output_format": fmt,
                "include_masks": fmt == "yolo_seg",
            },
            timeout=120,
        )
        assert resp.status_code == 200, resp.text
        assert "formatted_output" in resp.json()

    @skip_no_service
    def test_batch_job_real(self):
        import time
        batch = []
        for item in ITEMS[:5]:
            img, _ = _load(item)
            batch.append({"image": img, "filename": item["image"]})
        resp = requests.post(
            f"{SERVICE_URL}/jobs",
            json={
                "images": batch,
                "classes": CLASSES,
                "mode": "text",
                "output_format": "yolo",
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

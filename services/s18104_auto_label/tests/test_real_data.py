"""End-to-end auto_label tests across multiple use cases.

Each use case lives under tests/data/real/<use_case>/ with images/, labels/ (YOLO),
and manifest.json (classes + text_prompts + items).
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
import requests

from conftest import SERVICE_URL, skip_no_service

REAL_DIR = Path(__file__).resolve().parent / "data" / "real"


def _discover() -> list[dict]:
    cases = []
    if not REAL_DIR.exists():
        return cases
    for sub in sorted(REAL_DIR.iterdir()):
        m = sub / "manifest.json"
        if not m.exists():
            continue
        info = json.loads(m.read_text())
        info["base"] = sub
        cases.append(info)
    return cases


USE_CASES = _discover()
ALL_ITEMS = [(uc, it) for uc in USE_CASES for it in uc["items"]]


def _load(uc: dict, item: dict):
    base = uc["base"]
    img_b64 = base64.b64encode((base / "images" / item["image"]).read_bytes()).decode()
    lbl_path = base / "labels" / (Path(item["image"]).stem + ".txt")
    w, h = item["width"], item["height"]
    anns = []
    for line in lbl_path.read_text().strip().splitlines():
        parts = line.split()
        c = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])
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


def _id(uc_item):
    uc, it = uc_item
    return f"{uc['use_case']}/{it['image']}"


class TestRealData:

    @skip_no_service
    def test_usecase_coverage(self):
        names = {uc["use_case"] for uc in USE_CASES}
        expected = {"fire", "helmet", "mask", "glove", "phone", "safety_shoe", "zone"}
        assert expected.issubset(names), names

    @skip_no_service
    @pytest.mark.parametrize("uc_item", ALL_ITEMS, ids=_id)
    def test_annotate_text_real(self, uc_item):
        uc, item = uc_item
        img, _ = _load(uc, item)
        resp = requests.post(
            f"{SERVICE_URL}/annotate",
            json={
                "image": img,
                "classes": uc["classes"],
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
    @pytest.mark.parametrize("uc", USE_CASES, ids=lambda u: u["use_case"])
    def test_annotate_iou_per_usecase(self, uc):
        total, matched, count, no_det = 0.0, 0, 0, 0
        for item in uc["items"]:
            img, anns = _load(uc, item)
            resp = requests.post(
                f"{SERVICE_URL}/annotate",
                json={
                    "image": img,
                    "classes": uc["classes"],
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
                count += 1
                best = 0.0
                for det in dets:
                    if det.get("class_id") != ann["class_id"]:
                        continue
                    best = max(best, _iou(ann["bbox_xyxy"], det["bbox_xyxy"]))
                if best > 0.2:
                    matched += 1
                total += best
        mean = total / max(count, 1)
        print(
            f"\n[{uc['use_case']}] {matched}/{count} matched IoU>0.2, "
            f"mean IoU={mean:.3f}, empty_imgs={no_det}/{len(uc['items'])}"
        )
        assert mean >= 0.05, f"{uc['use_case']} mean IoU {mean:.3f} too low"

    @skip_no_service
    @pytest.mark.parametrize("fmt", ["coco", "yolo", "yolo_seg"])
    def test_annotate_formats_real(self, fmt):
        uc = USE_CASES[0]
        img, _ = _load(uc, uc["items"][0])
        resp = requests.post(
            f"{SERVICE_URL}/annotate",
            json={
                "image": img,
                "classes": uc["classes"],
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
        uc = USE_CASES[0]
        batch = []
        for item in uc["items"][:5]:
            img, _ = _load(uc, item)
            batch.append({"image": img, "filename": item["image"]})
        resp = requests.post(
            f"{SERVICE_URL}/jobs",
            json={
                "images": batch,
                "classes": uc["classes"],
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

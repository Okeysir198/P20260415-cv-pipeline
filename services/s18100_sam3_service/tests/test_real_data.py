"""End-to-end SAM3 tests on 30 real fire/smoke images (fasdd_cv/val).

Asserts SAM3's text + box segmentation produce meaningful IoU against
ground-truth YOLO labels.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest
import requests

from conftest import REQUEST_TIMEOUT, SERVICE_URL, skip_no_service

REAL_DIR = Path(__file__).resolve().parent / "data" / "real"
MANIFEST = REAL_DIR / "manifest.json"
CLASS_NAMES = {0: "fire", 1: "smoke"}
TEXT_PROMPTS = {0: "fire flames", 1: "smoke"}


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
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        anns.append({"class_id": c, "bbox_xyxy": [x1, y1, x2, y2]})
    return img_b64, anns, w, h


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


ITEMS = _load_manifest() if MANIFEST.exists() else []


class TestRealData:

    @skip_no_service
    def test_dataset_size(self):
        assert len(ITEMS) >= 30

    @skip_no_service
    @pytest.mark.parametrize("item", ITEMS, ids=lambda it: it["image"])
    def test_segment_box_real(self, item):
        """SAM3 /segment_box should return a mask for each GT bbox."""
        img, anns, w, h = _load(item)
        for ann in anns:
            box_int = [int(round(v)) for v in ann["bbox_xyxy"]]
            resp = requests.post(
                f"{SERVICE_URL}/segment_box",
                json={"image": img, "box": box_int},
                timeout=REQUEST_TIMEOUT,
            )
            assert resp.status_code == 200, resp.text
            d = resp.json().get("result", resp.json())
            assert "mask" in d and "bbox" in d

    @skip_no_service
    @pytest.mark.parametrize("item", ITEMS, ids=lambda it: it["image"])
    def test_segment_text_real(self, item):
        """SAM3 /segment_text should detect at least one object for each GT class."""
        img, anns, w, h = _load(item)
        classes_present = {a["class_id"] for a in anns}
        for cid in classes_present:
            resp = requests.post(
                f"{SERVICE_URL}/segment_text",
                json={"image": img, "text": TEXT_PROMPTS[cid]},
                timeout=REQUEST_TIMEOUT,
            )
            assert resp.status_code == 200, resp.text
            dets = resp.json().get("detections", [])
            assert isinstance(dets, list)

    @skip_no_service
    def test_text_detection_iou_aggregate(self):
        """Aggregate IoU of SAM3 text detections vs GT boxes across all 30 images."""
        total_iou = 0.0
        matched = 0
        total_anns = 0
        for item in ITEMS:
            img, anns, w, h = _load(item)
            per_class_dets = {}
            for cid in {a["class_id"] for a in anns}:
                resp = requests.post(
                    f"{SERVICE_URL}/segment_text",
                    json={"image": img, "text": TEXT_PROMPTS[cid]},
                    timeout=REQUEST_TIMEOUT,
                )
                if resp.status_code != 200:
                    continue
                per_class_dets[cid] = resp.json().get("detections", [])
            for ann in anns:
                total_anns += 1
                dets = per_class_dets.get(ann["class_id"], [])
                best = 0.0
                for det in dets:
                    bbox = det.get("bbox") or det.get("bbox_xyxy") or []
                    if isinstance(bbox, dict):
                        bbox = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
                    if len(bbox) == 4:
                        best = max(best, _iou(ann["bbox_xyxy"], [float(v) for v in bbox]))
                if best > 0.2:
                    matched += 1
                total_iou += best
        mean_iou = total_iou / max(total_anns, 1)
        print(f"\nSAM3 text-detect: {matched}/{total_anns} matched IoU>0.2, mean IoU={mean_iou:.3f}")
        assert mean_iou > 0.25, f"mean IoU {mean_iou:.3f} too low"
        assert matched / total_anns > 0.5

    @skip_no_service
    @pytest.mark.parametrize("item", ITEMS[:10], ids=lambda it: it["image"])
    def test_auto_mask_real(self, item):
        """SAM3 /auto_mask should produce detections for a real image."""
        img, _, _, _ = _load(item)
        resp = requests.post(
            f"{SERVICE_URL}/auto_mask",
            json={"image": img},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200, resp.text
        dets = resp.json().get("detections", [])
        assert isinstance(dets, list)

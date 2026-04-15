"""End-to-end SAM3 tests across multiple use cases (fire, helmet, mask, glove, phone).

Each use case lives under tests/data/real/<use_case>/ with images/, labels/ (YOLO),
and manifest.json (classes + text_prompts + items).
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
import requests

from conftest import REQUEST_TIMEOUT, SERVICE_URL, skip_no_service

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
    return img_b64, anns, w, h


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
        assert {"fire", "helmet", "mask", "glove", "phone", "safety_shoe", "zone"}.issubset(names)

    @skip_no_service
    @pytest.mark.parametrize("uc_item", ALL_ITEMS, ids=_id)
    def test_segment_box_real(self, uc_item):
        uc, item = uc_item
        img, anns, _, _ = _load(uc, item)
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
    @pytest.mark.parametrize("uc_item", ALL_ITEMS, ids=_id)
    def test_segment_text_real(self, uc_item):
        uc, item = uc_item
        img, anns, _, _ = _load(uc, item)
        prompts = uc.get("text_prompts", {})
        classes = uc["classes"]
        for cid in {a["class_id"] for a in anns}:
            text = prompts.get(classes[str(cid)]) or classes[str(cid)]
            resp = requests.post(
                f"{SERVICE_URL}/segment_text",
                json={"image": img, "text": text},
                timeout=REQUEST_TIMEOUT,
            )
            assert resp.status_code == 200, resp.text
            assert isinstance(resp.json().get("detections", []), list)

    @skip_no_service
    @pytest.mark.parametrize("uc", USE_CASES, ids=lambda u: u["use_case"])
    def test_text_iou_per_usecase(self, uc):
        """Per use case, text-prompt detection IoU vs GT should clearly exceed 0."""
        prompts = uc.get("text_prompts", {})
        classes = uc["classes"]
        total, matched, count = 0.0, 0, 0
        for item in uc["items"]:
            img, anns, _, _ = _load(uc, item)
            per_class = {}
            for cid in {a["class_id"] for a in anns}:
                text = prompts.get(classes[str(cid)]) or classes[str(cid)]
                r = requests.post(
                    f"{SERVICE_URL}/segment_text",
                    json={"image": img, "text": text},
                    timeout=REQUEST_TIMEOUT,
                )
                if r.status_code == 200:
                    per_class[cid] = r.json().get("detections", [])
            for ann in anns:
                count += 1
                best = 0.0
                for det in per_class.get(ann["class_id"], []):
                    b = det.get("bbox") or []
                    if isinstance(b, dict):
                        b = [b["x1"], b["y1"], b["x2"], b["y2"]]
                    if len(b) == 4:
                        best = max(best, _iou(ann["bbox_xyxy"], [float(v) for v in b]))
                if best > 0.2:
                    matched += 1
                total += best
        mean = total / max(count, 1)
        print(f"\n[{uc['use_case']}] {matched}/{count} matched IoU>0.2, mean IoU={mean:.3f}")
        assert mean > 0.1, f"{uc['use_case']} mean IoU {mean:.3f} too low"

    @skip_no_service
    @pytest.mark.parametrize("uc", USE_CASES, ids=lambda u: u["use_case"])
    def test_auto_mask_real(self, uc):
        img, _, _, _ = _load(uc, uc["items"][0])
        resp = requests.post(
            f"{SERVICE_URL}/auto_mask",
            json={"image": img},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200, resp.text
        assert isinstance(resp.json().get("detections", []), list)

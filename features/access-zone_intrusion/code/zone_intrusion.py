"""Production inference class for zone intrusion detection.

Uses an Ultralytics YOLO model (pretrained on COCO) to detect persons,
then tests each detection centroid against configured polygon zones.

Run smoke test:
    uv run features/access-zone_intrusion/code/zone_intrusion.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from utils.config import load_config  # noqa: E402

FEATURE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = FEATURE_DIR / "configs" / "10_inference.yaml"


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #

@dataclass
class PersonDetection:
    box_xyxy: np.ndarray  # (4,) float32
    score: float
    in_zone: bool
    zone_id: str


@dataclass
class ZoneResult:
    intruding: bool
    detections: list[PersonDetection]
    alert_zones: list[str]  # zone IDs that have at least one intrusion
    latency_ms: float


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #

class ZoneIntrusionDetector:
    def __init__(self, config_path: str | Path = DEFAULT_CONFIG) -> None:
        from ultralytics import YOLO

        cfg = load_config(config_path)
        config_dir = Path(config_path).resolve().parent

        weights_raw = cfg["model"]["weights"]
        weights_path = (config_dir / weights_raw).resolve()
        self._model = YOLO(str(weights_path))

        self._conf: float = cfg["model"].get("conf", 0.35)
        self._iou: float = cfg["model"].get("iou", 0.45)
        self._person_class_id: int = cfg["zone"].get("person_class_id", 0)

        # Parse named zones: list of {id, polygon} dicts
        raw_zones = cfg["zone"].get("zones", [])
        self._zones: list[dict] = raw_zones  # [{id, polygon: [[x,y], ...]}]

    # ---------------------------------------------------------------------- #
    # Core API
    # ---------------------------------------------------------------------- #

    def detect(self, image_bgr: np.ndarray) -> ZoneResult:
        h, w = image_bgr.shape[:2]

        t0 = time.perf_counter()
        results = self._model(
            image_bgr,
            conf=self._conf,
            iou=self._iou,
            classes=[self._person_class_id],
            verbose=False,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        detections: list[PersonDetection] = []
        alert_zones: set[str] = set()

        boxes = results[0].boxes
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()

            for box, score in zip(xyxy, scores):
                in_zone, zone_id = self._check_zones(box, w, h)
                detections.append(PersonDetection(
                    box_xyxy=box.astype(np.float32),
                    score=float(score),
                    in_zone=in_zone,
                    zone_id=zone_id,
                ))
                if in_zone:
                    alert_zones.add(zone_id)

        return ZoneResult(
            intruding=bool(alert_zones),
            detections=detections,
            alert_zones=sorted(alert_zones),
            latency_ms=round(latency_ms, 1),
        )

    def draw(self, image_bgr: np.ndarray, result: ZoneResult) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        vis = image_bgr.copy()

        for zone in self._zones:
            poly_px = _poly_to_pixel(zone["polygon"], w, h)
            cv2.polylines(vis, [poly_px], True, (0, 255, 255), 2)
            overlay = vis.copy()
            cv2.fillPoly(overlay, [poly_px], (0, 255, 255))
            vis = cv2.addWeighted(overlay, 0.15, vis, 0.85, 0)

        for det in result.detections:
            x1, y1, x2, y2 = det.box_xyxy.astype(int)
            color = (0, 0, 255) if det.in_zone else (0, 255, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                vis, f"person {det.score:.2f}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )

        verdict = "INTRUSION" if result.intruding else "CLEAR"
        verdict_color = (0, 0, 255) if result.intruding else (0, 200, 0)
        cv2.putText(
            vis, verdict,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, verdict_color, 2, cv2.LINE_AA,
        )

        return vis

    # ---------------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------------- #

    def _check_zones(
        self, box_xyxy: np.ndarray, w: int, h: int
    ) -> tuple[bool, str]:
        """Return (in_zone, zone_id) for the first zone that contains centroid."""
        cx = float((box_xyxy[0] + box_xyxy[2]) / 2)
        cy = float((box_xyxy[1] + box_xyxy[3]) / 2)

        for zone in self._zones:
            poly_px = _poly_to_pixel(zone["polygon"], w, h)
            hit = cv2.pointPolygonTest(poly_px.astype(np.int32), (cx, cy), False) >= 0
            if hit:
                return True, zone["id"]

        return False, ""


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #

def _poly_to_pixel(poly_norm: list[list[float]], w: int, h: int) -> np.ndarray:
    return np.array([[p[0] * w, p[1] * h] for p in poly_norm], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Smoke test CLI
# --------------------------------------------------------------------------- #

def _smoke_test() -> None:
    zones_json = FEATURE_DIR / "samples" / "zones.json"
    zones_cfg = json.loads(zones_json.read_text())

    detector = ZoneIntrusionDetector(DEFAULT_CONFIG)

    predict_dir = FEATURE_DIR / "predict" / "zone_intrusion"
    predict_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {}
    correct = 0
    total_with_gt = 0

    for img_name, meta in zones_cfg["samples"].items():
        img_path = FEATURE_DIR / "samples" / img_name
        if not img_path.exists():
            print(f"  [skip] missing {img_path.name}")
            continue

        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]

        # Override detector zones with per-sample polygon from zones.json
        detector._zones = [{"id": "sample_zone", "polygon": meta["polygon"]}]

        result = detector.detect(image)
        annotated = detector.draw(image, result)
        cv2.imwrite(str(predict_dir / img_name), annotated)

        expected = meta["expected_intrusion"]
        predicted = result.intruding
        match = None
        if expected is not None:
            match = predicted == expected
            total_with_gt += 1
            if match:
                correct += 1

        results[img_name] = {
            "scene": meta["scene"],
            "expected_intrusion": expected,
            "predicted_intrusion": predicted,
            "correct": match,
            "n_persons": len(result.detections),
            "alert_zones": result.alert_zones,
            "latency_ms": result.latency_ms,
        }
        status = "OK" if match else ("WRONG" if match is False else "SKIP(null)")
        print(
            f"  {img_name}: persons={len(result.detections)} "
            f"verdict={'INTRUSION' if predicted else 'CLEAR'} "
            f"({result.latency_ms:.0f} ms) [{status}]"
        )

    eval_dir = FEATURE_DIR / "eval"
    eval_dir.mkdir(exist_ok=True)
    out_json = eval_dir / "smoke_test_results.json"
    out_json.write_text(json.dumps(results, indent=2))

    print(f"\nAccuracy: {correct}/{total_with_gt} correct intrusion verdicts")
    print(f"Results written to {out_json}")
    print(f"Visualizations written to {predict_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zone intrusion detector")
    parser.add_argument("--smoke-test", action="store_true", help="Run on all samples/")
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG), help="Path to 10_inference.yaml"
    )
    args = parser.parse_args()

    if args.smoke_test:
        _smoke_test()
    else:
        parser.print_help()

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
import supervision as sv
from matplotlib.path import Path as _MplPath

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from utils.config import load_config  # noqa: E402
from utils.viz import (  # noqa: E402
    VizStyle,
    annotate_detections,
    annotate_polygons,
    classification_banner,
)

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
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
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

            # Vectorized point-in-polygon: (N, Z) inclusion matrix.
            centroids = np.stack(
                [(xyxy[:, 0] + xyxy[:, 2]) * 0.5, (xyxy[:, 1] + xyxy[:, 3]) * 0.5],
                axis=1,
            )
            inclusion = self._zones_contain(centroids, w, h)  # (N, Z) bool

            for i, (box, score) in enumerate(zip(xyxy, scores)):
                hits = inclusion[i]
                if hits.any():
                    zone_idx = int(np.argmax(hits))  # first matching zone
                    in_zone, zone_id = True, self._zones[zone_idx]["id"]
                    alert_zones.add(zone_id)
                else:
                    in_zone, zone_id = False, ""
                detections.append(PersonDetection(
                    box_xyxy=box.astype(np.float32),
                    score=float(score),
                    in_zone=in_zone,
                    zone_id=zone_id,
                ))

        return ZoneResult(
            intruding=bool(alert_zones),
            detections=detections,
            alert_zones=sorted(alert_zones),
            latency_ms=round(latency_ms, 1),
        )

    def draw(self, image_bgr: np.ndarray, result: ZoneResult) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        vis_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        zone_style = VizStyle(zone_fill_alpha=0.15, zone_outline_thickness=2)
        yellow = sv.Color(r=255, g=255, b=0)  # (0,255,255) BGR = yellow RGB
        for zone in self._zones:
            poly_px = _poly_to_pixel(zone["polygon"], w, h).astype(np.int32)
            vis_rgb = annotate_polygons(
                vis_rgb,
                polygons=[poly_px],
                labels=[zone["id"]],
                style=zone_style,
                color=yellow,
            )

        # Split detections by in/out-of-zone so each group can use a distinct color.
        for in_zone_flag, color_rgb in (
            (True, sv.Color(r=255, g=0, b=0)),     # red  (was BGR (0,0,255))
            (False, sv.Color(r=0, g=255, b=0)),    # green (was BGR (0,255,0))
        ):
            group = [d for d in result.detections if d.in_zone == in_zone_flag]
            if not group:
                continue
            xyxy = np.stack([d.box_xyxy for d in group], axis=0).astype(np.float32)
            scores = np.array([d.score for d in group], dtype=np.float32)
            dets = sv.Detections(
                xyxy=xyxy,
                confidence=scores,
                class_id=np.zeros(len(group), dtype=int),
            )
            labels = [f"person {d.score:.2f}" for d in group]
            vis_rgb = annotate_detections(vis_rgb, dets, labels=labels, color=color_rgb)

        verdict = "INTRUSION" if result.intruding else "CLEAR"
        banner_bg = (231, 76, 60) if result.intruding else (39, 174, 96)  # red / green RGB
        vis_rgb = classification_banner(
            vis_rgb, verdict, position="top", bg_color_rgb=banner_bg
        )

        return cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)

    # ---------------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------------- #

    def _zones_contain(
        self, points: np.ndarray, w: int, h: int
    ) -> np.ndarray:
        """Vectorized point-in-polygon over all zones.

        Args:
            points: (N, 2) float array of (x, y) pixel coords.
            w, h: frame dimensions for normalized-polygon scaling.

        Returns:
            (N, Z) bool array; cell (i, j) True iff point i lies in zone j.
        """
        if points.size == 0 or not self._zones:
            return np.zeros((len(points), len(self._zones)), dtype=bool)

        out = np.zeros((len(points), len(self._zones)), dtype=bool)
        for j, zone in enumerate(self._zones):
            poly_px = _poly_to_pixel(zone["polygon"], w, h)
            out[:, j] = _MplPath(poly_px).contains_points(points)
        return out


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

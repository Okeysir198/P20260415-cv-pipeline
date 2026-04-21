"""YOLOX-M fire/smoke inference via core/p10_inference.DetectionPredictor.

YOLOX checkpoints already save in the `{"config": ..., "model": sd}` format
the predictor expects — no repack needed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/home/ct-admin/Documents/Langgraph/TEST/ai")
sys.path.insert(0, str(REPO))

import cv2

from core.p10_inference.predictor import DetectionPredictor
from core.p10_inference.supervision_bridge import (
    annotate_frame,
    build_annotators,
    to_sv_detections,
)
from utils.config import load_config

RUN_DIR = REPO / "features/safety-fire_detection/runs/2026-04-21_141209_06_training"
CKPT = RUN_DIR / "best.pth"
DATA_CFG = RUN_DIR / "05_data.yaml"
SAMPLES = REPO / "features/safety-fire_detection/samples"
OUT = SAMPLES.parent / "samples_yolox_predictions"
OUT.mkdir(exist_ok=True)


def main() -> None:
    data_cfg = load_config(str(DATA_CFG))
    predictor = DetectionPredictor(
        model_path=CKPT,
        data_config=data_cfg,
        conf_threshold=0.03,
        iou_threshold=0.50,
    )
    class_names = predictor.class_names
    annotators = build_annotators()

    summary = []
    images = sorted(p for p in SAMPLES.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for path in images:
        img = cv2.imread(str(path))
        result = predictor.predict(img)
        sv_dets = to_sv_detections(result)
        annotated = annotate_frame(img, sv_dets, class_names, annotators)
        cv2.imwrite(str(OUT / path.name), annotated)

        dets = [
            {"class": str(n), "score": round(float(s), 3),
             "box": [round(float(v), 1) for v in b]}
            for n, s, b in zip(result["class_names"], result["scores"], result["boxes"])
        ]
        summary.append({"image": path.name, "detections": dets})
        tag = ", ".join(f"{d['class']}:{d['score']}" for d in dets) or "-"
        print(f"{path.name:30s}  {len(dets):2d} det  [{tag}]")

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {len(images)} annotated images + summary.json → {OUT}")


if __name__ == "__main__":
    main()

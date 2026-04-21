"""RT-DETRv2 fire/smoke inference via core/p10_inference.DetectionPredictor.

HF-Trainer checkpoints are a flat `pytorch_model.bin` with `hf_model.*` keys;
DetectionPredictor's `.pt` loader expects `{"config": ..., "model": state_dict}`.
Repack once into a `best_repacked.pt` alongside the checkpoint, then run the
canonical predictor.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/home/ct-admin/Documents/Langgraph/TEST/ai")
sys.path.insert(0, str(REPO))

import cv2
import numpy as np
import torch

from core.p10_inference.predictor import DetectionPredictor
from core.p10_inference.supervision_bridge import (
    annotate_frame,
    build_annotators,
    to_sv_detections,
)
from utils.config import load_config

RUN_DIR = REPO / "features/safety-fire_detection/runs/2026-04-21_174000_06_training"
HF_CKPT = RUN_DIR / "checkpoint-2930/pytorch_model.bin"
REPACKED = RUN_DIR / "checkpoint-2930/best_repacked_q100.pt"
TRAIN_CFG = RUN_DIR / "06_training.yaml"
DATA_CFG = RUN_DIR / "05_data.yaml"
SAMPLES = REPO / "features/safety-fire_detection/samples"
OUT = SAMPLES.parent / "samples_rtdetr_predictions"
OUT.mkdir(exist_ok=True)


def _ensure_repacked() -> Path:
    if REPACKED.exists():
        return REPACKED
    train_cfg = load_config(str(TRAIN_CFG))
    train_cfg["model"].pop("pretrained", None)
    sd = torch.load(HF_CKPT, map_location="cpu", weights_only=True)
    torch.save({"config": train_cfg, "model": sd}, REPACKED)
    print(f"Repacked HF checkpoint → {REPACKED}")
    return REPACKED


def main() -> None:
    ckpt = _ensure_repacked()
    data_cfg = load_config(str(DATA_CFG))
    # RT-DETRv2 preprocessor_config.json: do_normalize=false — model expects
    # raw [0,1] values, NOT ImageNet-normalized. DetectionPredictor always
    # normalizes unless we override mean/std.
    data_cfg["mean"] = [0.0, 0.0, 0.0]
    data_cfg["std"] = [1.0, 1.0, 1.0]

    predictor = DetectionPredictor(
        model_path=ckpt,
        data_config=data_cfg,
        conf_threshold=0.05,
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

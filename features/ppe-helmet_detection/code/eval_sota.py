#!/usr/bin/env python3
"""Evaluate pretrained SOTA models on ppe-helmet_detection samples.

Runs the top two candidate pretrained models against ``samples/`` and writes
annotated predictions to ``predict/<model_name>/``.

Model 1: YOLOS-tiny fine-tuned on Hard-Hat-Detection (Apache 2.0, HF).
Model 2: D-FINE-M Obj365->COCO (general person detector baseline).
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
FEATURE = Path(__file__).resolve().parents[1]
PRETRAINED = REPO / "pretrained" / "ppe-helmet_detection"
SAMPLES = FEATURE / "samples"
OUT = FEATURE / "predict"

# --- colors (BGR)
COLORS = {
    "hardhat": (0, 200, 0),
    "no-hardhat": (0, 0, 220),
    "person": (200, 150, 0),
    "_default": (180, 180, 0),
}


def draw(image: np.ndarray, boxes, labels, scores) -> np.ndarray:
    vis = image.copy()
    for (x1, y1, x2, y2), lab, sc in zip(boxes, labels, scores):
        color = COLORS.get(lab, COLORS["_default"])
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        text = f"{lab} {sc:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (int(x1), int(y1) - th - 4), (int(x1) + tw + 4, int(y1)), color, -1)
        cv2.putText(vis, text, (int(x1) + 2, int(y1) - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def save_predictions(img: Image.Image, boxes, labels, scores, out_path: Path) -> None:
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), draw(bgr, boxes, labels, scores))


def run_yolos(sample_paths: list[Path]) -> dict:
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    model_dir = PRETRAINED / "yolos-tiny-hardhat"
    proc = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForObjectDetection.from_pretrained(model_dir).eval()
    id2label = model.config.id2label

    out_dir = OUT / "yolos-tiny-hardhat"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, list] = {}
    for p in sample_paths:
        img = Image.open(p).convert("RGB")
        inputs = proc(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([img.size[::-1]])
        results = proc.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.5
        )[0]
        boxes = results["boxes"].cpu().numpy()
        labels = [id2label[int(x)] for x in results["labels"].cpu().numpy()]
        scores = results["scores"].cpu().numpy()
        save_predictions(img, boxes, labels, scores, out_dir / p.name)
        summary[p.name] = [
            {"label": l, "score": float(s), "box": [float(v) for v in b]}
            for l, s, b in zip(labels, scores, boxes)
        ]
        print(f"[yolos] {p.name}: {len(labels)} dets -> "
              + ", ".join(f"{l}:{s:.2f}" for l, s in zip(labels, scores)))
    return summary


def run_dfine(sample_paths: list[Path]) -> dict:
    """Best effort: D-FINE-M Obj365->COCO via HF Transformers."""
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    # Load from HF hub (weights already in pretrained dir won't auto-bind to arch here;
    # simpler to use the official HF checkpoint for inference).
    repo_id = "ustc-community/dfine-medium-coco"
    try:
        proc = AutoImageProcessor.from_pretrained(repo_id)
        model = AutoModelForObjectDetection.from_pretrained(repo_id).eval()
    except Exception as e:  # noqa: BLE001
        print(f"[dfine] load failed: {e}")
        return {}
    id2label = model.config.id2label

    out_dir = OUT / "dfine-medium-coco"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, list] = {}
    for p in sample_paths:
        img = Image.open(p).convert("RGB")
        inputs = proc(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([img.size[::-1]])
        results = proc.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.4
        )[0]
        boxes = results["boxes"].cpu().numpy()
        labels_all = [id2label[int(x)] for x in results["labels"].cpu().numpy()]
        scores_all = results["scores"].cpu().numpy()
        # Filter to person class (most relevant; D-FINE-COCO has no helmet class)
        keep = [i for i, l in enumerate(labels_all) if l == "person"]
        boxes = boxes[keep] if len(keep) else boxes[:0]
        labels = [labels_all[i] for i in keep]
        scores = scores_all[keep] if len(keep) else scores_all[:0]
        save_predictions(img, boxes, labels, scores, out_dir / p.name)
        summary[p.name] = [
            {"label": l, "score": float(s), "box": [float(v) for v in b]}
            for l, s, b in zip(labels, scores, boxes)
        ]
        print(f"[dfine] {p.name}: {len(labels)} person dets")
    return summary


def main() -> None:
    sample_paths = sorted(SAMPLES.glob("*.jpg")) + sorted(SAMPLES.glob("*.png"))
    if not sample_paths:
        print(f"No samples under {SAMPLES}")
        sys.exit(1)
    print(f"Running {len(sample_paths)} samples")

    print("\n=== Model 1: YOLOS-tiny Hard-Hat ===")
    run_yolos(sample_paths)

    print("\n=== Model 2: D-FINE-M COCO ===")
    run_dfine(sample_paths)

    print("\nDone. Outputs at", OUT)


if __name__ == "__main__":
    main()

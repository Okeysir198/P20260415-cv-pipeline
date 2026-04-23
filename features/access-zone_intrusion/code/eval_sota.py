"""Run SOTA pretrained person detectors on access-zone_intrusion samples.

Top-2 detectors per technical_study/access-zone_intrusion-sota.md:
    1. D-FINE-N (HF: ustc-community/dfine_n_coco) - primary recommendation
    2. YOLOX-Tiny (Megvii COCO .pth)              - fallback baseline

For each sample image we:
    * run the detector,
    * keep COCO 'person' class only (idx 0),
    * apply a polygon-zone test (centroid-in-polygon),
    * write a visualization (boxes + zone polygon + verdict) to
      .../predict/<model_name>/<image>.jpg,
    * accumulate per-sample results for the QUALITY_REPORT.

Run from project root (ai/):
    uv run python features/access-zone_intrusion/code/eval_sota.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import supervision as sv
import torch

REPO_AI = Path(__file__).resolve().parents[3]  # .../ai
sys.path.insert(0, str(REPO_AI))

from utils.viz import (  # noqa: E402
    VizStyle,
    annotate_detections,
    annotate_polygons,
    classification_banner,
)

FEATURE_DIR = Path(__file__).resolve().parents[1]
SAMPLES_DIR = FEATURE_DIR / "samples"
PREDICT_DIR = FEATURE_DIR / "predict"
ZONES_JSON = SAMPLES_DIR / "zones.json"

PERSON_CLASS = 0  # COCO class id for 'person'
CONF_THRESH = 0.35
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def polygon_to_pixel(poly_norm: list[list[float]], w: int, h: int) -> np.ndarray:
    return np.array([[p[0] * w, p[1] * h] for p in poly_norm], dtype=np.float32)


def centroid_in_polygon(box_xyxy: np.ndarray, poly_px: np.ndarray) -> bool:
    cx = float((box_xyxy[0] + box_xyxy[2]) / 2)
    cy = float((box_xyxy[1] + box_xyxy[3]) / 2)
    return cv2.pointPolygonTest(poly_px.astype(np.int32), (cx, cy), False) >= 0


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #
def load_dfine_n() -> tuple[Any, Any]:
    from transformers import AutoImageProcessor, DFineForObjectDetection

    repo = "ustc-community/dfine_n_coco"
    processor = AutoImageProcessor.from_pretrained(repo)
    model = DFineForObjectDetection.from_pretrained(repo).to(DEVICE).eval()
    return model, processor


def detect_dfine(model, processor, image_bgr: np.ndarray) -> list[tuple[np.ndarray, float]]:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image_bgr.shape[:2]], device=DEVICE)  # (H, W)
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=CONF_THRESH
    )[0]
    persons = []
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        if int(label) == PERSON_CLASS:
            persons.append((box.cpu().numpy(), float(score)))
    return persons


def load_yolox_tiny():
    from core.p06_models import build_model
    from core.p10_inference.predictor import _remap_megvii_state_dict

    cfg = {"model": {"arch": "yolox-tiny", "num_classes": 80, "input_size": [416, 416]}}
    model = build_model(cfg)
    ckpt_path = REPO_AI / "pretrained" / "access-zone_intrusion" / "yolox_tiny.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    try:
        model.load_state_dict(state)
    except RuntimeError:
        remapped = _remap_megvii_state_dict(state, set(model.state_dict().keys()))
        model.load_state_dict(remapped, strict=False)
    model.to(DEVICE).eval()
    return model


def detect_yolox(model, image_bgr: np.ndarray) -> list[tuple[np.ndarray, float]]:
    from core.p06_training.postprocess import postprocess

    h, w = image_bgr.shape[:2]
    in_h, in_w = 416, 416
    # YOLOX preprocess: BGR -> resize -> CHW (raw uint8 -> float, no norm)
    resized = cv2.resize(image_bgr, (in_w, in_h))
    chw = resized.astype(np.float32).transpose(2, 0, 1)
    x = torch.from_numpy(chw[np.newaxis, ...]).to(DEVICE)
    with torch.no_grad():
        raw = model(x)
    results = postprocess("yolox", model, raw, conf_threshold=CONF_THRESH, nms_threshold=0.45)[0]
    persons = []
    sx, sy = w / in_w, h / in_h
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        if int(label) == PERSON_CLASS:
            b = box.copy().astype(np.float32)
            b[[0, 2]] *= sx
            b[[1, 3]] *= sy
            b[[0, 2]] = np.clip(b[[0, 2]], 0, w)
            b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
            persons.append((b, float(score)))
    return persons


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def visualize(
    image_bgr: np.ndarray,
    detections: list[tuple[np.ndarray, float]],
    poly_px: np.ndarray,
    intrusion: bool,
    save_path: Path,
) -> None:
    vis_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    zone_style = VizStyle(zone_fill_alpha=0.15, zone_outline_thickness=2)
    yellow = sv.Color(r=255, g=255, b=0)
    vis_rgb = annotate_polygons(
        vis_rgb,
        polygons=[poly_px.astype(np.int32)],
        labels=["zone"],
        style=zone_style,
        color=yellow,
    )

    in_flags = [centroid_in_polygon(b, poly_px) for b, _ in detections]
    for in_zone_flag, color_rgb in (
        (True, sv.Color(r=255, g=0, b=0)),
        (False, sv.Color(r=0, g=255, b=0)),
    ):
        idxs = [i for i, f in enumerate(in_flags) if f == in_zone_flag]
        if not idxs:
            continue
        xyxy = np.stack([detections[i][0] for i in idxs], axis=0).astype(np.float32)
        scores = np.array([detections[i][1] for i in idxs], dtype=np.float32)
        dets_sv = sv.Detections(
            xyxy=xyxy,
            confidence=scores,
            class_id=np.zeros(len(idxs), dtype=int),
        )
        labels = [f"person {detections[i][1]:.2f}" for i in idxs]
        vis_rgb = annotate_detections(vis_rgb, dets_sv, labels=labels, color=color_rgb)

    verdict = "INTRUSION" if intrusion else "CLEAR"
    banner_bg = (231, 76, 60) if intrusion else (39, 174, 96)
    vis_rgb = classification_banner(
        vis_rgb, verdict, position="top", bg_color_rgb=banner_bg
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR))


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def run_model(model_name: str, detect_fn, zones_cfg: dict) -> dict:
    out_dir = PREDICT_DIR / model_name
    per_sample: dict[str, dict] = {}
    for img_name, meta in zones_cfg["samples"].items():
        img_path = SAMPLES_DIR / img_name
        if not img_path.exists():
            print(f"  [skip] missing {img_path}")
            continue
        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]
        poly_px = polygon_to_pixel(meta["polygon"], w, h)

        t0 = time.perf_counter()
        detections = detect_fn(image)
        latency_ms = (time.perf_counter() - t0) * 1000

        in_zone = [centroid_in_polygon(b, poly_px) for b, _ in detections]
        intrusion = any(in_zone)
        visualize(image, detections, poly_px, intrusion,
                  out_dir / img_name)

        per_sample[img_name] = {
            "scene": meta["scene"],
            "expected_intrusion": meta["expected_intrusion"],
            "predicted_intrusion": intrusion,
            "n_persons": len(detections),
            "n_persons_in_zone": int(sum(in_zone)),
            "max_score": max((s for _, s in detections), default=0.0),
            "latency_ms": round(latency_ms, 1),
        }
        print(f"  {img_name}: persons={len(detections)} in_zone={sum(in_zone)} "
              f"verdict={'INTRUSION' if intrusion else 'CLEAR'} ({latency_ms:.0f} ms)")
    return per_sample


def main() -> None:
    PREDICT_DIR.mkdir(parents=True, exist_ok=True)
    zones_cfg = json.loads(ZONES_JSON.read_text())

    all_results: dict[str, dict] = {}

    print("=== D-FINE-N (HF ustc-community/dfine_n_coco) ===")
    try:
        dfine_model, dfine_proc = load_dfine_n()
        all_results["dfine_n"] = run_model(
            "dfine_n", lambda im: detect_dfine(dfine_model, dfine_proc, im), zones_cfg
        )
    except Exception as e:
        print(f"  D-FINE-N failed: {e}")
        all_results["dfine_n"] = {"_error": str(e)}

    print("\n=== YOLOX-Tiny (Megvii COCO) ===")
    try:
        yolox_model = load_yolox_tiny()
        all_results["yolox_tiny"] = run_model(
            "yolox_tiny", lambda im: detect_yolox(yolox_model, im), zones_cfg
        )
    except Exception as e:
        print(f"  YOLOX-Tiny failed: {e}")
        all_results["yolox_tiny"] = {"_error": str(e)}

    out_json = PREDICT_DIR / "results.json"
    out_json.write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()

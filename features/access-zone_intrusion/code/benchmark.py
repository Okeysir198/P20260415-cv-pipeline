"""Benchmark pretrained person detectors for zone intrusion detection.

Evaluates YOLO .pt models (yolo11, yolov10, yolov12) plus YOLOX .pth baselines
on 10 sample images with ground-truth expected_intrusion from zones.json.

Usage:
    uv run features/access-zone_intrusion/code/benchmark.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

FEATURE_DIR = Path(__file__).resolve().parents[1]
SAMPLES_DIR = FEATURE_DIR / "samples"
EVAL_DIR = FEATURE_DIR / "eval"
ZONES_JSON = SAMPLES_DIR / "zones.json"

PRETRAINED_ZONE = REPO / "pretrained" / "access-zone_intrusion"
PRETRAINED_ROOT = REPO / "pretrained"

PERSON_CLASS = 0  # COCO class index for person
CONF_THRESH = 0.35
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# YOLO .pt models in the zone_intrusion pretrained dir
YOLO_MODELS = [
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt",
    "yolov10n.pt", "yolov10s.pt", "yolov10m.pt",
    "yolov12n.pt", "yolov12s.pt",
]

# YOLOX root-level baselines (COCO 80-class)
YOLOX_MODELS = [
    ("yolox_nano", "yolox-nano", PRETRAINED_ROOT / "yolox_nano.pth"),
    ("yolox_tiny", "yolox-tiny", PRETRAINED_ROOT / "yolox_tiny.pth"),
    ("yolox_s",    "yolox-s",    PRETRAINED_ROOT / "yolox_s.pth"),
    ("yolox_m",    "yolox-m",    PRETRAINED_ROOT / "yolox_m.pth"),
    ("yolox_l",    "yolox-l",    PRETRAINED_ROOT / "yolox_l.pth"),
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# Geometry helpers (from eval_sota.py)
# ---------------------------------------------------------------------------

def _polygon_to_pixel(poly_norm: list[list[float]], w: int, h: int) -> np.ndarray:
    return np.array([[p[0] * w, p[1] * h] for p in poly_norm], dtype=np.float32)


def _centroid_in_polygon(box_xyxy: np.ndarray, poly_px: np.ndarray) -> bool:
    cx = float((box_xyxy[0] + box_xyxy[2]) / 2)
    cy = float((box_xyxy[1] + box_xyxy[3]) / 2)
    return cv2.pointPolygonTest(poly_px.astype(np.int32), (cx, cy), False) >= 0


# ---------------------------------------------------------------------------
# YOLO .pt inference
# ---------------------------------------------------------------------------

def _make_yolo_detect_fn(model_path: Path) -> tuple[Callable, dict]:
    """Load Ultralytics YOLO and return (detect_fn, meta)."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))

    def detect_fn(image_bgr: np.ndarray) -> list[tuple[np.ndarray, float]]:
        results = model(image_bgr, conf=CONF_THRESH, verbose=False)[0]
        persons = []
        for box in results.boxes:
            if int(box.cls.item()) == PERSON_CLASS:
                xyxy = box.xyxy.cpu().numpy()[0]
                persons.append((xyxy, float(box.conf.item())))
        return persons

    # Latency warm-up (single pass)
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    model(dummy, conf=CONF_THRESH, verbose=False)

    return detect_fn, {"framework": "Ultralytics"}


# ---------------------------------------------------------------------------
# YOLOX .pth inference (COCO 80-class, person = class 0)
# ---------------------------------------------------------------------------

def _make_yolox_detect_fn(arch: str, ckpt_path: Path) -> tuple[Callable, dict] | None:
    """Load YOLOX PyTorch checkpoint and return (detect_fn, meta). Returns None if failed."""
    if not ckpt_path.exists():
        return None

    try:
        from core.p06_models import build_model
        from core.p10_inference.predictor import _remap_megvii_state_dict

        input_size = _yolox_input_size(arch)
        cfg = {"model": {"arch": arch, "num_classes": 80, "input_size": input_size}}
        model = build_model(cfg)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        try:
            model.load_state_dict(state)
        except RuntimeError:
            remapped = _remap_megvii_state_dict(state, set(model.state_dict().keys()))
            model.load_state_dict(remapped, strict=False)
        model.to(DEVICE).eval()

        in_h, in_w = input_size

        def detect_fn(image_bgr: np.ndarray) -> list[tuple[np.ndarray, float]]:
            from core.p06_training.postprocess import postprocess

            h, w = image_bgr.shape[:2]
            resized = cv2.resize(image_bgr, (in_w, in_h))
            chw = resized.astype(np.float32).transpose(2, 0, 1)
            x = torch.from_numpy(chw[np.newaxis, ...]).to(DEVICE)
            with torch.no_grad():
                raw = model(x)
            results = postprocess(
                "yolox", model, raw, conf_threshold=CONF_THRESH, nms_threshold=0.45
            )[0]
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

        return detect_fn, {"framework": "YOLOX (PyTorch)"}

    except Exception as exc:
        print(f"  Failed to load {arch}: {exc}", file=sys.stderr)
        return None


def _yolox_input_size(arch: str) -> list[int]:
    sizes = {
        "yolox-nano": [416, 416],
        "yolox-tiny": [416, 416],
        "yolox-s":    [640, 640],
        "yolox-m":    [640, 640],
        "yolox-l":    [640, 640],
    }
    return sizes.get(arch, [640, 640])


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def _eval_model(
    model_name: str,
    detect_fn: Callable,
    zones_cfg: dict,
) -> dict:
    """Run detector on all samples and compute intrusion metrics."""
    tp = fp = fn = tn = 0
    latencies: list[float] = []
    total_persons = 0
    skipped_edge = 0

    for img_name, meta in zones_cfg["samples"].items():
        img_path = SAMPLES_DIR / img_name
        if not img_path.exists():
            continue

        expected = meta["expected_intrusion"]
        if expected is None:
            # Edge case — undefined GT, count separately but skip from metrics
            skipped_edge += 1
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        poly_px = _polygon_to_pixel(meta["polygon"], w, h)

        t0 = time.perf_counter()
        detections = detect_fn(image)
        latencies.append((time.perf_counter() - t0) * 1000)

        in_zone = [_centroid_in_polygon(b, poly_px) for b, _ in detections]
        predicted = any(in_zone)
        total_persons += len(detections)

        if expected and predicted:
            tp += 1
        elif not expected and predicted:
            fp += 1
        elif expected and not predicted:
            fn += 1
        else:
            tn += 1

    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "status": "ok",
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "accuracy": round(accuracy, 3),
        "latency_ms": round(avg_latency, 1),
        "persons_detected": total_persons,
        "images_evaluated": total,
        "edge_cases_skipped": skipped_edge,
    }


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def _write_report(entries: list[dict]) -> str:
    ok = [e for e in entries if e.get("status") == "ok"]
    failed = [e for e in entries if e.get("status") != "ok"]

    sorted_ok = sorted(ok, key=lambda e: e.get("accuracy", -1.0), reverse=True)

    from datetime import date
    lines = [
        "# Zone Intrusion — Pretrained Model Benchmark",
        f"Date: {date.today().isoformat()}",
        f"Samples: {SAMPLES_DIR} (10 images, 2 edge-case excluded from metrics)",
        "",
        "## Results (sorted by intrusion accuracy)",
        "",
        "| Model | Framework | Accuracy | Precision | Recall | F1 | Latency ms | Persons detected |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for e in sorted_ok:
        lines.append(
            f"| {e['model']} | {e.get('framework', '—')} "
            f"| {e['accuracy']} | {e['precision']} | {e['recall']} | {e['f1']} "
            f"| {e['latency_ms']} | {e['persons_detected']} |"
        )

    if failed:
        lines += ["", "## Failed models", ""]
        for e in failed:
            lines.append(f"- **{e['model']}**: {e.get('error_msg', e.get('status'))}")

    # Recommendation
    lines += ["", "## Recommendation", ""]
    if sorted_ok:
        best = sorted_ok[0]
        lines.append(
            f"Best model for zone intrusion: **{best['model']}** "
            f"(accuracy={best['accuracy']}, F1={best['f1']}, latency={best['latency_ms']} ms)"
        )
        fast_ok = [e for e in sorted_ok if e["latency_ms"] > 0]
        if len(fast_ok) > 1:
            fastest = min(fast_ok, key=lambda e: e["latency_ms"])
            if fastest["model"] != best["model"]:
                lines.append(
                    f"Fastest model: **{fastest['model']}** "
                    f"(latency={fastest['latency_ms']} ms, accuracy={fastest['accuracy']})"
                )
    else:
        lines.append("No models evaluated successfully.")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    zones_cfg = json.loads(ZONES_JSON.read_text())
    all_entries: list[dict] = []

    # --- YOLO .pt models ---
    for model_file in YOLO_MODELS:
        model_path = PRETRAINED_ZONE / model_file
        model_name = Path(model_file).stem
        print(f"=== {model_name} ===")
        if not model_path.exists():
            print(f"  Not found: {model_path}")
            all_entries.append({"model": model_name, "status": "file_not_found",
                                 "framework": "Ultralytics"})
            continue
        try:
            detect_fn, meta = _make_yolo_detect_fn(model_path)
            metrics = _eval_model(model_name, detect_fn, zones_cfg)
            all_entries.append({"model": model_name, **meta, **metrics})
            print(f"  accuracy={metrics['accuracy']} F1={metrics['f1']} "
                  f"latency={metrics['latency_ms']} ms")
        except Exception as exc:
            print(f"  Error: {exc}", file=sys.stderr)
            all_entries.append({"model": model_name, "status": "error",
                                 "error_msg": str(exc), "framework": "Ultralytics"})

    # --- YOLOX .pth baselines ---
    for model_name, arch, ckpt_path in YOLOX_MODELS:
        print(f"=== {model_name} ===")
        result = _make_yolox_detect_fn(arch, ckpt_path)
        if result is None:
            all_entries.append({"model": model_name, "status": "load_failed",
                                 "framework": "YOLOX (PyTorch)"})
            continue
        detect_fn, meta = result
        try:
            metrics = _eval_model(model_name, detect_fn, zones_cfg)
            all_entries.append({"model": model_name, **meta, **metrics})
            print(f"  accuracy={metrics['accuracy']} F1={metrics['f1']} "
                  f"latency={metrics['latency_ms']} ms")
        except Exception as exc:
            print(f"  Error: {exc}", file=sys.stderr)
            all_entries.append({"model": model_name, "status": "error",
                                 "error_msg": str(exc), "framework": "YOLOX (PyTorch)"})

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    json_out = EVAL_DIR / "benchmark_results.json"
    json_out.write_text(json.dumps({"results": all_entries}, indent=2, default=str))
    print(f"\nJSON results: {json_out}")

    report = _write_report(all_entries)
    md_out = EVAL_DIR / "benchmark_report.md"
    md_out.write_text(report)
    print(f"Markdown report: {md_out}")

    ok = [e for e in all_entries if e.get("status") == "ok"]
    print(f"\n{len(ok)}/{len(all_entries)} models evaluated successfully.")
    if ok:
        best = max(ok, key=lambda e: e.get("accuracy", -1.0))
        print(f"Best: {best['model']} — accuracy={best['accuracy']} F1={best['f1']}")


if __name__ == "__main__":
    main()

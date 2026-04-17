"""Benchmark pretrained person/foot detection models on the shoes_detection val/test splits.

Strategy:
- rfdetr_small.onnx and dfine_small_coco.safetensors are COCO detectors.
  Evaluate person-class (COCO class 0) detection mAP against GT class 0 (person).
- No pretrained foot detector exists — foot-class metrics require fine-tuning.
- Classifiers (.bin, HF dino dirs) are skipped.

Usage:
    uv run features/ppe-shoes_detection/code/benchmark.py
    uv run features/ppe-shoes_detection/code/benchmark.py --split val
    uv run features/ppe-shoes_detection/code/benchmark.py --split test
"""

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from utils.config import load_config

PRETRAINED_DIR = REPO / "pretrained" / "ppe-shoes_detection"
DATA_CONFIG_PATH = REPO / "features" / "ppe-shoes_detection" / "configs" / "05_data.yaml"
EVAL_DIR = REPO / "features" / "ppe-shoes_detection" / "eval"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# COCO person class ID = 0
COCO_PERSON_ID = 0
# Our GT person class ID = 0
GT_PERSON_ID = 0


# ---------------------------------------------------------------------------
# Image loading + preprocessing
# ---------------------------------------------------------------------------

def _load_images_and_gt(dataset_path: Path, split: str) -> list[dict]:
    images_dir = dataset_path / split / "images"
    labels_dir = dataset_path / split / "labels"
    if not images_dir.exists():
        raise FileNotFoundError(f"Split dir not found: {images_dir}")

    samples = []
    for img_path in sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
        label_path = labels_dir / (img_path.stem + ".txt")
        gt_boxes = []   # xyxy pixel coords
        gt_labels = []
        if label_path.exists():
            img_for_size = cv2.imread(str(img_path))
            if img_for_size is None:
                continue
            h, w = img_for_size.shape[:2]
            for line in label_path.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                gt_boxes.append([x1, y1, x2, y2])
                gt_labels.append(cls_id)
        samples.append({
            "path": img_path,
            "gt_boxes": np.array(gt_boxes, dtype=np.float32).reshape(-1, 4),
            "gt_labels": np.array(gt_labels, dtype=np.int64),
        })
    return samples


def _preprocess_imagenet(img_bgr: np.ndarray, size: int = 640) -> np.ndarray:
    resized = cv2.resize(img_bgr, (size, size))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return normalized.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)


# ---------------------------------------------------------------------------
# mAP computation (simple IoU matching, no external deps)
# ---------------------------------------------------------------------------

def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _compute_map50(
    all_preds: list[dict],
    all_gts: list[dict],
    gt_class_id: int,
    iou_thresh: float = 0.5,
) -> tuple[float, float, float]:
    """Compute mAP50 for a single GT class. Returns (map50, precision, recall)."""
    # Gather all detections for this class across images, sorted by confidence desc
    det_list = []  # (score, tp, img_idx)
    n_gt = 0

    for img_idx, (pred, gt) in enumerate(zip(all_preds, all_gts)):
        gt_mask = gt["gt_labels"] == gt_class_id
        gt_boxes = gt["gt_boxes"][gt_mask]
        n_gt += len(gt_boxes)

        pred_mask = pred["labels"] == gt_class_id
        pred_boxes = pred["boxes"][pred_mask]
        pred_scores = pred["scores"][pred_mask]

        gt_matched = np.zeros(len(gt_boxes), dtype=bool)
        for pi in np.argsort(-pred_scores):
            best_iou = iou_thresh
            best_gi = -1
            for gi, gb in enumerate(gt_boxes):
                if gt_matched[gi]:
                    continue
                iou_val = _iou(pred_boxes[pi], gb)
                if iou_val >= best_iou:
                    best_iou = iou_val
                    best_gi = gi
            if best_gi >= 0:
                gt_matched[best_gi] = True
                det_list.append((float(pred_scores[pi]), 1, img_idx))
            else:
                det_list.append((float(pred_scores[pi]), 0, img_idx))

    if n_gt == 0:
        return 0.0, 0.0, 0.0

    det_list.sort(key=lambda x: -x[0])
    tp_cumsum = np.cumsum([d[1] for d in det_list]).astype(float)
    fp_cumsum = np.cumsum([1 - d[1] for d in det_list]).astype(float)

    recalls = tp_cumsum / n_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

    # All-point interpolation AP
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    ap = float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))

    # Best-F1 precision/recall
    f1 = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    best = int(np.argmax(f1)) if len(f1) > 0 else 0
    best_p = float(precisions[best]) if len(precisions) > 0 else 0.0
    best_r = float(recalls[best]) if len(recalls) > 0 else 0.0

    return ap, best_p, best_r


# ---------------------------------------------------------------------------
# RF-DETR ONNX inference
# ---------------------------------------------------------------------------

def _eval_rfdetr_onnx(
    onnx_path: Path, samples: list[dict], conf_thresh: float = 0.5
) -> tuple[list[dict], float]:
    import onnxruntime as ort

    providers = []
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    try:
        sess = ort.InferenceSession(str(onnx_path), providers=providers)
    except Exception:
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    input_name = sess.get_inputs()[0].name
    all_preds: list[dict] = []
    latencies: list[float] = []

    for sample in samples:
        img = cv2.imread(str(sample["path"]))
        if img is None:
            all_preds.append(_empty_pred())
            continue

        orig_h, orig_w = img.shape[:2]
        inp = _preprocess_imagenet(img, size=640)

        t0 = time.perf_counter()
        outputs = sess.run(None, {input_name: inp})
        latencies.append(time.perf_counter() - t0)

        # RF-DETR ONNX outputs: typically (1, N, 4) boxes + (1, N, C) logits
        # or (1, N, 5+C) combined — probe the shape
        pred = _decode_rfdetr_output(outputs, orig_h, orig_w, conf_thresh)
        all_preds.append(pred)

    avg_latency_ms = np.mean(latencies) * 1000 if latencies else 0.0
    return all_preds, avg_latency_ms


def _decode_rfdetr_output(
    outputs: list[np.ndarray], orig_h: int, orig_w: int, conf_thresh: float
) -> dict:
    """Decode RF-DETR ONNX output to boxes/scores/labels in original image coords."""
    if not outputs:
        return _empty_pred()

    out = outputs[0]  # (1, N, 5+C) or (1, N, 4+C+1) depending on export

    if out.ndim == 3:
        out = out[0]  # (N, ...)

    n = out.shape[0]
    if n == 0:
        return _empty_pred()

    # Probe: if last dim is 5+num_classes, col 4 is objectness
    # RT-DETR style: (N, 4+num_classes) with no objectness
    # RF-DETR often exports (N, 6): cx,cy,w,h,score,class_id
    ncols = out.shape[-1]

    if ncols == 6:
        # Standard RF-DETR export: cx,cy,w,h,conf,cls
        boxes_cxcywh = out[:, :4]
        scores = out[:, 4]
        class_ids = out[:, 5].astype(int)
        boxes_xyxy = _cxcywh_to_xyxy(boxes_cxcywh, orig_h, orig_w)
    elif ncols >= 5:
        # Assume (N, 4+C): boxes in [0,1] xyxy + class scores
        boxes_norm = out[:, :4]
        class_probs = out[:, 4:]
        scores = class_probs.max(axis=-1)
        class_ids = class_probs.argmax(axis=-1)
        # Scale boxes to original image
        boxes_xyxy = np.stack([
            boxes_norm[:, 0] * orig_w,
            boxes_norm[:, 1] * orig_h,
            boxes_norm[:, 2] * orig_w,
            boxes_norm[:, 3] * orig_h,
        ], axis=-1)
    else:
        return _empty_pred()

    mask = scores >= conf_thresh
    return {
        "boxes": boxes_xyxy[mask].astype(np.float32),
        "scores": scores[mask].astype(np.float32),
        "labels": class_ids[mask].astype(np.int64),
    }


def _cxcywh_to_xyxy(boxes: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
    """Convert cx,cy,w,h (normalized [0,1]) to x1,y1,x2,y2 (pixel)."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - w / 2) * orig_w
    y1 = (cy - h / 2) * orig_h
    x2 = (cx + w / 2) * orig_w
    y2 = (cy + h / 2) * orig_h
    return np.stack([x1, y1, x2, y2], axis=-1)


def _empty_pred() -> dict:
    return {
        "boxes": np.empty((0, 4), dtype=np.float32),
        "scores": np.empty((0,), dtype=np.float32),
        "labels": np.empty((0,), dtype=np.int64),
    }


# ---------------------------------------------------------------------------
# D-FINE safetensors inference via HF Transformers
# ---------------------------------------------------------------------------

def _eval_dfine_safetensors(
    model_dir: Path, samples: list[dict], conf_thresh: float = 0.5
) -> tuple[list[dict], float]:
    from transformers import AutoModelForObjectDetection, AutoProcessor
    import torch

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    try:
        processor = AutoProcessor.from_pretrained(str(model_dir))
        model = AutoModelForObjectDetection.from_pretrained(
            str(model_dir), ignore_mismatched_sizes=True
        ).to(device).eval()
    except Exception as exc:
        raise RuntimeError(f"Failed to load D-FINE from {model_dir}: {exc}") from exc

    all_preds: list[dict] = []
    latencies: list[float] = []

    with __import__("torch").no_grad():
        for sample in samples:
            from PIL import Image as PILImage
            img_pil = PILImage.open(str(sample["path"])).convert("RGB")
            orig_w, orig_h = img_pil.size

            inputs = processor(images=img_pil, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            t0 = time.perf_counter()
            output = model(**inputs)
            latencies.append(time.perf_counter() - t0)

            # Postprocess with processor
            target_sizes = __import__("torch").tensor([[orig_h, orig_w]])
            results = processor.post_process_object_detection(
                output, threshold=conf_thresh, target_sizes=target_sizes
            )[0]

            boxes = results["boxes"].cpu().numpy()  # (N, 4) xyxy
            scores = results["scores"].cpu().numpy()
            labels = results["labels"].cpu().numpy()

            all_preds.append({
                "boxes": boxes.astype(np.float32),
                "scores": scores.astype(np.float32),
                "labels": labels.astype(np.int64),
            })

    avg_ms = np.mean(latencies) * 1000 if latencies else 0.0
    return all_preds, avg_ms


# ---------------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------------

def _count_images(dataset_path: Path, split: str) -> int:
    d = dataset_path / split / "images"
    if not d.exists():
        return 0
    return sum(1 for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)


def _run_split(
    model_name: str,
    model_fn,
    dataset_path: Path,
    split: str,
    conf_thresh: float = 0.5,
) -> dict:
    try:
        samples = _load_images_and_gt(dataset_path, split)
        all_preds, latency_ms = model_fn(samples)
        map50, prec, rec = _compute_map50(all_preds, samples, gt_class_id=GT_PERSON_ID)
        return {
            "status": "ok",
            "map50_person": round(map50, 4),
            "precision_person": round(prec, 4),
            "recall_person": round(rec, 4),
            "latency_ms": round(latency_ms, 1),
            "n_images": len(samples),
        }
    except Exception as exc:
        return {"status": "error", "error_msg": str(exc)}


def _run_benchmark(splits: list[str]) -> tuple[list[dict], list[dict]]:
    data_config = load_config(DATA_CONFIG_PATH)
    dataset_path = (DATA_CONFIG_PATH.parent / data_config["path"]).resolve()

    results: list[dict] = []
    skipped: list[dict] = []

    # --- RF-DETR ONNX ---
    rfdetr_path = PRETRAINED_DIR / "rfdetr_small.onnx"
    if rfdetr_path.exists():
        entry: dict = {"model": "rfdetr_small", "framework": "ONNX Runtime (RF-DETR)", "note": "COCO, person-class only"}
        for split in splits:
            entry[split] = _run_split(
                "rfdetr_small",
                lambda s, p=rfdetr_path: _eval_rfdetr_onnx(p, s),
                dataset_path,
                split,
            )
        results.append(entry)
    else:
        skipped.append({"model": "rfdetr_small.onnx", "reason": "File not found"})

    # --- D-FINE safetensors ---
    # Look for the safetensors in a HF model dir or as a standalone file
    dfine_pt = PRETRAINED_DIR / "dfine_small_coco.safetensors"
    dfine_dir = _find_dfine_hf_dir()
    if dfine_dir is not None:
        entry = {"model": "dfine_small_coco", "framework": "HF Transformers (D-FINE)", "note": "COCO, person-class only"}
        for split in splits:
            entry[split] = _run_split(
                "dfine_small_coco",
                lambda s, d=dfine_dir: _eval_dfine_safetensors(d, s),
                dataset_path,
                split,
            )
        results.append(entry)
    elif dfine_pt.exists():
        skipped.append({
            "model": "dfine_small_coco",
            "reason": (
                "Found dfine_small_coco.safetensors but no companion config.json / "
                "preprocessor_config.json in the pretrained dir. Cannot load via HF "
                "AutoModelForObjectDetection without a config. Provide a full HF model dir "
                "or add config.json alongside the safetensors file."
            ),
        })
    else:
        skipped.append({"model": "dfine_small_coco", "reason": "File not found"})

    # --- Classifiers and non-detectors ---
    for name in ["fastvit_t12.bin", "fastvit_t8.bin", "efficientformerv2_s1.bin",
                  "mobilevitv2_100.bin"]:
        skipped.append({"model": name, "reason": "Image classifier, not a detector"})

    for d in PRETRAINED_DIR.iterdir():
        if d.is_dir() and d.name.startswith("_hf_facebook_dino"):
            skipped.append({"model": d.name, "reason": "Image feature extractor, not a detector"})

    return results, skipped


def _find_dfine_hf_dir() -> Optional[Path]:
    """Return a D-FINE HF model dir that has config.json alongside a safetensors file."""
    for candidate in PRETRAINED_DIR.iterdir():
        if not candidate.is_dir():
            continue
        if (candidate / "config.json").exists() and any(
            candidate.glob("*.safetensors")
        ):
            return candidate
    # Also check if the safetensors lives in an unnamed subdir
    safetensors = PRETRAINED_DIR / "dfine_small_coco.safetensors"
    if safetensors.exists():
        # Check if a companion config.json is in the same dir
        if (PRETRAINED_DIR / "config.json").exists():
            return PRETRAINED_DIR
    return None


def _write_report(
    results: list[dict],
    skipped: list[dict],
    splits: list[str],
    val_count: int,
    test_count: int,
) -> str:
    lines = [
        "# PPE Shoes Detection — Pretrained Model Benchmark",
        f"Date: {date.today().isoformat()}",
        f"Dataset: shoes_detection (val: {val_count} images, test: {test_count} images)",
        "",
        "## Scope",
        "",
        "All available pretrained models are COCO detectors. Evaluated on:",
        "- **Person detection** (GT class 0): COCO models detect `person` which maps to our class 0.",
        "- **Foot classes** (GT 1=foot_with_safety_shoes, 2=foot_without_safety_shoes): "
        "no pretrained foot detector exists. These classes require fine-tuning.",
        "",
        "## Person-Detection Results (sorted by val mAP50)",
        "",
    ]

    primary_split = splits[0]
    sorted_results = sorted(
        results,
        key=lambda e: e.get(primary_split, {}).get("map50_person") or -1.0,
        reverse=True,
    )

    headers = ["Model", "Framework", "val mAP50 (person)", "val P", "val R"]
    if "test" in splits:
        headers += ["test mAP50 (person)", "test P", "test R"]
    headers += ["Latency ms", "Status", "Note"]

    sep = " | "
    lines.append(sep.join(headers))
    lines.append(sep.join(["---"] * len(headers)))

    def fmt(v: object) -> str:
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    for entry in sorted_results:
        val_s = entry.get("val", {})
        status = val_s.get("status", "—")
        if status == "error":
            status = f"error: {val_s.get('error_msg', '')[:60]}"

        row = [
            entry["model"],
            entry.get("framework", "—"),
            fmt(val_s.get("map50_person")),
            fmt(val_s.get("precision_person")),
            fmt(val_s.get("recall_person")),
        ]
        if "test" in splits:
            test_s = entry.get("test", {})
            row += [
                fmt(test_s.get("map50_person")),
                fmt(test_s.get("precision_person")),
                fmt(test_s.get("recall_person")),
            ]
        row += [
            fmt(val_s.get("latency_ms")),
            status,
            entry.get("note", ""),
        ]
        lines.append(sep.join(row))

    lines += [
        "",
        "## Skipped models",
        "",
        "| Model | Reason |",
        "| --- | --- |",
    ]
    for s in skipped:
        lines.append(f"| {s['model']} | {s['reason']} |")

    lines += [
        "",
        "## Foot-Class Evaluation",
        "",
        "| Class | Status |",
        "| --- | --- |",
        "| foot_with_safety_shoes | No pretrained model — requires fine-tuning |",
        "| foot_without_safety_shoes | No pretrained model — requires fine-tuning |",
        "",
        "## Recommendation",
        "",
    ]

    ok_results = [e for e in sorted_results if e.get("val", {}).get("status") == "ok"]
    if ok_results:
        best = ok_results[0]
        map50 = best.get("val", {}).get("map50_person", 0.0)
        lines.append(
            f"Best person detector: **{best['model']}** (val mAP50 person: {map50:.3f}). "
            "Use as the person-detection backbone for the two-stage shoes pipeline. "
            "Fine-tune foot-class heads on shoes_detection training split."
        )
    else:
        lines.append(
            "No detector evaluated successfully. "
            "Check ONNX/HF model files and re-run after fixing errors."
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test", "both"], default="both")
    args = parser.parse_args()

    splits = ["val", "test"] if args.split == "both" else [args.split]

    data_config = load_config(DATA_CONFIG_PATH)
    dataset_path = (DATA_CONFIG_PATH.parent / data_config["path"]).resolve()
    val_count = _count_images(dataset_path, "val")
    test_count = _count_images(dataset_path, "test")

    print(f"Benchmarking on splits: {splits}")
    print(f"Dataset: {dataset_path} (val={val_count}, test={test_count})")

    results, skipped = _run_benchmark(splits)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    json_out = EVAL_DIR / "benchmark_results.json"
    json_out.write_text(
        json.dumps({"results": results, "skipped": skipped}, indent=2, default=str)
    )
    print(f"JSON results: {json_out}")

    report = _write_report(results, skipped, splits, val_count, test_count)
    md_out = EVAL_DIR / "benchmark_report.md"
    md_out.write_text(report)
    print(f"Markdown report: {md_out}")

    ok = [e for e in results if e.get("val", {}).get("status") == "ok"]
    print(f"\nEvaluated {len(ok)} models successfully, {len(skipped)} skipped.")
    if ok:
        best = max(ok, key=lambda e: e.get("val", {}).get("map50_person") or -1.0)
        print(f"Best person detector: {best['model']} — val mAP50={best['val'].get('map50_person', '?'):.3f}")


if __name__ == "__main__":
    main()

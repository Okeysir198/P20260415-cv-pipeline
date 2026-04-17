#!/usr/bin/env python3
"""Benchmark pretrained helmet-detection models on our val/test splits.

Evaluates every Ultralytics .pt / .onnx file in pretrained/ppe-helmet_detection/,
plus the YOLOS-tiny HF model.  For each model it:
  1. Loads the model and reads its class names.
  2. Fuzzy-maps model class names -> our 4 GT class IDs.
  3. Writes a temporary Ultralytics data.yaml using the model's OWN class names,
     runs .val() to get per-class APs, then remaps those APs to our class names.
  4. Writes benchmark_results.json + benchmark_report.md under eval/.

Usage:
    uv run features/ppe-helmet_detection/code/benchmark.py [--split val|test]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

PRETRAINED_DIR = REPO / "pretrained" / "ppe-helmet_detection"
EVAL_DIR = REPO / "features" / "ppe-helmet_detection" / "eval"
DATA_CONFIG_PATH = REPO / "features" / "ppe-helmet_detection" / "configs" / "05_data.yaml"

# Our canonical 4-class GT map
GT_CLASSES: dict[int, str] = {
    0: "person",
    1: "head_with_helmet",
    2: "head_without_helmet",
    3: "head_with_nitto_hat",
}

# --- Fuzzy class-name mapping rules ----------------------------------------
# Each entry: (list of substrings/patterns, GT class id)
# Matching is case-insensitive; first match wins.
_MATCH_RULES: list[tuple[list[str], int]] = [
    (["head_with_helmet", "hardhat", "hard-hat", "hard hat", "safety helmet",
      "helmet", "hard_hat", "safety_helmet"], 1),
    (["head_without_helmet", "no helmet", "no-helmet", "no hardhat", "no-hardhat",
      "no hard hat", "without helmet", "without_helmet", "nohelmet", "nohardhat",
      "bare head", "no_helmet", "no_hardhat"], 2),
    (["nitto", "bump cap", "bump_cap", "soft cap", "soft_cap"], 3),
    (["person", "human", "worker", "people"], 0),
    # vest / safety-vest → no GT class match; return -1 below via fallthrough
]

_VEST_SKIP = {"vest", "safety vest", "safety_vest", "hi-vis", "hivis"}


def map_model_class_to_gt(name: str) -> int:
    """Return GT class id for a model class name, or -1 if no match."""
    low = name.lower().strip()
    # fast vest skip
    if low in _VEST_SKIP or "vest" in low:
        return -1
    for patterns, gt_id in _MATCH_RULES:
        for pat in patterns:
            if pat in low:
                return gt_id
    return -1


def build_class_map(model_names: dict[int, str]) -> dict[int, int]:
    """Map model class ids -> GT class ids.  Unmapped classes get -1."""
    return {mid: map_model_class_to_gt(mname) for mid, mname in model_names.items()}


# --- Temp data.yaml helpers -------------------------------------------------

def _load_data_config() -> dict[str, Any]:
    with open(DATA_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _resolve_dataset_path(data_config: dict[str, Any]) -> Path:
    raw_path = data_config["path"]  # e.g. "../../../dataset_store/..."
    return (DATA_CONFIG_PATH.parent / raw_path).resolve()


def make_temp_yaml_for_model(model_names: dict[int, str], dataset_path: Path,
                              data_config: dict[str, Any], split: str) -> str:
    """Build a temp Ultralytics data.yaml using the model's OWN class names."""
    nc = len(model_names)
    names = [model_names[i] for i in range(nc)]
    d = {
        "path": str(dataset_path),
        "train": data_config["train"],
        "val": data_config["val"],
        "test": data_config.get("test", data_config["val"]),
        "nc": nc,
        "names": names,
    }
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(d, tmp)
    tmp.close()
    return tmp.name


def make_temp_yaml_gt(dataset_path: Path, data_config: dict[str, Any]) -> str:
    """Build a temp Ultralytics data.yaml using our GT class names."""
    names = [GT_CLASSES[i] for i in sorted(GT_CLASSES)]
    d = {
        "path": str(dataset_path),
        "train": data_config["train"],
        "val": data_config["val"],
        "test": data_config.get("test", data_config["val"]),
        "nc": len(GT_CLASSES),
        "names": names,
    }
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(d, tmp)
    tmp.close()
    return tmp.name


# --- Metrics extraction from Ultralytics results ----------------------------

def _extract_metrics(val_results: Any, class_map: dict[int, int],
                     model_names: dict[int, str]) -> dict[str, Any]:
    """Pull per-class APs from Ultralytics validator results and remap to GT classes."""
    # val_results is an ultralytics.utils.metrics.DetMetrics object
    # .results_dict has summary; .ap_class_index and .box.ap have per-class AP@.5
    try:
        box = val_results.box
        map50 = float(box.map50)
        map50_95 = float(box.map)
        per_class_ap50: dict[str, float] = {}

        ap_per_cls = box.ap50  # ndarray, one entry per class that appeared
        ap_cls_idx = val_results.ap_class_index  # model class ids

        # Remap each model-class AP to our GT class name
        gt_ap_accum: dict[int, list[float]] = {k: [] for k in GT_CLASSES}
        for model_cls_id, ap_val in zip(ap_cls_idx, ap_per_cls):
            gt_id = class_map.get(int(model_cls_id), -1)
            if gt_id >= 0:
                gt_ap_accum[gt_id].append(float(ap_val))

        for gt_id, aps in gt_ap_accum.items():
            gt_name = GT_CLASSES[gt_id]
            per_class_ap50[gt_name] = float(sum(aps) / len(aps)) if aps else None

        return {
            "map50": map50,
            "map50_95": map50_95,
            "per_class_ap50": per_class_ap50,
            "precision": float(box.mp),
            "recall": float(box.mr),
        }
    except Exception as e:
        return {"error": f"metrics extraction failed: {e}"}


# --- Ultralytics PT / ONNX evaluation ---------------------------------------

def evaluate_ultralytics(model_path: Path, dataset_path: Path,
                          data_config: dict[str, Any], split: str,
                          conf: float = 0.001) -> dict[str, Any]:
    """Load an Ultralytics model and run .val() with a temp data.yaml."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    model_names: dict[int, str] = model.names  # {0: "hardhat", 1: "person", ...}

    class_map = build_class_map(model_names)
    mapped_classes = {mid: GT_CLASSES[gid] for mid, gid in class_map.items() if gid >= 0}

    tmp_yaml = make_temp_yaml_for_model(model_names, dataset_path, data_config, split)
    try:
        results = model.val(
            data=tmp_yaml,
            split=split,
            conf=conf,
            iou=0.5,
            verbose=False,
            plots=False,
        )
        metrics = _extract_metrics(results, class_map, model_names)
    finally:
        Path(tmp_yaml).unlink(missing_ok=True)

    return {
        "model_class_names": model_names,
        "class_map": {str(k): v for k, v in class_map.items()},
        "mapped_gt_classes": mapped_classes,
        **metrics,
    }


# --- YOLOS-tiny HF evaluation -----------------------------------------------

def evaluate_yolos_tiny(model_dir: Path, dataset_path: Path,
                         data_config: dict[str, Any], split: str) -> dict[str, Any]:
    """Evaluate YOLOS-tiny-hardhat via HF transformers + manual mAP."""
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoFeatureExtractor, AutoModelForObjectDetection

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_extractor = AutoFeatureExtractor.from_pretrained(str(model_dir))
    hf_model = AutoModelForObjectDetection.from_pretrained(str(model_dir))
    hf_model.eval().to(device)

    id2label: dict[int, str] = hf_model.config.id2label
    class_map = build_class_map({int(k): v for k, v in id2label.items()})

    split_dir = dataset_path / data_config[split] if split in data_config else dataset_path / "val/images"
    label_dir = split_dir.parent / "labels"

    predictions: list[dict] = []
    ground_truths: list[dict] = []

    img_paths = sorted(p for p in split_dir.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})

    conf_thresh = 0.3

    with torch.no_grad():
        for img_path in img_paths:
            img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img.size

            inputs = feature_extractor(images=img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = hf_model(**inputs)

            target_sizes = torch.tensor([[orig_h, orig_w]], device=device)
            result = feature_extractor.post_process_object_detection(
                outputs, threshold=conf_thresh, target_sizes=target_sizes
            )[0]

            boxes_pred = result["boxes"].cpu().numpy()  # xyxy pixel
            scores_pred = result["scores"].cpu().numpy()
            labels_pred = result["labels"].cpu().numpy()

            # Remap model class ids -> GT class ids; drop unmatched
            keep_mask = np.array([class_map.get(int(l), -1) >= 0 for l in labels_pred])
            remapped_labels = np.array([class_map.get(int(l), -1) for l in labels_pred])

            predictions.append({
                "boxes": boxes_pred[keep_mask],
                "scores": scores_pred[keep_mask],
                "labels": remapped_labels[keep_mask],
            })

            # GT from YOLO .txt
            label_path = label_dir / (img_path.stem + ".txt")
            gt_boxes, gt_labels = [], []
            if label_path.exists():
                for line in label_path.read_text().splitlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    x1 = (cx - w / 2) * orig_w
                    y1 = (cy - h / 2) * orig_h
                    x2 = (cx + w / 2) * orig_w
                    y2 = (cy + h / 2) * orig_h
                    gt_boxes.append([x1, y1, x2, y2])
                    gt_labels.append(cls_id)

            ground_truths.append({
                "boxes": np.array(gt_boxes, dtype=np.float64).reshape(-1, 4),
                "labels": np.array(gt_labels, dtype=np.int64),
            })

    from core.p08_evaluation.sv_metrics import compute_map
    metrics = compute_map(predictions, ground_truths, iou_threshold=0.5, num_classes=4)

    # sv_metrics returns "mAP" (single IoU threshold), not "map50"/"map50_95"
    per_class_ap = metrics.get("per_class_ap", {})  # {int: float}
    per_class_ap50 = {cls_name: per_class_ap.get(cls_id) for cls_id, cls_name in GT_CLASSES.items()}

    # precision/recall in sv_metrics are dicts keyed by class id — take the mean
    prec_dict = metrics.get("precision", {})
    rec_dict = metrics.get("recall", {})
    mean_prec = float(sum(prec_dict.values()) / len(prec_dict)) if prec_dict else None
    mean_rec = float(sum(rec_dict.values()) / len(rec_dict)) if rec_dict else None

    return {
        "model_class_names": id2label,
        "class_map": {str(k): v for k, v in class_map.items()},
        "map50": metrics.get("mAP"),  # sv_metrics key is "mAP"
        "map50_95": None,             # sv_metrics computes at single IoU threshold
        "per_class_ap50": per_class_ap50,
        "precision": mean_prec,
        "recall": mean_rec,
    }


# --- Report writing ----------------------------------------------------------

def write_report(results: list[dict], out_dir: Path, split: str) -> None:
    """Write markdown benchmark report sorted by map50 descending."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sort: successful models by map50 desc, then skipped/errors at bottom
    ok = [r for r in results if r.get("status") == "ok"]
    err = [r for r in results if r.get("status") != "ok"]
    ok_sorted = sorted(ok, key=lambda r: r.get("map50") or 0.0, reverse=True)

    lines = [
        f"# Helmet Detection — Pretrained Model Benchmark ({split})",
        "",
        f"**Dataset:** `dataset_store/training_ready/helmet_detection`  ",
        f"**Split:** `{split}`  ",
        f"**GT classes:** {', '.join(f'`{v}`' for v in GT_CLASSES.values())}",
        "",
        "## Results (sorted by mAP50)",
        "",
        "| Rank | Model | mAP50 | mAP50-95 | P | R | person | head_w_helmet | head_wo_helmet | nitto |",
        "|------|-------|-------|----------|---|---|--------|---------------|----------------|-------|",
    ]

    def _fmt(v: float | None) -> str:
        return f"{v:.3f}" if v is not None else "—"

    for rank, r in enumerate(ok_sorted, 1):
        pc = r.get("per_class_ap50", {})
        lines.append(
            f"| {rank} | `{r['model']}` "
            f"| {_fmt(r.get('map50'))} "
            f"| {_fmt(r.get('map50_95'))} "
            f"| {_fmt(r.get('precision'))} "
            f"| {_fmt(r.get('recall'))} "
            f"| {_fmt(pc.get('person'))} "
            f"| {_fmt(pc.get('head_with_helmet'))} "
            f"| {_fmt(pc.get('head_without_helmet'))} "
            f"| {_fmt(pc.get('head_with_nitto_hat'))} "
            f"|"
        )

    if err:
        lines += ["", "## Skipped / Errors", ""]
        for r in err:
            lines.append(f"- `{r['model']}`: {r.get('error', r.get('status', 'unknown'))}")

    lines += ["", f"*Generated {time.strftime('%Y-%m-%d %H:%M')}*", ""]

    report_path = out_dir / "benchmark_report.md"
    report_path.write_text("\n".join(lines))
    print(f"Report written to {report_path}")


# --- Main -------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark pretrained helmet models")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--conf", type=float, default=0.001,
                        help="Confidence threshold for Ultralytics .val()")
    return parser.parse_args()


def _try_load_bin(path: Path, dataset_path: Path, data_config: dict[str, Any],
                   split: str) -> dict[str, Any]:
    """Attempt to evaluate a .bin file.

    The gghsgn_*.bin files are HuggingFace YOLOS (ViT-based) PyTorch state dicts.
    They require a config.json alongside them to instantiate the model architecture.
    Without it we cannot reconstruct the model, so we skip with a clear note.
    """
    # Check if a config.json exists alongside the .bin
    config_json = path.parent / "config.json"
    if not config_json.exists():
        # Try to find config.json in a same-named subdirectory
        subdir_config = path.parent / path.stem / "config.json"
        if not subdir_config.exists():
            return {
                "status": "skipped",
                "error": (
                    "HuggingFace YOLOS state dict — requires config.json to "
                    "reconstruct model architecture. No config.json found alongside "
                    f"{path.name}. Download the full HF repo to evaluate."
                ),
            }
        config_json = subdir_config

    # config.json exists — attempt to load via HF transformers
    try:
        from transformers import AutoFeatureExtractor, AutoModelForObjectDetection
        model_dir = config_json.parent
        metrics = evaluate_yolos_tiny(model_dir, dataset_path, data_config, split)
        return {"status": "ok", **metrics}
    except Exception as exc:
        return {"status": "error", "error": f"HF load failed: {exc}"}


def _try_load_safetensors(path: Path, dataset_path: Path, data_config: dict[str, Any],
                           split: str) -> dict[str, Any]:
    """Attempt to evaluate a .safetensors file via HF AutoModelForObjectDetection.

    Requires a config.json in the same directory as the safetensors file.
    """
    config_json = path.parent / "config.json"
    if not config_json.exists():
        return {
            "status": "skipped",
            "error": (
                "Missing config.json — cannot load DETA/HF model without it. "
                f"The {path.name} safetensors needs a HF model config to "
                "determine architecture. Download the full HF repo to evaluate."
            ),
        }
    try:
        metrics = evaluate_yolos_tiny(path.parent, dataset_path, data_config, split)
        return {"status": "ok", **metrics}
    except Exception as exc:
        return {"status": "error", "error": f"HF safetensors load failed: {exc}"}


def _collect_model_paths() -> list[tuple[str, Path]]:
    """Return (label, path) pairs for all evaluable models."""
    entries: list[tuple[str, Path]] = []
    skip_dirs = {
        "_hf_facebook_dinov3-vitb16-pretrain-lvd1689m",
        "_hf_facebook_dinov3-vits16-pretrain-lvd1689m",
        "_hf_Advantech-EIOT_qualcomm-ultralytics-ppe_detection",
    }

    for p in sorted(PRETRAINED_DIR.iterdir()):
        if p.is_dir():
            if p.name in skip_dirs:
                continue
            if p.name == "yolos-tiny-hardhat":
                entries.append(("yolos-tiny-hardhat", p))
            continue
        if p.suffix.lower() in {".pt", ".onnx", ".bin", ".safetensors"}:
            entries.append((p.name, p))

    return entries


def main() -> None:
    args = _parse_args()
    data_config = _load_data_config()
    dataset_path = _resolve_dataset_path(data_config)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    model_entries = _collect_model_paths()
    print(f"Found {len(model_entries)} models to evaluate on split='{args.split}'")

    all_results: list[dict] = []

    for label, model_path in model_entries:
        print(f"\n--- Evaluating: {label} ---")
        result: dict[str, Any] = {"model": label, "path": str(model_path)}

        try:
            t0 = time.time()
            suffix = model_path.suffix.lower() if not model_path.is_dir() else ""
            if model_path.is_dir():
                metrics = evaluate_yolos_tiny(model_path, dataset_path, data_config, args.split)
                elapsed = time.time() - t0
                result.update({"status": "ok", "elapsed_s": round(elapsed, 1), **metrics})
                print(f"  mAP50={metrics.get('map50', 'N/A'):.3f}  "
                      f"mAP50-95={metrics.get('map50_95', 'N/A'):.3f}  "
                      f"({elapsed:.0f}s)")
            elif suffix == ".bin":
                outcome = _try_load_bin(model_path, dataset_path, data_config, args.split)
                elapsed = time.time() - t0
                result.update({"elapsed_s": round(elapsed, 1), **outcome})
                print(f"  {outcome.get('status', 'unknown')}: {outcome.get('error', '')}")
            elif suffix == ".safetensors":
                outcome = _try_load_safetensors(model_path, dataset_path, data_config, args.split)
                elapsed = time.time() - t0
                result.update({"elapsed_s": round(elapsed, 1), **outcome})
                print(f"  {outcome.get('status', 'unknown')}: {outcome.get('error', '')}")
            else:
                metrics = evaluate_ultralytics(model_path, dataset_path, data_config,
                                               args.split, conf=args.conf)
                elapsed = time.time() - t0
                result.update({"status": "ok", "elapsed_s": round(elapsed, 1), **metrics})
                print(f"  mAP50={metrics.get('map50', 'N/A'):.3f}  "
                      f"mAP50-95={metrics.get('map50_95', 'N/A'):.3f}  "
                      f"({elapsed:.0f}s)")
        except Exception as exc:
            result["status"] = "error"
            result["error"] = str(exc)
            print(f"  ERROR: {exc}")

        all_results.append(result)

    # Serialise — convert any non-serialisable numpy scalars
    def _to_native(obj: Any) -> Any:
        if hasattr(obj, "item"):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_native(v) for v in obj]
        return obj

    serialisable = _to_native(all_results)

    json_path = EVAL_DIR / "benchmark_results.json"
    json_path.write_text(json.dumps(serialisable, indent=2))
    print(f"\nJSON results written to {json_path}")

    write_report(serialisable, EVAL_DIR, args.split)


if __name__ == "__main__":
    main()

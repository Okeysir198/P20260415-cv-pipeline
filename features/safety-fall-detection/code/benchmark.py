#!/usr/bin/env python3
"""Benchmark pretrained fall-detection models on our val/test splits.

Evaluates:
  - yolov11_fall_melihuzunoglu.pt   (Ultralytics detection)
  - yolov8_fall_kamalchibrani.pt    (Ultralytics detection)
  - fall_resnet18_popkek00.safetensors  (HF ResNet classification on GT crops)
  - yolox_s.pth / yolox_m.pth      (COCO pretrained; person-only baseline)

Video / action-recognition models (videomae, x3d, slowfast, movinet) are
skipped — they require multi-frame video input.

Image classifiers (dinov2, efficientnetv2, mobilenetv4) are general-purpose
and have no fall-class vocabulary; skipped with a note.

Usage:
    uv run features/safety-fall-detection/code/benchmark.py [--split val|test]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

PRETRAINED_DIR = REPO / "pretrained" / "safety-fall-detection"
PRETRAINED_ROOT = REPO / "pretrained"
EVAL_DIR = REPO / "features" / "safety-fall-detection" / "eval"
DATA_CONFIG_PATH = REPO / "features" / "safety-fall-detection" / "configs" / "05_data.yaml"

# Our 2-class GT map
GT_CLASSES: dict[int, str] = {
    0: "person",
    1: "fallen_person",
}

# --- Fuzzy class-name matching ----------------------------------------------
_FALL_PATTERNS = [
    "fallen", "fall", "fallen_person", "laying", "lying", "collapse",
    "on ground", "on the ground", "down",
]
_PERSON_PATTERNS = ["person", "human", "standing", "worker", "people", "pedestrian"]


def map_model_class_to_gt(name: str) -> int:
    """Return GT class id for a model class name, or -1 if no match."""
    low = name.lower().strip()
    for pat in _FALL_PATTERNS:
        if pat in low:
            return 1
    for pat in _PERSON_PATTERNS:
        if pat in low:
            return 0
    return -1


def build_class_map(model_names: dict[int, str]) -> dict[int, int]:
    return {mid: map_model_class_to_gt(mname) for mid, mname in model_names.items()}


# --- Data helpers -----------------------------------------------------------

def _load_data_config() -> dict[str, Any]:
    with open(DATA_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _resolve_dataset_path(data_config: dict[str, Any]) -> Path:
    return (DATA_CONFIG_PATH.parent / data_config["path"]).resolve()


def make_temp_yaml_for_model(model_names: dict[int, str], dataset_path: Path,
                              data_config: dict[str, Any]) -> str:
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
    """Temp yaml using our GT class names."""
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


# --- Ultralytics evaluation -------------------------------------------------

def _extract_metrics(val_results: Any, class_map: dict[int, int]) -> dict[str, Any]:
    """Remap per-class APs from Ultralytics DetMetrics to our GT class names."""
    try:
        box = val_results.box
        map50 = float(box.map50)
        map50_95 = float(box.map)

        ap_per_cls = box.ap50
        ap_cls_idx = val_results.ap_class_index

        gt_ap_accum: dict[int, list[float]] = {k: [] for k in GT_CLASSES}
        for model_cls_id, ap_val in zip(ap_cls_idx, ap_per_cls):
            gt_id = class_map.get(int(model_cls_id), -1)
            if gt_id >= 0:
                gt_ap_accum[gt_id].append(float(ap_val))

        per_class_ap50 = {
            GT_CLASSES[gid]: (float(sum(aps) / len(aps)) if aps else None)
            for gid, aps in gt_ap_accum.items()
        }

        return {
            "map50": map50,
            "map50_95": map50_95,
            "per_class_ap50": per_class_ap50,
            "precision": float(box.mp),
            "recall": float(box.mr),
        }
    except Exception as e:
        return {"error": f"metrics extraction failed: {e}"}


def evaluate_ultralytics(model_path: Path, dataset_path: Path,
                          data_config: dict[str, Any], split: str,
                          conf: float = 0.001) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    model_names: dict[int, str] = model.names
    class_map = build_class_map(model_names)

    tmp_yaml = make_temp_yaml_for_model(model_names, dataset_path, data_config)
    try:
        results = model.val(
            data=tmp_yaml,
            split=split,
            conf=conf,
            iou=0.5,
            verbose=False,
            plots=False,
        )
        metrics = _extract_metrics(results, class_map)
    finally:
        Path(tmp_yaml).unlink(missing_ok=True)

    return {
        "model_class_names": model_names,
        "class_map": {str(k): v for k, v in class_map.items()},
        **metrics,
    }


# --- YOLOX COCO baseline (person only) -------------------------------------

def evaluate_yolox_coco(ckpt_path: Path, dataset_path: Path,
                         data_config: dict[str, Any], split: str) -> dict[str, Any]:
    """YOLOX pretrained on COCO-80.  Person = class 0.  No fallen_person class.

    We run standard Ultralytics-style evaluation on the person class only,
    treating every GT fallen_person box as a person (since from YOLOX's
    perspective it's all person).
    """
    import torch
    from core.p06_models import build_model
    from core.p08_evaluation.evaluator import ModelEvaluator

    arch = ckpt_path.stem.replace("yolox_", "yolox-")  # yolox_s -> yolox-s
    cfg = {
        "model": {"arch": arch, "num_classes": 80, "pretrained": False},
        "training": {"input_size": [640, 640]},
    }
    model = build_model(cfg)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)

    # Build a data config that maps COCO class 0 (person) to our GT class 0
    eval_data_cfg = dict(data_config)
    eval_data_cfg["_config_dir"] = DATA_CONFIG_PATH.parent
    # The GT labels in our dataset are 0=person, 1=fallen_person.
    # YOLOX sees 80 COCO classes; class 0 = person.
    # We evaluate mAP for class 0 only (person), ignoring fallen_person.
    eval_data_cfg["num_classes"] = 80
    eval_data_cfg["names"] = {i: f"coco_{i}" for i in range(80)}
    eval_data_cfg["names"][0] = "person"

    evaluator = ModelEvaluator(
        model=model,
        data_config=eval_data_cfg,
        conf_threshold=0.25,
        iou_threshold=0.5,
        batch_size=8,
    )
    preds, gts = evaluator.get_predictions(split)

    # Collect AP for class 0 (person) only
    from core.p08_evaluation.sv_metrics import compute_precision_recall
    prec, rec, _ = compute_precision_recall(preds, gts, class_id=0, iou_threshold=0.5)
    if prec.size:
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([1.0], prec, [0.0]))
        mpre = np.maximum.accumulate(mpre[::-1])[::-1]
        person_ap50 = float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))
    else:
        person_ap50 = 0.0

    return {
        "note": "COCO pretrained — person class only; no fallen_person class",
        "map50": person_ap50,
        "map50_95": None,
        "per_class_ap50": {"person": person_ap50, "fallen_person": None},
        "precision": None,
        "recall": None,
    }


# --- ResNet18 classification on GT crops ------------------------------------

def _load_gt_crops(split_images_dir: Path, label_dir: Path,
                   target_size: int = 224) -> tuple[list, list[int]]:
    """Crop GT bboxes from val images.  Returns (pil_crops, gt_class_ids)."""
    from PIL import Image

    crops, labels = [], []
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for img_path in sorted(p for p in split_images_dir.iterdir()
                            if p.suffix.lower() in img_exts):
        label_path = label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            x1 = max(0, int((cx - w / 2) * orig_w))
            y1 = max(0, int((cy - h / 2) * orig_h))
            x2 = min(orig_w, int((cx + w / 2) * orig_w))
            y2 = min(orig_h, int((cy + h / 2) * orig_h))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img.crop((x1, y1, x2, y2)).resize(
                (target_size, target_size), resample=2  # BICUBIC
            )
            crops.append(crop)
            labels.append(cls_id)

    return crops, labels


def evaluate_resnet_classifier(safetensors_path: Path, config_path: Path,
                                dataset_path: Path, data_config: dict[str, Any],
                                split: str) -> dict[str, Any]:
    """Classify GT crops with ResNet18.  Model has 2 classes: no_fall(0), fall(1)."""
    import torch
    from PIL import Image
    from torchvision import transforms
    from transformers import AutoModelForImageClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = AutoModelForImageClassification.from_pretrained(
            str(safetensors_path.parent),
            ignore_mismatched_sizes=True,
        )
    except Exception:
        # Fallback: load with explicit config if model_type missing
        from transformers import AutoConfig, ResNetForImageClassification
        cfg = AutoConfig.for_model("resnet", num_labels=2)
        model = ResNetForImageClassification(cfg)
        from safetensors.torch import load_file
        state = load_file(str(safetensors_path))
        model.load_state_dict(state, strict=False)
    model.eval().to(device)

    # Model label map: {"0": "no_fall", "1": "fall"}
    id2label: dict[int, str] = {int(k): v for k, v in model.config.id2label.items()}
    # Map model output -> GT class: "fall"->1, "no_fall"->0
    model_to_gt = {
        mid: (1 if "fall" in lbl and "no" not in lbl else 0)
        for mid, lbl in id2label.items()
    }

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    split_dir = dataset_path / data_config[split]
    label_dir = split_dir.parent / "labels"
    crops, gt_labels = _load_gt_crops(split_dir, label_dir)

    if not crops:
        return {"error": "No GT crops found for classification eval"}

    batch_size = 32
    pred_labels = []
    with torch.no_grad():
        for i in range(0, len(crops), batch_size):
            batch_imgs = torch.stack([transform(c) for c in crops[i:i + batch_size]])
            logits = model(batch_imgs.to(device)).logits
            preds = logits.argmax(dim=-1).cpu().tolist()
            pred_labels.extend([model_to_gt.get(p, p) for p in preds])

    gt_arr = np.array(gt_labels)
    pred_arr = np.array(pred_labels)
    accuracy = float((gt_arr == pred_arr).mean())

    per_class: dict[str, dict] = {}
    for cls_id, cls_name in GT_CLASSES.items():
        tp = int(((pred_arr == cls_id) & (gt_arr == cls_id)).sum())
        fp = int(((pred_arr == cls_id) & (gt_arr != cls_id)).sum())
        fn = int(((pred_arr != cls_id) & (gt_arr == cls_id)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[cls_name] = {
            "precision": prec, "recall": rec, "f1": f1,
            "support": int((gt_arr == cls_id).sum()),
        }

    return {
        "task": "classification_on_gt_crops",
        "model_id2label": id2label,
        "num_crops": len(crops),
        "accuracy": accuracy,
        "per_class": per_class,
        # map50 not applicable for classification; set None so report handles it
        "map50": None,
        "map50_95": None,
    }


# --- Report writing ----------------------------------------------------------

def write_report(results: list[dict], out_dir: Path, split: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    detection_ok = [r for r in results if r.get("status") == "ok"
                    and r.get("task") != "classification_on_gt_crops"
                    and r.get("map50") is not None]
    clf_ok = [r for r in results if r.get("status") == "ok"
              and r.get("task") == "classification_on_gt_crops"]
    errors = [r for r in results if r.get("status") != "ok"]

    detection_ok.sort(key=lambda r: r.get("map50") or 0.0, reverse=True)

    def _fmt(v: float | None) -> str:
        return f"{v:.3f}" if v is not None else "—"

    lines = [
        f"# Fall Detection — Pretrained Model Benchmark ({split})",
        "",
        f"**Dataset:** `dataset_store/training_ready/fall_detection`  ",
        f"**Split:** `{split}`  ",
        f"**GT classes:** `person` (0), `fallen_person` (1)",
        "",
        "## Detection Models (sorted by mAP50)",
        "",
        "| Rank | Model | mAP50 | mAP50-95 | P | R | AP_person | AP_fallen |",
        "|------|-------|-------|----------|---|---|-----------|-----------|",
    ]

    for rank, r in enumerate(detection_ok, 1):
        pc = r.get("per_class_ap50", {})
        note = f" _{r['note']}_" if r.get("note") else ""
        lines.append(
            f"| {rank} | `{r['model']}`{note} "
            f"| {_fmt(r.get('map50'))} "
            f"| {_fmt(r.get('map50_95'))} "
            f"| {_fmt(r.get('precision'))} "
            f"| {_fmt(r.get('recall'))} "
            f"| {_fmt(pc.get('person'))} "
            f"| {_fmt(pc.get('fallen_person'))} "
            f"|"
        )

    if clf_ok:
        lines += ["", "## Classification Models (evaluated on GT crops)", ""]
        for r in clf_ok:
            lines += [
                f"### `{r['model']}`",
                f"- **Accuracy:** {_fmt(r.get('accuracy'))}  ",
                f"- **Crops evaluated:** {r.get('num_crops', '?')}",
            ]
            for cls_name, m in (r.get("per_class") or {}).items():
                lines.append(
                    f"  - `{cls_name}`: P={_fmt(m.get('precision'))}  "
                    f"R={_fmt(m.get('recall'))}  F1={_fmt(m.get('f1'))}  "
                    f"n={m.get('support', '?')}"
                )

    if errors:
        lines += ["", "## Skipped / Errors", ""]
        for r in errors:
            lines.append(f"- `{r['model']}`: {r.get('error', r.get('status', 'unknown'))}")

    lines += ["", f"*Generated {time.strftime('%Y-%m-%d %H:%M')}*", ""]

    report_path = out_dir / "benchmark_report.md"
    report_path.write_text("\n".join(lines))
    print(f"Report written to {report_path}")


# --- Main -------------------------------------------------------------------

_SKIP_MODELS = {
    # Video / action recognition — need multi-frame input
    "videomae-base-finetuned-kinetics.bin",
    "videomae-small-finetuned-kinetics.bin",
    "x3d_xs.pyth", "x3d_l.pyth", "x3d_m.pyth", "x3d_s.pyth",
    "slowfast_r50_k400.pyth", "slowfast_r101_k400.pyth",
    "movinet_a1_base.tar.gz", "movinet_a2_base.tar.gz",
    "movinet_a3_base.tar.gz", "movinet_a2_stream.tar.gz",
    # General-purpose image classifiers — no fall vocabulary
    "dinov2-small.bin",
    "efficientnetv2_rw_s.ra2_in1k.bin",
    "mobilenetv4_conv_small.e2400_r224_in1k.bin",
    "_imagenet_classes.json",
}

_SKIP_REASON: dict[str, str] = {
    "videomae-base-finetuned-kinetics.bin": "video model — needs multi-frame input",
    "videomae-small-finetuned-kinetics.bin": "video model — needs multi-frame input",
    "x3d_xs.pyth": "video model — skip",
    "x3d_l.pyth": "video model — skip",
    "x3d_m.pyth": "video model — skip",
    "x3d_s.pyth": "video model — skip",
    "slowfast_r50_k400.pyth": "video model — skip",
    "slowfast_r101_k400.pyth": "video model — skip",
    "movinet_a1_base.tar.gz": "video model — skip",
    "movinet_a2_base.tar.gz": "video model — skip",
    "movinet_a3_base.tar.gz": "video model — skip",
    "movinet_a2_stream.tar.gz": "video model — skip",
    "dinov2-small.bin": "general image classifier — no fall class vocabulary",
    "efficientnetv2_rw_s.ra2_in1k.bin": "general image classifier — no fall class vocabulary",
    "mobilenetv4_conv_small.e2400_r224_in1k.bin": "general image classifier — no fall class vocabulary",
    "_imagenet_classes.json": "metadata file — skip",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark pretrained fall detection models")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--skip-yolox", action="store_true",
                        help="Skip COCO-pretrained YOLOX baselines (slow)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_config = _load_data_config()
    dataset_path = _resolve_dataset_path(data_config)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []

    # 1. Skipped models — record reason
    for fname, reason in _SKIP_REASON.items():
        all_results.append({
            "model": fname,
            "status": "skipped",
            "error": reason,
        })

    # 2. Ultralytics detection models
    detection_pts = [
        PRETRAINED_DIR / "yolov11_fall_melihuzunoglu.pt",
        PRETRAINED_DIR / "yolov8_fall_kamalchibrani.pt",
    ]

    for pt_path in detection_pts:
        label = pt_path.name
        print(f"\n--- Evaluating: {label} ---")
        result: dict[str, Any] = {"model": label, "path": str(pt_path)}
        if not pt_path.exists():
            result.update({"status": "error", "error": "file not found"})
            all_results.append(result)
            continue
        try:
            t0 = time.time()
            metrics = evaluate_ultralytics(pt_path, dataset_path, data_config,
                                           args.split, conf=args.conf)
            elapsed = time.time() - t0
            result.update({"status": "ok", "elapsed_s": round(elapsed, 1), **metrics})
            print(f"  mAP50={metrics.get('map50', 'N/A'):.3f}  ({elapsed:.0f}s)")
        except Exception as exc:
            result.update({"status": "error", "error": str(exc)})
            print(f"  ERROR: {exc}")
        all_results.append(result)

    # 3. ResNet18 fall classifier
    safetensors_path = PRETRAINED_DIR / "fall_resnet18_popkek00.safetensors"
    config_path = PRETRAINED_DIR / "fall_resnet18_popkek00_config.json"
    label = "fall_resnet18_popkek00.safetensors"
    print(f"\n--- Evaluating: {label} (classification on GT crops) ---")
    result = {"model": label, "path": str(safetensors_path)}
    try:
        t0 = time.time()
        metrics = evaluate_resnet_classifier(safetensors_path, config_path,
                                             dataset_path, data_config, args.split)
        elapsed = time.time() - t0
        result.update({"status": "ok", "elapsed_s": round(elapsed, 1), **metrics})
        print(f"  Accuracy={metrics.get('accuracy', 'N/A'):.3f}  ({elapsed:.0f}s)")
    except Exception as exc:
        result.update({"status": "error", "error": str(exc)})
        print(f"  ERROR: {exc}")
    all_results.append(result)

    # 4. YOLOX COCO baselines (person only)
    if not args.skip_yolox:
        for ckpt_name in ["yolox_s.pth", "yolox_m.pth"]:
            ckpt_path = PRETRAINED_ROOT / ckpt_name
            label = ckpt_name
            print(f"\n--- Evaluating: {label} (COCO person baseline) ---")
            result = {"model": label, "path": str(ckpt_path)}
            if not ckpt_path.exists():
                result.update({"status": "error", "error": "file not found"})
                all_results.append(result)
                continue
            try:
                t0 = time.time()
                metrics = evaluate_yolox_coco(ckpt_path, dataset_path, data_config, args.split)
                elapsed = time.time() - t0
                result.update({"status": "ok", "elapsed_s": round(elapsed, 1), **metrics})
                print(f"  person AP50={metrics.get('map50', 'N/A'):.3f}  ({elapsed:.0f}s)")
            except Exception as exc:
                result.update({"status": "error", "error": str(exc)})
                print(f"  ERROR: {exc}")
            all_results.append(result)

    # Serialise
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

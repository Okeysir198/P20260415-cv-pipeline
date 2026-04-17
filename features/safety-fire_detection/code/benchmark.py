"""Benchmark pretrained fire/smoke detection models on the fire_detection val/test splits.

Usage:
    uv run features/safety-fire_detection/code/benchmark.py
    uv run features/safety-fire_detection/code/benchmark.py --split val
    uv run features/safety-fire_detection/code/benchmark.py --split test
"""

import argparse
import json
import sys
import tempfile
import time
from datetime import date
from pathlib import Path
from typing import Optional

import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from utils.config import load_config

PRETRAINED_DIR = REPO / "pretrained" / "safety-fire_detection"
DATA_CONFIG_PATH = REPO / "features" / "safety-fire_detection" / "configs" / "05_data.yaml"
EVAL_DIR = REPO / "features" / "safety-fire_detection" / "eval"

GT_NAMES = {0: "fire", 1: "smoke"}

# Fuzzy class name aliases → GT class ID
FIRE_ALIASES = {"fire", "flame", "wildfire", "flames", "burning"}
SMOKE_ALIASES = {"smoke", "haze", "smog"}

# Models that are COCO-only detectors (no fire/smoke class)
COCO_DETECTOR_DIRS = {"ustc-community_dfine-medium-coco", "ustc-community_dfine-small-coco",
                       "deim_dfine_s_coco", "deim_dfine_m_coco"}

# Models that are image classifiers (not detectors)
CLASSIFIER_DIRS = {
    "shawnmichael_convnext-tiny-fire-smoke",
    "shawnmichael_efficientnetb2-fire-smoke",
    "shawnmichael_swin-fire-smoke",
    "shawnmichael_vit-fire-smoke-v4",
    "shawnmichael_vit-large-fire-smoke",
    "pyronear_resnet18",
    "pyronear_resnet34",
    "pyronear_rexnet1_0x",
    "pyronear_rexnet1_3x",
    "pyronear_rexnet1_5x",
    "pyronear_mobilenet_v3_large",
    "pyronear_mobilenet_v3_small",
    "dima806_wildfire_types",
    "Shoriful025_wildfire_smoke_seg_vit",
    "sequoiaandrade_smoke-cloud-race-odin",
}

# YOLOX root-level checkpoints — COCO 80-class, no fire/smoke
YOLOX_CHECKPOINTS = {
    "yolox_s": REPO / "pretrained" / "yolox_s.pth",
    "yolox_m": REPO / "pretrained" / "yolox_m.pth",
}

# Additional YOLO-family .pt model dirs beyond the task spec (discovered on disk)
EXTRA_PT_DIRS = {
    "JJUNHYEOK_yolov8n_wildfire",
    "Mehedi-2-96_fire-smoke-yolo",
    "TommyNgx_YOLOv10-Fire-and-Smoke",
    "touati-kamel_yolov10n-forest-fire",
    "pyronear_yolo11s_nimble-narwhal_v6",
}


def _build_temp_data_yaml(dataset_path: Path) -> str:
    content = {
        "path": str(dataset_path),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 2,
        "names": ["fire", "smoke"],
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir=tempfile.gettempdir()
    ) as f:
        yaml.dump(content, f)
        return f.name


def _build_class_map(model_names: dict) -> Optional[dict]:
    """Map model class IDs to GT class IDs (0=fire, 1=smoke). Returns None if no match."""
    class_map: dict[int, int] = {}
    for model_cls_id, model_cls_name in model_names.items():
        name_lower = model_cls_name.lower()
        if name_lower in FIRE_ALIASES:
            class_map[int(model_cls_id)] = 0
        elif name_lower in SMOKE_ALIASES:
            class_map[int(model_cls_id)] = 1
    return class_map if class_map else None


def _eval_ultralytics(
    model_path: Path, temp_yaml: str, split: str
) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    class_map = _build_class_map(model.names)
    if class_map is None:
        return {"status": "no_matching_classes", "model_names": dict(model.names)}

    t0 = time.perf_counter()
    results = model.val(data=temp_yaml, split=split, verbose=False, plots=False)
    latency_ms = (time.perf_counter() - t0) / max(results.speed.get("n", 1), 1) * 1000

    # Extract metrics — Ultralytics returns box.map50 / map50-95 / mp / mr
    box = results.box
    return {
        "status": "ok",
        "class_map": {str(k): v for k, v in class_map.items()},
        "map50": float(box.map50) if hasattr(box, "map50") else None,
        "map50_95": float(box.map) if hasattr(box, "map") else None,
        "precision": float(box.mp) if hasattr(box, "mp") else None,
        "recall": float(box.mr) if hasattr(box, "mr") else None,
        "latency_ms": round(latency_ms, 1),
    }


def _eval_yolox_checkpoint(arch: str, ckpt_path: Path) -> dict:
    """YOLOX .pth checkpoints are COCO 80-class — no fire/smoke class."""
    if not ckpt_path.exists():
        return {"status": "file_not_found"}
    return {
        "status": "no_matching_classes",
        "note": "COCO 80-class model — no fire/smoke class. Use as backbone for fine-tuning.",
        "arch": arch,
    }


def _find_pt_file(model_dir: Path) -> Optional[Path]:
    """Recursively find the first .pt file in a model directory, skipping optimizer.pt."""
    for pt in sorted(model_dir.rglob("*.pt")):
        if pt.stem not in ("optimizer", "scheduler"):
            return pt
    return None


def _count_images(dataset_path: Path, split: str) -> int:
    split_dir = dataset_path / split / "images"
    if not split_dir.exists():
        return 0
    return sum(1 for p in split_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


# ONNX pyronear classifiers: single-class "Wildfire" binary sigmoid output
PYRONEAR_ONNX_DIRS = {
    "pyronear_resnet18",
    "pyronear_resnet34",
    "pyronear_rexnet1_0x",
    "pyronear_rexnet1_3x",
    "pyronear_rexnet1_5x",
    "pyronear_mobilenet_v3_large",
    "pyronear_mobilenet_v3_small",
}

SHAWNMICHAEL_DIRS = {
    "shawnmichael_convnext-tiny-fire-smoke",
    "shawnmichael_efficientnetb2-fire-smoke",
    "shawnmichael_swin-fire-smoke",
    "shawnmichael_vit-fire-smoke-v4",
    "shawnmichael_vit-large-fire-smoke",
}

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _preprocess_imagenet(img_bgr, size: int = 224):
    """Resize, convert BGR→RGB, normalize ImageNet, return (1,3,H,W) float32."""
    import cv2
    import numpy as np

    img = cv2.resize(img_bgr, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array(_IMAGENET_MEAN, dtype=np.float32)
    std = np.array(_IMAGENET_STD, dtype=np.float32)
    img = (img - mean) / std
    return img.transpose(2, 0, 1)[np.newaxis, ...]  # (1,3,H,W)


def _image_has_fire_or_smoke(label_path: Path) -> bool:
    """Return True if the YOLO label file has at least one fire (cls=0) or smoke (cls=1) box."""
    if not label_path.exists():
        return False
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if parts and int(parts[0]) in (0, 1):
            return True
    return False


def _collect_val_images(dataset_path: Path) -> list[tuple[Path, bool]]:
    """Return [(img_path, gt_positive), ...] for val split."""
    img_dir = dataset_path / "val" / "images"
    lbl_dir = dataset_path / "val" / "labels"
    if not img_dir.exists():
        return []
    pairs = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label_path = lbl_dir / (img_path.stem + ".txt")
        gt_positive = _image_has_fire_or_smoke(label_path)
        pairs.append((img_path, gt_positive))
    return pairs


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _eval_pyronear_onnx(model_dir: Path, val_pairs: list[tuple[Path, bool]]) -> dict:
    """Binary wildfire ONNX classifier: output shape (1,1), sigmoid → probability."""
    import cv2
    import numpy as np
    import onnxruntime as ort

    onnx_path = model_dir / "model.onnx"
    if not onnx_path.exists():
        return {"status": "no_onnx_file"}

    config_path = model_dir / "config.json"
    classes = ["Wildfire"]
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        classes = cfg.get("classes", classes)

    # Pyronear models: output (1,1) logit → sigmoid → wildfire probability
    is_wildfire_class = any(
        c.lower() in {"wildfire", "fire", "smoke", "flame"} for c in classes
    )

    try:
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        return {"status": "error", "error_msg": str(exc)}

    input_name = sess.get_inputs()[0].name
    threshold = 0.5  # sigmoid output threshold

    tp = fp = fn = tn = 0
    latencies: list[float] = []

    for img_path, gt_positive in val_pairs:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        blob = _preprocess_imagenet(img_bgr)

        t0 = time.perf_counter()
        try:
            raw = sess.run(None, {input_name: blob})[0]
        except Exception:
            continue
        latencies.append((time.perf_counter() - t0) * 1000)

        # Raw is (1,1) logit → sigmoid
        prob = float(1 / (1 + np.exp(-raw.flat[0])))
        predicted_positive = prob >= threshold if is_wildfire_class else False

        if gt_positive and predicted_positive:
            tp += 1
        elif not gt_positive and predicted_positive:
            fp += 1
        elif gt_positive and not predicted_positive:
            fn += 1
        else:
            tn += 1

    precision, recall, f1 = _precision_recall_f1(tp, fp, fn)
    total = tp + fp + fn + tn
    return {
        "status": "ok",
        "classes": classes,
        "threshold": threshold,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "accuracy": round((tp + tn) / total, 3) if total > 0 else None,
        "latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
        "n_images": total,
    }


def _run_classifier_benchmark(dataset_path: Path) -> tuple[list[dict], list[dict]]:
    """Evaluate ONNX pyronear classifiers; skip HF safetensors classifiers."""
    val_pairs = _collect_val_images(dataset_path)
    print(f"  Classifier eval on {len(val_pairs)} val images "
          f"({sum(1 for _, g in val_pairs if g)} positive / "
          f"{sum(1 for _, g in val_pairs if not g)} negative)")

    clf_results: list[dict] = []
    clf_skipped: list[dict] = []

    for name in sorted(CLASSIFIER_DIRS):
        model_dir = PRETRAINED_DIR / name

        if name in SHAWNMICHAEL_DIRS:
            clf_skipped.append({
                "model": name,
                "reason": "HF safetensors classifier — eval skipped for time",
            })
            continue

        if not model_dir.exists():
            clf_skipped.append({"model": name, "reason": "Directory not found"})
            continue

        if name in PYRONEAR_ONNX_DIRS:
            print(f"  Evaluating ONNX classifier: {name}")
            try:
                result = _eval_pyronear_onnx(model_dir, val_pairs)
            except Exception as exc:
                result = {"status": "error", "error_msg": str(exc)}
            clf_results.append({"model": name, "framework": "ONNX (pyronear)", **result})
        else:
            clf_skipped.append({"model": name, "reason": "No ONNX model — skipped"})

    return clf_results, clf_skipped


def _run_benchmark(splits: list[str]) -> tuple[list[dict], list[dict]]:
    data_config = load_config(DATA_CONFIG_PATH)
    dataset_path = (DATA_CONFIG_PATH.parent / data_config["path"]).resolve()
    temp_yaml = _build_temp_data_yaml(dataset_path)

    results: list[dict] = []
    skipped: list[dict] = []

    # --- YOLOX root checkpoints ---
    for arch, ckpt_path in YOLOX_CHECKPOINTS.items():
        entry = {"model": arch, "framework": "YOLOX (PyTorch)", "path": str(ckpt_path)}
        for split in splits:
            entry[split] = _eval_yolox_checkpoint(arch, ckpt_path)
        _route_entry(entry, splits, results, skipped,
                     skip_reason="COCO 80-class, no fire/smoke class — backbone baseline")

    # --- Pretrained model directories ---
    if not PRETRAINED_DIR.exists():
        print(f"Warning: pretrained dir not found: {PRETRAINED_DIR}", file=sys.stderr)
        return results, skipped

    for item in sorted(PRETRAINED_DIR.iterdir()):
        # Gap 1: handle .pth files directly in PRETRAINED_DIR (e.g. yolox_m.pth)
        if item.is_file() and item.suffix == ".pth":
            skipped.append({
                "model": item.stem,
                "reason": (
                    "YOLOX checkpoint directly in pretrained dir — COCO 80-class, "
                    "no fire/smoke class. Use as backbone for fine-tuning."
                ),
            })
            continue

        if not item.is_dir():
            continue
        name = item.name
        model_dir = item

        if name in COCO_DETECTOR_DIRS:
            skipped.append({"model": name, "reason": "COCO-only detector, no fire/smoke class"})
            continue

        if name in CLASSIFIER_DIRS:
            # Handled separately in _run_classifier_benchmark
            continue

        pt_file = _find_pt_file(model_dir)
        if pt_file is None:
            skipped.append({"model": name, "reason": "No .pt file found"})
            continue

        entry: dict = {"model": name, "framework": "Ultralytics", "path": str(pt_file)}
        all_no_match = True
        for split in splits:
            try:
                split_result = _eval_ultralytics(pt_file, temp_yaml, split)
                entry[split] = split_result
                if split_result.get("status") == "ok":
                    all_no_match = False
            except Exception as exc:
                entry[split] = {"status": "error", "error_msg": str(exc)}
                all_no_match = False  # count errors as attempted

        if all_no_match and all(
            entry.get(s, {}).get("status") == "no_matching_classes" for s in splits
        ):
            model_names = entry.get(splits[0], {}).get("model_names", {})
            skipped.append({
                "model": name,
                "reason": f"No fire/smoke class in model. Model classes: {model_names}",
            })
        else:
            results.append(entry)

    # Clean up temp yaml
    Path(temp_yaml).unlink(missing_ok=True)
    return results, skipped


def _route_entry(
    entry: dict,
    splits: list[str],
    results: list[dict],
    skipped: list[dict],
    skip_reason: str,
) -> None:
    all_no_match = all(
        entry.get(s, {}).get("status") == "no_matching_classes" for s in splits
    )
    if all_no_match:
        skipped.append({"model": entry["model"], "reason": skip_reason})
    else:
        results.append(entry)


def _primary_metric(entry: dict, split: str) -> float:
    s = entry.get(split, {})
    v = s.get("map50")
    return v if v is not None else -1.0


def _write_classifier_section(
    clf_results: list[dict],
    clf_skipped: list[dict],
) -> str:
    lines = [
        "",
        "## Classification Models",
        "",
        "Image-level binary classifiers evaluated on val split "
        "(GT positive = image has ≥1 fire or smoke bbox).",
        "",
    ]

    ok = [r for r in clf_results if r.get("status") == "ok"]
    if ok:
        sorted_ok = sorted(ok, key=lambda r: r.get("f1", -1.0), reverse=True)
        lines.append("| Model | Classes | Precision | Recall | F1 | Accuracy | Latency ms |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for r in sorted_ok:
            lines.append(
                f"| {r['model']} | {', '.join(r.get('classes', []))} "
                f"| {r.get('precision', '—')} | {r.get('recall', '—')} "
                f"| {r.get('f1', '—')} | {r.get('accuracy', '—')} "
                f"| {r.get('latency_ms', '—')} |"
            )
    else:
        lines.append("No ONNX classifiers evaluated successfully.")

    error_models = [r for r in clf_results if r.get("status") != "ok"]
    if error_models:
        lines += ["", "### Classifier errors", ""]
        for r in error_models:
            lines.append(f"- {r['model']}: {r.get('error_msg', r.get('status'))}")

    if clf_skipped:
        lines += ["", "### Skipped classifiers", "", "| Model | Reason |", "| --- | --- |"]
        for s in clf_skipped:
            lines.append(f"| {s['model']} | {s['reason']} |")

    return "\n".join(lines)


def _write_report(
    results: list[dict],
    skipped: list[dict],
    splits: list[str],
    val_count: int,
    test_count: int,
    clf_results: list[dict] | None = None,
    clf_skipped: list[dict] | None = None,
) -> str:
    primary_split = splits[0]
    sorted_results = sorted(results, key=lambda e: _primary_metric(e, primary_split), reverse=True)

    lines = [
        "# Fire Detection — Pretrained Model Benchmark",
        f"Date: {date.today().isoformat()}",
        f"Dataset: fire_detection (val: {val_count} images, test: {test_count} images)",
        "",
        "## Results (sorted by val mAP50)",
        "",
    ]

    headers = ["Model", "Framework", "val mAP50", "val mAP50-95", "val P", "val R"]
    if "test" in splits:
        headers += ["test mAP50", "test mAP50-95", "test P", "test R"]
    headers += ["Latency ms", "Status"]

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
            fmt(val_s.get("map50")),
            fmt(val_s.get("map50_95")),
            fmt(val_s.get("precision")),
            fmt(val_s.get("recall")),
        ]
        if "test" in splits:
            test_s = entry.get("test", {})
            row += [
                fmt(test_s.get("map50")),
                fmt(test_s.get("map50_95")),
                fmt(test_s.get("precision")),
                fmt(test_s.get("recall")),
            ]
        row += [
            fmt(val_s.get("latency_ms")),
            status,
        ]
        lines.append(sep.join(row))

    lines += ["", "## Skipped models", "", "| Model | Reason |", "| --- | --- |"]
    for s in skipped:
        lines.append(f"| {s['model']} | {s['reason']} |")

    # Recommendation
    ok_results = [e for e in sorted_results if e.get("val", {}).get("status") == "ok"]
    lines.append("")
    lines.append("## Recommendation")
    if ok_results:
        best = ok_results[0]
        best_map = best.get("val", {}).get("map50", 0.0)
        lines.append(f"Best backbone: **{best['model']}** (val mAP50: {best_map:.3f})")
    else:
        lines.append("No successfully evaluated models — check errors above.")

    if clf_results is not None:
        lines.append(_write_classifier_section(clf_results, clf_skipped or []))

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test", "both"], default="both")
    args = parser.parse_args()

    splits = ["val", "test"] if args.split == "both" else [args.split]

    dataset_path = (
        DATA_CONFIG_PATH.parent / load_config(DATA_CONFIG_PATH)["path"]
    ).resolve()
    val_count = _count_images(dataset_path, "val")
    test_count = _count_images(dataset_path, "test")

    print(f"Benchmarking on splits: {splits}")
    print(f"Dataset: {dataset_path} (val={val_count}, test={test_count})")

    results, skipped = _run_benchmark(splits)

    print("\n--- Classifier benchmark ---")
    clf_results, clf_skipped = _run_classifier_benchmark(dataset_path)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    json_out = EVAL_DIR / "benchmark_results.json"
    json_out.write_text(
        json.dumps(
            {"results": results, "skipped": skipped,
             "classifiers": clf_results, "classifiers_skipped": clf_skipped},
            indent=2, default=str,
        )
    )
    print(f"JSON results: {json_out}")

    report = _write_report(results, skipped, splits, val_count, test_count,
                           clf_results, clf_skipped)
    md_out = EVAL_DIR / "benchmark_report.md"
    md_out.write_text(report)
    print(f"Markdown report: {md_out}")

    # Quick summary to stdout
    ok = [e for e in results if e.get("val", {}).get("status") == "ok"]
    print(f"\nEvaluated {len(ok)} detector models successfully, {len(skipped)} skipped.")
    clf_ok = [r for r in clf_results if r.get("status") == "ok"]
    print(f"Evaluated {len(clf_ok)} ONNX classifiers, {len(clf_skipped)} skipped.")
    if ok:
        best = max(ok, key=lambda e: e.get("val", {}).get("map50") or -1.0)
        print(f"Best detector: {best['model']} — val mAP50={best['val'].get('map50', '?'):.3f}")
    if clf_ok:
        best_clf = max(clf_ok, key=lambda r: r.get("f1", -1.0))
        print(f"Best classifier: {best_clf['model']} — F1={best_clf.get('f1', '?')}")


if __name__ == "__main__":
    main()

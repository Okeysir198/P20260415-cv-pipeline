"""Pretrained model benchmark for safety-poketenashi-phone-usage.

Evaluates COCO-pretrained YOLOX models on the phone-usage detection dataset.
Since phone_usage is not a COCO class, only the person class (COCO class 0
→ our class 0) is evaluated. phone_usage (class 1) requires fine-tuning.

Also probes for any Ultralytics .pt files in the shared pretrained root.

Output:
    features/safety-poketenashi-phone-usage/eval/benchmark_results.json
    features/safety-poketenashi-phone-usage/eval/benchmark_report.md
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from utils.config import load_config
from core.p06_models import build_model
from core.p08_evaluation.evaluator import ModelEvaluator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRETRAINED_ROOT = REPO / "pretrained"
EVAL_DIR = REPO / "features" / "safety-poketenashi-phone-usage" / "eval"
DATA_CONFIG_PATH = REPO / "features" / "safety-poketenashi-phone-usage" / "configs" / "05_data.yaml"

# COCO class 0 = "person" → maps to our dataset class 0 (person)
# COCO has no equivalent for our class 1 (phone_usage)
COCO_PERSON_CLASS_ID = 0

YOLOX_MODELS = [
    {"name": "yolox_s", "arch": "yolox-s", "pth": PRETRAINED_ROOT / "yolox_s.pth"},
    {"name": "yolox_m", "arch": "yolox-m", "pth": PRETRAINED_ROOT / "yolox_m.pth"},
]

PHONE_PRETRAINED_DIR = PRETRAINED_ROOT / "safety-poketenashi-phone-usage"
POKETENASHI_PRETRAINED_DIR = PRETRAINED_ROOT / "safety-poketenashi"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_device() -> torch.device:
    # Use CPU to avoid fragmentation with multi-model benchmarks
    return torch.device("cpu")


def load_yolox_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    """Load a COCO-pretrained YOLOX .pth checkpoint into model in-place."""
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state = ckpt.get("model", ckpt)
    # Strip "module." prefix from DDP checkpoints
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys (head class mismatch expected for COCO→custom)")


def filter_predictions_to_coco_person(
    predictions: list[dict], coco_num_classes: int = 80
) -> list[dict]:
    """Keep only COCO class-0 (person) detections and remap label to 0.

    YOLOX with 80 COCO classes emits label indices 0–79.  Class 0 = person.
    We keep only those detections and relabel them as our class 0 (person).
    """
    filtered = []
    for pred in predictions:
        boxes = pred.get("boxes", np.empty((0, 4)))
        scores = pred.get("scores", np.empty(0))
        labels = pred.get("labels", np.empty(0, dtype=np.int64))

        mask = labels == COCO_PERSON_CLASS_ID
        filtered.append({
            "boxes": boxes[mask] if len(boxes) else boxes,
            "scores": scores[mask] if len(scores) else scores,
            "labels": labels[mask].copy() if len(labels) else labels,
        })
    return filtered


def filter_gt_to_person_class(ground_truths: list[dict]) -> list[dict]:
    """Keep only GT boxes with class 0 (person) for person-only evaluation."""
    filtered = []
    for gt in ground_truths:
        labels = gt.get("labels", np.empty(0, dtype=np.int64))
        boxes = gt.get("boxes", np.empty((0, 4)))
        mask = labels == 0
        filtered.append({
            "boxes": boxes[mask] if len(boxes) else boxes,
            "labels": labels[mask] if len(labels) else labels,
        })
    return filtered


def compute_person_only_map(
    predictions: list[dict],
    ground_truths: list[dict],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute mAP restricted to person class only."""
    from core.p08_evaluation.sv_metrics import compute_map

    # Wrap in a 1-class config: only class 0
    person_preds = filter_predictions_to_coco_person(predictions)
    person_gts = filter_gt_to_person_class(ground_truths)

    return compute_map(person_preds, person_gts, iou_threshold=iou_threshold, num_classes=1)


def probe_ultralytics_pt_files() -> list[Path]:
    """Return .pt files in pretrained root and feature sub-dirs (if any)."""
    candidates: list[Path] = []
    for search_dir in [PRETRAINED_ROOT, PHONE_PRETRAINED_DIR, POKETENASHI_PRETRAINED_DIR]:
        if search_dir.exists():
            candidates.extend(search_dir.glob("*.pt"))
    return candidates


def benchmark_yolox_model(spec: dict, data_config: dict, device: torch.device) -> dict[str, Any]:
    """Run a single YOLOX COCO-pretrained model benchmark.

    Returns a result dict suitable for the JSON report.
    """
    name = spec["name"]
    arch = spec["arch"]
    pth = spec["pth"]

    result: dict[str, Any] = {"name": name, "arch": arch, "checkpoint": str(pth)}

    if not pth.exists():
        result["status"] = "error"
        result["error"] = f"checkpoint not found: {pth}"
        return result

    try:
        print(f"\n[{name}] Building model (80 COCO classes)...")
        cfg = {"model": {"arch": arch, "num_classes": 80, "input_size": [640, 640]}}
        model = build_model(cfg)
        load_yolox_checkpoint(model, pth, device)
        model.to(device).eval()

        # Use ModelEvaluator with a 2-class data config (our dataset) but
        # we override the class count to 80 for the model forward pass.
        # We collect raw predictions then filter to person class before metrics.
        evaluator = ModelEvaluator(
            model,
            data_config,
            device=device,
            conf_threshold=0.35,
            iou_threshold=0.45,
            output_format="yolox",
        )
        # Temporarily override num_classes so evaluator doesn't apply 2-class
        # masking internally — we do the person-class filtering ourselves.
        evaluator.num_classes = 80
        evaluator.class_names = {i: f"coco_{i}" for i in range(80)}

        print(f"[{name}] Running inference on val split...")
        t0 = time.perf_counter()
        predictions, ground_truths = evaluator.get_predictions(split="val")
        latency_s = time.perf_counter() - t0
        n_images = len(predictions)

        print(f"[{name}] Computing person-class mAP...")
        metrics = compute_person_only_map(predictions, ground_truths, iou_threshold=0.5)

        result.update({
            "status": "ok",
            "num_images_evaluated": n_images,
            "inference_time_s": round(latency_s, 2),
            "ms_per_image": round(latency_s / max(n_images, 1) * 1000, 1),
            "coco_classes": 80,
            "evaluated_class": "person (COCO class 0 → our class 0)",
            "skipped_class": "phone_usage (class 1) — no COCO equivalent, fine-tuning required",
            "person_mAP50": round(float(metrics.get("map50", 0.0)), 4),
            "person_mAP50_95": round(float(metrics.get("map", 0.0)), 4),
        })

    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)

    return result


def benchmark_ultralytics_model(pt_path: Path, data_config: dict, device: torch.device) -> dict[str, Any]:
    """Try loading a .pt file via Ultralytics and run val on person class."""
    result: dict[str, Any] = {"name": pt_path.name, "checkpoint": str(pt_path)}
    try:
        from ultralytics import YOLO  # type: ignore[import]

        print(f"\n[ultralytics] Loading {pt_path.name}...")
        yolo = YOLO(str(pt_path))
        yolo.to(device)

        # Count detectable person instances on val split as a proxy metric
        val_images_dir = (
            REPO / "dataset_store" / "training_ready" / "safety_poketenashi_phone_usage"
            / "val" / "images"
        )
        if not val_images_dir.exists():
            result["status"] = "error"
            result["error"] = f"val images dir not found: {val_images_dir}"
            return result

        image_paths = list(val_images_dir.glob("*.*"))[:50]  # cap at 50 images
        person_detections = 0
        t0 = time.perf_counter()
        for img_path in image_paths:
            results = yolo(str(img_path), verbose=False, conf=0.35)
            for r in results:
                if r.boxes is not None:
                    cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                    person_detections += int((cls_ids == 0).sum())
        latency_s = time.perf_counter() - t0

        result.update({
            "status": "ok",
            "num_images_sampled": len(image_paths),
            "person_detections_total": person_detections,
            "ms_per_image": round(latency_s / max(len(image_paths), 1) * 1000, 1),
            "note": "Qualitative only — mAP not computed for Ultralytics .pt probe",
        })

    except ImportError:
        result["status"] = "skipped"
        result["error"] = "ultralytics not installed"
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def write_json(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON results → {path}")


def write_markdown_report(results: dict, path: Path) -> None:
    lines = [
        "# Benchmark Report — safety-poketenashi-phone-usage",
        "",
        "## Summary",
        "",
        "> **phone_usage class has no pretrained equivalent; fine-tuning required for class 1 detection.**",
        "> Person detection (class 0) metrics provided as baseline using COCO-pretrained YOLOX models.",
        "> COCO class 0 (`person`) maps directly to our dataset class 0 (`person`).",
        "> Our class 1 (`phone_usage`) represents the *act* of using a phone while walking — not a COCO class.",
        "",
        f"- Dataset: `{results.get('dataset_name')}`",
        f"- Split evaluated: `{results.get('split')}`",
        f"- Evaluation date: {results.get('date')}",
        "",
        "## YOLOX COCO-pretrained Results",
        "",
        "| Model | Status | Images | ms/img | person mAP@50 | person mAP@50:95 |",
        "|-------|--------|--------|--------|--------------|-----------------|",
    ]

    for r in results.get("yolox_models", []):
        status = r.get("status", "?")
        if status == "ok":
            lines.append(
                f"| {r['name']} | ok | {r.get('num_images_evaluated', '-')} "
                f"| {r.get('ms_per_image', '-')} | {r.get('person_mAP50', '-')} "
                f"| {r.get('person_mAP50_95', '-')} |"
            )
        else:
            lines.append(f"| {r['name']} | {status} | — | — | — | — |")

    lines += [
        "",
        "## Ultralytics .pt Probe Results",
        "",
    ]

    if results.get("ultralytics_pt_files"):
        lines += [
            "| File | Status | Images Sampled | Person Detections | ms/img |",
            "|------|--------|---------------|------------------|--------|",
        ]
        for r in results["ultralytics_pt_files"]:
            if r.get("status") == "ok":
                lines.append(
                    f"| {r['name']} | ok | {r.get('num_images_sampled', '-')} "
                    f"| {r.get('person_detections_total', '-')} | {r.get('ms_per_image', '-')} |"
                )
            else:
                err = r.get("error", r.get("status"))
                lines.append(f"| {r['name']} | {r.get('status')} | — | — | — |")
                lines.append(f"|  | _error: {err}_ | | | |")
    else:
        lines.append("_No .pt files found in pretrained root or feature sub-dirs._")

    lines += [
        "",
        "## Notes",
        "",
        "- **COCO pretrained baseline**: YOLOX-S/M were pretrained on COCO 80 classes.",
        "  Only `person` (COCO class 0) has a direct counterpart in our 2-class dataset.",
        "- **phone_usage requires fine-tuning**: `phone_usage` represents a behavioral action",
        "  (walking while using a phone), not a physical object class in COCO.",
        "  Expected mAP for phone_usage with zero-shot COCO models: ~0.00.",
        "- **Recommended next step**: Fine-tune YOLOX-S or YOLOX-M backbone on the",
        "  `safety_poketenashi_phone_usage` training split.",
        "  See `features/safety-poketenashi-phone-usage/configs/06_training.yaml` (to be created).",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"Markdown report → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import datetime

    device = get_device()
    print(f"Device: {device}")

    data_config = load_config(DATA_CONFIG_PATH)
    data_config["_config_dir"] = DATA_CONFIG_PATH.parent

    results: dict[str, Any] = {
        "dataset_name": data_config.get("dataset_name"),
        "split": "val",
        "date": datetime.date.today().isoformat(),
        "device": str(device),
        "note": (
            "phone_usage class has no pretrained equivalent; fine-tuning required "
            "for class 1 detection. Person detection (class 0) metrics provided as baseline."
        ),
        "yolox_models": [],
        "ultralytics_pt_files": [],
    }

    # --- YOLOX COCO-pretrained benchmarks ---
    for spec in YOLOX_MODELS:
        result = benchmark_yolox_model(spec, data_config, device)
        results["yolox_models"].append(result)

    # --- Ultralytics .pt probe ---
    pt_files = probe_ultralytics_pt_files()
    if pt_files:
        print(f"\nFound {len(pt_files)} .pt file(s) to probe: {[p.name for p in pt_files]}")
        for pt_path in pt_files:
            result = benchmark_ultralytics_model(pt_path, data_config, device)
            results["ultralytics_pt_files"].append(result)
    else:
        print("\nNo .pt files found in pretrained root or feature sub-dirs — skipping Ultralytics probe.")

    # --- Write outputs ---
    write_json(results, EVAL_DIR / "benchmark_results.json")
    write_markdown_report(results, EVAL_DIR / "benchmark_report.md")
    print("\nDone.")


if __name__ == "__main__":
    main()

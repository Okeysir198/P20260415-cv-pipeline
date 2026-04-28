"""CPPE-5 test-set benchmark: PyTorch fp32 vs ONNX fp32 vs ONNX int8 (QDQ).

All backends run on GPU. Reports per-image latency (median/P95) and COCO
mAP / mAP_50 against ground-truth test labels.

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run scripts/benchmark_detr_fp32_int8_vs_pytorch.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torchmetrics.detection import MeanAveragePrecision
from transformers import AutoConfig, AutoImageProcessor, AutoModelForObjectDetection

REPO = Path(__file__).resolve().parent.parent

RUNS = [
    (
        "RT-DETRv2-R50",
        REPO / "notebooks/detr_finetune_reference/our_rtdetr_v2_torchvision/runs/seed42",
    ),
    (
        "D-FINE-large (50ep)",
        REPO / "notebooks/detr_finetune_reference/our_dfine_torchvision/runs/seed42_50ep",
    ),
]

TEST_IMG_DIR = REPO / "dataset_store/training_ready/cppe5/test/images"
TEST_LBL_DIR = REPO / "dataset_store/training_ready/cppe5/test/labels"
CONF_THRESHOLD = 0.3
WARMUP = 10
LATENCY_REPEATS = 3  # per image, take median across repeats
DEVICE = "cuda"


def load_gt() -> list[dict]:
    """Load test set: list of dicts with image path, pixel-space GT boxes, labels."""
    items = []
    for img_path in sorted(TEST_IMG_DIR.glob("*.jpg")):
        with Image.open(img_path) as im:
            W, H = im.size
        boxes_xyxy, labels = [], []
        lbl_path = TEST_LBL_DIR / f"{img_path.stem}.txt"
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                p = line.strip().split()
                if len(p) != 5:
                    continue
                cls = int(float(p[0]))
                cx, cy, bw, bh = (float(x) for x in p[1:])
                x1 = (cx - bw / 2) * W
                y1 = (cy - bh / 2) * H
                x2 = (cx + bw / 2) * W
                y2 = (cy + bh / 2) * H
                boxes_xyxy.append([x1, y1, x2, y2])
                labels.append(cls)
        items.append({
            "path": img_path,
            "size": (W, H),
            "gt_boxes": torch.tensor(boxes_xyxy, dtype=torch.float32) if boxes_xyxy
                        else torch.zeros((0, 4), dtype=torch.float32),
            "gt_labels": torch.tensor(labels, dtype=torch.int64),
        })
    return items


def _cuda_time_ms(fn, repeats: int) -> float:
    """Median wall-clock ms across `repeats` calls, with CUDA sync each side."""
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return statistics.median(times)


def _load_hf_model(run_dir: Path):
    """Load a model saved by our in-repo HF backend — strips the `hf_model.`
    prefix that `core/p06_models/hf_model.py::HFModelWrapper` adds to every key.

    Plain `AutoModelForObjectDetection.from_pretrained(run_dir)` silently
    re-initializes every weight because the prefix prevents key matching.
    """
    config = AutoConfig.from_pretrained(run_dir)
    model = AutoModelForObjectDetection.from_config(config)
    sd = torch.load(run_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
    if any(k.startswith("hf_model.") for k in sd):
        sd = {k.removeprefix("hf_model."): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"    load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        if missing[:3]:
            print(f"      missing sample: {missing[:3]}")
        if unexpected[:3]:
            print(f"      unexpected sample: {unexpected[:3]}")
    return model


def bench_pytorch(run_dir: Path, gt: list[dict]) -> dict:
    model = _load_hf_model(run_dir).to(DEVICE).eval()
    processor = AutoImageProcessor.from_pretrained(run_dir)

    # Warmup on first image
    first = Image.open(gt[0]["path"]).convert("RGB")
    warm_inputs = processor(images=first, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        for _ in range(WARMUP):
            _ = model(**warm_inputs)
    torch.cuda.synchronize()

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    latencies = []

    with torch.inference_mode():
        for item in gt:
            img = Image.open(item["path"]).convert("RGB")
            W, H = item["size"]
            inputs = processor(images=img, return_tensors="pt").to(DEVICE)
            target_sizes = torch.tensor([[H, W]], device=DEVICE)

            def _fwd_and_post():
                out = model(**inputs)
                processor.post_process_object_detection(
                    out, target_sizes=target_sizes, threshold=0.001
                )

            lat = _cuda_time_ms(_fwd_and_post, LATENCY_REPEATS)
            latencies.append(lat)

            # Final pass for predictions (kept)
            out = model(**inputs)
            post = processor.post_process_object_detection(
                out, target_sizes=target_sizes, threshold=0.001
            )[0]
            metric.update(
                [{
                    "boxes": post["boxes"].detach().cpu(),
                    "scores": post["scores"].detach().cpu(),
                    "labels": post["labels"].detach().cpu(),
                }],
                [{"boxes": item["gt_boxes"], "labels": item["gt_labels"]}],
            )

    del model
    torch.cuda.empty_cache()
    m = metric.compute()
    return {
        "latency_median_ms": statistics.median(latencies),
        "latency_p95_ms": sorted(latencies)[int(len(latencies) * 0.95)],
        "fps": 1000 / statistics.median(latencies),
        "map": float(m["map"]),
        "map_50": float(m["map_50"]),
        "map_75": float(m["map_75"]),
    }


def bench_onnx(
    onnx_path: Path,
    run_dir: Path,
    gt: list[dict],
    tag: str,
) -> dict:
    processor = AutoImageProcessor.from_pretrained(run_dir)

    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3
    sess = ort.InferenceSession(
        str(onnx_path), sess_options=sess_opts,
        providers=["CUDAExecutionProvider"],
    )
    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]

    # Warmup on first image
    first = Image.open(gt[0]["path"]).convert("RGB")
    warm = processor(images=first, return_tensors="np")
    warm_pv = warm["pixel_values"].astype(np.float32)
    for _ in range(WARMUP):
        _ = sess.run(None, {input_name: warm_pv})

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    latencies = []

    for item in gt:
        img = Image.open(item["path"]).convert("RGB")
        W, H = item["size"]
        inputs = processor(images=img, return_tensors="np")
        pv = inputs["pixel_values"].astype(np.float32)

        def _fwd():
            sess.run(None, {input_name: pv})

        lat = _cuda_time_ms(_fwd, LATENCY_REPEATS)
        latencies.append(lat)

        # Final pass for predictions — reuse HF post-processor by wrapping
        # ORT outputs in a ModelOutput-like namespace.
        raw = sess.run(None, {input_name: pv})
        raw_dict = dict(zip(output_names, raw))
        out = SimpleNamespace(
            logits=torch.from_numpy(raw_dict["logits"]).to(DEVICE),
            pred_boxes=torch.from_numpy(raw_dict["pred_boxes"]).to(DEVICE),
        )
        target_sizes = torch.tensor([[H, W]], device=DEVICE)
        post = processor.post_process_object_detection(
            out, target_sizes=target_sizes, threshold=0.001
        )[0]
        metric.update(
            [{
                "boxes": post["boxes"].detach().cpu(),
                "scores": post["scores"].detach().cpu(),
                "labels": post["labels"].detach().cpu(),
            }],
            [{"boxes": item["gt_boxes"], "labels": item["gt_labels"]}],
        )

    del sess
    torch.cuda.empty_cache()
    m = metric.compute()
    return {
        "backend_tag": tag,
        "latency_median_ms": statistics.median(latencies),
        "latency_p95_ms": sorted(latencies)[int(len(latencies) * 0.95)],
        "fps": 1000 / statistics.median(latencies),
        "map": float(m["map"]),
        "map_50": float(m["map_50"]),
        "map_75": float(m["map_75"]),
    }


def _verdict(int8: dict, fp32: dict, *, speedup_threshold: float = 1.2,
             map_drop_threshold: float = 0.01) -> tuple[str, str]:
    """Decide whether INT8 beats fp32. Returns (label, reason)."""
    if not int8 or not fp32:
        return "INT8 N/A", "no fp32 baseline"
    speedup = fp32["latency_median_ms"] / int8["latency_median_ms"] if int8["latency_median_ms"] > 0 else 0.0
    map_drop = fp32["map"] - int8["map"]
    if speedup >= speedup_threshold and map_drop <= map_drop_threshold + 1e-6:
        return "INT8 OK", f"{speedup:.2f}x speedup, mAP drop {map_drop:+.3f}"
    return "INT8 NOT VIABLE", f"{speedup:.2f}x speedup, mAP drop {map_drop:+.3f} — use fp32"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true",
                        help="Exit nonzero if any INT8 backend is NOT VIABLE.")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    assert "CUDAExecutionProvider" in ort.get_available_providers(), "ORT CUDA EP required"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"ORT: {ort.__version__}")
    print(f"Test set: {TEST_IMG_DIR} ({len(list(TEST_IMG_DIR.glob('*.jpg')))} images)\n")

    gt = load_gt()
    print(f"Loaded {len(gt)} GT items")
    print(f"Note: latency measured per-image on native size; mAP uses HF post-proc at threshold=0.001.\n")

    all_results = []
    for name, run_dir in RUNS:
        if not run_dir.exists():
            print(f"SKIP {name}: {run_dir} not found")
            continue

        print(f"═══ {name} ═══")
        fp32_onnx = run_dir / "model.onnx"
        int8_onnx = run_dir / "int8_trt" / "model_quantized.onnx"

        print("  [1/3] PyTorch fp32 (CUDA) ...", flush=True)
        pt = bench_pytorch(run_dir, gt)
        pt["model"] = name; pt["backend"] = "pytorch-fp32"
        all_results.append(pt)

        if fp32_onnx.exists():
            print("  [2/3] ONNX fp32 (CUDA EP) ...", flush=True)
            r = bench_onnx(fp32_onnx, run_dir, gt, "onnx-fp32")
            r["model"] = name; r["backend"] = "onnx-fp32"
            all_results.append(r)
        else:
            print(f"  SKIP onnx-fp32: {fp32_onnx} missing")

        if int8_onnx.exists():
            print("  [3/3] ONNX int8 QDQ (CUDA EP) ...", flush=True)
            r = bench_onnx(int8_onnx, run_dir, gt, "onnx-int8")
            r["model"] = name; r["backend"] = "onnx-int8"
            all_results.append(r)
        else:
            print(f"  SKIP onnx-int8: {int8_onnx} missing")
        print()

    # Summary
    print("\n" + "=" * 96)
    header = f"{'Model':<22} {'Backend':<14} {'Latency med':>12} {'P95':>8} {'FPS':>7} {'mAP':>7} {'mAP50':>8} {'mAP75':>8}"
    print(header); print("-" * 96)
    for r in all_results:
        print(
            f"{r['model']:<22} {r['backend']:<14} "
            f"{r['latency_median_ms']:>9.2f} ms "
            f"{r['latency_p95_ms']:>5.2f} ms "
            f"{r['fps']:>5.1f}  "
            f"{r['map']:>.3f}  {r['map_50']:>.3f}   {r['map_75']:>.3f}"
        )

    # INT8 viability verdict per arch — fp32 ONNX is the baseline.
    print("\n" + "=" * 96)
    print(f"{'Model':<22} {'Verdict':<20} {'Detail':<50}")
    print("-" * 96)
    failed: list[str] = []
    by_arch: dict[str, dict[str, dict]] = {}
    for r in all_results:
        by_arch.setdefault(r["model"], {})[r["backend"]] = r
    for name, runs in by_arch.items():
        label, reason = _verdict(runs.get("onnx-int8", {}), runs.get("onnx-fp32", {}))
        print(f"{name:<22} {label:<20} {reason}")
        if label == "INT8 NOT VIABLE":
            failed.append(name)

    out_json = REPO / "bench_detr_fp32_int8_pytorch.json"
    out_json.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults → {out_json.relative_to(REPO)}")

    if args.strict and failed:
        print(f"\n--strict: {len(failed)} arch(s) failed INT8 viability — {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()

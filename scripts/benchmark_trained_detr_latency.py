"""Per-image latency benchmark for our trained RT-DETRv2 / D-FINE checkpoints.

Measures pure forward + post-process time on a single GPU at the training input
size, with CUDA synchronisation. Reports median / P95 latency and FPS.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run scripts/benchmark_trained_detr_latency.py
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection

RUNS = [
    (
        "RT-DETRv2-R50 TV (test mAP 0.558)",
        "notebooks/detr_finetune_reference/our_rtdetr_v2_torchvision/runs/seed42",
    ),
    (
        "D-FINE-large TV 30ep (test mAP 0.430)",
        "notebooks/detr_finetune_reference/our_dfine_torchvision/runs/seed42",
    ),
    (
        "D-FINE-large TV 50ep (test mAP 0.492)",
        "notebooks/detr_finetune_reference/our_dfine_torchvision/runs/seed42_50ep",
    ),
]

INPUT_SIZE = 480
WARMUP = 20
ITERS = 200
DEVICE = "cuda"
DTYPE = torch.float32  # match training precision (fp32 for D-FINE)


def bench(name: str, run_dir: Path) -> dict:
    model = AutoModelForObjectDetection.from_pretrained(run_dir).to(DEVICE).eval()
    processor = AutoImageProcessor.from_pretrained(run_dir)

    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=DEVICE, dtype=DTYPE)
    target_sizes = torch.tensor([[INPUT_SIZE, INPUT_SIZE]], device=DEVICE)

    # Warmup
    with torch.inference_mode():
        for _ in range(WARMUP):
            out = model(pixel_values=x)
            _ = processor.post_process_object_detection(
                out, target_sizes=target_sizes, threshold=0.3
            )
    torch.cuda.synchronize()

    # Timed
    fwd_times, post_times = [], []
    with torch.inference_mode():
        for _ in range(ITERS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(pixel_values=x)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            _ = processor.post_process_object_detection(
                out, target_sizes=target_sizes, threshold=0.3
            )
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            fwd_times.append((t1 - t0) * 1000)
            post_times.append((t2 - t1) * 1000)

    total = [f + p for f, p in zip(fwd_times, post_times)]
    del model
    torch.cuda.empty_cache()
    return {
        "name": name,
        "params_M": n_params / 1e6,
        "fwd_median_ms": statistics.median(fwd_times),
        "post_median_ms": statistics.median(post_times),
        "total_median_ms": statistics.median(total),
        "total_p95_ms": sorted(total)[int(len(total) * 0.95)],
        "fps": 1000 / statistics.median(total),
    }


def main() -> None:
    assert torch.cuda.is_available(), "CUDA required"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Input: 1×3×{INPUT_SIZE}×{INPUT_SIZE}, dtype={DTYPE}, iters={ITERS}\n")

    results = []
    for name, path in RUNS:
        run_dir = Path(path).resolve()
        if not run_dir.exists():
            print(f"SKIP: {name} — {run_dir} not found")
            continue
        print(f"Benchmarking: {name}")
        results.append(bench(name, run_dir))

    print(f"\n{'Model':<45} {'Params':>8} {'Fwd':>8} {'Post':>7} {'Total':>8} {'P95':>7} {'FPS':>7}")
    print("-" * 92)
    for r in results:
        print(
            f"{r['name']:<45} "
            f"{r['params_M']:>6.1f} M "
            f"{r['fwd_median_ms']:>6.2f} ms "
            f"{r['post_median_ms']:>5.2f} ms "
            f"{r['total_median_ms']:>6.2f} ms "
            f"{r['total_p95_ms']:>5.2f} ms "
            f"{r['fps']:>5.1f}"
        )


if __name__ == "__main__":
    main()

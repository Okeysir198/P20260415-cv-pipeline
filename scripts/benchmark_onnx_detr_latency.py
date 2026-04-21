"""ONNX Runtime latency benchmark for exported RT-DETRv2 / D-FINE.

Runs the same single-image timing loop as benchmark_trained_detr_latency.py
but on the exported ONNX models, for direct PyTorch-vs-ONNX comparison.

Usage:
    .venv-export/bin/python scripts/benchmark_onnx_detr_latency.py
"""

from __future__ import annotations

import os
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

RUNS = [
    (
        "RT-DETRv2-R50 fp32",
        "notebooks/detr_finetune_reference/our_rtdetr_v2_torchvision/runs/seed42/model.onnx",
    ),
    (
        "D-FINE-large fp32",
        "notebooks/detr_finetune_reference/our_dfine_torchvision/runs/seed42_50ep/model.onnx",
    ),
]

INPUT_SIZE = 480
WARMUP = 20
ITERS = 200


def bench(name: str, onnx_path: Path) -> dict:
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3  # suppress warnings
    sess = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_opts,
        providers=["CUDAExecutionProvider"],  # GPU only; CPU fallback disabled
    )

    input_name = sess.get_inputs()[0].name
    x = np.random.randn(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)

    for _ in range(WARMUP):
        _ = sess.run(None, {input_name: x})

    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        _ = sess.run(None, {input_name: x})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    del sess
    return {
        "name": name,
        "size_MB": size_mb,
        "median_ms": statistics.median(times),
        "p95_ms": sorted(times)[int(len(times) * 0.95)],
        "fps": 1000 / statistics.median(times),
    }


def main() -> None:
    print(f"ORT version: {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")
    print(f"Input: 1×3×{INPUT_SIZE}×{INPUT_SIZE}, fp32, iters={ITERS}\n")

    assert "CUDAExecutionProvider" in ort.get_available_providers(), "CUDA provider required"

    results = []
    for name, path in RUNS:
        p = Path(path)
        if not p.exists():
            print(f"SKIP: {name} — {p} not found")
            continue
        print(f"Benchmarking: {name}", flush=True)
        results.append(bench(name, p))

    print(f"\n{'Model':<25} {'Size':>8} {'Median':>9} {'P95':>8} {'FPS':>7}")
    print("-" * 64)
    for r in results:
        print(
            f"{r['name']:<25} "
            f"{r['size_MB']:>6.1f} MB "
            f"{r['median_ms']:>6.2f} ms "
            f"{r['p95_ms']:>5.2f} ms "
            f"{r['fps']:>5.1f}"
        )


if __name__ == "__main__":
    main()

"""Minimal benchmark — pose latency on samples/.

Loads DWPose ONNX, runs N warmup + M timed inferences over each sample image
at full-frame box, reports mean/p50/p95 ms. Skips cleanly if ONNX or samples
are missing — lets the feature folder land before we have field clips.

Run:
    uv run features/safety-poketenashi_stair_diagonal/code/benchmark.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
_CODE_DIR = Path(__file__).parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(_CODE_DIR))

from predictor import _DWPOSE_ONNX, _DWPose  # noqa: E402

FEATURE = REPO / "features" / "safety-poketenashi_stair_diagonal"
SAMPLES = FEATURE / "samples"
EVAL = FEATURE / "eval"


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, p))


def main() -> int:
    if not _DWPOSE_ONNX.exists():
        print(f"[benchmark] DWPose ONNX missing: {_DWPOSE_ONNX}")
        return 0

    images = sorted(p for p in SAMPLES.glob("*.jpg") if p.is_file())
    if not images:
        print(f"[benchmark] no samples in {SAMPLES} — skipping")
        return 0

    pose = _DWPose(_DWPOSE_ONNX)
    timings: dict[str, list[float]] = {}

    # Warmup once on first image (CUDA/ORT init dominates first call).
    img0 = cv2.imread(str(images[0]))
    if img0 is not None:
        h, w = img0.shape[:2]
        for _ in range(3):
            pose(img0, np.array([0, 0, w, h], dtype=np.float32))

    for path in images:
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        box = np.array([0, 0, w, h], dtype=np.float32)
        runs: list[float] = []
        for _ in range(10):
            t0 = time.perf_counter()
            pose(img, box)
            runs.append((time.perf_counter() - t0) * 1000)
        timings[path.name] = runs

    summary = {
        "n_images": len(timings),
        "per_image_mean_ms": {k: round(float(np.mean(v)), 2) for k, v in timings.items()},
        "overall": {
            "mean_ms": round(float(np.mean([v for vs in timings.values() for v in vs])), 2),
            "p50_ms": round(_percentile([v for vs in timings.values() for v in vs], 50), 2),
            "p95_ms": round(_percentile([v for vs in timings.values() for v in vs], 95), 2),
        },
    }

    EVAL.mkdir(parents=True, exist_ok=True)
    out = EVAL / "benchmark_results.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"[benchmark] {summary['n_images']} images → mean={summary['overall']['mean_ms']} ms, "
          f"p95={summary['overall']['p95_ms']} ms")
    print(f"[benchmark] → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

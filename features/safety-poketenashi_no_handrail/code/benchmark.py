"""Latency benchmark for the no_handrail predictor.

Iterates over ``samples/*.jpg`` (or ``samples/*.png``) and times the full
predictor pass per image. Writes ``eval/benchmark_results.json``.

Run:
    uv run features/safety-poketenashi_no_handrail/code/benchmark.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[3]
_CODE_DIR = Path(__file__).parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(_CODE_DIR))

from predictor import NoHandrailPredictor  # noqa: E402


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return s[k]


def main() -> None:
    feat = REPO / "features" / "safety-poketenashi_no_handrail"
    config_path = feat / "configs" / "10_inference.yaml"
    samples_dir = feat / "samples"
    eval_dir = feat / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([*samples_dir.glob("*.jpg"), *samples_dir.glob("*.png")])
    if not image_paths:
        print(f"[benchmark] no images in {samples_dir} — copy a stair frame in")
        report = {"status": "no_samples", "n_images": 0}
        (eval_dir / "benchmark_results.json").write_text(json.dumps(report, indent=2))
        return

    predictor = NoHandrailPredictor(config_path)

    # Warmup pass.
    img0 = cv2.imread(str(image_paths[0]))
    if img0 is not None:
        for _ in range(3):
            predictor.process_frame(img0)

    latencies_ms: list[float] = []
    det_count = 0
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        t0 = time.perf_counter()
        result = predictor.process_frame(img)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        det_count += len(result.persons)

    n = len(latencies_ms)
    report = {
        "status": "ok",
        "n_images": n,
        "det_rate": round(det_count / max(1, n), 3),
        "latency_ms": {
            "mean": round(sum(latencies_ms) / n, 2) if n else 0.0,
            "p50": round(_percentile(latencies_ms, 50), 2),
            "p95": round(_percentile(latencies_ms, 95), 2),
            "max": round(max(latencies_ms), 2) if latencies_ms else 0.0,
        },
    }
    out = eval_dir / "benchmark_results.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()

"""Per-image benchmark: triggered / not-triggered tally + mean latency.

Runs the HandsInPocketsPredictor against the sample images in
``configs/10_inference.yaml::samples.images`` and writes a flat report
to ``eval/benchmark_results.json``.

Usage:
    uv run features/safety-poketenashi_hands_in_pockets/code/benchmark.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2

_FEAT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_FEAT / "code"))

from predictor import HandsInPocketsPredictor, _resolve_path  # noqa: E402


def main() -> int:
    cfg_path = _FEAT / "configs" / "10_inference.yaml"
    predictor = HandsInPocketsPredictor(cfg_path)
    samples = (predictor.cfg.get("samples", {}) or {}).get("images", []) or []
    cfg_dir = predictor.config_path.parent

    rows: list[dict] = []
    latencies_ms: list[float] = []
    n_triggered = 0
    for rel in samples:
        img_path = _resolve_path(rel, cfg_dir)
        if not img_path.exists():
            rows.append({"image": str(img_path), "status": "missing"})
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            rows.append({"image": str(img_path), "status": "unreadable"})
            continue
        # Warm + measure (single pass; rule itself is microseconds).
        t0 = time.perf_counter()
        out = predictor.process_frame(img)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt)
        triggered = any(p.result.triggered for p in out["persons"])
        if triggered:
            n_triggered += 1
        rows.append(
            {
                "image": str(img_path),
                "status": "ok",
                "n_persons": len(out["persons"]),
                "triggered": triggered,
                "latency_ms": round(dt, 2),
            }
        )

    summary = {
        "n_samples": len(samples),
        "n_ok": sum(1 for r in rows if r.get("status") == "ok"),
        "n_triggered": n_triggered,
        "mean_latency_ms": round(sum(latencies_ms) / max(1, len(latencies_ms)), 2),
        "rows": rows,
    }
    out_path = _FEAT / "eval" / "benchmark_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

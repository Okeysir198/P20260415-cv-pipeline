"""Latency benchmark across all 4 pose backends.

For each backend in ``{dwpose_onnx, rtmpose, mediapipe, hf_keypoint}``:
  1. Try to build the orchestrator with that backend.
  2. Run inference over the sample images N times each.
  3. Record mean / std latency and the person-detection rate.

Output: ``features/safety-point_and_call/eval/benchmark_results.json``.
"""

from __future__ import annotations

import argparse
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

_BACKENDS = ["dwpose_onnx", "rtmpose", "mediapipe", "hf_keypoint"]


def _resolve_samples_dir(feat_dir: Path) -> Path:
    own = feat_dir / "samples"
    if own.exists() and any(p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                            for p in own.iterdir()):
        return own
    return REPO / "features" / "safety-poketenashi" / "samples"


def _benchmark_backend(
    backend: str,
    config_path: Path,
    images: list[np.ndarray],
    runs: int,
) -> dict:
    from orchestrator import PointAndCallOrchestrator

    try:
        orch = PointAndCallOrchestrator(config_path, pose_backend_override=backend)
    except Exception as exc:
        return {"backend": backend, "error": f"{type(exc).__name__}: {exc}"}

    latencies: list[float] = []
    detected = 0
    total = 0
    try:
        for _ in range(runs):
            for img in images:
                t0 = time.perf_counter()
                res = orch.process_frame(img)
                latencies.append((time.perf_counter() - t0) * 1000.0)
                total += 1
                if res.persons:
                    detected += 1
    except Exception as exc:
        return {"backend": backend, "error": f"{type(exc).__name__}: {exc}"}

    if not latencies:
        return {"backend": backend, "error": "no images processed"}

    return {
        "backend": backend,
        "det_rate": detected / total if total else 0.0,
        "latency_ms_mean": float(np.mean(latencies)),
        "latency_ms_std": float(np.std(latencies)),
        "n_runs": runs,
        "n_images": len(images),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pose-backend latency benchmark")
    parser.add_argument("--runs", type=int, default=10,
                        help="Repetitions per backend (default 10)")
    args = parser.parse_args()

    feat = REPO / "features" / "safety-point_and_call"
    config_path = feat / "configs" / "10_inference.yaml"
    eval_dir = feat / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    samples_dir = _resolve_samples_dir(feat)
    image_paths = sorted(
        p for p in samples_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        print(f"No sample images found in {samples_dir}", file=sys.stderr)
        sys.exit(1)

    images: list[np.ndarray] = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is not None:
            images.append(img)
    if not images:
        print("All sample images unreadable", file=sys.stderr)
        sys.exit(1)

    print(f"Samples dir: {samples_dir} ({len(images)} images)")
    print(f"Config     : {config_path}")
    print(f"Runs       : {args.runs}\n")

    results: list[dict] = []
    for backend in _BACKENDS:
        print(f"--- backend: {backend} ---")
        r = _benchmark_backend(backend, config_path, images, args.runs)
        if "error" in r:
            print(f"  error: {r['error']}")
        else:
            print(
                f"  det_rate={r['det_rate']:.2f} "
                f"latency={r['latency_ms_mean']:.1f}+-{r['latency_ms_std']:.1f}ms"
            )
        results.append(r)

    out_path = eval_dir / "benchmark_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults -> {out_path}")


if __name__ == "__main__":
    main()

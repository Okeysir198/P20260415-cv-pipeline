"""Benchmark: CPU augmentation vs GPU augmentation throughput.

Measures dataloader iteration + GPU transfer + augmentation time for two modes:
  - cpu_aug:  all transforms on CPU workers (gpu_augment=False)
  - gpu_aug:  Mosaic/Resize on CPU workers, Affine/ColorJitter/Flips on GPU (gpu_augment=True)

Reports ms/batch, imgs/sec, and speedup ratio for batch sizes 16, 32, 64.
Uses the real fire_detection training split (~11K images) if available;
falls back to test_fire_100 (80 images) with a warning.

Run standalone:
    uv run tests/test_p06_aug_benchmark.py

Run via pytest:
    uv run -m pytest tests/test_p06_aug_benchmark.py -v -s
"""

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import run_all
from utils.config import load_config
from core.p05_data.detection_dataset import build_dataloader
from core.p05_data.transforms import build_gpu_transforms

OUTPUTS = Path(__file__).resolve().parent / "outputs" / "p06_aug_benchmark"
OUTPUTS.mkdir(parents=True, exist_ok=True)

# Prefer full fire_detection dataset; fall back to test_fire_100
_FIRE_DATA_CFG = ROOT / "features" / "safety-fire_detection" / "configs" / "05_data.yaml"
_TEST_DATA_CFG = ROOT / "configs" / "_test" / "05_data.yaml"

BATCH_SIZES = [16, 32, 64]
WARMUP_BATCHES = 5
MEASURE_BATCHES = 20

AUG_CONFIG = {
    "mosaic": True,
    "mixup": False,
    "hsv_h": 0.015,
    "hsv_s": 0.5,
    "hsv_v": 0.4,
    "fliplr": 0.5,
    "flipud": 0.0,
    "scale": [0.5, 1.5],
    "degrees": 10.0,
    "translate": 0.1,
    "shear": 2.0,
}


def _time_dataloader(data_config, base_dir, batch_size, gpu_augment, device):
    """Return mean ms/batch over MEASURE_BATCHES after WARMUP_BATCHES warmup."""
    training_config = {
        "augmentation": AUG_CONFIG,
        "training": {"gpu_augment": gpu_augment},
        "data": {
            "batch_size": batch_size,
            "num_workers": 4,
            "pin_memory": True,
        },
    }

    loader = build_dataloader(
        data_config,
        split="train",
        training_config=training_config,
        base_dir=base_dir,
    )

    gpu_transform = None
    if gpu_augment:
        input_size = tuple(data_config["input_size"])
        gpu_transform = build_gpu_transforms(
            config=AUG_CONFIG,
            input_size=input_size,
            mean=data_config.get("mean"),
            std=data_config.get("std"),
        )

    # Infinite iterator so we never run out of batches
    def _cycle():
        while True:
            yield from loader

    it = _cycle()

    for _ in range(WARMUP_BATCHES):
        batch = next(it)
        imgs = batch["images"].to(device, non_blocking=True)
        tgts = [t.to(device, non_blocking=True) for t in batch["targets"]]
        if gpu_transform is not None:
            imgs, tgts = gpu_transform(imgs, tgts)
        torch.cuda.synchronize()

    times = []
    for _ in range(MEASURE_BATCHES):
        t0 = time.perf_counter()
        batch = next(it)
        imgs = batch["images"].to(device, non_blocking=True)
        tgts = [t.to(device, non_blocking=True) for t in batch["targets"]]
        if gpu_transform is not None:
            imgs, tgts = gpu_transform(imgs, tgts)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return sum(times) / len(times)


def test_aug_benchmark():
    if not torch.cuda.is_available():
        print("  SKIP: CUDA not available")
        return

    device = torch.device("cuda")

    # Pick dataset
    if _FIRE_DATA_CFG.exists():
        data_config_path = _FIRE_DATA_CFG
        base_dir = str(_FIRE_DATA_CFG.parent)
        print(f"  Dataset: fire_detection (full, ~11K train images)")
    else:
        data_config_path = _TEST_DATA_CFG
        base_dir = str(_TEST_DATA_CFG.parent)
        print(f"  Dataset: test_fire_100 (80 train images — results less representative)")

    data_config = load_config(str(data_config_path))

    header = (
        f"\n  {'batch':>5}  {'CPU ms/batch':>14}  {'GPU ms/batch':>14}"
        f"  {'speedup':>8}  {'GPU img/s':>10}"
    )
    separator = "  " + "-" * (len(header) - 2)
    print(header)
    print(separator)

    results = []
    for bs in BATCH_SIZES:
        cpu_ms = _time_dataloader(data_config, base_dir, bs, gpu_augment=False, device=device)
        gpu_ms = _time_dataloader(data_config, base_dir, bs, gpu_augment=True,  device=device)
        speedup = cpu_ms / gpu_ms
        gpu_imgs = bs / (gpu_ms / 1000)
        results.append(dict(
            batch_size=bs,
            cpu_ms=round(cpu_ms, 1),
            gpu_ms=round(gpu_ms, 1),
            speedup=round(speedup, 2),
            gpu_imgs_per_sec=round(gpu_imgs, 0),
        ))
        print(
            f"  {bs:>5}  {cpu_ms:>14.1f}  {gpu_ms:>14.1f}"
            f"  {speedup:>8.2f}x  {gpu_imgs:>10.0f}"
        )

    print(separator)

    out_path = OUTPUTS / "aug_benchmark.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved → {out_path}")

    for r in results:
        assert r["speedup"] > 1.0, (
            f"GPU aug slower than CPU at batch_size={r['batch_size']}: "
            f"{r['gpu_ms']}ms vs {r['cpu_ms']}ms"
        )


if __name__ == "__main__":
    run_all(
        [("aug_benchmark — CPU vs GPU (batch 16/32/64)", test_aug_benchmark)],
        title="p06 — Augmentation Throughput Benchmark",
    )

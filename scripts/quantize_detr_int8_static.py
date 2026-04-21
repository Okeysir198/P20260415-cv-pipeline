"""Static INT8 quantization of exported RT-DETRv2 / D-FINE via Optimum ORTQuantizer
with calibration on CPPE-5 images.

Produces a TensorRT-compatible quantized ONNX. Can be loaded via ORT's
TensorrtExecutionProvider for GPU INT8 inference.

Usage:
    .venv-export/bin/python scripts/quantize_detr_int8_static.py
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from datasets import Dataset
from PIL import Image
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoCalibrationConfig, AutoQuantizationConfig

REPO = Path(__file__).resolve().parent.parent

RUNS = [
    ("rtdetr_v2", REPO / "notebooks/detr_finetune_reference/our_rtdetr_v2_torchvision/runs/seed42"),
    ("dfine_50ep", REPO / "notebooks/detr_finetune_reference/our_dfine_torchvision/runs/seed42_50ep"),
]

CALIB_DIR = REPO / "dataset_store/training_ready/cppe5/train/images"
INPUT_SIZE = 480
N_CALIB = 32  # MinMax only needs ~32 diverse samples; keeps GPU VRAM + host RAM low


def _preprocess(image_path: Path) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return np.transpose(arr, (2, 0, 1))[None, ...]  # [1, 3, H, W]


def build_calib_dataset(n: int) -> Dataset:
    imgs = sorted(CALIB_DIR.glob("*.jpg"))
    random.seed(42)
    sampled = random.sample(imgs, min(n, len(imgs)))
    tensors = [_preprocess(p)[0] for p in sampled]  # strip batch dim
    return Dataset.from_dict({"pixel_values": tensors})


def quantize(name: str, run_dir: Path) -> None:
    print(f"\n=== Quantizing {name} ===")
    print(f"Source: {run_dir}/model.onnx")

    quantizer = ORTQuantizer.from_pretrained(run_dir, file_name="model.onnx")

    # TRT-target INT8: produces ONNX consumable by TensorRT/ORT-TRT.
    # .tensorrt() is always static by design (docs say "for TensorRT static quantization").
    qconfig = AutoQuantizationConfig.tensorrt(per_channel=True)

    print(f"Building calibration dataset (n={N_CALIB}) from {CALIB_DIR}")
    calib_dataset = build_calib_dataset(N_CALIB)

    # MinMax calibration — keeps 2 scalars per tensor (vs percentile's full histogram).
    # Percentile OOM'd the box previously on DETR-family graphs (hundreds of exposed
    # activations × 100 images × full histograms). MinMax is usually within ~0.5 mAP
    # of percentile for static INT8 and fits in memory trivially.
    cal_config = AutoCalibrationConfig.minmax(calib_dataset)

    # Write the augmented (all-activations-exposed) model inside run_dir so we don't
    # litter the repo root, and so it's cleaned alongside int8_trt/.
    aug_path = run_dir / "int8_trt" / "augmented_model.onnx"
    aug_path.parent.mkdir(exist_ok=True)

    print("Computing activation ranges on GPU (CUDAExecutionProvider)...")
    ranges = quantizer.fit(
        dataset=calib_dataset,
        calibration_config=cal_config,
        onnx_augmented_model_name=aug_path,
        operators_to_quantize=qconfig.operators_to_quantize,
        batch_size=1,
        use_gpu=True,
    )

    out_dir = run_dir / "int8_trt"
    out_dir.mkdir(exist_ok=True)
    print(f"Writing quantized ONNX to {out_dir}/")
    quantizer.quantize(
        save_dir=out_dir,
        quantization_config=qconfig,
        calibration_tensors_range=ranges,
    )

    # Report sizes
    fp32_mb = (run_dir / "model.onnx").stat().st_size / (1024 * 1024)
    int8_files = list(out_dir.glob("*.onnx"))
    if int8_files:
        int8_mb = int8_files[0].stat().st_size / (1024 * 1024)
        print(f"fp32 {fp32_mb:.1f} MB → int8 {int8_mb:.1f} MB  ({int8_mb / fp32_mb * 100:.0f}%)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["rtdetr_v2", "dfine_50ep"], help="Quantize just one model")
    args = parser.parse_args()

    for name, run_dir in RUNS:
        if args.only and args.only != name:
            continue
        if not run_dir.exists():
            print(f"SKIP {name}: {run_dir} not found")
            continue
        try:
            quantize(name, run_dir)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

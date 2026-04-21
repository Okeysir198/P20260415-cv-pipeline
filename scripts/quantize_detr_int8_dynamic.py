"""Dynamic INT8 quantization of our exported RT-DETRv2 / D-FINE ONNX models.

Weights → INT8, activations stay fp32. No calibration data required. Produces
smaller files (~4×) and faster CPU inference (~2-3×) with some accuracy cost
(typically 3-8% mAP for DETR-family).

Usage:
    .venv-export/bin/python scripts/quantize_detr_int8_dynamic.py
"""

from __future__ import annotations

from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

RUNS = [
    ("RT-DETRv2-R50", "/tmp/onnx_bench/rtdetr_v2/model.onnx"),
    ("D-FINE-large 50ep", "/tmp/onnx_bench/dfine_50ep/model.onnx"),
]


def main() -> None:
    for name, path in RUNS:
        src = Path(path)
        dst = src.parent / f"{src.stem}_int8{src.suffix}"
        print(f"Quantizing {name}: {src.name} → {dst.name}")
        quantize_dynamic(
            model_input=str(src),
            model_output=str(dst),
            weight_type=QuantType.QInt8,
        )
        src_mb = src.stat().st_size / (1024 * 1024)
        dst_mb = dst.stat().st_size / (1024 * 1024)
        print(f"  fp32 {src_mb:.1f} MB → int8 {dst_mb:.1f} MB  ({dst_mb / src_mb * 100:.0f}%)")


if __name__ == "__main__":
    main()

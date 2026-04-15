# p05_export — ONNX Export + Quantization + Benchmarking

## Purpose

Convert trained PyTorch models to ONNX for edge deployment, with optional INT8 quantization and latency benchmarking.

## Export Flow

```
.pth (PyTorch) → .onnx (ONNX) → onnxsim (simplified) → INT8 quantize (edge-ready)
```

## Files

| File | Purpose |
|---|---|
| `exporter.py` | `ModelExporter` — .pth to .onnx with onnxsim simplification, output validation against PyTorch, configurable naming template |
| `export.py` | CLI entry point |
| `quantize.py` | `ModelQuantizer` class with `quantize_dynamic()` and `quantize_static()` methods — dynamic (no calibration data) and static (with calibration dataset via `CalibrationDataReader`) INT8 quantization using `onnxruntime.quantization` |
| `benchmark.py` | `ModelBenchmark` — compare PyTorch vs ONNX vs quantized: latency (ms), throughput (FPS), model size (MB), memory usage |

## Target Edge Chips

- **AX650N**: 18 INT8 TOPS
- **CV186AH**: 7.2 INT8 TOPS

## CLI

```bash
uv run core/p09_export/export.py \
  --model runs/fire_detection/best.pth \
  --training-config features/safety-fire_detection/configs/06_training.yaml \
  --export-config configs/_shared/09_export.yaml
```

## Config Reference

- `configs/_shared/09_export.yaml` — ONNX opset, simplify flag, output directory, naming template (`{model}_{arch}_{imgsz}_v{version}`)

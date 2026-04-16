# Export & Edge Deployment Guide

## ONNX Export

```bash
FEATURE=features/safety-fire_detection

# FP32 ONNX (default) — runs in the main venv
uv run core/p09_export/export.py \
  --model $FEATURE/runs/best.pt \
  --training-config $FEATURE/configs/06_training.yaml \
  --skip-optimize
```

- **Config:** `configs/_shared/09_export.yaml`
- **Opset:** 17 (default)
- **Simplify:** enabled by default (onnx-simplifier)
- **Outputs:** `$FEATURE/export/<model>.onnx`

## Graph Optimization + INT8 Quantization

Optimum + `onnxruntime-quantize` conflict with the `transformers@git` pin
in the main venv, so quantization lives in a separate venv.

```bash
# One-time setup
bash scripts/setup-export-venv.sh     # creates .venv-export/

# Dynamic INT8 (no calibration data required)
.venv-export/bin/python core/p09_export/export.py \
  --model $FEATURE/runs/best.pt \
  --training-config $FEATURE/configs/06_training.yaml \
  --optimize O2 --quantize dynamic

# Static INT8 (uses ~100 val images for calibration)
.venv-export/bin/python core/p09_export/export.py \
  --model $FEATURE/runs/best.pt \
  --training-config $FEATURE/configs/06_training.yaml \
  --optimize O2 --quantize static
```

- Target: < 3% mAP drop from FP32
- Calibration images come from `05_data.yaml::val`
- `--optimize` levels: `O1` (basic), `O2` (default recommended), `O3`
  (GPU-friendly), `O4` (FP16 cast)

## Target Edge Chips

| Chip | INT8 TOPS | Use Cases |
|------|-----------|-----------|
| AX650N | 18 | Primary deployment target |
| CV186AH | 7.2 | Cost-optimized alternative |

## Benchmarking

`ModelBenchmark` in `core/p09_export/benchmark.py` is a library class
(no CLI). Invoke from a short script or from tests:

```python
from core.p09_export.benchmark import ModelBenchmark

bench = ModelBenchmark(onnx_path="features/safety-fire_detection/export/best.onnx")
bench.run(input_size=(640, 640), iterations=100)
```

On-device latency should be measured with the vendor SDK on the real
edge chip, not with onnxruntime on the host.

## Deployment Checklist

- [ ] ONNX export validates (output matches PyTorch within tolerance)
- [ ] INT8 quantization mAP drop < 3%
- [ ] Latency meets spec (< 40ms per frame on target chip)
- [ ] Memory footprint fits edge device RAM
- [ ] `model_card.yaml` generated via `utils/release.py --run-dir <ts_dir>`

# Export & Edge Deployment Guide

## ONNX Export

```bash
# Export trained model to ONNX
uv run core/p09_export/export.py \
  --model runs/fire_detection/best.pt \
  --training-config features/safety-fire_detection/configs/06_training.yaml \
  --export-config configs/_shared/09_export.yaml
```

- **Config:** `configs/_shared/09_export.yaml`
- **Opset:** 17 (default)
- **Simplify:** enabled by default (onnx-simplifier)

## INT8 Quantization

```bash
uv run core/p09_export/quantize.py \
  --model runs/fire_detection/best.onnx \
  --calibration-data dataset_store/fire_detection/val/images/ \
  --output runs/fire_detection/best_int8.onnx
```

- Calibration uses ~100 representative images from validation set
- Target: < 3% mAP drop from FP32

## Target Edge Chips

| Chip | INT8 TOPS | Use Cases |
|------|-----------|-----------|
| AX650N | 18 | Primary deployment target |
| CV186AH | 7.2 | Cost-optimized alternative |

## Benchmarking

```bash
uv run core/p09_export/benchmark.py \
  --model runs/fire_detection/best.onnx \
  --input-size 640 640 \
  --iterations 100
```

## Deployment Checklist

- [ ] ONNX export validates (output matches PyTorch within tolerance)
- [ ] INT8 quantization mAP drop < 3%
- [ ] Latency meets spec (< 40ms per frame)
- [ ] Memory footprint fits edge device RAM

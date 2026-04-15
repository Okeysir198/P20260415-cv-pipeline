# PicoDet (PP-PicoDet) — Model Research

Anchor-free, lightweight object detector by Baidu (PaddleDetection). Designed for mobile and edge devices.

## Quick Reference

| Variant | Params | FLOPs | mAP@COCO (input) | Latency (SD865) |
|---------|--------|-------|-------------------|-----------------|
| PicoDet-XS | 0.70M | 0.67G | 23.5 (320) / 26.2 (416) | ~8ms / ~12ms |
| PicoDet-S | 1.18M | 0.97G | 29.1 (320) / 32.5 (416) | ~10ms / ~15ms |
| PicoDet-M | 3.46M | 2.57G | 34.4 (320) / 37.5 (416) | ~8ms / ~28ms |
| PicoDet-L | 5.80M | 4.20G | 36.1 (320) / 39.4 (416) / 42.6 (640) | ~12ms / ~21ms / ~63ms |

**License:** Apache 2.0 — compatible with commercial use.
**Paper:** [arXiv 2111.00902](https://arxiv.org/abs/2111.00902)
**Official code:** [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) (PaddlePaddle)

## Comparisons vs Current Pipeline Models

| Model | Params | mAP@COCO | Notes |
|-------|--------|----------|-------|
| PicoDet-XS | 0.7M | 23.5 (320) | Smallest in pipeline |
| PicoDet-S | 1.2M | 32.5 (416) | Beats YOLOX-Nano (0.9M, 25.8) and Nanodet-M (1.0M, 23.5) |
| PicoDet-M | 3.5M | 37.5 (416) | Smaller than YOLOX-Tiny (5M, 32.8) with higher accuracy |
| PicoDet-L | 5.8M | 42.6 (640) | Matches YOLOv5s (7.2M, 37.2) at nearly 2x lower latency |
| YOLOX-Tiny | 5.1M | 32.8 (416) | Current pipeline lightweight option |
| YOLOX-M | 25.3M | 40.4 (640) | Current pipeline primary detector |
| D-FINE-S | 10M | 48.5 (640) | Current pipeline transformer option |

## Architecture

```
Input → ESNet/LCNet Backbone → CSP-PAN Neck → Task-Aligned Head → Output
           (stride 8,16,32)    (top-down +      (cls + reg)
                                  bottom-up)
                                                     ↓
                                              (B, N, 4+C) tensor
                                              [cx, cy, w, h, cls_0, ..., cls_C]
```

### Key Components

- **ESNet backbone** (Enhanced ShuffleNet) — optimized for low memory access cost, minimal transpose operations. LCNet variant in v2 models.
- **CSP-PAN neck** — Cross Stage Partial Path Aggregation Network. Fuses multi-scale features with CSP-style splitting.
- **Task-Aligned Head** — Classification + regression branches only. **No objectness branch** (key difference from YOLOX).
- **Task Alignment Learning (TAL)** — Assignment criterion: `t = s^alpha * IoU^beta`, selects top-k anchors per GT. Replaces SimOTA/YOLOX assignment.

### Key Differences from YOLOX

| Aspect | YOLOX | PicoDet |
|--------|-------|---------|
| Output tensor | `(B, N, 5+C)` — has objectness | `(B, N, 4+C)` — no objectness |
| Confidence | `obj_conf * cls_conf` | `max(softmax(cls_logits))` |
| Assignment | SimOTA (cost matrix) | TAL (alignment metric, top-k) |
| Loss | BCE obj + BCE cls + GIoU reg | BCE/Focal cls + GIoU reg |
| Backbone | CSPDarknet | ESNet/LCNet |
| Neck | PAFPN | CSP-PAN |

## ONNX Export & Quantization

### ONNX Export

Officially supported via `paddle2onnx`:

```bash
pip install paddle2onnx==0.9.2
paddle2onnx --model_dir output_inference/picodet_s_320_coco_lcnet/ \
            --model_filename model.pdmodel \
            --params_filename model.pdiparams \
            --opset_version 11 \
            --save_file picodet_s_320_coco.onnx
onnxsim picodet_s_320_coco.onnx picodet_s_processed.onnx
```

Supported inference backends: OpenVINO, NCNN, MNN, TensorRT, Paddle Lite.

### INT8 Quantization

| Precision | mAP@0.5:0.95 (PicoDet-S 416) |
|-----------|------------------------------|
| FP32 | 32.5 |
| INT8 | 31.5 (~1.0 mAP drop) |

Uses PaddleSlim for QAT/PTQ. PicoDet-S-NPU variant specifically optimized for INT8 NPU deployment.

## Where PicoDet Fits in This Pipeline

### Use Case Candidates

| Current Model | PicoDet Alternative | Benefit |
|--------------|-------------------|---------|
| YOLOX-Tiny for zone intrusion (5M) | PicoDet-M (3.5M) | 30% smaller, higher accuracy |
| YOLOX-Tiny for pose pipeline backbone (5M) | PicoDet-S (1.2M) | 4x smaller, faster for multi-model chains |
| YOLOX-M for fire detection (25M) | PicoDet-L (5.8M) | 4x smaller (but lower accuracy — 42.6 vs 40.4 mAP COCO, needs domain validation) |
| YOLOX-Tiny for poketenashi (5M) | PicoDet-M (3.5M) | Smaller + faster for phone detection + pose chain |

### Gap Filled

PicoDet-XS (0.7M) and PicoDet-S (1.2M) are significantly smaller than anything currently in the pipeline. This opens possibilities for **multi-model concurrent inference** on edge chips where memory and compute budget is tight.

## Integration Options

### Option A: Full PyTorch Integration (Recommended for production)

Port PicoDet to PyTorch and integrate into the full pipeline. Follows the same pattern as YOLOX (native model + external loss + registered postprocessor).

**Files needed:**
- NEW `core/p06_models/picodet.py` — ESNet + CSP-PAN + TaskAlignedHead + `@register_model("picodet")`
- NEW `core/p06_training/picodet_loss.py` — TAL loss with TaskAlignmentAssignment
- MODIFY `core/p06_models/__init__.py` — add import
- MODIFY `core/p06_training/losses.py` — register `PicoDetLoss` factory + `_ARCH_LOSS_MAP`
- MODIFY `core/p06_training/postprocess.py` — register `@register_postprocessor("picodet")`
- MODIFY `core/p06_training/metrics_registry.py` — `METRICS_REGISTRY["picodet"] = _detection_metrics`

**Key design decisions:**
- New `output_format = "picodet"` (not reusing `"yolox"` — PicoDet has no objectness channel, so YOLOX postprocessor would be incorrect)
- External loss (not `forward_with_loss`) — follows YOLOX pattern since PicoDet loss is structurally different
- Variant aliases: `picodet-xs`, `picodet-s`, `picodet-m`, `picodet-l`

**No changes needed** in: `trainer.py`, `exporter.py`, `predictor.py`, `detection_dataset.py`

**PyTorch reference:** [miemie2013/PicoDet-pytorch](https://github.com/miemie2013/PicoDet-pytorch) (~310 stars, most complete unofficial port)

**Pros:** Unified training/eval/export/inference, can fine-tune on custom datasets
**Cons:** Unofficial PyTorch port — may have minor accuracy differences vs PaddlePaddle reference

### Option B: ONNX-Only (Train with PaddlePaddle)

Train in PaddleDetection, export to ONNX, integrate only into `core/p10_inference/`.

```bash
# Training (PaddlePaddle)
python tools/train.py -c configs/picodet/picodet_s_320_coco.yml

# Export to ONNX
paddle2onnx --model_dir output_inference/picodet_s/ --save_file picodet_s.onnx
```

**Pros:** Official training accuracy, proven ONNX export, minimal code changes
**Cons:** Requires PaddlePaddle environment, separate training pipeline, cannot fine-tune within this pipeline

### Option C: Hybrid — ONNX Import for Inference Only

Use PicoDet ONNX models (pretrained COCO or custom-trained in PaddleDetection) as drop-in inference models. Integrate into `core/p10_inference/predictor.py` ONNX path only.

**Pros:** Zero training code, use official models directly
**Cons:** No fine-tuning capability, limited to available pretrained weights

## Recommendations

1. **Short-term (evaluation):** Use Option C — download official PicoDet ONNX models, test inference speed and accuracy on AX650N/CV186AH using `core/p10_inference/predictor.py` and `core/p09_export/benchmark.py`. Compare against YOLOX-Tiny for zone intrusion and phone detection use cases.

2. **Mid-term (if evaluation passes):** Use Option A — full PyTorch integration for fine-tuning on domain datasets. The pipeline architecture is ready for it (registry, loss, postprocessor patterns all exist). Main effort is porting the ESNet backbone and TAL loss to PyTorch.

3. **Key validation needed:**
   - Benchmark PicoDet-S vs YOLOX-Tiny on zone intrusion dataset (mAP@0.5, latency on AX650N)
   - Test INT8 quantized PicoDet accuracy drop on fire/helmet domain data
   - Verify multi-model concurrent inference feasibility (PicoDet person detector + YOLOX-M specialized detector)

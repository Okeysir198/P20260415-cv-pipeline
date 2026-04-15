# ppe-shoes_detection — Pretrained Weights

## Summary

Two-stage safety-shoes pipeline: a clean-license person/foot detector (Stage 1)
followed by a fine-grained shoe classifier (Stage 2). Total **54 files**
across **~760 MB**. See brief:
[../../docs/technical_study/ppe-shoes_detection-sota.md](../../docs/technical_study/ppe-shoes_detection-sota.md).

## Files on disk (as of 2026-04-14)

| File / Subfolder | Size | Type | License | Source | Notes |
|---|---|---|---|---|---|
| `dfine_nano_coco.safetensors` + `dfine_nano_coco_{config,preprocessor}.json` | 15 MB | D-FINE-N COCO | Apache-2.0 | ustc-community | **Stage-1 top pick.** |
| `dfine_nano_coco_hf/` | 16 KB | HF-style config mirror | Apache-2.0 | ustc-community | Convenience copy for HF Transformers loader. |
| `dfine_small_coco.safetensors` + `dfine_small_coco_{config,preprocessor}.json` | 40 MB | D-FINE-S COCO | Apache-2.0 | ustc-community | Stage-1 upgrade tier. |
| `dfine_small_coco_hf/` | 16 KB | HF config mirror | Apache-2.0 | ustc-community | |
| `rfdetr_small.onnx` | 110 MB | RF-DETR-Small ONNX | Apache-2.0 | roboflow | Single-stage / Stage-1 alternative. |
| `dinov2_small.bin` + `dinov2_small_{config,preprocessor}.json` | 85 MB | DINOv2 ViT-S/14 backbone | Apache-2.0 | facebook | **Stage-2 top pick** (needs 2-class linear head). |
| `dinov2_small_hf/` | 16 KB | HF config mirror | Apache-2.0 | facebook | |
| `efficientformerv2_s0.bin` | 15 MB | EfficientFormerV2-S0 (timm) | Apache-2.0 | snap-research | Stage-2 lightweight. |
| `efficientformerv2_s1.bin` | 25 MB | EfficientFormerV2-S1 (timm) | Apache-2.0 | snap-research | Stage-2 mid. |
| `fastvit_t8.bin` | 16 MB | FastViT-T8 (timm) | **Apple ASCL — non-commercial** | apple | Benchmark only. |
| `fastvit_t12.bin` | 30 MB | FastViT-T12 (timm) | **Apple ASCL — non-commercial** | apple | Benchmark only. |
| `mobilevitv2_100.bin` | 19 MB | MobileViTv2-1.0 (timm) | **Apple sample-code** | apple | Benchmark only. |
| `_hf_facebook_dinov3-vits16-pretrain-lvd1689m/` | 83 MB | DINOv3 ViT-S/16 | **Meta DINOv3 non-commercial** | facebook | Benchmark only — license blocks ship. |
| `_hf_facebook_dinov3-vitb16-pretrain-lvd1689m/` | 327 MB | DINOv3 ViT-B/16 | **Meta DINOv3 non-commercial** | facebook | Benchmark only. |
| `imagenet_classes.txt` | 12 KB | ImageNet-1k class list | — | — | Probe helper. |
| `DOWNLOAD_MANIFEST.md` | 1.5 KB | Manifest w/ SHA256s | — | — | |

## Recommended defaults (from SOTA brief)

- **Stage 1 top pick — D-FINE-N (COCO)** → `dfine_nano_coco.safetensors` (on disk).
- **Stage 2 top picks — DINOv2-small, EfficientFormerV2-S0** → `dinov2_small.bin`, `efficientformerv2_s0.bin` (on disk).
- **Single-stage alternative — RF-DETR-Small** → `rfdetr_small.onnx` (on disk).

## Gated / skipped / 404

- DINOv3 ViT-S/B were fetched through the gated-retry pass; **non-commercial licence**, kept for benchmark only.
- FastViT / MobileViTv2 remain on disk but Apple ASCL blocks commercial deployment.

## Related docs

- SOTA brief: `../../docs/technical_study/ppe-shoes_detection-sota.md`
- Deep dive: `../../docs/technical_study/ppe-oss-pretrained-deep-dive.md`
- Bulk log: `../../docs/technical_study/ppe-bulk-download-log.md`
- Quality report: `../../features/ppe-shoes_detection/predict/QUALITY_REPORT.md`
- SHA256s: `./DOWNLOAD_MANIFEST.md`

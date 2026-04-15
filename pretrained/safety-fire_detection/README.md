# safety-fire_detection — Pretrained Weights

## Summary

Fire/smoke detection candidates from the Phase-1 bulk download pass. Total
**413 files** across **~2.0 GB** — a mix of COCO/Obj365 generic detectors
(D-FINE S/M/L, DEIM-D-FINE, YOLOX-M) used as clean-license starting points
for fine-tuning, plus ~25 community fire/smoke fine-tunes (Pyronear,
Ultralytics-derivative, ViT / Swin / ConvNeXt classifiers) kept for
benchmarking only. See brief:
[../../docs/technical_study/safety-fire_detection-sota.md](../../docs/technical_study/safety-fire_detection-sota.md).

## Files on disk (as of 2026-04-14)

| File / Subfolder | Size | Type | License | Source | Notes |
|---|---|---|---|---|---|
| `deim_dfine_m_coco/deim_dfine_hgnetv2_m_coco_90e.pth` | 76 MB | DEIM-D-FINE-M COCO | Apache-2.0 | ShihuaHuang95/DEIM | **SOTA pick #1** for fine-tuning. |
| `deim_dfine_s_coco/deim_dfine_hgnetv2_s_coco_120e.pth` | 40 MB | DEIM-D-FINE-S COCO | Apache-2.0 | ShihuaHuang95/DEIM | SOTA pick #2. |
| `dfine_s_coco.pth` | 40 MB | D-FINE-S COCO | Apache-2.0 | Peterande/D-FINE | Fallback if DEIM training is skipped. |
| `dfine_m_coco.pth` | 58 MB | D-FINE-M COCO | Apache-2.0 | Peterande/D-FINE | Fallback. |
| `dfine_s_obj365.pth` | 42 MB | D-FINE-S Objects365 | Apache-2.0 | Peterande/D-FINE | Stronger open-vocab pretrain. |
| `dfine_m_obj365.pth` | 78 MB | D-FINE-M Objects365 | Apache-2.0 | Peterande/D-FINE | Stronger open-vocab pretrain. |
| `dfine_l_obj365.pth` | 73 MB | D-FINE-L Objects365 | Apache-2.0 | Peterande/D-FINE | Larger accuracy tier. |
| `ustc-community_dfine-small-coco/` | 40 MB | D-FINE-S HF port | Apache-2.0 | ustc-community | HF Transformers-native safetensors + config. |
| `ustc-community_dfine-medium-coco/` | 76 MB | D-FINE-M HF port | Apache-2.0 | ustc-community | HF Transformers-native. |
| `yolox_m.pth` | 102 MB | YOLOX-M COCO | Apache-2.0 | Megvii | Baseline detector retained as safety net. |
| `pyronear_yolov11n/` | 41 MB | YOLOv11n wildfire (.pt + ONNX/NCNN tars) | **AGPL-3.0** (Ultralytics) | pyronear | Benchmark only — AGPL blocks commercial ship. |
| `pyronear_yolov8s/` | 100 MB | YOLOv8s wildfire | **AGPL-3.0** | pyronear | Benchmark only. |
| `pyronear_yolo11s_sensitive-detector/` | 81 MB | YOLO11s wildfire + ONNX/NCNN | **AGPL-3.0** | pyronear | Benchmark only. |
| `pyronear_yolo11s_nimble-narwhal_v6/` | 19 MB | YOLO11s wildfire v6 | **AGPL-3.0** | pyronear | Benchmark only. |
| `pyronear_mobilenet_v3_small/` | 12 MB | MobileNetV3-S fire classifier | Apache-2.0 (check) | pyronear | Single-class classifier (fire vs no-fire). |
| `pyronear_mobilenet_v3_large/` | 33 MB | MobileNetV3-L fire classifier | Apache-2.0 (check) | pyronear | Includes ONNX. |
| `pyronear_resnet18/` | 86 MB | ResNet18 fire classifier | Apache-2.0 (check) | pyronear | Includes ONNX. |
| `pyronear_resnet34/` | 163 MB | ResNet34 fire classifier | Apache-2.0 (check) | pyronear | Includes ONNX. |
| `pyronear_rexnet1_0x/` | 28 MB | ReXNet-1.0× fire classifier | Apache-2.0 (check) | pyronear | |
| `pyronear_rexnet1_3x/` | 46 MB | ReXNet-1.3× fire classifier | Apache-2.0 (check) | pyronear | |
| `pyronear_rexnet1_5x/` | 60 MB | ReXNet-1.5× fire classifier | Apache-2.0 (check) | pyronear | |
| `JJUNHYEOK_yolov8n_wildfire/best.pt` | 22 MB | YOLOv8n wildfire | **AGPL-3.0** (Ultralytics) | JJUNHYEOK | Benchmark only. |
| `Mehedi-2-96_fire-smoke-yolo/` | 22 MB | YOLOv8s fire+smoke | **AGPL-3.0** | Mehedi-2-96 | Benchmark only. |
| `SalahALHaismawi_yolov26-fire-detection/` | 21 MB | YOLOv26 fire | **AGPL-3.0** (Ultralytics-derivative) | SalahALHaismawi | Benchmark only. |
| `TommyNgx_YOLOv10-Fire-and-Smoke/` | 119 MB | YOLOv10 fire+smoke | **AGPL-3.0** (THU-MIG) | TommyNgx | Benchmark only. |
| `touati-kamel_yolov8s-forest-fire/` | 22 MB | YOLOv8s forest fire | **AGPL-3.0** | touati-kamel | Benchmark only. |
| `touati-kamel_yolov10n-forest-fire/` | 12 MB | YOLOv10n forest fire | **AGPL-3.0** | touati-kamel | Benchmark only. |
| `touati-kamel_yolov12n-forest-fire/` | 11 MB | YOLOv12n forest fire | **AGPL-3.0** | touati-kamel | Benchmark only. |
| `shawnmichael_convnext-tiny-fire-smoke/` | 107 MB | ConvNeXt-T classifier | Apache-2.0 | shawnmichael | Fire/smoke classifier. |
| `shawnmichael_efficientnetb2-fire-smoke/` | 30 MB | EfficientNet-B2 classifier | Apache-2.0 | shawnmichael | |
| `shawnmichael_swin-fire-smoke/` | 106 MB | Swin-T classifier | Apache-2.0 | shawnmichael | |
| `shawnmichael_vit-fire-smoke-v4/` | 65 MB | ViT-B fire-smoke | Apache-2.0 | shawnmichael | |
| `shawnmichael_vit-large-fire-smoke/` | 65 MB | ViT-L fire-smoke (weights only; safetensors missing) | Apache-2.0 | shawnmichael | No weight file downloaded — config only. |
| `dima806_wildfire_types/` | 65 MB | ViT wildfire-type classifier | Apache-2.0 | dima806 | Multi-class wildfire types. |
| `sequoiaandrade_smoke-cloud-race-odin/` | 27 MB | TF SavedModel smoke-vs-cloud | Apache-2.0 (check) | sequoiaandrade | TF format, not wired into pipeline. |
| `Shoriful025_wildfire_smoke_seg_vit/` | 48 KB | ViT wildfire smoke segmentation (config only) | (check) | Shoriful025 | Weights missing — config only. |
| `DOWNLOAD_MANIFEST.md` | 6.7 KB | Download manifest with SHA256s | — | — | Source of truth for hashes. |
| `_logs/` | 4 KB | Download logs | — | — | |

## Recommended defaults (from SOTA brief)

- **#1 DEIM-D-FINE-M (COCO)** → `deim_dfine_m_coco/deim_dfine_hgnetv2_m_coco_90e.pth` (on disk).
- **#2 D-FINE-S / DEIM-D-FINE-S** → `deim_dfine_s_coco/deim_dfine_hgnetv2_s_coco_120e.pth` and `dfine_s_coco.pth` (on disk).
- **#3 RF-DETR-Small** → not on disk in this folder; see `ppe-shoes_detection/rfdetr_small.onnx` for the shared RF-DETR-Small ONNX artefact.

Caveat from `predict/QUALITY_REPORT.md`: both top picks are COCO/Obj365
pretrains with **no fire/smoke class**, so Phase-1 fine-tuning on FASDD +
D-Fire is required before production use.

## Gated / skipped / 404

None for this feature. All bulk-log entries downloaded successfully.

## Related docs

- SOTA brief: `../../docs/technical_study/safety-fire_detection-sota.md`
- Deep dive: `../../docs/technical_study/safety-fire_oss-pretrained-deep-dive.md`
- Bulk log: `../../docs/technical_study/safety-fire_bulk-download-log.md`
- Quality report: `../../features/safety-fire_detection/predict/QUALITY_REPORT.md`
- SHA256s: `./DOWNLOAD_MANIFEST.md`

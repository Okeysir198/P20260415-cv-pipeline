# CLAUDE.md — pretrained/

Pretrained model weights. Two layouts coexist:

- **Per-feature dirs** (canonical, actively used by benchmark.py and training): `pretrained/<feature>/` — e.g. `pretrained/ppe-helmet_detection/`, `pretrained/safety-fire_detection/`. See each feature's `CLAUDE.md` for benchmark rankings.
- **Legacy `nitto_denko/` subtree**: pre-benchmark exploration, kept for reference. Do not pick models from the Nitto Denko tables below — **authoritative "best pretrained" rankings live in `features/<feature>/CLAUDE.md`** (val mAP50 verified 2026-04-17).

Most weights are free-licensed (Apache-2.0, MIT, CC-BY-4.0, openrail). AGPL-3.0 rows are flagged — avoid for commercial deployment.

## Root Models (COCO-pretrained, multi-purpose)

| Model | File | Size | Classes | License | Source |
|-------|------|------|---------|---------|--------|
| YOLOX-Nano | `yolox_nano.pth` | 7.4 MB | 80 (COCO) | Apache-2.0 | [Megvii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX/releases) |
| YOLOX-Tiny | `yolox_tiny.pth` | 39 MB | 80 (COCO) | Apache-2.0 | Megvii YOLOX |
| YOLOX-S | `yolox_s.pth` | 69 MB | 80 (COCO) | Apache-2.0 | Megvii YOLOX |
| YOLOX-M | `yolox_m.pth` | 194 MB | 80 (COCO) | Apache-2.0 | Megvii YOLOX |
| YOLOX-L | `yolox_l.pth` | 415 MB | 80 (COCO) | Apache-2.0 | Megvii YOLOX |
| D-FINE-N | `dfine_n_coco.pt` | 14 MB | 80 (COCO) | Apache-2.0 | [ustc-community/dfine-nano-coco](https://huggingface.co/ustc-community/dfine-nano-coco) |
| D-FINE-S | `dfine_s_coco.pt` | 40 MB | 80 (COCO) | Apache-2.0 | [ustc-community/dfine-small-coco](https://huggingface.co/ustc-community/dfine-small-coco) |
| D-FINE-M | `dfine_m_coco.pt` | 76 MB | 80 (COCO) | Apache-2.0 | [ustc-community/dfine-medium-coco](https://huggingface.co/ustc-community/dfine-medium-coco) |
| RT-DETR-R18 | `rtdetr_v2_r18_coco.pt` | 78 MB | 80 (COCO) | Apache-2.0 | [PekingU/rtdetr_r18vd_coco_o365](https://huggingface.co/PekingU/rtdetr_r18vd_coco_o365) |
| RT-DETR-R50 | `rtdetr_v2_r50_coco.pt` | 165 MB | 80 (COCO) | Apache-2.0 | PekingU/rtdetr_r50vd_coco_o365 |
| SCRFD-500M | `scrfd_500m.onnx` | 2.5 MB | face | MIT | [immich-app/buffalo_s · detection/model.onnx](https://huggingface.co/immich-app/buffalo_s/resolve/main/detection/model.onnx) |
| MobileFaceNet (ArcFace) | `mobilefacenet_arcface.onnx` | 13 MB | 512-D embed | MIT | [immich-app/buffalo_s · recognition/model.onnx](https://huggingface.co/immich-app/buffalo_s/resolve/main/recognition/model.onnx) |

**COCO vehicle classes**: car (2), motorcycle (3), bus (5), truck (7) — all COCO models detect vehicles out-of-the-box.

### Edge Face Recognition Stack — SCRFD-500M + MobileFaceNet

Matched detector + recogniser pair from the InsightFace `buffalo_s` compact bundle (the edge/mobile variant of `buffalo_l`). Both files live at the repo root so the `core/p06_models/{scrfd,mobilefacenet}.py` default paths resolve without config changes. MIT-compatible license, production-friendly, total footprint <16 MB.

**Download commands** (re-run to refresh):

```bash
# Detector (9-output SCRFD-500M, 1-anchor per position + landmarks)
curl -L -o pretrained/scrfd_500m.onnx \
  https://huggingface.co/immich-app/buffalo_s/resolve/main/detection/model.onnx

# Recogniser (MobileFaceNet + ArcFace, WebFace600K)
curl -L -o pretrained/mobilefacenet_arcface.onnx \
  https://huggingface.co/immich-app/buffalo_s/resolve/main/recognition/model.onnx
```

**Specs**:

| Model | Input | Output | Preprocess | Notes |
|---|---|---|---|---|
| SCRFD-500M | `(1, 3, H, W)` RGB | 9 tensors: score/bbox/kps × 3 strides (8/16/32) | `(pixel - 127.5) / 128.0` | Dynamic H/W; `core/p06_models/scrfd.py` resizes to 640×640 |
| MobileFaceNet | `(1, 3, 112, 112)` RGB | `(1, 512)` embedding | `(pixel - 127.5) / 127.5` | Trained with ArcFace on WebFace600K (`w600k_mbf`) |

**Benchmarks** (published InsightFace model-zoo numbers):
- LFW: **99.7%** accuracy
- CFP-FP: **97.0%** accuracy
- IJB-C: **93.9%** TAR @ FAR=1e-4

**Alternatives considered** (not adopted):
- **EdgeFace-S (CVPR-W 2023)** — +1.7% IJB-C over buffalo_s, but ships PyTorch-only (no official ONNX) and is CC-BY-NC-SA-4.0 (non-commercial). Revisit only if (a) commercial use isn't required or (b) you export + re-QA the ONNX yourself.
- **buffalo_l** — ResNet-50 recogniser, 166 MB, ~+0.5% IJB-C over buffalo_s. Fine if accuracy > size, but 13× larger footprint.
- **YuNet + SFace** (`pretrained/access-face_recognition/`) — alternate pipeline already in use by the face-recognition benchmark; stays in place unchanged. SCRFD + MobileFaceNet is the `core/p06_models/` default that the p10 face tests exercise.

**Output-format note**: The prior repo `scrfd_500m.onnx` was a non-standard 12-output export (cls + obj + bbox + kps × 3 strides, 1 anchor). `core/p06_models/scrfd.py` supports both 9- and 12-output formats (+6-output legacy), but the 9-output buffalo_s weight is the one matched to the recogniser and has the cleanest decode path.

---

## Nitto Denko Safety Models (`pretrained/nitto_denko/`) — **DEPRECATED / legacy, pre-benchmark**

> ⚠️ **Deprecated section.** Populated during initial model exploration (Q1 2026). **Not** current benchmark results. For verified "best pretrained" per feature, use the per-feature `CLAUDE.md` or the "Quick Reference" at the bottom of this file. "Inference Quality" notes here are small-sample anecdotes (N=5–8), not val-split mAP. Do not pick checkpoints from these tables for new training runs.


### Fire Detection (`pretrained/nitto_denko/fire_detection/`)

| Model | File | Size | Classes | License | Source | Inference Quality |
|-------|------|------|---------|---------|--------|------------------|
| YOLOv10n fire | `best.pt` | 5.5 MB | fire-smoke, fog, sol, fire, factory-smoke | Apache-2.0 | [touati-kamel/yolov10n-forest-fire-detection](https://huggingface.co/touati-kamel/yolov10n-forest-fire-detection) | Fair (3/7 on test) |
| YOLOv8s fire | `yolov8s_fire.pt` | 22 MB | fire-smoke, fog, sol, fire, factory-smoke | Apache-2.0 | [touati-kamel/yolov8s-forest-fire-detection](https://huggingface.co/touati-kamel/yolov8s-forest-fire-detection) | Better (4/7 on test) |
| YOLOv10 fire-smoke | `yolov10_fire_smoke/best.pt` | 62 MB | fire, smoke | Apache-2.0 | [TommyNgx/YOLOv10-Fire-and-Smoke-Detection](https://huggingface.co/TommyNgx/YOLOv10-Fire-and-Smoke-Detection) | Not tested |
| ViT fire detection | `vit_fire_detection/pytorch_model.bin` | 328 MB | fire, no-fire (cls) | Apache-2.0 | [EdBianchi/vit-fire-detection](https://huggingface.co/EdBianchi/vit-fire-detection) | Classification |
| ViT fire-smoke v2 | `vit_fire_smoke_v2/model.safetensors` | 328 MB | fire, smoke, neither (cls) | Apache-2.0 | [shawnmichael/vit-fire-smoke-detection-v2](https://huggingface.co/shawnmichael/vit-fire-smoke-detection-v2) | Classification |

**Recommendation**: Benchmark `yolox_m.pth` and `dfine_m_coco.pt` (both COCO pretrained, architecture-compatible) on the fire dataset. The `nitto_denko/fire_detection/` YOLOv8/v10 models are useful for visual QA only — YOLOv8 weights cannot be loaded into a YOLOX or D-FINE architecture.

### Helmet/PPE Detection (`pretrained/nitto_denko/helmet_detection/`)

| Model | File | Size | Classes | License | Source | Inference Quality |
|-------|------|------|---------|---------|--------|------------------|
| YOLOv8n hardhat | `best.pt` | 6 MB | Hardhat, NO-Hardhat | — | [keremberke/yolov8n-hard-hat-detection](https://huggingface.co/keremberke/yolov8n-hard-hat-detection) | Good (8 hardhats on construction) |
| YOLOv8 PPE | `yolov8_ppe_detection/best.pt` | 6 MB | PPE classes | MIT | [Hansung-Cho/yolov8-ppe-detection](https://huggingface.co/Hansung-Cho/yolov8-ppe-detection) | Not tested |
| Vyra YOLO PPE | `vyra_yolo_ppe/best.pt` | 50 MB | PPE classes | CC-BY-4.0 | [Hexmon/vyra-yolo-ppe-detection](https://huggingface.co/Hexmon/vyra-yolo-ppe-detection) | Not tested |
| YOLOS-tiny hardhat | `yolos_tiny_hardhat/pytorch_model.bin` | 25 MB | hardhat classes | Apache-2.0 | [DunnBC22/yolos-tiny-Hard_Hat_Detection](https://huggingface.co/DunnBC22/yolos-tiny-Hard_Hat_Detection) | Not tested |

**Recommendation**: Use `best.pt` (keremberke) for quick inference. Fine-tune `yolov8_ppe_detection/best.pt` (MIT license) for production.

### Fall Detection (`pretrained/nitto_denko/fall_detection/`)

| Model | File | Size | Classes | License | Source | Inference Quality |
|-------|------|------|---------|---------|--------|------------------|
| YOLOv8n fall | `best.pt` | 6 MB | Fall-Detected | openrail | [kamalchibrani/yolov8_fall_detection_25](https://huggingface.co/kamalchibrani/yolov8_fall_detection_25) | **Excellent** (0.86 conf, no FP on standing) |

**Recommendation**: Production-ready for basic fall detection. Fine-tune on your factory data for better accuracy.

### Phone Detection (`pretrained/nitto_denko/phone_detection/`)

| Model | File | Size | Classes | License | Source | Inference Quality |
|-------|------|------|---------|---------|--------|------------------|
| YOLOv8n phone | `yolov8n-mobile-phone.pt` | 6 MB | mobile_phone | MIT | [IndUSV/yolov8n-mobile-phone](https://huggingface.co/IndUSV/yolov8n-mobile-phone) | Good (0.98 on clear phone) |
| YOLOv5s cellphone | `yolov5s_cellphone/yolov5s.pt` | 14 MB | cellphone | MIT | [MahekDharod/cellphone-detection-yolov5s](https://huggingface.co/MahekDharod/cellphone-detection-yolov5s) | Not tested |

**Recommendation**: Use `yolov8n-mobile-phone.pt` for inference. Fine-tune for factory-specific phone detection scenarios.

### Pose Estimation (`pretrained/nitto_denko/pose_estimation/`)

| Model | File | Size | Format | License | Source | Inference Quality |
|-------|------|------|--------|---------|--------|------------------|
| YOLOv8n-pose | `yolov8n-pose.pt` | 6.5 MB | PyTorch | AGPL-3.0 | Ultralytics official | **Excellent** (7/7 images, 17 keypoints) |
| YOLOv8n-pose ONNX | `onnx/model.onnx` | 13 MB | ONNX | AGPL-3.0 | [Xenova/yolov8n-pose](https://huggingface.co/Xenova/yolov8n-pose) | Excellent |

**Note**: AGPL-3.0 license — evaluate compliance for commercial deployment. Alternative: use existing `pretrained/rtmpose_s_256x192.onnx` (Apache-2.0, RTMPose).

### Face Detection (`pretrained/nitto_denko/face_recognition/`)

| Model | File | Size | License | Source | Inference Quality |
|-------|------|------|---------|--------|------------------|
| SCRFD-500M PyTorch | `models/scrfd_500m_bnkps/model.pth` | 2.5 MB | MIT | [public-data/insightface](https://huggingface.co/public-data/insightface) | Works (3 faces on group photo) |
| SCRFD-2.5G PyTorch | `models/scrfd_2.5g_bnkps/model.pth` | 3.2 MB | MIT | public-data/insightface | Higher accuracy |
| Buffalo-L det ONNX | `models/buffalo_l/det_10g.onnx` | 16 MB | MIT | public-data/insightface | ONNX only |
| YOLOv11n face | `yolov11n_face/model.pt` | 5.3 MB | Apache-2.0 | [AdamCodd/YOLOv11n-face-detection](https://huggingface.co/AdamCodd/YOLOv11n-face-detection) | Not tested |
| YOLOv11x face | `yolov11x_face/model.pt` | 110 MB | Apache-2.0 | [AdamCodd/YOLOv11x-face-detection](https://huggingface.co/AdamCodd/YOLOv11x-face-detection) | High accuracy |

### Person Detection (`pretrained/nitto_denko/person_detection/`)

| Model | File | Size | License | Source |
|-------|------|------|---------|--------|
| YOLO person | `yolo_person/yolo_person.pt` | 6 MB | MIT | [lazylearn/yolo_person_detection](https://huggingface.co/lazylearn/yolo_person_detection) |
| YOLO26 person (small) | `yolo26_person_prw/weights/small_full_ft.pt` | 20 MB | CC-BY-4.0 | [simoswish/PersonDetector_YOLO26_PRW](https://huggingface.co/simoswish/PersonDetector_YOLO26_PRW) |
| YOLO26 person (large) | `yolo26_person_prw/weights/large_full_ft.pt` | 51 MB | CC-BY-4.0 | simoswish/PersonDetector_YOLO26_PRW |

### Safety Shoes (No pretrained model available)

No open-source pretrained model exists for safety shoes detection/classification. Fine-tune from COCO pretrained weights (YOLOX-Nano for detection, MobileNetV3-Small for classification) on your own data.

---

## Smart Parking Models (`../smart_parking/pretrained/`)

### Vehicle Detection (`../smart_parking/pretrained/vehicle_detection_*`)

| Model | File | Size | Classes | License | Source |
|-------|------|------|---------|---------|--------|
| YOLOv8n COCO | `vehicle_detection_ultralytics/yolov8n.pt` | 6.3 MB | 80 (COCO) | ⚠️ AGPL-3.0 | Ultralytics |
| YOLOv8s COCO | `vehicle_detection_ultralytics/yolov8s.pt` | 22 MB | 80 (COCO) | ⚠️ AGPL-3.0 | Ultralytics |
| YOLOv8m COCO | `vehicle_detection_ultralytics/yolov8m.pt` | 50 MB | 80 (COCO) | ⚠️ AGPL-3.0 | Ultralytics |
| YOLO11n COCO | `vehicle_detection_ultralytics/yolo11n.pt` | 5.4 MB | 80 (COCO) | ⚠️ AGPL-3.0 | Ultralytics |
| YOLO11s COCO | `vehicle_detection_ultralytics/yolo11s.pt` | 19 MB | 80 (COCO) | ⚠️ AGPL-3.0 | Ultralytics |
| YOLO11m COCO | `vehicle_detection_ultralytics/yolo11m.pt` | 39 MB | 80 (COCO) | ⚠️ AGPL-3.0 | Ultralytics |
| YOLO26 vehicle | `vehicle_detection_yolo26/weights/best.pt` | 42 MB | car | — | [rujutashashikanjoshi/yolo26-testA-vehicle-detection](https://huggingface.co/rujutashashikanjoshi/yolo26-testA-vehicle-detection-4931_full-100m) |
| Highway vehicle | `highway_vehicle_detection/weights/best.pt` | 50 MB | vehicles | MIT | [vietnguyennn0705/highway-vehicle-detection](https://huggingface.co/vietnguyennn0705/highway-vehicle-detection) |
| YOLOX-Traffic nano | `traffic_yolox_nano/traffic_yolox_nano.onnx` | 3.6 MB | traffic | — | Wayne1227/traffic_yolox_nano |

**Note**: Ultralytics models are AGPL-3.0. For commercial use, prefer root YOLOX/D-FINE/RT-DETR models (Apache-2.0).

### License Plate Detection (`../smart_parking/pretrained/license_plate_*`)

| Model | File | Size | License | Source | Inference Quality |
|-------|------|------|---------|--------|------------------|
| YOLOv11 nano | `license_plate_yolov11/...-v1n.pt` | 5.3 MB | — | [morsetechlab/yolov11-license-plate-detection](https://huggingface.co/morsetechlab/yolov11-license-plate-detection) | **Excellent** (5/5 plates) |
| YOLOv11 small | `license_plate_yolov11/...-v1s.pt` | 19 MB | — | morsetechlab (17.7k downloads) | Excellent |
| YOLOv11 medium | `license_plate_yolov11/...-v1m.pt` | 39 MB | — | morsetechlab | — |
| YOLOv11 large | `license_plate_yolov11/...-v1l.pt` | 49 MB | — | morsetechlab | — |
| YOLOv11 xlarge | `license_plate_yolov11/...-v1x.pt` | 110 MB | — | morsetechlab | — |
| YOLOv8 plate | `license_plate_yolov8/best.pt` | 6 MB | MIT | [Koushim/yolov8-license-plate-detection](https://huggingface.co/Koushim/yolov8-license-plate-detection) | Good |
| YOLOv5n plate | `license_plate_yolov5n/best.pt` | 3.7 MB | — | [keremberke/yolov5n-license-plate](https://huggingface.co/keremberke/yolov5n-license-plate) | — |
| YOLOv26 plate | `license_plate_yolov26/best.pt` | 5.2 MB | Apache-2.0 | [CodexParas/car-plate-detection-yolov26](https://huggingface.co/CodexParas/car-plate-detection-yolov26) | — |
| YOLOS-small plate | `license_plate_yolos/pytorch_model.bin` | 117 MB | Apache-2.0 | [nickmuchi/yolos-small-finetuned-license-plate-detection](https://huggingface.co/nickmuchi/yolos-small-finetuned-license-plate-detection) | — |
| YOLOS rego plates | `license_plate_yolos_small/pytorch_model.bin` | 117 MB | Apache-2.0 | [nickmuchi/yolos-small-rego-plates-detection](https://huggingface.co/nickmuchi/yolos-small-rego-plates-detection) | — |

### Vehicle Classification (`../smart_parking/pretrained/vehicle_*classification*`)

| Model | File | Size | Classes | License | Source | Inference Quality |
|-------|------|------|---------|---------|--------|------------------|
| ViT classification | `vehicle_classification/model.safetensors` | 328 MB | 17 types | Apache-2.0 | [malifiahm/vehicle_classification](https://huggingface.co/malifiahm/vehicle_classification) | **Very good** (7/8 correct) |
| ViT 10-types | `vehicle_10types_classification/model.safetensors` | 328 MB | 10 types | Apache-2.0 | [dima806/vehicle_10_types_image_detection](https://huggingface.co/dima806/vehicle_10_types_image_detection) | Not tested |
| ViT multiclass | `vehicle_multiclass_classification/model.safetensors` | 328 MB | multi-class | Apache-2.0 | [Tianmu28/vehicle_multiclass_classification](https://huggingface.co/Tianmu28/vehicle_multiclass_classification) | Not tested |
| Car brands | `car_brands_classification/pytorch_model.bin` | 332 MB | car brands | Apache-2.0 | [lamnt2008/car_brands_classification](https://huggingface.co/lamnt2008/car_brands_classification) | Not tested |

### Parking Occupancy (`../smart_parking/pretrained/parking_segformer/`)

| Model | File | Size | License | Source |
|-------|------|------|---------|--------|
| SegFormer-Large parking | `best_model.ckpt` | ~500 MB | — | [UTEL-UIUC/SegFormer-large-parking](https://huggingface.co/UTEL-UIUC/SegFormer-large-parking) |

**Note**: Lightning checkpoint format. For training, use `nvidia/segformer-b0-finetuned-ade-512-512` (auto-downloads, Apache-2.0).

### Edge-Optimized ONNX Models (`../smart_parking/pretrained/edge/`)

| Model | File | Size | Format |
|-------|------|------|--------|
| YOLO26n INT8 | `vehicle_detection/yolo26n-onnx/onnx/model_int8.onnx` | 2.8 MB | ONNX INT8 |
| YOLOv10n INT8 | `vehicle_detection/yolov10n-onnx/onnx/model_int8.onnx` | 2.6 MB | ONNX INT8 |
| MobileNetV3-S INT8 | `vehicle_classification/mobilenetv3-small-onnx/onnx/model_int8.onnx` | 2.2 MB | ONNX INT8 |
| MobileNetV4-S INT8 | `vehicle_classification/mobilenetv4-small-onnx/onnx/model_int8.onnx` | 3.8 MB | ONNX INT8 |
| SegFormer-B0 | `parking_segmentation/segformer-b0-ade-onnx/onnx/model.onnx` | 15 MB | ONNX FP32 |
| YOLOv11n plate | `license_plate/yolov11n-lp/license-plate-finetune-v1n.pt` | 5.3 MB | PyTorch |

---

## Quick Reference: Best Model Per Use Case

> Updated 2026-04-17 based on val-split benchmark results. See each feature's `CLAUDE.md` for full tables.
>
> **Benchmark methodology**: every `val mAP50` in this section was produced by `uv run features/<feature>/code/benchmark.py --split val` and persisted to `features/<feature>/eval/benchmark_results.json`. mAP follows the COCO standard at IoU=0.50. To reproduce or refresh, re-run the benchmark for the relevant feature.

### For Fine-Tuning (start from these weights)

| Use Case | Best Pretrained Start | Path | val mAP50 |
|---|---|---|---|
| Fire Detection | SalahALHaismawi_yolov26-fire-detection | `pretrained/safety-fire_detection/SalahALHaismawi_yolov26-fire-detection/best.pt` | 0.153 |
| Helmet/PPE | melihuzunoglu_yolov11_ppe | `pretrained/ppe-helmet_detection/melihuzunoglu_yolov11_ppe.pt` | 0.105 |
| Fall Detection | yolov11_fall_melihuzunoglu | `pretrained/safety-fall-detection/yolov11_fall_melihuzunoglu.pt` | 0.050 |
| Safety Shoes | COCO YOLOX-S (no foot detector exists) | `pretrained/yolox_s.pth` | 0.000 |
| Phone Usage | COCO YOLOX-S (action class, no pretrained) | `pretrained/yolox_s.pth` | 0.000 |
| Pose Estimation (shared by 5 `safety-poketenashi_*` rule features) | DWPose ONNX (interim; RTMPose fine-tune pending) | `pretrained/safety-poketenashi/dwpose_384_pose.onnx` | det=1.0 |

### For Direct Inference (pretrained-only, no training needed)

| Use Case | Model | Path | Metric |
|---|---|---|---|
| Zone Intrusion | yolox_tiny | `pretrained/yolox_tiny.pth` | acc=1.0, 6.9ms |
| Zone Intrusion (edge) | yolov10n | `pretrained/access-zone_intrusion/yolov10n.pt` | acc=0.875, 4.6ms |
| Face Detection | yunet_2023mar | `pretrained/access-face_recognition/yunet_2023mar.onnx` | det_rate=0.933, 2.1ms |
| Face Recognition | yunet + sface_fp32 | `pretrained/access-face_recognition/{yunet_2023mar,sface_2021dec}.onnx` | rank-1=1.0 |
| Pose (`safety-poketenashi_*` rule family — 5 features) | dwpose_384_pose | `pretrained/safety-poketenashi/dwpose_384_pose.onnx` (shared storage) | det=1.0, 13ms |
| Helmet (ONNX serving) | HudatersU_safety_helmet | `pretrained/ppe-helmet_detection/HudatersU_safety_helmet.onnx` | mAP50=0.124 |

### COCO Backbone Pool (for custom fine-tuning)

| Model | File | Size | License |
|---|---|---|---|
| YOLOX-S | `yolox_s.pth` | 69 MB | Apache-2.0 |
| YOLOX-M | `yolox_m.pth` | 194 MB | Apache-2.0 |
| YOLOX-Tiny | `yolox_tiny.pth` | 39 MB | Apache-2.0 |
| YOLOX-Nano | `yolox_nano.pth` | 7.4 MB | Apache-2.0 |
| D-FINE-S | `dfine_s_coco.pt` | 40 MB | Apache-2.0 |
| D-FINE-M | `dfine_m_coco.pt` | 76 MB | Apache-2.0 |
| RT-DETR-R18 | `rtdetr_v2_r18_coco.pt` | 78 MB | Apache-2.0 |

---

## License Summary

| License | Models |
|---------|--------|
| Apache-2.0 | YOLOX (all sizes), D-FINE, RT-DETRv2, ViT classifiers, YOLOS plates, YOLOv10 fire, YOLOv11 face |
| MIT | SCRFD, MobileFaceNet, YOLOv8 PPE (Hansung-Cho), phone, person, license plate (keremberke/Koushim) |
| CC-BY-4.0 | Vyra PPE, YOLO26 person (simoswish PRW) |
| openrail | YOLOv8 fall (kamalchibrani) |
| AGPL-3.0 | Ultralytics YOLOv8 / YOLO11 (caution for commercial) |
| Unverified | keremberke YOLOv8 hardhat, morsetechlab plates, YOLO26 vehicle — check HF model cards before commercial use |

**For commercial deployment**: prefer Apache-2.0/MIT. Avoid AGPL-3.0 (Ultralytics, YOLOv8n-pose) unless the AGPL source-disclosure terms are acceptable. When in doubt, run `curl -s https://huggingface.co/api/models/<repo>` and check `cardData.license`.

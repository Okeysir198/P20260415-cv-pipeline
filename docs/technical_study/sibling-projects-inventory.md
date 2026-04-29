# Sibling-Project Pretrained Inventory

Recursive scan of two sibling repos for pretrained model weights relevant to the
Phase 1 use cases. Symlinks (no copies) into `ai/pretrained/<feature>/` so the
files stay read-only in their source tree but are usable from `edge_ai`.

## 1. Sources scanned

1. `/home/nthanhtrung/Documents/05_Team/02_Vietsol/01_Projects/visual_core/`
   (recursive, `.venv` and `site-packages` excluded — they only held pip
   editable-install `.pth` shims, not model weights).
2. `/home/nthanhtrung/Documents/05_Team/02_Vietsol/01_Projects/dms_oms/pretrained/`
   (recursive).

Extensions matched: `*.pth *.pt *.onnx *.safetensors *.h5 *.tflite *.bin`.

## 2. Totals

| Source        | Files (real weights) | Aggregate size |
| ------------- | -------------------- | -------------- |
| visual_core   | 25                   | ~3.6 GB        |
| dms_oms       | 19                   | ~1.36 GB       |

By extension across both sources: `.pt` 22, `.pth` 9, `.onnx` 6,
`.safetensors` 3, `.bin` 4, `.h5` 1.

## 3. Mapping table

### visual_core

| Source path | Size | Inferred feature | Symlink target | Notes |
| --- | --- | --- | --- | --- |
| `01_code/checkpoints/feat_detect/rfdetr/rf-detr-base.pth` | 372 MB | access-zone_intrusion | `visualcore_rf-detr-base.pth` | Generic COCO detector |
| `01_code/rf-detr-base.pth` | 372 MB | (dup) | — | Identical copy of above; not linked |
| `01_code/checkpoints/feat_detect/yolov8/yolov8l.pt` | 88 MB | access-zone_intrusion | `visualcore_yolov8l.pt` | Generic COCO YOLOv8-L |
| `01_code/yolov8l.pt` | 88 MB | (dup) | — | Same file, not linked |
| `checkpoints/stage1_step/asformer/gtea_split1.pt` | 4.7 MB | safety-fall-detection | `visualcore_asformer_gtea_split1.pt` | Temporal action segmentation (GTEA cooking) — feature extractor candidate for action models |
| `checkpoints/stage1_step/mstcn/gtea_split1.pt` | 3.3 MB | safety-fall-detection | `visualcore_mstcn_gtea_split1.pt` | MS-TCN segmentation head |
| `checkpoints/stage2_action/c2f_tcn/gtea_split1.pt` | 27 MB | safety-fall-detection | `visualcore_c2f_tcn_gtea_split1.pt` | C2F-TCN action seg |
| `checkpoints/stage2_action/diffact/gtea_split{1..4}.pt` | 5 MB ea | safety-fall-detection | `visualcore_diffact_gtea_split{1..4}.pt` | DiffAct (diffusion action) |
| `checkpoints/stage2_action/fact/gtea_split{1..4}.pt` | 62 MB ea | safety-fall-detection | `visualcore_fact_gtea_split{1..4}.pt` | FACT action transformer |
| `checkpoints/stage2_action/fact/breakfast_split{1..4}.pt` | 498 MB ea | safety-fall-detection | `visualcore_fact_breakfast_split{1..4}.pt` | FACT, Breakfast dataset |
| `checkpoints/stage2_action/fact/epic_kitchens_split1.pt` | 232 MB | safety-fall-detection | `visualcore_fact_epic_kitchens_split1.pt` | FACT, EpicKitchens |
| `checkpoints/stage2_action/fact/egoprocel_split1.pt` | 187 MB | safety-fall-detection | `visualcore_fact_egoprocel_split1.pt` | FACT, EgoProceL |
| `checkpoints/stage2_action/ltcontext/breakfast_split{1..4}.pth` | 3 MB ea | safety-fall-detection | `visualcore_ltcontext_breakfast_split{1..4}.pth` | LTContext head |
| `checkpoints/stage2_action/ltcontext/gtea_split1.pt` | 3 MB | safety-fall-detection | `visualcore_ltcontext_gtea_split1.pt` | LTContext head |

### dms_oms

| Source path | Size | Inferred feature | Symlink target | Notes |
| --- | --- | --- | --- | --- |
| `face_detection/RetinaFace-R50.pth` | 109 MB | access-face_recognition | `dmsoms_RetinaFace-R50.pth` | Strong off-the-shelf face detector |
| `face_detection/yolov11n_face/model.pt` | 5.5 MB | access-face_recognition | `dmsoms_yolov11n_face.pt` | YOLOv11-nano face |
| `face_detection/yolov11n_face/model.onnx` | 10.6 MB | access-face_recognition | `dmsoms_yolov11n_face.onnx` | ONNX export |
| `face_detection/yolov11n_face/model_fp16.onnx` | 5.4 MB | access-face_recognition | `dmsoms_yolov11n_face_fp16.onnx` | FP16 ONNX |
| `face_detection/yolov11n_face/model_gpu.onnx` | 10.5 MB | access-face_recognition | `dmsoms_yolov11n_face_gpu.onnx` | GPU-optim ONNX |
| `face_detection/yolov11n_face/model_nms.onnx` | 10.6 MB | access-face_recognition | `dmsoms_yolov11n_face_nms.onnx` | ONNX with NMS baked in |
| `person/rtdetr_r18vd/model.safetensors` | 81 MB | access-zone_intrusion | `dmsoms_rtdetr_r18vd.safetensors` | RT-DETR R18, generic person/COCO |

## 4. Files skipped (not Phase 1)

| Source path | Size | Reason |
| --- | --- | --- |
| `dms_oms/pretrained/head_pose/6DRepNet_300W_LP_AFLW2000.pth` | 157 MB | DMS head-pose (Phase 2+) |
| `dms_oms/pretrained/head_pose/WHENet.h5` | 18 MB | DMS head-pose |
| `dms_oms/pretrained/distraction/pytorch_model.bin` | 126 MB | DMS distraction classifier |
| `dms_oms/pretrained/distraction/convnext_tiny_driverbox/model.safetensors` | 111 MB | DMS distraction (ConvNeXt) |
| `dms_oms/pretrained/distraction/convnext_tiny_driverbox/training_args.bin` | 5 KB | HF training args, not weights |
| `dms_oms/pretrained/drowsiness/vit_drowsiness/pytorch_model.bin` | 343 MB | DMS drowsiness |
| `dms_oms/pretrained/drowsiness/vit_drowsiness/training_args.bin` | 4 KB | HF training args |
| `dms_oms/pretrained/drowsiness/mobilevit_drowsiness/best_model.pt` | 70 MB | DMS drowsiness |
| `dms_oms/pretrained/drowsiness/yolo_cls_drowsiness/{best,last}.pt` | 57 MB ea | DMS drowsiness classifier |
| `dms_oms/pretrained/seatbelt/yolov8_seatbelt/best.pt` | 52 MB | DMS seatbelt |
| `dms_oms/pretrained/emotion/FER_static_ResNet50_AffectNet.pt` | 99 MB | DMS emotion |
| `visual_core/01_code/rf-detr-base.pth` | 372 MB | Duplicate of `feat_detect/rfdetr/rf-detr-base.pth` |
| `visual_core/01_code/yolov8l.pt` | 88 MB | Duplicate of `feat_detect/yolov8/yolov8l.pt` |
| `visual_core/.../site-packages/*.pth` | small | Python pip editable shims, not model weights |
| `visual_core/.../onnxruntime/datasets/*.onnx` | <1 KB ea | ORT test fixtures |

### Ambiguous classifications (only one link created)

- The `feat_detect/{rfdetr,yolov8}` checkpoints are generic COCO models. They
  are linked to **access-zone_intrusion** (person detector role) but could
  also serve as the upstream detector for **ppe-helmet_detection** or
  **ppe-shoes_detection**. PPE folders already hold purpose-trained
  PPE/helmet weights, so duplication was not needed.
- The visual_core `stage2_action/fact` `breakfast_*` and `epic_kitchens_*`
  weights are large (500 MB+ each). They are linked under
  **safety-fall-detection** as candidate temporal-action backbones, but
  they are trained on cooking datasets, not falls — treat them as
  *transfer-learning starting points*, not ready inference models.

## 5. Per-destination summary

| Feature folder | New symlinks added |
| --- | --- |
| `safety-fire_detection/` | 0 |
| `ppe-helmet_detection/` | 0 |
| `ppe-shoes_detection/` | 0 |
| `safety-fall-detection/` | 22 |
| `safety-fall_pose_estimation/` | 0 |
| `safety-poketenashi_*` (5 rule features) | 0 |
| `access-zone_intrusion/` | 6 (3 new from sibling repos + 3 pre-existing local symlinks) |
| `access-face_recognition/` | 6 |

Total: **34 symlinks** across **3 features** (`safety-fall-detection`,
`access-zone_intrusion`, `access-face_recognition`).

The other five Phase 1 features (fire detection, helmet PPE, shoes PPE, fall
pose, poketenashi) already have purpose-built pretrained weights downloaded
locally; no sibling-repo files added meaningful coverage to those.

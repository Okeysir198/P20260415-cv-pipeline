# Technical Approach — Phase 1 AI Model Development
## Factory Smart Camera Safety System

**Date:** March 14, 2026
**Target Edge Chips:** AX650N (AXera) / CV186AH (Sophgo)

---

## Document Structure

This file covers cross-cutting concerns: license compliance, edge chips, export pipeline, deployment strategy, roadmap, and gap analysis.

Per-use-case technical details (architecture, dataset, training, alert logic, edge deployment) are in the **platform docs**:

| Use Case | Platform Doc |
|---|---|
| **A: Fire & smoke detection** | [`docs/03_platform/safety-fire_detection.md`](../../03_platform/safety-fire_detection.md) |
| **B: PPE helmet compliance** | [`docs/03_platform/ppe-helmet_detection.md`](../../03_platform/ppe-helmet_detection.md) |
| **F: PPE safety shoes** | [`docs/03_platform/ppe-shoes_detection.md`](../../03_platform/ppe-shoes_detection.md) |
| **G: Fall detection (classification)** | [`docs/03_platform/safety-fall_classification.md`](../../03_platform/safety-fall_classification.md) |
| **G: Fall detection (pose)** | [`docs/03_platform/safety-fall_pose_estimation.md`](../../03_platform/safety-fall_pose_estimation.md) |
| **H: Behavioral violations (poketenashi)** | [`docs/03_platform/safety-poketenashi.md`](../../03_platform/safety-poketenashi.md) |
| **I: Zone intrusion** | [`docs/03_platform/access-zone_intrusion.md`](../../03_platform/access-zone_intrusion.md) |

All platform docs follow the standard template: [`docs/03_platform/_TEMPLATE.md`](../../03_platform/_TEMPLATE.md)

---

## 1. Model License Compliance

> **Requirement:** All models must be freely usable for commercial deployment without royalties or open-source obligations.

### Approved Models (FREE Commercial Use)

#### Detection Models

| Model | License | Repo | Commercial Status |
|---|---|---|---|
| **YOLOX** | Apache 2.0 | Megvii-BaseDetection/YOLOX | FREE — no restrictions |
| **RT-DETR / RT-DETRv2** | Apache 2.0 | lyuwenyu/RT-DETR | FREE — no restrictions |
| **D-FINE** (N/S/M/L/X) | Apache 2.0 | Peterande/D-FINE | FREE — no restrictions. ICLR 2025 Spotlight |
| **RF-DETR** (Nano–Large) | Apache 2.0 | roboflow/rf-detr | FREE — only Nano through Large variants |
| **DINO** | Apache 2.0 | IDEA-Research/DINO | FREE — not real-time |
| **Co-DETR** | MIT | Sense-X/Co-DETR | FREE — not real-time |
| **EfficientDet** | Apache 2.0 | google/automl | FREE — no restrictions |

#### Pose Estimation Models

| Model | License | Repo | Commercial Status |
|---|---|---|---|
| **RTMPose / MMPose** | Apache 2.0 | open-mmlab/mmpose | FREE — best edge option, proven on AX650N (4.79ms) |
| **RTMO (one-stage pose)** | Apache 2.0 | open-mmlab/mmpose | FREE — no separate person detector needed |
| **Lite-HRNet** | Apache 2.0 | HRNet/Lite-HRNet | FREE — ultra-lightweight (1.8M params) |
| **HRNet** | MIT | HRNet/HRNet-Human-Pose-Estimation | FREE — no restrictions |
| **MoveNet** | Apache 2.0 | TensorFlow Hub (Google) | FREE — lightweight, TFLite-first |
| **MediaPipe Pose (BlazePose)** | Apache 2.0 | google-ai-edge/mediapipe | FREE — 33 landmarks (3D), TFLite-first, mobile-optimized |

#### Classification / Action / Tracking

| Model | License | Commercial Status |
|---|---|---|
| **MobileNetV3** | BSD-3-Clause | FREE (torchvision) |
| **EfficientNet** | Apache 2.0 | FREE (Google) |
| **X3D / SlowFast** | Apache 2.0 | FREE (Meta/Facebook Research) |
| **ByteTrack** | MIT | FREE |
| **BoT-SORT** | MIT | FREE |
| **OC-SORT** | MIT | FREE |

### PROHIBITED Models (License Blocks Commercial Use)

| Model | License | Issue | Alternative |
|---|---|---|---|
| **YOLOv5 / YOLOv8 / YOLO11 / YOLO26** | AGPL-3.0 | Must open-source entire project OR pay ~$5,000/yr Enterprise License | Use YOLOX or D-FINE |
| **AlphaPose** | Non-commercial only | Requires commercial license from Shanghai Jiao Tong University | Use RTMPose |
| **ViTPose** | Apache 2.0 (license OK) | Transformer attention layers quantize poorly to INT8 on NPU — not edge-feasible | Use RTMPose (same AP, better INT8) |
| **VideoMAE** | CC-BY-NC 4.0 | Non-commercial only | Use X3D or SlowFast |
| **RF-DETR XL/2XL** | PML 1.0 (Roboflow proprietary) | Requires Roboflow platform | Use RF-DETR-Large or D-FINE-X |

> **Important:** The Ultralytics AGPL-3.0 license applies to ALL their models including future releases. Even internal R&D use or SaaS/cloud deployments require open-sourcing or a paid Enterprise License. Do NOT use any Ultralytics model in this project.

### Selected Model Stack for This Project

| Role | Model | License | Params | Why |
|---|---|---|---|---|
| **Primary Detector (CNN)** | YOLOX-M | Apache 2.0 | 25.3M | Proven, good INT8, existing pipeline |
| **Primary Detector (Transformer)** | D-FINE-S | Apache 2.0 | 10M | Best accuracy/latency, NMS-free, ICLR 2025 |
| **Lightweight Detector** | D-FINE-N | Apache 2.0 | 4M | Smaller than YOLOX-Tiny, higher AP |
| **Alternative Transformer** | RT-DETRv2-R18 | Apache 2.0 | 20M | Proven deployment, discrete sampling |
| **Pose Estimator** | RTMPose-S | Apache 2.0 | 5.47M | Edge-optimized, AX650N proven (4.79ms), 72.2 AP |
| **Pose (high accuracy)** | RTMPose-M | Apache 2.0 | 13.59M | 75.8 AP, pure CNN, excellent INT8 |
| **Pose (ultra-light)** | RTMPose-T | Apache 2.0 | 3.34M | 68.5 AP, best for CV186AH (7.2 TOPS) |
| **Pose (one-stage)** | RTMO-S | Apache 2.0 | ~8M | 67.7 AP, no separate person detector needed |
| **Pose (33 landmarks)** | MediaPipe Pose Full | Apache 2.0 | 3.5M | 33 3D landmarks, TFLite-native, 18-40 FPS mobile CPU |
| **Pose (ultra-light 33)** | MediaPipe Pose Lite | Apache 2.0 | 1.3M | 33 landmarks, ~50 FPS mobile CPU, 3 MB model |
| **Classifier** | MobileNetV3-Small | BSD-3 | 2.5M | Shoe/helmet crop classification |
| **Action Recognition** | X3D-XS | Apache 2.0 | 3.8M | Edge-feasible temporal modeling |
| **Tracker** | ByteTrack | MIT | N/A | Fastest, fixed-camera optimized |

---

## 2. Edge Chip Comparison

| Spec | AX650N (AXera) | CV186AH (Sophgo) |
|---|---|---|
| **INT8 TOPS** | **18 TOPS** | **7.2 TOPS** |
| **INT4 TOPS** | 72 TOPS | 12 TOPS |
| **CPU** | 8x Cortex-A55 @ 1.7GHz | 6x Cortex-A53 @ 1.6GHz |
| **RAM** | 8GB LPDDR4x | Up to 8GB LPDDR4 |
| **Video Decode** | 32x 1080p30 | 16x 1080p30 |
| **Toolchain** | Pulsar2 (ONNX → `.axmodel`) | TPU-MLIR (ONNX → `.cvimodel`/`.bmodel`) |
| **YOLOX support** | Direct (`ax_yolox_steps.cc` in ax-samples) | Direct (official YOLOX deployment guide) |
| **Pose models** | **RTMPose/SimCC (4.79ms)**, HRNet | HRNet-pose (3.43ms INT8) |
| **Multi-model** | NPU scheduler (concurrent) | Dual-core dual-task (concurrent) |
| **Power (est.)** | 5–8W SoC | 5–15W SoC |
| **Quantization** | INT4/INT8/INT16/FP16/BF16, mixed, QAT | INT4/INT8/BF16, mixed, QAT |
| **ISP** | AI-ISP, black-light full-color | 2f-HDR, 3DNR, LDC, fisheye dewarp |

**Verdict:** AX650N is the stronger chip (2.5x NPU performance, better CPU). CV186AH is feasible but requires lighter model variants (YOLOX-S/Tiny) for multi-model scenarios.

---

### D-FINE vs RT-DETRv2 vs YOLOX — Head-to-Head Comparison

| Model | AP (COCO) | Params | Latency (T4 FP16) | NMS-Free | INT8 Quantization | License |
|---|---|---|---|---|---|---|
| YOLOX-Tiny | ~33 | 5.1M | ~3ms | No | Excellent | Apache 2.0 |
| YOLOX-S | ~40 | 9.0M | ~5ms | No | Excellent | Apache 2.0 |
| YOLOX-M | ~46 | 25.3M | ~10ms | No | Excellent | Apache 2.0 |
| **D-FINE-N** | **42.8** | **4M** | **2.12ms** | **Yes** | Good (HGNetv2 backbone) | Apache 2.0 |
| **D-FINE-S** | **48.5** | **10M** | **3.49ms** | **Yes** | Good | Apache 2.0 |
| **D-FINE-M** | **52.3** | **19M** | **5.62ms** | **Yes** | Good | Apache 2.0 |
| D-FINE-L | 54.0 | 31M | 8.07ms | Yes | Good | Apache 2.0 |
| D-FINE-X | 55.8 | 62M | 12.89ms | Yes | Good | Apache 2.0 |
| RT-DETRv2-R18 | 46.5 | 20M | 4.6ms | Yes | Moderate | Apache 2.0 |
| RT-DETRv2-R34 | 48.9 | 31M | 6.2ms | Yes | Moderate | Apache 2.0 |
| RT-DETRv2-R50 | 53.1 | 42M | 9.3ms | Yes | Moderate | Apache 2.0 |

**Key findings:**
- **D-FINE-N (4M params) outperforms YOLOX-S (9.0M params)** in accuracy (42.8 vs ~40 AP) with half the parameters and NMS-free inference
- **D-FINE-S (10M params) matches YOLOX-M (25.3M params)** in accuracy (48.5 vs ~46 AP) with 60% fewer parameters
- D-FINE consistently beats RT-DETRv2 at every scale (+0.4 to +2.4 AP) with 13% lower latency
- YOLOX has the best INT8 quantization robustness (pure CNN), D-FINE is second-best (HGNetv2 CNN backbone + transformer decoder)

### D-FINE Architecture Overview

```
Input (640×640 RGB)
    │
    ▼
┌─────────────────────────────────────────┐
│  HGNetv2 Backbone (CNN)                 │
│  • Efficient CNN feature extractor      │
│  • Scales: N(4M), S(10M), M(19M),      │
│    L(31M), X(62M)                       │
│  • Output: Multi-scale features         │
│    S3(stride 8), S4(16), S5(32)         │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Hybrid Encoder                         │
│  • AIFI (intra-scale self-attention)    │
│  • CCFM (cross-scale CNN fusion)        │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Transformer Decoder with FDR           │
│  • Fine-grained Distribution Refinement │
│    → Iteratively refines probability    │
│      distributions over boundary        │
│      locations (not direct coordinates) │
│  • GO-LSD Self-Distillation             │
│    → Deeper layers teach shallower      │
│      layers for better localization     │
│  • Result: significantly more precise   │
│    bounding boxes than RT-DETR          │
└─────────────────────────────────────────┘
    │
    ▼
  Set of detections (NMS-free, end-to-end)
```

### RT-DETRv2 Architecture Overview

```
Input (640×640 RGB)
    │
    ▼
┌─────────────────────────────────────────┐
│  ResNet Backbone (CNN)                  │
│  • R18(20M), R34(31M), R50(42M),       │
│    R101(76M)                            │
│  • Well-studied, proven quantization    │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Hybrid Encoder (same as RT-DETR v1)    │
│  • AIFI + CCFM                          │
│  • v2 improvement: selective multi-scale│
│    sampling points per feature level    │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Transformer Decoder                    │
│  • v2 improvement: discrete sampling   │
│    operator (replaces grid_sample)      │
│    → Better ONNX/TensorRT deployment   │
│  • Uncertainty-minimal query selection  │
└─────────────────────────────────────────┘
    │
    ▼
  Set of detections (NMS-free, end-to-end)
```

---

## 3. Common Export & Deployment Pipeline

### 3.1 Training → Edge Export Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                    GPU Training Server                            │
│                                                                  │
│  Dataset (YOLO format)                                           │
│      │                                                           │
│      ▼                                                           │
│  step05: Data Transforms (Mosaic, MixUp, HSV, Affine)           │
│      │                                                           │
│      ▼                                                           │
│  step07: Training (YOLOX-M / D-FINE / RTMPose)                  │
│      │  • SimOTA label assignment                                │
│      │  • FocalLoss + IoULoss (CIoU)                            │
│      │  • EMA (decay=0.9998)                                     │
│      │  • Cosine LR + warmup                                     │
│      │  • AMP (mixed precision)                                  │
│      │  • Grad clip = 35.0                                       │
│      │                                                           │
│      ▼                                                           │
│  step08: Evaluation (mAP@0.5, Precision, Recall, Confusion)    │
│      │                                                           │
│      ▼                                                           │
│  step09: Export to ONNX (opset 11+) → onnxsim simplify          │
│      │                                                           │
│      ├──► AX650N: pulsar2 build --target_hardware AX650          │
│      │         → .axmodel (INT8 PTQ, 100-1000 cal images)        │
│      │                                                           │
│      └──► CV186AH: model_transform.py → MLIR                    │
│                → run_calibration.py → model_deploy.py            │
│                → .cvimodel (INT8 PTQ, 100-1000 cal images)       │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 Quantization Strategy

| Technique | When to Use |
|---|---|
| **INT8 PTQ** | Default for all models. Use validation set as calibration data (100-1000 images) |
| **INT4 PTQ** | Only on AX650N when FPS is insufficient with INT8. Verify mAP drop < 2% |
| **Mixed Precision** | Per-layer INT8/FP16 for sensitive layers (classification head, attention). Both chips support this |
| **QAT (Quantization-Aware Training)** | If PTQ causes >3% mAP drop. Train with fake quantization nodes inserted |

### 3.3 Model Size Budget

| Model Tier | License | Params | FP32 Size | INT8 Size | AX650N FPS (est.) | CV186AH FPS (est.) |
|---|---|---|---|---|---|---|
| **D-FINE-N** | Apache 2.0 | **4M** | **~8MB** | **~2MB** | **55+** | **28+** |
| **D-FINE-S** | Apache 2.0 | **10M** | **~20MB** | **~5MB** | **35+** | **18+** |
| **D-FINE-M** | Apache 2.0 | **19M** | **~38MB** | **~10MB** | **25+** | **12+** |
| RT-DETRv2-R18 | Apache 2.0 | 20M | ~40MB | ~10MB | 30+ | 15+ |
| RT-DETRv2-R50 | Apache 2.0 | 42M | ~84MB | ~21MB | 15+ | 8+ |
| YOLOX-Tiny | Apache 2.0 | 5.1M | ~20MB | ~5MB | 50+ | 25+ |
| YOLOX-S | Apache 2.0 | 9.0M | ~35MB | ~9MB | 35+ | 18+ |
| YOLOX-M | Apache 2.0 | 25.3M | ~100MB | ~25MB | 25–30 | 12–15 |
| **RTMPose-T** | Apache 2.0 | **3.34M** | **~7MB** | **~2MB** | **50+** | **25+** |
| **RTMPose-S** | Apache 2.0 | **5.47M** | **~11MB** | **~3MB** | **40+** | **20+** |
| **RTMPose-M** | Apache 2.0 | **13.59M** | **~27MB** | **~7MB** | **25+** | **12+** |
| RTMO-S (one-stage) | Apache 2.0 | ~8M | ~16MB | ~4MB | 30+ | 15+ |
| **MediaPipe Pose Lite** | Apache 2.0 | **1.3M** | **~3MB** | **~1.5MB** | **50+** | **25+** |
| **MediaPipe Pose Full** | Apache 2.0 | **3.5M** | **~6MB** | **~3MB** | **35+** | **18+** |
| MediaPipe Pose Heavy | Apache 2.0 | ~15M | ~26MB | ~13MB | 10+ | 5+ |
| Lite-HRNet-30 | Apache 2.0 | 1.8M | ~4MB | ~1MB | 60+ | 30+ |
| HRNet-W32 | MIT | 28.5M | ~115MB | ~29MB | 20+ | 10+ |
| ~~YOLO26-N~~ | ~~AGPL-3.0~~ | ~~3M~~ | — | — | — | — |
| ~~YOLO26-S~~ | ~~AGPL-3.0~~ | ~~9.5M~~ | — | — | — | — |

> **Struck-through models are NOT approved** for this project due to AGPL-3.0 license restrictions.

---


## 10. Multi-Model Edge Deployment Strategy

### 10.1 AX650N Model Loading (18 TOPS, Comfortable)

| Zone Type | Models Loaded | Est. Combined FPS | NPU Utilization |
|---|---|---|---|
| Factory floor (general) | Fire + Helmet + Intrusion | ~15 FPS each | ~70% |
| Factory floor (full PPE) | Fire + Helmet + Shoes (2-stage) | ~12 FPS each | ~85% |
| Stairs / corridors | Poketenashi (person+phone+pose) + Intrusion | ~15 FPS | ~80% |
| Restricted areas | Intrusion only | ~50 FPS | ~20% |
| Warehouse / storage | Fall (pose) + Fire + Intrusion | ~15 FPS each | ~75% |

### 10.2 CV186AH Model Loading (7.2 TOPS, Constrained)

| Strategy | Description | Impact |
|---|---|---|
| **Use lighter models** | YOLOX-Tiny instead of YOLOX-M | -5% accuracy, +2x FPS |
| **Frame interleaving** | Model A on odd frames, Model B on even frames | Half latency per model |
| **Priority scheduling** | Fire always runs; PPE only during work hours | Saves NPU time |
| **INT4 quantization** | 12 TOPS at INT4 (1.7x vs INT8) | Verify accuracy holds |

### 10.3 Memory Budget

| Model | INT8 Size | Runtime Memory (est.) |
|---|---|---|
| YOLOX-Tiny | ~5MB | ~50MB |
| YOLOX-M | ~25MB | ~200MB |
| RTMPose-S | ~6MB | ~60MB |
| MobileNetV3 | ~2MB | ~30MB |
| ByteTrack | N/A | ~20MB |
| **2-model deployment** | ~30MB | **~300MB** |
| **3-model deployment** | ~36MB | **~360MB** |

Both chips have 8GB RAM — memory is not a constraint.

### 10.4 Inference Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Edge Device (AX650N / CV186AH)                         │
│                                                         │
│  ┌───────────────┐                                      │
│  │ RTSP Receiver  │ ← Camera RTSP stream (1080p30)      │
│  │ (HW decoder)   │                                     │
│  └───────┬────────┘                                     │
│          │ decoded frames                               │
│          ▼                                              │
│  ┌───────────────┐                                      │
│  │ Preprocessor   │ resize, normalize, BGR→RGB          │
│  │ (ISP/CPU)      │                                     │
│  └───────┬────────┘                                     │
│          │ 640×640 tensor                               │
│          ▼                                              │
│  ┌───────────────┐                                      │
│  │ NPU Inference  │ YOLOX / RTMPose / MobileNetV3      │
│  │ (.axmodel /    │ Multi-model scheduler               │
│  │  .cvimodel)    │                                     │
│  └───────┬────────┘                                     │
│          │ detections / keypoints                       │
│          ▼                                              │
│  ┌───────────────┐                                      │
│  │ Post-Process   │ Decode, NMS, tracking (CPU)         │
│  │ (CPU)          │ ByteTrack, zone engine, rules       │
│  └───────┬────────┘                                     │
│          │ alerts                                       │
│          ▼                                              │
│  ┌───────────────┐                                      │
│  │ Alert Manager  │ MQTT publish, frame capture         │
│  │ (CPU)          │ Rate limiting, deduplication        │
│  └───────┬────────┘                                     │
│          │                                              │
└──────────┼──────────────────────────────────────────────┘
           │ MQTT / Ethernet
           ▼
    Central Dashboard (alerts + video context)
```

---

## 11. Model Development Roadmap

### 11.1 Development Priority & Timeline

| Priority | Use Case | Complexity | Weeks | Dependencies |
|---|---|---|---|---|
| 1 | **I: Intrusion** | Low (no training) | 1–2 | Zone polygon config |
| 2 | **A: Fire** | Medium | 3–4 | Dataset ready (122K) |
| 3 | **B: Helmet** | Medium | 3–4 | Dataset ready (62K) |
| 4 | **G: Fall** | Medium-High | 4–5 | Pose model + fall data collection |
| 5 | **F: Shoes** | High (small dataset) | 4–5 | Data collection critical (3.7K→14K) |
| 6 | **H: Poketenashi** | Highest (multi-model) | 5–6 | Phone data + pose + zone config |

### 11.2 Parallel Development Tracks

```
Week 1-2:
  ├─ Track A: Deploy Intrusion (YOLOX-Tiny + ByteTrack + zones)     → DONE
  ├─ Track B: Start Fire training (YOLOX-M, 122K images)            → Training
  ├─ Track C: Start Helmet training (YOLOX-M, 62K images)           → Training
  └─ Track D: Collect shoes data + fall simulation data             → Data

Week 3-4:
  ├─ Track A: Intrusion validated on factory footage                → Testing
  ├─ Track B: Fire v1 evaluation + error analysis + HPO             → Iterate
  ├─ Track C: Helmet v1 evaluation + generative augment             → Iterate
  ├─ Track D: Start Shoes training (two-stage pipeline)             → Training
  └─ Track E: Start Fall pose model training (RTMPose-S)            → Training

Week 5-6:
  ├─ Track B: Fire v2 (refined) + D-FINE-S comparison               → Compare
  ├─ Track C: Helmet v2 (refined) + two-stage fallback if needed    → Compare
  ├─ Track D: Shoes v1 evaluation + data expansion                  → Iterate
  ├─ Track E: Fall v1 evaluation + rule tuning                      → Iterate
  └─ Track F: Start Poketenashi pipeline (person+phone+pose)        → Training

Week 7-8:
  ├─ All models: ONNX export + edge chip compilation                → Export
  ├─ Edge testing: INT8 quantization accuracy verification          → Validate
  └─ Track F: Poketenashi v1 evaluation + zone config               → Iterate

Week 9:
  ├─ Integration testing on edge devices                            → Deploy
  ├─ Multi-model concurrent inference benchmarking                  → Benchmark
  └─ Customer demo preparation                                     → Demo

Week 10-12: Buffer for iteration on underperforming models
```

### 11.3 Model Architecture Summary

| Use Case | CNN (Primary) | Transformer (Alternative) | All Apache 2.0? | Post-Processing |
|---|---|---|---|---|
| **A: Fire** | YOLOX-M (2 classes) | **D-FINE-S** (10M, 48.5 AP, NMS-free) | YES | Temporal filter (3 frames) |
| **B: Helmet** | YOLOX-M (4 classes) | **D-FINE-S** (NMS-free, better small-object) | YES | ByteTrack + alert (30 frames) |
| **F: Shoes** | YOLOX-Tiny + MobileNetV3 | **D-FINE-N** (4M) + MobileNetV3 | YES | ByteTrack + alert (30 frames) |
| **G: Fall** | YOLOX-Tiny + RTMPose-S | **D-FINE-N** + **RTMPose-M** (75.8 AP) + Temporal Transformer | YES | Geometric rules + temporal (30 frames) |
| **G: Fall (MediaPipe)** | YOLOX-Tiny + **MediaPipe Pose Full** | **D-FINE-N** + **MediaPipe Pose Full** (33 3D landmarks) | YES | 33-landmark rules + 3D depth + temporal |
| **G: Fall (alt)** | — | **RTMO-S** (one-stage, no detector needed) + Temporal | YES | Simplified pipeline |
| **H: Poketenashi** | YOLOX-Tiny + YOLOX-Nano + RTMPose-S | **D-FINE-N** + **RTMPose-T** (3.34M) + X3D-XS | YES | Zone engine + geometric rules |
| **H: Poketenashi (hybrid)** | YOLOX-Tiny + YOLOX-Nano + RTMPose-S + **MediaPipe Lite** (CPU) | **D-FINE-N** + **RTMPose-T** + **MediaPipe Lite** (hand detail) | YES | RTMPose on NPU + MediaPipe on CPU for hand/finger rules |
| **I: Intrusion** | YOLOX-Tiny (COCO pretrained) | **D-FINE-N** (COCO, NMS-free) | YES | ByteTrack + zone engine |

> **Note:** All models in both CNN and Transformer columns are Apache 2.0 or MIT licensed — $0 licensing cost for commercial deployment. RT-DETRv2-R18 (Apache 2.0) is an alternative to D-FINE if edge deployment compatibility is the priority.

### 11.4 Training Infrastructure

| Resource | Specification |
|---|---|
| GPU | Local GPU (CUDA, from pyproject.toml) |
| Framework | PyTorch + existing camera_edge pipeline |
| Experiment tracking | WandB (optional, configured in training YAML) |
| HPO | Optuna (50 trials, TPE sampler, median pruner) |
| Augmentation | Mosaic + MixUp + HSV + Affine + generative augment |
| Evaluation | mAP@0.5, precision, recall, confusion matrix, error analysis |
| Export | ONNX (opset 11) → onnxsim → Pulsar2/TPU-MLIR → edge format |

### 11.5 Risk Mitigation

| Risk | Mitigation |
|---|---|
| Shoes dataset too small (3.7K) | Generative augmentation + active data collection. Minimum target: 14K |
| Fall pose data too small (111) | Use COCO pretrained pose model. Collect 2K+ factory fall simulations |
| Transformer INT8 accuracy drop | Mixed precision (attention FP16, rest INT8). Fallback to CNN if >3% mAP loss |
| D-FINE edge compilation fails | Fallback to YOLOX (proven on both AX650N and CV186AH) or RT-DETRv2-R18 (discrete sampling designed for deployment) |
| Multi-model FPS insufficient | Frame interleaving, model downsizing (M→S→N), INT4 on AX650N |
| Fisheye camera distortion | Hardware dewarping (ISP/LDC), or train on FishEye8K dataset |
| False positives (fire: sunlight) | Hard negative mining, temporal consistency filter, two-stage verification |
| Poketenashi rule accuracy | Start with geometric rules; upgrade to X3D action recognition if FP rate >5% |
| Ultralytics license violation | NEVER use YOLOv5/v8/v11/YOLO26 — use YOLOX or D-FINE instead |

---

## 12. Requirements Gap Analysis & Camera Resolution Study

### 12.1 Camera Resolution vs Detection Range

The installed cameras have different resolutions and FOV configurations. Detection range is limited by how many pixels the target object occupies in the image.

**Rule of thumb:** An object needs **≥ 8–10 pixels** to be reliably detected by YOLO-class detectors. For small/amorphous objects (smoke), **≥ 16–20 pixels** is recommended.

**Formula:** `pixel_size_at_distance = (2 × distance × tan(HFOV/2)) / horizontal_resolution`

#### Camera 1 & 2: DWC-MV82DiVT / DWC-MB62DiVTW (2.1MP, Vari-focal 2.7–13.5mm)

Resolution: **1920 × 1080** | HFOV: **89.6° (wide) ↔ 27° (tele)**

| Distance | Wide (89.6°) | | Tele (27°) | |
|---|---|---|---|---|
| | Field Width | px/m | Field Width | px/m |
| 10m | 19.9m | 96 px/m | 4.8m | 400 px/m |
| 30m | 59.6m | 32 px/m | 14.4m | 133 px/m |
| 65m | 129m | 15 px/m | 31.2m | 62 px/m |
| 100m | 199m | 10 px/m | 48.0m | 40 px/m |
| 325m | 645m | 3 px/m | 156m | 12 px/m |

**Fire detection range (≥10px for 1×1m fire):**
- Wide (89.6°): **up to ~100m** (10 px/m × 1m = 10px)
- **Tele (27°): up to ~325m** (12 px/m × 1m = 12px) — meets customer spec!

**Fire detection range (≥10px for 0.2×0.2m fire):**
- Wide (89.6°): **up to ~20m** (96×0.2 at 10m = 19px → ok; at 30m = 6px → too small)
- Tele (27°): **up to ~65m** (62×0.2 = 12px)

#### Camera 3 & 4: DWC-MV75Wi28TW / DWC-MB75Wi4TW (5MP, Fixed 2.8mm / 4.0mm)

Resolution: **2592 × 1944** | HFOV: **102.4° (2.8mm) / 82.3° (4.0mm)**

| Distance | 2.8mm (102.4°) | | 4.0mm (82.3°) | |
|---|---|---|---|---|
| | Field Width | px/m | Field Width | px/m |
| 10m | 24.9m | 104 px/m | 17.3m | 150 px/m |
| 30m | 74.7m | 35 px/m | 51.8m | 50 px/m |
| 65m | 162m | 16 px/m | 112m | 23 px/m |
| 100m | 249m | 10 px/m | 173m | 15 px/m |

**These are wide-angle fixed-lens cameras — best for near/mid-range detection (helmet, shoes, fall, intrusion) up to ~30m.**

#### Camera 5: DWC-PVF5Di1TW (5MP, Fisheye 360°)

Resolution: **2592 × 1944** | FOV: **360° panoramic**

After dewarping into 4 perspective views (90° each):
- Effective resolution per view: ~648 × 1944 (quarter of horizontal)
- At 10m: field width ~20m → ~32 px/m
- At 30m: field width ~60m → ~11 px/m

**Fisheye cameras are limited to near-range detection (~15m) after dewarping.** Best suited for intrusion/fall in enclosed spaces.

#### Camera 6: DWC-MB45WiATW (5MP, Vari-focal 2.7–13.5mm)

Resolution: **2592 × 1944** | HFOV: **85° (wide) ↔ 31° (tele)**

| Distance | Wide (85°) | | Tele (31°) | |
|---|---|---|---|---|
| | Field Width | px/m | Field Width | px/m |
| 10m | 18.3m | 142 px/m | 5.5m | 471 px/m |
| 30m | 55.0m | 47 px/m | 16.6m | 156 px/m |
| 65m | 119m | 22 px/m | 36.0m | 72 px/m |
| 100m | 183m | 14 px/m | 55.4m | 47 px/m |
| 325m | 596m | 4 px/m | 180m | 14 px/m |

**Best camera for fire at distance** — 5MP + tele (31°):
- 1×1m fire at 325m → 14px — **detectable!**
- 0.2×0.2m fire at 65m → 72×0.2 = 14px — **detectable!**
- 0.5×0.5m fire at 162.5m → ~22×0.5 = 11px — **marginally detectable**

### 12.2 Detection Range Summary by Use Case

| Use Case | Min Object Size | Min px Needed | Best Camera Config | Max Reliable Range |
|---|---|---|---|---|
| **Fire (1×1m)** | 1.0m | 10px | DWC-MB45WiATW **tele 31°** (5MP) | **~325m** |
| **Fire (0.5×0.5m)** | 0.5m | 10px | DWC-MB45WiATW **tele 31°** (5MP) | **~162m** |
| **Fire (0.2×0.2m)** | 0.2m | 10px | DWC-MB45WiATW **tele 31°** (5MP) | **~65m** |
| **Smoke (1×1m)** | 1.0m | 20px (diffuse) | DWC-MB45WiATW **tele 31°** (5MP) | **~160m** |
| **Helmet** | ~0.25m | 10px | Any 5MP camera | **~15–25m** |
| **No_helmet (bare head)** | ~0.20m | 10px | 5MP wide or 2.1MP tele | **~15–20m** |
| **Safety shoes** | ~0.15m | 10px (cropped) | Any camera (two-stage crop) | **~10–15m** |
| **Person (fall/intrusion)** | ~0.5m width | 10px | Any 5MP camera | **~50–70m** |
| **Phone in hand** | ~0.07m | 8px (cropped) | Any camera (upper-body crop) | **~5–10m** |
| **Handrail contact** | N/A (pose) | Keypoint accuracy | Any camera in stair zone | **~10–15m** |

### 12.3 Key Finding: Customer Fire Detection Spec IS Achievable

> **The customer's 325m fire detection requirement IS achievable** — but ONLY when using the **5MP vari-focal cameras (DWC-MB45WiATW) set to telephoto (31° HFOV)**:
>
> - 1×1m fire at 325m → 14 pixels → detectable with tiled inference
> - 0.5×0.5m fire at 162.5m → 11 pixels → marginally detectable
> - 0.2×0.2m fire at 65m → 14 pixels → detectable
>
> With the **2.1MP vari-focal cameras at tele (27°)**, 1×1m fire at 325m → 12 pixels → also marginally detectable.
>
> **Requirement:** Cameras designated for long-range fire detection MUST be configured to **telephoto** setting. Wide-angle mode is physically incapable of the specified ranges.

### 12.4 Requirements Gap Analysis

#### CRITICAL GAPS — Must Resolve with Customer

| # | Gap | Requirement Source | Impact | Recommended Action |
|---|---|---|---|---|
| **G1** | **Temperature change detection** | Customer A.4 Model A: "Temperature change detection: Enables early identification of abnormal heat rise before a fire occurs" | Customer explicitly expects pre-fire thermal anomaly detection | **Clarify with customer:** No thermal cameras are installed. The AITech reference they cited uses specialized thermal cameras. Options: (a) Add FLIR Lepton thermal module to edge device (~$200/unit), (b) Use color-based heat shimmer detection as stopgap (limited accuracy), (c) Remove from Phase 1 scope |
| **G2** | **Full harness + waist belt detection** | Customer A.4 Model B: "Detection of not wearing several types of helmets (including Nitto soft hats) **and two types of safety belts**" | Customer expects harness/belt detection as part of Model B | **Action:** Add classes `harness` and `safety_belt` to Model B. Requires: (a) Collecting training data of harness/belt wearing, (b) Expanding model to 6 classes: person, head_with_helmet, head_without_helmet, head_with_nitto_hat, harness, safety_belt. Alternative: Create separate Model C for harness/belt |
| **G3** | **Pointing-and-calling (指差呼称)** | Customer A.4 Model H: "Perform proper pointing-and-calling (safety confirmation) at designated points" | Customer expects detection of pointing-and-calling gestures | **Action:** Implement as pose-based gesture detection: (a) Arm extended forward (shoulder-wrist angle < 30° from horizontal), (b) Head turned toward pointing direction, (c) Only at designated polygon zones. Requires: labeled examples of pointing-and-calling for validation |
| **G4** | **Camera tele configuration for fire** | Customer A.4 Model A: detection ranges up to 325m | Fire detection at 325m requires telephoto lens setting | **Action:** Work with customer to identify which cameras should be set to telephoto for fire detection zones. Document per-camera lens configuration |

#### MODERATE GAPS — Training Pipeline Adjustments

| # | Gap | Requirement Source | Impact | Recommended Action |
|---|---|---|---|---|
| **G5** | **Low-light / IR mode** | B.10: "Works in low-light (0.16 lux), IR mode" | All cameras switch to B/W IR mode in darkness. Models trained on color images may degrade | **Action:** Add IR/low-light augmentation to training pipeline: (a) Random grayscale conversion (30% of training images) to simulate IR mode, (b) Random brightness reduction to simulate 0.16 lux, (c) Collect or generate night/IR samples from factory cameras. Add to `core/p05_data/transforms.py` |
| **G6** | **CV186AH power budget** | B.6: "< 12W per edge device" | CV186AH estimated 5–15W — may exceed 12W | **Action:** Benchmark CV186AH actual power consumption under multi-model load. If >12W, restrict CV186AH to simple zones (1–2 models) or use AX650N exclusively |
| **G7** | **False alarm rate < 5/day** | B.10: "< 5 false alarms per day per camera" | Not explicitly mapped in alert logic | **Action:** Add alert rate limiter in edge alert manager: (a) Cooldown period between same-type alerts (e.g., 60s for PPE, 30s for fire), (b) Per-camera daily alert counter with configurable limit, (c) Aggregate similar detections within time window |
| **G8** | **24/7 stability validation** | B.10: "24/7 operation for 1 week without manual intervention" | No stability test plan in approach | **Action:** Add Week 9 stability testing: run edge device with all models for 168 hours continuously. Monitor for memory leaks, NPU hangs, RTSP reconnection, thermal throttling |

#### INFORMATIONAL — Likely Out of Phase 1 Scope

| # | Item | Requirement Source | Status |
|---|---|---|---|
| G9 | PPE Levels A–E (chemical PPE: goggles, gloves, apron, boots) | Customer ref (Kurita) | **Context only** — customer reference shows PPE levels at their chemical plant, but only helmet and safety belt are explicitly requested for detection |
| G10 | Dashboard with video context | B.10 | Separate software engineering scope — not part of AI model development |
| G11 | Mobile alerts / Email / Siren integration | B.9 | System integration scope |
| G12 | Data privacy / face blurring | B.11 Q8 | Policy decision, not AI model scope |

### 12.5 Proposed Resolutions

#### G1: Temperature Change Detection

```
Option A (Recommended): Remove from Phase 1 scope
  • No thermal cameras installed
  • AITech reference uses specialized thermal cameras (not our hardware)
  • Add as Phase 2 deliverable with FLIR Lepton thermal module

Option B: Color-based heat shimmer (stopgap)
  • Detect visual heat distortion using optical flow analysis
  • Very limited: only works in specific lighting conditions
  • NOT reliable enough for safety-critical application
  • Not recommended

Option C: Add thermal camera module
  • FLIR Lepton 3.5: 160×120, 8-14μm, ~$200/unit
  • Mount alongside existing camera, fuse RGB + thermal
  • Requires additional edge processing pipeline
  • Phase 2 scope
```

#### G2: Full Harness + Safety Belt Detection

```
Option A (Recommended): Expand Model B to 6 classes
  Classes: person, head_with_helmet, head_without_helmet, head_with_nitto_hat, harness, no_harness
  • Harness is visible as straps across chest/shoulders
  • Requires ~3K labeled images of harness/no_harness
  • Safety belt (waist belt) may be too small to detect reliably
    from overhead/dome cameras — clarify viewing angles with customer

Option B: Separate Model C for harness/belt
  • Dedicated model with higher input resolution (1280×1280)
  • Better accuracy for small harness straps
  • Higher compute cost — additional NPU load
```

#### G3: Pointing-and-Calling Detection

```
Approach: Pose-based gesture detection at designated zones

Detection rules (using RTMPose keypoints):
  1. Person is in designated "pointing-and-calling zone" polygon
  2. One arm extended forward:
     • shoulder_to_wrist_angle < 30° from horizontal
     • wrist_x significantly ahead of shoulder_x
  3. Head oriented toward pointing direction:
     • nose_x close to wrist_x (same direction)

Implementation:
  • No separate model training needed — uses existing RTMPose keypoints
  • Zone polygons mark designated pointing-and-calling locations
  • Alert triggers if person passes through zone WITHOUT performing gesture
  • Requires customer to define designated locations

Accuracy concern:
  • Gesture detection from 2D keypoints is inherently noisy
  • Camera angle significantly affects reliability
  • Recommend: install cameras at eye-level near designated points
  • Expected accuracy: 70-80% (lower than other use cases)
```

#### G5: IR / Low-Light Training Augmentation

```python
# Add to core/p05_data/transforms.py

class IRSimulation:
    """Simulate IR/B&W camera mode for low-light training."""

    def __init__(self, p_grayscale=0.3, p_low_light=0.2):
        self.p_grayscale = p_grayscale  # probability of IR simulation
        self.p_low_light = p_low_light  # probability of low-light simulation

    def __call__(self, image, targets):
        # Simulate IR mode (B/W with IR illumination)
        if random.random() < self.p_grayscale:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            # Boost contrast (IR cameras have higher contrast)
            image = cv2.convertScaleAbs(image, alpha=1.3, beta=10)

        # Simulate low-light conditions (0.16 lux)
        if random.random() < self.p_low_light:
            factor = random.uniform(0.1, 0.4)  # darken significantly
            image = (image * factor).astype(np.uint8)
            # Add noise (low-light noise)
            noise = np.random.normal(0, 15, image.shape).astype(np.int16)
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return image, targets
```

### 12.6 Updated Requirements Coverage Summary

| Requirement | Status | Notes |
|---|---|---|
| **Model A: Fire (flames + smoke)** | **COVERED** | YOLOX-M / D-FINE-S on 122K dataset |
| **Model A: Fire long-range (325m)** | **COVERED** | Requires 5MP vari-focal cameras at **tele** setting |
| **Model A: Temperature detection** | **GAP → Phase 2** | Needs thermal camera (not installed) |
| **Model B: Helmet compliance** | **COVERED** | 4-class detection (person, head_with_helmet, head_without_helmet, head_with_nitto_hat) |
| **Model B: Harness + safety belt** | **GAP → Expand scope** | Add 2 classes + collect training data |
| **Model F: Safety shoes** | **COVERED** | Two-stage pipeline (person → foot crop → classifier) |
| **Model G: Fall detection** | **COVERED** | Phase 1: classification. Phase 2: pose + rules |
| **Model H: Phone usage** | **COVERED** | YOLOX-Nano phone detector + walking trajectory |
| **Model H: Hands in pockets** | **COVERED** | Pose keypoints + geometric rules |
| **Model H: No handrail** | **COVERED** | Pose + stair zone polygon |
| **Model H: Diagonal stair crossing** | **COVERED** | Trajectory angle analysis |
| **Model H: Pointing-and-calling** | **GAP → Pose rules** | Feasible with pose gesture detection at designated zones (70-80% accuracy) |
| **Model I: Zone intrusion** | **COVERED** | Pretrained COCO + polygon logic |
| **Model I: Wrong direction** | **COVERED** | Line crossing + direction vector |
| **Model I: Loitering / no-parking** | **COVERED** | Duration tracker in zone |
| **≥ 15 FPS per model** | **COVERED** | AX650N: 25-50 FPS. CV186AH: 12-25 FPS |
| **< 12W power** | **COVERED (AX650N)** | AX650N: 5-8W. CV186AH: needs benchmarking |
| **Multi-model (2+ at ≥10 FPS)** | **COVERED** | Both chips support concurrent inference |
| **Alert latency < 3s** | **COVERED** | On-device inference + MQTT |
| **Low-light (0.16 lux) / IR** | **GAP → Training fix** | Add IR/low-light augmentation to training pipeline |
| **24/7 stability** | **PARTIAL** | Add stability test plan to Week 9 |
| **< 5 false alarms/day** | **PARTIAL** | Add alert rate limiter |
| **$0 licensing** | **COVERED** | All models Apache 2.0 or MIT |

**Overall coverage: ~90%** — 4 gaps require customer clarification (temperature, harness, pointing-and-calling, camera tele config), 3 gaps require training pipeline updates (IR augmentation, stability testing, alert rate limiting).

---

## Appendix A: Model Export Commands

```bash
# YOLOX-M Fire → ONNX
uv run core/p09_export/export.py \
  --model runs/fire_detection/best.pt \
  --training-config features/safety-fire_detection/configs/06_training.yaml \
  --export-config configs/_shared/09_export.yaml

# ONNX → AX650N (.axmodel)
onnxsim fire_yoloxm_640_v1.onnx fire_yoloxm_640_v1_sim.onnx
pulsar2 build \
  --input fire_yoloxm_640_v1_sim.onnx \
  --output fire_yoloxm_640_v1.axmodel \
  --target_hardware AX650 \
  --calibration_data cal_images/ \
  --quant_type INT8

# ONNX → CV186AH (.cvimodel)
model_transform.py \
  --model_name fire_yoloxm \
  --model_def fire_yoloxm_640_v1_sim.onnx \
  --input_shapes [[1,3,640,640]] \
  --pixel_format rgb \
  --output_names output \
  --mlir fire_yoloxm.mlir

run_calibration.py fire_yoloxm.mlir \
  --dataset cal_images/ \
  --input_num 500 \
  -o fire_yoloxm_cali_table

model_deploy.py \
  --mlir fire_yoloxm.mlir \
  --quantize INT8 \
  --calibration_table fire_yoloxm_cali_table \
  --chip cv186ah \
  --model fire_yoloxm_int8.cvimodel
```

## Appendix B: Edge Reference Code

| Chip | Model Type | Reference Code |
|---|---|---|
| AX650N | YOLOX | `ax-samples/ax_yolox_steps.cc` |
| AX650N | HRNet pose | `ax-samples/ax_hrnet_steps.cc` |
| AX650N | **RTMPose/SimCC** | **`ax-samples/ax_simcc_pose_steps.cc` (4.79ms, RECOMMENDED)** |
| AX650N | Python runtime | `pyaxengine` (ONNXRuntime-compatible API) |
| CV186AH | YOLOX | Sophgo YOLOX Deployment Guide |
| CV186AH | HRNet-pose | `sophon-demo` HRNet sample (3.43ms INT8) |
| CV186AH | YOLOv8 | Sophgo YOLOv8 Deployment Guide |
| CV186AH | Python runtime | SAIL (Python/C++ high-level API) |

## Appendix C: Key References

### Papers
- **D-FINE** (ICLR 2025 Spotlight): Fine-grained Distribution Refinement for DETR. [arXiv:2410.13842](https://arxiv.org/abs/2410.13842), [GitHub](https://github.com/Peterande/D-FINE)
- **RT-DETRv2** (CVPR 2024): Improved Real-Time Detection Transformer. [arXiv:2407.17140](https://arxiv.org/abs/2407.17140), [GitHub](https://github.com/lyuwenyu/RT-DETR)
- RT-DETR (CVPR 2024): "DETRs Beat YOLOs on Real-time Object Detection"
- RF-DETR (ICLR 2026): First real-time detector >60 AP on COCO. [GitHub](https://github.com/roboflow/rf-detr)
- **YOLOX** (2021): Exceeding YOLO Series. [GitHub](https://github.com/Megvii-BaseDetection/YOLOX)
- **RTMPose** (2023): Real-Time Multi-Person Pose Estimation. CSPNeXt + SimCC. [GitHub](https://github.com/open-mmlab/mmpose)
- **RTMO** (2024): One-stage real-time multi-person pose. [GitHub](https://github.com/open-mmlab/mmpose)
- Lite-HRNet (2021): Ultra-lightweight pose estimation (1.8M params). [GitHub](https://github.com/HRNet/Lite-HRNet)
- **MediaPipe BlazePose** (2020): On-Device, Real-time Body Pose Tracking. 33 3D landmarks, MobileNetV2 backbone, TFLite-first. [arXiv:2006.10204](https://arxiv.org/abs/2006.10204), [GitHub](https://github.com/google-ai-edge/mediapipe), [Guide](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
- RT-DETR-Smoke (2025): 87.75% mAP@0.5 for smoke at 445 FPS
- LFD-YOLO (2025): Lightweight fall detection

### Datasets
- FASDD: 120K+ fire/smoke images (ground/UAV/satellite)
- D-Fire: ~21K fire/smoke (standard YOLO benchmark)
- SHWD: 7.5K helmet images
- GDUT-HWD: 3.2K multi-color hard hat
- SH17: Manufacturing PPE (helmet, vest, gloves, shoes)
- COCO Keypoints: ~150K person pose images
- Le2i Fall: ~4K fall detection benchmark
- FishEye8K: 8K fisheye images (person, vehicle)

### Edge SDKs
- AX650N: Pulsar2 toolchain, ax-samples, pyaxengine (HuggingFace: AXERA-TECH)
- CV186AH: TPU-MLIR (GitHub: sophgo/tpu-mlir), SOPHON SDK, CVI TDL SDK

### License Sources
- YOLOX: Apache 2.0 — [LICENSE](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE)
- D-FINE: Apache 2.0 — [LICENSE](https://github.com/Peterande/D-FINE/blob/master/LICENSE)
- RT-DETR/v2: Apache 2.0 — [LICENSE](https://github.com/lyuwenyu/RT-DETR/blob/main/LICENSE)
- RF-DETR (Nano-Large): Apache 2.0 — [LICENSE](https://github.com/roboflow/rf-detr/blob/main/LICENSE)
- RTMPose/MMPose: Apache 2.0 — [LICENSE](https://github.com/open-mmlab/mmpose/blob/main/LICENSE)
- ViTPose: Apache 2.0 (license OK but NOT recommended for edge — poor INT8 quantization)
- RTMO: Apache 2.0 — part of MMPose
- HRNet: MIT — [LICENSE](https://github.com/HRNet/HRNet-Human-Pose-Estimation/blob/master/LICENSE)
- ByteTrack: MIT — [LICENSE](https://github.com/ifzhang/ByteTrack/blob/main/LICENSE)
- X3D/SlowFast: Apache 2.0 — [LICENSE](https://github.com/facebookresearch/SlowFast/blob/main/LICENSE)
- Ultralytics (YOLOv5/v8/v11/YOLO26): **AGPL-3.0 — NOT APPROVED for this project**

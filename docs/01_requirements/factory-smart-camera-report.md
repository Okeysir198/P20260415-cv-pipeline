# 🏭 Factory Smart Camera System
## Technical Summary Report — Non-NVIDIA Edge AI Deployment
**Date:** March 2026 | **Version:** 1.0 | **Scope:** Safety Action Detection via PoE Edge AI

> **Note:** This report was written during initial research (pre-planning). For finalized decisions on tools, costs, and training approach, see [03_phase1_development_plan.md](phase1_development_plan.md) and [04_tool_stack.md](tool_stack.md).
>
> **⚠ License Update (March 2026):** Final architecture uses **YOLOX (Apache 2.0)** instead of YOLO11/YOLO26 — **$0 licensing cost** for commercial deployment. See [07_phase1_executive_summary.md](phase1_executive_summary.md) for updated model selection.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Edge Hardware — Non-NVIDIA Options](#3-edge-hardware--non-nvidia-options)
4. [AI Models — State of the Art](#4-ai-models--state-of-the-art)
5. [Use Cases & Model Mapping](#5-use-cases--model-mapping)
6. [Datasets](#6-datasets)
7. [Training Resources](#7-training-resources)
8. [Deployment Pipeline](#8-deployment-pipeline)
9. [Decision Matrix](#9-decision-matrix)
10. [Recommended Stack](#10-recommended-stack)
11. [Cost Estimation](#11-cost-estimation)

---

## 1. Project Overview

The goal is to transform existing **normal IP cameras** into **smart AI cameras** capable of detecting unsafe worker actions in real time — without replacing the cameras or relying on cloud infrastructure.

### Core Principles

- **One edge board per camera** — independent, scalable, low-latency
- **Powered via PoE** — single cable per camera point (data + power)
- **On-device AI inference** — no video sent to cloud, only alert metadata
- **No NVIDIA hardware** — cost-competitive, vendor-diverse stack
- **RTSP stream input** — works with any existing IP camera brand

### Target Safety Use Cases

| # | Use Case | Priority |
|---|---|---|
| 1 | PPE compliance (helmet, vest, gloves, goggles) | 🔴 Critical |
| 2 | Fall & unsafe posture detection | 🔴 Critical |
| 3 | Restricted zone intrusion | 🔴 Critical |
| 4 | Forklift–pedestrian proximity | 🟠 High |

> **Updated:** Forklift–pedestrian proximity detection has been moved to Phase 2. Phase 1 models are: (a) Fire, (b) Helmet, (f) Safety Shoes, (g) Fall, (h) Poketenashi, (i) Zone Intrusion.

| 5 | Fire & smoke early detection | 🟠 High |
| 6 | Unsafe behavior classification | 🟡 Medium |

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     FACTORY FLOOR                               │
│                                                                 │
│  [IP Camera 1]──PoE──┐                                          │
│  [IP Camera 2]──PoE──┤                                          │
│  [IP Camera N]──PoE──┴──► [PoE Switch] ──► [Factory Network]    │
│                                │                                │
│                         ┌──────┘                                │
│                         │  PoE tap per camera point             │
│                         ▼                                       │
│              ┌─────────────────────┐                            │
│              │  EDGE BOARD         │  (one per camera)          │
│              │  RPi 5 + Hailo-8    │                            │
│              │  or RK3588 + Hailo  │                            │
│              │                     │                            │
│              │  ┌───────────────┐  │                            │
│              │  │ RTSP Capture  │  │                            │
│              │  │ AI Inference  │  │                            │
│              │  │ Alert Logic   │  │                            │
│              │  └───────────────┘  │                            │
│              └────────┬────────────┘                            │
│                       │ Alerts only (JSON/MQTT)                 │
│                       ▼                                         │
│              ┌─────────────────────┐                            │
│              │  Central Dashboard  │                            │
│              │  Alert Server       │                            │
│              │  (On-premise)       │                            │
│              └─────────────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

### Key Architecture Decisions

- **Edge-first** — All AI inference happens on the board; only events/alerts are transmitted
- **RTSP pull** — Board pulls stream from camera over PoE network
- **MQTT/REST** — Lightweight alert protocol to central dashboard
- **No cloud dependency** — fully air-gapped operation possible
- **Modular models** — Multiple lightweight models run in parallel per board

---

## 3. Edge Hardware — Non-NVIDIA Options

> **Constraint:** No NVIDIA Jetson. We need competitive, cost-effective, vendor-diverse alternatives.

---

### 3.1 Tier 1 — Budget ($80–$160/unit) ⭐ Recommended for Scale

#### Option A: Raspberry Pi 5 (8GB) + Hailo-8 + PoE HAT

| Spec | Detail |
|---|---|
| **Total Cost** | ~$140–160 |
| **AI Performance** | 26 TOPS (Hailo-8) |
| **CPU** | ARM Cortex-A76 quad-core @ 2.4GHz |
| **RAM** | 8GB LPDDR4X |
| **Power** | ~15W via PoE HAT |
| **PoE** | ✅ Native (with Waveshare/official HAT ~$15) |
| **FPS (YOLOv11s/1080p)** | 30–60 FPS |

> **Note:** Official Hailo benchmarks use 4-lane PCIe. Real-world edge devices (single-lane PCIe) achieve ~50% of official figures. Realistic estimates: s-variant 30-50 FPS, n-variant 60-80 FPS at 640px.

| **OS** | Raspberry Pi OS (Debian-based) |
| **Inference Runtime** | Hailo Runtime (HailoRT) + GStreamer |
| **Form Factor** | 85mm × 56mm |

**Why this is the top pick for mass deployment:**
- Largest community of any SBC globally
- Hailo-8 M.2 HAT plugs directly into Pi 5 PCIe slot
- Hailo provides pre-compiled HEF models (YOLO family ready-to-run)
- PoE HAT eliminates external power supply entirely
- Proven in production smart camera deployments worldwide

---

#### Option B: Raspberry Pi 5 (4GB) + Hailo-8L + PoE HAT

| Spec | Detail |
|---|---|
| **Total Cost** | ~$95–110 |
| **AI Performance** | 13 TOPS (Hailo-8L) |
| **Best for** | Single model deployments (PPE only, or zone intrusion only) |
| **FPS** | ~30 FPS @ 1080p with YOLO11n |

**Note:** Hailo-8L is the entry-level. Sufficient for simpler use cases; upgrade to Hailo-8 if running multiple models simultaneously.

---

### 3.2 Tier 2 — Performance ($130–$200/unit)

#### Option C: ArmSoM Sige7 — RK3588 + Hailo-8 (via PCIe)

| Spec | Detail |
|---|---|
| **Total Cost** | ~$140–180 |
| **AI Performance** | 32 TOPS (6 RK3588 NPU + 26 Hailo-8) |
| **CPU** | RK3588 — 4× Cortex-A76 + 4× Cortex-A55 |
| **RAM** | 8GB LPDDR4 |
| **Power** | ~15W |
| **PoE** | ✅ Via HAT |
| **FPS** | 30-channel 1080p object detection (vendor demo) |
| **Advantage** | Dual NPU: RK3588 handles one model, Hailo-8 handles another simultaneously |

**Best for:** Running PPE detection + fall detection simultaneously without frame rate drop.

---

#### Option D: Radxa ROCK 5B / Orange Pi 5 Plus (RK3588 standalone)

| Spec | Detail |
|---|---|
| **Total Cost** | ~$80–120 |
| **AI Performance** | 6 TOPS (built-in NPU only) |
| **RAM** | Up to 16GB |
| **PoE** | Via HAT |
| **Best for** | Budget-constrained, single lightweight model |
| **Limitation** | 6 TOPS is tight for multiple models; acceptable for YOLO26n at 30 FPS |

---

### 3.3 Tier 3 — High Performance ($250–$450/unit)

#### Option E: Hailo-15 OEM Integration (True Smart Camera)

| Spec | Detail |
|---|---|
| **AI Performance** | 20+ TOPS purpose-built for camera applications |
| **Power** | **< 5W** — lowest power of any option |
| **PoE** | ✅ Native — designed to live inside camera enclosure |
| **ONVIF/RTSP** | ✅ Native support |
| **Form Factor** | Chip-level — requires OEM PCB integration |
| **Best for** | High-volume production deployment, true embedded smart camera |
| **Limitation** | Not a dev board — requires hardware engineering for integration |

#### Option F: Renesas RZ/V2H Eval Kit

| Spec | Detail |
|---|---|
| **AI Performance** | **100 TOPS** — most powerful edge AI SoC as of July 2025 |
| **Power** | ~10W |
| **PoE** | Via external adapter |
| **Best for** | Future-proof deployments, multi-model stacks |
| **Status** | Ubuntu support via Canonical partnership (2025) |

#### Option G: Axelera Metis M.2

| Spec | Detail |
|---|---|
| **AI Performance** | **214 TOPS** — in-memory compute architecture |
| **Power** | ~15W |
| **Best for** | Multi-camera analytics from single board |
| **Status** | Maturing ecosystem — less production-proven |

---

### 3.4 Hardware Comparison Summary

| Board | TOPS | Price | PoE | Edge Maturity | Best Use |
|---|---|---|---|---|---|
| **RPi 5 + Hailo-8L** | 13 | ~$110 | ✅ | ⭐⭐⭐⭐⭐ | Budget, single model |
| **RPi 5 + Hailo-8** | 26 | ~$150 | ✅ | ⭐⭐⭐⭐⭐ | **Best overall** ✅ |
| **RK3588 standalone** | 6 | ~$90 | ✅ | ⭐⭐⭐ | Ultra-budget |
| **ArmSoM Sige7** | 32 | ~$160 | ✅ | ⭐⭐⭐ | Dual NPU tasks |
| **Hailo-15 (OEM)** | 20 | ~$70 chip | ✅ native | ⭐⭐ (HW needed) | Production embed |
| **Renesas RZ/V2H** | 100 | ~$300+ | ❌ | ⭐⭐ (new) | Future-proof |
| **Axelera Metis** | 214 | ~$250 | ❌ | ⭐⭐ (new) | Multi-stream |

---

## 4. AI Models — State of the Art

> **Key insight:** YOLO is no longer the only option. Transformer-based detectors (RF-DETR, D-FINE, RT-DETRv2) now clearly outperform YOLO in accuracy while maintaining real-time speed on GPU. However, for NPU-based edge (Hailo), CNN models still hold the advantage due to quantization compatibility.

---

### 4.1 Model Families Overview

#### Family 1: CNN-Based (YOLO Lineage)
- Fast, highly edge-compatible, NPU/Hailo-optimized
- Lower accuracy ceiling vs. transformers
- Best for: Hailo-8, RK3588 NPU, any CPU-only edge

**Top models:**
- `YOLO26n/s` — Sep 2025, fastest CPU inference (43% faster than YOLO11n), NMS-free, edge-first design
- `YOLO11n/s` — Oct 2024, current Ultralytics production standard
- `YOLOv12s` — Early 2025, attention-centric CNN hybrid (research, not recommended for production)

#### Family 2: Transformer-Based (DETR Variants)
- Higher accuracy than YOLO at all scales
- Best on GPU-equipped edge (Jetson replacements with GPU)
- Apache 2.0 license — more commercial-friendly than YOLO's AGPL-3.0
- Harder to quantize for pure NPU deployment

**Top models:**
- `RF-DETR` — Roboflow, March 2025. First real-time model to exceed 60 mAP on COCO
- `D-FINE` — ICLR 2025 Spotlight. Best accuracy/speed ratio for real-time detection
- `RT-DETRv2/v4` — Baidu/PaddlePaddle. Battle-tested DETR for production
- `DEIM / DEIMv2` — CVPR 2025 / Oct 2025. Cutting-edge, DINOv3 backbone

#### Family 3: Vision-Language Models (VLMs)
- Zero-shot detection capability (no training needed)
- Cloud/server only — too heavy for edge
- Examples: Grounding DINO, OWLv2, Florence-2

---

### 4.2 Full Benchmark Comparison (COCO Object Detection)

| Model | Family | mAP@50-95 | Latency (T4) | Edge (NPU) | License |
|---|---|---|---|---|---|
| YOLO26n | CNN | 39.8–40.3% | ~1.5ms | ✅ Excellent | AGPL-3.0 |
| YOLO26s | CNN | 47.2% | ~2.8ms | ✅ Excellent | AGPL-3.0 |
| YOLO11n | CNN | 39.5% | ~1.6ms | ✅ Excellent | AGPL-3.0 |
| YOLO11s | CNN | 47.0% | ~2.5ms | ✅ Excellent | AGPL-3.0 |
| YOLOv12s | Hybrid | 48.0% | ~2.6ms | ✅ Good | AGPL-3.0 |
| RT-DETRv2-S | Transformer | 51–53% | ~4ms | ⚠️ GPU | Apache 2.0 |
| **RF-DETR-N** | Transformer | ~42% | 3.5ms | ⚠️ GPU | **Apache 2.0** |
| **RF-DETR-S** | Transformer | **53.0%** | 3.5ms | ⚠️ GPU | **Apache 2.0** |
| **RF-DETR-B** | Transformer | **60.5%** | ~40ms | ❌ GPU only | **Apache 2.0** |
| **D-FINE-S** | Transformer | 49.5% | ~5ms | ⚠️ GPU | **Apache 2.0** |
| **D-FINE-L** | Transformer | **54.7%** | ~8ms | ⚠️ GPU | **Apache 2.0** |
| **D-FINE-X** | Transformer | **56.5%** | ~13ms | ⚠️ GPU | **Apache 2.0** |
| DEIMv2 | Transformer | Highest | ~10–15ms | ❌ GPU only | Apache 2.0 |

> **Key finding:** RF-DETR-S beats YOLO11-X (the largest YOLO) with higher mAP (53.0% vs 51.2%) AND faster speed (3.52ms vs 11.92ms). The era of "YOLO by default" is ending for GPU-capable deployments.

---

### 4.3 Honest Edge Deployment Reality

| Factor | YOLO26/11 | RF-DETR | D-FINE | RT-DETRv2 |
|---|---|---|---|---|
| **Accuracy** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Speed (small)** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Hailo-8 NPU** | ✅ Best | ⚠️ Difficult | ⚠️ Medium | ⚠️ Partial |
| **RK3588 NPU** | ✅ Good | ❌ Not yet | ❌ Not yet | ⚠️ Partial |
| **ONNX/TensorRT** | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| **Small objects** | ⚠️ Struggles | ✅ Excellent | ✅ Very good | ✅ Good |
| **License** | ⚠️ AGPL-3.0 | ✅ Apache 2.0 | ✅ Apache 2.0 | ✅ Apache 2.0 |
| **Training ease** | ✅ Easiest | ✅ Easy | ⚠️ Moderate | ✅ Easy |
| **Community** | ✅ Largest | Growing | Growing | Medium |

---

## 5. Use Cases & Model Mapping

### 5.1 PPE Compliance Detection
**Goal:** Detect helmet, vest, gloves, goggles — present vs. absent per worker

| Aspect | Detail |
|---|---|
| **CV Task** | Object Detection (multi-class) |
| **Classes** | `helmet`, `no_helmet`, `vest`, `no_vest`, `gloves`, `goggles`, `person` |

> **Phase 1 scope note:** Phase 1 Helmet model only covers: person, helmet, no_helmet, nitto_hat. Full PPE (vest, gloves, goggles) is Phase 2.

| **Best model (Hailo edge)** | `YOLO26s` — fastest CNN, edge-optimized |
| **Best model (GPU edge)** | `RF-DETR-S` — DINOv2 backbone handles small objects (goggles, gloves) far better |
| **Challenge** | Small PPE items at distance; low-light factory floors |
| **Solution** | Use 1280px input resolution (instead of 640px) for small object improvement |
| **mAP benchmark** | 0.92+ mAP@0.5 achievable on PPE datasets (2025 literature) |

> **Metric update:** Primary acceptance metrics are now Precision, Recall, FP Rate, FN Rate. mAP@0.5 is used as secondary internal tracking metric. See `docs/phase1_development_plan.md` for full targets.

---

### 5.2 Fall & Unsafe Posture Detection
**Goal:** Detect worker falling, lying on ground, unsafe bending near machinery

| Aspect | Detail |
|---|---|
| **CV Task** | Pose Estimation → Rule-based classifier |
| **Best model** | `YOLO11-Pose` or `DETRPose` (2025, outperforms YOLO11-X on CrowdPose) |
| **Keypoints** | 17-point COCO format (nose, shoulders, hips, knees, ankles) |
| **Fall logic** | `bbox_width / bbox_height > 1.5` AND `hip_y > shoulder_y + threshold` |
| **Edge** | YOLO11n-Pose runs well on Hailo-8 at 30+ FPS |

```
Person detected → Extract 17 keypoints → Calculate body angle → 
If horizontal ratio > threshold → FALL ALERT
```

---

### 5.3 Restricted Zone Intrusion
**Goal:** Alert when worker enters danger zones (near machines, conveyor belts, hazardous areas)

| Aspect | Detail |
|---|---|
| **CV Task** | Object Detection + Tracking + Zone Logic |
| **Detection model** | `YOLO26n` — lightweight person detector |
| **Tracker** | `ByteTrack` or `BotSORT` — assigns unique IDs per person |
| **Zone definition** | Software-defined polygons drawn over camera FOV |
| **Library** | `Supervision` (Roboflow) — built-in zone tools |
| **Training needed** | ❌ None — use COCO pretrained person detection weights |

---

### 5.4 Forklift–Pedestrian Proximity
**Goal:** Detect collision risk when forklift and worker are too close

| Aspect | Detail |
|---|---|
| **CV Task** | Multi-class Object Detection + distance estimation |
| **Classes** | `person`, `forklift`, `pallet_jack` |
| **Best model (Hailo)** | `YOLO11s` fine-tuned on factory vehicle data |
| **Best model (GPU)** | `D-FINE-S` — better multi-class localization accuracy |
| **Alert logic** | If bounding box IoU or centroid distance < threshold → proximity alert |
| **Training data** | Custom factory footage required (forklifts vary by facility) |

---

### 5.5 Fire & Smoke Detection
**Goal:** Early fire and smoke detection before traditional sensor triggers

| Aspect | Detail |
|---|---|
| **CV Task** | Object Detection |
| **Classes** | `fire`, `smoke` |
| **Best model (Hailo)** | `YOLO11s` fine-tuned on fire/smoke datasets |
| **Best model (GPU)** | `RF-DETR-S` — global attention captures diffuse smoke better than CNN local features |
| **Key challenge** | Smoke is semi-transparent and low-contrast — transformers handle this better |
| **Augmentation** | Include hazy, low-light, partial occlusion samples in training |

---

### 5.6 Unsafe Behavior Classification (Advanced)
**Goal:** Detect unsafe lifting posture, unauthorized machinery operation, overload carrying

| Aspect | Detail |
|---|---|
| **CV Task** | Pose Estimation → Temporal Action Recognition |
| **Stage 1** | `YOLO11-Pose` extracts 17 keypoints per frame |
| **Stage 2** | 16–30 frame sequence fed to lightweight `LSTM` or `1D-CNN` classifier |
| **Output classes** | `safe_lift`, `unsafe_lift`, `fall`, `normal_walk`, `unauthorized_operation` |
| **Training data** | Eskişehir Production Facility Dataset (691 video clips, 8 classes, 1080p) |

---

## 6. Datasets

### 6.1 PPE Detection

| Dataset | Images | Classes | Source | License |
|---|---|---|---|---|
| **Construction-PPE (Ultralytics 2025)** | 1,416 images (1,132 train / 143 val / 141 test) | 11 (worn + missing) | docs.ultralytics.com/dataset_store/detect/construction-ppe | MIT license |
| **CHV (Color Helmet & Vest)** | 5,000+ | helmet, vest, person | GitHub / Papers | Research |
| **SHEL5K** | 5,000 | safety helmets | Open research | Research |
| **CHVG** | 1,699 | 8 classes incl. safety glass | PeerJ | Open |
| **Roboflow Universe PPE** | 100,000+ | various | universe.roboflow.com | Mixed |
| **Hard Hat Workers** | ~7,000 | helmet, head, person | Kaggle | Open |

> **Recommendation:** Start with Roboflow Universe PPE (100K+ images), supplement with your own factory footage for domain adaptation.

---

### 6.2 Pose / Fall Detection

| Dataset | Description | Source |
|---|---|---|
| **COCO Keypoints** | 200K+ people, 17 keypoints | cocodataset.org |
| **Le2i Fall Dataset** | Indoor fall detection videos | University of Burgundy |
| **MPFDD** | Multi-person fall detection | Nature Scientific Reports 2025 |
| **UR Fall Detection** | 70 fall sequences | University of Rochester |

---

### 6.3 Fire & Smoke

| Dataset | Images | Source |
|---|---|---|
| **FASDD** | 120,000+ | Largest open fire/smoke dataset ([github.com/openrsgis/FASDD](https://github.com/openrsgis/FASDD)) |
| **D-Fire** | 21,000+ | fire + smoke, YOLO annotations ([github.com/gaiasd/DFireDataset](https://github.com/gaiasd/DFireDataset)) |
| **Roboflow Fire** | 10,000+ | universe.roboflow.com |
| **Dunnings 2018/2019** | ~10,000 | Durham University (open) |

---

### 6.4 Factory Vehicles & Behavior

| Dataset | Description | Source |
|---|---|---|
| **Roboflow Forklift** | Forklift detection | universe.roboflow.com |
| **Industrial Safety Dataset** | Industrial PPE + vehicles | Kaggle |
| **Eskişehir Production Facility** | 691 video clips, 8 safe/unsafe behavior classes, 1080p 24fps, real factory | PubMed 2024 |

> **Important:** For forklifts and factory-specific behaviors, expect to collect and annotate custom data from your specific facility. No single open dataset covers all factory environments.

---

### 6.5 Annotation Tools

| Tool | Cost | Best For |
|---|---|---|
| **Roboflow** | Free tier / paid | Easiest — auto-annotation, dataset management, export to YOLO/COCO format |
| **Label Studio** | Open source | Flexible, self-hosted, supports video |
| **CVAT** | Open source | Professional, team annotation, video support |
| **X-AnyLabeling** | Open source | Latest models integration (SAM, DEIMv2) |

---

## 7. Training Resources

### 7.1 GPU Requirements for Training YOLO / RF-DETR / D-FINE

| Scenario | GPU | VRAM | Training Time | Cloud Cost |
|---|---|---|---|---|
| **Prototyping** | RTX 3060 | 12GB | ~12–16h / 100 epochs | ~$3–6 |
| **Recommended** | RTX 3090 / 4090 | 24GB | ~2–4h / 100 epochs | ~$3–8 |
| **Professional** | A100 40GB | 40GB | <2h / 100 epochs | ~$10–15/hr |
| **Fastest** | A100 80GB / H100 | 80GB | <1h / 100 epochs | ~$20–30/hr |

> For YOLO11/26 and RF-DETR fine-tuning on 5,000–50,000 image datasets, a **single RTX 3090 or 4090 is sufficient**. No multi-GPU setup needed.

### 7.2 Training Configuration (Typical)

```
Dataset size:     5,000–20,000 images per use case
Epochs:           100–300
Batch size:       16–32 (on 24GB GPU)
Image resolution: 640×640 (standard) / 1280 (small PPE objects)
Pretrained from:  COCO weights (transfer learning — NOT training from scratch)
Estimated time:   2–6 hours per model on RTX 4090
Framework:        Ultralytics Python API (YOLO) / Roboflow SDK (RF-DETR)
```

### 7.3 Cloud Training Platforms (No Local GPU Required)

| Platform | GPU Options | Cost | Notes |
|---|---|---|---|
| **RunPod** | RTX 4090 / A100 | ~$0.5–3/hr | Good community GPUs |
| **Google Colab Pro** | A100 40GB | ~$10/mo | Easy setup |
| **Lightning AI** | A10G / A100 | Free tier available | Good DL tooling |
| **AWS EC2 (p3/p4)** | V100 / A100 | ~$3–10/hr | Enterprise reliability |

---

## 8. Deployment Pipeline

### 8.1 Model Training → Edge Deployment Flow

```
1. COLLECT DATA
   └─ Existing cameras + manual annotation (Roboflow/CVAT)
   
2. TRAIN MODEL
   └─ Local GPU (remote PC) / Google Colab Pro → fine-tune from COCO pretrained
   
3. EXPORT
   ├─ For Hailo-8:   PyTorch → ONNX → Hailo DFC → .HEF file
   └─ For RK3588:    PyTorch → ONNX → RKNN Toolkit → .rknn file
   
4. DEPLOY TO EDGE
   └─ Copy .HEF/.rknn to edge board
   
5. INFERENCE PIPELINE
   └─ GStreamer / OpenCV → RTSP capture → model inference → 
      alert logic → MQTT publish
      
6. CENTRAL DASHBOARD
   └─ Subscribe to MQTT topics → display alerts → logging → notifications
```

### 8.2 Hailo-8 Deployment Specifics

The Hailo workflow uses its own compiler toolchain:

```
PyTorch Model (.pt)
      │
      ▼ export to ONNX
ONNX Model (.onnx)
      │
      ▼ Hailo Dataflow Compiler (DFC)
Hailo Archive (.har) — quantization & optimization
      │
      ▼ compile
Hailo Executable File (.hef) — runs on Hailo-8 hardware
      │
      ▼ runtime
HailoRT + GStreamer pipeline → real-time inference
```

**Hailo Model Zoo (pre-built HEF files available for):**
- YOLOv8n/s/m
- YOLO11n/s
- Person detection
- Pose estimation
- Face detection

> RF-DETR and D-FINE are not yet in Hailo Model Zoo — custom compilation via DFC required.

### 8.3 RK3588 NPU Deployment Specifics

```
PyTorch/ONNX Model
      │
      ▼ RKNN Toolkit2
.rknn Model (quantized INT8)
      │
      ▼ RKNN Runtime
NPU Inference on RK3588 (all 3 NPU cores)
```

- Inference time: ~25–30ms for YOLO-NAS-S on all 3 RK3588 NPU cores
- Best supported: YOLO family, MobileNet, ResNet variants

### 8.4 Software Stack

| Layer | Technology |
|---|---|
| **OS** | Raspberry Pi OS / Ubuntu 22.04 (ARM) |
| **Inference runtime** | HailoRT (Hailo), RKNN Runtime (Rockchip) |
| **Video pipeline** | GStreamer with hailonet plugin |
| **CV library** | OpenCV 4.x |
| **Alert logic** | Python 3.11 |
| **Communication** | MQTT (Mosquitto broker) / REST API |
| **Dashboard** | Node-RED / Grafana / Custom web app |
| **OTA updates** | Balena Cloud / custom Ansible |

---

## 9. Decision Matrix

### 9.1 Hardware Selection by Use Case Count

| Models per Camera | Recommended Board | Reasoning |
|---|---|---|
| 1 model only | RPi 5 + Hailo-8L | 13 TOPS sufficient for single YOLO model |
| 2–3 models | **RPi 5 + Hailo-8** | 26 TOPS handles parallel inference |
| 3–4 models | ArmSoM Sige7 (RK3588+Hailo-8) | Dual NPU — split tasks across chips |
| 4+ models | Renesas RZ/V2H | 100 TOPS for complex multi-model stacks |

### 9.2 Model Selection by Deployment Target

| Use Case | Hailo-8 (CNN only) | GPU-capable Edge |
|---|---|---|
| PPE Detection | `YOLO26s` | `RF-DETR-S` |
| Fall Detection | `YOLO11n-Pose` | `YOLO11s-Pose` / `DETRPose` |
| Zone Intrusion | `YOLO26n` + ByteTrack | `YOLO26n` + ByteTrack |
| Forklift Proximity | `YOLO11s` (custom) | `D-FINE-S` (custom) |
| Fire/Smoke | `YOLO11s` | `RF-DETR-S` |
| Behavior Classification | `YOLO11n-Pose` + LSTM | `RF-DETR` + LSTM |

### 9.3 Model Family Selection Guide

```
Is accuracy the top priority, and do you have GPU on edge?
├── YES → RF-DETR-S or D-FINE-S (Apache 2.0, higher mAP)
└── NO  → Does NPU/Hailo quantization matter?
          ├── YES → YOLO26n/s (best CNN for edge, NMS-free)
          └── NO  → RT-DETRv2-S (good balance, Apache 2.0)

Is this a commercial product requiring proprietary code?
├── YES → RF-DETR, D-FINE, RT-DETRv2 (Apache 2.0)
└── NO  → Any model fine (YOLO AGPL-3.0 ok for internal use)

Are objects small (goggles, gloves far from camera)?
├── YES → RF-DETR (DINOv2 backbone excels at small objects)
└── NO  → YOLO26s sufficient
```

---

## 10. Recommended Stack

### 10.1 Primary Recommendation — RPi 5 + Hailo-8 Stack

**Hardware per camera point:**
```
Raspberry Pi 5 (8GB)          ~$80
Hailo-8 M.2 HAT               ~$70
Waveshare PoE HAT              ~$20
MicroSD (32GB) / NVMe SSD     ~$10–30
Enclosure (weatherproof)       ~$15–30
─────────────────────────────────────
Total per camera:           ~$195–230
```

**Software stack:**
```
OS:         Raspberry Pi OS Bookworm (64-bit)
Runtime:    HailoRT 4.x + GStreamer 1.x
Models:     YOLO26s (PPE) + YOLO11n-Pose (Fall) — parallel on Hailo-8
Language:   Python 3.11
Alerts:     MQTT → Mosquitto broker → Node-RED dashboard
Updates:    Balena Cloud (remote fleet management)
```

**Why this stack:**
- Largest SBC community — easiest hiring, debugging, documentation
- Hailo-8 at 26 TOPS handles 2–3 parallel models at 30 FPS
- PoE-powered — zero extra wiring
- Hailo Model Zoo provides pre-built HEF files for YOLO family
- Proven in production worldwide

---

### 10.2 High-Accuracy Alternative — GPU-Capable Non-NVIDIA Edge

For scenarios where YOLO accuracy is insufficient (small PPE items, complex scenes), and using non-NVIDIA GPU:

**Hardware options (non-NVIDIA GPU edge):**

| Board | GPU | AI TOPS | Price |
|---|---|---|---|
| **Apple Mac Mini M4** | Apple Silicon GPU | ~38 TOPS ANE | ~$600 (server use) |
| **Beelink EQ13 (Intel N100)** | Intel iGPU + NPU | ~11 TOPS | ~$150–200 |
| **AMD Ryzen embedded (Radxa X4)** | AMD RDNA iGPU | ~16 TOPS | ~$100–150 |
| **Intel NUC 15 (Core Ultra)** | Intel Arc + NPU | ~99 TOPS | ~$500–700 |

**For Intel-based edge:**
```
Model export: PyTorch → ONNX → OpenVINO IR format
Runtime:      OpenVINO Runtime (optimized for Intel CPU/GPU/NPU)
Supports:     RF-DETR, D-FINE, RT-DETRv2, all YOLO variants
Best for:     RF-DETR-S at full accuracy on Intel Arc GPU
```

---

### 10.3 Full System Topology (Recommended for 10+ Cameras)

```
Factory Floor (per camera zone):
┌─────────────────────────────────┐
│  IP Camera (existing)           │
│  │ RTSP over PoE cable          │
│  ▼                              │
│  RPi 5 (8GB) + Hailo-8          │
│  ├─ YOLOX-M → PPE detection     │
│  ├─ MoveNet → Fall detect  │
│  ├─ Zone logic (software)        │
│  └─ MQTT alerts → network        │
└────────────┬────────────────────┘
             │ Ethernet
             ▼
        PoE Switch
             │
             ▼
   ┌──────────────────────┐
   │  On-Premise Server   │
   │  (any Linux server)  │
   │                      │
   │  Mosquitto (MQTT)    │
   │  Node-RED (logic)    │
   │  Grafana (dashboard) │
   │  PostgreSQL (logs)   │
   └──────────────────────┘
             │
             ▼
   Mobile alerts / Email / Siren
```

---

## 11. Cost Estimation

### 11.1 Per-Camera Unit Cost

| Component | Budget Build | Recommended | High-Performance |
|---|---|---|---|
| Edge Board | RPi 5 4GB + Hailo-8L | RPi 5 8GB + Hailo-8 | ArmSoM Sige7 |
| Board cost | ~$110 | ~$150 | ~$170 |
| PoE HAT | $15 | $20 | $20 |
| Storage | $10 | $20 | $30 |
| Enclosure | $15 | $25 | $30 |
| **Total/camera** | **~$150** | **~$215** | **~$250** |

### 11.2 Training Cost (One-Time, All Models)

| Phase | GPU | Hours | Cost |
|---|---|---|---|
| PPE model training | Local GPU | ~4h | $0 |
| Fall/pose model | RTX 4090 | ~2h | ~$2 |
| Fire/smoke model | RTX 4090 | ~3h | ~$3 |
| Forklift model | RTX 4090 | ~4h | ~$4 |
| Behavior classifier | RTX 4090 | ~3h | ~$3 |
| **Total training** | | **~16h** | **~$20–30** |

> **Model count update:** Phase 1 now has 6 models: (a) Fire Detection, (b) PPE Helmet, (f) PPE Safety Shoes, (g) Fall Detection, (h) Poketenashi, (i) Zone Intrusion. The table above reflects the original 5-model estimate; Safety Shoes and Poketenashi training costs should be added.

> Training cost is nearly negligible. The main cost is hardware and annotation labor.

### 11.3 Annotation Cost Estimate

| Use Case | Images Needed | Hours (manual) | Cost @ $10/hr |
|---|---|---|---|
| PPE detection | 5,000–10,000 | 40–80h | ~$400–800 |
| Forklift/vehicles | 2,000–5,000 | 20–40h | ~$200–400 |
| Fire/smoke | Use open datasets | 0h | $0 |
| Fall/pose | Use open datasets | 0h | $0 |
| **Total annotation** | | **60–120h** | **~$600–1,200** |

> Use **Roboflow auto-annotation** with SAM 3 to reduce annotation time by 60–80%.

### 11.4 Total Project Cost (Example: 20 Cameras)

| Item | Cost |
|---|---|
| Edge hardware (20× RPi 5 + Hailo-8 stack) | ~$4,300 |
| PoE switch (24-port) | ~$200 |
| Central server (mini PC) | ~$300 |
| Model training (cloud GPU) | ~$50 |
| Dataset annotation | ~$1,000 |
| Software / dashboards (open source) | $0 |
| **Total (20 cameras)** | **~$5,850** |
| **Per camera** | **~$290** |

---

## Appendix A: Model License Summary

| Model | License | Commercial Use | Code Disclosure |
|---|---|---|---|
| YOLO11/26 | AGPL-3.0 | ⚠️ Restricted | Required |
| YOLOv12 | AGPL-3.0 | ⚠️ Restricted | Required |
| **RF-DETR** | **Apache 2.0** | ✅ Free | Not required |
| **D-FINE** | **Apache 2.0** | ✅ Free | Not required |
| **RT-DETRv2/v4** | **Apache 2.0** | ✅ Free | Not required |
| **DEIM / DEIMv2** | **Apache 2.0** | ✅ Free | Not required |

> **For commercial factory products:** Prefer Apache 2.0 models (RF-DETR, D-FINE, RT-DETRv2). YOLO under AGPL-3.0 requires open-sourcing any modifications, which may be problematic for proprietary deployments.

---

## Appendix B: Key GitHub Repositories

| Resource | URL |
|---|---|
| RF-DETR | github.com/roboflow/rf-detr |
| D-FINE | github.com/Peterande/D-FINE |
| RT-DETR (official) | github.com/lyuwenyu/RT-DETR |
| DEIMv2 | github.com/Intellindust-AI-Lab/DEIMv2 |
| Hailo Model Zoo | github.com/hailo-ai/hailo_model_zoo |
| Supervision (zone tools) | github.com/roboflow/supervision |
| RKNN Toolkit2 | github.com/airockchip/rknn-toolkit2 |
| Ultralytics (YOLO) | github.com/ultralytics/ultralytics |

---

## Appendix C: Glossary

| Term | Meaning |
|---|---|
| TOPS | Tera Operations Per Second — AI compute throughput measure |
| PoE | Power over Ethernet — delivers power + data via single cable |
| RTSP | Real-Time Streaming Protocol — standard IP camera video stream |
| NPU | Neural Processing Unit — dedicated AI inference chip |
| HEF | Hailo Executable File — compiled model format for Hailo hardware |
| RKNN | Rockchip Neural Network — model format for RK3588 NPU |
| mAP | Mean Average Precision — standard object detection accuracy metric |
| NMS | Non-Maximum Suppression — post-processing step to remove duplicate detections |
| ONNX | Open Neural Network Exchange — universal model interchange format |
| DINOv2 | Meta's self-supervised vision transformer used as backbone in RF-DETR |
| ByteTrack | Fast multi-object tracker for maintaining person IDs across frames |
| MQTT | Lightweight messaging protocol for IoT/edge → server communication |

---

*Report prepared March 2026 | Based on literature and benchmarks current to early 2026*
*Hardware prices are approximate and subject to market variation*

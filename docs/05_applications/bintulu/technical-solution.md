# Bintulu Smart City - Technical Solution Proposal

**Date**: 2026-03-19
**Author**: Nguyen Thanh Trung (AI Camera Lead, VIETSOL)
**Customer**: Bintulu Development Authority (BDA), Sarawak, Malaysia
**Partner**: Elektro Serve Power Sdn Bhd (ESP)

---

## Table of Contents

1. [Solution Overview](#solution-overview)
2. [UC2: Smart Parking Solution (Phase 1 — Lead)](#uc2-smart-parking-solution-phase-1--lead)
3. [UC1: AI Traffic Light Solution (Phase 2)](#uc1-ai-traffic-light-solution-phase-2)
4. [Shared Platform Architecture](#shared-platform-architecture)
5. [VIETSOL Existing Assets & Reuse Plan](#vietsol-existing-assets--reuse-plan)
6. [Development Roadmap](#development-roadmap)
7. [Bill of Materials & Cost Estimate](#bill-of-materials--cost-estimate)

---

## Solution Overview

### Design Principles

1. **Edge-First Architecture** — All AI inference runs on edge devices. Only metadata (JSON events, bay status, plate text) is sent to backend. This reduces bandwidth by 100-1000x and ensures privacy (no raw video leaves the premises).

2. **Reuse Existing CCTV** — BDA's existing camera infrastructure is leveraged wherever possible. Edge AI boxes are added alongside existing cameras, minimizing hardware cost.

3. **Camera-Based (No Ground Sensors)** — Vision-based detection replaces per-bay sensors. One camera covers 40-60 bays, making the system 3-5x cheaper at scale than sensor-based alternatives.

4. **Model-Agnostic Pipeline** — VIETSOL's proven `camera_edge` detection pipeline supports multiple architectures (YOLOX, D-FINE, RT-DETRv2). New models can be added via YAML config without code changes.

5. **Apache 2.0 Licensing** — All AI models use Apache 2.0 or MIT licenses. Zero licensing fees, full commercial freedom.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        BINTULU SMART CITY                           │
│                                                                     │
│  ┌──────────────────┐    ┌──────────────────┐                      │
│  │  PARKING ZONES   │    │  INTERSECTIONS   │                      │
│  │  (7,000 bays)    │    │  (Phase 2)       │                      │
│  └────────┬─────────┘    └────────┬─────────┘                      │
│           │                       │                                  │
│     ┌─────▼─────┐          ┌─────▼─────┐                           │
│     │ IP Cameras │          │ IP Cameras │                          │
│     │ + LPR Cams │          │ (4/intx)   │                          │
│     └─────┬─────┘          └─────┬─────┘                           │
│           │ RTSP                 │ RTSP                             │
│     ┌─────▼─────────┐    ┌─────▼─────────┐                        │
│     │ EDGE AI BOX   │    │ EDGE AI BOX   │                        │
│     │ (per zone)    │    │ (per intx)    │                        │
│     │               │    │               │                        │
│     │ • YOLOX/D-FINE│    │ • YOLOX/D-FINE│                        │
│     │ • ANPR (OCR)  │    │ • ByteTrack   │                        │
│     │ • ByteTrack   │    │ • Queue est.  │                        │
│     │ • Zone logic  │    │ • Lane ROI    │                        │
│     └───────┬───────┘    └───────┬───────┘                        │
│             │ MQTT/REST          │ MQTT/REST                       │
│             │ (metadata only)    │ (metadata only)                 │
│     ┌───────▼────────────────────▼───────┐                        │
│     │      CENTRAL MANAGEMENT PLATFORM    │                        │
│     │                                     │                        │
│     │  ┌─────────┐  ┌──────────────────┐ │                        │
│     │  │ REST API │  │ MQTT Broker      │ │                        │
│     │  └────┬────┘  └────────┬─────────┘ │                        │
│     │       │                │            │                        │
│     │  ┌────▼────────────────▼─────────┐ │                        │
│     │  │ Backend (FastAPI)             │ │                        │
│     │  │ • Occupancy engine            │ │                        │
│     │  │ • ANPR matching               │ │                        │
│     │  │ • Alert manager               │ │                        │
│     │  │ • Traffic analytics           │ │                        │
│     │  │ • Revenue calculator          │ │                        │
│     │  └────┬─────────────────────────┘ │                        │
│     │       │                            │                        │
│     │  ┌────▼────────────────────────┐  │                        │
│     │  │ PostgreSQL + TimescaleDB    │  │                        │
│     │  └─────────────────────────────┘  │                        │
│     └───────┬────────────────────────────┘                        │
│             │                                                      │
│     ┌───────▼────────────────────────────┐                        │
│     │      END-USER INTERFACES            │                        │
│     │                                     │                        │
│     │  • Operator Dashboard (Web)         │                        │
│     │  • LED Guidance Signs (serial/ETH)  │                        │
│     │  • Mobile App (driver-facing)       │                        │
│     │  • Enforcement Handheld App         │                        │
│     └─────────────────────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## UC2: Smart Parking Solution (Phase 1 — Lead)

### 2.1 AI Models

#### Vehicle Detection & Bay Occupancy

| Component | Model | Details | Status |
|-----------|-------|---------|--------|
| **Primary detector** | YOLOX-M (ONNX, INT8) | 25.3M params, Apache 2.0, proven INT8 quantization | **EXISTING** — retrain on COCO vehicle classes |
| **Alternative detector** | D-FINE-S (ONNX) | 10M params, NMS-free, Apache 2.0 | **EXISTING** — HF adapter ready |
| **Object tracker** | ByteTrack | MIT license, fixed-camera optimized | **EXISTING** — production-ready |
| **Occupancy classifier** | MobileNetV3-Small | 2.5M params, per-slot crop classification | **EXISTING** — proven for crop classification |

**Detection approach**: Hybrid (detection + classification)
1. **YOLOX/D-FINE** detects all vehicles in frame → bounding boxes
2. Each bounding box is matched against predefined bay polygon ROIs
3. Bays with overlapping vehicle detections are marked "occupied"
4. **MobileNetV3** serves as fallback per-slot classifier for ambiguous cases
5. Temporal smoothing (3-5 second window) prevents flicker

**What we reuse from camera_edge:**
- `core/models/registry.py` — model dispatch (YOLOX, D-FINE, RT-DETRv2)
- `core/inference/predictor.py` — `DetectionPredictor` with PyTorch + ONNX backends
- `core/inference/video_inference.py` — `VideoProcessor` with alert system + ByteTrack
- `core/export/exporter.py` — ONNX export with validation
- `core/export/quantize.py` — INT8 quantization (dynamic + static calibration)
- `core/data/dataset.py` + `core/data/transforms.py` — training pipeline
- `utils/supervision_bridge.py` — tracker creation, annotation rendering

**What we need to build:**
- Bay polygon ROI configuration UI
- Occupancy mapping logic (detection → bay → status)
- Temporal smoothing for stable occupancy reporting
- Vehicle type classification head (car/motorcycle/truck)

#### ANPR / License Plate Recognition

| Component | Model | Details | Status |
|-----------|-------|---------|--------|
| **Plate detector** | YOLOX-Tiny or D-FINE-N | Lightweight, detect plate bounding box | **NEW** — train on Malaysian plate dataset |
| **Plate OCR** | PaddleOCR (or TrOCR) | Open-source, supports custom alphabets | **NEW** — integrate + fine-tune for MY plates |
| **IR handling** | Preprocessing pipeline | Grayscale normalization for IR images | **NEW** |

**ANPR pipeline:**
```
Camera frame (IR/visible) → Plate Detection (YOLOX-Tiny) → Crop plate region
    → Perspective correction → OCR (PaddleOCR) → Plate text + confidence
    → Match against database → Entry/exit event
```

**Key design decisions:**
- Dedicated LPR cameras at entry/exit (5MP, IR illuminator, narrow FoV)
- Edge processing: plate text + timestamp + snapshot sent to backend (no video streaming)
- Malaysian plate format support: `ABC 1234`, `W 1234 A`, `Sarawak S*` variants

#### Safety Monitoring

| Feature | Model | Details | Status |
|---------|-------|---------|--------|
| **Person detection** | YOLOX-M (COCO person class) | Pretrained, no additional training needed | **EXISTING** — zone intrusion model |
| **Loitering detection** | ByteTrack + dwell time logic | Track person, trigger if stationary > threshold | **PARTIAL** — tracker exists, need dwell-time logic |
| **Intrusion detection** | Person detection + zone polygon | Detect person entering restricted polygon area | **PARTIAL** — config skeleton exists, need implementation |
| **Fall detection** | YOLOX-M (fallen_person class) | Trained on 17K images, mAP >= 0.85 | **EXISTING** — trained model available |

**What we reuse:**
- Zone intrusion experiment (`features/access-zone_intrusion/experiments/`) — person detection is ready
- Fall detection model (17K training images, 2-class: person, fallen_person)
- Alert system in `VideoProcessor` — configurable confidence thresholds, frame windows, cooldown
- ByteTrack tracking — already integrated in video pipeline

**What we need to build:**
- Functional zone polygon enforcement (config exists, logic not implemented)
- Dwell-time tracking (person stationary in zone > N seconds)
- Zone management UI for operators

#### Violation Detection

| Feature | Approach | Status |
|---------|----------|--------|
| **Double parking** | Vehicle detected outside any bay ROI > 2 min | **NEW** — combine detection + zone + timer |
| **Obstruction** | Vehicle in fire lane / access lane ROI > 1 min | **NEW** — same approach as double parking |
| **Overtime** | ANPR entry time + occupancy > max time | **NEW** — backend logic |

### 2.2 Edge Hardware Specification

**Recommended: NVIDIA Jetson Orin NX 16GB**

| Spec | Value | Rationale |
|------|-------|-----------|
| AI performance | 100 TOPS (INT8) | Handles 8-12 camera streams with YOLOX-M INT8 |
| Memory | 16 GB LPDDR5 | Sufficient for multi-model inference |
| Power | 10-25W | Low power for outdoor enclosure |
| Cameras per device | 8-12 streams @1080p | Covers ~400-700 bays per edge box |
| Price | ~$399-$599 (module) | Cost-effective at scale |
| ONNX Runtime | Full support | Our export pipeline targets ONNX |

**Alternative: AX650N (AXera) — VIETSOL's primary target chip**

| Spec | Value | Rationale |
|------|-------|-----------|
| AI performance | 18 TOPS (INT8) | Sufficient for 2-4 cameras with optimized models |
| CPU | 8x Cortex-A55 | Edge computing capable |
| Price | Lower than Jetson | Better for high-volume deployment |
| Compiler | AXera toolchain | ONNX → AXera format conversion needed |

**Deployment plan for 7,000 bays:**

| Component | Quantity | Unit Cost | Total |
|-----------|----------|-----------|-------|
| Occupancy cameras (2MP, IP67) | 140 (avg 50 bays/cam) | $200-$500 | $28K-$70K |
| ANPR cameras (5MP, IR, LPR) | 30 (entry/exit points) | $500-$1,500 | $15K-$45K |
| Edge AI boxes (Jetson Orin NX) | 15 (10 cams/box) | $1,500-$3,000 | $22.5K-$45K |
| PoE switches (16-port) | 15 | $200-$500 | $3K-$7.5K |
| LED guidance signs | 40 (per zone/floor) | $500-$2,000 | $20K-$80K |
| Network (fiber backbone) | 1 lot | — | $20K-$50K |
| **Hardware subtotal** | | | **$108K-$297K** |

### 2.3 Software Architecture

```
┌─────────────────────────────────────────────────┐
│                 EDGE AI BOX                      │
│                                                  │
│  ┌──────────────┐   ┌──────────────────────┐    │
│  │ RTSP Manager  │──▶│ Frame Buffer (ring)  │    │
│  │ (per camera)  │   └──────────┬───────────┘    │
│  └──────────────┘              │                 │
│                     ┌──────────▼───────────┐     │
│                     │ DetectionPredictor    │     │
│                     │ (YOLOX-M INT8 ONNX)  │     │
│                     └──────────┬───────────┘     │
│                                │                 │
│              ┌─────────────────┼──────────────┐  │
│              ▼                 ▼              ▼   │
│  ┌───────────────┐  ┌──────────────┐  ┌───────┐ │
│  │ Bay Occupancy  │  │ Safety Alert │  │ ANPR  │ │
│  │ Engine         │  │ Engine       │  │Engine │ │
│  │                │  │              │  │       │ │
│  │• ROI matching  │  │• Zone check  │  │• Plate│ │
│  │• Temporal avg  │  │• Dwell time  │  │ detect│ │
│  │• Status pub    │  │• Fall detect │  │• OCR  │ │
│  └───────┬───────┘  └──────┬───────┘  └───┬───┘ │
│          │                 │              │      │
│          ▼                 ▼              ▼      │
│  ┌───────────────────────────────────────────┐   │
│  │ MQTT Publisher (metadata only)            │   │
│  │ • bay/{zone_id}/{bay_id}/status           │   │
│  │ • alert/{camera_id}/{event_type}          │   │
│  │ • anpr/{gate_id}/{plate_text}             │   │
│  └───────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

**MQTT message examples:**

```json
// Bay status update
{
  "topic": "bay/zone_a/bay_042/status",
  "payload": {
    "status": "occupied",
    "vehicle_type": "car",
    "confidence": 0.94,
    "timestamp": "2026-03-19T14:32:05+08:00"
  }
}

// ANPR event
{
  "topic": "anpr/gate_north/entry",
  "payload": {
    "plate": "QSA 1234",
    "confidence": 0.97,
    "snapshot_url": "/snapshots/20260319_143205_gate_north.jpg",
    "timestamp": "2026-03-19T14:32:05+08:00"
  }
}

// Safety alert
{
  "topic": "alert/cam_b12/loitering",
  "payload": {
    "type": "loitering",
    "zone": "staff_only_area",
    "duration_seconds": 312,
    "snapshot_url": "/snapshots/20260319_143205_cam_b12.jpg",
    "confidence": 0.88,
    "timestamp": "2026-03-19T14:32:05+08:00"
  }
}
```

---

## UC1: AI Traffic Light Solution (Phase 2)

### 3.1 AI Models

#### Vehicle Detection & Classification

| Component | Model | Details | Status |
|-----------|-------|---------|--------|
| **Primary detector** | YOLOX-M (ONNX, INT8) | Retrain or fine-tune on COCO traffic classes | **EXISTING** — need traffic-class training |
| **Alternative** | D-FINE-S (ONNX) | NMS-free, better for dense intersections | **EXISTING** — HF adapter ready |
| **Tracker** | ByteTrack | Vehicle tracking across frames for counting + queue estimation | **EXISTING** |

**Vehicle classes**: car, motorcycle, truck, bus, bicycle, pedestrian (6 classes from COCO subset)

**Training plan**: Fine-tune YOLOX-M on intersection-specific data. Start with COCO pretrained weights (80 classes), fine-tune on 6 target classes. Our training pipeline (`core/training/trainer.py`) handles this natively.

#### Queue Length & Density Estimation

| Feature | Approach | Status |
|---------|----------|--------|
| **Queue length** | Count stopped vehicles per lane ROI (speed < threshold via tracker) | **NEW** — combine ByteTrack + lane ROI + speed estimate |
| **Traffic density** | Vehicle count per lane per time window | **NEW** — counting logic on existing detection |
| **Flow rate** | Vehicles crossing virtual line per minute | **NEW** — line-crossing counter |

**Reuse from camera_edge:**
- ByteTrack tracking provides per-vehicle trajectories
- `VideoProcessor` frame-by-frame processing with analytics collection
- Alert system for anomaly triggers

#### Adaptive Signal Timing

| Component | Approach | Status |
|-----------|----------|--------|
| **Phase optimizer** | Rule-based (Webster's formula) for initial deployment | **NEW** |
| **Advanced optimizer** | Reinforcement Learning (future, using SUMO-RL for training) | **FUTURE** |
| **Controller interface** | NTCIP 1202 (SNMP) or vendor-specific API | **NEW** |

**Phase 1 approach (rule-based):**
1. Measure real-time queue length and vehicle count per approach
2. Calculate optimal green time using modified Webster's formula:
   - `green_time = max(min_green, base_green * (queue_ratio / avg_queue_ratio))`
3. Distribute cycle time proportionally to demand
4. Respect min/max green constraints and all-red clearance

**Phase 2 approach (RL-based, future):**
- Train DQN/PPO agent in SUMO simulator with real intersection topology
- State: queue lengths, phase state, time-of-day
- Action: next phase selection + duration
- Reward: minimize total delay + maximize throughput

### 3.2 Edge Hardware

**Same as Smart Parking: NVIDIA Jetson Orin NX or AX650N**

Per intersection:
- 1 edge AI box (handles 4 approach cameras)
- 4 IP cameras (one per approach direction), or 1 panoramic 360-degree camera
- Serial/Ethernet connection to existing traffic signal controller

### 3.3 Signal Controller Integration

```
┌──────────────────────────────────────────┐
│              EDGE AI BOX                  │
│                                           │
│  [4 Camera Streams] → [YOLOX Detection]   │
│          ↓                                │
│  [ByteTrack Tracking]                     │
│          ↓                                │
│  [Per-Lane Analytics]                     │
│  • Queue length (stopped vehicles)        │
│  • Vehicle count (per class)              │
│  • Flow rate (vehicles/min)               │
│          ↓                                │
│  [Phase Optimizer]                        │
│  • Calculate optimal green times          │
│  • Apply min/max constraints              │
│  • Generate phase plan                    │
│          ↓                                │
│  [Controller Interface]                   │
│  • NTCIP 1202 (SNMP SET/GET)             │
│  • Or vendor-specific serial protocol     │
│          ↓                                │
│  [Failsafe Monitor]                       │
│  • If AI system fails → revert to         │
│    fixed-time plan in controller          │
└──────────┬───────────────────────────────┘
           │ SNMP / Serial
           ▼
┌──────────────────────┐
│ TRAFFIC SIGNAL       │
│ CONTROLLER           │
│ (existing hardware)  │
│ • NEMA TS2 / ATC     │
│ • Fixed-time backup  │
└──────────────────────┘
```

**Failsafe design:**
- Controller retains its own fixed-time program
- Edge AI sends timing recommendations, not direct signal commands
- If edge device is unresponsive for > 2 cycles, controller auto-reverts
- Manual override always available

---

## Shared Platform Architecture

### 4.1 Central Management Backend

```
┌──────────────────────────────────────────────────────┐
│                  BACKEND SERVER                        │
│                                                        │
│  ┌─────────────────────────────────────────────────┐  │
│  │  FastAPI Application                            │  │
│  │                                                  │  │
│  │  /api/v1/parking/                               │  │
│  │    GET  /zones                  — list zones     │  │
│  │    GET  /zones/{id}/occupancy   — live status    │  │
│  │    GET  /zones/{id}/history     — time series    │  │
│  │    GET  /anpr/entries           — entry log      │  │
│  │    GET  /anpr/plate/{plate}     — plate history  │  │
│  │    GET  /violations             — violation list  │  │
│  │    POST /violations/{id}/ack    — acknowledge    │  │
│  │                                                  │  │
│  │  /api/v1/traffic/                               │  │
│  │    GET  /intersections          — list all       │  │
│  │    GET  /intersections/{id}/live — live metrics  │  │
│  │    GET  /intersections/{id}/signals — phase info │  │
│  │    PUT  /intersections/{id}/plan — update plan   │  │
│  │                                                  │  │
│  │  /api/v1/alerts/                                │  │
│  │    GET  /                       — list alerts    │  │
│  │    POST /{id}/acknowledge       — ack alert      │  │
│  │    GET  /stats                  — alert summary  │  │
│  │                                                  │  │
│  │  /ws/v1/live                    — WebSocket feed │  │
│  └──────────────────────┬──────────────────────────┘  │
│                         │                              │
│  ┌──────────────────────▼──────────────────────────┐  │
│  │  PostgreSQL + TimescaleDB                       │  │
│  │                                                  │  │
│  │  • parking_bays         (status, zone, coords)  │  │
│  │  • occupancy_events     (hypertable, 5s grain)  │  │
│  │  • anpr_events          (plate, gate, time)     │  │
│  │  • parking_sessions     (entry, exit, duration) │  │
│  │  • traffic_metrics      (hypertable, 1min)      │  │
│  │  • signal_events        (phase changes)         │  │
│  │  • alerts               (type, status, time)    │  │
│  │  • violations           (type, evidence, plate) │  │
│  └─────────────────────────────────────────────────┘  │
│                                                        │
│  ┌─────────────────────────────────────────────────┐  │
│  │  Mosquitto MQTT Broker                          │  │
│  │  • bay/+/+/status        (occupancy updates)    │  │
│  │  • anpr/+/+              (plate events)         │  │
│  │  • alert/+/+             (safety alerts)        │  │
│  │  • traffic/+/metrics     (intersection data)    │  │
│  └─────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

### 4.2 Operator Dashboard (Web)

| Feature | UC2: Parking | UC1: Traffic |
|---------|-------------|-------------|
| **Live map** | Parking zone map with bay-level occupancy colors | Intersection map with signal phase status |
| **Camera feeds** | Live view with AI overlay (bboxes, bay status) | Live view with detection overlay |
| **Metrics** | Occupancy %, turnover rate, dwell time, revenue | Queue length, flow rate, delay, congestion index |
| **Alerts** | Loitering, intrusion, fall, violations | Stalled vehicle, anomaly, incident |
| **Analytics** | Peak hour trends, zone heatmaps, revenue reports | Traffic volume trends, signal performance, travel time |
| **Config** | Bay ROI editor, zone management, alert thresholds | Lane ROI editor, timing plan editor, min/max green |
| **Reports** | Daily revenue, violations, occupancy summary | Daily volume, congestion, incident summary |

### 4.3 Mobile App (Driver-Facing, Smart Parking)

| Feature | Priority |
|---------|----------|
| Find available parking by zone | Must |
| Navigate to open bay | Should |
| View real-time parking map | Must |
| Check parking fee / dwell time | Must |
| Mobile payment integration (Touch 'n Go, Boost) | Should |
| Pre-book parking spot | Could |
| Push notification: parking expiring | Should |

---

## VIETSOL Existing Assets & Reuse Plan

### 5.1 Direct Reuse (No Modification Needed)

| Asset | Path | Application |
|-------|------|-------------|
| **YOLOX-M detection** | `core/models/yolox.py` | Vehicle detection (retrain on traffic/parking classes) |
| **D-FINE-S/N detection** | `core/models/dfine.py` + `hf_model.py` | Alternative detector, NMS-free |
| **RT-DETRv2 detection** | `core/models/rtdetr.py` + `hf_model.py` | ONNX-reliable export |
| **Model registry** | `core/models/registry.py` | Architecture dispatch, variant mapping |
| **Training pipeline** | `core/training/trainer.py` | Config-driven training with EMA, AMP, callbacks |
| **Loss functions** | `core/training/losses.py` | YOLOXLoss, FocalLoss, IoULoss |
| **LR scheduler** | `core/training/lr_scheduler.py` | Cosine annealing with warmup |
| **Callbacks** | `core/training/callbacks.py` | Checkpoint, early stopping, WandB |
| **Dataset + transforms** | `core/data/dataset.py`, `transforms.py` | Mosaic, MixUp, HSV, affine augmentation |
| **ONNX export** | `core/export/exporter.py` | Validated export with simplification |
| **INT8 quantization** | `core/export/quantize.py` | Dynamic + static calibration |
| **Benchmarking** | `core/export/benchmark.py` | Latency, throughput, model size comparison |
| **PyTorch inference** | `core/inference/predictor.py` | `.pt` and `.onnx` backend auto-detection |
| **Video processing** | `core/inference/video_inference.py` | Frame-by-frame pipeline with alerts + analytics |
| **ByteTrack tracker** | `utils/supervision_bridge.py` | `create_tracker()`, `update_tracker()` |
| **Annotation rendering** | `utils/supervision_bridge.py` | `build_annotators()`, `annotate_frame()` |
| **Config system** | `utils/config.py` | YAML load/merge/validate/interpolate |
| **Device management** | `utils/device.py` | CUDA/MPS/CPU auto-detection |
| **Metrics** | `utils/metrics.py` | IoU, AP, mAP, confusion matrix |
| **Visualization** | `utils/visualization.py` | Bboxes, PR curves, training charts |
| **HPO** | `core/p07_hpo/` | Optuna hyperparameter search |
| **Auto-annotation** | `core/p01_auto_annotate/` | SAM3-based labeling for new datasets |

### 5.2 Partial Reuse (Extend Existing Code)

| Asset | Current State | Extension Needed |
|-------|---------------|-----------------|
| **Zone intrusion model** | Person detection (COCO pretrained) | Add vehicle detection, functional zone polygon enforcement |
| **Alert system** | Frame-window + cooldown based | Add dwell-time alerts, zone-specific alerts |
| **Gradio demo** | 5-tab safety demo | Extend to parking occupancy + ANPR demo |
| **Fall detection** | Trained (17K images, 2 classes) | Reuse as-is for parking safety monitoring |
| **MobileNetV3 classifier** | Shoe/helmet crop classification | Retrain for vehicle type or bay occupied/vacant |
| **Zone config** | `app_demo/config.yaml` (zones.enabled: false, polygons: []) | Implement polygon enforcement logic |

### 5.3 New Development Required

| Component | Effort | Description |
|-----------|--------|-------------|
| **Bay occupancy engine** | Medium | ROI polygon matching + temporal smoothing + status publishing |
| **ANPR pipeline** | Large | Plate detection (YOLOX-Tiny) + OCR (PaddleOCR) + MY plate format |
| **Violation detection** | Medium | Double parking timer, obstruction zones, overtime logic |
| **Dwell-time tracker** | Small | Extend ByteTrack with per-ID timer for loitering |
| **MQTT publisher** | Small | Edge-side MQTT client for metadata streaming |
| **Central backend** | Large | FastAPI + PostgreSQL + TimescaleDB + MQTT broker |
| **Operator dashboard** | Large | Web UI (React/Vue) with maps, live feeds, analytics |
| **LED sign driver** | Small | Serial/Ethernet protocol to drive guidance signs |
| **Mobile app** | Large | Driver-facing app (React Native or Flutter) |
| **Queue length estimator** | Medium | Stopped-vehicle detection per lane ROI |
| **Phase optimizer** | Medium | Rule-based signal timing (Webster's) |
| **Controller interface** | Large | NTCIP 1202 / SNMP integration |
| **Multi-camera manager** | Medium | RTSP stream manager for 150+ cameras |

---

## Development Roadmap

### Phase 1: Smart Parking MVP (Months 1-6)

```
Month 1-2: Foundation
├── Vehicle detection model (COCO classes, fine-tune YOLOX-M)
├── Bay occupancy engine (ROI matching + temporal smoothing)
├── ANPR pipeline (plate detection + PaddleOCR + MY plates)
├── Zone polygon enforcement (extend existing config)
└── MQTT publisher on edge

Month 3-4: Platform
├── Central backend (FastAPI + PostgreSQL + MQTT)
├── Occupancy API + ANPR matching engine
├── Operator dashboard MVP (live map, camera feeds, alerts)
├── LED guidance sign integration
└── Safety monitoring (loitering, intrusion, fall reuse)

Month 5-6: Integration & Pilot
├── Pilot deployment: 500-1,000 bays (1 zone)
├── Violation detection (double parking, obstruction)
├── Payment integration (Touch 'n Go API)
├── Dashboard analytics (trends, revenue, reports)
├── Model fine-tuning on Bintulu real data
└── Acceptance testing
```

### Phase 2: Smart Parking Full Rollout (Months 7-10)

```
Month 7-8: Scale
├── Full rollout: remaining zones (7,000 bays total)
├── Mobile app (driver-facing, find parking + pay)
├── Advanced analytics (heatmaps, prediction)
└── Performance optimization at scale

Month 9-10: Stabilize
├── 30-day acceptance testing at full scale
├── Operator training
├── Documentation and handover
└── SLA monitoring and support setup
```

### Phase 3: AI Traffic Light (Months 8-14)

```
Month 8-10: Foundation
├── Vehicle classification model (6 classes, COCO fine-tune)
├── Multi-lane ROI configuration
├── Queue length estimation (tracker + stopped-vehicle logic)
├── Traffic flow counting (line-crossing counter)
└── Traffic analytics backend module

Month 11-12: Signal Control
├── Phase optimizer (rule-based, Webster's formula)
├── NTCIP 1202 / controller interface
├── Failsafe mechanism
├── Traffic dashboard module
└── Pilot: 2-3 intersections

Month 13-14: Scale & Accept
├── Full intersection rollout
├── RL-based optimizer (if needed, trained in SUMO)
├── Emergency vehicle priority
├── Central coordination (multi-intersection)
└── 30-day acceptance testing
```

---

## Bill of Materials & Cost Estimate

### UC2: Smart Parking (7,000 bays)

| Category | Item | Qty | Unit Cost (USD) | Total (USD) |
|----------|------|-----|-----------------|-------------|
| **Hardware** | Occupancy cameras (2MP, IP67) | 140 | $300 | $42,000 |
| | ANPR cameras (5MP, IR, LPR) | 30 | $1,000 | $30,000 |
| | Edge AI boxes (Jetson Orin NX + enclosure) | 15 | $2,500 | $37,500 |
| | PoE switches (16-port, outdoor) | 15 | $400 | $6,000 |
| | LED guidance signs | 40 | $1,000 | $40,000 |
| | Network (fiber, switches, router) | 1 lot | — | $30,000 |
| | Backend server (on-premise) | 1 | $5,000 | $5,000 |
| **Hardware subtotal** | | | | **$190,500** |
| **Software** | AI model development & training | — | — | $40,000 |
| | ANPR pipeline development | — | — | $30,000 |
| | Central platform (backend + dashboard) | — | — | $60,000 |
| | Mobile app | — | — | $30,000 |
| | Integration & testing | — | — | $20,000 |
| **Software subtotal** | | | | **$180,000** |
| **Services** | Installation & commissioning | — | — | $40,000 |
| | Training & documentation | — | — | $10,000 |
| | 1-year support & maintenance | — | — | $30,000 |
| **Services subtotal** | | | | **$80,000** |
| **TOTAL** | | | | **$450,500** |

**Per-bay cost: ~$64/bay** (vs. $400-$750/bay for sensor-based solutions)

### UC1: AI Traffic Light (per intersection, estimate)

| Category | Item | Qty | Unit Cost (USD) | Total (USD) |
|----------|------|-----|-----------------|-------------|
| **Hardware** | IP cameras (2MP, IP67) | 4 | $500 | $2,000 |
| | Edge AI box (Jetson Orin NX + enclosure) | 1 | $2,500 | $2,500 |
| | Network equipment | 1 lot | — | $1,500 |
| **Hardware subtotal** | | | | **$6,000** |
| **Software** | AI model + signal optimizer (amortized) | — | — | $5,000 |
| | Controller interface integration | — | — | $3,000 |
| | Platform module (amortized per intersection) | — | — | $2,000 |
| **Software subtotal** | | | | **$10,000** |
| **Services** | Installation + commissioning | — | — | $4,000 |
| **Services subtotal** | | | | **$4,000** |
| **TOTAL per intersection** | | | | **$20,000** |

**Competitive positioning**: Market median is $45,000/intersection. Our solution at $20,000/intersection is highly cost-competitive, especially for Southeast Asian markets.

---

## Key Differentiators

| Factor | VIETSOL Solution | Typical Competitor |
|--------|-----------------|-------------------|
| **Licensing** | Apache 2.0 / MIT (zero fees) | Proprietary (per-camera/per-intersection annual fees) |
| **Edge hardware** | Jetson Orin NX / AX650N (commodity) | Proprietary hardware lock-in |
| **Model flexibility** | YOLOX + D-FINE + RT-DETRv2 (swap via YAML config) | Single model, vendor-locked |
| **CCTV reuse** | Native RTSP/ONVIF support on any camera | Often requires proprietary cameras |
| **Cost (parking, 7K bays)** | ~$450K total (~$64/bay) | $2M-$5M (sensor-based) |
| **Cost (traffic, per intx)** | ~$20K | $45K median (market) |
| **Local partnership** | ESP (Bintulu-based) + VIETSOL (Hanoi) | Foreign vendor, remote support |
| **Customization** | Full source code access, config-driven | Black box, limited customization |

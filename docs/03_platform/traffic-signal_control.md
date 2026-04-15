# Technical Approach — AI Adaptive Traffic Signal Control

## Bintulu, Sarawak Smart Traffic System

**Date:** March 19, 2026
**Target Deployment:** Bintulu Development Authority (BDA), Sarawak, Malaysia
**Target Edge Hardware:** NVIDIA Jetson Orin NX 16GB (primary) / AX650N (resource-constrained alternative)
**License Constraint:** Apache 2.0 or MIT only (no AGPL/Ultralytics)

---

## Customer Requirements

**Source:** Bintulu Development Authority (BDA) Smart Traffic Light System Plan; [BDA mulls installing smart system for traffic lights across Bintulu town (Borneo Post, 2019)](https://www.theborneopost.com/2019/07/29/bda-mulls-installing-smart-system-for-traffic-lights-across-bintulu-town/); [BDA Strategic Plan: Smart Liveable City (Borneo Post, 2023)](https://www.theborneopost.com/2023/12/22/bdas-strategic-plan-repositions-bintulu-into-a-smart-liveable-city/)

**Explicit Requirements:**
- Deploy an adaptive traffic signal control system across Bintulu intersections to replace fixed-time signaling
- Optimize traffic flow in real-time based on live camera observations of queue lengths, vehicle counts, and speeds
- Provide emergency vehicle priority (preemption) to reduce response times for ambulances and fire trucks
- Enable central monitoring and management of all intersections via a cloud dashboard for BDA operations staff
- Support corridor-level coordination between adjacent intersections (green wave) for major arterial roads
- Operate autonomously at the edge during network outages — traffic must never go dark
- Use resource-constrained hardware and open-source AI to deploy at scale without depending on proprietary commercial alternatives (NoTraffic, Miovision)

**Reference Data:**

| Metric | Value |
|---|---|
| Target edge hardware | NVIDIA Jetson Orin NX 16GB (primary) / AX650N (resource-constrained) |
| License constraint | Apache 2.0 or MIT only (no AGPL/Ultralytics) |
| Cameras per intersection | 4x PoE IP cameras (one per approach leg) |
| Pilot scope | 3 intersections in Bintulu town center |
| Scale target | 20-50+ intersections across Bintulu |
| System uptime target | >= 99.5% |

**Customer Reference:** [SASCOO AI Traffic System (Ledvision Malaysia)](https://ledvision.com.my/project/transforming-traffic-flow-with-ai-the-sascoo/) — existing AI traffic deployment in Malaysia; NoTraffic AI Mobility Platform; Miovision Surtrac

## Business Problem Statement

- **Traffic congestion:** Bintulu's growing urban population and industrial activity (port city, LNG, palm oil) create peak-hour congestion at key intersections, causing delays that frustrate residents and reduce economic productivity
- **Emergency response delays:** Fixed-time signals cannot prioritize emergency vehicles, adding critical seconds to ambulance and fire truck response times when every second matters for patient outcomes
- **Air quality and environmental impact:** Idling vehicles at congested intersections increase local emissions, contributing to poor air quality in a city already affected by industrial activity
- **Economic impact of congestion:** Unnecessary delays for commercial vehicles, commuters, and logistics fleets translate to measurable economic losses for a city positioning itself as a smart liveable city
- **Limited municipal resources:** BDA requires a resource-efficient solution — commercial adaptive traffic systems (NoTraffic, Miovision) are proprietary and require vendor lock-in, limiting flexibility and local control at scale for a mid-size Malaysian city
- **Aging infrastructure:** Existing traffic controllers run fixed-time plans that cannot adapt to changing demand patterns, special events, incidents, or seasonal traffic variations
- **Safety incidents:** Without real-time monitoring, stalled vehicles, wrong-way drivers, and red-light runners go undetected until a collision occurs

## Technical Problem Statement

- **Congestion → Real-time multi-stream perception:** Each intersection requires simultaneous processing of 4 camera feeds (one per approach leg) to measure queue lengths, vehicle counts, speeds, and density — all within a strict latency budget on edge hardware with limited compute
- **Emergency response → Low-latency signal preemption:** Emergency vehicle detection must trigger signal changes within 15 seconds total (detection + confirmation + controller transition), requiring fast inference and reliable NTCIP 1202 communication with existing traffic signal controllers
- **Hardware constraint → Resource-constrained edge devices:** The system must run on standard edge devices processing 4 camera streams simultaneously with INT8 quantization, while maintaining >= 10 FPS per stream for reliable analytics
- **Infrastructure integration → Protocol compatibility:** Must integrate with heterogeneous existing traffic signal controllers (Siemens, Swarco, Malaysian vendors) via NTCIP 1202 SNMP, vendor-specific serial protocols, or REST APIs — the exact hardware is unknown until site survey
- **Coordination → Multi-intersection optimization:** Green wave coordination between adjacent intersections requires inter-agent communication (MQTT) and multi-agent reinforcement learning, while remaining stable if network connectivity is lost
- **Resilience → Autonomous edge operation:** The edge device must continue making valid signal timing decisions during extended network outages (hours to days), falling back to rule-based control when RL policy is unavailable
- **Tropical environment → Harsh operating conditions:** Outdoor traffic cabinets in Bintulu reach 55-65 deg C internal temperature; edge hardware must operate reliably in IP67 enclosures with passive cooling at 25W sustained GPU load
- **Motorcycle-heavy traffic → Small-object detection:** Malaysian intersections have high motorcycle density with lane-splitting behavior, requiring detection models that can identify small, heavily occluded motorcycles weaving between larger vehicles
- **Data scarcity → Sim-to-real transfer:** No pre-existing labeled traffic dataset for Bintulu; RL policies must be trained in simulation (SUMO/CityFlow) and transferred to real intersections with minimal performance degradation

## Technical Solution Options

### Option 1: D-FINE-S Detection + ByteTrack + PPO Reinforcement Learning (Recommended)

- **Approach:** NMS-free D-FINE-S (10M params, Apache 2.0) for vehicle detection across 4 camera streams, ByteTrack for multi-object tracking and counting, PPO-based reinforcement learning for adaptive signal timing trained in SUMO/CityFlow simulation, with a rule-based Webster/actuated controller as safety fallback. Hybrid architecture: rule-based controller always provides a valid timing plan, RL optimizer suggests adjustments that pass through a safety validator before execution.
- **Addresses:** Real-time multi-stream perception (NMS-free eliminates bottleneck, ~10 FPS/stream on Orin NX), emergency preemption (visual detection of ambulance/fire_truck classes + NTCIP preemption command), hardware constraint (standard Jetson hardware, open-source models), multi-intersection coordination (MAPPO for multi-agent RL), autonomous operation (rule-based fallback when RL unavailable), sim-to-real (CityFlow pre-training + SUMO fine-tuning + domain randomization)
- **Pros:** D-FINE-S offers best accuracy-to-size ratio (48.5% mAP, 10M params); NMS-free critical for 4-stream latency; ICLR 2025 peer-reviewed; ByteTrack is 171 FPS with no Re-ID overhead; PPO is stable and scales to multi-agent; hybrid architecture ensures safety even if AI fails
- **Cons:** D-FINE transformer architecture may have lower NPU utilization on AX650N (resource-constrained target); RL training requires simulation infrastructure and sim-to-real validation; multi-agent RL adds complexity for corridor coordination; emergency vehicle visual detection has limited range compared to dedicated radio/GPS-based preemption systems

### Option 2: YOLOX-M Detection + ByteTrack + Rule-Based Adaptive (No RL)

- **Approach:** YOLOX-M (25.3M params, Apache 2.0) for vehicle detection, ByteTrack for tracking, and rule-based Webster + actuated control logic for signal timing — no reinforcement learning. Signal timing adapts in real-time based on live queue/demand data but does not learn from or optimize for future states.
- **Addresses:** Real-time perception (proven INT8 quantization on both Jetson and AX650N), hardware constraint (YOLOX-M is self-contained in existing pipeline), infrastructure integration (same NTCIP/SNMP layer), autonomous operation (rule-based controller is inherently stable), tropical environment (works on both target chips)
- **Pros:** Simpler architecture with fewer failure modes; no simulation training required; YOLOX-M has proven INT8 path on AX650N NPU; 15-25% delay reduction vs fixed-time is achievable with rule-based adaptive alone; faster to deploy (no RL training pipeline); lower engineering risk
- **Cons:** Cannot optimize for future consequences (queue spillover prevention, green wave coordination); no multi-intersection learning; limited to 15-25% improvement vs 20-40% with RL; cannot adapt to novel patterns (events, incidents) beyond what rules encode

### Option 3: Commercial Off-the-Shelf (Fallback)

- **Approach:** Deploy a commercial adaptive traffic system such as NoTraffic (camera + radar per approach), Miovision Surtrac (camera + decentralized AI), or LYT (existing camera + edge compute). These are proven, supported solutions with established track records.
- **When to use:** If edge AI development timeline is too aggressive, if BDA requires vendor-backed SLA and support contracts, or if on-site technical capabilities are insufficient for maintaining a custom system

**Decision:** Option 1 selected. D-FINE-S + ByteTrack + PPO hybrid provides the best balance of resource efficiency (open-source, no vendor lock-in), performance (20-40% delay reduction target), and safety (rule-based fallback). Option 2 serves as the Phase 2 deployment before RL is ready (Months 5-8). The phased rollout (data collection -> rule-based -> RL-assisted -> full adaptive) de-risks the transition at each stage.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [AI Models — What to Use and Why](#2-ai-models)
3. [Adaptive Signal Timing Algorithms](#3-adaptive-signal-timing-algorithms)
4. [Signal Controller Integration](#4-signal-controller-integration)
5. [Edge Deployment Optimization](#5-edge-deployment-optimization)
6. [Performance Metrics & Evaluation](#6-performance-metrics--evaluation)
7. [Implementation Phases](#7-implementation-phases)

---

## 1. System Architecture

### 1.1 Full System Diagram

```
                           ┌─────────────────────────────────┐
                           │       CLOUD / BACKEND            │
                           │  ┌─────────────────────────┐    │
                           │  │  Central Management      │    │
                           │  │  Dashboard (Grafana/Web) │    │
                           │  └──────────┬──────────────┘    │
                           │             │                    │
                           │  ┌──────────▼──────────────┐    │
                           │  │  Time-Series DB          │    │
                           │  │  (InfluxDB/TimescaleDB)  │    │
                           │  └──────────┬──────────────┘    │
                           │             │                    │
                           │  ┌──────────▼──────────────┐    │
                           │  │  RL Training Server      │    │
                           │  │  (SUMO/CityFlow + PPO)   │    │
                           │  └──────────┬──────────────┘    │
                           │             │                    │
                           │  ┌──────────▼──────────────┐    │
                           │  │  MQTT Broker             │    │
                           │  │  (EMQX / Mosquitto)      │    │
                           │  └──────────┬──────────────┘    │
                           └─────────────┼───────────────────┘
                                         │ MQTT over TLS
                                         │ (4G/5G / Fiber)
                    ┌────────────────────┼────────────────────┐
                    │                    │                     │
         ┌──────────▼─────────┐ ┌───────▼──────────┐ ┌───────▼──────────┐
         │  INTERSECTION 1    │ │  INTERSECTION 2   │ │  INTERSECTION N   │
         │  ┌──────────────┐  │ │  (same topology)  │ │  (same topology)  │
         │  │ Edge Device   │  │ └──────────────────┘ └──────────────────┘
         │  │ Jetson Orin NX│  │
         │  │               │  │
         │  │ ┌───────────┐ │  │
         │  │ │ Camera x4  │ │  │    4x PoE IP Cameras (1080p/4K)
         │  │ │ (RTSP In) │ │  │    one per approach leg
         │  │ └───────────┘ │  │
         │  │       │       │  │
         │  │ ┌─────▼─────┐ │  │
         │  │ │ Detection  │ │  │    TensorRT INT8 inference
         │  │ │ + Tracking │ │  │    YOLOX-M / D-FINE-S
         │  │ └─────┬─────┘ │  │    ByteTrack MOT
         │  │       │       │  │
         │  │ ┌─────▼─────┐ │  │
         │  │ │ Traffic    │ │  │    Queue length, density,
         │  │ │ Analytics  │ │  │    speed, counts, LOS
         │  │ └─────┬─────┘ │  │
         │  │       │       │  │
         │  │ ┌─────▼─────┐ │  │
         │  │ │ Signal     │ │  │    Rule-based + RL optimizer
         │  │ │ Optimizer  │ │  │    PPO policy inference
         │  │ └─────┬─────┘ │  │
         │  │       │       │  │
         │  └───────┼───────┘  │
         │          │ RS-485 / Ethernet
         │  ┌───────▼───────┐  │
         │  │ Traffic Signal │  │    NTCIP 1202 / vendor API
         │  │ Controller     │  │    Econolite / Siemens / local
         │  └───────────────┘  │
         └─────────────────────┘
```

### 1.2 Data Flow Specification

| Data Path | Format | Rate | Protocol | Size/msg |
|---|---|---|---|---|
| Camera → Edge Device | H.264/H.265 RTSP | 15-30 FPS per stream | RTSP over PoE | ~2-5 Mbps/stream |
| Edge → Signal Controller | Phase commands (SET) | On demand (~1-5/min) | SNMP v2c/v3 (NTCIP 1202) | <1 KB |
| Signal Controller → Edge | Phase state (GET) | 1 Hz polling | SNMP v2c/v3 (NTCIP 1202) | <1 KB |
| Edge → MQTT Broker | Traffic analytics JSON | 1 Hz aggregate | MQTT over TLS | ~2-5 KB |
| Edge → MQTT Broker | Anomaly alerts | Event-driven | MQTT over TLS (QoS 1) | <1 KB |
| Cloud → Edge | Updated RL policy weights | On demand (~1/week) | MQTT + HTTPS | ~5-20 MB |
| Cloud → Edge | Config updates | On demand | MQTT (QoS 2) | <10 KB |

**MQTT Topic Structure:**
```
bintulu/intersections/{intersection_id}/analytics      # 1 Hz traffic state
bintulu/intersections/{intersection_id}/alerts          # anomaly events
bintulu/intersections/{intersection_id}/signal/state    # current phase
bintulu/intersections/{intersection_id}/signal/command  # phase override
bintulu/intersections/{intersection_id}/health          # device health
bintulu/system/policy/update                            # RL model push
bintulu/system/config/update                            # config push
```

**Analytics JSON payload (1 Hz):**
```json
{
  "timestamp": "2026-03-19T14:30:00.000+08:00",
  "intersection_id": "BIN-INT-001",
  "current_phase": 2,
  "phase_elapsed_s": 18.5,
  "approaches": {
    "north": {
      "queue_length_m": 45.2,
      "queue_vehicles": 12,
      "flow_rate_vph": 420,
      "avg_speed_kmh": 8.3,
      "density_veh_per_km": 48.5,
      "los": "D",
      "counts": {"car": 8, "motorcycle": 3, "truck": 1, "bus": 0, "bicycle": 0, "pedestrian": 5}
    },
    "south": { ... },
    "east": { ... },
    "west": { ... }
  },
  "anomalies": [],
  "signal_decision": {
    "next_phase": 3,
    "green_duration_s": 25,
    "reason": "rl_optimizer"
  }
}
```

### 1.3 Edge Device Software Stack

```
┌─────────────────────────────────────────────────┐
│                  Application Layer                │
│  ┌───────────┐ ┌──────────┐ ┌────────────────┐  │
│  │ Detection  │ │ Tracking │ │ Signal Control │  │
│  │ Pipeline   │ │ Engine   │ │ (Rule+RL)      │  │
│  └───────────┘ └──────────┘ └────────────────┘  │
│  ┌───────────┐ ┌──────────┐ ┌────────────────┐  │
│  │ Analytics  │ │ Anomaly  │ │ MQTT Client    │  │
│  │ Engine     │ │ Detector │ │ (paho-mqtt)    │  │
│  └───────────┘ └──────────┘ └────────────────┘  │
├─────────────────────────────────────────────────┤
│                  ML Runtime Layer                 │
│  ┌───────────────────────────────────────────┐   │
│  │ TensorRT 10.x (INT8 inference engine)     │   │
│  └───────────────────────────────────────────┘   │
│  ┌───────────────────────────────────────────┐   │
│  │ ONNX Runtime (fallback / model loading)   │   │
│  └───────────────────────────────────────────┘   │
│  ┌───────────────────────────────────────────┐   │
│  │ Optional: DeepStream 7.x (multi-stream)   │   │
│  └───────────────────────────────────────────┘   │
├─────────────────────────────────────────────────┤
│                  System Layer                     │
│  ┌───────────┐ ┌──────────┐ ┌────────────────┐  │
│  │ GStreamer  │ │ OpenCV   │ │ NumPy/SciPy    │  │
│  │ (decode)   │ │ (vision) │ │ (compute)      │  │
│  └───────────┘ └──────────┘ └────────────────┘  │
│  ┌───────────┐ ┌──────────┐ ┌────────────────┐  │
│  │ PySNMP    │ │ SQLite   │ │ systemd        │  │
│  │ (NTCIP)   │ │ (local)  │ │ (watchdog)     │  │
│  └───────────┘ └──────────┘ └────────────────┘  │
├─────────────────────────────────────────────────┤
│  JetPack 6.x (L4T Ubuntu 22.04, CUDA 12.6)     │
│  Linux kernel 5.15 + NVIDIA GPU driver           │
└─────────────────────────────────────────────────┘
│  NVIDIA Jetson Orin NX 16GB                      │
│  100 TOPS INT8 | 8-core Arm Cortex-A78AE        │
│  1024-core Ampere GPU | 102 GB/s memory BW      │
└─────────────────────────────────────────────────┘
```

### 1.4 Redundancy and Failover Design

| Failure Mode | Mitigation | Recovery Time |
|---|---|---|
| Edge device crash | systemd watchdog auto-restart; signal controller reverts to fixed-time plan | <30 seconds |
| Camera stream loss | Graceful degradation — remaining cameras continue; alert to cloud | Immediate (partial) |
| All cameras down | Controller falls back to pre-programmed actuated plan stored locally | Immediate |
| Network to cloud lost | Edge operates fully autonomously; buffers analytics to SQLite for later sync | Indefinite |
| Signal controller comms lost | Edge stops sending commands; controller runs its own timing plan | Immediate |
| Power failure | UPS in traffic cabinet (30-60 min); controller has battery backup | UPS duration |
| RL policy corruption | Fall back to rule-based control; checksum verification on policy updates | Immediate |

**Critical design principle:** The edge device is an *advisor* to the signal controller, not a replacement. The traffic signal controller always has its own fallback timing plan. If the edge device disappears entirely, traffic reverts to a working fixed-time or actuated plan — never to a dark intersection.

### 1.5 Cloud/Backend Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                    Cloud Backend (VPS/On-Prem)                 │
│                                                               │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐  │
│  │ EMQX Broker  │   │ TimescaleDB  │   │ Grafana          │  │
│  │ (MQTT 5.0)   │──→│ (time-series │──→│ (dashboards,     │  │
│  │              │   │  analytics)  │   │  real-time maps)  │  │
│  └──────┬───────┘   └──────────────┘   └──────────────────┘  │
│         │                                                     │
│  ┌──────▼───────┐   ┌──────────────┐   ┌──────────────────┐  │
│  │ FastAPI      │   │ RL Training  │   │ Alert Service    │  │
│  │ (REST API    │   │ Server       │   │ (Telegram/SMS    │  │
│  │  + WebSocket)│   │ (SUMO + PPO) │   │  notifications)  │  │
│  └──────────────┘   └──────────────┘   └──────────────────┘  │
│                                                               │
│  ┌──────────────┐   ┌──────────────┐                         │
│  │ OTA Update   │   │ Device Mgmt  │                         │
│  │ Service      │   │ (health,     │                         │
│  │ (policy push)│   │  inventory)  │                         │
│  └──────────────┘   └──────────────┘                         │
└───────────────────────────────────────────────────────────────┘
```

**Key cloud functions:**
- **Real-time monitoring:** Live intersection status, phase timing, queue lengths, anomaly alerts on Grafana dashboards with map overlays
- **Historical analytics:** Trend analysis, peak-hour patterns, before/after comparisons, congestion heatmaps
- **RL training:** Periodic retraining of signal timing policy on SUMO simulation calibrated with real data; push updated weights to edge
- **OTA updates:** Firmware, model weights, and config updates via MQTT + HTTPS with rollback capability
- **Device management:** Health monitoring, uptime tracking, remote restart, log collection

---

## 2. AI Models — What to Use and Why

### 2.1 Vehicle Detection & Classification

#### Model Comparison for Intersection Cameras

| Model | mAP@50:95 (COCO) | Latency (T4 GPU) | Params | NMS-Free | License | Edge-Optimized |
|---|---|---|---|---|---|---|
| **YOLOX-M** | 46.9% | 6.5 ms | 25.3M | No | Apache 2.0 | Excellent INT8 |
| **YOLOX-S** | 40.5% | 3.2 ms | 9.0M | No | Apache 2.0 | Excellent INT8 |
| **D-FINE-S** | 48.5% | 8.1 ms | 10M | Yes | Apache 2.0 | Good TensorRT |
| **D-FINE-N** | 42.8% | 5.0 ms | 4M | Yes | Apache 2.0 | Excellent |
| **D-FINE-L** | 54.0% | ~12 ms | 31M | Yes | Apache 2.0 | Good |
| **RT-DETRv2-R18** | 47.9% | 5.4 ms | 20M | Yes | Apache 2.0 | Good TensorRT |
| **RT-DETRv2-R50** | 53.4% | ~10 ms | 42M | Yes | Apache 2.0 | Moderate |
| **RF-DETR-B** | 54.7% | 4.5 ms | ~30M | Yes | Apache 2.0 | Excellent |
| **RF-DETR-L** | 60.5% | ~40 ms | ~130M | Yes | Apache 2.0 | Too heavy |

*Note: YOLOv8/v11/YOLO26 (Ultralytics, AGPL-3.0) are prohibited per license policy.*

#### Why Overhead Intersection Cameras Are Different

Intersection cameras are typically mounted at 6-10 meters height looking down at 30-60 degree angles. This creates:

1. **Top-down perspective distortion** — vehicles appear foreshortened; motorcycle/bicycle detection is harder
2. **Small object density** — many vehicles in frame simultaneously, especially motorcycles in Southeast Asia
3. **Multi-class diversity** — cars, motorcycles (dominant in Malaysia), trucks, buses, bicycles, pedestrians all present
4. **Occlusion at stops** — vehicles queue closely, rear vehicles heavily occluded
5. **Lighting variation** — dawn/dusk glare, headlights at night, tropical rain

#### Recommended Model: D-FINE-S (Primary) + YOLOX-M (Fallback)

**D-FINE-S is the primary recommendation** for traffic intersection deployment:

- **48.5% mAP@50:95** — best accuracy-to-size ratio in its class
- **NMS-free** — eliminates a major latency bottleneck and tuning headache; critical when processing 4 streams simultaneously
- **10M parameters** — small enough for multi-stream on Jetson Orin NX
- **ICLR 2025 Spotlight** — peer-reviewed, well-documented architecture
- **Fine-grained localization** — D-FINE's fine-grained distribution refinement is particularly good for tight bounding boxes on occluded vehicles

**YOLOX-M as fallback** for the AX650N target:
- Self-contained implementation already in our pipeline (`core/models/yolox.py`)
- Proven INT8 quantization path for NPU chips
- CSPDarknet backbone well-suited to INT8 on non-NVIDIA accelerators

**Target classes (6):** `car`, `motorcycle`, `truck`, `bus`, `bicycle`, `pedestrian`

**Training strategy:**
- Pre-train on COCO (80 classes) → fine-tune on traffic-specific dataset
- Recommended datasets: BDD100K (100K driving images, 10 classes), UA-DETRAC (140K frames, vehicle detection), VisDrone (10K drone images — similar overhead angle)
- Fine-tune specifically for Malaysian traffic: high motorcycle density, tropical lighting conditions
- Target: **mAP@50 >= 0.90** for the 6 target classes

#### Multi-Stream Latency Budget on Jetson Orin NX

| Configuration | Model | Precision | Per-Stream FPS | 4-Stream FPS | Feasible? |
|---|---|---|---|---|---|
| Single model, batched | D-FINE-S | INT8 | ~40 | ~10 per stream | Yes |
| Single model, batched | YOLOX-M | INT8 | ~55 | ~14 per stream | Yes |
| Per-stream model | D-FINE-S | INT8 | ~40 | ~10 per stream | Yes (more memory) |
| DeepStream pipeline | D-FINE-S | INT8 | N/A | ~12 per stream | Yes (optimal) |

At 10+ FPS per stream across 4 cameras, detection is sufficient for traffic analytics (vehicles don't move fast at intersections).

### 2.2 Vehicle Tracking & Counting

#### Tracker Comparison for Traffic Scenarios

| Tracker | MOTA | IDF1 | FPS | Re-ID | Best For |
|---|---|---|---|---|---|
| **ByteTrack** | 77.3% | 77.8% | 171 | No | Real-time, low compute, fixed cameras |
| **BoT-SORT** | 80.5% | 80.2% | ~45 | Yes | Higher accuracy, more compute |
| **OC-SORT** | 78.0% | 77.5% | ~120 | No | Occlusion handling, gap recovery |
| **Deep OC-SORT** | 79.2% | 79.5% | ~60 | Yes | Best occlusion, highest compute |
| **StrongSORT** | 79.6% | 79.5% | ~30 | Yes | Highest accuracy, heavy Re-ID |

#### Recommendation: ByteTrack (Primary)

**ByteTrack is the clear winner for traffic intersections:**

1. **171 FPS** — negligible overhead; can track across all 4 streams without bottleneck
2. **No Re-ID model needed** — saves GPU memory for detection; Re-ID is unnecessary at fixed-camera intersections where vehicles don't leave and re-enter the frame
3. **Low-confidence rescue** — ByteTrack's signature feature: it associates low-confidence detections (motorcycles partially occluded) that other trackers would drop
4. **Proven in traffic** — 98% vehicle counting accuracy in validation studies; 77.3% MOTA on MOT17
5. **MIT license** — fully compatible
6. **Already integrated** in our pipeline (`utils/supervision_bridge.py`)

**Use `supervision` library for tracker integration:**
```python
import supervision as sv
tracker = sv.ByteTrack(
    track_activation_threshold=0.25,   # low threshold for motorcycle detection
    lost_track_buffer=30,              # keep track alive for 1s at 30fps
    minimum_matching_threshold=0.8,    # IoU threshold for association
    frame_rate=15                      # match camera FPS
)
```

#### Direction-Aware Vehicle Counting (Virtual Line Crossing)

```
Camera View (one approach):
┌──────────────────────────────┐
│                              │
│    ↓  ↓  ↓                   │   Incoming traffic
│    ↓  ↓  ↓                   │
│ ═══════════════  ← COUNT LINE│   Virtual counting line
│    ↓  ↓  ↓                   │
│         STOP LINE            │
│ ┌──┐ ┌──┐ ┌──┐              │   Queued vehicles
│ │  │ │  │ │  │              │
└──────────────────────────────┘
```

**Algorithm:**
1. Track each vehicle across frames using ByteTrack → get track_id + centroid trajectory
2. Define a virtual counting line (configured per camera via calibration)
3. For each track, record centroid position at frame N and N-1
4. If centroid crosses the line between frames: increment counter, record direction (incoming vs outgoing) based on which side the centroid came from
5. Store per-class counts: `{car: 42, motorcycle: 87, truck: 5, bus: 2, bicycle: 3}`
6. Aggregate into flow rate (vehicles per hour) using a sliding window

**Direction detection:** Compare the y-coordinate (or projected world coordinate) of the track centroid before and after crossing. Movement toward the intersection = incoming; away = outgoing.

#### Queue Length Estimation

**Method: Stopped vehicle detection via tracking**

1. For each tracked vehicle, compute instantaneous speed: `speed = |centroid(t) - centroid(t-1)| * fps * px_to_m_scale`
2. Vehicle is "stopped" if speed < 2 km/h for > 2 seconds
3. Queue length = distance from stop line to the farthest stopped vehicle in the approach lane
4. Queue vehicle count = number of stopped vehicles in the approach lane

**Lane assignment:** Use a pre-configured polygon mask per lane (set during camera calibration). Each detected vehicle is assigned to a lane based on centroid position.

**Queue state classification:**
- **Growing:** Queue length increasing over 10s window
- **Stable:** Queue length constant (+/- 5%)
- **Discharging:** Queue length decreasing (green phase serving this approach)
- **Cleared:** No stopped vehicles

#### Vehicle Speed Estimation (Pixel-to-World Calibration)

**Homography-based approach:**

1. **One-time calibration:** During installation, mark 4+ ground-truth points with known real-world coordinates (GPS or tape measure). Common approach: paint marks at known distances on the road surface.
2. **Compute homography matrix H:** Maps pixel coordinates to bird's-eye-view (BEV) ground plane coordinates using `cv2.findHomography(src_pts, dst_pts)`
3. **Transform tracked centroids to BEV:** `world_pt = H @ pixel_pt` (homogeneous coordinates)
4. **Compute speed:** `speed = euclidean_distance(world_pt_t, world_pt_{t-1}) * fps` (in m/s)
5. **Accuracy:** Typically 89-99% accuracy within 50m of the camera, degrading with distance

**Alternative (no manual calibration):** Use the known dimensions of standard vehicles (car ~4.5m, truck ~12m) detected in frame to auto-calibrate the scale factor. Less accurate but eliminates manual setup.

#### Multi-Lane Tracking Challenges

| Challenge | Solution |
|---|---|
| Lane changes mid-intersection | ByteTrack handles smoothly via IoU association; re-assign lane based on new centroid position |
| Motorcycle lane splitting | Southeast Asia specific — motorcycles weave between cars; lower IoU threshold (0.3) for motorcycle tracks |
| U-turns | Track direction reversal → count as both outgoing on one approach and incoming on the opposite |
| Heavy occlusion at stops | ByteTrack low-confidence rescue; D-FINE's NMS-free design avoids suppressing overlapping vehicles |
| Nighttime headlight glare | IR-cut filter cameras + training data augmentation with glare/bloom effects |

### 2.3 Traffic Density & Flow Estimation

#### Detection-Based Approach (Recommended)

Count detected vehicles in a defined region of interest (ROI) per lane, convert to density:

```
density (veh/km) = vehicle_count_in_ROI / ROI_length_km
flow_rate (veh/h) = vehicles_crossing_count_line / time_window_h
avg_speed (km/h) = mean(tracked_vehicle_speeds)
```

**Advantages:** Uses existing detection + tracking pipeline. No additional model needed. Per-class density available.

#### Regression-Based Approach (Alternative for Extreme Congestion)

When traffic is so dense that individual vehicles cannot be detected (gridlock, motorcycle swarms):

- Train a regression CNN (e.g., CSRNet, MCNN) on traffic density maps
- Input: camera frame → Output: density map (heatmap) where integral = vehicle count
- Useful as a fallback when detection mAP drops in extreme congestion

**Recommendation:** Start with detection-based; add regression fallback only if extreme congestion scenarios (unlikely in Bintulu) demand it.

#### Level of Service (LOS) Classification

LOS is the standard traffic engineering metric (HCM — Highway Capacity Manual):

| LOS | Delay/Vehicle (s) | V/C Ratio | Description |
|---|---|---|---|
| **A** | <= 10 | 0.00 - 0.60 | Free flow, no delay |
| **B** | 10 - 20 | 0.60 - 0.70 | Stable flow, slight delay |
| **C** | 20 - 35 | 0.70 - 0.80 | Stable flow, acceptable delay |
| **D** | 35 - 55 | 0.80 - 0.90 | Approaching unstable, tolerable delay |
| **E** | 55 - 80 | 0.90 - 1.00 | Unstable flow, significant delay |
| **F** | > 80 | > 1.00 | Forced flow, excessive delay |

**Estimation method:**
1. Compute average delay per vehicle: `delay = time_in_queue / vehicle_count` (measured via tracking — time between stop and crossing the stop line)
2. Compute V/C ratio: `volume / capacity` where capacity is the theoretical maximum throughput (from signal timing and lane count)
3. Classify LOS based on both metrics
4. Report per-approach and overall intersection LOS at 1 Hz

#### Industry Approaches

| Company | Technology | Approach |
|---|---|---|
| **NoTraffic** | Proprietary camera + radar sensor per approach; on-device ML | Detection-based density; runs millions of scenarios in real-time for signal optimization |
| **Miovision (Surtrac)** | Camera sensors at intersection; decentralized AI | Treats intersection as single-machine scheduling problem; optimizes every second; shares info with neighbor intersections |
| **Vivacity Labs** | AI-powered computer vision sensor; >6000 deployed | 97%+ accuracy across 12+ road user classifications; zone occupancy, speed, dwell times, turning counts |
| **LYT (formerly IntelliLight)** | Existing cameras + edge compute | Cloud-first, retrofits to existing infrastructure; uses NTCIP 1202 integration |

### 2.4 Anomaly & Incident Detection

All anomaly detection runs on top of the existing detection + tracking pipeline — no additional models needed.

#### Stalled Vehicle Detection

```python
# Pseudocode
for track in active_tracks:
    if track.speed < STALL_SPEED_THRESHOLD:  # < 2 km/h
        track.stall_timer += dt
    else:
        track.stall_timer = 0

    if track.stall_timer > STALL_TIME_THRESHOLD:  # > 120 seconds
        if track.position not in LEGAL_STOP_ZONES:  # not at stop line during red
            trigger_alert("stalled_vehicle", track)
```

**Thresholds:** Speed < 2 km/h for > 120 seconds, not in a legal stopping zone. Adjust based on red phase duration to avoid false positives during long reds.

#### Wrong-Way Detection

1. Define expected travel direction per lane (configured during setup)
2. Compute track heading: `heading = atan2(dy, dx)` from centroid displacement over 1-second window
3. If heading differs from expected direction by > 135 degrees for > 3 seconds → wrong-way alert
4. **Critical for safety** — can trigger immediate signal hold (all-red) to protect the wrong-way vehicle

#### Pedestrian in Vehicle Lane

1. Detect pedestrians (class `pedestrian` from the detection model)
2. Check if pedestrian centroid falls within vehicle lane polygon (not crosswalk zone)
3. If pedestrian is in vehicle lane for > 5 seconds → alert
4. Can extend pedestrian crossing phase if pedestrian detected on crosswalk during phase change

#### Red-Light Running

1. Know current signal phase (from NTCIP polling)
2. If a vehicle crosses the stop line during red phase for that approach → red-light event
3. Log with timestamp, track_id, vehicle class, frame capture
4. Useful for enforcement data (not real-time control, but valuable analytics)

### 2.5 Emergency Vehicle Detection

#### Visual Detection (Primary)

**Add emergency vehicle classes to detection model:**
- Extend the 6-class model with 2 additional classes: `ambulance`, `fire_truck`
- Malaysian emergency vehicles have distinctive visual features: red/blue light bars, specific livery
- Training data: Collect local Bintulu/Sarawak emergency vehicle images; augment with COCO/OpenImages emergency vehicle images

**Detection trigger:**
1. Detect `ambulance` or `fire_truck` class with confidence > 0.7
2. Confirm over 3 consecutive frames (debounce)
3. Determine approach direction from tracking
4. Trigger signal preemption for that approach

#### Siren Audio Detection (Secondary)

**Approach:** Deploy a small audio classification model alongside the vision pipeline:
- **Model:** Audio Spectrogram Transformer (AST) or YAMNet (TF-Lite, 3.7MB)
- **Input:** Mel spectrogram from microphone at each approach
- **Classes:** `siren`, `horn`, `ambient`
- **Advantage:** Can detect emergency vehicles before they are visible (around corners)
- **Challenge:** Urban noise, motorcycle engines, construction — high false positive risk
- **Hardware:** USB microphone per approach

**Recommendation:** Visual detection as primary; audio as optional secondary signal for earlier preemption. Fuse both: `confidence = max(visual_conf, audio_conf * 0.7)` (discount audio due to higher FP rate).

#### Signal Preemption Protocol

```
1. Emergency vehicle detected on approach X
2. Edge device sends PREEMPT command to signal controller
3. Controller executes preemption sequence:
   a. Current phase → transition to all-red (3-5s clearance)
   b. All-red → green for approach X
   c. Hold green until emergency vehicle clears intersection
   d. Emergency vehicle exits → restore normal operation
4. Total preemption latency budget: < 15 seconds
   - Detection: < 500 ms
   - Confirmation (3 frames): ~200 ms
   - Command transmission: < 100 ms
   - Controller transition: 3-5 seconds (all-red clearance)
   - Green activation: immediate after clearance
```

**NTCIP preemption objects:**
- `preemptControl` (OID 1.3.6.1.4.1.1206.4.2.1.6) — trigger preemption
- `preemptState` — read current preemption status
- Priority: Emergency preemption overrides all RL/adaptive timing decisions

---

## 3. Adaptive Signal Timing Algorithms

### 3.1 Rule-Based Approaches

#### Webster's Formula for Optimal Cycle Length

The foundational traffic signal timing formula (Webster, 1958):

```
Co = (1.5L + 5) / (1 - Y)

Where:
  Co = Optimal cycle length (seconds)
  L  = Total lost time per cycle (seconds) = Σ(start-up lost time + clearance lost time) per phase
  Y  = Sum of critical flow ratios = Σ(qi / si) for each phase
       qi = critical lane flow rate (veh/s) for phase i
       si = saturation flow rate (veh/s) for phase i
```

**Example for a 4-phase intersection:**
- L = 4 phases x (3s start-up + 2s clearance) = 20s
- Y = 0.15 + 0.20 + 0.10 + 0.12 = 0.57
- Co = (1.5 × 20 + 5) / (1 - 0.57) = 35 / 0.43 = **81 seconds**

**Practical constraints:**
- Minimum cycle: 40 seconds (pedestrian crossing time)
- Maximum cycle: 120-150 seconds (driver patience, coordination)
- Webster overestimates when V/C > 0.5; use modified formulas for high-saturation scenarios

#### Green Time Allocation

```
Effective green for phase i:  gi = (Co - L) × (yi / Y)

Where:
  gi = effective green time for phase i
  yi = critical flow ratio for phase i
  Y  = sum of all critical flow ratios
```

This distributes green time proportional to demand — the phase with the most traffic gets the most green.

#### Min/Max Green Constraints

| Constraint | Typical Value | Reason |
|---|---|---|
| Minimum green | 7-15 seconds | Pedestrian safety; driver reaction time |
| Maximum green | 40-60 seconds | Prevent excessive delay on other approaches |
| Minimum pedestrian green | Walk + clearance = 7 + (crossing_distance / 1.2 m/s) | ADA/accessibility compliance |
| All-red clearance | 2-5 seconds | Clear intersection before conflicting phase |
| Yellow change interval | 3-5 seconds | `t = reaction_time + v/(2*deceleration)` |
| Minimum cycle | 40 seconds | Accommodate all phases with minimum greens |
| Maximum cycle | 120-150 seconds | Driver patience, coordination with adjacent intersections |

#### Actuated vs Semi-Actuated Control

| Mode | Description | When to Use |
|---|---|---|
| **Pre-timed** | Fixed cycle, fixed green splits | Baseline fallback; predictable traffic |
| **Semi-actuated** | Minor street gets green only on demand; major street gets rest of time | Main road with low-volume side streets |
| **Fully actuated** | All phases demand-responsive; cycle length varies | Variable demand on all approaches |
| **Adaptive (our target)** | Real-time optimization using live traffic data | Peak hours, incident response |

**Our system implements fully actuated control as the baseline**, with adaptive optimization layered on top.

#### Rule-Based Adaptive Algorithm

```python
def compute_green_splits(approach_data, config):
    """Rule-based green time allocation from real-time detection data."""
    total_demand = sum(a.flow_rate for a in approach_data)
    if total_demand == 0:
        return config.default_splits

    # Webster-inspired proportional allocation
    splits = {}
    for phase in config.phases:
        critical_approach = phase.critical_approach
        demand_ratio = critical_approach.flow_rate / critical_approach.saturation_flow
        splits[phase.id] = demand_ratio

    Y = sum(splits.values())
    L = config.total_lost_time
    cycle_length = clamp((1.5 * L + 5) / (1 - Y), config.min_cycle, config.max_cycle)

    for phase_id, ratio in splits.items():
        green = (cycle_length - L) * (ratio / Y)
        green = clamp(green, config.phases[phase_id].min_green,
                             config.phases[phase_id].max_green)
        splits[phase_id] = green

    return cycle_length, splits
```

**Performance vs fixed-time:** Rule-based adaptive typically achieves **15-25% reduction in average delay** compared to fixed-time plans (Miovision/Surtrac reports 25% travel time reduction, 40% waiting time reduction).

### 3.2 Reinforcement Learning Approaches

#### Why RL for Traffic Signals?

Rule-based methods optimize for the *current* moment. RL optimizes for the *expected future* — it learns that giving a slightly longer green now may prevent a queue buildup that would cause more delay later. This is particularly valuable for:
- Coordinating adjacent intersections (green wave)
- Handling non-stationary demand patterns (peak hours, events)
- Adapting to incidents without manual intervention

#### RL Formulation as Markov Decision Process (MDP)

**State Space (what the RL agent observes):**

```python
state = {
    # Per-approach (4 approaches x features):
    "queue_length": [12, 5, 8, 3],          # vehicles in queue
    "queue_density": [0.45, 0.18, 0.32, 0.12],  # normalized
    "avg_speed": [8.3, 28.5, 12.1, 35.2],   # km/h
    "flow_rate": [420, 180, 350, 120],       # veh/h
    "waiting_time": [45.2, 12.1, 32.5, 5.8], # seconds avg

    # Signal state:
    "current_phase": 2,                       # one-hot encoded
    "phase_elapsed": 18.5,                    # seconds
    "time_of_day": 0.65,                      # normalized 0-1

    # Optional neighborhood (for multi-intersection):
    "neighbor_queues": [[8, 4], [6, 10]],     # upstream/downstream
}
# Total state dimension: ~30-50 features
```

**Action Space:**

| Design | Description | Pros | Cons |
|---|---|---|---|
| **Phase selection** | Choose next phase from {1,2,3,4} | Simple; standard in literature | Fixed green duration |
| **Phase + duration** | Choose phase and green time (discretized: 10/15/20/25/30s) | More control | Larger action space |
| **Keep or switch** | Binary: extend current phase by 5s or advance to next | Smallest action space; most stable training | Less flexible |
| **Continuous duration** | Output green time as continuous value [min, max] | Most flexible | Harder to train; PPO better than DQN |

**Recommended: Phase selection with fixed 5-second decision interval.** Every 5 seconds, the agent decides: keep current phase or switch to phase X. This is the approach used by PressLight and CoLight.

**Reward Function:**

```python
def compute_reward(state, next_state):
    """Multi-objective reward balancing delay, throughput, and fairness."""
    # Primary: minimize total intersection delay
    delay_reduction = state.total_waiting_time - next_state.total_waiting_time
    delay_reward = delay_reduction / max(state.total_waiting_time, 1.0)

    # Secondary: maximize throughput
    throughput_reward = next_state.vehicles_served / next_state.vehicles_total

    # Fairness: penalize max-approach-delay to prevent starvation
    max_delay_penalty = -max(next_state.per_approach_waiting_time) / 120.0

    # Safety: heavy penalty for exceeding max green or min pedestrian time
    safety_penalty = -10.0 if violates_constraints(next_state) else 0.0

    reward = (0.5 * delay_reward +
              0.3 * throughput_reward +
              0.15 * max_delay_penalty +
              0.05 * safety_penalty)
    return reward
```

#### Algorithm Comparison

| Algorithm | Type | Performance | Stability | Multi-Agent | Recommended |
|---|---|---|---|---|---|
| **DQN** | Value-based | Good | Moderate | With IDQN | For single intersection prototyping |
| **PPO** | Policy-gradient | Very Good | High | Yes (MAPPO) | **Yes — primary recommendation** |
| **A2C** | Actor-critic | Good | Low-moderate | Yes (MA2C) | Simpler but less stable than PPO |
| **SAC** | Off-policy | Very Good | High | Limited | For continuous action spaces |
| **Rainbow DQN** | Enhanced DQN | Excellent | High | Limited | For single intersection, high performance |

**Recommendation: PPO (Proximal Policy Optimization)**
- Most stable training in practice (clipped objective prevents catastrophic policy updates)
- Works well with discrete action spaces (phase selection)
- Scales to multi-agent (MAPPO for multi-intersection coordination)
- Well-supported in Stable-Baselines3 and RLlib

#### Key Papers and Their Contributions

| Paper | Year | Key Contribution | Approach |
|---|---|---|---|
| **IntelliLight** | 2018 | First practical RL traffic control; defined state/action/reward design | DQN with phase-duration action |
| **PressLight** | 2019 | Uses "pressure" (upstream - downstream queue) as reward; theoretically grounded in max-pressure control | DQN with pressure reward |
| **CoLight** | 2019 | Graph attention network for multi-intersection coordination; agents communicate via attention mechanism | Graph Attention + RL |
| **MPLight** | 2020 | Decentralized RL with parameter sharing; scales to 2500+ intersections | FRAP + parameter sharing |
| **GeneraLight** | 2020 | Transfer learning across intersections with different topologies | Meta-learning + RL |
| **Advanced-MP** | 2022 | Considers both running and queuing vehicles; state-of-the-art on CityFlow benchmarks | Enhanced max-pressure + RL |
| **LLMLight** | 2025 | LLM agents for traffic signal control; KDD 2025; won Geneva Invention Award 2025 | GPT-4 as signal controller |
| **FGLight** | 2025 | Fine-grained neighbor information for multi-agent coordination | AAMAS 2025 |

#### Training in Simulation

**SUMO (Simulation of Urban Mobility):**
- Industry standard open-source traffic microsimulator
- Simulates individual vehicle behavior with configurable driver models
- Python API (TraCI) for RL integration
- SUMO-RL library provides Gymnasium-compatible environment
- Slower but more realistic physics and traffic models

**CityFlow:**
- Purpose-built for RL experiments
- **20-25x faster than SUMO** — critical for RL training which requires millions of steps
- Open-source, Python API, PettingZoo multi-agent support
- Less detailed vehicle physics but sufficient for signal timing learning

**Recommended training pipeline:**
1. **Pre-train on CityFlow** (fast, millions of episodes)
2. **Fine-tune on SUMO** (calibrated with real Bintulu traffic data)
3. **Deploy to edge** (policy inference only, ~1ms per decision)

**Sim-to-Real Transfer Challenges:**

| Challenge | Mitigation |
|---|---|
| Simulated traffic doesn't match real demand patterns | Calibrate simulation with real traffic counts from initial deployment (Phase 1 data collection) |
| Driver behavior differs (e.g., motorcycle lane-splitting) | Add Malaysia-specific driver behavior model in SUMO |
| Sensor noise (detection misses, tracking errors) | Add noise injection to simulation state during training |
| Intersection geometry mismatch | Model actual Bintulu intersection geometry in simulator |
| Demand variability (events, weather, holidays) | Train with domain randomization — vary demand +/- 30% |

**Grounded Action Transformation (GAT):** Recent approach (2025) that trains a transformation model to map simulated actions to real-world-appropriate actions, significantly reducing the sim-to-real gap. JL-GAT extends this by integrating neighboring agent information.

### 3.3 Hybrid Approaches (Recommended)

#### Architecture: Rule-Based Baseline + RL Optimizer

```
┌─────────────────────────────────────────────────┐
│              Signal Timing Decision               │
│                                                   │
│  ┌──────────────┐     ┌──────────────────────┐   │
│  │ Rule-Based   │     │ RL Optimizer          │   │
│  │ Controller   │     │ (PPO policy)          │   │
│  │              │     │                       │   │
│  │ Webster +    │     │ State → Action        │   │
│  │ Actuated     │     │ (next phase +         │   │
│  │ Logic        │     │  green duration adj)  │   │
│  └──────┬───────┘     └──────────┬────────────┘   │
│         │                        │                 │
│         ▼                        ▼                 │
│  ┌──────────────────────────────────────────┐     │
│  │          Safety Validator                 │     │
│  │  • Min/max green enforcement              │     │
│  │  • All-red clearance guaranteed           │     │
│  │  • Pedestrian minimum crossing time       │     │
│  │  • Emergency preemption override          │     │
│  │  • Conflict monitor (no conflicting       │     │
│  │    greens ever)                            │     │
│  └──────────────────────────────────────────┘     │
│                      │                             │
│                      ▼                             │
│              Final Phase Command                   │
└─────────────────────────────────────────────────┘
```

**How it works:**

1. **Rule-based controller** runs continuously and always has a valid timing plan (Webster + actuated logic)
2. **RL optimizer** suggests adjustments: "extend phase 2 green by 8 seconds" or "switch to phase 3 now"
3. **Safety validator** checks every command against hard constraints before execution:
   - Minimum green time not violated
   - All-red clearance interval preserved
   - No conflicting phases active simultaneously
   - Pedestrian crossing time satisfied
4. If RL suggestion violates any constraint → fall back to rule-based output
5. If RL policy is unavailable (loading, crashed, not yet trained) → pure rule-based control

**This is how NoTraffic and Surtrac operate in practice** — they always have safety constraints that cannot be overridden by AI, and they always have a fallback timing plan.

#### Deployment Strategy

| Phase | Signal Control Mode | Duration |
|---|---|---|
| Phase 1: Data Collection | Fixed-time (existing) + AI analytics only (no control) | 2-3 months |
| Phase 2: Rule-Based Adaptive | Webster + actuated control using live detection data | 3-6 months |
| Phase 3: RL-Assisted | RL optimizer active, rule-based as fallback | 6-12 months |
| Phase 4: Full Adaptive | RL primary with learned multi-intersection coordination | 12+ months |

---

## 4. Signal Controller Integration

### 4.1 NTCIP 1202 Protocol Deep Dive

NTCIP 1202 (National Transportation Communications for ITS Protocol) is the North American standard for traffic signal controller communication. Version 4 (2024) is current. It defines SNMP MIB objects for reading and controlling signal state.

#### Key SNMP OIDs for Signal Control

| Object Group | OID Base | Description |
|---|---|---|
| **Phase Status** | 1.3.6.1.4.1.1206.4.2.1.1.4 | Current state of each phase (green/yellow/red) |
| **Phase Control** | 1.3.6.1.4.1.1206.4.2.1.1.5 | Force phase on/off, omit, hold |
| `phaseStatusGroupGreens` | .1.3.6.1.4.1.1206.4.2.1.1.4.1.4 | Bitmask of phases currently green |
| `phaseStatusGroupYellows` | .1.3.6.1.4.1.1206.4.2.1.1.4.1.5 | Bitmask of phases in yellow |
| `phaseStatusGroupReds` | .1.3.6.1.4.1.1206.4.2.1.1.4.1.6 | Bitmask of phases currently red |
| **Phase Timing** | 1.3.6.1.4.1.1206.4.2.1.1.2 | Min/max green, yellow, red clearance per phase |
| `phaseMinimumGreen` | .1.3.6.1.4.1.1206.4.2.1.1.2.1.4 | Minimum green time (tenths of second) |
| `phaseMaximumGreen` | .1.3.6.1.4.1.1206.4.2.1.1.2.1.8 | Maximum green time (tenths of second) |
| `phaseYellowChange` | .1.3.6.1.4.1.1206.4.2.1.1.2.1.9 | Yellow change interval |
| `phaseRedClear` | .1.3.6.1.4.1.1206.4.2.1.1.2.1.10 | Red clearance interval |
| **Detector** | 1.3.6.1.4.1.1206.4.2.1.2 | Vehicle/pedestrian detector status and config |
| **Preempt** | 1.3.6.1.4.1.1206.4.2.1.6 | Emergency vehicle preemption control |
| `preemptControl` | .1.3.6.1.4.1.1206.4.2.1.6.1.1 | Trigger/release preemption |
| **Unit** | 1.3.6.1.4.1.1206.4.2.1.3 | Controller unit status, mode, flash |
| **Coordination** | 1.3.6.1.4.1.1206.4.2.1.7 | Coordination plan, offset, cycle |

#### Communication Pattern (PySNMP)

```python
from pysnmp.hlapi import *

CONTROLLER_IP = "192.168.1.100"
COMMUNITY_READ = "public"
COMMUNITY_WRITE = "private"

# READ current phase status (which phases are green)
def get_phase_status():
    oid = ObjectIdentity('1.3.6.1.4.1.1206.4.2.1.1.4.1.4.1')
    error_indication, error_status, error_index, var_binds = next(
        getCmd(SnmpEngine(),
               CommunityData(COMMUNITY_READ),
               UdpTransportTarget((CONTROLLER_IP, 161)),
               ContextData(),
               ObjectType(oid))
    )
    if error_indication:
        raise ConnectionError(f"SNMP error: {error_indication}")
    green_bitmask = int(var_binds[0][1])
    return green_bitmask  # e.g., 0b00000100 = phase 3 is green

# WRITE: place call on a phase (request green)
def place_phase_call(phase_num):
    """Place a vehicle call on a phase, requesting green."""
    oid = ObjectIdentity('1.3.6.1.4.1.1206.4.2.1.1.5.1.2.1')
    # Set VehicleCallPhase bitmask
    bitmask = 1 << (phase_num - 1)
    error_indication, error_status, error_index, var_binds = next(
        setCmd(SnmpEngine(),
               CommunityData(COMMUNITY_WRITE),
               UdpTransportTarget((CONTROLLER_IP, 161)),
               ContextData(),
               ObjectType(oid, Integer32(bitmask)))
    )
    if error_indication:
        raise ConnectionError(f"SNMP SET error: {error_indication}")

# WRITE: trigger emergency preemption
def trigger_preemption(preempt_num=1):
    oid = ObjectIdentity(f'1.3.6.1.4.1.1206.4.2.1.6.1.1.{preempt_num}')
    next(setCmd(SnmpEngine(),
                CommunityData(COMMUNITY_WRITE),
                UdpTransportTarget((CONTROLLER_IP, 161)),
                ContextData(),
                ObjectType(oid, Integer32(1))))  # 1 = activate
```

### 4.2 Alternative Protocols

| Protocol | Region | Transport | Notes |
|---|---|---|---|
| **NTCIP 1202** (SNMP) | North America, adopted in SE Asia | UDP/IP port 161 | Standard choice for new deployments |
| **UTMC** (Urban Traffic Management & Control) | UK, some Commonwealth | TCP/IP | XML-based, more complex |
| **OCIT-O** | Germany, Europe | TCP/IP | Open Communication Interface for Traffic |
| **Vendor-specific serial** | Legacy controllers | RS-232/RS-485 | Older Siemens, Peek, etc. — may need serial-to-IP bridge |
| **MQTT (custom)** | Modern IoT systems | TCP/IP port 1883 | If controller supports IoT protocols directly |
| **REST API** | Newest controllers (Econolite EOS, LYT) | HTTP/HTTPS | Easiest to integrate; not universally available |

### 4.3 Failsafe Mechanisms

**Hardware conflict monitor:** Every NTCIP-compliant signal controller has a hardware-level conflict monitor (separate microcontroller) that will force the intersection into **flash mode** (all-red flashing) if conflicting greens are ever detected. This is independent of any software and cannot be overridden. Our AI system operates *within* the controller's coordination framework, never bypassing the conflict monitor.

**Software failsafes in edge device:**

```python
class SignalSafetyValidator:
    def validate_command(self, command, current_state):
        """Returns True only if command is safe to execute."""
        # 1. No conflicting greens
        if has_conflict(command.green_phases, self.conflict_matrix):
            return False

        # 2. Minimum green respected
        if current_state.phase_elapsed < self.min_green[current_state.phase]:
            if command.action == "switch":
                return False

        # 3. Yellow/all-red transition required
        if command.action == "switch" and not command.includes_transition:
            return False

        # 4. Maximum green not exceeded
        if current_state.phase_elapsed > self.max_green[current_state.phase]:
            command.action = "switch"  # force advance

        # 5. Pedestrian minimum crossing time
        if ped_phase_active and ped_elapsed < self.min_ped_crossing:
            return False

        return True
```

### 4.4 Controller Hardware in Southeast Asia

Malaysia and Southeast Asia commonly use:

| Manufacturer | Model | Protocol Support | Market Presence |
|---|---|---|---|
| **Siemens/Yunex Traffic** | Various | NTCIP, UTMC, proprietary | Strong in Malaysia (legacy installations) |
| **Swarco** | ITC-2 | NTCIP, OCIT | Growing in SE Asia |
| **Econolite** | Cobalt (ATC) | Full NTCIP 1202 v04 | Used where NEMA/ATC standard adopted |
| **Dynamic Traffic Systems** | Various | Varies | **Malaysian manufacturer** |
| **Sena Traffic Systems** | Various | Varies | **Malaysian manufacturer** |
| **Hikvision/Dahua** | Integrated camera+controller | Proprietary API | Growing in China-linked projects |
| **Pascal** | Various | Varies | **Malaysian traffic solutions provider** |

**Malaysia context:** JKR (Public Works Dept) standard SPJ 2013 covers traffic signal specifications. DBKL operates the Kuala Lumpur Command and Control Centre (KLCCC) for centralized traffic management. Bintulu, being under BDA jurisdiction, may have different standards — assess existing controller hardware during site survey.

**Recommendation:** If procuring new controllers, specify Econolite Cobalt or Yunex Traffic with full NTCIP 1202 v04 support. If integrating with existing controllers, conduct protocol audit during Phase 1.

---

## 5. Edge Deployment Optimization

### 5.1 ONNX to TensorRT Optimization on Jetson

**Conversion pipeline:**

```bash
# 1. Export from PyTorch to ONNX (using our existing pipeline)
python core/export/export.py \
    --model runs/traffic/best.pt \
    --training-config configs/traffic/06_training.yaml \
    --export-config configs/_shared/09_export.yaml

# 2. Convert ONNX to TensorRT on the Jetson device
trtexec \
    --onnx=traffic_dfine_s.onnx \
    --saveEngine=traffic_dfine_s_int8.engine \
    --int8 \
    --calib=calibration_cache.bin \
    --workspace=4096 \
    --maxBatch=4 \
    --minShapes=input:1x3x640x640 \
    --optShapes=input:4x3x640x640 \
    --maxShapes=input:4x3x640x640

# 3. Validate accuracy
python core/export/benchmark.py \
    --engine traffic_dfine_s_int8.engine \
    --dataset dataset_store/traffic/val \
    --compare-onnx traffic_dfine_s.onnx
```

**Key TensorRT flags:**
- `--int8` — INT8 quantization for maximum throughput
- `--calib` — Calibration cache from representative dataset (500-1000 images from Bintulu intersections)
- `--maxBatch=4` — Enable batching of 4 camera streams
- Dynamic shapes — support variable batch size for multi-stream

### 5.2 INT8 Quantization Strategy

**Post-Training Quantization (PTQ):** Primary approach. Collect 500-1000 representative frames from actual Bintulu intersection cameras. Run calibration to determine optimal quantization ranges.

**Quantization-Aware Training (QAT):** If PTQ accuracy drops > 1.5% mAP, fine-tune with QAT for the last 10 epochs.

| Model | FP32 mAP | INT8 PTQ mAP | INT8 QAT mAP | Speedup |
|---|---|---|---|---|
| D-FINE-S (expected) | 48.5% | ~47.0% | ~48.0% | 2-3x |
| YOLOX-M (expected) | 46.9% | ~45.5% | ~46.5% | 2.5-3.5x |

**For AX650N target:** Use the Axera Pulsar2 toolchain (ONNX → AX650N NPU model). The AX650N delivers 18 TOPS INT8 with excellent efficiency for CNN models (YOLOX). Transformer models (D-FINE) may have lower utilization on the AX650N NPU — test during Phase 1.

### 5.3 Multi-Stream Inference (4 Cameras on 1 Device)

**Option A: DeepStream SDK (Recommended for Jetson)**

```
┌────────────────────────────────────────────────────┐
│                  DeepStream Pipeline                │
│                                                    │
│  RTSP Cam 1 ──→ ┌──────────┐                      │
│  RTSP Cam 2 ──→ │ nvstreammux│ → nvinfer → tracker │
│  RTSP Cam 3 ──→ │ (batched  │   (TensorRT)  │     │
│  RTSP Cam 4 ──→ │  decode)  │                │     │
│                 └──────────┘           ┌─────▼───┐ │
│                                        │ Analytics│ │
│                                        │ Probe    │ │
│                                        └─────────┘ │
└────────────────────────────────────────────────────┘
```

**DeepStream advantages:**
- Hardware-accelerated H.264/H.265 decode (NVDEC) — zero GPU overhead for video decode
- Automatic batching of multiple streams for inference
- Built-in tracker (NvDCF, DeepSORT, IOU) — though ByteTrack via custom plugin preferred
- GStreamer-based — composable and configurable
- Supports 16-18 simultaneous streams on Orin NX 16GB

**Option B: Custom ONNX Runtime Pipeline**

```python
# Multi-stream with threading + batched inference
import onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor

class MultiStreamPipeline:
    def __init__(self, model_path, num_streams=4):
        self.session = ort.InferenceSession(
            model_path,
            providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider']
        )
        self.streams = [RTSPCapture(url) for url in camera_urls]
        self.tracker = [ByteTrack() for _ in range(num_streams)]

    def process_batch(self):
        """Grab frames from all cameras, batch inference, per-stream tracking."""
        frames = [stream.read() for stream in self.streams]
        batch = np.stack([preprocess(f) for f in frames])  # (4, 3, 640, 640)
        detections = self.session.run(None, {"input": batch})
        for i, (dets, tracker) in enumerate(zip(detections, self.trackers)):
            tracked = tracker.update(dets[i])
            self.analytics[i].update(tracked)
```

**Recommendation:** Use DeepStream for production deployment (hardware decode, proven stability, thermal management). Use custom ONNX Runtime pipeline for prototyping and development.

### 5.4 Latency Budget

```
Total budget per decision cycle: < 500 ms

┌─────────────────────────────────────────────────────┐
│ Stage                    │ Latency    │ Cumulative   │
├─────────────────────────────────────────────────────┤
│ Frame capture (RTSP)     │ ~33 ms     │ 33 ms        │
│ Decode (NVDEC)           │ ~5 ms      │ 38 ms        │
│ Preprocessing            │ ~2 ms      │ 40 ms        │
│ Detection (INT8, batch4) │ ~25 ms     │ 65 ms        │
│ Postprocessing           │ ~3 ms      │ 68 ms        │
│ Tracking (ByteTrack x4)  │ ~4 ms      │ 72 ms        │
│ Analytics computation    │ ~5 ms      │ 77 ms        │
│ RL policy inference      │ ~1 ms      │ 78 ms        │
│ Safety validation        │ ~1 ms      │ 79 ms        │
│ SNMP command (local LAN) │ ~10 ms     │ 89 ms        │
│ Controller response      │ ~50 ms     │ 139 ms       │
│ MQTT publish (async)     │ ~5 ms      │ 144 ms       │
├─────────────────────────────────────────────────────┤
│ TOTAL                    │            │ ~150 ms      │
│ Headroom                 │            │ ~350 ms      │
└─────────────────────────────────────────────────────┘
```

Actual signal decisions are made at 1 Hz (every ~1 second), not every frame. The 150 ms per-frame latency gives ample headroom.

### 5.5 Power and Thermal Management

**Jetson Orin NX power modes:**

| Mode | GPU Clock | CPU Clock | Power | Use Case |
|---|---|---|---|---|
| MAXN | 918 MHz | 2.0 GHz | 25W | Maximum performance; daytime peak hours |
| 15W | 612 MHz | 1.5 GHz | 15W | Balanced; most operating hours |
| 10W | 408 MHz | 1.2 GHz | 10W | Night/low-traffic; power saving |

**Thermal design for outdoor traffic cabinet:**
- **Operating temperature:** Traffic cabinets in Bintulu can reach 55-65C internal temperature (tropical climate, direct sun)
- **Jetson Orin NX max junction temp:** 105C; derate above 85C ambient
- **Enclosure:** IP67-rated fanless aluminum chassis (e.g., Connect Tech ThermiQ, Axiomtek AIE800, NEXCOM)
- **Heat dissipation:** Conduction cooling via thermal pads to aluminum chassis; chassis acts as heatsink
- **Passive cooling validation:** Required thermal simulation/testing at 65C ambient, 25W sustained GPU load
- **Power supply:** 12V/24V DC from traffic cabinet power rail; wide-input (9-36V DC) industrial PSU for surge protection

**Dynamic thermal management:**
```python
def adjust_power_mode(gpu_temp, traffic_volume):
    if gpu_temp > 90:
        set_power_mode("10W")  # thermal throttle
    elif traffic_volume < LOW_TRAFFIC_THRESHOLD:
        set_power_mode("10W")  # power save at night
    elif traffic_volume > HIGH_TRAFFIC_THRESHOLD:
        set_power_mode("MAXN")  # peak performance
    else:
        set_power_mode("15W")  # balanced
```

---

## 6. Performance Metrics & Evaluation

### 6.1 Key Performance Indicators (KPIs)

| KPI | Measurement Method | Target |
|---|---|---|
| **Average delay per vehicle** | Track time from queue entry to intersection clearance | Reduce by >= 20% vs fixed-time |
| **Average travel time** (corridor) | Track vehicles across multiple intersections | Reduce by >= 15% |
| **Queue length** | Max queue length per approach per cycle | Reduce by >= 25% |
| **Number of stops** | Count deceleration-to-stop events per track | Reduce by >= 20% |
| **Throughput** | Vehicles served per hour per approach | Increase by >= 10% |
| **Level of Service** | HCM LOS classification per approach | Improve by >= 1 grade |
| **Pedestrian wait time** | Time from ped button press to walk signal | Maintain or improve |
| **Emergency response time** | Time from EV detection to green | < 15 seconds |
| **System uptime** | Edge device operational hours / total hours | >= 99.5% |
| **Detection accuracy** | mAP@50 on Bintulu-specific validation set | >= 0.85 |
| **Counting accuracy** | Vehicle count vs manual ground truth | >= 95% |

### 6.2 Before/After Comparison Methodology

**Phase 1 baseline data collection (2-3 months):**
1. Deploy cameras and edge devices at target intersections
2. Run detection + tracking + analytics only (NO signal control)
3. Record: queue lengths, delays, throughput, LOS per 15-minute interval, 24/7
4. This becomes the "before" dataset

**Phase 2 A/B comparison:**
1. Enable adaptive control on odd days, revert to fixed-time on even days
2. Run for 4-6 weeks minimum (cover different day-of-week patterns)
3. Statistical comparison: paired t-test on matched time periods (Tuesday 8am-9am AI vs previous Tuesday 8am-9am fixed)
4. Control for external factors: weather, special events, road construction

**Metrics reporting for BDA stakeholders:**
- Monthly report with before/after comparison charts
- Real-time Grafana dashboard accessible via web
- Quarterly review presentation with KPI trend analysis
- Comparison to Miovision/NoTraffic published results (25-40% improvement benchmarks)

### 6.3 A/B Testing Protocol

```
Week 1-2: Calibration period (both modes, measure consistency)
Week 3-8: Alternating schedule

Monday:    AI Adaptive
Tuesday:   Fixed-Time (control)
Wednesday: AI Adaptive
Thursday:  Fixed-Time (control)
Friday:    AI Adaptive
Saturday:  Fixed-Time (control)
Sunday:    AI Adaptive

Compare: AI days vs Fixed days, matched by time-of-day
Statistical test: Welch's t-test, p < 0.05 for significance
Minimum sample: 100+ cycles per condition per approach
```

---

## 7. Implementation Phases

### Phase 1: Sensor Deployment & Data Collection (Months 1-4)

**Goal:** Deploy cameras and edge devices; collect baseline traffic data; validate detection pipeline.

**Scope:** 3 pilot intersections in Bintulu town center

| Task | Duration | Deliverable |
|---|---|---|
| Site survey: camera positions, power, network, controller audit | 2 weeks | Site assessment report |
| Procure hardware: Jetson Orin NX x3, IP cameras x12, enclosures, switches | 4 weeks | Hardware on-site |
| Install cameras, edge devices, network | 2 weeks | Operational installation |
| Camera calibration: homography, lane masks, counting lines | 1 week | Per-camera config YAML |
| Train detection model on Bintulu traffic data | 3 weeks | D-FINE-S fine-tuned, INT8 engine |
| Deploy detection + tracking pipeline | 1 week | Analytics flowing to cloud |
| Baseline data collection (no signal control) | 8 weeks | 2 months traffic dataset |
| Validate counting accuracy (manual vs AI) | 2 weeks | Accuracy report |

#### Model Card (Phase 1 Deliverables)

Each release produces a model card at `docs/model_cards/traffic_signal_control.md` and a YAML card at `releases/traffic_signal_control/v<N>/model_card.yaml`.

**Model artifacts:**

| Artifact | Format | Path |
|---|---|---|
| PyTorch model | `.pth` | `runs/traffic_signal_control/best.pt` |
| ONNX model | `.onnx` | `runs/traffic_signal_control/export/traffic_dfine_s_640_v{N}.onnx` |
| TensorRT engine | `.engine` | `runs/traffic_signal_control/export/traffic_dfine_s_640_v{N}_int8.engine` |
| Training config | `.yaml` | `configs/traffic_signal_control/06_training.yaml` |
| Metrics | `.json` | `runs/traffic_signal_control/metrics.json` |

**Hardware per intersection:**
- 1x NVIDIA Jetson Orin NX 16GB in IP67 enclosure
- 4x PoE IP cameras, 1080p, IR night vision, IP67
- 1x PoE switch, industrial grade
- 1x UPS for traffic cabinet
- Cabling and mounting hardware

### Phase 2: Rule-Based Adaptive Control (Months 5-8)

**Goal:** Enable rule-based adaptive signal timing; measure improvement over fixed-time.

| Task | Duration | Deliverable |
|---|---|---|
| Controller protocol audit (NTCIP/vendor-specific) | 2 weeks | Integration specification |
| Develop SNMP integration module | 3 weeks | Controller communication library |
| Implement Webster + actuated control logic | 2 weeks | Rule-based signal controller |
| Safety validator implementation + testing | 2 weeks | Validated safety layer |
| Lab testing with simulated controller | 2 weeks | Integration test report |
| Field deployment: rule-based control active | 1 week | Live adaptive control |
| A/B testing: adaptive vs fixed-time | 6 weeks | Performance comparison report |
| Tune parameters based on results | 2 weeks | Optimized configuration |

**Expected outcome:** 15-25% delay reduction vs fixed-time baseline.

#### Model Card (Phase 2 Deliverables)

Phase 2 focuses on signal control software rather than detection models. The detection model card from Phase 1 is updated with deployment validation metrics.

**Updated artifacts:**

| Artifact | Format | Path |
|---|---|---|
| Deployment metrics | `.json` | `runs/traffic_signal_control/deployment_metrics_v{N}.json` |
| Signal control config | `.yaml` | `configs/traffic_signal_control/signal_control.yaml` |
| Safety validator test report | `.pdf` | `docs/model_cards/traffic_signal_control/safety_report_v{N}.pdf` |

### Phase 3: RL Optimizer Integration (Months 9-14)

**Goal:** Train and deploy RL-based signal timing optimizer; validate improvement over rule-based.

| Task | Duration | Deliverable |
|---|---|---|
| Model Bintulu intersections in SUMO/CityFlow | 3 weeks | Calibrated simulation |
| Train PPO policy in simulation | 4 weeks | Trained RL model |
| Sim-to-real validation on historical data | 2 weeks | Transfer analysis report |
| Deploy RL as advisor (logging only, no control) | 4 weeks | RL decision log for review |
| Enable RL control with rule-based fallback | 2 weeks | Hybrid system live |
| A/B testing: RL-hybrid vs rule-based only | 6 weeks | Performance comparison |
| Iterate: retrain RL with real-world data | 3 weeks | Improved policy |

**Expected outcome:** Additional 5-15% delay reduction over rule-based alone.

#### Model Card (Phase 3 Deliverables)

Phase 3 introduces the RL signal timing policy model. A separate model card is produced for the RL policy alongside the updated detection model card.

**RL policy artifacts:**

| Artifact | Format | Path |
|---|---|---|
| RL policy model | `.pth` | `runs/traffic_signal_control/rl_policy/best_ppo_policy.pt` |
| RL training config | `.yaml` | `configs/traffic_signal_control/rl_training.yaml` |
| RL metrics | `.json` | `runs/traffic_signal_control/rl_policy/metrics.json` |
| Sim-to-real transfer report | `.pdf` | `docs/model_cards/traffic_signal_control/sim2real_report_v{N}.pdf` |
| Model card (RL) | `.yaml` | `releases/traffic_signal_control/rl_policy/v{N}/model_card.yaml` |

### Phase 4: Scale & Advanced Features (Months 15-24)

**Goal:** Scale to 20+ intersections; enable corridor coordination; add advanced analytics.

| Task | Duration | Deliverable |
|---|---|---|
| Scale to 10 additional intersections | 3 months | 13 intersections live |
| Multi-intersection coordination (CoLight/MAPPO) | 2 months | Coordinated green wave |
| Emergency vehicle preemption | 1 month | EVP system active |
| Anomaly detection (stalled vehicles, wrong-way) | 1 month | Alert system |
| Scale to 20+ intersections | 3 months | Full Bintulu coverage |
| Central management dashboard for BDA | 2 months | Operations center ready |
| Long-term optimization and maintenance | Ongoing | Continuous improvement |

#### Model Card (Phase 4 Deliverables)

Phase 4 scales to multi-intersection coordination. Model cards are produced per intersection cluster and a combined system-level card.

**Scaled deployment artifacts:**

| Artifact | Format | Path |
|---|---|---|
| Multi-agent RL policy | `.pth` | `runs/traffic_signal_control/multi_agent_rl/best_mappo_policy.pt` |
| Multi-agent config | `.yaml` | `configs/traffic_signal_control/multi_agent_rl.yaml` |
| Per-intersection model cards | `.yaml` | `releases/traffic_signal_control/intersection_{id}/v{N}/model_card.yaml` |
| System-level model card | `.yaml` | `releases/traffic_signal_control/system_v{N}/model_card.yaml` |
| Corridor coordination metrics | `.json` | `runs/traffic_signal_control/multi_agent_rl/corridor_metrics.json` |

### Minimum Viable Intersection (MVI) Deployment

The absolute minimum to demonstrate value at a single intersection:

```
Hardware:
  - 1x Jetson Orin NX 16GB
  - 2x IP cameras (main approaches only)
  - 1x PoE switch
  - 1x IP67 enclosure + mounting

Software:
  - D-FINE-S detection model (pre-trained on COCO, fine-tuned)
  - ByteTrack tracking
  - Vehicle counting + queue estimation + LOS classification
  - MQTT analytics to Grafana dashboard
  - Rule-based signal timing (Webster + actuated)
  - NO RL initially (add in Phase 3)

Timeline: 6-8 weeks from hardware arrival to live demo
```

### Scaling from 1 to 50+ Intersections

| Scale | Architecture Change | Network |
|---|---|---|
| 1-3 intersections | Independent edge devices; shared cloud dashboard | 4G per device |
| 4-10 intersections | Add corridor coordination (neighbor info sharing via MQTT) | 4G or fiber |
| 10-25 intersections | Multi-agent RL training; centralized policy optimization | Fiber backbone |
| 25-50+ intersections | Hierarchical control: zone coordinators + local agents | Fiber + redundant links |

**Per-intersection resource requirements at scale:**
- Edge hardware (standard components, decreasing with volume purchasing)
- Installation labor
- Cloud hosting per intersection
- Ongoing maintenance per intersection

Compare to commercial systems: NoTraffic and Miovision Surtrac are proprietary, vendor-locked solutions. Our approach uses standard hardware and open-source AI, eliminating vendor dependency and enabling full local control.

---

## References and Sources

### System Architecture & Edge AI
- [Lightweight Edge AI Framework for Adaptive Traffic Signal Control (MDPI)](https://www.mdpi.com/2071-1050/18/3/1147)
- [AI-Based Adaptive Traffic Signal Control System Review (MDPI)](https://www.mdpi.com/2079-9292/13/19/3875)
- [Intelligent Traffic Control Using Edge-Cloud Computing (IJERT)](https://www.ijert.org/an-intelligent-traffic-control-system-using-edge-cloud-computing-and-deep-learning-for-smart-cities)

### Object Detection Models
- [Best Object Detection Models 2025 — Roboflow](https://blog.roboflow.com/best-object-detection-models/)
- [D-FINE: Real-Time Object Detection — Ikomia](https://www.ikomia.ai/blog/d-fine-real-time-object-detection)
- [RF-DETR: SOTA Real-Time Object Detection — Roboflow](https://blog.roboflow.com/rf-detr/)
- [RT-DETR: DETRs Beat YOLOs](https://zhao-yian.github.io/RTDETR/)
- [RTDETRv2 vs YOLOX — Ultralytics](https://docs.ultralytics.com/compare/rtdetr-vs-yolox/)

### Vehicle Tracking & Counting
- [ByteTrack and BoT-SORT Vehicle Tracking — Medium](https://medium.com/@zain.18j2000/vehicles-tracking-using-botsort-and-bytetrack-tracking-algorithms-304759af3148)
- [Multi-Object Tracker Real-World Comparison — Veroke](https://www.veroke.com/insights/how-top-ai-multi-object-trackers-perform-in-real-world-scenarios/)
- [SORT, DeepSORT, ByteTrack Comparison for Highway Videos](https://vectoral.org/index.php/IJSICS/article/view/97)
- [Vehicle Queue Length Estimation (MDPI)](https://www.mdpi.com/2227-9717/9/10/1786)

### Speed Estimation & Calibration
- [Robust Automatic Monocular Vehicle Speed Estimation (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Revaud_Robust_Automatic_Monocular_Vehicle_Speed_Estimation_for_Traffic_Surveillance_ICCV_2021_paper.pdf)
- [Vision-Based Vehicle Speed Estimation Survey — IET](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/itr2.12079)

### Traffic Density & LOS
- [AI Tools for Level of Service Classification (De Gruyter)](https://www.degruyterbrill.com/document/doi/10.1515/comp-2025-0033/html)
- [Real-Time Traffic Density Estimation with YOLOv8 — Kaggle](https://www.kaggle.com/code/farzadnekouei/real-time-traffic-density-estimation-with-yolov8)
- [Level of Service — SMATS Traffic](https://www.smatstraffic.com/2021/07/26/level-of-service/)

### Reinforcement Learning for Traffic Signals
- [RL for Traffic Signal Control — Survey Portal](https://traffic-signal-control.github.io/)
- [PressLight — GitHub](https://github.com/wingsweihua/presslight)
- [CoLight: Network-Level Cooperation (ACM)](https://dl.acm.org/doi/10.1145/3357384.3357902)
- [LLMLight: LLM Agents for Traffic Control — GitHub](https://github.com/usail-hkust/LLMTSCS)
- [SUMO-RL: RL Environments for Traffic Signal Control — GitHub](https://github.com/LucasAlegre/sumo-rl)
- [CityFlow: Multi-Agent RL Environment — GitHub](https://github.com/cityflow-project/CityFlow)
- [Adaptive Traffic Signal Control with DQN/PPO (arXiv)](https://arxiv.org/abs/2602.12296)
- [Federated Deep RL for Urban Traffic Signal Control (Nature)](https://www.nature.com/articles/s41598-025-91966-1)
- [Sim-to-Real Transfer for Traffic Signal Control (RL Journal 2025)](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_256.pdf)

### Signal Controller Integration
- [NTCIP 1202: What Is It & Why Does It Matter — LYT](https://lyt.ai/blog/ntcip-1202-what-is-it-why-does-it-matter/)
- [NTCIP 1202 v02.19 Standard (PDF)](https://www.ntcip.org/wp-content/uploads/2018/11/NTCIP1202v0219f.pdf)
- [NTCIP 1202 Signal Controller Objects — ARC-IT](https://www.arc-it.net/html/standards/standard36.html)
- [Econolite ATC Traffic Signal Controllers](https://www.econolite.com/products/traffic-signal-controller/atc-traffic-signal-controller/)

### Signal Timing Algorithms
- [Webster's Formula for Optimum Cycle Length — APSED](https://www.apsed.in/post/traffic-signal-design-webster-s-formula-for-optimum-cycle-length)
- [Traffic Signals — Engineering LibreTexts](https://eng.libretexts.org/Bookshelves/Civil_Engineering/Fundamentals_of_Transportation/06:_Traffice_Control/6.02:_Traffic_Signals)

### Commercial Systems
- [NoTraffic AI Mobility Platform](https://www.notraffic.com/ai-mobility-platform/)
- [Miovision Surtrac](https://miovision.online/surtrac/)
- [Miovision Adaptive Signal Control](https://miovision.com/adaptive/)
- [Vivacity Labs AI Traffic Monitoring](https://vivacitylabs.com/)

### Emergency Vehicle Detection
- [AI-Powered Emergency Vehicle Sound Detection (MDPI Sensors)](https://www.mdpi.com/1424-8220/25/3/793)
- [Emergency Vehicle Preemption — Econolite](https://www.econolite.com/application-areas/emergency-vehicle-preemption/)
- [Emergency Vehicle Preemption — Commsignia](https://commsignia.com/solutions/emergency-preemption)

### Edge Deployment & Hardware
- [Jetson Benchmarks — NVIDIA](https://developer.nvidia.com/embedded/jetson-benchmarks)
- [DeepStream SDK Multi-Camera Pipelines — NVIDIA](https://developer.nvidia.com/blog/implementing-real-time-multi-camera-pipelines-with-nvidia-jetson/)
- [YOLOv8 Performance on Jetson (Seeed Studio)](https://www.seeedstudio.com/blog/2023/03/30/yolov8-performance-benchmarks-on-nvidia-jetson-devices/)
- [Connect Tech ThermiQ Edge AI Cooling](https://connecttech.com/connect-tech-launches-thermiq-edge-ai-cooling-solutions/)
- [IP67 Edge AI System — Axiomtek](https://www.axiomtek.com/Default.aspx?MenuId=Products&FunctionId=ProductView&ItemId=26135)
- [AX650N PSA Security Certification — Axera](https://en.axera-tech.com/News_desc/5/427.html)

### Bintulu / Malaysia Context
- [BDA Smart Traffic Light System Plan — Borneo Post](https://www.theborneopost.com/2019/07/29/bda-mulls-installing-smart-system-for-traffic-lights-across-bintulu-town/)
- [BDA Strategic Plan: Smart Liveable City — Borneo Post](https://www.theborneopost.com/2023/12/22/bdas-strategic-plan-repositions-bintulu-into-a-smart-liveable-city/)
- [Bintulu Development Authority](https://www.bda.gov.my/)
- [SASCOO AI Traffic System — Ledvision Malaysia](https://ledvision.com.my/project/transforming-traffic-flow-with-ai-the-sascoo/)

### MQTT & Communication
- [MQTT Protocol Guide — EMQ](https://www.emqx.com/en/blog/the-easiest-guide-to-getting-started-with-mqtt)
- [MQTT for Edge Device Connectivity — Amplicon](https://www.amplicon.com/actions/viewDoc.cfm?doc=mqtt-enabling-edge-device-connectivity-in-the-iiot-era-white-paper.pdf)

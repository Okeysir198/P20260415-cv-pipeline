# Technical Approach: AI Smart Parking System — Bintulu, Malaysia

**Scope**: 7,000 parking bays across outdoor surface lots and multi-level structures
**Edge hardware**: NVIDIA Jetson Orin NX 16GB (primary) / AX650N (alternative)
**Existing pipeline**: YOLOX-M, D-FINE-S, RT-DETRv2, ByteTrack, ONNX export, INT8 quantization
**Licensing constraint**: Apache 2.0 / MIT only — **no AGPL-3.0 (Ultralytics) models**

---

## Customer Requirements

**Source:** Bintulu Smart City — Bintulu Development Authority (BDA), Sarawak, Malaysia. Part of a broader smart traffic initiative alongside adaptive signal control. Existing digital parking infrastructure via BorneoParkBTU app.

**Explicit Requirements:**
- Monitor and report real-time occupancy status for **7,000 parking bays** across outdoor surface lots, multi-level structures, and covered areas in Bintulu
- Deploy **ANPR (Automatic Number Plate Recognition)** at all entry/exit gates for vehicle identification, session tracking, and payment enforcement
- Provide a **parking guidance system** with LED signs at zone entry points showing available bay counts, directional arrows, and per-bay indicators (green/red) in indoor structures
- Deliver a **mobile application** (Android/iOS) with real-time bay finder, turn-by-turn navigation, cashless payment via Malaysian e-wallets, and e-receipts
- Build an **operations dashboard** with live occupancy map, camera feeds, ANPR event monitoring, violation management, and analytics
- Integrate with existing **BorneoParkBTU** digital parking app rather than replace it, extending the current BDA-managed infrastructure
- Support **Malaysian payment methods**: Touch 'n Go eWallet, Boost, GrabPay, DuitNow QR, FPX (online banking), and cash (legacy pay stations)
- Detect and enforce **parking violations**: double parking, fire lane obstruction, overtime parking, disabled bay misuse, no-parking zone violations
- Ensure **safety monitoring** across parking areas: loitering detection, intrusion alerts, fall detection, abandoned objects, vehicle break-in detection, wrong-way driving
- Design for **24-month phased deployment** from lab prototype through full 7,000-bay rollout, with training and handover to BDA operations team

**Reference Data:**

| Metric | Value |
|---|---|
| Total parking bays | 7,000 |
| Area types | Outdoor surface (60%), multi-level (30%), covered (10%) |
| Estimated cameras needed | 195-266 occupancy + 30 ANPR |
| Estimated edge devices | 30-35 Jetson Orin NX 16GB |
| Entry/exit gate lanes | ~30 lanes |
| Target occupancy accuracy | >= 97% bay-level, >= 99% zone-level |
| Target ANPR plate-level accuracy | >= 95% (after fine-tuning) |
| Deployment timeline | 24 months (6 phases) |

**Customer Reference:** [BorneoParkBTU App — Bintulu Digital Parking](https://dayakdaily.com/borneoparkbtu-app-introduced-for-cashless-parking-payments-in-bintulu-starting-may-1/); [Parquery Parking Monitoring](https://parquery.com/how-it-works/); [Cleverciti Smart Parking](https://www.cleverciti.com/en/resources/blog/how-does-smart-parking-work)

## Business Problem Statement

- **Parking congestion:** Drivers waste 5-15 minutes circling lots looking for available spaces during peak hours, especially near commercial centers and government offices in Bintulu, leading to traffic congestion on surrounding roads
- **Driver frustration and experience:** The absence of real-time availability information means drivers cannot plan ahead or be guided to open spots, resulting in a poor experience that discourages visits to central business areas
- **Operational inefficiency:** Manual enforcement and cash-based payment create opportunities for unpaid parking, overstayed sessions, and fee collection gaps. Without automated session tracking, the BDA cannot accurately bill or enforce time limits
- **Safety and security concerns:** Parking areas lack automated monitoring for suspicious activity, vehicle break-ins, loitering, and other safety incidents, exposing the BDA to liability and visitors to risk
- **Operational inefficiency:** Manual patrols for violation enforcement and occupancy monitoring are labor-intensive, inconsistent, and cannot cover all 7,000 bays simultaneously, especially at night and during adverse weather
- **Environmental impact:** Vehicles circling for parking contribute to unnecessary fuel consumption and emissions in urban areas, counteracting smart city sustainability goals
- **Underutilized capacity:** Inefficient parking management reduces turnover — vehicles that stay too long or occupy premium spots without paying reduce available capacity for new visitors, hurting local businesses that depend on customer foot traffic

## Technical Problem Statement

- **Parking congestion -> Large-scale camera deployment:** Monitoring 7,000 bays across outdoor lots, multi-level structures, and covered areas requires an estimated 195-266 occupancy cameras and 30 ANPR cameras, demanding a robust three-tier network (access/aggregation/core), PoE+ power distribution, fiber backbone, and 30-35 edge AI devices — a significant infrastructure challenge in Bintulu's tropical climate (27-33C avg, 80-90% humidity)
- **Driver frustration -> Real-time occupancy detection at scale:** Achieving >= 97% bay-level accuracy across diverse conditions (outdoor sun/shadows/rain vs. indoor fluorescent/LED) requires distinguishing occupied from vacant bays reliably, with edge cases including motorcycles in car bays, partially visible vehicles, shadow artifacts, rain reflections, and vehicles overhanging multiple bays
- **Operational inefficiency -> ANPR accuracy for Malaysian plates:** Malaysian plate formats are complex (Peninsular, Sarawak Q-prefix, Sabah S-prefix, taxi, military, diplomatic, multi-line motorcycle plates) and Bintulu specifically uses Sarawak division codes. Achieving >= 95% plate-level accuracy requires fine-tuning on local plate fonts, handling dirty/damaged/angled plates, and managing a 3-7% initial misread rate
- **Safety and security -> Multi-model orchestration on edge:** Running occupancy detection, ANPR, safety monitoring (loitering, intrusion, fall detection, abandoned objects), and violation detection simultaneously on each edge device requires multi-stream inference at low FPS with shared GPU resources, plus ByteTrack-based person/vehicle tracking with dwell timers
- **Operational inefficiency -> Real-time guidance system integration:** Updating 80+ LED signs (RS485 protocol) and a mobile app in real-time requires low-latency MQTT telemetry, zone-level aggregation every 10 seconds, and bay-level state changes published within 3 seconds of detection
- **Environmental impact -> Temporal smoothing and false positive control:** Preventing flickering bay status (caused by shadows, passing vehicles, camera shake) while maintaining responsive detection requires asymmetric hysteresis state machines with configurable occupy/vacate thresholds across a sliding window
- **Underutilized capacity -> Payment integration and violation enforcement:** Matching entry/exit ANPR sessions, calculating fees against Malaysian rate structures, integrating with multiple e-wallet providers (Touch 'n Go, Boost, GrabPay, DuitNow), and capturing legally-defensible violation evidence with multi-angle snapshots requires a reliable backend with PostgreSQL + TimescaleDB

## Technical Solution Options

### Option 1: Hybrid Detection + Classification on Jetson Orin NX (Recommended)

- **Approach:** Use YOLOX-Tiny (or D-FINE-N as transformer alternative) as a vehicle detector on the full camera frame at 2-5 FPS, match detections to pre-defined bay polygons using area-overlap scoring, then fall back to MobileNetV3-Small classification on cropped bay regions only for ambiguous cases (confidence 0.4-0.7). ANPR uses YOLOX-Tiny plate detection + PaddleOCR PP-OCRv4 OCR. All inference on NVIDIA Jetson Orin NX 16GB with DeepStream 7.x multi-stream pipeline and TensorRT INT8.
- **Addresses:** Large-scale camera deployment (8 cameras per edge box, ~35 edge boxes total), real-time occupancy detection (hybrid approach balances accuracy and cost — detection on full frame + selective classification for ~5-10% ambiguous bays), ANPR accuracy (custom pipeline with fine-tuning on Malaysian plates), multi-model orchestration (DeepStream handles stream muxing, batched inference, and metadata routing), temporal smoothing (asymmetric hysteresis state machine with 5-frame sliding window)
- **Pros:** Leverages existing YOLOX training pipeline and ONNX export tooling; Apache 2.0 license throughout; Jetson Orin NX provides 100 TOPS INT8 with 11GB headroom for future models; DeepStream is production-proven for multi-stream; hybrid approach reduces compute by only classifying ambiguous bays (~5-10% vs. all 60 bays every frame)
- **Cons:** Jetson Orin NX is more expensive than AX650N alternatives; outdoor lighting variability in Bintulu (tropical sun, monsoon rain) requires extensive augmentation and local fine-tuning; RS485 LED sign integration is hardware-specific per manufacturer; 24-month deployment timeline is aggressive for 7,000 bays

### Option 2: Per-Slot Classification with Overhead Cameras

- **Approach:** Crop each bay region from the frame using pre-defined ROI polygons, then classify every crop as occupied/vacant using MobileNetV3-Small (98.01% accuracy on PKLot dataset). Requires a separate classification inference for every bay every frame (60 bays x 5 FPS = 300 classifications/sec/camera). Best suited for indoor/multi-level structures with controlled lighting and near-overhead camera angles.
- **Addresses:** Real-time occupancy detection (higher per-bay accuracy than detection-based approach in controlled environments); temporal smoothing (simpler — single classification per bay vs. detection + ROI matching)
- **Pros:** Highest per-bay accuracy (98%+) in controlled indoor environments; simpler pipeline (classify crop -> occupied/vacant, no ROI matching logic); well-studied approach with extensive benchmark datasets (PKLot, CNRPark-EXT)
- **Cons:** Expensive at scale — 300+ classifications per second per camera vs. 1 detection inference; requires individual crops for every bay every frame, consuming more memory bandwidth; poor performance outdoors with variable lighting and shadows; camera angle sensitivity (requires near-overhead view, limiting coverage to 25-40 bays/camera indoors vs. 40-60 outdoors with detection approach); not viable as primary approach for outdoor surface lots (60% of Bintulu's bays)

**Decision:** Option 1 (Hybrid Detection + Classification on Jetson Orin NX) selected as the primary approach. It handles both outdoor and indoor environments with a single pipeline, scales to 7,000 bays across 30-35 edge devices, and builds on existing YOLOX/D-FINE training infrastructure. Option 2 (Per-Slot Classification) is recommended as a secondary strategy for indoor multi-level structures where controlled lighting makes it superior. The hybrid approach in Option 1 already incorporates MobileNetV3 classification as a fallback for ambiguous bays, effectively combining both options. See System Architecture (Section 1) and AI Models (Section 2) for implementation details.

---

## 1. System Architecture

### 1.1 High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        FIELD LAYER (per zone)                          │
│                                                                        │
│  ┌──────────┐   RTSP    ┌──────────────────┐   MQTT/gRPC             │
│  │ IP Camera├──────────►│  Edge Box         │──────────┐              │
│  │ (PoE)    │           │  (Jetson Orin NX) │          │              │
│  └──────────┘           │                   │          │              │
│  ┌──────────┐   RTSP    │  • Occupancy AI   │          │              │
│  │ IR Camera├──────────►│  • ANPR AI        │          │              │
│  │ (gate)   │           │  • Safety AI      │          │              │
│  └──────────┘           │  • ByteTrack      │          │              │
│                         └──────────────────┘          │              │
│  ┌──────────┐   RS485                                  │              │
│  │ LED Sign │◄─────────── (zone controller) ◄──────────┤              │
│  └──────────┘                                          │              │
└────────────────────────────────────────────────────────┼──────────────┘
                                                         │
                          Fiber / 4G LTE                 │
                                                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         BACKEND LAYER                                   │
│                                                                        │
│  ┌────────────┐   ┌──────────────┐   ┌──────────────┐                │
│  │ EMQX MQTT  │──►│ Event Engine │──►│ TimescaleDB  │                │
│  │ Broker     │   │ (Go / Rust)  │   │ (time-series)│                │
│  └────────────┘   └──────┬───────┘   └──────────────┘                │
│                          │                                             │
│                          ▼                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ PostgreSQL   │  │ Redis        │  │ MinIO / S3   │               │
│  │ (transact.)  │  │ (cache/pub)  │  │ (snapshots)  │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
│                          │                                             │
│                          ▼                                             │
│  ┌──────────────────────────────────────────────┐                    │
│  │           API Gateway (FastAPI / Kong)        │                    │
│  │  /occupancy  /anpr  /violations  /analytics   │                    │
│  └──────────────┬──────────────┬────────────────┘                    │
│                 │              │                                       │
└─────────────────┼──────────────┼───────────────────────────────────────┘
                  │              │
        ┌─────────┘              └──────────┐
        ▼                                   ▼
┌───────────────┐                 ┌──────────────────┐
│ Dashboard     │                 │ Mobile App       │
│ (React + WS)  │                 │ (Flutter/RN)     │
│ • Live map    │                 │ • Bay finder     │
│ • Camera feed │                 │ • Navigation     │
│ • Analytics   │                 │ • Payment        │
│ • Alerts      │                 │ • E-receipt      │
└───────────────┘                 └──────────────────┘
```

### 1.2 ANPR Subsystem (Separate from Occupancy)

ANPR runs on dedicated IR cameras at entry/exit gates — physically and logically separated from the bay-occupancy cameras:

```
Entry Gate                                    Exit Gate
┌───────────┐                                ┌───────────┐
│ IR Camera  │                                │ IR Camera  │
│ 2MP, 940nm│                                │ 2MP, 940nm│
└─────┬─────┘                                └─────┬─────┘
      │ RTSP                                       │ RTSP
      ▼                                            ▼
┌─────────────────────────────────────────────────────────┐
│              ANPR Edge Box (Jetson Orin NX)              │
│                                                          │
│  1. Plate Detection  → YOLOX-Tiny or D-FINE-N (INT8)   │
│  2. Plate Alignment  → STN or affine warp               │
│  3. OCR              → PaddleOCR (CRNN+CTC, INT8)      │
│  4. Format Validator → Malaysian plate regex             │
│  5. Confidence Gate  → threshold ≥ 0.85                  │
│                                                          │
│  Output: {plate_text, confidence, timestamp, camera_id,  │
│           direction: entry|exit, snapshot_url}            │
└──────────────────────────┬──────────────────────────────┘
                           │ MQTT topic: parking/anpr/events
                           ▼
                    Backend Session Manager
                    • Match entry ↔ exit plates
                    • Calculate duration & fee
                    • Fuzzy match for misreads
                    • Whitelist/blacklist check
```

### 1.3 Data Flow Specifications

| Data Type | Format | Rate | Protocol | Topic / Endpoint |
|---|---|---|---|---|
| Bay status change | `{bay_id, status, confidence, timestamp}` | On-change (~0.5-2 msg/bay/hr avg) | MQTT QoS 1 | `parking/occupancy/{zone_id}` |
| Zone count summary | `{zone_id, total, occupied, vacant}` | Every 10s | MQTT QoS 0 | `parking/zones/{zone_id}/count` |
| ANPR event | `{plate, confidence, direction, camera_id, ts, snapshot_path}` | Per vehicle (~500-2000/day/gate) | MQTT QoS 1 | `parking/anpr/events` |
| Violation alert | `{type, bay_id, plate, evidence_urls[], ts}` | On-event | MQTT QoS 2 | `parking/violations` |
| Safety alert | `{type, person_id, location, dwell_s, snapshot}` | On-event | MQTT QoS 2 | `parking/safety/alerts` |
| Heartbeat | `{edge_id, cpu_temp, gpu_util, uptime}` | Every 30s | MQTT QoS 0 | `parking/health/{edge_id}` |

### 1.4 Edge Device Software Stack

```
┌──────────────────────────────────────────────────┐
│  JetPack 6.x (L4T + Ubuntu 22.04)                │
│  ├── TensorRT 10.x (INT8 inference)              │
│  ├── DeepStream 7.x (multi-stream pipeline)      │
│  ├── GStreamer (RTSP decode, frame mux)           │
│  ├── CUDA 12.x / cuDNN 9.x                       │
│  ├── Python 3.12 + ONNX Runtime (fallback)       │
│  ├── Mosquitto (local MQTT broker)                │
│  ├── SQLite (local state buffer for offline)      │
│  └── Mender OTA agent (remote updates)            │
│                                                    │
│  Application Layer:                                │
│  ├── occupancy_engine.py                          │
│  │   └── detect → ROI match → temporal smooth     │
│  ├── anpr_engine.py                               │
│  │   └── detect plate → align → OCR → validate    │
│  ├── safety_engine.py                             │
│  │   └── person detect → track → dwell timer      │
│  ├── stream_manager.py                            │
│  │   └── RTSP connect, reconnect, health check    │
│  └── publisher.py                                 │
│      └── MQTT publish, offline buffer, retry      │
└──────────────────────────────────────────────────┘
```

### 1.5 Network Topology for 7,000 Bays

**Camera count estimation:**

| Area Type | Bays per Camera | Total Bays | Cameras Needed | Edge Boxes (8 cam each) |
|---|---|---|---|---|
| Outdoor surface lot | 40-60 bays/cam | 4,200 (60%) | 84-105 | 11-14 |
| Multi-level structure | 25-40 bays/cam | 2,100 (30%) | 53-84 | 7-11 |
| Covered / tight layout | 15-25 bays/cam | 700 (10%) | 28-47 | 4-6 |
| Entry/exit gates (ANPR) | 1 lane/cam | ~30 lanes | 30 (IR) | 4 |
| **Total** | | **7,000** | **195-266** | **26-35** |

**Network topology:**

```
                     ┌─────────────────────┐
                     │  Core Switch (L3)    │
                     │  10GbE fiber uplinks │
                     │  (Data Center / NOC) │
                     └──────┬──────────────┘
                            │ 10G fiber
               ┌────────────┼────────────────┐
               ▼            ▼                ▼
        ┌────────────┐ ┌────────────┐  ┌────────────┐
        │ Agg Switch │ │ Agg Switch │  │ Agg Switch │
        │ Zone A     │ │ Zone B     │  │ Zone N     │
        │ 1G fiber   │ │ 1G fiber   │  │ 1G fiber   │
        └──────┬─────┘ └──────┬─────┘  └──────┬─────┘
               │              │                │
         ┌─────┴─────┐  ┌────┴──────┐   ┌────┴──────┐
         ▼           ▼  ▼          ▼   ▼          ▼
    ┌─────────┐ ┌─────────┐  ┌─────────┐  ┌─────────┐
    │PoE Sw   │ │PoE Sw   │  │PoE Sw   │  │PoE Sw   │
    │24-port  │ │24-port  │  │24-port  │  │24-port  │
    │802.3at  │ │802.3at  │  │802.3at  │  │802.3at  │
    └──┬──────┘ └──┬──────┘  └──┬──────┘  └──┬──────┘
       │           │            │            │
   8-12 IP     8-12 IP     8-12 IP      8-12 IP
   cameras     cameras     cameras      cameras
       │           │            │            │
    ┌──▼───┐   ┌──▼───┐     ┌──▼───┐    ┌──▼───┐
    │Jetson│   │Jetson│     │Jetson│    │Jetson│
    │Orin  │   │Orin  │     │Orin  │    │Orin  │
    └──────┘   └──────┘     └──────┘    └──────┘
```

**Bill of materials (network):**

| Component | Quantity | Purpose |
|---|---|---|
| Core L3 switch (10GbE, 48-port SFP+) | 2 (redundant) | Data center aggregation |
| Aggregation switch (1G fiber, 24 SFP) | 8-12 | Per-zone aggregation |
| PoE+ access switch (24-port 802.3at, 1G fiber uplink) | 24-30 | Camera power + data |
| Jetson Orin NX 16GB + enclosure | 30-35 | Edge AI processing |
| Single-mode fiber (OS2) | ~15 km total | Backbone runs |
| Cat6 outdoor-rated cable | ~50 km total | Camera to PoE switch (≤100m) |
| UPS (per switch cabinet) | 24-30 | 30-min battery backup |

---

## 2. AI Models

### 2.1 Parking Bay Occupancy Detection

#### Two Approaches Compared

**Approach A: Object Detection + ROI Matching**

Detect all vehicles in the full camera frame, then match each detection to pre-defined bay polygons using IoU or centroid overlap.

| Model | mAP@0.5 (COCO) | Params | INT8 Latency (Orin NX) | License |
|---|---|---|---|---|
| **YOLOX-M** | 46.9% | 25.3M | ~8ms | Apache 2.0 |
| **D-FINE-S** | 48.5% | 10M | ~12ms | Apache 2.0 |
| **YOLOX-Tiny** | 32.8% | 5.1M | ~3ms | Apache 2.0 |
| SSD-MobileNetV2 | ~22% | 3.4M | ~2ms | Apache 2.0 |
| RT-DETRv2-R18 | 47.9% | 20M | ~15ms | Apache 2.0 |

**Approach B: Per-Slot Classification**

Crop each bay region from the frame using pre-defined ROI polygons, then classify each crop as occupied/vacant.

| Model | Accuracy (PKLot+CNRPark) | Params | Latency per Crop | License |
|---|---|---|---|---|
| **MobileNetV3-Small (improved)** | 98.01% | 2.5M | ~0.3ms | Apache 2.0 |
| EfficientNet-B0 | ~97.5% | 5.3M | ~0.8ms | Apache 2.0 |
| ResNet-18 | ~96.5% | 11.7M | ~1.2ms | BSD |
| CarNet (custom) | 97.03% | ~1M | ~0.2ms | Research |

**Recommendation: Hybrid approach — Detection + Classification**

Use **YOLOX-Tiny** (or D-FINE-N for transformer alternative) as a vehicle detector on the full frame at 2-5 FPS, then use the detection results to confirm/deny per-bay occupancy via ROI overlap. For ambiguous cases (confidence 0.4-0.7), fall back to **MobileNetV3-Small** classification on the cropped bay region.

Rationale:
- Pure detection is robust to new/unseen bay configurations but can miss partially visible vehicles
- Pure classification is more accurate per-bay but requires a crop for every bay every frame (expensive at scale: 60 bays x 5 FPS = 300 classifications/sec/camera)
- The hybrid approach runs detection on the full frame (1 inference) and only classifies the 5-10% ambiguous bays
- This matches the approach used by Parquery (single camera → up to 300 bays, AI classifies occupancy)

**Outdoor lots vs indoor garages:**

| Factor | Outdoor | Indoor/Multi-level |
|---|---|---|
| Lighting | Variable (sun, shadows, rain) | Controlled (fluorescent, LED) |
| Best approach | Detection (handles variable conditions) | Classification (stable crops) |
| Key challenge | Shadows causing false occupied | Low ceilings limiting camera angle |
| Camera type | 4MP+ wide-angle, IP67 | 2MP, wider FoV, vandal-proof |
| Recommended model | YOLOX-Tiny + MobileNetV3 hybrid | MobileNetV3 classification (primary) |

**Edge cases and mitigations:**

| Edge Case | Mitigation |
|---|---|
| Motorcycles in car bays | Train detector on motorcycle class; classify as "occupied-motorcycle" |
| Partially visible vehicles | IoU threshold ≥ 0.15 (low) for bay matching; temporal smoothing confirms |
| Shadows mimicking vehicles | Use RGB+temporal: shadow moves with sun; train with shadow-augmented data |
| Rain puddles / reflections | Include rainy-day images in training set; temporal smoothing filters transients |
| Vehicle overhanging 2 bays | Assign to bay with highest centroid overlap; flag for review if IoU split >0.4/0.4 |
| Empty trailer / flatbed | Classify based on vehicle presence, not cargo; height not relevant for 2D |

**Temporal smoothing** (see Section 4 for full state machine):
- Sliding window of N=5 frames (1-2 seconds at 3-5 FPS)
- Majority vote: bay is "occupied" only if ≥3/5 frames agree
- Hysteresis: transition from vacant→occupied requires 3 consecutive "occupied" votes; occupied→vacant requires 4 consecutive "vacant" votes (asymmetric to prevent flicker)
- Commercial systems (Parquery, Cleverciti) report 95-99% accuracy using similar temporal filtering with AI classification

**Benchmark references:**
- PKLot dataset: 12,416 images, 695,899 parking space samples — MobileNetV3 achieves 98.01% accuracy
- CNRPark-EXT dataset: 157,549 patches — AUC 0.99 with optimized MobileNetV3
- Cleverciti: "industry's most accurate parking sensor" per independent studies (no public mAP, but rated >99% in controlled tests)
- Parquery: single camera monitors up to 300 bays; de-identified images (no LPR from occupancy cameras)

### 2.2 ANPR / License Plate Recognition

#### Pipeline Architecture

```
Frame (1080p IR) → Plate Detection → Alignment/Correction → OCR → Format Validation → Output
```

#### Stage 1: Plate Detection

| Model | mAP@0.5 | Speed (Orin NX INT8) | Notes |
|---|---|---|---|
| **YOLOX-Tiny** | ~92% (plate) | ~3ms | Our existing pipeline, easy to train |
| **D-FINE-N** | ~93% (plate) | ~5ms | NMS-free, transformer alternative |
| WPOD-Net | ~85% | ~8ms | Includes affine unwarp, but older architecture |
| SSD-MobileNetV2 | ~88% | ~2ms | Lightweight but lower accuracy on angled plates |

**Recommendation**: **YOLOX-Tiny** for plate detection — we already have the training pipeline, INT8 export, and it matches our Apache 2.0 licensing. D-FINE-N as transformer alternative for NMS-free inference.

#### Stage 2: Plate Alignment/Correction

Two options:
1. **Spatial Transformer Network (STN)**: Learned affine correction before OCR. Improves recognition from 88.3% to 95.0% on angled plates.
2. **Four-corner regression + perspective warp**: Detect plate corners (4 keypoints), apply `cv2.getPerspectiveTransform()` to rectify. Simpler, no extra model needed if plate detector outputs corners.

**Recommendation**: Four-corner keypoint regression added to the plate detection head (lightweight, no separate model). Fall back to STN for severely distorted plates.

#### Stage 3: OCR

| Engine | Accuracy (general) | Speed | Trainable | Malaysian Support | License |
|---|---|---|---|---|---|
| **PaddleOCR (PP-OCRv4)** | ~98% (printed) | ~5ms | Yes (fine-tune on plates) | Good (Latin) | Apache 2.0 |
| EasyOCR | 90-95% | ~15ms | Limited | Malay supported | Apache 2.0 |
| TrOCR (Microsoft) | ~95% (handwritten) | ~30ms | Yes (HF) | Needs fine-tuning | MIT |
| LPRNet | ~94% | ~2ms | Yes | Needs training | MIT |
| CRNN+CTC | ~93% | ~3ms | Yes | Needs training | Various |
| Tesseract 5 | ~85% | ~20ms | Config only | Poor on plates | Apache 2.0 |

**Recommendation**: **PaddleOCR PP-OCRv4** — highest accuracy on printed text, lightweight (<10MB model), Apache 2.0, supports fine-tuning on Malaysian plate fonts. For maximum speed with slightly lower accuracy, use **LPRNet** (2ms, trainable, MIT license). Deploy PaddleOCR as primary, LPRNet as fallback for high-throughput gates.

#### Malaysian Plate Format Handling

Malaysian plates follow several formats that the OCR post-processor must validate:

| Type | Format | Examples | Regex Pattern |
|---|---|---|---|
| Standard (Peninsular) | `[A-Z]{1,3} [0-9]{1,4} [A-Z]?` | `WA 1234 B`, `B 9876` | `^[A-Z]{1,3}\s?\d{1,4}\s?[A-Z]?$` |
| Sarawak (Q-prefix) | `Q[A-Z]{1,2} [0-9]{1,4} [A-Z]?` | `QSA 1234`, `QK 567 A` | `^Q[A-HJ-NP-Y]{1,2}\s?\d{1,4}\s?[A-Z]?$` |
| Sabah (S-prefix) | `S[A-Z]{1,2} [0-9]{1,4} [A-Z]?` | `SA 1234 B` | `^S[A-Z]{1,2}\s?\d{1,4}\s?[A-Z]?$` |
| Taxi | `H[A-Z] [0-9]{1,4}` | `HA 1234` | `^H[A-Z]\s?\d{1,4}$` |
| Special / Vanity | Variable | `MALAYSIA 1`, `PUTRAJAYA 1` | Whitelist lookup |
| Military | `ZM [0-9]+` | `ZM 1234` | `^Z[A-Z]\s?\d{1,4}$` |
| Diplomatic | `[0-9]+-[0-9]+-[0-9]+` | `16-23-1` | `^\d{1,3}-\d{1,3}-\d{1,3}$` |
| Multi-line | Top: prefix, Bottom: numbers | Common on motorcycles | OCR reads both lines, concat |

**Sarawak-specific (Bintulu)**: All Sarawak plates use `Q` prefix. Bintulu plates registered locally use division codes like `QSA`, `QSB`, etc. Letters I, O, Z are not used. No leading zeros in numbers.

**IR camera considerations:**
- 940nm IR illumination for nighttime: invisible to drivers, avoids glare
- IR-cut filter bypass: plate reflective material reflects IR strongly, creating high-contrast characters
- Shutter speed ≤ 1/500s to freeze motion at entry gates
- Distance: 3-5m from gate barrier, 15-30 degree angle
- Resolution: minimum 2MP, plate should be ≥100px wide in frame

**Dirty/damaged/angled plates:**
- Train on augmented data: blur, noise, occlusion, tilt up to 45 degrees
- Multi-frame aggregation: capture 3-5 frames per vehicle approach, take highest-confidence OCR result
- Confidence threshold: accept if ≥0.85; queue for manual review if 0.60-0.85; reject if <0.60
- Fuzzy matching for partial reads: Levenshtein distance ≤ 2 against known plates in the facility

**Commercial vs open-source comparison:**

| Solution | Accuracy | Edge Support | Malaysian Plates | License |
|---|---|---|---|---|
| Plate Recognizer SDK | 97%+ | Jetson, RPi, x86 | Yes (MY region) | Commercial |
| OpenALPR (legacy) | ~90% | Linux | Limited | Commercial |
| UltimateALPR SDK | 95%+ | Jetson, ARM, x86 | Configurable | Commercial |
| **Custom (YOLOX + PaddleOCR)** | ~93-96% (after fine-tune) | Full control | Fine-tune to MY | Apache 2.0 |

**Recommendation**: Build custom ANPR using **YOLOX-Tiny (plate detect) + PaddleOCR (OCR)** — full control, fine-tune on Malaysian plates. For the initial prototype phase, consider **Plate Recognizer** as a benchmark reference to validate our custom pipeline's accuracy. Target: ≥95% plate-level accuracy after fine-tuning on 5,000+ Malaysian plate images.

### 2.3 Safety and Anomaly Detection

All safety models use **person detection from COCO-pretrained YOLOX** (class 0: person), which is already part of our pipeline.

| Safety Feature | Detection Method | Model(s) | Trigger |
|---|---|---|---|
| **Loitering** | Person detect → ByteTrack ID → dwell timer | YOLOX-Tiny (person) + ByteTrack | `dwell_time > threshold` (e.g., 300s) |
| **Intrusion** | Person detect → zone ROI check | YOLOX-Tiny (person) | Person centroid inside restricted ROI |
| **Fall detection** | Dedicated fall model (already trained) | YOLOX-M (person, fallen_person) | `fallen_person` class detected |
| **Abandoned object** | Background subtraction + static blob timer | Frame diff + ByteTrack | Static foreground blob >600s, no person nearby |
| **Vehicle break-in** | Person near vehicle + unusual dwell | YOLOX (person) + ByteTrack + bay status | Person at occupied bay >120s, not driver |
| **Wrong-way driving** | Vehicle track direction vs zone rules | YOLOX (vehicle) + ByteTrack | Track direction opposite to zone `allowed_direction` |

**Loitering implementation:**

```python
# Pseudocode: per-track dwell time
for track in active_tracks:
    if track.class_id == PERSON:
        track.dwell_time += frame_interval
        if track.dwell_time > config.loiter_threshold_sec:
            if not track.alert_sent:
                publish_alert("loitering", track)
                track.alert_sent = True
```

**Anomaly detection approaches:**
- **Rule-based** (recommended for parking): Define zone-specific rules (loiter time, restricted areas, vehicle direction). Simple, explainable, auditable, no extra training data.
- **Learned** (future enhancement): Train autoencoder on "normal" parking activity; flag high reconstruction error. Requires months of baseline data. Better for detecting novel anomalies.

**Recommendation**: Start with rule-based detection for all safety features. It is fully auditable (important for Malaysian regulatory compliance) and requires no additional training data. Evaluate learned anomaly detection after 6 months of baseline data collection.

### 2.4 Violation Detection

| Violation Type | Detection Logic | Evidence |
|---|---|---|
| **Double parking** | Vehicle bbox outside all bay ROIs + `dwell > 60s` | 3 snapshots (initial, 30s, 60s) + plate OCR |
| **Fire lane obstruction** | Vehicle bbox overlaps fire_lane ROI + `dwell > 30s` | Snapshot + plate + zone metadata |
| **Overtime parking** | ANPR session duration > permitted time | Entry/exit timestamps + bay occupancy log |
| **Disabled bay misuse** | Vehicle in disabled_bay ROI + plate not in permit whitelist | Snapshot + plate + permit DB check |
| **No-parking zone** | Vehicle in no_parking ROI + `dwell > 30s` | Snapshot series + plate |

**Evidence capture protocol:**

```json
{
  "violation_id": "VIO-2026-03-19-001234",
  "type": "double_parking",
  "timestamp_start": "2026-03-19T10:23:15+08:00",
  "timestamp_confirmed": "2026-03-19T10:24:15+08:00",
  "location": {"zone": "A", "floor": 1, "near_bay": "A-142"},
  "plate": {"text": "QSA 1234", "confidence": 0.92},
  "evidence": [
    {"type": "snapshot", "url": "s3://violations/VIO-001234/t0.jpg", "timestamp": "..."},
    {"type": "snapshot", "url": "s3://violations/VIO-001234/t30.jpg", "timestamp": "..."},
    {"type": "snapshot", "url": "s3://violations/VIO-001234/t60.jpg", "timestamp": "..."},
    {"type": "video_clip", "url": "s3://violations/VIO-001234/clip.mp4", "duration_sec": 90}
  ],
  "reviewed": false,
  "officer_id": null,
  "fine_amount": null
}
```

**Commercial enforcement reference:**
- Genetec AutoVu: mobile ANPR + fixed cameras, chalking replacement, overtime enforcement
- Smart Parking Ltd (Australia): sensor + camera fusion, app-based enforcement, digital permits
- SenSen AI: curbside enforcement with automatic evidence capture and ticket issuance
- Parkify: AI enforcement cameras with multi-angle evidence packages

---

## 3. Bay ROI Configuration and Management

### 3.1 Defining Parking Bay Polygons

Each bay is defined as a 4+ point polygon in normalized image coordinates (0-1 range), stored per camera:

```yaml
# configs/parking/zone_a_cam_01.yaml
camera_id: "zone_a_cam_01"
resolution: [1920, 1080]
bays:
  - id: "A-001"
    type: "standard"        # standard | disabled | electric | motorcycle
    polygon: [[0.12, 0.45], [0.18, 0.45], [0.19, 0.62], [0.11, 0.62]]
    zone: "A"
    floor: 1
  - id: "A-002"
    type: "standard"
    polygon: [[0.18, 0.45], [0.24, 0.45], [0.25, 0.62], [0.19, 0.62]]
    zone: "A"
    floor: 1
restricted_zones:
  - id: "fire_lane_01"
    type: "fire_lane"
    polygon: [[0.0, 0.80], [1.0, 0.80], [1.0, 1.0], [0.0, 1.0]]
```

### 3.2 ROI Configuration Tools

**Option 1: Web-based drawing UI (recommended)**

Build a React-based configuration tool:
1. Stream live camera feed or load reference snapshot
2. Click to draw polygons for each bay
3. Auto-suggest grid layout for regular lots (user adjusts)
4. Import/export YAML or JSON
5. Overlay detection results for visual validation

**Option 2: Semi-automatic detection**

Use the vehicle detector on a set of parking lot images (across different times) to build a statistical map of where vehicles park. Cluster vehicle centroids in bird's-eye view to auto-detect bay positions. This is the approach described in the paper "Automatic Vision-Based Parking Slot Detection and Occupancy Classification" (ECCV 2023) — uses vehicle detection clustering in BEV to discover slots.

**Recommendation**: Start with the web-based drawing UI for initial deployment (deterministic, reliable). Add semi-automatic bay discovery as a v2 feature to accelerate new zone onboarding.

### 3.3 Perspective Transformation

For overhead analytics, reporting, and semi-auto bay detection, compute a homography matrix per camera:

```python
# Four reference points (image coords → real-world ground coords)
src_pts = np.float32([[x1,y1], [x2,y2], [x3,y3], [x4,y4]])  # pixel coords
dst_pts = np.float32([[X1,Y1], [X2,Y2], [X3,Y3], [X4,Y4]])  # meters on ground plane
H = cv2.getPerspectiveTransform(src_pts, dst_pts)

# Transform detection centroid to ground plane
ground_pos = cv2.perspectiveTransform(np.array([[[px, py]]]), H)
```

Store the homography matrix per camera. Recompute after any camera adjustment.

### 3.4 Camera Recalibration After Maintenance

When a camera is physically moved or re-aimed:
1. System detects calibration drift: bay accuracy drops below threshold
2. Alert triggers to maintenance team
3. Technician opens web UI, loads new camera view
4. System overlays old ROI polygons on new view
5. Technician adjusts polygons or re-runs semi-auto detection
6. Validation: run detection on 100 frames, compare to manual ground truth
7. Publish updated config to edge device via MQTT `parking/config/{camera_id}`

### 3.5 Bay Numbering Hierarchy

```
Facility → Floor → Zone → Bay

Example: Bintulu Central Parking
├── Floor G (Ground)
│   ├── Zone A (North outdoor)
│   │   ├── A-001 ... A-150 (standard)
│   │   ├── A-D01 ... A-D05 (disabled)
│   │   └── A-E01 ... A-E10 (EV charging)
│   ├── Zone B (South outdoor)
│   │   └── B-001 ... B-200
│   └── Zone C (Covered)
│       └── C-001 ... C-100
├── Floor 1
│   ├── Zone D ...
│   └── Zone E ...
└── Floor 2
    └── Zone F ...

Identifier format: {zone}-{type_prefix}{number}
  type_prefix: "" (standard), "D" (disabled), "E" (EV), "M" (motorcycle)
```

### 3.6 How Parquery and Cleverciti Handle ROI at Scale

- **Parquery**: Camera calibration performed during installation. Parking zones are drawn once on a reference image. AI processes de-identified snapshots (privacy-compliant). Single camera covers up to 300 bays. ROI config stored server-side.
- **Cleverciti**: Proprietary overhead sensors with built-in multi-space detection. No manual ROI drawing needed — sensor auto-detects bay grid from overhead view. Claimed to work in "nearly any environmental condition."

Our approach combines the best of both: web-based ROI drawing (like Parquery) with optional semi-automatic bay discovery (like Cleverciti's auto-detection but using our vehicle detector instead of proprietary hardware).

---

## 4. Occupancy Engine Design

### 4.1 Detection to ROI Matching

```python
def match_detections_to_bays(detections, bay_polygons, method="iou"):
    """
    Match vehicle detections to parking bay polygons.

    Methods:
    - "iou": Intersection over Union between detection bbox and bay polygon
    - "centroid": Detection centroid inside bay polygon
    - "area_overlap": Fraction of bay area covered by detection
    """
    bay_status = {}
    for bay in bay_polygons:
        best_score = 0.0
        best_det = None
        for det in detections:
            if method == "iou":
                score = polygon_iou(det.bbox_polygon, bay.polygon)
            elif method == "centroid":
                cx, cy = det.centroid
                score = 1.0 if point_in_polygon(cx, cy, bay.polygon) else 0.0
            elif method == "area_overlap":
                score = intersection_area(det.bbox_polygon, bay.polygon) / bay.area

            if score > best_score:
                best_score = score
                best_det = det

        bay_status[bay.id] = {
            "score": best_score,
            "detection": best_det,
            "raw_status": "occupied" if best_score > MATCH_THRESHOLD else "vacant"
        }
    return bay_status
```

**Threshold selection:**
- IoU method: threshold = 0.15 (low, because bay polygon and detection bbox don't perfectly align)
- Centroid method: binary (in/out), good for well-separated bays
- Area overlap: threshold = 0.30 (bay must be ≥30% covered)

**Recommendation**: Use **area_overlap** as primary method — it handles partial visibility better than IoU and is less sensitive to exact bbox shape than centroid. Fall back to centroid for motorcycle-sized detections.

### 4.2 Temporal Smoothing

```python
class TemporalSmoother:
    """
    Sliding window + asymmetric hysteresis for flicker prevention.
    """
    def __init__(self, window_size=5, occupy_threshold=3, vacate_threshold=4):
        self.window_size = window_size
        self.occupy_threshold = occupy_threshold  # frames needed: vacant → occupied
        self.vacate_threshold = vacate_threshold  # frames needed: occupied → vacant
        self.history = defaultdict(lambda: deque(maxlen=window_size))
        self.state = {}  # bay_id → "occupied" | "vacant"

    def update(self, bay_id, raw_status):
        self.history[bay_id].append(raw_status)

        occupied_count = sum(1 for s in self.history[bay_id] if s == "occupied")
        current = self.state.get(bay_id, "vacant")

        if current == "vacant" and occupied_count >= self.occupy_threshold:
            self.state[bay_id] = "occupied"
            return "occupied", True  # status, changed
        elif current == "occupied" and (self.window_size - occupied_count) >= self.vacate_threshold:
            self.state[bay_id] = "vacant"
            return "vacant", True

        return current, False  # no change
```

### 4.3 Per-Bay State Machine

```
                    ┌──────────────┐
                    │   VACANT     │
                    │  (green LED) │
                    └──────┬───────┘
                           │ detection score > threshold
                           │ for occupy_threshold frames
                           ▼
                    ┌──────────────┐
                    │  ENTERING    │  (transient, 2-5s)
                    │  (yellow LED)│
                    └──────┬───────┘
                           │ score persists > threshold
                           │ for confirm_frames (3)
                           ▼
                    ┌──────────────┐
                    │  OCCUPIED    │
                    │  (red LED)   │
                    └──────┬───────┘
                           │ detection score < threshold
                           │ for vacate_threshold frames
                           ▼
                    ┌──────────────┐
                    │  LEAVING     │  (transient, 2-5s)
                    │  (yellow LED)│
                    └──────┬───────┘
                           │ score persists < threshold
                           │ for confirm_frames (4)
                           ▼
                    ┌──────────────┐
                    │   VACANT     │
                    └──────────────┘

Special transitions:
  ENTERING → VACANT  (car drove through, did not park)
  OCCUPIED → OCCUPIED (re-detections maintain state)
  Any → UNKNOWN (camera offline / model failure)
```

### 4.4 Handling Transient Events

| Event | How Detected | How Filtered |
|---|---|---|
| Car driving through (not parking) | Detection appears for <3 frames then disappears | `ENTERING → VACANT` (never reaches OCCUPIED) |
| Pedestrian walking past | Detection class = "person" not "vehicle" | Excluded from bay matching (only match vehicle classes) |
| Delivery truck temporarily blocking view | Multiple bays go UNKNOWN simultaneously | If >50% bays in view change state simultaneously, hold previous state for 30s |
| Camera shake / vibration | All detections shift position | If >80% of detection centroids shift by similar vector, ignore frame |

### 4.5 Confidence Scoring

Each bay status update includes a confidence score:

```python
confidence = min(
    detection_confidence,           # model's objectness × class prob
    roi_match_score,                # area overlap or IoU with bay
    temporal_consistency,           # fraction of window agreeing
)
# Range: 0.0 - 1.0
# Publish only if confidence > 0.5
# Flag for review if 0.5 < confidence < 0.7
```

### 4.6 Zone/Floor/Lot Aggregation

```python
def aggregate_counts(bay_statuses, hierarchy):
    """Roll up bay status to zone → floor → lot level."""
    counts = defaultdict(lambda: {"total": 0, "occupied": 0, "vacant": 0, "unknown": 0})

    for bay_id, status in bay_statuses.items():
        zone = hierarchy[bay_id]["zone"]
        floor = hierarchy[bay_id]["floor"]
        lot = hierarchy[bay_id]["lot"]

        for level_key in [f"bay:{bay_id}", f"zone:{zone}", f"floor:{floor}", f"lot:{lot}"]:
            counts[level_key]["total"] += 1
            counts[level_key][status] += 1

    return counts
```

### 4.7 Update Frequency

| Event | Publish Rate |
|---|---|
| Bay status change | Immediately (on state transition) |
| Zone count summary | Every 10 seconds (periodic) |
| Full lot snapshot | Every 60 seconds (periodic) |
| Camera health | Every 30 seconds (periodic) |
| Analytics rollup | Every 5 minutes (batch) |

**Bandwidth estimate**: 7,000 bays × ~2 status changes/hour × ~100 bytes = ~1.4 MB/hour for bay events. Zone summaries add ~50 KB/hour. Total MQTT traffic: <5 MB/hour — negligible on any network.

---

## 5. ANPR Integration Design

### 5.1 Entry/Exit Gate Camera Placement

| Parameter | Recommended | Notes |
|---|---|---|
| Camera type | 2MP IR bullet, 940nm | Invisible IR, no driver distraction |
| Height | 1.0-1.5m (plate level) | Direct view of rear plate at entry, front at exit |
| Angle | 15-30 degrees horizontal | Avoid extreme perspective distortion |
| Distance | 3-5m from barrier | Plate occupies ≥100px width in frame |
| Shutter speed | ≤ 1/500s | Freeze motion at 5-20 km/h approach |
| Frame rate | 15-30 FPS | Capture 3-5 frames per vehicle approach |
| Illumination | 940nm IR LED array, 10-15m range | Supplement camera built-in IR |
| Trigger | Inductive loop or beam break | Trigger capture only when vehicle present |

### 5.2 Plate Matching: Entry to Exit

```python
class SessionManager:
    def on_entry(self, plate_text, confidence, timestamp, camera_id, snapshot_url):
        session = {
            "plate": plate_text,
            "entry_time": timestamp,
            "entry_camera": camera_id,
            "entry_confidence": confidence,
            "entry_snapshot": snapshot_url,
            "status": "active"
        }
        # Check permit whitelist
        permit = self.db.get_permit(plate_text)
        if permit:
            session["permit_type"] = permit.type  # monthly, seasonal, disabled
            session["prepaid"] = True

        self.db.create_session(session)
        # Open barrier (if gated)
        self.gate_controller.open(camera_id)

    def on_exit(self, plate_text, confidence, timestamp, camera_id, snapshot_url):
        # Exact match first
        session = self.db.find_active_session(plate_text)

        if not session:
            # Fuzzy match: Levenshtein distance ≤ 2
            candidates = self.db.find_fuzzy_sessions(plate_text, max_distance=2)
            if len(candidates) == 1:
                session = candidates[0]
                session["exit_fuzzy_matched"] = True
            elif len(candidates) > 1:
                # Ambiguous: queue for manual review
                self.queue_manual_review(plate_text, candidates, snapshot_url)
                self.gate_controller.open(camera_id)  # don't block traffic
                return
            else:
                # No match: log orphan exit, open gate
                self.log_orphan_exit(plate_text, timestamp, snapshot_url)
                self.gate_controller.open(camera_id)
                return

        # Calculate fee
        duration = timestamp - session["entry_time"]
        fee = self.calculate_fee(duration, session.get("permit_type"))

        session["exit_time"] = timestamp
        session["exit_camera"] = camera_id
        session["duration"] = duration
        session["fee"] = fee
        session["status"] = "completed"

        self.db.update_session(session)
        self.gate_controller.open(camera_id)
```

### 5.3 Handling Misreads

| Scenario | Strategy |
|---|---|
| Confidence 0.85-1.0 | Accept as-is |
| Confidence 0.60-0.85 | Accept + flag for review |
| Confidence <0.60 | Reject, capture multi-frame, take best |
| No match on exit | Fuzzy match (Levenshtein ≤ 2), then manual queue |
| Multiple entry matches | Disambiguate by entry time proximity |
| Complete OCR failure | Log snapshot, open gate, manual review queue |

**Expected misread rate**: 3-7% initially, dropping to 1-3% after fine-tuning on Malaysian plates for 2-3 months. Manual review queue handles 50-200 events/day initially.

### 5.4 Whitelist/Blacklist

```yaml
# configs/parking/permits.yaml
whitelists:
  monthly_pass:
    plates: ["QSA 1234", "QSB 5678", ...]
    bay_zone: "A"  # assigned zone (optional)
    valid_until: "2026-12-31"

  disabled:
    plates: ["QSA 9999"]
    bay_type: "disabled"
    valid_until: "2027-06-30"

  staff:
    plates: ["QSA 1111", "QSA 2222"]
    bay_zone: "STAFF"
    fee_exempt: true

blacklists:
  banned:
    plates: ["ABC 1234"]
    reason: "Repeated violations"
    action: "deny_entry"  # deny_entry | alert_only
```

### 5.5 Payment Integration

**Fee calculation:**

```python
def calculate_fee(duration_minutes, permit_type, rate_config):
    if permit_type in ("monthly", "staff"):
        return 0.0  # prepaid / exempt

    # Bintulu typical rates (example)
    if duration_minutes <= rate_config.free_minutes:  # e.g., 15 min free
        return 0.0

    hours = math.ceil(duration_minutes / 60)
    if hours <= rate_config.flat_rate_hours:  # e.g., first 2 hours
        return rate_config.flat_rate  # e.g., RM 2.00

    extra_hours = hours - rate_config.flat_rate_hours
    fee = rate_config.flat_rate + (extra_hours * rate_config.hourly_rate)  # e.g., RM 1.00/hr
    return min(fee, rate_config.daily_max)  # e.g., max RM 10.00/day
```

**Malaysian payment integration:**

| Payment Method | Integration Approach | API/SDK |
|---|---|---|
| **Touch 'n Go eWallet** | PPRO payment infrastructure or direct TNG API | TNG Open API |
| **Boost** | PPRO or direct Boost merchant API | Boost Merchant API |
| **GrabPay** | PPRO or GrabPay merchant integration | GrabPay API |
| **DuitNow QR** | Bank Negara Malaysia's universal QR standard | PayNet DuitNow API |
| **FPX (online banking)** | Direct bank transfer | PayNet FPX API |
| **Cash (legacy)** | Exit pay station with coin/note acceptor | Hardware integration |
| **BorneoParkBTU** | Existing Bintulu digital parking app | BDA partnership/API |

**Recommendation**: Integrate with **PPRO** as a single payment gateway — it already supports Touch 'n Go, Boost, and GrabPay in Malaysia. Add **DuitNow QR** for bank transfers. Explore partnership with **BorneoParkBTU** (Bintulu's existing digital parking app by BDA) to extend rather than replace the existing infrastructure.

---

## 6. Backend and Dashboard

### 6.1 REST API Design

```
Base URL: https://api.parking.example.com/v1

── Occupancy ──
GET    /occupancy/bays                    # All bays (filterable: zone, floor, status)
GET    /occupancy/bays/{bay_id}           # Single bay status + history
GET    /occupancy/zones                   # All zone summaries
GET    /occupancy/zones/{zone_id}         # Zone detail with bay list
GET    /occupancy/summary                 # Facility-wide counts
WS     /occupancy/stream                  # WebSocket: real-time bay changes

── ANPR ──
GET    /anpr/sessions                     # Active + recent sessions (paginated)
GET    /anpr/sessions/{session_id}        # Session detail (entry/exit/fee)
POST   /anpr/sessions/{session_id}/review # Manual plate correction
GET    /anpr/review-queue                 # Pending manual reviews
POST   /anpr/whitelist                    # Add plate to whitelist
DELETE /anpr/whitelist/{plate}            # Remove from whitelist

── Violations ──
GET    /violations                        # All violations (filterable)
GET    /violations/{id}                   # Violation detail + evidence
PATCH  /violations/{id}                   # Update status (reviewed, fined, dismissed)
GET    /violations/stats                  # Violation statistics

── Analytics ──
GET    /analytics/occupancy/history       # Historical occupancy (hourly/daily/weekly)
GET    /analytics/peak-hours              # Peak occupancy analysis
GET    /analytics/turnover                # Bay turnover rates
GET    /analytics/duration-distribution   # Parking duration histogram

── Alerts ──
GET    /alerts                            # All active alerts
POST   /alerts/{id}/acknowledge           # Acknowledge alert
GET    /alerts/safety                     # Safety-specific alerts

── Config ──
GET    /config/cameras                    # Camera list + status
GET    /config/cameras/{id}/rois          # Bay ROI polygons for camera
PUT    /config/cameras/{id}/rois          # Update ROI polygons
GET    /config/zones                      # Zone hierarchy

── Health ──
GET    /health                            # System health summary
GET    /health/edges                      # All edge devices status
GET    /health/edges/{id}                 # Single edge device metrics
```

### 6.2 Database Schema

**PostgreSQL (transactional data):**

```sql
-- Facility hierarchy
CREATE TABLE lots (id SERIAL PRIMARY KEY, name TEXT, address TEXT, timezone TEXT);
CREATE TABLE floors (id SERIAL PRIMARY KEY, lot_id INT REFERENCES lots, name TEXT, level INT);
CREATE TABLE zones (id SERIAL PRIMARY KEY, floor_id INT REFERENCES floors, name TEXT, capacity INT);
CREATE TABLE bays (
    id TEXT PRIMARY KEY,          -- e.g., "A-001"
    zone_id INT REFERENCES zones,
    type TEXT DEFAULT 'standard', -- standard, disabled, ev, motorcycle
    polygon JSONB,                -- [[x,y], ...] normalized coords
    camera_id TEXT,
    current_status TEXT DEFAULT 'vacant',
    last_updated TIMESTAMPTZ
);

-- ANPR sessions
CREATE TABLE parking_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plate_text TEXT NOT NULL,
    entry_time TIMESTAMPTZ NOT NULL,
    exit_time TIMESTAMPTZ,
    entry_camera TEXT,
    exit_camera TEXT,
    entry_confidence FLOAT,
    exit_confidence FLOAT,
    entry_snapshot TEXT,           -- S3 URL
    exit_snapshot TEXT,
    duration INTERVAL,
    fee DECIMAL(10,2),
    payment_status TEXT DEFAULT 'pending',
    payment_method TEXT,
    permit_type TEXT,
    fuzzy_matched BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_sessions_plate ON parking_sessions(plate_text);
CREATE INDEX idx_sessions_entry ON parking_sessions(entry_time);
CREATE INDEX idx_sessions_active ON parking_sessions(exit_time) WHERE exit_time IS NULL;

-- Violations
CREATE TABLE violations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    type TEXT NOT NULL,           -- double_parking, fire_lane, overtime, etc.
    bay_id TEXT REFERENCES bays,
    plate_text TEXT,
    evidence JSONB,               -- array of {type, url, timestamp}
    status TEXT DEFAULT 'pending', -- pending, reviewed, fined, dismissed
    detected_at TIMESTAMPTZ NOT NULL,
    reviewed_at TIMESTAMPTZ,
    reviewed_by TEXT,
    fine_amount DECIMAL(10,2),
    notes TEXT
);

-- Permits / Whitelists
CREATE TABLE permits (
    id SERIAL PRIMARY KEY,
    plate_text TEXT NOT NULL,
    type TEXT NOT NULL,           -- monthly, seasonal, staff, disabled
    zone_restriction TEXT,
    valid_from DATE,
    valid_until DATE,
    fee_exempt BOOLEAN DEFAULT FALSE,
    active BOOLEAN DEFAULT TRUE
);
CREATE INDEX idx_permits_plate ON permits(plate_text) WHERE active = TRUE;

-- Edge devices
CREATE TABLE edge_devices (
    id TEXT PRIMARY KEY,
    location TEXT,
    zone_id INT REFERENCES zones,
    ip_address INET,
    model TEXT,                   -- "jetson_orin_nx_16gb"
    firmware_version TEXT,
    last_heartbeat TIMESTAMPTZ,
    status TEXT DEFAULT 'online'
);

-- Cameras
CREATE TABLE cameras (
    id TEXT PRIMARY KEY,
    edge_device_id TEXT REFERENCES edge_devices,
    type TEXT,                    -- occupancy, anpr_ir, safety
    rtsp_url TEXT,
    resolution TEXT,              -- "1920x1080"
    bay_count INT,                -- number of bays visible
    roi_config JSONB,
    status TEXT DEFAULT 'online'
);
```

**TimescaleDB (time-series — extends PostgreSQL):**

```sql
-- Bay status history (hypertable, auto-partitioned by time)
CREATE TABLE bay_status_history (
    time TIMESTAMPTZ NOT NULL,
    bay_id TEXT NOT NULL,
    status TEXT NOT NULL,        -- vacant, occupied, unknown
    confidence FLOAT,
    detection_count INT,         -- vehicles detected in bay
    source TEXT                  -- camera_id
);
SELECT create_hypertable('bay_status_history', 'time');
CREATE INDEX idx_bay_history ON bay_status_history(bay_id, time DESC);

-- Zone occupancy snapshots (hypertable)
CREATE TABLE zone_occupancy (
    time TIMESTAMPTZ NOT NULL,
    zone_id INT NOT NULL,
    total INT,
    occupied INT,
    vacant INT,
    occupancy_rate FLOAT
);
SELECT create_hypertable('zone_occupancy', 'time');

-- Edge device metrics (hypertable)
CREATE TABLE edge_metrics (
    time TIMESTAMPTZ NOT NULL,
    edge_id TEXT NOT NULL,
    cpu_temp FLOAT,
    gpu_temp FLOAT,
    gpu_utilization FLOAT,
    memory_used_mb INT,
    inference_fps FLOAT,
    stream_count INT
);
SELECT create_hypertable('edge_metrics', 'time');

-- Continuous aggregates for fast analytics
CREATE MATERIALIZED VIEW hourly_occupancy
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS hour,
    zone_id,
    AVG(occupancy_rate) AS avg_occupancy,
    MAX(occupancy_rate) AS peak_occupancy,
    MIN(occupancy_rate) AS min_occupancy
FROM zone_occupancy
GROUP BY hour, zone_id;
```

### 6.3 Real-Time Updates

| Method | Use Case | Implementation |
|---|---|---|
| **WebSocket** | Dashboard live map, bay status changes | FastAPI WebSocket endpoint; Redis PubSub as broker |
| **SSE (Server-Sent Events)** | Mobile app occupancy feed (simpler) | FastAPI StreamingResponse; unidirectional |
| **Polling** | Low-priority analytics, health checks | REST endpoint, 10-60s interval |

**Recommendation**: WebSocket for the operations dashboard (bidirectional, low latency). SSE for the mobile app (simpler, works through proxies/CDNs). Polling only for health monitoring.

### 6.4 Dashboard Features

| Tab | Features |
|---|---|
| **Live Map** | Interactive lot map with bay-level color coding (green/yellow/red); click bay for detail; real-time updates via WebSocket |
| **Camera Feeds** | Grid view of camera streams; click to enlarge; overlay detection boxes and bay status |
| **ANPR Monitor** | Live entry/exit events; session search by plate; manual review queue |
| **Violations** | Pending violations with evidence; review workflow; fine issuance |
| **Analytics** | Occupancy trends (hourly/daily/weekly); peak hour heatmap; bay turnover |
| **Alerts** | Safety alerts (loitering, intrusion); system alerts (edge offline, camera down); acknowledge workflow |
| **Configuration** | Camera ROI editor; zone management; permit management; rate configuration |
| **Reports** | Occupancy summary; violation summary; session summary; export to CSV/PDF |

### 6.5 Multi-Tenant Support

If managing multiple parking facilities, use tenant isolation:

```
/v1/{tenant_id}/occupancy/...
/v1/{tenant_id}/anpr/...

Database: schema-per-tenant (PostgreSQL schemas) or row-level security
MQTT: topic prefix per tenant: {tenant_id}/parking/occupancy/...
```

---

## 7. Edge Deployment Optimization

### 7.1 Multi-Camera Inference on Jetson Orin NX

The Jetson Orin NX 16GB has:
- GPU: 1024 CUDA cores, 32 Tensor Cores (Ampere)
- DLA: 2x NVDLA v2.0 engines
- CPU: 8x Arm Cortex-A78AE
- Memory: 16GB LPDDR5 (102.4 GB/s)
- INT8 performance: ~100 TOPS (GPU+DLA combined)

**Stream capacity estimate:**

| Model | INT8 Latency | Streams @5FPS | Streams @3FPS | Streams @1FPS |
|---|---|---|---|---|
| YOLOX-Tiny (occupancy) | ~3ms | 12 (on GPU) | 20 | 60 |
| YOLOX-Tiny (plate det) | ~3ms | Shared with occupancy | — | — |
| PaddleOCR (plate OCR) | ~5ms | On-demand only | — | — |
| MobileNetV3 (slot classify) | ~0.3ms | On-demand only | — | — |
| YOLOX-Tiny (person safety) | ~3ms | Shared with occupancy | — | — |

**DeepStream pipeline architecture:**

```
┌──────────────────────────────────────────────────────────────┐
│  DeepStream Pipeline (single process, multi-stream)          │
│                                                               │
│  Stream 1 ─┐                                                 │
│  Stream 2 ─┤    ┌──────────┐    ┌──────────┐                │
│  Stream 3 ─┼───►│ nvstreammux├──►│ nvinfer  │──► Vehicle     │
│  ...       ─┤   │ (batch 8) │   │ YOLOX-T  │    detections  │
│  Stream 8 ─┘    └──────────┘    │ (TensorRT│                │
│                                  │  INT8)   │                │
│                                  └────┬─────┘                │
│                                       │                      │
│                          ┌────────────┼──────────┐           │
│                          ▼            ▼          ▼           │
│                   ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│                   │Occupancy │ │ANPR crop │ │Safety    │    │
│                   │Engine    │ │+ OCR     │ │Engine    │    │
│                   │(Python)  │ │(nvinfer2)│ │(Python)  │    │
│                   └──────────┘ └──────────┘ └──────────┘    │
│                          │            │          │           │
│                          └────────────┼──────────┘           │
│                                       ▼                      │
│                                ┌──────────┐                  │
│                                │ MQTT Pub │                  │
│                                └──────────┘                  │
└──────────────────────────────────────────────────────────────┘
```

### 7.2 Inference Runtime Comparison

| Runtime | Pros | Cons | Recommendation |
|---|---|---|---|
| **TensorRT** | Fastest INT8 on Jetson; DLA support; batch inference | Jetson-only; complex build | Primary for production |
| **DeepStream** | Multi-stream mux; GStreamer pipeline; TensorRT under hood | Learning curve; Jetson-focused | Use for stream management |
| **ONNX Runtime** | Cross-platform; easy development | Slower than TensorRT on Jetson | Development + fallback |

**Recommendation**: Use **DeepStream 7.x** for the production multi-stream pipeline (handles RTSP decode, batching, inference via TensorRT, and metadata routing). Use **ONNX Runtime** during development and testing on non-Jetson hardware.

### 7.3 Memory Management

| Component | Memory Usage | Notes |
|---|---|---|
| YOLOX-Tiny INT8 engine | ~50MB | Single model shared across streams |
| PaddleOCR INT8 | ~30MB | Loaded on-demand for ANPR gates |
| MobileNetV3 INT8 | ~10MB | Loaded on-demand for ambiguous bays |
| DeepStream pipeline (8 streams) | ~2GB | Decode buffers + metadata |
| CUDA context | ~500MB | Base overhead |
| OS + application | ~2GB | JetPack + Python + MQTT |
| **Total** | **~5GB of 16GB** | **~11GB headroom** |

Headroom allows for future model additions (e.g., action recognition) without hardware upgrade.

### 7.4 Frame Rate Optimization

| Task | Process Rate | Rationale |
|---|---|---|
| Occupancy detection | Every 3rd frame (≈3 FPS from 10 FPS stream) | Bay status changes slowly; temporal smoothing covers gaps |
| ANPR (gate cameras) | Every frame (15-30 FPS) | Vehicles pass quickly; need best frame for plate read |
| Safety (person tracking) | Every 5th frame (≈2 FPS) | Loitering threshold is minutes; 2 FPS sufficient for tracking |
| Violation evidence capture | Burst: 5 FPS for 10s | Triggered only on violation detection |

### 7.5 Power Budget and Thermal

| Mode | Power Draw | Thermal |
|---|---|---|
| Orin NX 16GB (15W mode) | 10-15W | Passive heatsink sufficient up to 35C ambient |
| Orin NX 16GB (25W mode, MAXN) | 15-25W | Active fan required in Bintulu climate (28-33C avg) |
| Full load (8 streams + 3 models) | ~20W | IP65 enclosure with fan; thermal throttle at 97C junction |

**Bintulu climate consideration**: Average temperature 27-33C, high humidity (80-90%). Use IP65-rated enclosures with active cooling fan and conformal-coated PCBs.

### 7.6 OTA Model Update Strategy

Using **Mender** (open-source OTA) for Jetson:

```
┌─────────────────────────────────────────────┐
│  OTA Update Pipeline                         │
│                                              │
│  1. Train new model in cloud                 │
│  2. Export to ONNX → TensorRT INT8           │
│  3. Validate on test set (accuracy ≥ prev)   │
│  4. Package as Mender artifact               │
│  5. Deploy to staging group (5% of fleet)    │
│  6. Monitor accuracy metrics for 48h         │
│  7. If OK → phased rollout (25% → 50% → 100%)│
│  8. If degraded → automatic rollback (A/B)   │
│                                              │
│  Artifact contents:                          │
│  ├── model.trt (TensorRT engine)             │
│  ├── model_config.yaml                       │
│  ├── version.json                            │
│  └── pre/post-install scripts                │
└─────────────────────────────────────────────┘
```

**Alternative**: **Allxon** — commercial fleet management for Jetson with GUI-based OTA, monitoring dashboard, and mass deployment. Simpler but is a commercial product.

---

## 8. Guidance System

### 8.1 LED Sign Types and Protocols

| Sign Type | Protocol | Data | Placement |
|---|---|---|---|
| **Zone count display** (7-segment, 4-digit) | RS485 @ 9600bps | Available count per zone | Zone entry points |
| **Directional arrow + count** | RS485 @ 9600bps | Arrow direction + count | Decision forks in driving aisles |
| **Individual bay indicator** (green/red LED) | RS485 daisy-chain or wired | 1-bit per bay | Above each bay (indoor structures) |
| **Full-color VMS** (LED matrix) | Ethernet or RS485 | Text/graphics ("FULL", "300 SPACES") | Facility entrance |
| **Exterior entrance sign** | Ethernet (more bandwidth) | Total available, "OPEN"/"FULL" | Street-facing |

**RS485 integration:**

```python
# Zone count display controller (RS485)
import serial

class LEDSignController:
    def __init__(self, port="/dev/ttyUSB0", baudrate=9600):
        self.serial = serial.Serial(port, baudrate, timeout=1)

    def update_zone_count(self, sign_address: int, available: int):
        """Send available count to RS485 LED sign."""
        if available <= 0:
            display_text = "FULL"
        else:
            display_text = f"{available:4d}"

        # Protocol: STX + ADDR + DATA + ETX + CHECKSUM
        # (Exact protocol varies by manufacturer — Signal-Tech, Nortech, etc.)
        packet = bytes([0x02, sign_address]) + display_text.encode() + bytes([0x03])
        checksum = sum(packet) & 0xFF
        self.serial.write(packet + bytes([checksum]))
```

Up to 8 signs on a single RS485 bus (300m max distance). Each sign has a unique address (1-8).

### 8.2 Color Coding Logic

```python
def get_zone_color(available, total):
    occupancy_rate = 1 - (available / total)
    if occupancy_rate < 0.50:
        return "GREEN"    # >50% free — plenty of space
    elif occupancy_rate < 0.80:
        return "YELLOW"   # 20-50% free — filling up
    elif occupancy_rate < 1.00:
        return "RED"      # <20% free — almost full
    else:
        return "RED_BLINK"  # completely full
```

### 8.3 Mobile App Features

| Feature | Description | Technology |
|---|---|---|
| **Real-time map** | Interactive lot map with bay-level colors | React Native MapView + WebSocket |
| **Bay finder** | "Find me a spot" → nearest available bay | Backend query + pathfinding |
| **Navigation** | Turn-by-turn to available bay | Indoor: BLE beacons + dead reckoning; Outdoor: GPS |
| **Payment** | In-app payment via TNG/Boost/GrabPay/DuitNow | PPRO SDK integration |
| **E-receipt** | Digital parking receipt | PDF generation + push notification |
| **Pre-booking** | Reserve a bay for future arrival | Session reservation with timeout |
| **Violation history** | View any fines linked to plate | Plate-based lookup |
| **Loyalty/rewards** | Points for cashless payment, off-peak parking | Gamification module |

### 8.4 External Navigation Integration

- **Google Maps**: Publish available counts via Google Maps Platform "Parking Availability" API (requires partnership)
- **Waze**: "Waze for Cities" data feed — publish lot location + availability
- **Apple Maps**: MapKit integration for iOS users
- **In-app deep link**: Navigation apps hand off to parking app on arrival for indoor guidance

---

## 9. Performance Metrics and Evaluation

### 9.1 Occupancy Accuracy

**Ground truth collection methodology:**

1. **Manual annotation**: Annotators label bay status on sampled frames (every 5 minutes for 24 hours across 20 representative cameras). This creates ~5,760 labeled frames.
2. **Stratified sampling**: Sample frames at peak, off-peak, night, rain, and transition periods.
3. **Inter-annotator agreement**: Two annotators per frame; resolve disagreements with third.
4. **Metrics**:
   - Per-bay accuracy: `correct_predictions / total_observations`
   - Per-bay precision and recall (for "occupied" class)
   - False positive rate: vacant bay classified as occupied
   - False negative rate: occupied bay classified as vacant
   - State transition accuracy: correct detection of entering/leaving events

**Targets:**

| Metric | Target | Measurement |
|---|---|---|
| Bay-level accuracy | ≥ 97% | Manual annotation on sampled frames |
| Zone count accuracy | ≥ 99% (±2 bays) | Compare to manual count at random intervals |
| False positive rate | < 2% | Vacant reported as occupied |
| False negative rate | < 3% | Occupied reported as vacant |
| State change latency | < 15 seconds | Time from vehicle stop/start to status update |

### 9.2 ANPR Accuracy

| Metric | Target | Method |
|---|---|---|
| Plate detection rate | ≥ 99% | Count plates detected / vehicles passed (inductive loop ground truth) |
| Character-level accuracy | ≥ 98% | Compare OCR output to manually transcribed plates |
| Plate-level accuracy (exact match) | ≥ 95% | Entire plate text correct |
| Session match rate | ≥ 97% | Entry-exit plate pairs matched automatically |
| Manual review queue | < 3% of sessions | Sessions requiring human intervention |

### 9.3 End-to-End Latency

| Path | Target | Components |
|---|---|---|
| Bay change → dashboard | < 3 seconds | Edge inference (100ms) + MQTT (50ms) + backend (50ms) + WebSocket (50ms) |
| Vehicle entry → gate open | < 2 seconds | ANPR inference (200ms) + validation (50ms) + gate controller (500ms) |
| Violation detection → alert | < 10 seconds | Detection + dwell timer + evidence capture + MQTT |
| Zone count → LED sign | < 5 seconds | Aggregation (1s) + RS485 transmission (100ms) |

### 9.4 Uptime and Reliability

| Component | Target Uptime | Failover |
|---|---|---|
| Edge device | 99.5% (43.8h downtime/yr) | Neighbor edge box takes over streams |
| Backend API | 99.9% (8.8h downtime/yr) | Load-balanced, multi-instance |
| Database | 99.99% (52.6 min/yr) | PostgreSQL streaming replication |
| MQTT broker | 99.9% | EMQX cluster (3 nodes) |
| LED signs | 99.0% | RS485 bus failure shows "---" |
| Camera | 98.0% | 10% overlap coverage mitigates single camera failure |

---

## 10. Deployment Plan for 7,000 Bays

### 10.1 Phased Rollout

```
Phase 0: Lab Prototype (Month 1-2)
├── Set up 1 Jetson Orin NX + 4 cameras in office/lab
├── Train occupancy model on PKLot + CNRPark + custom Bintulu images
├── Build ANPR pipeline with Malaysian plate dataset
├── Develop backend API + basic dashboard
├── Integration test: camera → edge → backend → dashboard
└── Deliverable: Working demo on 4 cameras

Phase 1: Pilot Zone — 200 bays (Month 3-5)
├── Deploy at one surface lot zone in Bintulu
├── 5-6 occupancy cameras + 2 ANPR gate cameras
├── 1 Jetson Orin NX edge box
├── 2 LED zone count signs
├── 1 PoE switch + fiber uplink
├── Validate: occupancy accuracy, ANPR accuracy, latency
├── Collect ground truth data for 4 weeks
├── Fine-tune models on Bintulu-specific conditions
├── User acceptance testing (UAT) with BDA stakeholders
└── Deliverable: Pilot report with accuracy metrics

Phase 2: Expanded Pilot — 1,000 bays (Month 6-8)
├── Expand to 4-5 zones (mix of outdoor + covered)
├── 20-25 cameras + 6 ANPR cameras
├── 4 Jetson Orin NX edge boxes
├── 10 LED signs
├── Mobile app beta launch (Android + iOS)
├── Payment integration (Touch 'n Go, Boost)
├── Safety features enabled (loitering, intrusion)
├── Violation detection pilot
├── Stress test: peak occupancy scenarios
└── Deliverable: Validated system ready for scale

Phase 3: Half Deployment — 3,500 bays (Month 9-13)
├── Deploy to 50% of all zones
├── ~80 cameras + 15 ANPR cameras
├── 15 edge boxes
├── 40 LED signs
├── Full payment integration
├── Violation enforcement active
├── Public mobile app launch
├── BorneoParkBTU app integration
├── Analytics dashboard for BDA management
└── Deliverable: Half-facility operational

Phase 4: Full Deployment — 7,000 bays (Month 14-18)
├── Complete remaining zones
├── ~200 cameras + 30 ANPR cameras
├── 30-35 edge boxes
├── 80+ LED signs
├── Full network backbone
├── OTA update system operational
├── Monitoring and alerting fully configured
├── Training and handover to BDA operations team
└── Deliverable: Full 7,000-bay operational system

Phase 5: Optimization & Handover (Month 19-24)
├── Model re-training on 6 months of Bintulu data
├── Edge case refinement (monsoon season, festivals)
├── Performance tuning based on analytics
├── Documentation and operations manual
├── Staff training (L1 support, L2 troubleshooting)
├── Warranty period begins
└── Deliverable: Fully optimized, handed over system
```

### 10.2 Camera Placement Guidelines

| Parameter | Outdoor Surface Lot | Multi-Level Structure | ANPR (Gate) |
|---|---|---|---|
| **Height** | 6-8m (light pole mount) | 3-4m (ceiling mount) | 1.0-1.5m (plate level) |
| **Angle** | 30-45 degrees down | 45-60 degrees down (near-overhead) | 15-30 degrees horizontal |
| **Coverage** | 40-60 bays per camera | 25-40 bays per camera | 1 lane per camera |
| **Resolution** | 4MP (2560x1440) | 2MP (1920x1080) | 2MP IR (1920x1080) |
| **Lens** | 2.8-8mm varifocal | 1.8-3.6mm wide-angle | 8-12mm fixed (plate zoom) |
| **IP rating** | IP67 (outdoor, rain) | IP54 (indoor, dust) | IP67 (gate, weather) |
| **PoE** | 802.3at (PoE+, 30W) | 802.3af (PoE, 15W) | 802.3at (PoE+, IR LEDs) |
| **Overlap** | 10-15% between cameras | 10-15% between cameras | N/A |
| **Night** | IR built-in or separate | Not needed (lit) | 940nm IR illuminator |

### 10.3 Network Design

**Three-tier architecture:**

| Tier | Equipment | Connectivity | Purpose |
|---|---|---|---|
| **Access** | 24-port PoE+ switches (802.3at) | Cat6 to cameras (≤100m) | Camera power + data |
| **Aggregation** | 24-port SFP switches (1G fiber) | Single-mode fiber to access switches | Per-zone aggregation |
| **Core** | 48-port SFP+ switch (10G) | Fiber to aggregation + WAN | Datacenter backbone |

**Redundancy:**
- Dual core switches (active-standby)
- Dual uplinks from aggregation to core
- Edge boxes on UPS (30 min battery)
- 4G LTE failover on each edge box (for critical ANPR events)

**Bandwidth estimate per edge box:**
- 8 RTSP streams × 4Mbps (1080p H.264) = 32 Mbps inbound (local network)
- MQTT telemetry: <1 Mbps outbound
- Evidence snapshots: ~5 Mbps burst (during violations)
- Total per edge box to backend: ~10 Mbps sustained, 40 Mbps peak

### 10.4 Timeline Summary

| Phase | Duration | Bays | Cameras | Edge Boxes | Key Milestone |
|---|---|---|---|---|---|
| 0: Lab | 2 months | 0 (simulated) | 4 | 1 | Working demo |
| 1: Pilot | 3 months | 200 | 8 | 1 | Accuracy validation |
| 2: Expanded | 3 months | 1,000 | 31 | 4 | Mobile app + payment |
| 3: Half | 5 months | 3,500 | 95 | 15 | Public launch |
| 4: Full | 5 months | 7,000 | 230 | 35 | Complete deployment |
| 5: Optimize | 6 months | 7,000 | 230 | 35 | Handover |
| **Total** | **24 months** | | | | |

### 10.5 Training and Handover Plan

| Training Module | Audience | Duration | Content |
|---|---|---|---|
| **System overview** | BDA management | 1 day | Architecture, capabilities, KPIs |
| **Dashboard operations** | Operations team | 3 days | Live map, ANPR review, violations, reports |
| **L1 support** | IT helpdesk | 2 days | Common issues, restart procedures, escalation |
| **L2 troubleshooting** | System admins | 5 days | Edge device access, log analysis, camera recalibration |
| **ROI configuration** | Trained operators | 2 days | Web UI for bay polygon editing, validation |
| **Model retraining** | ML engineers (if any) | 5 days | Data collection, annotation, training pipeline, OTA deploy |

### 10.6 Model Card (Deliverables)

Each release produces a model card at `docs/model_cards/traffic_smart_parking.md` and a YAML card at `releases/smart_parking/v<N>/model_card.yaml`.

**Occupancy Detection Model artifacts:**

| Artifact | Format | Path |
|---|---|---|
| PyTorch model | `.pth` | `runs/smart_parking/occupancy/best.pt` |
| ONNX model | `.onnx` | `runs/smart_parking/occupancy/export/parking_yolox_tiny_640_v{N}.onnx` |
| TensorRT engine | `.trt` | `runs/smart_parking/occupancy/export/parking_yolox_tiny_640_v{N}.trt` |
| Training config | `.yaml` | `configs/smart_parking/occupancy/06_training.yaml` |
| Metrics | `.json` | `runs/smart_parking/occupancy/metrics.json` |

**ANPR Model artifacts:**

| Artifact | Format | Path |
|---|---|---|
| Plate detection model | `.pth` | `runs/smart_parking/anpr/plate_detect/best.pt` |
| Plate detection ONNX | `.onnx` | `runs/smart_parking/anpr/plate_detect/export/anpr_plate_yolox_tiny_v{N}.onnx` |
| OCR model (PaddleOCR) | `.pdmodel` | `runs/smart_parking/anpr/ocr/ppocr_v4_finetuned/` |
| Training config | `.yaml` | `configs/smart_parking/anpr/06_training.yaml` |
| Metrics | `.json` | `runs/smart_parking/anpr/metrics.json` |

**Safety Detection Model artifacts:**

| Artifact | Format | Path |
|---|---|---|
| PyTorch model | `.pth` | `runs/smart_parking/safety/best.pt` |
| ONNX model | `.onnx` | `runs/smart_parking/safety/export/safety_yolox_tiny_640_v{N}.onnx` |
| Training config | `.yaml` | `configs/smart_parking/safety/06_training.yaml` |
| Metrics | `.json` | `runs/smart_parking/safety/metrics.json` |

**Documentation deliverables:**
- System architecture document (this document, expanded)
- Operations manual (day-to-day procedures)
- Troubleshooting guide (symptom → diagnosis → fix)
- API reference (OpenAPI/Swagger)
- Network diagram (physical + logical)
- Camera placement map (per-zone)

---

## Appendix A: Bill of Materials (Estimated)

| Item | Qty | Purpose |
|---|---|---|
| Jetson Orin NX 16GB + carrier board | 35 | Edge AI processing |
| IP camera 4MP outdoor (occupancy) | 150 | Outdoor lot bay monitoring |
| IP camera 2MP indoor (occupancy) | 50 | Indoor/multi-level bay monitoring |
| IR camera 2MP (ANPR) | 30 | Entry/exit gate plate capture |
| PoE+ switch 24-port | 28 | Camera power + data |
| Aggregation switch 24-port SFP | 10 | Per-zone aggregation |
| Core switch 48-port SFP+ | 2 | Data center aggregation |
| LED zone count sign (RS485) | 80 | Zone entry point displays |
| Single-mode fiber (OS2) | ~15 km | Backbone runs |
| Cat6 outdoor-rated cable | ~50 km | Camera to PoE switch (≤100m) |
| Edge enclosure (IP65, fan, DIN) | 35 | Outdoor protection for edge devices |
| UPS per cabinet (1kVA) | 28 | 30-min battery backup |
| Server (backend, 2U rack) | 3 | Backend API, database, analytics |

## Appendix B: Key References

- PKLot Dataset: Almeida et al., "PKLot — A robust dataset for parking lot classification" (2015)
- CNRPark-EXT: Amato et al., "Deep learning for decentralized parking lot occupancy detection" (2017)
- ByteTrack: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box" (ECCV 2022)
- YOLOX: Ge et al., "YOLOX: Exceeding YOLO Series in 2021" (arXiv 2021)
- D-FINE: Peng et al., "D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement" (ICLR 2025)
- PaddleOCR: Du et al., "PP-OCRv3: More Attempts for the Improvement of Ultra Lightweight OCR System" (2022)
- LPRNet: Zherzdev & Gruzdev, "LPRNet: License Plate Recognition via Deep Neural Networks" (2018)
- WPOD-Net: Silva & Jung, "License Plate Detection and Recognition in Unconstrained Scenarios" (ECCV 2018)
- Temporal smoothing for parking: "Resource-Efficient Design and Implementation of Real-Time Parking Monitoring System with Edge Device" (Sensors 2025)
- MobileNetV3 for parking: Yuldashev et al., "Parking Lot Occupancy Detection with Improved MobileNetV3" (Sensors 2023)

Sources:
- [Resource-Efficient Parking Monitoring with Edge Device (2025)](https://www.mdpi.com/1424-8220/25/7/2181)
- [Real-Time Parking Space Management on Low-Power Platform](https://pmc.ncbi.nlm.nih.gov/articles/PMC12656557/)
- [Automatic Vision-Based Parking Slot Detection and Occupancy Classification](https://arxiv.org/abs/2308.08192)
- [Parking Lot Occupancy Detection with Improved MobileNetV3](https://www.mdpi.com/1424-8220/23/17/7642)
- [Smart Parking with Pixel-Wise ROI Selection (YOLO variants)](https://arxiv.org/html/2412.01983v2)
- [Parquery — How Parking Space Monitoring Works](https://parquery.com/how-it-works/)
- [Cleverciti — How Smart Parking Works](https://www.cleverciti.com/en/resources/blog/how-does-smart-parking-work)
- [ANPR with YOLOv12 and PaddleOCR (2025)](https://www.mdpi.com/2076-3417/15/14/7833)
- [Malaysian Vehicle Registration Plates — Wikipedia](https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_Malaysia)
- [Plate Recognizer vs OpenALPR Comparison](https://platerecognizer.com/better-than-openalpr/)
- [NVIDIA DeepStream Parallel Inference App](https://github.com/NVIDIA-AI-IOT/deepstream_parallel_inference_app)
- [Maximizing DL Performance on Jetson Orin with DLA](https://developer.nvidia.com/blog/maximizing-deep-learning-performance-on-nvidia-jetson-orin-with-dla/)
- [Jetson Benchmarks](https://developer.nvidia.com/embedded/jetson-benchmarks)
- [EMQX vs Mosquitto MQTT Broker Comparison](https://www.emqx.com/en/blog/emqx-vs-mosquitto-2023-mqtt-broker-comparison)
- [Mender OTA for NVIDIA Jetson](https://mender.io/blog/how-to-leverage-over-the-air-ota-updates-with-nvidia-microservices-for-jetson)
- [Allxon BSP OTA for Jetson](https://www.allxon.com/bsp-ota-updates)
- [Signal-Tech RedStorm Parking Guidance System](https://www.signal-tech.com/redstorm/how-it-works)
- [BorneoParkBTU App — Bintulu Digital Parking](https://dayakdaily.com/borneoparkbtu-app-introduced-for-cashless-parking-payments-in-bintulu-starting-may-1/)
- [PaddleOCR vs EasyOCR vs TrOCR Comparison](https://mljourney.com/optical-character-recognition-trocr-vs-paddleocr-vs-easyocr/)
- [License Plate Detection: EasyOCR vs PaddleOCR vs Tesseract (IEEE 2024)](https://ieeexplore.ieee.org/document/10725878/)
- [PPRO Integrates GrabPay and Touch 'n Go in Malaysia](https://www.ppro.com/news/ppro-integrates-grabpay-and-touch-n-go-in-malaysia/)
- [SenSen AI Curbside Enforcement](https://sensen.ai/curbside-enforcement/)
- [Navigine Smart Parking Indoor Navigation](https://navigine.com/blog/how-does-an-intelligent-parking-system-work/)
- [Designing CCTV Cabling for Large-Scale Facilities](https://www.gcabling.com/designing-cctv-ip-surveillance-cabling-for-large-scale-facilities)
- [Benchmarking DL Models for Object Detection on Edge Devices](https://arxiv.org/html/2409.16808v1)
- [AX650N Third-Generation Intelligent Vision Chip](https://www.axera-tech.com/en/news/2819.html)
- [TimescaleDB for Real-Time Analytics](https://www.timescale.com/)
- [LPRNet via NVIDIA NGC (TensorRT)](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/lprnet)

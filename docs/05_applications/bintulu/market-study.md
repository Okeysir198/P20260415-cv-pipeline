# Bintulu Smart City - Market Study & Gap Analysis

**Date**: 2026-03-19
**Author**: Nguyen Thanh Trung (AI Camera Lead, VIETSOL)
**Purpose**: Technical market research for UC1 (AI Traffic Light) and UC2 (Smart Parking) — benchmark against VIETSOL camera_edge capabilities

---

## Table of Contents

1. [UC1: AI Traffic Light — Market Landscape](#uc1-ai-traffic-light--market-landscape)
2. [UC2: Smart Parking — Market Landscape](#uc2-smart-parking--market-landscape)
3. [VIETSOL camera_edge — Current Capabilities](#vietsol-camera_edge--current-capabilities)
4. [Gap Analysis: What We Can Fulfill vs. What's Missing](#gap-analysis-what-we-can-fulfill-vs-whats-missing)
5. [Recommendations & Roadmap](#recommendations--roadmap)

---

## UC1: AI Traffic Light — Market Landscape

### 1.1 Edge Hardware Platforms

| Platform | AI Performance | Power | Price | Production Deployments |
|----------|---------------|-------|-------|----------------------|
| **NVIDIA Jetson AGX Orin** | 275 TOPS (INT8) | 15-60W | ~$999 (module) | NoTraffic (confirmed NVIDIA partnership) |
| **NVIDIA Jetson Orin NX** | 100-157 TOPS | 10-40W | $399-$599 | Sintrones IBOX-600 (IP66 roadside) |
| **NVIDIA Jetson Orin Nano** | 40-67 TOPS | 7-25W | $199-$249 | Budget intersection deployments |
| **Ambarella CV72S** | Custom CVflow | <3W | $30-60 (OEM) | Smart traffic cameras (Hikvision, Dahua) |
| **Ambarella CV75S** | Supports ViTs/VLMs | <5W | $50-100 (OEM) | Enterprise multi-modal traffic analytics |
| **Hailo-8** | 26 TOPS | <2.5W | $70-100 (M.2) | Smart cameras, single-cam deployments |
| **Hailo-15H/M/L** | 7-20 TOPS (VPU) | Low | Mid-range | AI-centric 4K vision processors |

**Market dominant**: NVIDIA Jetson Orin family (for edge AI boxes), Ambarella CV-series (for smart cameras with embedded AI).

### 1.2 System Architecture

```
[Camera/Radar]  ──>  [Edge AI Processor]  ──>  [Traffic Signal Controller]  ──>  [Signals]
       │                     │                          │
       │                     ▼                          │
       │             Local Analytics                    │
       │          (detection, counting,                 │
       │           queue estimation)                    │
       ▼                     │                          ▼
  [Lidar/Thermal]            ▼                    [NTCIP 1202 / SNMP]
  (optional)         [Cloud Platform]                   │
                     (central ATMS,                     ▼
                      dashboards,              [Cabinet / ATC Unit]
                      analytics)
```

**Key integration protocols:**
| Protocol | Purpose | Used By |
|----------|---------|---------|
| **NTCIP 1202** | US standard for actuated traffic signal controllers (SNMP-based) | Econolite, Siemens, McCain controllers |
| **UTMC** | UK Urban Traffic Management and Control standard | UK SCOOT deployments |
| **MQTT** | Lightweight IoT edge-to-cloud telemetry | Newer AI platforms |
| **DSRC / C-V2X** | Vehicle-to-infrastructure communication | NoTraffic |

**Architecture trend**: Hybrid (edge + cloud) is winning — edge AI for real-time detection (<100ms latency), cloud for network-wide optimization and analytics.

### 1.3 Software & Platforms — Market Leaders

| Vendor | System | Architecture | Key Stats |
|--------|--------|-------------|-----------|
| **Siemens Mobility** | SCOOT/ATCS | Centralized | >11% global market share |
| **Econolite** | Centracs ATMS | Cloud ATMS | 60,000+ intersections, 500+ customers (NA) |
| **NoTraffic** | AI Mobility Platform | Hybrid (Jetson + Cloud) | 200+ agencies, 35 US states, FL statewide |
| **Miovision** | Adaptive + TrafficLink | Hybrid | 60,000+ intersections ecosystem |
| **Vivacity Labs** | Smart Signal Control | Edge sensor + UTC | 6,000+ sensors, 97% accuracy, UK-focused |
| **Surtrac** (Rapid Flow / Miovision) | Decentralized AI | Edge-only | Pittsburgh: 25% travel time reduction |
| **LYT** (formerly Rhythm Engineering) | LYT.speed / LYT.transit | Cloud + edge | Transit/emergency priority specialist |
| **SCATS** (NovaTrans) | SCATS | Two-level | Sydney, Beijing, Shanghai |
| **SWARCO** | Various ATCS | Centralized/hybrid | European market leader |
| **Huawei** | TrafficGo | AI + IoT + Big Data | Chinese market leader |

**No production-grade open-source adaptive signal control exists.** Open-source tools (SUMO-RL, PyTSC, CityFlow) are research/simulation only.

### 1.4 AI Models for Traffic

| Task | Model | Performance | Notes |
|------|-------|-------------|-------|
| **Vehicle detection** | YOLOv8-v12 | 97% precision, 82% mAP@50-95, 70+ FPS | De facto standard |
| **Vehicle detection (edge)** | YOLO-Lite variants | 29.8 FPS on Jetson Nano | Optimized for constrained HW |
| **Vehicle detection (NMS-free)** | RT-DETR, D-FINE | High accuracy, improving edge speed | Growing adoption |
| **Queue length estimation** | YOLO + DeepSORT | 73-88% accuracy | Detection + tracking approach |
| **Traffic density** | CNN regression | Direct density from overhead feed | Used in central systems |
| **Emergency vehicle** | YOLOv8 specialized | mAP 0.96 | + siren audio detection |
| **Multi-lane monitoring** | Per-lane ROI zones | Configurable in software | 360-degree cameras (Miovision) |

**Real-time requirements**: >=15 FPS minimum, 30 FPS preferred. End-to-end latency <200ms for signal phase decisions.

### 1.5 Benchmarks & Proven Results

| Deployment | System | Results |
|------------|--------|---------|
| Pittsburgh, PA | Surtrac | 25% travel time reduction, 40% idle time reduction |
| Tucson, AZ | NoTraffic | 23% average delay reduction, 1.25M+ driver hours saved |
| London, UK | SCOOT | 30% travel time reduction, 50% congestion decrease |
| Las Vegas, NV | Waycare | 17% primary crash reduction |
| China (100 cities) | Big-data adaptive | 11% peak-hour trip time reduction |
| Florida (statewide) | NoTraffic | FDOT-approved statewide AI traffic management |

### 1.6 Pricing & Market Size

| Metric | Value |
|--------|-------|
| Per-intersection cost (hardware + install) | $20,000 - $120,000 |
| Median installation cost (US) | ~$45,000 per intersection |
| Budget systems (ACS Lite) | ~$6,000 per intersection |
| Premium systems (SCOOT) | ~$60,000 per intersection |
| Global ATCS market (2025) | $7.07 billion |
| Projected (2030) | $14.37 billion |
| CAGR | 8.75% - 17.4% |
| Fastest-growing region | Asia-Pacific |

---

## UC2: Smart Parking — Market Landscape

### 2.1 Edge Hardware Platforms

| Platform | AI Performance | Power | Price | Cameras/Device |
|----------|---------------|-------|-------|----------------|
| **NVIDIA Jetson AGX Orin** | 275 TOPS (INT8) | 15-60W | $999-$1,999 | 16 streams @1080p30 |
| **NVIDIA Jetson Orin NX** | 100 TOPS (INT8) | 10-25W | $399-$599 | 18 streams @1080p30 |
| **NVIDIA Jetson Orin Nano** | 40-67 TOPS (INT8) | 7-15W | $199-$249 | 4-8 streams @1080p30 |
| **Hailo-8** | 26 TOPS (INT8) | 2.5W | $70-$100 | 4 streams @FHD |
| **Hailo-8L** | 13 TOPS (INT8) | ~1.5W | $30-$50 | 1-2 streams |
| **Ambarella CV72S** | Custom CVflow | <3W | OEM SoC | 4x 5MP @30fps |
| **Google Coral Edge TPU** | 4 TOPS (INT8) | 2W | $25-$60 | 1-2 streams |

**Sweet spot for parking**: Hailo-8 (Flash Parking already uses it), Ambarella CV72S (embedded in Hikvision/Dahua smart cameras), Jetson Orin NX (for multi-camera edge boxes).

### 2.2 System Architecture

```
[IP Cameras / Smart Cameras]
         │
         │  RTSP / ONVIF (H.265)
         ▼
[Edge AI Device]  ──── Local inference (occupancy detection, ANPR)
         │
         │  MQTT / REST API (metadata only: ~200 bytes/update)
         ▼
[Backend / Cloud Platform]
         │
         ├── Occupancy Database (real-time slot map)
         ├── ANPR Database (entry/exit logs, dwell time)
         ├── Analytics Engine (heatmaps, peak hours, revenue)
         └── Payment / Enforcement Integration
         │
         ▼
[End-User Interfaces]
         ├── Mobile App (find-my-spot, navigation, payment)
         ├── LED Guidance Signs (floor/zone counts)
         ├── Operator Dashboard (live occupancy, alerts)
         └── Enforcement Handheld (violation alerts)
```

**Camera vs Sensor comparison:**
| Aspect | Camera-Based | Ultrasonic/Magnetic Sensor |
|--------|-------------|--------------------------|
| Coverage | 1 camera = 40-100 bays | 1 sensor = 1 bay |
| Cost per bay | $3-$15 (amortized) | $300-$1,400 (installed) |
| Additional data | Vehicle type, color, plate, violations | Binary occupied/vacant only |
| Maintenance | Low (cleaning, software updates) | High (battery, pavement repair) |
| Lifespan | 7-10 years | 3-5 years |
| Accuracy | 95-99% (after learning) | 95-98% |

### 2.3 Software & Platforms — Market Leaders

**Camera-Based Occupancy Detection:**
| Vendor | Approach | Key Feature |
|--------|----------|-------------|
| **Parquery** (Switzerland) | AI on existing CCTV | Retrofit any camera, per-spot SaaS pricing |
| **Cleverciti + ParkHelp** | Proprietary overhead sensors + AI | 600K+ spaces, 700+ sites, 50 countries |
| **Parklio Detect** | AI camera software | 99% accuracy, 70 spots/camera |
| **CVEDIA** | AI video analytics | Synthetic data training, hardware-agnostic |
| **Isarsoft** | Video analytics platform | General-purpose smart parking |

**Full Parking Management (HW + SW):**
| Vendor | Focus |
|--------|-------|
| **Genetec AutoVu** | ALPR enforcement + access control |
| **Hikvision** | Smart cameras with embedded AI |
| **Milesight** | LPR + AI parking cameras, 5,600-camera deployment |
| **Dahua** | Embedded AI parking cameras |
| **Bosch** | AI parking cameras + IoT |
| **Smart Parking Ltd** | ANPR enforcement + guidance (ASX-listed) |

**Market share**: Camera & LPR segment = 42% of 2025 smart parking revenue, growing at 23.3% CAGR (fastest segment).

### 2.4 AI Models for Parking

| Task | Model | Performance |
|------|-------|-------------|
| **Bay occupancy (detection)** | YOLOv8-v11 | 92-98% accuracy, 30+ FPS |
| **Bay occupancy (classification)** | MobileNetV3 | 98% avg accuracy per slot |
| **ANPR/LPR detection** | YOLO v9/v10/v11 | Plate localization |
| **ANPR/LPR recognition** | TrOCR / PaddleOCR / Tesseract | 95-98% plate recognition |
| **Double parking** | YOLO + zone logic | Vehicle outside bay ROI >N seconds |
| **Loitering/intrusion** | Person detection + dwell time | Configurable time thresholds |

**Camera coverage per bay:**
| Scenario | Bays/Camera |
|----------|-------------|
| Outdoor surface lot (pole mount) | 50-100 bays |
| Indoor garage (ceiling mount) | 20-50 bays |
| On-street (pole mount) | 10-30 bays |
| Conservative planning estimate | 40-60 bays |

### 2.5 ANPR Accuracy Benchmarks

| Condition | Accuracy |
|-----------|----------|
| Daytime, clear, good angle | 97-99% |
| Nighttime with IR | 95-98% |
| Rain (with IR) | 93-96% |
| Extreme angle (>30 deg) | 85-92% |
| Dirty/damaged plates | 80-90% |

### 2.6 Scale Estimate for 7,000 Bays

| Component | Camera-Based | Sensor-Based |
|-----------|-------------|-------------|
| Detection hardware | 100-175 cameras x $500-$2K = **$50K-$350K** | 7,000 sensors x $300-$500 = **$2.1M-$3.5M** |
| Edge compute | 10-20 boxes x $500-$2K = **$5K-$40K** | Minimal = **$20K-$50K** |
| Installation labor | **$50K-$150K** | **$700K-$1.4M** (pavement drilling) |
| Backend/software | **$50K-$200K** | **$50K-$200K** |
| Guidance signs | **$100K-$300K** | **$100K-$300K** |
| **Year 1 total** | **$255K-$1.04M** | **$2.97M-$5.45M** |
| **5-year TCO** | **$1M-$3M** | **$5M-$10M** |

**Camera-based is 3-5x cheaper**, and edge processing reduces bandwidth by 100-1000x (metadata only = ~2 Mbps for entire 7,000-bay system).

### 2.7 Pricing & Market Size

| Metric | Value |
|--------|-------|
| Per-camera license (perpetual) | $500-$3,000 |
| Per-bay SaaS | $2-$10/spot/month |
| Enterprise site license | Custom pricing |
| Global smart parking market (2025) | $5.9-$11.2 billion |
| Projected (2033) | $21.5-$64.5 billion |
| CAGR | 10.8%-21.6% |
| Camera/LPR segment share | 42% (fastest growing at 23.3% CAGR) |

---

## VIETSOL camera_edge — Current Capabilities

### 3.1 Detection Models (Production-Ready)

| Model | Type | Params | License | Status |
|-------|------|--------|---------|--------|
| **YOLOX-M** | CNN | 25.3M | Apache 2.0 | Primary detector, self-contained implementation |
| **YOLOX-Tiny/S/L/X** | CNN | Various | Apache 2.0 | All variants available |
| **D-FINE-S** | Transformer | 10M | Apache 2.0 | HF adapter, NMS-free |
| **D-FINE-N/M** | Transformer | 4M/heavier | Apache 2.0 | Lightweight / escalation |
| **RT-DETRv2-R18/R50** | Transformer | 20M | Apache 2.0 | ONNX-reliable export |
| **Generic HF adapter** | Any | Varies | — | Any `ForObjectDetection` model works |

### 3.2 Auxiliary Models

| Model | Task | Params | Status |
|-------|------|--------|--------|
| **RTMPose-S/T/M** | Pose estimation | 3.3-13.6M | Proven on AX650N (4.79ms) |
| **MediaPipe Pose** | 33 3D landmarks | 1.3-3.5M | TFLite |
| **MobileNetV3-Small** | Crop classification | 2.5M | Shoe/helmet |
| **ByteTrack** | Object tracking | — | MIT, single-camera |

### 3.3 Trained Use Cases

| ID | Task | Classes | Dataset | Target mAP@0.5 |
|----|------|---------|---------|----------------|
| a | **Fire Detection** | fire, smoke | 122K | >= 0.85 |
| b | **Helmet Compliance** | person, head_with/without_helmet | 62K | >= 0.92 |
| f | **Safety Shoes** | person, foot_with/without_safety_shoes | 3.7K | >= 0.85 |
| g | **Fall Detection** | person, fallen_person | 17K | >= 0.85 |
| g | **Fall Detection (Pose)** | person (17/33 kpts) | 111+COCO | >= 0.85 |
| h | **Phone Detection** | phone + pose rules | 13K | >= 0.80 |
| i | **Zone Intrusion** | person (COCO pretrained) | — | >= 0.92 |

### 3.4 Pipeline Capabilities

| Capability | Status | Details |
|------------|--------|---------|
| Training pipeline | **Production** | Config-driven, EMA, AMP, grad clip, WandB |
| Evaluation | **Production** | mAP@0.5, mAP@0.5:0.95, per-class AP, confusion matrix |
| ONNX export | **Production** | Validated, FP16 mode, dynamic batch |
| INT8 quantization | **Production** | Dynamic + static (calibration) |
| PyTorch inference | **Production** | GPU/CPU |
| ONNX Runtime inference | **Production** | GPU/CPU |
| Video processing | **Production** | Frame-by-frame, alert system, analytics |
| ByteTrack tracking | **Production** | Configurable thresholds, trace visualization |
| Gradio demo | **Production** | 5 tabs (image, video, webcam, compare, analytics) |
| Annotation QA | **Beta** | LangGraph + SAM3 pipeline |
| Auto-annotation | **Beta** | SAM3, bbox + polygon output |
| Generative augmentation | **Beta** | SAM3 + diffusion inpainting |
| HPO | **Beta** | Optuna, 21 parameters |
| Label Studio bridge | **Partial** | YOLO <-> LS format |

### 3.5 Target Edge Hardware

| Chip | INT8 TOPS | CPU | Target Use |
|------|-----------|-----|-----------|
| **AX650N (AXera)** | 18 TOPS | 8x Cortex-A55 | Primary deployment target |
| **CV186AH (Sophgo)** | 7.2 TOPS | 6x Cortex-A53 | Requires lighter models |

### 3.6 What Is NOT Currently Implemented

- ANPR / License Plate Recognition
- Vehicle classification (car, truck, bus, motorcycle)
- Traffic density estimation / queue length
- Adaptive signal timing logic
- Traffic signal controller integration (NTCIP, SNMP)
- Parking bay occupancy detection
- Vehicle counting / flow analysis
- Speed estimation
- Multi-camera fusion / cross-camera tracking
- Zone polygon detection (config skeleton only, not functional)
- Central management platform / cloud backend
- Mobile app
- LED guidance sign integration
- Payment system integration
- TensorRT optimization

---

## Gap Analysis: What We Can Fulfill vs. What's Missing

### UC1: AI Traffic Light

| Requirement | VIETSOL Status | Gap Level | Notes |
|-------------|---------------|-----------|-------|
| **Real-time vehicle detection** | Partial | MEDIUM | YOLOX/D-FINE detect objects; need vehicle-specific classes (car, truck, bus, motorcycle) — requires new dataset + training |
| **Vehicle classification** | NOT PRESENT | HIGH | No vehicle taxonomy model; need COCO-pretrained or custom dataset |
| **Adaptive signal timing** | NOT PRESENT | CRITICAL | Core algorithm missing; need RL-based or rule-based phase optimizer |
| **Multi-lane monitoring** | Partial | MEDIUM | ByteTrack + zone config exists but not functional; need per-lane ROI zone implementation |
| **Queue length estimation** | NOT PRESENT | HIGH | Need detection + tracking + stopped-vehicle logic |
| **Incident/anomaly detection** | Partial | MEDIUM | Zone intrusion model exists; need stalled vehicle / blockage detection |
| **Emergency vehicle priority** | NOT PRESENT | HIGH | No EV detection model, no siren detection, no preemption logic |
| **Signal controller integration** | NOT PRESENT | CRITICAL | No NTCIP 1202 / SNMP / serial protocol support |
| **Central traffic management** | NOT PRESENT | CRITICAL | No cloud platform, ATMS, dashboard for traffic |
| **Edge deployment** | Partial | LOW | ONNX + INT8 quantization ready; AX650N/CV186AH targeted but no NTCIP firmware |
| **Scalable city-wide** | NOT PRESENT | HIGH | No multi-intersection coordination, no network optimization |

**Fulfillment: ~20%** — We have the base detection/tracking/export infrastructure. The entire traffic-specific logic layer (adaptive timing, controller integration, central management) is missing.

### UC2: Smart Parking

| Requirement | VIETSOL Status | Gap Level | Notes |
|-------------|---------------|-----------|-------|
| **Vehicle detection** | Partial | LOW | YOLOX detects objects; need vehicle-class training on parking data |
| **Bay occupancy monitoring** | NOT PRESENT | HIGH | Need per-slot ROI mapping + occupancy classification logic |
| **ANPR (license plate)** | NOT PRESENT | CRITICAL | No plate detection model, no OCR engine, no IR camera support |
| **Safety monitoring (loitering)** | Partial | MEDIUM | Zone intrusion + person detection exist; need dwell-time logic |
| **Safety monitoring (intrusion)** | Partial | LOW | Zone intrusion model exists; need zone polygon implementation |
| **Safety monitoring (accidents)** | Partial | MEDIUM | Fall detection exists; need vehicle-accident detection |
| **Violation detection (double parking)** | NOT PRESENT | HIGH | Need bay-zone logic + vehicle-outside-zone timer |
| **Real-time space guidance** | NOT PRESENT | CRITICAL | No guidance sign integration, no occupancy API, no mobile app |
| **Centralized dashboard** | NOT PRESENT | CRITICAL | No parking management platform, no analytics backend |
| **Reuse existing CCTV** | YES | NONE | Edge AI on RTSP streams is our core architecture |
| **Edge processing** | YES | LOW | ONNX + INT8 + AX650N targeting ready |
| **Tracking** | YES | NONE | ByteTrack production-ready |

**Fulfillment: ~25%** — We have strong edge AI inference + tracking infrastructure. Parking-specific features (ANPR, occupancy mapping, guidance, dashboard) are entirely missing.

### Shared Gaps (Both UCs)

| Gap | Impact | Effort Estimate |
|-----|--------|----------------|
| **Zone polygon detection** (functional) | Both UCs need per-zone logic | Medium (config exists, need UI + enforcement logic) |
| **Vehicle classification model** | Both need car/truck/bus/motorcycle | Medium (COCO pretrained YOLOX or D-FINE) |
| **Cloud backend / API** | Both need central platform | Large (new development) |
| **Dashboard / Web UI** | Both need operator interface | Large (new development) |
| **ANPR/LPR** | Smart Parking critical | Large (plate detection + OCR pipeline) |
| **Signal controller protocol** | AI Traffic Light critical | Large (NTCIP 1202 implementation) |
| **Mobile app** | Smart Parking needs it | Large (new development) |
| **Multi-camera management** | Both at scale | Medium (RTSP stream manager) |

---

## Recommendations & Roadmap

### Priority Assessment

| Priority | Item | UC | Rationale |
|----------|------|----|-----------|
| **P0** | Vehicle detection + classification (COCO classes) | Both | Foundation for everything; can use COCO-pretrained YOLOX-M/D-FINE-S immediately |
| **P0** | Zone polygon detection (functional) | Both | Config skeleton exists; implement boundary checking + per-zone alerts |
| **P1** | ANPR/LPR pipeline | Parking | Critical differentiator; YOLO plate detection + PaddleOCR |
| **P1** | Bay occupancy logic | Parking | ROI slot mapping + classification (occupied/vacant) |
| **P1** | Queue length + density estimation | Traffic | Detection + tracking + stopped-vehicle timer |
| **P2** | Adaptive signal timing algorithm | Traffic | Rule-based first, RL-based later |
| **P2** | Central management backend | Both | REST API + database + dashboard |
| **P2** | Real-time guidance (LED signs, mobile) | Parking | Integration layer |
| **P3** | NTCIP 1202 protocol | Traffic | Signal controller integration |
| **P3** | Emergency vehicle priority | Traffic | Specialized model + preemption logic |
| **P3** | City-wide coordination | Traffic | Multi-intersection optimizer |

### What We Can Leverage Immediately

| Existing Asset | Application |
|----------------|-------------|
| YOLOX-M / D-FINE-S detection pipeline | Vehicle detection at intersections and parking lots |
| ByteTrack tracking | Vehicle counting, queue estimation, dwell time |
| ONNX + INT8 quantization | Edge deployment on AX650N / Jetson |
| Video processing + alert system | Incident detection, violation alerts |
| Zone intrusion model (person detection) | Loitering and intrusion detection in parking |
| Gradio demo | Quick PoC demonstration to BDA |
| Config-driven architecture | Rapid adaptation to new use cases without code changes |
| HF generic adapter | Quickly integrate any new model (e.g., plate detection) |

### Competitive Positioning

| Factor | VIETSOL Advantage | VIETSOL Disadvantage |
|--------|-------------------|---------------------|
| **Edge AI expertise** | Strong — proven pipeline with AX650N/CV186AH targeting, INT8 quantization | No TensorRT optimization yet |
| **Model variety** | Strong — YOLOX + D-FINE + RT-DETRv2 + pose + classification | No vehicle taxonomy or ANPR model |
| **Cost** | Strong — camera-based approach is 3-5x cheaper than sensors | Need to build platform layer |
| **Existing CCTV reuse** | Strong — core architecture is RTSP → Edge AI | No smart camera SoC integration |
| **Traffic domain** | Weak — no adaptive timing, no controller integration | NoTraffic/Miovision years ahead |
| **Parking domain** | Weak — no ANPR, no occupancy mapping | Parquery/Cleverciti are mature |
| **Platform/dashboard** | Weak — Gradio demo only, no production platform | Need full-stack development |
| **Local presence** | Strong — ESP partnership provides local market access in Bintulu | Need to prove technical capabilities |

### Suggested Approach for Proposal

**Smart Parking (UC2) — Recommend leading with this:**
- Lower technical barrier (detection + tracking is our strength)
- Camera-based approach aligns perfectly with "reuse existing CCTV" requirement
- Clear cost advantage over sensor-based alternatives ($1-3M vs $5-10M for 7,000 bays)
- Can demo vehicle detection + tracking + zone alerts quickly
- ANPR is the main gap — can be addressed with YOLO + PaddleOCR in 4-8 weeks

**AI Traffic Light (UC1) — Recommend as Phase 2:**
- Much higher technical complexity (adaptive timing, controller integration)
- No existing open-source adaptive signal control to build on
- Competing against NoTraffic/Miovision with years of deployment experience
- Can offer detection + counting + analytics as first phase, adaptive control later
- Partnership with traffic controller vendor (e.g., Econolite, local Malaysian vendor) recommended

---

## References

### AI Traffic Light
- NoTraffic AI Mobility Platform — notraffic.com
- Miovision Adaptive Signal Control — miovision.com
- SCOOT 8 AI (TRL Software) — trlsoftware.com
- Surtrac (CMU/Rapid Flow) — 25% travel time reduction in Pittsburgh
- Econolite Centracs ATMS — econolite.com
- NVIDIA Jetson for Traffic (NoTraffic partnership) — blogs.nvidia.com
- US DOT ITS Costs Database — itskrs.its.dot.gov
- Adaptive Traffic Control Market Report (Spherical Insights, 2025)

### Smart Parking
- Parquery Smart Parking — parquery.com
- Cleverciti + ParkHelp — cleverciti.com
- Genetec AutoVu — genetec.com
- Milesight LPR — milesight.com
- Flash Parking (Hailo-8) — flashparking.com
- SFpark (San Francisco, 19,250 sensors case study)
- Smart Parking Market Report (Grand View Research, 2025)
- Allied Market Research: Smart Parking to $48.3B by 2033

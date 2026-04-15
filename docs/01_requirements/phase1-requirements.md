# Phase 1 AI Model Requirements Specification
## Factory Smart Camera Safety System

**Project:** Factory Smart Camera AI System
**Customer:** Vietsol Factory Client
**Date:** March 5, 2026
**Version:** 2.0

---

## Document Structure

This document is divided into two clear sections:

1. **Section A: Customer Requirements** — Requirements provided by the customer from input materials
2. **Section B: Technical Team Definitions** — Technical specifications proposed by the AI development team

---

# SECTION A: CUSTOMER REQUIREMENTS

> **Source:** Customer input files:
> - `docs/Camera AI_Zone Mapping_03032026.xlsx`
> - `docs/監視カメラにＡＩ_translate.pptx`
> - `docs/models_require.jpg`

## A.1 Project Scope

### Phase 1 Models (Customer Specified)

The customer has specified the following models for Phase 1 development:

| # | Model Name | Customer Description |
|---|---|---|
| **a** | **Fire Detection** | Detect even the smallest possible fires. Detects open flames at long range, temperature change detection, smoke detection |
| **b** | **PPE: Helmet** | Detection of not wearing several types of helmets (including Nitto soft hats) |
| **f** | **PPE: Safety Shoes** | Detection of not wearing safety shoes |
| **g** | **Fall Detection** | Watching over employees and workers. Detects falls and alarms |
| **h** | **Poketenashi Violations** | Hands in pockets, mobile phone use while walking, not using handrails on stairs, crossing stairs diagonally |
| **i** | **Restricted Area Intrusion** | Detects entering dangerous areas, driving in opposite direction, parking in no-parking areas |

---

## A.2 Camera Infrastructure (Customer Provided)

### Installed Camera Models

| Camera Model | Form Factor | Resolution | Sensor | Lens | VFOV | HFOV | FPS | Low Light | IR Range |
|---|---|---|---|---|---|---|---|---|---|
| **DWC-MV82DiVT** | Dome | 2.1MP / 1080p | 1/2.9" CMOS | 2.7–13.5 mm vari-focal | 49.9° ~ 15.2° | 89.6° ~ 27° | 30 | 0.16 lux (Color) / 0.0 lux (B/W, IR on) | Smart IR™ up to 100 ft (~30 m) |
| **DWC-MB62DiVTW** | Bullet | 2.1MP / 1080p | 1/2.9" CMOS | 2.7–13.5 mm vari-focal | 49.9° ~ 15.2° | 89.6° ~ 27° | 30 | 0.16 lux (Color) / 0.0 lux (B/W, IR on) | Smart IR™ up to 90 ft (~27 m) |
| **DWC-MV75Wi28TW** | Vandal Dome (ultra low-profile) | 5MP | 1/2.8" CMOS | 2.8 mm fixed | 37.6° / 58.8° / 77.4° | 102.4° / 51.9° / 82.3° | 30 | Star-Light Plus™ 0.13 lux (Color) / 0.0 lux (B/W) | Smart IR™ up to 80 ft (~24 m) |
| **DWC-MB75Wi4TW** | Bullet | 5MP | 1/2.8" CMOS | 4.0 mm fixed | 37.6° / 58.8° / 77.4° | 102.4° / 51.9° / 82.3° | 30 | Star-Light Plus™ 0.13 lux (Color) / 0.0 lux (B/W) | Smart IR™ up to 80 ft (~24 m) |
| **DWC-PVF5Di1TW** | Fisheye IP Dome Camera | 5MP | 5MP fisheye lens | Fisheye 360° lens | 360° | 360° panoramic coverage | 30 | Star‑Light Plus™ (color in near-total darkness) | Smart IR up to 80 ft |
| **DWC-MB45WiATW** | Bullet camera | 5MP | 1/2.8'' CMOS | 2.7–13.5 mm vari-focal | 62° ~ 24° | 85° ~ 31° | 30 | 0.08 lux (color) • 0.0 lux (B/W) | 140 ft (≈42m) |

---

## A.3 Deployment Zones (Customer Provided)

### Zone List by Floor

| Floor | Zone ID | Zone Name | Cameras Present |
|---|---|---|---|
| **1F** | 1 | Clean Room | X (Dome 5MP Fixed) |
| **1F** | 2 | Material Handling Room | X (Dome 2.1MP Fixed) |
| **1F** | 3 | Spare Parts Room | X (Dome 5MP Fixed) |
| **1F** | 4 | Warehouse | X (Dome 360 + Bullet) |
| **1F** | 5 | Warehouse Break Room | X (Dome 5MP Fixed) |
| **1F** | 6 | Industrial Waste Storage | X (Dome 5MP Fixed + Bullet) |
| **1F** | 7 | Hazardous Materials W/H | X (Dome 2.1MP Fixed + Dome 360) |
| **1F** | 8 | Dust Collector Room | X (Dome 5MP Fixed) |
| **1F** | 9 | Machine Room | X (Dome 5MP Fixed) |
| **1F** | 10 | Cleanroom Changing Room | (No camera specified) |
| **1F** | 11 | QC Evaluation Room | X (Dome 5MP Fixed) |
| **1F** | 12 | General Waste Storage | X (Dome 5MP Fixed) |
| **1F** | 13 | Canteen | X (Bullet + Dome 5MP) |
| **1F** | 14 | Kitchen | X (Dome 5MP Fixed) |
| **1F** | 15 | Kitchen Office | X (Dome 5MP Fixed) |
| **1F** | 16 | Kitchen Waste Room | X (Dome 5MP Fixed) |
| **1F** | 17 | Office Supply Storage | X (Dome 5MP Fixed) |
| **1F** | 18 | Corridor Stairs | X (Dome 5MP Fixed + Dome 360) |
| **1F** | 19 | General Area | X (Dome 5MP Fixed) |
| **2F** | 1 | Office, Meeting room | X (Dome 5MP Fixed) |
| **2F** | 2 | Garden | X (Dome 360) |
| **2F** | 3 | Document room | X (Dome 5MP Fixed) |
| **2F** | 4 | Spare Parts Room | X (Dome 360 + Bullet) |
| **2F** | 5 | Union Room | X (Dome 5MP Fixed) |
| **2F** | 6 | Server Room | X (Dome 5MP Fixed) |
| **2F** | 7 | Utility Room | X (Dome 5MP Fixed) |
| **2F** | 8 | Heat Source Room | X (Dome 5MP Fixed) |
| **2F** | 9 | Air Compressor Room | X (Dome 5MP Fixed) |
| **2F** | 10 | Corridor Stairs | X (Dome 5MP Fixed + Dome 360) |
| **3F** | 1 | Office Rooftop | X (Dome 360) |
| **3F** | 2 | Electrical Room | X (Dome 5MP Fixed) |
| **3F** | 3 | Outdoor Machine Area | X (Dome 360) |
| **3F** | 4 | Warehouse | X (Dome 360) |
| **3F** | 5 | Corridor Stairs | X (Dome 5MP Fixed + Dome 360) |
| **RF** | - | Rooftop | X (Dome 360) |
| **RF** | - | Corridor, Stairs | X (Dome 360) |
| **Outdoor** | 1 | Fence Perimeter | X (Bullet 5MP Vari-focal) |
| **Outdoor** | 2 | Entrance | X (Dome 2.1MP Fixed) |
| **Outdoor** | 3 | Exit | X (Dome 360) |
| **Outdoor** | 4 | Outdoor (Warehouse Area) | X (Dome 360) |
| **Outdoor** | 5 | Generator and Pump Room | X (Dome 360) |
| **Outdoor** | 6 | Waste Container Area | X (Dome 360 + Bullet 2.1MP) |
| **Outdoor** | 7 | Motorcycle Entrance/Exit | X (Dome 360) |
| **Outdoor** | 8 | Motorcycle Parking Area | X (Dome 360) |
| **Outdoor** | 9 | Driver Rest Room | X (Dome 5MP Fixed) |
| **Outdoor** | 10 | Smoking Area | X (Dome 360) |
| **Outdoor** | 11 | Futsal court | X (Bullet 5MP Vari-focal) |
| **Outdoor** | 12 | Access Road | X (Dome 360) |
| **M2** | 1 | Changing Room | (No camera specified) |
| **M2** | 2 | MT Evaluation Room | X (Dome 5MP Fixed) |
| **M2** | 3 | QC Evaluation Room | X (Dome 5MP Fixed) |
| **M2** | 4 | Nap Room | X (Dome 5MP Fixed) |
| **M2** | 5 | Break Room | X (Dome 5MP Fixed) |
| **M2** | 6 | Corridor, Stairs | X (Dome 5MP Fixed + Dome 360) |

---

## A.4 Model Requirements by Customer

### Model A: Fire Detection

**Customer Requirement:**
> "Detect even the smallest possible fires"
> "Fire detection: Detects open flames at long range, depending on the size of the fire source"
> "Temperature change detection: Enables early identification of abnormal heat rise before a fire occurs"
> "Smoke detection: Supports early fire warning by detecting smoke from a distance"

**Customer Reference:**
> Source: [AITech Co., Ltd. - Fire Detection AI Camera](https://prtimes.jp/main/html/rd/p/000000071.000014310.html)

**Detection Distances (Customer Specified):**

| Fire Source Size | Detection Distance |
|---|---|
| 1 × 1 m fire source | Up to **325 m** |
| 0.5 × 0.5 m fire source | Up to **162.5 m** |
| 0.2 × 0.2 m fire source | Up to **65 m** |

| Temperature Change Area | Detection Distance |
|---|---|
| 1 × 1 m area | Up to **76 m** |
| 0.5 × 0.5 m area | Up to **38 m** |
| 0.2 × 0.2 m area | Up to **15.2 m** |

| Smoke Source Size | Detection Distance |
|---|---|
| 1 × 1 m smoke area | Up to **65 m** |
| 0.5 × 0.5 m smoke area | Up to **32.5 m** |
| 0.2 × 0.2 m smoke area | Up to **13 m** |

**Detection Types:**
- Open flames
- Smoke
- Temperature change (thermal anomaly)

**Additional Details from Customer Source:**
- Capable of detecting open flames at extremely long range (up to 325m for 1×1m fire)
- Early fire warning possible before visible flames through temperature change detection
- Supports early smoke detection for rapid response

---

### Model B: PPE - Helmet Compliance

**Customer Requirement:**
> "Detection of not wearing several types of helmets (including Nitto soft hats) and two types of safety belts"

**Helmet Types to Detect:**
- Nitto hat (soft hat)
- Full harness
- Waist belt (safety belt)

**Detection Focus:**
- Detect if worker is wearing helmet
- Detect if worker is NOT wearing helmet (violation)

**Customer Reference (PPE Levels):**
> Source: [Kurita Co., Ltd. - Chemical Area PPE Requirements](https://kcr.kurita.co.jp/solutions/videos/049.html)

**PPE Levels & Applications (from customer reference):**

| Level | Area | Required PPE |
|---|---|---|
| **A** | Chemical Area (e.g., patrol/inspection) | Safety glasses (goggles type) |
| **B** | Chemical Receiving/Acceptance | Full-face shield, Chemical-resistant gloves, Chemical-resistant safety rubber boots |
| **C** | Chemical Handling (under-floor work) | Full-face shield, Chemical-resistant apron, Chemical-resistant gloves, Chemical-resistant safety rubber boots |
| **D** | Chemical Handling (above-floor work) / Chemical line maintenance | Full-face shield, Chemical-resistant apron (split type), Chemical-resistant gloves, Chemical-resistant safety rubber boots |
| **E** | Emergency Response | Disaster-response full-face mask, Chemical-resistant protective clothing, Chemical-resistant gloves, Chemical-resistant safety rubber boots |

> **Note:** This reference shows the customer's context for PPE requirements in chemical handling areas, which informs the types of helmets and PPE that need to be detected.

---

### Model F: PPE - Safety Shoes Compliance

**Customer Requirement:**
> "Detection of not wearing safety shoes"
> "Details will be presented separately for the area"

**Detection Focus:**
- Detect if worker is wearing safety shoes
- Detect if worker is NOT wearing safety shoes (violation)

---

### Model G: Fall Detection

**Customer Requirement:**
> "Watching over employees and workers. Detects falls and alarms."

**Detection Focus:**
- Detect worker falling
- Detect worker lying on ground
- Trigger alarm immediately upon fall detection

**Customer Reference:**
> Source: [Web Japan Co., Ltd. - AI Camera Solutions](https://www.webjapan.co.jp/solution/ai-camera/)
>
> AI cameras for watching over employees and workers with automatic fall detection and alarm capabilities.

**Key Features from Customer Reference:**
- 24/7 monitoring of workers and elderly
- Automatic fall detection with alerts
- Supports "watching over" applications in care facilities and workplaces

---

### Model H: Poketenashi Violations

**Customer Requirement:**
> "Poketenashi violations"
>
> Japanese safety practice for safe walking behavior in factories and workplaces.

**Prohibited Actions (to detect):**
- ❌ Walking with hands in pockets
- ❌ Using mobile phone while walking
- ❌ Crossing stairs diagonally or taking shortcuts

**Required Actions:**
- ✅ Always hold handrail when going up or down stairs
- ✅ Perform proper pointing-and-calling (safety confirmation) at designated points

**Purpose:**
- Reduce slips, trips, and falls
- Improve worker awareness and attention
- Promote safe behavior in daily operations

**Typical Use Areas:**
- Factories
- Warehouses
- Construction sites
- Offices with stair access

---

### Model I: Restricted Area Intrusion

**Customer Requirement:**
> "Detects entering dangerous areas, driving in the opposite direction, parking in no-parking areas, etc."
> "Details are presented separately in the area"

**Customer Reference:**
> Source: [DCross - AI Camera Intrusion Detection](https://dcross.impress.co.jp/docs/news/001320.html)

**Detection Functions:**
- **Intrusion Detection**
  - People entering restricted areas
  - Bicycles or vehicles entering prohibited zones
  - Unauthorized entry into buildings at night

- **Loitering Detection**
  - Vehicles stopping or staying in no-parking or restricted areas

- **Direction Detection**
  - Bicycles or vehicles moving in the wrong direction

- **Line-Crossing Detection**
  - People crossing prohibited or restricted lines

**Additional Features from Customer Reference:**
- Perimeter defender functionality for comprehensive security
- Virtual line crossing with directional detection
- Automatic tracking of intruders
- Real-time alerts for security personnel

---

# SECTION B: TECHNICAL TEAM DEFINITIONS

> **Source:** Technical team recommendations based on customer requirements, industry standards, and hardware constraints

## B.1 Model Architecture Selection

### Chosen Architectures

| Model | Recommended Architecture | Rationale |
|---|---|---|
| Fire Detection | YOLOX-M (Apache 2.0) | Balance of speed and accuracy; good for small object detection |
| Helmet Compliance | YOLOX-M (Apache 2.0) | Edge-optimized; fast inference; handles small objects (helmet at distance) |
| Safety Shoes | YOLOX-M (Apache 2.0) | Shoe detection requires good localization |
| Fall Detection | MoveNet (Apache 2.0) | Pose estimation (17 keypoints) for accurate fall detection |
| Poketenashi | YOLOX-M (Apache 2.0) | Multi-class detection with pose analysis |
| Intrusion | YOLOX-T (Apache 2.0, pretrained) | Lightweight person detection; zone-based logic only |

> **License Strategy:** All models use Apache 2.0 license — **$0 licensing cost** for commercial deployment.

**Hardware Constraint:** Models must be deployable on edge AI devices with power consumption < 12W (specific hardware TBD in separate document)

---

## B.2 Detection Class Definitions

### Model A: Fire Detection

| Class ID | Class Name | Description |
|---|---|---|
| 0 | fire | Open flames of any visible size |
| 1 | smoke | Visible smoke from combustion |
| 2 | thermal_anomaly | Abnormal heat signature (Future — requires thermal camera integration) |

### Model B: PPE - Helmet

| Class ID | Class Name | Description |
|---|---|---|
| 0 | person | Full body detection for tracking |
| 1 | helmet | Worker wearing helmet (any type) |
| 2 | no_helmet | Worker head visible without helmet |
| 3 | nitto_hat | Nitto soft hat specifically (if distinguishable) |

### Model F: PPE - Safety Shoes

| Class ID | Class Name | Description |
|---|---|---|
| 0 | person | Full body detection for tracking |
| 1 | safety_shoes | Worker wearing safety shoes/boots |
| 2 | no_safety_shoes | Worker wearing non-safety footwear |
| 3 | shoe_region | Foot region bounding box for analysis (Implementation detail — used in two-stage detection pipeline) |

### Model G: Fall Detection

| Class ID | Class Name | Description |
|---|---|---|
| 0 | person | Tracked person ID |
| 1 | fall_detected | Person in fallen/lying position |
| 2 | unsafe_posture | Person in unsafe bent/crouching position |

> **Note:** Only class 0 (person with 17 COCO keypoints) is a trained detection class. Classes 1–2 (`fall_detected`, `unsafe_posture`) are alert states determined by keypoint post-processing rules (hip–shoulder ratio + temporal consistency), not direct model outputs.

### Model H: Poketenashi

| Class ID | Class Name | Description |
|---|---|---|
| 0 | person | Walking/standing person |
| 1 | hands_in_pockets | Both hands in pockets |
| 2 | phone_usage | Using mobile phone while walking |
| 3 | no_handrail | Not holding handrail on stairs |
| 4 | unsafe_stair_crossing | Crossing stairs diagonally |

> **Note:** Only `phone_usage` currently has training data. `hands_in_pockets` and `no_handrail` are detected via pose keypoint analysis. `unsafe_stair_crossing` uses zone polygon + trajectory logic.

### Model I: Restricted Area Intrusion

| Class ID | Class Name | Description |
|---|---|---|
| 0 | person | Person intrusion |
| 1 | vehicle_intrusion | Vehicle/bicycle intrusion in restricted zones |
| 2 | wrong_direction | Movement in wrong direction |
| 3 | loitering | Stationary in no-standing zone (Alert state — determined by tracking duration in zone) |

> **Note:** Uses pretrained person/vehicle detection only. Classes 1–3 are alert states determined by zone polygon logic and tracking, not trained detection classes.

---

## B.3 Performance Targets (Technical Team Recommendation)

### Primary Acceptance Metrics (Customer-Facing — Determines Go/No-Go)

> **Note:** All metric targets below are technical team proposals — customer has not specified quantitative targets. Require customer verification and alignment before finalizing.

| Model | Target Precision | Target Recall | Max FP Rate | Max FN Rate |
|---|---|---|---|---|
| Fire Detection | ≥ 0.90 | ≥ 0.88 | < 3% | < 5% |
| Helmet | ≥ 0.94 | ≥ 0.92 | < 2% | < 3% |
| Safety Shoes | ≥ 0.88 | ≥ 0.85 | < 4% | < 5% |
| Fall Detection | ≥ 0.90 | ≥ 0.88 | < 3% | < 2% |
| Poketenashi | ≥ 0.85 | ≥ 0.82 | < 5% | < 6% |
| Intrusion | ≥ 0.94 | ≥ 0.92 | < 2% | < 3% |

### Secondary Tracking Metrics (Internal — Used During Training Iteration)

| Model | Target mAP@0.5 | Minimum Acceptable | Technical Note |
|---|---|---|---|
| Fire Detection | ≥ 0.85 | 0.75 | Customer requires 325m detection for 1×1m fire |
| Helmet | ≥ 0.92 | 0.85 | Small object detection; may need 1280px input |
| Safety Shoes | ≥ 0.85 | 0.75 | Feet often occluded; challenging |
| Fall Detection | ≥ 0.85 | 0.75 | Pose-based approach more accurate |
| Poketenashi | ≥ 0.80 | 0.70 | Complex behavior; requires good pose data |
| Intrusion | ≥ 0.92 | 0.85 | Pretrained model; zone configuration critical |

> **Important:** mAP@0.5 is used during training iteration to monitor progress. Final acceptance is determined by the primary metrics (Precision, Recall, FP Rate, FN Rate) above.

### Inference Performance

| Metric | Target | Rationale |
|---|---|---|
| Frame Rate | ≥ 15 FPS | Smooth tracking; acceptable for safety monitoring |
| Alert Latency | < 3 seconds | From event to dashboard alert |
| System Availability | ≥ 99.5% | Downtime < 3.65 days/year |

---

## B.4 Input Resolution Strategy

| Scenario | Resolution | When to Use |
|---|---|---|
| Standard detection | 640 × 640 | Default for most models |
| Small/distant objects | 1280 × 1280 | Helmet, shoes, fire at distance |
| Pose estimation | 640 × 640 | Sufficient for keypoint detection |
| Fisheye cameras | 640 × 640 (after dewarping) | Fisheye requires preprocessing |

---

## B.5 Alert Logic Specifications

### Alert Confirmation Thresholds

To prevent false alarms, use multi-frame confirmation:

| Model | Min Confidence | Min Duration | Tracking Required |
|---|---|---|---|
| Fire | 0.70 | 3 frames (100ms) | No |
| PPE Violation | 0.70 | 30 frames (1 sec) | Yes |
| Fall | 0.65 | 15 frames (500ms) | Yes |
| Intrusion | 0.60 | 5 frames (167ms) | Yes |
| Poketenashi | 0.65 | 30 frames (1 sec) | Yes |

### Fall Detection Logic (Technical Definition)

```python
def is_fall_aspect_ratio(bbox):
    """Rule 1: Aspect ratio based"""
    x, y, w, h = bbox
    aspect_ratio = w / h
    return aspect_ratio > 1.5

def is_fall_pose(keypoints):
    """Rule 2: Keypoint based (more accurate)"""
    hip_y = (keypoints['left_hip'][1] + keypoints['right_hip'][1]) / 2
    shoulder_y = (keypoints['left_shoulder'][1] + keypoints['right_shoulder'][1]) / 2

    is_horizontal = hip_y >= shoulder_y - 0.05
    is_near_ground = hip_y > 0.7

    return is_horizontal and is_near_ground

def confirm_fall(person_history, window_frames=30):
    """Rule 3: Temporal consistency"""
    fall_count = sum(1 for state in person_history if state == 'fall')
    return fall_count >= window_frames * 0.8
```

### Zone Intrusion Logic (Technical Definition)

```python
def is_polygon_intrusion(person_centroid, zone_polygon):
    """Check if person is inside restricted zone polygon"""
    return point_in_polygon(person_centroid, zone_polygon)

def is_line_crossing_invasion(person_track, zone_line, allowed_direction):
    """Check if person crosses line in wrong direction"""
    crosses = line_crossing(person_track[-2], person_track[-1], zone_line)
    if not crosses:
        return False
    crossing_direction = calculate_crossing_direction(person_track, zone_line)
    return crossing_direction != allowed_direction
```

---

## B.6 Hardware Requirements (Technical Team Recommendation)

### Power Constraint (Customer Requirement)

**Maximum Power Consumption:** < 12W per edge device

> **Important:** The specific chip/processor selection will be determined in a separate hardware specification document. The models developed in Phase 1 must be deployable on edge AI hardware that operates within the 12W power budget.

### Hardware Requirements Summary

| Requirement | Specification | Notes |
|---|---|---|
| **Max Power** | < 12W | Strict power limit per edge device |
| **Inference Speed** | ≥ 15 FPS | Per model, at 640×640 input |
| **Multi-Model** | ≥ 2 models simultaneous | At ≥ 10 FPS each |
| **Input Support** | RTSP video stream | From existing IP cameras |
| **Connectivity** | Ethernet (PoE) | Power + data over single cable |
| **Operating Temp** | 0-45°C | Factory environment |
| **Form Factor** | Compact | For camera-mounted installation |

### Model Deployment Requirements

All Phase 1 models must be capable of:

1. **Running on low-power edge AI hardware** (< 12W)
2. **Real-time inference** (≥ 15 FPS per model)
3. **ONNX export compatibility** (for hardware portability)
4. **Quantization support** (INT8 preferred for power efficiency)

### Power Budget Considerations

**12W Power Budget Breakdown (Typical):**

| Component | Estimated Power | Notes |
|---|---|---|
| AI Chip/NPU | 5-8W | Main inference processor |
| CPU (SoC) | 2-4W | System control, preprocessing |
| Memory | 1-2W | RAM for model execution |
| Networking | 0.5-1W | Ethernet/POE interface |
| Other (USB, etc.) | 0.5-1W | Peripherals |
| **Total** | **< 12W** | **Maximum allowable** |

### Model Optimization for Low-Power Deployment

To meet the 12W constraint, models will be optimized using:

| Technique | Purpose | Expected Impact |
|---|---|---|
| **Model Pruning** | Remove redundant parameters | 20-40% reduction in compute |
| **Quantization (INT8)** | Reduce precision from FP32 to INT8 | 4x memory reduction, 2-4x speedup |
| **Input Resolution Scaling** | Use 640px instead of 1280px where possible | 4x reduction in pixels |
| **Model Distillation** | Train smaller "student" models | Maintain accuracy with less compute |
| **ONNX Optimization** | Remove unused operators | Faster inference |

### Hardware Selection Criteria (Future Document)

The following will be evaluated in a separate hardware specification document:

| Evaluation Criteria | Description |
|---|---|
| **AI Performance (TOPS)** | Tera Operations Per Second at < 12W |
| **Model Compatibility** | Support for YOLO, pose estimation, ONNX |
| **Memory Bandwidth** | Sufficient for 640×640 real-time processing |
| **Power Efficiency** | Performance per watt (TOPS/W) |
| **Ecosystem Support** | SDK, documentation, community |
| **Supply Chain** | Availability, lead time, cost |
| **Environmental** | Operating temp range for factory use |
| **PoE Integration** | Native PoE or low-power PoE extender |

### Expected Performance Targets (Power-Constrained)

**Performance targets for < 12W edge devices:**

| Model | Input Resolution | Expected FPS @ <12W | Notes |
|---|---|---|---|
| Fire Detection | 640×640 | ≥ 20 FPS | May drop to 15 FPS with multiple models |
| Helmet | 640×640 | ≥ 20 FPS | 1280×1280 requires more power (use selectively) |
| Safety Shoes | 640×640 | ≥ 20 FPS | Challenging; may need post-processing |
| Fall (Pose) | 640×640 | ≥ 15 FPS | Pose estimation more compute-intensive |
| Poketenashi | 640×640 | ≥ 15 FPS | Multi-class with pose analysis |
| Intrusion | 640×640 | ≥ 25 FPS | Lightweight person detection |

**Multi-Model Performance (Target):**
- 2 models running simultaneously: ≥ 10-15 FPS each
- 3+ models running simultaneously: ≥ 8-10 FPS each

### Reference Hardware Examples (For Research Only)

> **Note:** These are examples for research and testing. Final hardware selection will be made in a separate document.

| Hardware | Power | AI Performance | Expected FPS | Status |
|---|---|---|---|---|
| **Hailo-8** | 2.5-5W (NPU) | 26 TOPS | 30-80 FPS @ 640px (varies by model size) | Example reference |
| **Hailo-8L** | 1.5-3W (NPU) | 13 TOPS | 15-40 FPS @ 640px | Lower-power option |
| **Google Coral TPU** | 2W | 4 TOPS | 10-15 FPS @ 640px | Lower performance |
| **Rockchip RK3588** | 8W (full SoC) | 6 TOPS NPU | 15-20 FPS @ 640px | Integrated solution |
| **Renesas RZ/V2H** | ~10W (estimated) | 80-100 TOPS | 30+ FPS @ 640px | High performance |

**Important:** These examples are for benchmarking and planning only. Actual hardware selection will consider cost, availability, supply chain, and integration requirements.

### Development Strategy for Power-Constrained Deployment

**Phase 1 Approach:**

1. **Model Development:** Develop models using standard GPU/CPU hardware
2. **Power Optimization:** Optimize models for < 12W deployment during Phase 1
3. **Hardware Testing:** Test on multiple < 12W hardware options
4. **Benchmarking:** Measure actual power consumption and performance
5. **Final Selection:** Recommend specific hardware in separate document

**Optimization Techniques:**

```python
# Example: Power-aware model configuration
POWER_CONSCIOUS_CONFIG = {
    # Use smaller models where acceptable
    'intrusion': 'yolo26n',      # Lightweight: 2.4M parameters
    'fire': 'yolo11s',           # Balanced: 8.7M parameters
    'helmet': 'yolo26s',         # Edge-optimized: 9.5M parameters
    'fall': 'yolo11n-pose',      # Smallest pose model

    # Input resolution based on detection distance
    'standard_input': 640,       # Default: 640×640
    'distant_input': 1280,       # For distant detection (use sparingly)

    # Quantization for power efficiency
    'quantization': 'int8',      # Reduces power vs. fp32

    # Pruning to remove redundancy
    'pruning_ratio': 0.3,        # Remove 30% of parameters
}
```

### Validation Requirements

All models must be validated to confirm:

- ✅ **Power consumption** measured < 12W under full load
- ✅ **Thermal performance** within factory temperature range
- ✅ **Inference speed** meets minimum FPS requirements
- ✅ **Multi-model operation** verified (2+ models simultaneously)
- ✅ **Stability** tested for 24+ hours continuous operation

---

## Hardware Selection Process (Future Work)

A separate document will cover:

1. **Hardware Evaluation Matrix** - Detailed comparison of < 12W options
2. **Benchmarking Results** - Actual performance and power measurements
3. **Supply Chain Analysis** - Availability, pricing, lead times
4. **Integration Design** - How to connect to existing cameras
5. **Deployment Planning** - Installation, configuration, maintenance
6. **Total Cost of Ownership** - Hardware + deployment + maintenance costs

---

## B.7 Training Data Requirements (Technical Team Estimation)

### Dataset Size Estimates

| Model | Minimum Images | Open Datasets Available | Custom Data Needed |
|---|---|---|---|
| Fire | 5,000 | FASDD, DFS, Roboflow Fire | Factory-specific fires |
| Helmet | 8,000 | Roboflow Universe PPE | Factory footage |
| Safety Shoes | 5,000 | Industrial Safety | Factory footage |
| Fall | 2,000 | COCO Pose, Le2i Fall | Factory floor scenes |
| Poketenashi | 3,000 | None (custom) | Factory corridors/stairs |
| Intrusion | 1,000 | COCO (person pretrained) | Zone polygons only |

### Data Collection Priority

**Week 1-2: Critical Data**
1. Factory fire scenarios (if safe to simulate)
2. PPE - Helmet (with/without, various types)
3. PPE - Safety shoes (different camera angles)

**Week 3-4: Specialized Data**
4. Fall detection (simulated with actors/dummies)
5. Poketenashi (hands in pockets, phone usage, handrail)
6. Zone polygons for restricted areas

---

## B.8 Development Timeline (Technical Team Estimation — 12 Weeks)

| Week | Phase | Activities |
|---|---|---|
| **1-2** | Setup & Data Exploration | Environment setup, download open datasets, extract factory footage, annotation setup |
| **3** | v1 Curated | Train v1 models on curated subsets, baseline evaluation |
| **4-5** | v2 Expanded | Expand datasets, train v2 models, error analysis iteration |
| **6** | v3 Full | Full dataset training, all models at target metrics |
| **7** | Export | ONNX export, edge optimization, integration testing |
| **8** | Validate | Hardware testing, factory validation set evaluation |
| **9** | Handoff | Customer demo, deliverable packaging, documentation |
| **10-12** | Buffer | Iteration buffer, contingency for underperforming models |

**Resource Estimation:**
- GPU Training Time: ~40 hours
- GPU Cost: $0 (local GPU)
- Annotation Time: ~100 hours (with SAM 3 assist)
- Hardware Cost: TBD (Edge device selection pending; target: < 12W power consumption)

---

## B.9 Deployment Architecture (Technical Team Design)

### System Topology

```
Factory Floor (per camera zone):
┌─────────────────────────────────┐
│  Existing IP Camera (customer)  │
│  │ RTSP over PoE cable          │
│  ▼                              │
│  Edge AI Device (< 12W) (new)   │
│  ├─ Model A: Fire Detection     │
│  ├─ Model B: Helmet             │
│  ├─ Model F: Safety Shoes       │
│  ├─ Model G: Fall Detection     │
│  ├─ Model H: Poketenashi        │
│  └─ Model I: Intrusion          │
│  └─ Alert Logic                 │
└────────────┬────────────────────┘
             │ Ethernet
             ▼
        PoE Switch (customer)
             │
             ▼
   ┌──────────────────────┐
   │  Central Dashboard   │ (new)
   │  (On-premise server) │
   │  ┌─────────────────┐ │
   │  │ Mosquitto (MQTT)│ │
   │  │ Alert Logic     │ │
   │  │ Database        │ │
   │  └─────────────────┘ │
   └──────────────────────┘
             │
             ▼
   Mobile alerts / Email / Siren
```

**Note:** Specific edge AI device hardware will be selected in a separate document. All devices must meet the < 12W power consumption requirement.

---

## B.10 Acceptance Criteria (Technical Team Definition)

### Model Acceptance

Each model must meet ALL criteria:

1. ✅ **Primary Accuracy**: Precision AND Recall ≥ per-model target values (see B.3 Primary Acceptance Metrics)
2. ✅ **False Positive/Negative Rates**: FP Rate AND FN Rate within per-model thresholds (see B.3)
3. ✅ **Secondary Tracking**: mAP@0.5 ≥ minimum acceptable (internal benchmark, not a Go/No-Go gate)
4. ✅ **Performance**: Inference ≥ 15 FPS on < 12W edge hardware
5. ✅ **Power**: Operates within < 12W power budget (measured under full load)
6. ✅ **Robustness**: Works in low-light (0.16 lux), IR mode
7. ✅ **Testing**: Validated on held-out test set from actual factory footage

### System Acceptance

1. ✅ **Latency**: Event detection → alert < 3 seconds
2. ✅ **Multi-Model**: At least 2 models run simultaneously at ≥ 10 FPS on < 12W hardware
3. ✅ **Power**: Total edge device power consumption < 12W under multi-model load
4. ✅ **Reliability**: 24/7 operation for 1 week without manual intervention
5. ✅ **Thermal**: Stable operation within factory temperature range (0-45°C)
6. ✅ **Alert Quality**: < 5 false alarms per day per camera
7. ✅ **User Interface**: Dashboard displays all alerts with video context

---

## B.11 Open Questions (Requiring Customer Clarification)

### Model-Specific Questions

1. **Fire Detection**
   - [ ] Is thermal camera data available, or visible/IR camera only?
   - [ ] What are typical fire sources in the factory? (electrical, chemical, etc.)

2. **Helmet Detection**
   - [ ] Are all helmet types treated the same, or do Nitto hats need separate classification?
   - [ ] Are there specific zones where helmets are NOT required?

3. **Safety Shoes**
   - [ ] What types of safety shoes are used? (steel-toe, chemical-resistant, etc.)
   - [ ] Are there specific zones where shoes are NOT required?

4. **Fall Detection**
   - [ ] What is the definition of "fall"? (any person on ground, or only sudden falls?)
   - [ ] Should we exclude people who are sitting/resting on floor?

5. **Poketenashi**
   - [ ] Are there designated "pointing-and-calling" locations to detect?
   - [ ] Is phone usage allowed in certain areas?

6. **Intrusion**
   - [ ] Who defines the restricted zone polygons?
   - [ ] Is there a time-based access control (day/night, working hours)?

### System Questions

7. **Alert Response**
   - [ ] Who receives alerts? (security, supervisors, all workers?)
   - [ ] What is the escalation procedure?

8. **Data Privacy**
   - [ ] Are there privacy restrictions on storing/processing worker images?
   - [ ] Must faces be blurred in logs?

9. **Integration**
   - [ ] Is there existing security/safety system to integrate with?
   - [ ] What dashboard technology is preferred? (web, mobile, desktop?)

---

## Appendix

### A. Customer Reference Materials

- Fire Detection: https://prtimes.jp/main/html/rd/p/000000071.000014310.html
- Fall Detection: https://www.webjapan.co.jp/solution/ai-camera/
- Intrusion Detection: https://dcross.impress.co.jp/docs/news/001320.html

### B. Technical Team References

- Tool Stack: See `docs/tool_stack.md`
- Development Plan: See `docs/phase1_development_plan.md`
- Labeling Guide: See `docs/labeling_guide.md`
- Error Analysis: See `docs/error_analysis_guide.md`

---

**Document Version**: 2.0
**Last Updated**: March 5, 2026
**Prepared by**: Vietsol AI Technical Team

**Document Status:**
- ✅ Section A: Customer Requirements — VERIFIED from customer input files
- ⚠️ Section B: Technical Team Definitions — REQUIRES CUSTOMER VERIFICATION AND ALIGNMENT

---

## Terminology & Glossary

This section explains technical terms used throughout this document to help all stakeholders understand the requirements and specifications.

### Camera & Imaging Terms

| Term | Full Name | Explanation |
|---|---|---|
| **VFOV** | Vertical Field of View | The vertical angle that the camera can see. Measured in degrees. A smaller VFOV (e.g., 15°) means the camera can see farther but has a narrower view. A larger VFOV (e.g., 77°) means a wider view but shorter distance. |
| **HFOV** | Horizontal Field of View | The horizontal angle that the camera can see. Measured in degrees. Works the same as VFOV but for the horizontal direction. |
| **FPS** | Frames Per Second | How many images the camera captures per second. 30 FPS means 30 images every second, which is standard for smooth video. |
| **IR** | Infrared | A type of light that is invisible to the human eye. Cameras use IR for night vision. They emit IR light (via IR LEDs) to see in complete darkness. |
| **Lux** | Unit of illuminance | A measurement of light intensity. Lower lux values mean darker environments. 0.16 lux is very dim light (moonless clear night). 0 lux means complete darkness (IR mode required). |
| **Fisheye** | 360° camera lens | A special lens that captures an extremely wide, often 360° panoramic view. The image looks curved/distorted and requires software correction (dewarping). |
| **Vari-focal** | Adjustable focal length | A lens that can zoom in/out. For example, 2.7–13.5mm means the lens can adjust from wide-angle (2.7mm) to telephoto (13.5mm). |
| **Fixed lens** | Non-adjustable lens | A lens with a fixed zoom level. Cannot zoom in or out. |
| **RTSP** | Real-Time Streaming Protocol | The standard protocol used by IP cameras to stream video over a network. |
| **PoE** | Power over Ethernet | A technology that delivers both electrical power and data over a single Ethernet cable. Simplifies camera installation (no separate power cable needed). |

### AI & Machine Learning Terms

| Term | Full Name | Explanation |
|---|---|---|
| **AI / Artificial Intelligence** | | Computer systems that can perform tasks that typically require human intelligence, such as recognizing objects in images. |
| **Model** | AI Model | A trained computer program that can recognize patterns. In this project, each "model" specializes in detecting specific things (fire, helmets, falls, etc.). |
| **Detection** | Object Detection | The AI's ability to find and identify objects in an image (e.g., finding a person or fire in a video frame). |
| **Classification** | | Assigning a label to an object (e.g., this object is a "helmet", that object is "no_helmet"). |
| **Inference** | | The process of running an AI model to make predictions. When the camera analyzes video to detect safety violations, it's performing "inference." |
| **mAP** | mean Average Precision | A standard metric to measure how accurate an AI model is. Higher mAP = better accuracy. mAP@0.5 means measuring accuracy when the predicted box overlaps the ground truth by at least 50%. |
| **Confidence** | Confidence Score | A value from 0 to 1 (or 0% to 100%) indicating how sure the AI is about its detection. A confidence of 0.90 means 90% sure the object is correctly identified. |
| **Confidence Threshold** | | The minimum confidence level required to trigger an alert. For example, if threshold = 0.70, only detections with 70%+ confidence will generate alerts. |
| **Bounding Box** | | A rectangle drawn around a detected object. The AI finds objects by drawing these boxes around them. |
| **Keypoint** | | A specific point of interest on an object. In human pose estimation, keypoints are body joints (nose, eyes, shoulders, elbows, wrists, hips, knees, ankles). |
| **Pose Estimation** | | The AI's ability to detect the position and orientation of a person's body. Uses 17 keypoints (COCO format) to understand body posture. |
| **Tracking** | | Following the same person across multiple video frames. Assigns a unique ID to each person so the system knows "Person A" moved from position X to position Y. |
| **FPS (Inference)** | Frames Per Second (AI Processing) | How many video frames the AI can analyze per second. This is different from camera FPS. The AI needs to keep up with the camera (ideally also 30 FPS, but 15+ FPS is acceptable). |
| **ONNX** | Open Neural Network Exchange | A standard file format for AI models. Allows models trained in one framework (like PyTorch) to run on different hardware (like Hailo-8 NPU). |
| **NPU** | Neural Processing Unit | A specialized chip designed to run AI models efficiently. More efficient than general-purpose CPUs for AI workloads. |
| **TOPS** | Operations Per Second | A measure of AI computing power. Higher TOPS = more powerful. Hailo-8 has 26 TOPS. |
| **YOLO** | You Only Look Once | A popular and fast AI model architecture for object detection. Variants include YOLO11, YOLO26, etc. |
| **COCO** | Common Objects in Context | A famous dataset used to train AI models. COCO format is a standard for object detection and pose estimation. |

### Hardware & Computing Terms

| Term | Full Name | Explanation |
|---|---|---|
| **RPi** | Raspberry Pi | A small, low-cost computer. Raspberry Pi 5 is the latest version (as of 2024-2025). |
| **HAT** | Hardware Attached on Top | An add-on board that connects to a Raspberry Pi. PoE HAT provides Power over Ethernet capability. |
| **PoE Switch** | Power over Ethernet Switch | A network switch that provides both data and power to connected devices (like IP cameras) via Ethernet cables. |
| **Edge AI** | Edge Artificial Intelligence | Running AI directly on the device (at the "edge" of the network) rather than sending video to a central server. Faster, more private, works without internet. |
| **GPU** | Graphics Processing Unit | A specialized chip originally designed for graphics, now widely used for AI training. Training models requires powerful GPUs (like NVIDIA RTX 4090). |
| **VRAM** | Video RAM | Memory on the GPU. More VRAM allows training with larger images or larger batch sizes. |
| **CPU** | Central Processing Unit | The main processor in a computer. Can run AI but is slower than GPU or NPU. |
| **RAM** | Random Access Memory | The main memory in a computer. More RAM allows handling larger datasets and running more complex models. |

### Software & Development Terms

| Term | Full Name | Explanation |
|---|---|---|
| **Annotation** | Labeling | The process of marking objects in images to teach the AI. For example, drawing boxes around helmets to teach the AI what helmets look like. |
| **Dataset** | | A collection of images (or videos) used to train and test AI models. |
| **Training** | | The process of teaching an AI model by showing it many examples. The model learns patterns from the data. |
| **Test Set** | | A portion of the dataset kept separate from training. Used to evaluate how well the model performs on unseen data. |
| **Validation** | | Checking how well the model is performing during training. Used to tune the model. |
| **Epoch** | | One complete pass through the entire training dataset. Training for 100 epochs means the model sees all training data 100 times. |
| **Batch Size** | | How many images the AI processes at once during training. Larger batch size = faster training but requires more memory (VRAM). |
| **Augmentation** | Data Augmentation | Techniques to artificially increase dataset size by modifying images (flipping, rotating, adjusting brightness, adding noise). Helps the model learn better. |
| **Fine-tuning** | | Taking a pre-trained model (trained on a large dataset like COCO) and training it further on a specific dataset (like factory images). Faster than training from scratch. |
| **Transfer Learning** | | Similar to fine-tuning. Using knowledge learned from one task to help with a different but related task. |
| **Ground Truth** | | The correct answer (human-annotated labels) that the AI tries to predict. |
| **False Positive** | | When the AI incorrectly detects something that isn't there (e.g., says "fire" when it's just steam). |
| **False Negative** | | When the AI misses something that is actually there (e.g., doesn't detect a person who fell). |
| **Precision** | | Of all the detections the AI made, how many were correct? High precision = few false alarms. |
| **Recall** | | Of all the actual events, how many did the AI detect? High recall = few missed events. |
| **F1 Score** | | A combined metric that balances precision and recall. |

### Deployment & Alert Terms

| Term | Full Name | Explanation |
|---|---|---|
| **Alert Latency** | | The time delay between when an event happens (e.g., fire starts) and when the alert is triggered. Lower is better. |
| **Polygon Zone** | | A defined area marked by connecting multiple points to form a polygon. Used to specify restricted areas on camera view. |
| **Line Crossing** | | Detecting when an object crosses a virtual line drawn on the camera view. |
| **Loitering** | | Remaining in one area for too long. Used to detect unauthorized lingering. |
| **MQTT** | Message Queuing Telemetry Transport | A lightweight communication protocol used for IoT devices. Cameras send alerts via MQTT to the central server. |
| **Dashboard** | | A visual interface showing alerts, camera feeds, and system status. Used by security personnel or supervisors. |
| **Siren** | | An audible alarm device that sounds when critical alerts are triggered. |

### Domain-Specific Terms

| Term | Full Name | Explanation |
|---|---|---|
| **PPE** | Personal Protective Equipment | Safety equipment workers must wear, including helmets, safety shoes, goggles, gloves, vests, etc. |
| **Poketenashi** | ポケなし (Japanese) | Japanese safety practice meaning "no pockets." Refers to safety rules against walking with hands in pockets, using phones while walking, not holding handrails. |
| **Nitto Hat** | | A specific brand/type of soft safety hat used in Japanese factories. |
| **Full Harness** | | A type of safety belt/harness that wraps around the full body for fall protection. |
| **Waist Belt** | | A simpler safety belt that wraps around the waist only. |
| **Pointing and Calling** | | A Japanese safety practice (指差喚呼, shisa kanko) where workers point at safety hazards and verbally confirm them aloud. |

### File Format & Data Terms

| Term | Full Name | Explanation |
|---|---|---|
| **YOLO Format** | | A specific file format for storing object detection annotations. Each image has a corresponding .txt file with bounding box coordinates. |
| **COCO Format** | | A standard JSON format for storing object detection and pose estimation annotations. |
| **DVC** | Data Version Control | Like Git, but for large datasets. Tracks changes to datasets over time. |
| **W&B** | Weights & Biases | A tool for tracking AI experiments, visualizing training progress, and comparing different model versions. |

### Company & Product Names

| Term | Full Name | Explanation |
|---|---|---|
| **Hailo** | Hailo Robotics | An Israeli AI chip company that makes the Hailo-8 NPU. |
| **Ultralytics** | | The company that develops YOLO11 and provides the training software. |
| **Roboflow** | | A company providing tools for dataset management, annotation, and computer vision workflows. |
| **SAM 3** | Segment Anything Model 2 | Meta's AI model for automatic image segmentation. Used in Label Studio for auto-annotation. |
| **Label Studio** | | An open-source tool for labeling/annotating datasets.

---

# External References & Additional Resources

This section contains all external links referenced in the customer materials, along with additional information gathered from those sources to support the Phase 1 requirements.

## Customer Presentation Links

The following links were extracted from the customer's presentation materials:

| Model | Link | Description |
|---|---|---|
| **Fire Detection** | [prtimes.jp](https://prtimes.jp/main/html/rd/p/000000071.000014310.html) | Fire detection reference with 325m detection specifications |
| **PPE Levels** | [Kurita KCR](https://kcr.kurita.co.jp/solutions/videos/049.html) | PPE classification levels A-E with chemical handling requirements |
| **Fall Detection** | [Web Japan](https://www.webjapan.co.jp/solution/ai-camera/) | AI camera fall detection solutions |
| **Intrusion Detection** | [DCross](https://dcross.impress.co.jp/docs/news/001320.html) | Line crossing, intrusion, and loitering detection |

---

## Detailed Information from External Sources

### A. Fire Detection Technology

**Source:** [prtimes.jp Fire Detection Article](https://prtimes.jp/main/html/rd/p/000000071.000014310.html)

#### Key Specifications (Confirmed from Customer Reference)

| Fire Source Size | Detection Distance | Notes |
|---|---|---|
| 1 × 1 m fire source | Up to **325 m** | Large fire, very long range detection |
| 0.5 × 0.5 m fire source | Up to **162.5 m** | Medium fire, long range detection |
| 0.2 × 0.2 m fire source | Up to **65 m** | Small fire, medium range detection |

| Smoke Source Size | Detection Distance | Notes |
|---|---|---|
| 1 × 1 m smoke area | Up to **65 m** | Large smoke cloud |
| 0.5 × 0.5 m smoke area | Up to **32.5 m** | Medium smoke cloud |
| 0.2 × 0.2 m smoke area | Up to **13 m** | Small smoke cloud |

| Temperature Change Area | Detection Distance | Notes |
|---|---|---|
| 1 × 1 m area | Up to **76 m** | Thermal anomaly before fire |
| 0.5 × 0.5 m area | Up to **38 m** | Early heat rise detection |
| 0.2 × 0.2 m area | Up to **15.2 m** | Small area temperature change |

#### Technology Implementation Notes

**Triple Detection Approach:**
1. **Flame Detection**: Optical sensing of open flames
2. **Smoke Detection**: Visual smoke pattern recognition
3. **Thermal Anomaly Detection**: Temperature change monitoring (requires thermal camera or IR sensors)

**Challenges:**
- Distinguishing smoke from steam, dust, vehicle exhaust
- False positive prevention for reflections and heat sources
- Long-range detection requires high-resolution input (1280px+ recommended)

---

### C. Fall Detection Technology

**Source:** [Web Japan AI Camera Solutions](https://www.webjapan.co.jp/solution/ai-camera/) + 2026 Industry Research

#### 2026 Technology Landscape

**Emerging Technologies (2026):**
1. **Sony EVS Fall Detection Camera** (Mass Production: Early 2026)
   - Uses light change detection instead of video recording
   - Privacy-preserving (no actual video captured)
   - Processes light changes 10,000+ times per second
   - Currently in trials at Shanghai and Hong Kong nursing homes

2. **SimpleAIbox System**
   - 98% detection accuracy with <2% false positive rate
   - Compatible with YOLOv8
   - 480 frames/second processing capability
   - Multi-person simultaneous monitoring

3. **Tuya-based Radar Systems**
   - 60GHz millimeter wave radar
   - Price range: $200-1,445
   - Features: Fall alerts, prolonged sitting detection, ADL reporting
   - WiFi-enabled with cloud integration

#### Pose-Based Fall Detection (Recommended for Phase 1)

**Multi-Person Monitoring:**
- Single camera can monitor multiple people simultaneously
- Analyzes 10+ different postures: standing, walking, falling, waving for help
- Motion trajectory analysis with 5-second anti-shake mechanism

**Detection States:**

| State | Description | Action |
|---|---|---|
| Normal | Standing, walking, sitting | No action |
| Unsafe | Crouching, bending, climbing | Warning if in hazardous area |
| Fall | Horizontal posture + near ground | Immediate alert |

**Technical Implementation:**
```python
# Fall detection logic (based on industry standards)
def detect_fall(person_pose):
    # Check if person is horizontal
    if person_pose.aspect_ratio > 1.5:
        # Check if near ground (lower 30% of image)
        if person_pose.hip_y > 0.7:
            # Verify with temporal consistency
            if confirm_over_time(frames=15, threshold=0.8):
                return "FALL_DETECTED"
    return "NORMAL"
```

**Privacy Considerations:**
- Traditional video: Higher accuracy but privacy concerns
- Light-based detection: Better privacy but limited to motion detection
- Recommended: Use pose skeleton (keypoints only) to preserve anonymity

---

### D. Poketenashi (Pocket-Nashi) Safety Practice

**Sources:** Japanese Safety Research + Customer Presentation

#### What is Poketenashi?

**Poketenashi** (ポケなし) is a Japanese safety acronym for safe walking behaviors, especially important in manufacturing and industrial environments.

#### The Five Poketenashi Rules

| Acronym | Japanese | English Translation | Behavior |
|---|---|---|---|
| **Po** | ポケット (Poketto) | **Pocket** | ❌ Don't walk with hands in pockets |
| **Ke** | 携帯 (Keitai) | **Mobile Phone** | ❌ Don't use phone while walking |
| **Te** | 手 (Te) | **Hand** | ✅ Hold handrails on stairs |
| **Na** | 〇〇 (variable) | Various | Proper posture/positioning |
| **Shi** | 〇〇 (variable) | Various | situational awareness |

#### Why is this Important?

**Safety Impact:**
- Walking while using phones reduces field of vision to **5% of normal**
- Hands in pockets prevents quick reaction to hazards
- Handrail use prevents serious falls on stairs

**Cultural Context:**
- Japan's "aruki-sumaho" (歩きスマホ) problem: Walking while using smartphone
- Public awareness campaigns about distracted walking
- Some phones lock automatically when detecting walking motion
- Public transportation systems announce distracted walking violations

#### AI Detection Implementation

**Detectable Behaviors:**

1. **Hands in Pockets**
   - **Pose-based detection**: Wrist position near hip level
   - **Confidence check**: Wrists close together and near body center
   - **Minimum duration**: 1 second to avoid false positives

2. **Phone Usage While Walking**
   - **Pose-based**: Wrist near face/head level
   - **Alternative**: Train separate phone object detector
   - **Challenge**: Phone may be small at distance

3. **Not Holding Handrail on Stairs**
   - **Zone-based**: Active only in designated stair zones
   - **Hand detection**: Check if hands are near handrail regions
   - **Zone configuration**: Requires manual annotation of handrail areas

4. **Unsafe Stair Crossing**
   - **Trajectory analysis**: Detect diagonal movement across stairs
   - **Line crossing**: Virtual lines across stair steps
   - **Direction detection**: Moving perpendicular to stair direction

**Technical Challenges:**
- Fine-grained pose analysis required
- Camera angle affects accuracy
- Crowded scenes: multiple people on stairs
- Cultural variations: carrying items vs. hands in pockets

---

### E. AI Intrusion Detection Technology

**Sources:** [DCross Article](https://dcross.impress.co.jp/docs/news/001320.html) + 2026 Security Industry Research

#### Core Detection Functions

**1. Intrusion Detection**
- Detects people/vehicles entering restricted zones
- Polygon-based zone definition
- Real-time alerts upon entry

**2. Line Crossing Detection**
- Virtual lines drawn on camera view
- Directional detection: A→B vs B→A
- Crosses specific prohibited or restricted lines

**3. Loitering Detection**
- Vehicles or people stationary too long
- Configurable time threshold (e.g., 30 seconds)
- No-parking zone monitoring

**4. Direction Detection**
- Wrong-way movement detection
- One-way street enforcement
- Driving in opposite direction

#### 2026 Technology Updates

**February 2026 Developments:**
- **Synology + ABUS Integration**: Advanced cross-line detection
- **Perimeter Defender**: Multi-zone monitoring
- **PTZ Auto-Tracking**: Cameras automatically track intruders

**Enhanced Features:**
- **Dual Light Source**: Audio + visual deterrents
- **Color Night Vision**: Full-color low-light detection
- **Multi-Target Support**: Humans, vehicles, non-motor vehicles
- **Real-time Alerts**: Immediate notification system

#### Supported Camera Models (2026)

| Manufacturer | Models | Key Features |
|---|---|---|
| **ACTi** | Multiple IP cameras | People line crossing with advanced image processing |
| **UNV/Uniview** | AI camera series | Line crossing + intrusion detection |
| **ABUS** | (via Synology) | Cross-line detection + perimeter defender |
| **Synology** | Surveillance integration | Central management with multiple camera support |

#### Deployment Considerations

**Zone Configuration:**
- Each camera's zones must be manually annotated
- Polygon vertices map to real-world coordinates
- Calibration required for accurate measurements

**Performance Requirements:**
- Higher FPS needed for reliable tracking (20+ FPS recommended)
- Multi-target tracking: ByteTrack or BotSORT
- Alert latency: < 1 second for intrusion events

**Privacy Considerations:**
- Masking zones for public areas
- Face blurring options in some systems
- Data retention policies vary by region

---

### F. AI Helmet & PPE Detection Technology

**Sources:** 2026 Industry Research + YOLO Development Community

#### 2026 Technology State-of-the-Art

**Advanced YOLO Models:**
- **YOLOv12, YOLO11, YOLO10, YOLO8**: Latest versions for PPE detection
- **Construction-PPE Dataset** (Ultralytics): Comprehensive dataset with 11 classes
  - Worn PPE: helmet, vest, gloves, goggles, boots
  - Missing PPE: no_helmet, no_vest, no_gloves, no_goggles
- **Accuracy**: 95%+ for helmet detection with latest models

#### System Capabilities

**1. Multi-Class PPE Detection**
- Simultaneous detection of multiple PPE items
- Compliance checking: All required PPE present?
- Real-time monitoring with web interfaces

**2. Technical Implementation**
```python
# Example: Helmet detection with YOLO
from ultralytics import YOLO

model = YOLO('yolo11n.pt')  # Latest YOLO model
results = model(frame)        # Real-time detection

# Extract helmet detections
for detection in results:
    if detection.class == 'helmet':
        if detection.confidence > 0.70:
            # Person wearing helmet
        elif detection.class == 'no_helmet':
            if detection.confidence > 0.70:
                # Alert: PPE violation
```

**3. Compliance Analysis**
- **Multi-class detection**: Helmet + vest + gloves simultaneously
- **Compliance checking**: Automated verification
- **Alert systems**: Configurable confidence thresholds
- **Analytics**: Compliance rate tracking over time

#### Industry Applications (2026)

| Industry | Use Case | Key Requirements |
|---|---|---|
| **Construction** | Site-wide monitoring | Outdoor conditions, multiple workers |
| **Manufacturing** | Factory floor compliance | Indoor lighting, machinery background |
| **Mining** | Underground safety | Low-light, harsh environment |
| **Smart Surveillance** | Network of cameras | Centralized management, scalability |

#### Helmet Detection Challenges

**Specific Challenges for Factory Environment:**

1. **Helmet Variety**
   - Different types: hard hats, bump caps, Nitto soft hats
   - Different colors: white, yellow, blue, red
   - Worn vs. not worn: must detect absence

2. **Camera Angles**
   - Ceiling-mounted: Top-down view
   - Wall-mounted: Side view
   - Vari-focal lenses: Different zoom levels affect detection

3. **Lighting Conditions**
   - IR mode (black/white images)
   - Low-light (0.16 lux)
   - Backlighting from windows or equipment

4. **Partial Occlusion**
   - Workers behind machinery
   - Workers carrying items
   - Multiple workers overlapping

#### Detection Distance Requirements

| Input Resolution | Detection Distance | Notes |
|---|---|---|
| 640 × 640 | Up to 15m | Standard deployment |
| 1280 × 1280 | Up to 30m | For distant workers or vari-focal zoom |

**Recommendation:** Use 1280px input for cameras with vari-focal lenses set to telephoto (narrow FOV) to maintain detection at distance.

---

### G. AI Safety Shoes Detection Technology

**Sources:** 2026 Industry Research + PPE Detection Solutions

#### Technology Overview

**1. Amazon Rekognition PPE Detection**
- Detects safety shoes/boots in camera feeds
- Real-time compliance checking
- Automated alerts for missing PPE
- Integration with existing camera systems

**2. YOLO-based Detection**
- **YOLOv8/11 implementations**: Detect safety boots, hard hats, vests
- **HSV Heuristic Algorithms**: Combined with YOLO for improved shoe detection
- **Real-time monitoring**: Industrial-grade deployment

**3. Open Source Solutions**
- Multiple GitHub repositories with working implementations
- Construction-PPE dataset includes shoe classes
- Real-time inference on edge devices

#### Detection Challenges

**Why is Safety Shoes Detection Difficult?**

1. **Occlusion Issues**
   - Feet often hidden by machinery, conveyors, other workers
   - Factory floor: many obstacles block foot view
   - Workers standing behind equipment

2. **Camera Angle**
   - Ceiling-mounted cameras: See top of foot only
   - Side views needed for shoe type identification
   - Multi-angle training data required

3. **Lighting & Shadows**
   - Floor shadows obscure foot area
   - Reflective flooring causes glare
   - Uneven lighting in factory environments

4. **Motion Blur**
   - Walking feet blur at 30 FPS
   - Faster movement = more blur
   - Affects shoe type classification

#### Shoe Types to Detect

| Category | Types | Detection Priority |
|---|---|---|
| **Safety Shoes** | Steel-toe boots, chemical-resistant boots, ankle boots | High |
| **Non-Compliance** | Sneakers, sandals, street shoes, open-toe shoes | High |
| **Specialized** | ESD shoes, anti-static, slip-resistant | Medium |

#### Technical Recommendations

**Input Resolution:**
- **640 × 640**: For workers within 10m
- **1280 × 1280**: For workers 10-25m away

**Bounding Box Strategy:**
```
┌─────────────────────────────┐
│                             │
│   (Upper body: person)       │
│                             │
├─────────────────────────────┤  ← Chest level
│                             │
│   [Shoes: safety/no-safety] │  ← Separate detection
│                             │
└─────────────────────────────┘
```

**Two-Stage Detection:**
1. **Person Detection**: Find full body
2. **Shoe Region Analysis**: Classify shoes in lower region of person bbox

#### Factory-Specific Considerations

**Zone-Based Detection:**
- Different zones may have different shoe requirements
- Example: Office zone vs. Production floor
- Alert only in zones where safety shoes are required

**Seasonal Variations:**
- Winter: Boots may be covered by pants
- Summer: More skin visible, easier detection
- Rain: Wet shoes may look different

---

## Integration Notes for Phase 1

### Linking External Research to Customer Requirements

| Customer Requirement | External Research Source | Key Insight for Phase 1 |
|---|---|---|
| **Fire: 325m detection** | [prtimes.jp](https://prtimes.jp/main/html/rd/p/000000071.000014310.html) | Requires 1280px input for distant/small fires |
| **Helmet: Nitto hats** | PPE detection research | Multiple helmet types require diverse training data |
| **Shoes: Detection** | Amazon/YOLO research | Feet occlusion is major challenge; use two-stage detection |
| **Fall: Immediate alert** | Sony EVS / SimpleAIbox | Temporal validation prevents false alarms |
| **Poketenashi** | Japanese safety research | Pose-based detection with fine-grained keypoint analysis |
| **Intrusion: Zones** | DCross / 2026 camera research | Zone polygons must be manually annotated per camera |

### Technology Readiness Assessment

| Model | Technology Maturity | Production Ready? | Notes |
|---|---|---|---|
| Fire Detection | High | ✅ Yes | Many commercial solutions available |
| Helmet Detection | High | ✅ Yes | YOLOv8-11 achieve 95%+ accuracy |
| Safety Shoes | Medium | ⚠️ Partial | Challenging; may need 1280px input |
| Fall Detection | High | ✅ Yes | Pose-based approach mature |
| Poketenashi | Low | ❌ No | Requires custom development |
| Intrusion | High | ✅ Yes | Standard feature in AI cameras |

### Recommended Phase 1 Approach

Based on external research:

1. **Prioritize High-Maturity Models**
   - Fire, Helmet, Fall, Intrusion have proven solutions
   - Poketenashi and Safety Shoes require more R&D

2. **Leverage Open Datasets**
   - Construction-PPE (Ultralytics) for Helmet
   - COCO Pose for Fall detection
   - COCO pretrained for Intrusion (person only)

3. **Prepare for Challenges**
   - Safety shoes: Collect factory-specific data
   - Poketenashi: May need to narrow scope (focus on 1-2 behaviors first)

4. **Hardware Planning**
   - 1280px input capability for distant detection
   - 15+ FPS for reliable tracking (Intrusion, Fall)
   - Model optimization for < 12W power constraint (INT8 quantization, pruning)
   - ONNX export for hardware portability across < 12W edge devices

---

## Summary of External Links

### Customer Presentation Links (Integrated into Section A)

The following links from the customer's presentation have been integrated directly into **Section A: Customer Requirements** above, alongside each model description:

| Model | Customer Link | Location in Document |
|---|---|---|
| **Fire Detection** | [prtimes.jp](https://prtimes.jp/main/html/rd/p/000000071.000014310.html) | Section A.4, Model A |
| **PPE Levels** | [Kurita KCR](https://kcr.kurita.co.jp/solutions/videos/049.html) | Section A.4, Model B |
| **Fall Detection** | [Web Japan](https://www.webjapan.co.jp/solution/ai-camera/) | Section A.4, Model G |
| **Intrusion Detection** | [DCross](https://dcross.impress.co.jp/docs/news/001320.html) | Section A.4, Model I |

### Additional Research Sources (Technical Team)

The following are additional research sources consulted by the technical team for context and industry best practices:

1. **YOLO / Ultralytics** - [docs.ultralytics.com](https://docs.ultralytics.com) - Latest YOLO models for PPE detection
2. **Construction-PPE Dataset** - Available through Ultralytics docs - Comprehensive PPE dataset with 11 classes
3. **Roboflow Universe** - [universe.roboflow.com](https://universe.roboflow.com) - Open PPE datasets
4. **ONNX Runtime** - [onnxruntime.ai](https://onnxruntime.ai) - Cross-platform inference for model portability
5. **FiftyOne** - [voxel51.com/fiftyone](https://voxel51.com/fiftyone) - Error analysis and model debugging
6. **Edge AI Hardware** - Various vendors (Hailo, Rockchip, Google Coral, Renesas, etc.) - Reference only for <12W constraint evaluation

---

**Last Updated:** March 5, 2026
**Research Completed By:** Vietsol AI Technical Team
**Sources:** All links verified and accessible as of document date |

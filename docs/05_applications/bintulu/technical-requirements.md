# Bintulu Smart City - Technical Requirements Definition

**Date**: 2026-03-19
**Author**: Nguyen Thanh Trung (AI Camera Lead, VIETSOL)
**Customer**: Bintulu Development Authority (BDA), Sarawak, Malaysia
**Partner**: Elektro Serve Power Sdn Bhd (ESP)

---

## Table of Contents

1. [UC1: AI Traffic Light — Technical Requirements](#uc1-ai-traffic-light--technical-requirements)
2. [UC2: Smart Parking — Technical Requirements](#uc2-smart-parking--technical-requirements)
3. [Shared Infrastructure Requirements](#shared-infrastructure-requirements)
4. [Acceptance Criteria](#acceptance-criteria)

---

## UC1: AI Traffic Light — Technical Requirements

### 1.1 Functional Requirements

#### FR-T01: Real-Time Vehicle Detection & Classification

| ID | Requirement | Priority |
|----|------------|----------|
| FR-T01.1 | Detect and classify vehicles into categories: car, motorcycle, truck, bus, bicycle, pedestrian | Must |
| FR-T01.2 | Operate at >= 15 FPS per camera stream on edge hardware | Must |
| FR-T01.3 | Achieve >= 90% mAP@0.5 for vehicle detection under normal conditions | Must |
| FR-T01.4 | Maintain >= 80% accuracy under adverse conditions (rain, night, glare) | Must |
| FR-T01.5 | Support input resolution of 1080p (1920x1080) from IP cameras | Must |
| FR-T01.6 | Process multiple camera streams simultaneously (1 per approach direction, typically 4 per intersection) | Must |

#### FR-T02: Adaptive Signal Timing

| ID | Requirement | Priority |
|----|------------|----------|
| FR-T02.1 | Dynamically adjust green/red phase durations based on real-time traffic density per approach | Must |
| FR-T02.2 | Support configurable minimum/maximum green times per phase | Must |
| FR-T02.3 | Support configurable all-red clearance intervals | Must |
| FR-T02.4 | Update signal timing decisions within 2 seconds of detection input | Must |
| FR-T02.5 | Support multiple intersection timing plans (peak, off-peak, weekend, special event) | Should |
| FR-T02.6 | Provide manual override capability for operators | Must |

#### FR-T03: Multi-Lane Traffic Flow Monitoring

| ID | Requirement | Priority |
|----|------------|----------|
| FR-T03.1 | Define configurable polygon ROI zones for each lane of each approach | Must |
| FR-T03.2 | Count vehicles per lane per time interval (1-min, 5-min, 15-min, 1-hour bins) | Must |
| FR-T03.3 | Estimate queue length per lane (number of stopped vehicles) | Must |
| FR-T03.4 | Measure lane-level average speed (stopped vs. flowing threshold) | Should |
| FR-T03.5 | Detect lane imbalance and report to operator | Should |

#### FR-T04: Incident & Anomaly Detection

| ID | Requirement | Priority |
|----|------------|----------|
| FR-T04.1 | Detect stalled/stopped vehicles in intersection area beyond a configurable time threshold | Must |
| FR-T04.2 | Detect wrong-way vehicles (vehicle moving against expected lane direction) | Should |
| FR-T04.3 | Detect pedestrians in vehicle lanes | Should |
| FR-T04.4 | Generate real-time alerts with frame snapshot and timestamp | Must |
| FR-T04.5 | Alert delivery via MQTT to central platform within 5 seconds of detection | Must |

#### FR-T05: Emergency Vehicle Priority

| ID | Requirement | Priority |
|----|------------|----------|
| FR-T05.1 | Detect emergency vehicles (ambulance, fire truck, police) approaching intersection | Should |
| FR-T05.2 | Trigger signal preemption (force green on emergency approach, all-red on others) | Should |
| FR-T05.3 | Return to normal signal plan within 1 cycle after emergency vehicle clears | Should |
| FR-T05.4 | Log all preemption events with timestamp, vehicle type, approach direction | Should |

#### FR-T06: Central Traffic Management Platform

| ID | Requirement | Priority |
|----|------------|----------|
| FR-T06.1 | Web-based dashboard showing real-time status of all managed intersections | Must |
| FR-T06.2 | Display live vehicle counts, queue lengths, signal phase status per intersection | Must |
| FR-T06.3 | Historical analytics: traffic volume trends, peak hours, congestion patterns | Must |
| FR-T06.4 | Alert management: view, acknowledge, resolve incident alerts | Must |
| FR-T06.5 | Signal timing configuration interface (create/edit timing plans remotely) | Must |
| FR-T06.6 | Map view with intersection health indicators (green/yellow/red status) | Should |
| FR-T06.7 | Report generation (daily, weekly, monthly traffic summaries) | Should |
| FR-T06.8 | Multi-user access with role-based permissions (admin, operator, viewer) | Must |

#### FR-T07: Signal Controller Integration

| ID | Requirement | Priority |
|----|------------|----------|
| FR-T07.1 | Interface with existing traffic signal controllers via standard protocol (NTCIP 1202 / SNMP preferred, or vendor-specific API) | Must |
| FR-T07.2 | Read current signal phase/timing state from controller | Must |
| FR-T07.3 | Send phase change commands to controller | Must |
| FR-T07.4 | Failsafe: revert to pre-programmed fixed-time plan if AI system becomes unavailable | Must |
| FR-T07.5 | Support graceful degradation (if one camera fails, use remaining cameras) | Must |

### 1.2 Non-Functional Requirements

| ID | Requirement | Category |
|----|------------|----------|
| NFR-T01 | Edge device must operate 24/7 in outdoor enclosure (IP65+, -10C to 60C) | Environmental |
| NFR-T02 | Edge device power consumption <= 60W per intersection | Power |
| NFR-T03 | System availability >= 99.5% (max 44 hours downtime/year) | Reliability |
| NFR-T04 | End-to-end latency (camera frame → signal decision) <= 500ms | Performance |
| NFR-T05 | Edge device must operate independently when cloud connection is lost | Reliability |
| NFR-T06 | All video processed locally; only metadata sent to cloud (privacy compliance) | Privacy |
| NFR-T07 | Support OTA firmware/model updates | Maintainability |
| NFR-T08 | System must be scalable to 50+ intersections city-wide | Scalability |
| NFR-T09 | Encrypted communication (TLS 1.3) between edge and cloud | Security |

---

## UC2: Smart Parking — Technical Requirements

### 2.1 Functional Requirements

#### FR-P01: Vehicle Detection & Bay Occupancy Monitoring

| ID | Requirement | Priority |
|----|------------|----------|
| FR-P01.1 | Detect vehicles in parking bays and classify each bay as occupied or vacant | Must |
| FR-P01.2 | Support configurable polygon ROI zones defining each parking bay | Must |
| FR-P01.3 | Achieve >= 95% occupancy detection accuracy under normal conditions | Must |
| FR-P01.4 | Maintain >= 90% accuracy under adverse conditions (rain, night, shadow, glare) | Must |
| FR-P01.5 | Update occupancy status within 5 seconds of vehicle arrival/departure | Must |
| FR-P01.6 | Support at least 40-60 bays per camera (outdoor lot, pole-mounted) | Must |
| FR-P01.7 | Handle total capacity of 7,000 parking spaces across multiple zones | Must |
| FR-P01.8 | Detect vehicle type (car, motorcycle, truck) for each occupied bay | Should |

#### FR-P02: Automatic Number Plate Recognition (ANPR)

| ID | Requirement | Priority |
|----|------------|----------|
| FR-P02.1 | Detect and read license plates at entry/exit points | Must |
| FR-P02.2 | Achieve >= 95% plate recognition accuracy under good conditions (day, clear) | Must |
| FR-P02.3 | Achieve >= 90% plate recognition accuracy at night with IR illumination | Must |
| FR-P02.4 | Support Malaysian license plate formats | Must |
| FR-P02.5 | Process plates within 1 second of vehicle detection | Must |
| FR-P02.6 | Record entry/exit timestamps per plate for dwell time calculation | Must |
| FR-P02.7 | Match plates against permit/whitelist database for access control | Should |
| FR-P02.8 | Capture plate image snapshot and store for manual review | Must |

#### FR-P03: AI-Based Safety Monitoring

| ID | Requirement | Priority |
|----|------------|----------|
| FR-P03.1 | Detect loitering: person stationary in restricted area beyond configurable time threshold (default 5 minutes) | Must |
| FR-P03.2 | Detect intrusion: person/vehicle entering restricted zone (after-hours areas, staff-only zones) | Must |
| FR-P03.3 | Detect accidents: fallen person, vehicle collision indicators | Should |
| FR-P03.4 | Generate real-time alerts with frame snapshot, location, timestamp | Must |
| FR-P03.5 | Configurable alert zones per camera via polygon ROI | Must |
| FR-P03.6 | Alert cooldown to prevent repeated notifications for the same event | Must |

#### FR-P04: Violation Detection

| ID | Requirement | Priority |
|----|------------|----------|
| FR-P04.1 | Detect double parking: vehicle detected outside marked bay zones for > configurable time (default 2 minutes) | Must |
| FR-P04.2 | Detect obstruction: vehicle blocking access lanes or fire lanes | Should |
| FR-P04.3 | Detect overtime parking: vehicle exceeding maximum allowed time in zone | Should |
| FR-P04.4 | Capture violation evidence: plate image + violation image + timestamp | Must |
| FR-P04.5 | Link violations to ANPR data when available | Should |

#### FR-P05: Real-Time Guidance

| ID | Requirement | Priority |
|----|------------|----------|
| FR-P05.1 | Provide real-time count of available spaces per zone/floor/area | Must |
| FR-P05.2 | Drive LED guidance signs showing available space counts per zone | Must |
| FR-P05.3 | Provide REST API for real-time occupancy data consumption by mobile app | Must |
| FR-P05.4 | Mobile app: show parking map with available bays, navigate to open spot | Should |
| FR-P05.5 | Mobile app: support pre-booking and mobile payment | Could |
| FR-P05.6 | Support color-coded guidance (green/yellow/red per zone fullness) | Should |

#### FR-P06: Centralized Dashboard & Analytics

| ID | Requirement | Priority |
|----|------------|----------|
| FR-P06.1 | Web-based operator dashboard with real-time occupancy map | Must |
| FR-P06.2 | Live camera feeds with AI overlay (bounding boxes, bay status) | Must |
| FR-P06.3 | Historical analytics: occupancy trends, peak hours, average dwell time | Must |
| FR-P06.4 | Revenue analytics: collected vs. expected revenue, payment compliance rate | Should |
| FR-P06.5 | Violation reporting: daily/weekly/monthly violation summaries | Must |
| FR-P06.6 | Alert management: view, acknowledge, escalate safety alerts | Must |
| FR-P06.7 | Export reports (PDF, CSV) for operations and planning | Should |
| FR-P06.8 | Multi-user access with role-based permissions | Must |

#### FR-P07: Payment Integration

| ID | Requirement | Priority |
|----|------------|----------|
| FR-P07.1 | Calculate parking fee based on ANPR entry/exit timestamps and zone rate (MYR 0.50/hour) | Must |
| FR-P07.2 | Integrate with at least one mobile payment provider (e.g., Touch 'n Go, Boost, GrabPay) | Should |
| FR-P07.3 | Support pre-paid/subscription permits with automatic gate control | Could |
| FR-P07.4 | Generate daily revenue reconciliation reports | Should |

### 2.2 Non-Functional Requirements

| ID | Requirement | Category |
|----|------------|----------|
| NFR-P01 | Edge device must operate 24/7 in outdoor or semi-outdoor environment (IP65+) | Environmental |
| NFR-P02 | Edge device power consumption <= 25W per camera group (8-12 cameras) | Power |
| NFR-P03 | System availability >= 99.5% | Reliability |
| NFR-P04 | Occupancy status update latency <= 5 seconds | Performance |
| NFR-P05 | ANPR processing latency <= 1 second | Performance |
| NFR-P06 | Edge processing: only metadata sent to backend (no raw video streaming required) | Privacy/Bandwidth |
| NFR-P07 | Network bandwidth <= 5 Mbps total for metadata from all 7,000 bays | Bandwidth |
| NFR-P08 | Support OTA model/firmware updates | Maintainability |
| NFR-P09 | Support phased deployment (zone-by-zone rollout) | Scalability |
| NFR-P10 | System must handle 10x peak events (major event parking surge) | Scalability |
| NFR-P11 | Data retention: 90 days for transaction logs, 30 days for event images | Storage |
| NFR-P12 | Encrypted communication (TLS 1.3) and API authentication | Security |

---

## Shared Infrastructure Requirements

### 3.1 Edge Hardware

| ID | Requirement | Applies To |
|----|------------|-----------|
| INF-01 | Edge AI processor with >= 18 TOPS INT8 performance | Both |
| INF-02 | Support RTSP/ONVIF IP camera input (H.264/H.265) | Both |
| INF-03 | Industrial-grade enclosure (IP65, -10C to 60C operating range) | Both |
| INF-04 | 4G/LTE or Ethernet WAN connectivity | Both |
| INF-05 | Local storage for 7-day event image buffer | Both |
| INF-06 | Hardware watchdog for auto-recovery on crash | Both |
| INF-07 | PoE support for camera power (optional, reduces cabling) | Both |

### 3.2 Camera Requirements

| Parameter | Traffic Light (UC1) | Smart Parking (UC2) |
|-----------|-------------------|---------------------|
| Resolution | >= 2MP (1080p) | >= 2MP (1080p) |
| Frame rate | >= 25 FPS | >= 15 FPS |
| Lens type | Varifocal (2.8-12mm) for intersection coverage | Varifocal or wide-angle for bay coverage |
| Night vision | IR illumination, >= 50m range | IR illumination, >= 30m range |
| Weather | IP67, IK10 vandal-proof | IP66 minimum |
| Protocol | RTSP, ONVIF Profile S | RTSP, ONVIF Profile S |
| Mounting | 6-8m pole height for intersection overview | 4-6m pole/ceiling for parking lot coverage |
| ANPR camera | Not required | Dedicated LPR camera at entry/exit (IR, 5MP, narrow FoV) |

### 3.3 Network Requirements

| Component | Specification |
|-----------|--------------|
| Edge-to-Cloud | >= 10 Mbps per intersection / parking zone |
| Camera-to-Edge | Local network (PoE switch), >= 100 Mbps |
| Cloud backend | Hosted or on-premise server, >= 1 Gbps LAN |
| Redundancy | Dual WAN (4G + fiber) for critical intersections |
| Latency | Edge-to-Cloud RTT <= 100ms |

### 3.4 Cloud/Backend Requirements

| Component | Specification |
|-----------|--------------|
| Compute | 4-8 vCPU, 16-32 GB RAM (scales with intersections/zones) |
| Database | PostgreSQL or TimescaleDB for time-series metrics |
| Message broker | MQTT (Mosquitto) or Kafka for event streaming |
| Storage | 1 TB SSD for 90-day data retention (metadata + event images) |
| Web server | Nginx + application backend (FastAPI or Node.js) |
| OS | Ubuntu 22.04 LTS or 24.04 LTS |

---

## Acceptance Criteria

### UC1: AI Traffic Light

| Metric | Target | Test Method |
|--------|--------|-------------|
| Vehicle detection mAP@0.5 | >= 90% | Evaluate on BDA intersection test dataset (500+ images) |
| Vehicle classification accuracy | >= 85% per class | Per-class AP evaluation |
| Detection FPS (per camera) | >= 15 FPS on edge hardware | Benchmark on target edge device |
| Queue length estimation accuracy | >= 80% (within +-2 vehicles) | Manual count vs. system count on 100 samples |
| Signal phase response time | <= 2 seconds from detection to command | Instrumented end-to-end timing test |
| Congestion reduction | >= 15% average delay reduction within 3 months | Before/after travel time comparison |
| System uptime | >= 99.5% over 30-day acceptance period | Monitoring logs |
| Incident detection rate | >= 80% of real incidents detected | Staged + real incident evaluation |

### UC2: Smart Parking

| Metric | Target | Test Method |
|--------|--------|-------------|
| Bay occupancy accuracy | >= 95% overall, >= 90% at night | Ground truth comparison over 7-day period |
| ANPR plate recognition | >= 95% (day), >= 90% (night) | Manual plate verification on 1,000+ entries |
| Occupancy update latency | <= 5 seconds | Timestamp comparison (real event vs. system update) |
| Guidance sign accuracy | Matches actual vacancy within +-3 spaces per zone | Physical count vs. displayed count |
| Safety alert detection rate | >= 85% for loitering/intrusion | Staged event testing (50+ scenarios) |
| Violation detection rate | >= 80% for double parking | Manual review vs. system detections over 14 days |
| System uptime | >= 99.5% over 30-day acceptance period | Monitoring logs |
| Revenue accuracy | >= 98% (calculated vs. actual) | Reconciliation audit over 30 days |

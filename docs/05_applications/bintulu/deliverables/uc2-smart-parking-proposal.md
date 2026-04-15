# AI-Powered Smart Parking Management System

## Proposal for Bintulu Development Authority (BDA)

**7,000 Parking Bays | Camera-Based AI Solution**

---

| | |
|---|---|
| **Prepared by** | VIETSOL (Technology Partner) + ESP — Elektro Serve Power Sdn Bhd (Local Partner) |
| **Prepared for** | Bintulu Development Authority (BDA), Sarawak, Malaysia |
| **Date** | March 2026 |
| **Classification** | STRICTLY CONFIDENTIAL |

*[VIETSOL Logo]  [ESP Logo]*

---
---

# PART A: EXECUTIVE SUMMARY

---

## 1. Overview

VIETSOL and ESP propose a complete AI camera-based smart parking management system covering 7,000 parking bays across Bintulu. The solution uses 170 IP cameras and 15 industrial edge AI processors to detect vehicle occupancy, read license plates, enforce violations, and guide drivers to available spaces — all for a total investment of approximately **$450,000 (MYR 2.1M)**. By processing video entirely at the edge, only lightweight metadata (bay status, plate text, alerts) is transmitted to the central platform, ensuring privacy compliance and minimal bandwidth requirements. This camera-based approach delivers the same functionality as sensor-based alternatives at **3-5x lower cost**, with the added benefit of visual verification, safety monitoring, and ANPR capabilities that sensors simply cannot provide.

---

## 2. The Opportunity

Bintulu's managed parking infrastructure spans **7,000 bays** across multiple zones, currently operating at a rate of **MYR 0.50/hr**. The existing manual enforcement model faces well-documented challenges:

- **Low compliance** — without real-time monitoring, parking violations go undetected and unenforced
- **Revenue leakage** — manual collection and limited oversight result in significant uncaptured revenue
- **No real-time data** — operators lack visibility into occupancy patterns, peak utilization, and zone-level demand
- **Driver frustration** — circling for parking increases congestion and reduces visitor satisfaction

The global smart parking market is projected to reach **$11.1 billion by 2029**, growing at a **23.3% CAGR**. Camera and LPR-based solutions represent **42% of this market** — the fastest-growing and most cost-effective segment. BDA has the opportunity to position Bintulu as a smart city leader in Sarawak by deploying a modern, AI-driven parking management system that pays for itself through improved revenue capture and operational efficiency.

---

## 3. Solution at a Glance

Our solution is built on a proven **3-layer architecture** designed for reliability, scalability, and ease of maintenance:

| Layer | Components | Function |
|-------|-----------|----------|
| **Layer 1: IP Cameras** | 140 occupancy cameras + 30 ANPR cameras | Capture video streams across all parking zones |
| **Layer 2: Edge AI Boxes** | 15 industrial edge processors | Run proven AI detection models locally — process video on-device, transmit only metadata (bay status, plate text, alerts) via MQTT |
| **Layer 3: Central Platform** | Cloud/on-premises management system | Occupancy engine, ANPR matching, operator dashboard, mobile app, LED guidance signs, payment integration |

**Key Design Principles:**

- **Edge-first processing** — video never leaves the edge device; only structured metadata is transmitted
- **Privacy by design** — no video streaming to cloud, no facial recognition, no personal data stored at the edge
- **Standard hardware** — all cameras are off-the-shelf industrial IP cameras (RTSP/ONVIF compatible)
- **Open architecture** — zero vendor lock-in on hardware, cameras, or AI models (all Apache 2.0 / MIT licensed)

---

## 4. Cost Advantage

The camera-based approach delivers a **dramatic cost advantage** over traditional sensor-based systems, while providing superior functionality:

| Approach | Year 1 Investment | Per-Bay Cost | 5-Year TCO | Capabilities |
|----------|-------------------|-------------|------------|-------------|
| **Camera-Based (Our Solution)** | **$450,000** | **~$64/bay** | **$1-3M** | Occupancy + ANPR + Safety + Violations + Guidance + Analytics |
| Sensor-Based (Alternative) | $2-5M | $400-750/bay | $5-10M | Occupancy only (no ANPR, no safety, no visual verification) |

**Why cameras win:**

- One camera covers **40-60 bays** vs. one sensor per bay
- ANPR, safety monitoring, and violation detection are **included at no additional cost** — sensors require separate systems
- Visual evidence for violations and disputes — sensors provide only binary occupied/empty data
- Cameras can be **reused for future smart city applications** (traffic monitoring, public safety, crowd analytics)

---

## 5. Key Performance Metrics

| Metric | Target |
|--------|--------|
| Bay occupancy detection accuracy | **>=95%** (overall), **>=90%** (night) |
| Occupancy update latency | **<=5 seconds** |
| ANPR recognition accuracy | **>=95%** (day), **>=90%** (night) |
| System uptime | **>=99.5%** |
| Guidance sign accuracy | **Within +-3 spaces** per zone |
| Safety alert detection rate | **>=85%** |
| Revenue reconciliation accuracy | **>=98%** |

---

## 6. Implementation Timeline

**Total duration: 10 months** from contract signing to final acceptance.

| Phase | Months | Activities | Deliverables |
|-------|--------|-----------|-------------|
| **Foundation** | 1-2 | Site survey, network design, hardware procurement, AI model training on Bintulu-specific conditions | Site assessment report, network architecture, trained AI models |
| **Platform Development** | 3-4 | Central management platform, operator dashboard, mobile app, LED sign integration, payment gateway integration | Platform beta, mobile app beta, API documentation |
| **Pilot Deployment** | 5-6 | Deploy to 500-1,000 bays in selected high-traffic zones, field testing, model fine-tuning, user acceptance testing | Pilot zone fully operational, performance baseline report |
| **Full Rollout** | 7-8 | Scale to all 7,000 bays, zone-by-zone commissioning, LED sign installation, enforcement workflow activation | All zones live, operator training complete |
| **Stabilization** | 9-10 | Performance optimization, 30-day acceptance testing, documentation handover, support transition | Final acceptance certificate, O&M documentation |

---
---

# PART B: TECHNICAL APPENDIX

---

## 7. Detailed Features

### FR-P01: Bay Occupancy Monitoring

| Aspect | Specification |
|--------|--------------|
| **Detection method** | Proven AI detection models identify vehicles in each camera frame and match detections against pre-configured bay ROI (Region of Interest) polygons |
| **Coverage** | 40-60 bays per camera, depending on viewing angle and bay layout |
| **Accuracy** | >=95% overall, >=90% in low-light / night conditions |
| **Update frequency** | Every 5 seconds per zone |
| **Temporal smoothing** | Bay status changes require sustained detection over multiple frames (configurable threshold) to prevent false transitions from momentary occlusions or passing vehicles |
| **Calibration** | One-time ROI polygon setup per camera via web-based calibration tool |
| **Output** | Per-bay status (occupied/vacant/unknown) with timestamp, published via MQTT |

### FR-P02: Automatic Number Plate Recognition (ANPR)

| Aspect | Specification |
|--------|--------------|
| **Camera placement** | Dedicated high-resolution cameras at zone entry/exit points |
| **Recognition accuracy** | >=95% (daytime), >=90% (nighttime with IR illumination) |
| **Plate format** | Full support for Malaysian plate formats (standard, special series, personalized) |
| **Processing** | Entirely at the edge — only plate text, confidence score, and timestamp are transmitted to the backend |
| **Matching** | Backend correlates entry/exit plate reads for duration-based fee calculation |
| **Evidence capture** | Cropped plate image stored locally for 30 days (configurable) for dispute resolution |
| **Throughput** | Handles peak vehicle flow rates at entry/exit points without queuing |

### FR-P03: Safety Monitoring

| Aspect | Specification |
|--------|--------------|
| **Loitering detection** | Alerts when a person remains stationary for more than 5 minutes (configurable) in a restricted or monitored zone |
| **Intrusion detection** | Detects unauthorized entry into closed parking areas, restricted zones, or after-hours facilities |
| **Fall detection** | Identifies fallen persons using advanced pose estimation and motion analysis |
| **Alert mechanism** | Real-time push notification to operator dashboard and enforcement handhelds, with snapshot image and location |
| **Detection rate** | >=85% across all safety event types |

### FR-P04: Violation Detection

| Aspect | Specification |
|--------|--------------|
| **Double parking** | Detects vehicles stopped outside designated bay boundaries for more than 2 minutes (configurable) |
| **Obstruction** | Identifies vehicles blocking access lanes, fire exits, or disabled bays |
| **Overtime parking** | Flags vehicles exceeding maximum allowed parking duration per zone policy |
| **Evidence capture** | Automatic snapshot with timestamp, bay ID, and plate linkage (where available) for enforcement action |
| **Detection rate** | >=80% for double parking violations |

### FR-P05: Real-Time Parking Guidance

| Aspect | Specification |
|--------|--------------|
| **LED guidance signs** | 40 signs deployed at key decision points (zone entries, intersections, multi-level ramps) |
| **Display content** | Available space count per zone, color-coded status (green = available, yellow = limited, red = full) |
| **Update frequency** | Real-time, synchronized with occupancy engine (<=5 second latency) |
| **Accuracy** | Within +-3 spaces of actual availability per zone |
| **Mobile integration** | REST API feeds real-time availability to the mobile app, enabling drivers to check availability before arriving |
| **Wayfinding** | Mobile app provides turn-by-turn guidance to nearest available bay |

### FR-P06: Dashboard & Analytics

| Aspect | Specification |
|--------|--------------|
| **Operator dashboard** | Web-based, responsive design, role-based access control (admin, operator, enforcement, viewer) |
| **Live view** | Real-time camera feeds with AI overlay showing bay status, detections, and alerts |
| **Occupancy analytics** | Historical trends, peak hours, zone utilization heat maps, day-of-week patterns |
| **Revenue analytics** | Daily/weekly/monthly revenue breakdown by zone, payment method, and duration band |
| **Violation reporting** | Violation logs with evidence, export to PDF/Excel, integration with enforcement workflow |
| **System health** | Camera status, edge device health, network connectivity, AI model performance metrics |

### FR-P07: Payment Integration

| Aspect | Specification |
|--------|--------------|
| **Fee calculation** | Automated duration-based calculation from ANPR entry/exit timestamps |
| **Payment methods** | Mobile payment integration (Touch 'n Go eWallet, Boost, DuitNow QR) |
| **Reconciliation** | Daily automated revenue reconciliation with >=98% accuracy |
| **Grace period** | Configurable grace period per zone (e.g., first 15 minutes free) |
| **Escalation** | Unpaid sessions flagged for enforcement follow-up with plate and evidence linkage |

---

## 8. System Architecture

```
+------------------------------------------------------------------+
|                     LAYER 1: IP CAMERAS                          |
|                                                                  |
|  [Occupancy Cameras x140]           [ANPR Cameras x30]          |
|   2MP, IP67, PoE                     5MP, IR, PoE               |
|   40-60 bays/camera                  Zone entry/exit             |
|        |                                    |                    |
|        +------------- RTSP Streams ---------+                    |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                  LAYER 2: EDGE AI PROCESSING                     |
|                                                                  |
|  [Edge AI Boxes x15]                                             |
|  +------------------------------------------------------------+  |
|  | RTSP Ingest --> Frame Decode --> AI Inference Pipeline      |  |
|  |                                                            |  |
|  |  +------------------+  +----------------+  +-------------+ |  |
|  |  | Bay Occupancy    |  | Safety Alert   |  | ANPR        | |  |
|  |  | Engine           |  | Engine         |  | Engine      | |  |
|  |  | - Vehicle detect |  | - Loitering    |  | - Plate     | |  |
|  |  | - ROI matching   |  | - Intrusion    |  |   detect    | |  |
|  |  | - Status smooth  |  | - Fall detect  |  | - OCR read  | |  |
|  |  +--------+---------+  +-------+--------+  +------+------+ |  |
|  |           |                    |                   |        |  |
|  |           +----------- MQTT Publisher -------------+        |  |
|  |              (metadata only: status, text, alerts)          |  |
|  +------------------------------------------------------------+  |
+------------------------------------------------------------------+
                              |
                         MQTT / REST
                              |
                              v
+------------------------------------------------------------------+
|                LAYER 3: CENTRAL MANAGEMENT PLATFORM              |
|                                                                  |
|  +------------------+  +------------------+  +----------------+  |
|  | API Gateway      |  | MQTT Broker      |  | Database       |  |
|  | (FastAPI)        |  | (Mosquitto)      |  | (PostgreSQL +  |  |
|  |                  |  |                  |  |  TimescaleDB)  |  |
|  +--------+---------+  +--------+---------+  +-------+--------+  |
|           |                     |                    |           |
|           +---------------------+--------------------+           |
|                                 |                                |
|  +------------------+  +------------------+  +----------------+  |
|  | Occupancy Engine |  | ANPR Matching    |  | Alert Manager  |  |
|  | - Zone aggregator|  | - Entry/exit     |  | - Notification |  |
|  | - Guidance calc  |  | - Fee calculator |  | - Escalation   |  |
|  | - Trend analysis |  | - Reconciliation |  | - Evidence log |  |
|  +------------------+  +------------------+  +----------------+  |
|                                                                  |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                      INTERFACES                                  |
|                                                                  |
|  [Operator Dashboard]  [Mobile App]  [LED Signs]  [Handhelds]   |
|   Web-based, RBAC      iOS/Android   40 units      Enforcement  |
|   Live view + analytics  Wayfinding   Zone counts   Alerts +    |
|   Violation mgmt        Payment       Color-coded   Evidence    |
+------------------------------------------------------------------+
```

**Data Flow Principles:**

- Video streams stay within the edge layer — **no video is transmitted** to the backend or cloud
- Edge devices publish only structured metadata: `{bay_id, status, timestamp}`, `{plate_text, confidence, camera_id, timestamp}`, `{alert_type, location, snapshot_url}`
- Backend processes metadata for aggregation, matching, analytics, and user-facing interfaces
- All communication encrypted (TLS 1.3 for REST/WebSocket, TLS for MQTT)

---

## 9. Edge Hardware Specification

### Edge AI Processor

| Specification | Value |
|--------------|-------|
| **Platform** | NVIDIA Jetson Orin NX (industrial-grade) |
| **AI Performance** | 100 TOPS (INT8) |
| **Memory** | 16 GB LPDDR5 |
| **Storage** | 256 GB NVMe SSD |
| **Power consumption** | 10-25W (configurable performance modes) |
| **Camera streams** | 8-12 simultaneous streams @ 1080p |
| **AI runtime** | ONNX Runtime with TensorRT acceleration |
| **Operating temperature** | -25C to +80C (industrial range) |
| **Form factor** | Fanless, DIN-rail mountable, IP40 enclosure |
| **Connectivity** | 2x GbE, WiFi 6, USB 3.2 |

### Camera Specifications

| Type | Occupancy Camera | ANPR Camera |
|------|-----------------|-------------|
| **Resolution** | 2MP (1920x1080) | 5MP (2592x1944) |
| **Protection** | IP67 weatherproof | IP67 weatherproof |
| **Night vision** | IR LEDs, 30m range | IR LEDs, 50m range, dedicated illuminator |
| **Lens** | Varifocal 2.8-12mm | Motorized 5-50mm |
| **Protocol** | RTSP / ONVIF | RTSP / ONVIF |
| **Power** | PoE (IEEE 802.3af) | PoE+ (IEEE 802.3at) |
| **Mounting** | Pole/wall bracket | Gantry/pole mount at 3-4m height |

### Network Infrastructure

| Component | Specification |
|-----------|--------------|
| **PoE switches** | 16-port managed PoE+ switches (15 units, co-located with edge boxes) |
| **Backbone** | Fiber optic ring connecting all parking zones to central server room |
| **Redundancy** | Dual-path fiber with automatic failover |
| **Bandwidth** | Internal: PoE LAN (camera to edge). Backbone: metadata only (~50 Kbps per edge device) |
| **UPS** | Per-zone UPS providing 30-minute battery backup for edge devices and switches |

---

## 10. Deployment Plan & Bill of Materials

### Full Bill of Materials

| Category | Item | Qty | Unit Cost (USD) | Total (USD) |
|----------|------|-----|-----------------|-------------|
| **Hardware** | Occupancy cameras (2MP IP67 PoE) | 140 | $300 | $42,000 |
| | ANPR cameras (5MP IR PoE+) | 30 | $1,000 | $30,000 |
| | Edge AI boxes (Jetson Orin NX + enclosure) | 15 | $2,500 | $37,500 |
| | PoE managed switches (16-port) | 15 | $400 | $6,000 |
| | LED guidance signs (outdoor rated) | 40 | $1,000 | $40,000 |
| | Network infrastructure (fiber, conduit, patch panels) | 1 lot | - | $30,000 |
| | Backend server (rack-mount, redundant PSU) | 1 | $5,000 | $5,000 |
| | **Hardware Subtotal** | | | **$190,500** |
| **Software** | AI model development & optimization | - | - | $40,000 |
| | ANPR pipeline development | - | - | $30,000 |
| | Central management platform | - | - | $60,000 |
| | Mobile application (iOS + Android) | - | - | $30,000 |
| | System integration & testing | - | - | $20,000 |
| | **Software Subtotal** | | | **$180,000** |
| **Services** | Site survey, installation & commissioning | - | - | $40,000 |
| | Operator training & documentation | - | - | $10,000 |
| | 1-year warranty support & maintenance | - | - | $30,000 |
| | **Services Subtotal** | | | **$80,000** |
| **TOTAL** | | | | **$450,500** |

*All prices in USD. MYR equivalent at prevailing exchange rate (~MYR 2.1M at 1 USD = 4.65 MYR).*

### Zone Deployment Strategy

| Phase | Zones | Bays | Cameras | Edge Boxes | Duration |
|-------|-------|------|---------|-----------|----------|
| **Pilot** | 2-3 high-traffic zones (town center, commercial district) | 500-1,000 | 20-25 | 2-3 | Months 5-6 |
| **Rollout Wave 1** | Remaining commercial + government zones | 3,000 | 60-70 | 6 | Month 7 |
| **Rollout Wave 2** | Residential, recreational, and peripheral zones | 3,000 | 80-85 | 6 | Month 8 |
| **Total** | All zones | 7,000 | 170 | 15 | - |

---

## 11. Acceptance Criteria

All acceptance tests will be conducted jointly by BDA and the project team during the stabilization phase (Months 9-10).

| Metric | Target | Test Method | Duration |
|--------|--------|-------------|----------|
| Bay occupancy accuracy (overall) | **>=95%** | Ground truth comparison: physical audit of 500+ bays vs. system status, sampled across all zones and times of day | 7 consecutive days |
| Bay occupancy accuracy (night) | **>=90%** | Same method, restricted to 7 PM - 7 AM period | 7 consecutive nights |
| ANPR recognition (day) | **>=95%** | Manual verification of 1,000+ entry/exit plate reads against ground truth (human review of captured plate images) | 14 days |
| ANPR recognition (night) | **>=90%** | Same method, restricted to 7 PM - 7 AM period | 14 nights |
| Occupancy update latency | **<=5 seconds** | Timestamp comparison: physical vehicle arrival/departure time vs. system status change time, 100+ events | 7 days |
| Guidance sign accuracy | **Within +-3 spaces per zone** | Physical count of available bays vs. LED sign display, 50+ spot checks across zones | 7 days |
| Safety alert detection rate | **>=85%** | Staged scenario testing: 50+ scenarios (loitering, intrusion, fall) executed by test personnel | 3 days |
| Double parking violation detection | **>=80%** | Manual review of flagged vs. actual double parking events over test period | 14 days |
| System uptime | **>=99.5%** | Continuous monitoring of all edge devices, cameras, platform services, and interfaces | 30 consecutive days |
| Revenue reconciliation accuracy | **>=98%** | Comparison of system-calculated fees vs. manual audit of 500+ parking sessions | 30 days |

### Acceptance Process

1. **Test Plan Approval** — joint agreement on test scenarios, sampling methodology, and pass/fail criteria (Month 9, Week 1)
2. **Test Execution** — 30-day monitored operation with daily performance reports (Month 9, Weeks 2-4 + Month 10, Week 1)
3. **Results Review** — joint analysis of all metrics, identification of any remediation items (Month 10, Week 2)
4. **Remediation** — resolution of any items not meeting target (if applicable, Month 10, Week 3)
5. **Final Acceptance Certificate** — signed by both parties upon all metrics meeting targets (Month 10, Week 4)

---

## 12. Why VIETSOL + ESP

| Factor | Our Solution | Typical Competitor |
|--------|-------------|-------------------|
| **Licensing** | Zero licensing fees — all AI models are Apache 2.0 / MIT licensed, no per-camera or annual software fees | Per-camera annual license fees ($50-200/camera/year) |
| **Hardware** | Standard industrial edge processors and off-the-shelf IP cameras | Proprietary hardware requiring vendor-specific replacements |
| **CCTV Reuse** | Works with any RTSP/ONVIF compatible camera — existing infrastructure can be integrated | Requires proprietary cameras from specific manufacturers |
| **Cost (7,000 bays)** | **~$450K (~$64/bay)** | $2-5M for sensor-based alternatives ($400-750/bay) |
| **Privacy** | Edge processing — no video leaves the device, only metadata transmitted | Cloud-based video processing raising privacy and bandwidth concerns |
| **Local Presence** | **ESP (Bintulu-based)** for on-ground installation, maintenance, and support + **VIETSOL (Hanoi)** for AI development and platform engineering | Foreign vendor with remote support, slow response times |
| **Customization** | Config-driven architecture — zone policies, detection thresholds, alert rules, and fee structures are fully configurable without code changes | Black-box systems with limited customization options |
| **Model Flexibility** | Multiple proven AI architectures optimized for different scenarios (occupancy, ANPR, safety) — can adapt and improve over time | Single vendor-locked model with no ability to upgrade or adapt |
| **Scalability** | Add cameras and edge devices incrementally — system scales linearly | Sensor-based systems require one sensor per bay for any expansion |
| **Future-Proof** | Same camera infrastructure supports future smart city applications (traffic, safety, crowd analytics) | Single-purpose sensors with no reuse potential |

### About VIETSOL

VIETSOL is a Hanoi-based AI and software engineering company specializing in edge AI solutions for smart city and industrial applications. Our team has deep expertise in computer vision, model optimization for edge deployment, and scalable platform architecture. We develop and maintain a proven library of AI detection models purpose-built for real-world deployment on industrial edge processors.

### About ESP — Elektro Serve Power Sdn Bhd

ESP is a Bintulu-based electrical and systems integration company with extensive experience in infrastructure projects across Sarawak. As the local implementation partner, ESP provides site survey, physical installation, network infrastructure, ongoing maintenance, and first-line support — ensuring rapid response times and local accountability.

---
---

## Next Steps

| Step | Action | Timeline |
|------|--------|----------|
| 1 | Review and feedback on this proposal | 2 weeks |
| 2 | Technical clarification session (in-person or virtual) | By arrangement |
| 3 | Site visit and detailed survey of priority zones | 1 week (upon approval) |
| 4 | Contract finalization and project kick-off | Target: Q2 2026 |

---

**For inquiries, please contact:**

| | VIETSOL | ESP |
|---|---------|-----|
| **Contact** | *[Name, Title]* | *[Name, Title]* |
| **Email** | *[email]* | *[email]* |
| **Phone** | *[phone]* | *[phone]* |

---

*Strictly Confidential | VIETSOL + ESP | March 2026*

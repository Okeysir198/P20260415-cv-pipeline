# AI-Powered Adaptive Traffic Signal Control System

## Proposal for Bintulu Development Authority (BDA)

**Bintulu Smart City Program — Phase 2: Intelligent Traffic Management**

---

*[VIETSOL Logo]* &nbsp;&nbsp;&nbsp;&nbsp; *[ESP Logo]*

**Prepared by:** VIETSOL & Elektro Serve Power Sdn Bhd (ESP)

**Date:** March 2026

**Classification:** STRICTLY CONFIDENTIAL

---

# PART A: EXECUTIVE SUMMARY

---

## 1. Overview

VIETSOL, in partnership with Elektro Serve Power Sdn Bhd (ESP), proposes an AI-powered adaptive traffic signal control system for the Bintulu Smart City program. The system uses edge-processed camera feeds at each intersection to detect and classify vehicles in real time, estimate queue lengths per approach, and dynamically optimize signal timing to reduce congestion and improve traffic flow. Deployed at approximately $20,000 per intersection — 55% below the market median — the solution leverages proven AI detection models running on industrial edge processors, with all video processed locally for privacy and low latency. This system forms Phase 2 of the Bintulu Smart City program, building on the shared platform validated during Smart Parking (Phase 1), reducing incremental cost and deployment risk.

---

## 2. The Opportunity

Bintulu is experiencing sustained growth driven by SCORE (Sarawak Corridor of Renewable Energy) industrial development, increasing population, and rising vehicle ownership. Fixed-time traffic signals — designed for static traffic patterns — cannot adapt to the dynamic, time-varying demand that comes with a growing city. The result is unnecessary congestion, wasted fuel, increased emissions, and frustrated commuters.

The global Adaptive Traffic Control System (ATCS) market reflects the urgency of this problem:

| Metric | Value |
|--------|-------|
| Global ATCS market (2025) | $7.07 billion |
| Projected market (2030) | $14.37 billion |
| Growth rate (CAGR) | 8.75% – 17.4% |
| Fastest growing region | Asia-Pacific |

Cities across Asia-Pacific are rapidly adopting AI-driven traffic management to address congestion before it becomes entrenched. Bintulu has the advantage of acting now — deploying modern adaptive control at a fraction of the cost that larger cities face when retrofitting legacy infrastructure.

---

## 3. Solution at a Glance

Each intersection is equipped with a self-contained AI traffic management unit:

```
4 IP Cameras (one per approach)
        |
        v
   Edge AI Box
   - Vehicle detection & classification
   - Queue length estimation
   - Phase optimization
        |
        v
 Traffic Signal Controller
   - Receives timing recommendations
   - Retains fixed-time backup (failsafe)
```

**How it works:**

1. **Four IP cameras** (one per intersection approach) capture 1080p video of approaching traffic.
2. **An edge AI processor** at the intersection runs proven AI detection models to identify, classify, and track every vehicle in real time — no cloud connection required.
3. **Per-lane analytics** compute queue length, vehicle count, and flow rate for each approach.
4. **A phase optimizer** calculates optimal green times based on real-time demand, applying safety constraints (minimum green, all-red clearance).
5. **Timing recommendations** are sent to the existing traffic signal controller via standard protocols (NTCIP 1202 / SNMP).
6. **Failsafe guarantee:** The signal controller retains its fixed-time backup plan and automatically reverts if the AI system becomes unresponsive — traffic never stops flowing.

Only metadata (vehicle counts, queue lengths, timing commands) leaves the intersection. No video is transmitted or stored centrally, ensuring full privacy compliance.

---

## 4. Proven Results Worldwide

AI-powered adaptive traffic signal control has been deployed and validated in cities across multiple continents:

| Deployment | System Type | Documented Results |
|------------|-------------|-------------------|
| Pittsburgh, PA, USA | AI adaptive signal control | 25% travel time reduction, 40% idle time reduction |
| Tucson, AZ, USA | AI traffic management platform | 23% average delay reduction, 1.25M+ driver hours saved |
| London, United Kingdom | Adaptive signal control network | 30% travel time reduction, 50% congestion decrease |
| Las Vegas, NV, USA | AI traffic analytics | 17% reduction in primary crashes |
| Florida, USA (statewide) | AI traffic management | State DOT-approved statewide deployment |

These results demonstrate that AI-driven adaptive signal control consistently delivers 15–40% improvements in travel time, delay, and congestion metrics across diverse traffic environments — from dense urban corridors to suburban intersections.

---

## 5. Cost Advantage

Our solution delivers adaptive traffic signal control at approximately **55% below the market median cost**:

| Component | Cost (USD) | Description |
|-----------|-----------|-------------|
| Hardware | $6,000 | Edge AI processor, enclosure, cabling, installation materials |
| Software | $10,000 | AI detection models, phase optimizer, controller interface, central platform |
| Services | $4,000 | Site survey, installation, configuration, commissioning, training |
| **Total per intersection** | **$20,000** | |

| | Our Solution | Market Median |
|---|---|---|
| **Cost per intersection** | **~$20,000** | **~$45,000** |
| Annual licensing fees | $0 | $3,000 – $10,000/year |
| Proprietary hardware | No (standard industrial) | Often required |

**No recurring licensing fees.** The AI models and software are fully owned — there are no per-intersection, per-year software charges that inflate the total cost of ownership over time.

---

## 6. Timeline

Phase 2 begins after Smart Parking (Phase 1) validates the shared platform, ensuring proven infrastructure before adding signal control complexity.

| Period | Phase | Activities |
|--------|-------|------------|
| **Months 8–10** | Foundation | Vehicle detection & classification model training and validation. Lane ROI configuration tool. Queue length estimation algorithm. Traffic analytics dashboard (counts, flow rates, historical trends). Integration with shared platform backend. |
| **Months 11–12** | Signal Control | Phase optimizer development and tuning. Signal controller interface (NTCIP 1202 / SNMP). Pilot deployment at 2–3 intersections. Before/after performance measurement. Safety validation and failsafe testing. |
| **Months 13–14** | Full Rollout | Deployment to remaining intersections. Acceptance testing against all metrics. Operator training and documentation. 30-day monitored operation period. Final handover and sign-off. |

---

# PART B: TECHNICAL APPENDIX

---

## 7. Detailed Features

### FR-T01: Vehicle Detection & Classification

| Specification | Target |
|---------------|--------|
| Vehicle classes | Car, motorcycle, truck, bus, bicycle, pedestrian |
| Cameras per intersection | 4 (one per approach) |
| Input resolution | 1920 x 1080 (1080p) |
| Detection frame rate | >= 15 FPS per camera |
| Detection accuracy (mAP@0.5) | >= 90% (normal conditions) |
| Detection accuracy (adverse) | >= 80% (rain, low light, glare) |

The system uses proven AI object detection models optimized for edge deployment. Each camera feed is processed independently by the edge AI processor, enabling simultaneous analysis of all four approaches. Vehicle classification enables differentiated traffic management — for example, giving priority to buses or adjusting clearance intervals for heavy vehicles.

### FR-T02: Adaptive Signal Timing

| Specification | Target |
|---------------|--------|
| Timing mode | Dynamic phase optimization based on real-time demand |
| Response time | <= 2 seconds from detection to phase adjustment |
| Timing plans | Multiple configurable plans (peak, off-peak, weekend, event) |
| Green time constraints | Configurable minimum and maximum green per phase |
| All-red clearance | Configurable per intersection geometry |
| Manual override | Operator can force specific phase via dashboard or on-site |
| Failsafe | Auto-revert to fixed-time plan if AI unresponsive |

The phase optimizer distributes cycle time proportionally to real-time queue demand across all approaches, while respecting safety constraints (minimum pedestrian crossing time, all-red clearance intervals, maximum green limits). Timing plans can be configured per time-of-day and day-of-week, with the AI optimization applied within each plan's parameter envelope.

### FR-T03: Multi-Lane Monitoring

| Specification | Target |
|---------------|--------|
| Lane definition | Configurable polygon ROI per lane |
| Vehicle counting | 1-minute, 5-minute, 15-minute, and 60-minute bins |
| Queue length | Stopped vehicle count per lane, accuracy >= 80% (within +/- 2 vehicles) |
| Speed estimation | Lane-level average speed measurement |
| Imbalance detection | Alert when adjacent lanes differ by > configurable threshold |

Operators configure lane boundaries using a visual ROI drawing tool on the dashboard. The system tracks vehicles across frames using advanced object tracking, enabling accurate counting even in dense traffic. Queue length is estimated by counting stationary or near-stationary vehicles within each lane ROI.

### FR-T04: Incident Detection

| Specification | Target |
|---------------|--------|
| Incident types | Stalled vehicles, wrong-way travel, pedestrians in vehicle lanes |
| Alert delivery | MQTT push notification within 5 seconds |
| Alert content | Incident type, location, timestamp, snapshot image |
| Detection accuracy | >= 80% of real incidents |

Abnormal events are detected by analyzing vehicle trajectories and behavior patterns. A vehicle stationary for an abnormal duration triggers a stalled vehicle alert. Vehicles moving against the expected flow direction trigger wrong-way alerts. Pedestrians detected within vehicle lane ROIs trigger safety alerts. All alerts include a snapshot image for operator verification.

### FR-T05: Emergency Vehicle Priority

| Specification | Target |
|---------------|--------|
| Emergency vehicle types | Ambulance, fire truck, police vehicle |
| Detection method | Visual classification from camera feed |
| Signal preemption | Force green on emergency vehicle approach |
| Recovery | Auto-return to normal timing after 1 signal cycle |
| Logging | Full event log (detection time, preemption start/end, vehicle type) |

When an emergency vehicle is detected approaching an intersection, the system sends a preemption command to the signal controller, forcing the emergency vehicle's approach to green while holding conflicting phases at red. After the emergency vehicle clears the intersection, normal adaptive timing resumes automatically after one complete cycle. All preemption events are logged for audit and performance review.

### FR-T06: Central Platform

| Specification | Target |
|---------------|--------|
| Dashboard | Web-based, real-time status of all intersections |
| Live data | Vehicle counts, queue lengths, signal phase status per intersection |
| Map view | Geographic overview with health indicators per intersection |
| Historical analytics | Traffic volume trends, peak hour analysis, congestion patterns |
| Alert management | Real-time alert feed, acknowledgment workflow, escalation rules |
| Timing configuration | Remote timing plan adjustment per intersection |
| Reporting | Automated daily/weekly/monthly traffic reports |
| Access control | Role-based access (administrator, operator, viewer) |

The central platform shares infrastructure with the Smart Parking system deployed in Phase 1 — the same backend, dashboard framework, and MQTT messaging bus. This reduces deployment cost and provides operators with a unified view of all smart city systems from a single interface.

### FR-T07: Signal Controller Integration

| Specification | Target |
|---------------|--------|
| Protocol | NTCIP 1202 / SNMP (standard), or vendor-specific API |
| Read capability | Current phase state, timing parameters, fault status |
| Write capability | Phase advance, phase hold, timing plan selection |
| Failsafe | Controller auto-reverts to fixed-time if AI communication lost |
| Graceful degradation | If one camera fails, system continues with remaining cameras |
| Heartbeat | AI system sends periodic heartbeat; controller monitors health |

The system interfaces with existing traffic signal controllers using industry-standard protocols. No proprietary controller hardware is required. The AI system operates in an advisory role — it sends timing recommendations to the controller, which executes them while maintaining its own safety interlocks. If communication is lost for a configurable timeout period, the controller seamlessly reverts to its built-in fixed-time plan.

---

## 8. System Architecture

### Per-Intersection Edge Architecture

```
 Camera 1 (North)    Camera 2 (East)    Camera 3 (South)    Camera 4 (West)
       |                   |                   |                   |
       +-------------------+-------------------+-------------------+
                                   |
                          [ Edge AI Processor ]
                          |                   |
                   +------+------+     +------+------+
                   |  AI Engine  |     |  Analytics  |
                   |             |     |             |
                   | - Detection |     | - Queue len |
                   | - Classify  |     | - Veh count |
                   | - Tracking  |     | - Flow rate |
                   +------+------+     | - Speed est |
                          |            +------+------+
                          |                   |
                   +------+-------------------+------+
                   |       Phase Optimizer           |
                   |                                 |
                   | - Calculate optimal green times |
                   | - Apply min/max constraints     |
                   | - Generate phase plan           |
                   | - Emergency preemption logic    |
                   +-----------------+---------------+
                                     |
                   +-----------------+---------------+
                   |    Controller Interface         |
                   |                                 |
                   | - NTCIP 1202 / SNMP             |
                   | - Send phase commands           |
                   | - Read controller state         |
                   | - Heartbeat monitoring          |
                   +-----------------+---------------+
                                     |
                          [ Signal Controller ]
                          (existing hardware)
                                     |
                   +-----------------+---------------+
                   |    Failsafe Monitor             |
                   |                                 |
                   | - If AI heartbeat lost:         |
                   |   auto-revert to fixed-time     |
                   | - If camera fails:              |
                   |   degrade gracefully            |
                   +---+-------------------------+---+
                       |                         |
                  [ MQTT ]                  [ MQTT ]
                       |                         |
              Metadata to Central        Alerts to Central
              Platform (counts,          Platform (incidents,
              queue, timing)             preemptions)
```

### Central Platform Architecture

```
 Intersection 1    Intersection 2    ...    Intersection N
       |                 |                        |
       +-----------------+------------------------+
                         |
                    [ MQTT Broker ]
                    (shared with Smart Parking)
                         |
              +----------+----------+
              |   Central Backend   |
              |                     |
              | - Data aggregation  |
              | - Historical store  |
              | - Alert routing     |
              | - Report generation |
              | - API gateway       |
              +----------+----------+
                         |
              +----------+----------+
              |    Web Dashboard    |
              |                     |
              | - Map view          |
              | - Real-time status  |
              | - Analytics         |
              | - Configuration     |
              | - Reports           |
              +---------------------+
```

---

## 9. Adaptive Timing Approach

### Phase 1 Deployment: Rule-Based Optimization

The initial deployment uses a proven rule-based approach that delivers immediate congestion reduction:

1. **Measure:** For each approach, count the number of queued (stationary) vehicles and the arrival rate using AI detection and tracking.
2. **Calculate demand ratio:** Compute the relative demand across all approaches (e.g., North has 40% of total queue, East has 30%, South has 20%, West has 10%).
3. **Distribute cycle time:** Allocate green time proportionally to demand within the current cycle length.
4. **Apply constraints:** Enforce minimum green (pedestrian safety), maximum green (fairness), and all-red clearance intervals.
5. **Execute:** Send the optimized phase plan to the signal controller for the next cycle.
6. **Repeat:** Recalculate every cycle (typically 60–120 seconds) to continuously adapt to changing conditions.

This approach is transparent, predictable, and easy to audit — critical for safety-critical traffic infrastructure.

### Future Enhancement: Reinforcement Learning Optimization

Once sufficient real-world intersection data has been collected (typically 3–6 months of operation), the system can be enhanced with reinforcement learning-based optimization:

- **Learning from real data:** The RL agent trains on actual traffic patterns specific to each Bintulu intersection, learning timing strategies that a rule-based system cannot discover.
- **Multi-intersection coordination:** RL can optimize across adjacent intersections to create "green waves" along corridors.
- **Projected improvement:** An additional 10–20% delay reduction beyond rule-based optimization, based on published research and field deployments.

The RL enhancement is a software update — no hardware changes required.

---

## 10. Edge Hardware

The traffic system uses the same edge hardware platform as Smart Parking (Phase 1), maximizing platform synergy and reducing costs:

| Specification | Value |
|---------------|-------|
| Processor type | Industrial edge AI processor |
| AI performance | 100 TOPS INT8 |
| Memory | 16 GB |
| Power consumption | 10–25W |
| Enclosure | IP65 weatherproof |
| Operating temperature | -10°C to 60°C |
| Connectivity | Gigabit Ethernet, Wi-Fi (optional), 4G/LTE (optional) |
| Storage | 256 GB SSD (local analytics buffer) |

**Per intersection deployment:**

| Item | Qty | Purpose |
|------|-----|---------|
| Edge AI processor | 1 | AI inference, analytics, phase optimization |
| IP camera (1080p) | 4 | One per intersection approach |
| Network switch (PoE) | 1 | Camera connectivity, power-over-ethernet |
| Network connection | 1 | Edge box to signal controller + central platform |
| Power supply | 1 | Edge box + switch (cameras powered via PoE) |
| Mounting hardware | 1 set | Pole-mount brackets for cameras and enclosure |

The same edge processor model used for parking enforcement is reused for traffic — spare parts, maintenance procedures, and operator familiarity are shared across both systems.

---

## 11. Acceptance Criteria

All metrics are measured during the acceptance testing period (Months 13–14) and during the 30-day monitored operation:

| Metric | Target | Test Method |
|--------|--------|-------------|
| Vehicle detection mAP@0.5 | >= 90% | BDA intersection test dataset (500+ annotated images) |
| Vehicle classification accuracy | >= 85% per class | Per-class Average Precision evaluation |
| Detection frame rate | >= 15 FPS per camera | Benchmark on deployed edge hardware |
| Queue length accuracy | >= 80% (within +/- 2 vehicles) | Manual count vs. system count (100 samples per intersection) |
| Signal phase response time | <= 2 seconds | End-to-end timing measurement (detection to controller command) |
| Congestion reduction | >= 15% delay reduction within 3 months | Before/after average travel time comparison (GPS probe data or manual measurement) |
| System uptime | >= 99.5% over 30 days | Automated monitoring and health check logs |
| Incident detection rate | >= 80% of real incidents | Combination of staged test scenarios and real incident evaluation |
| Emergency preemption | 100% of detected emergency vehicles | Staged emergency vehicle tests (minimum 10 per intersection) |
| Failsafe revert | <= 5 seconds to fixed-time | Simulated AI failure test |
| Adverse weather detection | >= 80% mAP@0.5 | Testing during rain and low-light conditions |

**Acceptance process:**

1. Factory acceptance test (FAT) — system validation in lab environment
2. Site acceptance test (SAT) — per-intersection commissioning and calibration
3. 30-day monitored operation — continuous uptime and performance monitoring
4. Final acceptance — review of all metrics against targets, sign-off by BDA

---

## 12. Why Phase 2 — The Phased Approach

The Bintulu Smart City program is designed as a progressive deployment, where each phase builds on the validated infrastructure of the previous phase:

| Aspect | Benefit of Phased Approach |
|--------|---------------------------|
| **Platform validation** | Smart Parking (Phase 1) proves the edge hardware, backend infrastructure, MQTT messaging, and web dashboard before traffic signal control adds complexity. |
| **Shared hardware** | Traffic uses the same edge processor platform as parking — shared spare parts, maintenance procedures, and vendor relationships reduce cost. |
| **Proven AI pipeline** | Vehicle detection and tracking capabilities developed and validated for parking transfer directly to traffic, reducing development time and risk. |
| **Incremental investment** | BDA invests progressively, with each phase delivering standalone value, rather than a high-risk big-bang deployment. |
| **Operator readiness** | Operators trained on the parking system dashboard are already familiar with the platform when traffic management is added. |
| **Lower risk** | If any platform issues arise during Phase 1, they are resolved before Phase 2 depends on them. |

```
Phase 1: Smart Parking              Phase 2: Traffic Signal Control
(Months 1-7)                         (Months 8-14)

 - Edge hardware validated           - Same edge hardware platform
 - Backend + MQTT proven              - Same backend + MQTT
 - Dashboard operational              - Extended dashboard
 - Detection models trained           - Models adapted for traffic
 - Operator training complete         - Incremental operator training

        Platform Proven ──────────────> Platform Reused
```

---

## 13. Why VIETSOL + ESP

| Factor | Our Solution | Typical Competitor |
|--------|-------------|-------------------|
| **Licensing** | Zero licensing fees — perpetual ownership | $3,000 – $10,000 per intersection per year |
| **Hardware** | Standard industrial edge processors — no vendor lock-in | Proprietary hardware, single-source dependency |
| **Cost per intersection** | ~$20,000 (55% below market) | ~$45,000 market median |
| **Total cost (10 intersections, 5 years)** | ~$200,000 | ~$450,000 + $150K–$500K licensing |
| **Failsafe design** | Controller retains full fixed-time backup, auto-reverts | Vendor-dependent failsafe implementation |
| **Local presence** | ESP based in Bintulu — same-day on-site support | Foreign vendor, remote support with travel delays |
| **Platform synergy** | Shared infrastructure with Smart Parking — unified dashboard, shared maintenance | Separate vendor per system, no integration |
| **Privacy** | All video processed on-site at edge — only metadata transmitted | Often cloud-based video processing |
| **Customization** | Full source code ownership — BDA can extend and modify | Black-box software, vendor controls roadmap |
| **Scalability** | Add intersections incrementally at $20K each | Volume discounts often require large upfront commitments |

### Local Partnership Advantage

**ESP (Elektro Serve Power Sdn Bhd)** provides:
- Physical presence in Bintulu for installation, maintenance, and emergency support
- Existing relationships with local infrastructure authorities
- Electrical and networking installation capabilities
- First-line support and maintenance

**VIETSOL** provides:
- AI model development, training, and optimization
- Edge software platform and central dashboard
- System integration and signal controller interfacing
- Continuous model improvement and feature development

This partnership ensures that BDA has both the technical depth of a specialized AI company and the local presence of a Bintulu-based engineering firm — eliminating the support gaps that plague deployments by foreign vendors without local representation.

---

## Summary

The AI-Powered Adaptive Traffic Signal Control System delivers:

- **Immediate impact:** 15–25% congestion reduction through real-time adaptive signal timing
- **Cost efficiency:** $20,000 per intersection, zero licensing fees, 55% below market
- **Safety first:** Failsafe design ensures traffic never stops flowing, even if AI fails
- **Privacy by design:** Edge processing, metadata only — no video leaves the intersection
- **Platform synergy:** Shared infrastructure with Smart Parking reduces cost and risk
- **Local support:** ESP in Bintulu provides same-day on-site service
- **Future-ready:** Reinforcement learning enhancement available as a software update

We look forward to demonstrating how this system will transform Bintulu's traffic infrastructure into an intelligent, adaptive network that grows with the city.

---

*Strictly Confidential | VIETSOL + ESP | March 2026*

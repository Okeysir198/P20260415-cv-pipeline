# Edge AI Performance Specification — Nitto Denko Proposal

## 1. Market Benchmark Summary

| Use Case | Market Best (Research) | Commercial Claims | Vietsol Target |
|---|---|---|---|
| Fire/Smoke | 94.8% precision, 93% recall (DetectNet_v2) | Up to 98.5% (Scylla AI) | 92% precision, 90% recall |
| Helmet/PPE | 92.4% AP@50 (Swin+YOLOv10) | 96–98% (FuweeVision, Incoresoft) | 94% precision, 92% recall |
| Safety Shoes | 83.5% AP@50 (hardest PPE class, Nature 2025) | Few vendors — niche market | 88% precision, 85% recall |
| Fall Detection | Up to 98% (lab setting, JMIR/IEEE) | Visionify: 48% fewer near-miss incidents | 90% precision, 88% recall |
| Zone Intrusion | >95% across complex environments | >90% FP reduction vs traditional systems | 94% precision, 92% recall |
| Phone Usage | 96% (controlled, YOLOv8+ResNet-50); 49.5% mAP (FPI-Det wild) | Very few competitors offer this | 85% precision, 82% recall |
| Fire Response Time | 2–5 sec (AI video) vs 90+ sec (ceiling-mount sensors) | "90 sec faster than traditional sensors" (AxxonSoft) | < 3 sec |

Sources: Frontiers in Computer Science 2025, Nature Scientific Reports 2025, arXiv (SH17, FPI-Det), Scylla AI, Visionify, AxxonSoft, Viso.ai, IEEE/JMIR.

---

## 2. Edge Hardware Competitive Landscape

| Chip | INT8 TOPS | Power | Price Range | Key Note |
|---|---|---|---|---|
| NVIDIA Jetson AGX Orin | 275 | 15–60W | $1,000–2,000 | Best ecosystem, highest cost |
| NVIDIA Jetson Orin Nano Super | 67 | 7–25W | ~$250 | Popular dev platform |
| Qualcomm QCS8550 | 48 | Low (4nm) | Module pricing | Wi-Fi 7, excellent ISP |
| Hailo-8 | 26 | 2.5W | $100–250 | PCIe M.2, ultra low power |
| **AX650N (Vietsol)** | **18** | **5–8W** | **Budget-friendly** | **8K ISP, 32ch decode, YOLOX-S@130FPS** |
| Hailo-8L | 13 | <2W | Budget | Entry-level |
| CV186AH (backup) | 7.2 | 5–15W | Budget | 16ch decode, LLM support |
| Google Coral (Edge TPU) | 4 | 2W | $25–60 | TFLite only |

AX650N advantage: YOLOX-S at 130 FPS (7.66ms), MobileNetV2 at 1,798 FPS — competitive with chips 3x the price.

---

## 3. Vietsol Proposed Performance Specification

### 3.1 Latency

| Metric | Specification | Context |
|---|---|---|
| Single-model inference | **< 40ms** (25+ FPS) | YOLOX-M on AX650N INT8 |
| Multi-model per camera | **< 67ms** (15+ FPS) | Fire + Helmet + Intrusion simultaneous |
| Pose estimation | **< 5ms** (40+ FPS) | RTMPose-S proven 4.79ms on AX650N |
| Alert notification | **< 3 seconds** | Detection → MQTT → Microsoft Teams |
| Traditional sensor comparison | **60–90x faster** | AI video vs ceiling-mount fire sensors |

### 3.2 Accuracy (mAP@0.5)

| Use Case | Model | mAP@0.5 | Precision | Recall |
|---|---|---|---|---|
| Fire/Smoke | YOLOX-M | **≥ 90%** | **≥ 92%** | **≥ 90%** |
| Helmet (incl. Nitto hat) | YOLOX-M | **≥ 92%** | **≥ 94%** | **≥ 92%** |
| Safety Shoes | YOLOX-Tiny + MobileNetV3 | **≥ 85%** | **≥ 88%** | **≥ 85%** |
| Fall Detection (classify) | YOLOX-M | **≥ 88%** | **≥ 90%** | **≥ 88%** |
| Fall Detection (pose) | YOLOX-Tiny + RTMPose-S | **≥ 88%** | **≥ 90%** | **≥ 88%** |
| Phone Usage | YOLOX-Tiny + RTMPose-S | **≥ 82%** | **≥ 85%** | **≥ 82%** |
| Zone Intrusion | YOLOX-Tiny (COCO pretrained) | **≥ 95%** | **≥ 95%** | **≥ 94%** |

### 3.3 False Positive Rate

| Use Case | FP Rate | Rationale |
|---|---|---|
| Fire/Smoke | **< 3%** | Factory environment — minimize alert fatigue |
| Helmet | **< 2%** | Large dataset (62K images) enables high precision |
| Safety Shoes | **< 4%** | Small dataset (3.7K) — conservative target |
| Fall Detection | **< 3%** | Dual-path (classify + pose) cross-validates |
| Zone Intrusion | **< 2%** | Person detection is mature |
| Phone Usage | **< 5%** | Small object — hardest task |

### 3.4 False Negative Rate

| Use Case | FN Rate | Mitigation Strategy |
|---|---|---|
| Fire/Smoke | **< 5%** | Multi-frame temporal smoothing; redundant with physical sensors |
| Helmet | **< 3%** | 62K dataset, high recall achievable |
| Safety Shoes | **< 5%** | 2-stage detection (person → foot crop → classify) |
| Fall Detection | **< 3%** | Life-safety critical — dual classify + pose path reduces misses |
| Zone Intrusion | **< 2%** | Person detection near-zero miss; multi-camera overlap |
| Phone Usage | **< 8%** | Hardest task — small object, frequent occlusion |

Mitigation design: Temporal smoothing (N-frame confirmation), multi-camera coverage zones, human-in-the-loop escalation for ambiguous detections.

### 3.5 System Specifications

| Metric | Specification |
|---|---|
| Processing | 100% on-device — no cloud dependency |
| Power consumption | < 12W per edge box (SoC) |
| Cameras per box | Up to 4 simultaneous streams |
| Video decode | Up to 32ch 1080p@30fps |
| Model format | INT8 quantized ONNX → AX650N native |
| Notification | MQTT → Microsoft Teams (< 3 sec) |
| Storage | Local event clips + metadata; optional NAS export |
| Licensing | Apache 2.0 (YOLOX, D-FINE) — $0 commercial license fee |

---

## 4. Multi-Model Zone Performance (AX650N at 18 TOPS)

| Zone Type | Models Running | Est. FPS Each | NPU Utilization |
|---|---|---|---|
| Factory floor (general) | Fire + Helmet + Intrusion | ~15 FPS | ~70% |
| Factory floor (full PPE) | Fire + Helmet + Shoes (2-stage) | ~12 FPS | ~85% |
| Stairs / corridors | Phone + Pose + Intrusion | ~15 FPS | ~80% |
| Restricted areas | Intrusion only | ~50 FPS | ~20% |
| Warehouse / storage | Fall (pose) + Fire + Intrusion | ~15 FPS | ~75% |

---

## 5. Competitive Differentiators

| Factor | Vietsol Edge AI | Typical Competitor |
|---|---|---|
| **Licensing** | Apache 2.0 (YOLOX/D-FINE) — $0 royalty | AGPL-3.0 (Ultralytics YOLO) — licensing risk |
| **Power** | < 12W (AX650N) | 15–60W (Jetson Orin family) |
| **Alert speed** | < 3 sec (vs 90+ sec traditional sensors) | 2–5 sec (comparable) |
| **On-device** | 100% edge, no cloud required | Many require cloud (Verkada, Landing AI) |
| **Cost** | Budget-friendly AX650N | Jetson Orin $250–2,000 per unit |
| **Phone detection** | Included (pose + detection) | Rarely offered by competitors |
| **Nitto hat support** | Custom-trained class | Not available from generic vendors |
| **Multi-model** | 3+ models simultaneous per camera | Most run 1 model per stream |

---

## 6. Industry Standards Alignment

| Standard | Scope | Status |
|---|---|---|
| EN 54-10 | Fire flame detector requirements | AI video detection emerging; no vendor certified yet |
| IEC 62676-4:2025 | Video surveillance visual performance | Updated from DORI to 7-level framework |
| IEC 62676-6 (draft) | Video content analytics performance grading | Key future standard — not yet finalized |
| ISO 13849 | Machine safety control systems (PL a–e) | Applies if AI is part of safety function |
| IEC 62443 | Industrial cybersecurity for network devices | Applicable to edge box network security |

Note: No competitor has EN 54 or IEC 62676-6 certification for AI video analytics yet — early compliance is a market differentiator.

---

## 7. Performance Summary for PM (Sophgo CV186AH — Worst Case)

| Use Case | Worst-Case Latency | Alert Latency | mAP@0.5 | Precision | Recall | FP Rate | FN Rate |
|---|---|---|---|---|---|---|---|
| Fire/Smoke | ~167ms (6 FPS) | < 5 sec | ≥ 90% | ≥ 92% | ≥ 90% | < 3% | < 5% |
| Helmet (incl. Nitto hat) | ~167ms (6 FPS) | < 5 sec | ≥ 92% | ≥ 94% | ≥ 92% | < 2% | < 3% |
| Safety Shoes (2-stage) | ~200ms (5 FPS) | < 5 sec | ≥ 85% | ≥ 88% | ≥ 85% | < 4% | < 5% |
| Fall Detection | ~167ms (6 FPS) | < 5 sec | ≥ 88% | ≥ 90% | ≥ 88% | < 3% | < 3% |
| Zone Intrusion | ~167ms (6 FPS) | < 5 sec | ≥ 95% | ≥ 95% | ≥ 94% | < 2% | < 2% |
| Phone Usage | ~167ms (6 FPS) | < 5 sec | ≥ 82% | ≥ 85% | ≥ 82% | < 5% | < 8% |

Notes: Worst-case = 3 models simultaneous per camera on CV186AH (7.2 INT8 TOPS). All accuracy/error rates are **per frame**.

---

## 8. Headline Specification (Plain Text — Copy-Paste Ready)

```
Vietsol Edge AI Performance (Sophgo CV186AH)

- Single model latency: < 80ms per frame (12–15 FPS)
- Multi-model latency (3 models simultaneous): < 200ms per frame (5–6 FPS)
- 6–15 FPS real-time processing per camera (depending on model count)
- < 5 second end-to-end alert latency (detection → MQTT → Teams notification)
- 85–95% detection accuracy per frame across all use cases
- < 2–5% false positive rate per frame — minimizes alert fatigue
- < 2–8% false negative rate per frame — mitigated by:
  - Temporal smoothing: consecutive-frame voting (e.g., 3/5 frames confirm) reduces effective FN rate to < 1–2% over a 1-second window
  - Multi-camera overlap: critical zones covered by 2+ cameras — combined miss probability drops exponentially (e.g., 5% × 5% = 0.25%)
- 100% on-device — no cloud, no data leaves the factory
- < 15W power per edge box
```

# Product Roadmap

```
  TIMELINE        PPE              Safety           Access          Traffic         Infra
  ════════════════════════════════════════════════════════════════════════════════════════════

  Phase 1 ── 12 weeks ─────────────────────────────────────────────────────────────────────

  W1–2         ┌─────────────────── Setup, Data Transform, Labeling Start ──────────────────┐
  mid-Mar      │  nitto_hat collection    SAM3 pre-annotation    Label Studio setup         │
               └────────────────────────────────────────────────────────────────────────────┘

  W3–4         [b] Helmet ──┐     [a] Fire ──────┐  [i] Zone ────┐                 SCRFD-500M
  late Mar     [f] Shoes ───┤     [g] Fall ──────┤   Intrusion ──┤                 MobileFaceNet
               (curated)    │      ├─ classify   │  (pretrained) │                 ByteTrack
                            │      └─ pose       │               │                 ONNX Runtime
                            │     [h] Poketenashi┤               │
                            │                    │               │
  W5–6                      │                    │               │
  mid-Apr      ─── expanded datasets ── midpoint review ─────────┘
               Evaluate multi-model inference on edge hardware

  W7–8
  late Apr     ─── full pipeline + customer factory data integration ───────────

  W9
  mid-May      ─── ONNX export ── INT8 quantize ── acceptance testing ── docs ──

  W10–12
  late May     ─── contingency / escalation buffer ────────────────────────────

  Phase 2 ── TBD ──────────────────────────────────────────────────────────────────────────

               [c] Glasses       [l] Forklift
               [d] Masks          Proximity
               [e] Gloves        [m] Near-Miss
               [j] Aprons         Behavior
               [k] Harness

  ═════════════════════════════════════════════════════════════════════════════════════════

  APPLICATIONS
  ─────────────────────────────────────────────────────────────────────────────────────────
  ┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
  │  Nitto Denko     │       │    Bintulu       │       │   Future...      │
  │  Factory         │       │  Smart City      │       │                  │
  ├──────────────────┤       ├──────────────────┤       ├──────────────────┤
  │ PPE: b,f,c*,     │       │ Traffic:         │       │ TBD              │
  │   d*,e*,j*,k*    │       │  parking,        │       │                  │
  │ Safety: a,g,h,   │       │  signal          │       │                  │
  │   l*,m*          │       │ Access: zone     │       │                  │
  │ Access: zone,    │       │                  │       │                  │
  │   face           │       │                  │       │                  │
  └──────────────────┘       └──────────────────┘       └──────────────────┘

  * = Phase 2
```

```
  PRODUCT FEATURES
  ═════════════════════════════════════════════════════════════════════════════════════════════════
                          Real-time   Object                  Identity   Compliance   Edge
  Feature                  Alert     Tracking   Pose/Rules    Match      Report      Ready
  ─────────────────────── ───────── ───────── ──────────── ────────── ─────────── ──────────

  PPE Cluster
  [b] Helmet Detection       ○         ○                        ○          ○          ○
  [f] Safety Shoes           ○         ○                        ○          ○          ○
  [c] Glasses/Goggles*       ○         ○                        ○          ○          ○
  [d] Masks*                 ○         ○                        ○          ○          ○
  [e] Gloves*                ○         ○                        ○          ○          ○
  [j] Aprons*                ○         ○                        ○          ○          ○
  [k] Safety Harness*        ○         ○          ○             ○          ○          ○

  Safety Cluster
  [a] Fire Detection         ○                                             ○          ○
  [g] Fall (classify)        ○         ○                        ○          ○          ○
  [g] Fall (pose)            ○         ○          ○             ○          ○          ○
  [h] Poketenashi            ○         ○          ○             ○          ○          ○
  [l] Forklift Proximity*    ○         ○          ○                        ○          ○
  [m] Near-Miss Behavior*    ○         ○          ○                        ○          ○

  Access Cluster
  [i] Zone Intrusion         ○         ○                                   ○          ○
  [—] Face Recognition                                          ○                     ○

  Traffic Cluster
  [—] Smart Parking          ○         ○                                   ○          ○
  [—] Signal Control         ○         ○                                   ○          ○

  ─────────────────────── ───────── ───────── ──────────── ────────── ─────────── ──────────
  ● = ready    ○ = in development    * = Phase 2
```

---

## Phase 1 — Factory Safety Core (12 weeks)

Core detection features for Nitto Denko factory safety system.

| ID | Feature | Cluster | Architecture | Difficulty | Platform Doc |
|----|---------|---------|-------------|------------|-------------|
| a | Fire Detection | Safety | YOLOX-M | LOW | [safety-fire_detection](03_platform/safety-fire_detection.md) |
| b | Helmet Detection | PPE | YOLOX-M | LOW-MED | [ppe-helmet_detection](03_platform/ppe-helmet_detection.md) |
| f | Safety Shoes | PPE | YOLOX-Tiny + MobileNetV3 | HIGH | [ppe-shoes_detection](03_platform/ppe-shoes_detection.md) |
| g | Fall Classification | Safety | YOLOX-M | HIGH | [safety-fall_classification](03_platform/safety-fall_classification.md) |
| g | Fall Pose Estimation | Safety | YOLOX-Tiny + RTMPose-S | HIGH | [safety-fall_pose_estimation](03_platform/safety-fall_pose_estimation.md) |
| h | Poketenashi Violations | Safety | YOLOX-Tiny + RTMPose-S | HIGH | [safety-poketenashi](03_platform/safety-poketenashi.md) |
| i | Zone Intrusion | Access | YOLOX-Tiny (pretrained) | LOW | [access-zone_intrusion](03_platform/access-zone_intrusion.md) |
| — | Face Recognition | Access | SCRFD-500M + MobileFaceNet | LOW | [access-face_recognition](03_platform/access-face_recognition.md) |

**Timeline:**

| Week | Month | Milestone |
|------|-------|-----------|
| 1–2 | mid-Mar | Environment setup, data transformation pipelines, labeling start (nitto_hat, safety shoes) |
| **3–4** | **late Mar** | **v1 training on curated subsets** ◀ NOW |
| 5–6 | mid-Apr | v2 training on expanded datasets, midpoint review |
| 7–8 | late Apr | v3 training + customer data integration |
| 9 | mid-May | ONNX export, acceptance testing, documentation |
| 10–12 | late May | Buffer / contingency |

**Targets:**

| Metric | Range |
|--------|-------|
| mAP@0.5 | 0.80 – 0.92 (per feature) |
| Precision | 0.85 – 0.94 |
| Recall | 0.82 – 0.92 |
| FP Rate | < 2–5% |
| Edge latency | < 40ms per frame |

**Details:** [phase1-executive-summary](02_project/phase1-executive-summary.md) | [phase1-development-plan](02_project/phase1-development-plan.md) | [phase1-requirements](01_requirements/phase1-requirements.md)

---

## Phase 2 — Extended PPE & Behavioral Safety

Additional PPE classes and advanced behavioral detection. Builds on Phase 1 infrastructure.

| ID | Feature | Cluster | Description | Difficulty |
|----|---------|---------|-------------|------------|
| c | Safety Glasses/Goggles | PPE | Glasses, goggles, face shields — small object, reflections | HIGH |
| d | Masks | PPE | Surgical, N95, respirator, face shield+mask — type classification | MED-HIGH |
| e | Gloves | PPE | Chemical-resistant, cut-resistant, insulated — hand region detect | HIGH |
| j | Aprons | PPE | Chemical aprons, split-type — body region detect | MED |
| k | Safety Belt/Harness | PPE | Harness usage + hook detection, smart harness integration | HIGH |
| l | Forklift/Pedestrian Proximity | Safety | Collision risk detection between forklifts and workers | HIGH |
| m | Near-Miss Behavior | Safety | Unsafe behavior patterns, near-miss incident detection | HIGH |

**Key challenges vs Phase 1:**
- Smaller objects (glasses, gloves, harness hooks) — may need higher input resolution (1280px)
- Fine-grained classification (mask type, glove type, apron type)
- Temporal reasoning for behavioral models (l, m)
- Limited public datasets for most Phase 2 classes — heavier custom annotation

**Prerequisites from Phase 1:**
- Training pipeline proven and config-driven
- Edge deployment pipeline validated (ONNX export, INT8 quantization)
- Annotation workflow established (Label Studio + SAM3 pre-annotation)
- Person detection models reusable as Stage 1 for two-stage pipelines

**Details:** [phase2-requirements](01_requirements/phase2-requirements.md)

---

## Applications

| Customer | Phase | Platform Features | Docs |
|----------|-------|-------------------|------|
| Nitto Denko | 1 + 2 | PPE + Safety + Access (all features) | [nitto-denko/](05_applications/nitto-denko/) |
| Bintulu | — | Traffic + Access | [bintulu/](05_applications/bintulu/) |

---

## Feature Clusters

```
PPE ─────────── helmet, shoes, glasses*, masks*, gloves*, aprons*
Safety ──────── fire, fall (classify + pose), poketenashi, forklift*, near-miss*
Access ──────── zone intrusion, face recognition
Traffic ─────── smart parking, signal control
```

\* = Phase 2

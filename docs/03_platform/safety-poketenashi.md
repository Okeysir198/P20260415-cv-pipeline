# Safety: Poketenashi Violations (Platform Overview)
> ID: h | Owner: TBD | Phase: 1 | Status: training
> Code split into 5 sibling feature folders — see "Feature Layout" below.

## Feature Layout (split 2026-04-29)

The single `features/safety-poketenashi/` umbrella has been replaced by
5 self-contained per-rule feature folders. This page remains the
**platform overview** for the rule family; per-rule details (configs,
training, eval) live in each feature's own `CLAUDE.md`.

| Feature folder | Rule | Mode | Config root |
|---|---|---|---|
| `features/safety-poketenashi_phone_usage` | phone usage while walking | 🎯 Fine-tune (FPI-Det) | `features/safety-poketenashi_phone_usage/configs/` |
| `features/safety-poketenashi_hands_in_pockets` | wrists inside torso band | 🔧 Pretrained pose + rule | `features/safety-poketenashi_hands_in_pockets/configs/` |
| `features/safety-poketenashi_no_handrail` | wrist outside handrail zone | 🔧 Pretrained pose + zone | `features/safety-poketenashi_no_handrail/configs/` |
| `features/safety-poketenashi_stair_diagonal` | trajectory angle vs stair | 🔧 Pretrained pose + tracking | `features/safety-poketenashi_stair_diagonal/configs/` |
| `features/safety-poketenashi_point_and_call` | 指差呼称 crosswalk gesture | 🔧 Pretrained pose + FSM | `features/safety-poketenashi_point_and_call/configs/` |

All five share the DWPose ONNX in `pretrained/safety-poketenashi/`
(directory name unchanged — it is the shared storage path, not a
feature folder).

## Customer Requirements

**Source:** Nitto Denko factory safety specification

**Explicit Requirements:**
- Deploy an automated system that detects Japanese factory safety violations known as "poketenashi" (literally "don't do that") across designated camera zones
- Detect four specific unsafe behaviors in Phase 1: phone usage while walking, hands in pockets while walking, failure to hold handrails on stairs, and diagonal/unsafe stair crossing
- **POKETENASHI** is a Japanese factory safety acronym covering safe walking behavior:
  - **Po** (pocket) -- Do not walk with hands in pockets
  - **Ke** (keitai) -- Do not use cell phone while walking
  - **Te** (tesuri) -- Hold the handrail on stairs
  - **Na** (naname) -- Do not jaywalk/cross diagonally
  - **Shi** (yubisashi) -- Point and call (shisa kanko)
- Phase 1 covers: **phone_usage**, **hands_in_pockets**, **no_handrail**, and **unsafe_stair_crossing**
- Purpose: Reduce slips, trips, and falls; improve worker awareness and attention; promote safe behavior in daily operations
- Typical use areas: Factories, warehouses, construction sites, offices with stair access
- System must operate on edge hardware (AX650N / CV186AH) with real-time alerts

## Business Problem Statement

- Japanese factory safety culture requires strict adherence to "poketenashi" behavioral norms -- a framework widely adopted in manufacturing to prevent workplace injuries through standardized safe walking and awareness practices
- Workers frequently violate these norms: walking with hands in pockets (tripping hazard), using mobile phones while walking (distraction hazard), failing to grip handrails on stairs (fall hazard), and crossing stairs diagonally instead of straight ahead (slipping hazard)
- Current enforcement relies entirely on manual observation by floor supervisors, which is inconsistent, labor-intensive, and cannot cover all camera zones simultaneously across shifts
- The customer (Nitto Denko) requires automated, continuous monitoring at designated factory zones to detect these four behavioral violations in real time and generate alerts
- Injuries from slips, trips, and falls are among the most common workplace incidents in Japanese manufacturing; preventing even a fraction of these through behavioral compliance delivers measurable reduction in lost-time incidents and safety consequences
- Cultural expectations are high: poketenashi is not optional -- it is a core safety discipline. The detection system must respect this by maintaining low false alarm rates so that alerts remain credible and workers do not become desensitized

## Technical Problem Statement

- **{Behavioral diversity} -> {Multi-behavior detection complexity}:** Four distinct unsafe behaviors must be detected simultaneously, each requiring different sensing modalities (object detection for phones, body pose analysis for hands-in-pockets and handrail usage, spatial zone and trajectory analysis for stair crossing), making this the most complex Phase 1 model by far
- **{Small-action recognition} -> {Hands-in-pockets detection ambiguity}:** Detecting hands inside pockets is inherently ambiguous from a 2D camera view because arms resting naturally at the sides produce nearly identical silhouettes; wrist keypoints are occluded by fabric, and no public dataset exists for this behavior, so detection must rely entirely on heuristic rules applied to pose keypoints
- **{Distraction detection} -> {Phone as small, occluded object}:** Mobile phones are small objects (often under 10% of the person bounding box area), frequently held at the ear or in front of the face where they are partially occluded by the hand, and vary widely in appearance across device models and protective cases
- **{Stair zone context} -> {Zone-aware reasoning required}:** Handrail and diagonal crossing violations are only meaningful within stair zones, requiring per-camera polygon configuration for stair areas, handrail positions, and stair direction vectors -- misconfiguration directly produces false positives or missed violations
- **{Temporal consistency} -> {Sustained behavior confirmation needed}:** Momentary poses (e.g., briefly lowering hands to adjust clothing) must not trigger false alarms; violations require sustained behavior over multiple frames (0.5--1.5 seconds depending on violation type), necessitating frame-level tracking and temporal accumulation logic
- **{Edge compute constraints} -> {Three concurrent models on limited NPU}:** Running a person detector, pose estimator, and phone detector simultaneously on edge chips (AX650N at 18 INT8 TOPS, CV186AH at 7.2 INT8 TOPS) requires careful model sizing and frame-skip scheduling to maintain usable FPS while covering all violation types

## Technical Solution Options

### Option 1: Multi-Model Pipeline -- YOLOX-Tiny + YOLOX-Nano + RTMPose-S + Rule Engine (Recommended)

- **Approach:** YOLOX-Tiny (COCO pretrained) detects persons and feeds crops to YOLOX-Nano (fine-tuned on FPI-Det) for phone detection and RTMPose-S (COCO-Pose pretrained) for 17-keypoint pose estimation. A CPU-based rule engine classifies all four violations using phone bounding boxes, wrist-hip keypoint relationships, stair zone polygons, and trajectory angles. ByteTrack provides per-person tracking for temporal confirmation.
- **Addresses:** Multi-behavior diversity (dedicated models per sensing modality), phone as small object (dedicated phone detector on upper-body crops), hands-in-pockets ambiguity (pose keypoint heuristics with low-confidence wrist detection), stair zone context (polygon-based zone configuration), temporal consistency (frame-count accumulation per tracked person), edge compute constraints (YOLOX-Tiny 5.1M + YOLOX-Nano 0.9M + RTMPose-S 5.47M = ~11.5M total params, ~15-20 FPS on AX650N)
- **Pros:** Each model is small and well-suited to its sub-task; only phone detection requires training (8-10 hours); person detector shared with Zone Intrusion (Model I) reduces compute; rule engine is fully configurable via YAML without retraining; all components Apache 2.0
- **Cons:** Three sequential inference passes add latency; rule-based pose heuristics for hands-in-pockets have inherent ambiguity ceiling (~70-80% expected accuracy); no public dataset for hands-in-pockets or handrail usage; per-camera zone configuration required

### Option 2: All-Transformer Pipeline -- D-FINE-N + RTMPose-T + X3D-XS

- **Approach:** D-FINE-N (4M params, 42.8 AP, NMS-free) replaces YOLOX-Tiny for person detection. RTMPose-T (3.34M, ultra-light) replaces RTMPose-S for pose. X3D-XS (3.8M params, 3D CNN) replaces rule-based classification for hands-in-pockets using 16-frame video clips for temporal action recognition.
- **Addresses:** Multi-behavior diversity (dedicated transformer + action recognition per modality), hands-in-pockets ambiguity (learned temporal patterns replace noisy pose heuristics), edge compute constraints (D-FINE-N 4M + RTMPose-T 3.34M + X3D-XS 3.8M = ~11.1M total params)
- **Pros:** D-FINE-N achieves higher AP than YOLOX-Tiny with fewer params; X3D-XS captures temporal motion patterns that static frames miss, potentially improving hands-in-pockets accuracy; NMS-free D-FINE reduces post-processing overhead
- **Cons:** X3D-XS requires ~3K labeled video clips (no public dataset exists); action recognition training significantly increases development time; 16-frame clip buffering adds latency; all components less proven on target edge chips than CNN counterparts

### Option 3: Single-Model End-to-End -- YOLOX-M Multi-Class

- **Approach:** Train a single YOLOX-M (25.3M params) with all four violation types as object detection classes: person, hands_in_pockets, phone, no_handrail, unsafe_stair_crossing. Each violation is treated as a bounding-box classification problem.
- **Addresses:** Multi-behavior diversity (single model handles all classes), edge compute constraints (single inference pass, ~50+ FPS on AX650N)
- **Pros:** Simplest architecture -- one model, one inference pass; highest FPS ceiling; easiest to deploy and maintain; no zone configuration or rule engine needed
- **Cons:** Cannot detect hands-in-pockets or handrail violations from static frames (these require pose/zone context, not just bounding boxes); no public dataset for 3 of 4 classes; phone detection accuracy degraded by not using person crops; loses temporal confirmation capability; effectively unworkable for 3 of 4 violation types

**Decision:** Option 1 (Multi-Model Pipeline) is recommended. It is the only approach that can detect all four violation types within edge compute constraints, requires training only one model (phone detection), and provides configurable rule-based thresholds for the three pose/zone-dependent behaviors. Options 2 and 3 either require unavailable training data or cannot detect the required violation types.

## Approach

**Architecture:** Hybrid pipeline -- YOLOX-M (phone detection, trained) + YOLOX-Tiny (person detection, pretrained) + RTMPose-S (pose, pretrained) + Rule engine (CPU). All components Apache 2.0.

**Performance Targets:**
- mAP@0.5 >= 0.80 (overall)
- Precision >= 0.85
- Recall >= 0.82
- FP Rate < 5%

---

## Detection Classes

| Class ID | Class Name | Detection Method | Description |
|---|---|---|---|
| 0 | person | Pretrained (YOLOX-Tiny) | Walking/standing person |
| 1 | hands_in_pockets | Rule-based (pose keypoints) | Both hands in pockets -- wrist-hip proximity + low wrist confidence |
| 2 | phone_usage | Trained (YOLOX-M) | Using mobile phone while walking -- phone bbox detection |
| 3 | no_handrail | Rule-based (pose + zone) | Not holding handrail on stairs -- wrist position vs zone polygon |
| 4 | unsafe_stair_crossing | Rule-based (zone + trajectory) | Crossing stairs diagonally -- trajectory angle vs stair direction |

> Only `phone_usage` (class 2) requires model training. Classes 1, 3, 4 are detected via pretrained pose estimation + rule-based logic.

### Alert Logic

| Model | Min Confidence | Min Duration | Tracking Required |
|---|---|---|---|
| Poketenashi | 0.65 | 30 frames (1 sec) | Yes |

### Temporal Confirmation Frames

| Violation Type | Frames | Duration @ 30fps |
|---|---|---|
| phone_usage | 15 | ~0.5 sec |
| hands_in_pockets | 30 | ~1 sec |
| no_handrail | 45 | ~1.5 sec |
| unsafe_stair_crossing | 15 | ~0.5 sec |

---

## Dataset

### Phone Detection Datasets

| Dataset | Images | Format | License | Quality | Notes |
|---|---|---|---|---|---|
| **FPI-Det** | 22,879 | YOLO | Academic | Best | Workplace/education/transport; faces (29K) + phones (10K); 18,800 train / 1,730 val / 2,349 test |
| Phone Call Usage (Roboflow) | 3,115 | YOLO | Open | Good | Pretrained model available |
| Mobile Phone Detection (Roboflow) | 1,674 | YOLO | Open | Decent | Real-time usage detection project |
| Cellphone Dataset (Roboflow) | 1,901 | YOLO | Open | Decent | Subset of COCO 2017 |
| COCO "cell phone" class | ~5,000+ | COCO JSON | CC BY 4.0 | Good | Already in pretrained YOLO weights |

**Key Research -- FPI-Det Paper (Sep 2025, arXiv:2509.09111):**
- Dual-class detection (face + phone) enables determining *who* is using *which* phone
- YOLOv8-x achieves AP@50 = 92.4% for phone detection on FPI-Det
- Extreme scale variation, frequent occlusions, varied capture conditions
- URL: [github.com/KvCgRv/FPI-Det](https://github.com/KvCgRv/FPI-Det)

### Consolidated Data Strategy

| Sub-behavior | Detection Method | Public Data | Custom Data Needed |
|---|---|---|---|
| Phone usage | Object detection (YOLOX-M) | FPI-Det 22,879 + Roboflow ~4,789 = **~28K** | 200-400 factory-specific |
| Hands in pockets | Pose estimation + rules | COCO-Pose 200K+ (pretrained) | 100-200 for threshold calibration |
| No handrail | Pose + zone logic | COCO-Pose 200K+ (pretrained) | 100-200 for threshold calibration |

**No dedicated hands-in-pockets or handrail usage detection dataset exists anywhere.** The hybrid approach (pose keypoint rules) eliminates the need for custom annotation of these behaviors.

### Custom Data Requirements

| Scenario | Images Needed | Collection Method | Priority |
|---|---|---|---|
| Phone usage while walking (factory) | 200-400 | Staged + natural | HIGH |
| Hands in pockets (walking) | 100-200 | Staged | HIGH (calibration only) |
| No handrail on stairs | 100-200 | Staged | HIGH (calibration only) |
| Proper handrail use (negative) | 100-200 | Staged | HIGH |
| Normal walking (negative baseline) | 200-400 | Natural footage | MEDIUM |

**Total custom data: 800-1,600 images** (reduced from 4,000-6,100 by using the hybrid pose-based approach).

### Annotation Guidelines

Only `phone` requires bounding box annotation:
1. Draw bounding box tightly around the phone object (not the hand or person)
2. Phone must be visible -- do not annotate fully occluded phones
3. Include phones in any orientation: held to ear, held in front of face, in hand at side
4. Include partially occluded phones (hand covering part of phone)

Rule calibration data (hands_in_pockets, no_handrail) needs only binary labels (violation / no violation), not bounding boxes.

---

## Architecture

### Multi-Model Pipeline

```
Camera Frame (1080p)
    |
    v
+------------------------------------------+
|  Model 1: Person Detector + Tracker      |
|  YOLOX-Tiny -> person bboxes             |
|  ByteTrack -> person IDs + trajectories  |
|  Speed: 50+ FPS (AX650N)                |
+-------------------+----------------------+
                    |
        +-----------+-----------+
        v                       v                    v
+----------+  +-----------+  +---------------+
| Model 2  |  | Model 3   |  | Zone Engine   |
| Phone    |  | Pose      |  | (CPU only)    |
| Detector |  | Estimator |  |               |
| YOLOX-N  |  | RTMPose-S |  | - Stair poly  |
| 1 class  |  | 17 kpts   |  | - Handrail    |
| crop:    |  | 256x192   |  | - Direction   |
| upper    |  | per person |  | - Trajectory  |
| body     |  |           |  |               |
+----+-----+  +-----+-----+  +-------+-------+
     |              |                |
     v              v                v
+----------------------------------------------+
|  Rule Engine (CPU, <1ms)                     |
|                                              |
|  (1) Phone Usage:                            |
|      phone bbox overlaps hand/ear region     |
|      + person is walking (trajectory > thr)  |
|      -> PHONE_VIOLATION                      |
|                                              |
|  (2) Hands in Pockets:                       |
|      wrist_y between hip_y +/- margin        |
|      AND wrist_confidence < 0.3 (occluded)   |
|      AND sustained >= 30 frames              |
|      -> HANDS_IN_POCKETS                     |
|                                              |
|  (3) No Handrail (stair zone only):          |
|      person centroid inside stair_polygon    |
|      AND no wrist keypoint within 50px of    |
|         handrail_polygon edge                |
|      AND sustained >= 30 frames              |
|      -> NO_HANDRAIL                          |
|                                              |
|  (4) Diagonal Stair Crossing:                |
|      person in stair_polygon                 |
|      trajectory_angle vs stair_direction >30 |
|      AND sustained >= 15 frames              |
|      -> DIAGONAL_CROSSING                    |
+----------------------------------------------+
```

### Component Models

| Component | Model | Purpose | Training Required? |
|---|---|---|---|
| Person detection + tracking | YOLOX-Tiny (COCO pretrained) | Detect and track persons | NO (pretrained) |
| Pose estimation | RTMPose-S (COCO-Pose pretrained) | 17 body keypoints per person | NO (pretrained) |
| Phone detection | YOLOX-M (fine-tuned on FPI-Det + Roboflow) | Detect phone objects | YES (~8-10h training) |
| Stair zone detection | Supervision PolygonZone | Define stair/corridor regions | NO (configuration) |
| Rule engine | Python logic | Classify violations from keypoints | NO (threshold calibration only) |

### Sub-Behavior Detection Logic

**Phone Usage (CNN-based):** YOLOX-M detects phone objects in upper-body crops. Rule engine checks if phone bbox overlaps hand/ear region and person is walking (displacement > threshold per frame).

**Hands in Pockets (Pose-based):** Uses COCO-Pose keypoints -- wrist (idx 9, 10) proximity to hips (idx 11, 12) combined with low wrist confidence (< 0.3 = occluded by pockets). Both hands must be detected as in-pockets. Wrists must be below shoulders (to exclude crossed arms).

**No Handrail (Pose + Zone):** Person centroid must be inside stair_polygon. Neither wrist keypoint is within proximity threshold of handrail_polygon edge. Sustained over 45 frames.

**Diagonal Stair Crossing (Trajectory):** Person trajectory angle vs stair direction vector > 30 degrees, sustained over 15 frames.

### Transformer-Based Alternative

**D-FINE-N as Person Detector (Apache 2.0):** 4M params (smaller than YOLOX-Tiny at 5.1M), 42.8 AP (vs ~33 AP), NMS-free.

**MediaPipe Pose as Alternative Pose Estimator:** 33 landmarks (vs 17 for RTMPose) with hand/foot detail. Better for hands-in-pockets (wrist + pinky/index/thumb positions) and phone usage (finger positions near ear). Trade-off: single-person tracker requires running one instance per detected person crop.

**Recommended hybrid:** RTMPose-S on NPU for primary pose. MediaPipe Pose Lite on CPU for selected persons requiring finer hand/finger analysis.

### Action Recognition Options (If Rules Insufficient)

**Option A -- X3D-XS (3D CNN):** 3.8M params, 0.6G FLOPs, 16-frame video clips. Captures temporal motion patterns. Requires ~3K labeled video clips.

**Option B -- SlowFast Pre-Training + Distillation:** Pre-train SlowFast (Apache 2.0) on Kinetics-400, fine-tune on ~500-1000 labeled Poketenashi clips, distill to X3D-XS for edge.

> VideoMAE (CC-BY-NC 4.0) is NOT approved for commercial use. Use SlowFast (Apache 2.0) instead.

---

## Training Results

### Industry Benchmarks

| Behavior | Typical mAP@0.5 | Typical Precision | Typical Recall | Challenge Level |
|---|---|---|---|---|
| Phone usage (clear view) | 0.85-0.92 | 0.88-0.92 | 0.87-0.90 | Easy-Medium |
| Phone usage (occluded) | 0.70-0.80 | 0.75 | 0.72 | Medium-Hard |
| Hands in pockets (pose-based) | 0.70-0.80 | 0.73 | 0.71 | Hard (no public dataset) |
| No handrail (zone + pose) | 0.75-0.85 | 0.78 | 0.76 | Medium |
| Unsafe stair crossing | 0.75-0.85 | 0.78 | 0.76 | Medium |

### Acceptance Metrics

| Metric | Target | Business Impact |
|---|---|---|
| Precision | >= 0.85 | Of 100 poketenashi alarms, 85 are real violations |
| Recall | >= 0.82 | Of 100 actual violations, 82 are detected |
| FP Rate | < 5% | ~3-5 false alarms/day |
| FN Rate | < 6% | ~4-6 missed violations/day (assuming 100 violations/day) |
| mAP@0.5 | >= 0.80 | Overall system performance |

### Business Impact (Daily Operations)

| Metric | Value |
|---|---|
| Daily violations (est.) | 50 |
| Detected correctly | 41 |
| Missed | 9 |
| False alarms/day | 2-3 |
| Annual false alarms | 730-1,095 |

*Assumptions: Typical factory with 100-200 workers per shift across multiple camera zones.*

### Training Requirements

| Component | Training Time | Dataset | Notes |
|---|---|---|---|
| Person detection | 0 hours | COCO pretrained | No training |
| Pose estimation | 0 hours | COCO-Pose pretrained | No training |
| Phone detection | 8-10 hours | FPI-Det (22,879) | Fine-tune YOLOX-M on phone class |
| Rule calibration | 2-4 hours | Custom (200-400 images) | Threshold tuning, no GPU needed |

**Total training time: 10-14 hours**

---

## Edge Deployment

### Performance Budget

| Model | AX650N FPS | CV186AH FPS | Runs |
|---|---|---|---|
| YOLOX-Tiny (person) | 50+ | 25+ | Every frame |
| YOLOX-Nano (phone) | 80+ | 40+ | On upper-body crops |
| RTMPose-S (pose) | 40+ | 20+ | On detected persons |
| Rule engine (CPU) | Negligible | Negligible | Every frame |
| **Combined pipeline** | **~15-20** | **~8-12** | |

### Inference Requirements

| Component | FLOPs | Parameters | Latency (T4) | Latency (Edge) |
|---|---|---|---|---|
| Person (YOLOX-Tiny) | 6.5 GFLOPS | 4.4M | ~2ms | ~6-8ms |
| Pose (RTMPose-S) | ~4 GFLOPS | 5.47M | ~3ms | ~5ms (AX650N) |
| Phone (YOLOX-M) | 20.7 GFLOPS | 25.3M | 2.5ms | ~10-12ms |
| Rule engine | ~0 | 0 | <0.1ms | <0.1ms |
| **Total** | ~31 GFLOPS | ~35M | ~7.5ms | ~21-25ms |

### CV186AH Frame-Skip Optimization

Run pose every 2nd frame, phone detection every 3rd frame:
```
Frame 1: person detection + tracking
Frame 2: person detection + tracking + phone detection
Frame 3: person detection + tracking + pose estimation
Frame 4: person detection + tracking
Frame 5: person detection + tracking + phone detection
Frame 6: person detection + tracking + pose estimation
...
```

### Target Edge Chips

- **AX650N** (18 INT8 TOPS) -- primary target
- **CV186AH** (7.2 INT8 TOPS) -- secondary target
- Format: ONNX (INT8 quantized for edge)
- Power: 4-6W (3 models in pipeline)

### Shared Infrastructure with Zone Intrusion (Model I)

Person detection (YOLOX-Tiny) and ByteTrack tracking can be shared with Model I to reduce total system compute. Run detection once, feed both pipelines.

### Model Card (Deliverables)

Each release produces a model card at `docs/model_cards/h_poketenashi.md` and a YAML card at `releases/poketenashi/v<N>/model_card.yaml`.

**Model artifacts:**

| Artifact | Format | Path |
|---|---|---|
| PyTorch model | `.pth` | `runs/poketenashi/best.pt` |
| ONNX model | `.onnx` | `runs/poketenashi/export/h_phone_yoloxm_640_v{N}.onnx` |
| Training config | `.yaml` | `configs/poketenashi/06_training.yaml` |
| Metrics | `.json` | `runs/poketenashi/metrics.json` |

**Model card fields (auto-populated by `utils/release.py`):**

| Field | Source |
|---|---|
| Architecture, classes, input size | `06_training.yaml` |
| Dataset version (DVC tag) | `dvc tags list` |
| Training run path | `runs/poketenashi/` |
| mAP@0.5, Precision, Recall | `metrics.json` / `eval_results.json` |
| Per-class breakdown | Evaluation output |

---

## Zone Configuration

Each camera needs per-zone configuration for stair areas and handrails:

```yaml
camera_id: 1F_Z18_001
zones:
  stair_area:
    type: polygon
    points: [[120, 300], [520, 280], [580, 720], [80, 720]]

  handrail_left:
    type: polyline
    points: [[100, 310], [60, 720]]
    proximity_threshold: 50  # pixels

  handrail_right:
    type: polyline
    points: [[540, 290], [600, 720]]
    proximity_threshold: 50

  stair_direction:
    vector: [0.1, 1.0]
    diagonal_angle_threshold: 30  # degrees

alert_config:
  phone_usage:
    confidence: 0.65
    min_duration_frames: 30
    require_walking: true

  hands_in_pockets:
    wrist_confidence_threshold: 0.3
    min_duration_frames: 30

  no_handrail:
    require_stair_zone: true
    min_duration_frames: 30

  diagonal_crossing:
    angle_threshold: 30
    min_duration_frames: 15
```

---

## Development Plan

### Recommended Strategy

```
Phase 1: Pose + Rules (Weeks 1-4)
  - D-FINE-N (person, Apache 2.0) + RTMPose-S (pose) + YOLOX-Nano (phone)
  - Rule-based detection for all 4 sub-behaviors
  - Fastest to develop, no action recognition training needed

Phase 2: Validation (Weeks 5-6)
  - Test on factory footage
  - Measure FP/FN rates per sub-behavior
  - Identify which rules need refinement

Phase 3: Action Recognition (if needed)
  - If rule-based approach has high FP rate for hands-in-pockets
  - Train X3D-XS on factory video clips
  - Replace rule-based with learned classification
```

### Week-by-Week

- **Weeks 1-2:** Setup and data review. Review FPI-Det quality, verify YOLOX-Tiny and RTMPose-S on factory footage. Define stair zone polygons. Dataset quality review and label error audit. Curate ~5K balanced subset.
- **Weeks 3-4:** Phone detection v1 training on curated subset. Evaluate mAP/Precision/Recall. Error analysis.
- **Weeks 5-6:** Phone detection v2 with cleaned data + HPO. Parallel pose rule calibration (threshold tuning on 100-200 staged images).
- **Weeks 7-8:** Phone detection v3 if needed. Full pipeline integration (YOLOX-Tiny + RTMPose-S + YOLOX-M + rule engine + temporal consistency). End-to-end testing.
- **Week 9:** ONNX export, finalize pipeline, build alert dispatcher, per-behavior P/R/FP/FN reports, threshold fixes.

### Timeline

**3-4 weeks** (reduced from 4-5 weeks by using the hybrid pose-based approach)

---

## Gap G3: Pointing-and-Calling Detection

**Customer expects:** Detection of pointing-and-calling (指差呼称) gestures at designated points.

**Approach:** Pose-based gesture detection:
1. Person is in designated "pointing-and-calling zone" polygon
2. One arm extended forward (shoulder-wrist angle < 30 deg from horizontal)
3. Head oriented toward pointing direction (nose_x close to wrist_x)
4. Alert triggers if person passes through zone WITHOUT performing gesture

**Expected accuracy:** 70-80% (lower than other behaviors due to 2D pose limitations). Camera angle significantly affects reliability — recommend eye-level cameras near designated points.

## Key Commands

The only rule that requires its own training is `phone_usage`; the
other four are pretrained-only (DWPose) + per-rule logic. Replace the
config path with the feature folder for any rule-only command (e.g.
`features/safety-poketenashi_hands_in_pockets/configs/10_inference.yaml`).

```bash
# Train phone detection (feature: safety-poketenashi_phone_usage)
uv run core/p06_training/train.py --config features/safety-poketenashi_phone_usage/configs/06_training_yolox.yaml

# Evaluate
uv run core/p08_evaluation/evaluate.py \
  --model features/safety-poketenashi_phone_usage/runs/<ts>/best.pth \
  --config features/safety-poketenashi_phone_usage/configs/05_data.yaml \
  --split test

# Data preparation (phone_usage only — pose-rule features have no training data)
uv run core/p00_data_prep/run.py --config features/safety-poketenashi_phone_usage/configs/00_data_preparation.yaml

# Export
uv run core/p09_export/export.py \
  --model features/safety-poketenashi_phone_usage/runs/<ts>/best.pth \
  --training-config features/safety-poketenashi_phone_usage/configs/06_training_yolox.yaml \
  --export-config configs/_shared/09_export.yaml

# Pose-rule features (hands_in_pockets, no_handrail, stair_diagonal, point_and_call)
# have no training step — wire them into the demo via 10_inference.yaml only:
uv run demo  # multi-tab Gradio loads each feature's 10_inference.yaml
```

## Limitations

- **Hands-in-pockets false positives:** Arms naturally at sides can be confused with hands in pockets. Mitigated by requiring BOTH low wrist confidence AND proximity, plus temporal smoothing.
- **Phone detection in low light:** Mitigated by FPI-Det's varied lighting conditions and factory-specific augmentation images.
- **Handrail zone misconfiguration:** Requires per-camera calibration. Validation with test footage is essential.
- **Pose estimation with heavy clothing:** Factory uniforms may affect keypoint confidence. Threshold tuning may be needed.
- **Multi-model pipeline latency:** 3 models in sequence limits FPS. Shared person detection with Model I and frame-skip scheduling help.
- **No public dataset for hands-in-pockets or handrail usage:** Entirely rule-based detection for these behaviors.
- **Most challenging Phase 1 model:** Multi-behavior detection + lack of public data for 2 of 4 behaviors.

### Contingency Plans

**Phone detection underperforms (mAP < 0.75 after v2):**
1. Add FPI-Det full dataset (22,879 images) if not already included
2. Try 1280px input size
3. Switch to RT-DETRv2-S via HuggingFace Transformers

**Pose rules too inaccurate (FP rate > 10% after calibration):**
1. Tighten temporal consistency (increase confirmation frames from 30 to 60)
2. Add additional calibration data
3. Defer hands-in-pockets to Phase 2, keep phone + no-handrail only
4. Last resort: collect 1,000+ custom annotated images and train a dedicated detector

---

## Changelog

| Date | Change |
|---|---|
| 2026-03-26 | Initial platform feature doc merged from requirements, model card, and technical approach |

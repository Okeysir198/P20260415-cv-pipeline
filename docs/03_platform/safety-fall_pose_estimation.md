# Safety: Fall Pose Estimation
> ID: g | Owner: TBD | Phase: 1 | Status: training | Config: `features/safety-fall_pose_estimation/configs/06_training.yaml`
> See also: [Safety: Fall Classification](safety-fall_classification.md) — complementary classification-based approach

## Customer Requirements

**Source:** [Web Japan Co., Ltd. - AI Camera Solutions](https://www.webjapan.co.jp/solution/ai-camera/) -- AI cameras for watching over employees and workers with automatic fall detection and alarm capabilities. Supports 24/7 monitoring in care facilities and workplaces.

**Explicit Requirements:**
- "Watching over employees and workers. Detects falls and alarms."
- 24/7 monitoring of workers and elderly
- Automatic fall detection with alerts
- Supports "watching over" applications in care facilities and workplaces

## Business Problem Statement

- Falls in industrial workplaces cause severe injury and death, and rapid detection is critical for triggering emergency medical response
- Classification-based fall detection cannot reliably distinguish falls from normal work activities like sitting, crouching, or kneeling, leading to frequent false alarms that erode worker trust and cause alert fatigue
- Delayed fall detection increases the severity of injuries and exposes the organization to significant legal liability and workers' compensation claims
- Facilities require continuous monitoring across all shifts, including nights and weekends, where human observation is limited or absent
- False alarms from existing classification approaches disrupt workflows and may cause safety staff to ignore or disable the system entirely

## Technical Problem Statement

- **Worker fall safety -> Two-stage pipeline complexity:** Detecting falls via pose estimation requires a two-stage pipeline (person detection followed by per-person keypoint estimation), which adds latency and deployment complexity compared to a single classification model
- **False alarm reduction -> Keypoint visibility:** Rule-based fall detection relies on accurate body keypoint positions, but occlusion by equipment, other workers, or unusual camera angles can corrupt keypoint data and trigger false positives or missed detections
- **Distinguish falls from normal activities -> Temporal analysis:** Falls must be distinguished from intentional lying down or sitting based on temporal movement patterns, requiring frame-to-frame tracking and consistency checks over a sliding window
- **Continuous 24/7 coverage -> Edge deployment of multi-model pipeline:** Running two sequential models (detector + pose estimator) plus rule-based logic on resource-constrained edge chips (AX650N / CV186AH) must still achieve real-time performance while leaving headroom for other safety models running concurrently

## Technical Solution Options

### Option 1: YOLOX-Tiny + RTMPose-S (Two-Stage -- Recommended)

- **Approach:** Two-stage pipeline: YOLOX-Tiny (5.1M params) detects person bounding boxes, then RTMPose-S (5.47M params, CSPNeXt backbone + SimCC head) estimates 17 COCO keypoints per person crop at 256x192. Geometric and temporal rules on keypoint positions determine fall state. Apache 2.0 throughout.
- **Addresses:** Two-stage pipeline complexity (YOLOX-Tiny at 50+ FPS + RTMPose-S at 4.79ms proven on AX650N = ~15ms combined for 1 person), false alarm reduction (keypoint geometry distinguishes sitting/crouching from falling), temporal analysis (80% of 30-frame window + velocity check), edge deployment (both models have proven ONNX export and INT8 quantization on AX650N)
- **Pros:** Proven on AX650N (4.79ms RTMPose-S reference code); pure CNN architecture with excellent INT8 quantization; multi-person support via external detector; higher COCO keypoint AP (72.2) than alternatives; PyTorch ecosystem integration for fine-tuning
- **Cons:** Two-model pipeline increases deployment complexity; requires per-person crop processing; limited to 17 keypoints (no hand/foot detail or 3D z-coordinates)

### Option 2: D-FINE-N + RTMPose-S (Transformer Alternative)

- **Approach:** Replace YOLOX-Tiny with D-FINE-N (4M params, 42.8 AP, NMS-free, Apache 2.0) as the person detector, keeping RTMPose-S for pose estimation. Same geometric/temporal rule-based fall detection logic.
- **Addresses:** Two-stage pipeline complexity (D-FINE-N has fewer params and higher AP than YOLOX-Tiny, NMS-free output simplifies pipeline), edge deployment (4M params vs 5.1M for YOLOX-Tiny, lighter on NPU memory)
- **Pros:** NMS-free output simplifies post-processing; 20% fewer parameters than YOLOX-Tiny with higher COCO AP; transformer hybrid encoder captures global context for better person detection in crowded scenes
- **Cons:** Less proven on AX650N compared to YOLOX-Tiny; transformer attention layers require mixed-precision quantization (attention in FP16) on edge chips; newer architecture with less production track record

### Option 3: YOLOX-Tiny + MediaPipe Pose (Ultra-Light Alternative)

- **Approach:** YOLOX-Tiny detects persons, MediaPipe Pose Lite (1.3M params) or Full (3.5M params) estimates 33 landmarks with 3D z-coordinates per person. Runs on ARM CPU via TFLite + XNNPack, freeing NPU for other models. Apache 2.0.
- **Addresses:** Two-stage pipeline complexity (CPU-based pose avoids NPU contention -- detector runs on NPU while pose runs on Cortex-A55 concurrently), false alarm reduction (33 landmarks with hand/foot detail and 3D z-coordinates enable more robust ground-contact and depth-aware fall detection), edge deployment (TFLite-native on ARM CPU, no NPU compilation needed)
- **Pros:** 33 landmarks provide hand/foot detail and 3D z-coordinates for depth-aware fall detection; runs on CPU leaving NPU free for other safety models; proven mobile performance (~50 FPS Lite); shares pose model with Poketenashi (phone detection) use case
- **Cons:** Built-in tracker is single-person only (multi-person requires external detector); TFLite-to-ONNX conversion needed for NPU deployment introduces accuracy risk; no AX650N NPU reference code; lower keypoint accuracy on COCO benchmark compared to RTMPose-S

**Decision:** Start with Option 1 (YOLOX-Tiny + RTMPose-S) as the primary path -- proven on AX650N with reference code, multi-person native support, and excellent INT8 quantization. Train Option 2 (D-FINE-N + RTMPose-S) in parallel as the transformer alternative. Evaluate Option 3 (MediaPipe Pose) for scenarios where 33 landmarks or CPU-offloading provide measurable FP/FN improvement.

## Architecture

The pose estimation approach uses a two-stage pipeline: (1) detect persons with YOLOX-Tiny, (2) estimate 17 or 33 keypoints per person with RTMPose-S or MediaPipe Pose, then (3) apply geometric rules on keypoint positions to determine fall state. This is the recommended approach for high-accuracy fall detection with low false positive rates.

**Pipeline (YOLOX-Tiny + RTMPose-S):**

```
Input (640x640)
    |
    v
+------------------------------------------+
|  Stage 1: Person Detection               |
|  YOLOX-Tiny -> person bounding boxes     |
|  Speed: 50+ FPS on AX650N               |
+------------------------------------------+
    | person crops
    v
+------------------------------------------+
|  Stage 2: Pose Estimation                |
|  RTMPose-S (5.47M params, Apache 2.0)   |
|  CSPNeXt backbone + SimCC head           |
|  Input: 256x192 (per person crop)        |
|  Output: 17 COCO keypoints (x, y, conf) |
|  Speed: 40+ FPS per person on AX650N     |
+------------------------------------------+
    | 17 keypoints per person
    v
+------------------------------------------+
|  Fall Detection Rules (CPU, ~0.1ms)      |
|                                          |
|  Rule 1: Aspect Ratio                    |
|  bbox_w / bbox_h > 1.5 -> horizontal    |
|                                          |
|  Rule 2: Keypoint Geometry               |
|  hip_y >= shoulder_y - 0.05             |
|  AND hip_y > 0.7 (near ground)          |
|                                          |
|  Rule 3: Temporal Consistency            |
|  fall persists >= 80% of 30-frame window |
|                                          |
|  Rule 4: Velocity Check                  |
|  rapid hip_y descent > threshold         |
|  (Distinguishes fall vs lying down)      |
+------------------------------------------+
    |
    v
  Alert: FALL_DETECTED (with person ID from tracker)
```

**Pros:** Lower false positive rate than classification (pose geometry distinguishes sitting/crouching from falling). Rich keypoint features enable multiple independent fall indicators.
**Cons:** Two-model pipeline, slightly higher latency, requires per-person crop processing.

### Acceptance Metrics

| Metric | Target | Min Acceptable |
|---|---|---|
| mAP@0.5 | >= 0.85 | 0.75 |
| Precision | >= 0.90 | >= 0.87 |
| Recall | >= 0.88 | >= 0.85 |
| FP Rate | < 3% | < 5% |
| FN Rate | < 2% | < 4% |

Fall detection prioritizes recall over precision (life safety). Missing a fall is more critical than a false alarm.

---

## 4. Detection Classes

The pose approach does not directly detect "fall" as a class. Instead, it detects `person` and extracts keypoints, then applies rule-based logic to determine fall state.

### Trained Detection Class

| Class ID | Class Name | Description |
|---|---|---|
| 0 | person | Tracked person (17 COCO keypoints from RTMPose, or 33 landmarks from MediaPipe) |

### Alert States (Rule-Based, Not Model Outputs)

| Alert State | Description | Determined By |
|---|---|---|
| fall_detected | Person in fallen/lying position | Keypoint geometry rules + temporal consistency |
| unsafe_posture | Person in unsafe bent/crouching position | Keypoint angle thresholds |

### Alert Logic

| Model | Min Confidence | Min Duration | Tracking Required |
|---|---|---|---|
| Fall Pose | 0.65 | 15 frames (500ms) | Yes |

### Fall Detection Rules (Detailed)

```python
def is_fall_aspect_ratio(bbox):
    """Rule 1: Aspect ratio based"""
    x, y, w, h = bbox
    return (w / h) > 1.5

def is_fall_pose(keypoints):
    """Rule 2: Keypoint based (most accurate)"""
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

**Rule 4 (Velocity Check):** Track `delta_hip_y` between frames. A rapid descent (`delta_hip_y > threshold`) distinguishes an active fall from a person already lying down.

---

## 5. Dataset

### Primary Training Data

| Source | Images | Purpose |
|---|---|---|
| g_fall_pose dataset | 111 | Factory fall keypoints (threshold calibration) |
| COCO Keypoints | ~150K | Pose estimator pretraining (17 keypoints, CC BY 4.0) |

Dataset path: `../../dataset_store/fall_pose_estimation/` with `{train,val,test}/{images,labels}/` structure.

**Critical note:** 111 pose images is insufficient for fine-tuning. Must collect or generate ~2K+ factory-specific fall simulation videos. The COCO pretrained pose model (RTMPose-S) serves as the base; factory data is used for threshold calibration, not full retraining.

### Public Datasets (Reference / Expansion)

| Dataset | Type | Size | License | Quality | Notes |
|---|---|---|---|---|---|
| COCO Keypoints | Pose | 200K+ images | CC BY 4.0 | 5/5 | General pose; 17 keypoints |
| Le2i Fall Detection | Video | ~130 sequences | Academic | 4/5 | 4 fall scenarios; frame extraction needed |
| UR Fall Detection | Video | 70 sequences | Academic | 4/5 | 30 falls + 40 ADLs; RGB + Kinect depth |
| MPDD Fall Dataset | Images | 1,200+ | Academic | 4/5 | Multi-person falls; Nature Scientific Data 2025 |
| UP-Fall Detection | Video | 11 activities | Academic | 3/5 | Multi-modal; 17 young subjects (ages 18-24) |

**Dataset URLs:**
- COCO Keypoints: [cocodataset.org](https://cocodataset.org)
- Le2i Fall: Contact Universite de Bourgogne (research use)
- UR Fall: University of Rzeszow (contact)
- MPDD: Nature Scientific Data (search "MPDD fall detection dataset 2025")

### Custom Data Requirements

**Staged Fall Collection:**

| Scenario | Sequences Needed | Collection Method | Priority |
|---|---|---|---|
| Forward falls | 50-80 | Staged with mats | High |
| Backward falls | 50-80 | Staged with mats | High |
| Sideways falls | 40-60 | Staged with mats | Medium |
| Near-falls (stumble recovery) | 30-50 | Staged | Medium |
| Factory context falls | 20-30 | Staged in factory setting | Medium |

Total: 200-300 fall sequences (~2,000-3,000 extracted frames).

**Non-Fall Balance Data:**
- Sitting: 500-800 images
- Lying down: 300-500 images
- Kneeling/crouching: 500-700 images
- Standing/walking: 500-700 images
- Total non-fall: 1,800-2,700 images

### Annotation Guidelines

**Pose Keypoints (COCO 17-keypoint format):**
Nose, left/right eye, left/right ear, left/right shoulder, left/right elbow, left/right wrist, left/right hip, left/right knee, left/right ankle.

**Rules:**
1. Use pose annotation (17 COCO keypoints) for all person instances
2. Annotate body orientation (horizontal vs vertical)
3. Mark hip-shoulder height difference
4. Include temporal context (3-5 frames before/after fall event)
5. For unsafe posture: mark awkward positions (deep squat, over-reaching); exclude normal work-related sitting/kneeling/crouching
6. Use video sequences (not single frames) for temporal context

**Quality Check:**
- Minimum 100 annotated fall sequences
- Verify temporal consistency (fall status should not flicker between frames)
- 20% expert review

---

## Architecture Details

### RTMPose-S

RTMPose-S is the primary pose estimator -- proven on AX650N at 4.79ms (reference code: `ax_simcc_pose_steps.cc`), pure CNN architecture with excellent INT8 quantization.

| Property | Value |
|---|---|
| Person detector | YOLOX-Tiny (5.1M params, Apache 2.0) |
| Pose estimator | RTMPose-S (CSPNeXt backbone + SimCC head) |
| Pose params | 5.47M |
| Pose FLOPs | 0.68G |
| Pose input | 256x192 (per person crop) |
| Keypoints | 17 COCO keypoints (x, y, confidence) |
| COCO AP | 72.2 |
| AX650N latency | 4.79ms (proven) |
| License | Apache 2.0 |

**RTMPose variant comparison (all Apache 2.0):**

| Model | Params | FLOPs | COCO AP | AX650N FPS (est.) | CV186AH FPS (est.) |
|---|---|---|---|---|---|
| RTMPose-T | 3.34M | 0.36G | 68.5 | ~50+ | ~25+ |
| **RTMPose-S** | **5.47M** | **0.68G** | **72.2** | **~40+ (4.79ms proven)** | **~20+** |
| RTMPose-M | 13.59M | 1.93G | 75.8 | ~25+ | ~12+ |

**Why NOT ViTPose:** ViTPose-S (24M params, 73.8 AP) uses transformer attention layers that quantize poorly to INT8 on NPU chips. RTMPose-S achieves comparable accuracy (72.2 AP) with 5x fewer parameters and excellent INT8 quantization.

#### RTMPose Architecture

```
Person crop (256x192)
    |
    v
+------------------------------------------+
|  CSPNeXt Backbone (pure CNN)             |
|  - Cross Stage Partial blocks            |
|  - Depthwise separable convolutions      |
|  - NO attention layers                   |
|  - INT8 quantization: EXCELLENT          |
+------------------------------------------+
    |
    v
+------------------------------------------+
|  SimCC Head (1-D Coordinate Classif.)    |
|  - Classifies x,y independently          |
|  - No large 2D heatmap decode            |
|  - Faster post-processing than heatmaps  |
|  - Better quantization properties        |
+------------------------------------------+
    |
    v
  17 keypoints (x, y, confidence)
    |
    v
  Fall detection rules (geometric + temporal)
```

#### RTMPose Training Configuration

```yaml
model:
  architecture: rtmpose-s
  num_keypoints: 17
  pretrained: rtmpose_s_coco.pth  # COCO pose pretrained (Apache 2.0)

dataset:
  coco_pose: ~150K images (person keypoints)
  factory_fall: 111 images + collected fall simulations

training:
  epochs: 100
  batch_size: 32
  optimizer: AdamW
  lr: 0.001
  loss: mse  # or combined MSE + bone-length loss

augmentation:
  random_flip: 0.5
  random_rotation: [-30, 30]
  random_scale: [0.75, 1.25]
  color_jitter: true
  half_body_transform: 0.3
```

### MediaPipe Pose

MediaPipe Pose Landmarker is Google's ML solution for high-fidelity body pose tracking. Built on BlazePose research, it uses a two-stage detector-tracker architecture:

1. **Person Detector (Stage 1):** SSD-based detector runs on first frame or when tracking is lost -- predicts hip midpoint, circumscribing circle radius, and shoulder-hip incline angle. Input: 224x224.
2. **Landmark Model (Stage 2):** MobileNetV2-like CNN predicts 33 landmarks within the tracked ROI. Each landmark has 5 values: x, y, z (3D), visibility, presence.

The tracker-based design means the heavy detector only runs intermittently -- subsequent frames use lightweight landmark prediction on the tracked ROI.

#### 33-Landmark Topology (vs 17 COCO Keypoints)

```
MediaPipe Pose: 33 landmarks                    COCO: 17 keypoints
------------------------------                   ------------------
 0: nose                                          0: nose
 1: left_eye_inner      (extra)                   1: left_eye
 2: left_eye                                      2: right_eye
 3: left_eye_outer      (extra)                   3: left_ear
 4: right_eye_inner     (extra)                   4: right_ear
 5: right_eye                                     5: left_shoulder
 6: right_eye_outer     (extra)                   6: right_shoulder
 7: left_ear                                      7: left_elbow
 8: right_ear                                     8: right_elbow
 9: mouth_left          (extra)                   9: left_wrist
10: mouth_right         (extra)                  10: right_wrist
11: left_shoulder                                11: left_hip
12: right_shoulder                               12: right_hip
13: left_elbow                                   13: left_knee
14: right_elbow                                  14: right_knee
15: left_wrist                                   15: left_ankle
16: right_wrist                                  16: right_ankle
17: left_pinky          (extra -- hand detail)
18: right_pinky         (extra -- hand detail)
19: left_index          (extra -- hand detail)
20: right_index         (extra -- hand detail)
21: left_thumb          (extra -- hand detail)
22: right_thumb         (extra -- hand detail)
23: left_hip
24: right_hip
25: left_knee
26: right_knee
27: left_ankle
28: right_ankle
29: left_heel           (extra -- foot detail)
30: right_heel          (extra -- foot detail)
31: left_foot_index     (extra -- foot detail)
32: right_foot_index    (extra -- foot detail)
```

**Key advantage over COCO 17:** 16 additional landmarks -- hand detail (pinky/index/thumb per hand), foot detail (heel/foot_index per foot), face detail (eye inner/outer, mouth corners). Particularly valuable for:
- **Fall detection:** Heel/foot landmarks improve ground-contact detection
- **3D z-coordinate:** Enables depth-aware fall detection without stereo cameras
- **Shared with Poketenashi:** Finger landmarks for phone grip, hands-in-pockets detection

#### MediaPipe Model Variants

| Variant | Size | Params | MFLOPs | Mobile CPU FPS | Mobile GPU FPS | Use Case |
|---------|------|--------|--------|----------------|----------------|----------|
| **Lite** | **3 MB** | **1.3M** | **2.7** | **~50** | **~49** | Edge deployment (AX650N/CV186AH) |
| **Full** | **6 MB** | **3.5M** | **6.9** | **~18** | **~40** | Balanced accuracy/speed |
| Heavy | 26 MB | ~15M | N/A | ~4 | ~19 | Desktop/high-accuracy only |

#### MediaPipe Fall Detection Features (33 Landmarks)

| Feature | Landmarks Used | Description |
|---------|---------------|-------------|
| **Torso angle** | shoulders (11,12), hips (23,24) | Angle between shoulder-hip midpoint line and vertical. Upright ~0-15 deg, fallen ~60-90 deg |
| **BBox aspect ratio** | All 33 landmarks | Width/height ratio of landmark bounding box. Standing: W/H < 1. Fallen: W/H > 1 |
| **Hip descent velocity** | hips (23,24) | Vertical descent speed of hip midpoint. Sudden drop = potential fall |
| **Ground contact** | heels (29,30), foot_index (31,32) | Distance between hips and feet shrinks during fall -- only available with 33 landmarks |
| **Knee collapse** | hips (23/24), knees (25/26), ankles (27/28) | Knee angle beyond normal threshold indicates leg buckle |
| **Head-to-feet distance** | nose (0), ankles (27,28) | Vertical distance compresses dramatically during fall |
| **3D depth change** | z-coordinates of hips | Forward/backward fall detection using z-axis (unique to MediaPipe) |

Published accuracy: 95.84% fall detection accuracy on NTU RGB+D dataset using hip descent + bbox W/H ratio + skeletal keypoints.

#### MediaPipe Fall Detection Rules

```
Fall Detection Rules (CPU, ~0.1ms)

Rule 1: Torso Angle
  shoulder_mid = avg(11, 12)
  hip_mid = avg(23, 24)
  torso_angle = angle_to_vertical(shoulder_mid, hip_mid)
  is_fallen = torso_angle > 60 deg

Rule 2: Landmark BBox Aspect Ratio
  bbox = bounding_box(all 33 landmarks)
  is_horizontal = bbox_w / bbox_h > 1.5

Rule 3: Hip Descent Velocity
  hip_center = avg(23, 24)
  velocity = delta_hip_y between frames
  rapid_fall = velocity > threshold

Rule 4: Ground Contact (33-landmark exclusive)
  heel_y = avg(29, 30)
  foot_y = avg(31, 32)
  hip_near_feet = abs(hip_y - foot_y) < 0.15 * person_height

Rule 5: Temporal Consistency
  window = last 30 frames
  confirm_fall = sum(fall) >= 24 (80%)
```

#### MediaPipe vs RTMPose Comparison

| Feature | MediaPipe Pose Full | MediaPipe Pose Lite | RTMPose-S | RTMPose-T |
|---------|--------------------|--------------------|-----------|-----------|
| **Landmarks** | **33 (3D)** | **33 (3D)** | 17 (2D) | 17 (2D) |
| **Params** | 3.5M | **1.3M** | 5.47M | 3.34M |
| **Model size** | 6 MB | **3 MB** | ~11 MB | ~7 MB |
| **INT8 size** | ~3 MB | **~1.5 MB** | ~3 MB | ~2 MB |
| **Mobile CPU FPS** | ~18 | **~50** | ~40 (4.79ms AX650N) | ~50 |
| **Multi-person** | No (single tracker) | No (single tracker) | Yes (with detector) | Yes (with detector) |
| **3D landmarks** | **Yes (z-axis)** | **Yes (z-axis)** | No | No |
| **Hand detail** | **6 landmarks (per hand)** | **6 landmarks (per hand)** | 1 (wrist only) | 1 (wrist only) |
| **Foot detail** | **4 landmarks (heel + toe)** | **4 landmarks (heel + toe)** | 1 (ankle only) | 1 (ankle only) |
| **Native format** | TFLite | TFLite | PyTorch/ONNX | PyTorch/ONNX |
| **AX650N proven** | No (TFLite -> ONNX conversion needed) | No | **Yes (4.79ms, reference code)** | Yes |
| **License** | Apache 2.0 | Apache 2.0 | Apache 2.0 | Apache 2.0 |

**When to choose MediaPipe Pose over RTMPose:**
- Need 33 landmarks (hand/foot detail for ground-contact detection)
- Need 3D z-coordinates for depth-aware fall detection
- Running on standard ARM CPU (TFLite-native, no NPU compilation needed)
- Single-person or low-density scenes (MediaPipe's tracker is single-person)

**When to choose RTMPose over MediaPipe:**
- Multi-person dense factory scenes (RTMPose with external detector handles multi-person natively)
- AX650N/CV186AH NPU deployment (RTMPose has proven reference code, direct ONNX -> axmodel path)
- Need PyTorch ecosystem integration (training, fine-tuning within existing pipeline)
- Higher keypoint accuracy on COCO benchmark (72.2 AP vs MediaPipe's mobile-optimized accuracy)

### One-Stage Alternative: RTMO-S

```
Full frame (640x640)
    |
    v
  RTMO-S (Apache 2.0, ~8M params)
  - YOLO-style CNN backbone
  - Dual 1-D heatmap head
  - Detects persons AND keypoints in one forward pass
  - 67.7 AP on COCO
    |
    v
  All persons + 17 keypoints per person
```

**Trade-off:** Lower AP (67.7 vs 72.2) but simpler pipeline -- eliminates person detector entirely. Best when multiple people are in frame (>4 persons).

### Temporal Transformer (Phase 3, If Needed)

For cases where geometric rules have too high a false positive rate:

```
Per-frame keypoints (from RTMPose)
    |
    v
  Temporal Transformer (runs on CPU)
  Input: 30-frame keypoint sequence [30 x 17 x 3]
  - Positional encoding for frame order
  - Self-attention across time dimension
  - Captures movement patterns:
    - Rapid descent (fall)
    - Gradual descent (sitting down)
    - Stationary horizontal (lying)
  Output: fall_probability (0-1)
  Params: ~0.5M (tiny model)
  Runs on Cortex-A55 CPU: ~2ms
```

Achieves 97.6% accuracy in published research (2024). The lightweight temporal transformer runs on CPU with negligible overhead.

### Transformer Person Detector Alternative

D-FINE-N (4M params, 42.8 AP, NMS-free, Apache 2.0) can replace YOLOX-Tiny as the person detector -- better accuracy, fewer params, NMS-free.

### Development Plan

| Phase | Description |
|---|---|
| Week 1-2 | Verify RTMPose-S/MediaPipe pretrained model; run inference on 111 CCTV fall images; analyze keypoint accuracy and calibrate hip-shoulder ratio thresholds |
| Week 3-4 | Collect 200-300 additional calibration images from factory footage; validate keypoint accuracy on factory scenarios (low light, occlusion); tune fall thresholds |
| Week 5-6 | Implement fall detection rules (hip_y >= shoulder_y - 0.05); build temporal consistency (80% of sliding window); integrate with classification pipeline for ensemble |
| Week 7-8 | Export to ONNX/TFLite for edge; finalize ensemble logic with classification approach |

### Recommended Strategy

```
Phase 1 (Quick Win):
  YOLOX-M classification (person/fallen_person) -- already configured
  Use existing 17K dataset
  Target: mAP >= 0.80 baseline

Phase 2 (Accuracy Boost -- choose one pose approach):
  Option A: D-FINE-N (person) + RTMPose-S (17 keypoints, NPU)
    - Proven on AX650N (4.79ms), ONNX -> axmodel path validated
    - Best for multi-person dense scenes, NPU-native execution
    - Geometric fall rules (hip-shoulder ratio + ground proximity)

  Option B: D-FINE-N (person) + MediaPipe Pose Full (33 landmarks, CPU)
    - 33 3D landmarks -- heel/foot detail improves ground-contact detection
    - Runs on Cortex-A55 CPU (TFLite + XNNPack) -- zero NPU contention
    - 3D z-coordinates enable depth-aware fall vs sitting discrimination
    - Better hand detail for Poketenashi integration (shared pose model)
    - Needs TFLite -> ONNX conversion if NPU deployment is required

  Recommendation: Start with Option A (proven path), evaluate Option B
  in parallel for cases where 33 landmarks improve FP/FN rates.

  Target: Precision >= 0.90, Recall >= 0.88

Phase 3 (if FP rate too high):
  Add temporal transformer on keypoint sequences
  Distinguishes fall vs sit-down vs lie-down
  Target: FP Rate < 1%
```

---

## 6. Training Results

Training is in progress. Results will be documented here upon completion.

Target benchmarks:

| Metric | Target |
|---|---|
| mAP@0.5 | >= 0.85 |
| Precision | >= 0.90 |
| Recall | >= 0.88 |
| FP Rate | < 3% |
| FN Rate | < 2% |

### Industry Benchmarks (Reference)

| Approach | Accuracy | Latency | FP Rate |
|---|---|---|---|
| Pose + Rules (keypoints + geometric rules) | 90-95% | Medium | Low |
| Pose + Temporal (keypoints + LSTM/TCN) | 95-99% | Higher | Very Low |
| Hybrid (classify + pose + temporal) | Best | Highest | Lowest |
| YOLO-Pose baseline | 83-87% mAP | 15-20ms | ~4% |
| 3D CNN + LSTM | 91% F1 | 120ms | ~3% |

Temporal consistency (3-5 frames) reduces FP rate from ~8% to ~3%. Hip-shoulder height ratio is the most reliable fall indicator.

---

## 7. Edge Deployment

### Target Hardware

| Chip | INT8 TOPS | Notes |
|---|---|---|
| AX650N | 18 INT8 TOPS | Primary target |
| CV186AH | 7.2 INT8 TOPS | Resource-constrained alternative |

### Deployment Artifacts

| Artifact | Format | Purpose |
|---|---|---|
| Person detector (YOLOX-Tiny) | ONNX | Person bounding box detection |
| RTMPose-S | ONNX | 17 keypoints per person |
| MediaPipe Pose Lite/Full | TFLite | 33 landmarks per person (alternative) |
| Fall detection rules | Python | Geometric + temporal logic (CPU) |

### Expected Performance (RTMPose-S Path)

| Component | AX650N Latency | AX650N FPS |
|---|---|---|
| YOLOX-Tiny (person detection) | ~10ms | 50+ FPS |
| RTMPose-S (per person) | 4.79ms (proven) | ~40+ FPS |
| Fall rules (CPU) | ~0.1ms | Negligible |
| **Combined (1 person)** | **~15ms** | **~30+ FPS** |

Power consumption: 2.5-5W (NPU, depending on load). Temporal window: 3-5 frames (60-150ms latency acceptable).

### MediaPipe Edge Deployment Options

```
MediaPipe Pose (.task bundle)
    |
    +---> Option A: TFLite Runtime (Direct)
    |    - Use TFLite interpreter on ARM CPU (Cortex-A55/A53)
    |    - XNNPack delegate for CPU acceleration
    |    - Simplest path, no conversion needed
    |    - FPS: ~18-50 depending on variant
    |
    +---> Option B: TFLite -> ONNX -> NPU
    |    - Convert via tf2onnx or tflite2tensorflow
    |    - ONNX -> Pulsar2 (AX650N) or TPU-MLIR (CV186AH)
    |    - Runs on NPU for maximum throughput
    |    - Requires validation of conversion accuracy
    |
    +---> Option C: TFLite -> ONNX -> TensorRT (Jetson)
         - For NVIDIA Jetson edge devices
         - FP16/INT8 quantization supported
```

**Recommended on AX650N/CV186AH:**
- **CPU path (Option A):** Run MediaPipe Pose Lite on Cortex-A55 CPU via TFLite + XNNPack while YOLOX runs on NPU. CPU and NPU run concurrently -- zero NPU contention.
- **NPU path (Option B):** Convert to ONNX and compile to .axmodel/.cvimodel for NPU execution. Higher throughput but requires conversion validation.

---

## 8. Limitations

### Known Issues

1. **Limited factory-specific pose data:** Only 111 keypoint-annotated images for factory fall scenarios. COCO pretrained models may not generalize well to factory-specific conditions (low light, occlusion by equipment, unusual camera angles).
2. **Two-model pipeline complexity:** Requires running both person detector and pose estimator, increasing deployment complexity and total latency compared to classification approach.
3. **MediaPipe single-person limitation:** MediaPipe's built-in tracker is single-person only. Multi-person scenes require external person detection (YOLOX-Tiny), adding pipeline complexity.
4. **Occlusion sensitivity:** Keypoint estimation degrades when body parts are occluded by equipment, furniture, or other persons. Partial visibility can cause rule-based logic to fail.
5. **TFLite-to-ONNX conversion risk:** MediaPipe models require TFLite -> ONNX conversion for NPU deployment, which may introduce accuracy loss.

### Mitigation

- **Data expansion:** Collect 2K+ factory-specific fall simulation videos for threshold calibration and pose model fine-tuning.
- **Ensemble with classification** (see [Fall Classification](safety-fall_classification.md)): both models must agree for alert, significantly reducing false positives.
- **Temporal consistency** (80% of 30-frame window) filters transient keypoint errors.
- **Velocity check** (Rule 4) distinguishes active falls from persons already lying down.
- **RTMPose proven path:** RTMPose-S has reference code running on AX650N at 4.79ms -- avoids MediaPipe conversion risk.

### Contingency

- If g-pose underperforms (only 111 keypoint images): auto-annotation pipeline with ViTPose++ Huge generates pseudo-labels to expand the dataset. Additionally, g-classify (17K images) provides a fallback.
- If both approaches underperform: ensemble (both must agree) reduces false positives; temporal consistency (3-5 frames) further filters noise.
- Architecture fallback: if YOLOX license becomes an issue, switch to D-FINE-N (Apache 2.0) as person detector.

### Business Impact

**Scenario:** 10 falls/month in facility, ~20 person-down events/day.
- **Expected detection:** 8-9 falls detected correctly (88% recall)
- **Expected false alarms:** fewer than classification alone (pose geometry filters sitting/crouching)
- **Expected missed falls:** 1-2 per month
- **Annual false alarms:** significantly lower than classification-only approach

---

## Key Commands

```bash
# Pose estimation training
uv run core/p06_training/train.py --config features/safety-fall_pose_estimation/configs/06_training.yaml

# Evaluate
uv run core/p08_evaluation/evaluate.py --model runs/fall_pose_estimation/best.pt --config features/safety-fall_pose_estimation/configs/05_data.yaml --split test
```

---

## Model Card (Deliverables)

Each release produces a model card at `docs/model_cards/g_fall_detection.md` and a YAML card at `releases/fall_pose_estimation/v<N>/model_card.yaml`.

> **Note:** Fall pose estimation shares its model card with fall classification under the unified `g_fall_detection` model ID.

**Model artifacts:**

| Artifact | Format | Path |
|---|---|---|
| PyTorch model | `.pth` | `runs/fall_pose_estimation/best.pt` |
| ONNX model | `.onnx` | `runs/fall_pose_estimation/export/g_fallpose_yoloxt_640_v{N}.onnx` |
| Training config | `.yaml` | `features/safety-fall_pose_estimation/configs/06_training.yaml` |
| Metrics | `.json` | `runs/fall_pose_estimation/metrics.json` |

**Model card fields (auto-populated by `utils/release.py`):**

| Field | Source |
|---|---|
| Architecture, classes, input size | `06_training.yaml` |
| Dataset version (DVC tag) | `dvc tags list` |
| Training run path | `runs/fall_pose_estimation/` |
| mAP@0.5, Precision, Recall | `metrics.json` / `eval_results.json` |
| Per-class breakdown | Evaluation output |

---

## 9. Changelog

| Date | Version | Change |
|---|---|---|
| 2026-03-26 | 0.1 | Initial platform doc created from requirements and technical approach |

# Safety: Fall Classification
> ID: g | Owner: TBD | Phase: 1 | Status: training | Config: `features/safety-fall_classification/configs/06_training.yaml`
> See also: [Safety: Fall Pose Estimation](safety-fall_pose_estimation.md) — complementary pose-based approach

## 1. Customer Requirements

**Source:** [Web Japan Co., Ltd. - AI Camera Solutions](https://www.webjapan.co.jp/solution/ai-camera/) -- AI cameras for watching over employees and workers with automatic fall detection and alarm capabilities.

**Customer Specification:**

> "Watching over employees and workers. Detects falls and alarms."

**Explicit Requirements:**
- 24/7 monitoring of workers and elderly
- Automatic fall detection with alerts
- Supports "watching over" applications in care facilities and workplaces

## 2. Business Problem Statement

- Falls in factory environments can cause severe injury or death, creating direct risk to worker safety and operational continuity.
- Delayed emergency response after a fall significantly worsens medical outcomes -- every minute without intervention increases the severity of injuries.
- Unmonitored falls expose the organization to regulatory liability, workers' compensation claims, and reputational damage.
- Manual monitoring is not scalable -- factories operate across multiple zones and shifts where human supervisors cannot maintain constant visual coverage.
- Missing a fall (false negative) is unacceptable in life-safety applications; the operational impact of a false alarm is far lower than the consequences of a missed fall.

## 3. Technical Problem Statement

- **Worker fall safety -> distinguishing falls from normal postures:** Workers routinely sit, crouch, kneel, and bend during normal tasks. A system that cannot differentiate these from actual falls will produce excessive false alarms that desensitize operators, or suppress real alerts.
- **Emergency response time -> reliable real-time detection:** The system must detect falls within seconds and sustain detection across frames to avoid transient false triggers while still alerting quickly enough for effective response.
- **Regulatory liability -> eliminating missed falls:** Zero tolerance for false negatives means the system must maintain high recall even under challenging conditions (occlusion, poor lighting, unusual camera angles).
- **Scalable monitoring -> edge deployment:** The solution must run on low-power edge chips (AX650N/CV186AH) without cloud dependency, limiting available compute and memory.
- **False alarm fatigue -> temporal filtering:** Momentary detection glitches from lighting changes, passing objects, or partial occlusion must be filtered out without introducing dangerous delays in real fall alerts.

## 4. Technical Solution Options

### Option 1: YOLOX-M Single-Model Classification (Recommended)

**Approach:** Train a single YOLOX-M model to detect two classes -- `person` (upright/normal) and `fallen_person` (on the ground). Add ByteTrack for persistent person tracking across frames and a temporal filter requiring `fallen_person` to persist for 15 consecutive frames (500ms at 30 FPS) before alerting.

**Addresses:**
- Worker fall safety (posture confusion) -- mitigated by temporal filtering and hard negative mining (sitting/crouching/lying images labeled as `person`)
- Emergency response time -- single-model pipeline runs at 30-50 FPS on AX650N with sub-20ms latency
- Edge deployment -- YOLOX-M has excellent INT8 quantization properties, ~6-7 MB model size

**Pros:**
- Simplest pipeline: single model, no multi-stage coordination
- Fast inference and low deployment complexity
- 17K training images readily available
- Proven Apache 2.0 license (commercial use safe)

**Cons:**
- Cannot distinguish sitting/crouching from falling by body geometry alone -- relies on learned visual appearance
- Higher false positive rate than pose-based approach
- No depth information -- cannot distinguish elevated surface vs ground

### Option 2: Pose-Based Fall Detection (RTMPose-S)

**Approach:** Two-stage pipeline: (1) detect persons with YOLOX-Tiny, (2) estimate 17 body keypoints per person with RTMPose-S, (3) apply geometric rules on keypoint positions and orientations to determine fall state. See [Fall Pose Estimation](safety-fall_pose_estimation.md) for full details.

**Addresses:**
- Worker fall safety (posture confusion) -- keypoint geometry provides explicit body orientation, significantly reducing sitting/crouching false positives
- Eliminating missed falls -- geometric rules based on body center-of-mass height and torso angle are more robust than appearance-only classification
- False alarm fatigue -- 80% of 30-frame window temporal consistency provides strong noise suppression

**Pros:**
- Significantly lower false positive rate by analyzing body geometry rather than appearance alone
- Explicit reasoning about body orientation (torso angle, head-to-hip ratio) makes the system more interpretable
- Effective even with limited factory-specific training data (111 annotated images) due to COCO pretraining

**Cons:**
- Two-stage pipeline adds latency compared to single-model approach
- Only 111 factory-specific keypoint-annotated images -- heavily dependent on COCO pretraining generalization
- Rule-based fall detection from 2D keypoints is inherently noisy under heavy occlusion
- More complex deployment and maintenance

**Decision:** Start with Option 1 (YOLOX-M classification) for a quick baseline with lower complexity. If acceptance metrics are not met after v2 training, escalate to Option 2 or an ensemble of both approaches (both must agree for alert).

## 5. Approach

The classification approach treats fall detection as a standard object detection problem. A single YOLOX-M model is trained to distinguish `person` (upright/normal) from `fallen_person` (on the ground / in a fallen state). This is the simpler of two fall detection approaches (the other being pose estimation -- see [Fall Pose Estimation](safety-fall_pose_estimation.md)).

The classification pipeline:

```
Input (640x640)
    |
    v
  YOLOX-M -> detect "person" and "fallen_person"
    |
    v
  ByteTrack (track person states over time)
    |
    v
  Temporal filter: fallen_person sustained >= 15 frames (500ms)
    |
    v
  Alert: FALL_DETECTED
```

**Pros:** Simple pipeline, single model, fast inference, lower deployment complexity.
**Cons:** Can confuse sitting/crouching with falling. Higher false positive rate than pose-based approach.

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

## 6. Detection Classes

| Class ID | Class Name | Description |
|---|---|---|
| 0 | person | Normal upright person (standing, walking, sitting normally) |
| 1 | fallen_person | Person in fallen/lying position on the ground |

### Alert Logic

| Model | Min Confidence | Min Duration | Tracking Required |
|---|---|---|---|
| Fall Classification | 0.65 | 15 frames (500ms) | Yes |

The temporal filter requires `fallen_person` detections to persist for at least 15 consecutive frames (500ms at 30 FPS) before triggering an alert. This reduces false positives from momentary misclassifications.

---

## 7. Dataset

### Primary Training Data

The classification model trains on a 17K image dataset (`g_fall_classify`):

| Source | Images | Purpose |
|---|---|---|
| g_fall_classify dataset | 17K | Person / fallen_person classification |

Dataset path: `../../dataset_store/fall_classification/` with `{train,val,test}/{images,labels}/` structure. YOLO label format: `class_id cx cy w h` (normalized 0-1).

### Public Datasets (Reference / Expansion)

| Dataset | Type | Size | License | Quality | Notes |
|---|---|---|---|---|---|
| Le2i Fall Detection | Video | ~130 sequences | Academic | 4/5 | Contact Univ. of Burgundy; 4 fall scenarios |
| UR Fall Detection | Video | 70 sequences | Academic | 4/5 | 30 falls + 40 ADLs; RGB + Kinect depth |
| MPDD Fall Dataset | Images | 1,200+ | Academic | 4/5 | Multi-person falls; Nature Scientific Data 2025 |
| UP-Fall Detection | Video | 11 activities | Academic | 3/5 | Multi-modal; 17 young subjects (ages 18-24) |

### Custom Data Requirements

**Non-Fall Balance Data (to reduce sitting/lying false positives):**
- Sitting: 500-800 images
- Lying down: 300-500 images
- Kneeling/crouching: 500-700 images
- Standing/walking: 500-700 images
- Total non-fall: 1,800-2,700 images

**Staged Fall Collection (if expanding dataset):**

| Scenario | Sequences Needed | Priority |
|---|---|---|
| Forward falls | 50-80 | High |
| Backward falls | 50-80 | High |
| Sideways falls | 40-60 | Medium |
| Near-falls (stumble recovery) | 30-50 | Medium |
| Factory context falls | 20-30 | Medium |

### Annotation Guidelines

**Classes:**
- `person` (0): Normal upright/walking/standing person
- `fallen_person` (1): Person in fallen/lying position

**Rules:**
1. Annotate body orientation (horizontal vs vertical)
2. Exclude normal sitting, kneeling, crouching as `fallen_person` -- these are `person`
3. Include temporal context where available (use video sequences, not isolated frames)
4. Mark ambiguous cases (sitting that looks like falling) for expert review

**Quality Check:**
- Minimum 100 annotated fall sequences
- 20% expert review
- Verify class balance between person and fallen_person

---

## 8. Architecture

| Property | Value |
|---|---|
| Architecture | YOLOX-M (CSPDarknet + PAFPN + Decoupled Head) |
| Parameters | 25.3M (depth=0.67, width=0.75) |
| Input size | 640x640 |
| Classes | 2 (person, fallen_person) |
| Pretrained | `pretrained/yolox_m.pth` (COCO pretrained, Megvii) |
| License | Apache 2.0 |
| Tracker | ByteTrack (MIT) |

### Pipeline

1. **Detection:** YOLOX-M processes each frame, outputs bounding boxes for `person` and `fallen_person` with confidence scores.
2. **Tracking:** ByteTrack assigns persistent IDs to each detected person across frames.
3. **Temporal filtering:** A `fallen_person` detection must persist for >= 15 frames (500ms) for the same tracked person before triggering an alert.
4. **Alert:** `FALL_DETECTED` with person ID from tracker.

### Training Configuration

Training uses the pipeline config at `features/safety-fall_classification/configs/06_training.yaml`.

```bash
# Train
uv run core/p06_training/train.py --config features/safety-fall_classification/configs/06_training.yaml

# Evaluate
uv run core/p08_evaluation/evaluate.py \
  --model runs/fall_classification/best.pt \
  --config features/safety-fall_classification/configs/05_data.yaml \
  --split test --conf 0.25

# Export to ONNX
uv run core/p09_export/export.py \
  --model runs/fall_classification/best.pt \
  --training-config features/safety-fall_classification/configs/06_training.yaml \
  --export-config configs/_shared/09_export.yaml
```

### Model Card (Deliverables)

Each release produces a model card at `docs/model_cards/g_fall_detection.md` and a YAML card at `releases/fall_classification/v<N>/model_card.yaml`.

**Model artifacts:**

| Artifact | Format | Path |
|---|---|---|
| PyTorch model | `.pth` | `runs/fall_classification/best.pt` |
| ONNX model | `.onnx` | `runs/fall_classification/export/g_fall_yoloxm_{imgsz}_v{N}.onnx` |
| Training config | `.yaml` | `features/safety-fall_classification/configs/06_training.yaml` |
| Metrics | `.json` | `runs/fall_classification/metrics.json` |

**Model card fields (auto-populated by `utils/release.py`):**

| Field | Source |
|---|---|
| Architecture, classes, input size | `06_training.yaml` |
| Dataset version (DVC tag) | `dvc tags list` |
| Training run path | `runs/fall_classification/` |
| mAP@0.5, Precision, Recall | `metrics.json` / `eval_results.json` |
| Per-class breakdown | Evaluation output |

---

## 9. Training Results

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
| Classification (YOLOX: person/fallen) | 85-90% | Low | Medium (sitting ~ fallen) |
| 3D CNN + LSTM | 91% F1 | 120ms | ~3% |
| Le2i Baseline (Pose + temporal) | 83% AP | 50ms | ~5% |

---

## 10. Edge Deployment

### Target Hardware

| Chip | INT8 TOPS | Notes |
|---|---|---|
| AX650N | 18 INT8 TOPS | Primary target |
| CV186AH | 7.2 INT8 TOPS | Resource-constrained alternative |

### Deployment Artifacts

| Artifact | Format | Purpose |
|---|---|---|
| `g_classify_yoloxm_640_v{N}.pth` | PyTorch | Original weights |
| `g_classify_yoloxm_640_v{N}.onnx` | ONNX | Edge deployment |

### Expected Performance

| Property | Estimate |
|---|---|
| Model size (INT8) | ~6-7 MB |
| Inference latency (AX650N) | ~15-20ms |
| Expected FPS | 30-50 FPS |
| Power consumption | 2.5-5W (NPU) |

YOLOX-M has excellent INT8 quantization properties (pure CNN, no attention layers).

---

## 11. Limitations

### Known Issues

1. **Sitting/crouching confusion:** The classification approach can confuse sitting or crouching persons with fallen persons. The aspect ratio and body shape of a seated person can resemble a fallen person, leading to false positives.
2. **Lying down ambiguity:** Workers intentionally lying down (e.g., resting, working under equipment) may trigger false alarms.
3. **Single-model limitation:** Without pose keypoints, the model relies solely on visual appearance. Occlusion, unusual camera angles, or poor lighting degrade accuracy.
4. **No depth information:** Cannot distinguish between a person lying on an elevated surface vs on the ground.

### Mitigation

- **Temporal filtering** (15-frame persistence) reduces transient false positives.
- **Ensemble with pose estimation** (see [Fall Pose Estimation](safety-fall_pose_estimation.md)) significantly reduces false positives by requiring both models to agree.
- **Hard negative mining** during training: include sitting/crouching/lying-down images labeled as `person` to teach the model the distinction.

### Contingency

- If classification alone does not meet acceptance metrics, combine with the pose estimation approach in an ensemble (both must agree for alert).
- If YOLOX-M license becomes an issue, switch to D-FINE-S (Apache 2.0).
- Annotation quality is the primary bottleneck -- dataset quality audits are built into the development plan.

### Business Impact

**Scenario:** 10 falls/month in facility, ~20 person-down events/day.
- **Expected detection:** 8-9 falls detected correctly (88% recall)
- **Expected false alarms:** 1-2 per day (~365-730/year, acceptable for life safety)
- **Expected missed falls:** 1-2 per month

---

## Recommendation

**Phase 1:** Deploy classification model (YOLOX-M, 17K images) for quick baseline.
**Phase 2:** Add pose pipeline (D-FINE-N + RTMPose-S) for improved accuracy and lower FP rate.
**Phase 3 (if needed):** Temporal transformer on keypoint sequences for 97.6% accuracy.

Both approaches should run in parallel during Weeks 3-6, then the best performer is selected for production.

---

## 12. Changelog

| Date | Version | Change |
|---|---|---|
| 2026-03-26 | 0.1 | Initial platform doc created from requirements and technical approach |

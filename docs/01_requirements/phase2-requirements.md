# Phase 2 AI Model Requirements Specification
## Factory Smart Camera Safety System

**Project:** Factory Smart Camera AI System
**Customer:** Vietsol Factory Client
**Date:** March 5, 2026
**Version:** 1.0

---

## Document Structure

This document covers **Phase 2** of the Factory Smart Camera AI development. For Phase 1 requirements, see `../docs/phase1_requirements_spec.md`.

---

# SECTION A: CUSTOMER REQUIREMENTS

> **Source:** Customer input files and Phase 2 scope definition

## A.1 Project Scope

### Phase 2 Models (Planned Development)

The following models are planned for Phase 2 development:

| # | Model Name | Description |
|---|---|---|
| **c** | **PPE: Safety Glasses/Goggles** | Detection of safety glasses, goggles, and eye protection compliance |
| **d** | **PPE: Masks** | Detection of face masks and respirators (mask types: N95, surgical, respirator) |
| **e** | **PPE: Gloves** | Detection of safety gloves (various types: chemical-resistant, cut-resistant, insulated) |
| **j** | **PPE: Aprons** | Detection of protective aprons and body covering |
| **k** | **PPE: Safety Belt/Harness Usage** | Detection of proper safety belt/harness usage with smart harness integration |
| **l** | **Forklift/Pedestrian Proximity** | Detection of collision risk between forklifts and pedestrians |
| **m** | **Near-Miss Dangerous Behavior** | Detection of unsafe behaviors and near-miss incidents |

> **Note:** Phase 1 models (a, b, f, g, h, i) are documented in `../docs/phase1_requirements_spec.md`

---

## A.2 Model Requirements by Customer

### Model C: PPE - Safety Glasses/Goggles

**Customer Requirement:**
> "Detection of protective glasses, masks, gloves, and aprons. Details of the area are presented separately"
> Source: Customer presentation (Slide 3)

**Detection Classes:**

| Class ID | Class Name | Description |
|---|---|---|
| 0 | person | Full body detection for tracking |
| 1 | safety_glasses | Safety glasses or goggles present and worn correctly |
| 2 | no_safety_glasses | Face visible without eye protection |
| 3 | goggles | Full-face goggles (distinct from glasses) |
| 4 | face_shield | Full-face shield (for chemical handling areas) |

**Context from Customer Reference (PPE Levels):**
> Source: [Kurita Co., Ltd. - Chemical Area PPE Requirements](https://kcr.kurita.co.jp/solutions/videos/049.html)

| Area | Eye Protection Required | Detection Priority |
|---|---|---|
| **A** | Chemical Area (patrol/inspection) | Safety glasses (goggles type) |
| **B** | Chemical Receiving/Acceptance | Full-face shield |
| **C** | Chemical Handling (under-floor) | Full-face shield |
| **D** | Chemical Handling (above-floor) | Full-face shield |
| **E** | Emergency Response | Disaster-response full-face mask |

**Detection Challenges:**
- Small object detection (glasses are small compared to face)
- Distinguishing between glasses vs. goggles vs. face shield
- Reflections from lenses causing false negatives
- Differentiating clear/transparent glasses from no glasses

---

### Model D: PPE - Masks

**Customer Requirement:**
> "Detection of protective glasses, masks, gloves, and aprons."
> Source: Customer presentation (Slide 3)

**Detection Classes:**

| Class ID | Class Name | Description |
|---|---|---|
| 0 | person | Full body detection for tracking |
| 1 | mask_present | Face mask/surgical mask/ respirator detected |
| 2 | no_mask | Face visible without mask |
| 3 | n95_mask | N95 respirator mask |
| 4 | full_respirator | Full-face respirator |
| 5 | face_shield_with_mask | Face shield + mask combination |

**Mask Types to Detect (based on customer PPE levels):**

| Mask Type | Use Case | Priority |
|---|---|---|
| **Surgical Mask** | General protection, food handling | Medium |
| **N95 Respirator** | Dust protection, chemical handling | High |
| **Full-Face Respirator** | Chemical areas, emergency response | High |
| **Full-Face Shield** | Chemical receiving, handling (over mask) | High |

**Detection Challenges:**
- Masks covering most of face → facial features obscured
- Distinguishing mask types (similar appearance)
- Masks with exhalation valves (different appearance)
- Face shields → transparent and reflective
- Beard/hair obscuring mask fit check

---

### Model E: PPE - Gloves

**Customer Requirement:**
> "Detection of protective glasses, masks, gloves, and aprons."
> Source: Customer presentation (Slide 3)

**Detection Classes:**

| Class ID | Class Name | Description |
|---|---|---|
| 0 | person | Full body detection for tracking |
| 1 | gloves_present | Gloves detected on hands |
| 2 | no_gloves | Bare hands detected (no gloves) |
| 3 | chemical_resistant_gloves | Chemical-resistant gloves (thick, distinctive) |
| 4 | cut_resistant_gloves | Cut-resistant gloves (often metal mesh or coated) |
| 5 | insulated_gloves | Thermal/insulated gloves for heat protection |

**Glove Types to Detect (based on customer PPE levels):**

| Glove Type | Use Case | Color/Appearance | Detection Priority |
|---|---|---|---|
| **Chemical-Resistant** | Chemical handling areas | Often thick, rubber-like (green, blue, orange) | Critical |
| **Cut-Resistant** | Handling sharp objects, metal work | May have mesh coating, distinctive texture | High |
| **Insulated/Thermal** | Heat source areas | Thick, often leather or multi-layer | High |
| **General Safety** | General handling | Various colors (blue, white, black) | Medium |

**Detection Challenges:**
- Hands often in motion → motion blur
- Hands holding tools/materials → occlusion
- Small object (hands at distance)
- Gloves similar color as background or clothing
- Distinguishing glove types by appearance only
- Complex hand poses (fingers wrapped, gripping objects)

**Special Consideration:**
- Hand region detection first, then classify gloves on hand region
- May require higher input resolution (1280px) for distant workers
- Integration with hand tracking for improved accuracy

---

### Model J: PPE - Aprons

**Customer Requirement:**
> "Detection of protective glasses, masks, gloves, and aprons."
> Source: Customer presentation (Slide 3)

**Detection Classes:**

| Class ID | Class Name | Description |
|---|---|---|
| 0 | person | Full body detection for tracking |
| 1 | apron_present | Apron detected on torso |
| 2 | no_apron | Torso visible without apron |
| 3 | chemical_apron | Chemical-resistant apron (full coverage) |
| 4 | split_apron | Split-type apron (as mentioned in PPE Level D) |

**Apron Types to Detect (based on customer PPE levels):**

| Apron Type | Use Case | Coverage | Detection Priority |
|---|---|---|---|
| **Chemical-Resistant Apron** | Chemical handling (under-floor) | Full torso coverage | Critical |
| **Chemical-Resistant Apron (Split)** | Chemical handling (above-floor) | Upper/lower body | Critical |
| **General Protective Apron** | General handling, food processing | Partial torso coverage | Medium |

**Detection Challenges:**
- Apron shape varies (full, half, split-type)
- Different colors (white, blue, green, transparent)
- Similar appearance to regular clothing
- Apron may be partially obscured by worker's arms or equipment
- Workers wearing apron over other clothing (layering)

**Context from Customer Reference:**
- **Level C** (Chemical Handling - under-floor): Full-face shield + Chemical-resistant apron
- **Level D** (Chemical Handling - above-floor): Full-face shield + Chemical-resistant apron (split type)

---

### Model K: PPE - Safety Belt/Harness Usage

**Customer Requirement:**
> "Detection of non-use of safety belts at heights"
> "It even detects whether it is in use (whether it is hooked). Can high-altitude detection be detected in conjunction with smart safety belts?"
> Source: Customer presentation (Slide 5-6)

**Detection Classes:**

| Class ID | Class Name | Description |
|---|---|---|
| 0 | person | Full body detection for tracking |
| 1 | harness_worn | Safety harness detected on body |
| 2 | harness_not_worn | Person without safety harness |
| 3 | harness_hooked | Harness connected to anchor point (hooked) |
| 4 | harness_not_hooked | Harness not connected to anchor point |
| 5 | working_at_height | Person at height ≥1.5m |

**Customer References:**

**From Hitachi Solutions 2023 Press Release:**
> Source: [Hitachi Solutions - Smart Safety Harness](https://www.hitachi-solutions.co.jp/company/press/news/2023/0516.html)

**From Fujii Denko Smart Harness PDF:**
> Source: [Fujii Denko - Epron Smart Harness System](https://www.fujii-denko.co.jp/wp-content/themes/fujii-denko/images/product/harness/epron2/V1.0.5hex.pdf)

**Detection Focus:**

| Detection Type | Description | Alert Condition |
|---|---|---|
| **Wearing Detection** | Is harness on body? | Alarm if not fastened within 5 min of work start |
| **Hook Detection** | Is harness connected to anchor? | Immediate alarm if unhooked at height ≥1.5m |
| **Height Detection** | Is person at dangerous height? | Activate hook monitoring when ≥1.5m |
| **Zone Detection** | Is person in designated work zone? | Disable height detection in non-height zones |

**Smart Harness Functions (from customer references):**

1. **Harness Wearing Detection (Locks)**
   - Detects whether smart safety harness is worn correctly
   - Alarm if harness not fastened 5 minutes after work starts
   - Alarm if worker releases harness during work

2. **Height Detection Sensor**
   - Measures working height
   - Work above **1.5 m** is recognized as work at height
   - Activates additional safety protocols when threshold exceeded

3. **Hook Status Detection Sensors**
   - Dual hook monitoring (two attachment points required)
   - Both hooks must be properly attached when working at height
   - Immediate alarm if hooks become detached

4. **Contact Sensor (Non-Height Work Area)**
   - Configure non-height work zones
   - Disable height detection in these zones
   - Prevents false alarms in ground-level areas

5. **Emergency SOS Button**
   - Immediate supervisor notification
   - Real-time location tracking of workers

**Integration with Smart Harness Systems (Phase 2):**

**Combined Approach:**
- **Smart Harness** (worn by worker): Internal sensors (wearing, hook, height)
- **AI Camera** (ceiling/wall-mounted): Visual confirmation of harness usage
- **Data Fusion**: Cross-validate between sensor data and visual detection

**AI Camera Requirements:**

1. **Harness Detection**
   - Detect harness straps on worker's body (shoulders, chest, waist)
   - Distinguish harness types: full body harness vs. waist belt
   - Verify harness is worn (not just carried)

2. **Hook Detection**
   - Detect hook or carabiner connected to anchor point
   - Verify hook is properly engaged (not just resting on anchor)
   - Dual-hook systems: detect both hooks attached

3. **Height Estimation**
   - Estimate worker's height above ground/floor
   - Detect when worker crosses 1.5m threshold
   - Zone-based: Define "height work zones" vs. "ground-level zones"

4. **State Machine**
   ```
   Ground level → No hook monitoring required
   Height ≥1.5m → Activate hook monitoring
   Hook not attached → Immediate alert
   Harness not worn → Alert (if in hazardous area)
   ```

**Detection Challenges:**
- Harness straps may be thin and hard to see from distance
- Worker's body may partially obscure harness
- Different harness types (full body vs. waist belt)
- Determining if hook is actually supporting weight vs. just resting on anchor
- Height estimation from camera angle (perspective distortion)
- Distinguishing between "carrying harness" vs. "wearing harness"

---

### Model L: Forklift/Pedestrian Proximity

**Customer Requirement:**
> Detection of collision risk between forklifts and pedestrians in factory environments
> Based on customer safety needs and industry best practices

**Detection Classes:**

| Class ID | Class Name | Description |
|---|---|---|
| 0 | person | Pedestrian/worker detection |
| 1 | forklift | Forklift detection |
| 2 | pallet_jack | Pallet jack or manual lift |
| 3 | pedestrian_in_danger_zone | Person in forklift operating area |
| 4 | proximity_alert | Person and forklift too close (safety threshold violated) |

**Detection Focus:**

| Aspect | Description | Safety Threshold |
|---|---|---|
| **Detection** | Detect forklifts and people in same area | Any forklift + person in camera view |
| **Distance Estimation** | Calculate distance between person and forklift | Alert if < safety threshold |
| **Zone-Based** | Define forklift operating zones | Alert if person enters forklift zone |
| **Tracking** | Track IDs for multiple forklifts and people | Monitor proximity over time |

**Forklift Types to Detect:**

| Type | Description | Detection Priority |
|---|---|---|
| **Counterbalance Forklift** | Standard forklift with forks at rear | High |
| **Reach Truck** | Forks extend forward | Medium |
| **Pallet Jack** | Manual/Electric pallet mover | Medium |
| **Order Picker** | Worker elevated on forks | High |

**Safety Thresholds (Typical):**

| Context | Safe Distance | Alert Condition |
|---|---|---|
| **Static** | > 2-3 meters | Alert if closer for extended period |
| **Moving Forklift** | > 3-5 meters | Alert if forklift approaching person |
| **Person Walking** | > 2 meters | Alert if person walks toward forklift |
| **Blind Spot** | N/A | Alert if person enters forklift blind spot |

**Distance Estimation Methods:**

1. **Bounding Box IoU**
   - Calculate overlap between person and forklift bounding boxes
   - Alert if IoU > threshold (indicating too close)

2. **Centroid Distance**
   - Calculate distance between center points
   - Account for camera perspective (requires calibration)

3. **Zone-Based**
   - Define circular or rectangular zones around forklift
   - Alert if person centroid enters zone

**Detection Challenges:**

| Challenge | Description | Mitigation |
|---|---|---|
| **Occlusion** | Person or forklift partially hidden | Multi-camera fusion, tracking |
| **Perspective** | Distance varies with camera angle | Camera calibration, depth estimation |
| **Lighting** | Indoor/outdoor transitions, shadows | Low-light detection, IR mode support |
| **Multiple Workers** | Crowded scenes with many people | Multi-object tracking, ID persistence |
| **Forklift Variety** | Different types and sizes | Diverse training data, generic classes |
| **Speed** | Fast-moving forklifts | Higher FPS inference (20+ FPS) |

**Alert Logic:**

```python
# Example: Proximity alert logic
def check_proximity_alert(person_bbox, forklift_bbox, camera_calib):
    # Method 1: IoU-based
    iou = calculate_iou(person_bbox, forklift_bbox)
    if iou > 0.1:  # Threshold for "too close"
        return True, "IoU proximity violation"

    # Method 2: Centroid distance with perspective correction
    person_centroid = get_centroid(person_bbox)
    forklift_centroid = get_centroid(forklift_bbox)
    distance_px = euclidean_distance(person_centroid, forklift_centroid)
    distance_m = pixel_to_meters(distance_px, camera_calib,
                                     depth_at(person_centroid))

    if distance_m < SAFETY_THRESHOLD:
        return True, f"Distance: {distance_m:.1f}m < {SAFETY_THRESHOLD}m"

    return False, "Safe distance maintained"
```

---

### Model M: Near-Miss Dangerous Behavior

**Customer Requirement:**
> Detection of unsafe behaviors and near-miss incidents before they result in accidents
> Sources: Customer presentation + industry research

**Customer References:**
> - [Ken-IT World - AI Detects Near-Misses (2024)](https://ken-it.world/it/2024/12/ai-detects-near-misses.html)
> - [VIACT AI - Near-Miss Detection Solutions](https://www.viact.ai/video-analytics-solution/near-miss-detection)

**Detection Classes:**

| Class ID | Class Name | Description |
|---|---|---|
| 0 | unsafe_behavior | Generic unsafe behavior detected |
| 1 | near_miss | Near-miss incident (almost accident) |
| 2 | slip_trip_risk | Person slipping or tripping (but caught balance) |
| 3 | unsafe_lifting | Lifting with back bent, no knee bend |
| 4 | carrying_overweight | Carrying load too heavy for single person |
| 5 | working_under_load | Working under suspended load |
| 6 | unsafe_posture | Other unsafe body posture or positioning |

**Near-Miss vs. Accident:**

| Category | Definition | Example |
|---|---|---|
| **Near-Miss** | Almost caused an accident but avoided | Worker almost hit by falling object, but dodged |
| **Unsafe Behavior** | Action that could lead to accident | Worker climbing on equipment without fall protection |
| **Accident** | Actual incident occurred (not Phase 2 scope) | Worker fell, was injured |

**Detection Categories:**

#### 1. Slip/Trip Risk Detection

**Detection Focus:**
- Person stumbling but regaining balance
- Person slipping on wet/oily surface but not falling
- Person tripping over obstacle but catching themselves

**Detection Methods:**
- **Pose Estimation**: Detect sudden body position changes
- **Trajectory Analysis**: Detect unusual movement patterns (stumble, wobble)
- **Temporal Consistency**: Verify recovery (return to normal posture)

#### 2. Unsafe Lifting Detection

**Detection Focus:**
- Worker lifting with bent back (not using knees)
- Worker lifting load too heavy (bent over, struggling)
- Worker twisting while lifting

**Detection Methods:**
- **Pose Analysis**: Analyze body posture during lifting
- **Object Detection**: Detect boxes/objects being carried
- **Temporal Analysis**: Monitor lifting motion over time

**Safe Lifting Criteria:**
- Back straight (spine aligned)
- Knees bent (squatting, not bending at waist)
- Load close to body
- No twisting motion during lift

**Unsafe Lifting Indicators:**
- ❌ Bent at waist (back curved, knees straight)
- ❌ Holding load away from body
- ❌ Twisting torso while lifting
- ❌ Struggling to lift (load too heavy)

#### 3. Working Under Suspended Load

**Detection Focus:**
- Person standing/working under suspended object
- Forklift carrying load above person's head
- Crane operations with people underneath

**Detection Methods:**
- **Zone Definition**: Define danger zones under lifting operations
- **Object Detection**: Detect forklifts/cranes with suspended loads
- **Person Detection**: Alert if person in danger zone

#### 4. Carrying Overweight

**Detection Focus:**
- Single person carrying load that appears too heavy
- Person struggling or showing signs of strain
- Multiple people required for load (but only 1 person carrying)

**Detection Methods:**
- **Pose Analysis**: Detect body strain indicators
- **Size Estimation**: Estimate object size relative to person
- **Temporal Analysis**: Monitor movement (struggling vs. smooth)

**Indicators:**
- Leaning to one side (load imbalance)
- Exaggerated body sway while walking
- Stopping frequently to rest
- Load appears > 50% of person's body width

#### 5. Other Unsafe Behaviors

**Additional Behaviors to Detect:**
- Running in factory areas
- Climbing on equipment/machinery
- Walking while distracted (phone, conversation)
- Not using designated walkways
- Reaching into machinery while running

**Detection Challenges:**

| Challenge | Description | Mitigation |
|---|---|---|
| **Context Dependence** | "Running" may be OK in some areas | Zone-based rules |
| **Subtle Behaviors** | Unsafe lifting may look similar to safe lifting | Fine-grained pose analysis |
| **False Positives** | Normal movement may look unsafe | Temporal validation, confidence thresholds |
| **Lighting** | Poor lighting affects pose estimation | Low-light models, IR mode |
| **Crowded Scenes** | Multiple people interacting | Multi-person tracking |

---

# SECTION B: TECHNICAL TEAM DEFINITIONS

> **Source:** Technical team recommendations based on customer requirements, industry standards, and hardware constraints

## B.1 Model Architecture Selection (Phase 2)

### Chosen Architectures

| Model | Recommended Architecture | Rationale |
|---|---|---|
| Safety Glasses/Goggles | YOLO26s or YOLO11s | Small object detection; may need 1280px input |
| Masks | YOLO26s | Face region detection; handle partial occlusion |
| Gloves | YOLO26s + Region Proposal | Hand region detection + classification |
| Aprons | YOLO11s | Torso region detection; distinguish from clothing |
| Harness Usage | YOLO11s-Pose + Custom | Pose for hook detection + classification |
| Forklift Proximity | YOLO11s + Tracking | Multi-object tracking + distance estimation |
| Near-Miss Behavior | YOLO11s-Pose + Custom | Pose-based behavior analysis |

**Hardware Constraint:** All models must be deployable on edge AI devices with power consumption < 12W (specific hardware TBD in separate document)

---

## B.2 Detection Class Definitions (Phase 2)

### Model C: Safety Glasses/Goggles

| Class ID | Class Name | Description |
|---|---|---|
| 0 | person | Full body detection for tracking |
| 1 | safety_glasses | Safety glasses or goggles present and worn correctly |
| 2 | no_safety_glasses | Face visible without eye protection |
| 3 | goggles | Full-face goggles (distinct from glasses) |
| 4 | face_shield | Full-face shield (for chemical handling areas) |

### Model D: Masks

| Class ID | Class Name | Description |
|---|---|---|
| 0 | person | Full body detection for tracking |
| 1 | mask_present | Face mask/surgical mask/ respirator detected |
| 2 | no_mask | Face visible without mask |
| 3 | n95_mask | N95 respirator mask |
| 4 | full_respirator | Full-face respirator |
| 5 | face_shield_with_mask | Face shield + mask combination |

### Model E: Gloves

| Class ID | Class Name | Description |
|---|---|---|
| 0 | person | Full body detection for tracking |
| 1 | gloves_present | Gloves detected on hands |
| 2 | no_gloves | Bare hands detected (no gloves) |
| 3 | chemical_resistant_gloves | Chemical-resistant gloves (thick, distinctive) |
| 4 | cut_resistant_gloves | Cut-resistant gloves (often metal mesh or coated) |
| 5 | insulated_gloves | Thermal/insulated gloves for heat protection |

### Model J: Aprons

| Class ID | Class Name | Description |
|---|---|---|
| 0 | person | Full body detection for tracking |
| 1 | apron_present | Apron detected on torso |
| 2 | no_apron | Torso visible without apron |
| 3 | chemical_apron | Chemical-resistant apron (full coverage) |
| 4 | split_apron | Split-type apron (as mentioned in PPE Level D) |

### Model K: Safety Belt/Harness Usage

| Class ID | Class Name | Description |
|---|---|---|
| 0 | person | Full body detection for tracking |
| 1 | harness_worn | Safety harness detected on body |
| 2 | harness_not_worn | Person without safety harness |
| 3 | harness_hooked | Harness connected to anchor point (hooked) |
| 4 | harness_not_hooked | Harness not connected to anchor point |
| 5 | working_at_height | Person at height ≥1.5m |

### Model L: Forklift/Pedestrian Proximity

| Class ID | Class Name | Description |
|---|---|---|
| 0 | person | Pedestrian/worker detection |
| 1 | forklift | Forklift detection |
| 2 | pallet_jack | Pallet jack or manual lift |
| 3 | pedestrian_in_danger_zone | Person in forklift operating area |
| 4 | proximity_alert | Person and forklift too close (safety threshold violated) |

### Model M: Near-Miss Dangerous Behavior

| Class ID | Class Name | Description |
|---|---|---|
| 0 | unsafe_behavior | Generic unsafe behavior detected |
| 1 | near_miss | Near-miss incident (almost accident) |
| 2 | slip_trip_risk | Person slipping or tripping (but caught balance) |
| 3 | unsafe_lifting | Lifting with back bent, no knee bend |
| 4 | carrying_overweight | Carrying load too heavy for single person |
| 5 | working_under_load | Working under suspended load |
| 6 | unsafe_posture | Other unsafe body posture or positioning |

---

## B.3 Performance Targets (Phase 2)

### Accuracy Metrics

| Model | Target mAP@0.5 | Minimum Acceptable | Technical Note |
|---|---|---|---|
| Safety Glasses/Goggles | ≥ 0.80 | 0.70 | Small object detection (challenging) |
| Masks | ≥ 0.85 | 0.75 | Face partially covered |
| Gloves | ≥ 0.80 | 0.70 | Hands often occluded |
| Aprons | ≥ 0.82 | 0.72 | May blend with clothing |
| Harness Usage | ≥ 0.85 | 0.75 | Complex (wearing + hooked + height) |
| Forklift Proximity | ≥ 0.85 | 0.75 | Requires accurate distance estimation |
| Near-Miss Behavior | ≥ 0.75 | 0.65 | Complex behavioral analysis |

### Inference Performance

| Metric | Target | Rationale |
|---|---|---|
| Frame Rate | ≥ 15 FPS | Standard for safety monitoring |
| Multi-Model | ≥ 2 models @ ≥ 10 FPS each | Some cameras may need 3+ models |
| Alert Latency | < 2 seconds | Near-miss detection requires fast response |
| Power Consumption | < 12W | Must meet same power constraint as Phase 1 |

---

## B.4 Input Resolution Strategy (Phase 2)

| Scenario | Resolution | When to Use |
|---|---|---|
| Standard detection | 640 × 640 | Default for most models |
| Small/distant objects | 1280 × 1280 | Glasses, gloves at distance |
| Face/Hand regions | 640 × 640 (cropped) | Focus on specific body regions |
| Fast motion | 640 × 640 | Near-miss, moving forklifts |

---

## B.5 Specialized Detection Logic (Phase 2)

### Harness Usage Detection Logic

```python
def detect_harness_violation(person_pose, harness_detections, height_estimate):
    """
    Complete harness usage violation detection
    Returns: (violation_type, confidence, should_alert)
    """

    # 1. Check if harness is worn
    harness_on_body = detect_harness_on_body(person_pose)
    if not harness_on_body:
        if is_in_hazardous_area(person_pose['location']):
            return "HARNESS_NOT_WORN", 0.9, True

    # 2. Check height
    if height_estimate >= 1.5:  # At height
        # Check if harness is hooked
        if not is_hooked_to_anchor(harness_detections):
            return "HARNESS_NOT_HOOKED_AT_HEIGHT", 0.95, True
    else:
        # On ground level - hook status not required
        return "NO_VIOLATION", 0.0, False

    return "NO_VIOLATION", 0.0, False
```

### Forklift Proximity Detection Logic

```python
def check_forklift_proximity(persons, forklifts, camera_calib):
    """
    Check all person-forklift pairs for proximity violations
    """
    violations = []

    for person in persons:
        for forklift in forklifts:
            # Method 1: Zone-based
            if is_person_in_forklift_zone(person['centroid']):
                violations.append({
                    'type': 'person_in_forklift_zone',
                    'person_id': person['id'],
                    'forklift_id': forklift['id'],
                    'confidence': 0.9
                })
                continue

            # Method 2: Distance-based
            distance = calculate_distance(
                person['centroid'],
                forklift['centroid'],
                camera_calib
            )

            if distance < SAFETY_DISTANCE:
                violations.append({
                    'type': 'proximity_violation',
                    'person_id': person['id'],
                    'forklift_id': forklift['id'],
                    'distance_m': distance,
                    'confidence': 0.85
                })

    return violations
```

### Near-Miss Detection Logic

```python
def detect_slip_trip(person_trajectory, pose_keypoints):
    """
    Detect slip/trip events: person stumbles but recovers
    """
    # Sudden drop in hip height (stumble)
    hip_y_current = pose_keypoints['left_hip'][1]
    hip_y_previous = pose_keypoints['left_hip'][1].t_minus_1

    hip_drop = hip_y_previous - hip_y_current
    if hip_drop > 0.1:  # Sudden 10% drop
        # Check recovery - did they return to normal within 1 second?
        hip_y_future = pose_keypoints['left_hip'][1].t_plus_30
        if hip_y_future >= hip_y_current - 0.05:  # Recovered
            return "SLIP_TRIP_RECOVERED", 0.8

    return "NO_SLIP_TRIP", 0.0


def detect_unsafe_lifting(person_pose, object_bbox):
    """
    Detect unsafe lifting: bent back, not using knees
    """
    # Extract relevant keypoints
    left_shoulder = pose_keypoints['left_shoulder']
    right_shoulder = pose_keypoints['right_shoulder']
    left_hip = pose_keypoints['left_hip']
    right_hip = pose_keypoints['right_hip']
    left_knee = pose_keypoints['left_knee']
    right_knee = pose_keypoints['right_knee']

    # Check if knees are bent (knees lower than hips)
    knees_bent = (left_knee[1] > left_hip[1] or
                  right_knee[1] > right_hip[1])

    if not knees_bent:
        # Knees not bent - lifting with straight legs
        back_angle = calculate_torso_angle(left_shoulder, left_hip)
        if back_angle > 45:  # Bent over
            return "UNSAFE_LIFTING_NO_KNEES", 0.85

    # Check if back is curved (not straight)
    spinal_alignment = calculate_spinal_curvature(left_shoulder, right_shoulder, left_hip, right_hip)
    if spinal_alignment > CURVATURE_THRESHOLD:
        return "UNSAFE_LIFTING_CURVED_BACK", 0.75

    return "SAFE_LIFTING", 0.0
```

---

## B.6 Hardware Requirements (Phase 2)

### Power Constraint (Same as Phase 1)

**Maximum Power Consumption:** < 12W per edge device

All Phase 2 models must meet the same < 12W power constraint as Phase 1 models.

### Model Complexity Considerations

| Model | Computational Demand | Optimization Priority |
|---|---|---|
| Safety Glasses | Low-Medium | Small object focus |
| Masks | Medium | Face region analysis |
| Gloves | Medium-High | Hand region detection |
| Aprons | Low-Medium | Torso region analysis |
| Harness Usage | High | Pose + classification + height |
| Forklift Proximity | Medium | Detection + tracking |
| Near-Miss | High | Pose + temporal analysis |

### Multi-Model Deployment Strategy

For cameras requiring 3+ models simultaneously:
- Use model pruning to reduce compute requirements
- Prioritize models by safety criticality
- Consider zone-based model activation (only run models when relevant)
- Frame skipping where appropriate (e.g., check every 2nd frame for non-critical models)

---

## B.7 Training Data Requirements (Phase 2)

### Dataset Size Estimates

| Model | Minimum Images | Open Datasets Available | Custom Data Needed | Priority |
|---|---|---|---|---|
| Safety Glasses | 3,000-5,000 | Part of broader PPE dataset | Factory-specific eye protection | High |
| Masks | 3,000-5,000 | Part of broader PPE dataset | Various mask types | High |
| Gloves | 4,000-6,000 | Construction-PPE (partial) | Factory-specific gloves | High |
| Aprons | 2,000-3,000 | Limited open data | Factory aprons | Medium |
| Harness Usage | 1,500-3,000 | None (highly specialized) | Staged harness usage | High |
| Forklift Proximity | 5,000-8,000 | Industrial Safety (partial) | Factory forklift footage | High |
| Near-Miss Behavior | 2,000-3,000 | None (highly specialized) | Staged unsafe behaviors | Medium |

### Data Collection Strategy

**Week 1-2: Foundation Data**
1. Extend Phase 1 PPE dataset to include glasses, masks, gloves, aprons
2. Begin collecting factory footage for forklift operations
3. Stage harness usage scenarios (with harness if available)

**Week 3-4: Specialized Data**
4. Collect specialized apron images (chemical handling contexts)
5. Record forklift operations in various factory areas
6. Stage near-miss scenarios (with proper safety precautions)

**Week 5-6: Validation Data**
7. Collect test footage from actual factory operations
8. Validate models on held-out test sets
9. Error analysis and additional data collection as needed

---

## B.8 Development Timeline (Phase 2)

### Schedule

| Week | Phase | Activities |
|---|---|---|
| **1-2** | Data Preparation | Extend Phase 1 datasets, collect Phase 2 data, setup annotation |
| **3-4** | Model Training | Train Safety Glasses, Masks, Gloves models |
| **5** | Model Training | Train Aprons, Harness Usage models |
| **6** | Model Training | Train Forklift Proximity, Near-Miss models |
| **7** | Integration | Multi-model testing, optimization |
| **8** | Validation | ONNX export, hardware testing, customer demo |

**Resource Estimation:**
- GPU Training Time: ~50 hours (increased due to more models)
- GPU Cost: $0 (local GPU)
- Annotation Time: ~120 hours (more specialized data needed)
- Hardware Cost: TBD (Edge device selection pending; target: < 12W)

---

## B.9 Acceptance Criteria (Phase 2)

### Model Acceptance

Each Phase 2 model must meet ALL criteria:

1. ✅ **Accuracy**: mAP@0.5 ≥ target value
2. ✅ **Performance**: Inference ≥ 15 FPS on < 12W edge hardware
3. ✅ **Power**: Operates within < 12W power budget (measured under full load)
4. ✅ **Robustness**: Works in low-light (0.16 lux), IR mode
5. ✅ **False Positive Rate**: Below specified threshold
6. ✅ **Testing**: Validated on held-out test set from actual factory footage

### System Acceptance

1. ✅ **Latency**: Event detection → alert < 2 seconds (near-miss needs fast response)
2. ✅ **Multi-Model**: At least 3 models run simultaneously at ≥ 8 FPS on < 12W hardware
3. ✅ **Power**: Total edge device power consumption < 12W under multi-model load
4. ✅ **Reliability**: 24/7 operation for 1 week without manual intervention
5. ✅ **Thermal**: Stable operation within factory temperature range (0-45°C)
6. ✅ **Alert Quality**: < 5 false alarms per day per camera
7. ✅ **User Interface**: Dashboard displays all alerts with video context

---

## B.10 Open Questions (Phase 2 - Requiring Customer Clarification)

### Model-Specific Questions

1. **Safety Glasses/Goggles**
   - [ ] Are all workers required to wear eye protection, or only in certain zones?
   - [ ] Should we distinguish between different types of glasses (reading glasses vs. safety glasses)?
   - [ ] Are transparent safety glasses difficult to distinguish from no glasses?

2. **Masks**
   - [ ] What types of masks are used? (N95, surgical, cloth, respirator?)
   - [ ] Are masks required in all areas or only chemical handling zones?
   - [ ] Should we detect mask wearing correctly (covering nose and mouth)?

3. **Gloves**
   - [ ] What glove types are most critical to detect?
   - [ ] Are there specific colors that indicate safety gloves?
   - [ ] How do we handle workers carrying items (which may obscure gloves)?

4. **Aprons**
   - [ ] Are aprons required in all production areas or only chemical handling?
   - [ ] Should we detect apron condition (torn, dirty, contaminated)?

5. **Harness Usage**
   - [ ] What height threshold should be used? (Customer ref says 1.5m - confirm this is correct)
   - [ ] Are there smart harness systems already deployed or planned?
   - [ ] Should we integrate with existing harness IoT sensors?

6. **Forklift Proximity**
   - [ ] What is the specific safety distance requirement? (2m, 3m, 5m?)
   - [ ] Should alerts differ by forklift speed (moving vs. stationary)?
   - [ ] Are there designated pedestrian-only vs. forklift-only zones?

7. **Near-Miss Behavior**
   - [ ] What specific behaviors are most critical to detect?
   - [ ] Should near-miss events trigger immediate alerts or logged for review?
   - [ ] How should detected near-misses be used for training and improvement?

### System Questions

8. **Alert Priorities**
   - [ ] How should we prioritize alerts when multiple models trigger?
   - [ ] Should critical alerts override lower-priority ones?

9. **Data Privacy**
   - [ ] Are there any restrictions on collecting/staging unsafe behavior videos?
   - [] Must faces be blurred in logs and dashboards?

10. **Integration**
    - [ ] Will Phase 2 models integrate with Phase 1 deployed systems?
    - [ ] What dashboard technology is preferred?

---

## Appendix

### A. Customer Reference Materials (Phase 2)

- PPE Levels (Chemical Handling): [Kurita KCR](https://kcr.kurita.co.jp/solutions/videos/049.html)
- Smart Harness: [Hitachi Solutions 2023](https://www.hitachi-solutions.co.jp/company/press/news/2023/0516.html)
- Smart Harness PDF: [Fujii Denko](https://www.fujii-denko.co.jp/wp-content/themes/fujii-denko/images/product/harness/epron2/V1.0.5hex.pdf)
- Near-Miss Detection: [Ken-IT World](https://ken-it.world/it/2024/12/ai-detects-near-misses.html)
- Near-Miss Detection: [VIACT AI](https://www.viact.ai/video-analytics-solution/near-miss-detection)

### B. Related Documents

- `../docs/phase1_requirements_spec.md` — Phase 1 requirements (Models a, b, f, g, h, i)
- `docs/tool_stack.md` — Complete tool stack overview
- `../docs/phase1_development_plan.md` — 12-week timeline, metrics, validation
- `docs/labeling_guide.md` — Label Studio setup, annotation rules
- `docs/error_analysis_guide.md` — FiftyOne/Cleanlab workflows

---

**Document Version**: 1.0
**Last Updated:** March 5, 2026
**Prepared by:** Vietsol AI Technical Team

**Document Status:**
- ✅ Section A: Customer Requirements — VERIFIED from customer input files
- ⚠️ Section B: Technical Team Definitions — REQUIRES CUSTOMER VERIFICATION AND ALIGNMENT

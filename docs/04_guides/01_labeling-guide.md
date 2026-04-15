# Labeling Guide — Factory Smart Camera
## Using Label Studio for Object Detection & Pose Annotation

**Project:** Factory Safety AI Models
**Tool:** Label Studio (self-hosted, open source)
**Annotation Format:** YOLO (detection), COCO JSON (pose)

---

## Table of Contents

1. [Label Studio Setup](#1-label-studio-setup)
2. [Project Setup](#2-project-setup)
3. [Per-Model Labeling Requirements](#3-per-model-labeling-requirements) — what to label, how, quantity, pre-annotation approach
4. [Labeling Configuration](#4-labeling-configuration) — Label Studio XML configs
5. [Annotation Guidelines](#5-annotation-guidelines) — general rules, PPE, fire, pose
6. [Pre-Annotation Stack](#6-pre-annotation-stack-sam-3--rf-detr--vitpose) — SAM 3 + RF-DETR + MoveNet
7. [Export Settings](#7-export-settings) — YOLO/COCO format, class ID mappings
8. [Annotation Workflow & QA](#8-annotation-workflow--qa) — review process, quality metrics
9. [Dataset Split Script](#9-dataset-split-script)
10. [YOLO Dataset YAML](#10-yolo-dataset-yaml-for-training)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Label Studio Setup

### 1.1 Installation

```bash
# Option 1: uv (recommended — fast, isolated)
uv tool install label-studio
label-studio start --port 8080

# Or run directly without installing globally:
uvx label-studio start --port 8080

# Option 2: uv with virtual environment (project-level)
uv venv .venv
source .venv/bin/activate
uv pip install label-studio
label-studio start --port 8080

# Option 3: Docker (recommended for team/server use)
docker run -it -p 8080:8080 \
  -v $(pwd)/label-studio-data:/label-studio/data \
  heartexlabs/label-studio:latest
```

Access at: `http://localhost:8080`

### 1.2 Create Account & Organization

1. Open `http://localhost:8080` in browser
2. Create admin account
3. Create organization (e.g., "Vietsol-SmartCamera")
4. Invite annotators via Settings > Members

---

## 2. Project Setup

### 2.1 Create Projects (one per use case)

Create the following projects in Label Studio:

| Project Name | Task Type | Description |
|---|---|---|
| `PPE-Helmet` | Object Detection | Helmet detection (Phase 1 — Model B) |
| `PPE-SafetyShoes` | Object Detection | Safety shoes detection (Phase 1 — Model F) |
| `Poketenashi` | Object Detection | Behavior violations (Phase 1 — Model H) |
| `Fire-Smoke` | Object Detection | Fire and smoke detection (Phase 1 — Model A) |
| `Fall-Pose` | Keypoint / Pose | 17-point COCO pose annotation (Phase 1 — Model G) |
| `PPE-Detection` | Object Detection | Helmet, vest, gloves, goggles detection (Phase 2 — full PPE) |
| `Forklift-Proximity` | Object Detection | Forklift, person, pallet jack detection (Phase 2) |

### 2.2 Import Data

1. Go to project > Settings > Cloud Storage (or drag-drop upload)
2. Supported formats: JPG, PNG, MP4 (video frames)
3. For video: extract frames first using:

```bash
# Extract 1 frame per second from factory footage
ffmpeg -i factory_cam01.mp4 -vf "fps=1" -q:v 2 output/cam01_frame_%06d.jpg
```

---

## 3. Per-Model Labeling Requirements

### 3.1 Overview: What Needs Labeling

Not all models need custom labeling. Some use pretrained weights, pose rules, or public datasets that are already labeled.

| Model | What to Label | Annotation Type | Public Data (ready) | Custom Labeling Needed | Priority | Pre-Annotation Approach |
|---|---|---|---|---|---|---|
| **a** Fire | fire, smoke | Bbox detection | 122,525 images | 100-200 factory-specific images | LOW | RF-DETR-L (detect fire/smoke bboxes) + SAM 3 (text: "fire", "smoke") |
| **b** Helmet | person, helmet, no_helmet, nitto_hat | Bbox detection | 62,602 images | ~2,500 nitto_hat images | MEDIUM | RF-DETR-L (detect person/helmet) + SAM 3 (text: "helmet", "hard hat", "head") |
| **f** Shoes | person, safety_shoes, no_safety_shoes | Bbox detection | 3,772 images | 2,000+ factory shoe images | **HIGH** | SAM 3 (text: "shoes", "feet", "safety shoes") — small objects benefit from SAM 3 segmentation |
| **g-pose** Fall Pose | person (17 COCO keypoints) | Keypoint / pose | 111 keypoint images + COCO 58K | ~1,000 verified auto-annotated | MEDIUM | **MoveNet Huge** auto-annotates 17 keypoints on bbox-only fall images → AT reviews |
| **g-classify** Fall Classify | person, fallen_person | Bbox detection | 17,383 images | Minimal (100-200 factory) | LOW | RF-DETR-L (detect persons) → manual class assignment (fallen vs standing) |
| **h** Poketenashi (phone) | phone | Bbox detection | ~13,470 images (FPI-Det + Roboflow) | 200-400 factory phone images | LOW | RF-DETR-L (detect phone bboxes) + COCO pretrained "cell phone" class |
| **h** Poketenashi (pose rules) | — | **No labeling needed** | COCO-Pose pretrained | 100-200 calibration images (threshold tuning only, no annotation) | — | Pretrained MoveNet (Apache 2.0) — only threshold calibration |
| **i** Zone Intrusion | — | **No labeling needed** | COCO pretrained (person) | None | — | Pretrained YOLOX-T (Apache 2.0) — only zone polygon configuration |

**Total Custom Labeling Effort:**

| Data Type | Quantity | Estimated Time (with pre-annotation) | Estimated Time (manual) |
|---|---|---|---|
| Nitto hat (helmet model) | ~2,500 images | 15-20 hours | 60-80 hours |
| Safety shoes (factory) | 2,000+ images | 12-15 hours | 50-65 hours |
| Fall pose (auto-annotated, review only) | ~1,000 images | 8-12 hours | 40-50 hours |
| Factory-specific (fire, phone, fall classify) | 400-800 images | 3-5 hours | 10-20 hours |
| **Total** | **~6,000 images** | **~40-50 hours** | **~160-215 hours** |

> Pre-annotation (RF-DETR-L + SAM 3 + MoveNet) reduces labeling effort by **70-85%**.

---

### 3.2 Per-Model Labeling Details

#### 3.2.1 Model A: Fire Detection

**What to label:** Bounding boxes around fire and smoke regions.

**Classes:**

| Class | Description | How to Label |
|---|---|---|
| `fire` | Visible open flames | Tight bbox around flame area only (not surrounding glow) |
| `smoke` | Combustion smoke | Bbox around densest smoke concentration |

**Quantity:** 122K public images ready. Custom: 100-200 factory-specific images (outdoor areas, controlled fire tests).

**Pre-annotation approach:**
1. Run **RF-DETR-L** on factory images → auto-detect fire/smoke bboxes
2. Run **SAM 3** with text prompts "fire" and "smoke" → segment irregular flame/smoke shapes → convert to bboxes
3. Import pre-annotations into Label Studio → AT reviews and corrects (~15% manual effort)

**Common mistakes to avoid:**
- Do NOT label steam, dust, or fog as smoke
- Do NOT include surrounding glow — only actual visible flames
- Label thin smoke wisps only if clearly visible combustion smoke

---

#### 3.2.2 Model B: Helmet Detection

**What to label:** Bounding boxes around persons and their head PPE status.

**Classes:**

| Class | Description | How to Label |
|---|---|---|
| `person` | Full body of each worker | Tight bbox head-to-feet (or visible portion) |
| `helmet` | Head wearing a hard hat/safety helmet | Tight bbox around head area with helmet |
| `no_helmet` | Head clearly visible without helmet | Tight bbox around bare head area |
| `nitto_hat` | Head wearing Nitto soft safety hat (Japanese style) | Tight bbox around head with Nitto hat — **custom class, no public data** |

**Quantity:** 62K public images ready. Custom: ~2,500 nitto_hat images from factory collection.

**Pre-annotation approach:**
1. Run **RF-DETR-L** on factory images → auto-detect person bboxes
2. Run **SAM 3** with text prompts "helmet", "hard hat", "head" → detect head regions
3. Import into Label Studio → AT assigns class (helmet/no_helmet/nitto_hat)
4. **Nitto hat images:** SAM 3 may not recognize nitto hats — AT must manually label these. Collect from factory footage with workers wearing nitto hats.

**Common mistakes to avoid:**
- If head is not visible (turned away, fully occluded), do NOT label
- Baseball caps and beanies = `no_helmet` (not safety equipment)
- Nitto hats look like soft caps — distinct from hard helmets

---

#### 3.2.3 Model F: Safety Shoes Detection

**What to label:** Bounding boxes around persons and their shoe PPE status. **Highest priority for custom labeling** — only 3.7K public images.

**Classes:**

| Class | Description | How to Label |
|---|---|---|
| `person` | Full body of each worker | Tight bbox head-to-feet |
| `safety_shoes` | Feet wearing safety shoes (steel toe, etc.) | Tight bbox around shoe/foot area — both feet if visible |
| `no_safety_shoes` | Feet wearing non-safety footwear (sneakers, sandals) | Tight bbox around shoe/foot area |

**Quantity:** 3.7K public images ready. Custom: **2,000+ factory shoe images** (critical gap).

**Pre-annotation approach:**
1. Run **SAM 3** with text prompts "shoes", "feet", "safety shoes" → segment foot regions (SAM 3 handles small objects better than RF-DETR for this task)
2. Convert SAM 3 masks to bboxes
3. Import into Label Studio → AT assigns class (safety_shoes/no_safety_shoes)
4. **Focus on diversity:** different shoe types, angles, lighting, occlusion levels

**Common mistakes to avoid:**
- Shoes may be partially occluded by machinery/tables — label if >30% visible
- Safety shoes vs regular shoes can be subtle — annotator training needed on shoe types
- Label at **1280px resolution** — shoe details are lost at 640px

---

#### 3.2.4 Model G-Pose: Fall Detection (Pose)

**What to label:** 17 COCO keypoints per person (pose estimation).

**Keypoints (COCO format):**

| Index | Keypoint | Index | Keypoint |
|---|---|---|---|
| 0 | nose | 9 | left_wrist |
| 1 | left_eye | 10 | right_wrist |
| 2 | right_eye | 11 | left_hip |
| 3 | left_ear | 12 | right_hip |
| 4 | right_ear | 13 | left_knee |
| 5 | left_shoulder | 14 | right_knee |
| 6 | right_shoulder | 15 | left_ankle |
| 7 | left_elbow | 16 | right_ankle |
| 8 | right_elbow | | |

**Quantity:** 111 keypoint-annotated fall images + COCO 58K pretrained. Custom: ~1,000 auto-annotated images needing verification.

**Pre-annotation approach (auto-annotation pipeline):**
1. Take 11K bbox-only fall detection images (from public datasets)
2. Run **MoveNet Huge** on each image → generates 17 keypoints per person automatically
3. MoveNet is more accurate than YOLO-Pose for offline annotation — produces high-quality pseudo-labels
4. Import auto-annotations into Label Studio → AT **reviews and corrects** (not annotates from scratch)
5. Target: 1,000 verified images after review

**Common mistakes to avoid:**
- Occluded keypoints: mark as "not visible" (visibility=0), do NOT guess positions
- Fallen persons may have unusual poses — ensure hip/shoulder keypoints are correct (critical for fall rule: hip_y >= shoulder_y)
- Workers in bulky clothing: keypoints may shift — annotate actual joint position, not clothing edge

---

#### 3.2.5 Model G-Classify: Fall Detection (Classification)

**What to label:** Bounding boxes around persons, classified as standing or fallen.

**Classes:**

| Class | Description | How to Label |
|---|---|---|
| `person` | Standing/walking/sitting person (normal) | Tight bbox around full body |
| `fallen_person` | Person lying on ground (fall event) | Tight bbox around full body on ground |

**Quantity:** 17K public images ready. Custom: 100-200 factory-specific images.

**Pre-annotation approach:**
1. Run **RF-DETR-L** on images → auto-detect all person bboxes
2. Import into Label Studio → AT assigns class: `person` (upright) or `fallen_person` (on ground)
3. Use aspect ratio as hint: fallen persons have wider-than-tall bboxes

**Common mistakes to avoid:**
- Person bending down to pick something up ≠ `fallen_person` (must be lying/collapsed)
- Person sitting on chair ≠ `fallen_person`
- If ambiguous (crouching, kneeling), label as `person`

---

#### 3.2.6 Model H: Poketenashi (Phone Detection Only)

**What to label:** Bounding boxes around phones only. Other behaviors (hands-in-pockets, no-handrail) use **pretrained pose models + rules — NO labeling needed**.

**Classes:**

| Class | Description | How to Label |
|---|---|---|
| `phone` | Mobile phone visible in hand or near face | Tight bbox around the phone object |

**Quantity:** ~13K public images ready (FPI-Det + Roboflow). Custom: 200-400 factory phone images.

**Pre-annotation approach:**
1. Run **RF-DETR-L** on factory images → COCO pretrained includes "cell phone" class (class 67/80)
2. Review and correct detections in Label Studio
3. For FPI-Det dataset: already in YOLO format, no re-annotation needed

**What does NOT need labeling (pose rules):**
- `hands_in_pockets` — detected by MoveNet pretrained (wrist-hip keypoint proximity rule). Only needs 100-200 calibration images for threshold tuning (no annotation, just visual verification).
- `no_handrail` — detected by MoveNet pretrained (wrist position relative to handrail zone polygon). Only needs zone polygon configuration per camera.

**Common mistakes to avoid:**
- Label the phone object, not the hand holding it
- Include phones held to ear (phone call) and phones held in front (texting/browsing)
- Do NOT label phones in pockets or bags (not visible = not detectable)

---

#### 3.2.7 Model I: Zone Intrusion

**No labeling needed.** Uses pretrained YOLOX-T person detection (COCO pretrained, Apache 2.0). Only requires:
- Zone polygon configuration per camera (from factory floor plans)
- No custom training data

---

### 3.3 Factory Validation Set (Separate from Training)

In addition to training data, AT must collect and label a **factory validation set** — real camera footage used as the true acceptance test. This data is NEVER mixed into training.

| Timeline | Target | Purpose |
|---|---|---|
| Week 2-3 | 50 images per model | Initial validation baseline |
| Week 4-5 | 100-200 images per model | DG2 midpoint validation |
| Week 6 | 300-500 images per model (frozen) | Final acceptance test set |
| Week 7+ | Active learning additions | Models flag low-confidence → AT labels → feeds back |

**Pre-annotation for validation set:** Same approach as training data (RF-DETR-L + SAM 3 + MoveNet), but validation labels must be **100% human-verified** — no auto-annotation accepted without review.

---

## 4. Labeling Configuration

### 4.1 PPE: Helmet Detection — Label Config XML (Phase 1 — Model B)

Go to Project > Settings > Labeling Interface > Code, paste:

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="person" background="#FF0000"/>
    <Label value="helmet" background="#00FF00"/>
    <Label value="no_helmet" background="#FF6600"/>
    <Label value="nitto_hat" background="#0066FF"/>
  </RectangleLabels>
</View>
```

### 4.2 PPE: Safety Shoes Detection — Label Config XML (Phase 1 — Model F)

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="person" background="#FF0000"/>
    <Label value="safety_shoes" background="#00FF00"/>
    <Label value="no_safety_shoes" background="#FF6600"/>
  </RectangleLabels>
</View>
```

### 4.3 Poketenashi: Phone Detection — Label Config XML (Phase 1 — Model H)

> **Note:** Only `phone` needs detection labeling. `hands_in_pockets` and `no_handrail` are detected via pose keypoint rules (YOLO11n-Pose pretrained) — no custom labeling needed for those behaviors.

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="phone" background="#0066FF"/>
  </RectangleLabels>
</View>
```

### 4.4 PPE Full Detection — Label Config XML (Phase 2 — includes vest, gloves, goggles)

> **Phase 2** — Not in scope for Phase 1 development. Includes all PPE classes for future full-PPE model.

Go to Project > Settings > Labeling Interface > Code, paste:

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="person" background="#FF0000"/>
    <Label value="helmet" background="#00FF00"/>
    <Label value="no_helmet" background="#FF6600"/>
    <Label value="vest" background="#0066FF"/>          <!-- Phase 2 -->
    <Label value="no_vest" background="#FF00FF"/>        <!-- Phase 2 -->
    <Label value="gloves" background="#00FFFF"/>         <!-- Phase 2 -->
    <Label value="no_gloves" background="#996633"/>      <!-- Phase 2 -->
    <Label value="goggles" background="#FFFF00"/>        <!-- Phase 2 -->
    <Label value="no_goggles" background="#FF3399"/>     <!-- Phase 2 -->
  </RectangleLabels>
</View>
```

### 4.5 Forklift Proximity — Label Config XML (Phase 2)

> **Phase 2** — Not in scope for Phase 1 development.

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="person" background="#FF0000"/>
    <Label value="forklift" background="#0000FF"/>
    <Label value="pallet_jack" background="#00FF00"/>
  </RectangleLabels>
</View>
```

### 4.6 Fire & Smoke — Label Config XML (Phase 1 — Model A)

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="fire" background="#FF4400"/>
    <Label value="smoke" background="#888888"/>
  </RectangleLabels>
</View>
```

### 4.7 Fall/Pose — Label Config XML (Phase 1 — Model G, Keypoint Annotation)

```xml
<View>
  <Image name="image" value="$image" zoom="true"/>
  <KeyPointLabels name="keypoints" toName="image" smart="true">
    <Label value="nose" background="#FF0000"/>
    <Label value="left_eye" background="#FF3300"/>
    <Label value="right_eye" background="#FF6600"/>
    <Label value="left_ear" background="#FF9900"/>
    <Label value="right_ear" background="#FFCC00"/>
    <Label value="left_shoulder" background="#CCFF00"/>
    <Label value="right_shoulder" background="#66FF00"/>
    <Label value="left_elbow" background="#00FF33"/>
    <Label value="right_elbow" background="#00FF99"/>
    <Label value="left_wrist" background="#00FFCC"/>
    <Label value="right_wrist" background="#00CCFF"/>
    <Label value="left_hip" background="#0066FF"/>
    <Label value="right_hip" background="#0033FF"/>
    <Label value="left_knee" background="#3300FF"/>
    <Label value="right_knee" background="#6600FF"/>
    <Label value="left_ankle" background="#9900FF"/>
    <Label value="right_ankle" background="#CC00FF"/>
  </KeyPointLabels>
</View>
```

> Note: For pose/fall, prefer using COCO Keypoints pretrained data. Only annotate custom factory footage if workers wear bulky equipment that changes body proportions.

---

## 5. Annotation Guidelines

### 5.1 General Rules

1. **Bounding boxes must be tight** — edges of the box should touch the object boundary
2. **Label every instance** — don't skip partially visible objects
3. **Occluded objects** — label if >30% of the object is visible
4. **Minimum size** — skip objects smaller than 20x20 pixels (too small for detection)
5. **One label per box** — each bounding box gets exactly one class label

### 5.2 PPE Detection Rules

#### Person
- Draw box around the **full body** of each worker
- Include head to feet when visible
- If only upper body visible, draw box around visible portion

#### Helmet / No-Helmet
- Draw tight box around the **head area**
- `helmet`: head clearly wearing a hard hat/safety helmet
- `no_helmet`: head clearly visible without any helmet
- If head is not visible (turned away, occluded), **do not label**

#### Vest / No-Vest (Phase 2)

> **Phase 2** — Not in scope for Phase 1 development.

- Draw box around the **torso area**
- `vest`: high-visibility vest clearly visible on torso
- `no_vest`: torso clearly visible without safety vest
- If torso is heavily occluded, skip

#### Gloves / No-Gloves (Phase 2)

> **Phase 2** — Not in scope for Phase 1 development.

- Draw box around **both hands together** or **individual hands**
- Only label if hands are clearly visible
- Skip if hands are not in frame or too small (<15px)

#### Goggles / No-Goggles (Phase 2)

> **Phase 2** — Not in scope for Phase 1 development.

- Draw box around the **eye/face area**
- `goggles`: safety glasses or goggles visible on face
- `no_goggles`: face visible without eye protection
- Skip if face is not visible

#### PPE Visual Reference

```
    ┌─────────┐
    │ helmet/ │  ← Head region: helmet or no_helmet (Phase 1)
    │no_helmet│
    ├─────────┤
    │ goggles/│  ← Face region: goggles or no_goggles (Phase 2)
    │no_goggle│
    ├─────────┤
    │  vest/  │  ← Torso region: vest or no_vest (Phase 2)
    │ no_vest │
    ├─────────┤
    │         │
    ├─────────┤
    │ gloves/ │  ← Hand region: gloves or no_gloves (Phase 2)
    │no_gloves│
    └─────────┘

    ← Full body box = "person"
```

### 5.3 Forklift Proximity Rules (Phase 2)

> **Phase 2** — Not in scope for Phase 1 development.

#### Person
- Same as PPE person labeling
- Include all workers, even those far from forklifts

#### Forklift
- Draw box around the **entire forklift** including forks
- Include the driver if seated
- Label forklifts even when stationary/parked

#### Pallet Jack
- Draw box around the entire pallet jack
- Include manual and electric pallet jacks

### 5.4 Fire & Smoke Rules

#### Fire
- Draw box around the **visible flame area**
- Include all separate fire patches as individual boxes
- Do NOT include surrounding glow (only actual flames)

#### Smoke
- Draw box around the **densest visible smoke area**
- Smoke can be large and diffuse — draw box around the main concentration
- Label thin wisps only if clearly visible
- **Common false positives to avoid:** steam, dust clouds, fog — label only if it looks like combustion smoke

### 5.5 Quality Checklist (per image)

- [ ] All visible persons labeled
- [ ] All PPE items labeled (present AND absent)
- [ ] Bounding boxes are tight (no excess padding)
- [ ] No duplicate labels on same object
- [ ] Occluded objects (>30% visible) are labeled
- [ ] Very small objects (<20px) are skipped

---

## 6. Pre-Annotation Stack (SAM 3 + RF-DETR + MoveNet)

Use three complementary models for pre-annotation to minimize manual labeling effort (70-85% reduction).

### 6.1 Install Pre-Annotation Tools

```bash
# Install all three pre-annotation models
uv pip install rfdetr sam3 easy-vitpose
```

### 6.2 Pre-Annotation Models

| Model | Task | When to Use |
|---|---|---|
| **RF-DETR-L** | Bbox detection (person, fire, objects) | All detection models (a, b, f, g-classify, h) |
| **SAM 3** | Open-vocabulary segmentation | Novel classes (nitto_hat, safety shoes), smoke/fire (irregular shapes) |
| **MoveNet Huge** | 17-keypoint pose estimation | Fall pose (g-pose), Poketenashi pose rules |

### 6.3 Setup SAM 3 as Label Studio Backend

```bash
# Clone Label Studio ML Backend
git clone https://github.com/HumanSignal/label-studio-ml-backend.git
cd label-studio-ml-backend

# Run SAM 3 backend (replaces SAM 2)
docker compose -f docker-compose.sam3.yml up -d
```

**Connect to Label Studio:**
1. Go to Project > Settings > Machine Learning
2. Add ML Backend URL: `http://localhost:9090`
3. Enable "Use for interactive preannotations"

### 6.4 Workflow: Interactive Annotation with SAM 3

1. Open an image in Label Studio
2. Type the class name (e.g., "helmet") — SAM 3 finds **all** instances automatically
3. Review and correct detected instances
4. Confirm and move to next image

> **SAM 3 vs SAM 2:** SAM 3 (Meta, ICLR 2026) adds open-vocabulary instance detection — type a class name and it segments all matching objects. SAM 2 required clicking each object individually. SAM 3 reduces per-image annotation time from ~2 min to ~15 sec.

### 6.5 Workflow: Batch Pre-Annotation (Offline)

For large batches, run pre-annotation offline before importing to Label Studio:

```python
# Step 1: RF-DETR — detect all objects, generate bbox labels
from rfdetr import RFDETRLarge
det_model = RFDETRLarge()
for image in images:
    detections = det_model.predict(image)
    save_yolo_labels(detections)  # → labels/*.txt

# Step 2: SAM 3 — segment novel classes by text prompt
from sam3 import SAM3
sam = SAM3.from_pretrained("facebook/sam3-hiera-large")
for image in images:
    masks = sam.predict(image, text_prompt="nitto hat")
    save_yolo_labels(masks_to_bboxes(masks))

# Step 3: MoveNet — generate keypoint annotations for pose models
from easy_vitpose import VitPoseModel
pose_model = VitPoseModel("huge")
for image, person_bboxes in person_crops:
    keypoints = pose_model.predict(image, bboxes=person_bboxes)
    save_coco_pose(keypoints)  # → 17 COCO keypoints per person

# Step 4: Import pre-annotations into Label Studio for human review
```

### 6.6 Which Model for Which Project?

| Label Studio Project | Pre-Annotation Model | Output |
|---|---|---|
| `Fire-Smoke` | RF-DETR-L + SAM 3 ("fire", "smoke") | Bboxes |
| `PPE-Helmet` | RF-DETR-L + SAM 3 ("helmet", "head") | Bboxes |
| `PPE-SafetyShoes` | SAM 3 ("shoes", "feet") + RF-DETR-L | Bboxes (small objects benefit from SAM 3) |
| `Fall-Pose` | MoveNet Huge | 17 COCO keypoints |
| `Poketenashi` | RF-DETR-L ("phone") + MoveNet (pose rules) | Bboxes + keypoints |

> Pre-annotation reduces human effort to **review & correct** only — typically 15-30% of manual labeling time.

---

## 7. Export Settings

### 7.1 Export to YOLO Format (Detection Models)

1. Go to Project > Export
2. Select format: **YOLO**
3. Export creates:
   ```
   export/
   ├── images/
   │   ├── image001.jpg
   │   └── image002.jpg
   ├── labels/
   │   ├── image001.txt    # class_id cx cy w h (normalized)
   │   └── image002.txt
   └── classes.txt          # class names
   ```

### 7.2 Export to COCO JSON (Pose Models)

1. Go to Project > Export
2. Select format: **COCO**
3. Export creates:
   ```
   export/
   ├── images/
   │   ├── image001.jpg
   │   └── image002.jpg
   └── result.json          # COCO format with keypoints
   ```

### 7.3 Class ID Mapping

#### PPE: Helmet Detection (Phase 1 — Model B)
```
0: person
1: helmet
2: no_helmet
3: nitto_hat
```

#### PPE: Safety Shoes Detection (Phase 1 — Model F)
```
0: person
1: safety_shoes
2: no_safety_shoes
```

#### Poketenashi: Phone Detection (Phase 1 — Model H)
```
0: phone
```
> Only phone class needs labeling. hands_in_pockets and no_handrail use pose keypoint rules (pretrained, no labeling).

#### Fire & Smoke (Phase 1 — Model A)
```
0: fire
1: smoke
```

#### PPE Full Detection (Phase 2 — all PPE classes)
```
0: person
1: helmet
2: no_helmet
3: vest
4: no_vest
5: gloves
6: no_gloves
7: goggles
8: no_goggles
```

#### Forklift Proximity (Phase 2)
```
0: person
1: forklift
2: pallet_jack
```

---

## 8. Annotation Workflow & QA

### 8.1 Recommended Workflow

```
Step 1: Import images into Label Studio project
           │
Step 2: Run pre-annotation: RF-DETR + SAM 3 + MoveNet (auto-labels ~70-85%)
           │
Step 3: Annotator reviews & corrects pre-annotations
           │
Step 4: Senior ML engineer reviews 10% random sample
           │
Step 5: If error rate > 5%, annotator re-reviews batch
           │
Step 6: Export to YOLO/COCO format
           │
Step 7: Run dataset validation script (check distribution, duplicates)
```

### 8.2 Quality Metrics

| Metric | Target |
|---|---|
| Missing labels (objects not annotated) | < 3% |
| Wrong class labels | < 2% |
| Loose bounding boxes (>10px padding) | < 5% |
| Inter-annotator agreement (IoU) | > 0.80 |

### 8.3 Review Process in Label Studio

1. Go to Project > Settings > Instructions — paste this guide's section 4
2. Enable **Review Stream**: Settings > Annotation > Enable review
3. Assign reviewer role to senior ML engineer
4. Reviewer approves/rejects each annotation
5. Rejected annotations go back to annotator queue

---

## 9. Dataset Split Script

After export, split into train/val/test:

```bash
#!/bin/bash
# split_dataset.sh — Split YOLO dataset into train/val/test

DATASET_DIR="./export"
OUTPUT_DIR="./dataset_split"

mkdir -p $OUTPUT_DIR/{train,val,test}/{images,labels}

# Count total images
TOTAL=$(ls $DATASET_DIR/images/*.jpg | wc -l)
TRAIN_END=$((TOTAL * 70 / 100))
VAL_END=$((TOTAL * 90 / 100))

# Shuffle and split
ls $DATASET_DIR/images/*.jpg | shuf > /tmp/file_list.txt

head -n $TRAIN_END /tmp/file_list.txt | while read f; do
  base=$(basename "$f" .jpg)
  cp "$f" $OUTPUT_DIR/train/images/
  cp "$DATASET_DIR/labels/${base}.txt" $OUTPUT_DIR/train/labels/ 2>/dev/null
done

sed -n "$((TRAIN_END+1)),${VAL_END}p" /tmp/file_list.txt | while read f; do
  base=$(basename "$f" .jpg)
  cp "$f" $OUTPUT_DIR/val/images/
  cp "$DATASET_DIR/labels/${base}.txt" $OUTPUT_DIR/val/labels/ 2>/dev/null
done

tail -n +$((VAL_END+1)) /tmp/file_list.txt | while read f; do
  base=$(basename "$f" .jpg)
  cp "$f" $OUTPUT_DIR/test/images/
  cp "$DATASET_DIR/labels/${base}.txt" $OUTPUT_DIR/test/labels/ 2>/dev/null
done

echo "Split complete:"
echo "  Train: $(ls $OUTPUT_DIR/train/images/ | wc -l) images"
echo "  Val:   $(ls $OUTPUT_DIR/val/images/ | wc -l) images"
echo "  Test:  $(ls $OUTPUT_DIR/test/images/ | wc -l) images"
```

---

## 10. YOLO Dataset YAML (for training)

Phase 1 dataset configs already exist in `configs/<usecase>/`. These are the **actual files** used for training:

#### Phase 1 Dataset YAMLs (in `configs/<usecase>/`)

```yaml
# features/safety-fire_detection/configs/05_data.yaml
path: ../merged/fire_detection
train: train/images
val: val/images
test: test/images
names:
  0: fire
  1: smoke
```

```yaml
# features/ppe-helmet_detection/configs/05_data.yaml
path: ../merged/helmet_detection
train: train/images
val: val/images
test: test/images
names:
  0: person
  1: helmet
  2: no_helmet
  # 3: nitto_hat  # TODO: add after custom data collection
```

```yaml
# features/ppe-shoes_detection/configs/05_data.yaml
path: ../merged/shoes_detection
train: train/images
val: val/images
test: test/images
names:
  0: person
  1: safety_shoes
  2: no_safety_shoes
```

```yaml
# features/safety-fall_pose_estimation/configs/05_data.yaml
path: ../merged/fall_pose_estimation
train: train/images
val: val/images
test: test/images
names:
  0: person
kpt_shape: [17, 3]
```

```yaml
# features/safety-fall-detection/configs/05_data.yaml
path: ../merged/fall_detection
train: train/images
val: val/images
test: test/images
names:
  0: person
  1: fallen_person
```

```yaml
# features/safety-poketenashi/configs/05_data.yaml (phone detection only)
path: ../merged/phone_detection
train: train/images
val: val/images
test: test/images
names:
  0: phone
```
> Poketenashi uses a hybrid pipeline: phone detection (this config) + MoveNet (pretrained, no config needed) + rule engine. See `03h_model_poketenashi.md` Section 1.6.

> **Note:** Zone Intrusion (Model I) has no dataset config — uses pretrained YOLOX-T (Apache 2.0) with no training.

#### Phase 2 Dataset YAMLs

```yaml
# ppe_full_dataset.yaml (Phase 2 — all PPE classes)
path: ./dataset_split
train: train/images
val: val/images
test: test/images

names:
  0: person
  1: helmet
  2: no_helmet
  3: vest        # Phase 2
  4: no_vest     # Phase 2
  5: gloves      # Phase 2
  6: no_gloves   # Phase 2
  7: goggles     # Phase 2
  8: no_goggles  # Phase 2
```

```yaml
# forklift_dataset.yaml (Phase 2)
path: ./dataset_split
train: train/images
val: val/images
test: test/images

names:
  0: person
  1: forklift
  2: pallet_jack
```

---

## 11. Troubleshooting

| Issue | Solution |
|---|---|
| Label Studio slow with many images | Use cloud storage (S3/MinIO) instead of local upload |
| SAM 3 backend not connecting | Check Docker logs: `docker logs label-studio-ml-backend` |
| Export missing labels | Ensure all annotations are "submitted" (not draft) |
| YOLO format has wrong class IDs | Check classes.txt order matches your training YAML |
| Images too large (>5MB each) | Resize to max 1920px before import: `mogrify -resize 1920x1920 *.jpg` |
| Multiple annotators disagreeing | Run inter-annotator agreement check, hold calibration session |

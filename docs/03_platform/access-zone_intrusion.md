# Access Control: Zone Intrusion
> ID: i | Owner: TBD | Phase: 1 | Status: training | Config: N/A (pretrained COCO)

## Customer Requirements

**Source:** [DCross - AI Camera Intrusion Detection](https://dcross.impress.co.jp/docs/news/001320.html)

**Explicit Requirements:**
- Detect persons entering dangerous or restricted areas (hazardous storage, machinery zones, loading docks)
- Detect vehicles and bicycles entering prohibited zones or driving in the wrong direction
- Detect loitering -- vehicles stopping or staying in no-parking and restricted areas beyond a time threshold
- Detect line-crossing -- persons crossing prohibited or restricted virtual lines
- Detect unauthorized entry during night hours
- Provide real-time alerts with automatic tracking of violators
- Support virtual line crossing with directional detection (allowed vs. prohibited direction)

## Business Problem Statement

- Unauthorized entry into restricted zones (hazardous storage, machinery areas, loading docks) puts workers at risk of injury and exposes the company to regulatory fines
- Lack of real-time monitoring means violations go unnoticed until after an incident occurs, increasing liability and operational consequences
- Safety zone compliance must be enforced consistently across shifts and camera locations without relying on manual patrols
- Vehicle misuse in pedestrian-only or no-parking areas creates congestion, obstructs emergency routes, and endangers foot traffic

## Technical Problem Statement

- **Unauthorized access prevention -> Zone definition and real-time detection:** Cameras must define restricted areas as polygons and lines in pixel coordinates, then detect when persons or vehicles enter those zones at high confidence and low latency
- **Safety zone compliance -> Tracking across zones:** Individual persons and vehicles must be tracked over time using unique IDs so the system can determine zone entry, dwell time, and exit -- not just per-frame presence
- **Security monitoring -> Loitering and temporal detection:** The system must measure how long a tracked object remains inside a zone and raise alerts only when the dwell time exceeds configurable thresholds, distinguishing brief transits from actual violations
- **Liability reduction -> False positive suppression:** Authorized personnel (e.g., maintenance workers with valid access) and transient detections (shadows, reflections, passing traffic) must be filtered out to keep false alarm rates below 2%, maintaining operator trust in the alert system
- **Multi-camera coverage -> Scalable zone configuration:** Each camera requires its own zone polygon definitions mapped to pixel coordinates, with time-based rules that change restrictions by shift or time of day

## Technical Solution Options

### Option 1: Pretrained YOLOX-Tiny COCO + ByteTrack + Zone Rules (Recommended)

**Approach:** Deploy YOLOX-Tiny with COCO pretrained weights (person, bicycle, car, motorcycle, bus, truck classes) as the detector. No custom training is needed -- COCO contains 200K+ person annotations achieving 90%+ mAP@0.5 out of the box. Pair with ByteTrack (MIT) for multi-object tracking with persistent IDs, and Supervision PolygonZone/LineZone for zone logic. All alert states (intrusion, loitering, wrong direction, line crossing) are computed as rule-based checks on tracked detections -- no additional model training required.

**Addresses:** Zone definition (PolygonZone), tracking (ByteTrack), loitering (dwell time on tracked IDs), false positive suppression (confidence thresholds + temporal smoothing), multi-camera scalability (per-camera JSON zone configs).

**Pros:**
- Minimal setup overhead and near-zero deployment time -- fastest Phase 1 use case
- 50-70 FPS on AX650N, 25-35 FPS on CV186AH -- headroom for additional logic
- All components Apache 2.0 / MIT -- no licensing risk
- Shared person detection pipeline with Model H (Poketenashi) reduces total system compute
- Mature COCO person detection with well-understood accuracy characteristics

**Cons:**
- Fixed to COCO class set -- cannot detect domain-specific objects without fine-tuning
- YOLOX-Tiny has lower overall AP (~33) compared to transformer alternatives
- Zone configuration requires manual per-camera calibration using factory floor plans

### Option 2: D-FINE-N COCO Pretrained + Zone Rules (Transformer Alternative)

**Approach:** Replace YOLOX-Tiny with D-FINE-N (Apache 2.0, 4M params, 42.8 AP). Same ByteTrack + PolygonZone/LineZone pipeline. D-FINE-N is NMS-free, smaller than YOLOX-Tiny, and more accurate. Falls back to YOLOX-Tiny if edge compilation issues arise.

**Addresses:** All the same technical challenges as Option 1, with higher detection accuracy especially for small/distant persons thanks to STAL (Small-Target-Aware Label Assignment).

**Pros:**
- Higher accuracy than YOLOX-Tiny (42.8 AP vs ~33 AP) with fewer parameters (4M vs 4.4M)
- NMS-free inference simplifies the pipeline
- Same Apache 2.0 license, minimal setup overhead

**Cons:**
- Transformer models may have longer edge compilation cycles
- Less battle-tested on AX650N/CV186AH compared to YOLOX-Tiny
- If edge deployment fails, requires fallback to Option 1

### Option 3: Custom-Trained Intrusion Detector

**Approach:** Fine-tune a detection model on factory-specific imagery with custom classes (e.g., "person_in_hazard_zone", "vehicle_in_restricted_area") annotated in context. Only pursued if COCO pretrained person detection proves insufficient for specific camera angles, lighting conditions, or object types.

**Addresses:** Domain gaps not covered by COCO pretrained weights (unusual camera angles, fisheye distortion, factory-specific vehicles, PPE-specific appearance).

**Pros:**
- Can adapt to factory-specific conditions that pretrained models miss
- Potential to add domain-specific classes beyond COCO vocabulary

**Cons:**
- Requires dataset collection, annotation, training, and evaluation -- weeks of additional work
- Risk of overfitting to specific camera angles or lighting conditions
- Introduces ongoing maintenance burden for model updates

**Decision:** Option 1 (Pretrained YOLOX-Tiny COCO + ByteTrack + Zone Rules) as the primary approach. Option 2 (D-FINE-N) as a drop-in accuracy upgrade if edge compilation succeeds. Option 3 reserved as a contingency if pretrained detection proves insufficient in specific factory environments.

## Approach

**Architecture:** YOLOX-Tiny (pretrained COCO, Apache 2.0) + Supervision PolygonZone + ByteTrack tracking. All components Apache 2.0 / MIT.

**Performance Targets:**
- mAP@0.5 >= 0.92
- Precision >= 0.94
- Recall >= 0.92
- FP Rate < 2%

---

## Detection Classes

| Class ID | Class Name | Type | Description |
|---|---|---|---|
| 0 | person | Pretrained detection | Person intrusion |
| 1 | vehicle_intrusion | Alert state (zone logic) | Vehicle/bicycle intrusion in restricted zones |
| 2 | wrong_direction | Alert state (tracking logic) | Movement in wrong direction |
| 3 | loitering | Alert state (dwell time logic) | Stationary in no-standing zone -- determined by tracking duration |

> Uses pretrained person/vehicle detection only. Classes 1-3 are alert states determined by zone polygon logic and tracking, not trained detection classes.

### Alert Logic

| Model | Min Confidence | Min Duration | Tracking Required |
|---|---|---|---|
| Intrusion | 0.60 | 5 frames (167ms) | Yes |

### Zone Intrusion Logic

```python
def is_polygon_intrusion(person_centroid, zone_polygon):
    """Check if person is inside restricted zone polygon"""
    return point_in_polygon(person_centroid, zone_polygon)

def is_line_crossing_invasion(person_track, zone_line, allowed_direction):
    """Check if person crosses line in wrong direction"""
    crosses = line_crossing(person_track[-2], person_track[-1], zone_line)
    if not crosses:
        return False
    crossing_direction = calculate_crossing_direction(person_track, zone_line)
    return crossing_direction != allowed_direction
```

---

## Dataset

**No custom dataset required.** Uses pretrained COCO person detection (class 0, 200K+ person annotations in pretraining). COCO pretrained models achieve 90%+ mAP@0.5 for person detection out of the box.

### Public Dataset References

| Dataset | Images | Classes | License | Notes |
|---|---|---|---|---|
| **COCO Person** | 200K+ | person (class 0/80) | CC BY 4.0 | Pretrained weights; person AP50-95 = 63.6% |
| Drone Security | 9,999 | intrusion, person | Varied | Drone-based security |
| Climbing/Fence | 10,000 | climb, no_climb | Varied | Fence violation specific |

### Zone Configuration Requirements (Not Training)

| Requirement | Quantity | Method | Priority |
|---|---|---|---|
| Zone polygons | 20-50 zones | Factory floor plan + PolygonZone tool | CRITICAL |
| Camera calibration | 5-10 cameras | Manual mapping | CRITICAL |
| Time-based rules | 10-20 rules | Configuration JSON | High |
| Alert thresholds | 10-20 parameters | Tuning with test footage | Medium |

**No annotation required.** All custom work is zone configuration and alert logic.

---

## Architecture

### Pipeline

```
Camera Frame (1080p)
    |
    v
+------------------------------------------+
|  YOLOX-Tiny (COCO pretrained)            |
|  Input: 640x640                          |
|  Classes: person, bicycle, car,          |
|           motorcycle, bus, truck         |
|  Speed: 50+ FPS (AX650N), 25+ (CV186AH) |
+------------------------------------------+
    | detections
    v
+------------------------------------------+
|  ByteTrack Multi-Object Tracker          |
|  - Track person/vehicle IDs             |
|  - Maintain trajectory history          |
|  - Handle brief occlusions             |
|  - Runs on CPU: ~2ms per frame          |
|                                          |
|  Config:                                |
|  track_buffer: 60 frames (2 seconds)   |
|  match_threshold: 0.8 (IoU)            |
|  low_threshold: 0.1                     |
+------------------------------------------+
    | tracked objects with IDs + trajectories
    v
+------------------------------------------+
|  Zone Intelligence Engine (CPU, <1ms)    |
|                                          |
|  (1) Polygon Intrusion:                  |
|      centroid inside restricted polygon  |
|      -> INTRUSION_ALERT                  |
|                                          |
|  (2) Line Crossing:                      |
|      trajectory segment crosses virtual  |
|      line boundary                       |
|      -> LINE_CROSSING_ALERT             |
|                                          |
|  (3) Wrong Direction:                    |
|      crossing direction != allowed_dir   |
|      -> WRONG_DIRECTION_ALERT           |
|                                          |
|  (4) Loitering:                          |
|      object stays in zone > N seconds   |
|      displacement < threshold           |
|      -> LOITERING_ALERT                 |
|                                          |
|  (5) No-Parking:                         |
|      vehicle stationary in zone         |
|      duration > threshold               |
|      -> NO_PARKING_ALERT                |
+------------------------------------------+
```

### Component Stack

| Component | Model/Tool | Purpose | Training Required? |
|---|---|---|---|
| Person detection | YOLOX-Tiny (COCO pretrained) | Detect persons in frame | NO (pretrained) |
| Person tracking | ByteTrack (Supervision) | Track persons across frames | NO (algorithm) |
| Zone polygons | Supervision PolygonZone | Define restricted areas per camera | NO (configuration) |
| Alert logic | Python | Intrusion, loitering, direction, line-crossing | NO (rule-based) |

### Alternative Detectors

**D-FINE-N (Apache 2.0):** 4M params, 42.8 AP (vs YOLOX-Tiny ~33 AP), NMS-free, 2.12ms on T4. Smaller, faster, and more accurate than YOLOX-Tiny. Drop-in replacement for the YOLOX-Tiny detector in the pipeline above.

**RT-DETRv2-R18 (Apache 2.0):** 20M params, 46.5 AP, NMS-free. Higher accuracy but larger model. Discrete sampling for reliable ONNX export.

### Tracker Comparison

| Tracker | Speed | ID Switches | MOTA | Best For |
|---|---|---|---|---|
| **ByteTrack** | Fastest (~2ms) | Low | Good | Fixed cameras, simple scenes |
| OC-SORT | Fast (~3ms) | Lower | Better | Handles occlusion better |
| BoT-SORT | Slower (~8ms) | Lowest | Best | Crowded scenes, ReID needed |

**Recommendation:** ByteTrack for fixed factory cameras. Upgrade to BoT-SORT only if cross-camera re-identification is needed.

### Supervision Library Components

| Component | Class | Purpose | License |
|---|---|---|---|
| PolygonZone | `sv.PolygonZone` | Define restricted area polygons | MIT |
| PolygonZoneAnnotator | `sv.PolygonZoneAnnotator` | Visualize zones on video | MIT |
| ByteTrack | `sv.ByteTrack` | Multi-object tracking with IDs | MIT |
| Detections | `sv.Detections` | Unified detection format | MIT |
| LineZone | `sv.LineZone` | Line crossing detection | MIT |

### Advanced Features

**Time-Based Zone Rules:** Zone restrictions can be scheduled (e.g., danger_zone_A restricted 6AM-10PM, danger_zone_B restricted 24/7) with authorized roles.

**Wrong Direction Detection:** Tracks person trajectory over 10+ frames, computes movement angle, compares against allowed direction. Alert if deviation > 90 degrees.

**Dwell Time Alerting:** Tracks zone entry timestamps. Alert if person stays in zone longer than configured threshold (e.g., 10 seconds).

**Vehicle Intrusion:** Uses COCO vehicle classes (car=2, motorcycle=3, bus=5, truck=7) to detect vehicles entering pedestrian-only zones.

**Temporal Smoothing:** 15-frame confirmation threshold (~0.5 sec at 30fps) with decay to avoid flicker alerts.

### Fisheye Camera Handling

For 360-degree fisheye cameras (DWC-PVF5Di1TW):
- **Option A (Recommended):** Hardware dewarping via AX650N ISP or CV186AH LDC module into 4 perspective views (90 degrees each)
- **Option B:** Software dewarping (equirectangular to 4 virtual cameras, ~5ms per view on CPU)
- **Option C:** Direct fisheye detection -- train YOLOX on FishEye8K dataset (157K boxes), eliminates dewarping overhead

### Shared Infrastructure with Poketenashi (Model H)

Person detection and ByteTrack tracking are shared with Model H to reduce total system compute:

```
Camera Feed -> YOLOX-Tiny (person) -> ByteTrack -> [Shared Detections]
                                                        |
                                    +-------------------+-------------------+
                                    |                                       |
                            Model H: Poketenashi                   Model I: Zone Intrusion
                            - RTMPose-S (keypoints)                - PolygonZone.trigger()
                            - YOLOX-M (phone detect)               - Time-based rules
                            - Rule engine (violations)              - Direction check
                            - Temporal smoothing                    - Dwell time alert
```

---

## Training Results

**No custom training performed.** Uses pretrained COCO person detection.

### Person Detection Benchmarks (COCO Pretrained)

| Model | mAP@0.5 (Person) | Parameters | Speed (T4) | License |
|---|---|---|---|---|
| **YOLOX-Tiny** | 0.90+ | 4.4M | ~2ms | Apache 2.0 |
| YOLOX-S | 0.92+ | 8.5M | ~3ms | Apache 2.0 |
| D-FINE-N | 0.92+ (42.8 AP overall) | 4M | 2.12ms | Apache 2.0 |
| RF-DETR-B | 0.94+ | 29M | ~5ms | Apache 2.0 |

High accuracy targets are achievable because:
1. Person detection is a mature, well-researched problem (COCO has 200K+ person annotations)
2. Pretrained weights are highly optimized (90%+ mAP@0.5 for person)
3. Zone logic adds additional filtering layer (reduces false positives)
4. ByteTrack tracking provides temporal consistency (reduces flicker)

### Acceptance Metrics

| Metric | Target | Business Impact |
|---|---|---|
| Precision | >= 0.94 | Of 100 zone intrusion alarms, 94 are real intrusions |
| Recall | >= 0.92 | Of 100 actual intrusions, 92 are detected |
| FP Rate | < 2% | ~1-2 false alarms/day |
| FN Rate | < 3% | ~1-3 missed intrusions/month |
| mAP@0.5 | >= 0.92 | Internal tracking metric |

### Business Impact (Daily Operations)

| Metric | Value |
|---|---|
| Daily unauthorized entries (est.) | 50 |
| Detected correctly | 46 |
| Missed | 4 |
| False alarms/day | 1-2 |
| Annual false alarms | 365-730 |

*Assumptions: Typical factory with 100-200 workers per shift across multiple camera zones.*

---

## Edge Deployment

### Inference Requirements

| Component | FLOPs | Parameters | Latency (T4) | Latency (Edge) |
|---|---|---|---|---|
| YOLOX-Tiny | 6.5 GFLOPS | 4.4M | ~2ms | ~6-8ms |
| Zone logic | ~0 | 0 | <1ms | <1ms |
| ByteTrack | ~0 | 0 | <0.5ms | <0.5ms |

- **Expected FPS:** 50-70 FPS on AX650N, 25-35 FPS on CV186AH
- **Power:** 2.5-3W (NPU under load)
- **Target edge chips:** AX650N (18 INT8 TOPS), CV186AH (7.2 INT8 TOPS)
- **Format:** ONNX (INT8 quantized)

### Model Card (Deliverables)

Each release produces a model card at `docs/model_cards/i_zone_intrusion.md` and a YAML card at `releases/zone_intrusion/v<N>/model_card.yaml`.

**Model artifacts:**

| Artifact | Format | Path |
|---|---|---|
| PyTorch model | `.pth` | `runs/zone_intrusion/best.pt` |
| ONNX model | `.onnx` | `runs/zone_intrusion/export/i_zone_yoloxt_640_v{N}.onnx` |
| Training config | `.yaml` | `features/access-zone_intrusion/configs/06_training.yaml` |
| Metrics | `.json` | `runs/zone_intrusion/metrics.json` |

**Model card fields (auto-populated by `utils/release.py`):**

| Field | Source |
|---|---|
| Architecture, classes, input size | `06_training.yaml` |
| Dataset version (DVC tag) | `dvc tags list` |
| Training run path | `runs/zone_intrusion/` |
| mAP@0.5, Precision, Recall | `metrics.json` / `eval_results.json` |
| Per-class breakdown | Evaluation output |

---

## Zone Configuration

### Per-Camera Zone Config Example

```yaml
camera_id: 1F_Z7_001
zones:
  - name: "Hazardous Storage Perimeter"
    type: polygon
    points: [[100, 200], [500, 180], [520, 600], [80, 620]]
    alert_type: intrusion
    target_classes: [person]
    confidence_threshold: 0.60
    min_duration_frames: 5  # 167ms

  - name: "Vehicle Entry Gate"
    type: line
    points: [[0, 400], [640, 380]]
    alert_type: wrong_direction
    allowed_direction: "left_to_right"
    target_classes: [car, truck, motorcycle]

  - name: "No Parking Zone"
    type: polygon
    points: [[200, 100], [600, 100], [600, 500], [200, 500]]
    alert_type: loitering
    target_classes: [car, truck]
    duration_threshold_seconds: 30
    displacement_threshold_pixels: 50

  - name: "Emergency Exit Line"
    type: line
    points: [[320, 0], [320, 480]]
    alert_type: line_crossing
    target_classes: [person]
    bidirectional: false
```

### PolygonZone Configuration Tool

Roboflow provides a web tool to visually draw zone polygons on camera frames:
1. Upload a frame from the camera
2. Click points to draw the zone polygon
3. Press Enter to close the polygon
4. Export as NumPy array of coordinates

This generates the polygon coordinates needed for `sv.PolygonZone(polygon=np.array([...]))`.

---

## Development Plan

### Timeline: 2-3 weeks (fastest of all Phase 1 use cases)

```
Step 1: Deploy pretrained D-FINE-N (COCO, Apache 2.0) + ByteTrack (MIT)
  - If D-FINE-N edge compilation has issues -> fallback to YOLOX-Tiny (Apache 2.0)
Step 2: Configure zone polygons per camera (customer-provided or on-site calibration)
Step 3: Implement zone engine (CPU logic)
Step 4: Test with factory footage, tune thresholds
```

### Week-by-Week

- **Weeks 1-2:** Obtain factory floor plans and camera positions. Define restricted zones. Map zone polygons to camera pixel coordinates. Implement zone intrusion logic (PolygonZone, ByteTrack, loitering, direction, line-crossing). Test with sample footage.
- **Weeks 3-4:** Export YOLOX-Tiny to ONNX. Validate on sample factory footage. Initial integration testing.
- **Weeks 5-6:** Integrate with unified alert system. Test with multiple camera streams. Share person detection pipeline with Model H (Poketenashi).
- **Weeks 7-8:** Validate against factory validation set (200-500 images). Tune loitering thresholds, direction sensitivity. Per-class P/R/FP/FN.
- **Week 9:** Final acceptance test. Document calibrated thresholds. Zone management documentation complete.

### Resource Estimates

| Category | Effort |
|---|---|
| Data | None (pretrained COCO) |
| Compute | Minimal (no training) |
| Human | 2-3 weeks engineering |

**Risk level: NEAR ZERO** -- uses pretrained models, no training required.

---

## Limitations

- **Camera angle sensitivity:** Unusual camera angles or extreme lighting conditions may degrade person detection. Consider fine-tuning on factory-specific footage if needed.
- **Fisheye distortion:** Standard models do not handle fisheye distortion natively. Requires dewarping (hardware or software) or fisheye-specific training.
- **Zone configuration effort:** Each camera requires manual polygon configuration. Changes to physical layout require reconfiguration.
- **Cross-camera tracking:** ByteTrack tracks within a single camera. Cross-camera re-identification requires BoT-SORT or a dedicated ReID model.
- **Small/distant persons:** Very small persons at long distances may be missed. D-FINE-N with STAL (Small-Target-Aware Label Assignment) helps.

### Contingency

- If person detection accuracy is insufficient in specific camera angles/lighting, fine-tune on factory-specific footage
- If AGPL license becomes a concern for any component, all recommended models are Apache 2.0

---

## Demo Application

The zone intrusion tab is implemented in the Gradio demo app (`app_demo/tabs/tab_zone.py`):
- Visual zone drawing
- Image and video inference
- ByteTrack tracking toggle
- Per-zone intrusion frame statistics
- Red/cyan zone coloring (intruded/clear)

```bash
# Run demo
uv run app_demo/run.py  # :7861
```

## Key Implementation References

| Component | Library | API |
|-----------|---------|-----|
| Polygon zone | `supervision` | `sv.PolygonZone(polygon=np.array([...]))` |
| Line zone | `supervision` | `sv.LineZone(start=Point, end=Point)` |
| Zone trigger | `supervision` | `zone.trigger(detections)` → bool mask |
| Tracker | `supervision` | `sv.ByteTrack(...)` |
| Zone annotator | `supervision` | `sv.PolygonZoneAnnotator(zone=zone)` |

---

## Changelog

| Date | Change |
|---|---|
| 2026-03-26 | Initial platform feature doc merged from requirements, model card, and technical approach |

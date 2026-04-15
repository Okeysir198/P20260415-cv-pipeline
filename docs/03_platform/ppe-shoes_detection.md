# PPE: Safety Shoes Detection
> ID: f | Owner: E2 | Phase: 1 | Status: training

## Customer Requirements

**Source:** Nitto Denko factory safety specification (verbatim quote)

**Explicit Requirements:**
- Detect workers not wearing safety shoes in designated factory areas
- Area-specific detection zones to be defined separately
- Automated, camera-based monitoring -- no manual inspection

**Key Config:** `features/ppe-shoes_detection/configs/06_training.yaml`
**License:** Apache 2.0 -- commercial use permitted. No AGPL/Ultralytics models.

## Business Problem Statement

- Workers entering production areas without safety shoes are exposed to foot injuries from heavy objects, sharp materials, and chemical spills
- Regulatory bodies (OSHA, local labor law) mandate safety footwear in designated zones; violations carry fines and liability
- Manual shoe-compliance checks by supervisors are inconsistent, labor-intensive, and cannot cover all entry points or shifts
- Factory-specific safety shoe requirements (steel-toe, chemical-resistant, ESD) vary by zone, making generic enforcement insufficient
- Incidents from missing or incorrect safety footwear result in lost workdays, regulatory liability, and production downtime

## Technical Problem Statement

- **Foot safety enforcement -> Shoes are the smallest detectable objects:** Feet occupy only ~60x40 pixels in a typical 1080p frame (~20x13 pixels after 640x640 resize), making them the hardest objects to detect reliably among all Phase 1 models
- **Regulatory compliance accuracy -> Fine-grained shoe classification at low resolution:** Distinguishing safety shoes (steel-toe, closed-toe leather, safety sneakers) from non-safety footwear (sneakers, sandals, open-toed shoes) requires texture-level detail that is lost at small pixel counts
- **Factory floor occlusion -> Frequent visual obstruction:** Feet are regularly hidden behind machinery, conveyors, pallets, and other workers, leading to missed detections and false negatives
- **Inconsistent manual checks -> Automated per-person tracking:** The system must associate shoe compliance status with individual workers over time using multi-frame confirmation (>= 30 frames) to avoid alert fatigue from transient occlusion
- **Small dataset availability -> Limited training data:** Only 3.7K annotated images exist against a 14K minimum target, with no dedicated public safety shoes dataset available for pre-training or transfer learning

## Technical Solution Options

### Option 1: YOLOX-Tiny + MobileNetV3 Classifier (Two-Stage, Recommended)

**Approach:** Stage 1 uses YOLOX-Tiny (5.1M params, INT8) at 640x640 for person detection. Stage 2 crops the bottom 30% of each person bounding box, resizes to 224x224, and classifies foot crops with MobileNetV3-Small (2.5M params) into safety/non-safety categories.

**Addresses:** Small object problem (foot crop gives classifier a high-resolution 224x224 input), fine-grained classification (MobileNetV3 trained specifically on shoe textures), and occlusion (ByteTrack associates shoe status across frames).

**Pros:**
- Total 7.6M params -- fits comfortably on AX650N and CV186AH with headroom for other models
- Stage 2 runs at 100+ FPS per crop -- negligible latency impact
- Proven architecture; YOLOX-Tiny INT8 quantization is stable on edge chips
- Modular -- Stage 2 can be upgraded independently (e.g., to ViT-Tiny) if accuracy is insufficient

**Cons:**
- Two-stage pipeline adds engineering complexity (crop logic, coordinate mapping, per-person state tracking)
- Person detection failures cascade to shoe classification (missed person = missed shoe check)
- Additional training pipeline for Stage 2 classifier

### Option 2: D-FINE-N + MobileNetV3 (Transformer Alternative)

**Approach:** Replace YOLOX-Tiny with D-FINE-N (4M params, NMS-free, 2.12ms) as the Stage 1 person detector. Same MobileNetV3-Small Stage 2 for shoe classification.

**Addresses:** Same as Option 1, with better person detection accuracy due to D-FINE-N's global attention mechanism, which handles partially visible persons better at frame edges and near machinery.

**Pros:**
- Smaller total footprint (4M + 2.5M = 6.5M params)
- NMS-free inference simplifies the detection pipeline
- Global attention improves detection of partially occluded persons
- Recommended for best accuracy/size ratio

**Cons:**
- D-FINE-N INT8 quantization on AX650N/CV186AH is less proven than YOLOX-Tiny
- Transformer models may have longer first-inference latency on edge chips
- Less operational experience with D-FINE-N in production compared to YOLOX-Tiny

### Option 3: Single-Stage YOLOX-M End-to-End (Simpler)

**Approach:** Run YOLOX-M (25.3M params) at 1280x1280 input resolution to detect person, shoe regions, and shoe class directly in one pass.

**Addresses:** Simplifies pipeline -- no crop logic, no second model, no coordinate mapping. All shoe classification happens via the detection head's class predictions.

**Pros:**
- Simplest deployment -- single ONNX model, single inference call
- No cascade failure mode -- person and shoe detected together
- No need for separate Stage 2 training pipeline

**Cons:**
- ~50 GFLOPS -- 2x the compute of two-stage approach
- Poor recall for foot-level objects even at 1280px (feet still < 40px effective resolution)
- Less accurate fine-grained shoe classification (detection head vs dedicated classifier)
- Larger model leaves less headroom for concurrent model pipelines on edge chips

**Decision:** Option 1 (YOLOX-Tiny + MobileNetV3) as primary deployment for proven INT8 stability on target chips. Option 2 (D-FINE-N + MobileNetV3) as upgrade path if person detection accuracy is insufficient. Option 3 as last resort only if two-stage latency is unacceptable.

## Detection Classes

| Class ID | Name | Description |
|----------|------|-------------|
| 0 | person | Full body detection for tracking |
| 1 | foot_with_safety_shoes | Worker wearing safety shoes/boots (steel-toe, closed-toe leather, safety sneakers) |
| 2 | foot_without_safety_shoes | Worker wearing non-safety footwear (sneakers, sandals, open-toed shoes) |
| 3 | shoe_region | Foot region bounding box for analysis (used in two-stage pipeline) |

## Dataset

### Current State

| Attribute | Value |
|-----------|-------|
| Total images | 3.7K (small -- target 14K minimum) |
| Format | YOLO (`class_id cx cy w h`, normalized 0-1) |
| Path | `../../dataset_store/shoes_detection/{train,val,test}/{images,labels}/` |
| Config | `features/ppe-shoes_detection/configs/05_data.yaml` |
| Input size | 640x640 (two-stage) or 1280x1280 (single-stage) |

**Critical: No dedicated safety shoes dataset exists in public repositories.** Shoes/boots only appear as a minor class within larger PPE datasets, with limited variety and poor occlusion coverage.

### Public Dataset Sources

| Dataset | Images | Classes | License | Quality | Notes |
|---------|--------|---------|---------|---------|-------|
| Construction-PPE | 1,416 | Includes "boots" | MIT | 3/5 | Small; part of multi-class PPE |
| Safety-Guard | 5,000+ | Includes footwear | MIT | 3/5 | Limited shoe-specific data |
| PPE Detection | 10,151 | Includes "boots" | Varied | 3/5 | Construction sites |
| Industrial Safety | Varied | Varied | Varied | 2/5 | Search Roboflow/Kaggle |

### Data Expansion Strategy

| Source | Images | Purpose |
|--------|--------|---------|
| Current dataset (f_safety_shoes) | 3.7K | Primary training data |
| Generative augment (shoes_expansion) | ~5K | Generative augmentation for data expansion |
| SH17 dataset (foot class) | ~2K | Supplementary foot images |
| Factory footage collection | ~3K | Real-world safety shoe variety |
| **Total target** | **~14K** | **Minimum for reliable training** |

### Custom Data Collection Priorities

| Scenario | Images Needed | Collection Method | Priority |
|----------|---------------|-------------------|----------|
| Safety shoes (clear view) | 1,000-1,500 | Factory footage | High |
| Occluded shoes | 1,000-1,500 | Factory footage | **CRITICAL** |
| Different shoe types | 500-1,000 | Various safety shoes | Medium |
| Non-safety shoes | 500-1,000 | Factory footage | High |
| Various angles | 500-1,000 | Staged collection | Medium |
| Long-distance shots | 500 | Far camera angles | Medium |

**Camera angles:** overhead, 45-degree, eye-level. **Shoe styles:** steel-toe, chemical-resistant, ESD, sneakers, sandals, slippers. **Distances:** near, mid, far.

### Hard Negative Images

Add ~500-1,000 images containing only loose shoes/boots (on floors, shelves, shoe racks) with **empty `.txt` label files**. These teach the model to suppress detections on shoe-like objects not worn on a person's feet.

## Annotation Guidelines

1. Annotate full `person` body first
2. Annotate `shoe_region` (both feet together if visible)
3. Classify shoes:
   - `foot_with_safety_shoes`: Steel-toe boots, closed-toe leather shoes, safety sneakers
   - `foot_without_safety_shoes`: Sneakers, sandals, open-toed shoes
4. If feet occluded by objects/machinery -- annotate `person` only
5. For people sitting/kneeling -- still annotate shoes if visible

**Ambiguous Cases:**
- Shoe covers -- `foot_with_safety_shoes` (safety footwear assumed underneath)
- Boots partially visible -- annotate visible region only
- Shadows obscuring feet -- use `shoe_region` with caution

## Architecture

### Why Two-Stage Detection

Safety shoes occupy a tiny portion of the frame:
- Full person at 1080p: ~200x500 pixels
- Foot region: ~60x40 pixels (bottom 8% of person)
- After resize to 640x640: foot is approximately 20x13 pixels

**Single-stage detection at 640x640 has poor recall for foot-level objects.** Two-stage with cropping is strongly recommended.

### Two-Stage Pipeline (YOLOX-Tiny + MobileNetV3)

```
Camera Frame (1080p)
    |
    v
+---------------------------------------------+
|  Stage 1: Person Detection                  |
|  YOLOX-Tiny (5.1M params, INT8)            |
|  Input: 640x640                             |
|  Output: person bounding boxes              |
|  Speed: 50+ FPS on AX650N                  |
+---------------------------------------------+
    | person bboxes
    v
+---------------------------------------------+
|  Foot Region Extraction (CPU)               |
|  Crop bottom 30% of each person bbox        |
|  Expand by 15% padding                      |
|  Resize to 224x224                          |
+---------------------------------------------+
    | foot crops
    v
+---------------------------------------------+
|  Stage 2: Shoe Classification               |
|  MobileNetV3-Small (2.5M params, INT8)      |
|  Input: 224x224 foot crop                   |
|  Output: safety_shoes / no_safety_shoes     |
|  Speed: 100+ FPS per crop on AX650N        |
+---------------------------------------------+
    |
    v
  ByteTrack (associate shoe status with person ID)
    |
    v
  Alert: foot_without_safety_shoes sustained >= 30 frames
```

### Stage 1 Model Options

| Stage | Model | Params | Notes |
|-------|-------|--------|-------|
| 1 (Person) | YOLOX-Tiny | 5.1M | Proven INT8 stability on AX650N/CV186AH, 50+ FPS |
| 1 (Person) | D-FINE-N | 4M | NMS-free, 2.12ms, global attention for partially visible persons -- upgrade if YOLOX-Tiny person accuracy insufficient |

### Stage 2 Model Options

| Stage | Model | Params | Notes |
|-------|-------|--------|-------|
| 2 (Shoes) | MobileNetV3-Small | 2.5M | 224x224 foot crop, BSD-3 license, 100+ FPS |
| 2 (Shoes) | ViT-Tiny | 5.7M | Upgrade if MobileNetV3 shoe accuracy < target -- better fine-grained texture/detail classification (toe cap, sole pattern, ankle height) |

### Single-Stage Fallback

If two-stage latency is unacceptable, crop the bottom 60% of the 1080p frame (where feet appear), resize to 640x640, and run YOLOX-M for direct shoe detection. Simpler but less accurate -- feet may appear anywhere if camera angle varies.

### Stage 2 Training Config (Shoe Classifier)

```yaml
model:
  architecture: mobilenetv3-small
  num_classes: 2  # foot_with_safety_shoes, foot_without_safety_shoes
  pretrained: imagenet
  input_size: 224

training:
  epochs: 100
  batch_size: 64
  optimizer: AdamW
  lr: 0.001
  scheduler: cosine
  warmup_epochs: 5

augmentation:
  random_crop: true
  color_jitter: [0.3, 0.3, 0.3, 0.1]
  random_perspective: true
  random_erasing: 0.3
  horizontal_flip: 0.5
```

## Alert Logic

| Parameter | Value |
|-----------|-------|
| Min Confidence | 0.70 |
| Min Duration | 30 frames (1 sec at 30 FPS) |
| Tracking Required | Yes (ByteTrack) |

## Training Results

| Metric | Target | Min Acceptable | v1 | v2 | v3 |
|--------|--------|----------------|----|----|-----|
| mAP@0.5 | >= 0.85 | >= 0.75 | | | |
| Precision | >= 0.88 | >= 0.85 | | | |
| Recall | >= 0.85 | >= 0.80 | | | |
| FP Rate | < 4% | < 6% | | | |
| FN Rate | < 5% | < 7% | | | |

### Industry Benchmarks (Reference)

> These benchmark figures were gathered via AI-assisted research and are provided as reference only. They have not been independently verified.

| Context | mAP@0.5 | Precision | Recall | FP Rate |
|---------|---------|-----------|--------|---------|
| Shoes (clear view) | 0.85-0.90 | 0.90 | 0.89 | ~2% |
| Shoes (occluded) | 0.70-0.80 | 0.75 | 0.72 | ~5% |
| Overall (mixed) | 0.80-0.85 | ~0.85 | ~0.82 | ~4% |

### Per-Class Benchmarks

| Class | Typical mAP@0.5 | Precision | Recall | Challenge Level |
|-------|-----------------|-----------|--------|-----------------|
| foot_with_safety_shoes (clear) | 0.88-0.92 | 0.90 | 0.89 | Easy-Medium |
| foot_with_safety_shoes (occluded) | 0.70-0.80 | 0.75 | 0.72 | Hard |
| foot_without_safety_shoes | 0.85-0.90 | 0.88 | 0.87 | Easy-Medium |

## Development Plan

### Week 1-2: Setup & Data Review
- Verify merged shoe dataset (3.7K images)
- Check annotation quality -- shoe bboxes are small, verify accuracy
- Plan augmentation strategy: mosaic, copy-paste, TTA (critical for small dataset)
- Begin factory shoe image collection (target: 500 initial)
- Configure `features/ppe-shoes_detection/configs/05_data.yaml`

### Week 3-4: v1 Training
- Train at 1280px default (single-stage) or two-stage pipeline
- Evaluate v1: mAP, P/R per class
- Error analysis: occlusion cases, regular-shoes-vs-safety-shoes confusion

### Week 5-6: v2 Training (augmentation + new data)
- Merge new factory shoe images (target: 1,500 total)
- Heavy augmentation: mosaic=1.0, copy_paste=0.3, mixup=0.15

### Week 7-8: v3 + Decision Gate
- Merge all available data (target: 2,000+)
- **DG3 check:** if mAP < 0.75 after v3, switch to D-FINE-S or RT-DETRv2-S
- If DG3 triggered: retrain with alternative architecture (uses buffer weeks W10-12)

### Week 9: Export & Handoff
- Export to ONNX
- Build PPE alert logic (shoes compliance: confidence + multi-frame confirmation)
- Run against factory validation set (300-500 images)
- Test on heavily occluded scenes specifically

### Model Card (Deliverables)

Each release produces a model card at `docs/model_cards/f_safety_shoes.md` and a YAML card at `releases/shoes_detection/v<N>/model_card.yaml`.

**Model artifacts:**

| Artifact | Format | Path |
|---|---|---|
| PyTorch model | `.pth` | `runs/shoes_detection/best.pt` |
| ONNX model | `.onnx` | `runs/shoes_detection/export/f_shoes_yoloxm_{imgsz}_v{N}.onnx` |
| Training config | `.yaml` | `features/ppe-shoes_detection/configs/06_training.yaml` |
| Metrics | `.json` | `runs/shoes_detection/metrics.json` |

**Model card fields (auto-populated by `utils/release.py`):**

| Field | Source |
|---|---|
| Architecture, classes, input size | `06_training.yaml` |
| Dataset version (DVC tag) | `dvc tags list` |
| Training run path | `runs/shoes_detection/` |
| mAP@0.5, Precision, Recall | `metrics.json` / `eval_results.json` |
| Per-class breakdown | Evaluation output |

## Edge Deployment

| Attribute | Value |
|-----------|-------|
| Target chips | AX650N (18 INT8 TOPS), CV186AH (7.2 INT8 TOPS) |
| Export format | ONNX (then INT8 quantized for target chip) |
| Expected FPS (two-stage) | 25-40 FPS on target hardware |
| Power (two-stage) | 3-5W |
| Stage 1 latency | ~10-12ms (YOLOX-Tiny on Hailo-8 equivalent) |
| Stage 2 latency | <10ms per crop (MobileNetV3-Small) |

### Inference Requirements

| Model | FLOPS | Parameters | Latency (T4) |
|-------|-------|------------|--------------|
| Single-stage (YOLOX-M, 1280px) | ~50 GFLOPS | 25.3M | ~8ms |
| Two-stage (YOLOX-Tiny + MobileNetV3) | ~26 GFLOPS | 7.6M total | 4-5ms |

## Resource Estimation

| Category | Details |
|----------|---------|
| Data (custom collection) | Factory footage collection (1,000-1,500 images) |
| Compute | Local GPU training |
| Human effort | ML Engineer 50-60h + Annotator 20-30h |

**Timeline: 3-4 weeks** (challenging due to occlusion handling and small dataset)

## Contingency Plan

**Trigger:** Precision < 0.85 or Recall < 0.80 (tracking: mAP < 0.75) after 2 training iterations.

**Actions:**
1. Implement two-stage detection (person -> feet region -> shoe classification)
2. Add temporal tracking (use shoe position from previous frames)
3. Use pose keypoints to estimate foot position
4. Consider shoe detection from lower body only (crop image at waist)
5. If all fail after DG3, evaluate outsourcing this single model to a specialized CV vendor

## Limitations & Known Issues

- **Occlusion is the primary challenge.** Feet are frequently hidden behind machinery, other workers, or objects. Expect lower recall in occluded scenes compared to other PPE models.
- **Small dataset (3.7K).** Well below the 14K minimum target. Generative augmentation and factory footage collection are critical.
- **No public benchmark dataset** exists specifically for safety shoes, making it difficult to compare against published results.
- **Fine-grained classification.** Distinguishing safety shoes from regular shoes at low resolution requires texture-level detail that may be lost after resize.
- **Camera angle dependency.** Overhead cameras may not capture feet at all; system works best with 45-degree or eye-level camera angles.
- **Higher FN rate acceptable.** Due to physical constraints of foot visibility, a higher false negative rate (< 5%) is accepted compared to helmet detection.

## Key Commands

```bash
# Train detection (Stage 1)
uv run core/p06_training/train.py --config features/ppe-shoes_detection/configs/06_training.yaml

# Train classifier (Stage 2)
uv run core/p06_training/train.py --config features/ppe-shoes_detection/configs/06_training.yaml

# Evaluate
uv run core/p08_evaluation/evaluate.py --model runs/shoes_detection/best.pt --config features/ppe-shoes_detection/configs/05_data.yaml --split test

# Data preparation (merge sources)
uv run core/p00_data_prep/run.py --config features/ppe-shoes_detection/configs/00_data_preparation.yaml

# Generative augmentation
uv run core/p03_generative_aug/run_augment.py --config features/ppe-shoes_detection/configs/03_generative_augment.yaml
```

## Changelog

| Date | Version | Change |
|------|---------|--------|
| 2026-04-01 | 0.2 | Migrate top sections to new template: Customer Requirements, Business Problem, Technical Problem, Technical Solution Options; restructure Architecture to remove Primary/Alternative/Fallback labels |
| 2026-03-26 | 0.1 | Initial platform doc merged from model plan, model card, and technical approach |

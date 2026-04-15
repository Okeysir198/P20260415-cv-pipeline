# PPE: Helmet Detection
> ID: b | Owner: E2 | Phase: 1 | Status: training

## Customer Requirements

**Source:** [Phase 1 Requirements](../01_requirements/phase1-requirements.md) -- Model B: Helmet Detection

**Explicit Requirements:**
- Detect workers not wearing several types of helmets (including Nitto soft hats)
- Detect two types of safety belts (deferred to Gap G2 -- separate model expansion)
- Cover all PPE levels (A through E) defined by the customer's chemical area safety standards

**Customer Reference (PPE Levels):** [Kurita Co., Ltd. - Chemical Area PPE Requirements](https://kcr.kurita.co.jp/solutions/videos/049.html)

| Level | Area | Required PPE |
|---|---|---|
| **A** | Chemical Area (patrol/inspection) | Safety glasses (goggles type) |
| **B** | Chemical Receiving/Acceptance | Full-face shield, chemical-resistant gloves, chemical-resistant safety rubber boots |
| **C** | Chemical Handling (under-floor) | Full-face shield, chemical-resistant apron, gloves, safety rubber boots |
| **D** | Chemical Handling (above-floor) / line maintenance | Full-face shield, chemical-resistant apron (split type), gloves, safety rubber boots |
| **E** | Emergency Response | Disaster-response full-face mask, chemical-resistant protective clothing, gloves, safety rubber boots |

## Business Problem Statement

- Factory workers must wear appropriate head protection to prevent serious head injuries from falling objects, overhead equipment, and workplace accidents
- Non-compliance with PPE regulations exposes Nitto Denko to regulatory violations, fines, and legal liability under Japanese industrial safety standards
- Nitto Denko requires detection of hard hats, bare heads, and factory-specific Nitto soft cloth hats -- a unique requirement not met by off-the-shelf solutions
- Safety belt/harness detection also requested but deferred to Gap G2 (separate model expansion)
- Non-compliance with PPE regulations leads to workplace injuries, production shutdowns, regulatory penalties, and reputational damage

## Technical Problem Statement

- **Worker safety / head injury prevention → Nitto soft hat visual similarity:** Nitto soft hats are visually similar to other headwear (beanies, baseball caps, bandanas), making them difficult to distinguish reliably without dedicated training data and fine-grained classification
- **Regulatory compliance → Absent from public datasets:** Nitto soft hats are a factory-specific item absent from all public PPE detection datasets (SHWD, Hard Hat Workers, Construction-PPE, Safety-Guard), requiring custom data collection from actual factory footage
- **Nitto Denko factory-specific requirements → 4-class fine-grained classification:** The model must discriminate between `head_with_helmet`, `head_without_helmet`, `head_with_nitto_hat`, and `person` simultaneously -- a harder task than binary helmet/no-helmet detection
- **Regulatory compliance → Varying distances and viewing angles:** Factory cameras capture workers at varying distances (close-up to far away) and from multiple angles (front, side, back, overhead), requiring robust detection across all perspectives including partially occluded heads
- **Worker safety / head injury prevention → NMS suppression in crowds:** Crowded factory scenes with workers in close proximity risk NMS (non-maximum suppression) suppressing nearby head detections, causing missed violations when multiple workers stand together

## Technical Solution Options

### Option 1: YOLOX-M -- Single-Stage 4-Class Detector (Recommended)

- **Approach:** Single YOLOX-M model (25.3M params, CSPDarknet + PAFPN + Decoupled Head) trained on 4 classes simultaneously: `person`, `head_with_helmet`, `head_without_helmet`, `head_with_nitto_hat`. Input 640x640 (1280x1280 if small-helmet recall is low).
- **Addresses:** All 5 technical challenges. Good INT8 quantization behavior on AX650N/CV186AH edge chips. Apache 2.0 license.
- **Pros:** Fastest development cycle, single model to deploy and maintain, well-understood training pipeline, proven baseline for PPE detection (mAP >0.90 achievable with sufficient data)
- **Cons:** Nitto hat subclass may have lower precision (~0.75) due to visual similarity with other headwear; may require 1280px input for distant helmets which reduces edge FPS

### Option 2: D-FINE-S -- Transformer Single-Stage Detector (Alternative)

- **Approach:** D-FINE-S (10M params, HGNetv2-S backbone) trained on the same 4 classes. Uses hybrid encoder (AIFI + CCFM) and NMS-free FDR output.
- **Addresses:** NMS suppression in crowds (NMS-free), varying distances (FDR fine-grained refinement), fine-grained classification (hybrid encoder captures person-helmet spatial relationships). Higher accuracy than YOLOX-M with fewer parameters.
- **Pros:** Better accuracy than YOLOX-M with fewer parameters; NMS-free output eliminates duplicate person/helmet box detections in crowds; Apache 2.0 license
- **Cons:** Transformer models are newer and less tested on edge INT8 quantization; higher engineering risk for deployment

### Option 3: Two-Stage Pipeline -- Person Detector + Crop Classifier (Fallback)

- **Approach:** Stage 1 uses lightweight detector (YOLOX-Tiny or D-FINE-N, 4-5M params) for `person` detection. Stage 2 crops the head region (top 25% of person bbox, expanded 20%) and classifies with MobileNetV3-Small (2.5M params, 224x224 input) into `head_with_helmet` / `head_without_helmet` / `head_with_nitto_hat`.
- **Addresses:** Fine-grained classification (classifier sees zoomed-in head crop), NMS suppression (person-level tracking, not head-level), visual similarity (dedicated classifier on high-resolution crop).
- **Pros:** Higher per-object accuracy because the classifier sees a zoomed-in head; lightweight person detector runs at 50+ FPS on AX650N; classifier inference ~1ms per crop
- **Cons:** Added pipeline complexity (two models to train, deploy, and maintain); slightly higher total latency; error cascades if person detector misses a worker

**Decision:** Start with Option 1 (YOLOX-M single-stage) for fastest baseline. Train Option 2 (D-FINE-S) in parallel for comparison. Escalate to Option 3 (two-stage) only if single-stage mAP < 0.92 after 2 training iterations.

## Detection Classes

| Class ID | Name | Description |
|----------|------|-------------|
| 0 | person | Full body detection for tracking |
| 1 | head_with_helmet | Worker wearing helmet (any type including hard hats, safety helmets) |
| 2 | head_without_helmet | Worker head visible without helmet (bare head, baseball caps, beanies) |
| 3 | head_with_nitto_hat | Nitto soft hat specifically (typically blue/white cloth cap) |

## Dataset

- **Sources:**
  - Current merged dataset (b_helmet): 62K images (person, head_with_helmet, head_without_helmet)
  - SHWD: 7,581 images (9,044 positive, 111,514 negative instances) -- Open license
  - Hard Hat Workers: 7,000 images (helmet, person, vest) -- CC BY 4.0
  - Safety-Guard: 5,000+ images (8 PPE classes) -- MIT
  - Construction-PPE: 1,416 images (11 PPE classes, 1,132 train / 143 val / 141 test) -- MIT
  - PPE Detection: 10,151 images (11 PPE classes) -- Varied
  - SH17: 8,099 images, 75,994 instances (manufacturing environments) -- Academic
  - GDUT-HWD: 3,200 images (multi-color hard hat variety)
  - Generative augment: ~5K (head_with_helmet -> head_without_helmet transformation)
  - Factory footage: ~3K (Nitto hat specific samples)
- **Size:** train/val/test splits from 62K merged dataset
- **DVC tag:** `helmet-data-v{N}`
- **Path:** `dataset_store/helmet_detection/{train,val,test}/{images,labels}/`

**Dataset Acquisition:**
```bash
# Construction-PPE (via Ultralytics)
yolo export data=construction-ppe.yaml
# Hard Hat Workers (Kaggle)
kaggle datasets download -d andrewmvd/hard-hat-detection
# Safety-Guard
git clone https://github.com/pfeifer/safety-guard-dataset
# SHWD: Search "Safety-Helmet-Wearing-Dataset SHWD GitCode"
```

### Custom Data Requirements

**Nitto Hat Challenge:** No public Nitto hat dataset exists. Nitto hats (soft cloth caps) are underrepresented in open datasets and require custom collection for the Japanese factory context. This is the critical data gap for this model.

| Scenario | Images Needed | Collection Method | Priority |
|---|---|---|---|
| Nitto soft hats | 500-1,000 | Factory workers, staged | **CRITICAL** |
| Various helmet colors | 500-1,000 | Factory footage | High |
| Back/side views | 500 | Staged collection | Medium |
| Poor lighting | 300-500 | Night shift, dim areas | Medium |
| Long-distance shots | 300-500 | Far camera angles | Medium |

**Total custom data estimate: 2,100-3,500 images**

## Architecture

- **Model:** YOLOX-M (CSPDarknet + PAFPN + Decoupled Head), 25.3M params, depth=0.67, width=0.75, Apache 2.0
- **Input size:** 640x640 (try 1280 if small-helmet recall is low)
- **Key config:** `features/ppe-helmet_detection/configs/06_training.yaml`
- **Tracker:** ByteTrack (MIT) on CPU
- **License:** Apache 2.0 -- commercial use permitted

### Pipeline Flow

```
Input (640x640)
    |
    v
  YOLOX-M (CSPDarknet + PAFPN + Decoupled Head)
    |
    v
  4-class output: person, head_with_helmet, head_without_helmet, head_with_nitto_hat
    |
    v
  ByteTrack (person tracking on CPU)
    |
    v
  Alert: head_without_helmet sustained >= 30 frames on tracked person
```

### Training Configuration

```yaml
model:
  architecture: yolox-m
  num_classes: 4  # person, head_with_helmet, head_without_helmet, head_with_nitto_hat
  pretrained: yolox_m.pth

training:
  epochs: 200
  batch_size: 16
  input_size: 640  # try 1280 if small-helmet recall is low

loss:
  cls_loss: focal
  reg_loss: ciou
  cls_weight: 1.5  # increased for better head_with_helmet/head_without_helmet discrimination
  obj_weight: 1.0
  reg_weight: 5.0
```

### Helmet-Specific Training Tricks

1. **Generative augmentation for hard negatives**: Use generative augment with `helmet_to_no_helmet.yaml` -- replace helmet with bare head to generate `head_without_helmet` samples
2. **Multi-scale training**: Helmets at distance can be <10 pixels. Train with random input 448-1024
3. **Attention module**: Add CBAM or EMA (Efficient Multi-Scale Attention) after the PAFPN neck to improve small helmet detection
4. **Nitto hat differentiation**: Ensure balanced samples of each type. If differentiation is too hard, merge `head_with_nitto_hat` into `head_with_helmet` class
5. **Hard negative images (~500-1000)**: Add images containing only loose helmets (on tables, shelves, hooks) with empty `.txt` label files to suppress detections on helmet-like objects not worn on a person's head

### Two-Stage Fallback (Higher Accuracy)

If single-stage mAP < 0.92:

```
Stage 1: YOLOX-Tiny -> detect "person" (lightweight, 50+ FPS on AX650N)
    |
    v
  Crop head region (top 25% of person bbox, expanded by 20%)
    |
    v
Stage 2: MobileNetV3-Small classifier -> head_with_helmet / head_without_helmet / head_with_nitto_hat
    Input: 224x224 crop
    Params: 2.5M
    INT8: ~1MB
    Inference: ~1ms per crop on AX650N
```

**Advantage:** Higher per-object accuracy because the classifier sees a zoomed-in head.
**Disadvantage:** Added pipeline complexity, slightly higher latency.

## Alert Logic

| Model | Min Confidence | Min Duration | Tracking Required |
|---|---|---|---|
| PPE Violation | 0.70 | 30 frames (1 sec) | Yes |

## Training Results

| Metric | Target | Min Acceptable | v1 | v2 | v3 |
|--------|--------|----------------|----|----|-----|
| mAP@0.5 | >= 0.92 | 0.85 | | | |
| Precision | >= 0.94 | -- | | | |
| Recall | >= 0.92 | -- | | | |
| FP Rate | < 2% | -- | | | |
| FN Rate | < 3% | -- | | | |

### Industry Benchmarks (Reference)

> **Disclaimer:** Benchmark figures were gathered via AI-assisted research. They are provided as reference only and have not been independently verified.

| Solution | Precision | Recall | FP Rate | mAP@0.5 | Notes |
|---|---|---|---|---|---|
| YOLOv8 + Construction-PPE | 0.94 | 0.93 | ~1% | 0.92-0.94 | Best benchmark -- 11-class PPE |
| YOLOv8 + SHWD | 0.91 | 0.87 | ~2% | 0.885 | DarkNet53 backbone |
| Improved YOLOv5s (2024) | 0.91 | 0.89 | ~2% | 0.89+ | +3.9% improvement over baseline |
| Hard Hat Workers (Kaggle) | ~0.90 | ~0.88 | ~2% | 0.91 | Helmet-only dataset |

### Per-Class Benchmarks (Reference)

| Class | Typical Precision | Typical Recall | mAP@0.5 | Challenge Level |
|---|---|---|---|---|
| head_with_helmet (front view) | 0.94 | 0.93 | 0.92-0.95 | Easy |
| head_with_helmet (side/back view) | 0.87 | 0.86 | 0.85-0.90 | Medium |
| head_with_nitto_hat (soft cap) | **0.75** | **0.72** | **0.70-0.80** | **Hard** (underrepresented) |
| head_without_helmet | 0.93 | 0.92 | 0.90-0.94 | Easy-Medium |

**Key Insights:**
- Helmet detection achieves Precision >0.94 with sufficient training data (5,000+ images)
- `head_with_nitto_hat` (cloth caps) are underrepresented: Precision ~0.75, needs custom data collection

### Business Impact Scenario

- **Target:** 94% precision, 92% recall
- **Per Day:** 100 workers enter without helmets
- **Expected Detection:** 92 workers detected correctly
- **Expected Missed:** 8 workers (FN rate: 8%)
- **Expected False Alarms:** 2-3 false alarms per day (assuming ~100 workers with helmets)
- **Annual Impact:** ~730-1,095 false alarms/year (low, manageable)

## Annotation Guidelines

**Rules:**
1. Always annotate `person` first (full body or visible part)
2. Annotate head region for helmet classes (not full body)
3. Distinguish between:
   - `head_with_helmet`: Hard hats, safety helmets
   - `head_with_nitto_hat`: Soft cloth caps (typically blue/white)
   - `head_without_helmet`: Bare head, baseball caps, beanies
4. For partially visible heads, annotate visible portion
5. Exclude people in background (> 15m distance) if unclear

**Special Cases:**
- Worker facing away: Annotate as `person` only (head not visible)
- Back of head visible: Annotate as `head_with_helmet` if shape indicates helmet
- Reflections in mirrors/glass: Exclude

## Edge Deployment

- **Target chips:** AX650N (18 INT8 TOPS) / CV186AH (7.2 INT8 TOPS)
- **Export format:** ONNX (then INT8 quantization for edge)
- **Expected performance at 640px:** 30-50 FPS on edge NPU (s-variant), 60-80 FPS (n-variant)
- **Expected performance at 1280px:** 10-20 FPS on edge NPU
- **Power:** 2.5-5W (NPU, depending on load)

### Inference Requirements

| Model | FLOPS | Parameters | Memory | Latency (T4) |
|---|---|---|---|---|
| YOLOX-M | -- | 25.3M | -- | -- |
| D-FINE-S | -- | 10M | -- | -- |
| RT-DETRv2-R18 | -- | 20M | -- | ~4.6ms |

### Model Card (Deliverables)

Each release produces a model card at `docs/model_cards/b_helmet.md` and a YAML card at `releases/helmet_detection/v<N>/model_card.yaml`.

**Model artifacts:**

| Artifact | Format | Path |
|---|---|---|
| PyTorch model | `.pth` | `runs/helmet_detection/best.pt` |
| ONNX model | `.onnx` | `runs/helmet_detection/export/b_helmet_yoloxm_{imgsz}_v{N}.onnx` |
| Training config | `.yaml` | `features/ppe-helmet_detection/configs/06_training.yaml` |
| Metrics | `.json` | `runs/helmet_detection/metrics.json` |

**Model card fields (auto-populated by `utils/release.py`):**

| Field | Source |
|---|---|
| Architecture, classes, input size | `06_training.yaml` |
| Dataset version (DVC tag) | `dvc tags list` |
| Training run path | `runs/helmet_detection/` |
| mAP@0.5, Precision, Recall | `metrics.json` / `eval_results.json` |
| Per-class breakdown | Evaluation output |

## Development Plan

### Week 1: Setup & Specs Review
- Verify Construction-PPE, Hard Hat Workers merged dataset (62K images)
- Check class distribution: person, head_with_helmet, head_without_helmet
- Begin nitto_hat factory image collection with AT (target: 300-500 initial)
- Configure `features/ppe-helmet_detection/configs/05_data.yaml`
- Set up training environment (GPU, W&B, DVC)

### Week 2: Data Exploration & Curation
- Dataset quality review: visualize dataset, check label consistency, identify duplicates
- Label error audit: find label errors, near-duplicates, outliers in the 62K dataset
- Curate v1 subset: ~8K high-quality images from the 62K pool (balanced classes, diverse scenes)
- Document data quality findings and curation criteria

### Week 3-4: v1 Training (640px, curated subset)
- Train on curated high-quality subset (~8K images from 62K) to validate pipeline and establish clean baseline
- Evaluate v1: mAP, P/R per class
- Error analysis: confusable cases (baseball caps, beanies, back-of-head views)

### Week 5-6: v2 Training (expanded dataset)
- Fix dataset from error analysis
- Expand training set with additional curated images from the full 62K pool
- Merge initial nitto_hat custom data from AT (target: 1,000-1,500 images)
- If mAP < 0.75 at 640px: switch to 1280px

### Week 7-8: v3 Training (full dataset + Nitto hat data)
- Train on full cleaned dataset
- Merge all nitto_hat custom data from AT (target: 2,000-2,500 images by W8)
- Oversample nitto_hat class to balance dataset
- Test nitto_hat Precision/Recall specifically

### Week 9: Export & Handoff
- Export best model to ONNX
- Build PPE alert logic (confidence + multi-frame confirmation)
- Run against factory validation set (300-500 images)
- Per-class P/R/FP/FN: head_with_helmet, head_without_helmet, head_with_nitto_hat
- Threshold tuning for production deployment

**Timeline: 3-4 weeks baseline + Nitto hat custom data collection**

## Limitations & Known Issues

- **Nitto hat detection** is the hardest class (Precision ~0.75, Recall ~0.72) due to underrepresentation in public datasets; requires custom factory data collection
- **Side/back view helmets** are harder to detect (mAP 0.85-0.90 vs 0.92-0.95 for front views)
- **Long-distance shots** -- helmets at distance can be <10 pixels; may require 1280px input
- **Confusable headwear** -- baseball caps and beanies may be confused with no-helmet or nitto-hat classes
- **Hard negatives** -- loose helmets on tables/shelves may trigger false positives without dedicated hard-negative training images

## Contingency Plans

### Contingency 1: Nitto Hat Detection Fails

**Trigger:** Nitto hat Precision < 0.90 or Recall < 0.88 (tracking: mAP < 0.85) after 2 training iterations

**Actions:**
1. Collect additional 500-1,000 Nitto hat examples
2. Use higher input resolution (1280px)
3. Try different architecture: D-FINE-S / RT-DETRv2-S (higher accuracy on small objects, HGNetV2 CNN -- NPU-friendly)
4. Consider separate Nitto hat model (ensemble approach)

### Contingency 2: License Issues

**Trigger:** Customer requires commercial-friendly license

**Actions:**
1. YOLOX-M is already Apache 2.0 -- no issue
2. If switching architectures: D-FINE-S (Apache 2.0) or RT-DETRv2-S (Apache 2.0)

## Build vs Outsource

| Model | Data Volume | Risk Level | Recommendation |
|---|---|---|---|
| **b** Helmet | 62K | LOW | Build in-house -- standard PPE detection |

**Decision:** Build in-house. The architecture (YOLOX-M, Apache 2.0) and tooling are mature. Outsourcing adds communication overhead and does not solve the real bottleneck: data quality and quantity. Decision gates (DG1-DG5) handle failure escalation.

**What to outsource if needed:** Annotation (already covered by AT) and additional factory data collection -- not model training.

## Gap G2: Harness & Safety Belt

**Customer also expects:** "Detection of not wearing safety belts"

**Recommended approach:** Expand Model B to 6 classes:
- Existing: person, head_with_helmet, head_without_helmet, head_with_nitto_hat
- New: harness, no_harness

**Requirements:** ~3K labeled images of harness/no_harness. Safety belt (waist belt) may be too small for overhead cameras — clarify viewing angles with customer.

**Alternative:** Separate Model C with higher input resolution (1280x1280) for better harness strap detection.

## Key Commands

```bash
# Train
uv run core/p06_training/train.py --config features/ppe-helmet_detection/configs/06_training.yaml

# Evaluate
uv run core/p08_evaluation/evaluate.py --model runs/helmet_detection/best.pt --config features/ppe-helmet_detection/configs/05_data.yaml --split test --conf 0.25

# Export
uv run core/p09_export/export.py --model runs/helmet_detection/best.pt --training-config features/ppe-helmet_detection/configs/06_training.yaml --export-config configs/_shared/09_export.yaml

# Generative augmentation (start SAM3 service first)
uv run core/p03_generative_aug/run_augment.py --config features/ppe-helmet_detection/configs/03_generative_augment.yaml
```

## Changelog

| Date | Version | Change |
|------|---------|--------|
| | | |

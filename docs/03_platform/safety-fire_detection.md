# Safety: Fire Detection
> ID: a | Owner: E1 | Phase: 1 | Status: training

## Customer Requirements

**Source:** [AITech Co., Ltd. - Fire Detection AI Camera](https://prtimes.jp/main/html/rd/p/000000071.000014310.html)

**Explicit Requirements:**
- Detect even the smallest possible fires
- Detect open flames at long range, depending on the size of the fire source
- Enable early identification of abnormal heat rise before a fire occurs (temperature change detection)
- Support early fire warning by detecting smoke from a distance

**Detection Distances (Customer Specified):**

| Fire Source Size | Detection Distance |
|---|---|
| 1 x 1 m fire source | Up to **325 m** |
| 0.5 x 0.5 m fire source | Up to **162.5 m** |
| 0.2 x 0.2 m fire source | Up to **65 m** |

| Temperature Change Area | Detection Distance |
|---|---|
| 1 x 1 m area | Up to **76 m** |
| 0.5 x 0.5 m area | Up to **38 m** |
| 0.2 x 0.2 m area | Up to **15.2 m** |

| Smoke Source Size | Detection Distance |
|---|---|
| 1 x 1 m smoke area | Up to **65 m** |
| 0.5 x 0.5 m smoke area | Up to **32.5 m** |
| 0.2 x 0.2 m smoke area | Up to **13 m** |

## Business Problem Statement

- Factory fires and smoke incidents pose an immediate risk to worker lives and can escalate rapidly without early warning
- Property and equipment damage from undetected fires results in significant production downtime and operational disruption
- Regulatory and insurance requirements mandate fire monitoring systems in industrial environments
- Late detection increases the severity of incidents, leading to more extensive remediation and potential legal liability
- Without automated monitoring, reliance on human observation creates gaps in coverage during off-hours and across large facility areas

## Technical Problem Statement

- **Life safety -> Long-range detection:** Fires as small as 0.2 x 0.2 m must be detected at distances up to 65 m, and 1 x 1 m fires at up to 325 m, requiring the system to identify objects as small as 10-14 pixels in the camera frame
- **Early warning -> Smoke detection difficulty:** Smoke is diffuse, transparent, and amorphous with no rigid boundaries, making it significantly harder to detect than solid objects like flames
- **Property protection -> False positive control:** Common factory visual elements (sunlight reflections, orange machinery, steam vents, fog) produce fire-like appearances that must be reliably distinguished from actual fire and smoke
- **Pre-fire prevention -> RGB-only constraint:** The customer requires temperature anomaly detection but the current camera hardware provides only RGB imagery (no thermal imaging), limiting the ability to detect heat signatures before visible flames appear
- **Facility-wide coverage -> Edge deployment:** The detection system must run on edge chips (AX650N / CV186AH) with limited compute, requiring models that balance accuracy against real-time inference constraints

## Technical Solution Options

### Option 1: YOLOX-M (CNN -- Primary)

- **Approach:** CSPDarknet53 backbone with PAFPN neck and decoupled detection head. 25.3M params, Apache 2.0 license. Proven pipeline with excellent INT8 quantization on target edge chips.
- **Addresses:** Long-range detection (via tiled inference), false positive control (strong augmentation pipeline), edge deployment (excellent INT8 quantization, ~30 FPS at 640x640 on AX650N)
- **Pros:** Well-understood architecture, reliable ONNX export, proven quantization with minimal accuracy loss, existing training pipeline and tooling
- **Cons:** Local convolutions may struggle with diffuse smoke boundaries compared to global attention mechanisms; anchor-based design adds tuning complexity

### Option 2: D-FINE-S (Transformer -- Alternative)

- **Approach:** HGNetv2-S CNN backbone with hybrid encoder (AIFI self-attention + CCFM cross-scale fusion) and transformer decoder with Fine-grained Distribution Refinement (FDR). 10M params, 48.5 AP, Apache 2.0, NMS-free output.
- **Addresses:** Smoke detection difficulty (global attention captures diffuse patterns), false positive control (FDR refines boundaries), edge deployment (60% fewer params than YOLOX-M, 3.49ms on T4)
- **Pros:** Global attention better suited for amorphous smoke; NMS-free output simplifies edge pipeline; 60% fewer parameters than YOLOX-M with higher COCO AP; quantization-friendly CNN backbone
- **Cons:** Transformer decoder requires mixed-precision quantization (attention in FP16) on edge chips; newer architecture with less production track record; ONNX export complexity for deformable attention

### Option 3: RT-DETRv2-R18 (Transformer -- Fallback)

- **Approach:** ResNet-18 backbone with discrete sampling operator designed for reliable ONNX/TensorRT export. 20M params, 46.5 AP, Apache 2.0.
- **Addresses:** Edge deployment (discrete sampling operator specifically solves ONNX export issues with deformable attention), smoke detection (attention-based encoder)
- **Pros:** Deployment reliability is the design priority -- discrete sampling operator avoids common ONNX export failures; ResNet-18 backbone is widely supported across all edge toolchains
- **Cons:** Lower AP (46.5) than both YOLOX-M (46) and D-FINE-S (48.5); larger than D-FINE-S (20M vs 10M params) with lower accuracy; slower inference than D-FINE-S (4.6ms vs 3.49ms on T4)

**Decision:** Start with YOLOX-M as primary (proven pipeline, best INT8 quantization). Train D-FINE-S in parallel as alternative (stronger on smoke, smaller model). Fall back to RT-DETRv2-R18 only if D-FINE-S encounters edge deployment issues.

## Detection Classes

| Class ID | Name | Description |
|----------|------|-------------|
| 0 | fire | Open flames of any visible size |
| 1 | smoke | Visible smoke from combustion |
| 2 | thermal_anomaly | Abnormal heat signature (future -- requires thermal camera integration) |

## Alert Logic

| Model | Min Confidence | Min Duration | Tracking Required |
|---|---|---|---|
| Fire | 0.70 | 3 frames (100ms) | No |

## Dataset

- **Sources:**
  - **FASDD** (120,000+ images) -- Multi-sensor (ground/UAV/satellite), COCO/YOLO format, Open Access. [github.com/openrsgis/FASDD](https://github.com/openrsgis/FASDD)
  - **FASDD_CV** (40,000+) -- Ground-based sensors, mAP: 84.9%. [Kaggle](https://www.kaggle.com/dataset_store/yuulind/fasdd-cv-coco)
  - **FASDD_UAV** (40,000+) -- Drone/UAV imagery, mAP: 89.7%
  - **FASDD_RS** (40,000+) -- Satellite imagery, mAP: 74.0%
  - **D-Fire** (21,000+ images) -- YOLO-format annotations, fire+smoke, Academic license. [github.com/gaiasd/DFireDataset](https://github.com/gaiasd/DFireDataset)
  - **DFS Dataset** (~999 images) -- Classification only (no bboxes, needs annotation), CC0. [Kaggle](https://www.kaggle.com/dataset_store/phylake1337/fire-dataset)
  - **Roboflow Fire** (varied) -- Community-contributed, variable quality, YOLO/VOC format
  - **Bow Fire Dataset** (300+) -- Small but clean, CC BY 4.0
- **Size:** ~142K total (120K+ FASDD + 21K D-Fire + 1.1-2K custom)
- **Split:** 80% train / 10% val / 10% test
- **Calibration set:** 500 images from validation set (for INT8 PTQ on edge)
- **DVC tag:** TBD

### Custom Data Requirements

| Scenario | Images Needed | Collection Method | Priority |
|---|---|---|---|
| Indoor fires (controlled) | 200-300 | Staged with fire safety team | High |
| Smoke tests | 100-200 | Smoke generators | Medium |
| Factory fire footage | 500-1,000 | Historical/archival footage (if available) | High |
| Long-range calibration | 300-500 | Outdoor controlled burns | Medium |

**Total custom data estimate: 1,100-2,000 images**

### Data Strategy

| Source | Images | Purpose |
|---|---|---|
| Current dataset (a_fire) | 122K | Primary training data |
| D-Fire | ~21K | Domain diversity (outdoor/indoor fire) |
| FASDD | 120K+ | Multi-domain (ground/UAV/satellite) |
| Hard negatives | ~5K | Sunlight, orange objects, steam, fog |
| Generative augment | ~10K | Generative augmentation for rare scenarios |

## Annotation Guidelines

**Bounding Box Rules:**
1. Draw bounding box around entire flame/smoke area
2. Include smoke plume base even if faint
3. For distant fires (< 50px), mark entire affected region
4. Exclude reflections and false positives (sunlight, heaters)
5. Label thermal anomalies (heat haze) if visible

**Quality Check:**
- Minimum 50 annotated images per reviewer
- Inter-annotator agreement > 0.85 (IoU)
- 10% random sample audit

## Architecture

- **YOLOX-M** (25.3M params, depth=0.67, width=0.75) -- Apache 2.0
- **D-FINE-S** (10M params, 48.5 AP, NMS-free) -- Apache 2.0
- **RT-DETRv2-R18** (20M params, 46.5 AP) -- Apache 2.0, discrete sampling for reliable ONNX export
- **Input size:** 640x640 (standard), 1280x1280 (long-range mode)
- **Key config:** `features/safety-fire_detection/configs/06_training.yaml`

### YOLOX-M Architecture (CNN)

```
Input (640x640 RGB)
    |
    v
+-------------------------------------------+
|  CSPDarknet53 Backbone                    |
|  depth=0.67, width=0.75                   |
|  +- Stem: 3->48ch, 6x6 conv, stride 2    |
|  +- Dark2: stride 4, CSP blocks          |
|  +- Dark3: stride 8  -> P3 feature       |
|  +- Dark4: stride 16 -> P4 feature       |
|  +- Dark5: stride 32 -> P5 feature       |
+-------------------------------------------+
    | P3, P4, P5
    v
+-------------------------------------------+
|  PAFPN Neck                               |
|  Top-down: P5->P4->P3 (upsample + CSP)   |
|  Bottom-up: P3->N3->N4 (downsample + CSP)|
|  Output: 3 fused feature maps             |
+-------------------------------------------+
    |
    v
+-------------------------------------------+
|  Decoupled Head (x3 scales)               |
|  +- cls branch: 2xConv -> 2 classes       |
|  +- reg branch: 2xConv -> 4 coords        |
|  +- obj branch: 2xConv -> 1 objectness    |
|  Total anchors: ~8400 (80^2 + 40^2 + 20^2)|
+-------------------------------------------+
    |
    v
  SimOTA assignment -> FocalLoss + CIoU Loss
```

### D-FINE-S Architecture (Transformer)

```
Input (640x640 RGB)
    |
    v
+-------------------------------------------+
|  HGNetv2-S Backbone (CNN)                 |
|  10M params, multi-scale features         |
|  Quantization-friendly (pure CNN)         |
+-------------------------------------------+
    |
    v
+-------------------------------------------+
|  Hybrid Encoder (AIFI + CCFM)             |
|  Intra-scale self-attention captures      |
|  smoke's diffuse, spread-out nature       |
|  Cross-scale CNN fusion for multi-res     |
+-------------------------------------------+
    |
    v
+-------------------------------------------+
|  Transformer Decoder with FDR             |
|  Fine-grained Distribution Refinement:    |
|  -> Iteratively refines boundary          |
|     probability distributions             |
|  -> More precise smoke/fire boundaries    |
|  GO-LSD self-distillation:               |
|  -> Deeper layers teach shallower layers  |
|  Output: Set of detections (NMS-free)     |
+-------------------------------------------+
    |
    v
  Hungarian matching -> Focal + L1 + GIoU Loss
```

**Why D-FINE for fire:**
- 48.5 AP with only 10M params -- matches YOLOX-M (46 AP, 25.3M) with 60% fewer parameters
- Global attention captures smoke's diffuse, spread-out nature better than local convolutions
- FDR provides more precise bounding boxes for amorphous fire/smoke shapes
- NMS-free output simplifies edge deployment pipeline
- HGNetv2 backbone quantizes well to INT8 (pure CNN)
- 3.49ms on T4 FP16 -- faster than YOLOX-M

### RT-DETRv2-R18 (Fallback)

| Feature | RT-DETRv2-R18 | D-FINE-S |
|---|---|---|
| AP (COCO) | 46.5 | 48.5 |
| Params | 20M | 10M |
| Latency (T4) | 4.6ms | 3.49ms |
| Backbone | ResNet-18 | HGNetv2 |
| Deployment | Discrete sampling op (designed for ONNX) | Standard ops |

RT-DETRv2's discrete sampling operator specifically solves ONNX/TensorRT deployment issues with deformable attention. Choose RT-DETRv2-R18 if edge deployment compatibility is the top priority.

### RF-DETR (Accuracy Ceiling -- Apache 2.0, Teacher Only)

- 60.5% AP on COCO (current SOTA for real-time detection)
- DINOv2 backbone with strong domain adaptation
- Best fine-tuning performance across 100 diverse datasets
- Too heavy for direct edge deployment -- use as teacher model for knowledge distillation
- RF-DETR-Large (teacher) -> D-FINE-S (student) distillation

### RT-DETR-Smoke (Specialized, 2025)

87.75% mAP@0.5 at 445 FPS for smoke detection:
- Modified hybrid encoder preserving smoke detail features
- Lightweight attention for transparent/diffuse objects
- Could be adapted for factory fire/smoke detection

## Training Strategy

### Training Configuration

```yaml
# features/safety-fire_detection/configs/06_training.yaml
model:
  architecture: yolox-m
  depth: 0.67
  width: 0.75
  num_classes: 2
  pretrained: yolox_m.pth  # COCO pretrained

training:
  epochs: 200
  batch_size: 16
  optimizer: SGD
  lr: 0.01
  momentum: 0.9
  weight_decay: 0.0005
  scheduler: cosine
  warmup_epochs: 5
  ema: true
  ema_decay: 0.9998
  amp: true
  grad_clip: 35.0
  early_stopping_patience: 50

loss:
  cls_loss: focal       # alpha=0.25, gamma=2.0
  reg_loss: ciou        # CIoU for better convergence
  cls_weight: 1.0
  obj_weight: 1.0
  reg_weight: 5.0
  warmup_epochs: 3      # loss warmup for stable start

augmentation:
  mosaic: true           # disable last 15 epochs
  mixup: true            # disable last 15 epochs
  mosaic_disable_epoch: 185
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  scale: [0.1, 2.0]
  degrees: 10
  translate: 0.1
  shear: 2.0
  flipud: 0.0
  fliplr: 0.5
```

### Fire-Specific Training Tricks

1. **Multi-scale training**: Random input size between 448-896 during training for scale robustness
2. **Copy-paste augmentation**: Paste fire/smoke instances onto clean factory backgrounds using segmentation masks
3. **Hard negative mining**: Include images of sunlight reflections, orange machinery, steam, fog in training set as background-only images (no fire labels)
4. **Disable mosaic/mixup last 15 epochs**: Let model fine-tune on clean images for sharper localization
5. **Class-weighted focal loss**: Increase smoke weight (smoke is harder -- transparent, diffuse boundaries)

### Small/Distant Fire Detection Strategy

```
For distant fires, detection range depends heavily on camera configuration:

- 2.1MP WIDE (89.6 deg) at 325m: 1x1m fire ~ 3x3px -- NOT detectable
- 5MP TELE (31 deg) at 325m: 1x1m fire ~ 14x14px -- DETECTABLE
- 5MP TELE (31 deg) at 65m: 0.2x0.2m fire ~ 14x14px -- DETECTABLE

>> Critical: Cameras for long-range fire detection MUST be set to telephoto.

Solution: Tiled Inference (SAHI approach)
+-----------------------------------------------+
|  1080p Frame (1920x1080)                      |
|  +--------+--------+--------+                 |
|  | Tile1  | Tile2  | Tile3  |  640x640 tiles  |
|  +--------+--------+--------+  with 15% overlap|
|  | Tile4  | Tile5  | Tile6  |                 |
|  +--------+--------+--------+                 |
|  Each tile -> YOLOX-M inference               |
|  Merge detections -> Global NMS               |
+-----------------------------------------------+

- Standard mode: full-frame 640x640 -> ~30 FPS (AX650N)
- High-res mode: 6 tiles x 640x640 -> ~5 FPS (AX650N)
- Adaptive: switch to tiled mode only when smoke haze detected
```

### Recommended Phased Strategy

```
Phase 1 (Weeks 1-6):
  Primary:   YOLOX-M (CNN) -- existing pipeline, proven, excellent INT8
  Secondary: D-FINE-S (Transformer) -- train in parallel as comparison
  Baseline:  RT-DETRv2-R18 -- if D-FINE edge deployment has issues

Phase 2 (Weeks 7-8):
  Compare mAP, precision, recall, FP rate on factory test set
  If YOLOX-M >= targets -> deploy YOLOX-M (simpler, best quantization)
  If D-FINE-S >> YOLOX-M -> deploy with mixed-precision quantization
  D-FINE-S advantage: NMS-free, smaller model, likely better on smoke

Phase 3 (if needed):
  Knowledge distillation: RF-DETR-Large (teacher) -> D-FINE-S (student)
  Generative augmentation for hard negatives
```

## Training Results

| Metric | Target | v1 | v2 | v3 |
|--------|--------|----|----|-----|
| mAP@0.5 | >= 0.85 | | | |
| Precision | >= 0.90 | | | |
| Recall | >= 0.88 | | | |
| FP Rate | < 3% | | | |
| FN Rate | < 5% | | | |

## Research: Industry Benchmarks

> All benchmark figures in this section were gathered via AI-assisted research. They are provided as reference only and have not been independently verified.

| Solution | Precision | Recall | FP Rate | mAP@0.5 | Detection Range | Notes |
|---|---|---|---|---|---|---|
| DetectNet_v2 (2024 benchmark) | 0.95 | 0.94 | ~3.5% | 0.949 | -- | Best reported overall |
| Ship-Fire Net (YOLOv8, 2024) | 0.92 | 0.89 | ~2% | 0.90+ | -- | Multi-scale + attention |
| CFS-YOLO (2024) | 0.91 | 0.90 | ~2% | ~0.90 | -- | Coarse-fine grained |
| FASDD Baseline (Swin) | 0.86 | 0.83 | ~4% | 0.849 | 50-100m | Research benchmark |
| AITech Co. (Customer Ref) | N/D | N/D | N/D | N/D | 325m (1x1m fire) | Commercial, long-range optimized |
| YOLOv8-Fire (GitHub) | ~0.88 | ~0.85 | ~3% | 0.85 | 50m | Open source |

**Key Insights:**
- Long-range detection requires high input resolution (1280px+) and strong augmentation
- Smoke detection is more challenging than flame detection (lower contrast)
- Temperature change detection requires thermal camera (not in current hardware spec)
- Industry best: Precision 95%, Recall 94% -- our targets are conservative and realistic

### Business Impact Scenario

- **Target:** 90% precision, 88% recall
- **Per Day:** 10 fire incidents occur
- **Expected Detection:** 8-9 fires detected correctly
- **Expected Missed:** 1-2 fires (FN rate: 12%)
- **Expected False Alarms:** 1-2 false alarms per day (assuming ~100 fire-like events)
- **Annual Impact:** ~365-730 false alarms/year (manageable with verification)

## Edge Deployment

- **Target chips:** AX650N (18 INT8 TOPS) / CV186AH (7.2 INT8 TOPS)
- **Export:** ONNX opset (see `configs/_shared/09_export.yaml`)
- **Format:** `.onnx` -> Pulsar2/TPU-MLIR -> INT8
- **Latency (AX650N):**
  - Standard mode (640x640): ~30 FPS
  - High-res tiled mode (6x 640x640): ~5 FPS
- **D-FINE-S edge note:** ONNX -> INT8 with mixed precision (decoder attention in FP16). Both AX650N and CV186AH support per-layer mixed precision.
- **INT8 mAP drop:** TBD

### Compute Requirements

| Hardware | GPU Memory | Training Time (640px) | Training Time (1280px) |
|---|---|---|---|
| RTX 4090 | 24GB | ~6-8 hours | ~10-12 hours |
| RTX 3090 | 24GB | ~8-10 hours | ~10-12 hours |
| A100 | 40GB | ~4-6 hours | ~8-10 hours |

**Primary: Local GPU (remote PC).**

## Development Plan

### Week 1: Setup & Specs Review
- Download and verify FASDD, DFS, Roboflow Fire datasets (122K total)
- Check annotation quality, class distribution (fire vs smoke balance)
- Configure `features/safety-fire_detection/configs/05_data.yaml`
- Split: 70% train, 20% val, 10% test
- Set up training environment (GPU, W&B, DVC)

### Week 2: Data Exploration & Curation
- Dataset quality review: visualize dataset, check label consistency, identify duplicates
- Label error audit: find label errors, near-duplicates, outliers in the 122K dataset
- Curate v1 subset: ~10K high-quality images from the 122K pool (balanced fire/smoke, diverse scenes)
- Document data quality findings and curation criteria

### Weeks 3-4: v1 Training (640px, curated subset)
> v1 uses a curated high-quality subset (~10K images from 122K) to validate the pipeline and establish a clean baseline. Full dataset added progressively in v2/v3.

- Train YOLOX-M on curated subset
- Evaluate v1: mAP, P/R per class (fire, smoke)
- Error analysis: identify mislabels, confusable negatives (lights, reflections, steam)

### Weeks 5-6: v2 Training (expanded dataset + 1280px if needed)
- Fix dataset from error analysis (remove mislabels, add hard negatives)
- Expand training set with additional curated images from the full 122K pool
- If mAP < 0.75 at 640px: switch to 1280px for long-range fire detection
- If acceptance met: prepare for v3

### Weeks 7-8: v3 Training (full dataset) + Factory Data
- Train on full cleaned dataset
- Add factory-specific fire/smoke images (if available)
- Domain adaptation: lower LR, fewer epochs on mixed data
- Test long-range detection accuracy at various distances

### Week 9: Export & Handoff
- Export best model to ONNX
- Build fire alert module with temporal filtering (multi-frame confirmation)
- Run against factory validation set (300-500 images)
- Per-class P/R/FP/FN reports (fire, smoke)
- Threshold tuning for production deployment

### Model Card (Deliverables)

Each release produces a model card at `docs/model_cards/a_fire.md` and a YAML card at `releases/fire_detection/v<N>/model_card.yaml`.

**Model artifacts:**

| Artifact | Format | Path |
|---|---|---|
| PyTorch model | `.pth` | `runs/fire_detection/best.pt` |
| ONNX model | `.onnx` | `runs/fire_detection/export/a_fire_yoloxm_{imgsz}_v{N}.onnx` |
| Training config | `.yaml` | `features/safety-fire_detection/configs/06_training.yaml` |
| Metrics | `.json` | `runs/fire_detection/metrics.json` |

**Model card fields (auto-populated by `utils/release.py`):**

| Field | Source |
|---|---|
| Architecture, classes, input size | `06_training.yaml` |
| Dataset version (DVC tag) | `dvc tags list` |
| Training run path | `runs/fire_detection/` |
| mAP@0.5, Precision, Recall | `metrics.json` / `eval_results.json` |
| Per-class breakdown | Evaluation output |

**Timeline: 9 weeks (2-3 weeks setup/data, 6 weeks training iterations, 1 week export)**

## Temperature Anomaly (Future Scope)

Requires thermal camera integration. Two paths:
1. **Thermal camera add-on**: FLIR Lepton or similar, fuse RGB + thermal features
2. **Color-based heat shimmer**: Detect visual heat distortion using optical flow analysis (stopgap, limited accuracy)

## Contingency Plan

- If mAP < 0.75 after 2 training iterations:
  1. Switch to 1280px input size
  2. If still failing, consider D-FINE-S (higher accuracy, Apache 2.0)
- If edge deployment issues with D-FINE:
  1. Fall back to RT-DETRv2-R18 (discrete sampling op designed for ONNX)
- If accuracy ceiling hit:
  1. Knowledge distillation: RF-DETR-Large (teacher) -> D-FINE-S (student)
  2. Generative augmentation for hard negatives

## Limitations & Known Issues

- Temperature change detection requires thermal camera integration (not in current hardware spec)
- Long-range detection (325m for 1x1m fire) requires telephoto camera configuration; wide-angle cameras cannot resolve fires at that distance (~3x3px)
- Smoke detection is inherently more challenging than flame detection due to lower contrast and diffuse boundaries
- DFS Dataset (999 images) is classification-only and requires annotation before use in detection training
- All research benchmark numbers are AI-gathered and not independently verified

## Build vs Outsource

| Model | Data Volume | Risk Level | Recommendation |
|---|---|---|---|
| a Fire | 122K | LOW | Build in-house -- abundant data, well-studied problem |

**Decision:** Build in-house. Outsource annotation and factory data collection only if needed.

## Key Commands

```bash
# Train
uv run core/p06_training/train.py --config features/safety-fire_detection/configs/06_training.yaml

# Resume
uv run core/p06_training/train.py --config features/safety-fire_detection/configs/06_training.yaml --resume runs/fire_detection/last.pth

# Evaluate
uv run core/p08_evaluation/evaluate.py --model runs/fire_detection/best.pt --config features/safety-fire_detection/configs/05_data.yaml --split test --conf 0.25

# Export
uv run core/p09_export/export.py --model runs/fire_detection/best.pt --training-config features/safety-fire_detection/configs/06_training.yaml --export-config configs/_shared/09_export.yaml
```

## Changelog

| Date | Version | Change |
|------|---------|--------|
| 2026-03-26 | -- | Initial platform doc merged from model plan, model card, and technical approach |

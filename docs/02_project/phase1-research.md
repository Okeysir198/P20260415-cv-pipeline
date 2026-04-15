# Phase 1 Deep Research: Datasets, Standards & Open Source Models
## Factory Smart Camera AI System - Technical Research Document

**Version:** 3.0
**Date:** March 6, 2026
**Scope:** Comprehensive research for 6 Phase 1 AI models

---

## Table of Contents

1. [Research Methodology](#1-research-methodology)
2. [Cross-Model Resource Estimation](#2-cross-model-resource-estimation)
3. [Hardware Performance Benchmarks](#3-hardware-performance-benchmarks)
4. [Recommendations & Decision Matrix](#4-recommendations--decision-matrix)
5. [Sources & References](#5-sources--references)

## Per-Model Deep Research (Separate Files)

| Model | File | Lines |
|---|---|---|
| **a** Fire Detection | [03a_model_fire.md](plans/a_fire.md) | ~322 |
| **b** Helmet Detection | [03b_model_helmet.md](plans/b_helmet.md) | ~327 |
| **f** Safety Shoes Detection | [03f_model_shoes.md](plans/f_safety_shoes.md) | ~294 |
| **g** Fall Detection | [03g_model_fall.md](plans/g_fall_detection.md) | ~341 |
| **h** Poketenashi Behavior | [03h_model_poketenashi.md](plans/h_poketenashi.md) | ~621 |
| **i** Zone Intrusion | [03i_model_zone_intrusion.md](plans/i_zone_intrusion.md) | ~539 |

> Per-model research (datasets, benchmarks, open-source models, compute requirements, resource estimation) has been moved to the model files above. Each model file merges research + development plan into a single reference.

---

## 1. Research Methodology

> **⚠ Disclaimer:** All benchmark figures, dataset statistics, and model performance numbers in this section were gathered via AI-assisted research (deep research by AI). They are provided as **reference only** and have **not been independently verified**. Always confirm critical numbers against the original source before making decisions.

### 1.1 Research Sources

**Primary Sources:**
- Academic papers (IEEE, MDPI, Springer, arXiv) - 2024-2025
- Open source repositories (GitHub, GitLab, Hugging Face)
- Public datasets (Kaggle, Roboflow Universe, Mendeley Data)
- Industrial benchmarks (FASDD, COCO, Le2i)
- Vendor documentation (Hailo, Ultralytics, Roboflow)

**Search Strategy:**
- Keywords: "{use case} + detection + dataset + 2024/2025"
- Keywords: "YOLO + {use case} + mAP + benchmark"
- Keywords: "{use case} + open source + model + performance"
- Keywords: "{use case} + computational requirements + FLOPS"

**Quality Criteria:**
- Datasets: > 1,000 images, clear annotations, permissive license
- Models: Peer-reviewed or GitHub stars > 100, mAP reported
- Benchmarks: Standard evaluation metrics (Precision, Recall, FP/FN rates; mAP@0.5 for training tracking)

### 1.2 Evaluation Metrics

**Dataset Quality Metrics:**

| Metric | Good | Acceptable | Poor |
|---|---|---|---|
| Size | > 5,000 images | 1,000-5,000 | < 1,000 |
| Annotation Quality | Verified, IoU > 0.85 | Crowd-sourced | Unverified |
| License | Apache 2.0 / MIT / CC BY | Academic use only | Proprietary |
| Class Balance | All classes > 10% | Imbalanced < 5% | Extreme imbalance |
| Diversity | Multiple scenarios | Limited | Single scenario |

**Model Quality Metrics:**

*Primary Acceptance Metrics (customer-facing — determines Go/No-Go):*

| Metric | State-of-Art | Production Ready | Research Only |
|---|---|---|---|
| Precision | > 0.94 | 0.85-0.94 | < 0.85 |
| Recall | > 0.92 | 0.82-0.92 | < 0.82 |
| False Positive Rate | < 2% | 2-5% | > 5% |
| False Negative Rate | < 3% | 3-6% | > 6% |

*Secondary Tracking Metrics (internal — used during training iteration):*

| Metric | State-of-Art | Production Ready | Research Only |
|---|---|---|---|
| mAP@0.5 | > 0.90 | 0.80-0.90 | < 0.80 |
| Inference Speed (T4) | < 5ms | 5-20ms | > 20ms |
| Parameters | < 10M | 10-50M | > 50M |
| FLOPS | < 10 GFLOPS | 10-50 GFLOPS | > 50 GFLOPS |

> **Note:** All metric targets are technical team proposals — customer has not specified quantitative targets. Require customer verification and alignment before finalizing.

---

## 2. Cross-Model Resource Estimation


### 2.1 Team Structure (3-Engineer Parallel Execution)

> See [03_phase1_development_plan.md](phase1_development_plan.md) Section 5.0 for team structure, model ownership rationale, and Section 8 for full resource planning.

### 2.2 Development Timeline

> See [03_phase1_development_plan.md](phase1_development_plan.md) Section 5.1 for the finalized 12-week development timeline (W1-2 = setup + data exploration, W3 = v1 training on curated subsets) and GPU hour estimates per engineer.

### 2.5 Dataset Requirements Summary

| Model | Public Data | Custom Data | Total | Annotation Effort |
|---|---|---|---|---|
| Fire | 141,000+ (FASDD + D-Fire) | 1,100-2,000 | ~142,000+ | 10-15 hours |
| Helmet | ~34,000+ (SHWD + others) | 2,100-3,500 | ~36,000+ | 15-20 hours |
| Shoes | ~16,500+ (partial PPE sets) | 4,000-6,500 | ~20,500+ | 20-30 hours |
| Fall | 200K+ (COCO) + ~200 videos | 2,000-3,000 | ~202,000+ | 15-25 hours |
| Poketenashi | 228,000+ (FPI-Det + COCO-Pose + Roboflow) | 800-1,600 | ~229,000+ | 5-10 hours |
| Intrusion | 200K+ (COCO pretrained) | 0 | ~200,000+ | 0 hours (pretrained) |
| **TOTAL** | **~620,000+** | **13,200-21,100** | **~632,500+** | **85-125 hours** |

> **Note:** These are initial research estimates of available data. For actual collected/merged dataset sizes used in training, see [03_phase1_development_plan.md](phase1_development_plan.md) Section 3.1 and `data/README.md`.

**Annotation with SAM 3 + RF-DETR-L + MoveNet:** 85-125 hours → **15-25 hours** (70-85% reduction)

---

## 3. Hardware Performance Benchmarks

### 3.1 GPU Training Performance

| Hardware | GPU Memory | TFLOPS | Cost/hr | Recommended For |
|---|---|---|---|---|
| **RTX 4090** | 24GB | 82.6 (FP32) | $0.40-1.00 | **All models** |
| RTX 3090 | 24GB | 35.6 (FP32) | $0.30-0.70 | Alternative |
| A100 | 40GB | 312 (TF16) | $2.00-3.00 | Large batch training |
| V100 | 16GB | 14 (TF32) | $1.00-1.50 | Not recommended |

**Primary:** Local GPU (remote PC). **Overflow:** Google Colab Pro (< 6h jobs).

### 3.2 Edge Deployment Performance (Hailo-8 NPU)

| Model | Input Size | Parameters | FPS (Hailo-8) | Power | Notes |
|---|---|---|---|---|---|
| YOLOX-T | 640×640 | 5.1M | 60-80 | 2.5-5W | Fastest, person detection |
| YOLOX-S | 640×640 | 9.0M | 50-70 | 2.5-5W | Fast |
| YOLOX-M | 640×640 | 25.3M | 30-50 | 3-5W | **Production choice** |
| YOLOX-L | 640×640 | 54.2M | 20-35 | 4-6W | High accuracy |

> **Note:** FPS figures are real-world estimates on single-lane PCIe (typical edge deployment). Official Hailo benchmarks on 4-lane PCIe show 2-4× higher FPS but are not representative of edge device configurations.

**Multi-Model Performance:**

| Models | Combined FPS | Power | Feasibility |
|---|---|---|---|
| 1 model | 30-80 | 2.5-5W | ✅ Easy |
| 2 models | 15-40 | 3-6W | ✅ Feasible |
| 3 models | 10-25 | 4-7W | ⚠️ May need frame skipping |
| 4+ models | 5-15 | 5-8W | ⚠️ Needs optimization (frame skip, model pruning) |

**12W Power Budget Breakdown:**

| Component | Power | Notes |
|---|---|---|
| Hailo-8 NPU (3-4 models) | 5-8W | Main inference (peak load) |
| CPU (RK3588) | 3-5W | Preprocessing, tracking |
| Memory | 1-2W | Model storage, buffers |
| Networking | 0.5-1W | Ethernet, PoE |
| Other | 0.5-1W | USB, peripherals |
| **TOTAL** | **10-17W** | **Likely exceeds 12W with 4+ models — optimization required** |

**Optimization Strategies:**
- Use YOLO11n/26n (smaller models)
- Frame skipping (process every 2nd frame for non-critical models)
- Model pruning (remove 20-30% parameters)
- INT8 quantization (default, 4x memory reduction)

### 3.3 Alternative Edge Hardware

| Hardware | TOPS | Power | Cost | Performance/W | Notes |
|---|---|---|---|---|---|
| **Hailo-8** | 26 | 2.5W | $249 | 10.4 | **RECOMMENDED** |
| Hailo-8L | 13 | 1.5W | ~$150 | 8.7 | Lower power option |
| Rockchip RK3588 | 6 (NPU) | 8W (SoC) | ~$80 | 0.75 | Integrated solution |
| Google Coral TPU | 4 | 2W | $60 | 2.0 | Limited performance |
| Renesas RZ/V2H | 80-100 | ~10W | TBD | 8-10 | High performance |

---

## 4. Recommendations & Decision Matrix

### 4.1 Model Architecture Recommendations

| Model | Architecture | Input Size | Justification |
|---|---|---|---|
| **Fire Detection** | YOLOX-M (Apache 2.0) | 640 + 1280 | Proven accuracy, 1280px for long-range |
| **Helmet Detection** | YOLOX-M (Apache 2.0) | 640 + 1280 | Edge-optimized, 1280px for small hats |
| **Safety Shoes** | YOLOX-M (Apache 2.0) | 640 + 1280 | Best occlusion handling |
| **Fall Detection** | YOLOX-M (Apache 2.0, classification) | 640 | Direct fallen_person classification |
| **Poketenashi** | YOLOX-M (phone) + YOLOX-T (person) | 640 | Phone detection + person tracking |
| **Zone Intrusion** | YOLOX-T (Apache 2.0, pretrained) | 640 | No training needed, end-to-end inference |

> **License Strategy:** All models use Apache 2.0 license — **$0 licensing cost** for commercial deployment.
> **Edge Constraint:** MoveNet is edge-deployable via TensorFlow Lite (designed for mobile). Used for fall detection pose + poketenashi keypoint rules.

### 4.2 Dataset Strategy Recommendations

| Priority | Action | Timeline |
|---|---|---|
| **HIGH** | Download FASDD, Construction-PPE, COCO Keypoints, FPI-Det | Week 1 |
| **HIGH** | Collect Nitto hat custom data (500-1,000 images) | Week 1-2 |
| **HIGH** | Collect Poketenashi staged data (4,000+ images) | Week 2-3 |
| **MEDIUM** | Collect safety shoes occlusion data (1,000+ images) | Week 2-3 |
| **MEDIUM** | Collect staged fall data (200-300 sequences) | Week 2-3 |
| **LOW** | Collect fire custom data (1,000-2,000 images) | Week 3-4 (optional) |

### 4.3 Risk Mitigation Recommendations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| **Nitto hat Precision < 0.75 or Recall < 0.72** | High | High | Collect 1,000+ examples; use 1280px; consider separate model |
| **Safety shoes Precision < 0.88 (occlusion)** | Medium | High | Use two-stage detection; add temporal tracking |
| **Poketenashi data bottleneck** | High | High | Use SAM 3; prioritize critical violations; staged collection |
| **Fall Detection FN Rate > 2%** | Medium | Critical | Life-safety priority — add more fall sequences; tune recall-first thresholds |
| **Power budget > 12W** | Low | High | Use smaller models; frame skipping; INT8 quantization |
| **YOLO AGPL-3.0 license** | Low | High | Switch to RT-DETRv2-S / D-FINE-S (Apache 2.0) if needed |

### 4.4 Go/No-Go Decision Points

> **Note:** These decision points were from initial research. For the finalized decision gate framework (DG1-DG5), see [03_phase1_development_plan.md](phase1_development_plan.md) Section 5.1.2.

> **Metric hierarchy:** Primary acceptance = Precision, Recall, FP Rate, FN Rate. Secondary tracking = mAP@0.5.

#### Decision Point 1: Nitto Hat Detection (End of Week 2)

**Primary Criteria (Acceptance):**
- Nitto hat Precision ≥ 0.75 AND Recall ≥ 0.72? → **GO** (continue)
- Nitto hat Precision < 0.75 OR Recall < 0.72? → Re-evaluate:
  - Collect 500 more examples
  - Try 1280px input
  - Consider separate Nitto hat model
  - If still below targets after 2 iterations → **NO-GO** (merge with "no_helmet" class)

**Secondary Tracking:** mAP@0.5 ≥ 0.85 (internal benchmark; not a Go/No-Go trigger)

#### Decision Point 2: Safety Shoes Occlusion (End of Week 3)

**Primary Criteria (Acceptance):**
- Overall Precision ≥ 0.88 AND Recall ≥ 0.85? → **GO** (continue)
- Occluded shoes Precision < 0.75 OR Recall < 0.72? → Re-evaluate:
  - Implement two-stage detection
  - Add temporal tracking
  - If still below targets after 1 iteration → **NO-GO** (accept lower accuracy for occluded class)

**Secondary Tracking:** mAP@0.5 ≥ 0.82 (internal benchmark)

#### Decision Point 3: Poketenashi Data Collection (End of Week 3)

**Criteria:**
- Annotation ≥ 70% complete? → **GO** (continue)
- Annotation < 70% complete? → Re-evaluate:
  - Prioritize critical violations (phone, no handrail)
  - Defer hands-in-pockets to Phase 2
  - Reduce dataset size to 3,000 minimum

#### Decision Point 4: Fall Detection Safety Validation (End of Week 5)

**Primary Criteria (Life-Safety):**
- Recall ≥ 0.88 AND FN Rate < 2%? → **GO** (life safety validated)
- Recall < 0.88 OR FN Rate > 2%? → Re-evaluate:
  - Collect more fall sequences (prioritize diversity)
  - Lower confidence threshold (trade FP for FN improvement)
  - Add temporal consistency (multi-frame confirmation)
  - If FN Rate > 5% after 2 iterations → **ESCALATE** (consult customer on acceptable risk)

**Secondary Tracking:** mAP@0.5 ≥ 0.85 (internal benchmark)

#### Decision Point 5: Power Budget (End of Week 7)

**Criteria:**
- All models run at ≥ 10 FPS within 12W? → **GO** (deployment ready)
- Power > 12W or FPS < 10? → Re-evaluate:
  - Use smaller models (YOLO11n, YOLO26n)
  - Implement frame skipping
  - Limit concurrent models to 3

### 4.5 Success Metrics Summary

> See [03_phase1_development_plan.md](phase1_development_plan.md) Section 4.0 for the finalized acceptance metrics (primary and secondary) and operational targets. All targets are technical team proposals — customer has not specified quantitative targets.

---

## 5. Sources & References

### Datasets (Verified)

- **FASDD**: Flame And Smoke Detection Dataset (120K+ images) — Wuhan University
  - GitHub: [github.com/openrsgis/FASDD](https://github.com/openrsgis/FASDD)
  - FASDD_CV on Kaggle: [kaggle.com/dataset_store/yuulind/fasdd-cv-coco](https://www.kaggle.com/dataset_store/yuulind/fasdd-cv-coco)
- **D-Fire**: Fire and Smoke Detection (21K+ images, YOLO format) — [github.com/gaiasd/DFireDataset](https://github.com/gaiasd/DFireDataset)
- **Construction-PPE**: Ultralytics Official Dataset (1,416 images, 11 classes) — [docs.ultralytics.com/dataset_store/detect/construction-ppe](https://docs.ultralytics.com/dataset_store/detect/construction-ppe/)
- **SHWD**: Safety-Helmet-Wearing-Dataset (7,581 images) — GitCode/Gitee
- **SH17**: Manufacturing PPE Dataset (8,099 images, 75,994 instances) — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S266644962400077X)
- **COCO Keypoints**: COCO Dataset (200K+ images) — [cocodataset.org](https://cocodataset.org)
- **FPI-Det**: Face-Phone Interaction Dataset (22,879 images) — [github.com/KvCgRv/FPI-Det](https://github.com/KvCgRv/FPI-Det) | Paper: [arXiv:2509.09111](https://arxiv.org/abs/2509.09111)
- **Le2i Fall Detection**: ~130 sequences — Université de Bourgogne (contact lab)
- **UR Fall Detection**: 70 sequences — University of Rzeszow

### Models & Frameworks

- **Ultralytics YOLO26**: Released Jan 14, 2026 — [docs.ultralytics.com/models/yolo26](https://docs.ultralytics.com/models/yolo26/) | Paper: [arXiv:2509.25164](https://arxiv.org/abs/2509.25164)
- **Ultralytics YOLO11**: [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **Hailo-8**: [hailo.ai](https://hailo.ai) | Community benchmarks: [community.hailo.ai](https://community.hailo.ai)
- **Label Studio**: [label-studio.com](https://label-studio.com)

### Research Papers (2024-2026)

- YOLO26: Key Architectural Enhancements and Performance Benchmarking — [arXiv:2509.25164](https://arxiv.org/abs/2509.25164)
- Ultralytics YOLO Evolution (YOLO26/11/v8/v5 overview) — [arXiv:2510.09653](https://arxiv.org/abs/2510.09653)
- FPI-Det: Face-Phone Interaction Dataset — [arXiv:2509.09111](https://arxiv.org/abs/2509.09111)
- D-Fire: An automatic fire detection system — Neural Computing and Applications, 2022
- Various IEEE Access, MDPI, and Springer publications on fire/smoke detection, PPE detection, fall detection, YOLO optimizations

### Notes

- **Search dates**: March 5, 2026
- **Coverage**: Focus on 2024-2026 state-of-the-art
- **Verified items**: YOLO26 specs, FASDD, D-Fire, FPI-Det, Construction-PPE size, Hailo-8 benchmarks
- **Limitations**: Some niche datasets (Nitto hats, hands-in-pockets, handrail) have no public availability
- **Recommendation**: Combine public datasets with targeted custom data collection
- **Hailo-8 FPS caveat**: Official benchmarks use 4-lane PCIe; real-world edge devices (single-lane) achieve ~50% of official figures

---

**Document Version:** 3.0
**Last Updated:** March 7, 2026
**Author:** Vietsol AI Technical Team
**Status:** Research Complete — Verified & Updated with Primary Acceptance Metrics

# Phase 1 Executive Summary

**Factory Smart Camera AI System** | Safety Action Detection via Edge AI
**Date:** March 7, 2026 | **Timeline:** 12 Weeks | **Status:** Planning Complete

---

## 1. Project Overview

We are building an AI-powered safety monitoring system that transforms existing factory IP cameras into smart cameras capable of detecting unsafe worker actions in real time. Phase 1 covers 6 detection use cases: fire, helmet compliance, safety shoes, fall detection, poketenashi violations (hands in pockets, phone while walking, no handrail), and restricted area intrusion.

**Target Hardware:** TBD — NPU-based edge device (similar to Hailo-8 or RK3588, <12W power)
**License Strategy:** YOLOX models (Apache 2.0) — free commercial use, no licensing fees

---

## 2. Timeline

| Week | Phase | Milestone |
|------|-------|-----------|
| 1-2 | Setup & Data Transformation | Environment ready, **data transformation pipelines** on all open datasets (prepare_*.py scripts), labeling started for **nitto_hat** and **safety shoes** |
| 3-4 | v1 Training | First models trained on curated subsets |
| 5-6 | v2 Training | Expanded datasets, **midpoint review**, evaluate multi-head model option* |
| 7-8 | v3 Training | Full pipeline + customer data integration (if available) |
| 9 | Export & Handoff | ONNX export, final acceptance, documentation delivered |
| 10-12 | Buffer | Contingency for escalations |

> **Week 1-2 Detail:**
> - **Data Transformation:** Run `prepare_*.py` scripts on all open datasets (fire, helmet, shoes, fall, phone) to unify class IDs, formats, and splits
> - **Labeling Start:** Begin collecting and labeling **nitto_hat** (custom class, no public data) and **safety shoes** factory images (supplement small 3.7K dataset)
> - **Pre-annotation:** SAM 3 + RF-DETR-L reduces manual labeling effort by 70-85%

> *\*Week 5 checkpoint: Evaluate if multi-model inference is feasible on target hardware. If not, train single multi-head model (challenge: dataset imbalance).*

---

## 3. Resources

| Resource | Details |
|----------|---------|
| **Team** | 3 AI Engineers (E1, E2, E3) + Annotation Team (AT) |
| **GPU** | Shared local remote PC (priority) + Google Colab Pro (overflow) |
| **Tools** | Ultralytics (training), W&B (tracking), Label Studio (annotation), DVC (versioning) |
| **Annotation Tools** | SAM 3 + RF-DETR-L + MoveNet (pose) pre-annotation (reduces manual effort 70-85%) |

---

## 4. Difficulty Assessment

| Model | Difficulty | Why |
|-------|------------|-----|
| Fire Detection | **LOW** | Large dataset (122K images), straightforward detection |
| Helmet Detection | **LOW-MED** | Good data (62K), but custom "nitto_hat" class needs factory collection |
| Zone Intrusion | **LOW** | No training needed — uses pretrained person detection |
| Fall Detection | **HIGH** | Dual-model approach (MoveNet pose + classify), limited pose data (111 images) |
| Safety Shoes | **HIGH** | Small pixel area (feet), frequent occlusion, small dataset (3.7K) |
| Poketenashi | **HIGH** | Hybrid pipeline: detection + pose rules + multi-behavior logic |

---

## 5. Build vs Outsource Risk

| Model | Strategy | Rationale | Outsource Trigger |
|-------|----------|-----------|-------------------|
| Fire Detection | **Build** | Large dataset, straightforward | — |
| Helmet Detection | **Build** | Good data, standard detection | — |
| Zone Intrusion | **Build** | Uses pretrained model | — |
| Fall Detection | **Build** | Dual approach provides fallback | — |
| Poketenashi | **Build** | Custom rules require in-house expertise | — |
| Safety Shoes | **Build** (with fallback) | Highest risk model | If DG3 fails, consider outsourcing |

> **Default:** All models built in-house using config-driven pipeline. Only safety shoes has outsource fallback if all accuracy escalation paths fail at DG3.

---

## 6. Labeling Strategy

> **Not all models need custom labeling.** Some use pretrained weights, pose rules, or public datasets that are already labeled.

### 6.1 Per-Model Labeling Requirements

| Model | What to Label | Public Data (ready) | Custom Labeling Needed | Priority | Pre-Annotation |
|------|----------------|---------------------|------------------------|----------|----------------|
| **a** Fire | fire, smoke (bbox) | 122,525 images | 100-200 factory images | LOW | RF-DETR-L + SAM 3 |
| **b** Helmet | person, helmet, no_helmet, nitto_hat | 62,602 images | ~2,500 nitto_hat images | MEDIUM | RF-DETR-L + SAM 3 |
| **f** Shoes | person, safety_shoes, no_safety_shoes | 3,772 images | 2,000+ factory images | **HIGH** | SAM 3 (small objects) |
| **g-pose** Fall Pose | person (17 keypoints) | 111 + COCO 58K | ~1,000 verified (auto-annotated) | MEDIUM | MoveNet auto-annotate → review |
| **g-classify** Fall Classify | person, fallen_person | 17,383 images | 100-200 factory images | LOW | RF-DETR-L |
| **h** Phone | phone (bbox) | ~13,470 images | 200-400 factory images | LOW | RF-DETR-L + COCO "cell phone" |
| **h** Pose Rules | — | COCO-Pose pretrained | 100-200 calibration only (no annotation) | — | Pretrained MoveNet |
| **i** Zone Intrusion | — | COCO pretrained (person) | None | — | Pretrained YOLOX-T |

### 6.2 Total Labeling Effort

| Data Type | Quantity | With Pre-Annotation | Manual Only |
|-----------|----------|---------------------|-------------|
| Nitto hat (helmet model) | ~2,500 images | 15-20 hours | 60-80 hours |
| Safety shoes (factory) | 2,000+ images | 12-15 hours | 50-65 hours |
| Fall pose (auto-annotated, review only) | ~1,000 images | 8-12 hours | 40-50 hours |
| Factory-specific (fire, phone, fall classify) | 400-800 images | 3-5 hours | 10-20 hours |
| **Total** | ~6,000 images | **~40-50 hours** | ~160-215 hours |

> **Pre-annotation (RF-DETR-L + SAM 3 + MoveNet) reduces labeling effort by 70-85%.**

**References:**
- Labeling guide: [05_labeling_guide.md](labeling_guide.md)
- Dataset catalog: [../data/README.md](../data/README.md)

---

## 7. Technical Approach

| Strategy | Description |
|----------|-------------|
| **Default Model** | YOLOX-M (Apache 2.0, free commercial use) |
| **Custom Network Option** | Reuse YOLOX backbone + custom multi-task head (single model for all classes) |
| **Escalation Path** | YOLOX-L → RT-DETRv2-S / D-FINE-S (all Apache 2.0, free) |
| **Decision Gates** | DG1 (W4): If mAP < 0.75, switch to larger model or higher resolution |
| **Validation** | Two-tier: public test (internal tracking) + factory test (acceptance) |

> **License Strategy:** All models use Apache 2.0 license — **$0 licensing cost**. No AGPL-3.0 models (YOLO11/YOLO26) to avoid Enterprise License fees (~$5K/yr).

> **Custom Multi-Head Model:** If multi-model inference exceeds edge capacity, build custom network by reusing YOLOX backbone with custom detection head for all classes.
> - **Challenge:** Dataset imbalance between classes (fire: 122K vs shoes: 3.7K)
> - **Strategy:** Class-balanced sampling + weighted loss during training

> **Continuous Improvement (All Models):** Customer factory data enables fine-tuning for **all models** (fire, helmet, shoes, fall, phone, zone) to improve real-world accuracy.
> - **Workflow:** Add new images → Re-run training command → No code changes needed — pipeline is config-driven

---

## 8. Key Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Safety shoes data insufficient | HIGH | HIGH | Factory collection + aggressive augmentation + DG3 fallback |
| Domain gap (public data ≠ factory) | HIGH | HIGH | Factory validation once customer/similar factory data available |
| Fall pose model fails | MEDIUM | MEDIUM | Fallback to classification-only approach |
| GPU contention (3 engineers) | MEDIUM | MEDIUM | Job scheduling + Colab Pro overflow |
| Late factory footage access | MEDIUM | HIGH | Coordinate from Day 1; config-driven pipeline enables easy data extension once available |
| Multi-model inference exceeds edge capacity | MEDIUM | HIGH | Fallback: train single multi-head model (challenge: dataset imbalance) |

---

## 9. Deliverables

| Deliverable | Format |
|-------------|--------|
| **Trained Models** | `.pt` (original) + `.onnx` (edge deployment) |
| **Training Pipeline** | Config-driven (YAML), single-command reproducible, **no hardcoded values** — easy to extend with new datasets |
| **Dataset Pipeline** | Prepare scripts, class mappings, versioned via DVC |
| **Alert Logic** | Rule-based modules + zone configs (JSON) |
| **Documentation** | Model cards, training guide, deployment guide |
| **W&B Artifacts** | Full run history, production-tagged models |

---

## Quick Reference

| Metric | Target |
|--------|--------|
| Precision | ≥ 0.85-0.94 (per model) |
| Recall | ≥ 0.82-0.92 (per model) |
| False Positive Rate | < 2-5% |
| False Negative Rate | < 2-6% |
| Edge Power Consumption | < 12W |

> **Note:** All metrics are technical team proposals — customer verification required.

---

**For detailed engineering plans, see:** [03_phase1_development_plan.md](phase1_development_plan.md)

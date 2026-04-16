# features/CLAUDE.md — Phase 1 Development Plan

> Living document. Update status after each iteration. Use `[x]` for done, `[-]` for skipped/N/A.

---

## Phase 1 Feature Inventory

**Training mode key:**
- 🎯 **Fine-tune** — custom classes not in COCO; must collect data + train (always via transfer learning, never from scratch)
- 🔧 **Pretrained only** — vendor pretrained weights used as-is; enrollment or config only

| Folder | Type | Training Mode | Model | Status |
|---|---|---|---|---|
| `safety-fire_detection` | Detection | 🎯 Fine-tune | YOLOX / D-FINE | ⬜ not started |
| `safety-fall-detection` | Detection | 🎯 Fine-tune | YOLOX | ⬜ not started |
| `safety-fall_pose_estimation` | Pose keypoints | 🎯 Fine-tune | RTMPose-S | ⬜ not started |
| `safety-poketenashi` | Orchestrator | 🔧 Pretrained only | (no own model — delegates to sub-models + pose rules) | ⬜ not started |
| `safety-poketenashi-phone-usage` | Detection sub-model | 🎯 Fine-tune | YOLOX | ⬜ not started |
| `ppe-helmet_detection` | Detection | 🎯 Fine-tune | YOLOX / D-FINE | ⬜ not started |
| `ppe-shoes_detection` | Detection | 🎯 Fine-tune | YOLOX | ⬜ not started |
| `access-face_recognition` | Face recognition | 🔧 Pretrained only | InsightFace SCRFD + ArcFace | ⬜ not started |
| `access-zone_intrusion` | Detection + zone logic | 🔧 Pretrained only | COCO pretrained person/vehicle | ⬜ not started |

### Why fine-tune vs pretrained only

| Feature | Reason |
|---|---|
| `safety-fire_detection` | `fire`, `smoke` — not in COCO 80 |
| `safety-fall-detection` | `fallen_person` — COCO `person` is upright; fallen is a separate learned class |
| `safety-fall_pose_estimation` | Custom keypoint skeleton tuned for industrial fall angle detection |
| `safety-poketenashi-phone-usage` | `phone_usage` — COCO has `cell phone` object, not the act of using it while walking |
| `ppe-helmet_detection` | `head_with_helmet` / `head_without_helmet` — no PPE compliance classes in COCO |
| `ppe-shoes_detection` | `foot_with_safety_shoes` / `foot_without_safety_shoes` — not in COCO |
| `safety-poketenashi` | No training; orchestrates sub-models + runs pose rules on RTMPose output |
| `access-face_recognition` | Universal ArcFace encoder; only needs gallery enrollment (no retraining) |
| `access-zone_intrusion` | COCO `person` class is sufficient; zone logic is pure polygon geometry |

Status icons: ⬜ not started · 🔄 in progress · ✅ done · ⏸ blocked · ❌ skipped

---

## Training Strategy

### Transfer Learning (mandatory for all 🎯 fine-tune features)

We never train from scratch. Every fine-tuned model follows a two-step process:

**Step 1 — Benchmark pretrained candidates**

Before committing to a backbone, run a short benchmark across candidate architectures on the target dataset. Evaluate zero-shot or few-epoch mAP to identify which pretrained backbone generalizes best to this domain.

Candidate pool per task:

| Task | Candidates to benchmark |
|---|---|
| Detection (fire, fall, PPE, phone) | YOLOX-S, YOLOX-M, D-FINE-S, D-FINE-M, RT-DETRv2-R18 |
| Pose keypoints (fall pose) | RTMPose-S, RTMPose-M (via MMPose pretrained COCO) |

Selection criteria: highest mAP on val split after 5–10 epochs of head-only fine-tuning. Do not select solely on COCO benchmark numbers — domain gap matters more than COCO rank.

**Step 2 — Full fine-tune on the winning backbone**

1. Load COCO pretrained weights
2. Freeze backbone, train head/neck for N epochs (warm-up phase)
3. Unfreeze all layers, train with lower LR (full fine-tune phase)
4. Evaluate final checkpoint on test split

Config knobs in `06_training.yaml`:
```yaml
training:
  freeze_backbone_epochs: 5    # head-only warm-up
  lr: 0.001                    # initial LR (head warm-up)
  lr_backbone: 0.0001          # backbone LR after unfreeze
  pretrained: true             # always true — load COCO weights
```

---

## Future: Unified Multi-Task Model (Post Phase 1)

After individual models are trained and validated, we plan to develop a single shared-backbone model that covers as many use cases as possible in one forward pass.

**Architecture concept:**

```
Shared frozen backbone (e.g. D-FINE-M or RT-DETRv2 backbone, pretrained on COCO)
        │
        ├── Detection head A  →  fire + smoke
        ├── Detection head B  →  helmet, shoes (PPE)
        ├── Detection head C  →  phone_usage, fallen_person
        └── Pose head         →  keypoints → poketenashi rules
```

**Why:**
- One backbone inference pass instead of 4–6 separate model calls
- Lower GPU memory footprint at deployment
- Single ONNX export for edge devices (Jetson, Hailo)

**How — freeze backbone, fine-tune task heads only:**
1. Start from the best-performing backbone identified in Phase 1 benchmarking
2. Freeze all backbone weights
3. Add and train task-specific heads/necks on combined multi-task dataset
4. Per-task loss weighting to prevent interference between tasks

**Risks and tradeoffs:**

| Risk | Mitigation |
|---|---|
| Task interference (diverse tasks hurt each other's mAP) | Loss weighting + gradient surgery; fall back to separate heads if mAP drops >3% |
| Backbone becomes a single point of failure | Keep individual Phase 1 models as fallback; unified model is additive, not replacement |
| Hard to update one task without retraining all heads | Modular head design — each head can be detached and retrained independently |
| Pose + detection on one backbone may conflict | Use separate neck branches per task family (detection neck vs pose neck) |

**Prerequisites before starting:**
- All 6 fine-tuned Phase 1 models trained and evaluated
- Per-task mAP baselines locked (unified model must match or exceed these)
- Unified dataset split created (combined from all feature `training_ready/` dirs)

This is a **Phase 2** item. Do not start until Phase 1 individual models are stable.

---

## Architecture Notes

### Poketenashi (prohibited / required action suite)

One umbrella feature (`safety-poketenashi`) orchestrates 5 behaviors via two backends:

| Behavior | Backend | Sub-folder |
|---|---|---|
| `phone_usage` | ML detection | `safety-poketenashi-phone-usage` (trained separately) |
| `hands_in_pockets` | Pose rule | `safety-poketenashi/code/hands_in_pockets_detector.py` |
| `stair_diagonal` | Pose + tracking rule | `safety-poketenashi/code/stair_safety_detector.py` |
| `no_handrail` | Pose + zone rule | `safety-poketenashi/code/handrail_detector.py` |
| `no_pointing_calling` | Pose rule | `safety-poketenashi/code/pointing_calling_detector.py` |

Pose backend = `safety-fall_pose_estimation` RTMPose-S weights (shared).
Alert thresholds and frame windows are in `safety-poketenashi/configs/10_inference.yaml`.

### Zone Intrusion

Uses COCO-pretrained person/vehicle detector — no custom training needed. Only config:
- Define site polygon zones in `access-zone_intrusion/configs/10_inference.yaml`

### Face Recognition

Enrollment workflow (run once per person per site):
```bash
uv run core/p10_inference/face_enroll.py \
  --config features/access-face_recognition/configs/face.yaml \
  --gallery data/face_gallery/demo.npz \
  --images features/access-face_recognition/samples/
```

---

## Per-Feature Pipeline Checklists

⚠️ **`06_training.yaml` does not exist yet in any feature folder** — must be created before training can start. This is the immediate next blocker after data prep.

### safety-fire_detection

- [x] `00_data_preparation.yaml` — sources locked, class map verified
- [x] `p00_data_prep` — 17,373 imgs (fire 53.7% / smoke 46.3%), DATASET_REPORT ✅
- [x] `p02_annotation_qa` — 95.1% good / 1.1% bad → ✅ ACCEPT; LS project 13
- [ ] **Benchmark pretrained candidates** — 5–10 epoch head-only run (YOLOX-M, D-FINE-M, RT-DETRv2-R18), pick best backbone (see Training Strategy above)
- [ ] `06_training.yaml` — set winning arch, `pretrained`, `freeze_backbone_epochs`, `lr_backbone`
- [ ] `p06_training` — train, monitor mAP
- [ ] `p08_evaluation` — evaluate on held-out test split
- [ ] `p09_export` — ONNX export
- [ ] `release/` — tag and package via `utils/release.py`

### safety-fall-detection

- [x] `00_data_preparation.yaml` — sources locked
- [x] `p00_data_prep` — 12,402 imgs (person 62.4% / fallen_person 37.6%), DATASET_REPORT ✅
- [x] `p02_annotation_qa` — 90.6% good / 0.2% bad → ✅ ACCEPT; LS project 16
- [ ] **Benchmark pretrained candidates** — YOLOX-S, YOLOX-M, D-FINE-S
- [ ] `06_training.yaml`
- [ ] `p06_training`
- [ ] `p08_evaluation`
- [ ] `p09_export`
- [ ] `release/`

### safety-fall_pose_estimation

- [ ] Confirm RTMPose-S pretrained weights in `pretrained/safety-fall_pose_estimation/`
- [ ] `00_data_preparation.yaml` — COCO keypoint sources
- [ ] `p00_data_prep`
- [ ] `p02_annotation_qa`
- [ ] **Benchmark pretrained candidates** — RTMPose-S vs RTMPose-M (head-only runs)
- [ ] `06_training.yaml` — keypoint task, winning RTMPose arch
- [ ] `p06_training`
- [ ] `p08_evaluation`
- [ ] `p09_export`
- [ ] `release/`

### safety-poketenashi (orchestrator — no model training)

- [ ] `code/hands_in_pockets_detector.py` — implement pose rule
- [ ] `code/stair_safety_detector.py` — implement trajectory angle rule
- [ ] `code/handrail_detector.py` — implement hand-to-zone rule
- [ ] `code/pointing_calling_detector.py` — implement arm extension rule
- [ ] `code/orchestrator.py` — wire sub-models + pose rules + zone logic
- [ ] Smoke test against `samples/` with all 5 behaviors
- [ ] Tune `10_inference.yaml` frame windows + thresholds from smoke test

### safety-poketenashi-phone-usage

- [x] `00_data_preparation.yaml` — sources locked
- [x] `p00_data_prep` — 22,975 imgs (phone_usage 94.6% / person 5.4%), DATASET_REPORT ✅
- [x] `p02_annotation_qa` — 90.6% good / 5.4% bad → ⚠️ borderline ACCEPT; SAM3 struggles with action-class semantics on small phone bboxes; LS project 17
- [ ] **Benchmark pretrained candidates** — YOLOX-S, YOLOX-M, D-FINE-S
- [ ] `06_training.yaml`
- [ ] `p06_training`
- [ ] `p08_evaluation`
- [ ] `p09_export`
- [ ] `release/`

### ppe-helmet_detection

- [x] `00_data_preparation.yaml` — sources locked; 4 classes: person, head_with_helmet, head_without_helmet, head_with_nitto_hat
- [x] `p00_data_prep` — 22,323 imgs (head_with_helmet 74% / head_without_helmet 21% / person 3% / head_with_nitto_hat 1.6%), DATASET_REPORT ✅
- [x] `p02_annotation_qa` — 94.7% good / 2.4% bad → ✅ ACCEPT; LS project 14
- [ ] **Benchmark pretrained candidates** — YOLOX-M, D-FINE-M, RT-DETRv2-R18
- [ ] `06_training.yaml`
- [ ] `p06_training`
- [ ] `p08_evaluation`
- [ ] `p09_export`
- [ ] `release/`

### ppe-shoes_detection

- [x] `00_data_preparation.yaml` — sources locked
- [x] `p00_data_prep` — 37,026 imgs (foot_with_safety_shoes 68% / foot_without_safety_shoes 29% / person 2.4%), DATASET_REPORT ✅
- [x] `p02_annotation_qa` — 88.5% good / 1.8% bad → ✅ ACCEPT; LS project 15
- [ ] **Benchmark pretrained candidates** — YOLOX-S, YOLOX-M (no domain-specific base exists)
- [ ] `06_training.yaml`
- [ ] `p06_training`
- [ ] `p08_evaluation`
- [ ] `p09_export`
- [ ] `release/`

### access-face_recognition

- [ ] `configs/face.yaml` — enrollment + similarity threshold set
- [ ] Enrollment run on site samples
- [ ] Smoke test via `app_demo` face tab
- [ ] `release/` — package gallery + config

### access-zone_intrusion

- [ ] `configs/10_inference.yaml` — polygon zones defined for target site
- [ ] Smoke test via `app_demo` zone tab
- [ ] No training needed (COCO pretrained)

---

## Recommended Sequence (GPU-aware)

GPU 2 has ~28 GB — run one training job at a time to avoid OOM.

**Phase A — Data prep (all 5 ML features in parallel):**
```
safety-fire_detection / ppe-helmet_detection / ppe-shoes_detection /
safety-fall-detection / safety-poketenashi-phone-usage
```
Run `p00_data_prep` → `p02_annotation_qa` for each. Can run concurrently (CPU-bound).

**Phase B — Training (sequential, one at a time on GPU 2):**
1. `safety-fire_detection` (largest, most data)
2. `ppe-helmet_detection`
3. `ppe-shoes_detection`
4. `safety-fall-detection`
5. `safety-poketenashi-phone-usage` (smallest)
6. `safety-fall_pose_estimation` (keypoints — after detection models done)

**Phase C — Inference / config only (no GPU needed):**
- `safety-poketenashi` pose rule modules
- `access-face_recognition` enrollment
- `access-zone_intrusion` zone polygon config

**Phase D — Eval + export + release (all features):**
- `p08_evaluation` → `p09_export` → `utils/release.py` for each trained model

---

## Iteration Log

### Iteration 2 — 2026-04-17

**Phase A complete — all 5 ML features data-ready.**

| Feature | Dataset | Images | QA | LS project |
|---|---|---|---|---|
| safety-fire_detection | fire_detection | 17,373 | 95.1% good / 1.1% bad ✅ | id=13 |
| ppe-helmet_detection | helmet_detection | 22,323 | 94.7% good / 2.4% bad ✅ | id=14 |
| ppe-shoes_detection | shoes_detection | 37,026 | 88.5% good / 1.8% bad ✅ | id=15 |
| safety-fall-detection | fall_detection | 12,402 | 90.6% good / 0.2% bad ✅ | id=16 |
| safety-poketenashi-phone-usage | safety_poketenashi_phone_usage | 22,975 | 90.6% good / 5.4% bad ⚠️ | id=17 |

- [x] `ppe-helmet_detection` class decision: `person` retained (source datasets have person labels); `head_with_nitto_hat` kept at 1.6% — site_collected will strengthen it
- [x] `ppe-shoes_detection` `person` at 2.4% — accepted; foot-centric datasets naturally lack person bboxes
- [x] `safety-poketenashi-phone-usage` borderline QA — SAM3 action-semantic limitation on small phone bboxes; re-label only if post-training mAP is low
- [x] All 5 datasets loaded into Label Studio at http://localhost:18103 for human review before training

**Next (Phase B):** Backbone benchmarking — 5–10 epoch head-only runs per feature on GPU 2. Start with `safety-fire_detection`.

---

### Iteration 1 — 2026-04-17

- [x] p00 DATASET_REPORT: `tiny` bbox tier added (w×h < 0.000479, <14² px on 640 px); `small` adjusted to 14²–32² px range
- [x] p02 `run_qa.py`: auto-appends Label Quality section to feature `DATASET_REPORT.md` after each QA run; re-run replaces section only
- [x] p02 `pipeline.py`: `sam3.include_missing_detection` wired from shared config (default `false` to prevent FP on class-restricted datasets)
- [x] `cv-dataset-prep` skill: sampling size hint + missing-detection flag documented + step 9 handoff strengthened

~~**Next:** run `p00_data_prep` → `p02_annotation_qa` for all 5 ML features (Phase A — data prep).~~ → Done in Iteration 2.

---

### Iteration 0 — 2026-04-16

- [x] All 5 ML feature `00_data_preparation.yaml` configs authored with `license/notes/dropped_classes/held_back`
- [x] `DATASET_REPORT.md` generator rewritten (8 sections, per-split breakdown, class mapping, imbalance bar)
- [x] `cv-dataset-prep` skill updated to lean command+decision format; template + references updated
- [x] `features/README.md` restructured for Phase 1 scope
- [x] `features/safety-poketenashi/configs/05_data.yaml` — 6-class umbrella config
- [x] `features/safety-poketenashi/configs/10_inference.yaml` — full alert + pose_rules config
- [x] `app_demo/config/config.yaml` — all Phase 1 tabs + use_cases verified
- [ ] `training_ready/` — empty; all 5 datasets need `p00_data_prep` run
- [ ] No trained weights yet

**Blockers:** None — data prep can start immediately.

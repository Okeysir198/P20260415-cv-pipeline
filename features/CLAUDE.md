# features/CLAUDE.md — Phase 1 Development Plan

> Living document. Update status after each iteration. Use `[x]` for done, `[-]` for skipped/N/A.
> Detailed benchmark results and per-feature checklists live in each feature's own `CLAUDE.md`.

---

## Phase 1 Feature Inventory

**Training mode key:**
- 🎯 **Fine-tune** — custom classes not in COCO; must collect data + train (always via transfer learning, never from scratch)
- 🔧 **Pretrained only** — vendor pretrained weights used as-is; enrollment or config only

| Folder | Type | Mode | Best Pretrained | Pretrained mAP50 | Status |
|---|---|---|---|---|---|
| `safety-fire_detection` | Detection | 🎯 Fine-tune | RT-DETRv2-R18 (fine-tuned) | 0.541 (10% data) | 🔄 arch selected, full train pending |
| `safety-fall-detection` | Detection | 🎯 Fine-tune | yolov11_fall_melihuzunoglu.pt | 0.050 | ⬜ not started |
| `safety-fall_pose_estimation` | Pose keypoints | 🎯 Fine-tune | dwpose_384_pose (ONNX, interim) | — | ⬜ not started |
| `safety-poketenashi` | Orchestrator | 🔧 Pretrained only | dwpose_384_pose (det_rate=1.0) | — | 🔄 pipelines done |
| `safety-poketenashi-phone-usage` | Detection sub-model | 🎯 Fine-tune | none (action class) | 0.000 | ⬜ not started |
| `ppe-helmet_detection` | Detection | 🎯 Fine-tune | melihuzunoglu_yolov11_ppe.pt | 0.105 | ⬜ not started |
| `ppe-shoes_detection` | Detection | 🎯 Fine-tune | none (no foot detector) | 0.000 | ⬜ not started |
| `access-face_recognition` | Face recognition | 🔧 Pretrained only | yunet + sface (rank-1=1.0) | — | 🔄 pipelines done |
| `access-zone_intrusion` | Detection + zone logic | 🔧 Pretrained only | yolox_tiny (acc=1.0, 6.9ms) | — | 🔄 pipelines done |

Status icons: ⬜ not started · 🔄 in progress · ✅ done · ⏸ blocked · ❌ skipped

---

## Training Strategy

### Transfer Learning (mandatory for all 🎯 fine-tune features)

We never train from scratch. Every fine-tuned model follows a two-step process:

**Step 1 — Benchmark pretrained candidates** (✅ done for all features — see per-feature CLAUDE.md)

**Step 2 — Full fine-tune on the winning backbone**

1. Load best pretrained weights (see inventory table above)
2. Freeze backbone, train head/neck for N epochs (warm-up phase)
3. Unfreeze all layers, train with lower LR (full fine-tune phase)
4. Evaluate final checkpoint on test split

Config knobs in `06_training.yaml`:
```yaml
training:
  freeze_backbone_epochs: 5    # head-only warm-up
  lr: 0.001                    # initial LR (head warm-up)
  lr_backbone: 0.0001          # backbone LR after unfreeze
  pretrained: true             # always true — load best pretrained weights
```

---

## Recommended Sequence (GPU-aware)

GPU 2 has ~28 GB — run one training job at a time to avoid OOM.

**Phase A — Data prep:** ✅ Complete (all 5 ML features)

**Phase B — Training (sequential, one at a time on GPU 2):**
1. `safety-fire_detection` — 🔄 `06_training.yaml` done (gpu_augment enabled, 3 arch configs); ready to train
2. `ppe-helmet_detection` — 4 classes, start from melihuzunoglu_yolov11_ppe.pt
3. `safety-fall-detection` — specialized class, start from yolov11_fall_melihuzunoglu.pt
4. `ppe-shoes_detection` — largest dataset (37k imgs), COCO backbone only
5. `safety-poketenashi-phone-usage` — action class, COCO backbone only
6. `safety-fall_pose_estimation` — keypoints, after detection models done

**Phase C — Config only (no GPU needed):** ✅ Pipelines implemented
- `safety-poketenashi` — pose rule modules + orchestrator done
- `access-face_recognition` — enrollment pipeline done
- `access-zone_intrusion` — zone detector done

**Phase D — Eval + export + release (all features):**
- `p08_evaluation` → `p09_export` → `utils/release.py` for each trained model

---

## Future: Unified Multi-Task Model (Phase 2)

After individual models are trained and validated, develop a single shared-backbone model covering all use cases in one forward pass.

```
Shared frozen backbone (D-FINE-M or RT-DETRv2)
        ├── Detection head A  →  fire + smoke
        ├── Detection head B  →  helmet, shoes (PPE)
        ├── Detection head C  →  phone_usage, fallen_person
        └── Pose head         →  keypoints → poketenashi rules
```

Do not start until all Phase 1 individual models are stable and mAP baselines are locked.

---

## Iteration Log

### Iteration 5 — 2026-04-18

Arch comparison for `safety-fire_detection` on 10% data (1,737 imgs), 15 epochs. RT-DETRv2-R18 wins decisively.

| Arch | best val/mAP50 | Notes |
|---|---|---|
| **RT-DETRv2-R18** | **0.541** (ep 15, still rising) | Winner — use `06_training_rtdetr.yaml` |
| D-FINE-S | 0.190 (ep 9, plateau) | `amp: false` required (fp16 NaN crash) |
| YOLOX-M | 0.113 (ep 73, early stop) | Previous run |

Max safe batch size on RTX 5090 (28 GB free, fp32): **bs=32** (14.7 GB peak).
Next: full training — `06_training_rtdetr.yaml`, bs=32, 150 epochs, 100% dataset.

---

### Iteration 4 — 2026-04-17

Full re-run of all 8 benchmark scripts. All exit 0. Results stable. See per-feature CLAUDE.md for full tables.

| Feature | Best Model | mAP50 / Metric |
|---|---|---|
| access-zone_intrusion | yolox_tiny | acc=1.0, F1=1.0 |
| access-face_recognition | yunet + sface_fp32 | rank-1=1.0 |
| safety-poketenashi | dwpose_384_pose | det_rate=1.0, 13ms |
| safety-fall-detection | yolov11_fall_melihuzunoglu.pt | mAP50=0.050 |
| ppe-shoes_detection | rfdetr_small (person only) | mAP50=0.000 |
| safety-poketenashi-phone-usage | yolox_s/m (COCO) | mAP50=0.000 |
| safety-fire_detection | SalahALHaismawi_yolov26 | mAP50=0.153 |
| ppe-helmet_detection | HudatersU.onnx / melihuzunoglu.pt | mAP50=0.124/0.105 |

**Next (Phase B):** Create `06_training.yaml` for each fine-tune feature and begin training. Priority: fire → helmet → fall → shoes → phone-usage.

---

### Iteration 3 — 2026-04-17

Pretrained model benchmark complete (all 9 features). Inference pipelines implemented for 3 pretrained-only features (`access-zone_intrusion`, `access-face_recognition`, `safety-poketenashi`).

- All `features/<feature>/code/benchmark.py` scripts written and run
- `ZoneIntrusionDetector`, `FaceRecognitionPipeline`, `PoketanashiOrchestrator` + 4 rule modules implemented
- Results written to `features/<feature>/eval/`

---

### Iteration 2 — 2026-04-17

Phase A complete — all 5 ML features data-ready.

| Feature | Images | QA | LS project |
|---|---|---|---|
| safety-fire_detection | 17,373 | 95.1% good ✅ | id=13 |
| ppe-helmet_detection | 22,323 | 94.7% good ✅ | id=14 |
| ppe-shoes_detection | 37,026 | 88.5% good ✅ | id=15 |
| safety-fall-detection | 12,402 | 90.6% good ✅ | id=16 |
| safety-poketenashi-phone-usage | 22,975 | 90.6% good ⚠️ | id=17 |

---

### Iteration 1 — 2026-04-17

- p00 DATASET_REPORT: `tiny` bbox tier added; `small` range adjusted
- p02 `run_qa.py`: auto-appends Label Quality section to feature `DATASET_REPORT.md`
- p02 `pipeline.py`: `sam3.include_missing_detection` wired from shared config

---

### Iteration 0 — 2026-04-16

- All 5 ML feature `00_data_preparation.yaml` configs authored
- `DATASET_REPORT.md` generator rewritten (8 sections)
- `features/README.md` restructured for Phase 1 scope
- `safety-poketenashi/configs/05_data.yaml` + `10_inference.yaml` created
- `app_demo/config/config.yaml` — all Phase 1 tabs verified

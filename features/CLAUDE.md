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
| `safety-fire_detection` | Detection | 🎯 Fine-tune | YOLOX-M (canonical lr) | 0.478 (5% val) / 0.978 (5% train) | ✅ YOLOX-M memorizes 5%; RT-DETRv2 plateaus at 0.28 train mAP (architectural small-data limit) |
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

Each feature has arch-specific configs (`06_training_{yolox,rtdetr,dfine}.yaml`); YOLOX-M is the small-data default (per Iteration 7 — dataset sizes here are all < 50k). Common knobs:
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
1. `safety-fire_detection` — 🔄 3 arch configs (`06_training_{yolox,rtdetr,dfine}.yaml`); YOLOX-M is the production choice (Iteration 7)
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

### Iteration 7 — 2026-04-20

Learning-capability / overfit analysis on 5% of `safety-fire_detection` (585 train / 130 val). Goal: verify which arch can memorize a small subset with augmentation OFF.

- **YOLOX-M wins** at `lr=0.0025` (Megvii scaling rule `basic_lr × bs/64` for bs=16). Train mAP = **0.978**, val = 0.478, 150 ep in 21 min. Default `lr=0.01` at bs=16 was 4× too hot → plateau at train loss ~4, val mAP 0.38.
- **RT-DETRv2-R18 cannot memorize 585 images** — tested 5 configs (lr=5e-5 / 1e-4 / 1.6e-4, `matcher_class_cost=2/5`, `num_denoising=20/100`, `num_queries=100/300`). Best train mAP = 0.28. Single-batch overfit works (loss 215 → 2.75 in 300 steps) so the pipeline is correct; the 5%-data plateau is a **steps-per-image shortage** compounded by bipartite-matcher instability on low-GT-density data.
- Fixed `id2label`/`label2id` missing in `build_hf_model` — unblocked single-batch class memorization (fire top-1 conf 0.118 → 0.993) but didn't move 5%-data val numbers materially.
- New: `notebooks/detr_finetune_reference/` — isolated `.venv-notebook/` with byte-for-byte ports of qubvel's RT-DETRv2 + D-FINE reference notebooks. Use as the known-good baseline for future DETR-family debugging. See that folder's `CLAUDE.md`.

**Rule of thumb**: prefer YOLOX-M when train data < ~5k images for 2–4 classes. RT-DETR is expected to work at 17k+ images; small-data weakness is a DETR-family architectural trait (matches the D-FINE findings in Iteration 6).

---

### Iteration 6 — 2026-04-19

Full training on `safety-fire_detection`, parallel on 2× RTX 5090. Main outcomes:

- **YOLOX-M (official Megvii impl)** via new `.venv-yolox-official/` + `model.impl=official`. Early-stopped ep101, best quick-val mAP=0.510 @ ep51 (full val at that epoch: 0.442; see gotcha about quick-val overstating mAP). TTA eval (3 scales × h-flip) brings it to **0.492 mAP@0.5** (+11%), smoke AP +24%. Error analysis: 99.9% of errors are background FPs, F1-vs-conf curve is pinched at conf ≈ 0.42, hardest val images are indoor warehouses. See `features/safety-fire_detection/eval/yolox_official_ep51/`.
- **RT-DETRv2-R18 diverges at published HPO LR (0.00016) on full data**: full-val peaks 0.303 @ ep10, collapses to 0.063 @ ep35. The HPO sweet-spot (from 5% data) is too hot for full data. Retraining with `lr=0.0001`, `patience=50`, `val_full_interval=0` (in progress).
- **D-FINE-S class collapse confirmed structural, not hparam**: two runs with different hparams both collapsed one class to AP≈0; the tuned run flipped which class collapsed. Load-report reinits and HF loss coefficients are *identical* to RT-DETRv2 — the remaining architectural difference is D-FINE's DFL reg head, which is too unstable at startup for bipartite matching on a 2-class, ~17k-image fine-tune. Recommendation: use RT-DETRv2 for small-class detection tasks; save D-FINE for COCO-scale class counts. Full investigation in `features/safety-fire_detection/CLAUDE.md`.

Dual-venv pattern added (custom vs official YOLOX); `scripts/setup-yolox-venv.sh`, `scripts/compare_yolox_impls.py`, `scripts/yolox_tta_eval.py`, `scripts/yolox_failure_cases.py` are the new tooling. Root CLAUDE.md updated with new gotchas (p08 preprocessing parity, quick-val overstatement, TTA utility, RT-DETR lr regression).

Fixed along the way: (1) YOLOX custom `+ 0.5` half-grid-cell decode bug — bit-identical parity with official now; (2) p08 evaluator used its own `_MinimalDetectionDataset` with letterbox + `/255` only → 5× mAP gap vs training; now uses YOLOXDataset + `build_transforms(is_train=False)`; (3) HF detection logs `cls_loss=0.0` because loss-dict key filter missed `loss_vfl` / `loss_dfl` — fixed.

---

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

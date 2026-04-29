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
| `safety-fire_detection` | Detection | 🎯 Fine-tune | SalahALHaismawi_yolov26 | 0.153 | 🔄 Phase B: YOLOX-M + RT-DETRv2-R50 PASS, D-FINE-M 50ep rerun pending · Phase C RT-DETRv2-R50 60ep complete 2026-04-22 (mAP@0.5=0.6844) |
| `safety-fall-detection` | Detection | 🎯 Fine-tune | yolov11_fall_melihuzunoglu.pt | 0.050 | ⬜ not started |
| `safety-fall_pose_estimation` | Pose keypoints | 🎯 Fine-tune | dwpose_384_pose (ONNX, interim) | — | ⬜ not started |
| `safety-poketenashi` | Orchestrator | 🔧 Pretrained only | dwpose_384_pose (det_rate=1.0) | — | 🔄 pipelines done |
| `safety-poketenashi-phone-usage` | Detection sub-model | 🎯 Fine-tune | none (action class) | 0.000 | ⬜ not started |
| `safety-point_and_call` | Pose orchestrator | 🔧 Pretrained only | dwpose_384_pose | — | 🔄 v1 rule-based MVP (DWPose + per-frame direction classifier + L→R→F sequence matcher) |
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

**Phase B — 20% smoke training (one feature at a time, one arch at a time on a single GPU):**
Each feature's CLAUDE.md carries the Phase-B plan: 20% train + 20% val (full test) × {YOLOX-M, RT-DETRv2-R50, D-FINE-M}, PASS-gated by loss drop + mAP above pretrained baseline + no class collapse. Execution order:
1. `safety-fire_detection` — 2 classes, 17k imgs (fastest signal)
2. `safety-fall-detection` — 2 classes, 12k imgs
3. `ppe-helmet_detection` — 4 classes, 22k imgs (multi-class stress)
4. `safety-poketenashi-phone-usage` — 2 classes, 23k imgs, 94.6/5.4 imbalance (class-collapse stress)
5. `ppe-shoes_detection` — 3 classes, 37k imgs (largest; may short-circuit if winner is obvious)
6. `safety-fall_pose_estimation` — keypoints, after detection models done; blocked on mmpose + no training_ready data

**Phase C — Config only (no GPU needed):** ✅ Pipelines implemented
- `safety-poketenashi` — pose rule modules + orchestrator done
- `access-face_recognition` — enrollment pipeline done
- `access-zone_intrusion` — zone detector done

**Phase D — Eval + export + release (all features):**
- `p08_evaluation` → `p09_export` → `utils/release.py` for each trained model

---

## Phase-B recipe (detection feature)

Shared 20% smoke recipe used by every Phase-B detection feature CLAUDE.md (fall, helmet, shoes, phone-usage). Each leaf file lists only its classes, dataset, baseline, and unique risk callout — commands and PASS criteria live here.

**Goal:** Sanity-check each arch can learn on the dataset. 20% train + 20% val (full test) × {YOLOX-M, RT-DETRv2-R50, D-FINE-M}.

**PASS criteria (all 4 must hold):**
1. `train/loss` drops >= 50% between epoch 1 and final epoch (no divergence, no NaN)
2. `val mAP@0.5` exceeds the pretrained baseline (or > 0.05 if no usable baseline exists)
3. Confusion matrix diagonal > 0.5 for each class (no class collapse)
4. `error_breakdown.png` shows FP mix != 100% background

### Commands

Replace `<feature>` with the feature folder name (e.g. `safety-fall-detection`).

```bash
# YOLOX-M (config defaults: impl=official, library=torchvision, mosaic=true,
# mixup=false, normalize=false, lr=0.0025, val_full_interval=0 — Phase B only
# overrides epochs->30 + subset)
CUDA_VISIBLE_DEVICES=0 .venv-yolox-official/bin/python core/p06_training/train.py \
  --config features/<feature>/configs/06_training_yolox.yaml \
  --override training.epochs=30 \
    data.subset.train=0.2 data.subset.val=0.2 \
    training.data_viz.enabled=false training.aug_viz.enabled=false training.val_viz.enabled=false

# RT-DETRv2-R50 (config defaults: arch=r50, pretrained=r50vd, bf16=true, amp=false,
# mosaic=false, warmup_steps=300, lr=1e-4 — Phase B overrides lr->5e-5 + bs->8 + epochs->30)
CUDA_VISIBLE_DEVICES=0 uv run core/p06_training/train.py \
  --config features/<feature>/configs/06_training_rtdetr.yaml \
  --override training.lr=5e-5 training.epochs=30 \
    data.batch_size=8 data.subset.train=0.2 data.subset.val=0.2 \
    training.data_viz.enabled=false training.aug_viz.enabled=false training.val_viz.enabled=false

# D-FINE-M (config defaults: arch=dfine-m, pretrained=dfine_m_coco, bf16=false, amp=false,
# weight_decay=0, lr=5e-5, bs=8 — match the reference recipe; Phase B only overrides epochs->30)
CUDA_VISIBLE_DEVICES=0 uv run core/p06_training/train.py \
  --config features/<feature>/configs/06_training_dfine.yaml \
  --override training.epochs=30 \
    data.subset.train=0.2 data.subset.val=0.2 \
    training.data_viz.enabled=false training.aug_viz.enabled=false training.val_viz.enabled=false
```

### Error analysis (run after each training)

```bash
CUDA_VISIBLE_DEVICES=0 uv run core/p08_evaluation/evaluate.py \
  --model features/<feature>/runs/<ts>/best.pth \
  --config features/<feature>/configs/05_data.yaml \
  --split test --conf 0.3 --iou 0.5
```

Outputs: `metrics.json`, `confusion_matrix.png`, per-class PR curves, `error_breakdown.png`, `size_recall.png`, `optimal_thresholds.json`.

### OOM notes
- 30 epochs at bs=8/16 -> ~15-80 min/arch on RTX 5090 depending on dataset size; consider 15 epochs for the largest (~37k img) features.
- Pre-flight: `nvidia-smi --query-gpu=memory.free --format=csv` -> need >=20 GB free.
- Kill if first-epoch VRAM > 24 GB or `train/loss` NaNs -> halve `data.batch_size` and retry.
- **bf16 policy**: YOLOX `amp=true`; RT-DETRv2 `bf16=true amp=false`; D-FINE `bf16=false amp=false`.
- **Never launch two trainings on the same GPU** — system hang risk.

### Results table template (fill after each run)

| Arch | epochs | best val mAP@0.5 | train loss drop | Class collapse? | PASS? | runs/ dir | eval/ dir |
|---|---|---|---|---|---|---|---|
| YOLOX-M | | | | | | | |
| RT-DETRv2-R50 | | | | | | | |
| D-FINE-M | | | | | | | |

### Error analysis summary template (per arch, fill after p08)
- Dominant FP type (background / class confusion / localization / duplicate)
- Worst class + per-class AP gap
- Size bucket where recall collapses
- Top 3 failure cases

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

### Phase-B investigation history — fire_detection, 2026-04-18 to 2026-04-20 (superseded)

Multiple training iterations ran on fire_detection exploring arch choice (YOLOX-M / RT-DETRv2-R18 / D-FINE-S) and scaling behavior. All run dirs, checkpoints, and eval artifacts were cleared on 2026-04-21 when the feature was reset; the fire CLAUDE.md was rewritten to a clean state. **Generalizable lessons kept** (now encoded in root `CLAUDE.md` Gotchas and per-feature Phase-B plans):

- **Megvii LR rule**: YOLOX `basic_lr × bs/64` — default `lr=0.01` at `bs=16` is 4× too hot; use `lr=0.0025`.
- **DETR-family requirements**: `amp: false` mandatory (fp16 → NaN pred_boxes); D-FINE further requires `bf16: false` (DFL stalls under bf16); DETR does not support Mosaic.
- **HPO LR does not generalize**: LR tuned on 5%-data HPO is typically too hot for full data; use conservative `5e-5` for small-class DETR fine-tune.
- **D-FINE-S collapses on 2-class fine-tune**: distribution-focal reg head is unstable at startup with reinit'd 2-query cls head; not hparam-fixable at small N_classes. Re-evaluate D-FINE-M fresh in Phase B.
- **YOLOX small-data overfit**: at 5% data with aug off, YOLOX-M memorizes (train mAP > 0.9); RT-DETRv2 cannot break ~0.3 train mAP (bipartite-matcher instability at low GT density). For any feature < 5k train imgs, YOLOX is the safer pick.
- **Dual YOLOX impls**: `.venv-yolox-official/` + `model.impl=official` for Megvii parity; `custom` impl for GPU-aug and per-component LR. Scripts in place: `setup-yolox-venv.sh`, `compare_yolox_impls.py`, `yolox_tta_eval.py`, `yolox_failure_cases.py`.
- **Code fixes made during the investigation** (still in the codebase): YOLOX `+ 0.5` decode bug removed; p08 evaluator preprocessing parity with training val; HF `cls_loss=0.0` logging artefact (key filter now matches `loss_vfl` / `loss_dfl`); `id2label`/`label2id` auto-populated from 05_data `names:` dict in `build_hf_model`.

Current Phase B plan restarts on all 5 detection features from scratch at 20% data, comparing YOLOX-M / RT-DETRv2-R50 / D-FINE-M — see each feature's CLAUDE.md.

---

### Iteration 9 — 2026-04-29 (new feature: safety-point_and_call)

New feature folder `features/safety-point_and_call/` for the Japanese 指差呼称 (shisa-kanko / point-and-call) crosswalk gesture: worker stops at the curb, points right (右ヨシ!), points left (左ヨシ!), optionally points front (前ヨシ!), then crosses. Distinct from `safety-poketenashi/code/pointing_calling_detector.py`, which only detects whether *any* arm is extended horizontally (binary, non-directional, no L→R→F sequence).

**v1 design (this iteration):**
- Pretrained-only — no own training set, no `06_training_*.yaml`, no `00_data_preparation.yaml`.
- Swappable pose backend (DWPose ONNX / RTMPose / MediaPipe / HF ViTPose) behind a thin `PoseBackend` Protocol in `code/pose_backend.py`. Default backend is DWPose for parity with `safety-poketenashi`.
- Per-frame direction classifier (`PointingDirectionDetector`): COCO-17 keypoints → torso-frame arm azimuth → label ∈ {point_left, point_right, point_front, neutral, invalid}.
- Temporal `CrosswalkSequenceMatcher`: emits `point_and_call_done` when an ordered subsequence (LR / RL / LRF / RLF) is held within `window_seconds`, each direction sustained ≥ `hold_frames`. Optional `cross_zone:` polygon triggers `missing_directions` when person crosses without a recent successful match.

**v2 roadmap** (not implemented):
- Bootstrap dataset from DP Dataset (Kyoto U, ~2M frames, 3D pointing direction) + Roboflow `wayceys-workspace/hand-pointing-directions` (1.7k images). Self-collect ~200 crosswalk clips for fine-tune.
- Replace per-frame geometric rule with a 5-class MLP head on (kpt, kpt_score) features.
- Upgrade to ST-GCN / 1D-TCN if rule recall < 90% in field testing.
- Full pipeline: `06_training_*.yaml`, ONNX export, release packaging.

---

### Iteration 8 — 2026-04-21 (config defaults alignment)

All 15 detection-feature training configs (5 YOLOX + 5 RT-DETR + 5 D-FINE across 5 Phase-B features) aligned to the validated recipes in `notebooks/detr_finetune_reference/`:

- **RT-DETR**: default arch bumped r18 → **r50** (`PekingU/rtdetr_v2_r50vd`); `augmentation.library: torchvision`; `logging.report_to: none`; stale `run_name` dropped.
- **D-FINE**: migrated from `backend: pytorch` (broken on small-class per Iteration 7) → `backend: hf`; default arch s → **dfine-m**; `bf16: false`, `amp: false`, `weight_decay: 0`, `lr: 5e-5`, `warmup_steps: 300`, `scheduler: linear`, `epochs: 50` (reference showed +0.06 test mAP over 30 ep on CPPE-5).
- **YOLOX**: explicit `model.impl: official`, `augmentation.library: torchvision`, `mosaic: true`, `mixup: false`, `normalize: false`, `lr: 0.0025` (Megvii rule `0.01 × bs/64`), `val_full_interval: 0`, `val_subset_fraction: 1.0`; `contrast` key removed. Scale stays `[0.8, 1.2]` (tight for tiny-object features — do NOT widen to Megvii's `[0.1, 2.0]` default; on 0.01–0.1% bboxes, aggressive scale jitter pushes objects below the training grid).

**Effect on Phase B commands**: each feature's CLAUDE.md now overrides only `training.epochs=30 data.subset.train=0.2 data.subset.val=0.2` plus viz-off flags — the previous 8–10 arch/backend/aug overrides are now config defaults. New HF-Trainer footguns (`run_name` ghost folder, `report_to` wandb/TB crash) documented in root CLAUDE.md.

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

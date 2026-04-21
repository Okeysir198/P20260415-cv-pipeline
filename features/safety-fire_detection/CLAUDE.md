# safety-fire_detection

**Type:** Detection | **Training:** Fine-tune required (fire/smoke not in COCO 80)

## Overview

Detects fire and smoke in images/video. Both classes are absent from COCO — pretrained models show low mAP (best: 0.153) confirming fine-tuning is mandatory.

## Classes

| ID | Name | Train split % |
|---|---|---|
| 0 | fire | 53.7% |
| 1 | smoke | 46.3% |

## Dataset

- **Images:** 17,373 (val: 2,609 | test: ~3,000)
- **QA:** 95.1% good / 1.1% bad → ACCEPT
- **Label Studio:** project id=13
- **Training ready:** `dataset_store/training_ready/fire_detection/`

## Pipeline Checklist

- [x] `00_data_preparation.yaml` — sources locked, class map verified
- [x] `p00_data_prep` — 17,373 imgs, DATASET_REPORT
- [x] `p02_annotation_qa` — LS project 13
- [x] `code/benchmark.py` — pretrained benchmark complete
- [x] Arch comparison configs — `06_training_dfine.yaml`, `06_training_rtdetr.yaml`, `06_training_yolox.yaml` created and tested
- [x] Arch comparison (Iteration 5, 10% data / 15 ep): RT-DETRv2-R18 led at val mAP50=0.541, but full-data runs (Iteration 6) showed RT-DETRv2 diverges at HPO LR and plateaus at ~0.335 at safer LR.
- [x] Overfit-capability analysis (Iteration 7, 5% data, aug off): **YOLOX-M memorizes (train 0.978 / val 0.478); RT-DETRv2 cannot (best train 0.28 across 5 configs)**. → **YOLOX-M is production arch for fire_detection.**
- [x] `p07_hpo` — RT-DETRv2 best (historical): lr=1.6e-4, warmup=15, wd=1.15e-5. Kept in `06_training_rtdetr.yaml` for reference; not the production config.
- [x] Full training — YOLOX-M (official) ep51: full val mAP=0.442, TTA=0.492. YOLOX-M v2 retrain with `val_full_interval=0` in progress (see below).
- [ ] `p06_training` — production run: `06_training_yolox.yaml --override model.impl=official augmentation.normalize=false training.val_full_interval=0`
- [ ] `p08_evaluation` — evaluate on test split
- [ ] `p09_export` — ONNX export
- [ ] `release/` — `utils/release.py`

## Benchmark Results — val split (2026-04-17, 2609 images)

### Detection Models

| Model | mAP50 | mAP50-95 | P | R | Latency ms | Status |
|---|---|---|---|---|---|---|
| **SalahALHaismawi_yolov26-fire-detection** | **0.153** | **0.062** | 0.241 | 0.173 | 3805 | ok |
| touati-kamel_yolov12n-forest-fire | 0.026 | 0.013 | 0.036 | 0.127 | 3528 | ok |
| JJUNHYEOK_yolov8n_wildfire | 0.025 | 0.017 | 0.035 | 0.396 | 3440 | ok |
| touati-kamel_yolov10n-forest-fire | 0.025 | 0.011 | 0.037 | 0.103 | 3007 | ok |
| touati-kamel_yolov8s-forest-fire | 0.022 | 0.009 | 0.072 | 0.053 | 4034 | ok |
| Mehedi-2-96_fire-smoke-yolo | 0.012 | 0.004 | 0.017 | 0.013 | 3529 | ok |
| TommyNgx_YOLOv10-Fire-and-Smoke | error | — | — | — | — | CUDA OOM |
| pyronear_yolov8s | error | — | — | — | — | CUDA OOM |

### Skipped Models

| Model | Reason |
|---|---|
| yolox_s/m | COCO 80-class, no fire/smoke class |
| deim_dfine_m/s_coco | COCO-only detector |
| pyronear_yolo11s_nimble-narwhal_v6 | No .pt file found |
| pyronear_yolo11s_sensitive-detector | No fire/smoke class |
| pyronear_yolov11n | COCO 80-class |
| ustc-community_dfine-medium/small-coco | COCO-only |

### Classification Models (pyronear — image-level binary)

| Model | F1 | Precision | Recall | Latency ms |
|---|---|---|---|---|
| **pyronear_resnet18** | **0.806** | 1.000 | 0.675 | 3.3 |
| pyronear_resnet34 | 0.792 | 1.000 | 0.656 | 6.6 |
| pyronear_mobilenet_v3_large | 0.775 | 1.000 | 0.632 | 2.0 |
| pyronear_mobilenet_v3_small | 0.691 | 1.000 | 0.527 | 1.3 |
| pyronear_rexnet1_0x | 0.000 | 0.000 | 0.000 | — |
| pyronear_rexnet1_3x | 0.000 | 0.000 | 0.000 | — |
| pyronear_rexnet1_5x | 0.000 | 0.000 | 0.000 | — |

**Recommendation:** Fine-tune from `SalahALHaismawi_yolov26-fire-detection` (highest mAP50=0.153). Use `pyronear_resnet18` as a fast pre-filter (F1=0.806, 3.3ms) to gate the detector.

Full results: `eval/benchmark_results.json` | `eval/benchmark_report.md`

## Data Findings

- **Test class imbalance**: test split is ~35% fire / 65% smoke vs ~53/47 in train+val — test set was not stratified by class. Per-class AP50 on test will be biased; interpret AP50_cls0 (fire) vs AP50_cls1 (smoke) with this skew in mind.
- **Objects are predominantly tiny**: bbox area distribution peaks at 0.01–0.1% of image area. Most objects are below the "tiny (<1%)" tier — do not reduce input resolution below 640×640.
- **~18% empty images**: significant background-only image fraction across all splits — good for false-positive suppression, keep them in training.

## GPU Augmentation Benchmark (2026-04-18, 640×640, 4 workers, updated post-vectorization)

| batch_size | CPU ms/batch | GPU ms/batch | Speedup | GPU img/s |
|---|---|---|---|---|
| 16 | 192 ms | 60 ms | 3.23x | 269 |
| 32 | 397 ms | 137 ms | 2.90x | 234 |
| 64 | 782 ms | 273 ms | 2.86x | 234 |

Mosaic stays CPU. GPU path: batched `affine_grid+grid_sample`, vectorized HSV, randomized ColorJitter order. Enabled via `training.gpu_augment: true`.

## Training Commands

```bash
# Production — YOLOX-M (official Megvii impl, per Iteration 7 overfit analysis)
.venv-yolox-official/bin/python core/p06_training/train.py \
  --config features/safety-fire_detection/configs/06_training_yolox.yaml \
  --override model.impl=official augmentation.normalize=false training.val_full_interval=0

# Reference-only — RT-DETRv2 (plateaus ~0.335 on full data; use at 50k+ images)
uv run core/p06_training/train.py --config features/safety-fire_detection/configs/06_training_rtdetr.yaml

# Reference-only — D-FINE-S (class collapse on 2-class fine-tune; see Iteration 6)
uv run core/p06_training/train.py --config features/safety-fire_detection/configs/06_training_dfine.yaml
```

## Key Files

```
configs/00_data_preparation.yaml   — data sources + class map
configs/05_data.yaml               — dataset paths + class names
configs/06_training_yolox.yaml     — YOLOX-M (production — per Iteration 7)
configs/06_training_rtdetr.yaml    — RT-DETRv2-R18 (reference — small-data plateau)
configs/06_training_dfine.yaml     — D-FINE-S (reference — class collapse on 2-class)
code/benchmark.py                  — pretrained benchmark
eval/benchmark_results.json        — benchmark output
eval/benchmark_report.md           — benchmark summary
```

## Arch Comparison Results (2026-04-18, 10% data, 15 epochs)

| Arch | best val/mAP50 | train/loss drop | Notes |
|---|---|---|---|
| **RT-DETRv2-R18** | **0.541** (ep 15, still rising) | 32.5 → 11.7 (-64%) | Winner |
| D-FINE-S | 0.190 (ep 9, plateau) | 32.5 → 18.3 (-44%) | — |
| YOLOX-M | 0.113 (ep 73, early stop) | — | Previous run |

Max safe batch size on RTX 5090 (28 GB free, fp32): **bs=32** (14.7 GB peak). bs=48 fits but leaves only 0.3 GB headroom.

## Full Training Run — YOLOX-M (official), 2026-04-19

**Setup:** `06_training_yolox.yaml` + `--override model.impl=official augmentation.normalize=false`. Ran in `.venv-yolox-official/`. Trainer unchanged — adapter uses existing `forward_with_loss()` hook.

**Result (ep51 / early stopped ep101):**

| Metric | Quick val (20% subset, drove best.pth) | Full val (true) | With TTA (3 scales × h-flip) |
|---|---|---|---|
| mAP@0.5 | 0.510 | **0.442** | **0.492** |
| Fire AP | 0.607 | 0.530 | 0.546 |
| Smoke AP | 0.413 | 0.354 | 0.438 |
| Best F1 @ conf 0.42 | — | fire 0.561 / smoke 0.446 | — |

**Headline lessons from error analysis (`eval/yolox_official_ep51/`)**:

1. **99.9% of errors are background false positives** (9.05M FPs vs 579 missed GT, 126 class confusions). When the model matches a real object it classifies correctly ~85% of the time (confusion diagonal 0.87 fire / 0.83 smoke).
2. **F1-vs-confidence curve is pinched** — useful only in `[0.33, 0.52]`, peak at 0.42. Model is miscalibrated: overconfident on many FPs.
3. **Hardest val images are indoor warehouses** (4/6 top hardest). Packaging / stacked boxes / reflective surfaces get confused with smoke. Landscape scenes with distant small fires are handled better than warehouse clutter.
4. **Smoke is systematically weaker** (AP 0.35 vs fire 0.53). Scale-variant and fuzzy-boundary → TTA closes most of the gap (+24% smoke AP).
5. **Max achievable recall ~0.85** for fire, ~0.80 for smoke — some true boxes are literally never proposed even at conf=0.

**YOLOX-M v2 retrain (in progress)**: `val_full_interval=0` → best-checkpoint selection on full val; expecting the true best epoch, not the lucky-quick-val peak.

## Full Training Run — RT-DETRv2-R18, 2026-04-19

**First run** (config defaults: lr=0.00016, patience=30): best quick val 0.345 @ ep9; **full val diverged** — 0.303 @ ep10 → 0.194 @ ep15 → 0.063 @ ep35 → early stopped ep39. LR from 5%-data HPO was too hot for full data; after 15-epoch warmup completed at ep15, model oscillated then collapsed.

**Retrain** (lr=0.0001, patience=50, `val_full_interval=0`): plateaued ~0.335 mAP by ep17. Even at the lower LR, RT-DETRv2 cannot break the mid-0.30s ceiling on this dataset at scale. See Iteration 7 overfit-capability analysis below.

## Arch Learnings — D-FINE-S, 2026-04-19

**Tried twice, both failed with class collapse** (one class → AP≈0, other inches up):

| Run | Config | Outcome |
|---|---|---|
| #1 | Defaults (lr=0.0001, bs=8, warmup=10) | ep38 mAP 0.034, **fire AP 0.0002** (smoke-only) |
| #2 | RT-DETR-style (lr=0.00016, bs=16, wd=1.15e-5, warmup=15) | ep22 mAP 0.016, fire AP 0.000 (class flip — now smoke-only, even lower) |

The hparam tune (2.6× stronger gradient signal per step) **flipped which class collapsed** but did not fix the underlying problem — this is the signature of mode collapse, not undertraining.

**Root-cause investigation** (what we verified, not what we guessed):

1. **Load-report reinits are identical between D-FINE and RT-DETRv2** — both reinit `decoder.class_embed.{0,1,2}.weight + bias`, `denoising_class_embed.weight` (`[81,256]→[3,256]`), and `enc_score_head.weight + bias`. The earlier guess that D-FINE reinits more was wrong. `.venv-yolox-official/bin/python -c "from transformers import DFineForObjectDetection, RTDetrV2ForObjectDetection; ..."` — diff is zero on the reinit keys.
2. **HF loss coefficients are identical**: `weight_loss_vfl=1.0, weight_loss_bbox=5.0, weight_loss_giou=2.0`, `matcher_class_cost=2.0, matcher_bbox_cost=5.0, matcher_giou_cost=2.0`, `eos_coefficient=0.0001`, `focal_loss_α=0.75 γ=2.0`, `num_queries=300`, `num_denoising=100`. Same for both archs.
3. **Only architectural difference left** is D-FINE's **Distribution Focal Loss (DFL) + Fine-grained Distribution Refinement** on the reg head: D-FINE predicts a distribution over 16 bins per box offset (4 × 16 × N_queries extra reg params to train), while RT-DETRv2 predicts 4 direct coordinates. With reinit'd cls head and ~17k-image fine-tune, D-FINE's reg head is too unstable at startup for the bipartite matcher to find consistent matches → one class's queries win and the other starves.

**Hypotheses to try if you want to rescue D-FINE** (not done — RT-DETRv2 is the pragmatic choice):
- `num_denoising: 300` (3× default) to force more cls-head training via noise queries
- `warmup_epochs: 40` (2–4× default) to stabilise before reg head dominates
- Freeze backbone + neck for first 10 epochs via `training.freeze: ["model.backbone", "model.encoder"]`
- Set `matcher_class_cost: 5.0` (up from 2.0) to make matcher weight class correctness more than box accuracy — should force balanced query specialization

**Also found/fixed: `cls_loss` logging was 0 for DETR-family.** `hf_model.py::forward_with_loss` filtered `raw_loss_dict` keys for `"ce"` or `"class"` — but HF DETR returns keys `loss_vfl` (Varifocal Loss) and `loss_dfl` (Distribution Focal Loss). So the per-component loss breakdown was silently empty. Fixed to also match `vfl / focal / dfl / reg`. This means **any prior DETR training log where `train/cls_loss=0.0` was an artifact**, not actual zero.

**Rule of thumb (for this repo's detection tasks)**: prefer RT-DETRv2-R18 over D-FINE-S when `num_classes ≤ 4` and dataset is under 50k images. D-FINE is probably fine for COCO-sized problems with 80+ classes where the cls head has enough signal to specialize queries.

## Overfit-capability analysis — 2026-04-20 (Iteration 7)

On 5% subset (585 train / 130 val, augmentation OFF, 150 epochs):

| Arch | Config | train/mAP50 | val/mAP50 | Memorizes? |
|---|---|---|---|---|
| YOLOX-M | `lr=0.0025`, bs=16 (Megvii scaling) | **0.978** | 0.478 | ✅ yes |
| RT-DETRv2-R18 | `matcher_class_cost=5`, `num_denoising=20`, `lr=5e-5`, bs=8 | 0.282 | 0.138 | ❌ no |
| RT-DETRv2-R18 | `num_queries=100` variant | 0.234 | 0.116 | ❌ no |
| RT-DETRv2-R18 | `lr=1e-4` variant | 0.194 | 0.133 | ❌ no |

**Key fix (YOLOX)**: Megvii's `basic_lr = 0.01 × bs/64` rule means `lr=0.01` at `bs=16` is 4× too hot. Use `lr=0.0025` at `bs=16` for small-batch training. With this fix + aug off, YOLOX-M memorizes the 5% train subset to near-perfection (train mAP 0.978) while also generalizing (val 0.478).

**Key finding (RT-DETR)**: pipeline is correct (verified via single-batch overfit: loss 215 → 2.75 in 300 steps on 8 images with predictions matching GT at 0.99+ confidence after the `id2label` fix), but the bipartite matcher cannot stabilize query specialization on 585 images regardless of matcher cost / query count / LR / num_denoising. Not a tunable hparam issue — architectural. Use **YOLOX-M as the production arch for fire_detection**; re-evaluate RT-DETRv2 only on full 17k data.

See `notebooks/detr_finetune_reference/CLAUDE.md` for the isolated reference-notebook environment used to sanity-check this finding against qubvel's canonical recipe.

## Gotchas

- **D-FINE/RT-DETR require `amp: false`** — HF DETR decoder overflows in fp16, producing NaN `pred_boxes` that crash `generalized_box_iou` on first forward pass. Both configs already set this.
- **`ValPredictionLogger` class names** — reads `trainer._loaded_data_cfg` (the resolved `05_data.yaml`). Using `trainer._data_cfg` (the `data:` section of the training YAML) returns no `names` key and labels show as IDs only. Fixed in `callbacks.py`.
- **`gpu_augment: true` is mandatory** for all three arch configs — always verify it's present. DETR-family (D-FINE, RT-DETRv2) additionally require `augmentation.mosaic: false` since DETR does not support mosaic.
- **HPO commands** (parallel on two GPUs):
  ```bash
  # GPU 0 — YOLOX
  CUDA_VISIBLE_DEVICES=0 uv run core/p07_hpo/run_hpo.py \
    --config features/safety-fire_detection/configs/06_training_yolox.yaml \
    --hpo-config configs/_shared/08_hpo_yolox.yaml \
    --override data.subset.train=0.05 data.subset.val=0.10 data.batch_size=32 \
      augmentation.mosaic=false \
      training.data_viz.enabled=false training.aug_viz.enabled=false training.val_viz.enabled=false
  # GPU 1 — D-FINE (then RT-DETRv2 sequentially)
  CUDA_VISIBLE_DEVICES=1 uv run core/p07_hpo/run_hpo.py \
    --config features/safety-fire_detection/configs/06_training_dfine.yaml \
    --hpo-config configs/_shared/08_hpo_detr.yaml \
    --override data.subset.train=0.05 data.subset.val=0.10 data.batch_size=32 \
      training.data_viz.enabled=false training.aug_viz.enabled=false training.val_viz.enabled=false
  ```

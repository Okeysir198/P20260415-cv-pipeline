# safety-fire_smoke_fall

**Type:** Detection (combined) | **Training:** Fine-tune (3 classes, none in COCO)

3-class joint detector merging `safety-fire_detection` (fire, smoke) + `safety-fall-detection` (fallen_person only — `person` dropped). Built to run a single forward pass instead of two separate models on shared cameras.

## Classes

| ID | Name | Source |
|---|---|---|
| 0 | fire | fire_detection |
| 1 | smoke | fire_detection |
| 2 | fallen_person | fall_detection (`person` class dropped) |

## Dataset

`dataset_store/training_ready/fire_smoke_fall/` — built via `scripts/merge_fire_smoke_fall_training_ready.py` (NOT p00 — symlinks both upstream training_ready/ trees, rewrites labels with class remap).

| Split | Imgs | fire | smoke | fallen |
|---|---:|---:|---:|---:|
| train | 20,376 | 13,717 | 11,232 | 9,586 |
| val | 4,382 | 2,567 | 2,525 | 2,029 |
| test | 4,390 | 2,615 | 2,557 | 2,019 |

Filenames are prefixed `fire__` / `fall__` to avoid collisions. Re-run the merge script (idempotent) when either upstream training_ready/ refreshes.

## Benchmark Results (2026-05-15)

All four configs trained on the per-source-temporal split inherited from upstream features. Test set evaluated on best ckpt by `eval_map_50`.

| Arch | params | bs | LR (head/bb) | epochs | best val mAP50 | **test mAP50** | fire | smoke | fallen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **D-FINE-N** | 4M | 32 | 1e-4 | 60 (resumed) | 0.575 @ ep37 | **0.584** 🏆 | **0.190** | **0.273** | 0.417 |
| D-FINE-S | 16M | 32 | 1e-4 | 60 (resumed) | 0.531 @ ep31 | 0.522 | 0.149 | 0.257 | **0.451** 🏆 |
| D-FINE-M | 31M | 16 | 5e-5 | 60 | 0.364 @ ep45 | 0.340 | 0.081 | 0.085 | 0.362 |
| RT-DETRv2-R50 | 42M | 8 | 5e-5 / 1e-5 | killed @ ep45 | 0.395 @ ep35 | 0.339 | 0.129 | 0.150 | 0.192 |

Per-class mAP shown is COCO-style mAP@[.5:.95] (`test_map_per_class_*`).

## Deployment Recommendation

- **Single overall winner → D-FINE-N** (test mAP50 = 0.584, only 4M params, fastest inference).
- **Fallen-priority deployment → D-FINE-S** (+0.034 fallen vs N, but loses 0.062 overall).
- Best ckpts:
  - `runs/dfine_n_2026-05-14_222459/checkpoint-23184` (best.pt resolves to this)
  - `runs/dfine_s_2026-05-15_030458/checkpoint-19747`

## Self-contained inference (`predict/`)

`features/safety-fire_smoke_fall/predict/` is a portable inference bundle — copy it
anywhere with `torch + torchvision + transformers + cv2 + ffmpeg` and it runs
without the rest of the repo:

- `inference.py` — three-stage CLI (extract / infer / render). **Per-class NMS**
  via `torchvision.ops.nms` (DETR is "NMS-free" in theory; in practice noisy
  real-world video produces overlapping low-conf proposals — NMS at IoU=0.5
  removes 88-99% of them). Output mp4 is **native H.264 via ffmpeg pipe**
  (VS Code playable; cv2 mp4v isn't).
- `model/best_overall/` — D-FINE-N original recipe, **test mAP50=0.584**
  (overall winner).
- `model/best_fire/` — D-FINE-N γ=3 α=0.9 EMA recipe, **fire 0.203** (fire
  winner; overall mAP50=0.534). Default model (production prefers fire recall).
- Each `model/<variant>/` includes `config.json`, `pytorch_model.bin`,
  `preprocessor_config.json`, `06_training.yaml`, `05_data.yaml`,
  `training_args.bin`, and `checkpoint-N/` subdir for full HF Trainer resume.
- Class names come from `config.json::id2label` — no separate yaml needed.
- Raw predictions saved at `conf=0.0` → re-render at any threshold without GPU.

## Load-bearing facts

1. **D-FINE-N beats every bigger D-FINE variant by a wide margin** (0.584 vs S 0.522 vs M 0.340) — fire and smoke are small/diffuse, and N's 4M-param backbone with EMA-style smoothing fits the dataset's small-object distribution best. **Do not assume bigger D-FINE = better here.**

2. **Effective LR×batch_size matters more than nominal LR.** Bigger configs paired lower LR with smaller batch:

   | Arch | LR × bs | (relative) |
   |---|---|---|
   | dfine-n / dfine-s | 1e-4 × 32 = 3.2e-3 | 1.0× |
   | dfine-m | 5e-5 × 16 = 8e-4 | 0.25× |
   | rtdetr-r50 | 5e-5 × 8 = 4e-4 | 0.125× |

   The 8× gradient-signal deficit on M/r50 explains why those undertrained at 60 epochs. Either increase LR (1e-4) or batch (32 if VRAM allows) before re-running M/r50.

3. **DETR late-epoch calibration drift confirmed here too** — N peaked at ep37 (0.575), then slid back into 0.50–0.55 plateau through ep60. Best ckpt (HF `load_best_model_at_end`) is bit-identical to checkpoint-<best_step>/pytorch_model.bin and recovers 0.584 on test. Do not extend epochs >60.

4. **Resumes work cleanly via `--resume <ckpt-dir>`.** Both N and S were OOM-killed mid-training (VLLM co-tenant on GPU1) and resumed correctly: optimizer/scheduler/epoch restored bit-exactly, post-resume mAP continued the pre-crash trajectory.

5. **Augmentation knobs that matter (already in configs):**
   - `flipud: 0.0` — vertical flip destroys upright-vs-fallen signal. MANDATORY 0.
   - `translate: 0.15` — bumped from 0.05 in upstream fire configs for stronger position invariance (combined dataset has more frame-edge boxes than fire-only).
   - `weight_loss_vfl: 2.0` + `matcher_class_cost: 4.0` — penalizes wrong-class predictions (smoke↔fire, fallen↔upright). Verified on N/S; over-regularizing on M (lower LR×bs amplifies the cls-loss dominance).

6. **`normalize: false` retained from fire feature** — combined dataset inherits fire's darker pixel statistics; ImageNet normalization would hurt the fire half. Untested whether dataset-specific stats would help further.

## Pipeline Checklist

- [x] Feature scaffolded from `safety-fire_smoke_fall_phone` template
- [x] `scripts/merge_fire_smoke_fall_training_ready.py` — symlink + class remap merge
- [x] `configs/05_data.yaml` — 3 classes, dataset path
- [x] `configs/06_training_{dfine_n,dfine_s,dfine_m,rtdetr_r50}.yaml`
- [x] `configs/10_inference.yaml` — alert thresholds (3 classes; phone removed)
- [x] All 4 archs trained (D-FINE-N best, RT-DETRv2-R50 manually killed at LR floor)
- [ ] Threshold sweep (`scripts/threshold_sweep.py`) on D-FINE-N to set per-class deployment confidence
- [ ] `p09_export` — ONNX export of D-FINE-N best
- [ ] `release/` — `utils/release.py` for D-FINE-N

## Key Files

```
configs/00_data_preparation.yaml      — NOT used (merge done via scripts/ instead)
configs/05_data.yaml                  — 3 classes, names: {0: fire, 1: smoke, 2: fallen_person}
configs/06_training_dfine_n.yaml      — 4M, bs=32, lr=1e-4 ✅ winner
configs/06_training_dfine_s.yaml      — 16M, bs=32, lr=1e-4
configs/06_training_dfine_m.yaml      — 31M, bs=16, lr=5e-5  (under-LR'd)
configs/06_training_rtdetr_r50.yaml   — 42M, bs=8, lr=5e-5/1e-5  (under-LR'd)
configs/10_inference.yaml             — per-class alert thresholds
runs/<arch>_<ts>/                     — checkpoints + test_results.json
runs/_logs/                           — training stdout
../../scripts/merge_fire_smoke_fall_training_ready.py  — dataset merge tool
```

## Gotchas

- **Dataset is built by a script, not p00.** Re-run `scripts/merge_fire_smoke_fall_training_ready.py` after either upstream training_ready/ refreshes. The symlinks resolve absolute paths so relocating the upstream dirs breaks training.
- **Fall images with only `person` boxes are skipped** (738 dropped: 543 train / 102 val / 93 test) — fall set effectively contributes ~85% of its images. If retraining `fallen_person` priority is critical, consider keeping a `person` class to provide negative-class context (currently not done).
- **D-FINE-M and RT-DETRv2-R50 are NOT done** — they ran 60 ep but at LR×bs that likely couldn't converge. To re-attempt, set `lr: 1e-4 batch_size: 32` for D-FINE-M (will fit on RTX 5090 32 GB) and `lr: 1e-4 batch_size: 16` for RT-DETRv2-R50. Expect 0.50+ mAP50 territory.
- **Bigger D-FINE pretrained needs matching arch** (root CLAUDE.md gotcha): `dfine_m_coco` weights only fit `dfine-m`. Each variant uses its own `ustc-community/dfine_<n|s|m>_coco` (or `dfine-medium-obj2coco` for M).
- **VLLM co-tenancy on GPU1 caused OOM during eval** in the initial S run — eval-time VRAM peak (32 GB) is much higher than train-time (19 GB). If a co-tenant holds even 4 GB on GPU1, prefer scheduling on GPU0 or kill co-tenants first.

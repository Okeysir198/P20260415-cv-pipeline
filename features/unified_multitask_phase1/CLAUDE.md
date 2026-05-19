# unified_multitask_phase1

**Type:** Multi-task detection (5 Phase 1 tasks, shared backbone)
**Architecture:** D-FINE-N shared trunk + 5 per-task classification heads
**Goal:** One ONNX file delivering all Phase 1 detection capabilities in a single forward pass.

> **Deep dive**: see `ARCHITECTURE.md` for the full design rationale,
> why R1/R2 (single 19-class head, frozen backbone) both failed at 0.30 mAP,
> code organization, training dynamics, and deployment notes.

## Tasks

All 5 task paths route through the **symlink farm** at
`dataset_store/training_ready/unified_detection/` (manifest: `tasks.yaml`,
aggregate report: `DATASET_REPORT.md`). Real data stays at the original
per-task dirs — the farm just gives one root.

| Head | Classes | Dataset (via farm) | Single-task baseline |
|---|---|---|---:|
| fire_smoke | fire, smoke | `unified_detection/fire_smoke` → `fire_detection/` | ~0.45 mAP50 (safety-fire_detection) |
| fall | person, fallen_person | `unified_detection/fall` → `fall_detection/` | none |
| helmet | person, head_with_helmet, head_without_helmet, head_with_nitto_hat | `unified_detection/helmet` → `helmet_detection/` (29,991 imgs after 2026-05-19 sh17 rebuild) | none |
| shoes | person, foot_with_safety_shoes, foot_without_safety_shoes | `unified_detection/shoes` → `shoes_detection/` | none |
| phone | phone_usage | `unified_detection/phone` → `safety_poketenashi_phone_usage/` | 0.529 mAP50 (poketenashi_phone) |

## Benchmark Results — dfine-n-multitask (2026-05-17, 40 ep)

Test set (held-out, never seen during training):

| Task | Test mAP@50 | Test mAP (challenge) | Single-task baseline | Δ |
|---|---:|---:|---:|---:|
| fire_smoke | 0.313 | 0.134 | ~0.45 | **-30%** |
| fall | 0.308 | 0.136 | (no baseline) | n/a |
| helmet | 0.353 | 0.192 | (no baseline) | n/a |
| shoes | 0.472 | 0.270 | (no baseline) | n/a |
| phone | 0.486 | 0.226 | 0.529 | **-8%** |
| **mean** | **0.386** | 0.192 | — | — |

Best val checkpoint: ep37, `eval_mean_mAP_50=0.398`. Run dir:
`runs/dfine_n_multitask_2026-05-17_141112/`.

**Verdict**: architecture validated (+29% past R1's 0.30 plateau). Phone within 8% of
single-task → production-acceptable. Fire dropped 30% — backbone capacity is split 5
ways while single-task fire owned the entire 4M trunk. Recommend dfine-s next to
recover fire (more trunk capacity = less per-task starvation).

## Architecture

```
Image (B,3,640,640)
   │
   ▼ HGNetv2 backbone (shared)
   ▼ HybridEncoder (shared)
   ▼ Transformer decoder × 6 layers (shared)
   ▼ DFL box head (shared — box geometry is task-agnostic)
   │
   ├── cls_head_fire_smoke    (Linear hidden_dim → 2)
   ├── cls_head_fall          (Linear hidden_dim → 2)
   ├── cls_head_helmet        (Linear hidden_dim → 4)
   ├── cls_head_shoes         (Linear hidden_dim → 3)
   └── cls_head_phone         (Linear hidden_dim → 1)
```

Shared params: ~4 M (HGNetv2-N + encoder + decoder + box head)
Per-task params: ~0.05 M × 5 = 0.25 M
Total: ~4.25 M (≈ single-task D-FINE-N)

## Why this approach

- **Failed alternatives**:
  - Single 19-class head (`unified_detection/`) → backbone learns class-averaged features → 0.30 plateau
  - Frozen unified backbone + per-task head (R2 experiment) → 0.30 plateau (frozen features too generic)
- **Multi-task heads** give each task its own discriminative head while sharing the expensive trunk → backbone gets direct task-specific gradient from every task.

## Training notes

- **Sampling**: `round_robin_sqrt` — task sampled with prob ∝ √(N_train). Prevents helmet/shoes from drowning fall/fire.
- **Batches stay single-task** (each batch routes to one cls head) — mixed-task batches are not currently supported.
- **Backbone init**: COCO (`ustc-community/dfine_n_coco`). Do NOT init from `unified_detection/runs/` (verified worse).
- **CDN disabled initially** (`num_denoising: 0`) — re-enable after baseline works to avoid interaction effects.
- **bf16: false** — D-FINE invariant.

## Per-task eval

`MultitaskHFTrainer.evaluation_loop` runs full eval separately per task, logs:
- `eval_fire_smoke_mAP_50`
- `eval_fall_mAP_50`
- ...
- `eval_mean_mAP_50` (checkpoint selection metric)

Best-checkpoint selection uses `mean_mAP_50` (balanced across all tasks).

## Files

```
configs/
  05_data.yaml                              — 5 tasks, dataset paths, sampling weights
  06_training_dfine_n_multitask.yaml        — N (4M) — first to validate architecture
  06_training_dfine_s_multitask.yaml        — S (11M) — run after N validates
  06_training_dfine_m_multitask.yaml        — M (19M, bs=16) — run after S beats N
code/                                       — (empty; uses core/ implementations)
runs/<arch>_<ts>/                           — checkpoints + per-task eval logs
```

## Arch progression strategy

Multi-task changes the size-vs-data calculus: 5 tasks share ~75k images of
backbone supervision, so bigger D-FINE is justified here (unlike single-task
fire_smoke_fall where M underfit at ~20k images).

| Step | Arch | Train time (est) | Gate to advance |
|---|---|---:|---|
| 1 | dfine-n | ~6 h | ≥80% of single-task baselines on all 5 tasks |
| 2 | dfine-s | ~14 h | beat N by ≥5% mean mAP_50 |
| 3 | dfine-m | ~24 h | beat S by ≥2% mean mAP_50 |
| 4 | dfine-l | (skip Phase 1) | revisit at Phase 2 (13 tasks) |

Stop at first arch that fails its gate — no point burning GPU on diminishing returns.

## Required core/ implementations (Days 2-3)

- `core/p06_models/dfine_multitask.py`   — DFineMultitaskModel wrapping DFineForObjectDetection
- `core/p05_data/multitask_dataset.py`   — MultitaskInterleaver (round-robin sampler)
- `core/p06_training/multitask_trainer.py` — MultitaskHFTrainer (per-task loss + eval)
- Register `dfine-n-multitask` arch in `core/p06_models/__init__.py`

## Pipeline checklist

- [x] Pre-work P1: kill R2 v2
- [x] Pre-work P2: verify all 5 datasets exist in training_ready/
- [x] Pre-work P3: collect available baselines (phone=0.529; others TBD)
- [x] Pre-work P5: feature folder scaffold + configs
- [x] Implement DFineMultitaskModel
- [x] Implement MultitaskInterleaver dataset
- [x] Implement MultitaskHFTrainer
- [x] Full 5-task training dfine-n (40 ep) — best val 0.398 / test 0.386
- [ ] dfine-s training (scale-up — configs ready at `06_training_dfine_s_multitask.yaml`)
- [ ] Post-hoc per-task error_analysis on dfine-n best ckpt
- [ ] Per-task ONNX export
- [ ] Inference adapter for multi-head outputs

## Gotchas (load-bearing)

1. **Each batch must be single-task** — cls head routing requires it. Collator groups by task_name.
2. **Person class appears in 3 task heads** (fall, helmet, shoes) — these are *independent* heads with their own person definitions. They do NOT collide; the backbone simply gets reinforced person-shape signal from all 3.
3. **Sampling strategy matters** — `round_robin_sqrt` is the default. `uniform` over-samples small tasks (helmet→4× phone effect); `proportional` drowns small tasks.
4. **Eval runs N forward passes per image** when computing per-task mAP — eval time scales linearly with task count.
5. **Don't init backbone from unified_detection** — that backbone is mixed-specific, not generic, and underperforms COCO init.
6. **Per-task data starvation is real** — with sqrt-weighted sampling and N=5 tasks, each task gets ~20% of batches. For tasks with rich single-task baselines (fire_smoke: 12k imgs, single-task 0.45), multi-task starves them ~5× → observed 30% mAP regression on dfine-n. The trade is real: 5× deployment efficiency vs N% per-task accuracy loss. Mitigate via bigger arch (dfine-s/m has more shared trunk capacity to absorb the 5-way split) or per-task loss reweighting (`task_loss_weights`). Tasks with no single-task baseline (helmet/shoes/fall here) lose nothing because they had no baseline to lose against.
7. **`class_metrics=True` is mandatory in `compute_metrics`** — without it, `MeanAveragePrecision` returns `map_per_class=-1.0` and per-class breakdowns are invisible. `_build_per_task_compute_metrics` in `multitask_trainer.py` enforces this and unpacks the per-class vector with each task's `id2label` so keys land as `eval_<task>_map_50_per_class_<classname>`. Verified post-fix on dfine-n best ckpt.

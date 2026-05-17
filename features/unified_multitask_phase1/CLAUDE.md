# unified_multitask_phase1

**Type:** Multi-task detection (5 Phase 1 tasks, shared backbone)
**Architecture:** D-FINE-N shared trunk + 5 per-task classification heads
**Goal:** One ONNX file delivering all Phase 1 detection capabilities in a single forward pass.

## Tasks

| Head | Classes | Dataset | Single-task baseline |
|---|---|---|---:|
| fire_smoke | fire, smoke | `training_ready/fire_detection` | TBD |
| fall | person, fallen_person | `training_ready/fall_detection` | TBD |
| helmet | person, head_with_helmet, head_without_helmet, head_with_nitto_hat | `training_ready/helmet_detection` | TBD |
| shoes | person, foot_with_safety_shoes, foot_without_safety_shoes | `training_ready/shoes_detection` | TBD |
| phone | phone_usage | `training_ready/safety_poketenashi_phone_usage` | 0.529 mAP50 |

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
  06_training_dfine_n_multitask.yaml        — multi-task training recipe
code/                                       — (empty; uses core/ implementations)
runs/<arch>_<ts>/                           — checkpoints + per-task eval logs
```

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
- [ ] Implement DFineMultitaskModel
- [ ] Implement MultitaskInterleaver dataset
- [ ] Implement MultitaskHFTrainer
- [ ] PoC: train fire_smoke + helmet (cross-domain, 20 ep)
- [ ] Full 5-task training (40 ep)
- [ ] Per-task ONNX export
- [ ] Inference adapter for multi-head outputs

## Gotchas (load-bearing)

1. **Each batch must be single-task** — cls head routing requires it. Collator groups by task_name.
2. **Person class appears in 3 task heads** (fall, helmet, shoes) — these are *independent* heads with their own person definitions. They do NOT collide; the backbone simply gets reinforced person-shape signal from all 3.
3. **Sampling strategy matters** — `round_robin_sqrt` is the default. `uniform` over-samples small tasks (helmet→4× phone effect); `proportional` drowns small tasks.
4. **Eval runs N forward passes per image** when computing per-task mAP — eval time scales linearly with task count.
5. **Don't init backbone from unified_detection** — that backbone is mixed-specific, not generic, and underperforms COCO init.

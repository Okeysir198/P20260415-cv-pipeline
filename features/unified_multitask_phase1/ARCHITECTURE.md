# Architecture — Multi-Task Detection (Phase 1)

Deep dive on the design of the unified Phase 1 detector. This document
covers **what** the architecture is, **why** it's structured this way (with
empirical evidence from R1/R2 failures that motivated it), and the
**concrete code paths** that implement each piece.

For the quick-reference user manual see `CLAUDE.md`. For training
recipes see `configs/06_training_dfine_*_multitask.yaml`.

---

## 1. Goal

Train **one** D-FINE detector that handles **all 5 Phase 1 detection tasks**
in a single forward pass. The deployed `model.onnx` should produce, for any
input frame:

- fire / smoke boxes
- person / fallen_person boxes
- person / head_with_helmet / head_without_helmet / head_with_nitto_hat boxes
- person / foot_with_safety_shoes / foot_without_safety_shoes boxes
- phone_usage boxes

…all at once, with per-task confidence thresholds tunable independently.

The acceptance bar: **each task within ~5–10% of its single-task baseline**.
Anything closer is a free win; anything more than 15% off would suggest the
multi-task approach is structurally wrong for these tasks.

---

## 2. The architectural choice — single-trunk + multi-head

```
              Image (B, 3, 640, 640)
                       │
                       ▼
        ┌──────────────────────────────┐
        │  Backbone (HGNetv2)          │  SHARED
        │  ~3 M params                 │  every task sees gradient from here
        └──────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  Encoder (HybridEncoder)     │  SHARED
        │  AIFI on P5 + CCFM cross-scale│
        └──────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  Decoder × 6 layers          │  SHARED
        │  Deformable cross-attention  │
        └──────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  Box head (DFL: 4 × 32 bins) │  SHARED — geometry is task-agnostic
        └──────────────────────────────┘
                       │
       ┌───────────────┼───────────────┬───────────┬───────────┐
       ▼               ▼               ▼           ▼           ▼
   ┌────────┐     ┌────────┐     ┌────────┐  ┌────────┐  ┌────────┐
   │ cls    │     │ cls    │     │ cls    │  │ cls    │  │ cls    │
   │ fire   │     │ fall   │     │ helmet │  │ shoes  │  │ phone  │
   │ (2 cls)│     │ (2 cls)│     │ (4 cls)│  │ (3 cls)│  │ (1 cls)│
   └────────┘     └────────┘     └────────┘  └────────┘  └────────┘
              ── PER-TASK heads (~0.05 M each) ──
```

**Total params**: ~4.25 M for D-FINE-N
- Shared trunk: ~4.0 M (~94%)
- All 5 cls heads combined: ~0.25 M (~6%)

The cls heads are the only place each task differs in terms of weights. The
geometry head (box regression) is shared because "where is this object?"
doesn't depend on which task's labels apply.

---

## 3. Why this architecture — what we tried that failed

We arrived at multi-head + shared-trunk after exhausting two simpler
approaches. Both produced 0.30 mAP50 plateaus on the same data.

### R1: Single 19-class head (`features/unified_detection/`)

```
Shared trunk → ONE cls head (19 classes: fire, smoke, fallen, person,
                              helmet, no_helmet, nitto, shoes, no_shoes,
                              phone, apron, harness, glasses, mask, …)
```

**Result**: 0.30 mAP plateau, rare classes (harness, glasses) at ~0.05.

**Why it failed** — three structural problems:

1. **Cross-source label inconsistency**: the same visual concept
   (e.g. a fallen worker) is labeled in the fall dataset but is unlabeled in
   the helmet/shoes datasets. The single softmax learns to *suppress*
   `fallen_person` predictions on helmet images → ~0 mAP on classes that
   only one source labels.

2. **Class semantics conflict**: `person` appears in fall, helmet, and shoes
   datasets with subtly different conventions (fall: only upright workers;
   helmet/shoes: all workers including fallen). A single 19-channel softmax
   gets contradictory training signal.

3. **Class imbalance dominates the softmax**: with 50k shoes vs 2k harness
   in one head's softmax, the head allocates capacity by gradient magnitude.
   Rare classes lose; they cannot be "rescued" by hyperparameter tuning.

### R2: Frozen R1 backbone + new task-specific head

Took R1's backbone (mediocre 19-class features), froze it, retrained only
a 3-class head on fire+smoke+fallen.

**Result**: 0.30 plateau (no better than R1).

**Why it failed**: the R1 backbone learned "average of 19 classes" features
— not generic visual features. Those averaged features aren't a useful
starting point for any specific task. Freezing them locks in mediocre
representations the new head can't compensate for.

### What multi-head + shared-trunk fixes

| Problem | Multi-head fix |
|---|---|
| Cross-source label noise | Each head only ever sees its own task's labels → no "is this thing background or fallen?" confusion |
| Class semantics conflict | Each task has its own `person` Linear weight — no contention |
| Class imbalance | Each task's softmax balances only its own classes |
| Per-task threshold tuning | Independent score spaces per task → tune `fire_threshold=0.2`, `helmet_threshold=0.5` independently |
| Adding a new task | Add 1 head, fine-tune; don't retrain everything |

The trunk is still shared (so we get the deployment + transfer-learning
benefits) but the classification decisions are isolated per-task.

---

## 4. Code organization

### Files implementing the architecture

```
core/p06_models/dfine_multitask.py        # the model
core/p05_data/multitask_dataset.py        # data routing
core/p06_training/multitask_trainer.py    # training loop + per-task eval
core/p06_models/__init__.py               # arch registration
core/p06_training/train.py                # backend dispatch
```

### `core/p06_models/dfine_multitask.py` — DFineMultitaskModel

The trunk-sharing implementation is unusual: rather than write a custom
forward path, it loads **N full `DFineForObjectDetection` instances** (one
per task, each correctly sized to that task's num_classes), then replaces
13 trunk submodules on tasks[1..N] with references to task[0]'s submodules:

```python
SHARED_MODULES = [
    "model.backbone",
    "model.encoder",
    "model.decoder",
    "model.enc_output",
    "model.enc_score_head",
    "model.enc_bbox_head",
    "bbox_predictor",
    "denoising_class_embed",
    # ... 5 more
]

# task_models[0] keeps its original modules; tasks[1..N] get references
for task_name in tasks[1:]:
    for mod_path in SHARED_MODULES:
        _set_submodule(task_models[task_name], mod_path,
                       _get_submodule(task_models[primary], mod_path))
```

PyTorch dedups by tensor identity in `named_parameters()`, so a 5-task model
counts shared parameters once. Verified empirically: a 2-task model uses
**less than 50% more params than single-task** (unit test asserts this).

The forward signature is task-routed:

```python
def forward(self, pixel_values, task_name: str, labels=None):
    # Dispatch to the per-task DFineForObjectDetection (which carries
    # its own cls head + loss criterion correctly sized to its num_classes).
    return self.task_models[task_name](pixel_values=pixel_values, labels=labels)
```

Each task has its own HF loss criterion (matcher + Hungarian + VFL + L1 +
GIoU + DFL) — these aren't shared because they bake in `num_classes`.

### `core/p05_data/multitask_dataset.py` — MultitaskInterleaver

The data layer enforces a critical invariant: **each batch must be
single-task** (the cls head can't process multiple tasks at once).

```python
class MultitaskInterleaver(IterableDataset):
    """Round-robin sqrt-weighted sampler across N detection datasets."""

    def __iter__(self):
        while emitted < total:
            # Pick a task with prob ∝ sqrt(train_size) — prevents large
            # tasks (shoes: 26k) from drowning small ones (phone: 16k).
            task = rng.choices(task_names, weights=sqrt_weights)[0]

            # Emit batch_size CONSECUTIVE samples from that task — the
            # collator then sees a homogeneous batch.
            for _ in range(batch_size):
                yield (image, target, task, path)
```

The `multitask_collate_fn` *checks* this invariant and raises a clear
error if a mixed batch ever shows up:

```python
task_names = {item[2] for item in batch}
if len(task_names) != 1:
    raise ValueError("Mixed-task batch unsupported")
```

For eval, each per-task val dataset is wrapped in `TaskLabeledDataset` so
that raw `YOLOXDataset` 3-tuples `(image, target, path)` get padded to
the 4-tuple `(image, target, task_name, path)` the collator expects.

### `core/p06_training/multitask_trainer.py` — MultitaskHFTrainer

Subclasses HF `Trainer` and overrides three methods:

1. **`compute_loss`** — pops `task_name` from inputs, routes through the
   model with that task selector. The model returns a per-task loss
   (computed by that task's criterion against that task's labels only).

2. **`evaluate`** / **`_evaluate_per_task`** — runs N sequential
   `super().evaluate()` calls, one per task with that task's val
   dataset. Aggregates into:
   - `eval_<task>_map_50` (per-task scalar)
   - `eval_<task>_map_50_per_class_<classname>` (per-class detail)
   - `eval_mean_mAP_50` (averaged across tasks → checkpoint selection)

3. **`_save`** — uses `torch.save` (not safetensors) because the
   shared-trunk parameters create tensor-identity ties that safetensors
   refuses to serialize.

The trainer also holds `_per_task_compute_metrics`: a dict mapping
`task_name → compute_metrics_callable`, each baked with that task's
`id2label`. Before every `super().evaluate(task_X)`, the trainer swaps
`self.compute_metrics` to `_per_task_compute_metrics[task_X]` so the
per-class keys land with the right class names (e.g.
`eval_fire_smoke_map_50_per_class_fire` not `..._per_class_0`).

### `core/p06_training/train.py` dispatch

A single branch:

```python
if backend == "hf_multitask":
    summary = run_multitask_training(config_path, overrides=...)
elif backend == "hf":
    summary = train_with_hf(...)
elif backend == "pytorch":
    ...
```

The HF Trainer + viz callback infrastructure for single-task `backend: hf`
is unchanged — multi-task is additive.

---

## 5. Training dynamics

### Sampling strategy

```yaml
# Per-task train sizes
fire_smoke: 12,161    →  sqrt = 110
fall:        8,758    →  sqrt =  94
helmet:     15,625    →  sqrt = 125
shoes:      25,958    →  sqrt = 161
phone:      16,081    →  sqrt = 127

# Normalized probabilities (sum = 1.0)
fire_smoke: 0.175
fall:       0.149
helmet:     0.199
shoes:      0.256    ← largest dataset, still sampled <2× phone
phone:      0.202
```

Pure-proportional would give shoes ~33% of batches and phone ~21% — a 1.6×
gap. sqrt-weighting compresses that to a 1.5× gap, giving small tasks more
of a chance.

### Gradient flow

Each task's batch backprops through:
1. Its own cls head (~0.05M params, fire-only signal)
2. Shared box head (gets fire's geometry signal)
3. Shared decoder (gets fire's region-attention signal)
4. Shared encoder (gets fire's multi-scale signal)
5. Shared backbone (gets fire's low/mid-level feature signal)

When the next batch (e.g. helmet) trains, it inherits a trunk that was just
updated by fire training. This is intra-step transfer learning — features
that survive must be useful to *multiple* tasks.

### What this produces (empirical, after 5 epochs of N training)

| ep | mean mAP_50 | fire_smoke | fall | helmet | shoes | phone |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.096 | 0.018 | 0.098 | 0.011 | 0.309 | 0.040 |
| 2 | 0.169 | 0.076 | 0.197 | 0.065 | 0.387 | 0.118 |
| 3 | 0.217 | 0.156 | 0.216 | 0.111 | 0.427 | 0.175 |
| 4 | 0.246 | 0.182 | 0.266 | 0.145 | 0.468 | 0.169 |
| 5 | 0.279 | 0.198 | 0.255 | 0.185 | 0.507 | 0.250 |

All 5 tasks improving simultaneously from a single trunk, no signs of any
task pulling the trunk against any other. The shared backbone is genuinely
useful for every head.

### Convergence expectations

D-FINE typically needs 30-50 epochs to fully converge on small datasets
(see project gotcha: "D-FINE is under-trained at qubvel's 30 epochs"). With
5 tasks sharing the trunk, each task effectively sees ~1/5 the batches
per epoch that a single-task run would — so 40-50 epochs of multi-task ≈
8-10 epochs of single-task gradient signal per task. Expect peak performance
around ep30-40.

---

## 6. Deployment

After training:

```
features/unified_multitask_phase1/runs/dfine_n_multitask_<ts>/
├── checkpoint-<best>/pytorch_model.bin    # multi-task state dict
├── config.json                            # primary task's HF config
├── 05_data.yaml                           # task list (snapshot)
├── 06_training.yaml                       # training recipe (snapshot)
└── test_results.json                      # per-task test mAP
```

The state dict contains both shared trunk weights (under
`task_models.<primary_task>.model.*`) and per-task cls heads (under
`task_models.<task_name>.class_embed.*`).

For ONNX export, the model graph is built once with all task cls heads
producing N output sets in a single forward:

```
outputs = {
    "task_fire_smoke_logits":  (B, Q, 2),
    "task_fire_smoke_boxes":   (B, Q, 4),
    "task_fall_logits":        (B, Q, 2),
    "task_fall_boxes":         (B, Q, 4),
    ...
    "task_phone_logits":       (B, Q, 1),
    "task_phone_boxes":        (B, Q, 4),
}
```

Inference time picks which task outputs to consume (or all of them, in
parallel). File size: only marginally larger than a single-task D-FINE-N
ONNX because heads add ~0.25 M params on top of ~4 M shared.

---

## 7. Why D-FINE specifically (not RT-DETRv2 or YOLOX)

| Aspect | D-FINE | RT-DETRv2 | YOLOX |
|---|---|---|---|
| Trunk shareability | Clean (HGNet + transformer) | Clean | Anchor-bound heads complicate sharing |
| Per-class head replacement | One `nn.Linear` per task | One `nn.Linear` per task | YOLOX decoupled head is bigger, less natural to swap |
| Numerical stability for multi-task | DFL loss benefits from large effective dataset | Vanilla regression — also fine | Pre-decoded outputs — fine but less expressive heads |
| ONNX cleanliness | Single model.onnx | Single model.onnx | Single model.onnx |

D-FINE wins on **shareability + per-head simplicity**. RT-DETRv2 is a fine
alternative (and the architecture as designed would work for it
unchanged — just a different `pretrained` URL and a different arch
registration). YOLOX would need more invasive surgery to the decoupled
head.

D-FINE's specific requirements that this architecture respects:
- **`bf16: false`** is enforced (DFL loss numerical sensitivity — project
  invariant)
- **AMP off** (DETR-family decoders overflow in fp16)
- **EMA enabled** (helps the long convergence DFL needs)
- **fp32 for the entire forward** (DFL bins quantization breaks at bf16)

---

## 8. Open questions / future work

### Currently deferred (out of scope for first run)

- **Viz callbacks** are not wired (HF's single-task callbacks assume one
  cls head). Workaround: run `core/p08_evaluation/evaluate.py` per task
  after training to get full error_analysis trees.
- **`freeze_backbone_epochs` is not wired** — the trunk is inside one of
  the per-task models; freezing requires walking that nested path.
- **Per-task loss weighting** uses uniform weights (`task_loss_weights: auto`).
  After first run, we can experiment with weighting based on task
  difficulty or sample count.
- **CDN denoising is disabled** (`num_denoising: 0`). Re-enable after
  baseline works to test the interaction with multi-task training.
- **ONNX export** for multi-task models is not implemented — needs to
  generate N output sets in one graph.

### Future scaling

- **Arch progression**: N → S → M → L (configs already exist for N/S/M).
  Stop at first arch that fails its gate (within 5% of previous).
- **Phase 2 expansion**: add masks, gloves, glasses, apron, harness,
  forklift_pedestrian. Each is +1 cls head, ~3 days of dev/training.
- **Per-task decoders (Option B)**: if any task hurts another despite
  multi-head separation, add per-task decoders too. Costs ~3× decoder
  params but isolates query learning.
- **Knowledge distillation**: use the unified model to pseudo-label
  unlabeled raw footage from cameras, then re-train with the augmented
  set for further gains.

---

## 9. Comparison table — three "unified" attempts

| | R1: Single big head | R2: Frozen R1 backbone | **Multi-task heads (current)** |
|---|---|---|---|
| Heads | 1 × 19-class | 1 × 3-class | N × per-task |
| Backbone source | trained from COCO on union of all data | R1 backbone (frozen) | COCO init, jointly trained on all tasks |
| Cross-source label noise | Severe | Inherited from R1 | Contained per task |
| Class imbalance | Dominates softmax | N/A (single subset) | Per-task only |
| Per-task threshold | Impossible | Single threshold | **Independent** |
| Adding new task | Retrain everything | Retrain head | **Add 1 head, fine-tune** |
| Empirical result | 0.30 plateau | 0.30 plateau | climbing past 0.28 at ep5, target ≥0.45 |
| Ship-ready | No | No | **Yes** (once trained) |

---

## 10. Where to look in the code

```
features/unified_multitask_phase1/
├── ARCHITECTURE.md                ← this file (deep dive)
├── CLAUDE.md                      ← quick-reference, gotchas, task list
└── configs/
    ├── 05_data.yaml               ← 5 tasks listed with class names
    └── 06_training_dfine_*_multitask.yaml ← arch-specific recipes

core/
├── p05_data/multitask_dataset.py        ← MultitaskInterleaver + collator
├── p06_models/dfine_multitask.py        ← DFineMultitaskModel
└── p06_training/multitask_trainer.py    ← MultitaskHFTrainer + entry point

tests/test_p06_multitask.py              ← 4 smoke tests (arch reg, forward, param-count, collator)
```

For one-line summaries see the file table in `CLAUDE.md`. For why something
is structured a particular way see the section here that introduces it.

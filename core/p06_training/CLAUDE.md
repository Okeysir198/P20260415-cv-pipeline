# CLAUDE.md — core/p06_training/

Authoritative notes on the training loop(s), callbacks, and the choice between
the pytorch and HF Trainer backends. Companion to `README.md` — this file
covers what's *between* files and the gotchas learned the hard way.

## Two backends, one config

`training.backend` in `06_training.yaml` picks the execution path:

| Backend | File | Use when |
|---|---|---|
| `pytorch` (default) | `trainer.py::DetectionTrainer` | You need EMA + per-component LR + the full custom callback suite (`DatasetStatsLogger`, `AugLabelGridLogger`, `DataLabelGridLogger`, `ValPredictionLogger`, `CheckpointSaver`, `EarlyStopping`, `WandBLogger`). Covers every task type (detection, classification, segmentation, pose, keypoint). |
| `hf` | `hf_trainer.py::train_with_hf` | Detection / classification / segmentation when you want DDP / DeepSpeed / gradient accumulation / bf16 tensor-core paths "for free", HF-Trainer-standard output layout (`checkpoint-N/`, `runs/<ts>_<host>/` TB, `trainer_state.json`), and the reference-notebook code pattern for DETR-family fine-tuning. |

Both respect the same YAML config keys; the HF backend falls back or warns
on features it doesn't implement yet. See `_validate_hf_backend_config`
(`hf_trainer.py`) for the allow/deny list.

### HF backend support matrix (detection)

| Config key | Respected | Notes |
|---|---|---|
| `training.backend: hf` | ✓ | dispatches through `train_with_hf` |
| `training.epochs` | ✓ | `num_train_epochs` |
| `training.lr`, `weight_decay` | ✓ | |
| `training.scheduler` | ✓ | passed as `lr_scheduler_type` |
| `training.warmup_steps` | ✓ (preferred) | HF `warmup_steps` directly; if missing, `warmup_epochs` → `warmup_ratio` |
| `training.max_grad_norm` / `grad_clip` | ✓ | |
| `training.bf16` | ✓ | `True` OK for RT-DETRv2; **must be `False` for D-FINE** (DFL stalls val at ~0.15 under bf16). fp16/`amp` overflows both — use bf16 or fp32 only. |
| `training.amp` | ✓ | validator **hard-errors** if True for detection |
| `training.patience` | ✓ | `EarlyStoppingCallback` |
| `training.ema` | ✓ | native `EMACallback` wrapping our `ModelEMA` — swaps weights in/out around each eval |
| `training.gpu_augment` | ✗ | HF Trainer uses its own DataLoader; warning emitted. Use `augmentation.library: albumentations` for CPU-aug speed parity (~2× faster than torchvision v2) |
| `training.val_full_interval` | partial | HF evaluates every epoch by default; this knob is effectively ignored |
| `data.subset.{train,val}` | ✓ | wraps in `torch.utils.data.Subset` with deterministic seed |
| `augmentation.library: albumentations` | ✓ | fast CPU aug backend with probability-gated transforms |
| `checkpoint.metric: val/mAP50` | ✓ | auto-translated to HF's `eval_map_50` |
| `checkpoint.save_best: true` | ✓ | uses HF's `load_best_model_at_end` |
| Viz callbacks | ✓ native | `hf_callbacks.py` — four first-class `TrainerCallback` subclasses mirroring the pytorch-backend loggers. No proxy-trainer hack. |
| Final test-set eval | ✓ auto | writes `<output_dir>/test_results.json` when a test split is present |

If a task / config combo isn't supported, `_validate_hf_backend_config` fails
fast at the top of `train_with_hf` rather than silently degrading.

## Files

| File | Purpose |
|---|---|
| `trainer.py` | `DetectionTrainer` — main training loop (pytorch backend). Auto-detects HF vs YOLOX model path, per-component LR groups, EMA, gradient clipping, callback dispatch. |
| `hf_trainer.py` | `train_with_hf`, `_DetectionTrainer` (Trainer subclass with shared-weight-safe `_save`), `EMACallback`, detection collator + real mAP `compute_metrics` (torchmetrics-based). Config validator enforces hard incompatibilities up-front. |
| `hf_callbacks.py` | Native `TrainerCallback` subclasses — `HFDatasetStatsCallback`, `HFDataLabelGridCallback`, `HFAugLabelGridCallback`, `HFValPredictionCallback` — that run the same viz outputs as the pytorch backend's `callbacks.py` counterparts but consume HF's documented hook kwargs (`model`, `eval_dataloader`, `state.log_history`) rather than a proxy trainer object. |
| `train.py` | CLI entry point — `auto_select_gpu`, determinism knobs (CUBLAS env var + `torch.use_deterministic_algorithms(True, warn_only=True)`), 3-warning filter for known-harmless PyTorch messages, dispatches to backend. |
| `callbacks.py` | `Callback` base class (pytorch backend only), `CheckpointSaver`, `EarlyStopping`, `WandBLogger`, `ValPredictionLogger`, `DatasetStatsLogger`, `DataLabelGridLogger`, `AugLabelGridLogger`, `CallbackRunner`. Also `_run_splits_and_subsets(trainer)` — now iterates train/val/test so the HF bridge's stub test-loader shows up in data_preview. |
| `callbacks_viz.py` | `TransformPipelineCallback` — fires once on train-start, renders `data_preview/04_transform_pipeline.png` (K rows × N cols: one representative sample per class walked step-by-step through the CPU transform pipeline; last col = Denormalize(Normalize) inverse check). Dual-backend (permissive `_AnyHook` base satisfies both pytorch `CallbackRunner` and HF `CallbackHandler` hook surfaces). |
| `post_train.py` | Backend-agnostic post-train runner. `run_post_train_artifacts(model, save_dir, val_dataset, test_dataset, task, class_names, input_size, style, training_config, …)` renders best-checkpoint val+test grids and dispatches to `error_analysis_runner`. `render_prediction_grid` is the **single** grid renderer (per-epoch, best, hardest-overview all route here via `annotate_gt_pred`). |
| `_common.py` | Shared helpers: `unwrap_subset`, `task_from_output_format`, `yolo_targets_to_xyxy`. Dedupes logic that previously had 3 copies across HF + pytorch backends. |
| `losses.py` | `DetectionLoss` ABC, `YOLOXLoss` (SimOTA), `FocalLoss`, `IoULoss`, `_DETRPassthroughLoss`, registry + `build_loss()`. |
| `lr_scheduler.py` | `WarmupScheduler`, `CosineScheduler`, `PlateauScheduler`, `StepScheduler`, `OneCycleScheduler` + `build_scheduler()`. |
| `postprocess.py` | `POSTPROCESSOR_REGISTRY`, YOLOX-only decoding (HF models use built-in `post_process_object_detection`). |
| `metrics_registry.py` | `METRICS_REGISTRY`, `register_metrics()`, per-format validation metrics dispatch (pytorch backend only). |

## Post-train observability (on `on_train_end`, both backends)

Every training run produces a uniform per-run artifact tree — no per-config opt-in. Driven by `post_train.run_post_train_artifacts` + `core/p08_evaluation/error_analysis_runner.run_error_analysis`.

```
runs/<ts>/
├── data_preview/               (on_train_start, ~2 s total — numbered by data-flow step)
│   ├── 00_dataset_info.{md,json}         ← provenance: feature, dataset, classes, splits, input_size, aug
│   ├── 01_dataset_stats.{png,json}       split sizes + class balance + bbox tiers
│   ├── 02_data_labels_{train,val,test}.png   raw images with GT
│   ├── 03_aug_labels_train[_mosaic].png  CPU augmentation output
│   └── 04_transform_pipeline.png    step-by-step transform walk (per class, max 5 samples); last col = Denormalize(Normalize) sanity check
├── val_predictions/
│   ├── epochs/epoch_NNN.png    (per-epoch, ~2 s each — the only mid-run hook)
│   ├── best.png                (on_train_end, best-checkpoint weights)
│   └── error_analysis/         (task-dispatched; ~15 s total)
│       ├── summary.{json,md}       3-axis: data_distribution + training_config + model_metrics
│       │                             summary.md opens with the failure-mode ranking table (Δ mAP50 per mode)
│       │                             summary.json::model_metrics.failure_mode has
│       │                               {error_types, per_class, confusion_pairs_top, contribution,
│       │                                modes_per_class, fn_attribution, miss_by_attribute, diagnosis}
│       ├── data_distribution.png   class count + per-class × size-tier
│       ├── boxes_per_image.png     crowdedness (mean/median/p95/max)
│       ├── bbox_aspect_ratio.png   per-class log-scale w/h
│       ├── per_class_pr_f1.png     P / R / F1 bars
│       ├── confusion_matrix.png    GT×Pred (last col/row = background)
│       ├── confidence_calibration.png  TP vs FP score histogram
│       ├── size_recall.png         small / medium / large with explicit COCO px² thresholds
│       ├── threshold_analysis.png  4-panel: PR curves + F1/P/R vs conf threshold (step 0.01)
│       ├── map_vs_iou.png          mAP at IoU 0.50 → 0.95 (AP50 / AP75 / AP[.5:.95])
│       ├── failure_mode_contribution.png   2-panel: global Δ mAP ranked bars + per-class × mode heatmap + glossary
│       ├── failure_by_attribute.png        2×2: miss-rate by size / aspect-ratio / crowdedness + top confusion pairs
│       ├── confidence_attribution.png      FN causality: true_miss / under_confidence / localization_fail (abs + normalized)
│       ├── recoverable_map_vs_iou.png      per-mode Δ mAP across IoU 0.5 → 0.9 (shows where localization dominates COCO mAP)
│       ├── hardest_images.png      top-12 overview
│       └── failure_mode_examples/  5-mode × per-class GT-vs-Pred galleries (replaces old hard_images/)
│           ├── missed/<class>/<stem>__missed__area_<px>.png
│           ├── localization/<class>/<stem>__loc__iou_<x>_score_<y>.png
│           ├── class_confusion/<pred>__from__<gt>/<stem>__conf__iou_<x>_score_<y>.png
│           ├── duplicate/<class>/<stem>__dup__iou_<x>_score_<y>.png
│           └── background_fp/<class>/<stem>__bgfp__score_<y>.png
├── test_predictions/           same layout as val_predictions/
└── test_results.json           HF Trainer metrics on the test split
```

`VizStyle` (core/p10_inference/supervision_bridge.py) is the single source of truth for colors/thickness/text — no per-site drawing constants. `training_config` in `summary.json` snapshots arch / params / lr / optimizer / scheduler / bf16 / best-metric / test-metrics from both backends.

Opt out per block in YAML (all default true):
```yaml
training:
  data_viz:  { enabled: false }
  aug_viz:   { enabled: false }
  norm_viz:  { enabled: false }
  val_viz:   { enabled: false }    # still leaves best_viz + error_analysis on
  best_viz:  { enabled: false }
  error_analysis: { enabled: false }
  post_train: { enabled: false }   # pytorch-backend only: skip best-reload + test eval
```

**pytorch-backend `_finalize_training`** (trainer.py): on train-end, reloads `best.pth`, auto-builds the test-split loader via `YOLOXDataset(split="test")` with `base_dir=self.config_path.parent` (so `05_data.yaml::path: "../../../dataset_store/..."` resolves correctly), runs `_validate(test_loader)` → writes `test_results.json`, then dispatches to `run_post_train_artifacts`. Brings pytorch backend to parity with HF's `load_best_model_at_end` + auto-test convention.

## Config templates

- **Detection, HF backend** → `configs/_shared/06_training_detection_hf.yaml`
  (the recipe that reproduced qubvel's CPPE-5 result). Copy + set the
  `model.num_classes` + `model.input_size` + point `dataset_config:` at the
  feature's `05_data.yaml`.
- **Detection, pytorch backend** → per-feature (`features/<f>/configs/06_training_rtdetr.yaml` etc.)
  using `DetectionTrainer` directly; see `safety-fire_detection`.
- **Classification/segmentation** → existing per-feature templates under
  `features/<f>/configs/`; both backends supported but pytorch is the primary.

## Determinism

`train.py` sets at import time:
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` env var (before torch imports)
- `torch.use_deterministic_algorithms(True, warn_only=True)` + cuDNN deterministic / non-benchmark

`warn_only=True` because RT-DETRv2's multi-scale deformable attention and
memory-efficient attention backward kernels lack deterministic CUDA impls
— strict mode crashes; warn-only locks the rest of the graph. Same tradeoff
as the reference notebook. Three specific warnings are filtered out as
known-harmless (`train.py` bottom of the determinism block); real errors
still surface.

HF Trainer's own `args.seed` + `data_seed` handle Python/NumPy/Torch RNG
and DataLoader sampler seed — all plumbed through from our YAML
`seed:` key.

**Early `set_seed` hook (hf_trainer.py)**: `train_with_hf` calls
`transformers.set_seed(config['seed'])` immediately before `build_model`,
because `from_pretrained(ignore_mismatched_sizes=True)` reinits class/bbox/
denoising heads inside that call. HF Trainer's own `args.seed` fires later
inside `Trainer.__init__` — too late. Without the early seed, D-FINE's
6 decoder `class_embed` heads picked up OS-entropy init and stalled val at
0.15; RT-DETRv2 still converged but with wider run-to-run variance.
Matches qubvel's convention from the reference notebooks.

## When to use which path — decision tree

```
Need EMA / per-component LR / custom callback registry?
├── Yes → backend: pytorch
└── No
    │
    Is the task detection / classification / segmentation?
    ├── Yes → backend: hf   (DDP/DeepSpeed/bf16 available, HF Trainer output layout)
    └── No  → backend: pytorch (pose/keypoint/face)
```

## Adding a new loss function

```python
from core.p06_training.losses import DetectionLoss, register_loss

@register_loss("my_loss")
class MyLoss(DetectionLoss):
    def forward(self, predictions, targets):
        return total_loss, {"cls_loss": ..., "reg_loss": ...}
```

Works on the pytorch backend. HF backend uses the model's internal
`forward_with_loss` (via `HFDetectionModel`) so it's loss-function-agnostic.

## Integration test

`tests/test_p06_training_hf_detection.py` runs four checks in ~20s on one
RTX GPU:
- Config validator rejects unsupported output_format
- Config validator rejects `amp=True` on detection
- One-epoch `rtdetr-r18` training on `test_fire_100` fixture: asserts
  HF-Trainer-standard file layout, `best_metric` finite, viz bridge dumps
  all three splits, `test_results.json` written with real mAP keys
- EMA callback produces `ema_model.bin`

Gotchas
-------

- **`resume_from_checkpoint=<path>` on HF backend**: supported, but note
  the checkpoint must have been saved by our `_DetectionTrainer._save`
  (wrapper-prefixed state dict `hf_model.*`) — not a bare `hf_model.save_pretrained`.
- **Viz callbacks on HF backend are native `TrainerCallback` subclasses**
  (`core/p06_training/hf_callbacks.py`). They share rendering helpers
  (`annotate_detections`, `save_image_grid`, `annotate_gt_pred` — all from
  `utils.viz`) with the pytorch-backend loggers but consume HF's documented
  kwargs (`model`, `eval_dataloader`, `state.log_history`) directly — no
  proxy-trainer attribute surface. Earlier bridge-adapter design dropped as
  of the native-callback migration.
- **Detection `compute_metrics` requires `eval_do_concat_batches=False`** —
  set automatically for detection in `_config_to_training_args`. Classifier/
  segmenter paths keep the default.
- **RT-DETRv2 shared-weights save** — the `_DetectionTrainer._save` override
  uses plain `torch.save` (not safetensors) for detection because
  RT-DETRv2 / D-FINE share `class_embed`/`bbox_embed` across decoder layers
  and safetensors rejects that. HF removed the `save_safetensors=False`
  TrainingArguments knob in 5.x.
- **Data-prep parsers trust actual image dims**, not annotation metadata
  — `core/p00_data_prep/parsers/_image_dims.py::actual_image_dims`. Verified
  necessary: ~6% of CPPE-5 validation rows have wrong HF metadata width/height.
  Use this helper in any new COCO/VOC/HF-dataset parser.
- **HF detection analyzer requires `model.processor`**: the error-analysis
  runner calls `_preprocess_for_model(image, input_size, model=model)`, which
  delegates to `model.processor` (HF `AutoImageProcessor`) when present.
  Without this path, DETR-family decoders receive un-normalized [0, 1] inputs
  and produce **zero predictions** — summary.json shows all-FN, pr_curves are
  empty, best.png shows only GT boxes. `HFDetectionModel` sets `self.processor`
  in `build_hf_model`; any new HF detection wrapper MUST do the same or wire
  a custom preprocess path. YOLOX (`output_format == "yolox"`) bypasses this
  and feeds raw [0, 255] to match the Megvii recipe.
- **Default `error_analysis_conf_threshold = 0.05`, not 0.3** — DETR-family
  models (RT-DETRv2, D-FINE) routinely produce correct predictions at scores
  in the 0.05–0.20 range; on CPPE-5 RT-DETRv2's max score on a typical val
  image was 0.176. Using 0.3 as the analyzer threshold dropped virtually
  every prediction, leaving TP≈0 / FN≈all in summary.json even when HF
  Trainer's own eval reported mAP50=0.82. With 0.05, TP counts land in the
  100s and PR curves / mAP-vs-IoU reflect real model behavior. Override via
  `training.post_train.error_conf_threshold` if you need different ops calibration.
- **Failure-mode Δ mAP is a counterfactual simulation, not a ground-truth gain**
  — `error_analysis_runner._compute_recoverable_map` iterates each of the 5
  modes, mutates the detection list (inject synthetic TPs for `missed`, flip
  class for `class_confusion`, bump same-class IoU to the eval threshold for
  `localization`, drop `duplicate` / `background_fp`), then recomputes AP.
  Numbers assume a *perfect* fix of one mode in isolation; real-world fixes
  interact (e.g. fixing localization shifts the PR-curve knee, which changes
  the Δ for background_fp). The ranking still correctly orders the biggest
  levers — just don't read "+0.23 on missed" as "augmenting will buy you 0.23
  mAP." Modes are tagged once at IoU 0.5 by the matcher; `recoverable_map_vs_iou`
  reuses those same tags but recomputes mAP at each IoU step — so `localization`
  Δ climbs at stricter IoUs because more of the "correct-at-0.5" bucket falls
  back into the fix-list as the threshold tightens.
- **HF `load_best_model_at_end` + wrapper-prefixed state dict** — our
  `_DetectionTrainer._save` writes state dicts with `hf_model.*` prefix.
  HF Trainer's `_load_best_model` loads into the **inner** `hf_model` module,
  not the wrapper, so it silently misses every prefixed key and reinits weights.
  The final `pytorch_model.bin` at run root therefore has **untrained class
  heads**; the true best weights live in `<run_dir>/checkpoint-N/pytorch_model.bin`
  where `N` is the step pointed to by `trainer_state.json::best_model_checkpoint`.
  HF Trainer's OWN test eval (run in-memory after the broken save) still sees
  the correct weights — so `test_results.json` is trustworthy even when
  `pytorch_model.bin` is not. Any analyzer-rerun script must load from the
  best-checkpoint folder, not the root.
- **`self.save_dir` is an instance attribute** (set inside `_build_callbacks`)
  so `_finalize_training` and `_build_pytorch_training_config` can read it
  after the main loop. Do not convert it back to a local variable — it's the
  only link between the callback setup phase and post-train finalization.

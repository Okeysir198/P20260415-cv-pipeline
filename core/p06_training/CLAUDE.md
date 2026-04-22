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
| `callbacks_viz.py` | `NormalizedInputPreviewCallback` — fires once on train-start, denormalizes one collated batch, renders `data_preview/normalized_input_preview.png`. Dual-backend (permissive `_AnyHook` base satisfies both pytorch `CallbackRunner` and HF `CallbackHandler` hook surfaces). |
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
├── data_preview/               (on_train_start, ~2 s total)
│   ├── dataset_stats.{png,json}
│   ├── data_labels_{train,val,test}.png
│   ├── aug_labels_train[_mosaic].png
│   └── normalized_input_preview.png  ← stage-3 sanity: denormalize(batch) + GT
├── val_predictions/
│   ├── epochs/epoch_NNN.png    (per-epoch, ~2 s each — the only mid-run hook)
│   ├── best.png                (on_train_end, best-checkpoint weights)
│   └── error_analysis/         (task-dispatched; ~10 s total)
│       ├── summary.{json,md}       3-axis: data_distribution + training_config + model_metrics
│       ├── data_distribution.png   class count + per-class × size-tier
│       ├── boxes_per_image.png     crowdedness (mean/median/p95/max)
│       ├── bbox_aspect_ratio.png   per-class log-scale w/h
│       ├── per_class_pr_f1.png     P / R / F1 bars
│       ├── confusion_matrix.png    GT×Pred (last col/row = background)
│       ├── confidence_calibration.png  TP vs FP score histogram
│       ├── size_recall.png         small / medium / large with explicit COCO px² thresholds
│       ├── pr_curves.png           per-class PR curve + AP in legend
│       ├── f1_vs_threshold.png     per-class F1 sweep + best-F1 threshold marker
│       ├── map_vs_iou.png          mAP at IoU 0.50 → 0.95 (AP50 / AP75 / AP[.5:.95])
│       ├── hardest_images.png      top-12 overview
│       └── hard_images/            per-error-type × per-class GT-vs-Pred galleries
│           ├── false_positives/<class>/<stem>__fp_score_0.87.png
│           ├── false_negatives/<class>/<stem>__fn.png
│           └── class_confusion/<pred>__from__<gt>/<stem>__iou_0.62.png
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
  (`_draw_gt_boxes`, `_save_image_grid`, `annotate_gt_pred`) with the
  pytorch-backend loggers but consume HF's documented kwargs (`model`,
  `eval_dataloader`, `state.log_history`) directly — no proxy-trainer
  attribute surface. Earlier bridge-adapter design dropped as of the
  native-callback migration.
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
- **`self.save_dir` is an instance attribute** (set inside `_build_callbacks`)
  so `_finalize_training` and `_build_pytorch_training_config` can read it
  after the main loop. Do not convert it back to a local variable — it's the
  only link between the callback setup phase and post-train finalization.

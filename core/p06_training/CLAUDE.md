# CLAUDE.md — core/p06_training/

Authoritative notes on the training loop(s), callbacks, and the choice between
the pytorch and HF Trainer backends. Companion to `README.md` — this file
covers what's *between* files and the gotchas learned the hard way.

## Read First — Critical Pitfalls

- **HF DETR-family val OOM**: set `eval_accumulation_steps=4` for val sets ≥1000 imgs — see [Gotchas: `eval_accumulation_steps=4` is mandatory](#gotchas) (line ~322).
- **One switch for rescale+normalize+input_size**: the `tensor_prep` block in `06_training.yaml` is authoritative — see [Gotchas: `tensor_prep` is the single switch](#gotchas) (line ~243).
- **Early `set_seed` before `build_model`**: required for D-FINE/RT-DETRv2 head-reinit reproducibility (prevents 0.15-mAP stall) — see [Determinism: Early `set_seed` hook](#determinism) (line ~180).

## Three backends, one config

`training.backend` in `06_training.yaml` picks the execution path:

| Backend | File | Venv | Use when |
|---|---|---|---|
| `pytorch` (default) | `trainer.py::DetectionTrainer` | main `.venv/` | You need EMA + per-component LR + the full custom callback suite (`DatasetStatsLogger`, `AugLabelGridLogger`, `DataLabelGridLogger`, `ValPredictionLogger`, `CheckpointSaver`, `EarlyStopping`, `WandBLogger`). Covers every task type (detection, classification, segmentation, pose, keypoint). |
| `hf` | `hf_trainer.py::train_with_hf` | main `.venv/` | Detection / classification / segmentation when you want DDP / DeepSpeed / gradient accumulation / bf16 tensor-core paths "for free", HF-Trainer-standard output layout (`checkpoint-N/`, `runs/<ts>_<host>/` TB, `trainer_state.json`), and the reference-notebook code pattern for DETR-family fine-tuning. |
| `paddle` | NOT in this dispatcher — see `core/p06_paddle/CLAUDE.md` | `.venv-paddle/` | Selecting `backend: paddle` here prints a redirect to `.venv-paddle/bin/python core/p06_paddle/train.py`. Paddle drives upstream `ppdet.engine.Trainer` directly; convergence with the pipeline happens at ONNX (`core/p06_paddle/export.py`). v1 = detection only (PicoDet, PP-YOLOE). |

Both respect the same YAML config keys; the HF backend falls back or warns
on features it doesn't implement yet. See `_validate_hf_backend_config`
(`hf_trainer.py`) for the allow/deny list.

### HF backend support matrix (all CV tasks)

`hf_keypoint` (top-down ViTPose-family) requires `KeypointTopDownDataset`,
NOT the bottom-up `KeypointDataset`. The two emit different sample shapes —
top-down returns `{pixel_values, target_heatmap, target_weight}` per-person
crops; bottom-up returns `{boxes, keypoints}` full-frame. The HF Trainer
dispatcher (`_build_datasets`) and post-train callbacks branch on this.


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
| `training.gpu_augment` | ✗ | HF Trainer uses its own DataLoader; warning emitted. Both `augmentation.library` options run on CPU under HF — torchvision v2 is the recommended default (parity with albumentations after the 2026-04-20 resize-first reorder; see fire RT-DETR config notes) and additionally supports Mosaic / MixUp / CopyPaste / IRSimulation |
| `training.val_full_interval` | partial | HF evaluates every epoch by default; this knob is effectively ignored |
| `data.subset.{train,val}` | ✓ | wraps in `torch.utils.data.Subset` with deterministic seed |
| `augmentation.library: albumentations` | ✓ | fast CPU aug backend with probability-gated transforms |
| `checkpoint.metric: val/mAP50` | ✓ | auto-translated to HF's `eval_map_50` |
| `checkpoint.save_best: true` | ✓ | uses HF's `load_best_model_at_end` |
| Viz callbacks | ✓ native, **all CV tasks** | `hf_callbacks.py` — first-class `TrainerCallback` subclasses. `_build_callbacks` allows detection / classification / segmentation / keypoint via `task_from_output_format`. `HFDataLabelGridCallback` + `HFAugLabelGridCallback` dispatch via `core.p06_training._common.build_dataset_for_viz` and route GT overlays through `_render_gt_panel` (per-task primitive: bbox / mask / banner / keypoints). `03_aug_labels_train.png` runs for every task via `_build_task_transforms` (detection→`build_transforms`, cls→`build_classification_transforms`, seg→`build_segmentation_transforms`, kpt→`build_keypoint_transforms`). `04_transform_pipeline.png` dispatches to `render_transform_pipeline` for detection (full per-step walker) and `render_transform_pipeline_task` for cls/seg/kpt (2-row raw↔denorm grid — the paired-box walker assumes YOLO targets). The 04 chart includes a `Denormalize(Normalize)` sanity-check column so a separate flat normalized-input grid is unnecessary (was previously emitted as `05_normalized_input_preview.png` — removed). |
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
| `post_train.py` | Backend-agnostic post-train runner. `run_post_train_artifacts(model, save_dir, val_dataset, test_dataset, task, class_names, input_size, style, training_config, …)` renders best-checkpoint val+test grids and dispatches to `error_analysis_runner`. `render_prediction_grid` is the grid renderer for per-epoch + best-checkpoint previews (routes via `annotate_gt_pred`). The hardest-images overview (`08_hardest_images.png`) is rendered by `_plot_hardest_images_grid` in `error_analysis_runner.py`. |
| `_common.py` | Shared helpers: `unwrap_subset`, `task_from_output_format`. Re-exports `yolo_targets_to_xyxy` from `utils/metrics.py` for back-compat. Dedupes logic that previously had 3 copies across HF + pytorch backends. |
| `losses.py` | `DetectionLoss` ABC, `YOLOXLoss` (SimOTA), `FocalLoss`, `IoULoss`, `_DETRPassthroughLoss`, registry + `build_loss()`. |
| `lr_scheduler.py` | `WarmupScheduler`, `CosineScheduler`, `PlateauScheduler`, `StepScheduler`, `OneCycleScheduler` + `build_scheduler()`. |
| `postprocess.py` | `POSTPROCESSOR_REGISTRY`, YOLOX-only decoding (HF models use built-in `post_process_object_detection`). |
| `metrics_registry.py` | `METRICS_REGISTRY`, `register_metrics()`, per-format validation metrics dispatch (pytorch backend only). |

## Post-train observability (on `on_train_end`, both backends)

Every training run produces a uniform per-run artifact tree — no per-config opt-in. Driven by `post_train.run_post_train_artifacts` + `core/p08_evaluation/error_analysis_runner.run_error_analysis`.

> **Detection threshold-sensitive charts use per-class F1-optimal cutoffs** — `run_post_train_artifacts` calls the analyzer with `threshold_policy="f1_optimal_per_class"` for detection. `run_error_analysis` collects raw predictions once at an eps floor (1e-3) for the threshold-AGNOSTIC charts (07 per-class AP, 09 calibration, 14 robustness, 15 threshold-analysis, 16–21 deep dives), then derives per-class thresholds from the same sweep chart 15 plots and applies them to the threshold-SENSITIVE charts (08 confusion, 10 failure_mode_contribution, 11 failure_by_attribute, 12 hardest, 13 mode-examples gallery). The dict lands in `summary.json::operating_point.thresholds` so anyone reading the report knows which charts reflect deploy cutoffs. Override per-call with `threshold_policy ∈ {fixed, f1_optimal, f1_optimal_per_class}`. The legacy per-arch scalar (`error_analysis_conf_threshold`) is now used **only for cls/seg/kpt** (still 0.05 for DETR-family / 0.25 for YOLOX). Train-split error analysis is hard-forced to `fixed` — picking thresholds from train would be overfit. Helper lives in `core/p08_evaluation/threshold_policy.py`.

```
runs/<ts>/
├── data_preview/               (on_train_start, ~2 s total — task-aware for det/cls/seg/kpt)
│   ├── 00_dataset_info.{md,json}                provenance: feature, dataset, classes, splits, input_size, aug
│   ├── 01_dataset_stats.{png,json}              task-aware: detection→bbox tiers + boxes-per-image;
│   │                                              cls→class hist + resolution + per-channel mean/std;
│   │                                              seg→pixel-class hist + mask coverage + components;
│   │                                              kpt→per-joint visibility + spatial heatmap + edge lengths
│   ├── 02_data_labels_{train,val,test}.png      raw images with GT (boxes/masks/banners/keypoints by task)
│   ├── 03_aug_labels_train.png                  CPU augmentation output
│   └── 04_transform_pipeline.png                step-by-step transform walk; last col = Denorm(Norm) sanity check
├── val_predictions/
│   ├── epochs/epoch_NNN.png    (per-epoch, ~2 s each — the only mid-run hook)
│   ├── best.png                (on_train_end, best-checkpoint weights)
│   └── error_analysis/         flat 01..20 layout, all diagnostics at depth 0 — both backends, all tasks
│       ├── summary.{json,md}       3-axis: data_distribution + training_config + model_metrics
│       │                           summary.md auto-iterates 01→20 with description + signal +
│       │                             suggested-next-step driven by `chart_annotations.py::CHART_META`
│       │   (chart PNGs carry a numeric `NN_` prefix; authoritative name map is `CHART_FILENAMES`
│       │    in `core/p08_evaluation/error_analysis_runner.py` — do not hardcode filenames)
│       ├── 01_overview.png                       headline metric + per-mode Δ ranked bars
│       ├── 02_data_distribution.png              val class/sample balance
│       ├── 03_distribution_mismatch.{png,json}   train↔val/test drift (class %, JS div, image-stats KS)
│       ├── 04_label_quality.{png,json}           per-class confident-disagreement rate
│       ├── 04_label_quality_gallery.png          top-N suspected mislabels GT|Pred
│       ├── 04_suspected_mislabels.csv            Label Studio re-import format
│       ├── 05_duplicates_leakage.{png,json}      pHash near-dupes within / across splits (loader-based enumeration)
│       ├── 06_learning_ability.{png,json}        train-vs-val regime + learning curves (det reuses main mAP evaluator)
│       ├── 07_per_class_performance.png          P/R/F1 (det/cls), IoU (seg), PCK (kpt)
│       ├── 08_confusion_matrix.png   OR  08_top_confused_pairs.png   (det/cls/seg; ≤20 classes vs >20)
│       ├── 09_confidence_calibration.png         TP vs FP histogram (det/cls)
│       │   OR 09_confidence_vs_error.png         heatmap-peak vs pixel error (kpt variant)
│       ├── 10_failure_mode_contribution.png      global Δ + per-class × mode heatmap
│       ├── 11_failure_by_attribute.png           task-aware attrs (size/aspect/crowdedness for det; resolution/brightness for cls; etc.)
│       ├── 12_hardest_images.png                 top-12 worst GT|Pred grid
│       ├── 13_failure_mode_examples/             per-task galleries; subfolder taxonomy:
│       │   │  detection: missed/, localization/, class_confusion/, duplicate/, background_fp/   (each /<class>/)
│       │   │  classification: misclassified/<gt>__as__<pred>/, low_confidence_correct/<cls>/, high_confidence_wrong/<gt>__as__<pred>/
│       │   │  segmentation: low_iou/<cls>/, missed/<cls>/, false_positive/<cls>/, boundary_error/<cls>/
│       │   │  keypoint: high_error/kp_<k>_<name>/, ghost/kp_<k>_<name>/, swapped_pair/<L>__<R>/
│       ├── 14_robustness_sweep.{png,json}        metric vs corruption — det: blur/jpeg/brightness/rotation;
│       │                                          kpt: blur/brightness/jpeg (no rotation — heatmap geometry breaks); 3 severities
│       ├── 15_threshold_analysis.png             (detection) 2×2: PR + F1/P/R vs conf threshold; F1-optimal marked
│       ├── 16_recoverable_map_vs_iou.png         (detection) per-mode Δ mAP across IoU 0.5→0.9
│       ├── 17_confidence_attribution.png         (detection) FN causality: true_miss / under_conf / loc_fail
│       ├── 18_boxes_per_image.png                (detection) crowdedness
│       ├── 19_bbox_aspect_ratio.png              (detection) per-class log-scale w/h
│       ├── 20_size_recall.png                    (detection) recall by COCO size bands
│       └── 21_pixel_confusion_matrix.png         (segmentation) row-normalised C×C pixel cross-tab
│           OR 21_bbox_padding_sweep.png          (keypoint top-down) AP/PCK vs bbox_padding ∈ {1.0..2.0}
├── test_predictions/           same flat 01..21 layout as val_predictions/error_analysis/
└── test_results.json           HF Trainer metrics on the test split
```

No more sibling `distribution_mismatch/`, `learning_ability/`, or `label_quality/` folders at run root. The `DM_`/`LA_`/`LQ_` filename prefixes are retired — every diagnostic uses the flat `NN_` numeric prefix. Reading order matches debugging order: data → labels → splits → capacity → per-class → confusion → calibration → failure decomposition → slices → instances → galleries → robustness → task-specific deep dives.

**Interpretation layer (Phase 3)** — every chart in `summary.md` is enriched with a description, a current-signal snapshot, and a rule-driven next-step suggestion. Lookup table is `core/p08_evaluation/chart_annotations.py::CHART_META` keyed by filename stem (e.g. `"05_confidence_calibration"`). Each entry has `title`, `description` (< 80 words plain English), an optional `signal_template`, and `next_step_rules` (each a `(when_metric_dict→bool, advice_template)` pair). The first matching rule's advice wins; if none fires the default is `"No action — signal is within acceptable band."`. Analyzers populate the per-chart `metrics` dict and pass it to `_write_json_md(..., chart_metrics=...)` — that drives both the signal snapshot and the rule selection.

**Adding a new chart**: (1) add entry in `CHART_FILENAMES` (numbered prefix, never hardcode the filename downstream), (2) write the chart in the relevant `_analyze_*` function, (3) add a `ChartMeta` entry in `CHART_META` describing it + 1–3 `Rule`s referencing already-computed metrics, (4) populate the corresponding `chart_metrics["<stem>"]` dict so signals + rules fire.

`VizStyle` (core/p10_inference/supervision_bridge.py) is the single source of truth for colors/thickness/text — no per-site drawing constants. `training_config` in `summary.json` snapshots arch / params / lr / optimizer / scheduler / bf16 / best-metric / test-metrics from both backends.

Opt out per block in YAML (all default true):
```yaml
training:
  data_viz:  { enabled: false }
  aug_viz:   { enabled: false }
  val_viz:   { enabled: false }    # still leaves best_viz + error_analysis on
  best_viz:  { enabled: false }
  error_analysis: { enabled: false }
  post_train: { enabled: false }   # pytorch-backend only: skip best-reload + test eval
```

> **Backend asymmetry**: pytorch-backend post_train (best-reload + test eval) is opt-out via `training.post_train: { enabled: false }`; HF backend auto-enables via `load_best_model_at_end` + auto-test eval (no separate knob).

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
    Is the task detection / classification / segmentation / keypoint (top-down)?
    ├── Yes → backend: hf   (DDP/DeepSpeed/bf16 available, HF Trainer output layout,
    │                        full data_preview/val_predictions/error_analysis tree
    │                        for every supported task — keypoint goes through the
    │                        `hf_keypoint` arch + `KeypointTopDownDataset`)
    └── No  → backend: pytorch (pose/face — top-down kpt now works on HF too)
```

## Adding a new backend

The dispatch happens in `train.py::main` keyed on `training.backend`. To add a fourth backend:

1. Write `core/p06_training/<backend>_trainer.py` exposing a `train_with_<backend>(config, config_path)` entry point — same signature as `train_with_hf`. Reuse `_common.py` helpers (`task_from_output_format`, `unwrap_subset`).
2. Add the dispatch branch in `train.py::main`. Keep heavy framework imports lazy (inside the function, not at module top) so the main venv doesn't need the framework installed to import `train.py`.
3. Reuse `post_train.run_post_train_artifacts` so the `data_preview/` + `val_predictions/error_analysis/` tree comes out identical. The runner is backend-agnostic — it takes a `model` with a `__call__(image)` interface and a dataset.
4. Document in this file's backend table (file path, venv, when-to-use) and in the root `CLAUDE.md` "Training Backends" bullet list.

**Paddle is not a backend in this dispatcher** — the redesign moved paddle into its own package (`core/p06_paddle/`), running in `.venv-paddle/` and driving upstream `ppdet.engine.Trainer` directly. The dispatcher in `train.py` prints a redirect when `backend: paddle` is selected. Convergence with the unified pipeline happens at ONNX, not at `nn.Module`. See `core/p06_paddle/__init__.py` for the entry contract.

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

- **`tensor_prep` is the single switch for rescale+normalize+input_size** (added 2026-04-23). The `tensor_prep:` block in `06_training.yaml` is authoritative: `input_size`, `rescale`, `normalize`, `mean`, `std`, `applied_by` (`hf_processor` | `v2_pipeline`). `build_hf_model` FORCES the HF processor's `do_rescale`/`do_normalize`/`image_mean`/`image_std`/`do_resize`/`size` to match — no more checkpoint-default leakage. `build_transforms(..., tensor_prep=...)` in `core/p05_data/transforms.py` appends `v2.Normalize` only when `applied_by == "v2_pipeline"`; skips it on `hf_processor`. `_validate_tensor_prep` in `utils/config.py` hard-errors on backend mismatch, double-normalize, missing-normalize, or missing mean/std; called from both `DetectionTrainer.__init__` (pytorch backend) and `train_with_hf` (HF backend) right after the model builds so the processor is observable. Legacy configs (no `tensor_prep`) auto-migrate on load via `_migrate_legacy_tensor_prep` with a one-line WARNING — add an explicit block to suppress it. CPPE-5 (`notebooks/detr_finetune_reference/our_rtdetr_v2_torchvision/06_training.yaml`) is the smoke-test target; other feature configs continue working unchanged via the shim.

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
- **Detection error-analysis: per-class F1-optimal threshold policy** (replaces the per-arch scalar auto-pick).
  `run_error_analysis(..., threshold_policy="f1_optimal_per_class")` is the new default for detection.
  Pass 1 collects raw predictions once at the eps floor (`core.p08_evaluation.threshold_policy.EPS_FLOOR = 1e-3`)
  → drives the agnostic charts (07 / 09 / 14 / 15 / 16 / 17 / 18 / 19 / 20).
  Pass 2 applies the per-class dict computed from the same sweep that chart 15 plots
  (3-point smoothed argmax over the F1 curve) → drives the sensitive charts (08 / 10 / 11 / 12 / 13).
  The dict + the list of charts it gates lands in `summary.json::operating_point` and the head
  of `summary.md`. Set `threshold_policy="fixed"` for deterministic CI / back-compat — the old
  scalar `conf_threshold` then applies to all charts. Set `"f1_optimal"` for a single global
  cutoff. `split="train"` forces `"fixed"` (avoid overfitting). The per-arch scalar
  (`error_analysis_conf_threshold`: 0.05 DETR-family / 0.25 YOLOX) survives in `post_train.py`
  but only flows into cls/seg/kpt analyzers; detection ignores it. Override via
  `training.post_train.error_conf_threshold` for cls/seg/kpt only.
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
- **HF `load_best_model_at_end` works correctly** (verified 2026-05-04 via
  md5sum: root `pytorch_model.bin` is bit-identical to
  `checkpoint-<best_step>/pytorch_model.bin`). The earlier note in this file
  claiming the in-process `_load_best_model` silently fails was based on
  incomplete diagnosis — the wrapper-prefix path actually round-trips
  correctly through `self.model.load_state_dict(state)` because both the
  wrapper module and the saved state dict use the same `hf_model.*` prefix.
  Root `pytorch_model.bin` and `test_results.json` are both trustworthy.
  Downstream reload sites (`core/p08_evaluation/evaluate.py`,
  `core/p09_export/export.py`, `core/p10_inference/predictor.py`) still use
  `utils.checkpoint.strip_hf_prefix` defensively — keep that, it makes
  cross-format checkpoints robust.
- **`EarlyStoppingCallback` warning during final test eval** — HF's callback
  fires `on_evaluate` regardless of `metric_key_prefix`, so during the final
  test eval (prefix `test_*`) it looks for `eval_map_50` and emits
  `"early stopping required metric_for_best_model, but did not find
  eval_map_50 so early stopping is disabled"`. Benign (training is over).
  `train_with_hf` strips `EarlyStoppingCallback` from the callback handler
  before invoking `trainer.evaluate(test_dataset, ...)` to suppress the noise
  (2026-05-04). Do not re-add it after — test eval is the last action.
- **`eval_accumulation_steps=4` is mandatory for DETR-family with val ≥ 1000**
  (verified 2026-05-04 on R18 fire OOM). Without it HF Trainer accumulates
  ALL eval batches' `ModelOutput`s on GPU before transferring to CPU. RT-DETRv2
  / D-FINE return logits + pred_boxes + intermediate decoder layer outputs +
  encoder feature maps + auxiliary outputs per batch — at val=2606 imgs /
  eval_batch_size=4 = 651 batches, that's several GB of eval-time growth on
  top of train state. End-of-epoch eval then OOMs even though train phase
  fits. `_config_to_training_args` sets this unconditionally; do not unset.
  Speed cost of more frequent CPU flushes is negligible (<0.5s per eval).
- **`_DetectionTrainer.evaluate()` runs `gc.collect() + torch.cuda.empty_cache()`
  before super.evaluate()** — partial defense against fragmentation when
  pre-existing GPU tenants take ≥10 GB. ~200ms overhead per eval cycle. Useful
  but NOT a fix on its own — `eval_accumulation_steps` above is the structural
  fix; the cache reset only helps when fragmentation (not actual usage) is
  the bottleneck.
- **CPU-RAM corollary to `eval_accumulation_steps=4`** (added 2026-05-05).
  The `eval_accumulation_steps` knob flushes GPU→CPU but doesn't shrink what's
  on CPU. With HF returning the full `ModelOutput` per batch, CPU RSS grew
  ~+30 GB per epoch on RT-DETR @ 960² (encoder hidden states + decoder layers
  + 6 aux outputs accumulating until `compute_metrics` returns and Python's
  allocator releases the pages, which it often defers). The structural fix
  is in `core/p06_models/hf_model.py::HFDetectionModel.forward` — eval-mode
  strip of unused `ModelOutput` fields (encoder_last_hidden_state,
  intermediate_*, decoder_*, auxiliary_outputs, etc). Pair fix:
  `HFValPredictionCallback.on_epoch_end` ends with `gc.collect() +
  torch.cuda.empty_cache()` to release the per-epoch viz forward tensors.
  Both edits are load-bearing for any DETR-family run > 5 epochs at 960²;
  without them, CPU RAM OOMs by epoch 3 even on a 128 GB box. See
  `core/p06_models/CLAUDE.md` for the wrapper-side invariant.
- **Paddle: separate package, not a backend in this dispatcher.** Train via `.venv-paddle/bin/python core/p06_paddle/train.py`; the unified dispatcher just prints a redirect when `backend: paddle` is selected. Paddle ships its own bundled CUDA incompatible with the main venv's CUDA 13 PyTorch wheels — sibling venv keeps them apart. Bootstrap: `bash scripts/setup-paddle-venv.sh`.
- **Paddle ↔ pipeline convergence is ONNX.** `core/p06_paddle/export.py` writes `model.onnx`; from there the standard main-venv path handles eval, error analysis, inference, demo. No torch ↔ paddle tensor bridge. Don't try to `torch.load` a `.pdparams` file.
- **Paddle ORT INT8** — PicoDet / PP-YOLOE (CNN) generally hit real INT8 speedup on ORT CUDA EP. For any future transformer-y paddle arch, reuse the DETR-family exclusion list in `core/p09_export/quantize.py` (`LayerNormalization`/`Softmax`/`Gather` opt-out + `percentile` calibration default).
- **`self.save_dir` is an instance attribute** (set inside `_build_callbacks`)
  so `_finalize_training` and `_build_pytorch_training_config` can read it
  after the main loop. Do not convert it back to a local variable — it's the
  only link between the callback setup phase and post-train finalization.

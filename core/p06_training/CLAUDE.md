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
| `training.bf16` | ✓ | set `True` for detection (DETR decoder overflows in fp16) |
| `training.amp` | ✓ | validator **hard-errors** if True for detection |
| `training.patience` | ✓ | `EarlyStoppingCallback` |
| `training.ema` | ✓ | native `EMACallback` wrapping our `ModelEMA` — swaps weights in/out around each eval |
| `training.gpu_augment` | ✗ | HF Trainer uses its own DataLoader; warning emitted. Use `augmentation.library: albumentations` for CPU-aug speed parity (~2× faster than torchvision v2) |
| `training.val_full_interval` | partial | HF evaluates every epoch by default; this knob is effectively ignored |
| `data.subset.{train,val}` | ✓ | wraps in `torch.utils.data.Subset` with deterministic seed |
| `augmentation.library: albumentations` | ✓ | fast CPU aug backend with probability-gated transforms |
| `checkpoint.metric: val/mAP50` | ✓ | auto-translated to HF's `eval_map_50` |
| `checkpoint.save_best: true` | ✓ | uses HF's `load_best_model_at_end` |
| Viz callbacks | ✓ via bridge | `_HFVizBridge` wraps our four loggers so `data_preview/` + `val_predictions/` both work |
| Final test-set eval | ✓ auto | writes `<output_dir>/test_results.json` when a test split is present |

If a task / config combo isn't supported, `_validate_hf_backend_config` fails
fast at the top of `train_with_hf` rather than silently degrading.

## Files

| File | Purpose |
|---|---|
| `trainer.py` | `DetectionTrainer` — main training loop (pytorch backend). Auto-detects HF vs YOLOX model path, per-component LR groups, EMA, gradient clipping, callback dispatch. |
| `hf_trainer.py` | `train_with_hf`, `_DetectionTrainer` (Trainer subclass with shared-weight-safe `_save`), `EMACallback`, `_HFVizBridge` (runs our callbacks inside HF), detection collator + real mAP `compute_metrics` (torchmetrics-based). Config validator enforces hard incompatibilities up-front. |
| `train.py` | CLI entry point — `auto_select_gpu`, determinism knobs (CUBLAS env var + `torch.use_deterministic_algorithms(True, warn_only=True)`), 3-warning filter for known-harmless PyTorch messages, dispatches to backend. |
| `callbacks.py` | `Callback` base class (pytorch backend only), `CheckpointSaver`, `EarlyStopping`, `WandBLogger`, `ValPredictionLogger`, `DatasetStatsLogger`, `DataLabelGridLogger`, `AugLabelGridLogger`, `CallbackRunner`. Also `_run_splits_and_subsets(trainer)` — now iterates train/val/test so the HF bridge's stub test-loader shows up in data_preview. |
| `losses.py` | `DetectionLoss` ABC, `YOLOXLoss` (SimOTA), `FocalLoss`, `IoULoss`, `_DETRPassthroughLoss`, registry + `build_loss()`. |
| `lr_scheduler.py` | `WarmupScheduler`, `CosineScheduler`, `PlateauScheduler`, `StepScheduler`, `OneCycleScheduler` + `build_scheduler()`. |
| `postprocess.py` | `POSTPROCESSOR_REGISTRY`, YOLOX-only decoding (HF models use built-in `post_process_object_detection`). |
| `metrics_registry.py` | `METRICS_REGISTRY`, `register_metrics()`, per-format validation metrics dispatch (pytorch backend only). |

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
- **Viz callbacks on HF backend are via an attribute-proxy adapter**,
  not native `TrainerCallback` subclasses. Risk: future HF Trainer API
  changes might break the proxy surface (`trainer.model`, `trainer.device`,
  `trainer.train_loader`, `trainer.val_loader`, `trainer._model_cfg`,
  `trainer._loaded_data_cfg`, `trainer._decode_predictions`,
  `trainer.callback_runner.get_callback`). If that happens, rewrite each
  viz callback as a native `TrainerCallback` subclass.
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

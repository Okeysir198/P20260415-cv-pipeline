# CLAUDE.md — configs/

YAML-driven configuration for the entire pipeline. **No hardcoded values in code** — everything comes from these files.

## Layout

This directory holds only cross-feature infra — **not** authoritative for any feature:

- `configs/_shared/` — pipeline templates (01–04, 07/08-HPO, 09); non-authoritative
- `configs/_test/` — CI smoke-test configs

Feature configs are authoritative under each feature:

```
features/<name>/configs/
  00_data_preparation.yaml   (when applicable)
  05_data.yaml
  06_training.yaml
  08_evaluation.yaml         (optional)
  09_export.yaml             (optional)
  10_inference.yaml          carries `alerts:`, `tracker:`, `samples:`
  03_generative_augment.yaml (feature-specific when used)
```

Folder names follow a **`<category>-<name>`** convention aligned with
`docs/03_platform/` (e.g. `safety-fire_detection`, `ppe-helmet_detection`,
`access-face_recognition`). See `features/README.md` for the current list.

## Numbering Convention

| # | Step | Location |
|---|------|----------|
| 00 | Data preparation (source datasets, class mapping, splits, output dir) | per-usecase |
| 01 | Auto-annotate (mode, NMS, output format) | `_shared/` |
| 02 | Annotation QA (sampling, scoring, SAM3 thresholds) | `_shared/` |
| 03 | Generative augment (SAM3 + diffusion settings) | `_shared/` or per-usecase override |
| 04 | Label Studio (URL, API key, import/export) | `_shared/` |
| 05 | Data definition (paths, classes, input_size, normalization) | per-usecase |
| 06 | Training (model arch, hyperparams, augmentation, loss, logging) | per-usecase |
| 07 | HPO (Optuna search space, pruner, sampler) | `_shared/` (file: `08_hyperparameter_tuning.yaml`) |
| 08 | Evaluation & Error Analysis (CLI-only, no config YAML) | -- |
| 09 | Export (ONNX opset, optimization, quantization) | `_shared/` |
| 10 | Inference (CLI-only, no config YAML; face/pose use feature YAMLs) | -- |

## Key Rules

1. **No inheritance/merging** — each config file is complete and self-contained. Per-usecase overrides of `_shared/` configs are full copies, not partial diffs.
2. **Paths are relative from project root** — dataset paths use `../../dataset_store/` (two levels up from config file location). Pretrained weights use `../../pretrained/`.
3. **`data.dataset_config: 05_data.yaml`** — training configs reference their sibling data config by filename only (resolved relative to the training config's directory).
4. **CLI override** — any value can be overridden: `--override training.lr=0.005 training.epochs=100`.
5. **No `${var}` interpolation between files** — `utils/config.py` supports variable interpolation within a single file only.

## 00_data_preparation.yaml Schema

Used by `core/p00_data_prep/run.py` to merge raw datasets into training-ready YOLO format.

```yaml
task: detection                          # detection (classification coming)
dataset_name: "fire_detection"
output_dir: "../../dataset_store/fire_detection"
output_format: "yolo"
classes:                                  # canonical target class names
  - fire
  - smoke
sources:                                  # list of raw datasets to merge
  - name: "roboflow_fire"
    path: "../../dataset_store/raw/fire_detection/roboflow_fire"
    format: "yolo"                        # yolo | coco
    has_splits: true                      # true = train/valid/test subdirs already exist
    splits_to_use: ["train", "valid", "test"]
    class_map:                            # source class name → canonical name
      "fire": "fire"
      "smoke": "smoke"
splits:                                   # output split ratios
  train: 0.8
  val: 0.1
  test: 0.1
  seed: 42
options:
  copy_images: true
  handle_duplicates: "rename"             # rename | skip | overwrite
  validate_labels: true
```

## 05_data.yaml Schema

```yaml
dataset_name: "fire_detection"
path: "../../dataset_store/fire_detection"     # relative from project root
train: "train/images"
val: "val/images"
test: "test/images"
names: {0: fire, 1: smoke}               # class_id → class_name
num_classes: 2
input_size: [640, 640]                    # [H, W]
mean: [0.485, 0.456, 0.406]              # RGB normalization (ImageNet default)
std: [0.229, 0.224, 0.225]
```

**Segmentation data layout**: Segmentation datasets use `images/` + `masks/` subdirectories (instead of `images/` + `labels/`). Masks are grayscale PNGs where pixel values are class IDs (0 to num_classes-1). The `train` and `val` keys point to split directories containing both subdirs:

```
dataset_store/my_segmentation/
  train/
    images/   ← RGB images
    masks/    ← grayscale PNGs (pixel value = class ID)
  val/
    images/
    masks/
```

## 06_training.yaml Schema

```yaml
model:
  arch: yolox-m                           # registry key (yolox-m, dfine-s, rtdetrv2-r18, timm, hf-detection, hf-classification)
  pretrained: ../../pretrained/yolox_m.pth # or true (for timm/HF auto-download) or null
  num_classes: 2
  input_size: [640, 640]
  depth: 0.67                             # YOLOX-specific
  width: 0.75                             # YOLOX-specific
  # timm_name: mobilenetv3_small_100      # timm-specific
  # hf_model_id: hf-internal-testing/...  # HF-specific

data:
  dataset_config: 05_data.yaml
  batch_size: 16
  num_workers: 4
  pin_memory: true

augmentation:                             # detection only (classification uses timm transforms)
  mosaic: true
  mixup: true
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  fliplr: 0.5
  scale: [0.1, 2.0]
  degrees: 10.0
  translate: 0.1
  shear: 2.0

training:
  backend: pytorch                        # pytorch (default), hf (HF Trainer + DDP/DeepSpeed), custom (dynamic import via training.custom_trainer_class)
  epochs: 200
  optimizer: sgd                          # sgd, adam, adamw
  lr: 0.01
  lr_backbone: 0.001                      # backbone LR after unfreeze (set ~10x lower than lr; only used when freeze_backbone_epochs > 0)
  freeze_backbone_epochs: 5               # train head/neck only for N epochs, then unfreeze all layers (transfer learning warm-up)
  weight_decay: 0.0005
  warmup_epochs: 5
  scheduler: cosine                       # cosine, plateau, step, onecycle
  patience: 50                            # early stopping
  amp: true
  grad_clip: 35.0
  gradient_accumulation_steps: 1          # >1 to simulate larger batch sizes
  nms_threshold: 0.45                     # NMS IoU threshold for validation
  ema: true
  ema_decay: 0.9998
  # Step scheduler options (when scheduler: step):
  # step_size: 30                         # epochs between LR reductions
  # gamma: 0.1                            # multiplicative LR decay factor
  # OneCycle options (when scheduler: onecycle):
  # pct_start: 0.3                        # fraction of cycle increasing LR

loss:
  type: yolox                             # yolox, detr-passthrough (HF models use built-in loss)

logging:
  wandb_project: smart-camera
  run_name: fire_yoloxm_v1

checkpoint:
  save_best: true
  metric: val/mAP50                       # or val/accuracy for classification
  mode: max
  save_interval: 10

seed: 42
```

## Adding a New Feature

```bash
bash scripts/new_feature.sh my_new_feature
# edit features/my_new_feature/configs/{05_data,06_training,10_inference}.yaml
```

Copies `features/_TEMPLATE/` and substitutes the name. No need to touch
`core/` or `app_demo/tabs/` — everything resolves from configs.

## CLI Overrides

Any config value can be overridden at runtime using `--override key=value`:

```bash
# Override single hyperparameter
uv run core/p06_training/train.py \
  --config features/safety-fire_detection/configs/06_training.yaml \
  --override training.lr=0.005

# Override multiple values
uv run core/p06_training/train.py \
  --config features/safety-fire_detection/configs/06_training.yaml \
  --override training.lr=0.005 training.epochs=100 training.batch_size=32

# Override nested values (use dot notation)
uv run core/p06_training/train.py \
  --config features/safety-fire_detection/configs/06_training.yaml \
  --override data.num_workers=8 augmentation.mosaic=false

# Override model architecture
uv run core/p06_training/train.py \
  --config features/safety-fire_detection/configs/06_training.yaml \
  --override model.arch=dfine-s model.pretrained=true

# Override logging
uv run core/p06_training/train.py \
  --config features/safety-fire_detection/configs/06_training.yaml \
  --override logging.wandb_project=my_project logging.run_name=experiment_v2

# Override evaluation settings
uv run core/p08_evaluation/evaluate.py \
  --model features/safety-fire_detection/runs/best.pt \
  --config features/safety-fire_detection/configs/05_data.yaml \
  --override conf_threshold=0.3
```

**Override format:** `--override <section>.<key>=<value>`
- Single-level: `training.lr=0.01`
- Nested: `augmentation.mosaic=true`
- Multiple: space-separated: `--override a=1 b=2 c=3`

## Gotchas

- **Classification and segmentation configs have no `augmentation` or `loss` section** — timm/HF models handle transforms and loss internally. Segmentation configs use `checkpoint.metric: val/mIoU` and `checkpoint.mode: max`.
- **`save_dir` is optional in logging** — defaults to `runs/<dataset_name>/` if omitted.
- **HF model configs pass extra keys directly to `from_pretrained()`** — any HF config param (`decoder_layers`, `num_queries`, etc.) works in the `model:` section without code changes.
- **`_test/` configs use `pretrained: null`** — CI tests don't download weights.

# Configs -- Pipeline-Flow YAML Configuration

All hyperparameters, paths, and settings are YAML-driven. No hardcoded values in code.

> **This directory holds only cross-feature infra:**
> - `_shared/` — non-authoritative pipeline templates
> - `_test/` — CI smoke fixtures
>
> **Feature configs live at `features/<name>/configs/`** — authoritative per feature.
> Edit `features/safety-fire_detection/configs/06_training.yaml`, `features/ppe-helmet_detection/configs/05_data.yaml`, etc.
>
> Every feature's configs folder is self-contained. Features never fall back
> to `_shared` at runtime — copy a template into your feature if you need it.

## Table of Contents

- [Quick Start -- Which Configs Do I Need?](#quick-start--which-configs-do-i-need)
- [Structure](#structure)
- [Step-by-Step Configuration Guide](#step-by-step-configuration-guide)
  - [Step 00: Data Preparation](#step-00-data-preparation)
  - [Step 01: Auto-Annotate](#step-01-auto-annotate)
  - [Step 02: Annotation Quality](#step-02-annotation-quality)
  - [Step 03: Generative Augment](#step-03-generative-augment)
  - [Step 04: Label Studio](#step-04-label-studio)
  - [Step 05: Data Definition](#step-05-data-definition)
  - [Step 06: Training](#step-06-training)
  - [Step 07: Hyperparameter Tuning](#step-07-hyperparameter-tuning)
  - [Step 08: Evaluation & Error Analysis](#step-08-evaluation--error-analysis)
  - [Step 09: Export](#step-09-export)
  - [Step 10: Inference](#step-10-inference)
- [Architecture Choices](#architecture-choices)
- [Segmentation (Detection & Classification)](#segmentation-detection--classification)
- [Pose / Keypoint Detection](#pose--keypoint-detection)
- [Feature Configs: Face and Pose](#feature-configs-face-and-pose)
- [Per-Directory Overrides](#per-directory-overrides)
- [Adding a New Use Case](#adding-a-new-use-case)
- [CLI Overrides](#cli-overrides)
- [Common Mistakes](#common-mistakes)
- [Key Rules at a Glance](#key-rules-at-a-glance)

---

## Quick Start -- Which Configs Do I Need?

| Task Type | 00 | 01 | 02 | 05 | 06 | Feature YAML | Example folder |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| Detection (raw data from multiple sources) | YES | -- | -- | YES | YES | -- | `fire_detection/`, `helmet_detection/` |
| Detection (pre-merged dataset) | -- | -- | -- | YES | YES | -- | `vehicle_detection/` |
| Classification | -- | -- | -- | YES | YES | -- | `shoes_classification/` |
| Segmentation | -- | -- | -- | YES | YES | -- | See [Segmentation](#segmentation-detection--classification) below |
| Pose training (detector + keypoints) | -- | -- | -- | YES | YES | -- | `fall_pose_estimation/` |
| Pose inference only | -- | -- | -- | -- | -- | `rtmpose_*.yaml` / `mediapipe_*.yaml` | `pose_estimation/` |
| Face recognition (inference only) | -- | -- | -- | -- | -- | `face.yaml` | `face_recognition/` |

**Steps 01--04** are shared pipeline tools in `_shared/`. They work out of the box for all use cases. Only create a per-directory copy when you need to override specific values (most commonly step 03 for generative augment). **Step 08 (evaluation) has no config YAML** -- it is CLI-only, using the data config (`05_data.yaml`) and CLI arguments. **Step 10 (inference) has no config YAML** -- it is CLI-only, using `05_data.yaml` and the trained model checkpoint.

---

## Structure

```
configs/
├── fire_detection/            # 00_data_preparation.yaml, 05_data.yaml, 06_training.yaml
├── helmet_detection/          # 00, 05, 06, 03_generative_augment.yaml (override)
├── shoes_detection/           # 00, 05, 06, 03_generative_augment.yaml (override)
├── shoes_classification/      # 05_data.yaml, 06_training.yaml (timm MobileNetV3)
├── fall_detection/       # 05, 06
├── fall_pose_estimation/      # 05, 06 (with kpt_shape)
├── phone_detection/           # 00, 05, 06
├── vehicle_detection/         # 05, 06
├── face_recognition/          # face.yaml (SCRFD + MobileFaceNet + gallery)
├── pose_estimation/           # rtmpose_s.yaml, rtmpose_t.yaml, mediapipe_full.yaml, mediapipe_lite.yaml
│
├── _shared/                   # DO NOT EDIT -- shared pipeline tool configs (steps 01--04, 07, 09)
└── _test/                     # DO NOT EDIT -- CI smoke test configs
```

---

## Step-by-Step Configuration Guide

### Step 00: Data Preparation

**File:** `00_data_preparation.yaml`
**When needed:** You have raw datasets from multiple sources (Roboflow, COCO JSON, site-collected) and need to merge them into a single YOLO-format dataset with consistent class IDs.
**Skip if:** Your dataset is already prepared in YOLO format with train/val/test splits.
**Command:** `uv run core/p00_data_prep/run.py --config configs/<name>/00_data_preparation.yaml`
**Example:** `features/safety-fire_detection/configs/00_data_preparation.yaml`

**Required fields:**

| Field | Type | Description |
|---|---|---|
| `task` | `detection` | Task type (only detection supported currently) |
| `dataset_name` | string | Name for the merged dataset (e.g. `"fire_detection"`) |
| `output_dir` | path | Where to write the merged dataset. Use `../../dataset_store/<name>` |
| `output_format` | `"yolo"` | Output label format |
| `classes` | list | Canonical target class names, in order (index = class ID) |
| `sources` | list | Raw datasets to merge (see below) |
| `splits` | dict | Output split ratios: `train`, `val`, `test`, `seed` |

**Each source entry:**

| Field | Type | Description |
|---|---|---|
| `name` | string | Identifier for this source |
| `path` | path | Path to raw data. Use `../../dataset_store/raw/<name>/...` |
| `format` | `"yolo"` or `"coco"` | Label format of this source |
| `has_splits` | bool | `true` if source already has train/valid/test subdirs |
| `splits_to_use` | list | Which splits to import (e.g. `["train", "valid", "test"]`) |
| `class_map` | dict | Maps source class names to your canonical class names |

**Optional fields:**

| Field | Default | Description |
|---|---|---|
| `options.copy_images` | `true` | Copy images (vs symlink) |
| `options.handle_duplicates` | `"rename"` | How to handle filename collisions: `rename`, `skip`, `overwrite` |
| `options.validate_labels` | `true` | Validate label coordinates are in range |

---

### Step 01: Auto-Annotate

**File:** `_shared/01_auto_annotate.yaml` (do not edit)
**When needed:** To automatically label unlabeled images using SAM3.
**Prerequisite:** Auto-label service running on `:18104`.
**Command:** `uv run core/p01_auto_annotate/run_auto_annotate.py --data-config configs/<name>/05_data.yaml --mode text`

**Key settings (for reference):**

| Setting | Default | What it does |
|---|---|---|
| `processing.mode` | `text` | Labeling mode: `text` (text prompts), `auto` (automatic), `hybrid` (both) |
| `processing.filter_mode` | `missing` | Only label images without existing labels |
| `processing.confidence_threshold` | `0.5` | Minimum confidence to keep a prediction |
| `nms.per_class_iou_threshold` | `0.5` | NMS threshold to remove duplicates |
| `text_prompts` | `{}` | Override per use-case or pass via CLI `--classes "0:fire,1:smoke"` |
| `output_format` | `"bbox"` | Label output format: `"bbox"` (YOLO-det), `"polygon"` (YOLO-seg), or `"both"` |

> **Note on YOLO-seg (`"polygon"`):** The auto-label service can produce polygon segmentation labels (`class_id x1 y1 ... xN yN`), which are compatible with QA review and Label Studio import. However, the training pipeline does **not** read YOLO-seg labels -- only mask PNGs (via `SegmentationDataset`). Convert polygon labels to mask PNGs before training if you need semantic segmentation.

---

### Step 02: Annotation Quality

**File:** `_shared/02_annotation_quality.yaml` (do not edit)
**When needed:** To assess the quality of your dataset annotations using SAM3 verification.
**Prerequisite:** SAM3 service running on `:18100` and QA service on `:18105`.
**Command:** `uv run core/p02_annotation_qa/run_qa.py --data-config configs/<name>/05_data.yaml`

**Key settings (for reference):**

| Setting | Default | What it does |
|---|---|---|
| `sampling.sample_size` | `2000` | Number of images to sample for QA |
| `scoring.thresholds.good` | `0.8` | Score above this = good annotation |
| `scoring.thresholds.review` | `0.5` | Score below this = needs review |
| `text_prompts` | per-class | SAM3 text prompts for verification (pre-populated for common classes) |

**When to override:** If you add new classes not in the default `text_prompts` list, copy to your folder and add entries for your classes.

---

### Step 03: Generative Augment

**File:** `_shared/03_generative_augment.yaml` (defaults) -- **this is the step most commonly overridden**
**When needed:** To generate synthetic training data by replacing objects in existing images (e.g. replacing helmets with bare heads to create hard negatives).
**Prerequisite:** SAM3 service on `:18100`.
**Command:** `uv run core/p03_generative_aug/run_generative_augment.py --data-config configs/<name>/05_data.yaml --config configs/<name>/03_generative_augment.yaml`

**How to create a per-directory override:**

Copy `_shared/03_generative_augment.yaml` to your folder and customize:

```yaml
# features/ppe-helmet_detection/configs/03_generative_augment.yaml
source:
  class_name: "head_with_helmet"       # what to find in existing images
  class_id: 1

target:
  class_name: "head_without_helmet"    # what to replace it with (label)
  class_id: 2

replacement_prompts:                   # text descriptions for inpainting
  - "knit hat, woolly beanie on head"
  - "baseball cap on head"
  - "bare head with hair"

inpainting:
  defaults:
    num_variants: 2                    # how many synthetic images per source
```

**Key settings:**

| Setting | Description |
|---|---|
| `source.class_name` / `class_id` | The object class to find and mask in source images |
| `target.class_name` / `class_id` | The class label assigned to the inpainted replacement |
| `replacement_prompts` | List of text descriptions -- the inpainting model generates these |
| `inpainting.defaults.num_variants` | Number of synthetic images generated per source image |
| `inpainting.defaults.strength` | Inpainting strength (0.85 default -- higher = more change) |

---

### Step 04: Label Studio

**File:** `_shared/04_label_studio.yaml` (do not edit)
**When needed:** To import model predictions into Label Studio for human review, or export reviewed annotations back to YOLO format.
**Prerequisite:** Label Studio service on `:8080` (`services/s18103_label_studio/`).
**Command:** `uv run core/p04_label_studio/bridge.py setup --data-config configs/<name>/05_data.yaml`

**Key settings (for reference):**

| Setting | Default | What it does |
|---|---|---|
| `label_studio.url` | `http://localhost:8080` | Label Studio server URL |
| `label_studio.api_key` | `""` | Set via `LS_API_KEY` env var -- **never hardcode** |
| `label_studio.import.include_predictions` | `true` | Show model pre-annotations in review UI |
| `label_studio.export.only_reviewed` | `true` | Only export annotations that have been human-reviewed |
| `label_studio.export.format` | `"yolo"` | Export format |

---

### Step 05: Data Definition

**File:** `05_data.yaml`
**When needed:** Always required for any training task.
**Command:** Referenced by `06_training.yaml` via `data.dataset_config: 05_data.yaml` -- not run directly.
**Example (detection):** `features/safety-fire_detection/configs/05_data.yaml`
**Example (classification):** `features/ppe-shoes_detection/configs/05_data.yaml`

**Required fields:**

| Field | Type | Description |
|---|---|---|
| `dataset_name` | string | Must match the dataset folder name |
| `path` | path | Dataset root. Use `../../dataset_store/<name>` |
| `train` | path | Training images subdirectory (see note below) |
| `val` | path | Validation images subdirectory |
| `names` | dict | Class ID to class name mapping: `{0: fire, 1: smoke}` |
| `num_classes` | int | Must match the number of entries in `names` |
| `input_size` | `[H, W]` | `[640, 640]` for detection, `[224, 224]` for classification |

**Optional fields:**

| Field | Default | Description |
|---|---|---|
| `test` | -- | Test images subdirectory (if you have a test split) |
| `mean` | `[0.485, 0.456, 0.406]` | RGB normalization mean (ImageNet default) |
| `std` | `[0.229, 0.224, 0.225]` | RGB normalization std (ImageNet default) |
| `layout` | -- | Set to `folder` for classification (folder-per-class structure) |

**Detection vs Classification vs Segmentation:**

| | Detection | Classification | Segmentation |
|---|---|---|---|
| `train` value | `"train/images"` | `"train"` | `"train/images"` |
| `layout` | not set | `folder` | not set |
| Label format | YOLO `.txt` (`class cx cy w h`) | Folder name = class | Mask PNG (pixel = class ID) |
| Dataset structure | `train/{images,labels}/` | `train/{class_a,class_b,...}/` | `train/{images,masks}/` |
| Label sibling dir | `labels/` | -- | `masks/` |

**Segmentation mask format:** Each image has a corresponding grayscale PNG in a sibling `masks/` directory. Pixel values are integer class IDs (0-indexed). Mask filename must match the image stem (e.g., `img001.jpg` -> `img001.png`). Missing masks default to all-zeros.

**YOLO-seg polygon format (annotation only -- NOT supported for training):** The auto-label service can output polygon labels in `labels/<stem>.txt` as `class_id x1 y1 x2 y2 ... xN yN` (normalized 0--1). Set `output_format: "polygon"` in `01_auto_annotate.yaml`. These labels can be reviewed in QA and Label Studio, but the training pipeline does not read them -- convert to mask PNGs for training.

**Pose data format:** Uses standard YOLO detection layout (`train/{images,labels}/`) with YOLO-pose labels:

```
class_id cx cy w h kx1 ky1 v1 kx2 ky2 v2 ... kxK kyK vK
```

- All coordinates normalized 0--1
- Each keypoint: `kx ky v` (x, y, visibility)
- Visibility: `0 = not labeled`, `1 = occluded`, `2 = visible`
- Total columns: `5 + num_keypoints * 3`

---

### Step 06: Training

**File:** `06_training.yaml`
**When needed:** Always required for any training task.
**Command:** `uv run core/p06_training/train.py --config configs/<name>/06_training.yaml`
**Example (detection):** `features/safety-fire_detection/configs/06_training.yaml`
**Example (classification):** `features/ppe-shoes_detection/configs/06_training.yaml`
**Example (pose):** `features/safety-fall_pose_estimation/configs/06_training.yaml`

This is the most complex config. It has several sections:

#### `model:` section

| Field | Required | Description |
|---|---|---|
| `arch` | YES | Architecture key (see [Architecture Choices](#architecture-choices)) |
| `num_classes` | YES | Must match `num_classes` in `05_data.yaml` |
| `input_size` | YES | Must match `input_size` in `05_data.yaml` |
| `pretrained` | YES | Path to local weights (`../../pretrained/yolox_m.pth`) or `true` (auto-download for timm/HF) or `null` (random init) |
| `depth` | YOLOX only | Network depth multiplier (e.g. `0.67` for YOLOX-M) |
| `width` | YOLOX only | Network width multiplier (e.g. `0.75` for YOLOX-M) |
| `timm_name` | timm only | timm model name (e.g. `mobilenetv3_small_100`) |
| `hf_model_id` | HF only | HuggingFace model ID (e.g. `PekingU/rtdetr_v2_r18vd`) |
| `kpt_shape` | pose only | Keypoint shape `[num_keypoints, dims]` (e.g. `[17, 3]`) |
| `num_keypoints` | pose only | Number of keypoints (must match `kpt_shape[0]`) -- read by `KeypointDataset` |

#### `data:` section

| Field | Default | Description |
|---|---|---|
| `dataset_config` | -- | **Filename only** -- `05_data.yaml` (resolved relative to this file's directory) |
| `batch_size` | -- | Batch size. Reduce if OOM (8 for pose, 16 for detection, 32 for classification) |
| `num_workers` | `4` | DataLoader workers |
| `pin_memory` | `true` | Pin memory for GPU transfer |

#### `augmentation:` section (detection only -- do NOT include for classification)

| Field | Default | Description |
|---|---|---|
| `mosaic` | `true` | 4-image mosaic augmentation |
| `mixup` | `true` | MixUp blending (set `false` for pose to preserve keypoints) |
| `hsv_h` | `0.015` | Hue jitter |
| `hsv_s` | `0.7` | Saturation jitter |
| `hsv_v` | `0.4` | Value jitter |
| `fliplr` | `0.5` | Horizontal flip probability |
| `flipud` | `0.0` | Vertical flip probability |
| `scale` | `[0.1, 2.0]` | Random scale range |
| `degrees` | `10.0` | Random rotation degrees |
| `translate` | `0.1` | Random translation fraction |
| `shear` | `2.0` | Random shear degrees |

**Pose augmentation restrictions:** Keypoint transforms support only `hsv_h/s/v` (color jitter) and `fliplr`. Geometric augmentations (mosaic, mixup, scale, degrees, translate, shear, flipud) either break keypoint alignment or are not yet implemented. Set `mosaic: false` and `mixup: false` for pose training. Use a narrower `scale` range (e.g., `[0.5, 1.5]`) if geometric augmentations are added later.

**Segmentation:** No `augmentation:` section -- HF segmentation models (SegFormer, Mask2Former) handle transforms internally via `AutoImageProcessor`.

#### `training:` section

| Field | Default | Description |
|---|---|---|
| `backend` | `pytorch` | `pytorch` (native loop) or `hf` (HF Trainer with DDP/DeepSpeed) |
| `epochs` | -- | Total training epochs (200 for detection, 50 for classification) |
| `optimizer` | -- | `sgd` (detection) or `adamw` (classification) |
| `lr` | -- | Learning rate (`0.01` for SGD, `0.001` for AdamW) |
| `momentum` | `0.9` | SGD momentum (ignored for Adam variants) |
| `weight_decay` | `0.0005` | Weight decay (`0.01` for AdamW) |
| `warmup_epochs` | `5` | LR warmup epochs |
| `scheduler` | `cosine` | LR scheduler: `cosine`, `plateau`, `step`, `onecycle` |
| `patience` | `50` | Early stopping patience (epochs without improvement) |
| `amp` | `true` | Mixed precision training |
| `grad_clip` | `35.0` | Gradient clipping max norm (set `0` to disable) |
| `gradient_accumulation_steps` | `1` | >1 simulates larger batch sizes on small GPUs |
| `nms_threshold` | `0.45` | NMS IoU threshold for validation (detection only) |
| `ema` | `true` | Exponential moving average |
| `ema_decay` | `0.9998` | EMA decay factor |

#### `loss:` section (detection only -- do NOT include for classification or segmentation)

| Field | Description |
|---|---|
| `type` | `yolox` for YOLOX models, `detr-passthrough` for D-FINE/RT-DETRv2 (HF models use built-in loss) |

**Segmentation:** No `loss:` section -- HF segmentation models use built-in cross-entropy pixel loss via `forward_with_loss()`.

#### `logging:` section

| Field | Description |
|---|---|
| `wandb_project` | Always `smart-camera` |
| `run_name` | Convention: `<dataset>_<arch>_v<N>` (e.g. `fire_yoloxm_v1`) |

#### `checkpoint:` section

| Field | Description |
|---|---|
| `save_best` | `true` -- save best model based on metric |
| `metric` | `val/mAP50` for detection, `val/accuracy` for classification, `val/mIoU` for segmentation |
| `mode` | `max` |
| `save_interval` | Save checkpoint every N epochs (e.g. `10`) |

#### `seed:` (top-level)

Always set to `42` for reproducibility.

---

### Step 07: Hyperparameter Tuning

**File:** `_shared/08_hyperparameter_tuning.yaml` (do not edit -- override in your folder if needed)
**When needed:** After initial training, when you want Optuna to automatically search for better hyperparameters.
**Command:** `uv run core/p07_hpo/run_hpo.py --config configs/<name>/06_training.yaml`

**Key settings (for reference, do not change in `_shared/`):**

| Setting | Default | What it does |
|---|---|---|
| `study.n_trials` | `50` | Number of Optuna trials |
| `trial.epochs` | `30` | Epochs per trial (shorter than full training) |
| `sampler` | `tpe` | TPE sampler (efficient Bayesian search) |
| `pruning.type` | `median` | Prune underperforming trials early |
| `search_space` | 21 params | Searches over lr, optimizer, augmentation, loss weights, etc. |

**When to override:** If your dataset is very small or your task is unusual, copy to your folder and adjust `search_space` ranges or `n_trials`.

---

### Step 08: Evaluation & Error Analysis

**Config:** None -- evaluation is CLI-driven, no config YAML.
**When needed:** After training (or after HPO), to measure model performance on the test split.
**Command:** `uv run core/p08_evaluation/evaluate.py --model runs/<name>/best.pt --config configs/<name>/05_data.yaml --task detection`

Evaluation uses the data config (`05_data.yaml`) for dataset paths/class info and CLI arguments for thresholds. Per-task metrics are auto-selected by the `--task` flag.

**Per-task metrics:**

| Task | `--task` | Primary Metric | Secondary Metrics |
|---|---|---|---|
| Detection | `detection` | mAP@0.5 | per-class AP, precision, recall |
| Classification | `classification` | top-1 accuracy | top-5 accuracy, per-class accuracy |
| Segmentation | `segmentation` | mIoU | per-class IoU |
| Pose | `detection` | mAP@0.5 (box-level) | per-class AP (evaluates person box detection) |

**CLI arguments:**

| Arg | Default | Description |
|---|---|---|
| `--model` | required | Path to `.pt` checkpoint |
| `--config` | required | Path to `05_data.yaml` |
| `--task` | `detection` | `detection`, `classification`, or `segmentation` |
| `--split` | `test` | `train`, `val`, or `test` |
| `--conf` | `0.5` | Confidence threshold |
| `--iou` | `0.5` | IoU threshold for NMS/matching |
| `--batch-size` | `16` | Batch size |
| `--error-analysis` | flag | Opt-in: error breakdown, confusion pairs, optimal thresholds |

**Examples:**

```bash
# Detection evaluation
uv run core/p08_evaluation/evaluate.py \
  --model runs/fire_detection/best.pt \
  --config features/safety-fire_detection/configs/05_data.yaml \
  --task detection

# Classification evaluation
uv run core/p08_evaluation/evaluate.py \
  --model runs/shoes_classification/best.pt \
  --config features/ppe-shoes_detection/configs/05_data.yaml \
  --task classification

# Segmentation evaluation
uv run core/p08_evaluation/evaluate.py \
  --model runs/road_segmentation/best.pt \
  --config configs/road_segmentation/05_data.yaml \
  --task segmentation

# With error analysis
uv run core/p08_evaluation/evaluate.py \
  --model runs/fire_detection/best.pt \
  --config features/safety-fire_detection/configs/05_data.yaml \
  --task detection --error-analysis
```

---

### Step 09: Export

**File:** `_shared/09_export.yaml` (do not edit)
**When needed:** To convert a trained `.pt` model to ONNX for edge deployment.
**Command:** `uv run core/p09_export/export.py --model runs/<name>/best.pt --training-config configs/<name>/06_training.yaml --export-config configs/_shared/09_export.yaml`

**Key settings (for reference):**

| Setting | Default | What it does |
|---|---|---|
| `opset` | `18` | ONNX opset version |
| `simplify` | `true` | Run onnxsim optimization |
| `input_size` | `[640, 640]` | Must match your training `input_size` |
| `optimization_level` | `O2` | ONNX Runtime graph optimization level |
| `quantization.enabled` | `false` | Enable INT8 quantization for edge chips |
| `quantization.preset` | `avx512_vnni` | Hardware target: `avx512_vnni` (server), `arm64` (edge) |

**When to override:** If your model uses a different `input_size` (e.g. 224x224 for classification), copy to your folder and update `input_size`.

---

### Step 10: Inference

**Config:** None -- inference is CLI-driven, no config YAML. Uses `05_data.yaml` for class names and input size, plus the trained model checkpoint.
**When needed:** After training and export, to run predictions on images or videos.

**Detection inference (single image):**

```bash
uv run core/p10_inference/predictor.py \
  --model runs/<name>/best.pt \
  --config configs/<name>/05_data.yaml \
  --image path/to/image.jpg
```

**Video inference:**

```bash
uv run core/p10_inference/video_inference.py \
  --model runs/<name>/best.pt \
  --config configs/<name>/05_data.yaml \
  --video path/to/video.mp4
```

**Face recognition** uses `features/access-face_recognition/configs/face.yaml` (see [Feature Configs: Face and Pose](#feature-configs-face-and-pose) for details).

---

## Architecture Choices

All architectures are **Apache 2.0 or MIT**. AGPL-3.0 models (YOLOv5/v8/v11/YOLO26 from Ultralytics) are **prohibited**.

### Training architectures (used in `06_training.yaml`)

| `model.arch` | Model | Task | `pretrained` | Extra fields |
|---|---|---|---|---|
| `yolox-m` | YOLOX-M (25.3M) | Detection | `../../pretrained/yolox_m.pth` | `depth: 0.67`, `width: 0.75` |
| `yolox-s` | YOLOX-S (9.0M) | Detection | local `.pth` | `depth: 0.33`, `width: 0.50` |
| `yolox-tiny` | YOLOX-Tiny (5.1M) | Detection | local `.pth` | `depth: 0.33`, `width: 0.375` |
| `yolox-l` | YOLOX-L (54.2M) | Detection | local `.pth` | `depth: 1.0`, `width: 1.0` |
| `dfine-s` | D-FINE-S (10M) | Detection | `true` | -- |
| `dfine-n` | D-FINE-N (4M) | Detection | `true` | -- |
| `dfine-m` | D-FINE-M (19M) | Detection | `true` | -- |
| `rtdetr-r18` | RT-DETRv2-R18 (20M) | Detection | `true` | -- |
| `rtdetr-r50` | RT-DETRv2-R50 (42M) | Detection | `true` | -- |
| `timm` | Any timm model | Classification | `true` | `timm_name: mobilenetv3_small_100` |
| `hf-detection` | Any HF detection model | Detection | `true` | `hf_model_id: ...` |
| `hf-classification` | Any HF classification model | Classification | `true` | `hf_model_id: ...` |
| `hf-segformer` | SegFormer | Segmentation | `true` | `hf_model_id: ...` |

**Recommended defaults:**
- **Detection:** Start with `yolox-m`. Escalate to `dfine-s` if you need higher accuracy.
- **Classification:** Use `timm` with `timm_name: mobilenetv3_small_100`.
- **Lightweight detection:** `yolox-tiny` or `dfine-n` for edge deployment.

### Inference-only architectures (used in feature YAML configs)

| Arch | Model | Config file |
|---|---|---|
| `scrfd-500m` | SCRFD face detector | `face_recognition/face.yaml` |
| `mobilefacenet` | MobileFaceNet embedder | `face_recognition/face.yaml` |
| `rtmpose-s` / `rtmpose-t` | RTMPose keypoint | `pose_estimation/rtmpose_*.yaml` |
| `mediapipe-full` / `mediapipe-lite` | MediaPipe Pose | `pose_estimation/mediapipe_*.yaml` |

---

## Segmentation (Detection & Classification)

Segmentation uses HuggingFace models with `AutoModelForSemanticSegmentation`. The pipeline is fully wired -- the trainer auto-detects `output_format == "segmentation"` and uses pixel-level cross-entropy loss + mIoU validation metrics.

### Data layout

```
dataset_store/<usecase>/
├── train/
│   ├── images/          # source images
│   │   ├── img001.jpg
│   │   └── ...
│   └── masks/           # grayscale PNGs, pixel value = class ID (0-indexed)
│       ├── img001.png   # must match image stem
│       └── ...
├── val/{images,masks}/
└── test/{images,masks}/
```

### `05_data.yaml` (segmentation)

```yaml
dataset_name: "road_segmentation"
path: "../../dataset_store/road_segmentation"
train: "train/images"
val: "val/images"
test: "test/images"
names: {0: background, 1: road, 2: sidewalk, 3: vehicle}
num_classes: 4
input_size: [512, 512]
```

### `06_training.yaml` (segmentation)

```yaml
model:
  arch: hf-segformer                    # or hf-mask2former, hf-dinov2-seg
  pretrained: nvidia/segformer-b2-finetuned-ade-512-512
  num_classes: 4
  input_size: [512, 512]
  # Any HF config param works -- forwarded to from_pretrained():
  # decode_head.num_classes: 4

data:
  dataset_config: 05_data.yaml
  batch_size: 8
  num_workers: 4

# NO augmentation: section -- HF handles transforms via AutoImageProcessor
# NO loss: section -- HF uses built-in cross-entropy pixel loss

training:
  backend: hf                          # use HF Trainer for segmentation
  epochs: 50
  optimizer: adamw
  lr: 0.00006
  weight_decay: 0.01
  warmup_epochs: 5
  scheduler: cosine
  patience: 20
  amp: true

logging:
  wandb_project: smart-camera
  run_name: road_segformer_v1

checkpoint:
  save_best: true
  metric: val/mIoU                     # segmentation uses mIoU, not mAP
  mode: max
  save_interval: 5

seed: 42
```

### Available segmentation architectures

| `model.arch` | Model | Params | Notes |
|---|---|---|---|
| `hf-segformer` | SegFormer (B0--B5) | 4--85M | Lightweight, efficient; use B0/B2 for edge |
| `hf-mask2former` | Mask2Former | 45--200M | Instance + panoptic + semantic; heavier |
| `hf-dinov2-seg` | DINOv2 + Seg head | varies | Strong features, good for few-shot |

**Recommended:** Start with `nvidia/segformer-b2-finetuned-ade-512-512`. Escalate to B5 or Mask2Former for higher accuracy.

---

## Pose / Keypoint Detection

Pose training produces a single-stage detector that predicts both bounding boxes and keypoints (YOLO-pose format). Inference then chains this detector with a specialized pose estimator (RTMPose or MediaPipe) for the two-stage pipeline.

### Single-stage: YOLOX with keypoints (training)

This trains a YOLOX-M model to output both boxes and keypoints simultaneously.

**`05_data.yaml` (pose):**

```yaml
dataset_name: "fall_pose_estimation"
path: "../../dataset_store/fall_pose_estimation"
train: "train/images"
val: "val/images"
test: "test/images"
names: {0: person}
num_classes: 1
input_size: [640, 640]
kpt_shape: [17, 3]                     # 17 COCO keypoints, 3 values (x, y, visibility)
num_keypoints: 17                      # required by KeypointDataset
```

**`06_training.yaml` (pose):**

```yaml
model:
  arch: yolox-m
  pretrained: ../../pretrained/yolox_m.pth
  num_classes: 1
  input_size: [640, 640]
  depth: 0.67
  width: 0.75
  kpt_shape: [17, 3]

data:
  dataset_config: 05_data.yaml
  batch_size: 8                        # reduce for pose (larger labels per sample)
  num_workers: 4

augmentation:
  mosaic: false                        # DISABLED -- breaks keypoint alignment
  mixup: false                         # DISABLED -- breaks keypoint alignment
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  fliplr: 0.5                          # OK -- KeypointTransform mirrors coordinates
  flipud: 0.0                          # DISABLED -- vertical flip invalid for pose
  scale: [0.5, 1.5]                    # narrower range if geometric augment added later
  degrees: 0.0                         # DISABLED -- not implemented for keypoints
  translate: 0.0                       # DISABLED -- not implemented for keypoints
  shear: 0.0                           # DISABLED -- not implemented for keypoints

training:
  backend: pytorch
  epochs: 150
  optimizer: sgd
  lr: 0.005                            # lower than detection (0.01)
  momentum: 0.9
  weight_decay: 0.0005
  warmup_epochs: 10
  scheduler: cosine
  patience: 60
  amp: true
  grad_clip: 35.0
  ema: true
  ema_decay: 0.9998

loss:
  type: yolox

logging:
  wandb_project: smart-camera
  run_name: fall_pose_yoloxm_v1

checkpoint:
  save_best: true
  metric: val/mAP50                    # evaluates person box detection quality
  mode: max
  save_interval: 10

seed: 42
```

**Pose augmentation rules:**

| Augmentation | Supported? | Notes |
|---|---|---|
| `hsv_h/s/v` | Yes | Color jitter -- does not affect coordinates |
| `fliplr` | Yes | KeypointTransform mirrors x-coordinates and swaps left/right indices |
| `mosaic`, `mixup` | No | Breaks keypoint alignment between image and labels |
| `scale`, `degrees`, `translate`, `shear` | No | Geometric transforms not yet implemented for keypoints |
| `flipud` | No | Vertical flip is semantically invalid for human pose |

### Two-stage: Detector + Pose Estimator (inference)

After training a person detector, chain it with a pre-trained pose model for keypoint estimation. This is configured via feature YAML files, not training configs.

```yaml
# features/safety-fall_pose_estimation/configs/rtmpose_s.yaml
pose_model:
  arch: rtmpose-s                      # or rtmpose-t, mediapipe-lite, mediapipe-full
  model_path: pretrained/rtmpose_s_256x192.onnx
  input_size: [256, 192]

detector:
  model_path: runs/fall_pose/best.pt   # your trained person detector
  data_config: ../../fall_pose_estimation/05_data.yaml
  conf_threshold: 0.5
  iou_threshold: 0.45

inference:
  person_conf_threshold: 0.5
  person_class_ids: [0]
  keypoint_conf_threshold: 0.3
  output_format: native                # "coco" to auto-convert MediaPipe 33->17 keypoints
```

| Pose model | Keypoints | Backend | Batch | 3D? |
|---|---|---|---|---|
| `rtmpose-s` | 17 (COCO) | ONNX Runtime | Yes | No |
| `rtmpose-t` | 17 (COCO) | ONNX Runtime | Yes | No |
| `mediapipe-lite` | 33 | TFLite | No | Yes |
| `mediapipe-full` | 33 | TFLite | No | Yes |

---

## Feature Configs: Face and Pose

These are **inference-only** configs -- you do not train these models. They configure pre-trained ONNX models with detection/recognition thresholds.

### Face Recognition (`face_recognition/face.yaml`)

```yaml
face_detector:
  arch: scrfd-500m
  model_path: pretrained/scrfd_500m.onnx
  conf_threshold: 0.5

face_embedder:
  arch: mobilefacenet
  model_path: pretrained/mobilefacenet_arcface.onnx
  embedding_dim: 512

gallery:
  path: data/face_gallery/default.npz
  similarity_threshold: 0.4            # lower = stricter matching
```

### Pose Estimation (`pose_estimation/*.yaml`)

Each pose config specifies a pose model + a person detector reference:

```yaml
pose_model:
  arch: rtmpose-s                      # or rtmpose-t, mediapipe-full, mediapipe-lite
  model_path: pretrained/rtmpose_s_256x192.onnx

detector:
  model_path: runs/fall/best.pt        # person detector checkpoint
  conf_threshold: 0.5
```

---

## Per-Directory Overrides

Any directory can include its own version of a shared config. The file is **complete and self-contained** -- no inheritance or merging with `_shared/`.

```
configs/
├── helmet_detection/
│   ├── 05_data.yaml
│   ├── 06_training.yaml
│   └── 03_generative_augment.yaml   # Full copy with helmet-specific prompts
```

The script loads whichever path you pass:

```bash
# Default generative augment (null source/target)
uv run core/p03_generative_aug/run_generative_augment.py \
  --config configs/_shared/03_generative_augment.yaml

# Helmet-specific override (helmet -> non-helmet replacement)
uv run core/p03_generative_aug/run_generative_augment.py \
  --config features/ppe-helmet_detection/configs/03_generative_augment.yaml
```

---

## Adding a New Use Case

### Checklist

1. **Create your directory:**
   ```bash
   mkdir configs/<your_usecase>
   ```

2. **Decide which configs you need** using the [decision table](#quick-start--which-configs-do-i-need) above.

3. **Copy from the closest existing use case:**

   | Your task | Copy from |
   |---|---|
   | Detection | `fire_detection/` |
   | Classification | `shoes_classification/` |
   | Segmentation | Create new -- see [Segmentation](#segmentation-detection--classification) section |
   | Detection + pose | `fall_pose_estimation/` |

   ```bash
   # Example: new detection use case
   cp features/safety-fire_detection/configs/05_data.yaml configs/<your_usecase>/
   cp features/safety-fire_detection/configs/06_training.yaml configs/<your_usecase>/
   ```

4. **Edit `05_data.yaml`** -- update these fields:
   - `dataset_name` -- your dataset name
   - `path` -- path to your dataset in `dataset_store/`
   - `names` -- your class ID to name mapping
   - `num_classes` -- must match the count of `names`
   - `input_size` -- `[640, 640]` for detection, `[224, 224]` for classification

5. **Edit `06_training.yaml`** -- update these fields:
   - `model.num_classes` -- must match `05_data.yaml`
   - `model.pretrained` -- correct path or `true`
   - `logging.run_name` -- descriptive name for your experiment
   - `data.batch_size` -- reduce if you get OOM errors

6. **Validate your configs:**
   ```bash
   uv run utils/validate_config.py configs/<your_usecase>/
   ```

7. **Train:**
   ```bash
   uv run core/p06_training/train.py --config configs/<your_usecase>/06_training.yaml
   ```

---

## CLI Overrides

Any config value can be overridden from the command line:

```bash
uv run core/p06_training/train.py \
  --config features/safety-fire_detection/configs/06_training.yaml \
  --override training.lr=0.005 training.epochs=100
```

**Override format:** `--override <section>.<key>=<value>`
- Single-level: `training.lr=0.01`
- Nested: `augmentation.mosaic=false`
- Multiple: space-separated: `--override a=1 b=2 c=3`

---

## Common Mistakes

- **`num_classes` mismatch** -- If `model.num_classes` in `06_training.yaml` does not match `num_classes` in `05_data.yaml`, training crashes or produces garbage. Always keep them in sync.

- **Wrong `pretrained` format** -- YOLOX uses a local `.pth` file (`../../pretrained/yolox_m.pth`). timm/HF models use `true` for auto-download. Using `true` for YOLOX will fail silently.

- **Including `augmentation` or `loss` in classification configs** -- timm/HF classification models handle transforms and loss internally. Including these sections causes errors or is silently ignored. Only detection configs have `augmentation:` and `loss:` sections.

- **Wrong `dataset_config` path** -- `data.dataset_config: 05_data.yaml` is filename-only, resolved relative to the training config's directory. Do not use full paths or `../` prefixes.

- **Editing `_shared/` or `_test/`** -- These are maintained by the infra team. Your changes will be overwritten on next merge. Copy to your use-case directory instead.

- **Missing `layout: folder` for classification** -- Classification datasets use folder-per-class structure. Without `layout: folder` in `05_data.yaml`, the loader assumes YOLO detection format and fails.

- **`input_size` inconsistency** -- The `input_size` in `05_data.yaml`, `06_training.yaml` (`model.input_size`), and `09_export.yaml` must all match.

- **Forgetting `kpt_shape` for pose** -- Pose training configs need `model.kpt_shape: [17, 3]` (17 COCO keypoints, 3 values each: x, y, visibility). Also set `num_keypoints: 17` in `05_data.yaml`.

- **Enabling mosaic/mixup for pose** -- These augmentations break keypoint alignment. Always set `mosaic: false` and `mixup: false` in pose training configs.

- **Using geometric augmentations for pose** -- `scale`, `degrees`, `translate`, `shear` are not yet implemented for keypoints. Set them to `0` or omit them.

- **Wrong mask format for segmentation** -- Masks must be grayscale PNGs with pixel values equal to class IDs (0-indexed). Mask filenames must match image stems. Missing masks silently default to all-zeros.

- **Including `augmentation` or `loss` in segmentation configs** -- HF segmentation models handle transforms and loss internally via `AutoImageProcessor` and `forward_with_loss()`. Including these sections causes errors or is silently ignored.

- **AGPL/Ultralytics models** -- Prohibited. Never use YOLOv5/v8/v9/v10/v11/YOLO26 from Ultralytics.

- **Looking for an evaluation config YAML** -- Step 08 (evaluation) has no config YAML. It uses `05_data.yaml` for dataset info and CLI arguments (`--task`, `--conf`, `--iou`, `--split`, `--error-analysis`) for thresholds and options.

---

## Key Rules at a Glance

1. **No inheritance/merging** -- each config file is complete and self-contained. Per-directory overrides are full copies, not partial diffs.
2. **Paths relative from project root** -- dataset paths use `../../dataset_store/`, pretrained weights use `../../pretrained/`.
3. **`dataset_config` is filename-only** -- `05_data.yaml`, resolved relative to the training config's directory.
4. **CLI overrides use dot notation** -- `--override section.key=value`.
5. **No `${var}` interpolation across files** -- variable interpolation works within a single YAML file only.
6. **Do not modify `_shared/` or `_test/`** -- copy to your use-case folder to override.
7. **Evaluation (step 08) has no config YAML** -- it is CLI-only, using `05_data.yaml` + `--task` flag.
8. **Inference (step 10) has no config YAML** -- it is CLI-only, using `05_data.yaml` + trained model checkpoint. Face and pose use their own feature YAML configs.
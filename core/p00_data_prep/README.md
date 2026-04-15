# Generic CV Data Preparation Tool

A generic tool for combining multiple source datasets into training-ready datasets for computer vision tasks.

## Features

- **Multi-format support**: YOLO, COCO JSON, Pascal VOC XML
- **Multi-task support**: Object detection, classification, segmentation, pose estimation
- **Flexible splitting**: Stratified train/val/test splits via `splits.json` catalog
- **Class mapping**: Map different source class names to canonical target classes
- **Duplicate handling**: Automatic renaming of duplicate files
- **Re-split capability**: Change split ratios without recombining data
- **Dry-run mode**: Preview changes before writing files

## Installation

The tool is part of the `camera_edge` project. All dependencies are already installed.

```bash
# Already installed via uv sync
```

## Usage

### Basic Usage

```bash
# Combine helmet detection datasets
uv run core/p00_data_prep/run.py \
  --config features/ppe-helmet_detection/configs/00_data_preparation.yaml

# Dry run to preview
uv run core/p00_data_prep/run.py \
  --config features/ppe-helmet_detection/configs/00_data_preparation.yaml \
  --dry-run

# Force overwrite existing dataset
uv run core/p00_data_prep/run.py \
  --config features/ppe-helmet_detection/configs/00_data_preparation.yaml \
  --force
```

### Re-split Only

Change split ratios without recombining data:

```bash
uv run core/p00_data_prep/run.py \
  --config features/ppe-helmet_detection/configs/00_data_preparation.yaml \
  --resplit-only --splits 0.85 0.1 0.05
```

## Config Format

Each use case has a `00_data_preparation.yaml` config file in its config folder:

```yaml
# features/ppe-helmet_detection/configs/00_data_preparation.yaml

# Task type (detection, classification, segmentation, pose)
task: detection

# Dataset identification
dataset_name: "helmet_detection"
output_dir: "../../dataset_store/helmet_detection"

# Output format
output_format: "yolo"

# Target classes (canonical names)
classes:
  - person
  - head_with_helmet
  - head_without_helmet
  - head_with_nitto_hat

# Source datasets to combine
sources:
  - name: "ppe_yolov8"
    path: "../../dataset_store/raw/helmet_detection/ppe_yolov8"
    format: "yolo"
    has_splits: true
    splits_to_use: ["train", "valid", "test"]
    class_map:
      "Hardhat": "head_with_helmet"
      "NO-Hardhat": "head_without_helmet"
      "Person": "person"

  - name: "hard_hat_workers"
    path: "../../dataset_store/raw/helmet_detection/hard_hat_workers/data"
    format: "voc"
    has_splits: false
    class_map:
      "helmet": "head_with_helmet"
      "no_helmet": "head_without_helmet"

# Split configuration (stratified by default)
splits:
  train: 0.8
  val: 0.1
  test: 0.1
  seed: 42

# Processing options
options:
  copy_images: true
  handle_duplicates: "rename"
  validate_labels: true
```

## Config Fields

| Field | Description |
|-------|-------------|
| `task` | Task type: detection, classification, segmentation, pose |
| `dataset_name` | Name of the output dataset |
| `output_dir` | Output directory path (relative to config file) |
| `output_format` | Output format: yolo, coco, flat |
| `classes` | List of canonical target class names |
| `sources` | List of source datasets to combine |
| `sources[].name` | Source dataset name (for duplicate prefix) |
| `sources[].path` | Path to source dataset |
| `sources[].format` | Source format: yolo, voc, coco, folder |
| `sources[].has_splits` | Whether source has train/val/test subdirs |
| `sources[].splits_to_use` | Which splits to include (if has_splits=true) |
| `sources[].class_map` | Map source class names to target names |
| `splits.train` | Training set ratio (0-1) |
| `splits.val` | Validation set ratio (0-1) |
| `splits.test` | Test set ratio (0-1) |
| `splits.seed` | Random seed for reproducibility |
| `options.copy_images` | Copy images (true) or create symlinks (false) |
| `options.handle_duplicates` | How to handle duplicates: skip, rename, overwrite |
| `options.validate_labels` | Validate annotations |

## Output Structure

The tool creates a split directory layout ready for training:

```
dataset_store/helmet_detection/
├── train/
│   ├── images/              # Training images
│   │   ├── img001.jpg
│   │   └── ...
│   └── labels/              # YOLO format annotations
│       ├── img001.txt
│       └── ...
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## Supported Formats

| Format | Description | Detection | Classification | Segmentation | Pose |
|--------|-------------|-----------|----------------|--------------|------|
| **YOLO** | .txt files with normalized bboxes | ✅ | ✅ | ✅ | ✅ |
| **COCO** | JSON annotations | ✅ | ✅ | ✅ | ✅ |
| **VOC** | Pascal VOC XML | ✅ | - | - | - |
| **Folder** | Class-named folders | - | ✅ | - | - |

## Examples

### Combine Helmet Detection Datasets

```bash
uv run core/p00_data_prep/run.py \
  --config features/ppe-helmet_detection/configs/00_data_preparation.yaml
```

### Combine Fire Detection Datasets

```bash
uv run core/p00_data_prep/run.py \
  --config features/safety-fire_detection/configs/00_data_preparation.yaml
```

### Re-split with Different Ratios

```bash
# 85% train, 10% val, 5% test (stratified)
uv run core/p00_data_prep/run.py \
  --config features/ppe-helmet_detection/configs/00_data_preparation.yaml \
  --resplit-only --splits 0.85 0.1 0.05
```

### Dry Run to Preview

```bash
uv run core/p00_data_prep/run.py \
  --config features/ppe-helmet_detection/configs/00_data_preparation.yaml \
  --dry-run
```

## Architecture

```
core/p00_data_prep/
├── run.py                       # CLI entry point
├── core/
│   ├── base.py                  # TaskAdapter abstract base
│   └── splitter.py              # Split catalog generation
├── adapters/
│   └── detection.py             # Detection task adapter
├── parsers/
│   ├── yolo.py                  # YOLO format parser
│   ├── voc.py                   # Pascal VOC parser
│   └── coco.py                  # COCO JSON parser
├── converters/
│   └── to_yolo.py               # Convert to YOLO format
└── utils/
    ├── class_mapper.py          # Class mapping utilities
    └── file_ops.py              # File operations
```

## Integration with Training

After running data preparation, the dataset is ready for training:

```bash
# 1. Prepare dataset
uv run core/p00_data_prep/run.py \
  --config features/ppe-helmet_detection/configs/00_data_preparation.yaml

# 2. Train (uses the prepared dataset)
uv run core/p06_training/train.py \
  --config features/ppe-helmet_detection/configs/06_training.yaml
```

## TODO

- [ ] Add classification adapter
- [ ] Add segmentation adapter
- [ ] Add pose estimation adapter
- [ ] Add format auto-detection
- [ ] Add validation module
- [ ] Add progress bar for large datasets

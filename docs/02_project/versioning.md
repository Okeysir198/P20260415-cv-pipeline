# Dataset + Model Versioning

## Overview

We use **DVC** (Data Version Control) to version datasets and model weights locally. DVC tracks large files alongside git — each git commit reproduces the exact dataset + model state.

## What We Version

| Artifact | Location | Why |
|---|---|---|
| **Datasets** | `../../dataset_store/{model}/` | Reproducible training — know exactly which images trained which model |
| **Pretrained weights** | `pretrained/` | Base weights for fine-tuning (YOLOX-M, SCRFD, MobileFaceNet) |
| **Trained models** | `runs/{model}/` | Deployment artifacts — best.pth, exported .onnx |

## Storage

```
Local remote (default):     ../../dvc-storage/
Cloud remote (future):      s3://bucket/dvc/ or gdrive:// or ssh://
```

To add a cloud remote later:
```bash
dvc remote add cloud s3://your-bucket/dvc-storage
dvc push -r cloud     # push to cloud
dvc pull -r cloud     # pull from cloud
```

## Naming Convention

| Artifact | Git Tag Format | Example |
|---|---|---|
| Dataset version | `{model}-data-v{N}` | `fire-data-v1`, `helmet-data-v3` |
| Model version | `{model}-model-v{N}` | `fire-model-v1`, `shoes-model-v2` |
| Combined release | `{model}-v{N}` | `fire-v1` (dataset + model together) |

## Workflows

### New Dataset Version

When new labeled images are added:

```bash
# 1. Add/update images in dataset directory
# 2. Track with DVC
dvc add ../../dataset_store/fire_detection/

# 3. Commit the .dvc file (tiny pointer)
git add ../../dataset_store/fire_detection.dvc
git commit -m "fire dataset v2: +5K images from Bintulu site"

# 4. Tag for easy reference
git tag fire-data-v2

# 5. Push data to storage
dvc push
```

### New Model Version

After training produces a better model:

```bash
# 1. Train
uv run core/p06_training/train.py --config features/safety-fire_detection/configs/06_training.yaml

# 2. Evaluate (confirm improvement)
uv run core/p08_evaluation/evaluate.py --model runs/fire_detection/best.pth --config features/safety-fire_detection/configs/05_data.yaml

# 3. Track with DVC
dvc add runs/fire_detection/

# 4. Commit
git add runs/fire.dvc
git commit -m "fire model v2: mAP 0.87 -> 0.91, trained on fire-data-v2"

# 5. Tag
git tag fire-model-v2
dvc push
```

### Reproduce a Specific Version

Any team member can get the exact state:

```bash
# Get code + DVC pointers at that version
git checkout fire-model-v2

# Download the actual data/model files
dvc checkout

# Now runs/fire_detection/best.pth and datasets are at v2
```

### Compare Versions

```bash
# See what changed between dataset versions
dvc diff fire-data-v1 fire-data-v2

# See git log of model improvements
git log --oneline --grep="fire model"
```

## DVC Files in Git

DVC creates small `.dvc` files (pointer files) that git tracks:

```
runs/fire.dvc              # Points to runs/fire_detection/ in DVC storage
pretrained.dvc             # Points to pretrained/ in DVC storage
../../dataset_store/fire_detection.dvc  # Points to dataset in DVC storage
```

These `.dvc` files are ~100 bytes each. The actual data lives in DVC storage.

## Quick Reference

```bash
dvc add <path>         # Track a file/directory
dvc push               # Upload to remote storage
dvc pull               # Download from remote storage
dvc checkout           # Restore files to match current .dvc pointers
dvc diff <tag1> <tag2> # Show what changed between versions
dvc status             # Show which tracked files have changed
```

# p01_data — Data Loading + Online Augmentation

## Purpose

Load images and labels from disk, apply online augmentation during training.

## Files

| File | Purpose |
|---|---|
| `base_dataset.py` | `BaseDataset(Dataset, ABC)` — task-agnostic base class for any CV dataset. Handles image discovery and loading; subclasses implement `load_target()` and `format_target()`. |
| `detection_dataset.py` | `YOLOXDataset(BaseDataset)` — YOLO-format detection dataset (`class_id cx cy w h` per line). Also exports `build_dataloader()` with forkserver multiprocessing and `collate_fn()`. |
| `classification_dataset.py` | `ClassificationDataset`, `build_classification_dataloader()` — folder-based or label-file classification layouts. |
| `segmentation_dataset.py` | `SegmentationDataset`, `build_segmentation_dataloader()` — mask PNG segmentation (pixel value = class ID). |
| `keypoint_dataset.py` | `KeypointDataset`, `build_keypoint_dataloader()` — YOLO-pose format keypoint/pose data. |
| `transforms.py` | `build_transforms(config, is_train, input_size, mean, std)` — torchvision.transforms.v2 pipeline with Mosaic, MixUp, CopyPaste, IRSimulation as v2.Transform subclasses. BGR-to-RGB conversion happens automatically in `_to_v2_sample()`. |

## Adding a New Dataset Format

Implement `BaseDataset` with two methods:

```python
from core.p05_data.base_dataset import BaseDataset

class MyDataset(BaseDataset):
    def load_target(self, label_path):
        # Read your label format (COCO JSON, XML, folder name, etc.)
        ...

    def format_target(self, raw_target, image_size):
        # Convert to model-ready format
        ...
```

## Customizing Augmentation Per Model

Edit the `augmentation:` section in `configs/<model>/06_training.yaml`:

```yaml
augmentation:
  mosaic: true       # 4-image mosaic
  mixup: true        # image blending
  hsv_h: 0.015       # hue jitter
  scale: [0.1, 2.0]  # random scale range
  degrees: 10.0      # rotation
```

## Config Reference

- Dataset paths: `features/<name>/configs/05_data.yaml`
- Augmentation params: `features/<name>/configs/06_training.yaml` → `augmentation:` section

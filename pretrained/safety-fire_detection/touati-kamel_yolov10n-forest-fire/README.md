
---
library_name: ultralytics
tags:
- object-detection
- forest-fire
- fire-detection
- yolo
- yolov10
datasets:
- forest-fire-detection
---

# YOLOv10n Forest Fire Detection Model

This model is trained to detect forest fires and related phenomena including:
- Fire
- Fire-smoke  
- Fog
- Factory-smoke

## Model Details
- Architecture: YOLOv10n
- Input size: 640x640
- Classes: 6 classes
- Training: Enhanced with class separation techniques and checkpointing

## Usage
```python
from ultralytics import YOLO

model = YOLO('touatikamel/yolov10n-forest-fire-detection')
results = model.predict('path/to/image.jpg')
```

## Checkpoints
This repository includes training checkpoints saved every 10 epochs:
- `checkpoints/best.pt` - Best performing model
- `checkpoints/latest.pt` - Latest checkpoint
- `checkpoints/epoch_N.pt` - Checkpoint from epoch N

## Training Resume
To resume training from a checkpoint:
```python
# Download checkpoint and resume training
checkpoint_path, metadata = download_checkpoint_from_hf("latest")
model = YOLO(checkpoint_path)
model.train(data="path/to/dataset.yaml", resume=True)
```

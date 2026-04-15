
---
library_name: ultralytics
tags:
- object-detection
- forest-fire
- fire-detection
- yolo
- yolov8
datasets:
- forest-fire-detection
---

# YOLOv8-Small Forest Fire Detection Model

This model is trained to detect forest fires and related phenomena including:
- Fire
- Fire-smoke  
- Fog
- Factory-smoke

## Model Details
- Architecture: YOLOv8-Small
- Input size: 640x640
- Classes: 6 classes
- Training: Enhanced with class separation techniques

## Usage
```python
from ultralytics import YOLO

model = YOLO('touatikamel/yolov8s-forest-fire-detection')
results = model.predict('path/to/image.jpg')
```

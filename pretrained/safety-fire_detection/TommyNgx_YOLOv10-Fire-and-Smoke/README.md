---
tags:
- object-detection
- fire-detection
- smoke-detection
license: apache-2.0
datasets:
- fire-smoke-dataset
model-index:
- name: YOLOv10-Fire-Smoke-Detection
  results:
  - task:
      type: object-detection
      name: Object Detection
    dataset:
      name: Fire and Smoke Dataset
      type: fire-smoke-dataset
    metrics:
    - type: mAP
      value: 0.85
widget:
- src: >-
    https://huggingface.co/TommyNgx/YOLOv10-Fire-and-Smoke-Detection/resolve/main/examples/example1.jpg
  example_title: Fire
- src: >-
    https://huggingface.co/TommyNgx/YOLOv10-Fire-and-Smoke-Detection/resolve/main/examples/example1.jpg
  example_title: Smoke
library_name: pytorch
base_model:
- Ultralytics/YOLO11
metrics:
- recall
---

  
# YOLOv10: Real-Time Fire and Smoke Detection

This repository contains a YOLOv10 model trained for real-time fire and smoke detection. The model uses the Ultralytics YOLO framework to perform object detection with high accuracy and efficiency. Users can adjust the confidence and IoU thresholds for optimal detection results.

## Model Details

- **Model Type**: YOLOv8 (adapted for YOLOv10 features)
- **Task**: Object Detection
- **Framework**: PyTorch
- **Input Size**: Adjustable (default: 640x640)
- **Classes Detected**: Fire, Smoke
- **File**: `best.pt`

## How to Use the Model

This model is hosted on Hugging Face and can be accessed via the **Inference Widget** or programmatically using the Hugging Face Transformers pipeline.

### Inference Widget

Upload an image to the widget below and adjust the following:
- **Confidence Threshold**: Minimum confidence level for predictions (default: 0.25).
- **IoU Threshold**: Minimum IoU level for object matching (default: 0.45).
- **Image Size**: Resize input image (default: 640x640).

### Usage with Python

To use the model programmatically:

```python
import torch
from ultralytics import YOLO
from PIL import Image

# Load the model
model_path = "pytorch_model.bin"
state_dict = torch.load(model_path, map_location="cpu")

# Initialize the YOLO model
model = YOLO()  # Replace with the correct YOLO class
model.load_state_dict(state_dict)

# Run inference
image = Image.open("path/to/image.jpg")
results = model.predict(image, conf=0.25, iou=0.45)
results.show()
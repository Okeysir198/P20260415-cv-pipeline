---
language: en
license: agpl-3.0
tags:
  - computer-vision
  - object-detection
  - license-plate
  - yolov11
  - ultralytics
  - finetuned
datasets:
  - roboflow/license-plate-recognition-rxg4e
metrics:
  - precision
  - recall
  - mAP@50
  - mAP@50-95
---

# YOLOv11-License-Plate Detection

This is a fine-tuned version of YOLOv11 (n, s, m, l, x) specialized for **License Plate Detection**, using a public dataset from Roboflow Universe:  
[License Plate Recognition Dataset (10,125 images)](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/11)

## 🚀 Use Cases

- Smart Parking Systems
- Tollgate / Access Control Automation
- Traffic Surveillance & Enforcement
- ALPR with OCR Integration

## 🏋️ Training Details

- Base Model: YOLOv11 (`n`, `s`, `m`, `l`, `x`)
- Training Epochs: 300
- Input Size: 640x640
- Optimizer: SGD (Ultralytics default)
- Device: NVIDIA A100
- Data Format: YOLOv5-compatible (images + labels in txt)

## 📊 Evaluation Metrics (YOLOv11x)

| Metric        | Value   |
|---------------|---------|
| Precision     | 0.9893  |
| Recall        | 0.9508  |
| mAP@50        | 0.9813  |
| mAP@50-95     | 0.7260  |

> For full table across models (n to x), please see the [README](README.md)

## 📦 Model Variants

- PyTorch (.pt) — for use with Ultralytics CLI and Python API
- ONNX (.onnx) — for cross-platform inference

## 🧠 How to Use

With Python (Ultralytics API):
```python
from ultralytics import YOLO
model = YOLO('yolov11x-license-plate.pt')
results = model.predict(source='image.jpg')
```

## 📜 License

- Base Model (YOLOv11): AGPLv3 by [Ultralytics](https://github.com/ultralytics/ultralytics)
- Dataset: CC BY 4.0 by Roboflow Universe
- This model: AGPLv3 (due to YOLOv11 license inheritance)

## ✅ License Compliance Reminder

In accordance with the AGPLv3 license:
- If you **use this model** in a service or project, you must **open source** the code that uses it.
- Please give proper attribution to Roboflow, Ultralytics, and MorseTechLab when using or deploying.

For license details, refer to [GNU AGPLv3 License](https://www.gnu.org/licenses/agpl-3.0.en.html)

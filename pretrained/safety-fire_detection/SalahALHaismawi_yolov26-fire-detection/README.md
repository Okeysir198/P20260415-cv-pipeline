---
license: mit
library_name: ultralytics
pipeline_tag: object-detection
tags:
  - yolo
  - yolov26
  - fire-detection
  - smoke-detection
  - computer-vision
  - pytorch
  - ultralytics
  - real-time
  - safety
datasets:
  - custom
metrics:
  - mAP
  - precision
  - recall
model-index:
  - name: yolov26-fire-detection
    results:
      - task:
          type: object-detection
        metrics:
          - name: mAP@50
            type: mAP
            value: 94.9
          - name: mAP@50-95
            type: mAP
            value: 68.0
          - name: Precision
            type: precision
            value: 89.6
          - name: Recall
            type: recall
            value: 88.8
---

# YOLOv26 Fire Detection

Real-time fire and smoke detection model based on YOLOv26 (Ultralytics). Achieves **94.9% mAP@50** on fire/smoke detection tasks.

## Model Description

This model detects fire, smoke, and related fire indicators in images and videos. Built on YOLOv26-S architecture and trained on 8,939 annotated images.

### Classes
- **fire** - Active flames
- **smoke** - Smoke plumes
- **other** - Related fire indicators

## Performance

| Metric | Score |
|--------|-------|
| mAP@50 | **94.9%** |
| mAP@50-95 | 68.0% |
| Precision | 89.6% |
| Recall | 88.8% |

## Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | YOLOv26-S |
| Epochs | 100 |
| Batch Size | 16 |
| Image Size | 640x640 |
| Optimizer | AdamW |
| Learning Rate | 0.01 |

## Usage

### Installation

```bash
pip install ultralytics
```

### Inference

```python
from ultralytics import YOLO

# Load model
model = YOLO("path/to/best.pt")

# Run inference
results = model.predict("image.jpg", conf=0.25)

# Process results
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls]
        print(f"Detected: {label} ({conf:.2f})")
```

### Video Inference

```python
from ultralytics import YOLO

model = YOLO("path/to/best.pt")
results = model.predict("video.mp4", save=True, conf=0.25)
```

### Webcam (Real-time)

```python
from ultralytics import YOLO

model = YOLO("path/to/best.pt")
results = model.predict(source=0, show=True, conf=0.25)
```

## Detection Samples

![Detection Sample](detection_sample.jpg)

## Training Curves

![Training Results](results.png)

## Use Cases

- Building fire safety monitoring
- Wildfire early detection systems
- Industrial safety surveillance
- Smart home fire detection
- Drone-based fire monitoring

## Limitations

- May have reduced accuracy in low-light conditions
- Smoke detection can be affected by fog/steam
- Best performance on images similar to training data

## Dataset

Trained on fire detection dataset from [Roboflow Universe](https://universe.roboflow.com/personal-bodxv/fire-detection-sejra-fognw/dataset/1):
- 8,939 images
- License: CC BY 4.0

## Citation

```bibtex
@misc{yolov26-fire-detection,
  author = {Salah AL-Haismawi},
  title = {YOLOv26 Fire Detection},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/SalahALHaismawi/yolov26-fire-detection}}
}
```

## License

MIT License

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO implementation
- [Roboflow](https://roboflow.com) for the dataset

---
license: agpl-3.0
tags:
- object-detection
- YOLO26
- computer-vision
- ultralytics
library_name: ultralytics
pipeline_tag: object-detection
---

# YOLO26 MEDIUM Model

Fine-tuned YOLO26 model for object detection.

## Model Details

- **Architecture**: YOLO26Medium
- **Framework**: Ultralytics YOLO26
- **Resolution**: 640x640
- **Epochs**: 100
- **Batch Size**: 16

## Classes

`car`

## Usage

```python
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="rujutashashikanjoshi/yolo26-testA-vehicle-detection-4931_full-100m", filename="best.pt")
model = YOLO(model_path)

results = model("your_image.jpg", conf=0.25)
results[0].show()
```

## Training Config

```json
{
    "model_type": "medium",
    "epochs": 100,
    "batch_size": 16,
    "image_size": 640,
    "learning_rate": 0.001,
    "weight_decay": 0.0005,
    "momentum": 0.937,
    "warmup_epochs": 5,
    "workers": 4,
    "patience": 75,
    "save_period": 10,
    "conf_threshold": 0.25,
    "iou_threshold": 0.45,
    "output_dir": "./output/4139_full_y26_testA",
    "optimizer": "AdamW"
}
```

## License

AGPL-3.0

---
datasets:
- pyronear/pyro-dataset
license: apache-2.0
tags:
- wildfire
- fire-detection
- yolo
- object-detection
- pyronear
---

# pyronear/yolo11s_sensitive-detector_v1.0.0

Pyronear YOLO model for early wildfire smoke detection.

**Release name:** Sensitive Detector
**Version:** v1.0.0

## Model details

| Field | Value |
|---|---|
| Architecture | yolo11s |
| Image size | 1024 |
| Epochs | 50 |
| Optimizer | AdamW |
| Weights SHA-256 | `f6f7868833804965...` |
| Training data MD5 | `fcd56c8728d160e9...` |

## Files

| File | Description |
|---|---|
| `best.pt` | PyTorch weights |
| `onnx_cpu.tar.gz` | ONNX export (cpu) |
| `ncnn_cpu.tar.gz` | NCNN export (cpu) |
| `manifest.yaml` | Full training manifest |

## Usage

### PyTorch (ultralytics)

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict("image.jpg", imgsz=1024, conf=0.2, iou=0.01)
for r in results:
    print(r.boxes)  # bounding boxes + confidences
```

### ONNX (onnxruntime)

```python
from huggingface_hub import hf_hub_download
import onnxruntime as ort
import numpy as np
from PIL import Image

path = hf_hub_download(repo_id="pyronear/yolo11s_sensitive-detector_v1.0.0", filename="onnx_cpu.tar.gz")
session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

img = Image.open("image.jpg").resize((1024, 1024))
x = np.array(img).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
outputs = session.run(None, {session.get_inputs()[0].name: x})
```

### NCNN

```bash
# Unzip first
tar -xzf ncnn_cpu.tar.gz
```

### Download with huggingface_hub

```python
from huggingface_hub import snapshot_download

local_dir = snapshot_download(repo_id="pyronear/yolo11s_sensitive-detector_v1.0.0")
```

### Pyronear engine (sequential smoke detection)

```python
from pyroengine.engine import Engine

engine = Engine(
    conf_thresh=0.20,
    nb_consecutive_frames=5,
)
# feed frames one by one — engine.predict() returns a score
score = engine.predict(pil_image, cam_id="camera_01")
if score > engine.conf_thresh:
    print("Smoke detected!")
```

## About Pyronear

[Pyronear](https://pyronear.org) builds open-source tools for early wildfire detection.

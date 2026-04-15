---
license: apache-2.0
language:
- en
tags:
- computer vision
- wildfire
---
# Pyronear YOLOv8s for Early Wildfire Detection

This repository contains the custom YOLOv8s model trained by Pyronear for early wildfire detection. Our model leverages the powerful capabilities of the Ultralytics YOLOv8 framework to accurately identify potential wildfire hotspots in real-time.

## Installation

Install the ultralytics package including all requirements in a Python>=3.8 environment with PyTorch>=1.8.

```bash
pip install ultralytics
```

For alternative installation methods including Conda, Docker, and Git, please refer to the [Ultralytics Quickstart Guide](https://docs.ultralytics.com/quickstart).

## Usage

### Python

You can also use the YOLOv8 model in a Python environment:

```python
from ultralytics import YOLO
from PIL import Image

# Load the model
model = YOLO('pyronear/yolov8s.pt')

# Run inference
results = model('path/to/your/image.jpg', conf=0.2, iou=0.1)

# Display results
results.show()
```

For more examples and detailed usage, see the [YOLOv8 Python Docs](https://docs.ultralytics.com/usage/python).

## Acknowledgements

This project utilizes the [Ultralytics YOLOv8 framework](https://ultralytics.com/yolov8). Special thanks to the Ultralytics team for their support and contributions to the open-source community.
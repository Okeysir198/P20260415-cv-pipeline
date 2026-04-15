---
pipeline_tag: object-detection
tags:
- ultralytics
- yolo
- yolov11
- qualcomm
- advantech
- edge-ai
- computer-vision
license: agpl-3.0
---

# qualcomm-ultralytics-ppe_detection

* **Model creator:** [Ultralytics](https://github.com/ultralytics/ultralytics)
* **Original model:** [YOLOv11n](https://huggingface.co/qualcomm/YOLOv11-Detection)

This repository contains the **YOLOv11n** model optimized for Qualcomm hardware using **Qualcomm® AI Engine Direct (SNPE)**.

It is designed for high-performance, real-time object detection inference on edge devices powered by Qualcomm Snapdragon platforms, enabling efficient on-device AI capabilities with low latency and reduced power consumption.

## Model Details

* **Developed by:** Advantech-EIOT / Ultralytics
* **Architecture:** YOLOv11n
* **Task:** PPE Detection
* **Precision:** Quantized (w8a16) for NPU optimization
* **Input Resolution:** 160 x 160
* **Calibration Data:** Calibrated using 1,000 images
* **Optimization:** Qualcomm® AI Stack / SNPE SDK

## Hardware Compatibility

This model is highly optimized for Advantech Edge AI platforms powered by Qualcomm processors:

* **Linux:** Dragonwing® Platforms (e.g. Dragonwing® IQ-9075)

## Limitations and Disclaimer

YOLOv11 is a powerful real-time object detection model, but it may exhibit limitations depending on the environment and deployment context.

* **Accuracy:** The model's accuracy, especially for small objects or in low-light conditions, may be impacted by the reduced input resolution (160x160) and quantization (w8a16). Users should validate outputs for critical applications or safety-critical systems.
* **Usage:** Please refer to the [Ultralytics AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) for usage restrictions and acceptable use policies.
* **Edge Optimization:** Inference performance (FPS) and bounding box precision may vary depending on the specific hardware configuration, camera ISP pipelines, and thermal constraints of the edge device.
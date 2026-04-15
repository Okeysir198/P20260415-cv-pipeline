---
language: en
license: mit
tags:
- environment
- computer-vision
- vit
- climate-change
---

# wildfire_smoke_segmentation_vit

## Overview
This model is a Vision Transformer (ViT) designed for the early detection of wildfires via satellite and aerial imagery. By identifying specific smoke patterns and thermal anomalies, it provides real-time alerts for environmental monitoring agencies.



## Model Architecture
The model is based on the **ViT-Base** (Patch 16) architecture:
- **Patching**: Divides input images into 16x16 patches to capture global spatial dependencies.
- **Attention**: Uses multi-head self-attention to distinguish between cloud cover and low-density smoke plumes.
- **Pre-training**: Initialized on ImageNet-21k and fine-tuned on the FIRESAT dataset.

## Intended Use
- **Remote Sensing**: Automated monitoring of vast forested areas via Sentinel-2 or Landsat imagery.
- **Early Warning Systems**: Integration into IoT-enabled lookout towers for local fire departments.
- **Post-Fire Analysis**: Assessing the spread and intensity of smoke for environmental impact studies.

## Limitations
- **Atmospheric Conditions**: Heavy cloud cover or fog can lead to false positives.
- **Resolution**: Accuracy drops significantly for images where the smoke plume is smaller than 32x32 pixels.
- **Time of Day**: Optimized for daytime multi-spectral imagery; night-time performance relies on thermal band availability.
---
license: apache-2.0
tags:
- segmentation
- computer-vision
- satellite-imagery
- parking-lot-detection
- segformer
library_name: pytorch
pipeline_tag: image-segmentation
---

# Model Description

This is a SegFormer model trained for parking lot boundary segmentation. The model was trained on the dataset introduced in the paper “A Pipeline and NIR-Enhanced Dataset for Parking Lot Segmentation” (WACV 2025).

# Intended Use

The model can be used for segmentation tasks to detect and delineate parking lots from satellite imagery.

# Citation

If you use this model, please cite the following paper:

@inproceedings{qiam2025pipeline,
  title={A Pipeline and NIR-Enhanced Dataset for Parking Lot Segmentation},
  author={Qiam, Shirin and Devunuri, Saipraneeth and Lehe, Lewis J},
  booktitle={2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages={1227--1236},
  year={2025},
  organization={IEEE}
}
# Core Pipeline

Model-agnostic ML pipeline engine. Supports **detection**, **classification**, and **segmentation** via config-driven architecture dispatch.

```
p01_data/        → Load images + labels, apply augmentation (detection + classification)
p02_models/      → Model architectures + registries (YOLOX, HF Transformers, timm, face, pose)
p03_training/    → Training loop (native + HF Trainer), losses, callbacks, LR scheduling
p04_evaluation/  → Evaluate models, compute metrics, error analysis, visualize results
p05_export/      → Export to ONNX, quantize, benchmark for edge deployment
p06_inference/   → Predictions, video processing, face recognition, pose estimation
```

## Architecture

Models declare an `output_format` property. The trainer, evaluator, and predictor dispatch behavior based on it:

| output_format | Task | Validation Metric | Example Models |
|---------------|------|-------------------|----------------|
| `"yolox"` | Detection | mAP@0.5 | YOLOX-M/S/L/Tiny/Nano |
| `"detr"` | Detection | mAP@0.5 | D-FINE-S/N/M, RT-DETRv2 |
| `"classification"` | Classification | accuracy, top-5 | timm (MobileNetV3, EfficientNet, ViT), HF classifiers |
| `"segmentation"` | Segmentation | mIoU | HF SegFormer, Mask2Former |

### Training Paths

The trainer auto-detects which path to use:

- **Models with `forward_with_loss()`** (HF, timm) — model computes its own loss internally
- **Models without it** (YOLOX) — trainer calls `forward()` then a separate loss function

Two backends available:
- **Native** (`DetectionTrainer`) — PyTorch training loop with ModelEMA, callbacks, checkpointing
- **HF Trainer** (`train_with_hf()`) — maps YAML config to HF `TrainingArguments` for DDP, DeepSpeed, gradient accumulation

## Module Reference

### p01_data/ — Datasets & Transforms

| File | Key Exports | Purpose |
|------|-------------|---------|
| `base_dataset.py` | `BaseDataset` ABC | Interface: `load_target()`, `format_target()` |
| `detection_dataset.py` | `YOLOXDataset`, `build_dataloader()`, `collate_fn()` | YOLO-format detection data |
| `classification_dataset.py` | `ClassificationDataset`, `build_classification_dataloader()` | Folder-based or label-file classification |
| `transforms.py` | `DetectionTransform`, `Mosaic`, `MixUp`, `CopyPaste`, `IRSimulation`, `build_transforms()` | torchvision v2 augmentation pipeline |

### p02_models/ — Architectures & Registries

| File | Key Exports | Purpose |
|------|-------------|---------|
| `registry.py` | `MODEL_REGISTRY`, `_VARIANT_MAP`, `build_model()`, `register_model()` | Main detection model dispatch |
| `base.py` | `DetectionModel` ABC | Interface: `forward()`, `output_format`, `strides`, `get_param_groups()` |
| `yolox.py` | `YOLOXModel`, `build_yolox()` | Self-contained CSPDarknet + PAFPN + Decoupled Head |
| `hf_model.py` | `HFDetectionModel`, `HFClassificationModel`, `HFSegmentationModel` | Generic HF Transformers adapters (detection/classification/segmentation) |
| `timm_model.py` | `TimmModel`, `build_timm_model()` | Thin timm adapter for any timm classification model |
| `dfine.py` | Variant registration | D-FINE-S/N/M → `"hf_detection"` |
| `rtdetr.py` | Variant registration | RT-DETRv2-R18/R50 → `"hf_detection"` |
| `timm_variants.py` | Variant registration | mobilenetv3, efficientnet, resnet, vit, convnext → `"timm"` |
| `hf_classification_variants.py` | Variant registration | hf-vit-cls, hf-dinov2-cls, etc. → `"hf_classification"` |
| `hf_segmentation_variants.py` | Variant registration | hf-segformer, hf-mask2former, etc. → `"hf_segmentation"` |

**Inference-only models** (ONNX Runtime / TFLite, separate registries):

| File | Key Exports | Purpose |
|------|-------------|---------|
| `face_base.py` | `FaceDetector` ABC, `FaceEmbedder` ABC | Face detection/embedding interfaces |
| `face_registry.py` | `FACE_DETECTOR_REGISTRY`, `FACE_EMBEDDER_REGISTRY`, builders | Face model dispatch |
| `scrfd.py` | `SCRFDModel` | SCRFD-500M/2.5G face detector (ONNX Runtime) |
| `mobilefacenet.py` | `MobileFaceNetModel` | MobileFaceNet-ArcFace 512-d embedder (ONNX Runtime) |
| `pose_base.py` | `PoseModel` ABC, `COCO_KEYPOINT_NAMES`, `COCO_SKELETON` | Pose estimation interface + keypoint schemas |
| `pose_registry.py` | `POSE_MODEL_REGISTRY`, `build_pose_model()` | Pose model dispatch |
| `rtmpose.py` | `RTMPoseModel` | RTMPose-S/T, 17 COCO keypoints (ONNX Runtime) |
| `mediapipe_pose.py` | `MediaPipePoseModel` | MediaPipe Pose, 33 3D landmarks (TFLite) |

### p03_training/ — Training Loop

| File | Key Exports | Purpose |
|------|-------------|---------|
| `trainer.py` | `DetectionTrainer`, `ModelEMA` | Native PyTorch training loop with EMA, dispatches by `output_format` |
| `hf_trainer.py` | `train_with_hf()` | HF Trainer backend — maps YAML config to `TrainingArguments` (DDP, DeepSpeed) |
| `losses.py` | `LOSS_REGISTRY`, `build_loss()`, `YOLOXLoss`, `FocalLoss`, `IoULoss` | Loss functions (YOLOX only — HF/timm use built-in) |
| `lr_scheduler.py` | `build_scheduler()`, `WarmupScheduler`, `CosineScheduler`, `PlateauScheduler`, `StepScheduler`, `OneCycleScheduler` | LR scheduling with linear warmup |
| `callbacks.py` | `CheckpointSaver`, `EarlyStopping`, `WandBLogger`, `CallbackRunner` | Training lifecycle hooks |
| `postprocess.py` | `POSTPROCESSOR_REGISTRY`, `register_postprocessor()`, `postprocess()` | YOLOX decode + NMS (HF models use built-in) |
| `metrics_registry.py` | `METRICS_REGISTRY`, `register_metrics()`, `compute_metrics()` | Per-format validation metrics dispatch |
| `train.py` | `main()` | CLI entry point — dispatches native vs HF backend |

### p04_evaluation/ — Metrics & Visualization

| File | Key Exports | Purpose |
|------|-------------|---------|
| `evaluator.py` | `ModelEvaluator` | Batched inference + metrics (mAP, accuracy, F1); supports .pt and .onnx |
| `sv_metrics.py` | `compute_map()`, `compute_confusion_matrix()`, `compute_precision_recall()` | supervision-based detection metrics |
| `visualization.py` | `draw_bboxes()`, `plot_confusion_matrix()`, `plot_pr_curve()`, `plot_training_curves()` | Plotting and annotation |
| `error_analysis.py` | `ErrorAnalyzer`, `ErrorCase`, `ErrorReport` | FP/FN classification, per-class optimal thresholds, hardest images |
| `evaluate.py` | `main()` | CLI entry point |

### p05_export/ — ONNX Export & Optimization

| File | Key Exports | Purpose |
|------|-------------|---------|
| `exporter.py` | `ModelExporter` | .pt → .onnx (HF via Optimum, YOLOX via torch.onnx); onnxsim simplification |
| `quantize.py` | `ModelQuantizer` | Dynamic/static INT8 quantization + ORTOptimizer graph optimization; presets (avx512_vnni, avx2, arm64) |
| `benchmark.py` | `ModelBenchmark` | Latency, throughput, model size comparison (PyTorch vs ONNX vs quantized) |
| `export.py` | `main()` | CLI entry point |

### p06_inference/ — Prediction & Video

| File | Key Exports | Purpose |
|------|-------------|---------|
| `predictor.py` | `DetectionPredictor` | Single/batch inference (.pt and .onnx backends) |
| `video_inference.py` | `VideoProcessor` | Frame-by-frame detection + alert logic + ByteTrack tracking |
| `supervision_bridge.py` | `to_sv_detections()`, `from_sv_detections()`, `build_annotators()`, `annotate_frame()`, `create_tracker()`, `update_tracker()` | Bridge to supervision library + tracker helpers |
| `face_gallery.py` | `FaceGallery` | Face enrollment, cosine matching, persistence (.npz) |
| `face_predictor.py` | `FacePredictor` | Violation detection → face detection → embedding → gallery match |
| `pose_predictor.py` | `PosePredictor` | Person detection → pose estimation → keypoint analysis |

## Import Convention

All imports use the full package path from the project root:

```python
# Detection
from core.p01_data.detection_dataset import YOLOXDataset, build_dataloader
from core.p02_models import build_model
from core.p03_training.trainer import DetectionTrainer
from core.p04_evaluation.evaluator import ModelEvaluator
from core.p04_evaluation.error_analysis import ErrorAnalyzer
from core.p05_export.exporter import ModelExporter
from core.p06_inference.predictor import DetectionPredictor

# Classification
from core.p01_data.classification_dataset import ClassificationDataset, build_classification_dataloader
from core.p02_models.timm_model import TimmModel

# Face recognition (inference-only)
from core.p02_models.face_registry import build_face_detector, build_face_embedder
from core.p06_inference.face_predictor import FacePredictor

# Pose estimation (inference-only)
from core.p02_models.pose_registry import build_pose_model
from core.p06_inference.pose_predictor import PosePredictor
```

## Adding a New Model

1. Create `core/p06_models/<name>.py` — implement `DetectionModel` ABC with `forward()`, `output_format`, `strides`
2. Optionally add `forward_with_loss()` for built-in loss (recommended for non-detection tasks)
3. Register with `@register_model("name")` decorator
4. Add variant aliases in `<name>_variants.py`
5. Import in `core/p06_models/__init__.py` to trigger registration

## Adding a New Task Type

1. Add a dataset class in `p01_data/` for the data format
2. Set a new `output_format` string on your model
3. Add metrics branch in `trainer.py:_validate()` for that format
4. Add decode branch in `trainer.py:_decode_predictions()` for that format

## Adding a New Pipeline Step

1. Create `core/p07_<name>/` with `__init__.py`
2. Implement your module
3. Update `CLAUDE.md` folder structure section

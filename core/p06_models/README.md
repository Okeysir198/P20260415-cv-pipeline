# p02_models — Model Architectures + Registries

## Purpose

Define model architectures and provide registries for building models from config.

## Registries

| Registry | Builder | Models |
|---|---|---|
| `MODEL_REGISTRY` | `build_model(config)` | YOLOX-M/Tiny/Nano, D-FINE-S/M/N, RT-DETRv2-R18/R50 |
| `POSE_MODEL_REGISTRY` | `build_pose_model(config)` | RTMPose-S/T, MediaPipe Pose |
| `FACE_DETECTOR_REGISTRY` | `build_face_detector(config)` | SCRFD-500M |
| `FACE_EMBEDDER_REGISTRY` | `build_face_embedder(config)` | MobileFaceNet-ArcFace |

## Files

| File | Purpose |
|---|---|
| `registry.py` | `MODEL_REGISTRY`, `register_model()`, `build_model()` — arch dispatch via `config["model"]["arch"]` |
| `base.py` | `DetectionModel(ABC, nn.Module)` — interface all detection models implement (`forward`, `output_format`, `strides`, `get_param_groups`) |
| `yolox.py` | Self-contained YOLOX (CSPDarknet + PAFPN + Decoupled Head) — M/Tiny/Nano variants |
| `hf_model.py` | `HFDetectionModel` — generic adapter for any HuggingFace `ForObjectDetection` model. All HF config keys pass through to `from_pretrained()` unchanged. |
| `dfine.py` | D-FINE-S/M/N variant registration (delegates to `hf_model.py`) |
| `rtdetr.py` | RT-DETRv2-R18/R50 variant registration (delegates to `hf_model.py`) |
| `timm_variants.py` | mobilenetv3, efficientnet, resnet, vit, convnext variant registration → `"timm"` |
| `hf_classification_variants.py` | hf-vit-cls, hf-dinov2-cls, etc. variant registration → `"hf_classification"` |
| `hf_segmentation_variants.py` | hf-segformer, hf-mask2former, etc. variant registration → `"hf_segmentation"` |
| `pose_base.py` | `PoseModel(ABC)` — interface for pose estimation models |
| `pose_registry.py` | `POSE_MODEL_REGISTRY`, `register_pose_model()`, `build_pose_model()` |
| `rtmpose.py` | RTMPose-S/T ONNX pose estimation |
| `mediapipe_pose.py` | MediaPipe Pose (33 3D landmarks, TFLite) |
| `face_base.py` | `FaceDetector` / `FaceEmbedder` ABCs, `ARCFACE_REF_LANDMARKS` |
| `face_registry.py` | `FACE_DETECTOR_REGISTRY`, `FACE_EMBEDDER_REGISTRY`, `build_face_detector()`, `build_face_embedder()` |
| `scrfd.py` | SCRFD-500M/2.5G face detector (ONNX Runtime, 5-point landmarks) |
| `mobilefacenet.py` | MobileFaceNet-ArcFace 512-d embedder (ONNX Runtime, affine alignment) |

## Adding a New Detection Model

### Option A: HuggingFace Model (registration only)

Create a new file or add to an existing one:

```python
from core.p02_models.hf_model import HF_MODEL_REGISTRY

HF_MODEL_REGISTRY["my-model"] = (MyModelClass, MyConfigClass, "org/model-name")
```

Then in config YAML:

```yaml
model:
  arch: my-model
  num_classes: 2
```

All HF config keys (e.g., `decoder_layers`, `num_queries`, `focal_loss_alpha`) work in YAML without code changes.

### Option B: Custom Model

```python
from core.p02_models.base import DetectionModel
from core.p02_models.registry import register_model

@register_model("my-custom-model")
def build_my_model(config):
    return MyModel(config)

class MyModel(DetectionModel):
    def forward(self, x): ...
    @property
    def output_format(self): return "my_format"
    @property
    def strides(self): return [8, 16, 32]
```

## Config Reference

- `model.arch` in `features/<name>/configs/06_training.yaml` selects the model
- All HF config keys pass through to `from_pretrained()`
- Pose models: `features/safety-fall_pose_estimation/configs/*.yaml`
- Face models: `features/access-face_recognition/configs/face.yaml`

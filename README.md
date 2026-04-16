# Camera Edge — Edge AI Computer Vision Platform

Config-driven, model-agnostic computer vision platform for edge deployment. Transforms existing IP cameras into intelligent sensors with real-time detection, pose estimation, face recognition, and behavioral analysis — all running on low-power edge chips.

**[Product Roadmap](docs/ROADMAP.md)** | **[Documentation](docs/README.md)**

## Platform Features

| Cluster | Features | Models |
|---------|----------|--------|
| **PPE** | Helmet, safety shoes, glasses*, masks*, gloves*, aprons*, harness* | YOLOX-M, D-FINE-S, MobileNetV3 |
| **Safety** | Fire/smoke, fall detection, poketenashi violations, forklift proximity*, near-miss* | YOLOX-M, RTMPose-S, MediaPipe Pose |
| **Access Control** | Zone intrusion, face recognition | YOLOX-Tiny, SCRFD-500M, MobileFaceNet |
| **Traffic** | Smart parking, adaptive signal control | YOLOX-Tiny, D-FINE-N |

\* = Phase 2 &nbsp;|&nbsp; All models Apache 2.0 or MIT — $0 licensing. **AGPL-3.0 models (Ultralytics) prohibited.**

## Architecture

```
IP Camera → Edge Chip → Detection / Pose / Face → Alert + Tracking + Identity → Backend
                │
          ┌─────┴──────────────────────────────────────┐
          │  Detection    YOLOX-M / D-FINE-S / RT-DETRv2│
          │  Pose         RTMPose-S / MediaPipe Pose    │
          │  Face         SCRFD-500M + MobileFaceNet    │
          │  Tracking     ByteTrack                     │
          │  Classify     MobileNetV3-Small             │
          └────────────────────────────────────────────┘
```

**Target edge chips:** AX650N (18 INT8 TOPS) · CV186AH (7.2 INT8 TOPS)

### Pipeline Overview

```
Raw images
    │
    ├── [P00] Data prep       Format conversion (COCO/VOC/YOLO → YOLO)
    ├── [P01] Auto-annotate   SAM3 + text prompts → YOLO labels
    ├── [P02] Annotation QA   SAM3 structural + quality verification
    ├── [P03] Gen. augment    SAM3 + Flux diffusion → synthetic data
    ├── [P04] Label Studio    Human review bridge
    │
    ├── [P05] Data            Dataset loaders (detection, cls, seg, keypoint)
    ├── [P06] Models          Registry: YOLOX, D-FINE, RT-DETRv2, timm, HF seg
    ├── [P06] Training        PyTorch native + HF Trainer + custom backend
    ├── [P07] HPO             Optuna hyperparameter optimization
    ├── [P08] Evaluation      mAP, mIoU, accuracy + error analysis
    ├── [P09] Export          ONNX + INT8 quantization
    └── [P10] Inference       PyTorch + ONNX predictor, video, face, pose
```

Adding a new detection task = **2 YAML files + 0 code changes**. Adding a new model = 1 file implementing `DetectionModel` ABC.

### Task Dispatch

The pipeline dispatches behavior based on `model.output_format`:

```
config.model.arch → build_model() → output_format
                                         │
                              ┌──────────┼──────────────┬──────────────┐
                           yolox      detr        classification   segmentation
                              │          │                │                │
                       YOLOXLoss   HF built-in    CrossEntropy      HF built-in
                       + SimOTA       loss         → Accuracy          loss
                              │          │                │            → mIoU
                              └──────────┴────────────────┴────────────────┘
                                                    │
                                          ONNX Export → INT8 → Edge Deploy
```

## Project Structure

```
ai/
├── core/                              # Model-agnostic engine (registry-based dispatch)
│   ├── p00_data_prep/                 # Format conversion (COCO/VOC/YOLO → YOLO)
│   ├── p01_auto_annotate/             # LangGraph + SAM3 auto-labeling (text/rule/hybrid)
│   ├── p02_annotation_qa/             # LangGraph + SAM3 annotation QA + scoring
│   ├── p03_generative_aug/            # SAM3 + Flux diffusion data augmentation
│   ├── p04_label_studio/              # Label Studio bridge (YOLO ↔ LS format)
│   ├── p05_data/                      # Dataset loaders + transforms (torchvision v2)
│   │   ├── detection_dataset.py       # YOLOXDataset
│   │   ├── classification_dataset.py  # ClassificationDataset
│   │   ├── segmentation_dataset.py    # SegmentationDataset
│   │   └── keypoint_dataset.py        # KeypointDataset
│   ├── p06_models/                    # Model registry (build_model, @register_model)
│   │   ├── yolox.py                   # YOLOX-M/S/L (self-contained, no official pkg)
│   │   ├── dfine.py                   # D-FINE-S/N/M (HF Transformers)
│   │   ├── rtdetr.py                  # RT-DETRv2-R18/R50 (HF Transformers)
│   │   ├── hf_model.py                # Generic HF detection adapter
│   │   ├── scrfd.py                   # SCRFD-500M face detector (ONNX)
│   │   ├── mobilefacenet.py           # MobileFaceNet-ArcFace embedder (ONNX)
│   │   ├── rtmpose.py                 # RTMPose-S/T pose estimator (ONNX)
│   │   └── mediapipe_pose.py          # MediaPipe Pose (TFLite, 33 landmarks)
│   ├── p06_training/                  # Training loop + losses + EMA + callbacks
│   │   ├── trainer.py                 # DetectionTrainer (PyTorch native)
│   │   ├── hf_trainer.py              # HF Trainer wrapper (DDP, DeepSpeed)
│   │   ├── losses.py                  # YOLOXLoss + SimOTA, FocalLoss, IoULoss
│   │   ├── lr_scheduler.py            # Cosine / Plateau / Step / OneCycle
│   │   └── callbacks.py               # CheckpointSaver, EarlyStopping, WandBLogger
│   ├── p07_hpo/                       # Optuna hyperparameter optimization
│   ├── p08_evaluation/                # Evaluator + mAP/mIoU/accuracy + error analysis
│   ├── p09_export/                    # ONNX export + INT8 quantization + benchmark
│   └── p10_inference/                 # Predictor + VideoProcessor + face + pose
│
├── utils/                             # Shared utilities (no PyTorch/HF dependency)
│   ├── config.py                      # YAML load/merge/validate, ${var} interpolation
│   ├── exploration.py                 # Dataset stats, class distribution, normalization
│   ├── scaffold.py                    # build_05_data_yaml, build_06_training_yaml
│   ├── validate_config.py             # Config validation CLI
│   ├── yolo_io.py                     # YOLO label read/write helpers
│   └── service_health.py             # Service availability checks
│
├── app_demo/                          # Gradio demo (multi-tab, all use cases)
│
├── features/                          # Self-contained per-feature folders
│   ├── _TEMPLATE/                     # Copy via scripts/new_feature.sh
│   ├── safety-*/                      # fire, fall, poketenashi (phone usage)
│   ├── ppe-*/                         # helmet, shoes, gloves
│   ├── access-*/                      # face_recognition, zone_intrusion
│   └── detect_vehicle/                # uniform layout: configs, code, samples,
│                                      # notebooks, tests, runs, eval, export,
│                                      # predict — see features/README.md
│
├── configs/                           # Non-authoritative shared infra
│   ├── _shared/                       # Pipeline templates (01–04, HPO, export)
│   └── _test/                         # CI smoke test fixtures
│
├── scripts/
│   └── new_feature.sh                 # Scaffold a new feature from _TEMPLATE
│
├── services/                          # Docker microservices
│   ├── s18100_sam3_service/           # SAM3 segmentation :18100 (GPU)
│   ├── s18101_flux_nim/               # Flux NIM diffusion :18101 (GPU)
│   ├── s18102_image_editor/           # Image editor orchestrator :18102 (CPU)
│   ├── s18103_label_studio/           # Label Studio :18103
│   ├── s18104_auto_label/             # Auto-labeling :18104 (CPU, needs SAM3)
│   ├── s18105_annotation_quality_assessment/  # Annotation QA :18105 (CPU, needs SAM3)
│   └── s18106_sam3_1_service/         # SAM3.1 drop-in :18106 (GPU)
│
├── tests/                             # Integration test suite — see tests/CLAUDE.md
├── dataset_store/                     # Training-ready datasets (YOLO format, gitignored)
├── pretrained/                        # Model weights (gitignored)
├── outputs/test_raw_pipeline/         # CI pipeline fixture (gitignored)
└── docs/                              # Documentation
```

## Requirements

- Python >= 3.12, [uv](https://docs.astral.sh/uv/) package manager
- CUDA-capable GPU (recommended for training; services require GPU for SAM3/Flux)

## Quick Start

```bash
# Install — bare sync installs the full p00→p10 pipeline, demo UI,
# Jupyter, MediaPipe, Playwright, pytest/ruff/dvc. No extras needed.
uv sync

# Optional: isolated venv for ONNX INT8 quantization (optimum +
# onnxruntime-quantize conflict with transformers@git; kept out of the
# main env on purpose).
bash scripts/setup-export-venv.sh     # creates .venv-export/

# Optional: heavyweight analytics (fiftyone + cleanlab)
uv sync --extra analysis

# Download YOLOX-M pretrained weights
wget -P pretrained/ https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth

# Copy .env and fill in tokens
cp .env.example .env        # add HF_TOKEN, WANDB_API_KEY
```

### Naming contract

Two conventions you'll see everywhere — `scripts/new_feature.sh` derives
the mapping automatically, so you never edit it by hand:

| Concept | Convention | Example |
|---|---|---|
| Feature folder | `<category>-<name>` (kebab-hyphen) | `safety-fire_detection` |
| `dataset_name` in 05_data.yaml | snake_case (folder `-` → `_`) | `safety_fire_detection` |
| `dataset_store/training_ready/` subdir | = `dataset_name` | `training_ready/safety_fire_detection/` |

Categories: `access-`, `ppe-`, `safety-`, `traffic-` (aligned with
`docs/03_platform/`). Feature-level details: [`features/README.md`](features/README.md).

### End-to-end: labeled dataset → deployed ONNX

```bash
FEATURE=features/safety-fire_detection

# 1. Merge raw sources into training_ready (split-subdir YOLO layout)
uv run core/p00_data_prep/run.py --config $FEATURE/configs/00_data_preparation.yaml

# 2. Explore — class distribution, mean/std, pixel stats
uv run utils/exploration.py --config $FEATURE/configs/05_data.yaml

# 3. Train
uv run core/p06_training/train.py --config $FEATURE/configs/06_training.yaml

# 4. Evaluate on test split + error analysis (FP/FN categorization)
uv run core/p08_evaluation/evaluate.py \
  --model $FEATURE/runs/best.pt --config $FEATURE/configs/05_data.yaml --split test

# 5. Export to ONNX (float32; use .venv-export/ for INT8 quantization)
uv run core/p09_export/export.py \
  --model $FEATURE/runs/best.pt --training-config $FEATURE/configs/06_training.yaml

# 6. Interactive demo (multi-tab Gradio UI — per-feature tabs read
#    10_inference.yaml for alert thresholds, tracker, and sample videos)
uv run demo
```

### End-to-end: raw unlabeled images → deployed ONNX

```bash
FEATURE=features/safety-fire_detection

# 0. Bring up annotation services
cd services/s18100_sam3_service && docker compose up -d      # SAM3 :18100 (GPU)
cd services/s18103_label_studio && bash bootstrap.sh         # Label Studio :18103
cd services/s18104_auto_label && docker compose up -d        # auto-label :18104
cd services/s18105_annotation_quality_assessment && docker compose up -d  # QA :18105

for p in 18100 18103 18104 18105; do curl -s http://localhost:$p/health; done

# 1. Auto-annotate with SAM3 text prompts → YOLO labels (flat-dir mode)
uv run core/p01_auto_annotate/run_auto_annotate.py \
  --image-dir /path/to/unlabeled/images --classes fire,smoke --mode text

# 2. Merge into training_ready (80/10/10 stratified split)
uv run core/p00_data_prep/run.py --config $FEATURE/configs/00_data_preparation.yaml

# 3. Validate: structural + SAM3 verification + score 0–1 + fix suggestions
uv run core/p02_annotation_qa/run_qa.py --data-config $FEATURE/configs/05_data.yaml

# 4. (Optional) Human review in Label Studio with split-aware round-trip
uv run core/p04_label_studio/bridge.py setup  --data-config $FEATURE/configs/05_data.yaml
uv run core/p04_label_studio/bridge.py import --data-config $FEATURE/configs/05_data.yaml
#    … reviewers edit labels + split assignment at http://localhost:18103 …
uv run core/p04_label_studio/bridge.py export --data-config $FEATURE/configs/05_data.yaml

# 5. (Optional) Generative augmentation — SAM3 + Flux inpainting
uv run core/p03_generative_aug/run_generative_augment.py \
  --config $FEATURE/configs/03_generative_augment.yaml

# 6. Re-QA, then train / eval / export / infer as above.
```

### Override any parameter from CLI

```bash
uv run core/p06_training/train.py \
  --config features/safety-fire_detection/configs/06_training.yaml \
  --override training.lr=0.005 training.epochs=100
```

## Adding a New Detection Task

**Zero code changes required.** Create 2 YAML configs:

```bash
uv run utils/scaffold.py vehicle_detection \
  --model yolox-m --classes "0:car,1:truck,2:bus"
# Creates features/detect_vehicle/configs/05_data.yaml + 06_training.yaml
#         features/detect_vehicle/code/train.py, evaluate.py, export.py, inference.py
```

Then edit `features/detect_vehicle/configs/05_data.yaml` to point at your dataset:

```yaml
dataset_name: vehicle_detection
path: ../../dataset_store/vehicle_detection
names: { 0: car, 1: truck, 2: bus }
num_classes: 3
input_size: [640, 640]
```

```bash
uv run core/p06_training/train.py --config features/detect_vehicle/configs/06_training.yaml
```

### Extension Difficulty

| What to add | Effort | Changes |
|-------------|--------|---------|
| New detection task (new classes) | **0 code** | 2 YAML configs |
| New HuggingFace model | **1 line** | Add to `HF_MODEL_REGISTRY` |
| Custom model architecture | **~100–300 lines** | 1 file implementing `DetectionModel` ABC |
| New pose model | **~50–150 lines** | 1 file implementing `PoseModel` ABC |

## Bringing Your Own Code

`core/` is optional. Three levels of independence:

### Level 1 — Custom trainer, keep the CLI

```yaml
# features/detect_vehicle/configs/06_training.yaml
training:
  backend: custom
  custom_trainer_class: experiments.vehicle_detection.my_trainer.MyTrainer
```

```python
class MyTrainer:
    def __init__(self, config_path: str, overrides=None): ...
    def train(self) -> dict: ...
```

### Level 2 — Custom model, keep core's trainer

```python
from core.p06_models.registry import register_model
from core.p06_models.base import DetectionModel

@register_model("my-detector")
class MyDetector(DetectionModel):
    def forward(self, x): ...
    @property
    def output_format(self): return "yolox"
    @property
    def strides(self): return [8, 16, 32]
```

### Level 3 — Fully standalone

Replace `features/<name>/code/train.py` with any framework. `utils/config.py` has no PyTorch/HF dependency — safe to import from any environment.

### Choosing the Right Level

| Situation | Level |
|-----------|-------|
| Supported model (YOLOX, D-FINE, RT-DETRv2, timm, HF) | Config only |
| Custom architecture, standard detection training | Level 2 — `@register_model` |
| Custom loss / training strategy / multi-GPU | Level 1 — `backend: custom` |
| Wrapping MMDetection / Detectron2 / Lightning | Level 1 or Level 3 |
| Completely different framework | Level 3 — fully standalone |

## Configuration

All parameters from YAML — no hardcoded values.

| Config | Location | Purpose |
|--------|----------|---------|
| Data | `features/<name>/configs/05_data.yaml` | Dataset paths, class names, normalization |
| Training | `features/<name>/configs/06_training.yaml` | Model arch, optimizer, scheduler, augmentation |
| Inference + alerts | `features/<name>/configs/10_inference.yaml` | Per-feature alert thresholds, tracker, samples |
| Auto-annotate | `configs/_shared/01_auto_annotate.yaml` | Mode, output format, NMS, service URL |
| QA | `configs/_shared/02_annotation_quality.yaml` | Sampling, SAM3 thresholds, scoring |
| Export | `configs/_shared/09_export.yaml` | ONNX opset, output dir |
| HPO | `configs/_shared/08_hyperparameter_tuning.yaml` | Optuna search space, pruning |
| Face | `features/access-face_recognition/configs/face.yaml` | SCRFD/MobileFaceNet, gallery, threshold |
| Demo | `app_demo/config/config.yaml` | Annotators, tracker, Gradio server |

Full schema reference: [`configs/CLAUDE.md`](configs/CLAUDE.md).

## More Commands

The Quick Start blocks above cover the primary workflow. A few additional
commands that don't fit there:

```bash
# Hyperparameter tuning (Optuna)
uv run core/p07_hpo/run_hpo.py --config $FEATURE/configs/06_training.yaml

# Resume from checkpoint
uv run core/p06_training/train.py --config $FEATURE/configs/06_training.yaml \
  --resume $FEATURE/runs/last.pth

# INT8 quantization (requires .venv-export/)
source .venv-export/bin/activate
python core/p09_export/export.py --model $FEATURE/runs/best.pt \
  --training-config $FEATURE/configs/06_training.yaml --quantize

# Verify environment
uv run tests/run_all.py
```

**WandB**: auto-logs to `smart-camera` if `WANDB_API_KEY` is set. Disable
with `--override logging.wandb_project=null`.

### Multi-Person Development

Each developer works in their own `features/<name>/` directory — zero
file conflicts. `core/` is shared and model-agnostic. New models extend
via `@register_model`; no editing existing files. Changes to `core/`
must pass `uv run tests/run_all.py`.

## Testing

```bash
# Full pipeline integration tests (33 files, sequential)
uv run tests/run_all.py

# Single test file
uv run tests/test_p06_training.py

# Via pytest
uv run -m pytest tests/test_p06_training.py -v
```

Tests use real data only (no mocks). Services (SAM3 :18100, QA :18105, auto-label :18104) skip gracefully when down. See [`tests/CLAUDE.md`](tests/CLAUDE.md) for the full test map.

## Design Principles

1. **Config-driven** — all values from YAML, override from CLI
2. **Model-agnostic** — registry dispatch for models, losses, postprocessors
3. **Edge-first** — optimized for INT8 quantization on NPU chips
4. **Zero licensing cost** — Apache 2.0 / MIT only, AGPL-3.0 prohibited
5. **Self-contained** — no external project imports
6. **Service-oriented** — Docker microservices for annotation tooling

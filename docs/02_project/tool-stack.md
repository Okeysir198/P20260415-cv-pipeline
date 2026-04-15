# Tool Stack — Factory Smart Camera AI Development
## Complete MLOps & Development Environment

---

## Stack Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     AI MODEL DEVELOPMENT STACK                   │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ DATA LAYER   │  │ TRAIN LAYER  │  (+ HF Transformers at DG3)  │ ANALYSIS LAYER       │   │
│  │              │  │              │  │                      │   │
│  │ Label Studio │  │ Ultralytics  │  │ FiftyOne             │   │
│  │ DVC          │  │ W&B          │  │ Cleanlab             │   │
│  │ Roboflow*    │  │ Local GPU    │  │ Supervision          │   │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘   │
│         │                 │                      │               │
│         └────────────┬────┴──────────────────────┘               │
│                      │                                           │
│              ┌───────▼───────┐                                   │
│              │ EXPORT LAYER  │                                   │
│              │               │                                   │
│              │ ONNX Runtime  │                                   │
│              │ Hailo DFC*    │                                   │
│              └───────┬───────┘                                   │
│                      │                                           │
│              ┌───────▼───────┐                                   │
│              │   HANDOFF     │                                   │
│              │               │                                   │
│              │ .onnx models  │ → Deployment team                 │
│              │ Alert logic   │                                   │
│              │ Model cards   │                                   │
│              └───────────────┘                                   │
└─────────────────────────────────────────────────────────────────┘

* Roboflow Universe = dataset download only (free)
* Hailo DFC = deployment team's responsibility
```

---

## 1. Data Layer

### 1.1 Label Studio — Annotation

| | Detail |
|---|---|
| **Role** | Image annotation for object detection & pose |
| **Why** | Free, self-hosted, offline, team collaboration, SAM 3 + RF-DETR pre-annotation |
| **License** | Apache 2.0 (open source) |
| **Cost** | Free |
| **Alternatives considered** | Roboflow (paid for teams), CVAT (heavier setup), X-AnyLabeling (no team features) |

```bash
# Install
uv tool install label-studio
label-studio start --port 8080

# Or run without global install
uvx label-studio start --port 8080
```

**Used for:**
- PPE bounding box annotation (9 classes)
- Forklift/vehicle annotation (3 classes)
- Fire/smoke annotation (2 classes)
- Pose keypoint annotation (17 COCO keypoints)
- QA review workflow (reviewer approves/rejects)

**Integrations:**
- SAM 3 ML backend for open-vocabulary auto-annotation
- RF-DETR-L for detection pre-annotation (SOTA bbox accuracy)
- MoveNet Huge for keypoint pre-annotation (pose models)
- Export to YOLO format (detection) and COCO JSON (pose)
- Connects to cloud storage (S3, MinIO, GCS)

**Documentation:** See `05_labeling_guide.md`

---

### 1.1.1 Pre-Annotation Model Stack

Three models used offline to generate high-quality pre-annotations before human review:

| Tool | Role | Model | License | Install |
|---|---|---|---|---|
| **RF-DETR-L** | Object detection pre-annotation (bboxes) | DINOv2 backbone, COCO AP ~58 | Apache 2.0 | `uv pip install rfdetr` |
| **SAM 3** | Open-vocabulary segmentation (text-prompted) | Meta ICLR 2026, 4M+ concepts | Apache 2.0 | `uv pip install sam3` |
| **MoveNet Huge** | Keypoint pre-annotation (17 COCO keypoints) | ViT + MoE, COCO pose record | Apache 2.0 | `uv pip install easy-vitpose` |

**Why this stack?**
- **RF-DETR-L** beats YOLO11-x by 1.8 AP on COCO — best offline bbox accuracy (Apache 2.0 up to Large)
- **SAM 3** replaces SAM 2: type "helmet" → finds ALL helmets (SAM 2 required per-object clicking)
- **MoveNet Huge** is far more accurate than YOLO-Pose for annotation — speed doesn't matter offline

```bash
# Install pre-annotation stack
uv pip install rfdetr sam3 easy-vitpose

# Pre-annotate a batch of images:
# 1. RF-DETR: detect objects → export as YOLO bbox labels
# 2. SAM 3: text prompt "person", "helmet" → segment instances → convert to bbox
# 3. MoveNet: detect person bboxes → predict 17 keypoints → export COCO pose format
# 4. Import into Label Studio → human review & correct
```

> **Note:** RF-DETR XL/2XL uses PML 1.0 license (not Apache 2.0) — stay with Nano-Large for commercial use.

---

### 1.2 DVC — Dataset Versioning

| | Detail |
|---|---|
| **Role** | Version control for large datasets (git for data) |
| **Why** | Track dataset changes over time, reproduce any training run |
| **License** | Apache 2.0 (open source) |
| **Cost** | Free |
| **Alternatives considered** | Roboflow (vendor lock-in), Git LFS (limited), LakeFS (overkill) |

```bash
# Install
uv pip install dvc dvc-s3  # or dvc-gdrive, dvc-ssh

# Initialize
cd smart_camera && dvc init

# Track a dataset
dvc add dataset_store/ppe_v1/
git add dataset_store/ppe_v1.dvc .gitignore
git commit -m "dataset: PPE v1 - 8,500 images"
```

**Used for:**
- Versioning annotated datasets (ppe_v1, ppe_v2, ppe_v2-cleaned)
- Linking dataset version to git commit (full reproducibility)
- Team data sharing without uploading to git
- Rollback to previous dataset versions

**Storage backends:**
- Local NAS / shared drive (simplest)
- MinIO (self-hosted S3, recommended for team)
- Google Drive (free, good for small teams)
- AWS S3 / Azure Blob (enterprise)

---

### 1.3 Roboflow Universe — Public Dataset Download

| | Detail |
|---|---|
| **Role** | Download pre-annotated open datasets |
| **Why** | Fastest way to get PPE, fire, forklift datasets in YOLO format |
| **License** | Mixed (per dataset) |
| **Cost** | Free (public datasets, no account required) |
| **Note** | We do NOT use Roboflow for annotation or training — Label Studio + Ultralytics replace those |

**Datasets to download:**
- PPE detection (100K+ images)
- Fire/smoke (FASDD, DFS, Roboflow Fire)
- Forklift detection
- Industrial safety

---

## 2. Training Layer

### 2.1 YOLOX Official — Model Training & Export (Apache 2.0)

| | Detail |
|---|---|
| **Role** | Train YOLOX models, evaluate, export to ONNX |
| **Why** | Apache 2.0 license — **FREE commercial use**, no licensing fees |
| **License** | Apache 2.0 (permissive, commercial-friendly) |
| **Cost** | **$0** — Free forever, no license fees |
| **Alternatives considered** | HF Transformers (used as DG3 escalation, see 2.1b), Ultralytics (AGPL-3.0 — requires $5K/yr Enterprise License) |

```bash
# Clone YOLOX repo
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip install -r requirements.txt
```

```bash
# Train YOLOX-M on custom dataset
python tools/train.py -f exps/default/yolox_m.py -d 8 -b 64 --fp16 -o

# Export to ONNX
python tools/export_onnx.py -n yolox-m -c weights/best.pth --output-name model.onnx
```

**Used for:**
- Training all 6 detection models (Fire, Helmet, Shoes, Fall Classify, Poketenashi, Person)
- Transfer learning from COCO pretrained weights
- ONNX export for edge deployment
- ByteTrack tracking integration (zone intrusion use case)

**Models trained with YOLOX:**

| Model | Architecture | Input | License |
|---|---|---|---|
| Fire Detection (a) | YOLOX-M | 640/1280 | Apache 2.0 (FREE) |
| Helmet Detection (b) | YOLOX-M | 640/1280 | Apache 2.0 (FREE) |
| Safety Shoes (f) | YOLOX-M | 640/1280 | Apache 2.0 (FREE) |
| Fall Pose (g-pose) | MoveNet | 640 | Apache 2.0 (FREE) |
| Fall Classify (g-classify) | YOLOX-M | 640 | Apache 2.0 (FREE) |
| Poketenashi Phone (h) | YOLOX-M | 640 | Apache 2.0 (FREE) |
| Zone Intrusion (i) | YOLOX-T (pretrained) | 640 | Apache 2.0 (FREE) |

---

### 2.1b HuggingFace Transformers — DG3 Escalation Framework

| | Detail |
|---|---|
| **Role** | Train RT-DETRv2-S or D-FINE-S if YOLO underperforms at Decision Gate 3 |
| **Why** | Native support for `RTDetrV2ForObjectDetection` and `DFineForObjectDetection` — Apache 2.0 models with HGNetV2 CNN backbone (NPU-friendly) |
| **License** | Apache 2.0 (open source) |
| **Cost** | Free |
| **When to install** | Only at DG3 (end of W6) — if any model mAP < 0.75 at 1280px after v3 |

```bash
# Install only when DG3 is triggered — NOT part of default setup
uv pip install transformers[torch] datasets accelerate
```

```python
from transformers import RTDetrV2ForObjectDetection, RTDetrV2ImageProcessor
from transformers import Trainer, TrainingArguments

# Load pretrained RT-DETRv2-S
model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetrv2_r18vd")
processor = RTDetrV2ImageProcessor.from_pretrained("PekingU/rtdetrv2_r18vd")

# Fine-tune with Trainer API
training_args = TrainingArguments(
    output_dir="runs/rtdetrv2_fire",
    num_train_epochs=50,
    per_device_train_batch_size=8,
    learning_rate=1e-4,
    fp16=True,
)
trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds)
trainer.train()
```

**Supported models:**

| Model | HF Class | COCO AP | License | Added to HF |
|---|---|---|---|---|
| **RT-DETRv2-S** | `RTDetrV2ForObjectDetection` | ~48 | Apache 2.0 | Feb 2025 |
| **D-FINE-S** | `DFineForObjectDetection` | ~48.5 | Apache 2.0 | Apr 2025 |

**Key differences from Ultralytics:**
- Dataset format: COCO JSON (convert from YOLO via `scripts/data/`)
- Data augmentation: `torchvision.transforms.v2` (GPU-accelerated, see Section 2.1c)
- No built-in W&B integration — use `transformers` W&B callback or manual logging
- Export to ONNX: `model.export(format="onnx")` or `torch.onnx.export()`

> **Note:** This is a fallback — most models should pass with Ultralytics YOLO. Buffer weeks (W10-12) are available for DG3 retraining.

---

### 2.1c PyTorch / torchvision — Native ML Framework

| | Detail |
|---|---|
| **Role** | GPU-accelerated data augmentation fallback + foundation for HF Transformers |
| **Why** | `torchvision.transforms.v2` runs augmentations on GPU tensors natively — eliminates CPU bottleneck when needed |
| **License** | BSD-3-Clause (open source) |
| **Cost** | Free |
| **Note** | Installed automatically with `ultralytics` — no separate install needed |

**Default augmentation (Ultralytics built-in, CPU):**
- All models use Ultralytics built-in augmentation by default
- Runs on CPU workers, pipelined with GPU training (`cache=True`, `workers=8`)
- Sufficient for all current dataset sizes

**Fallback augmentation (`torchvision.transforms.v2`, GPU):**
- Trigger: GPU utilization < 80% during training (check `nvidia-smi`) — CPU augmentation is the bottleneck
- Likely for: Fire (a) at 1280px, Safety Shoes (f) with heavy augmentation

```python
import torchvision.transforms.v2 as T
import torch

# GPU-accelerated augmentation pipeline
transform = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomResizedCrop(size=(640, 640), scale=(0.5, 1.0)),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.7, hue=0.015),
    T.RandomErasing(p=0.3),
])

# Apply on GPU — no CPU-GPU transfer overhead
image_gpu = image.to("cuda")
augmented = transform(image_gpu)  # runs entirely on GPU
```

**Also used for:**
- HF Transformers `Trainer` data pipeline (at DG3)
- Custom dataset classes if needed for non-YOLO training
- Image preprocessing for inference scripts

| Priority | Tool | Runs On | When |
|---|---|---|---|
| **Default** | Ultralytics built-in augmentation | CPU (pipelined, `workers=8`) | All models — sufficient for current dataset sizes |
| **Fallback** | `torchvision.transforms.v2` | GPU | If GPU util < 80% — CPU augmentation bottleneck |
| **DG3 only** | HF Transformers `Trainer` + `torchvision.transforms.v2` | GPU | RT-DETRv2-S / D-FINE-S training |

---

### 2.2 W&B (Weights & Biases) — Experiment Tracking

| | Detail |
|---|---|
| **Role** | Track training runs, compare experiments, model registry |
| **Why** | Native Ultralytics integration (zero config), best-in-class visualization |
| **License** | Proprietary (cloud service) |
| **Cost** | Free tier: 200 tracked hours/mo, unlimited projects, 100GB storage |
| **Alternatives considered** | MLflow (more setup, weaker YOLO integration), ClearML (more complex), Neptune (smaller community) |

```bash
# Install & login
uv pip install wandb
wandb login
```

**Used for:**
- Auto-logging all training metrics (loss, mAP, learning rate)
- Side-by-side run comparison (PPE v1 vs v2 vs v3)
- Sample prediction visualization (bboxes on real images)
- Hyperparameter tracking
- Model artifact storage (.pt files with metadata)
- Model registry (tag "production" versions for deployment team)
- Error analysis iteration tracking

**What W&B auto-captures from Ultralytics:**
- Training/validation loss curves
- mAP@0.5, mAP@0.5:0.95 per epoch
- Precision/Recall curves
- Sample predictions with bounding boxes
- Model weights as downloadable artifacts
- Hardware utilization (GPU memory, temperature)
- Full hyperparameter config

---

### 2.3 GPU Training Infrastructure

| | Detail |
|---|---|
| **Role** | GPU compute for model training |
| **Primary** | Local GPU (remote PC) — no cost, no upload latency, no session limits |
| **Overflow** | Google Colab Pro ($12/mo subscription + compute units) — for short jobs (< 6h) when local GPU is contended |
| **Alternatives considered** | RunPod (requires data upload), HuggingFace (limited free GPU) |

**Local GPU (Priority):**
- All long training jobs (> 6h): Fire 122K, Helmet 62K, COCO pretrain
- Schedule large jobs overnight to avoid contention between 3 engineers
- No session limits, no disconnect risk

**Google Colab Pro (Overflow, $12/mo + compute units):**
- Short jobs (< 6h): Shoes, Poketenashi, g-classify, g-pose finetune
- GPU: T4/V100/A100 (varies by availability)
- **Cost model:** $12/mo subscription gives 100 compute units. Additional units cost extra. A100 consumes units ~6x faster than T4 — budget accordingly.
- Session limit: ~24h, idle timeout: ~90 min
- **Always use checkpointing** to survive disconnects:
  ```bash
  # Resume from last checkpoint after disconnect:
  yolo detect train resume=True model=runs/detect/train/weights/last.pt
  ```

**Workflow:**
1. Train on local remote PC (priority — all long jobs)
2. If local GPU is occupied, use Colab Pro for jobs < 6h
3. Always save checkpoints; use `resume=True` on disconnect
4. Download trained weights (.pt) to local repo
5. Log all runs to W&B regardless of compute source

---

## 3. Analysis Layer

### 3.1 FiftyOne — Error Analysis & Model Debugging

| | Detail |
|---|---|
| **Role** | Visual debugging of model predictions, failure analysis |
| **Why** | Purpose-built for CV error analysis, interactive browser UI, Ultralytics compatible |
| **License** | Apache 2.0 (open source) |
| **Cost** | Free |
| **Alternatives considered** | Manual matplotlib scripts (slow, not interactive), W&B Tables (limited detection support) |

```bash
# Install
uv pip install fiftyone

# Launch UI
fiftyone app launch
# → http://localhost:5151
```

**Used for:**
- Per-class mAP evaluation
- False positive analysis (what is model hallucinating?)
- False negative analysis (what is model missing?)
- Object size analysis (small goggles/gloves failing?)
- Confusion matrix (which classes are confused?)
- Hardest image identification (for hard-example mining)
- Confidence threshold tuning (per-class optimal thresholds)
- Side-by-side ground truth vs. prediction visualization

**Documentation:** See `06_error_analysis_guide.md`

---

### 3.2 Cleanlab — Dataset Quality Audit

| | Detail |
|---|---|
| **Role** | Automatically find annotation mistakes in training data |
| **Why** | Label errors silently degrade model accuracy; Cleanlab finds them automatically |
| **License** | AGPL-3.0 (open source) |
| **Cost** | Free |
| **Alternatives considered** | Manual review (slow, misses subtle errors), FiftyOne mistakenness (less sophisticated) |

```bash
# Install
uv pip install cleanlab
```

**Used for:**
- Find wrong class labels (helmet mislabeled as no_helmet)
- Find missing annotations (unlabeled objects)
- Rank annotations by likelihood of being wrong
- Prioritize review effort (fix worst errors first)
- Typically finds 2–5% label errors → fixing gives 1–3% free mAP boost

---

### 3.3 Supervision — Visualization & Alert Logic

| | Detail |
|---|---|
| **Role** | Lightweight detection visualization, zone logic, tracking utilities |
| **Why** | Built-in zone polygon tools, clean bbox/mask visualization, ByteTrack integration |
| **License** | MIT (open source) |
| **Cost** | Free |
| **Alternatives considered** | Raw OpenCV drawing (verbose), custom code (reinventing the wheel) |

```bash
# Install
uv pip install supervision
```

**Used for:**
- Zone intrusion polygon definition & checking
- Bounding box / mask visualization for debugging
- Tracking visualization (person ID annotations)
- Alert logic utilities (centroid distance, IoU calculation)
- FPS counter and performance benchmarking

---

## 4. Export Layer

### 4.1 ONNX Runtime — Model Validation

| | Detail |
|---|---|
| **Role** | Validate exported ONNX models before handoff |
| **Why** | Ensure ONNX export is correct, test INT8 quantization accuracy |
| **License** | MIT (open source) |
| **Cost** | Free |

```bash
# Install
uv pip install onnxruntime onnx
```

**Used for:**
- Validate .onnx model loads and runs correctly
- Compare ONNX output vs. PyTorch output (should be identical)
- Test INT8 quantized ONNX accuracy (must be <3% mAP drop)
- Benchmark inference speed on CPU (sanity check)

---

## 5. Infrastructure & Utilities

### 5.1 Python Environment

| | Detail |
|---|---|
| **Python** | 3.11+ |
| **Package manager** | **uv** (fast, reliable — replaces pip/pip-tools/virtualenv) |
| **Virtual env** | uv venv per project |

> **Why uv?** 10-100x faster than pip for dependency resolution and installs. Drop-in replacement: use `uv pip install` everywhere instead of `pip install`. Also handles venv creation (`uv venv`), tool installation (`uv tool install`), and running tools without install (`uvx`). See [docs.astral.sh/uv](https://docs.astral.sh/uv/).

```bash
# Project setup
uv venv .venv
source .venv/bin/activate
uv pip install ultralytics wandb dvc fiftyone cleanlab supervision onnxruntime label-studio
```

### 5.2 OpenCV — Image Processing

| | Detail |
|---|---|
| **Role** | Image/video I/O, preprocessing, augmentation |
| **License** | Apache 2.0 |
| **Cost** | Free |

```bash
# Included with ultralytics, but can install separately
uv pip install opencv-python-headless
```

### 5.3 FFmpeg — Video Frame Extraction

| | Detail |
|---|---|
| **Role** | Extract frames from factory RTSP recordings |
| **License** | LGPL / GPL |
| **Cost** | Free |

```bash
# Install (system-level)
sudo apt install ffmpeg

# Extract 1 frame/sec from factory footage
ffmpeg -i factory_cam01.mp4 -vf "fps=1" -q:v 2 frames/cam01_%06d.jpg
```

### 5.4 Git — Version Control

| | Detail |
|---|---|
| **Role** | Code versioning, works with DVC for data versioning |
| **Hosting** | GitHub / GitLab / self-hosted |

---

## 6. One-Command Install

```bash
# Create project environment
uv venv .venv
source .venv/bin/activate

# Install all tools
uv pip install \
  ultralytics \
  wandb \
  dvc \
  fiftyone \
  cleanlab \
  supervision \
  onnxruntime \
  onnx \
  label-studio

# System dependencies
sudo apt install ffmpeg

# DG3 escalation only (install when needed, not Day 1):
# uv pip install transformers[torch] datasets accelerate

# Login to services
wandb login

# Initialize versioning
git init
dvc init
```

---

## 7. Tool Interaction Map

```
                    ┌─────────────────┐
                    │  Factory Cameras │
                    │  (RTSP footage)  │
                    └────────┬────────┘
                             │ ffmpeg (frame extraction)
                             ▼
                    ┌─────────────────┐
                    │  Label Studio   │──── SAM 3 + RF-DETR + MoveNet
                    │  (annotate)     │
                    └────────┬────────┘
                             │ export YOLO format
                             ▼
┌────────────┐      ┌─────────────────┐
│  Roboflow  │─────►│      DVC        │
│  Universe  │      │ (version data)  │
│ (download) │      └────────┬────────┘
└────────────┘               │ dataset yaml
                             ▼
                    ┌─────────────────┐      ┌─────────────┐
                    │  Ultralytics    │─────►│    W&B      │
                    │  (train YOLO)   │      │ (track runs)│
                    │  on Local GPU   │      └─────────────┘
                    └────────┬────────┘
                             │ if DG3: switch to
                    ┌────────▼────────┐
                    │ HF Transformers │
                    │ (RT-DETRv2 /    │
                    │  D-FINE) + tv2  │
                    └────────┬────────┘
                             │ .pt model
                             ▼
                    ┌─────────────────┐      ┌─────────────┐
                    │    FiftyOne     │      │  Cleanlab   │
                    │ (error analysis)│      │(label audit)│
                    └────────┬────────┘      └──────┬──────┘
                             │                      │
                             │  ◄── fix labels ─────┘
                             │      retrain if needed
                             ▼
                    ┌─────────────────┐
                    │  Ultralytics    │
                    │  (export ONNX)  │
                    └────────┬────────┘
                             │ .onnx model
                             ▼
                    ┌─────────────────┐
                    │  ONNX Runtime   │
                    │  (validate)     │
                    └────────┬────────┘
                             │ validated .onnx
                             ▼
                    ┌─────────────────┐
                    │    HANDOFF      │
                    │ to deployment   │
                    │ team            │
                    └─────────────────┘
```

---

## 8. Cost Summary

| Tool | Type | Cost |
|---|---|---|
| Label Studio | Annotation | Free |
| DVC | Dataset versioning | Free |
| Roboflow Universe | Dataset download | Free |
| Ultralytics | Training framework (default) | Free (AGPL) or ~$5K/yr (Enterprise) |
| HF Transformers | Training framework (DG3 escalation) | Free (Apache 2.0) |
| PyTorch / torchvision | GPU augmentation fallback | Free (BSD-3) |
| W&B | Experiment tracking | Free tier (200h/mo) |
| Local GPU | Training compute | Free (existing hardware) |
| Google Colab Pro | Overflow GPU | $12/mo subscription + compute units (A100 drains ~6x faster than T4) |
| FiftyOne | Error analysis | Free |
| Cleanlab | Dataset QA | Free |
| Supervision | Visualization/logic | Free |
| ONNX Runtime | Model validation | Free |
| OpenCV | Image processing | Free |
| Git + GitHub | Code versioning | Free |
| **Total** | | **~$12–24/mo** subscription + compute units (Colab overflow only) |

---

## 9. License Summary

| Tool | License | Commercial OK | Code Disclosure |
|---|---|---|---|
| Label Studio | Apache 2.0 | Yes | No |
| DVC | Apache 2.0 | Yes | No |
| **YOLOX Official** | **Apache 2.0** | **Yes** | **No** |
| HF Transformers | Apache 2.0 | Yes | No |
| PyTorch / torchvision | BSD-3-Clause | Yes | No |
| W&B | Proprietary (SaaS) | Yes | N/A |
| FiftyOne | Apache 2.0 | Yes | No |
| Cleanlab | AGPL-3.0 | Restricted | Required |
| Supervision | MIT | Yes | No |
| ONNX Runtime | MIT | Yes | No |
| OpenCV | Apache 2.0 | Yes | No |

> **Strategy: YOLOX First (Apache 2.0, FREE).** YOLOX is the default architecture for best Hailo-8/RK3588 NPU compatibility with **free commercial use** (Apache 2.0 license). No Enterprise License fees required. Transformer models (RT-DETRv2-S/D-FINE-S) are only used as DG3 accuracy fallback. Cleanlab AGPL is fine for internal tooling (not distributed to customers).

---

## 10. Minimum Viable Stack

If budget or setup time is constrained, this is the absolute minimum:

```bash
# Minimum install — covers train + evaluate + export
uv pip install ultralytics wandb

# That's it. Ultralytics handles everything from training to ONNX export.
# W&B auto-tracks all runs with zero config.
# Add other tools as needed.
```

| Priority | Tool | When to Add |
|---|---|---|
| **Must have** | Ultralytics + W&B | Day 1 |
| **Must have** | Label Studio | Day 1 (if annotating custom data) |
| **Should have** | DVC | Week 2 (when datasets start changing) |
| **Should have** | FiftyOne | Week 4 (after first training runs) |
| **Nice to have** | Cleanlab | Week 5 (before retraining) |
| **Nice to have** | Supervision | Week 6 (alert logic development) |
| **DG3 only** | HF Transformers + datasets + accelerate | Week 6+ (only if YOLO fails at DG3) |

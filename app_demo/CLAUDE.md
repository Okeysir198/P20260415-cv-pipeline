# CLAUDE.md -- app_demo/

Flexible, config-driven Gradio demo for any computer vision project. No hardcoded class names — all loaded from YAML configs. Supports generic detection, safety use cases, and video analytics.

## Architecture

```
app_demo/
    config/                     # All YAML configs (self-contained)
        config.yaml            # Main demo config (models, tabs, use_cases)
        coco_names.yaml        # COCO 80 class names
    src/                        # Generic demo components (reusable)
        class_filter.py        # ClassFilterComponent: checkbox filtering
        metrics_display.py     # MetricsDisplay: per-class metrics
        model_loader.py        # ModelLoader: file browser + dropdown
        generic_tab.py         # Generic Detection tab (model agnostic)
    tabs/                       # Tab implementations
        tab_stream.py          # Live camera streaming
        tab_detection.py       # Generic Object Detection (COCO)
        tab_fire.py            # Fire & Smoke Detection
        tab_ppe.py             # PPE Compliance (Helmet + Shoes)
        tab_fall.py            # Fall Detection (Classify + Pose)
        tab_phone.py           # Phone Detection
        tab_zone.py            # Zone Intrusion
        tab_face.py            # Face Recognition (Enroll + Identify)
        tab_analytics.py       # Video Analytics Dashboard
    run.py                      # CLI entry point
    app.py                      # create_app() factory
    model_manager.py            # ModelManager: lazy loading, caching, warmup
    utils.py                    # Shared helpers (color conv, annotation, badges)
```

## How It Works

### Startup Flow
1. `run.py` loads config from `app_demo/config/config.yaml`
2. `create_app()` creates `ModelManager` and calls `manager.warmup()`
3. `warmup()` preloads all COCO pretrained models + face ONNX models to GPU
4. Tab builders are loaded dynamically from the `tabs` list in config (via `importlib`)
5. `app.launch()` starts the server with Citrus theme

### Config System — No Hardcoding

**All class names come from YAML configs, not hardcoded in code.**

- `config/coco_names.yaml` — COCO 80 class names (externalized, not in config.yaml)
- `config/config.yaml` → `coco_names_config` — path to class names file
- `config/config.yaml` → `use_cases.<use_case>.alert_classes` — dynamic alert classes
- `config/config.yaml` → `use_cases.<use_case>.violation_classes` — dynamic violation classes
- `config/config.yaml` → `use_cases.<use_case>.compliance_classes` — dynamic compliance classes

Tab files load classes dynamically:
```python
# Old way (hardcoded — REMOVED):
_FIRE_ALERT_CLASSES = {"fire", "smoke"}

# New way (config-driven):
alert_classes = set(config.get("use_cases", {}).get("fire", {}).get("alert_classes", ["fire", "smoke"]))
```

### Model Loading
`ModelManager` handles all model lifecycle. **All model specs come from `config/config.yaml`**:

- **COCO pretrained** — `models.coco_pretrained` (name, path, normalize flag, hf_config)
- **Fine-tuned** — `models.fine_tuned.<use_case>.model_paths` (ordered list; first existing wins)
- **Face recognition** — SCRFD-500M + MobileFaceNet ONNX from `pretrained/`
- **Pose estimation** — RTMPose/MediaPipe loaded on first use

### Generic Components (src/)

#### ClassFilterComponent
Auto-generated checkbox group for class filtering:
```python
from app_demo.src.class_filter import ClassFilterComponent

filter_comp = ClassFilterComponent(predictor.class_names)
checkbox = filter_comp.build_ui(label="Filter Classes")
filtered_detections = filter_comp.filter_detections(detections, selected_names, name_to_id)
```

#### MetricsDisplay
Per-class precision/recall/F1 calculation and display:
```python
from app_demo.src.metrics_display import MetricsDisplay

metrics = MetricsDisplay.compute_from_detections(predictions, ground_truths, class_names)
table = MetricsDisplay.create_metrics_table(metrics)
chart = MetricsDisplay.create_metrics_chart(metrics)
```

#### ModelLoader
Generic model loading UI (file browser + pre-configured dropdown):
```python
from app_demo.src.model_loader import ModelLoader

loader = ModelLoader(config, manager)
model_ui = loader.build_model_selection_ui()
predictor = loader.load_predictor(model_path, num_classes, class_names)
```

#### Generic Detection Tab
Universal tab for any CV project:
- Model loading via file browser OR pre-configured dropdown
- Class filter checkboxes (auto-generated from model)
- Ground truth upload for metrics
- Image/video inference with class filtering
- Per-class metrics display (precision/recall/F1)
- Export results button

## Key Interfaces

### ModelManager
```python
manager = ModelManager(config)
manager.warmup()

predictor = manager.get_coco_predictor(0.25, "YOLOX-M")
predictor, type_ = manager.get_use_case_predictor("fire")
alert_cfg = manager.get_feature_alert_config("fire")          # loads features/safety-fire_detection/configs/10_inference.yaml
infer_cfg = manager.get_feature_inference_config("fire")      # full dict (alerts, tracker, samples)
predictor, type_ = manager.get_predictor_by_choice(dropdown_str, 0.25)

pose_pred = manager.get_pose_predictor("features/safety-fall_pose_estimation/configs/rtmpose_s.yaml")
face_pred = manager.get_face_predictor()
gallery = manager.get_face_gallery()

models = manager.list_available_models()
found = manager.discover_fine_tuned()
```

### Shared Utilities (utils.py)
```python
from app_demo.utils import (
    rgb_to_bgr, bgr_to_rgb,
    format_results_json,
    draw_keypoints,
    create_status_html,
    COCO_SKELETON_EDGES,
)
```

### Video Processing Pattern
```python
from core.p10_inference.video_inference import VideoProcessor

processor = VideoProcessor(
    predictor=predictor,
    alert_config=config.get("alerts"),
    enable_tracking=enable_tracking,
    tracker_config=config.get("tracker"),
)
summary = processor.process_video(video_path, output_path)
```

## Quick Start

### Launch the Demo

```bash
# Install dependencies (first time)
uv sync --extra demo --extra face --extra train

# Launch demo on default port 7861
uv run app_demo/run.py

# Launch with public sharing
uv run app_demo/run.py --share

# Custom port
uv run app_demo/run.py --server-port 8080
```

### Common Workflows

**1. Use Generic Detection Tab with Custom Model**
- Click "Generic Detection" tab
- Upload model file (.pt or .onnx) OR select from pre-configured dropdown
- Enter number of classes and class names
- Upload test image/video
- Adjust confidence threshold
- View results with class filtering and per-class metrics

**2. Add a New Pretrained Model**

Edit `app_demo/config/config.yaml`:
```yaml
models:
  coco_pretrained:
    My-Custom-Model:
      model_path: pretrained/my_custom_model.pth
      normalize: false  # true=ImageNet, false=0-255 (YOLOX)
```

Restart demo to see model in dropdown.

**3. Add Alert Classes for Use Case**

Edit `app_demo/config/config.yaml`:
```yaml
use_cases:
  my_use_case:
    alert_classes:
      - my_alert_class
      - another_alert
```

Tab will auto-detect these classes for alert triggering.

**4. Use Fine-Tuned Model**

Place model in `release/<use_case>/best.pt` or `runs/<use_case>/best.pt`. Demo auto-discovers it on startup.

**5. Debug Detection Issues**

```bash
# Check which models are available
# Open browser console, look for "Warming up models..." log messages

# Verify config loading
uv run -c "
from utils.config import load_config
config = load_config('app_demo/config/config.yaml')
print(config.get('models', {}).get('coco_pretrained', {}).keys())
"

# Test model loading directly
uv run -c "
from app_demo.model_manager import ModelManager
from utils.config import load_config
config = load_config('app_demo/config/config.yaml')
manager = ModelManager(config)
predictor = manager.get_coco_predictor(0.25, 'YOLOX-M')
print('Model loaded:', predictor is not None)
"
```

## Config (config/config.yaml)

Key sections:
- `models.coco_pretrained` — COCO pretrained model specs (path, normalize, hf_config)
- `models.fine_tuned` — Per-use-case model paths (searched in order)
- `tabs` — Ordered list of `{id, builder}` entries for dynamic tab loading
- `coco_names_config` — Path to COCO class names (relative to config/)
- `use_cases` — Per-use-case data config paths + alert/violation/compliance classes
- `face` — Face recognition config + gallery path
- `supervision` — Annotator settings (thickness, text scale, etc.)
- `tracker` — ByteTrack params
- `alerts` — Per-class confidence thresholds and frame windows
- `gradio` — Server settings (host, port, title)

## Commands

```bash
# Install dependencies
uv sync --extra demo --extra face --extra train

# Launch demo
uv run app_demo/run.py
uv run app_demo/run.py --share
uv run app_demo/run.py --server-port 8080

# Custom config
uv run app_demo/run.py --config app_demo/config/config.yaml
```

## Adding a New Tab

1. Create `app_demo/tabs/tab_<name>.py`
2. Define `build_tab_<name>(manager: Any, config: dict) -> None:`
3. Create `with gr.Tab("Tab Name"):` and build UI
4. Use `manager.get_use_case_predictor()` or `manager.get_coco_predictor()`
5. Add Image + Video sub-tabs with `VideoProcessor`
6. Add entry to `tabs` list in `config/config.yaml`

## Gotchas

- **⚠ Module-level state leaks across sessions** — `_zones` in `tab_zone.py` are shared across ALL Gradio sessions and are NOT thread-safe. One user's zone overwrites another's. For production, move to per-session storage (`gr.State`) or use locks.
- **YOLOX-M needs 0-255 input** — `ModelManager` sets `std=[1/255, 1/255, 1/255]` for raw pixel values
- **No hardcoded classes** — All class names loaded from config files
- **Config paths relative** — `coco_names_config: coco_names.yaml` is relative to `app_demo/config/`
- **Temp files not cleaned** — Video processing creates temp files that persist
- **HF model caching** — D-FINE-N and RT-DETRv2 auto-download, then cache to `pretrained/*.pt`

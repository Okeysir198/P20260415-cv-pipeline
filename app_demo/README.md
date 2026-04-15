# App Demo — Generic CV Detection Demo

Flexible Gradio demo for any computer vision project. Supports 10 tabs including generic detection, safety use cases, and analytics.

## Features

- **Generic Detection Tab** — Works with any model/dataset combo
- **Dynamic Class Filtering** — Auto-generated checkboxes from model classes
- **Per-Class Metrics** — Precision/recall/F1 with GT annotations
- **Config-Driven** — All class names from YAML, no hardcoded values
- **Multi-Model Support** — COCO pretrained + fine-tuned models
- **Video Processing** — Full video inference with tracking

## Run

```bash
uv sync --extra demo --extra face --extra train
uv run app_demo/run.py
```

## Tabs

| Tab | Use Case | Input |
|-----|----------|-------|
| Generic Detection | Any model/dataset (model agnostic) | Image, Video |
| Stream | Live camera streaming | Camera |
| Object Detection | Generic COCO 80-class | Image, Video |
| Fire Detection | Fire & smoke | Image, Video |
| PPE Compliance | Helmet + shoes | Image, Video |
| Fall Detection | Classification + pose | Image, Video |
| Phone Detection | Phone usage | Image, Video |
| Zone Intrusion | Restricted area | Image, Video |
| Face Recognition | Enroll + identify | Image, Video |
| Analytics | Video summary dashboard | Video |

## Structure

```
app_demo/
├── config/                    # All YAML configs
│   ├── config.yaml           # Main demo config (models, tabs, use_cases)
│   └── coco_names.yaml       # COCO 80 class names
├── src/                       # Generic demo components
│   ├── class_filter.py       # Checkbox-based class filtering
│   ├── metrics_display.py    # Per-class precision/recall/F1
│   ├── model_loader.py       # Generic model loading UI
│   └── generic_tab.py        # Generic Detection tab
├── tabs/                      # Tab implementations
│   ├── tab_detection.py      # Generic Object Detection
│   ├── tab_fire.py           # Fire & Smoke
│   ├── tab_ppe.py            # Helmet + Shoes (PPE)
│   ├── tab_fall.py           # Fall Detection
│   ├── tab_phone.py          # Phone Detection
│   ├── tab_zone.py           # Zone Intrusion
│   ├── tab_face.py           # Face Recognition
│   ├── tab_stream.py         # Live Camera Stream
│   └── tab_analytics.py      # Video Analytics
├── app.py                     # create_app() factory
├── run.py                     # CLI entry point
├── model_manager.py           # Model loading and caching
└── utils.py                   # Shared helpers
```

## Config

All demo configuration is in `app_demo/config/config.yaml`:

- `models.coco_pretrained` — COCO pretrained models (YOLOX-M, D-FINE-N, RT-DETRv2)
- `models.fine_tuned` — Fine-tuned model paths per use case
- `tabs` — Ordered tab list with builder functions
- `coco_names_config` — Path to COCO class names (relative to config/)
- `use_cases` — Per-use-case alert/violation/compliance classes
- `supervision` — Annotator settings
- `tracker` — ByteTrack params
- `alerts` — Per-class confidence thresholds
- `gradio` — Server settings

## Adding a New Tab

1. Create `app_demo/tabs/tab_<name>.py`
2. Define `build_tab_<name>(manager, config)` with `gr.Tab()` inside
3. Add entry to `tabs` list in `config/config.yaml`:
   ```yaml
   - id: my_tab
     builder: app_demo.tabs.tab_<name>.build_tab_<name>
   ```

## Adding a New Model

**COCO pretrained:**
```yaml
models:
  coco_pretrained:
    My-Model:
      model_path: pretrained/my_model.pth
      normalize: false  # true=ImageNet, false=0-255 (YOLOX)
```

**Fine-tuned:**
```yaml
models:
  fine_tuned:
    my_use_case:
      model_paths:
        - release/my_use_case/best.pt
        - runs/my_use_case/best.pt
```

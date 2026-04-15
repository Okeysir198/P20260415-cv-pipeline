# SAM3 Segmentation Service

REST API for SAM3 (Segment Anything Model 3) providing image segmentation and video tracking. Wraps three HuggingFace `facebook/sam3` model architectures in a FastAPI service with stateful video session support.

## Architecture

- **Port**: 18100
- **GPU**: 1x NVIDIA (~5GB VRAM for all 3 models)
- **Base**: ubuntu:24.04 + uv
- **Models**: 3 lazy-loaded singletons from `facebook/sam3`:
  - **Sam3Model** + Sam3Processor — text/image open-vocab segmentation (~2GB)
  - **Sam3TrackerVideoModel** + Sam3TrackerVideoProcessor — interactive point/box video tracking (~1.5GB bf16)
  - **Sam3VideoModel** + Sam3VideoProcessor — text-driven video detection + tracking (~1.5GB bf16)

## Endpoints

### Image (stateless)

| Method | Path | Model | Description |
|--------|------|-------|-------------|
| GET | `/health` | — | Health check — loaded models, device, dtype, active sessions |
| POST | `/segment_box` | Sam3TrackerVideoModel | Box-prompted segmentation `[x1,y1,x2,y2]` |
| POST | `/segment_text` | Sam3Model | Text-prompted open-vocab segmentation |
| POST | `/segment_text_batch` | Sam3Model | Batch text segmentation (up to 16 images) |
| POST | `/auto_mask` | Sam3Model | Segment everything (multi-prompt, `get_vision_features` optimized) |
| POST | `/auto_mask_batch` | Sam3Model | Batch auto-mask (up to 16 images) |

All image endpoints accept single request or `list[request]` for batch processing.

### Video Sessions (stateful)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/sessions` | Create tracker or text-driven video session |
| POST | `/sessions/{id}/frames` | Add frame to streaming tracker session |
| POST | `/sessions/{id}/prompts` | Add points/boxes/masks on a frame (tracker mode) |
| POST | `/sessions/{id}/propagate` | Propagate tracked objects through all frames |
| DELETE | `/sessions/{id}` | Delete session, free GPU memory |

### Request/Response Examples

**POST /segment_box**
```json
// Request
{"image": "<base64>", "box": [100, 200, 300, 400]}
// Response
{"result": {"mask": "<base64 PNG>", "bbox": {"x1": 98, "y1": 195, "x2": 305, "y2": 410}, "score": 0.95, "iou_score": 0.95, "area": 0.12}}
```

**POST /segment_text**
```json
// Request
{"image": "<base64>", "text": "helmet", "detection_threshold": 0.5, "mask_threshold": 0.5}
// Response
{"detections": [{"mask": "<base64>", "bbox": {...}, "score": 0.92, "area": 0.05}]}
```

**POST /auto_mask**
```json
// Request
{"image": "<base64>", "threshold": 0.2, "prompts": ["person. car. dog."]}
// Response
{"detections": [{"mask": "<base64>", "bbox": {...}, "score": 0.83, "area": 0.07}, ...]}
```

**POST /sessions** (tracker mode)
```json
// Request
{"mode": "tracker", "frames": ["<base64>", "<base64>", ...]}
// Response
{"session_id": "abc123", "mode": "tracker", "num_frames": 5, "width": 960, "height": 540}
```

**POST /sessions** (video mode — text-driven)
```json
// Request
{"mode": "video", "frames": ["<base64>", ...], "text": "person"}
// Response
{"session_id": "def456", "mode": "video", "num_frames": 5, "width": 960, "height": 540}
```

**POST /sessions/{id}/prompts**
```json
// Request
{"frame_idx": 0, "obj_ids": [1], "boxes": [[[100, 200, 300, 400]]]}
// Response
{"frame_idx": 0, "detections": [{"obj_id": 1, "mask": "<base64>", "bbox": {...}, "score": 1.0}]}
```

**POST /sessions/{id}/propagate**
```json
// Request
{"max_frames": 10}
// Response
{"frames": [{"frame_idx": 0, "detections": [{"obj_id": 1, "mask": "<base64>", ...}]}, ...]}
```

### Session Lifecycle Examples

**Interactive tracker (annotate video):**
```
POST /sessions           {mode: "tracker", frames: [f0, f1, ..., f99]}
POST /sessions/{id}/prompts  {frame_idx: 0, obj_ids: [1], boxes: [[[100,200,300,400]]]}
POST /sessions/{id}/propagate {}
DELETE /sessions/{id}
```

**Text-driven video detection:**
```
POST /sessions           {mode: "video", frames: [f0, ..., f99], text: "person"}
POST /sessions/{id}/propagate {}
DELETE /sessions/{id}
```

**Streaming tracker (webcam/RTSP):**
```
POST /sessions           {mode: "tracker"}
POST /sessions/{id}/frames   {frame: f0}
POST /sessions/{id}/prompts  {frame_idx: 0, obj_ids: [1], boxes: [[[x1,y1,x2,y2]]]}
POST /sessions/{id}/frames   {frame: f1}  → returns masks for tracked objects
POST /sessions/{id}/frames   {frame: f2}  → returns masks
...
DELETE /sessions/{id}
```

## Quick Start

```bash
# Docker
docker compose up -d
curl http://localhost:18100/health

# Local development
uv sync
uv run uvicorn app:app --host 0.0.0.0 --port 18100
```

## Gradio Demo

Interactive web UI for SAM3 segmentation, exposing all 5 segmentation modes as sub-tabs: text prompt, box prompt, auto mask, text-driven video tracking, and box-driven video tracking.

### Local Run

```bash
uv sync --extra demo
uv run python demo.py
# Open http://localhost:7860
```

### Docker

```bash
docker compose --profile demo up demo
# Open http://localhost:7860
```

The demo uses `profiles: [demo]` so it does not start with a plain `docker compose up`. Only starts when explicitly requested.

### Sub-tabs

| Tab | Mode | Description |
|-----|------|-------------|
| Text Prompt | Image (stateless) | Describe an object in natural language to segment it (e.g., "person", "fire") |
| Box Prompt | Image (stateless) | Draw a bounding box around an object to get a precise mask |
| Auto Mask | Image (stateless) | Segment everything in the image using multi-prompt detection |
| Text Video | Video (stateful) | Track an object across video frames using a text description |
| Box Video | Video (stateful) | Track an object across video frames using an initial bounding box |

### GPU Memory Note

Running the demo alongside the API service requires ~10 GB VRAM (each loads the same 3 models independently). For GPU-constrained environments, run the demo standalone -- it loads models lazily on first use via singletons, so only the models you interact with consume VRAM.

## Module Structure

```
s18100_sam3_service/
├── app.py                      # Thin entry point: imports app from src.routes
├── src/                        # Package with all service logic
│   ├── config.py               # YAML config, constants, logger
│   ├── schemas.py              # All Pydantic request/response models
│   ├── helpers.py              # Image decode, mask conversion utilities
│   ├── models.py               # 3 lazy-loaded model singletons + locks
│   ├── segmentation.py         # Stateless: segment_box, segment_text, segment_auto
│   ├── sessions.py             # Stateful: session store, init, propagate, CRUD
│   └── routes.py               # FastAPI app, 11 endpoint handlers
├── configs/default.yaml        # Runtime config (model, thresholds, session limits)
├── tests/                      # Integration tests (pytest + supervision)
│   ├── conftest.py             # Fixtures, skip_no_service, sv annotators
│   ├── data/                   # truck.jpg, cat.jpg, bedroom.mp4
│   ├── test00_health.py        # GET /health (3 tests)
│   ├── test01_segment_box.py   # POST /segment_box (5 tests)
│   ├── test02_segment_text.py  # POST /segment_text (5 tests)
│   ├── test03_auto_mask.py     # POST /auto_mask (4 tests)
│   ├── test04_tracker_session.py  # Tracker lifecycle + .mp4 output (6 tests)
│   ├── test05_video_session.py    # Video lifecycle + .mp4 output (4 tests)
│   ├── test06_error_and_edge_cases.py # Error handling + edge cases
│   ├── test07_batch_endpoints.py     # Batch text/auto_mask endpoints
│   └── outputs/                # Generated overlays (.png) and videos (.mp4)
├── Dockerfile
├── docker-compose.yaml
├── pyproject.toml
└── pytest.ini
```

## Configuration

`configs/default.yaml`:
```yaml
server:
  host: "0.0.0.0"
  port: 18100

model:
  name: "facebook/sam3"
  device: null              # null = auto-detect (cuda > mps > cpu)
  dtype: bfloat16

segmentation:
  mask_threshold: 0.5
  detection_threshold: 0.5
  auto_mask_threshold: 0.35
  auto_mask_nms_threshold: 0.5
  auto_mask_max_area: 0.95
  max_hole_area: 100.0
  max_sprinkle_area: 100.0
  auto_mask_prompts:        # split to stay under CLIP's 32-token limit
    - "person. car. truck. dog. cat. bird."
    - "chair. table. laptop. phone. bottle. cup."
    - "tree. building. road. sky. door. window."
    - "fire. smoke. helmet. shoe. bag. sign."

sessions:
  max_active: 10
  ttl_seconds: 3600

server:
  workers: 1
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes | HuggingFace token for model download |
| `NVIDIA_VISIBLE_DEVICES` | No | GPU visibility (default: all) |

## Tests

Integration tests covering all 11 endpoints. Tests require the service running at `localhost:18100` — they auto-skip if unreachable. Visualization uses [supervision](https://github.com/roboflow/supervision) for mask/bbox/label overlays. Video session tests output annotated `.mp4` files.

```bash
# Install test dependencies
uv sync --extra test

# Run all tests
uv run pytest tests/ -v

# Run individual test files
uv run pytest tests/test00_health.py -v
uv run pytest tests/test04_tracker_session.py -v
```

### Test Outputs

| File | Description |
|------|-------------|
| `tests/outputs/test01_segment_box.png` | Truck with box-prompted mask overlay |
| `tests/outputs/test02_segment_text.png` | Cat with text-prompted mask overlay |
| `tests/outputs/test03_auto_mask.png` | Cat with auto-mask segment-everything overlay |
| `tests/outputs/test04_tracker_session.mp4` | Bedroom video with box-prompted tracking overlay |
| `tests/outputs/test05_video_session.mp4` | Bedroom video with text-driven "bed" detection overlay |

### Endpoint Coverage

| # | Endpoint | Test File | Tests |
|---|----------|-----------|-------|
| 1 | `GET /health` | test00_health.py | 3 |
| 2 | `POST /segment_box` | test01_segment_box.py | 5 |
| 3 | `POST /segment_text` | test02_segment_text.py | 5 |
| 4 | `POST /segment_text_batch` | test07_batch_endpoints.py | — |
| 5 | `POST /auto_mask` | test03_auto_mask.py | 4 |
| 6 | `POST /auto_mask_batch` | test07_batch_endpoints.py | — |
| 7 | `POST /sessions` | test04 + test05 | 2 |
| 8 | `POST /sessions/{id}/frames` | test04_tracker_session.py | 1 |
| 9 | `POST /sessions/{id}/prompts` | test04_tracker_session.py | 1 |
| 10 | `POST /sessions/{id}/propagate` | test04 + test05 | 2 |
| 11 | `DELETE /sessions/{id}` | test04 + test05 | 2 |

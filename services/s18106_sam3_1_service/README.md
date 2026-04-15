# SAM 3.1 Segmentation Service

REST API for SAM 3.1 (Segment Anything Model 3.1) providing image segmentation and video tracking. Wraps three HuggingFace `facebook/sam3.1` model architectures in a FastAPI service with stateful video session support.

## What's New in SAM 3.1

**Object Multiplex** — shared-memory joint multi-object tracking. SAM 3.1 parallelizes inference across multiple objects using a single shared memory bank rather than one memory bank per object. Key improvements:

- **~7x faster** multi-object tracking inference at 128 objects on a single H100 GPU without accuracy loss
- **Improved VOS** (Video Object Segmentation) performance on 6 out of 7 benchmarks compared to SAM 3
- Scales efficiently with object count — throughput advantage grows as more objects are tracked simultaneously
- Same API and endpoint signatures as SAM 3 (s18100) — drop-in replacement at port 18106

## Architecture

- **Port**: 18106
- **GPU**: 1x NVIDIA (~5GB VRAM for all 3 models)
- **Base**: ubuntu:24.04 + uv
- **Models**: 3 lazy-loaded singletons from `facebook/sam3.1`:
  - **Sam3Model** + Sam3Processor — text/image open-vocab segmentation (~2GB)
  - **Sam3TrackerVideoModel** + Sam3TrackerVideoProcessor — interactive point/box video tracking with Object Multiplex (~1.5GB bf16)
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
# Set HF_TOKEN in .env
echo "HF_TOKEN=your_token_here" > .env

# Start service (Docker)
cd services/s18106_sam3_1_service && docker compose up -d
curl http://localhost:18106/health

# Local development
uv sync
uv run uvicorn app:app --host 0.0.0.0 --port 18106
```

## Module Structure

```
s18106_sam3_1_service/
├── app.py                      # Thin entry point: imports app from src.routes
├── src/                        # Package with all service logic
│   ├── config.py               # YAML config, constants, logger
│   ├── schemas.py              # All Pydantic request/response models
│   ├── helpers.py              # Image decode, mask conversion utilities
│   ├── models.py               # 3 lazy-loaded model singletons + locks
│   ├── segmentation.py         # Stateless: segment_box, segment_text, segment_auto
│   ├── sessions.py             # Stateful: session store, init, propagate, CRUD
│   └── routes.py               # FastAPI app, 11 endpoint handlers
├── configs/default.yaml        # Runtime config (model, port, thresholds, sessions)
├── tests/                      # Integration tests (pytest + supervision)
│   ├── conftest.py             # Fixtures, skip_no_service, sv annotators
│   ├── data/                   # truck.jpg, cat.jpg, bedroom.mp4
│   ├── test00_health.py        # GET /health
│   ├── test01_segment_box.py   # POST /segment_box
│   ├── test02_segment_text.py  # POST /segment_text
│   ├── test03_auto_mask.py     # POST /auto_mask
│   ├── test04_tracker_session.py  # Tracker lifecycle + .mp4 output
│   ├── test05_video_session.py    # Video lifecycle + .mp4 output
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
  port: 18106

model:
  name: "facebook/sam3.1"
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

Override config path via env: `SAM3_CONFIG=/path/to/config.yaml`

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes | HuggingFace token for model download |
| `NVIDIA_VISIBLE_DEVICES` | No | GPU visibility (default: all) |

## GPU Requirements

~5GB VRAM total for all three models loaded simultaneously:
- Sam3Model (text/image): ~2GB
- Sam3TrackerVideoModel (box tracking): ~1.5GB bf16
- Sam3VideoModel (text-driven video): ~1.5GB bf16

Object Multiplex in SAM 3.1 does not increase VRAM — the shared memory bank replaces per-object memory banks, keeping footprint flat while dramatically reducing compute for multi-object scenarios.

## Running Tests

Tests require the service running at `localhost:18106`. They auto-skip if unreachable.

```bash
# Install test dependencies
uv sync --extra test

# Run all tests
uv run pytest tests/ -v

# Run individual test files
uv run pytest tests/test00_health.py -v
uv run pytest tests/test04_tracker_session.py -v
uv run pytest tests/test07_batch_endpoints.py -v
```

### Test Outputs

| File | Description |
|------|-------------|
| `tests/outputs/test01_segment_box.png` | Truck with box-prompted mask overlay |
| `tests/outputs/test02_segment_text.png` | Cat with text-prompted mask overlay |
| `tests/outputs/test03_auto_mask.png` | Cat with auto-mask segment-everything overlay |
| `tests/outputs/test04_tracker_session.mp4` | Bedroom video with box-prompted tracking overlay |
| `tests/outputs/test05_video_session.mp4` | Bedroom video with text-driven "bed" detection overlay |

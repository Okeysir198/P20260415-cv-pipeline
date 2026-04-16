# CLAUDE.md — SAM3 Segmentation Service

Self-contained SAM3 segmentation REST API. Wraps three HuggingFace `facebook/sam3` model architectures (text/image, tracker, video) in a FastAPI service with stateful video session support. GPU required (~5GB VRAM).

## Architecture

Modular `src/` package with thin `app.py` entry point. All ML runs on-device (no external service calls).

```
Client → sam3_service (:18100, GPU)
         ├── Sam3Model           — text/image open-vocab segmentation
         ├── Sam3TrackerVideoModel — interactive box/point video tracking
         └── Sam3VideoModel      — text-driven video detection + tracking
```

## Endpoints

### Image (stateless)
| Method | Path | Model | Purpose |
|--------|------|-------|---------|
| POST | `/segment_box` | Sam3TrackerVideoModel | Box-prompted segmentation `[x1,y1,x2,y2]` |
| POST | `/segment_text` | Sam3Model | Text-prompted open-vocab segmentation |
| POST | `/auto_mask` | Sam3Model | Segment everything (multi-prompt) |

All image endpoints accept single request or `list[request]` for batch processing.

### Video Sessions (stateful)
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/sessions` | Create tracker or text-driven video session |
| POST | `/sessions/{id}/frames` | Add frame to streaming tracker session |
| POST | `/sessions/{id}/prompts` | Add points/boxes/masks on a frame (tracker mode) |
| POST | `/sessions/{id}/propagate` | Propagate tracked objects through all frames |
| DELETE | `/sessions/{id}` | Delete session, free GPU memory |

### Utility
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Loaded models, device, dtype, active sessions |

## Module Structure

```
s18100_sam3_service/
├── app.py                  # (~15 lines) Thin entry point: imports app from src.routes
├── src/                    # Package with all service logic
│   ├── __init__.py
│   ├── config.py           # (~35 lines) YAML config, constants, logger
│   ├── schemas.py          # (~100 lines) All 14 Pydantic request/response models
│   ├── helpers.py          # (~45 lines) decode_image, decode_frames, mask_to_detection
│   ├── models.py           # (~80 lines) 3 lazy-loaded singletons + per-model locks
│   ├── segmentation.py     # (~100 lines) segment_box, segment_text, segment_auto (stateless)
│   ├── sessions.py         # (~335 lines) SessionState, store, init, propagate, CRUD
│   └── routes.py           # (~190 lines) FastAPI app, 9 endpoint handlers
├── configs/default.yaml
├── tests/                  # Integration tests (pytest, requires service running)
│   ├── conftest.py         # Shared fixtures, skip_no_service, image/video loaders
│   ├── data/               # truck.jpg, cat.jpg, bedroom.mp4
│   ├── test00_health.py    # GET /health (3 tests)
│   ├── test01_segment_box.py   # POST /segment_box (5 tests)
│   ├── test02_segment_text.py  # POST /segment_text (5 tests)
│   ├── test03_auto_mask.py     # POST /auto_mask (4 tests)
│   ├── test04_tracker_session.py # Tracker lifecycle (6 tests)
│   ├── test05_video_session.py   # Video lifecycle (4 tests)
│   ├── test06_error_and_edge_cases.py # Error handling + edge cases
│   ├── test07_batch_endpoints.py     # Batch text/auto_mask endpoints
│   └── outputs/            # Generated overlays (gitignored)
├── Dockerfile, docker-compose.yaml, pyproject.toml, pytest.ini
```

### Import DAG (no cycles)

```
src/config.py           ← leaf, no internal imports
src/schemas.py          ← standalone (pydantic only)
src/helpers.py          ← standalone (PIL, numpy, base64)
src/models.py           ← config
src/segmentation.py     ← config, models, helpers
src/sessions.py         ← config, models, helpers
src/routes.py           ← config, schemas, helpers, models, segmentation, sessions
app.py                  ← src.routes (re-export only)
```

## Config

`configs/default.yaml` — all runtime settings:
- `server.port`: 18100
- `model.name`: `facebook/sam3`
- `model.device`: null (auto-detect cuda > mps > cpu)
- `model.dtype`: bfloat16
- `segmentation.mask_threshold`: 0.5
- `segmentation.detection_threshold`: 0.5
- `segmentation.auto_mask_threshold`: 0.2
- `segmentation.auto_mask_nms_threshold`: 0.5 (IoU threshold for NMS across prompts)
- `segmentation.auto_mask_max_area`: 0.95 (discard masks covering more than this fraction)
- `segmentation.max_hole_area`: 100.0 (fill mask holes smaller than this)
- `segmentation.max_sprinkle_area`: 100.0 (remove disconnected blobs smaller than this)
- `segmentation.auto_mask_prompts`: 4 prompt groups (split to stay under CLIP's 32-token limit)
- `sessions.max_active`: 10
- `sessions.ttl_seconds`: 3600

Override config path via env: `SAM3_CONFIG=/path/to/config.yaml`

## Running

```bash
# Docker
docker compose up -d
curl http://localhost:18100/health

# Local development
uv sync
uv run uvicorn app:app --host 0.0.0.0 --port 18100

# Run tests (requires service running at :18100)
uv sync --extra test
uv run pytest tests/ -v
```

## Demo

Interactive Gradio UI at port 7860. Calls the REST service — no local model loading.

```bash
# Service must be running first
docker compose up -d

# Launch demo (from services/s18100_sam3_service/)
uv run demo.py
uv run demo.py --url http://localhost:18100 --server-port 7861 --share

# Custom SAM3 URL via env
SAM3_URL=http://192.168.1.5:18100 uv run demo.py
```

**Tabs**: Text Prompt · Box Prompt · Auto Mask · Text Tracking · Box Tracker
**Demo assets** (`demo/`): `truck.jpg`, `cars.jpg`, `groceries.jpg`, `people-detection.mp4`, `person-bicycle-car-detection.mp4`
**Pattern**: Run inference once → masks cached in `gr.State` → viz controls (opacity, object toggle) re-render instantly without GPU.

## Dependencies

**Service**: fastapi, uvicorn, torch, torchvision, transformers (git HEAD), accelerate, huggingface-hub, Pillow, numpy, pydantic, pyyaml, matplotlib, requests, aiohttp, av, supervision, opencv-python-headless, gradio

**Tests** (`[test]` extra): pytest, opencv-python-headless

## Gotchas

### Critical (wire-protocol / data-corruption)

- **SAM3 session obj_ids mutation**: HF's `add_inputs_to_inference_session` stores a reference to the `obj_ids` list and may mutate it internally. Always `copy.deepcopy()` prompts before storing or re-applying — otherwise streaming frame addition silently breaks because stored obj_ids become empty.
- **Box tracker `input_boxes` requires 3-level nesting**: HF processor expects `[[[x1,y1,x2,y2]]]` (image→boxes→coords). Client sends 2-level `[[x1,y1,x2,y2]]`. `add_prompts_sync()` wraps automatically — do not unwrap when calling the processor directly.
- **Input validation limits**: `decode_image()` rejects images >20MB (decoded) or >8192×8192 pixels (returns **413**). Box coordinates must be non-negative with x1<x2, y1<y2 (**422** on violation).
- **Video mode requires text**: Creating a session with `mode="video"` requires a `text` prompt. Tracker mode is interactive (prompts added after creation).
- **Session mode validation**: `mode` uses `Literal["tracker", "video"]` in the Pydantic schema — invalid mode returns **422** (Pydantic validation), not 400.

### Nice-to-know (performance / operational)

- **Model lazy loading**: First request to each endpoint triggers model download + loading (2–5 min). Subsequent requests are fast. Test timeout is 120s — run `curl /health` before tests to verify readiness.
- **Session reinit on frame add**: Adding a frame to a streaming tracker session reinitializes the entire inference session with all frames + re-applies all stored prompts. This is required by the HF API.
- **auto_mask area filter**: `segment_auto` filters out masks with area > `auto_mask_max_area` (default 0.95, configurable in `configs/default.yaml`).
- **Batch endpoints are sequential**: Dedicated batch endpoints (`/segment_text_batch`, `/auto_mask_batch`) accept up to 16 items. Sam3's DETR cross-attention does not support batch>1 forward pass — images are processed sequentially; batch endpoints exist for API convenience, not GPU-level parallelism.
- **Atomic session creation**: `_reserve_session()` atomically cleans expired sessions, checks the active limit, and inserts under a single lock — prevents race conditions exceeding `max_active`.
- **GPU memory cleanup**: `delete_session_sync()` calls `torch.cuda.empty_cache()` after clearing session state to prevent VRAM fragmentation.
- **Request timing middleware**: All requests logged with `{METHOD} {PATH} {elapsed}s {status_code}` via FastAPI middleware.
- **`groceries.jpg` text prompt**: Image contains packaged grocery bags, not fresh produce. Use `"package. bag. food."` — `"fruit. vegetable. bottle."` scores near zero.
- **Pyright venv**: `pyrightconfig.json` at service root points to `.venv`. Without it, `torch`/`transformers`/`gradio`/`supervision` show `reportMissingImports` and cascade `reportOptionalMemberAccess`.

# CLAUDE.md — SAM 3.1 Segmentation Service

Self-contained SAM 3.1 segmentation REST API. Wraps three HuggingFace `facebook/sam3.1` model architectures (text/image, tracker, video) in a FastAPI service with stateful video session support. GPU required (~5GB VRAM). Port 18106.

## What's Different from SAM 3 (s18100)

**Object Multiplex** — SAM 3.1 uses a shared-memory joint multi-object tracking architecture:
- ~7x faster inference when tracking many objects in parallel (benchmarked at 128 objects on H100)
- Improved VOS (Video Object Segmentation) on 6/7 benchmarks vs SAM 3
- Same API endpoints and request/response format as s18100 — drop-in replacement at port 18106
- Config `model.name` is `facebook/sam3.1` (not `facebook/sam3`)

## Architecture

Modular `src/` package with thin `app.py` entry point. All ML runs on-device (no external service calls).

```
Client → sam3_1_service (:18106, GPU)
         ├── Sam3Model           — text/image open-vocab segmentation
         ├── Sam3TrackerVideoModel — interactive box/point video tracking (Object Multiplex)
         └── Sam3VideoModel      — text-driven video detection + tracking
```

## Endpoints

### Image (stateless)
| Method | Path | Model | Purpose |
|--------|------|-------|---------|
| GET | `/health` | — | Loaded models, device, dtype, active sessions |
| POST | `/segment_box` | Sam3TrackerVideoModel | Box-prompted segmentation `[x1,y1,x2,y2]` |
| POST | `/segment_text` | Sam3Model | Text-prompted open-vocab segmentation |
| POST | `/segment_text_batch` | Sam3Model | Batch text segmentation (up to 16 images) |
| POST | `/auto_mask` | Sam3Model | Segment everything (multi-prompt) |
| POST | `/auto_mask_batch` | Sam3Model | Batch auto-mask (up to 16 images) |

All image endpoints accept single request or `list[request]` for batch processing.

### Video Sessions (stateful)
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/sessions` | Create tracker or text-driven video session |
| POST | `/sessions/{id}/frames` | Add frame to streaming tracker session |
| POST | `/sessions/{id}/prompts` | Add points/boxes/masks on a frame (tracker mode) |
| POST | `/sessions/{id}/propagate` | Propagate tracked objects through all frames |
| DELETE | `/sessions/{id}` | Delete session, free GPU memory |

## Module Structure

```
s18106_sam3_1_service/
├── app.py                  # (~15 lines) Thin entry point: imports app from src.routes
├── src/                    # Package with all service logic
│   ├── __init__.py
│   ├── config.py           # YAML config, constants, logger
│   ├── schemas.py          # All Pydantic request/response models
│   ├── helpers.py          # decode_image, decode_frames, mask_to_detection
│   ├── models.py           # 3 lazy-loaded singletons + per-model locks
│   ├── segmentation.py     # segment_box, segment_text, segment_auto (stateless)
│   ├── sessions.py         # SessionState, store, init, propagate, CRUD
│   └── routes.py           # FastAPI app, 11 endpoint handlers
├── configs/default.yaml    # server.port: 18106, model.name: facebook/sam3.1
├── tests/                  # Integration tests (pytest, requires service running)
│   ├── conftest.py
│   ├── data/               # truck.jpg, cat.jpg, bedroom.mp4
│   ├── test00_health.py through test07_batch_endpoints.py
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

`configs/default.yaml` key settings:
- `server.port`: 18106
- `model.name`: `facebook/sam3.1`
- `model.device`: null (auto-detect cuda > mps > cpu)
- `model.dtype`: bfloat16
- `segmentation.mask_threshold`: 0.5
- `segmentation.detection_threshold`: 0.5
- `segmentation.auto_mask_threshold`: 0.2
- `sessions.max_active`: 10
- `sessions.ttl_seconds`: 3600

Override config path via env: `SAM3_CONFIG=/path/to/config.yaml`

## Running

```bash
# Docker
cd services/s18106_sam3_1_service && docker compose up -d
curl http://localhost:18106/health

# Local development
uv sync
uv run uvicorn app:app --host 0.0.0.0 --port 18106

# Run tests (requires service running at :18106)
uv sync --extra test
uv run pytest tests/ -v
```

## Dependencies

**Service**: fastapi, uvicorn, torch, torchvision, transformers (git HEAD), accelerate, huggingface-hub, Pillow, numpy, pydantic, pyyaml, matplotlib, requests, aiohttp, av

**Tests** (`[test]` extra): pytest, supervision, opencv-python-headless

## Gotchas

### Model Loading
- **Model lazy loading**: First request triggers download + loading (2-5 min). Test timeout is 120s. Run `curl /health` before tests to pre-warm.
- **Native SAM3.1 library only**: `facebook/sam3.1` has NO `model.safetensors` — HF Transformers integration doesn't work. Must use `pip install git+https://github.com/facebookresearch/sam3.git` and `build_sam3_multiplex_video_predictor`.
- **BPE tokenizer compression**: `merges.txt` from HF hub is plain text; SAM3 tokenizer requires gzip-compressed `merges.txt.gz`. `models.py` handles this automatically on first load.
- **Double-check locking for model loading**: `models.py` uses double-check locking to prevent multiple threads from loading the same model simultaneously on first request.

### Threading & Async
- **Bfloat16 autocast thread-local**: The native predictor enters `torch.autocast("cuda", bfloat16)` globally in its `__init__` thread, but this context is NOT inherited by `asyncio.to_thread` workers. All inference calls explicitly enter `torch.autocast("cuda", dtype=torch.bfloat16)`.
- **Global inference lock**: The native SAM3.1 predictor is not thread-safe. `inference_lock` in `models.py` serializes all GPU inference operations.
- **asyncio.to_thread for inference**: All model inference is wrapped in `asyncio.to_thread()` to avoid blocking the FastAPI event loop.
- **Atomic session creation**: `_reserve_session()` atomically cleans expired sessions, checks the active limit, and inserts the new session under a single lock.

### Request Formats
- **⚠ Box format is `[xmin, ymin, w, h]` normalized — NOT YOLO-style `[x1, y1, x2, y2]`**: The native `add_prompt` API expects `[xmin, ymin, width, height]` in normalized [0,1] coords. YOLO/COCO tools typically emit `[x1, y1, x2, y2]`; convert before passing. `bounding_box_labels` must accompany boxes (one label=1 per box).
- **Box API format wrapping**: Tests send boxes as `[[[x1,y1,x2,y2]]]` (HF batch format). `sessions.py` unwraps one nesting level.
- **Point/box tensors**: Pass `bounding_boxes=torch.tensor(..., dtype=torch.float32)` and `point_labels=torch.tensor(..., dtype=torch.int32)` — not raw Python lists.
- **`frame_index` key**: Request dict for `add_prompt` uses `frame_index` (not `frame_idx`). `propagate_in_video` uses `start_frame_index` (not `start_frame_idx`).
- **Input validation limits**: `decode_image()` rejects images >20MB or >8192x8192 pixels (413). Box coords must have x1<x2, y1<y2 (422).

### API Behavior
- **add_prompt resets state on each call**: SAM3.1 multiplex `add_prompt` calls `reset_state` before processing. For multi-object tracking, ALL boxes must be merged into a SINGLE `add_prompt` call. `sessions.py` accumulates and replays them combined.
- **Official API is `handle_request()` / `handle_stream_request()`**: Always use the high-level request-dispatch API. Request dicts use `type=` to dispatch: `"start_session"`, `"add_prompt"`, `"propagate_in_video"`, etc.
- **Point prompts use SAM2 path**: Point prompts route through `add_sam2_new_points`. For propagation after point-only prompts, `start_frame_index=0` must be passed to bypass the "no prompts received" check.
- **Batch endpoints sequential**: SAM3.1's DETR cross-attention does not support batch>1 forward pass — batch endpoints exist for API convenience only.
- **Video mode requires text**: `mode="video"` requires a `text` prompt. Tracker mode is interactive.
- **Session mode validation**: `mode` uses `Literal["tracker", "video"]` — invalid mode returns 422, not 400.

## Troubleshooting

**First request takes 2–5 min** — Native SAM3.1 model downloads on first use. Test timeout is 120s; if you run tests immediately after docker start you'll hit it. Wait for `/health` to show all 3 models loaded.

**`ModuleNotFoundError: sam3` on startup** — Native library missing. The Dockerfile installs it via `pip install git+https://github.com/facebookresearch/sam3.git` — check Docker build logs for this step.

**`KeyError: 'merges.txt'` on first text/video request** — `models.py` auto-handles this by converting `merges.txt` → `merges.txt.gz`. If you see this error with a stale cache, delete `~/.cache/huggingface/hub/models--facebook--sam3.1/` and restart.

**Concurrent session interference** — `inference_lock` serializes all GPU ops. If a long propagation is running, other requests queue behind it. This is by design (SAM3.1 predictor is not thread-safe).

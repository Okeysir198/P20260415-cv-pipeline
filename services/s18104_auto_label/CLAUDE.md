# CLAUDE.md — services/s18104_auto_label/

Self-contained auto-labeling REST API. Pure orchestrator — delegates all ML to SAM3 (:18100), does post-processing (NMS, polygon extraction, format conversion) locally. CPU-only, no torch.

## Architecture

```
Client → auto_label (:18104, CPU) → SAM3 (:18100, GPU)
```

Modular `src/` package with clear separation of concerns. `app.py` is a thin entry point that re-exports the FastAPI app from `src/routes`.

## Endpoints

### Image (stateless)
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/annotate` | Single-image annotation (text/auto/hybrid mode) |

### Batch Jobs (async)
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/jobs` | Create batch job (multiple images, async) |
| GET | `/jobs` | List all jobs |
| GET | `/jobs/{id}` | Poll job status + results |
| DELETE | `/jobs/{id}` | Cancel/remove job |

### Video Sessions (stateful, wraps SAM3 tracker)
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/video/sessions` | Create session (tracker or video mode) |
| POST | `/video/sessions/{id}/frames` | Add frame + optional prompts |
| POST | `/video/sessions/{id}/propagate` | Propagate tracked objects through all frames (video mode) |
| DELETE | `/video/sessions/{id}` | Close session |

### Utility
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/convert` | Format conversion (no SAM3 call) |
| GET | `/health` | Status + SAM3 connectivity |

### Output Formats
All annotation endpoints accept `output_format`: `coco` (default), `yolo`, `yolo_seg`, `label_studio`.

## Module Structure

```
auto_label/
├── app.py                  # (~15 lines) Thin entry point: imports app from src, re-exports for uvicorn
├── src/                    # Package with all service logic
│   ├── __init__.py         # Package marker
│   ├── config.py           # (~65 lines) YAML config, constants, logger
│   ├── schemas.py          # (~150 lines) All Pydantic models
│   ├── geometry.py         # (~170 lines) Image decode, polygon, bbox, NMS
│   ├── formatters.py       # (~100 lines) COCO/YOLO/YOLO-seg/LS converters
│   ├── annotator.py        # (~260 lines) SAM3 HTTP callers + annotation modes
│   ├── state.py            # (~480 lines) Mutable state (jobs + video sessions)
│   └── routes.py           # (~230 lines) FastAPI app, lifespan, all endpoints
├── configs/
│   └── default.yaml        # Runtime config
├── tests/                  # Integration tests (pytest, self-contained)
│   ├── conftest.py         # Shared fixtures, skip_no_service, visualization helpers
│   ├── data/               # Test images + video (fire_sample_{1,2,3}.jpg, indoor_fire.mp4)
│   ├── test00_health.py    # GET /health
│   ├── test01_annotate.py  # POST /annotate (all modes + formats)
│   ├── test02_jobs.py      # POST/GET/DELETE /jobs
│   ├── test03_video.py     # POST/DELETE /video/sessions, POST frames, POST propagate
│   ├── test04_convert.py   # POST /convert (all formats, roundtrip)
│   ├── outputs/            # Generated overlays + JSON responses
│   └── CLAUDE.md           # Test-specific docs
├── Dockerfile              # ubuntu:24.04 + uv
├── docker-compose.yaml     # Port 18104, SAM3 via host.docker.internal
├── pytest.ini              # pytest config (testpaths, python_files pattern)
├── pyproject.toml          # Deps + optional [test] extra (pytest, supervision)
└── uv.lock
```

### Import DAG (no cycles)

```
src/config.py           ← leaf, no internal imports
src/schemas.py          ← config
src/geometry.py         ← config, schemas
src/formatters.py       ← schemas
src/annotator.py        ← config, schemas, geometry
src/state.py            ← config, schemas, geometry, formatters, annotator
src/routes.py           ← config, schemas, geometry, formatters, annotator, state
app.py                  ← src.routes (re-export only)
```

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `src/config.py` | YAML config loading, all constants (`SAM3_URL`, `REQUEST_TIMEOUT`, defaults), logger setup |
| `src/schemas.py` | All Pydantic models: request/response for annotate, jobs, video, convert, health |
| `src/geometry.py` | `strip_data_uri`, `decode_image`, `mask_to_polygon`, bbox helpers, `nms_numpy`, `apply_nms` |
| `src/formatters.py` | `to_coco`, `to_yolo`, `to_yolo_seg`, `to_label_studio`, `format_output` |
| `src/annotator.py` | SAM3 HTTP callers (`call_sam3_text/auto/box`) + annotation modes (`annotate_text/auto/hybrid_mode`, `annotate_single`) |
| `src/state.py` | Job dict/lock/semaphore + video sessions dict/lock + all state operations |
| `src/routes.py` | FastAPI app instance, lifespan, all 11 endpoint handlers |

## Config

`configs/default.yaml` — all runtime settings:
- `server.port`: 18104
- `services.sam3_url`: SAM3 endpoint
- `processing.*`: timeout, default confidence/mode/format
- `nms.*`: per-class IoU, `cross_class_enabled`, `cross_class_threshold`
- `polygon.*`: simplify tolerance, min vertices
- `jobs.*`: max concurrent, TTL, max images per batch
- `video_sessions.*`: `max_active`, `ttl_seconds`

Override SAM3 URL via env: `SAM3_URL=http://host.docker.internal:18100`

## Running

```bash
# Requires SAM3 running at :18100
cd services/s18100_sam3_service && docker compose up -d

# Start auto_label
cd services/s18104_auto_label && docker compose up -d
curl http://localhost:18104/health

# Run tests (from services/s18104_auto_label/)
uv sync --extra test
uv run pytest -v                          # all 28 tests
uv run pytest tests/test00_health.py -v   # single file
```

## Dependencies

**Service** (in Docker): fastapi, uvicorn, pydantic, requests, httpx, Pillow, numpy, opencv-python-headless, pyyaml

**Tests** (local only, `[test]` extra): pytest, supervision (for bbox/mask/label visualization)

No torch, no HF transformers — all ML is delegated to SAM3.

## Video Session Prompts

Video frame prompts are forwarded directly to SAM3's `/sessions/{id}/prompts` endpoint. The field is always **`frame_idx`** (not `frame_index`) — it matches SAM3's Pydantic schema exactly; any other name returns 422:
```json
{"frame_idx": 0, "obj_ids": [1], "boxes": [[[x1, y1, x2, y2]]]}
```

The `obj_class_map` maps `obj_id → (class_id, class_name)` using the `classes` dict from session creation. If an obj_id isn't in the map, the detection returns `class_id=-1`.

## Gotchas

- **SAM3 must be running** — all annotation endpoints call SAM3. If SAM3 is down, requests return 500 with connection error details.
- **Video frames must be same size** — SAM3 tracker reinitializes with all frames on each frame add; different-sized frames cause `ValueError: all input arrays must have the same shape`.
- **SAM3 prompt format** — prompts are forwarded as-is to SAM3. Use SAM3's schema (`frame_idx`, `obj_ids`, `boxes`), not auto_label's detection schema (`class_id`, `bbox_xyxy`).
- **Polygon coords are normalized** (0–1) in API responses. `bbox_xyxy` is in pixel coords.
- **Batch jobs are async** — POST `/jobs` returns immediately with `job_id`, poll GET `/jobs/{id}` until `status=completed`.
- **Docker SAM3 URL** — inside Docker, SAM3 is at `http://host.docker.internal:18100` (set via docker-compose env). Locally, it's `http://localhost:18100`.

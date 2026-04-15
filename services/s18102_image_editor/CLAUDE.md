# CLAUDE.md — services/s18102_image_editor/

Self-contained image inpainting orchestrator REST API. Pure orchestrator — delegates segmentation to SAM3 (:18100) and image generation to Flux NIM (:18101), does mask compositing locally. CPU-only, no torch.

## Architecture

```
Client → image_editor (:18102, CPU) → SAM3 (:18100, GPU)    [for bbox/text mask]
                                    → Flux NIM (:18101, GPU) [for image generation]
```

Four mask modes (priority order):
1. **`mask` provided** — composite directly (skip SAM3)
2. **`bbox` provided** — calls SAM3 `/segment_box`, gets mask, composites
3. **`text_prompt` provided** — calls SAM3 `/segment_text`, picks highest-scoring detection mask, composites
4. **None of above** — direct Flux edit, no compositing (returns Flux output as-is)

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/inpaint` | Image inpainting — 4 mask modes, returns base64 images. Runs in thread executor. |
| GET | `/health` | Probes Flux NIM + SAM3, returns `"ok"` or `"degraded"` |

### /inpaint Request Fields
- `image` (str, required) — base64 input image
- `prompt` (str, required) — text prompt for Flux generation
- `mask` (str, optional) — base64 mask (white = inpaint region)
- `bbox` (list[float], optional) — `[x1, y1, x2, y2]` for SAM3 box segmentation
- `text_prompt` (str, optional) — text for SAM3 text segmentation
- `num_variants` (int, 1-4, default 1) — number of output images
- `seed` (int, default 0) — random seed (incremented per variant)
- `steps` (int, 1-50, default 4) — Flux inference steps

## Module Structure

```
s18102_image_editor/
├── app.py                  # (~15 lines) Thin entry point: imports app from src.routes
├── src/
│   ├── __init__.py
│   ├── config.py           # (~37 lines) YAML config, env var overrides, logger
│   ├── schemas.py          # (~36 lines) InpaintRequest, InpaintResponse, HealthResponse
│   ├── helpers.py          # (~23 lines) encode_image, decode_mask (imports _shared.image_utils)
│   ├── clients.py          # (~60 lines) call_flux, call_sam3_box, call_sam3_text
│   ├── compositing.py      # (~36 lines) mask_composite (alpha blending)
│   └── routes.py           # (~86 lines) FastAPI app, /inpaint + /health handlers
├── configs/default.yaml    # server, services URLs, generation defaults, timeout
├── tests/
│   ├── conftest.py         # Shared fixtures, skip_no_service, generate_source_image via Flux
│   ├── test00_health.py    # GET /health (3 tests)
│   ├── test01_inpaint_direct.py   # Direct edit, no mask (3 tests)
│   ├── test02_inpaint_bbox.py     # Bbox-prompted inpaint (3 tests)
│   ├── test03_inpaint_text.py     # Text-prompted inpaint (3 tests)
│   └── outputs/            # Generated overlays (gitignored)
├── Dockerfile              # ubuntu:24.04 + uv, CPU-only, context=services/ (copies _shared/)
├── docker-compose.yaml     # Port 18102, SAM3/Flux via host.docker.internal
├── pyproject.toml
└── uv.lock
```

### Import DAG (no cycles)

```
src/config.py           ← leaf (yaml, os, logging)
src/schemas.py          ← standalone (pydantic only)
src/helpers.py          ← _shared.image_utils (strip_data_uri)
src/clients.py          ← config (FLUX_NIM_URL, SAM3_URL, REQUEST_TIMEOUT)
src/compositing.py      ← standalone (numpy, PIL)
src/routes.py           ← config, schemas, helpers, clients, compositing
app.py                  ← src.routes (re-export only)
```

## Config

`configs/default.yaml`:
- `server.port`: 18102
- `services.flux_nim_url`: `http://localhost:18101`
- `services.sam3_url`: `http://localhost:18100`
- `generation.default_steps`: 4
- `generation.default_seed`: 0
- `generation.max_num_variants`: 4
- `request_timeout`: 120

Override via env: `FLUX_NIM_URL`, `SAM3_URL`, `IMAGE_EDITOR_CONFIG=/path/to/config.yaml`

## Running

```bash
# Start dependencies first
cd services/s18100_sam3_service && docker compose up -d   # SAM3 :18100
cd services/s18101_flux_nim && docker compose up -d       # Flux NIM :18101

# Start image editor
cd services/s18102_image_editor && docker compose up -d
curl http://localhost:18102/health

# Local development
cd services/s18102_image_editor
uv sync
uv run uvicorn app:app --host 0.0.0.0 --port 18102

# Tests (requires all 3 services running — conftest generates test images via Flux)
uv sync --extra test
uv run pytest tests/ -v
```

## Dependencies

**Service**: fastapi, uvicorn, pydantic, requests, Pillow, numpy, pyyaml

**Tests** (`[test]` extra): pytest, matplotlib

No torch — all ML delegated to SAM3 and Flux NIM.

## Gotchas

- **Both SAM3 and Flux must be running** — `/health` probes both and reports `"degraded"` if either is down. Only the "no mask" mode works without SAM3.
- **Synchronous HTTP in async endpoint** — `/inpaint` delegates to `_inpaint_sync` via `run_in_executor`. All downstream calls use `requests` (synchronous). Each request occupies one thread pool thread.
- **Flux content filter** — If Flux returns `finishReason: "CONTENT_FILTERED"`, the service raises HTTP 422 (not 502). Error message suggests trying a different prompt.
- **Flux payload format** — Image sent as `[f"data:image/png;base64,{b64}"]` (list with data URI prefix). This is specific to the Flux NIM API contract.
- **SAM3 text picks highest-scoring detection** — `call_sam3_text` selects `max(detections, key=score)["mask"]`. Returns 422 if SAM3 finds nothing.
- **Mask compositing resizes** — `compositing.py` resizes both edited image (LANCZOS) and mask (NEAREST) to match original dimensions if they differ (Flux may return different resolution).
- **Variant seeds are sequential** — For `num_variants > 1`, seeds are `seed+0, seed+1, ...`. Each variant is a separate sequential Flux call (no parallelism).
- **Docker build context is parent** — `context: ..` in docker-compose allows copying `services/_shared/` into the image.
- **Tests require Flux for fixture generation** — `conftest.py` calls Flux NIM to generate source test images, so all 3 services must be running.

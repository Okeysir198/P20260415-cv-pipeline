# Image Editor Orchestrator

CPU-only FastAPI orchestrator that combines SAM3 segmentation and Flux NIM generation for mask-based inpainting. Calls downstream microservices — no GPU required.

## Architecture

```
                    image_editor (orchestrator :18102)
  POST /inpaint — 4 mask modes:
    mask provided  -> composite directly
    bbox provided  -> SAM3 /segment_box -> composite
    text provided  -> SAM3 /segment_text -> composite
    none           -> direct Flux edit (no composite)

  Dependencies:
    sam3 :18100   — /segment_box, /segment_text
    flux_nim :18101 — /v1/infer
```

- **Port**: 18102
- **GPU**: None (CPU-only, lightweight)
- **Base**: ubuntu:24.04 + uv
- **Dependencies**: SAM3 service (:18100) + Flux NIM (:18101) running independently

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/inpaint` | POST | Inpainting with 4 mask modes |
| `/health` | GET | Health check + downstream service status |

### POST /inpaint — 4 Mask Modes

| Mode | Field | Behavior |
|------|-------|----------|
| Direct mask | `mask` | Use provided base64 mask as-is |
| Box -> SAM3 | `bbox` | Send `[x1,y1,x2,y2]` to SAM3 `/segment_box` |
| Text -> SAM3 | `text_prompt` | Send text to SAM3 `/segment_text` |
| No mask | (none) | Direct Flux edit, no compositing |

```json
{
  "image": "<base64-encoded PNG>",
  "prompt": "replace with a red hat",
  "bbox": [100, 50, 200, 150],
  "num_variants": 1,
  "seed": 0,
  "steps": 4
}
```
Response:
```json
{
  "images": ["<base64 result>"],
  "mask_used": "<base64 mask from SAM3>",
  "seed": 0
}
```

## Quick Start

### Docker (standalone — requires SAM3 + Flux NIM running independently)

```bash
cd services/s18102_image_editor
docker compose up -d
curl http://localhost:18102/health
```

### Local Development

```bash
cd services/s18102_image_editor
uv sync
FLUX_NIM_URL=http://localhost:18101 SAM3_URL=http://localhost:18100 \
  uv run uvicorn app:app --host 0.0.0.0 --port 18102
```

## Tests

Integration tests in `tests/` cover all endpoints (requires all 3 services running: image_editor :18102, SAM3 :18100, Flux NIM :18101):

| File | Endpoint | Tests |
|------|----------|-------|
| `test00_health.py` | `GET /health` | Health check, downstream status, status values (3 tests) |
| `test01_inpaint_direct.py` | `POST /inpaint` | Direct edit (no mask): basic, multiple variants, visualization (3 tests) |
| `test02_inpaint_bbox.py` | `POST /inpaint` | BBox -> SAM3: basic, compositing check, visualization (3 tests) |
| `test03_inpaint_text.py` | `POST /inpaint` | Text -> SAM3: basic, compositing check, visualization (3 tests) |

```bash
# Start all 3 services first
cd services/s18100_sam3_service && docker compose up -d
cd services/s18101_flux_nim && docker compose up -d
cd services/s18102_image_editor && docker compose up -d

# Run tests (from this directory)
uv run pytest tests/ -v

# Run individual test files
uv run pytest tests/test00_health.py -v
uv run pytest tests/test01_inpaint_direct.py -v
uv run pytest tests/test02_inpaint_bbox.py -v
uv run pytest tests/test03_inpaint_text.py -v
```

Tests skip gracefully if the service is not running. Test data lives in `tests/data/`, visualizations are saved to `tests/outputs/`.

## Configuration

`configs/default.yaml`:
```yaml
server:
  host: "0.0.0.0"
  port: 18102

services:
  flux_nim_url: "http://localhost:18101"
  sam3_url: "http://localhost:18100"

generation:
  default_steps: 4
  default_seed: 0
  max_num_variants: 4

request_timeout: 120
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `FLUX_NIM_URL` | No | Flux NIM URL (default: `http://localhost:18101`) |
| `SAM3_URL` | No | SAM3 service URL (default: `http://localhost:18100`) |
| `IMAGE_EDITOR_CONFIG` | No | Path to config YAML (default: `configs/default.yaml`) |

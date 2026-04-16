# CLAUDE.md — services/s18105_annotation_quality_assessment/

Self-contained annotation quality assessment REST API. Pure orchestrator — delegates ML to SAM3 (:18100) for visual verification and optionally to Ollama (:11434) for VLM semantic checking. All local processing is structural validation + scoring. CPU-only, no torch.

## Architecture

```
Client → annotation_quality_assessment (:18105, CPU) → SAM3 (:18100, GPU)    [for /verify]
                                     → Ollama (:11434, GPU)   [for /verify with enable_vlm=true]
```

Three operational modes:
1. **Structural validation** (`/validate`) — zero external dependencies, pure local checks
2. **SAM3 verification** (`/verify`) — structural + box/text/auto_mask verification via SAM3
3. **VLM semantic verification** (`/verify` + `enable_vlm=true`) — adds LangGraph crop+scene verification via Ollama

## Endpoints

### Validation (stateless, no SAM3 needed)
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/validate` | Structural checks: bbox bounds, class IDs, duplicates, aspect ratios, polygon integrity |

### SAM3 Verification (stateless, requires SAM3)
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/verify` | Structural + SAM3 box IoU + text verification + missing detection. Optional VLM via `enable_vlm` |

### Fix (stateless)
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/fix` | Auto-fix labels: clip out-of-bounds, remove duplicates/degenerates, return corrected labels |

### Batch Jobs (async)
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/jobs` | Create batch QA job (multiple images, async) |
| GET | `/jobs` | List all jobs (optional `?status=` filter) |
| GET | `/jobs/{id}` | Poll job status + partial results |
| DELETE | `/jobs/{id}` | Cancel/remove job |

### Report
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/report` | Aggregate per-image results into dataset-level summary |

### Utility
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Service status + SAM3/Ollama connectivity + active jobs |

### Label Formats
All endpoints accept `label_format`: `yolo` (default), `yolo_seg`, `coco`.

## Module Structure

```
s18105_annotation_quality_assessment/
├── app.py                  # (~15 lines) Thin entry point: imports app from src.routes
├── src/
│   ├── __init__.py
│   ├── config.py           # (~92 lines) YAML config, all constants, logger
│   ├── schemas.py          # (~199 lines) 17 Pydantic models (request/response/internal)
│   ├── geometry.py         # (~163 lines) Image decode, IoU matrix, bbox conversion, polygon geometry
│   ├── parsers.py          # (~148 lines) YOLO, YOLO-seg, COCO format parsers
│   ├── validators.py       # (~223 lines) 10 structural validation checks
│   ├── sam3.py             # (~322 lines) SAM3 HTTP callers + mask helpers + verification logic
│   ├── vlm.py              # (~369 lines) Ollama/LangGraph VLM: crop verify + scene verify + combine
│   ├── scoring.py          # (~441 lines) Score formula, fix generation, fix application, report aggregation
│   ├── state.py            # (~419 lines) Job state, per-image processing, TTL cleanup
│   └── routes.py           # (~279 lines) FastAPI app, lifespan, all endpoint handlers
├── configs/
│   └── default.yaml        # All runtime settings (64 params)
├── tests/
│   ├── conftest.py         # Shared fixtures, skip_no_service/sam3/ollama, visualization helpers
│   ├── test_endpoints.py   # Full endpoint coverage (validate, verify, fix, jobs, report)
│   └── test_vlm.py         # VLM unit tests (parsers, crop) + integration tests
├── Dockerfile              # ubuntu:24.04 + uv, CPU-only
├── docker-compose.yaml     # Port 18105, SAM3/Ollama via host.docker.internal
├── pyproject.toml
└── uv.lock
```

### Import DAG (no cycles)

```
src/config.py           ← leaf, no internal imports
src/schemas.py          ← standalone (pydantic only)
src/geometry.py         ← standalone (PIL, numpy, base64)
src/parsers.py          ← schemas
src/validators.py       ← config, geometry, schemas
src/sam3.py             ← config, geometry, schemas
src/vlm.py              ← config, geometry, schemas (+ langchain, langgraph)
src/scoring.py          ← config, parsers, schemas
src/state.py            ← config, geometry, parsers, sam3, scoring, schemas, validators, vlm
src/routes.py           ← config, geometry, scoring, schemas, state
app.py                  ← src.routes (re-export only)
```

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `src/config.py` | YAML config loading, all constants (`SAM3_URL`, `OLLAMA_URL`, validation/scoring/VLM defaults), logger |
| `src/schemas.py` | All Pydantic models: `ValidateRequest/Response`, `VerifyRequest/Response`, `FixRequest/Response`, `QAJobRequest`, `JobState`, `ReportRequest/Response`, `SAM3Verification`, `VLMVerification`, `HealthResponse` |
| `src/geometry.py` | `strip_data_uri`, `decode_image`, `compute_iou_matrix`, `compute_single_iou`, bbox conversions (`norm_cxcywh_to_xyxy`, `norm_xyxy_to_pixel`, `pixel_xyxy_to_norm_cxcywh`), `shoelace_area`, `polygon_self_intersects` |
| `src/parsers.py` | `parse_yolo`, `parse_yolo_seg`, `parse_coco` — all return `list[ParsedAnnotation]` with normalized cxcywh bbox + optional polygon |
| `src/validators.py` | `validate_annotations()` — 10 structural checks, returns `(issues, suggested_fixes)` |
| `src/sam3.py` | SAM3 HTTP callers (`call_sam3_box/text/auto`), mask helpers (`rasterize_polygon_mask`, `decode_sam3_mask`, `compute_mask_iou`), `verify_with_sam3()` |
| `src/vlm.py` | LangGraph VLM flow: `crop_verify_node` (fan-out per annotation) + `scene_verify_node` → `combine_node`. Compiled graph singleton. Public API: `verify_with_vlm()` |
| `src/scoring.py` | `score_image()` (weighted penalty formula), `generate_sam3_fixes()`, `generate_vlm_fixes()`, `apply_fixes()`, `annotations_to_labels()`, `aggregate_results()` |
| `src/state.py` | Job dict + lock + semaphore, `process_single_image_validate/verify/verify_async`, `process_job()`, `ttl_cleanup_loop()`, CRUD (`create/get/list/cancel_job`) |
| `src/routes.py` | FastAPI app, lifespan (init semaphore + TTL cleanup), 9 endpoint handlers |

## Verification Pipeline

### Structural Checks (10 checks, no GPU)
1. Out-of-bounds bbox coordinates (generates `clip_bbox` fix)
2. Invalid class IDs
3. Degenerate boxes — w or h < `min_box_size` (generates `remove_degenerate` fix)
4. Large boxes — w or h > `max_box_size`
5. Duplicate annotations — same class + IoU > `duplicate_iou_threshold` (generates `remove_duplicate` fix)
6. Extreme aspect ratios
7. Polygon vertex count < 3
8. Polygon self-intersection
9. Polygon-bbox area consistency
10. Polygon out-of-bounds vertices

### SAM3 Verification (4 steps)
1. **Box IoU** — For each annotation, call `/segment_box`, compute IoU between annotation and SAM3 bbox
2. **Mask IoU** — For seg formats with polygon data, rasterize annotation polygon and compute pixel-level IoU vs SAM3 mask
3. **Text verification** — For annotations with box IoU < `text_verify_threshold` (0.6), call `/segment_text` with class name; flag as misclassified if no overlap > 0.1
4. **Missing detection** — Call `/auto_mask`, filter by area bounds, flag unannotated objects that don't overlap existing annotations

### VLM Verification (LangGraph, 2 parallel nodes)
- **crop_verify_node** — Crop each annotation bbox, ask Ollama "Is this a {class_name}?" concurrently via `asyncio.gather`
- **scene_verify_node** — Send full image + annotation list, ask for incorrect indices + missing objects + quality score
- **combine_node** — Merge results into `VLMVerification`
- **Trigger modes**: `all` (always run), `selective` (only if SAM3 grade is `review` or `bad`), `standalone`

### Scoring Formula

```
score = 1.0 - (w_structural * structural_penalty
             + w_bbox * bbox_penalty
             + w_classification * classification_penalty
             + w_coverage * coverage_penalty
             + w_vlm * vlm_penalty)
```

Default weights: structural=0.25, bbox_quality=0.30, classification=0.15, coverage=0.15, vlm=0.15. Grade: `good` >= 0.8, `review` >= 0.5, `bad` < 0.5.

## Config

`configs/default.yaml` — all runtime settings:
- `server.*`: host, port (18105)
- `services.*`: `sam3_url`, `ollama_url`
- `processing.request_timeout`: 120s
- `validation.*`: `min_box_size` (0.005), `max_box_size` (0.95), `duplicate_iou_threshold` (0.95), `max_aspect_ratio` (20), polygon area ratio bounds
- `sam3.*`: `text_verify_threshold` (0.6), `auto_mask_min_area` (0.001), `auto_mask_max_area` (0.8)
- `vlm.*`: `model` (qwen3.5:9b), `trigger`, `crop_padding`, `request_timeout`, `crop_prompt`, `scene_prompt`
- `scoring.weights.*`: structural, bbox_quality, classification, coverage, vlm
- `scoring.thresholds.*`: good (0.8), review (0.5)
- `jobs.*`: `max_concurrent_jobs` (2), `ttl_seconds` (3600), `max_images_per_job` (500)

Override via env: `SAM3_URL`, `OLLAMA_URL`, `ANNOTATION_QA_CONFIG=/path/to/config.yaml`

## Running

```bash
# Requires SAM3 running at :18100 (for /verify endpoint only)
cd services/s18100_sam3_service && docker compose up -d

# Optional: Start Ollama for VLM semantic verification
docker run -d --gpus all -p 11434:11434 --name ollama ollama/ollama
docker exec ollama ollama pull qwen3.5:9b

# Start annotation_quality_assessment
cd services/s18105_annotation_quality_assessment && docker compose up -d
curl http://localhost:18105/health

# Local development
uv sync
uv run uvicorn app:app --host 0.0.0.0 --port 18105

# Run tests (from services/s18105_annotation_quality_assessment/)
uv sync --extra test
uv run pytest tests/ -v
uv run pytest tests/test_vlm.py::TestParseCropResponse -v  # unit tests only (no service needed)
```

## Dependencies

**Service** (in Docker): fastapi, uvicorn, pydantic, requests, httpx, Pillow, numpy, pyyaml, langchain-openai, langchain-core, langgraph

**Tests** (local only, `[test]` extra): pytest, supervision, opencv-python-headless

No torch — all ML delegated to SAM3 and Ollama.

## Gotchas

- **`include_missing_detection` is config-driven and off by default** — `configs/default.yaml` (and the pipeline's shared `02_annotation_quality.yaml`) sets `sam3.include_missing_detection: false`. The missing-detection check (step 4 in SAM3 verification) fires only when explicitly enabled. Default-off prevents false-positive "unlabeled object" flags on class-restricted datasets where non-target objects are intentionally left unannotated.
- **SAM3 is optional** — `/validate` and `/fix` work without SAM3. Only `/verify` calls SAM3; it returns results with zero IoUs if SAM3 is down (individual call failures are caught and defaulted to 0.0).
- **Ollama is optional** — VLM verification only runs when `enable_vlm=true` in the request. If Ollama is down, `verify_with_vlm()` returns `VLMVerification(available=False)`.
- **VLM fail-open** — On VLM error (crop failure or LLM invocation failure), annotations are marked `is_correct=True` with `confidence=0.0`. This is intentional: transient VLM errors don't flag good annotations as bad.
- **Batch job config** — `QAJobRequest` has a `config` field for threshold overrides (same as single-image endpoints). Without it, all thresholds use defaults from `configs/default.yaml`.
- **COCO in `/fix`** — The `/fix` endpoint uses placeholder image dimensions (1000x1000) for COCO format since no image is provided. COCO pixel coords will be approximate if the actual image is a different size.
- **LangGraph singleton** — The VLM graph is compiled once and cached in `_compiled_graph`. Graph topology is static; only state varies per call.
- **Docker SAM3/Ollama URLs** — Inside Docker, services are at `http://host.docker.internal:18100` and `:11434` (set via docker-compose env). Locally, use `http://localhost:*`.
- **Job TTL cleanup** — Background task runs every 60s, removes completed/failed/cancelled jobs older than `ttl_seconds` (default 1 hour). Uses threading lock for thread safety.
- **YOLO-seg odd coords** — Parser drops trailing odd coordinate with a warning. This handles malformed labels gracefully but silently modifies input data.

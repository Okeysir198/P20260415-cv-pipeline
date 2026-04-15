# CLAUDE.md — services/

Microservices for the camera_edge pipeline. Three services work together for generative augmentation (SAM3 segmentation + Flux NIM generation + orchestrator compositing). SAM3 (s18100) and SAM 3.1 (s18106) serve as standalone segmentation + video tracking services — SAM 3.1 adds Object Multiplex for ~7x faster multi-object tracking. The auto_label service provides a REST API for SAM3-based auto-labeling. The annotation_quality_assessment service provides structural + SAM3-verified annotation quality assessment.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                image_editor (orchestrator :18102)         │
│  POST /inpaint — 4 mask modes:                           │
│    mask provided → composite directly                    │
│    bbox provided → SAM3 /segment_box → composite         │
│    text provided → SAM3 /segment_text → composite        │
│    none          → direct Flux edit (no composite)       │
└──────────┬───────────────────────────────┬───────────────┘
           ▼                               ▼
  ┌─────────────────┐  ┌──────────────────────┐  ┌─────────────────────┐
  │  sam3_service    │  │  sam3_1_service       │  │  flux_nim            │
  │  :18100 (GPU)   │  │  :18106 (GPU)         │  │  :18101 (GPU)        │
  │  3 models:      │  │  3 models (SAM 3.1):  │  │  /v1/infer           │
  │   Sam3Model     │  │   Sam3Model           │  │  NVIDIA NIM container│
  │   Sam3Tracker   │  │   Sam3Tracker         │  │  (pre-built)         │
  │   Sam3Video     │  │   Sam3Video           │  │                      │
  │  9 endpoints    │  │  11 endpoints         │  │                      │
  │                 │  │  Object Multiplex:    │  │                      │
  │                 │  │  ~7x faster multi-obj │  │                      │
  └────────┬────────┘  └──────────┬───────────┘  └─────────────────────┘
           ▲
           │  SAM3 calls (/segment_text, /segment_box, /auto_mask)
           │
┌──────────┴───────────────────────────────────────────────┐
│               auto_label (orchestrator :18104)            │
│  POST /annotate — single-image annotation (text/auto/box)│
│  POST /jobs     — batch annotation (async)               │
│  POST /sessions — video annotation sessions              │
│  Output formats: COCO, YOLO, YOLO-seg, Label Studio      │
└──────────────────────────────────────────────────────────┘
           ▲
           │  SAM3 calls (/segment_text, /auto_mask) for verification
           │
┌──────────┴───────────────────────────────────────────────┐
│             annotation_quality_assessment (orchestrator :18105)            │
│  POST /validate — structural checks (no SAM3 needed)     │
│  POST /verify   — SAM3-based verification (optional)     │
│  POST /verify   — + VLM semantic check via Ollama        │
│  POST /fix      — auto-fix labels (clip, remove dups)    │
│  POST /jobs     — batch QA (async)                       │
│  POST /report   — aggregate QA summary                   │
└──────────┬───────────────────────────────────────────────┘
           │  Ollama calls (crop verify + scene verify)
           ▼
  ┌─────────────────────┐
  │  Ollama              │
  │  :11434 (GPU)        │
  │  VLM semantic QA     │
  │  (optional)          │
  └─────────────────────┘
```

## Services

| Service | Port | GPU | Description |
|---------|------|-----|-------------|
| `s18101_flux_nim/` | 18101 | Yes | Flux 2 Klein 4B — NVIDIA NIM pre-built container, no custom code |
| `s18100_sam3_service/` | 18100 | Yes | SAM3 REST API — 3 models, 9 endpoints (image + video sessions), ~5GB VRAM |
| `s18106_sam3_1_service/` | 18106 | Yes | SAM 3.1 REST API — Object Multiplex parallel multi-object tracking, ~7x faster, 11 endpoints, ~5GB VRAM |
| `s18102_image_editor/` | 18102 | No | Orchestrator — calls SAM3 + Flux NIM, mask compositing (CPU-only) |
| `s18104_auto_label/` | 18104 | No | Auto-labeling orchestrator — SAM3-based segmentation + detection REST API (CPU-only) |
| `s18105_annotation_quality_assessment/` | 18105 | No | Annotation QA orchestrator — structural validation + SAM3 verification + optional VLM via Ollama (:11434) REST API (CPU-only) |
| `s18103_label_studio/` | 18103 | No | Label Studio — annotation review UI (independent) |

## SAM3 Endpoints

SAM3 (s18100) and SAM 3.1 (s18106) expose the same endpoint signatures. SAM 3.1 is available at port 18106 and adds `POST /segment_text_batch` and `POST /auto_mask_batch` dedicated batch endpoints. Use s18106 when tracking many objects simultaneously — Object Multiplex gives ~7x throughput improvement.

### Image (stateless)
| Endpoint | Model | Purpose |
|----------|-------|---------|
| `POST /segment_box` | Sam3TrackerVideoModel | Box-prompted segmentation |
| `POST /segment_text` | Sam3Model | Text-prompted open-vocab segmentation |
| `POST /auto_mask` | Sam3Model | Segment everything (multi-prompt) |

### Video Sessions (stateful)
| Endpoint | Purpose |
|----------|---------|
| `POST /sessions` | Create tracker or text-driven video session |
| `POST /sessions/{id}/frames` | Add frame to streaming session |
| `POST /sessions/{id}/prompts` | Add points/boxes/masks (tracker mode) |
| `POST /sessions/{id}/propagate` | Propagate tracked objects through frames |
| `DELETE /sessions/{id}` | Delete session, free GPU memory |

### Health
| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Reports loaded models, device, VRAM, active sessions |

## Auto-Label Endpoints (s18104)

11 endpoints: `/annotate` (single image), `/jobs` CRUD (batch async), `/video/sessions` CRUD (stateful tracking), `/convert`, `/health`. Output formats: `coco`, `yolo`, `yolo_seg`, `label_studio`. See `s18104_auto_label/CLAUDE.md` for full endpoint reference.

## Annotation QA Endpoints (s18105)

9 endpoints: `/validate` (structural, no SAM3), `/verify` (SAM3 + optional VLM), `/fix` (auto-correct), `/jobs` CRUD (batch async), `/report` (aggregate summary), `/health`. Label formats: `yolo`, `yolo_seg`, `coco`. See `s18105_annotation_quality_assessment/CLAUDE.md` for full endpoint reference.

## Image Editor Endpoints (s18102)

2 endpoints: `/inpaint` (4 mask modes: mask, bbox, text, none), `/health`. See `s18102_image_editor/CLAUDE.md` for full endpoint reference.

## Quick Start

### Inpainting services (start each independently)

```bash
cd services/s18100_sam3_service && docker compose up -d    # SAM3 (:18100, GPU)
cd services/s18101_flux_nim && docker compose up -d        # Flux NIM (:18101, GPU)
cd services/s18102_image_editor && docker compose up -d    # Orchestrator (:18102, CPU)
curl http://localhost:18102/health                   # check orchestrator + dependencies
```

### Individual services

```bash
# SAM3 only
cd services/s18100_sam3_service && docker compose up -d
curl http://localhost:18100/health

# SAM 3.1 (same features, ~7x faster multi-object tracking via Object Multiplex)
cd services/s18106_sam3_1_service && docker compose up -d
curl http://localhost:18106/health

# Flux NIM only (needs NGC_API_KEY in .env)
cd services/s18101_flux_nim && docker compose up -d
curl http://localhost:18101/v1/health/ready

# Auto-Label (requires SAM3 running at :18100)
cd services/s18104_auto_label && docker compose up -d
curl http://localhost:18104/health

# Annotation QA (requires SAM3 running at :18100 for /verify endpoint)
cd services/s18105_annotation_quality_assessment && docker compose up -d
curl http://localhost:18105/health

# Label Studio (independent)
cd services/s18103_label_studio && docker compose up -d
curl http://localhost:18103/health
```

### Auto-Label usage examples

```bash
# Start SAM3 + auto_label together
cd services/s18100_sam3_service && docker compose up -d
cd services/s18104_auto_label && docker compose up -d

# Single image annotation (text mode, COCO output)
curl -X POST http://localhost:18104/annotate \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64>", "classes": {"0": "fire", "1": "smoke"}, "mode": "text"}'

# Batch job
curl -X POST http://localhost:18104/jobs \
  -H "Content-Type: application/json" \
  -d '{"images": [{"image": "<base64>", "filename": "img1.jpg"}], "classes": {"0": "fire"}, "mode": "text"}'

# Poll job status
curl http://localhost:18104/jobs/<job_id>
```

### Annotation QA usage examples

```bash
# Start SAM3 + annotation_quality_assessment together
cd services/s18100_sam3_service && docker compose up -d
cd services/s18105_annotation_quality_assessment && docker compose up -d

# Structural validation only (no SAM3 needed)
curl -X POST http://localhost:18105/validate \
  -H "Content-Type: application/json" \
  -d '{"labels": ["0 0.5 0.5 0.3 0.2", "1 0.2 0.3 0.1 0.15"], "label_format": "yolo", "classes": {"0": "fire", "1": "smoke"}}'

# SAM3-based verification (requires SAM3 at :18100)
curl -X POST http://localhost:18105/verify \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64>", "labels": ["0 0.5 0.5 0.3 0.2"], "label_format": "yolo", "classes": {"0": "fire"}}'

# Auto-fix labels
curl -X POST http://localhost:18105/fix \
  -H "Content-Type: application/json" \
  -d '{"labels": ["0 1.05 0.5 0.2 0.3"], "label_format": "yolo", "classes": {"0": "fire"}}'

# Aggregate report
curl -X POST http://localhost:18105/report \
  -H "Content-Type: application/json" \
  -d '{"results": [{"filename": "a.jpg", "score": 0.9, "grade": "good", "num_issues": 0, "num_annotations": 3}]}'
```

### SAM3 tests (requires SAM3 service running)

```bash
cd services/s18100_sam3_service
uv sync --extra test
uv run pytest tests/ -v                          # all 9 endpoints covered
uv run pytest tests/test01_segment_box.py -v     # box segmentation
uv run pytest tests/test02_segment_text.py -v    # text segmentation
uv run pytest tests/test03_auto_mask.py -v       # segment everything
uv run pytest tests/test04_tracker_session.py -v # video tracking (bedroom.mp4)
uv run pytest tests/test05_video_session.py -v   # text-driven video (bedroom.mp4)
```

### SAM 3.1 tests (requires SAM 3.1 service running at :18106)

```bash
cd services/s18106_sam3_1_service
uv sync --extra test
uv run pytest tests/ -v                               # all endpoints covered
uv run pytest tests/test01_segment_box.py -v          # box segmentation
uv run pytest tests/test02_segment_text.py -v         # text segmentation
uv run pytest tests/test03_auto_mask.py -v            # segment everything
uv run pytest tests/test04_tracker_session.py -v      # video tracking (bedroom.mp4)
uv run pytest tests/test05_video_session.py -v        # text-driven video (bedroom.mp4)
uv run pytest tests/test07_batch_endpoints.py -v      # batch text/auto_mask endpoints
```

### Auto-Label tests (requires SAM3 + auto_label running)

```bash
cd services/s18104_auto_label
uv sync --extra test
uv run pytest tests/ -v                          # all 28 tests
```

### Annotation QA tests (requires SAM3 + annotation_quality_assessment running)

```bash
cd services/s18105_annotation_quality_assessment
uv sync --extra test
uv run pytest tests/ -v
```

## Environment Variables

Each service that needs secrets has its own local `.env` file (not the project root `.env`):

| Variable | Service `.env` | Required |
|----------|---------------|----------|
| `HF_TOKEN` | `s18100_sam3_service/.env` | Yes — HuggingFace model download |
| `HF_TOKEN` | `s18106_sam3_1_service/.env` | Yes — HuggingFace model download |
| `NGC_API_KEY` | `s18101_flux_nim/.env` | Yes — NVIDIA NIM container auth |

`image_editor`, `auto_label`, and `annotation_quality_assessment` are CPU-only orchestrators with no secrets — they have no `.env` file.

## Design Decisions

- **ubuntu:24.04 base** for all custom Dockerfiles — NOT `nvidia/cuda` (PyTorch pip wheels ship their own CUDA libs)
- **uv** for dependency management inside containers (same as project root)
- **Orchestrators are CPU-only** — image_editor, auto_label, and annotation_quality_assessment only make HTTP calls + numpy processing, no torch
- **SAM3 is self-contained** — modular `src/` package with thin `app.py` entry point, no cross-directory imports in Docker
- **Flux NIM is pre-built** — no custom code, just docker-compose pointing to NVIDIA's container
- **Label Studio is independent** — not part of the inpainting or labeling pipeline
- **Auto-Label delegates all ML to SAM3** — pure orchestrator pattern: receives images, forwards to SAM3 for segmentation/detection, post-processes (NMS, polygon extraction, format conversion), returns results. Same architecture as image_editor.
- **Annotation QA has two modes** — `/validate` runs structural checks (bounds, duplicates, aspect ratios) with zero external dependencies; `/verify` optionally calls SAM3 for visual verification. Service works fully without SAM3 for structural-only QA.
- **SAM3 session state** — server-side `dict[str, SessionState]` with TTL auto-cleanup; `copy.deepcopy()` stored prompts to avoid HF inference session mutating obj_ids references

## Troubleshooting

### SAM3 Service Issues

**Problem:** SAM3 service returns 503 or connection refused
```bash
# Check if SAM3 is running
curl http://localhost:18100/health

# Restart SAM3
cd services/s18100_sam3_service && docker compose restart

# Check logs
docker compose logs -f sam3
```

**Problem:** OOM (Out of Memory) errors
```bash
# Check SAM3 VRAM usage (should be ~5GB)
curl http://localhost:18100/health | grep vram_used

# Reduce batch size or use smaller model
# Edit docker-compose.yml: set SAM3_MODEL=sam3_h_small (1.2GB vs 5GB)
```

### Session Cleanup

**Problem:** Stale video sessions consuming memory
```bash
# List active sessions (response includes active_sessions count)
curl http://localhost:18100/health

# Delete specific session
curl -X DELETE http://localhost:18100/sessions/{session_id}

# Delete all sessions (restart service)
cd services/s18100_sam3_service && docker compose restart
```

### Dependency Issues

**Problem:** Auto-label can't reach SAM3
```bash
# Check SAM3 health from auto_label container
docker exec s18104_auto_label curl http://host.docker.internal:18100/health

# If failing, check docker network (both should be on 'services' network)
docker network ls
docker network inspect services
```

**Problem:** Image editor returns 503 for SAM3-dependent modes
```bash
# Verify SAM3 is running before starting image_editor
curl http://localhost:18100/health

# Image editor's /inpaint with mode="none" (direct Flux) doesn't need SAM3
curl -X POST http://localhost:18102/inpaint \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64>", "mode": "none"}'
```

### Cleanup Commands

```bash
# Stop all services and remove volumes
cd services
for svc in s18100_sam3_service s18106_sam3_1_service s18101_flux_nim s18102_image_editor s18104_auto_label s18105_annotation_quality_assessment; do
  cd $svc && docker compose down -v && cd ..
done

# Remove all service containers
docker ps -a | grep s181 | awk '{print $1}' | xargs docker rm -f

# Remove orphaned docker networks
docker network prune -f
```

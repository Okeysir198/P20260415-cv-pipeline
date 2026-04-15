# Services

Docker microservices for the camera_edge annotation and data augmentation pipelines. Each service is self-contained with its own Dockerfile, configs, and tests.

## Architecture

```
s18100_sam3_service (GPU, :18100)  ─── Core segmentation engine
s18106_sam3_1_service (GPU, :18106) ── SAM 3.1 (drop-in replacement, ~7x faster multi-object)
    ↑
    ├── s18102_image_editor (CPU, :18102) ──→ s18101_flux_nim (GPU, :18101)
    ├── s18104_auto_label   (CPU, :18104)
    └── s18105_annotation_quality_assessment(CPU, :18105) ──→ Ollama (GPU, :11434, optional)

s18103_label_studio (CPU, :18103)  ─── Independent annotation UI
```

Only SAM3 and Flux NIM require a GPU. The orchestrators (image_editor, auto_label, annotation_quality_assessment) are CPU-only and delegate all ML inference to SAM3/Flux via HTTP.

## Services

### s18100_sam3_service — SAM3 Segmentation API
**Port:** 18100 | **GPU:** Required (~5 GB VRAM)

Core segmentation engine wrapping three SAM3 model variants. Provides box-prompted, text-prompted, and automatic segmentation plus stateful video tracking sessions.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Status, loaded models, VRAM, active sessions |
| `/segment_box` | POST | Box-prompted segmentation → mask + bbox |
| `/segment_text` | POST | Text-prompted open-vocab segmentation |
| `/auto_mask` | POST | Segment everything → list of masks |
| `/sessions` | POST | Create video tracking or text-driven session |
| `/sessions/{id}/frames` | POST | Add frame to streaming session |
| `/sessions/{id}/prompts` | POST | Add point/box prompts on a frame |
| `/sessions/{id}/propagate` | POST | Propagate tracked objects through frames |
| `/sessions/{id}` | DELETE | Delete session, free GPU memory |

**Env:** `HF_TOKEN` in `.env` for model download.

---

### s18101_flux_nim — Flux 2 Klein 4B Image Generation
**Port:** 18101 | **GPU:** Required (~8 GB VRAM)

Pre-built NVIDIA NIM container for text-to-image and image-to-image generation. Used by the image editor for inpainting.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/health/ready` | GET | Health check |
| `/v1/infer` | POST | Text-to-image or image-to-image (base64) |

**Env:** `NGC_API_KEY` in `.env` for NVIDIA NGC authentication.

**Notes:** Max 4 generation steps. Output is always 1024x1024 JPEG.

---

### s18102_image_editor — Generative Inpainting Orchestrator
**Port:** 18102 | **GPU:** None (CPU-only)

Orchestrates SAM3 segmentation + Flux NIM generation for mask-based inpainting. Supports four mask modes: direct mask, box → SAM3, text → SAM3, or no mask (direct edit).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Status + downstream service connectivity |
| `/inpaint` | POST | Inpaint with mask/bbox/text/direct mode |

**Requires:** SAM3 (:18100) + Flux NIM (:18101) running.

---

### s18106_sam3_1_service — SAM 3.1 Segmentation API
**Port:** 18106 | **GPU:** Required (~5 GB VRAM)

Drop-in replacement for s18100 using `facebook/sam3.1`. Adds Object Multiplex (~7x faster multi-object tracking) and two dedicated batch endpoints. Same API signatures and request/response format as s18100.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Status, loaded models, VRAM, active sessions |
| `/segment_box` | POST | Box-prompted segmentation → mask + bbox |
| `/segment_text` | POST | Text-prompted open-vocab segmentation |
| `/segment_text_batch` | POST | Batch text segmentation (up to 16 images) |
| `/auto_mask` | POST | Segment everything → list of masks |
| `/auto_mask_batch` | POST | Batch auto-mask (up to 16 images) |
| `/sessions` | POST | Create video tracking or text-driven session |
| `/sessions/{id}/frames` | POST | Add frame to streaming session |
| `/sessions/{id}/prompts` | POST | Add point/box prompts on a frame |
| `/sessions/{id}/propagate` | POST | Propagate tracked objects through frames |
| `/sessions/{id}` | DELETE | Delete session, free GPU memory |

**Env:** `HF_TOKEN` in `.env` for model download.

---

### s18103_label_studio — Annotation Review UI
**Port:** 18103 | **GPU:** None (CPU-only)

Open-source annotation UI ([Label Studio](https://labelstud.io/)) for visually reviewing and correcting annotations. Mounts dataset images read-only from the host.

**Typical workflow:**
1. Run auto-annotation → `runs/auto_annotate/<dataset>/`
2. Create project: `uv run core/p04_label_studio/bridge.py setup --data-config ...`
3. Import annotations: `uv run core/p04_label_studio/bridge.py import ...`
4. Review and correct in the web UI at `http://localhost:18103`
5. Export back to YOLO: `uv run core/p04_label_studio/bridge.py export ...`

**First launch:** Create admin account at `http://localhost:18103/user/signup`.

---

### s18104_auto_label — Auto-Labeling Orchestrator
**Port:** 18104 | **GPU:** None (CPU-only)

SAM3-based auto-labeling. Processes images in three modes and outputs annotations in multiple formats.

| Mode | Method | Use case |
|------|--------|----------|
| `text` | Text-prompted SAM3 per class | Known classes |
| `auto` | Automatic SAM3, then classify | Discovery |
| `hybrid` | Text first, then auto for uncovered regions | Maximum recall |

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Status + SAM3 connectivity |
| `/annotate` | POST | Single-image annotation |
| `/jobs` | POST | Create async batch job |
| `/jobs` | GET | List all jobs |
| `/jobs/{id}` | GET | Poll job status + results |
| `/jobs/{id}` | DELETE | Cancel/remove job |
| `/video/sessions` | POST | Create video annotation session |
| `/video/sessions/{id}/frames` | POST | Add frame + prompts |
| `/video/sessions/{id}` | DELETE | Close session |
| `/convert` | POST | Format conversion (no SAM3) |

**Output formats:** COCO, YOLO, YOLO-seg, Label Studio.

**Requires:** SAM3 (:18100) running.

---

### s18105_annotation_quality_assessment — Annotation Quality Assessment
**Port:** 18105 | **GPU:** None (CPU-only)

Multi-mode annotation quality assessment: structural validation (no dependencies), SAM3 visual verification (optional), and Ollama VLM semantic verification (optional).

**Validation modes:**
- **Structural** (10 checks) — out-of-bounds, invalid class IDs, degenerate boxes, duplicates, aspect ratios, polygon integrity
- **SAM3 verification** — box IoU, mask IoU, text-based classification check, missing detection scan
- **VLM verification** (optional) — crop + scene review via Qwen 3.5-9B on Ollama

**Scoring:** Each image gets a score (0–1) and grade (`good` >= 0.8, `review` 0.5–0.8, `bad` < 0.5).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Status + SAM3/Ollama connectivity |
| `/validate` | POST | Structural checks only (no SAM3 needed) |
| `/verify` | POST | Structural + SAM3 + optional VLM |
| `/fix` | POST | Auto-fix labels (clip, deduplicate, remove degenerate) |
| `/jobs` | POST | Create async batch QA job |
| `/jobs` | GET | List jobs |
| `/jobs/{id}` | GET | Poll job status + results |
| `/jobs/{id}` | DELETE | Cancel/remove job |
| `/report` | POST | Aggregate per-image results into dataset summary |

**Requires:** SAM3 (:18100) for `/verify`. Ollama (:11434) optional for VLM.

## Quick Start

```bash
# Start services (run from project root)
cd services/s18100_sam3_service && docker compose up -d    # SAM3 (GPU)
cd services/s18101_flux_nim && docker compose up -d        # Flux NIM (GPU)
cd services/s18102_image_editor && docker compose up -d    # Inpainting orchestrator
cd services/s18103_label_studio && docker compose up -d    # Annotation UI
cd services/s18104_auto_label && docker compose up -d      # Auto-labeling
cd services/s18105_annotation_quality_assessment && docker compose up -d   # QA
cd services/s18106_sam3_1_service && docker compose up -d                 # SAM 3.1 (GPU, optional)

# Health checks
curl http://localhost:18100/health
curl http://localhost:18101/v1/health/ready
curl http://localhost:18102/health
curl http://localhost:18103/health
curl http://localhost:18104/health
curl http://localhost:18105/health
curl http://localhost:18106/health
```

## Common Patterns

- **Configuration:** Each service has `configs/default.yaml`. Environment variables override config values.
- **Docker networking:** Services use `host.docker.internal` to reach each other when running in separate containers.
- **State:** Jobs and sessions are in-memory (no external database). Label Studio uses a Docker named volume for persistence.
- **Tests:** Each service has integration tests that skip gracefully when the service is not running.

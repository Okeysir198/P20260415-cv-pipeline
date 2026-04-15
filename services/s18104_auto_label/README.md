# Auto-Label Service

SAM3-based auto-labeling REST API. Pure orchestrator — delegates all ML to SAM3 (:18100), does post-processing (NMS, polygon extraction, format conversion) locally. CPU-only, no torch.

## Architecture

```
                  +------------------+
                  |   Client / CLI   |
                  +--------+---------+
                           |
                    HTTP requests
                           |
                  +--------v---------+
                  |   auto_label     |
                  |   :18104 (CPU)   |
                  |                  |
                  |  POST /annotate  |
                  |  POST /jobs      |
                  |  POST /video/..  |
                  |  POST /convert   |
                  |  GET  /health    |
                  +--------+---------+
                           |
                  SAM3 API calls
                           |
                  +--------v---------+
                  |   sam3_service   |
                  |   :18100 (GPU)   |
                  |                  |
                  |  /segment_text   |
                  |  /segment_box    |
                  |  /auto_mask      |
                  |  /sessions/...   |
                  +------------------+
```

## Quick Start

```bash
# Requires SAM3 running at :18100
cd services/s18100_sam3_service && docker compose up -d

# Start auto_label
cd services/s18104_auto_label && docker compose up -d

# Verify health
curl http://localhost:18104/health
# {"status": "ok", "sam3": "ok", "active_jobs": 0, "active_video_sessions": 0}
```

## Endpoints

### GET /health

Health check for the service and its SAM3 dependency.

**Response:**
```json
{
  "status": "ok",
  "sam3": "ok",
  "active_jobs": 0,
  "active_video_sessions": 0
}
```

---

### POST /annotate

Annotate a single image synchronously. Returns detections with optional masks.

**Request:**
```json
{
  "image": "<BASE64_IMAGE>",
  "classes": {"0": "fire", "1": "smoke"},
  "text_prompts": {"fire": "flames and burning"},
  "mode": "text",
  "confidence_threshold": 0.5,
  "nms_iou_threshold": 0.5,
  "output_format": "coco",
  "include_masks": true
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | string | required | Base64-encoded image (JPEG or PNG), data URI prefix optional |
| `classes` | object | required | Mapping of class ID to class name, e.g. `{"0": "fire"}` |
| `text_prompts` | object | `{}` | Optional refined prompts per class name |
| `mode` | string | `"text"` | Annotation mode: `text`, `auto`, or `hybrid` |
| `confidence_threshold` | float | `0.5` | Minimum confidence threshold (0.0–1.0) |
| `nms_iou_threshold` | float | `0.5` | Per-class NMS IoU threshold (0.0–1.0) |
| `output_format` | string | `"coco"` | Output format: `coco`, `yolo`, `yolo_seg`, `label_studio` |
| `include_masks` | bool | `true` | Include base64 PNG masks in response |

**Response:**
```json
{
  "detections": [
    {
      "class_id": 0,
      "class_name": "fire",
      "score": 0.92,
      "bbox_xyxy": [222, 135, 319, 326],
      "bbox_norm": [0.422266, 0.656534, 0.151172, 0.542614],
      "polygon": [[0.347656, 0.384943], [0.498047, 0.384943], ...],
      "mask": "<BASE64_PNG or null>",
      "area": 0.024909
    }
  ],
  "image_width": 640,
  "image_height": 352,
  "num_detections": 1,
  "processing_time_s": 2.145,
  "formatted_output": [...]
}
```

The `formatted_output` field contains detections converted to the requested `output_format`.

---

### POST /jobs

Create an asynchronous batch annotation job for multiple images.

**Request:**
```json
{
  "images": [
    {"image": "<BASE64_IMAGE>", "filename": "img_001.jpg"},
    {"image": "<BASE64_IMAGE>", "filename": "img_002.jpg"}
  ],
  "classes": {"0": "fire", "1": "smoke"},
  "text_prompts": {},
  "mode": "text",
  "confidence_threshold": 0.5,
  "nms_iou_threshold": 0.5,
  "output_format": "coco",
  "include_masks": false,
  "webhook_url": null
}
```

**Response:**
```json
{
  "job_id": "0d4eb6e70c16",
  "total_images": 2,
  "status": "queued"
}
```

---

### GET /jobs

List all batch jobs, optionally filtered by status.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `status` | string | Filter by status: `queued`, `running`, `completed`, `failed`, `cancelled` |

**Response:**
```json
[
  {
    "job_id": "0d4eb6e70c16",
    "status": "completed",
    "total_images": 2,
    "processed_images": 2,
    "created_at": 1710924600.0
  }
]
```

---

### GET /jobs/{job_id}

Get the status and results of a specific batch job.

**Response (in progress):**
```json
{
  "job_id": "0d4eb6e70c16",
  "status": "running",
  "total_images": 10,
  "processed_images": 4,
  "results": [...],
  "error": null,
  "created_at": 1710924600.0
}
```

**Response (completed):**
```json
{
  "job_id": "0d4eb6e70c16",
  "status": "completed",
  "total_images": 2,
  "processed_images": 2,
  "results": [
    {
      "filename": "img_001.jpg",
      "num_detections": 1,
      "detections": [
        {
          "class_id": 0,
          "class_name": "fire",
          "score": 0.92,
          "bbox_xyxy": [222, 135, 319, 326],
          "bbox_norm": [0.422266, 0.656534, 0.151172, 0.542614],
          "polygon": [...],
          "mask": null,
          "area": 0.024909
        }
      ],
      "formatted_output": [...],
      "image_width": 640,
      "image_height": 352
    }
  ],
  "error": null,
  "created_at": 1710924600.0
}
```

---

### DELETE /jobs/{job_id}

Cancel a running or queued job.

**Response:**
```json
{
  "job_id": "0d4eb6e70c16",
  "status": "cancelled"
}
```

---

### POST /video/sessions

Create a video annotation session backed by SAM3 tracker.

**Request:**
```json
{
  "mode": "tracker",
  "classes": {"0": "fire", "1": "smoke"},
  "text": null,
  "output_format": "coco"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | string | `"tracker"` | SAM3 session mode: `tracker` (box/point prompts + propagation) or `video` |
| `classes` | object | `{}` | Mapping of class ID to class name |
| `text` | string | `null` | Text prompt for text-driven video mode |
| `output_format` | string | `"coco"` | Output format for detections |

**Response:**
```json
{
  "session_id": "e1d8dbbd3b1f",
  "sam3_session_id": "abc123def456",
  "mode": "tracker"
}
```

---

### POST /video/sessions/{session_id}/frames

Add a frame to an active video session with optional prompts.

**Request (first frame with box prompt):**
```json
{
  "frame": "<BASE64_IMAGE>",
  "prompts": [
    {"frame_idx": 0, "obj_ids": [1], "boxes": [[[150, 100, 500, 400]]]}
  ]
}
```

**Request (subsequent frame, propagation only):**
```json
{
  "frame": "<BASE64_IMAGE>"
}
```

Prompts are forwarded as-is to SAM3's `/sessions/{id}/prompts` endpoint.

**Response:**
```json
{
  "frame_idx": 0,
  "detections": [
    {
      "obj_id": 1,
      "class_id": 0,
      "class_name": "fire",
      "score": 0.95,
      "bbox_xyxy": [150, 102, 507, 391],
      "bbox_norm": [0.513281, 0.351563, 0.556250, 0.411932],
      "polygon": [...],
      "mask": "<BASE64_PNG>",
      "area": 0.153
    }
  ]
}
```

---

### DELETE /video/sessions/{session_id}

Close and clean up a video annotation session and its backing SAM3 session.

**Response:**
```json
{
  "deleted": true,
  "session_id": "e1d8dbbd3b1f"
}
```

---

### POST /convert

Convert detections between output formats without re-running inference (no SAM3 call needed).

**Request:**
```json
{
  "detections": [
    {
      "class_id": 0,
      "class_name": "fire",
      "score": 0.92,
      "bbox_xyxy": [222, 135, 319, 326],
      "bbox_norm": [0.422266, 0.656534, 0.151172, 0.542614],
      "polygon": [],
      "mask": null,
      "area": 0.024909
    }
  ],
  "output_format": "yolo",
  "image_width": 640,
  "image_height": 352
}
```

**Response:**
```json
{
  "formatted_output": ["0 0.422266 0.656534 0.151172 0.542614"]
}
```

---

## Output Formats

| Format | Key | Bbox Format | Masks | Use Case |
|--------|-----|-------------|-------|----------|
| COCO | `coco` | `[x, y, w, h]` (pixels) | Flattened polygon vertices (pixels) | Standard annotation interchange |
| YOLO | `yolo` | `class_id cx cy w h` (normalized 0–1) | None | Detection training |
| YOLO-seg | `yolo_seg` | `class_id x1 y1 x2 y2 ... xN yN` (normalized) | Normalized polygon | Segmentation training |
| Label Studio | `label_studio` | Percentage-based (0–100) | Rectangle labels | Import into Label Studio for review |

## Annotation Modes

| Mode | Description | Speed | Accuracy | When to Use |
|------|-------------|-------|----------|-------------|
| `text` | Text-prompted segmentation via SAM3 `/segment_text` for each class | Medium | High for known classes | Classes with clear text descriptions |
| `auto` | Automatic mask generation via SAM3 `/auto_mask`, then classify each crop | Slow | Medium | Discovery / unknown classes |
| `hybrid` | Text mode first, then auto mode for uncovered regions | Slowest | Highest | Maximum recall needed |

## Project Structure

```
auto_label/
├── app.py                  # Thin entry point — imports app from src, re-exports for uvicorn
├── src/                    # Package with all service logic
│   ├── __init__.py
│   ├── config.py           # YAML config loading, constants, logger
│   ├── schemas.py          # All Pydantic request/response models
│   ├── geometry.py         # Image decode, polygon extraction, bbox helpers, NMS
│   ├── formatters.py       # COCO/YOLO/YOLO-seg/Label Studio converters
│   ├── annotator.py        # SAM3 HTTP callers + annotation mode logic
│   ├── state.py            # Batch job engine + video session management
│   └── routes.py           # FastAPI app, lifespan, all endpoint handlers
├── configs/
│   └── default.yaml        # Runtime config
├── tests/                  # Integration tests (pytest, requires service running)
│   ├── conftest.py         # Shared fixtures, skip_no_service, visualization helpers
│   ├── data/               # Test images + video
│   ├── test00_health.py    # GET /health
│   ├── test01_annotate.py  # POST /annotate (all modes + formats)
│   ├── test02_jobs.py      # POST/GET/DELETE /jobs
│   ├── test03_video.py     # POST/DELETE /video/sessions, POST frames, POST propagate
│   ├── test04_convert.py   # POST /convert (all formats, roundtrip)
│   └── test05_parallel_batch.py  # Parallel batch job stress tests
├── Dockerfile
├── docker-compose.yaml
├── pyproject.toml
└── uv.lock
```

## Configuration

The service reads configuration from `configs/default.yaml`. Key sections:

| Section | Description |
|---------|-------------|
| `server` | Host and port binding |
| `services` | SAM3 service URL |
| `annotation` | Default mode, confidence threshold, output format |
| `nms` | Per-class and cross-class NMS thresholds |
| `polygon` | Simplification tolerance and minimum vertices |
| `jobs` | Batch job concurrency and TTL |
| `video_sessions` | Maximum concurrent sessions and session TTL |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SAM3_URL` | `http://localhost:18100` | SAM3 service base URL (overrides config) |
| `AUTO_LABEL_CONFIG` | `configs/default.yaml` | Path to config YAML file |

Inside Docker, SAM3 is reached at `http://host.docker.internal:18100` (set via docker-compose env).

## Dependencies

**Service** (in Docker): fastapi, uvicorn, pydantic, requests, httpx, Pillow, numpy, opencv-python-headless, pyyaml

**Tests** (local only, `[test]` extra): pytest, supervision (for bbox/mask/label visualization)

No torch, no HF transformers — all ML is delegated to SAM3.

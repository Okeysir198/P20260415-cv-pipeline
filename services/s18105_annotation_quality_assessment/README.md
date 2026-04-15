# Annotation QA Service

CPU-only annotation quality assurance REST API. Performs structural validation of existing labels, optionally verifies annotations against SAM3 segmentation at `:18100`, and supports VLM-based semantic verification via Ollama at `:11434`.

## Architecture

```
                  +------------------+
                  |   Client / CLI   |
                  +--------+---------+
                           |
                    HTTP requests
                           |
                  +--------v---------+
                  |  annotation_quality_assessment   |
                  |  :18105          |
                  |                  |
                  |  - /validate     |
                  |  - /verify       |
                  |  - /fix          |
                  |  - /jobs         |
                  |  - /report       |
                  |  - /health       |
                  +---+---------+----+
                      |         |
             SAM3 API calls  Ollama API
             (/verify only)  (/verify + VLM)
                      |         |
              +-------v---+ +---v----------+
              | sam3_svc   | |   Ollama     |
              | :18100     | |   :11434     |
              |            | |              |
              | /seg_text  | | qwen3.5:9b   |
              | /seg_box   | | (VLM verify) |
              | /auto_mask | |              |
              +------------+ +--------------+
```

The service has three modes of operation:

- **Structural validation** (`/validate`): Checks label files for formatting errors, out-of-bounds coordinates, duplicate boxes, extreme aspect ratios, and other structural issues. No GPU or SAM3 dependency required.
- **SAM3 verification** (`/verify`): In addition to structural checks, sends the image to SAM3 for independent segmentation and compares the results against the provided annotations. Detects misclassified objects, missing annotations, and poor localization.
- **VLM verification** (`/verify` with `enable_vlm: true`): Uses Qwen3.5-9B via Ollama for semantic verification. Crops each annotation and asks the VLM if the class label is correct, then performs scene-level verification. Adds a third quality dimension beyond structural and visual.

## Endpoints

### GET /health

Health check for the service, SAM3, and Ollama connectivity.

```bash
curl http://localhost:18105/health
```

**Response:**
```json
{
  "status": "ok",
  "sam3": "ok",
  "ollama": "unreachable (Connection refused)",
  "active_jobs": 0
}
```

---

### POST /validate

Structural validation only (no SAM3 needed). Checks a single image's annotations for formatting and geometric issues.

```bash
curl -X POST http://localhost:18105/validate \
  -H "Content-Type: application/json" \
  -d '{
    "image": "<BASE64_IMAGE>",
    "labels": ["0 0.5 0.5 0.3 0.4", "1 0.2 0.3 0.1 0.15"],
    "label_format": "yolo",
    "classes": {"0": "fire", "1": "smoke"}
  }'
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | string | required | Base64-encoded image (for extracting dimensions) |
| `labels` | array | required | YOLO strings, YOLO-seg strings, or COCO dicts |
| `label_format` | string | `"yolo"` | Label format: `yolo`, `yolo_seg`, `coco` |
| `classes` | object | required | Mapping of class ID to class name, e.g. `{"0": "fire"}` |
| `config` | object | `{}` | Optional threshold overrides (min_box_size, etc.) |

**Response:**
```json
{
  "issues": [
    {
      "type": "degenerate_box",
      "severity": "medium",
      "annotation_idx": 1,
      "detail": "Box too small: w=0.0020, h=0.0030 (min=0.005)"
    }
  ],
  "num_annotations": 2,
  "num_issues": 1,
  "quality_score": 0.85,
  "grade": "good",
  "suggested_fixes": [
    {
      "type": "remove_degenerate",
      "annotation_idx": 1,
      "original": {"class_id": 1, "bbox_norm": [0.2, 0.3, 0.002, 0.003]},
      "suggested": null,
      "reason": "Bounding box too small (near-zero area)."
    }
  ],
  "label_format": "yolo",
  "processing_time_s": 0.012
}
```

---

### POST /verify

Full QA with SAM3 verification. Performs structural validation plus SAM3-based cross-checking. Optionally runs VLM semantic verification.

```bash
curl -X POST http://localhost:18105/verify \
  -H "Content-Type: application/json" \
  -d '{
    "image": "<BASE64_IMAGE>",
    "labels": ["0 0.5 0.5 0.3 0.4"],
    "label_format": "yolo",
    "classes": {"0": "fire", "1": "smoke"},
    "text_prompts": {"fire": "fire flames", "smoke": "smoke plume"},
    "include_missing_detection": true
  }'
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | string | required | Base64-encoded image |
| `labels` | array | required | YOLO strings, YOLO-seg strings, or COCO dicts |
| `label_format` | string | `"yolo"` | Label format: `yolo`, `yolo_seg`, `coco` |
| `classes` | object | required | Mapping of class ID to class name |
| `text_prompts` | object | `{}` | Refined text prompts per class name for SAM3 text verification |
| `include_missing_detection` | bool | `true` | Run SAM3 auto_mask to detect unannotated objects |
| `config` | object | `{}` | Optional threshold overrides |
| `enable_vlm` | bool | `false` | Enable VLM semantic verification via Ollama |
| `vlm_trigger` | string | `"selective"` | When to run VLM: `all`, `selective` (only review/bad), `standalone` |

**Response:**
```json
{
  "issues": [
    {
      "type": "misclassified",
      "severity": "high",
      "annotation_idx": 0,
      "detail": "SAM3 text verification found no matching class for annotation 0"
    }
  ],
  "sam3_verification": {
    "box_ious": [0.85],
    "mask_ious": [],
    "mean_box_iou": 0.85,
    "mean_mask_iou": 0.0,
    "misclassified": [],
    "missing_detections": []
  },
  "num_annotations": 1,
  "num_issues": 0,
  "quality_score": 0.92,
  "grade": "good",
  "suggested_fixes": [],
  "label_format": "yolo",
  "processing_time_s": 2.341,
  "vlm_verification": null
}
```

When `enable_vlm=true` and VLM runs, the response includes:
```json
{
  "vlm_verification": {
    "crop_verification": {
      "results": [
        {"annotation_idx": 0, "class_name": "fire", "is_correct": true, "confidence": 0.95, "reason": "clearly shows flames"}
      ],
      "num_checked": 1,
      "num_incorrect": 0,
      "mean_confidence": 0.95
    },
    "scene_verification": {
      "incorrect_indices": [],
      "missing_descriptions": [],
      "quality_score": 0.9,
      "raw_response": "INCORRECT: NONE\nMISSING: NONE\nQUALITY: 0.9"
    },
    "available": true
  }
}
```

---

### POST /fix

Apply suggested fixes to labels. Accepts the `suggested_fixes` from a previous `/validate` or `/verify` call.

```bash
curl -X POST http://localhost:18105/fix \
  -H "Content-Type: application/json" \
  -d '{
    "labels": ["0 1.05 0.5 0.2 0.3", "0 0.5 0.5 0.2 0.2"],
    "label_format": "yolo",
    "classes": {"0": "fire"},
    "suggested_fixes": [
      {
        "type": "clip_bbox",
        "annotation_idx": 0,
        "suggested": {"bbox_norm": [0.95, 0.5, 0.1, 0.3]}
      }
    ]
  }'
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `labels` | array | required | Original labels |
| `label_format` | string | `"yolo"` | Label format |
| `classes` | object | required | Class mapping |
| `suggested_fixes` | array | `[]` | Fix suggestions from validate/verify |
| `auto_apply` | array | `["clip_bbox", "remove_duplicate", "remove_degenerate"]` | Fix types to auto-apply |

**Response:**
```json
{
  "corrected_labels": ["0 0.950000 0.500000 0.100000 0.300000", "0 0.500000 0.500000 0.200000 0.200000"],
  "applied_fixes": [
    {"type": "clip_bbox", "annotation_idx": 0, "suggested": {"bbox_norm": [0.95, 0.5, 0.1, 0.3]}}
  ],
  "needs_review": [],
  "num_applied": 1,
  "num_needs_review": 0,
  "num_annotations_before": 2,
  "num_annotations_after": 2
}
```

---

### POST /jobs

Create an asynchronous batch QA job for multiple images.

```bash
curl -X POST http://localhost:18105/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "images": [
      {
        "image": "<BASE64_IMAGE_1>",
        "filename": "img_001.jpg",
        "labels": ["0 0.5 0.5 0.3 0.4"]
      },
      {
        "image": "<BASE64_IMAGE_2>",
        "filename": "img_002.jpg",
        "labels": ["1 0.3 0.6 0.2 0.25"]
      }
    ],
    "classes": {"0": "fire", "1": "smoke"},
    "mode": "validate",
    "label_format": "yolo"
  }'
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `images` | array | required | Array of `{image, filename, labels}` objects |
| `classes` | object | required | Class mapping |
| `mode` | string | `"verify"` | QA mode: `validate` or `verify` |
| `label_format` | string | `"yolo"` | Label format |
| `text_prompts` | object | `{}` | Text prompts for verify mode |
| `include_missing_detection` | bool | `false` | Detect unannotated objects (verify mode) |
| `config` | object | `{}` | Optional threshold overrides |
| `webhook_url` | string | `null` | URL to POST when job completes |
| `enable_vlm` | bool | `false` | Enable VLM verification |
| `vlm_trigger` | string | `"selective"` | VLM trigger mode |

**Response:**
```json
{
  "job_id": "a1b2c3d4e5f6",
  "total_images": 2,
  "status": "queued"
}
```

---

### GET /jobs

List all QA jobs.

```bash
curl http://localhost:18105/jobs
curl http://localhost:18105/jobs?status=completed
```

**Response:**
```json
[
  {
    "job_id": "a1b2c3d4e5f6",
    "status": "completed",
    "total_images": 2,
    "processed_images": 2,
    "created_at": 1711000000.0
  }
]
```

---

### GET /jobs/{job_id}

Get the status and results of a specific QA job.

```bash
curl http://localhost:18105/jobs/a1b2c3d4e5f6
```

**Response (completed):**
```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "completed",
  "total_images": 2,
  "processed_images": 2,
  "results": [
    {
      "filename": "img_001.jpg",
      "quality_score": 0.92,
      "grade": "good",
      "num_annotations": 1,
      "num_issues": 0,
      "issues": [],
      "suggested_fixes": []
    },
    {
      "filename": "img_002.jpg",
      "quality_score": 0.65,
      "grade": "review",
      "num_annotations": 1,
      "num_issues": 1,
      "issues": [
        {"type": "duplicate", "severity": "medium", "annotation_idx": 1, "detail": "Annotation 1 duplicates 0 (class=1, IoU=0.970)"}
      ],
      "suggested_fixes": [
        {"type": "remove_duplicate", "annotation_idx": 1, "original": {"class_id": 1, "bbox_norm": [0.301, 0.601, 0.2, 0.25]}, "suggested": null, "reason": "Duplicate annotation detected."}
      ]
    }
  ],
  "error": null,
  "created_at": 1711000000.0
}
```

---

### DELETE /jobs/{job_id}

Cancel a running or queued job.

```bash
curl -X DELETE http://localhost:18105/jobs/a1b2c3d4e5f6
```

**Response:**
```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "cancelled"
}
```

---

### POST /report

Generate a summary report from batch QA results.

```bash
curl -X POST http://localhost:18105/report \
  -H "Content-Type: application/json" \
  -d '{
    "results": [
      {"filename": "a.jpg", "quality_score": 0.9, "grade": "good", "issues": [], "suggested_fixes": [], "num_issues": 0, "num_annotations": 3},
      {"filename": "b.jpg", "quality_score": 0.4, "grade": "bad", "issues": [{"type": "out_of_bounds"}], "suggested_fixes": [], "num_issues": 1, "num_annotations": 2}
    ],
    "classes": {"0": "fire", "1": "smoke"},
    "dataset_name": "fire_v2"
  }'
```

**Response:**
```json
{
  "dataset": "fire_v2",
  "total_checked": 2,
  "grades": {"good": 1, "review": 0, "bad": 1},
  "avg_quality_score": 0.65,
  "issue_breakdown": {"out_of_bounds": 1},
  "per_class_stats": {"fire": {"count": 3, "issues": 0}, "smoke": {"count": 2, "issues": 0}},
  "worst_images": [
    {"filename": "b.jpg", "quality_score": 0.4, "grade": "bad", "num_issues": 1}
  ],
  "auto_fixable_count": 0
}
```

---

## Scoring

The overall quality score for each image is computed as a weighted penalty formula:

```
score = 1.0 - (w_structural * structural_penalty
             + w_bbox * bbox_penalty
             + w_classification * classification_penalty
             + w_coverage * coverage_penalty
             + w_vlm * vlm_penalty)
```

Default weights (from `configs/default.yaml`):

| Component | Weight | Description |
|-----------|--------|-------------|
| `structural` | 0.25 | Issues found / num_annotations |
| `bbox_quality` | 0.30 | 1 - mean(box_ious) from SAM3 verification |
| `classification` | 0.15 | Misclassified / num_annotations via SAM3 text check |
| `coverage` | 0.15 | Missing detections / (num_annotations + missing) |
| `vlm` | 0.15 | VLM crop + scene penalty (when enabled) |

Grades based on thresholds:

| Grade | Score Range | Action |
|-------|------------|--------|
| `good` | >= 0.8 | No action needed |
| `review` | 0.5 -- 0.8 | Manual review recommended |
| `bad` | < 0.5 | Re-annotation likely needed |

## Fix Types

| Fix Type | Description | Auto-applied | When Generated |
|----------|-------------|:---:|----------------|
| `clip_bbox` | Clip coordinates to [0, 1] | Yes | Out-of-bounds bbox |
| `remove_duplicate` | Remove duplicate annotation | Yes | Same-class pair with IoU > 0.95 |
| `remove_degenerate` | Remove tiny annotation | Yes | Box width or height < 0.005 |
| `tighten_bbox` | SAM3 suggests tighter bbox | No | Box IoU with SAM3 < 0.6 |
| `remove_annotation` | Remove misclassified | No | SAM3 text verification found no match |
| `vlm_flagged` | VLM disagrees with label | No | VLM crop/scene verification says incorrect |

## Label Formats

### YOLO bbox
Labels as strings: `"class_id cx cy w h"` (all normalized 0-1).
```json
["0 0.785 0.359 0.338 0.312", "1 0.234 0.567 0.100 0.200"]
```

### YOLO-seg
Labels as strings: `"class_id x1 y1 x2 y2 ... xN yN"` (normalized polygon).
```json
["0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8"]
```

### COCO
Labels as dicts with pixel-coordinate bounding boxes `[x, y, w, h]` and optional segmentation.
```json
[{"category_id": 0, "bbox": [120, 45, 230, 180], "segmentation": [[100, 50, 300, 50, 300, 200, 100, 200]]}]
```

## Quick Start

```bash
# Start the service (structural validation only, no dependencies)
cd services/s18105_annotation_quality_assessment && docker compose up -d

# Health check
curl http://localhost:18105/health

# For SAM3 verification, also start SAM3
cd services/s18100_sam3_service && docker compose up -d

# For VLM verification, also start Ollama
ollama serve  # or run via Docker
ollama pull qwen3.5:9b
```

## Tests

```bash
cd services/s18105_annotation_quality_assessment

# Unit tests only (VLM parsers, crop extraction — no service needed)
uv sync --extra test
uv run pytest tests/test_vlm.py -v -k "not Integration"

# All tests (requires service running at :18105)
uv run pytest tests/ -v
```

## Configuration

The service reads `configs/default.yaml`. Key sections:

| Section | Description |
|---------|-------------|
| `server` | Host and port binding |
| `services` | SAM3 service URL, Ollama URL |
| `processing` | Request timeout |
| `validation` | Structural check thresholds (min/max box size, duplicate IoU, aspect ratio, polygon rules) |
| `sam3` | SAM3 verification thresholds (text verify confidence, auto mask area range) |
| `vlm` | VLM verification settings: model name, trigger mode (selective/all/standalone), crop padding, prompt templates, request timeout |
| `scoring` | Component weights (including VLM weight) and grade thresholds |
| `jobs` | Batch job concurrency, storage limits, TTL, max images per job |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SAM3_URL` | `http://localhost:18100` | SAM3 service URL (overrides config) |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama URL for VLM verification (overrides config) |
| `ANNOTATION_QA_CONFIG` | `configs/default.yaml` | Path to config YAML file |

## Docker

```bash
# Build and start
cd services/s18105_annotation_quality_assessment
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

The service is CPU-only and does not require GPU. SAM3 verification requires `sam3_service` running at `:18100`.

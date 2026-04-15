# Label Studio — Annotation Review Service

Docker-based [Label Studio](https://labelstud.io/) instance for visually reviewing and correcting annotations. It is the **human-in-the-loop** step between automated labeling and model training.

## Pipeline Overview

Label Studio sits at the intersection of three upstream pipelines and one downstream consumer:

```
                         UPSTREAM SOURCES
              ┌───────────────────────────────────────┐
              │                                       │
              │  ┌─────────────┐  ┌────────────────┐  │
              │  │  Auto-Label │  │  Annotation QA │  │
              │  │   (s18104)  │  │    (s18105)    │  │
              │  │  SAM3-based │  │  structural +  │  │
              │  │  detection  │  │  SAM3 verify   │  │
              │  └──────┬──────┘  └───────┬────────┘  │
              │         │                 │            │
              │         ▼                 ▼            │
              │  runs/auto_annotate/   runs/qa/*/fixes.json
              │         │                 │            │
              │         │    ┌────────────┘            │
              │         ▼    ▼                         │
              │  bridge.py import                     │
              │         │                              │
              │         ▼                              │
              │  ┌──────────────────┐                  │
              │  │   Label Studio   │  ← human review  │
              │  │     (s18103)     │                  │
              │  │  :18103 (CPU)    │                  │
              │  └────────┬─────────┘                  │
              │           │                            │
              │           ▼                            │
              │  bridge.py export                      │
              │           │                            │
              │           ▼                            │
              │  dataset_store/<usecase>/<split>/labels/
              │           │                            │
              └───────────┼────────────────────────────┘
                          │
                          ▼
                   DOWNSTREAM CONSUMER
              ┌────────────────────┐
              │   Training         │
              │   (p03_training)   │
              └────────────────────┘
```

### Data Flow

1. **Auto-Label** (`s18104`) calls SAM3 to generate bounding boxes/polygons → writes YOLO `.txt` files to `runs/auto_annotate/<dataset>/`
2. **Annotation QA** (`s18105`) flags problematic labels → writes `fixes.json`
3. **Bridge CLI** (`core/p04_label_studio/bridge.py`) pushes those labels into Label Studio as pre-annotations
4. **Human reviewer** accepts, adjusts, or rejects each annotation in the Label Studio web UI
5. **Bridge CLI** exports reviewed annotations back to YOLO `.txt` files in `dataset_store/`
6. **Training** pipeline reads the reviewed labels from `dataset_store/`

### How Images Are Served

Images are **not uploaded** into Label Studio. The `dataset_store/` directory is mounted read-only into the container at `/datasets`, and Label Studio serves them directly via local file storage. This keeps the dataset as the single source of truth.

## Quick Start

```bash
# Start Label Studio (CPU only, no GPU required)
cd services/s18103_label_studio && docker compose up -d

# Verify it's running
curl http://localhost:18103/health

# Open in browser — create an admin account on first visit
xdg-open http://localhost:18103
```

## Workflow: Review Auto-Annotations

### Step 1 — Start Required Services

```bash
# SAM3 (GPU required, ~5GB VRAM)
cd services/s18100_sam3_service && docker compose up -d

# Auto-Label orchestrator (CPU, calls SAM3)
cd services/s18104_auto_label && docker compose up -d

# Label Studio review UI (CPU)
cd services/s18103_label_studio && docker compose up -d
```

### Step 2 — Run Auto-Annotation

```bash
# Using a data config (YOLO dataset layout)
uv run core/p01_auto_annotate/run_auto_annotate.py \
  --data-config features/safety-fire_detection/configs/05_data.yaml \
  --mode text

# Or on a flat image directory with ad-hoc classes
uv run core/p01_auto_annotate/run_auto_annotate.py \
  --image-dir /path/to/images \
  --classes "0:fire,1:smoke" \
  --mode text

# Polygon output (instance segmentation)
uv run core/p01_auto_annotate/run_auto_annotate.py \
  --data-config features/ppe-shoes_detection/configs/05_data.yaml \
  --mode text --output-format polygon

# Dry-run first to preview without writing
uv run core/p01_auto_annotate/run_auto_annotate.py \
  --data-config features/safety-fire_detection/configs/05_data.yaml --dry-run --mode text
```

Output is written to `runs/auto_annotate/<dataset_name>/`:
- `labels/*.txt` — YOLO label files (one per image)
- `report.json` — detection counts, per-class stats, timing
- `summary.txt` — human-readable summary
- `preview/` — sample images with drawn bounding boxes

### Step 3 — Set Up a Label Studio Project

```bash
uv run core/p04_label_studio/bridge.py setup \
  --data-config features/safety-fire_detection/configs/05_data.yaml
```

This creates a project named `<dataset_name>_review` (e.g. `fire_detection_review`) with the correct class labels and colors configured. Images are linked via local file storage — no files are copied.

### Step 4 — Import Annotations for Review

```bash
# Import SAM3 auto-annotations as pre-annotations
uv run core/p04_label_studio/bridge.py import \
  --data-config features/safety-fire_detection/configs/05_data.yaml \
  --from-auto-annotate runs/auto_annotate/fire_detection/
```

Each image appears in Label Studio with pre-filled bounding boxes. Reviewers can:
- **Accept** correct annotations as-is
- **Adjust** bounding box positions/sizes
- **Delete** false positives
- **Add** missed objects

### Step 5 — Export Reviewed Annotations

```bash
# Export only human-reviewed annotations back to YOLO format
uv run core/p04_label_studio/bridge.py export \
  --project fire_detection_review \
  --output-dir ../../dataset_store/fire_detection/train/labels \
  --data-config features/safety-fire_detection/configs/05_data.yaml \
  --only-reviewed --backup
```

The `--backup` flag saves existing labels with a timestamp before overwriting. The `--only-reviewed` flag ensures only human-verified annotations are exported (skips tasks that were never opened).

## Workflow: Review Existing Training Data

Import labels already in `dataset_store/` directly (no auto-annotation step needed):

```bash
# Set up project (same as above)
uv run core/p04_label_studio/bridge.py setup \
  --data-config features/ppe-shoes_detection/configs/05_data.yaml

# Import from specific splits
uv run core/p04_label_studio/bridge.py import \
  --data-config features/ppe-shoes_detection/configs/05_data.yaml \
  --splits train val

# After reviewing in the browser, export back
uv run core/p04_label_studio/bridge.py export \
  --project shoes_detection_review \
  --output-dir ../../dataset_store/shoes_detection/train/labels \
  --data-config features/ppe-shoes_detection/configs/05_data.yaml \
  --only-reviewed --backup
```

## Workflow: Review QA-Flagged Images

Import only images that the Annotation QA service (`s18105`) flagged as problematic:

```bash
# Run QA first (requires SAM3 at :18100)
cd services/s18105_annotation_quality_assessment && docker compose up -d
uv run core/p02_annotation_qa/run_qa.py \
  --data-config features/ppe-shoes_detection/configs/05_data.yaml

# Import only the flagged fixes into Label Studio
uv run core/p04_label_studio/bridge.py import \
  --data-config features/ppe-shoes_detection/configs/05_data.yaml \
  --from-qa-fixes runs/qa/shoes_detection/fixes.json
```

## Configuration

All settings live in `configs/_shared/08_label_studio.yaml`:

```yaml
label_studio:
  url: "http://localhost:8080"         # Container-internal URL (host uses :18103)
  api_key: ""                          # Or set LS_API_KEY env var, or --api-key CLI arg
  local_files_root: "/datasets"        # Must match docker-compose volume mount
  project_suffix: "_review"            # Appended to dataset_name for project naming
  import:
    batch_size: 100                    # Tasks per API call
    include_predictions: true          # Show pre-annotations in the UI
  export:
    only_reviewed: true                # Skip tasks without human annotations
    backup: true                       # Backup existing labels before overwrite
    format: "yolo"                     # Output format
  colors: [...]                        # Label colors (cycles if more classes than colors)
```

**API key priority:** `--api-key` CLI arg > `LS_API_KEY` env var > config file value.

## Volume Mounts

| Host Path | Container Path | Access | Purpose |
|-----------|---------------|--------|---------|
| Docker named volume `ls-data` | `/label-studio/data` | read-write | LS database, user data, exports |
| `../../dataset_store` | `/datasets` | read-only | Dataset images served via local file storage |
| `../../tests/fixtures/data` | `/dataset_store/test_fixtures` | read-only | Test fixtures for E2E tests |

## Stopping

```bash
cd services/s18103_label_studio && docker compose down
```

Data persists in the Docker named volume `ls-data` between restarts. To remove it entirely: `docker compose down -v`.

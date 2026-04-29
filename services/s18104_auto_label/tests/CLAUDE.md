# CLAUDE.md — services/s18104_auto_label/tests/

Integration tests for the auto_label service. Self-contained — run from within `services/s18104_auto_label/`, no imports from the project root.

## What these tests verify

- **Every one of the 11 REST endpoints is reachable and returns the documented schema** (see Endpoint Coverage table below) — no mocks, real HTTP to a live service.
- **All 4 output formats round-trip cleanly**: `coco`, `yolo`, `yolo_seg`, `label_studio` — annotate → convert → re-annotate produces equivalent structure.
- **SAM3 delegation works end-to-end**: each `/annotate` mode (text / auto / hybrid) actually calls SAM3 at `:18100` and post-processes (NMS, polygon extraction) locally.
- **Video session lifecycle is correct**: create → add frames → propagate → delete. Frame-size invariance is enforced (SAM3 tracker reinitializes on each frame add).
- **Graceful skip when service is down**: `skip_no_service` marker auto-skips the whole suite if `GET /health` fails — CI stays green when SAM3 or auto_label isn't up.

## Prerequisites

- auto_label service running at `localhost:18104`
- SAM3 service running at `localhost:18100`

```bash
cd services/s18100_sam3_service && docker compose up -d
cd services/s18104_auto_label && docker compose up -d
```

## Running Tests

```bash
cd services/s18104_auto_label

# Install test dependencies
uv sync --extra test

# Run all tests
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/test00_health.py -v
uv run pytest tests/test01_annotate.py -v
uv run pytest tests/test02_jobs.py -v
uv run pytest tests/test03_video.py -v
uv run pytest tests/test04_convert.py -v
uv run pytest tests/test05_parallel_batch.py -v
uv run pytest tests/test06_rule_classify_vlm.py -v
uv run pytest tests/test_real_data.py -v
```

Tests auto-skip if the service is not running.

## Test Files

| File | Endpoints / Coverage |
|------|----------------------|
| `test00_health.py` | `GET /health` |
| `test01_annotate.py` | `POST /annotate` (text mode, all 4 output formats, invalid mode/format, overlay) |
| `test02_jobs.py` | `POST /jobs`, `GET /jobs`, `GET /jobs/{id}`, `DELETE /jobs/{id}` |
| `test03_video.py` | `POST /video/sessions`, `POST /video/sessions/{id}/frames`, `POST /video/sessions/{id}/propagate`, `DELETE /video/sessions/{id}` |
| `test04_convert.py` | `POST /convert` (all 4 formats, empty, invalid, roundtrip from annotate) |
| `test05_parallel_batch.py` | Parallel batch annotation throughput / job-queue concurrency |
| `test06_rule_classify_vlm.py` | Rule-based classification + VLM-assisted annotation modes |
| `test_real_data.py` | End-to-end smoke against real `data/` images (not a synthetic fixture) |

## Endpoint Coverage

All 11 endpoints are covered:

| # | Method | Path | Test File |
|---|--------|------|-----------|
| 1 | GET | `/health` | `test00_health.py` |
| 2 | POST | `/annotate` | `test01_annotate.py` |
| 3 | POST | `/jobs` | `test02_jobs.py` |
| 4 | GET | `/jobs` | `test02_jobs.py` |
| 5 | GET | `/jobs/{id}` | `test02_jobs.py` |
| 6 | DELETE | `/jobs/{id}` | `test02_jobs.py` |
| 7 | POST | `/video/sessions` | `test03_video.py` |
| 8 | POST | `/video/sessions/{id}/frames` | `test03_video.py` |
| 9 | POST | `/video/sessions/{id}/propagate` | `test03_video.py` |
| 10 | DELETE | `/video/sessions/{id}` | `test03_video.py` |
| 11 | POST | `/convert` | `test04_convert.py` |

## Shared Helpers

`conftest.py` provides:
- `skip_no_service` — pytest mark that skips tests when service is unreachable
- `load_image_b64(filename)` — load test image from `data/` as base64
- `detections_to_sv(detections, w, h)` — convert API detections to `sv.Detections`
- `annotate_image(img, detections, class_names)` — draw bboxes/masks/labels with supervision
- `SERVICE_URL`, `DATA_DIR`, `OUTPUT_DIR` — shared constants

## Test Data

`tests/data/` contains real images and video for integration tests:
- `fire_sample_{1,2,3}.jpg` — fire detection test images
- `indoor_fire.mp4` — short video for video session tests

## Outputs

Test visualizations (overlays, JSON responses) are saved to `tests/outputs/` (gitignored). Delete between runs to force a clean snapshot: `rm -rf tests/outputs/`.

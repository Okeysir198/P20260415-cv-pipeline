# Tests — Flat Numbered Test Suite

Tests are numbered by pipeline phase. Core tests 08→10→12→14 run sequentially (later tests depend on earlier outputs). All other tests are independent.

## Test Categories

| Tests | Pipeline Phase | Scope | Dependencies |
|---|---|---|---|
| `test_utils00-09` | — | Config/device/metrics (00), supervision metrics (01), exploration (02), scaffold (03), keypoint geometry (04), langgraph reducers (05), progress bars (06), paddle_bridge YOLO→COCO (07), release pipeline (08), yolo_io (09) | None |
| `test_core00` | p00 prep | Data conversion (VOC → YOLO) | None |
| `test_core01-05` | p01 data | Detection, classification, segmentation, keypoint, COCO datasets | None |
| `test_core06-07` | p02 models | Model registry, model variants | None |
| `test_core08-09` | p03 training | Training loop, losses, callbacks, schedulers, EMA | test_core01 |
| `test_core10-11` | p04 evaluation | Evaluation metrics, error analysis | test_core08 |
| `test_core12-13` | p05 export | ONNX export, quantization, numerical validation | test_core08 |
| `test_core14-15` | p06 inference | Predictor, video inference, alerts, tracking | test_core08 + test_core12 |
| `test_core16` | p02+p03 | Classification/segmentation training (timm, HF) | None |
| `test_core17` | p02+p06 | Face recognition (SCRFD, MobileFaceNet) | None |
| `test_tools00-07` | — | Exploration, data prep, preview, QA, auto-annotate, augment, HPO, scaffold | None |

## Running Tests

```bash
make test              # All tests
make test-core         # Core pipeline only
make test-tools        # Tools only
make test-utils        # Utils only

# Individual file
uv run pytest tests/test_core08_training.py -v

# Full sequential pipeline
uv run tests/run_all.py
```

## Test Fixtures

Real images in `tests/fixtures/data/` (15 fire images with YOLO labels). Never use synthetic/random images for tests that touch model inference.

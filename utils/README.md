# Utils — Task-Agnostic Shared Utilities

Generic utilities used across the entire pipeline. Only task-agnostic code lives here. Detection-specific modules have been relocated:

| Module | Now in | Was |
|---|---|---|
| `sv_metrics.py` | `core/p08_evaluation/` | `utils/` |
| `visualization.py` | `core/p08_evaluation/` | `utils/` |
| `supervision_bridge.py` | `core/p10_inference/` | `utils/` |
| `bridge.py` | `core/p04_label_studio/` | `utils/` |

## Files

| File | Purpose |
|---|---|
| `config.py` | YAML load/merge/validate, `${var}` interpolation, `resolve_path()` |
| `device.py` | `get_device()` (CUDA/MPS/CPU), `set_seed()`, `get_gpu_info()` |
| `progress.py` | Progress bars for training and data processing |
| `metrics.py` | Generic bbox geometry: `compute_iou()`, `nms_numpy()`, `cxcywh_to_xyxy()` |
| `keypoint_utils.py` | Keypoint geometry utilities for pose estimation |
| `service_health.py` | Health check helpers for microservices |
| `validate_config.py` | CLI tool to validate YAML configs before training (exit 0 = OK) |

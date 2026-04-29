# safety-poketenashi_stair_diagonal

Single-rule pose feature: detects unsafe **diagonal** stair traversal — a worker
ascending/descending stairs whose hip-midpoint trajectory deviates from the
stair axis (assumed roughly horizontal in the camera frame) by more than a
configurable angle threshold over a multi-frame window.

This rule was split out from the umbrella `features/safety-poketenashi/` so it
can be deployed, calibrated, and benchmarked independently of the other 4
poketenashi rules.

| Field | Value |
|-------|-------|
| Task | Pose estimation + per-track stateful rule |
| Pose backend | DWPose-L 384x288 ONNX (`pretrained/safety-poketenashi/dw-ll_ucoco_384.onnx`) |
| Person detector | YOLO11n (`pretrained/access-zone_intrusion/yolo11n.pt`) |
| Training | None — pretrained-only orchestrator |
| Stateful? | **Yes** — buffers hip positions across `min_frames`; tracking required for multi-person |

## Run

```bash
# Synthetic-keypoint smoke (no media)
uv run features/safety-poketenashi_stair_diagonal/code/predictor.py --smoke-test

# Single-track inference on a video
uv run features/safety-poketenashi_stair_diagonal/code/predictor.py \
  --video path/to/clip.mp4

# Tests
uv run -m pytest features/safety-poketenashi_stair_diagonal/tests/ -v
```

## Config knobs

`configs/10_inference.yaml::pose_rules.stair_diagonal:`

| Key | Default | Meaning |
|---|---|---|
| `max_diagonal_angle_deg` | `20` | Trajectory deviation from horizontal that fires the rule |
| `min_frames` | `5` | Hip positions required before evaluating trajectory |

See `CLAUDE.md` for deployment architecture and site-calibration notes.

# safety-poketenashi_stair_diagonal

**Type:** Pose orchestrator (single-rule) | **Training:** Pretrained only (no own model)

## Overview

Stateful pose rule that flags workers traversing stairs diagonally. The rule
buffers per-track mid-hip positions across `min_frames` (default 5) consecutive
frames, then computes the trajectory angle from the oldest to the newest
position. If the deviation from horizontal (assumed stair axis) exceeds
`max_diagonal_angle_deg` (default 20°), the rule fires.

Source: split from `features/safety-poketenashi/code/stair_safety_detector.py`
(`StairSafetyDetector`, behavior=`stair_diagonal`). The detector class is copied
verbatim — the umbrella feature still owns the original until U11 retires it.

## Pipeline Checklist

- [x] `code/_base.py` — `PoseRule` base class + `RuleResult` dataclass (copied)
- [x] `code/stair_safety_detector.py` — diagonal trajectory rule (copied verbatim)
- [x] `code/predictor.py` — DWPose ONNX + per-track `StairSafetyDetector` orchestrator
- [x] `code/benchmark.py` — minimal pose-latency benchmark on samples
- [x] `tests/test_stair_safety.py` — synthetic-keypoint trajectory tests
- [ ] Tune `max_diagonal_angle_deg` per camera angle on field samples
- [ ] Add real samples to `samples/` for benchmark + smoke runs

## Deployment Architecture

### Single-camera pipeline
Frame ingestion → person detector (YOLO11n at `pretrained/access-zone_intrusion/yolo11n.pt`) → top-down crop → DWPose ONNX (shared at `pretrained/safety-poketenashi/dw-ll_ucoco_384.onnx`) → COCO-17 keypoints → `StairSafetyDetector` (per-track) → trajectory-angle check over a 5-frame buffer → triggered flag → event sink.

### Stateful rule — track persistence required
This rule buffers hip-midpoint positions per person across `min_frames` (default 5) frames. In multi-person deployments you MUST use ByteTrack so each worker's `StairSafetyDetector` instance keeps its own buffer. Without tracking, the buffer mixes positions from different people and fires false positives.

### When to enable ByteTrack
ALWAYS for this rule when more than one person can be in frame. Configured in `configs/10_inference.yaml::tracker:`; wire by passing `VideoProcessor(enable_tracking=True, tracker_config=cfg["tracker"])` — see `core/p10_inference/video_inference.py:166-172`.

### When you need person detector + pose detector together
DWPose top-down requires person boxes first. Required here for camera angles where stairs span < 60% of frame.

### Shared pose backbone
Same DWPose ONNX as the other `safety-poketenashi_*` rules. See `features/CLAUDE.md` for the multi-feature deployment recipe.

### Site calibration
- `max_diagonal_angle_deg` (default 20°): camera-angle dependent. Steeper camera angles mean apparent trajectories tilt further from horizontal.
- `min_frames` (default 5): trajectory window. Higher = lower FP rate but slower response.
- The rule assumes the stair axis is roughly horizontal in the camera frame. For tilted cameras, calibrate by recording someone walking straight up and measuring the apparent angle.

## Key Files

```
configs/05_data.yaml             — single-class (stair_diagonal) rule manifest
configs/10_inference.yaml        — alert thresholds + pose_rules.stair_diagonal + bytetrack stub
code/_base.py                    — PoseRule + RuleResult (copied from umbrella)
code/stair_safety_detector.py    — verbatim copy from umbrella (~101 lines)
code/predictor.py                — DWPose + per-track StairSafetyDetector orchestrator
code/benchmark.py                — pose-latency micro-benchmark on samples
tests/test_stair_safety.py       — synthetic trajectory tests (4 cases)
```

## Notes

- `code/` may import from `core/` and `utils/` only — never another feature's `code/`.
- The detector source remains in `features/safety-poketenashi/code/stair_safety_detector.py`. U11 owns retiring the umbrella.
- Pose model is shared with the other `safety-poketenashi_*` rule splits — load once if running them together.
- Person detector path: `pretrained/access-zone_intrusion/yolo11n.pt` (falls back to whole-frame if absent).
- Never use bare `YOLO("model.pt")` — Ultralytics auto-downloads to cwd; always use an absolute path from `pretrained/`.

# safety-point_and_call

**Type:** Pose orchestrator | **Training:** 🔧 Pretrained only (v1) — rule-based on top of pose keypoints

## Overview

Detects the Japanese **指差呼称** (*shisa-kanko*, "point-and-call") crosswalk gesture:
worker stops at the curb, points **right** ("右ヨシ!"), points **left** ("左ヨシ!"),
optionally points **front** ("前ヨシ!"), then crosses. v1 pipeline:

```
image_bgr ─► person detector ─► pose backend ─► COCO-17 keypoints
                                                       │
                                                       ▼
                              PointingDirectionDetector (per-frame, geometric)
                                                       │
                                          label ∈ {point_left, point_right,
                                                   point_front, neutral, invalid}
                                                       │
                                                       ▼
                              CrosswalkSequenceMatcher (temporal state machine)
                                                       │
                                                       ▼
                              alerts: [point_and_call_done | missing_directions]
```

Distinct from `features/safety-poketenashi/code/pointing_calling_detector.py`,
which only checks whether *any* arm is extended horizontally (binary,
non-directional, no L→R→F sequence).

## Pose backend (swappable)

The orchestrator never touches a specific pose model — it asks an interchangeable
adapter for `(keypoints_17, scores_17, person_box)` per detected person. Switch
backends by changing `pose.backend` in `configs/10_inference.yaml`:

| `pose.backend` | Adapter | Implementation |
|---|---|---|
| `dwpose_onnx` *(default)* | `_DWPoseAdapter` in `code/pose_backend.py` | Inlined ONNX SimCC wrapper, ~60 LOC. Uses `pretrained/safety-poketenashi/dw-ll_ucoco_384.onnx`, slices first 17 of 133 wholebody keypoints. |
| `rtmpose` | `_GenericPoseAdapter` | Delegates to `core.p10_inference.pose_predictor.PosePredictor` loaded from `pose.config` YAML. |
| `mediapipe` | `_GenericPoseAdapter` | Same path, MediaPipe YAML. |
| `hf_keypoint` | `_GenericPoseAdapter` | Same path, ViTPose-family YAML. |

Adding a fifth backend = one ~50-LOC adapter + one branch in
`build_pose_backend()`. No orchestrator or rule changes.

Person detector (top-down crops): `pretrained/access-zone_intrusion/yolo11n.pt`,
shared across backends. Same constraint as `safety-poketenashi`: never use bare
`YOLO("model.pt")` (Ultralytics auto-downloads to cwd) — always pass an explicit
absolute path from `pretrained/`.

## Quick Start

```bash
# Smoke-test the orchestrator (default DWPose backend)
cp features/safety-poketenashi/samples/*.jpg features/safety-point_and_call/samples/
uv run features/safety-point_and_call/code/orchestrator.py --smoke-test
# Writes eval/orchestrator_smoke_test.json + eval/smoke_*.jpg

# Smoke-test with a different backend (verifies the swap)
uv run features/safety-point_and_call/code/orchestrator.py --smoke-test --pose-backend rtmpose

# Latency benchmark across all 4 backends
uv run features/safety-point_and_call/code/benchmark.py
# Writes eval/benchmark_results.json

# Unit tests (synthetic keypoints — no GPU/data needed)
uv run -m pytest features/safety-point_and_call/tests/ -v

# Demo tab (Image + Video sub-tabs, pose-backend dropdown)
uv run app_demo/run.py
# Open http://localhost:7861 → "Point & Call" tab
```

## v1 Pipeline Status

- [x] Feature scaffold + docs (`README.md`, this `CLAUDE.md`, `.gitignore`,
      empty `samples/ eval/ predict/ export/ notebooks/`).
- [x] Configs (`configs/05_data.yaml`, `configs/10_inference.yaml`,
      `configs/pose_dwpose.yaml`).
- [x] Code: pose-backend abstraction + per-frame direction classifier +
      temporal sequence matcher + orchestrator + benchmark CLI.
- [x] Unit tests (22 passing): `test_pose_backend.py`,
      `test_pointing_direction_detector.py`,
      `test_crosswalk_sequence_matcher.py`.
- [x] Gradio demo tab + registration in `app_demo/config/config.yaml`.
- [ ] Tune sequence-matcher windows + per-frame thresholds against field clips
      (smoke samples are factory-worker frames, not crosswalk gestures —
      tuning needs real footage).
- [ ] Optional `cross_zone:` polygon configured per deployment site.

## Files

```
configs/
  05_data.yaml              5 viz classes (point_left/right/front, neutral, invalid) + dataset_name
  10_inference.yaml         pose: backend selector + thresholds + sequence params + alerts
  pose_dwpose.yaml          standalone DWPose YAML (parallel handle for app_demo)

code/
  _base.py                  PoseRule + RuleResult (self-contained — does NOT import safety-poketenashi)
  _geometry.py              elbow_angle_deg, arm_elevation_deg, torso_frame_basis,
                            arm_azimuth_torso_frame — all pure numpy
  pose_backend.py           PoseBackend Protocol + build_pose_backend() factory +
                            _DWPoseAdapter, _GenericPoseAdapter, _PersonDetector
  pointing_direction_detector.py
                            PointingDirectionDetector(PoseRule) — operates only on
                            (kpts17, scores17), backend-agnostic
  crosswalk_sequence_matcher.py
                            CrosswalkSequenceMatcher — sliding-window run-length
                            matcher, supports modes LR/RL/LRF/RLF/LFR/RFL/FLR/FRL,
                            cooldown after success
  orchestrator.py           PointAndCallOrchestrator + dataclasses + draw() +
                            --smoke-test CLI with --pose-backend override
  benchmark.py              Loops 4 backends; per-backend latency + det_rate,
                            graceful error capture per backend

tests/
  test_pose_backend.py
  test_pointing_direction_detector.py
  test_crosswalk_sequence_matcher.py

samples/                    point-and-call clips / frames (curate per deployment site)
eval/orchestrator_smoke_test.json     gitignored — written by --smoke-test
eval/benchmark_results.json           gitignored — written by benchmark.py
```

## Config knobs (`configs/10_inference.yaml`)

```yaml
pose:
  backend: dwpose_onnx                  # swap here — see "Pose backend" above
  weights: ../../../pretrained/...      # for dwpose_onnx
  config: null                          # YAML path for rtmpose/mediapipe/hf_keypoint
  person_detector: ../../../pretrained/access-zone_intrusion/yolo11n.pt
  person_conf: 0.35

pose_rules:
  point_and_call:
    elbow_angle_min_deg: 160            # arm must be nearly straight
    arm_elevation_max_deg: 30           # arm roughly horizontal (reject "pointing up")
    front_half_angle_deg: 30            # ±30° around body forward = "front"
    side_half_angle_deg: 45             # off-axis half-angle for left/right buckets
    min_keypoint_score: 0.3
    hold_frames: 6                      # each direction sustained ≥ 6 frames
    window_seconds: 5.0                 # full sequence must complete within 5 s
    sequence_modes: [LR, RL, LRF, RLF]  # accepted ordered sequences

alerts:
  frame_windows:
    point_and_call_done: 1              # emit immediately on match
    missing_directions: 90              # ~3 s at 30 fps before missing-directions fires
  window_ratio: 1.0
  cooldown_frames: 90

cross_zone: []                          # optional polygon (norm coords); enables missing_directions
```

## v2 Roadmap

If field testing reveals limitations of the geometric rule (target: rule
recall < 90% of human-judged complete gestures, or operational FP tolerance
exceeded), upgrade in this order:

1. **Dataset collection.**
   - **DP Dataset (Kyoto U)** — ~2M frames of pedestrian/worker behavior with
     3D pointing direction labels; primary source for in-distribution
     Japanese workplace footage.
   - **Roboflow `wayceys-workspace/hand-pointing-directions`** — ~1.7k images,
     5-class hand-pointing labels (Down/Forward/Left/Right/Up/Palm/Fist);
     useful for the per-frame direction head.
   - Self-collected: ~200 short crosswalk clips for fine-tune. Ingest via the
     existing `dataset_store/` MCP-driven flow (see `dataset_store/CLAUDE.md`);
     never bootstrap-script downloads.
2. **5-class MLP direction head** replacing the geometric rule. Inputs: pose
   keypoints in torso frame. Output: `{point_left, point_right, point_front,
   neutral, invalid}`. Trained via `core/p06_training/` (`hf-classification`
   or `timm` head). Adds `06_training_*.yaml`, ONNX-exported into
   `pretrained/safety-point_and_call/`.
3. **Sequence model upgrade if rule recall < 90%** — replace
   `CrosswalkSequenceMatcher` with **ST-GCN** (skeleton-based action recog)
   or **1D-TCN** over a sliding window of pose frames. Output:
   `{point_and_call_complete, incomplete_sequence, no_gesture}`.
4. **Release flow**: `utils/release.py --run-dir <ts_dir>` →
   `releases/safety_point_and_call/v<N>_<YYYY-MM-DD>/`.

## Notes

- **Folder is kebab** (`safety-point_and_call`); `dataset_name` in
  `05_data.yaml` is **snake** (`safety_point_and_call`).
- DWPose ONNX is shared with `safety-poketenashi` and
  `safety-fall_pose_estimation` — reuse the same checkpoint, do not duplicate.
- `code/_base.py` redefines `PoseRule` + `RuleResult` locally instead of
  importing from `safety-poketenashi` (project rule: `code/` may import
  `core/` + `utils/` only, never another feature's `code/`).
- `_DWPoseAdapter` is a ~60-LOC copy from `safety-poketenashi/code/orchestrator.py`
  for the same reason — same project rule applies.
- The geometric direction classifier consumes COCO-17 keypoints (not
  WholeBody-133): shoulders 5/6, elbows 7/8, wrists 9/10, hips 11/12.
  DWPose returns 133 wholebody points; the adapter slices the first 17 (body).

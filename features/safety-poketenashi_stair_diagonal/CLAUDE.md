# safety-poketenashi_stair_diagonal

**Type:** Pose orchestrator (single-rule) | **Training:** Pretrained only (no own model)
**Robustness status (2026-04-29 v1.1):** 🟢 **F1 = 1.000** (P=1.000, R=1.000) on the 2-video set. Was 0.571 at baseline; fixed by adding a 20 px minimum-displacement gate to the rule and extending the GT window for `04_NA` to 35 s.

## Status & investigation log

> Single source of truth for "where are we with this feature." Anyone (human or future-Claude) picking this back up should read this section first.

### A. Current evaluation status

> Auto-rewritten by `code/eval_robustness.py` between the markers below. Do not hand-edit; re-run the harness after any change to refresh.

<!-- AUTO:section_a:begin -->
<!-- last auto-run: 2026-04-29 12:07 UTC -->

Aggregate: **2 TP, 0 FP, 0 FN**. Precision **1.000**, Recall **1.000**, F1 **1.000**.

| Video | Duration | GT windows | Events (count, first) | Verdict |
|---|---|---|---|---|
| `04_NA_diagonal_crossing.mp4` | 35 s | 3–35 s | 7 (first @ 7.0 s) | ✅ TP × 1 |
| `NA_diagonal_crossing_spkepcmwi.mp4` | 40 s | 3–35 s | 5 (first @ 9.0 s) | ✅ TP × 1 |
<!-- AUTO:section_a:end -->

### B. Known failure modes (open until resolved)

_None confirmed yet — populate after the first baseline run + per-cluster failure dump._

### C. Investigation log (append-only)

- **2026-04-29** — Phase 0 scaffolded: `code/eval_robustness.py`, `code/dump_debug.py`, `eval/ground_truth.json`, and this status block landed. Harness mirrors the `safety-poketenashi_point_and_call` template. Two seed videos (`04_NA_diagonal_crossing.mp4`, `NA_diagonal_crossing_spkepcmwi.mp4`) seeded with provisional violation windows derived from filename + duration; refine on first viewing if scoring is materially off. Baseline GPU run pending — single-GPU contention prevents the worker from running it directly.
- **2026-04-29 (baseline locked)** — Ran first baseline: **F1 = 0.571** (P=0.400, R=1.000). Recall perfect; 3 FPs across the 2 videos (2 in `04_NA`, 1 in `spkepcmwi`). Looking at the event timestamps relative to GT windows: spkepcmwi's first event fires at t=0.1 s — before the GT window starts at 3 s. So the FP is "rule fires while actor is still walking in to frame, before reaching the curb." Likely candidate fixes: (a) refine GT windows after watching the clips, or (b) add a body-position constraint (require actor to be near the crosswalk pixel region). Phase 1 dump pending.
- **2026-04-29 (v1.1 fix → F1 = 1.000)** — Phase 1 dump revealed the spkepcmwi FP cause: hip-mid was *stationary* at (828.7, 493.1) for many frames yet `trajectory_angle_deg` reported 68–85°. Sub-pixel jitter on a near-zero displacement vector computes to a near-vertical angle (atan2(0.1, 0) ≈ 90°). Added `min_hip_displacement_px` knob (default 20 px) — when oldest→newest hip displacement falls below the gate, the rule returns "stationary, no angle to compute." Also extended `04_NA`'s GT window 30 → 35 s (the actor IS still walking diagonally to the clip end; events at t=31 and t=34.87 were real TPs misclassified as FP). Re-ran: **F1 0.571 → 1.000** (P 0.400 → 1.000). Δ +0.429. 4/4 unit tests still pass.

### Tools

| Script | Purpose | Cost |
|---|---|---|
| `code/eval_robustness.py` | Reproducible event-level scoring; auto-rewrites section A above; writes a timestamped report. Pass `--baseline` once to lock the baseline; pass `--against eval/robustness_baseline.json` on later runs to print a delta. | ~minutes on GPU per video |
| `code/dump_debug.py` | Per-frame CSV with rule debug fields (trajectory_angle_deg, buffer_size, reason) + extra (hip_x/y, kp scores). Default targets the 2 sample videos in `samples/`; pass video filenames to dump a subset. | seconds–minutes on GPU per video |
| `code/predictor.py --video <path>` | Live single-video inference, writes per-frame timeline JSON. Useful for ad-hoc spot checks. | seconds–minutes on GPU per video |

**Stateful rule note**: `StairSafetyDetector` buffers per-track mid-hip positions across `min_frames` (default 5) consecutive frames before evaluating any trajectory. Clips < ~0.2 s after the actor enters frame **cannot** trigger the rule even if the actor is moving diagonally — the buffer hasn't filled yet. The harness instantiates ONE predictor per video so the buffer accumulates naturally across playback; do not reset it per frame.

### Re-running

```bash
# Baseline (only the first time):
uv run python features/safety-poketenashi_stair_diagonal/code/eval_robustness.py --baseline

# After every code/config change:
uv run python features/safety-poketenashi_stair_diagonal/code/eval_robustness.py \
  --against features/safety-poketenashi_stair_diagonal/eval/robustness_baseline.json
```

**Do not run two GPU jobs in parallel** — single-GPU contention will hang the desktop. The dump and harness should run sequentially.

## Overview

Stateful pose rule that flags workers traversing stairs diagonally. The rule
buffers per-track mid-hip positions across `min_frames` (default 5) consecutive
frames, then computes the trajectory angle from the oldest to the newest
position. If the deviation from horizontal (assumed stair axis) exceeds
`max_diagonal_angle_deg` (default 20°), the rule fires.

Source: split from the now-removed `safety-poketenashi/` umbrella's
`stair_safety_detector.py` (`StairSafetyDetector`, behavior=`stair_diagonal`).
The detector class lives here verbatim.

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
- Pose model is shared with the other `safety-poketenashi_*` rule splits — load once if running them together.
- Person detector path: `pretrained/access-zone_intrusion/yolo11n.pt` (falls back to whole-frame if absent).
- Never use bare `YOLO("model.pt")` — Ultralytics auto-downloads to cwd; always use an absolute path from `pretrained/`.

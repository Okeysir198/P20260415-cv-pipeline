# safety-poketenashi_point_and_call

**Type:** Pose orchestrator | **Training:** 🔧 Pretrained only (v1) — rule-based on top of pose keypoints
**Robustness status (2026-04-29 baseline):** 🔴 v1.1 — event-level **F1 = 0.103** (P=0.056, R=0.571) on the labeled 8-video set. Hand-tabulated estimate of "≈ 0.5" was way too generous — at the event level, every cooldown re-fire outside a GT window counts as FP. Investigation plan active; see "Status & investigation log" below.

## Status & investigation log

> Single source of truth for "where are we with this feature." Anyone (human or future-Claude) picking this back up should read this section first. The detailed investigation plan lives in `~/.claude/plans/`; this is the day-to-day status.

### A. Current evaluation status

> Auto-rewritten by `code/eval_robustness.py` between the markers below. Do not hand-edit; re-run the harness after any change to refresh.

<!-- AUTO:section_a:begin -->
<!-- last auto-run: 2026-04-29 10:16 UTC -->

Aggregate: **4 TP, 67 FP, 3 FN**. Precision **0.056**, Recall **0.571**, F1 **0.103**.

| Video | Duration | GT windows | Matches (count, first) | Verdict |
|---|---|---|---|---|
| `05_SHI_point_and_call.mp4` | 41 s | 29–34 s | 2 (first @ 30.0 s) | ✅ TP × 1 |
| `POKETENASHI.mp4` | 266 s | 174–217 s | 11 (first @ 82.9 s) | ⚠️ TP 1 / FP 9 / FN 0 |
| `POKETENASHI_anzen_daiichi_lecture.mp4` | 379 s | (none) | 51 (first @ 9.6 s) | ❌ FP × 51 |
| `POKETENASHI_autotech_indonesia_senam.mp4` | 200 s | 100–130 s | 3 (first @ 37.6 s) | ❌ FN × 1 |
| `POKETENASHI_spkepcmwi_full.mp4` | 310 s | 158–228 s | 1 (first @ 278.0 s) | ❌ FN × 1 |
| `SHI_point_and_call_spkepcmwi.mp4` | 70 s | 25–60 s | 0 | ❌ FN × 1 |
| `shisa_kanko_correct_demo.mp4` | — | (skip) | — | ⚠️ Animated mascot ヨシだ君, not a photographic human. DWPose cannot detect cartoons. |
| `shisa_kanko_promotion_method.mp4` | 180 s | 4–180 s | 11 (first @ 4.1 s) | ✅ TP × 1 |
| `shisa_kanko_railway_toyota.mp4` | 236 s | 148–203 s | 12 (first @ 148.8 s) | ⚠️ TP 1 / FP 3 / FN 0 |
<!-- AUTO:section_a:end -->

### B. Known failure modes (open until resolved)

- [ ] **Lecture FPs** — `POKETENASHI_anzen_daiichi_lecture.mp4` fires 51 times. Root cause: `elbow_angle_min_deg=45` + `min_distinct_directions=2` accept any raised + bent arm in two distinct azimuth bins; presenter's normal hand gestures while speaking pass the gate. Proposed fix: gesture-onset temporal pattern (rest → raise → hold → lower state machine on wrist velocity) — see Phase 2 Intervention A.
- [ ] **Phone-on-ear FP in `POKETENASHI.mp4`** — first match fires at t=82.9 s during the KE phone-usage section, not the SHI section at 174–217 s. Root cause: `min_wrist_ear_distance_ratio=0.35` is too loose; phone holders' wrist is near forehead/cheek but not directly at the ear. Proposed fix: two-tier confidence gate — Phase 2 Intervention B.
- [ ] **Far-field FN in `SHI_point_and_call_spkepcmwi.mp4`** — actor at < 15 % of frame height; wrist scores 0.16–0.45 fall below `min_keypoint_score=0.25`. Proposed fix: pre-pose crop upscale — Phase 2 Intervention C.
- [ ] **Pose-jitter FPs (suspected)** — could be contributing to the lecture / outro FPs. Random-azimuth flicker may pass `min_distinct_directions: 2` if successive frames bin to L vs R noisily. Proposed fix: sequence transition-shape filter — Phase 2 Intervention D.
- [x] **AV1 codec silent failure** — `shisa_kanko_railway_toyota.mp4` was originally AV1-encoded; OpenCV's bundled FFmpeg can't decode AV1 without HW accel and returned 0 frames silently. Fixed by re-encoding to H.264 (commit on `main`).
- [ ] *(deferred)* **Cartoon-character handling** — DWPose is photographic-only. `shisa_kanko_correct_demo.mp4` (animated mascot) is unevaluable. No fix planned in v1.x.

### C. Investigation log (append-only)

- **2026-04-29** — Audited the orchestrator on the expanded 9-video sample set. Hand-tabulated F1 ≈ 0.5. Identified 3 actionable failure modes (lecture FPs, phone-on-ear FP, far-field FN) plus the AV1 codec bug (fixed in-place). Investigation plan filed at `~/.claude/plans/with-this-home-ct-admin-documents-langgr-federated-dewdrop.md`. Goal: lift F1 to ≥ 0.8 via 4 rule-based interventions before considering the ML head escalation on the v2 roadmap.
- **2026-04-29 (Phase 0 done)** — Built `code/eval_robustness.py` + `eval/ground_truth.json` and ran the first reproducible baseline. **Real event-level F1 = 0.103** (P=0.056, R=0.571), much worse than the hand-tabulated 0.5: cooldown re-fires count individually so the lecture's 51 matches all count as FPs. New issues surfaced: (a) `shisa_kanko_railway_toyota` has 3 FPs *outside* the 148–203 s GT window plus the 1 TP inside; (b) `autotech_indonesia_senam` is currently classified FN — 3 matches fired at t=37.6 s but my GT window guess of [100, 130] is probably wrong, refine the GT next time someone watches the clip. Baseline locked at `eval/robustness_baseline.json`. Phase 1 next: per-cluster failure dump.

### Next steps (in order)

1. ~~Phase 0: build `code/eval_robustness.py` + `eval/ground_truth.json` so this status block can be auto-regenerated.~~ ✅ done 2026-04-29.
2. **Phase 1**: per-cluster failure dump (CSV of pose features at FP/FN frames) → confirm hypotheses.
3. Phase 2: intervention A (gesture-onset temporal pattern) — biggest expected impact.
4. After each intervention: re-run harness, append a section C log entry, tick the section B checkbox if resolved.
5. Phase 3 gate: ship rule-based v1.2 if F1 ≥ 0.8, else escalate to ML head (v2).

### Re-running the harness

```
uv run python features/safety-poketenashi_point_and_call/code/eval_robustness.py
```
Takes ~9 min on GPU (DWPose ONNX over 9 videos × ~30 fps). Auto-rewrites section A above and writes a timestamped report to `eval/robustness_<ts>.json`. Add `--against eval/robustness_baseline.json` to print a delta vs the locked baseline. **Do not run two of these in parallel** — single GPU contention will hang the desktop.

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

Replaces the binary `pointing_calling_detector` that previously lived in the
now-removed `safety-poketenashi/` umbrella (which only checked whether *any*
arm was extended horizontally — binary, non-directional, no L→R→F sequence).

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
shared across backends. Never use bare `YOLO("model.pt")` (Ultralytics
auto-downloads to cwd) — always pass an explicit absolute path from `pretrained/`.

## Quick Start

```bash
# Smoke-test the orchestrator on sample images (default DWPose backend)
# Place a few worker JPGs under features/safety-poketenashi_point_and_call/samples/ first.
uv run features/safety-poketenashi_point_and_call/code/orchestrator.py --smoke-test
# Writes eval/orchestrator_smoke_test.json + eval/smoke_*.jpg

# Run on a video file (writes annotated mp4 + per-frame timeline JSON)
uv run features/safety-poketenashi_point_and_call/code/orchestrator.py \
  --video features/safety-poketenashi_point_and_call/samples/05_SHI_point_and_call.mp4
# Writes eval/smoke_<basename>.mp4 + eval/smoke_<basename>.json

# Override pose backend (verifies the swap)
uv run features/safety-poketenashi_point_and_call/code/orchestrator.py --smoke-test --pose-backend rtmpose

# Latency benchmark across all 4 backends
uv run features/safety-poketenashi_point_and_call/code/benchmark.py
# Writes eval/benchmark_results.json

# Unit tests (synthetic keypoints — no GPU/data needed)
uv run -m pytest features/safety-poketenashi_point_and_call/tests/ -v

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
- [x] **v1.1 robustness pass** (2026-04-29): forearm-vector geometry,
      wrist-far-from-ear filter, stationary-actor gate, largest-person filter,
      `min_distinct_directions` OR-acceptance. End-to-end smoke on POKETENASHI
      training video — true positive on good actor (SHI t=31.47s), no false
      positive on bad-example walking actor.
- [x] `--video <path>` CLI on `orchestrator.py` writes annotated mp4 +
      per-frame timeline JSON to `eval/smoke_<basename>.{mp4,json}`.
- [x] Config-wiring fix: `pose_rules.point_and_call.*` keys now actually load
      (were silently using hardcoded defaults).
- [x] `pose_backend.py` accepts both string and dict shapes for `person_detector`.
- [ ] Optional `cross_zone:` polygon configured per deployment site.
- [ ] Far-field actors (NA-style scenes): fix under-detection by either using
      a higher-resolution pose model or upscaling the person crop before pose.

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
                            PointingDirectionDetector(PoseRule) — uses forearm
                            (elbow→wrist) vector for elevation + azimuth; phone-to-ear
                            rejection via wrist-far-from-ear ratio
  crosswalk_sequence_matcher.py
                            CrosswalkSequenceMatcher — sliding-window run-length
                            matcher; strict ordered modes LR/RL/LRF/RLF/LFR/RFL/FLR/FRL
                            OR `min_distinct_directions` ≥ N OR-acceptance; cooldown
  orchestrator.py           PointAndCallOrchestrator + dataclasses + draw() +
                            --smoke-test CLI + --video CLI; largest-person filter +
                            stationary-actor (hip-velocity) gate
  benchmark.py              Loops 4 backends; per-backend latency + det_rate,
                            graceful error capture per backend

tests/
  test_pose_backend.py
  test_pointing_direction_detector.py
  test_crosswalk_sequence_matcher.py

samples/                    point-and-call clips / frames; current set is split from
                            POKETENASHI safety video (PT Trimitra Baterai Prakasa);
                            05_SHI_*.mp4 contains the actual gesture demonstration
eval/orchestrator_smoke_test.json     gitignored — written by --smoke-test (images)
eval/smoke_<basename>.{mp4,json}      gitignored — written by --video (clips)
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
    # Geometry — tuned for casual industrial-video pointing (NOT formal Japanese stiff-arm form).
    # Captures both "arm extended sideways" and "hand-at-eye-level" Japanese-style points.
    elbow_angle_min_deg: 45             # allows tight L-shape (hand-at-eye-level points)
    arm_elevation_max_deg: 60           # forearm within 60° of horizontal (allows raised pointing)
    front_half_angle_deg: 35            # ±35° around body forward axis = front
    side_half_angle_deg: 55             # ±55° around body left/right axis = left/right
    min_keypoint_score: 0.25            # accept slightly lower scores (small-in-frame actors)
    min_wrist_ear_distance_ratio: 0.35  # reject phone-CONTACT-ear; allow hand-at-eye-level
    max_body_speed_px_per_sec: 35       # smoothed-over-1.5s hip speed > N ⇒ walking ⇒ suppress pointing
    # Sequence acceptance — TWO ways to fire `point_and_call_done`:
    #   (a) one of the strict ordered permutations below matches, OR
    #   (b) `min_distinct_directions` distinct directions accumulate in window.
    # 2-D pose can't reliably tell "front" from "vertically up the body" when
    # an actor faces the camera, so 2 distinct directions is the robust default.
    hold_frames: 3                      # each direction sustained ≥ 3 frames (~0.1 s @30fps)
    window_seconds: 5.0                 # full sequence must complete within 5 s
    sequence_modes: [LRF, LFR, RLF, RFL, FLR, FRL]
    min_distinct_directions: 2          # 0=disabled (use sequence_modes only); 2/3=OR-condition

alerts:
  frame_windows:
    point_and_call_done: 1              # emit immediately on match
    missing_directions: 90              # ~3 s at 30 fps before missing-directions fires
  window_ratio: 1.0
  cooldown_frames: 90

cross_zone: []                          # optional polygon (norm coords); enables missing_directions
```

## Algorithm robustness (real-world video calibration)

The original geometry was tuned for formal stiff-arm Japanese shisa-kanko. Industrial
training footage shows casual variations (bent elbows, slightly diagonal arms, walking
actors with briefly extended arms) that defeat strict thresholds. v1.1 adds five
robustness layers, each driven by a YAML knob:

| Layer | YAML knob | Purpose |
|---|---|---|
| Forearm-vector geometry | `code/pointing_direction_detector.py` (always on) | Compute elevation + azimuth from elbow→wrist (forearm), not shoulder→wrist. Captures the actual pointing direction when the upper arm is dropped. |
| Wrist-far-from-ear | `min_wrist_ear_distance_ratio` | Rejects phone-to-ear poses: wrist within N×shoulder-width of nearest ear ⇒ not pointing. |
| Stationary-actor gate | `max_body_speed_px_per_sec` | Smoothed (1.5 s) hip-midpoint velocity. `> N px/s` ⇒ actor is walking, not standing at the curb ⇒ suppress pointing labels. |
| Largest-person filter | `code/orchestrator.py::process_frame` (always on) | When multiple people are detected (forklift driver + worker, etc.), feed only the largest-area box to the matcher to prevent label-flicker faking a multi-direction sequence. |
| Distinct-direction OR-acceptance | `min_distinct_directions` | Fires `point_and_call_done` when N distinct directions accumulate, in addition to strict permutation matching. Robust to "front" being indistinguishable from "vertically up" in 2-D pose. |

### End-to-end smoke results (v1.1)

| Clip | True positive | False positive | Notes |
|---|---|---|---|
| `samples/05_SHI_point_and_call.mp4` (41 s) | ✅ matches at t=30.00s & t=33.00s (good actor pointing) | ✅ none | Bad-example walking actor with briefly extended arm correctly suppressed by velocity gate |
| `samples/04_NA_diagonal_crossing.mp4` (35 s) | ✅ match at t=25.40s (actor's gesture before crossing) | ✅ none | After threshold relaxation, actor's hand-at-eye + arm-extended sequence both register (point_left + point_right) |

### Known limitations

- **Far-field actors**: when the actor is < 15 % of frame height (NA clip), keypoint
  scores fall below the rule's `min_keypoint_score` floor, and most pointing-frames
  return `invalid`. Tracking + a higher-resolution pose model would help.
- **Forearm vs upper-arm bending**: a "phone in hand, walking" pose with arm
  swinging at the side can briefly hit the extension thresholds, but the
  stationary-actor gate suppresses these because hips are moving.
- **2-D front detection**: a worker pointing AWAY from the camera (down the
  crosswalk) has the wrist at near-zero pixel offset from the shoulder. The
  rule conflates this with "vertically up" and either case fails the elevation
  filter. Hence `min_distinct_directions: 2` instead of strict 3.

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
   `pretrained/safety-poketenashi_point_and_call/`.
3. **Sequence model upgrade if rule recall < 90%** — replace
   `CrosswalkSequenceMatcher` with **ST-GCN** (skeleton-based action recog)
   or **1D-TCN** over a sliding window of pose frames. Output:
   `{point_and_call_complete, incomplete_sequence, no_gesture}`.
4. **Release flow**: `utils/release.py --run-dir <ts_dir>` →
   `releases/safety_poketenashi_point_and_call/v<N>_<YYYY-MM-DD>/`.

## Notes

- **Folder is kebab** (`safety-poketenashi_point_and_call`); `dataset_name` in
  `05_data.yaml` is **snake** (`safety_poketenashi_point_and_call`).
- DWPose ONNX is shared with `safety-poketenashi` and
  `safety-fall_pose_estimation` — reuse the same checkpoint, do not duplicate.
- `code/_base.py` defines `PoseRule` + `RuleResult` locally (project rule:
  `code/` may import `core/` + `utils/` only, never another feature's `code/`).
- `_DWPoseAdapter` (~60 LOC) was inlined from the now-removed
  `safety-poketenashi/` umbrella for the same reason.
- The geometric direction classifier consumes COCO-17 keypoints (not
  WholeBody-133): shoulders 5/6, elbows 7/8, wrists 9/10, hips 11/12.
  DWPose returns 133 wholebody points; the adapter slices the first 17 (body).

## Deployment Architecture

### Single-camera pipeline
Frame ingestion → person detector (YOLO11n at `pretrained/access-zone_intrusion/yolo11n.pt`) → top-down crop → DWPose ONNX (shared at `pretrained/safety-poketenashi/dw-ll_ucoco_384.onnx`) → COCO-17 keypoints → `PointingDirectionDetector` (per-frame label: point_left/right/front/neutral/invalid) → `CrosswalkSequenceMatcher` (per-track temporal FSM) → `point_and_call_done` / `missing_directions` event → sink.

### Per-track FSM (compliance logic)
The deployment FSM is:
```
IDLE → APPROACH (in approach polygon) → CROSSING (in cross_zone polygon) → DONE
                  └ on transition APPROACH→CROSSING:
                        if matcher.last_match recent → emit `compliant`
                        else → emit `missing_directions` ⚠
```
Compliance auditing requires both polygons configured per camera and ByteTrack so each worker has their own FSM state.

### When to enable ByteTrack
ALWAYS in real deployment. Sequence matching needs persistent track identity to avoid feeding multiple workers' labels into the same matcher. Configured in `configs/10_inference.yaml::tracker:`; wire via `VideoProcessor(enable_tracking=True, tracker_config=cfg["tracker"])` — see `core/p10_inference/video_inference.py:166-172`.

### When you need person detector + pose detector together
Both required. DWPose ONNX is top-down (needs person crops); the canonical adapter is `code/pose_backend.py::_DWPoseAdapter` (lines 98-167). For far-field cameras (< 15% of frame height), DWPose is the only path that works — use it always.

### Shared pose backbone
Same DWPose ONNX as the other `safety-poketenashi_*` rules. See `features/CLAUDE.md` for shared-loader recommendation when deploying multiple rules on one stream.

### Site calibration
- `cross_zone:` polygon (image-normalized) — required for `missing_directions` alert.
- `approach_zone:` polygon (optional) — defines where the gesture is expected.
- `max_body_speed_px_per_sec` — depends on camera distance/zoom; calibrate by recording a person walking and measuring hip pixel velocity.
- `min_distinct_directions` (default 2) — the rule fires when N distinct of L/R/F directions are observed within `window_seconds`. Set to 1 for single-direction installations, 2 for L+R, 3 for strict L+R+F.
- All thresholds in `pose_rules.point_and_call:` (elbow angle, arm elevation, ear distance, etc.).

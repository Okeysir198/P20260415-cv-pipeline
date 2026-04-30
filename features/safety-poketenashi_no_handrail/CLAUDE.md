# safety-poketenashi_no_handrail

**Type:** Pose rule | **Training:** 🔧 Pretrained only (rule-based on top of DWPose keypoints)
**Robustness status (2026-04-29):** 🟡 baseline pending — eval BLOCKED until per-video handrail polygons annotated. Harness scaffolded; both sample videos currently report `skipped — no polygon configured`.

## Status & investigation log

> Single source of truth for "where are we with this feature." Anyone (human or future-Claude) picking this back up should read this section first.

### A. Current evaluation status

> Auto-rewritten by `code/eval_robustness.py` between the markers below. Do not hand-edit.

<!-- AUTO:section_a:begin -->
<!-- last auto-run: (never) -->

Evaluable videos: **0**. Aggregate: **0 TP, 0 FP, 0 FN**. Precision **—**, Recall **—**, F1 **—**.

| Video | Duration | GT windows | Matches (count, first) | Verdict |
|---|---|---|---|---|
| `03_TE_no_handrail.mp4` | — | (skip) | — | ⚠️ no polygon configured |
| `TE_no_handrail_spkepcmwi.mp4` | — | (skip) | — | ⚠️ no polygon configured |
<!-- AUTO:section_a:end -->

### B. Known failure modes (open until resolved)

- [ ] **Eval BLOCKED on polygon annotation** — until handrail polygons are drawn per video, the rule cannot fire and harness output is meaningless. Edit `eval/ground_truth.json` and fill in `handrail_zones_norm` (image-normalized [0,1] polygons, one per visible handrail) plus `violation_windows` (when the actor is on stairs without touching the rail) for each video. Then re-run `code/eval_robustness.py`.
- [ ] *(placeholder — populate after first real baseline run)*

### C. Investigation log (append-only)

- **2026-04-29** — Robustness harness scaffolded (`code/eval_robustness.py`, `code/dump_debug.py`, `eval/ground_truth.json`). Polygons NOT yet annotated for either sample video; both currently report `skipped — no polygon configured`. Status block locked at BLOCKED until someone watches the clips and fills in `handrail_zones_norm` + `violation_windows`. GPU baseline will follow once polygons are in.

### Tools

| Script | Purpose | Cost |
|---|---|---|
| `code/eval_robustness.py` | Reproducible event-level scoring; auto-rewrites section A above; writes a timestamped report. Reads `handrail_zones_norm` per video from `eval/ground_truth.json` and overrides the predictor's polygons before processing — videos without a polygon are skipped. Pass `--baseline` once to lock the baseline; pass `--against eval/robustness_baseline.json` on later runs to print a delta. | ~1 min/video on GPU once polygons are configured |
| `code/dump_debug.py` | Per-frame CSV with wrist coords, distance to nearest zone edge per side, and all relevant kp scores (wrists, hips, knees, ankles). Skips videos with no polygon. | ~30 s per 35-s video on GPU |
| `code/handrail_detector.py` | Underlying rule (cv2.pointPolygonTest); see `_base.py` for the `PoseRule` ABC. Already imported by both scripts above; do not duplicate. | — |

### Re-running

```
# After editing eval/ground_truth.json with polygons + violation windows:
uv run python features/safety-poketenashi_no_handrail/code/eval_robustness.py
uv run python features/safety-poketenashi_no_handrail/code/eval_robustness.py --baseline   # once polygons stabilise
uv run python features/safety-poketenashi_no_handrail/code/dump_debug.py                   # per-frame CSVs
```

**Do not run two GPU jobs in parallel** — single-GPU contention will hang the desktop. The dump and harness should run sequentially.
**Polygon-config note**: this rule is disabled when `handrail_zones` is empty. The harness lifts polygons out of `eval/ground_truth.json` per video so the YAML config doesn't need to be edited per video — but if you change the camera framing, update the polygon there.

## Overview

Single-rule feature split out of the `safety-poketenashi` umbrella. Detects
when a worker on the stairs is **not** holding the handrail by checking the
pixel distance from each visible wrist to the nearest configured handrail
zone polygon. The rule fires when both wrists exceed `hand_to_railing_px`
from every zone.

This is a thin wrapper around `HandrailDetector` (~95 LOC, copied verbatim
from the umbrella) plus a DWPose pose backend and a per-frame orchestrator.

## Deployment Architecture

### Single-camera pipeline
Frame ingestion → person detector (YOLO11n at `pretrained/access-zone_intrusion/yolo11n.pt`) → top-down crop → DWPose ONNX (shared at `pretrained/safety-poketenashi/dw-ll_ucoco_384.onnx`) → COCO-17 keypoints → `HandrailDetector` (with site-configured zone polygons) → triggered flag → event sink.

### Site polygon required
This rule is disabled until `handrail_zones:` is populated in `configs/10_inference.yaml`. Polygons are image-normalized [0,1] coordinates; one polygon per visible handrail in the camera frame.

**Polygon SOP** (do this before any robustness run):

1. Open the sample clip; pause on a frame where the handrail is fully visible.
2. Read off pixel coords of the handrail's bounding outline (e.g. for a 1280×720 frame, a left-side rail running diagonally might be at pixels `(450, 720) → (520, 720) → (660, 100) → (590, 100)`).
3. Divide each `(x, y)` by `(W, H)` to normalize: `(450/1280, 720/720) → (0.352, 1.000)`.
4. Append to `eval/ground_truth.json` under the video's `handrail_zones_norm: [...]` (a list of polygons, each a list of `[x, y]` pairs).
5. Add `violation_windows: [[t_start_s, t_end_s], ...]` for stair sections where the actor is *not* touching any rail.

Minimal example for a 1280×720 frame with one left-rail polygon:
```json
{
  "video": "03_TE_no_handrail.mp4",
  "handrail_zones_norm": [[[0.352, 1.000], [0.406, 1.000], [0.516, 0.139], [0.461, 0.139]]],
  "violation_windows": [[2.5, 5.8]]
}
```

After editing, re-run `code/eval_robustness.py`. Use the same zone-annotation tool that `access-zone_intrusion` uses if you prefer drawing over hand-coding.

### When to enable ByteTrack
Enable when multiple workers can ascend the stairs simultaneously and you need per-worker compliance logging. Otherwise the per-frame check is enough. Wire via `VideoProcessor(enable_tracking=True, tracker_config=cfg["tracker"])` — see `core/p10_inference/video_inference.py:166-172`.

### When you need person detector + pose detector together
DWPose top-down requires person boxes. Required here whenever the camera shows the stairs from a distance.

### Shared pose backbone
Same DWPose ONNX as other `safety-poketenashi_*` rules. See `features/CLAUDE.md` for the multi-feature deployment recipe.

### Site calibration
- `handrail_zones`: must be populated per camera. Empty list disables the rule.
- `hand_to_railing_px` (default 60): pixel distance threshold. Calibrate by holding a hand on the rail in a recorded clip and measuring the wrist-to-zone-edge pixel distance.
- `tracker.frame_rate`: match camera fps.

## Pipeline Checklist

- [x] `code/_base.py` — `PoseRule` base class + `RuleResult` dataclass (self-contained)
- [x] `code/handrail_detector.py` — wrist-to-zone polygon distance rule (verbatim copy)
- [x] `code/predictor.py` — DWPose + rule orchestrator + `--smoke-test` / `--video` CLI
- [x] `code/benchmark.py` — latency benchmark on sample images
- [x] `tests/test_handrail.py` — synthetic-keypoint unit tests
- [ ] Configure site-specific `handrail_zones:` polygon in `10_inference.yaml`
- [ ] Smoke-test against site clips; tune `hand_to_railing_px`

## Files

```
configs/
  05_data.yaml              dataset_name + viz classes (no_handrail / handrail_ok)
  10_inference.yaml         pose_rules.no_handrail + ByteTrack stub + alerts
code/
  _base.py                  PoseRule + RuleResult — does NOT import safety-poketenashi
  handrail_detector.py      HandrailDetector (verbatim copy from umbrella)
  predictor.py              Thin per-frame orchestrator + CLI
  benchmark.py              Per-image latency
tests/
  test_handrail.py          Synthetic keypoints — wrist-in-zone, wrist-far, no-zones
samples/                    Stair clips (gitignored except .gitkeep)
eval/                       Smoke-test outputs (gitignored except .gitkeep)
```

## Notes

- **Folder is kebab** (`safety-poketenashi_no_handrail`); `dataset_name` in
  `05_data.yaml` is **snake** (`safety_poketenashi_no_handrail`).
- `code/_base.py` redefines `PoseRule` + `RuleResult` locally instead of
  importing from `safety-poketenashi` (project rule: `code/` may import
  `core/` + `utils/` only, never another feature's `code/`).
- `handrail_detector.py` was copied verbatim from the now-removed
  `safety-poketenashi/` umbrella during the per-rule split.
- `access-zone_intrusion/code/zone_intrusion.py` has a polygon evaluator
  using `matplotlib.path.Path`; this feature keeps the umbrella's
  `cv2.pointPolygonTest` to avoid cross-feature imports. Consolidation can
  happen later if the team wants a shared polygon utility in `utils/`.
- DWPose ONNX is shared with other `safety-poketenashi_*` rules — reuse the
  same checkpoint, do not duplicate.
- Person detector for top-down crops: `pretrained/access-zone_intrusion/yolo11n.pt`.
  Never use bare `YOLO("model.pt")` (Ultralytics auto-downloads to cwd) —
  always pass an explicit absolute path from `pretrained/`.

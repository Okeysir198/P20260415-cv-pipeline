# safety-poketenashi_no_handrail

**Type:** Pose rule | **Training:** 🔧 Pretrained only (rule-based on top of DWPose keypoints)

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
This rule is disabled until `handrail_zones:` is populated in `configs/10_inference.yaml`. Polygons are image-normalized [0,1] coordinates; one polygon per visible handrail in the camera frame. Draw them during install using whatever zone-annotation tool the site uses (the same one used for `access-zone_intrusion`).

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

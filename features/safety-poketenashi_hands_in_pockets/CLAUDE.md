# safety-poketenashi_hands_in_pockets

**Type:** Pose rule (single behavior) | **Training:** 🔧 Pretrained only — rule on top of DWPose ONNX

## Overview

Per-frame "hands in pockets" detector split from the umbrella
`safety-poketenashi/` orchestrator. Single rule, single behavior class
(`hands_in_pockets`). Pretrained-only — no training data, no `06_training.yaml`.

```
image_bgr ─► person detector (YOLO11n) ─► DWPose ONNX (shared) ─► COCO-17 keypoints
                                                                           │
                                                                           ▼
                                              HandsInPocketsDetector (per-frame)
                                                                           │
                                                                           ▼
                                                  alerts: [hands_in_pockets]
```

The rule fires when, for either side, **wrist y > hip y + ratio * torso_height**
AND **|wrist x − torso_cx| < margin * torso_width**.

## Deployment Architecture

### Single-camera pipeline
Frame ingestion → person detector (YOLO11n at `pretrained/access-zone_intrusion/yolo11n.pt`) → top-down crop → DWPose ONNX (shared at `pretrained/safety-poketenashi/dw-ll_ucoco_384.onnx`) → COCO-17 keypoints → `HandsInPocketsDetector` → per-frame `triggered` flag → optional matcher / FSM → event sink.

### When to enable ByteTrack
Enable when more than one worker can appear in the camera's field of view OR you need per-worker compliance logging. ByteTrack is configured in `configs/10_inference.yaml::tracker:` but is only wired when the caller passes `VideoProcessor(enable_tracking=True, tracker_config=cfg["tracker"])` — see `core/p10_inference/video_inference.py:166-172`. The YAML stub alone is not enough.

### When you need person detector + pose detector together
This rule reads COCO-17 keypoints, so it always needs pose. Whether you need a separate person detector depends on the pose backend: DWPose ONNX requires top-down crops (so YES, person detector first), MediaPipe / hf_keypoint internally handle full-frame (so NO, pose-only is enough). For far-field actors (< 15% of frame height) the top-down DWPose path is the only one that works.

### Shared pose backbone
`pretrained/safety-poketenashi/dw-ll_ucoco_384.onnx` is shared across all five `safety-poketenashi_*` features. Each feature today instantiates its own ORT session — when deploying multiple rules on one frame stream, that's redundant. See `features/CLAUDE.md` for the recommended shared-loader pattern.

### Site calibration
- `wrist_below_hip_ratio` (default 0.05): how far below the hip line the wrist must be for the rule to fire. Calibrate per camera angle.
- `wrist_inside_torso_margin` (default 0.08): horizontal proximity to torso centerline. Wider = more permissive.
- `tracker.frame_rate` should match the camera's actual fps.

## Files

```
configs/
  05_data.yaml              single-class umbrella (hands_in_pockets)
  10_inference.yaml         pose backend + pose_rules.hands_in_pockets + alerts + tracker stub
code/
  __init__.py
  _base.py                  PoseRule + RuleResult (self-contained — no cross-feature import)
  hands_in_pockets_detector.py
                            HandsInPocketsDetector(PoseRule) — wrist-below-hip + torso-margin
  predictor.py              orchestrator: person detector + DWPose adapter + this rule;
                            CLI --smoke-test + --video <path>
  benchmark.py              per-image triggered/not-triggered tally
tests/
  test_hands_in_pockets.py  synthetic COCO-17 keypoints, no GPU/data needed
samples/                    rule-specific clips (.gitkeep)
eval/                       gitignored; smoke-test + benchmark output (.gitkeep)
```

## Notes

- **Folder is kebab + underscore** (`safety-poketenashi_hands_in_pockets`); `dataset_name` in `05_data.yaml` is **snake** (`safety_poketenashi_hands_in_pockets`).
- `code/_base.py` and `code/hands_in_pockets_detector.py` are self-contained; project rule: `code/` may import from `core/` and `utils/` only, never another feature's `code/`.
- The DWPose adapter in `code/predictor.py` follows the cleaner pattern from `safety-poketenashi_point_and_call/code/pose_backend.py`.
- Person detector path is hard-coded to `pretrained/access-zone_intrusion/yolo11n.pt`; falls back to whole-frame box if missing. Never use bare `YOLO("model.pt")` (Ultralytics auto-downloads to cwd).

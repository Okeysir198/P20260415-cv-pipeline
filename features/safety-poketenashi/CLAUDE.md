# safety-poketenashi

**Type:** Orchestrator | **Training:** 🔧 Pretrained only (no own model; delegates to pose + sub-models)

## Overview

Umbrella feature that orchestrates detection of 5 prohibited/required workplace behaviors using RTMPose keypoints + zone rules + phone detection sub-model. No custom model training needed for this feature.

## Behaviors

| Behavior | Backend | Module |
|---|---|---|
| `phone_usage` | ML detection | `safety-poketenashi-phone-usage` (separate feature) |
| `hands_in_pockets` | Pose rule | `code/hands_in_pockets_detector.py` |
| `stair_diagonal` | Pose + tracking rule | `code/stair_safety_detector.py` |
| `no_handrail` | Pose + zone rule | `code/handrail_detector.py` |
| `no_pointing_calling` | Pose rule | `code/pointing_calling_detector.py` |

Pose backend: `dwpose_384_pose` ONNX (best: det_rate=1.0, 13ms). Falls back to MediaPipe if ONNX unavailable.

## Pipeline Checklist

- [x] `code/_base.py` — `PoseRule` base class + `RuleResult` dataclass
- [x] `code/hands_in_pockets_detector.py` — wrist-near-hip rule (COCO kps 9,10,11,12)
- [x] `code/stair_safety_detector.py` — diagonal trajectory rule with frame buffer
- [x] `code/handrail_detector.py` — wrist-to-zone distance rule
- [x] `code/pointing_calling_detector.py` — elbow angle + arm elevation rule
- [x] `code/orchestrator.py` — `PoketanashiOrchestrator` wiring all rules + DWPose ONNX
- [x] `code/benchmark.py` — pose model benchmark on samples
- [ ] Smoke test orchestrator against `samples/` → `eval/orchestrator_smoke_test.json`
- [ ] Tune `configs/10_inference.yaml` frame windows + thresholds from smoke test results

## Benchmark Results — samples (2026-04-17, 10 sample images)

| Model | Det Rate | Latency ms (mean) | Notes |
|---|---|---|---|
| **dwpose_384_pose** | **1.000** | **13.2** | Best: ONNX, no mmpose dependency |
| yolo_nas_pose_s | 1.000 | 37.7 | |
| yolo_nas_pose_m | 1.000 | 86.0 | |
| yolo_nas_pose_l | 1.000 | 110.4 | |
| dwpose_384_poke | 1.000 | 162.9 | High latency variance |
| pose_landmarker_lite | 0.900 | 19.4 | MediaPipe fallback |
| pose_landmarker_full | 0.900 | 34.x | MediaPipe |
| pose_landmarker_heavy | 0.900 | 56.x | MediaPipe |
| rtmpose-s_coco-wholebody | skipped | — | mmpose not installed |
| rtmw-l_cocktail14_384x288 | skipped | — | mmpose not installed |
| rtmpose-s_coco_256x192 | skipped | — | mmpose not installed |
| rtmo-s_body7_640x640 | skipped | — | mmpose not installed |
| rtmo-l_body7_640x640 | skipped | — | mmpose not installed |

**Recommendation:** `dwpose_384_pose` (ONNX, det_rate=1.0, 13ms, no extra dependencies).

> **Note (2026-04-29):** the latest `eval/benchmark_results.json` shows
> `dwpose_384_pose` with `status="error"` (CUDA alloc failure during batched
> benchmark — likely transient memory pressure). The model itself works in
> production: `safety-point_and_call` U3 smoke test confirmed ~10 ms/frame on
> the same ONNX. Re-run the benchmark on a less-loaded GPU to refresh the
> table. The recommendation stands.

Full results: `eval/benchmark_results.json`

## Key Files

```
configs/05_data.yaml            — 6-class umbrella config
configs/10_inference.yaml       — alert thresholds + frame windows + zone polygons
code/_base.py                   — PoseRule base class
code/hands_in_pockets_detector.py
code/stair_safety_detector.py
code/handrail_detector.py
code/pointing_calling_detector.py
code/orchestrator.py            — PoketanashiOrchestrator
code/benchmark.py               — pose model benchmark
eval/benchmark_results.json
```

## Notes

- Pose model is shared with `safety-fall_pose_estimation` — use same ONNX checkpoint
- `no_handrail` rule requires site-specific zone polygon in `10_inference.yaml` (handrail zone position)
- mmpose RTMPose models require `pip install mmpose` — currently skipped; DWPose ONNX is the production choice
- Person detector (for top-down DWPose cropping) uses `pretrained/access-zone_intrusion/yolo11n.pt`; falls back to whole-frame if unavailable
- Never use bare `YOLO("model.pt")` — Ultralytics auto-downloads to cwd; always use an explicit absolute path from `pretrained/`

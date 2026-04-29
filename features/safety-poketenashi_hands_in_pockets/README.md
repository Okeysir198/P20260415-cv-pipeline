# safety-poketenashi_hands_in_pockets

Single-rule pose feature: detects a worker walking with **hands in pockets**.
Split from the umbrella `safety-poketenashi/` orchestrator into a self-contained
folder so the rule can be deployed, calibrated, and demoed independently.

| Field | Value |
|-------|-------|
| Task | Pose estimation + CPU rule (per-frame) |
| Recommended model | DWPose-L (Apache-2.0, 133-kpt COCO-WholeBody, 384x288 ONNX); body-17 slice consumed by the rule |
| Detector (top-down crop) | YOLO11n (`pretrained/access-zone_intrusion/yolo11n.pt`) |
| Datasets | None (pretrained-only); custom factory video for threshold calibration |
| Behavior class | `hands_in_pockets` (alert when triggered) |

The rule consumes COCO-17 keypoints (shoulders 5/6, wrists 9/10, hips 11/12)
and fires when a wrist is BOTH below the hip line AND within the torso
horizontal band (close to the body centerline). Either side triggers.

## Pretrained weights

Shared with the other `safety-poketenashi_*` features:

| File | Purpose |
|---|---|
| `pretrained/safety-poketenashi/dw-ll_ucoco_384.onnx` | DWPose-L wholebody ONNX |
| `pretrained/access-zone_intrusion/yolo11n.pt` | Person detector for top-down crops |

## Quick Start

```bash
# Smoke-test the rule on synthetic keypoints + sample images
uv run features/safety-poketenashi_hands_in_pockets/code/predictor.py --smoke-test

# Run on a video file
uv run features/safety-poketenashi_hands_in_pockets/code/predictor.py \
  --video features/safety-poketenashi_hands_in_pockets/samples/<clip>.mp4

# Per-image trigger benchmark
uv run features/safety-poketenashi_hands_in_pockets/code/benchmark.py

# Unit tests
uv run -m pytest features/safety-poketenashi_hands_in_pockets/tests/ -v
```

For the standard feature layout and the end-to-end pipeline CLI, see
[`features/README.md`](../README.md) and the [root README](../../README.md).

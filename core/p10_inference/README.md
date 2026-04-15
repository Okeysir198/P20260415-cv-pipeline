# p06_inference — Prediction + Video + Tracking + Face Recognition

## Purpose

Run inference with trained models (PyTorch or ONNX), process video streams with tracking, and perform face recognition on detected violations.

## Files

| File | Purpose |
|---|---|
| `predictor.py` | `DetectionPredictor` — single-image and batch inference with dual backend (.pth PyTorch / .onnx ONNX Runtime), auto-detected by file extension. Works with any model in the registry. |
| `video_inference.py` | `VideoProcessor` — frame-by-frame detection with per-class alert logic (configurable confidence thresholds and confirmation windows) + optional ByteTrack tracking via supervision bridge |
| `pose_predictor.py` | `PosePredictor` — top-down pipeline: detect persons with `DetectionPredictor`, then estimate keypoints on each crop with any `PoseModel` |
| `face_gallery.py` | `FaceGallery` — enroll/match/remove face identities, `.npz` storage, cosine similarity matching, batch support |
| `face_predictor.py` | `FacePredictor` — violation bbox → expand crop → face detect (SCRFD) → embed (MobileFaceNet) → gallery match → identity |
| `supervision_bridge.py` | Convert between pipeline prediction dicts and `sv.Detections`, build annotators from config, ByteTrack tracker helpers |

## Dual Backend

- **PyTorch** (`.pth`): Full model with weights, used during development
- **ONNX Runtime** (`.onnx`): Optimized for edge deployment, auto-detected by file extension

## Config Reference

- Face recognition: `features/access-face_recognition/configs/face.yaml` — detector/embedder arch, gallery path, similarity threshold, violation class IDs
- Demo/tracking: `app_demo/config/config.yaml` — annotator settings, ByteTrack params, zone config
- Pose estimation: `features/safety-fall_pose_estimation/configs/*.yaml` — model arch, input size, keypoint definitions

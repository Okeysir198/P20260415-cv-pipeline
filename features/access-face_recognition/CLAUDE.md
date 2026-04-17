# access-face_recognition

**Type:** Face recognition | **Training:** Pretrained only (SCRFD detector + ArcFace embeddings; gallery enrollment only)

## Overview

Two-stage pipeline: face detection (YuNet / SCRFD) → ArcFace embedding match against enrolled gallery. No retraining needed. Run enrollment once per site/person set.

## Pipeline Checklist

- [x] `configs/face.yaml` — enrollment config created (buffalo_l primary, yunet+sface fallback)
- [x] `code/face_recognition.py` — `FaceRecognitionPipeline` class (enroll / recognize / draw)
- [x] `code/benchmark.py` — benchmark script run on samples
- [ ] Run enrollment on real site identities → `eval/gallery.npz`
- [ ] Smoke test via `app_demo` face tab
- [ ] Install InsightFace for production (buffalo_l pipeline; current fallback: OpenCV yunet+sface)
- [ ] `release/` — package gallery.npz + face.yaml

## Benchmark Results — val split (2026-04-17, 15 sample images, 8 enrolled identities)

### Face Detectors

| Model | Detection Rate | Latency ms |
|---|---|---|
| yunet_2023mar | 0.933 | 2.1 |
| yunet_2023mar_int8 | 0.933 | 2.0 |
| yolov8n-face | 0.933 | 33.4 |

### Recognition Pipelines

| Pipeline | Rank-1 Accuracy | Spoof Reject Rate | Notes |
|---|---|---|---|
| **yunet_fp32 + sface_fp32** | **1.000** | 0.000 | 5/5 correct; spoof_print_alice wrongly matched |
| yunet_int8 + sface_int8 | 0.800 | 0.000 | 4/5 correct |
| InsightFace buffalo_l/m/s, antelopev2 | — | — | Not installed |

**Recommendation:** `yunet_2023mar + sface_2021dec` (rank-1=1.0, 2ms detection). Install InsightFace (`buffalo_l`) for production — anti-spoof support and higher accuracy on diverse faces.

Full results: `eval/benchmark_results.json`

## Key Files

```
configs/face.yaml               — enrollment config (model paths, threshold, gallery path)
code/face_recognition.py        — FaceRecognitionPipeline(config_path) → enroll() / recognize() / draw()
code/benchmark.py               — benchmark script
eval/benchmark_results.json     — raw benchmark output
eval/gallery.npz                — enrolled face embeddings (created by enroll())
```

## Enrollment

```bash
# Enroll identities from a directory of images
uv run features/access-face_recognition/code/face_recognition.py \
  --enroll --config features/access-face_recognition/configs/face.yaml \
  --images features/access-face_recognition/samples/
```

## Notes

- Spoof rejection requires InsightFace anti-spoofing module — not available with OpenCV fallback
- Similarity threshold (default 0.5) may need tuning per environment (lighting, camera angle)
- Gallery is portable: copy `gallery.npz` + `face.yaml` to deploy to new site

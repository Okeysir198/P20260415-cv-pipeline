# access-face_recognition — Pretrained Weights

## Summary

Face detector + embedder candidates for access-control pipelines, plus a
liveness stub. Total **24 entries** (18 files + 6 symlinks into
`dms_oms/pretrained/face_detection/`) across **~1.2 GB**. See brief:
[../../docs/technical_study/access-face_recognition-sota.md](../../docs/technical_study/access-face_recognition-sota.md).

## Files on disk (as of 2026-04-14)

| File / Subfolder | Size | Type | License | Source | Notes |
|---|---|---|---|---|---|
| `yunet_2023mar.onnx` | 228 KB | YuNet face detector | Apache-2.0 | OpenCV Zoo | **Option C detector** — Apache-clean. |
| `yunet_2023mar_int8.onnx` | 100 KB | YuNet INT8 | Apache-2.0 | OpenCV Zoo | Production INT8. |
| `sface_2021dec.onnx` | 37 MB | SFace embedder | Apache-2.0 | OpenCV Zoo | **Option C embedder.** 128-d. |
| `sface_2021dec_int8.onnx` | 9.5 MB | SFace INT8 | Apache-2.0 | OpenCV Zoo | Production INT8. |
| `det_500m.onnx` | 2.5 MB | SCRFD-500M face detector | **InsightFace non-commercial** | InsightFace `buffalo_sc` | Baseline detector — ship-blocker license. |
| `w600k_mbf.onnx` | 13 MB | MobileFaceNet ArcFace 512-d | **InsightFace non-commercial** | InsightFace `buffalo_sc` | Baseline embedder — ship-blocker license. |
| `buffalo_sc.zip` | 15 MB | InsightFace buffalo_sc bundle | **InsightFace non-commercial** | InsightFace | Contains SCRFD-500M + MBF. |
| `buffalo_s.zip` | 122 MB | InsightFace buffalo_s bundle | **InsightFace non-commercial** | InsightFace | Benchmark only. |
| `buffalo_m.zip` | 264 MB | InsightFace buffalo_m bundle | **InsightFace non-commercial** | InsightFace | Benchmark only. |
| `buffalo_l.zip` | 276 MB | InsightFace buffalo_l bundle | **InsightFace non-commercial** | InsightFace | Benchmark only. |
| `antelopev2.zip` | 344 MB | InsightFace antelopev2 (RetinaFace-R50 + R100) | **InsightFace non-commercial** | InsightFace | Benchmark only. |
| `edgeface_base.pt` | 70 MB | EdgeFace-Base embedder | **research-only** (Idiap) | Idiap | Benchmark only — license blocker. |
| `yolov8n-face.pt` | 6 MB | YOLOv8n-Face | **AGPL-3.0** (Ultralytics) | derronqi | Benchmark only. |
| `anti_spoof_2_7_80x80_MiniFASNetV2.pth` | 1.8 MB | MiniFASNetV2 anti-spoof | Apache-2.0 | minivision | Liveness stub (not yet wired). |
| `dmsoms_RetinaFace-R50.pth` | symlink | RetinaFace-R50 | (sibling) | → `dms_oms/pretrained/face_detection/RetinaFace-R50.pth` | Sibling-project artefact. |
| `dmsoms_yolov11n_face.pt` | symlink | YOLOv11n-Face (.pt) | (sibling, **AGPL-3.0**) | → `dms_oms/pretrained/face_detection/yolov11n_face/model.pt` | Sibling — license-flagged. |
| `dmsoms_yolov11n_face.onnx` | symlink | YOLOv11n-Face ONNX | (sibling, **AGPL-3.0**) | → `dms_oms/.../model.onnx` | Sibling. |
| `dmsoms_yolov11n_face_fp16.onnx` | symlink | YOLOv11n-Face FP16 | (sibling, **AGPL-3.0**) | → `dms_oms/.../model_fp16.onnx` | Sibling. |
| `dmsoms_yolov11n_face_gpu.onnx` | symlink | YOLOv11n-Face GPU | (sibling, **AGPL-3.0**) | → `dms_oms/.../model_gpu.onnx` | Sibling. |
| `dmsoms_yolov11n_face_nms.onnx` | symlink | YOLOv11n-Face NMS | (sibling, **AGPL-3.0**) | → `dms_oms/.../model_nms.onnx` | Sibling. |
| `.cache/huggingface/` | — | HF download metadata | — | — | Tooling cache. |
| `DOWNLOAD_MANIFEST.md` | 1.5 KB | Manifest w/ SHA256s | — | — | |

## Recommended defaults (from SOTA brief)

- **Option C (lightest, strictest licence — default)** — Detector `yunet_2023mar_int8.onnx`, Embedder `sface_2021dec_int8.onnx` (both on disk, Apache-2.0).
- **Option A (baseline-style, retrain required)** — SCRFD-500M arch (weights `det_500m.onnx` on disk but **non-commercial — retrain before ship**); pair with AdaFace-R18 (not on disk — retrain needed).
- **Option B (accuracy-first, gated)** — EdgeFace-Base (`edgeface_base.pt` on disk) + YuNet; **Idiap research-only licence blocks customer ship without written permission**.
- Liveness: `anti_spoof_2_7_80x80_MiniFASNetV2.pth` (on disk, Apache-2.0, not yet wired).

## Gated / skipped / 404

- AdaFace MS1MV2 R50/R18 weights — Google Drive gated; **not on disk**. Retrain path recommended.
- EdgeFace — on disk, but Idiap licence is research-only → not production-shippable to Nitto Denko customer without written clearance.
- InsightFace `buffalo_*` + `antelopev2` — on disk but all **non-commercial**; baseline/benchmark only.

## Related docs

- SOTA brief: `../../docs/technical_study/access-face_recognition-sota.md`
- Sibling inventory: `../../docs/technical_study/sibling-projects-inventory.md`
- Quality report: `../../features/access-face_recognition/predict/QUALITY_REPORT.md`
- SHA256s: `./DOWNLOAD_MANIFEST.md`

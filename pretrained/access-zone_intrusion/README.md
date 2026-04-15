# access-zone_intrusion — Pretrained Weights

## Summary

Person/COCO detector candidates for zone-intrusion (polygon + tracker).
Total **17 entries** (10 files + 7 symlinks, of which 4 point at shared
ai/pretrained/ artefacts and 2 at sibling projects) across **~192 MB** of
unique bytes (siblings not counted). See brief:
[../../docs/technical_study/access-zone_intrusion-sota.md](../../docs/technical_study/access-zone_intrusion-sota.md).

## Files on disk (as of 2026-04-14)

| File / Subfolder | Size | Type | License | Source | Notes |
|---|---|---|---|---|---|
| `dfine_n_coco.pt` | symlink | D-FINE-N COCO | Apache-2.0 | → `../dfine_n_coco.pt` | **SOTA pick #1 — primary detector.** Shared weight. |
| `rtdetr_v2_r18_coco.pt` | symlink | RT-DETRv2-R18 COCO | Apache-2.0 | → `../rtdetr_v2_r18_coco.pt` | Tier-3 transformer fallback. |
| `yolox_tiny.pth` | symlink | YOLOX-Tiny COCO | Apache-2.0 | → `../yolox_tiny.pth` | Clean-license fallback detector. |
| `yolo11n.pt` | 5.4 MB | YOLO11n COCO | **AGPL-3.0** (Ultralytics) | Ultralytics | Benchmark only — license blocks ship. |
| `yolo11s.pt` | 19 MB | YOLO11s COCO | **AGPL-3.0** | Ultralytics | Benchmark only. |
| `yolo11m.pt` | 39 MB | YOLO11m COCO | **AGPL-3.0** | Ultralytics | Benchmark only. |
| `yolov10n.pt` | 11 MB | YOLOv10n COCO | **AGPL-3.0** (THU-MIG) | THU-MIG | Benchmark only. |
| `yolov10s.pt` | 32 MB | YOLOv10s COCO | **AGPL-3.0** | THU-MIG | Benchmark only. |
| `yolov10m.pt` | 64 MB | YOLOv10m COCO | **AGPL-3.0** | THU-MIG | Benchmark only. |
| `yolov12n.pt` | 5.3 MB | YOLOv12n COCO | **AGPL-3.0** | sunsmarterjie | Benchmark only. |
| `yolov12s.pt` | 18 MB | YOLOv12s COCO | **AGPL-3.0** | sunsmarterjie | Benchmark only. |
| `dmsoms_rtdetr_r18vd.safetensors` | symlink | RT-DETR R18vd person | (sibling) | → `dms_oms/pretrained/person/rtdetr_r18vd/model.safetensors` | Sibling-project artefact. |
| `visualcore_rf-detr-base.pth` | symlink | RF-DETR-Base | (sibling) | → `visual_core/01_code/checkpoints/feat_detect/rfdetr/rf-detr-base.pth` | Sibling-project artefact. |
| `visualcore_yolov8l.pt` | symlink | YOLOv8-L | (sibling, **AGPL-3.0**) | → `visual_core/.../yolov8/yolov8l.pt` | Benchmark only. |
| `.cache/huggingface/` | — | HF download metadata | — | — | Tooling cache. |
| `DOWNLOAD_MANIFEST.md` | 883 B | Manifest w/ SHA256s | — | — | |

## Recommended defaults (from SOTA brief)

- **#1 D-FINE-N (COCO)** → `dfine_n_coco.pt` (symlink to shared file).
- **#2 YOLOX-Tiny (baseline retained)** → `yolox_tiny.pth` (symlink to shared file).
- **#3 RT-DETRv2-R18 (tier 3, staged)** → `rtdetr_v2_r18_coco.pt` (symlink to shared file).
- Tracker: **ByteTrack** (MIT, no learned weights) — no files required here.

## Gated / skipped / 404

None. All Ultralytics YOLOv10/11/12 checkpoints are excluded from shipping by
the AGPL-3.0 licence constraint (kept for benchmark reference only).

## Related docs

- SOTA brief: `../../docs/technical_study/access-zone_intrusion-sota.md`
- Sibling inventory: `../../docs/technical_study/sibling-projects-inventory.md`
- Quality report: `../../features/access-zone_intrusion/predict/QUALITY_REPORT.md`
- SHA256s: `./DOWNLOAD_MANIFEST.md`

# safety-fall_pose_estimation — Pretrained Weights

## Summary

17-keypoint body pose candidates for fall detection, covering RTMPose /
RTMO / ViTPose / HRNet / DWPose / MediaPipe / YOLO-NAS-Pose / Sapiens.
Total **28 files** across **~2.9 GB**. See brief:
[../../docs/technical_study/safety-fall_pose_estimation-sota.md](../../docs/technical_study/safety-fall_pose_estimation-sota.md).

## Files on disk (as of 2026-04-14)

| File / Subfolder | Size | Type | License | Source | Notes |
|---|---|---|---|---|---|
| `rtmpose-s_coco_256x192.pth` | 21 MB | RTMPose-S COCO 17-kpt | Apache-2.0 | MMPose | **SOTA pick #1 — baseline pose head.** |
| `rtmdet-nano_person.pth` | 4.1 MB | RTMDet-nano person detector | Apache-2.0 | MMPose | **Paired Stage-1 detector** for top-down pose. |
| `rtmo-s_body7_640x640.pth` | 38 MB | RTMO-S Body7 one-stage | Apache-2.0 | MMPose | One-stage alternative. |
| `rtmo-l_body7_640x640.pth` | 171 MB | RTMO-L Body7 one-stage | Apache-2.0 | MMPose | Accuracy tier. |
| `hrnet_w48_coco_256x192.pth` | 244 MB | HRNet-W48 COCO | Apache-2.0 / MIT | MMPose | Classic high-accuracy reference. |
| `vitpose-plus-small.pth` | 127 MB | ViTPose++ Small | Apache-2.0 | ViTAE | Transformer pose reference. |
| `vitpose-plus-base.pth` | 479 MB | ViTPose++ Base | Apache-2.0 | ViTAE | Larger reference. |
| `dw-ll_ucoco_384.onnx` | 129 MB | DWPose-L WholeBody 384×288 | Apache-2.0 | DWPose | Wholebody (shared with poketenashi). |
| `pose_landmarker_lite.task` | 5.6 MB | MediaPipe Pose Landmarker (Lite) | Apache-2.0 | Google | Mobile/CPU baseline. |
| `pose_landmarker_full.task` | 9 MB | MediaPipe Pose Landmarker (Full) | Apache-2.0 | Google | |
| `pose_landmarker_heavy.task` | 30 MB | MediaPipe Pose Landmarker (Heavy) | Apache-2.0 | Google | |
| `yolo_nas_pose_s.onnx` | 59 MB | YOLO-NAS-Pose S | **Deci non-commercial weights** (code Apache-2.0) | Deci | Benchmark only. |
| `yolo_nas_pose_m.onnx` | 149 MB | YOLO-NAS-Pose M | **Deci non-commercial weights** | Deci | Benchmark only. |
| `yolo_nas_pose_l.onnx` | 208 MB | YOLO-NAS-Pose L | **Deci non-commercial weights** | Deci | Benchmark only. |
| `posec3d_slowonly_r50_ntu60_xsub_keypoint.pth` | 7.9 MB | PoseC3D SlowOnly NTU60 | Apache-2.0 | MMAction2 | Skeleton-action second stage. |
| `stgcn_80e_ntu60_xsub_keypoint.pth` | 12 MB | ST-GCN NTU60 | Apache-2.0 | MMAction2 | Skeleton-action second stage. |
| `_hf_facebook_sapiens-pose-0.3b/sapiens_0.3b_goliath_best_goliath_AP_573.pth` | 1.3 GB | Sapiens 0.3B pose | **Sapiens License (non-commercial)** | facebook | Benchmark only — size + license blocker. |
| `.cache/huggingface/` | — | HF download metadata | — | — | Tooling cache. |
| `DOWNLOAD_MANIFEST.md` | 1.6 KB | Manifest w/ SHA256s | — | — | |

## Recommended defaults (from SOTA brief)

- **#1 RTMDet-nano-person + RTMPose-S (256×192)** →
  `rtmdet-nano_person.pth` + `rtmpose-s_coco_256x192.pth` (both on disk).
- **#2 RTMO-S (one-stage Body7)** → `rtmo-s_body7_640x640.pth` (on disk).
- **#3 ViTPose++ Small** → `vitpose-plus-small.pth` (on disk) — accuracy reference, heavier INT8 path.

## Gated / skipped / 404

- **Sapiens 0.3B** — fetched via gated-retry pass; kept for benchmark only (Sapiens License blocks commercial ship; also exceeds 18-TOPS budget).
- **Sapiens 1B** — skipped (4.68 GB > 3 GB policy cap).
- **YOLO-NAS-Pose (S/M/L)** — on disk, but Deci pre-trained-weights licence blocks production.

## Related docs

- SOTA brief: `../../docs/technical_study/safety-fall_pose_estimation-sota.md`
- Pose bulk log: `../../docs/technical_study/access-pose-bulk-download-log.md`
- Gated retry log: `../../docs/technical_study/gated-retry-download-log.md`
- Quality report: `../../features/safety-fall_pose_estimation/predict/QUALITY_REPORT.md`
- SHA256s: `./DOWNLOAD_MANIFEST.md`

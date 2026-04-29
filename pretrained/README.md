# pretrained/

Pretrained weights used as starting checkpoints for fine-tuning and as
SOTA benchmark candidates across Phase-1 features. This directory is
**gitignored** except for `README.md` files (per-feature + this rollup) —
download weights manually or via each feature's bulk script before
training.

Target deployment: generic ~18 TOPS INT8 edge accelerator, chip-agnostic,
standard ONNX. Commercial-friendly licences (Apache-2.0 / MIT / BSD)
strongly preferred; AGPL / GPL / CC-BY-NC / research-only artefacts are
kept for benchmarking but flagged as ship-blockers.

## Phase-1 feature rollup (as of 2026-04-14)

| Feature | Top pick on disk | # Files | Total size | License status | Quality verdict |
|---|---|---|---|---|---|
| [safety-fire_detection](./safety-fire_detection/README.md) | `deim_dfine_m_coco/deim_dfine_hgnetv2_m_coco_90e.pth` (Apache-2.0) | 413 | 2.0 GB | Clean for top-picks; ~25 AGPL fire-YOLO fine-tunes retained as benchmark only | Top-2 run on 10 real images; COCO/Obj365 pretrains have no fire class — Phase-1 fine-tune required (FASDD + D-Fire). |
| [ppe-helmet_detection](./ppe-helmet_detection/README.md) | `dfine_medium_obj2coco.safetensors` (Apache-2.0) | 90 | 3.3 GB | Clean for top-picks; most YOLO helmet fine-tunes are AGPL; DINOv3 is non-commercial | D-FINE-M + YOLOS-tiny hardhat + YOLOX-M top-2 evaluated; D-FINE-M produces COCO-only boxes — helmet fine-tune pending. |
| [ppe-shoes_detection](./ppe-shoes_detection/README.md) | Stage 1 `dfine_nano_coco.safetensors`; Stage 2 `dinov2_small.bin` (both Apache-2.0) | 54 | 760 MB | Clean for top-picks; DINOv3 (Meta non-commercial), FastViT / MobileViTv2 (Apple ASCL) flagged | Stage-1 detectors + Stage-2 backbones downloaded; 2-class shoe head needs fine-tune. |
| [safety-fall-detection](./safety-fall-detection/README.md) | `efficientnetv2_rw_s.ra2_in1k.bin` single-frame (Apache-2.0); `slowfast_r50_k400.pyth` temporal (Apache-2.0) | 46 (24 files + 22 symlinks) | 3.5 GB | Clean single-frame + SlowFast / X3D / MoViNet; VideoMAE / V2, UniFormerV2 are CC-BY-NC (benchmark only) | EfficientNetV2-S + VideoMAE-S evaluated on stills; temporal models downloaded but not yet run. |
| [safety-fall_pose_estimation](./safety-fall_pose_estimation/README.md) | `rtmdet-nano_person.pth` + `rtmpose-s_coco_256x192.pth` (Apache-2.0) | 28 | 2.9 GB | Clean for RTM family; Sapiens 0.3B + YOLO-NAS-Pose flagged non-commercial | Two-stage (YOLOX-tiny + RTMPose-S) + one-stage (RTMO-S) pipelines pass on real images. |
| [safety-poketenashi](./safety-poketenashi/README.md) (shared storage for `safety-poketenashi_*` rule family — 5 features) | `dw-ll_ucoco_384.onnx` WholeBody-133 (Apache-2.0) | 16 | 3.3 GB | Clean for DWPose / RTMW / RTMPose; Sapiens 0.6B non-commercial (benchmark only) | DWPose-L evaluated against body-17 baseline on samples; wholebody runs as expected. |
| [access-zone_intrusion](./access-zone_intrusion/README.md) | `dfine_n_coco.pt` (Apache-2.0, symlink to shared) | 17 (10 files + 7 symlinks) | 192 MB unique | Clean D-FINE-N / YOLOX-Tiny / RT-DETRv2-R18; YOLOv10 / 11 / 12 AGPL (benchmark only) | D-FINE-N + YOLOX-Tiny run on inside/outside samples; one missed-detection FAIL logged — confidence-tuning / retrain pending. |
| [access-face_recognition](./access-face_recognition/README.md) | `yunet_2023mar_int8.onnx` + `sface_2021dec_int8.onnx` (Apache-2.0) | 24 (18 files + 6 symlinks) | 1.2 GB | YuNet + SFace clean; InsightFace buffalo / antelopev2 + EdgeFace non-commercial (ship-blockers) | Baseline (SCRFD+MBF) vs Apache fallback (YuNet+SFace) evaluated on pair-cosine protocol; see QUALITY_REPORT. |

**Combined on-disk footprint across the 8 feature folders:
~17 GB, 688 entries** (files + symlinks; sibling-project symlinks do not
count toward the byte total).

## Gated items pending user action

- **`Advantech-EIOT/qualcomm-ultralytics-ppe_detection`** — previously gated; access is **now resolved** and the file is on disk at
  `ppe-helmet_detection/_hf_Advantech-EIOT_qualcomm-ultralytics-ppe_detection/ppe_yolov11n_w8a16_160x160_pics1000.dlc`. It is a vendor-specific DLC format (AGPL-3.0 Ultralytics base), so it is excluded from the chip-agnostic ONNX production path; kept for reference only.
- **Permanently 404** (repo deleted upstream, no user action can recover):
  - `keremberke/yolov5n-hard-hat-detection`
  - `keremberke/yolov5s-hard-hat-detection`
  - `keremberke/yolov5m-hard-hat-detection`

  Placeholder empty folders remain under `ppe-helmet_detection/_hf_keremberke_yolov5*-hard-hat-detection/` and should be treated as "not recoverable". Equivalent clean-license hard-hat weights are already covered by `yolos-tiny-hardhat/` (DunnBC22, Apache-2.0) and D-FINE-M.
- **Size-policy skip**: `facebook/sapiens-pose-1b-torchscript` (4.68 GB > 3 GB cap) — intentionally not downloaded; the 0.3B (fall-pose) and 0.6B (poketenashi) TorchScript variants are on disk for benchmark only.
- **Google-Drive-gated**: AdaFace MS1MV2 R18 / R50 weights — not on disk; retrain path preferred for production face recognition.

See `../docs/technical_study/gated-retry-download-log.md` for full retry-pass details.

## Shared top-level weights

Files directly under `ai/pretrained/` (not inside a feature folder) are
shared checkpoints reused across features:

| File | Size | Model | Used by |
|---|---|---|---|
| `yolox_nano.pth` / `yolox_tiny.pth` / `yolox_s.pth` | 7.7 / 40 / 72 MB | YOLOX Nano / Tiny / S (COCO) | baselines, pose Stage 1 |
| `yolox_m.pth` / `yolox_m_coco.pt` | 203 / 102 MB | YOLOX-M (COCO) | fire / helmet baseline |
| `yolox_l.pth` / `yolox_l_coco.pt` | 415 / 208 MB | YOLOX-L (COCO) | accuracy tier |
| `dfine_n_coco.pt` / `dfine_s_coco.pt` / `dfine_m_coco.pt` | 15 / 40 / 76 MB | D-FINE N / S / M (COCO) | shoes / zone_intrusion / helmet |
| `rtdetr_v2_r18_coco.pt` / `rtdetr_v2_r50_coco.pt` | 78 / 165 MB | RT-DETRv2 R18 / R50 | zone_intrusion tier 3 |
| `scrfd_500m.onnx` | 2.5 MB | SCRFD-500M face detector | face_recognition (non-commercial) |
| `mobilefacenet_arcface.onnx` | 13 MB | MobileFaceNet ArcFace 512-d | face_recognition (non-commercial) |

D-FINE (N/S/M) and RT-DETRv2 (R18/R50) weights are auto-cached by HF
Transformers at `~/.cache/huggingface/` on first use; the `.pt` copies
here exist for offline / air-gapped environments.

## Constraints reminder

- All files under `pretrained/` are gitignored except per-feature `README.md` and this top-level rollup (see `../.gitignore`).
- Every ship candidate must clear licence review; AGPL / GPL / CC-BY-NC / research-only / vendor non-commercial entries are catalogued here but **never deployed to customers** without written clearance.

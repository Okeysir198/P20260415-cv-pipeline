# Safety / Fall — OSS Pretrained Deep-Dive (2025–2026)

Deeper follow-up to `safety-fall-detection-sota.md` and `safety-fall_pose_estimation-sota.md`. Those first passes concluded that license-clean backbones exist (EfficientNetV2-S, MobileNetV4, DINOv2, X3D-M, RTMPose/RTMO) but no off-the-shelf **fall-specific** checkpoint was evaluated. This document surveys the open-source landscape for weights trained *directly* on fall datasets across all three methodology families and records what is downloadable, verified, and license-clean.

## 1. Scope

Already covered by prior passes (not repeated here):

- Generic pose: RTMPose-S, RTMO-S, DWPose, RTMDet-nano person
- Generic classification backbones: EfficientNetV2-S, MobileNetV4, DINOv2, ConvNeXt-V2, EVA-02
- Generic video backbones: X3D-M, VideoMAE, MoViNet, TSM, UniFormerV2

New in this pass:

- **RGB single-frame** fall classifiers / detectors with weights on HF/GitHub
- **Skeleton** fall classifiers (ST-GCN, PoseC3D) pretrained on NTU-RGBD
- **Temporal video** fall classifiers (VideoMAE, I3D, 3D-CNN Sports-1M fine-tunes)
- OmniFall 2025 unified benchmark
- Roboflow Universe + HF fall-specific weights census

The prior `fall_detection` internal eval already established that every surveyed public checkpoint fails out-of-box on the internal factory hard-negatives; this pass is therefore about **fine-tune starting points**, not plug-and-play models.

## 2. Single-frame fall classification weights

| Model | Repo | Arch | Dataset | License | Notes |
|---|---|---|---|---|---|
| `popkek00/fall_detection_model` | hf.co/popkek00/fall_detection_model | ResNet-18 HF classifier, 11.2 M, 2-class (`fall` / `no_fall`) | `hiennguyen9874/fall-detection-dataset` (HF) | **MIT** | Trainable in one line with `AutoModelForImageClassification`; no reported accuracy. Good warm-start for single-frame head. |
| `Siddhartha276/Fall_Detection` | hf.co/Siddhartha276/Fall_Detection | EfficientNet-B0 fine-tune, TFLite only | Custom | **MIT** | TFLite, no PyTorch weights in repo (404 on `.pt`/`safetensors`); reference only. |
| `melihuzunoglu/human-fall-detection` | hf.co/melihuzunoglu/human-fall-detection | YOLOv11 Ultralytics, `best.pt` 5.47 MB | Custom | **AGPL-3.0** — flag | Ultralytics lineage ⇒ AGPL; incompatible with closed-source product. Benchmark only. |
| `kamalchibrani/yolov8_fall_detection_25` | hf.co/kamalchibrani/yolov8_fall_detection_25 | YOLOv8, `best.pt` 6.2 MB | `kamalchibrani/fall_detection` | **OpenRAIL** — ambiguous, and YOLOv8 code is AGPL — flag | Same AGPL caveat as above. |
| `pahaht/YOLOv8-Fall-detection` | github.com/pahaht/YOLOv8-Fall-detection | YOLOv8 | Custom | AGPL via Ultralytics — flag | GitHub-release weights; same licence issue. |
| `hiennguyen9874/fall-detection-dataset` | hf.co/datasets/hiennguyen9874/fall-detection-dataset | Dataset only | — | Dataset-card dependent | Useful pairing with ResNet-18 head above. |

**Verdict (single-frame).** Only `popkek00/fall_detection_model` is a license-clean PyTorch warm-start. Every other public fall detector is Ultralytics-lineage AGPL. For production, fine-tune EfficientNetV2-S / MobileNetV4 *ourselves* using the internal 17 k set + `hiennguyen9874` dataset; use `popkek00` as sanity-check baseline.

## 3. Skeleton-based fall classification weights

Skeleton fall classifiers ride the NTU-RGBD ecosystem: train on NTU-60/120 (action #43 = "falling down"), then fine-tune on URFall / Le2i skeletons. mmaction2 hosts the relevant weights.

| Model | URL | Params | Top-1 NTU60-XSub | License | Notes |
|---|---|---|---|---|---|
| PoseC3D (SlowOnly-R50, NTU60-XSub, joint) | download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint/slowonly_r50_u48_240e_ntu60_xsub_keypoint-f3adabf1.pth | 2.0 M | 93.7 % | **Apache-2.0 code; NTU weights inherit NTU non-commercial redistribution restriction — flag** | Best skeleton-based fall accuracy among sub-10 M models. 8.2 MB. |
| ST-GCN (NTU60-XSub, joint) | download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint/stgcn_80e_ntu60_xsub_keypoint-e7bb9653.pth | 3.1 M | 86.9 % | **Apache-2.0 code; NTU redistribution flag** | 12.4 MB. Classic baseline, GCN. |
| ST-GCN++ v1.0 (`stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d`) | download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/.../...20221129-484a394a.pth | 3.1 M | 88.8 % | Apache-2.0 code; NTU flag | Verified 200 OK (~12 MB). |
| `GajuuzZ/Human-Falling-Detect-Tracks` — `tsstg` ST-GCN 7-class | Google-Drive (see refs) | 1.7 M | Le2i-reported ~95 % F1 | License unspecified — treat as research-only | 7-class head already trained **on Le2i fall** (coffee-room + home). Pairs with AlphaPose. Most directly applicable off-the-shelf skeleton fall classifier. |
| MotionBERT action fine-tune NTU60-XSub | github.com/Walter0807/MotionBERT (OneDrive links) | 42 M | 97.2 % | Apache-2.0 code; NTU flag | OneDrive download gated; too heavy for 18 TOPS edge budget without distillation. |
| `simplexsigil2/omnifall` dataset | hf.co/datasets/simplexsigil2/omnifall | — | — | CC-BY-NC-4.0 — flag | 1 M-row unified benchmark (URFall + Le2i + OOPS-Fall + more). Dataset only; paper (arXiv:2505.19889) reports I3D + VideoMAE baselines. |

**Verdict (skeleton).** PoseC3D-SlowOnly is the best Apache-code / small-footprint starting point. The GajuuzZ `tsstg` head is the only pre-existing skeleton model already classed as fall/not on Le2i — ideal sanity-check plug-in behind our existing RTMPose pipeline, despite unclear licence. Both should only be used via **code re-use + re-train on internal data** to dodge the NTU redistribution clause.

## 4. Temporal video fall-detection weights

| Model | Repo | Dataset | License | Notes |
|---|---|---|---|---|
| `zohaibshahid/videomae-base-finetuned-fall-detection` | hf.co/zohaibshahid/videomae-base-finetuned-fall-detection | unspecified fall set | unspecified; VideoMAE upstream CC-BY-NC-4.0 — flag | Repo contains only `.gitattributes` — **weights were not actually uploaded**. Dead end. |
| OmniFall baselines (I3D-R50, VideoMAE-Base) | paper arXiv:2505.19889, simplexsigil2/omnifall | OmniFall unified (URFall + Le2i + more) | CC-BY-NC-4.0 (dataset); VideoMAE CC-BY-NC-4.0 — flag | Paper publishes staged-to-wild eval; weights not released at time of writing (Apr 2026). |
| Alam et al. 2025 "Fall Detection using Transfer Learning-based 3D CNN" (arXiv:2506.03193) | — | Sports-1M pretrain → GMDCSA + CAUCAFall | weights not released | SVM on Sports-1M 3D-CNN features; reference design only. |
| MoViNet-A2-Stream K600 | Apache-2.0 | Generic K600 | Apache-2.0 | (covered in prior doc) — needs fine-tune on fall; best streaming candidate. |
| X3D-M K400 | Apache-2.0, pytorchvideo zoo | K400 | Apache-2.0 | (covered in prior doc) — current recommended temporal head; needs fine-tune on URFall + Le2i + OmniFall subset. |

**Verdict (temporal).** No license-clean, weights-available, fall-finetuned temporal checkpoint exists today. VideoMAE fall fine-tunes that do exist on HF are either empty repos (`zohaibshahid`) or CC-BY-NC. Practical path: X3D-M + self-train on OmniFall (research-only eval) / URFall + Le2i + internal tubes (production weights).

## 5. Top 3 recommendations

| # | Family | Model | URL | License | Fine-tune path |
|---|---|---|---|---|---|
| 1 | Skeleton | **PoseC3D SlowOnly-R50 NTU60-XSub** | download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint/slowonly_r50_u48_240e_ntu60_xsub_keypoint-f3adabf1.pth | Apache-2.0 code; NTU non-commercial flag on weights | Replace 60-class head with 2-class (fall / not-fall); fine-tune on RTMPose-inferred keypoints over internal clips + Le2i + URFall. Lightweight (2 M, 8 MB), INT8-friendly 3D heatmap CNN. Pairs directly with the existing RTMPose-S / RTMO-S pipeline. |
| 2 | Single-frame RGB | **popkek00 ResNet-18 fall classifier** | hf.co/popkek00/fall_detection_model | MIT | Fine-tune head on internal 17 k + hard-negative crops. Use as license-clean baseline before committing to EfficientNetV2-S. 44 MB safetensors. |
| 3 | Temporal video | **X3D-M K400 (already on disk) + OmniFall / URFall fine-tune** | dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_M.pyth | Apache-2.0 (weights); OmniFall dataset CC-BY-NC-4.0 so only use OmniFall for *evaluation*, train on URFall + Le2i + internal | X3D-M is the only 18-TOPS-friendly temporal backbone with clean weights. No fall-specific checkpoint exists; self-train. |

Deliberately not top-3:

- Ultralytics YOLOv8/v11 fall weights (AGPL)
- VideoMAE fall fine-tunes (upstream CC-BY-NC-4.0; plus `zohaibshahid` repo is empty)
- MotionBERT action NTU (Apache code, but 42 M + NTU redistribution flag + OneDrive-gated)

## 6. Download manifest

All downloads verified with `curl -IL` (HTTP 200 after redirect). File paths are absolute.

### `ai/pretrained/safety-fall-detection/`

| File | Size (B) | SHA256 | Source | License |
|---|---|---|---|---|
| `/home/nthanhtrung/Documents/05_Team/02_Vietsol/01_Projects/edge_ai/ai/pretrained/safety-fall-detection/fall_resnet18_popkek00.safetensors` | 44,764,336 | `b806b650fcfe490e1c438cd45614dcdbe4dfb6960163915a02393abf73ab039b` | hf.co/popkek00/fall_detection_model | MIT |
| `/home/nthanhtrung/Documents/05_Team/02_Vietsol/01_Projects/edge_ai/ai/pretrained/safety-fall-detection/fall_resnet18_popkek00_config.json` | 690 | `69e9159f30c9ca852feb32b8196d6c64b9f8724f47eb34ee844784ca28bb729f` | hf.co/popkek00/fall_detection_model | MIT |

Previously-downloaded files (prior pass, unchanged) remain in the same directory: `efficientnetv2_rw_s.ra2_in1k.bin`, `mobilenetv4_conv_small.e2400_r224_in1k.bin`, `dinov2-small.bin`, `videomae-small-finetuned-kinetics.bin`, `x3d_m.pyth`.

### `ai/pretrained/safety-fall_pose_estimation/`

| File | Size (B) | SHA256 | Source | License |
|---|---|---|---|---|
| `/home/nthanhtrung/Documents/05_Team/02_Vietsol/01_Projects/edge_ai/ai/pretrained/safety-fall_pose_estimation/posec3d_slowonly_r50_ntu60_xsub_keypoint.pth` | 8,191,762 | `f3adabf19d56bd4fb458e59570d5bbe0208f1e8a9a79c3d5f7fe03a0d5825d2a` | download.openmmlab.com/mmaction/skeleton/posec3d/... | Apache-2.0 code; NTU weights non-commercial flag |
| `/home/nthanhtrung/Documents/05_Team/02_Vietsol/01_Projects/edge_ai/ai/pretrained/safety-fall_pose_estimation/stgcn_80e_ntu60_xsub_keypoint.pth` | 12,443,433 | `e7bb965330622f3eb602406af995add6de3f679ca08ff4d1686d984c2084bebe` | download.openmmlab.com/mmaction/skeleton/stgcn/... | Apache-2.0 code; NTU weights non-commercial flag |

Previously-downloaded files remain: `rtmdet-nano_person.pth`, `rtmo-s_body7_640x640.pth`, `rtmpose-s_coco_256x192.pth`.

## 7. Licensing landmines

- **NTU-RGBD redistribution restriction.** NTU-60 / NTU-120 are licensed for *non-commercial academic research only* (ROSE Lab, NTU Singapore). Any checkpoint trained on NTU (PoseC3D, ST-GCN, ST-GCN++, MotionBERT-action variants, most skeleton SOTA on mmaction2) inherits this restriction at the weights level even when the code is Apache-2.0. Do **not** ship NTU-trained weights in a commercial product; use them as initialisation and re-train on licence-clean data (internal + Le2i if licence allows + URFall academic permission), then replace / overwrite the weights before release. The PoseC3D + ST-GCN files downloaded here are for internal benchmarking and warm-start only.
- **Ultralytics YOLOv8 / YOLOv11 fall fine-tunes (`melihuzunoglu`, `kamalchibrani`, `pahaht`).** Ultralytics library is **AGPL-3.0**; redistributing (or *running as a network service*) an Ultralytics-derived model obligates AGPL source disclosure for the entire serving stack. Incompatible with closed-source edge firmware. Our baseline YOLOX-M (Apache-2.0) stays in place. Treat all Ultralytics fall weights as reference benchmarks only, never ship.
- **VideoMAE / VideoMAE v2 weights.** Upstream `MCG-NJU/videomae-*` checkpoints are CC-BY-NC-4.0. The `zohaibshahid/videomae-base-finetuned-fall-detection` HF repo contains only `.gitattributes` (weights were never uploaded) — a dead link. Any downstream VideoMAE fall fine-tune inherits CC-BY-NC unless retrained from-scratch on permissive data. Benchmark only.
- **OmniFall dataset (arXiv:2505.19889, `simplexsigil2/omnifall`).** CC-BY-NC-4.0 — excellent unified benchmark (URFall + Le2i + OOPS-Fall + more with dense temporal labels) but non-commercial. Use for *evaluation* of our own models; do not train a shipped model directly on it.
- **GajuuzZ `tsstg` ST-GCN Le2i weights.** No explicit licence on the repo or the Google-Drive downloads — legally ambiguous. Treat as research-only.
- **Le2i dataset.** Academic, request-access (Univ. Burgundy). URFall and Charfi academic-only; CAUCAFall has a more permissive CC license.

## 8. References

- OmniFall — Schneider et al., arXiv:2505.19889 (May 2025). Dataset: hf.co/datasets/simplexsigil2/omnifall.
- "Human Fall Detection using Transfer Learning-based 3D CNN" — Alam et al., arXiv:2506.03193 (May 2025).
- PoseC3D — Duan et al., arXiv:2104.13586; mmaction2 model zoo `skeleton/posec3d/`.
- ST-GCN — Yan et al., AAAI 2018; ST-GCN++ — Duan et al., ACM MM 2022; mmaction2 `skeleton/stgcn/`, `skeleton/stgcnpp/`.
- MotionBERT — Zhu et al., ICCV 2023, arXiv:2210.06551; github.com/Walter0807/MotionBERT.
- VideoMAE — Tong et al., arXiv:2203.12602; VideoMAE v2 — Wang et al., arXiv:2303.16727 (both CC-BY-NC-4.0).
- X3D — Feichtenhofer, CVPR 2020; PyTorchVideo model zoo.
- GajuuzZ Human-Falling-Detect-Tracks (AlphaPose + ST-GCN + SORT): github.com/GajuuzZ/Human-Falling-Detect-Tracks.
- pahaht YOLOv8-Fall-detection: github.com/pahaht/YOLOv8-Fall-detection (AGPL).
- Roboflow Universe "fall" search: universe.roboflow.com/search?q=class:fall.
- popkek00 ResNet-18 fall classifier: hf.co/popkek00/fall_detection_model (MIT).
- hiennguyen9874 fall dataset: hf.co/datasets/hiennguyen9874/fall-detection-dataset.
- NTU-RGBD license: github.com/shahroudy/NTURGB-D (non-commercial academic).
- Companion prior surveys: `ai/docs/technical_study/safety-fall-detection-sota.md`, `ai/docs/technical_study/safety-fall_pose_estimation-sota.md`.

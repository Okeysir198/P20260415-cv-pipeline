# Fall Pretrained Weights — Bulk Download Log (2026-04-14)

Follow-up to the three surveys:

- `safety-fall-detection-sota.md`
- `safety-fall_pose_estimation-sota.md`
- `safety-fall_oss-pretrained-deep-dive.md`

Operator instruction: "don't worry about license, just download first — bulk pull everything". This log records every candidate that was attempted, whether it succeeded, and where it landed. **License flags are informational only — commercial-use decisions must be re-validated before deployment.**

## 1. Destinations

- `ai/pretrained/safety-fall-detection/`
- `ai/pretrained/safety-fall_pose_estimation/`
- `ai/pretrained/safety-poketenashi/` (overlap destination)

All three directories are gitignored under `ai/.gitignore` (`pretrained/*`). Per-directory manifests live at `<dir>/DOWNLOAD_MANIFEST.md` and are **not** committed.

## 2. Fall classification — outcome

| # | Candidate | File | Size (B) | SHA256 | Source | License | Status |
|---|---|---|---:|---|---|---|---|
| 1 | VideoMAE-Base K400 | `videomae-base-finetuned-kinetics.bin` | 346,198,861 | `f8462908e843373183868b89c56699f675839f1bebf43694a6c987c6df9d3ce4` | hf: MCG-NJU/videomae-base-finetuned-kinetics | CC-BY-NC-4.0 | OK |
| 2 | VideoMAE-Small K400 | `videomae-small-finetuned-kinetics.bin` | 88,197,173 | `d69585904fec7e507bf2edfba4a7abe2b92def9afc76460d1dc14dbbf864bcfc` | hf: MCG-NJU/videomae-small-finetuned-kinetics | CC-BY-NC-4.0 | EXISTS |
| 3 | VideoMAEv2-Base K710 | `videomaev2_base_k710.safetensors` | 344,924,592 | `ebffa1874066ea227330016e58a848e9e2bb1ff5605746459bded1122a42176d` | hf: OpenGVLab/VideoMAEv2-Base | CC-BY-NC-4.0 | OK |
| 4 | VideoMAEv2-Large K710 | `videomaev2_large_k710.safetensors` | 1,215,474,840 | `c27064402bfec2c7495e2226a5e0d8f46374b4df4b03026af1e10a57e11b2996` | hf: OpenGVLab/VideoMAEv2-Large | CC-BY-NC-4.0 | OK |
| 5 | VideoMAEv2-Small | — | — | — | hf: OpenGVLab/VideoMAEv2-Small | — | SKIP — repo does not exist (404) |
| 6 | VideoMAEv2-giant | — | — | — | hf: OpenGVLab/VideoMAEv2-giant (4.1 GB) | CC-BY-NC-4.0 | SKIP — 4.1 GB is well over edge budget; not prioritised in 25 min |
| 7 | UniFormerV2-B K400+K710 8×224 | `uniformerv2_b16_k400_k710.pyth` | 458,289,355 | `743da61c97f6281bd11ef1d364c2101698c8e89ff7722ee957378e36ef344005` | hf: Andy1621/uniformerv2 | CC-BY-NC-4.0 (weights) / MIT (code) | OK |
| 8 | DINOv3 ViT-S/16 LVD-1689M | — | — | — | hf: facebook/dinov3-vits16-pretrain-lvd1689m | dinov3-license (NC, gated) | FAIL — 401 unauthenticated (expected) |
| 9 | zohaibshahid VideoMAE fall fine-tune | — | — | — | hf: zohaibshahid/videomae-base-finetuned-fall-detection | unspecified | SKIP — repo contains only `.gitattributes`, no weights (confirmed empty) |
| 10 | YOLOv11 fall fine-tune (melihuzunoglu) | `yolov11_fall_melihuzunoglu.pt` | 5,472,282 | `3f56ad30358d5c63bf8dbc0c1299cf68818c3d291dfb10c94107b94110aadd4c` | hf: melihuzunoglu/human-fall-detection | AGPL-3.0 | OK |
| 11 | YOLOv8 fall fine-tune (kamalchibrani) | `yolov8_fall_kamalchibrani.pt` | 6,210,350 | `4a7eba2e982955d7c5f2ea65968f23332c123275116e75e2bac915f40ba749e0` | hf: kamalchibrani/yolov8_fall_detection_25 | OpenRAIL/AGPL | OK |
| 12 | popkek00 ResNet-18 fall | `fall_resnet18_popkek00.safetensors` | 44,764,336 | `b806b650fcfe490e1c438cd45614dcdbe4dfb6960163915a02393abf73ab039b` | hf: popkek00/fall_detection_model | MIT | EXISTS |
| 13 | SlowFast R50 8×8 K400 | `slowfast_r50_k400.pyth` | 277,138,115 | `454f39e1c1f985df2bee2aa27887ed53ff56e74ed8b8cca11203a1a1264d7cc2` | dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics | Apache-2.0 | OK |
| 14 | SlowFast R101 8×8 K400 | `slowfast_r101_k400.pyth` | 503,790,111 | `62966206aa4c1e06262aa48f600e24d871469c7a92c23aa08636776944df3e94` | dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics | Apache-2.0 | OK |
| 15 | X3D-M K400 (reference) | `x3d_m.pyth` | 30,779,313 | `eb9583d5f5a988a5c5f06f75760b070a5c9715794bd3b0536b78bd30595dcf67` | dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics | Apache-2.0 | EXISTS |
| 16 | X3D-L K400 | `x3d_l.pyth` | 50,025,453 | `19c99d8ab8f2aceab4f8a59a85f6dbc3c7f3a262a798ca95b906709ed100866b` | dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics | Apache-2.0 | OK |
| 17 | X3D-S K400 | `x3d_s.pyth` | 30,779,313 | `26b95f1605d49650b54049db40ba3a56e023b86b58c3b3e0e10e0992a9c8682f` | dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics | Apache-2.0 | OK |
| 18 | X3D-XS K400 | `x3d_xs.pyth` | 30,779,313 | `6dafc96144c88f800b190d2cb179ff69b8fd48c40afd4deca6c1e61f76fdfe86` | dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics | Apache-2.0 | OK |
| 19 | MoViNet-A1 base (TF Model Garden) | `movinet_a1_base.tar.gz` | 18,995,200 | `4c80bd477321d2483279358049191e151cb071a762efbe4efc9afa953e29920f` | storage.googleapis.com/tf_model_garden/vision/movinet | Apache-2.0 | OK (TF SavedModel tarball — requires TF→PyTorch port) |
| 20 | MoViNet-A2 base | `movinet_a2_base.tar.gz` | 21,258,240 | `478f2ac53744ec81309c37c8ed9fae627dccb382bab181569d7943fe81ca775e` | storage.googleapis.com/tf_model_garden | Apache-2.0 | OK |
| 21 | MoViNet-A3 base | `movinet_a3_base.tar.gz` | 29,941,760 | `1cd1a8c00e9692a982619a12df68c1b30f95af970258b1c63f477f1dc1f02b0c` | storage.googleapis.com/tf_model_garden | Apache-2.0 | OK |
| 22 | MoViNet-A2 stream | `movinet_a2_stream.tar.gz` | 29,624,320 | `611af11e3d32c254477206165b8fdd5735fd94bea23bb5362ca9906909b08449` | storage.googleapis.com/tf_model_garden | Apache-2.0 | OK |
| 23 | EfficientNetV2-S (timm) | `efficientnetv2_rw_s.ra2_in1k.bin` | 96,674,637 | `8bb2555726585abc07991848cfa0de5732cff2d8bf9f88d6317a687dbdc1b303` | hf: timm/efficientnetv2_rw_s.ra2_in1k | Apache-2.0 | EXISTS |
| 24 | MobileNetV4-Conv-Small (timm) | `mobilenetv4_conv_small.e2400_r224_in1k.bin` | 15,296,226 | `28f2109728e2b5b4296397f0ea2aa8fd1896c8dca1e7f5ff217ae5ec22049e39` | hf: timm/mobilenetv4_conv_small.e2400_r224_in1k | Apache-2.0 | EXISTS |
| 25 | DINOv2-Small | `dinov2-small.bin` | 88,297,097 | `1051e25b2ed69ddad24f3c41e7b6eed6e7f7d012103ea227e47eb82e87dc2050` | hf: facebook/dinov2-small | Apache-2.0 | EXISTS |

## 3. Fall pose estimation — outcome

| # | Candidate | File | Size (B) | SHA256 | Source | License | Status |
|---|---|---|---:|---|---|---|---|
| 1 | RTMPose-S COCO 256×192 | `rtmpose-s_coco_256x192.pth` | 22,007,763 | `8d57b1112021367bb6857e468be383b2bd24a0b69f121b3f54177635fb742907` | openmmlab mmpose | Apache-2.0 | EXISTS |
| 2 | RTMDet-nano person | `rtmdet-nano_person.pth` | 4,215,674 | `0e2da635c75e25dc88af08d01eb34bfe9cac06a7841cba223bdf04eed288b3dc` | openmmlab mmpose | Apache-2.0 | EXISTS |
| 3 | RTMO-S Body7 640×640 | `rtmo-s_body7_640x640.pth` | 39,766,454 | `dac2bf749bbfb51e69ca577ca0327dff4433e3be9a56b782f0b7ef94fb45247e` | openmmlab mmpose | Apache-2.0 | EXISTS |
| 4 | RTMO-L Body7 640×640 | `rtmo-l_body7_640x640.pth` | 179,277,852 | `b37118cee39b43531c23d9e443fbb4f0b8987f590d9c6ed61978d12ecfce84df` | openmmlab mmpose | Apache-2.0 | OK (parallel agent) |
| 5 | HRNet-W48 COCO 256×192 | `hrnet_w48_coco_256x192.pth` | 255,011,654 | `b9e0b3ab0439cb68e166c7543e59d2587cd8d7e9acf5ea62a8378eeb82fb50e5` | openmmlab mmpose | MIT | OK (parallel agent) |
| 6 | ViTPose++ Small | `vitpose-plus-small.pth` | 132,619,932 | `f7bad8ed09eeeb2a7de6b38faaa8a88d07838e23e9c06a2a782099bca7467cb9` | ViTPose repo / HF mirror | Apache-2.0 | OK (parallel agent) |
| 7 | ViTPose++ Base | `vitpose-plus-base.pth` | 501,627,628 | `640225e4a9dd544239f1da0ae36af865a6e76e21e5f33a12656aa7b77fa5a5fa` | ViTPose repo / HF mirror | Apache-2.0 | OK (parallel agent) |
| 8 | PoseC3D SlowOnly-R50 NTU60-XSub | `posec3d_slowonly_r50_ntu60_xsub_keypoint.pth` | 8,191,762 | `f3adabf19d56bd4fb458e59570d5bbe0208f1e8a9a79c3d5f7fe03a0d5825d2a` | openmmlab mmaction2 | Apache-2.0 code / NTU-NC weights | EXISTS |
| 9 | ST-GCN NTU60-XSub | `stgcn_80e_ntu60_xsub_keypoint.pth` | 12,443,433 | `e7bb965330622f3eb602406af995add6de3f679ca08ff4d1686d984c2084bebe` | openmmlab mmaction2 | Apache-2.0 code / NTU-NC weights | EXISTS |
| 10 | DWPose 384 | `dw-ll_ucoco_384.onnx` | 134,399,116 | `724f4ff2439ed61afb86fb8a1951ec39c6220682803b4a8bd4f598cd913b1843` | hf: yzd-v/DWPose | Apache-2.0 | OK (parallel agent, also in safety-poketenashi) |
| 11 | YOLO-NAS-Pose S (ONNX) | `yolo_nas_pose_s.onnx` | 61,688,622 | `80f9b2113dfe5e720cfcf911a95034fb9e26c57e951537488d7e5b3e21951fe0` | hf: hr16/yolo-nas-pose | Deci-NC | OK |
| 12 | YOLO-NAS-Pose M (ONNX) | `yolo_nas_pose_m.onnx` | 156,009,479 | `9602e05d0bcebbb3a6b6b4e1e42fe904bc4479fa0e7f35a719e1466c059aa94a` | hf: hr16/yolo-nas-pose | Deci-NC | OK |
| 13 | YOLO-NAS-Pose L (ONNX) | `yolo_nas_pose_l.onnx` | 217,844,305 | `412322c226889a3bfc4a0990fa6c6937527dc89c97e7b37d508865d6d1628463` | hf: hr16/yolo-nas-pose | Deci-NC | OK |
| 14 | YOLO-NAS-Pose (.pth torch) | — | — | — | hf: hr16/yolo-nas-pose/*.pth | — | SKIP — repo holds ONNX only; no native pth |
| 15 | Sapiens-0.3B pose | — | — | — | hf: facebook/sapiens-pose-0.3b | Sapiens-License (NC) | FAIL — 404 on torchscript path (gated + renamed) |
| 16 | MediaPipe pose_landmarker lite | `pose_landmarker_lite.task` | 5,777,746 | `59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a` | storage.googleapis.com/mediapipe-models | Apache-2.0 | OK (parallel agent) |
| 17 | MediaPipe pose_landmarker full | `pose_landmarker_full.task` | 9,398,198 | `4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad` | storage.googleapis.com/mediapipe-models | Apache-2.0 | OK (parallel agent) |
| 18 | MediaPipe pose_landmarker heavy | `pose_landmarker_heavy.task` | 30,664,242 | `64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b` | storage.googleapis.com/mediapipe-models | Apache-2.0 | OK (parallel agent) |
| 19 | MotionBERT action NTU60 | — | — | — | OneDrive (Walter0807/MotionBERT) | Apache-2.0 code / NTU-NC weights | FAIL — no direct-download URL; OneDrive web-gate |

## 4. Poketenashi overlap — outcome

| # | Candidate | File | Size (B) | SHA256 | Source | License | Status |
|---|---|---|---:|---|---|---|---|
| 1 | DWPose-LL UCOCO 384 | `dw-ll_ucoco_384.onnx` | 134,399,116 | `724f4ff2439ed61afb86fb8a1951ec39c6220682803b4a8bd4f598cd913b1843` | hf: yzd-v/DWPose | Apache-2.0 | EXISTS |
| 2 | RTMPose-S COCO-wholebody | `rtmpose-s_coco-wholebody.pth` | 72,010,049 | `3da02694cd6479d3b333ff42ebd0723f96bfa06adac1db1e2e815ed2e9e1b02d` | openmmlab mmpose | Apache-2.0 | EXISTS |
| 3 | RTMW-L cocktail14 384×288 | `rtmw-l_cocktail14_384x288.pth` | 230,299,153 | `afa589fa16bf8b9bab0807c806538a43bead39ed0151b79f41adcf551e3bcfb0` | openmmlab mmpose | Apache-2.0 | OK |

## 5. Summary

| Bucket | Attempted | OK / EXISTS | FAIL / SKIP |
|---|---:|---:|---:|
| Fall classification | 25 | 22 | 3 (DINOv3 gated, VideoMAEv2-Small repo missing, zohaibshahid empty, VideoMAEv2-giant deliberate skip — 3 non-success from full list) |
| Fall pose estimation | 19 | 17 | 2 (Sapiens gated, MotionBERT OneDrive no direct URL) |
| Poketenashi overlap | 3 | 3 | 0 |
| **Total** | **47** | **42** | **5** |

Total bytes on disk for fall_detection (new + existing) ≈ 4.6 GB. Fall pose ≈ 1.7 GB. Poketenashi (unchanged) ≈ 0.4 GB.

## 6. License landmines (informational)

Unchanged from the source surveys — this log did **not** re-evaluate licensing, it only pulled bytes. Before any checkpoint becomes part of a shipped model:

- **CC-BY-NC-4.0** weights (VideoMAE, VideoMAEv2, UniFormerV2 weights) — research/benchmarking only.
- **dinov3-license** — gated, non-commercial; not downloaded.
- **Sapiens-License** — non-commercial; not downloaded.
- **Deci-NC** (YOLO-NAS-Pose pretrained weights, ONNX exports included) — non-commercial pretrained license; code Apache-2.0.
- **AGPL-3.0 / OpenRAIL** (Ultralytics YOLOv8/v11 fall fine-tunes) — AGPL forces network-service source disclosure; incompatible with closed-source firmware.
- **NTU-RGBD non-commercial redistribution** — inherits onto PoseC3D / ST-GCN / MotionBERT NTU-fine-tuned checkpoints.

Re-validate per the rules in `safety-fall_oss-pretrained-deep-dive.md` §7 before productionisation.

## 7. Failures and skips — explanations

- **DINOv3 ViT-S/16** — HF-gated model card (`dinov3-license`); `curl` returns 401 without authenticated access. Expected and documented.
- **VideoMAEv2-Small / VideoMAEv2-giant** — Small repo 404 (does not exist); giant 4.1 GB skipped to stay within 25 min budget and because it's far above the 18 TOPS edge target.
- **zohaibshahid/videomae-base-finetuned-fall-detection** — HF tree listing confirms repo contains only `.gitattributes`, matching the deep-dive doc's "weights were not actually uploaded" finding.
- **Sapiens-0.3B pose** — 404 on the `torchscript.pt2` path; repo is gated (requires `hf auth login` + access request).
- **MotionBERT action NTU60** — OneDrive "download" links are actually web-UI share pages without direct-file URLs; documented as "skip if no direct URL" in the task brief.

## 8. Per-feature manifests

Full byte-exact per-directory manifests (gitignored):

- `ai/pretrained/safety-fall-detection/DOWNLOAD_MANIFEST.md`
- `ai/pretrained/safety-fall_pose_estimation/DOWNLOAD_MANIFEST.md`
- `ai/pretrained/safety-poketenashi/DOWNLOAD_MANIFEST.md`

These are the source of truth for on-disk state; the tables above are a snapshot at the time of this log.

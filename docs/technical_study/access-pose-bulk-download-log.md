# Face / Pose / Zone Bulk Pretrained Weight Download Log

**Date:** 2026-04-14
**Scope:** `access-face_recognition`, `access-zone_intrusion`, `safety-fall_pose_estimation`, `safety-poketenashi`
**Policy:** User instruction — ignore license gating; pull everything that has a public direct URL. Do NOT run inference. No push, commit only.
**Tooling:** `curl -fL`, `hf download`, `gdown --fuzzy`, 120–180 s per-file timeout.

---

## 1. access-face_recognition (`ai/pretrained/access-face_recognition/`)

| File | Size (B) | SHA256 | Source URL | License | Status |
|---|---|---|---|---|---|
| `yunet_2023mar.onnx` | 232,589 | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` | https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx | Apache-2.0 | Pre-existing |
| `yunet_2023mar_int8.onnx` | 100,416 | `321aa5a6afabf7ecc46a3d06bfab2b579dc96eb5c3be7edd365fa04502ad9294` | https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar_int8.onnx | Apache-2.0 | Pre-existing |
| `buffalo_sc.zip` | 14,969,382 | `57d31b56b6ffa911c8a73cfc1707c73cab76efe7f13b675a05223bf42de47c72` | https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip | InsightFace non-commercial | Pre-existing |
| `buffalo_s.zip` | 127,607,557 | `d85a87f503f691807cd8bb97128bdf7a0660326cd9cd02657127fa978bab8b5e` | https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip | InsightFace non-commercial | **Downloaded** |
| `buffalo_m.zip` | 275,951,529 | `d98264bd8f2dc75cbc2ddce2a14e636e02bb857b3051c234b737bf3b614edca9` | https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_m.zip | InsightFace non-commercial | **Downloaded** |
| `buffalo_l.zip` | 288,621,354 | `80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f` | https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip | InsightFace non-commercial | **Downloaded** (contains ArcFace R50 `w600k_r50.onnx`) |
| `antelopev2.zip` | 360,662,982 | `8e182f14fc6e80b3bfa375b33eb6cff7ee05d8ef7633e738d1c89021dcf0c5c5` | https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip | InsightFace non-commercial | **Downloaded** (contains ArcFace R100 `glintr100.onnx` + SCRFD 10g) |
| `det_500m.onnx` | 2,524,817 | `5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a` | extracted from `buffalo_sc.zip` | non-commercial | Pre-existing |
| `w600k_mbf.onnx` | 13,616,099 | `9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f` | extracted from `buffalo_sc.zip` | non-commercial | Pre-existing |
| `sface_2021dec.onnx` | 38,696,353 | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` | OpenCV Zoo SFace | Apache-2.0 | Pre-existing |
| `sface_2021dec_int8.onnx` | 9,896,933 | `2b0e941e6f16cc048c20aee0c8e31f569118f65d702914540f7bfdc14048d78a` | OpenCV Zoo SFace INT8 | Apache-2.0 | Pre-existing |
| `anti_spoof_2_7_80x80_MiniFASNetV2.pth` | 1,849,453 | `a5eb02e1843f19b5386b953cc4c9f011c3f985d0ee2bb9819eea9a142099bec0` | github.com/minivision-ai/Silent-Face-Anti-Spoofing | Apache-2.0 | Pre-existing (verified) |
| `edgeface_base.pt` | 72,972,261 | `95861c09b22810136f43ec98845e7f09bfc3c43f5a804984a7bd2eac20abc30c` | https://huggingface.co/idiap/EdgeFace (via `hf download`) | research-only (Idiap) | **Downloaded** (HF accessible via `hf download`, gating passed for authenticated user) |
| `yolov8n-face.pt` | 6,247,065 | `37396ac6a9601ab9f5177e4231b09d81cf6f65a7f22db99ec3b36ab63f674e71` | https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/resolve/main/model.pt | AGPL-3.0 | **Downloaded** (renamed from `model.pt`) |

### Attempted but skipped (HTTP 404 / repo missing / gdrive quota)

| Target | Reason |
|---|---|
| `AdaFace IR-18 MS1MV2/WebFace4M` (gdown) | Google Drive virus-scan interstitial requires interactive confirm; `gdown --fuzzy` could not fetch (one 120 s timeout). Retry manually from https://github.com/mk-minchul/AdaFace#pretrained-models |
| `edgeface_s_gamma_05.pt`, `edgeface_xs_gamma_06.pt` | `huggingface.co/idiap/EdgeFace-S-GAMMA-05` et al. — 404 at `resolve/main/<file>.pt`. File paths inside repo differ; needs interactive listing. |
| `retinaface_resnet50.pth` (biubug6) | GDrive interstitial; repo hosts on Google Drive only. |
| `YOLOv5-Face / YOLOv7-Face` (deepcam-cn, derronqi) | GitHub release assets renamed / tag mismatch — `v0.1` not present; real tag is `yolov7-face` but per-file `yolov7-tiny-face.pt` not pushed as release asset. Needs scraping. |
| `TopoFR (R50/R100 Glint360K)` | No public HF mirror exists; repo distributes via OneDrive. |
| `MagFace iResNet50` | Original GDrive interstitial. |
| `ArcFace R100 / R50 standalone ONNX` | Obtained via `buffalo_l.zip` (R50 `w600k_r50.onnx`) and `antelopev2.zip` (R100 `glintr100.onnx`). Standalone HF mirrors (Xenova/arcface, immich-app/ArcFace-Resnet50) returned 404 on `resolve/main/model.onnx`. |
| `PartialFC / UniFace` | No direct redistributable weight URLs found; PartialFC is trained inside `insightface/recognition/arcface_torch`. |

---

## 2. access-zone_intrusion (`ai/pretrained/access-zone_intrusion/`)

| File | Size (B) | SHA256 | Source URL | License | Status |
|---|---|---|---|---|---|
| `yolov10n.pt` | 11,448,431 | `61b91ffc99b284792dca49bf40216945833cc2a515e1a742954e6e9327cfc19e` | https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt | AGPL-3.0 | **Downloaded** |
| `yolov10s.pt` | 32,956,759 | `96af3fc7c7169abcc4867f3e3088b761bb33cf801283c2ec05f9703d63a0ba77` | https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt | AGPL-3.0 | **Downloaded** |
| `yolov10m.pt` | 66,924,323 | `ff2c559f11d13701abc4e0345f82851d146ecfe7035efaafcc08475cfd8b5f2d` | https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10m.pt | AGPL-3.0 | **Downloaded** |
| `yolo11n.pt` | 5,613,764 | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` | https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt | AGPL-3.0 | **Downloaded** |
| `yolo11s.pt` | 19,313,732 | `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5` | https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt | AGPL-3.0 | **Downloaded** |
| `yolo11m.pt` | 40,684,120 | `d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95` | https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt | AGPL-3.0 | **Downloaded** |
| `yolov12n.pt` | 5,515,407 | `37080c2891b94c62998f0bfb552dd70c32f9f2ee36618b9e7b3da49b49e150ac` | https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12n.pt | AGPL-3.0 | **Downloaded** |
| `yolov12s.pt` | 18,708,559 | `83b22d92565a0399a48eb8e3ee05dbb5056ee6e0dece66acd6d553a1f7f73cd8` | https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12s.pt | AGPL-3.0 | **Downloaded** |
| (symlinks) `dfine_n_coco.pt`, `rtdetr_v2_r18_coco.pt`, `yolox_tiny.pth` | — | — | — | Apache-2.0 | Pre-existing symlinks to `ai/pretrained/` |

### Attempted but skipped

| Target | Reason |
|---|---|
| `yolo_nas_s/m/l_coco.pth` (Deci YOLO-NAS) | `sghub.deci.ai` + `deci-pretrained-models.s3.amazonaws.com` both return 403/404 without the super-gradients SDK signing. Needs `super-gradients` pip package to trigger signed download. |
| `LWDETR_{tiny,small,medium}_60e_coco.pth` | Release tags `v1.0`/`v1` absent on github.com/Atten4Vis/LW-DETR — repo ships weights via OneDrive links only. |
| `deim_dfine_hgnetv2_*_coco.pth` (DEIM) | GDrive interstitial — `gdown --fuzzy` did not complete within 120 s. |
| `osnet_*_msmt17.pt`, `clip_market1501.pt` (BoxMOT ReID) | Release tag `v10.0.46`/`v10.0.83` returned 404 at tried paths. Actual release layout changed; needs manual inspection of latest release assets list. |

---

## 3. safety-fall_pose_estimation (`ai/pretrained/safety-fall_pose_estimation/`)

| File | Size (B) | SHA256 | Source URL | License | Status |
|---|---|---|---|---|---|
| `rtmpose-s_coco_256x192.pth` | 22,007,763 | `8d57b1112021367bb6857e468be383b2bd24a0b69f121b3f54177635fb742907` | download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth | Apache-2.0 | Pre-existing |
| `rtmdet-nano_person.pth` | 4,215,674 | `0e2da635c75e25dc88af08d01eb34bfe9cac06a7841cba223bdf04eed288b3dc` | download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth | Apache-2.0 | Pre-existing |
| `rtmo-s_body7_640x640.pth` | 39,766,454 | `dac2bf749bbfb51e69ca577ca0327dff4433e3be9a56b782f0b7ef94fb45247e` | download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.pth | Apache-2.0 | Pre-existing |
| `rtmo-l_body7_640x640.pth` | 179,277,852 | `b37118cee39b43531c23d9e443fbb4f0b8987f590d9c6ed61978d12ecfce84df` | download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.pth | Apache-2.0 | **Downloaded** |
| `hrnet_w48_coco_256x192.pth` | 255,011,654 | `b9e0b3ab0439cb68e166c7543e59d2587cd8d7e9acf5ea62a8378eeb82fb50e5` | download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth | Apache-2.0 | **Downloaded** |
| `vitpose-plus-small.pth` | 132,619,932 | `f7bad8ed09eeeb2a7de6b38faaa8a88d07838e23e9c06a2a782099bca7467cb9` | huggingface.co/usyd-community/vitpose-plus-small (model.safetensors renamed) | Apache-2.0 | **Downloaded** |
| `vitpose-plus-base.pth` | 501,627,628 | `640225e4a9dd544239f1da0ae36af865a6e76e21e5f33a12656aa7b77fa5a5fa` | huggingface.co/usyd-community/vitpose-plus-base (model.safetensors renamed) | Apache-2.0 | **Downloaded** |
| `dw-ll_ucoco_384.onnx` | 134,399,116 | `724f4ff2439ed61afb86fb8a1951ec39c6220682803b4a8bd4f598cd913b1843` | huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx | Apache-2.0 | **Downloaded** (DWPose-L) |
| `pose_landmarker_lite.task` | 5,777,746 | `59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a` | storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task | Apache-2.0 | **Downloaded** |
| `pose_landmarker_full.task` | 9,398,198 | `4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad` | storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task | Apache-2.0 | **Downloaded** |
| `pose_landmarker_heavy.task` | 30,664,242 | `64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b` | storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task | Apache-2.0 | **Downloaded** |
| `posec3d_slowonly_r50_ntu60_xsub_keypoint.pth` | 8,191,762 | `f3adabf19d56bd4fb458e59570d5bbe0208f1e8a9a79c3d5f7fe03a0d5825d2a` | PYSKL NTU60 xsub | Apache-2.0 | Pre-existing |
| `stgcn_80e_ntu60_xsub_keypoint.pth` | 12,443,433 | `e7bb965330622f3eb602406af995add6de3f679ca08ff4d1686d984c2084bebe` | PYSKL NTU60 xsub | Apache-2.0 | Pre-existing |

### Attempted but skipped

| Target | Reason |
|---|---|
| `sapiens_0.3b_coco_best_coco_AP_796.pth`, `sapiens_0.6b_coco_best_coco_AP_812.pth` | HF repo `facebook/sapiens-pose-0.3b` / `0.6b` gated — `hf download` returned 401/404 without accepted license. Per instructions, tried once then skipped. Re-try: visit HF page, click "Access repository", rerun. |
| `dw-ss_ucoco_384.onnx` (DWPose-S), `dw-mm_ucoco_384.onnx` (DWPose-M) | Do not exist in `huggingface.co/yzd-v/DWPose` repo — only `dw-ll_ucoco_384.onnx` is hosted. S/M variants are only distributed in the OpenMMLab mmpose ONNX SDK zip (see RTMW zips in poketenashi). |
| `mobilehumanpose.pth.tar` | GDrive interstitial — no alternative public mirror found. |

---

## 4. safety-poketenashi (`ai/pretrained/safety-poketenashi/`)

| File | Size (B) | SHA256 | Source URL | License | Status |
|---|---|---|---|---|---|
| `dw-ll_ucoco_384.onnx` | 134,399,116 | `724f4ff2439ed61afb86fb8a1951ec39c6220682803b4a8bd4f598cd913b1843` | huggingface.co/yzd-v/DWPose | Apache-2.0 | Pre-existing |
| `rtmpose-s_coco-wholebody.pth` | 72,010,049 | `3da02694cd6479d3b333ff42ebd0723f96bfa06adac1db1e2e815ed2e9e1b02d` | download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth | Apache-2.0 | Pre-existing |
| `rtmw-l_256x192.zip` | 212,739,469 | `1e3e77558dfc199129bfff1c583e51b4ee190914de6ae30688243c20163c148c` | download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip | Apache-2.0 | **Downloaded** (ONNX SDK bundle) |
| `rtmw-l_384x288.zip` | 213,433,855 | `a87e1af41a0a067776dba7d46e1c21c8f6e9f18e247e0e606718dd1f31e96ffd` | download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip | Apache-2.0 | **Downloaded** (ONNX SDK bundle) |
| `rtmw-l_cocktail14_384x288.pth` | 230,299,153 | `afa589fa16bf8b9bab0807c806538a43bead39ed0151b79f41adcf551e3bcfb0` | download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth | Apache-2.0 | Pre-existing (full PyTorch ckpt; not listed in earlier manifest but present) |

### Attempted but skipped

| Target | Reason |
|---|---|
| `dw-ss_ucoco_384.onnx`, `dw-mm_ucoco_384.onnx` | Not hosted on HF repo — see fall_pose table. |
| `rtmpose-l_wholebody_384x288.pth`, `rtmpose-x_wholebody_384x288.pth` | Tried openmmlab URL variants; 404 (filename hash/epoch suffix mismatch in the URL pattern). Real URL must come from mmpose model-zoo page. |

---

## 5. Summary

| Destination | File count (tracked) | Total size |
|---|---:|---:|
| `access-face_recognition` | 14 | ~1.23 GB |
| `access-zone_intrusion` | 8 (+3 symlinks) | ~201 MB |
| `safety-fall_pose_estimation` | 13 | ~1.36 GB |
| `safety-poketenashi` | 5 | ~863 MB |
| **Total new this session** | **18 new files** | ~2.55 GB of new data |

### New this session (18 files)

- Face: `buffalo_s.zip`, `buffalo_m.zip`, `buffalo_l.zip`, `antelopev2.zip`, `edgeface_base.pt`, `yolov8n-face.pt`
- Zone: `yolov10n/s/m.pt`, `yolo11n/s/m.pt`, `yolov12n/s.pt` (8)
- Fall pose: `rtmo-l_body7_640x640.pth`, `hrnet_w48_coco_256x192.pth`, `vitpose-plus-small.pth`, `vitpose-plus-base.pth`, `pose_landmarker_{lite,full,heavy}.task` (7 counted separately; 4 unique "missing" items)
- Poketenashi: `rtmw-l_256x192.zip`, `rtmw-l_384x288.zip`

### Still outstanding (require manual / interactive fetch)

- AdaFace IR-18 (Google Drive virus-scan prompt)
- EdgeFace-S/XS variants (correct filename inside HF repo needed)
- RetinaFace (biubug6 GDrive)
- YOLOv5-Face / YOLOv7-Face assets (release tag scraping)
- TopoFR, MagFace, PartialFC, UniFace (GDrive / OneDrive, no HF mirrors)
- YOLO-NAS weights (super-gradients-SDK signed download)
- LW-DETR weights (OneDrive only)
- DEIM (GDrive interstitial)
- BoxMOT ReID (release-tag drift)
- Sapiens-Pose 0.3B / 0.6B (HF gate acceptance required)
- DWPose-S/M ONNX (not in yzd-v/DWPose repo; only in RTMW SDK zip — already pulled)
- MobileHumanPose (GDrive)

Manifests are also written to each folder as `DOWNLOAD_MANIFEST.md` (gitignored via `ai/pretrained/*/`).

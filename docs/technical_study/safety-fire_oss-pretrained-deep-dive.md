# Fire / Smoke — Deep OSS Pretrained Model Survey

> Follow-up to `safety-fire_detection-sota.md`. That pass covered generic
> COCO detectors (YOLOX, D-FINE, RT-DETRv2, RF-DETR). Our fire evaluation
> confirmed no COCO-only model works out of the box on fire/smoke. This
> deep dive looks specifically for **weights pretrained on fire/smoke
> data** — detection, classification, and segmentation — usable either
> directly or as a warm start for fine-tuning.

## 1. Scope

Targets searched (HF Hub, Roboflow Universe, GitHub, Zenodo, SciDB,
IEEE DataPort):

- Detection: YOLO-family / DETR-family / RT-DETR / D-FINE / YOLOX fine-tunes
  on D-Fire, FASDD, FIRESENSE, Pyro-SDIS, OpenFire, FLAME, Roboflow fire sets.
- Classification: CNN / ViT / EfficientNet / MobileNet fire vs no-fire.
- Segmentation: U-Net / Mask2Former / ViT-based smoke segmenters.

Every URL was verified with `curl -I` before download. All downloads
landed at
`/home/nthanhtrung/Documents/05_Team/02_Vietsol/01_Projects/edge_ai/ai/pretrained/safety-fire_detection/`
with SHA256 recorded in §6.

## 2. Fire/smoke **detection** pretrained weights

| Name | Arch | Source | Dataset | License (weights) | Reported metrics | Commercial? |
|---|---|---|---|---|---|---|
| **pyronear/yolo11s_nimble-narwhal_v6.0.0** | YOLO11-S (Ultralytics) @ 1024 | [hf.co](https://hf.co/pyronear/yolo11s_nimble-narwhal_v6.0.0) | pyronear/pyro-dataset (Pyro-SDIS, smoke-focused) | Weights tagged **Apache-2.0**; base/framework Ultralytics **AGPL-3.0** (see §7) | single-class wildfire smoke; manifest only | **Conditional** — AGPL contamination risk |
| pyronear/yolo11s_sensitive-detector_v1.0.0 | YOLO11-S | [hf.co](https://hf.co/pyronear/yolo11s_sensitive-detector_v1.0.0) | pyro-dataset | Apache-2.0 / AGPL via Ultralytics | higher recall variant | Conditional |
| pyronear/yolov8s | YOLOv8-S | [hf.co](https://hf.co/pyronear/yolov8s) | Pyronear wildfire | Apache-2.0 tag / Ultralytics AGPL base | ~640 input, ONNX provided | Conditional |
| pyronear/yolov11n | YOLOv11-N | [hf.co](https://hf.co/pyronear/yolov11n) | pyro-dataset | Apache-2.0 / AGPL | nano variant | Conditional |
| TommyNgx/YOLOv10-Fire-and-Smoke-Detection | YOLOv11/v10 fine-tune | [hf.co](https://hf.co/TommyNgx/YOLOv10-Fire-and-Smoke-Detection) | "fire-smoke-dataset" | Apache-2.0 tag / **Ultralytics AGPL** | model-index in card; **gated** | No — AGPL + gated |
| SalahALHaismawi/yolov26-fire-detection | YOLOv26-S (Ultralytics) | [hf.co](https://hf.co/SalahALHaismawi/yolov26-fire-detection) | Roboflow Universe "Fire Detection" 8.9K (CC-BY 4.0) | **MIT** tag / Ultralytics AGPL base | mAP50 94.9, mAP50-95 68.0, P 89.6, R 88.8 | No — AGPL base |
| touati-kamel/yolov8s-forest-fire-detection | YOLOv8-S | [hf.co](https://hf.co/touati-kamel/yolov8s-forest-fire-detection) | forest-fire-detection | no license tag / Ultralytics AGPL | none reported | No |
| touati-kamel/yolov10n-forest-fire-detection | YOLOv10-N | [hf.co](https://hf.co/touati-kamel/yolov10n-forest-fire-detection) | forest-fire | Ultralytics AGPL | none | No |
| touati-kamel/yolov12n-forest-fire-detection | YOLOv12-N | [hf.co](https://hf.co/touati-kamel/yolov12n-forest-fire-detection) | forest-fire | MIT tag / Ultralytics AGPL | none | No |
| JJUNHYEOK/yolov8n_wildfire_detection | YOLOv8-N | [hf.co](https://hf.co/JJUNHYEOK/yolov8n_wildfire_detection) | wildfire | Ultralytics AGPL | none | No |
| Mehedi-2-96/fire-smoke-detection-yolo | YOLO (unspecified) | [hf.co](https://hf.co/Mehedi-2-96/fire-smoke-detection-yolo) | undocumented | unknown | none | No — unlabeled |
| pedbrgs/Fire-Detection (YOLOv5-S/L) | YOLOv5 | [GitHub](https://github.com/pedbrgs/Fire-Detection) | D-Fire (21K) | Repo MIT / **YOLOv5 is GPL-3.0 / AGPL-era Ultralytics** | D-Fire splits in paper | Conditional — copy-left base |
| spacewalk01/yolov5-fire-detection | YOLOv5 | [GitHub](https://github.com/spacewalk01/yolov5-fire-detection) | custom ~10K | MIT / YOLOv5 GPL | mAP ~0.85 reported | Conditional |
| daimakram/Forest-Fire-Detection (Deformable DETR) | Deformable DETR | [GitHub](https://github.com/daimakram/Forest-Fire-Detection) | Forest Fire (custom) | Apache-2.0 (DETR) | no public eval | Yes — academic, non-Ultralytics |
| FASDD-trained Swin Transformer Object Detection | Swin-T + Cascade / Mask R-CNN | [FASDD paper](https://doi.org/10.1080/10095020.2024.2347922) / [openrsgis/FASDD](https://github.com/openrsgis/FASDD) | FASDD_CV / UAV / RS | MMDet Apache-2.0 / dataset license **not specified** in repo (see §7) | mAP 84.9 / 89.7 / 74.0 on sub-datasets | No redistributable weights published; code + data only |
| MS-FSDB benchmark models | RT-DETR, DINO, Co-DETR, etc. | [XiaoyiHan6/MS-FSDB](https://github.com/xiaoyihan6/ms-fsdb) / [FSD page](https://xiaoyihan6.github.io/FSD/) | Multi-scene FSD benchmark | Apache-2.0 (MMDet) | Paper [2410.16631](https://hf.co/papers/2410.16631) + [2410.16642](https://hf.co/papers/2410.16642) | Code + configs, weights require request |
| RT-DETR-Smoke | RT-DETR-R18 + hybrid encoder | [MDPI paper](https://www.mdpi.com/2571-6255/8/5/170) | Forest smoke | Paper-only; code not public at time of writing | mAP50 87.75, 445 FPS reported | No weights released |
| FSH-DETR | Deformable DETR + ECPConv | [MDPI](https://www.mdpi.com/1424-8220/24/13/4077) | Fire+Smoke+Human | Paper + code (GitHub referenced in PMC) | mAP50 84.2 | Weights availability unclear |
| Smoke-DETR | RT-DETR variant | [MDPI](https://www.mdpi.com/2571-6255/7/12/488) | Early fire smoke | Paper | mAP50 ~86 | No weights |

**Bottom line for detection.** Almost every HF "fire-detection" listing
with meaningful download count is an **Ultralytics YOLO fine-tune** (v5
through v26). The only HF-hosted, non-Ultralytics, fire-specific
detection weights we found are **pyronear's** ONNX exports (still
trained via Ultralytics so only their ONNX is cleanly distributable) and
academic Deformable-DETR / RT-DETR variants that did not publish weights.

## 3. Fire/smoke **classification** pretrained weights

| Name | Arch | Source | Dataset | License | Metrics | Commercial? |
|---|---|---|---|---|---|---|
| **pyronear/rexnet1_3x** | ReXNet-1.3x + SE | [hf.co](https://hf.co/pyronear/rexnet1_3x) (38K DL) | pyronear/openfire | **Apache-2.0** | not in card; ONNX + PyTorch provided | **Yes** |
| pyronear/rexnet1_0x | ReXNet-1.0x | [hf.co](https://hf.co/pyronear/rexnet1_0x) | openfire | Apache-2.0 | ONNX shipped | Yes |
| pyronear/rexnet1_5x | ReXNet-1.5x | [hf.co](https://hf.co/pyronear/rexnet1_5x) | openfire | Apache-2.0 | ONNX shipped | Yes |
| **pyronear/mobilenet_v3_small** | MobileNetV3-S | [hf.co](https://hf.co/pyronear/mobilenet_v3_small) | openfire | Apache-2.0 | ONNX + PT | Yes — lightest edge option |
| pyronear/mobilenet_v3_large | MobileNetV3-L | [hf.co](https://hf.co/pyronear/mobilenet_v3_large) | openfire | Apache-2.0 | ONNX + PT | Yes |
| pyronear/resnet18 | ResNet-18 | [hf.co](https://hf.co/pyronear/resnet18) | openfire | Apache-2.0 | ONNX + PT | Yes |
| pyronear/resnet34 | ResNet-34 | [hf.co](https://hf.co/pyronear/resnet34) | openfire | Apache-2.0 | ONNX + PT | Yes |
| shawnmichael/vit-fire-smoke-detection-v4 | ViT-B/16 | [hf.co](https://hf.co/shawnmichael/vit-fire-smoke-detection-v4) | undocumented | Apache-2.0 | metrics blank in card | Yes but under-documented |
| shawnmichael/convnext-tiny-fire-smoke-detection-v1 | ConvNeXt-V2-T | [hf.co](https://hf.co/shawnmichael/convnext-tiny-fire-smoke-detection-v1) | undocumented | Apache-2.0 | blank | Conditional |
| shawnmichael/swin-fire-smoke-detection-v1 | Swin-T | [hf.co](https://hf.co/shawnmichael/swin-fire-smoke-detection-v1) | undocumented | Apache-2.0 | blank | Conditional |
| shawnmichael/efficienetb2-fire-smoke-detection-v1 | EfficientNet-B2 | [hf.co](https://hf.co/shawnmichael/efficienetb2-fire-smoke-detection-v1) | undocumented | Apache-2.0 | blank | Conditional |
| shawnmichael/vit-large-fire-smoke-detection-v1 | ViT-L/16 | [hf.co](https://hf.co/shawnmichael/vit-large-fire-smoke-detection-v1) | undocumented | Apache-2.0 | blank | Conditional |
| dima806/wildfire_types_image_detection | ViT-B/16 | [hf.co](https://hf.co/dima806/wildfire_types_image_detection) | custom | Apache-2.0 | multi-class wildfire types | Yes |
| FireRisk MAE | MAE / ViT | [paper 2303.07035](https://hf.co/papers/2303.07035) | FireRisk (91,872 imgs, 7 classes) | code/weights — academic | top-1 65.3% | Conditional |

## 4. Fire/smoke **segmentation** pretrained weights

| Name | Arch | Source | Dataset | License | Notes |
|---|---|---|---|---|---|
| **sequoiaandrade/Smoke-Cloud-Segmentation-RACE-ODIN** | U-Net-style (TF-Keras) | [hf.co](https://hf.co/sequoiaandrade/Smoke-Cloud-Segmentation-RACE-ODIN) | NASA RACE-ODIN | **Apache-2.0** (NASA) | Published in Computers & Geosciences 2025 (`10.1016/j.cageo.2025.105960`). Needs conversion to PyTorch. |
| Shoriful025/wildfire_smoke_segmentation_vit | ViT segmenter | [hf.co](https://hf.co/Shoriful025/wildfire_smoke_segmentation_vit) | custom wildfire | MIT | very small, unverified metrics |
| SmokeyNet | spatiotemporal CNN | [paper 2112.08598](https://hf.co/papers/2112.08598) | FIgLib | repo MIT | wildland smoke time-series |
| UAV real-time smoke segmenter | distilled student from SAM-family teacher | [paper 2408.10843](https://hf.co/papers/2408.10843) | proprietary | paper-only at time of writing | Jetson Orin NX benchmarks |
| Hyperspectral Smoke MoP | ViT + MoE | [paper 2602.10858](https://hf.co/papers/2602.10858) | hyperspectral | paper-only | needs HS input — not RGB-usable |
| Unsupervised IR smoke seg | MRF + optical flow | [paper 1909.12937](https://hf.co/papers/1909.12937) | IR | classical, no NN weights | — |

## 5. Top 3 recommendations

Criteria: (a) weights actually downloadable today, (b) license cleanly
usable commercially, (c) relevant to factory-scene fire/smoke, (d) can
be used directly for eval or as warm-start for fine-tuning.

### #1 — Pyronear OpenFire classifier ensemble (warm-start for our classifier head / false-positive suppressor)

- **Repos (Apache-2.0):**
  [pyronear/rexnet1_3x](https://hf.co/pyronear/rexnet1_3x),
  [pyronear/mobilenet_v3_small](https://hf.co/pyronear/mobilenet_v3_small),
  [pyronear/resnet18](https://hf.co/pyronear/resnet18).
- **Why.** These are the **only widely-downloaded (38K+), non-Ultralytics,
  fire-domain checkpoints** on HF Hub. Clean Apache-2.0 chain (ReXNet,
  MobileNetV3, ResNet backbones — all permissive), ONNX + PyTorch shipped.
  Trained on OpenFire (web-crawled wildfire classification). Excellent
  candidates for a **"is-fire?" post-classifier** that cuts the FP rate
  of our detector without touching the detector's license.
- **Fine-tune path.** Load via `pyrovision.models.model_from_hf_hub` or
  directly with transformers `AutoModelForImageClassification`. Fine-tune
  on cropped positive+negative patches from our FASDD+D-Fire+factory
  footage. Use as 2-stage filter behind YOLOX / D-FINE.

### #2 — Pyronear YOLO11s/YOLOv8s wildfire detector (reference eval only; license-gated for production)

- **Repos (Apache-2.0 tag):**
  [pyronear/yolo11s_nimble-narwhal_v6.0.0](https://hf.co/pyronear/yolo11s_nimble-narwhal_v6.0.0),
  [pyronear/yolov8s](https://hf.co/pyronear/yolov8s).
- **Why.** Only fire-domain detection checkpoint on HF Hub with a fully
  documented training manifest (Pyro-SDIS @ 1024×1024, 50 ep, AdamW,
  single-class smoke). Best candidate to benchmark **"what can a pure
  smoke-specialist do on our scenes?"** — tells us the achievable
  recall floor on diffuse smoke.
- **Fine-tune path.** Use **only for offline evaluation and for
  distilling soft labels** onto our Apache-2.0 detector. Do **NOT** ship
  YOLO11 weights — see §7 (Ultralytics AGPL).

### #3 — NASA Smoke-Cloud-Segmentation-RACE-ODIN (segmentation prior / training signal)

- **Repo (Apache-2.0):** [sequoiaandrade/Smoke-Cloud-Segmentation-RACE-ODIN](https://hf.co/sequoiaandrade/Smoke-Cloud-Segmentation-RACE-ODIN).
- **Why.** Rare pixel-level smoke segmenter with a clean government-origin
  Apache-2.0 license. Useful as a **pseudo-mask generator** for our
  unlabeled factory smoke footage, feeding auxiliary training signal
  into a detector's segmentation head (or a separate Mask2Former smoke
  head) without pulling in AGPL YOLO code.
- **Fine-tune path.** Convert Keras weights → ONNX → PyTorch (or
  reimplement the U-Net head), predict masks on in-domain smoke frames,
  use as auxiliary supervision.

## 6. Download manifest

Absolute base path:
`/home/nthanhtrung/Documents/05_Team/02_Vietsol/01_Projects/edge_ai/ai/pretrained/safety-fire_detection/`

| Relative path | Source | Bytes | SHA256 |
|---|---|---|---|
| `pyronear_rexnet1_3x/pytorch_model.bin` | hf.co/pyronear/rexnet1_3x | 23,909,793 | `6445365e219389fb3bbe6e37f3a8442e30a5faebc014a3d18719d552efb2296d` |
| `pyronear_rexnet1_3x/model.onnx` | same | 23,588,871 | `5332bcb6bdf9d5296b5604488478b6104316e3aac0d1da96adae6ff13cd36050` |
| `pyronear_rexnet1_3x/config.json` | same | 167 | `b2f079b96e62c097d98fbf57596b0e42504e0719bcfc118da041ad88b4bab54d` |
| `pyronear_mobilenet_v3_small/pytorch_model.bin` | hf.co/pyronear/mobilenet_v3_small | 6,204,889 | `619caf973a8a509a247068dbfa24e546e391d2242c0639a490adc21c3a05febe` |
| `pyronear_mobilenet_v3_small/model.onnx` | same | 6,068,953 | `8fd451f919499e30e879eda19bfd2b249ceec77e4f1c02f7b022730e365ae897` |
| `pyronear_mobilenet_v3_small/config.json` | same | 175 | `b1a497877bca220a56798f2aac2cd085dd1ffa1f111536440ad8a4828b411e59` |
| `pyronear_yolo11s_nimble-narwhal_v6/best.pt` | hf.co/pyronear/yolo11s_nimble-narwhal_v6.0.0 | 19,225,626 | `0bf3c7ee9f720c26613c30719fea32f47ed04fc384e443de72414d9f8148ac9d` |
| `pyronear_yolo11s_nimble-narwhal_v6/manifest.yaml` | same | 13,490 | `2726ee21891cd2cdd705b3b8f76fe041bfc43f4816c382983a39980e69a8e617` |

Four checkpoints pulled: two classifiers (Apache-2.0, clean) and one
detector (Apache-2.0 tag but AGPL-contaminated — for eval only; see §7).

## 7. Licensing landmines

- **Ultralytics YOLO fine-tunes dominate HF Hub for "fire-detection".**
  YOLOv5 through YOLOv12 and the recently published YOLOv26 are all
  released by Ultralytics under **AGPL-3.0**. Any `best.pt` produced by
  `from ultralytics import YOLO; model.train(...)` is a **derivative
  work** of Ultralytics' AGPL code. Re-tagging the model card
  `license: apache-2.0` or `license: mit` does **not** change that.
  Distributing, or even network-serving, that `best.pt` triggers AGPL
  copyleft on our entire inference stack unless we buy an Ultralytics
  Enterprise license.
  - Dozens of HF repos fall into this trap: `touati-kamel/*`,
    `SalahALHaismawi/yolov26-fire-detection` (MIT-tagged),
    `TommyNgx/YOLOv10-Fire-and-Smoke-Detection` (Apache-tagged, gated),
    `Mehedi-2-96/fire-smoke-detection-yolo` (untagged),
    `JJUNHYEOK/yolov8n_wildfire_detection`, `Ziad-AI19/yolov8-fire-detection`,
    `galaxypulsar/yolov11-fire-detection`, `touati-kamel/yolov12n-*`,
    `pedbrgs/Fire-Detection` (YOLOv5), `spacewalk01/yolov5-fire-detection`.
  - **Pyronear's YOLO repos are no exception.** `manifest.yaml` in
    `pyronear/yolo11s_nimble-narwhal_v6.0.0` shows
    `model_type: yolo11s.pt`, `mode: train` via the Ultralytics CLI. The
    Pyronear Apache-2.0 tag covers *their* additions; the base weights
    and training framework remain AGPL. Treat these as **eval-only**
    until legal confirms Ultralytics' "no copyleft on weight values"
    carve-out (which they do **not** grant in writing).
- **FASDD dataset license is unspecified** in the openrsgis GitHub repo
  (no `LICENSE` file). The SciDB mirror has terms of use but is not an
  OSI license. Training and *using* the dataset internally is likely
  fine; **redistributing derived weights as commercial product carries
  risk**. Assume "research use" until we contact the authors.
- **Roboflow Universe fire/smoke datasets** default to CC-BY 4.0 or
  CC-BY-NC; check each dataset's card before fine-tuning a commercial
  model on them. The YOLOv26 model in §2 used an 8.9K CC-BY 4.0 fire
  set — commercial-usable only if we preserve attribution.
- **D-Fire dataset** (gaiasd) ships a LICENSE in-repo (not fully
  extracted above) — verify terms before production.
- **FLAME / FIgLib / WIT-UAS / FireRisk** are academic datasets.
  Weights trained on them are typically **research-only** unless the
  authors state otherwise. FireRisk MAE (paper 2303.07035) weights
  inherit that constraint.
- **Paper-only checkpoints** (RT-DETR-Smoke, FSH-DETR, Smoke-DETR,
  hyperspectral MoP) have no publicly-downloadable weights today; each
  requires author contact. Do not plan the roadmap around them.

**Green-light tier** (clean, downloadable, Apache-2.0, non-AGPL):
pyronear classifiers (ReXNet, MobileNetV3, ResNet-18/34 on OpenFire),
NASA RACE-ODIN smoke segmenter, dima806 ViT wildfire classifier, and
academic Deformable-DETR / MMDet-Swin code where weights can be re-
trained from scratch on D-Fire/FASDD under Apache-2.0.

## 8. References

- Pyronear Hub: [org page](https://hf.co/pyronear),
  [pyro-sdis dataset](https://hf.co/datasets/pyronear/pyro-sdis),
  [openfire dataset](https://hf.co/datasets/pyronear/openfire),
  [pyrovision GitHub](https://github.com/pyronear/pyro-vision).
- D-Fire dataset: [gaiasd/DFireDataset](https://github.com/gaiasd/DFireDataset),
  trained-model fork [pedbrgs/Fire-Detection](https://github.com/pedbrgs/Fire-Detection).
- FASDD: [openrsgis/FASDD](https://github.com/openrsgis/FASDD),
  paper [10.1080/10095020.2024.2347922](https://doi.org/10.1080/10095020.2024.2347922),
  [SciDB mirror](https://www.scidb.cn/en/detail?dataSetId=ce9c9400b44148e1b0a749f5c3eb0bda).
- MS-FSDB: [repo](https://github.com/xiaoyihan6/ms-fsdb),
  [site](https://xiaoyihan6.github.io/FSD/),
  papers [2410.16631](https://hf.co/papers/2410.16631) /
  [2410.16642](https://hf.co/papers/2410.16642).
- DETR-family fire: [RT-DETR-Smoke](https://www.mdpi.com/2571-6255/8/5/170),
  [FSH-DETR](https://www.mdpi.com/1424-8220/24/13/4077),
  [Smoke-DETR](https://www.mdpi.com/2571-6255/7/12/488),
  [Deformable-DETR fine-tune](https://github.com/daimakram/Forest-Fire-Detection).
- Segmentation: [NASA RACE-ODIN](https://hf.co/sequoiaandrade/Smoke-Cloud-Segmentation-RACE-ODIN),
  [SmokeyNet / FIgLib](https://hf.co/papers/2112.08598),
  [UAV real-time segmenter](https://hf.co/papers/2408.10843),
  [FireRisk MAE](https://hf.co/papers/2303.07035).
- Ultralytics licensing: [ultralytics/ultralytics LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
  (AGPL-3.0).
- Prior internal study: `ai/docs/technical_study/safety-fire_detection-sota.md`.

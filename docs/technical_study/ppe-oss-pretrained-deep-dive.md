# PPE OSS Pretrained Deep-Dive (2026-04-14)

> Follow-up to `ppe-helmet_detection-sota.md` and `ppe-shoes_detection-sota.md`.
> Those surveys covered generic COCO / Objects365 backbones (YOLOX, D-FINE,
> RT-DETRv2, DINOv2, EfficientFormerV2, RF-DETR). This pass looks specifically
> for **PPE-dataset-native pretrained weights** we can use as warm-starts or
> drop-in fine-tunes for helmet / vest / shoes / mask / gloves / goggles /
> harness.

## 1. Scope

| Already covered (generic COCO/Obj365) | New focus (PPE-native finetunes) |
|---|---|
| YOLOX-M/S/Tiny (Apache-2.0) | YOLOS-tiny on Hard-Hat (Apache-2.0) |
| D-FINE-N/S/M (Apache-2.0) | DETR-R50 on 5-class PPE (Apache-2.0) |
| RT-DETRv2-R18/R50 (Apache-2.0) | DETR-R50 on Safety-Vest (Apache-2.0) |
| RF-DETR Nano/Small (Apache-2.0) | YOLOS-tiny on 3-class PPE (Apache-2.0) |
| DINOv2-S, EfficientFormerV2-S0/S1 | DETA-on-Helmet (MIT, large) |
| Grounding-DINO (teacher only) | Keremberke YOLOv5 (public domain weights) |
|  | **Flagged AGPL forest**: Keremberke YOLOv8, Hexmon YOLO, Tanishjain YOLOv8n-6cls, leeyunjai YOLO11/YOLO26, Bhavani-23 Ocularone (YOLOv8/11), SH17 YOLOv8/v9/v10 |

Source channels queried: Hugging Face Hub (search + API), GitHub topic
`ppe-detection`, Roboflow Universe, Zenodo, arXiv-linked releases.

## 2. Helmet-specific pretrained weights

| Name | Arch | Source | Dataset | License | Reported mAP | Commercial? |
|---|---|---|---|---|---|---|
| **DunnBC22/yolos-tiny-Hard_Hat_Detection** | YOLOS-tiny (6.5 M) | https://huggingface.co/DunnBC22/yolos-tiny-Hard_Hat_Detection | keremberke/hard-hat-detection (5 K imgs, 2 cls) | **Apache-2.0** | AP 0.346 / AP50 0.747 | **Yes** |
| gghsgn/safety_helmet_detection | YOLOS (6.5 M) | https://huggingface.co/gghsgn/safety_helmet_detection | Motorcycle helmet (D/P1/P2/motorbike, 7 cls) | MIT | — (no card) | Yes — but **motorbike-rider domain**, not industrial |
| gghsgn/helmet_detection | YOLOS | https://huggingface.co/gghsgn/helmet_detection | Same motorcycle dataset (6 cls) | Apache-2.0 | — | Yes — motorbike domain |
| TheNobody-12/HelmetDETA | DETA (219 M) | https://huggingface.co/TheNobody-12/HelmetDETA | Helmet (9 labels, LABEL_0..8 anon) | MIT | — | Yes but **labels unlabelled, 878 MB weights — too heavy for edge** |
| Keremberke YOLOv5n/m/s — hard-hat | YOLOv5 | https://huggingface.co/keremberke/yolov5n-hard-hat-detection etc. | keremberke/hard-hat-detection | **GPL-3.0** (YOLOv5 Ultralytics pre-AGPL legacy — still copyleft) | mAP50 ~0.78 | **Flag — copyleft** |
| Keremberke YOLOv8n/s/m — hard-hat | YOLOv8 | https://huggingface.co/keremberke/yolov8m-hard-hat-detection | same | **AGPL-3.0** | mAP50 0.88 | **No — AGPL** |
| uisikdag/yolo-v5-hard-hat-detection | YOLOv5 | https://huggingface.co/uisikdag/yolo-v5-hard-hat-detection | SHWD subset | GPL-3.0 | — | **Flag — copyleft** |
| lanseria/yolov8n-hard-hat-detection_web_model | YOLOv8 tfjs | https://huggingface.co/lanseria/yolov8n-hard-hat-detection_web_model | hard-hat | MIT card / AGPL base | — | **Flag — derivative of AGPL** |
| dxvyaaa/yolo_helmet | YOLOv3/5/8/9/10 | https://huggingface.co/dxvyaaa/yolo_helmet | undisclosed | AGPL-3.0 | — | **No** |
| HudatersU/Safety_helmet | YOLOv8 ONNX | https://huggingface.co/HudatersU/Safety_helmet | — | GPL-3.0 | — | **Flag** |
| leeyunjai/yolo11-helmet, yolo26-helmet | YOLOv11 / YOLO26 | https://huggingface.co/leeyunjai/yolo11-helmet | factory helmet | Tagged undefined, inherits **AGPL** base | — | **Flag — AGPL base** |

## 3. Vest-specific pretrained weights

| Name | Arch | Source | Dataset | License | mAP | Commercial? |
|---|---|---|---|---|---|---|
| **uisikdag/autotrain-detr-resnet-50-safety-vest-detection** | DETR-R50 (41.6 M) | https://huggingface.co/uisikdag/autotrain-detr-resnet-50-safety-vest-detection | uisikdag/safetyvest (3 cls) | no header (base = DETR Apache-2.0, dataset author's) — treat as **Apache-2.0-inherited** | map 0.43 / map50 0.75 | Yes (base permissive) |
| Bhavani-23/Ocularone-Hazard-Vest-Dataset-Models | YOLOv8n/m/x + YOLOv11n/m/x | https://huggingface.co/Bhavani-23/Ocularone-Hazard-Vest-Dataset-Models | Ocularone hazard-vest | **Apache-2.0 card, but weights are Ultralytics YOLOv8/11 → AGPL-inherited** | — | **Flag — AGPL base** |
| wesjos/Yolo-hard-hat-safety-vest | YOLO11 | https://huggingface.co/wesjos/Yolo-hard-hat-safety-vest | construction vest+hat | Apache-2.0 card / AGPL base | — | **Flag — AGPL base** |
| gungniir/yolo11-vest | YOLO11 | https://huggingface.co/gungniir/yolo11-vest | vest | MIT card / AGPL base | — | **Flag** |

## 4. Shoes / boots pretrained weights

| Name | Arch | Source | Dataset | License | mAP | Commercial? |
|---|---|---|---|---|---|---|
| **No permissive PPE-native shoes/boots detector found on HF Hub.** | — | — | — | — | — | Stick with the two-stage plan in `ppe-shoes_detection-sota.md` (D-FINE-N + DINOv2-small). |
| M3GHAN/YOLOv8 (includes `boots`) | YOLOv8 | https://github.com/M3GHAN/YOLOv8-Object-Detection | custom 8-class | AGPL | — | **Flag** |
| VoxDroid / snehilsanyal YOLOv8 Construction-PPE | YOLOv8 | https://github.com/VoxDroid/Construction-Site-Safety-PPE-Detection | Roboflow PPE (11 cls incl. boots) | AGPL | — | **Flag** |
| Roboflow Universe PPE (many forks) | mostly YOLOv8/v11 | https://universe.roboflow.com/browse/manufacturing/ppe | CHV / Pictor / custom | per-project; majority AGPL-inherited | — | Case-by-case; prefer Apache-2.0 cards |

Shoes remain the weakest-covered PPE class in public OSS. Recommendation stays:
Stage 2 = DINOv2-small fine-tuned on our internal 3.7 K crop set (already
documented in `ppe-shoes_detection-sota.md §3`).

## 5. Multi-class PPE bundles (4+ classes)

| Name | Arch | Source | Classes | License | mAP | Commercial? |
|---|---|---|---|---|---|---|
| **Dricz/ppe-obj-detection** | DETR-R50 (41.6 M) | https://huggingface.co/Dricz/ppe-obj-detection | Coverall, Face_Shield, Gloves, Goggles, Mask (5) | **Apache-2.0** | — (no table) | **Yes** |
| **ikigaiii/yolos-tiny-ppe-detection** | YOLOS-tiny (6.5 M) | https://huggingface.co/ikigaiii/yolos-tiny-ppe-detection | head, helmet, person (3) | no header (base hustvl/yolos-tiny = **Apache-2.0**) | — | Yes (base permissive) |
| Hansung-Cho/yolov8-ppe-detection | YOLOv8 | https://huggingface.co/Hansung-Cho/yolov8-ppe-detection | hardhat, mask, construction PPE | MIT card / **AGPL base** | — | **Flag — AGPL base** |
| Hexmon/vyra-yolo-ppe-detection | YOLOv8 ONNX | https://huggingface.co/Hexmon/vyra-yolo-ppe-detection | PPE combined | CC-BY-4.0 card / **AGPL base** | — | **Flag** |
| melihuzunoglu/ppe-detection | YOLOv11 | https://huggingface.co/melihuzunoglu/ppe-detection | PPE | **AGPL-3.0** (self-declared) | — | **No** |
| Tanishjain9 / itsadityabaniya yolov8n-ppe-6classes | YOLOv8 | https://huggingface.co/Tanishjain9/yolov8n-ppe-detection-6classes | 6 PPE classes | MIT card / **AGPL base** | — | **Flag** |
| VirnectUnityTeam/rfdetr-ppe-detection | (empty placeholder) | https://huggingface.co/VirnectUnityTeam/rfdetr-ppe-detection | — | Apache-2.0 | — | Weights not uploaded (README only) |
| Advantech-EIOT/qualcomm-ultralytics-ppe_detection | YOLOv11 | https://huggingface.co/Advantech-EIOT/qualcomm-ultralytics-ppe_detection | — | **AGPL-3.0** | — | **No** |
| qualcomm/PPE-Detection | proprietary | https://huggingface.co/qualcomm/PPE-Detection | — | **license:other** (Qualcomm AI-Hub) | — | **Flag — vendor EULA** |
| SH17 dataset release (ahmadmughees) | YOLOv8 / v9-e / v10 | https://github.com/ahmadmughees/SH17dataset | 17 manufacturing-PPE classes | **AGPL** (YOLOv8/9/10 base) | YOLOv9-e mAP 70.9 % | **Flag — AGPL**, but **dataset (8099 imgs) itself is CC-BY-4.0 academic** — use for training a permissive model |
| DarthRegicid1/YOLOv5_PPE-Detection | YOLOv5 | https://huggingface.co/DarthRegicid1/YOLOv5_PPE-Detection | Pictor-PPE + VOC2028 + CHV | MIT card / **GPL-3.0 YOLOv5 base** | — | **Flag — copyleft** |

## 6. Top 3 recommendations per class

### Helmet
1. **DunnBC22/yolos-tiny-Hard_Hat_Detection** (Apache-2.0, 26 MB) — only fully
   permissive helmet-native checkpoint on HF with published metrics
   (AP50 0.747 on keremberke/hard-hat). Use as a warm-start for a
   D-FINE-M or RT-DETRv2 head-swap fine-tune — or directly as a validation
   baseline via `AutoModelForObjectDetection.from_pretrained(...)`.
   Fine-tune path: freeze backbone → 2→4-class head swap (add
   `head_with_nitto_hat`, retain `hardhat/no-hardhat`) → 40 epochs AdamW 3e-5
   on merged SHWD + our nitto set. Compare against D-FINE-M warm-start; if
   within ±1 AP, prefer D-FINE-M for edge headroom.
2. **ikigaiii/yolos-tiny-ppe-detection** (Apache-2.0 base, 26 MB) — alternate
   YOLOS-tiny with `{head, helmet, person}` — aligns with our 4-class schema
   after adding `head_with_nitto_hat`. Same fine-tune recipe.
3. **Train-our-own D-FINE-M on SHWD + Hard-Hat-Workers + SH17** (Apache-2.0,
   from `ustc-community/dfine-medium-obj2coco`) — the actually-recommended
   path. No public Apache-licensed D-FINE/RT-DETR PPE finetune exists yet;
   DunnBC22/ikigaiii serve as sanity-check baselines, not the production
   model.

### Vest
1. **uisikdag/autotrain-detr-resnet-50-safety-vest-detection** (~Apache-2.0
   inherited from DETR base, 166 MB) — only non-AGPL vest-native detector on
   HF with published metrics (map50 0.75). DETR-R50 is heavier than D-FINE-M
   but its head weights transfer cleanly to our vest class via
   HF Transformers.
2. **Train-our-own D-FINE-M on SH17 `high-vis-vest` subset** (dataset
   CC-BY-4.0; ignore SH17's AGPL weight release — use the data only).
3. **Grounding-DINO-T pseudo-label → D-FINE-M distillation** for vest —
   documented teacher path in helmet SOTA doc §3.

### Shoes / Boots
1. **Two-stage baseline (unchanged)** — D-FINE-N detection + DINOv2-small
   classifier, both Apache-2.0 (see `ppe-shoes_detection-sota.md §3`). No
   permissive PPE-shoe-native checkpoint exists publicly.
2. **SH17 `safety-boots` subset** as fine-tune data (CC-BY-4.0 data; do not
   pull SH17 AGPL weights).
3. **Roboflow Universe `construction-ppe` subsets** that ship with Apache-2.0
   or MIT dataset licenses — curate per-dataset before merging.

### Mask / Gloves / Goggles / Coverall (bonus — all in Dricz)
1. **Dricz/ppe-obj-detection** (DETR-R50, Apache-2.0, 5 classes) — only
   permissive detector covering coverall + face_shield + gloves + goggles +
   mask. Use as warm-start for any of these derived features.

### Harness
- **No permissive public checkpoint located.** SH17 is the only dataset with a
  harness class (CC-BY-4.0). Train from scratch on SH17 + in-house.

## 7. Download manifest

All downloads verified `HTTP 200` on `curl -IL` and SHA256-hashed.
Destination: absolute paths under `ai/pretrained/ppe-*`. Files are gitignored
and tracked via DVC.

| Local path | Size (B) | SHA256 | License |
|---|---|---|---|
| `/home/nthanhtrung/Documents/05_Team/02_Vietsol/01_Projects/edge_ai/ai/pretrained/ppe-helmet_detection/dunnbc22_yolos_tiny_hardhat.bin` | 25,957,273 | `a02db545b5ccd38cef96f1d98a1d32d88629fa761fea13c2778e084c51e5d67f` | Apache-2.0 |
| `/home/nthanhtrung/Documents/05_Team/02_Vietsol/01_Projects/edge_ai/ai/pretrained/ppe-helmet_detection/ikigaiii_yolos_tiny_ppe.safetensors` | 25,910,944 | `74bad0972ab0d121ac301e9af7ded3e257f64319736e1bc0a9ab854e7cfff865` | Apache-2.0 (inherited) |
| `/home/nthanhtrung/Documents/05_Team/02_Vietsol/01_Projects/edge_ai/ai/pretrained/ppe-helmet_detection/dricz_detr_r50_ppe5.safetensors` | 166,498,936 | `a7203e1b2ae436b38d3dbf2468b732377b967130649e928fce968a80d3fdaed6` | Apache-2.0 |
| `/home/nthanhtrung/Documents/05_Team/02_Vietsol/01_Projects/edge_ai/ai/pretrained/ppe-helmet_detection/uisikdag_detr_r50_vest.safetensors` | 166,496,880 | `b07f1161df4a500c4eabe2757a0aa445db884932ee43d13b288ad6214701e927` | Apache-2.0 (inherited from DETR) |

Pre-existing directory `ai/pretrained/ppe-helmet_detection/yolos-tiny-hardhat/`
already contains the full DunnBC22 snapshot (pytorch_model.bin +
config.json + preprocessor_config.json) from a previous clone — retained.

No shoes-specific file downloaded this pass (no permissive native checkpoint
found). Existing shoes pretrained tree (D-FINE-N/S, DINOv2-S,
EfficientFormerV2-S0) remains the canonical stack.

## 8. Licensing landmines

1. **AGPL-3.0 on Ultralytics YOLOv5 ≥ v6 / v8 / v10 / v11 / v12 / v26**
   Many of the HF Hub PPE finetunes re-upload these weights under a "more
   permissive" card license (MIT, Apache, CC-BY). The card license **does not
   override** the upstream AGPL obligation — if the weights were produced by
   Ultralytics code, AGPL-3.0 applies to any derivative product. Call-outs:
   Keremberke (YOLOv8 hard-hat), Hexmon, Hansung-Cho, Tanishjain9,
   itsadityabaniya, Advantech-EIOT, melihuzunoglu, dxvyaaa, leeyunjai,
   wesjos, gungniir, Bhavani-23, SH17 weight release.
2. **GPL-3.0 on legacy YOLOv5 (pre-Ultralytics-AGPL era)** — still copyleft,
   incompatible with Apache-2.0 linking. Affects Keremberke YOLOv5,
   uisikdag YOLOv5, HudatersU YOLOv8-via-GPL, kavinel/Helmet_Detector,
   DarthRegicid1.
3. **`license: other`** — qualcomm/PPE-Detection is gated to Qualcomm AI-Hub
   EULA (per-chip). Skip for chip-agnostic delivery.
4. **Dataset licenses vs model licenses**:
   SHWD, Pictor-v3, GDUT-HWD have **no explicit license** — academic use
   only; commercial redistribution of trained weights from these is legally
   murky. SH17 is **CC-BY-4.0** (safe to train on with attribution).
   keremberke/hard-hat-detection on HF is unlabeled — inherit caution.
5. **DINOv3** (not used for PPE but relevant) — gated + Meta DINOv3 license
   restricts commercial use; stay on DINOv2 (Apache-2.0).
6. **Un-versioned card licenses** — several repos (ikigaiii, gghsgn, wesjos)
   omit the `license:` YAML key. Default is **no license granted** (all rights
   reserved). Treat these as research-only until the author clarifies.

## 9. References

- DunnBC22 YOLOS-tiny hardhat — https://huggingface.co/DunnBC22/yolos-tiny-Hard_Hat_Detection
- Dricz PPE DETR-R50 — https://huggingface.co/Dricz/ppe-obj-detection
- uisikdag safety-vest DETR-R50 — https://huggingface.co/uisikdag/autotrain-detr-resnet-50-safety-vest-detection
- ikigaiii YOLOS-tiny PPE — https://huggingface.co/ikigaiii/yolos-tiny-ppe-detection
- gghsgn YOLOS helmet — https://huggingface.co/gghsgn/safety_helmet_detection
- TheNobody-12 HelmetDETA — https://huggingface.co/TheNobody-12/HelmetDETA
- Keremberke hard-hat collection — https://huggingface.co/keremberke?search_models=hard-hat
- keremberke hard-hat-detection dataset — https://huggingface.co/datasets/keremberke/hard-hat-detection
- Bhavani-23 Ocularone — https://huggingface.co/Bhavani-23/Ocularone-Hazard-Vest-Dataset-Models
- SH17 dataset + weights — https://github.com/ahmadmughees/SH17dataset , arXiv 2407.04590
- CHV dataset / ZijianWang — https://github.com/ZijianWang-ZW/PPE_detection
- Roboflow Universe PPE — https://universe.roboflow.com/browse/manufacturing/ppe
- RF-DETR (base) — https://github.com/roboflow/rf-detr
- Ultralytics AGPL license — https://www.ultralytics.com/license
- Qualcomm AI-Hub PPE — https://huggingface.co/qualcomm/PPE-Detection
- IITU Safety-Helmet dataset (CC-BY-4.0) — https://huggingface.co/datasets/ersace/IITU_Safety-Helmet_Dataset_v1.0_Demo
- Prior surveys: `ai/docs/technical_study/ppe-helmet_detection-sota.md`, `ai/docs/technical_study/ppe-shoes_detection-sota.md`

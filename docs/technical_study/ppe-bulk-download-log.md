# PPE Bulk Pretrained Download Log (2026-04-14)

Follow-up to:
- `ai/docs/technical_study/ppe-helmet_detection-sota.md`
- `ai/docs/technical_study/ppe-shoes_detection-sota.md`
- `ai/docs/technical_study/ppe-oss-pretrained-deep-dive.md`

Directive: pull every PPE-related pretrained weight that the prior surveys had
skipped for license reasons (AGPL-3.0 Ultralytics fine-tunes, Apple-ASCL
FastViT/MobileViTv2, gated Meta DINOv3, Qualcomm `license:other`, etc.).
License posture is deferred — this pass is bulk-download + catalog only.

## 1. Headline numbers

- **Candidates attempted:** 50 unique (file, repo) targets across two
  destination directories.
- **Succeeded this session:** 29 new downloads.
- **Already present (skipped):** 7 (prior-session checkpoints retained).
- **Superseded:** 7 — wrong-path probe URL, correct path in the same repo
  succeeded on the round-2 retry (HF API file-listing used to resolve).
- **True failures (gated / missing):** 7.
- **Total disk after pass:** ~3.18 GB across
  `ai/pretrained/ppe-helmet_detection/` (2.9 GB) and
  `ai/pretrained/ppe-shoes_detection/` (350 MB).

Per-file detail tables live in (gitignored) manifests:
- `ai/pretrained/ppe-helmet_detection/DOWNLOAD_MANIFEST.md`
- `ai/pretrained/ppe-shoes_detection/DOWNLOAD_MANIFEST.md`

## 2. What was newly downloaded

### Helmet / PPE multi-class (helmet dir)

| File | License | Notes |
|---|---|---|
| `keremberke_yolov8n_hardhat.pt` (6.2 MB) | AGPL-3.0 | Ultralytics v8 fine-tune on keremberke/hard-hat |
| `keremberke_yolov8s_hardhat.pt` (22 MB) | AGPL-3.0 | — |
| `keremberke_yolov8m_hardhat.pt` (52 MB) | AGPL-3.0 | mAP50 ≈ 0.88 per card |
| `uisikdag_yolov5_hardhat.pt` (42 MB) | GPL-3.0 | YOLOv5 pre-AGPL copyleft |
| `gghsgn_safety_helmet.bin` (26 MB) | MIT (card) | YOLOS motorbike-helmet |
| `gghsgn_helmet_detection.bin` (26 MB) | Apache-2.0 (card) | YOLOS motorbike-helmet 6-cls |
| `thenobody12_helmet_deta.safetensors` (879 MB) | MIT | DETA 219 M, too heavy for edge; keep as teacher |
| `hansung_yolov8_ppe.pt` (6.2 MB) | AGPL-inherited | YOLOv8 PPE |
| `tanishjain_yolov8n_ppe6.pt` (5.6 MB) | AGPL-inherited | 6-class PPE |
| `melihuzunoglu_yolov11_ppe.pt` (5.5 MB) | AGPL-3.0 | |
| `gungniir_yolo11_vest.pt` (5.5 MB) | AGPL-inherited | |
| `dxvyaaa_yolo_helmet.pt` (22 MB) | AGPL-inherited | Single `best.pt`; yolov8/9/10 subfile probes 404'd |
| `darthregicid_yolov5_ppe_m896.pt` (119 MB) | GPL-3.0 | Pictor-PPE + VOC2028 + CHV, 896px |
| `bhavani23_ocularone_yolov8{n,m,x}.pt` + `yolov11{n,m,x}.pt` (6) | AGPL-inherited | Ocularone hazard-vest, from `models/<arch>/<arch>.pt` |
| `wesjos_yolo11{n,m}_hardhat_vest.pt` (2) | AGPL-inherited | Hardhat+vest |
| `leeyunjai_yolo11{m,s,x}_helmet.pt` + `yolo26{m,s}_helmet.pt` (5) | AGPL-3.0 | Factory helmet YOLO11/26 |
| `HudatersU_safety_helmet.{onnx,pt}` (2) | GPL-3.0 | Resolved under `safety_helmet_251209.*` |
| `hexmon_vyra_yolo_ppe.{onnx,pt}` (2) | AGPL-inherited | `best.{onnx,pt}` |

### Shoes / stage-2 classifier (shoes dir)

| File | License | Notes |
|---|---|---|
| `fastvit_t8.bin` (16 MB) | Apple-ASCL | timm port |
| `fastvit_t12.bin` (30 MB) | Apple-ASCL | timm port |
| `mobilevitv2_100.bin` (19 MB) | Apple sample-code | timm port |
| `efficientformerv2_s1.bin` (25 MB) | Apache-2.0 | S1 variant (S0 was already present) |
| `rfdetr_small.onnx` (109 MB) | Apache-2.0 | ONNX-community export, Roboflow RF-DETR-S |

### Already-present (preserved, not re-downloaded)

`dfine_medium_obj2coco.safetensors`, `yolox_m.pth`, `dunnbc22_yolos_tiny_hardhat.bin`,
`ikigaiii_yolos_tiny_ppe.safetensors`, `dricz_detr_r50_ppe5.safetensors`,
`uisikdag_detr_r50_vest.safetensors`, `yolos-tiny-hardhat/` snapshot,
and full shoes stack (D-FINE-N/S, DINOv2-S, EfficientFormerV2-S0, plus
D-FINE/DINOv2 HF snapshot subdirs).

## 3. True failures — root cause

| File | Status | Reason |
|---|---|---|
| `keremberke_yolov5{n,s,m}_hardhat.pt` | 401 | Repos gated to HF login; requires auth token. `hf` CLI not installed, no token available in this shell. |
| `advantech_qualcomm_yolo11_ppe.pt` / `.dlc` | 401 | `Advantech-EIOT/qualcomm-ultralytics-ppe_detection` gated; Qualcomm AI-Hub EULA. |
| `dinov3_vits16.bin` (shoes dir) | 401 | `facebook/dinov3-vits16-pretrain-lvd1689m` — Meta DINOv3 gated license; must accept terms on HF Hub. |
| `lanseria_yolov8n_hardhat.onnx` | 404 | Upstream repo only ships TFJS shards (`group1-shard{1,2,3}of3.bin`), no single ONNX file exists. |

Mitigation for the 401s: user can run `hf auth login` (or set `HF_TOKEN`),
re-accept gates for `facebook/dinov3-*` / `keremberke/yolov5*` /
`Advantech-EIOT/*`, then re-run this same script. Bash helper at
`/tmp/ppe_dl_logs/dl_one.sh` + target tables at
`/tmp/ppe_dl_logs/targets*.tsv`.

## 4. Notes on corrected-path detections

Seven initial probe URLs returned 404 because the repos use non-standard
paths (`models/<arch>/<arch>.pt`, `helmet-11m.pt`, `yolo11m_safety.pt`,
`safety_helmet_251209.*`, `best.{pt,onnx}` at root or in `train/<exp>/weights/`).
These were resolved by calling `GET https://huggingface.co/api/models/<repo>`
and parsing `siblings[].rfilename`, then re-issuing curl against the real
path. All seven succeeded on round 2 and the stale rows are marked
`SUPERSEDED` in the manifests (kept for audit; not double-counted).

## 5. License posture reminder (not addressed in this pass)

All AGPL-3.0, Apple-ASCL, and GPL-3.0 downloads above are **disqualified**
for Nitto Denko commercial delivery per the helmet/shoes SOTA surveys.
They are downloaded here strictly as:

- benchmark comparators (accuracy A/B vs D-FINE-M / YOLOX-M),
- pseudo-label teachers for unlabelled factory footage,
- internal-only evaluation before the clean-room retrain on
  Apache-2.0 base (D-FINE-M / RT-DETRv2) plus in-house + CC-BY data.

Any production use requires routing through the permissive stack already
downloaded in prior passes (D-FINE-M/N/S, RT-DETRv2-R18/R50, DINOv2-small,
EfficientFormerV2-S0/S1, RF-DETR-Small, DunnBC22 YOLOS-tiny, Dricz DETR,
uisikdag DETR-vest, ikigaiii YOLOS). License-safe choice is unchanged.

# ppe-helmet_detection — Pretrained Weights

## Summary

Helmet / hard-hat / vest detection candidates. Total **90 files** across
**~3.3 GB** — clean-license transformer detectors (D-FINE-M, DETR-R50,
YOLOS-tiny, DINOv3 backbones) as the preferred pool, plus ~35
Ultralytics-based (AGPL) YOLOv5/8/10/11/12/26 community fine-tunes kept
only for benchmarking. See brief:
[../../docs/technical_study/ppe-helmet_detection-sota.md](../../docs/technical_study/ppe-helmet_detection-sota.md).

## Files on disk (as of 2026-04-14)

| File / Subfolder | Size | Type | License | Source | Notes |
|---|---|---|---|---|---|
| `dfine_medium_obj2coco.safetensors` | 76 MB | D-FINE-M Obj365→COCO | Apache-2.0 | ustc-community | **SOTA pick #1** — replace YOLOX-M. |
| `dricz_detr_r50_ppe5.safetensors` | 159 MB | DETR-R50 PPE-5 fine-tune | Apache-2.0 | dricz | Transformer PPE baseline. |
| `uisikdag_detr_r50_vest.safetensors` | 159 MB | DETR-R50 vest fine-tune | Apache-2.0 | uisikdag | Vest-specific. |
| `thenobody12_helmet_deta.safetensors` | 838 MB | DETA helmet | Apache-2.0 (check) | thenobody12 | Largest transformer ckpt. |
| `yolos-tiny-hardhat/` | 25 MB | YOLOS-tiny hard-hat | Apache-2.0 | DunnBC22 | Transformer detector, small & clean. |
| `dunnbc22_yolos_tiny_hardhat.bin` | 25 MB | YOLOS-tiny hard-hat (bin only) | Apache-2.0 | DunnBC22 | Mirror of above. |
| `ikigaiii_yolos_tiny_ppe.safetensors` | 25 MB | YOLOS-tiny PPE | Apache-2.0 | ikigaiii | |
| `gghsgn_helmet_detection.bin` | 25 MB | YOLOS-tiny helmet variant | Apache-2.0 | gghsgn | |
| `gghsgn_safety_helmet.bin` | 25 MB | YOLOS-tiny helmet variant | Apache-2.0 | gghsgn | |
| `yolox_m.pth` | 194 MB | YOLOX-M COCO | Apache-2.0 | Megvii | Baseline retained as safety net. |
| `_hf_facebook_dinov3-vits16-pretrain-lvd1689m/` | 83 MB | DINOv3 ViT-S/16 backbone | **Meta DINOv3 non-commercial** | facebook | Benchmark only — license blocks ship. |
| `_hf_facebook_dinov3-vitb16-pretrain-lvd1689m/` | 327 MB | DINOv3 ViT-B/16 backbone | **Meta DINOv3 non-commercial** | facebook | Benchmark only. |
| `_hf_Advantech-EIOT_qualcomm-ultralytics-ppe_detection/ppe_yolov11n_w8a16_160x160_pics1000.dlc` | 6.2 MB | Qualcomm DLC PPE YOLO | **AGPL-3.0** (Ultralytics) + vendor-specific format | Advantech-EIOT | Was gated — now accessible. DLC is chip-specific, excluded from generic ONNX path. |
| `_hf_keremberke_yolov5{n,s,m}-hard-hat-detection/` | 4 KB each | (empty) | — | keremberke | Upstream repos now 404 — placeholder dirs only. |
| `keremberke_yolov8n_hardhat.pt` | 6 MB | YOLOv8n hardhat | **AGPL-3.0** | keremberke | Benchmark only. |
| `keremberke_yolov8s_hardhat.pt` | 22 MB | YOLOv8s hardhat | **AGPL-3.0** | keremberke | Benchmark only. |
| `keremberke_yolov8m_hardhat.pt` | 50 MB | YOLOv8m hardhat | **AGPL-3.0** | keremberke | Benchmark only. |
| `bhavani23_ocularone_yolov8{n,m,x}.pt` | 6 / 50 / 131 MB | YOLOv8 PPE (ocularone) | **AGPL-3.0** | bhavani23 | Benchmark only. |
| `bhavani23_ocularone_yolov11{n,m,x}.pt` | 5.3 / 39 / 110 MB | YOLOv11 PPE | **AGPL-3.0** | bhavani23 | Benchmark only. |
| `leeyunjai_yolo11{s,m,x}_helmet.pt` | 19 / 39 / 110 MB | YOLOv11 helmet | **AGPL-3.0** | leeyunjai | Benchmark only. |
| `leeyunjai_yolo26{s,m}_helmet.pt` | 20 / 42 MB | YOLOv26 helmet | **AGPL-3.0** | leeyunjai | Benchmark only. |
| `wesjos_yolo11{n,m}_hardhat_vest.pt` | 5.1 / 116 MB | YOLOv11 hardhat+vest | **AGPL-3.0** | wesjos | Benchmark only. |
| `darthregicid_yolov5_ppe_m896.pt` | 114 MB | YOLOv5 PPE | **AGPL/GPL** (YOLOv5) | darthregicid | Benchmark only. |
| `uisikdag_yolov5_hardhat.pt` | 41 MB | YOLOv5 hardhat | **AGPL/GPL** | uisikdag | Benchmark only. |
| `hexmon_vyra_yolo_ppe.pt` / `.onnx` | 50 / 99 MB | YOLO PPE | **AGPL-3.0** | hexmon | Benchmark only. |
| `HudatersU_safety_helmet.pt` / `.onnx` | 84 / 167 MB | YOLO safety helmet | **AGPL-3.0** | HudatersU | Benchmark only. |
| `hansung_yolov8_ppe.pt` | 6 MB | YOLOv8 PPE | **AGPL-3.0** | hansung | Benchmark only. |
| `tanishjain_yolov8n_ppe6.pt` | 5.4 MB | YOLOv8n 6-class PPE | **AGPL-3.0** | tanishjain | Benchmark only. |
| `gungniir_yolo11_vest.pt` | 5.3 MB | YOLOv11 vest | **AGPL-3.0** | gungniir | Benchmark only. |
| `melihuzunoglu_yolov11_ppe.pt` | 5.3 MB | YOLOv11 PPE | **AGPL-3.0** | melihuzunoglu | Benchmark only. |
| `dxvyaaa_yolo_helmet.pt` | 22 MB | YOLO helmet | **AGPL-3.0** | dxvyaaa | Benchmark only. |
| `DOWNLOAD_MANIFEST.md` | 7.6 KB | Manifest w/ SHA256s | — | — | |

## Recommended defaults (from SOTA brief)

- **#1 D-FINE-M (Obj365→COCO)** → `dfine_medium_obj2coco.safetensors` (on disk).
- **#2 RT-DETRv2-R18** → not in this folder; shared at `ai/pretrained/rtdetr_v2_r18_coco.pt`.
- **#3 YOLOX-M (baseline retained)** → `yolox_m.pth` (on disk).

## Gated / skipped / 404

- `Advantech-EIOT/qualcomm-ultralytics-ppe_detection` — **now resolved** (downloaded into `_hf_Advantech-EIOT_qualcomm-ultralytics-ppe_detection/`). Vendor DLC format; not used in generic pipeline.
- `keremberke/yolov5{n,s,m}-hard-hat-detection` — **permanently 404** (repos deleted upstream).
- DINOv3 backbones downloaded but **license blocks commercial use**; benchmark only.

## Related docs

- SOTA brief: `../../docs/technical_study/ppe-helmet_detection-sota.md`
- Deep dive: `../../docs/technical_study/ppe-oss-pretrained-deep-dive.md`
- Bulk log: `../../docs/technical_study/ppe-bulk-download-log.md`
- Gated retry log: `../../docs/technical_study/gated-retry-download-log.md`
- Quality report: `../../features/ppe-helmet_detection/predict/QUALITY_REPORT.md`
- SHA256s: `./DOWNLOAD_MANIFEST.md`

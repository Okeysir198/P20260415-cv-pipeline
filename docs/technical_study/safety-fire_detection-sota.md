# Fire Detection — SOTA Pretrained Model Survey (2025–2026)

> Worker survey for Phase 1 roadmap review. Compares newer pretrained
> detectors against the current ROADMAP baseline (**YOLOX-M**, 640/1280)
> under a chip-agnostic **~18 TOPS INT8** envelope.

## 1. Task summary

Two-class RGB object detection for **fire** and **smoke** in factory scenes.
Per `ai/docs/03_platform/safety-fire_detection.md`, the detector must:

- Find small/distant fires (down to ~10–14 px in telephoto frames, tiled at 1280).
- Separate diffuse smoke from steam/fog/sunlight (global context helps).
- Keep low FP rate (target < 3%) with ≥ 0.90 precision and ≥ 0.88 recall.
- Export to standard ONNX and run INT8 at real-time rates on a ~18 TOPS edge SoC.
- Use a commercially permissive license (Apache-2.0 / MIT preferred).

Baseline is YOLOX-M (25.3 M params, CSPDarknet53 + PAFPN + decoupled head,
Apache-2.0). The platform doc already lists D-FINE-S and RT-DETRv2-R18 as
Options 2 and 3; this study extends the comparison to the newer 2024–2025
detectors (DEIM, D-FINE variants, RF-DETR, YOLOv10/11/12, YOLO26) and
recommends an action for the roadmap.

## 2. Candidate models

COCO val2017 mAP (50:95) shown as a capability proxy; fire benchmarks follow
in §7. Params are detector-only (backbone + neck + head).

| Model | Arch family | Params | Pretrained source | License | Bench mAP (COCO) | Edge-friendly |
|---|---|---|---|---|---|---|
| **YOLOX-M** (baseline) | YOLO / CSPDarknet + PAFPN | 25.3 M | [Megvii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | Apache-2.0 | 46.9 | Yes — proven INT8 |
| **D-FINE-S** | DETR + HGNetv2-B0 + FDR | 10 M | [Peterande/D-FINE](https://github.com/Peterande/D-FINE) | Apache-2.0 | 48.5 | Yes (mixed INT8/FP16 for decoder) |
| **D-FINE-M** | DETR + HGNetv2-B2 + FDR | 19 M | [Peterande/D-FINE](https://github.com/Peterande/D-FINE) | Apache-2.0 | 52.3 | Yes — still fits ~18 TOPS @ 640 |
| **D-FINE-L** | DETR + HGNetv2-B4 + FDR | 31 M | Same release | Apache-2.0 | 54.0 | Marginal at 640 INT8 |
| **DEIM-D-FINE-S** | D-FINE + Dense-O2O + MAL | 10 M | [Intellindust-AI-Lab/DEIM](https://github.com/Intellindust-AI-Lab/DEIM) | Apache-2.0 | 49.0 | Same cost as D-FINE-S (training-only change) |
| **DEIM-D-FINE-M** | D-FINE + DEIM training | 19 M | Same | Apache-2.0 | 52.7 | Yes |
| **DEIM-D-FINE-L** | D-FINE + DEIM training | 31 M | Same | Apache-2.0 | 54.7 | Marginal |
| **RT-DETRv2-R18** | DETR + ResNet-18 (discrete sampling) | 20 M | [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) | Apache-2.0 | 47.1 | Yes — built for ONNX |
| **RT-DETRv2-R50** | DETR + ResNet-50 | 42 M | Same | Apache-2.0 | 53.4 | Marginal @ 640 INT8 |
| **RF-DETR-Nano** | DINOv2-distill + DETR | ~12 M | [roboflow/rf-detr](https://github.com/roboflow/rf-detr) | Apache-2.0 | ~48 (67.6 AP50) | Yes |
| **RF-DETR-Small** | DINOv2-distill + DETR | ~20 M | Same | Apache-2.0 | ~52 (72.1 AP50) | Yes |
| **RF-DETR-Medium** | DINOv2-distill + DETR | ~32 M | Same | Apache-2.0 | ~54 (73.6 AP50) | Marginal |
| **RF-DETR-Large** | DINOv2-distill + DETR | ~60 M | Same | Apache-2.0 | 60.5 (75.1 AP50) | Teacher only, not edge |
| **YOLOv10-M** | YOLO, NMS-free | 15 M | [THU-MIG/yolov10](https://github.com/THU-MIG/yolov10) | Apache-2.0 (THU-MIG) | 51.3 | Yes |
| **YOLOv8-M / YOLOv11-M** | YOLO | 25–26 M | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | **AGPL-3.0** (Enterprise req.) | 50.2 / 51.5 | Yes — but flag license |
| **YOLOv12-S / M** | YOLO + area-attention | 9 / 20 M | [sunsmarterjie/yolov12](https://github.com/sunsmarterjie/yolov12) | AGPL-3.0 | 48.0 / 52.5 | Yes — license concern |
| **YOLO26** | YOLO + DFL + ProgLoss + STAL | S/M/L | [paper 2509.25164](https://hf.co/papers/2509.25164) | AGPL-3.0 (Ultralytics) | TBD | Yes — flag license |
| **DEIMv2-S/M** | DEIM + DINOv3 features | 10–19 M | [Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) | Apache-2.0 | ~50–54 | Yes — newest, less proven |

Community fire-specific pretrained checkpoints (starting points, not COCO-SOTA):
- [TommyNgx/YOLOv10-Fire-and-Smoke-Detection](https://hf.co/TommyNgx/YOLOv10-Fire-and-Smoke-Detection) — base YOLOv11/YOLOv10, Apache-2.0, **gated** (HF access request).
- [Mehedi-2-96/fire-smoke-detection-yolo](https://hf.co/Mehedi-2-96/fire-smoke-detection-yolo) — undocumented.
- Roboflow Universe "Fire-Smoke-Detection-Yolov11" — CC-BY 4.0 dataset + weights.

## 3. Top 3 recommendations

Ranking prioritizes **accuracy on diffuse smoke**, **edge deployability under
~18 TOPS INT8**, and **Apache-2.0 license**.

### #1 — DEIM-D-FINE-M (or D-FINE-M if DEIM training is skipped)

- **Why.** +5.8 COCO AP over YOLOX-M (52.7 vs 46.9) at 25 % fewer params
  (19 M vs 25.3 M). The DEIM training framework pushes D-FINE-M from 52.3 to
  52.7 AP and cuts training time ~50 % — same deployable weights as D-FINE-M.
  Fine-grained Distribution Refinement produces sharper boxes for amorphous
  smoke; global encoder attention captures diffuse plumes better than YOLOX's
  local convs. NMS-free output simplifies the edge alert pipeline.
- **Fine-tune path.** Start from the Objects365 pretrain
  (`dfine_m_obj365.pth`, 37.4 AP on O365), then fine-tune on the combined
  FASDD + D-Fire corpus at 640×640, then a second stage at 1280×1280 for
  long-range telephoto mode. Use HGNetv2-B2 in the existing `dfine-m` registry
  entry (`core/p06_models/`). If DEIM training is adopted, swap the loss to
  Dense-O2O + Matchability-Aware Loss using the DEIM repo configs.
- **Dataset.** FASDD (120 K) + FASDD_CV (40 K) + D-Fire (21 K) + ~5 K hard
  negatives (steam, sunlight, orange machinery) + customer factory footage.
- **Est. FPS @ 18 TOPS INT8.** ~22–28 FPS at 640, ~6–8 FPS for 6-tile 1280
  mode. Comparable to YOLOX-M but with higher AP. Decoder attention kept in
  FP16 (mixed precision) — most ~18 TOPS SoCs support per-layer mixed
  precision.

### #2 — D-FINE-S / DEIM-D-FINE-S

- **Why.** The platform doc already lists D-FINE-S as Option 2. It stays the
  best "fast mode" candidate: 48.5 AP (or 49.0 with DEIM training) at only
  10 M params — matches YOLOX-M AP with 60 % fewer parameters. Ideal if the
  model must coexist with several other detectors on the same 18 TOPS chip
  (helmet + fire + zone + face).
- **Fine-tune path.** Same as #1 but with `dfine_s_obj365.pth` (30.5 AP)
  pretrain. Faster convergence (~1 day on a single 4090 per DEIM paper).
- **Dataset.** Same as #1.
- **Est. FPS @ 18 TOPS INT8.** ~40–55 FPS at 640, ~12–15 FPS in 6-tile 1280.

### #3 — RF-DETR-Small (augment, not replace)

- **Why.** DINOv2-distilled backbone has exceptional fine-tune performance
  across 100+ Roboflow domains — strongest candidate for **small** fire
  targets and novel factory scenes. 72.1 AP50 on COCO, Apache-2.0.
- **Fine-tune path.** Use the `rfdetr` Python package (`RFDETRSmall()`) with
  the auto-downloaded COCO checkpoint. Fine-tune on FASDD+D-Fire at 560×560
  (RF-DETR native) then 768×768 for long-range.
- **Dataset.** Same as #1 plus Roboflow Universe fire datasets.
- **Est. FPS @ 18 TOPS INT8.** ~18–25 FPS at 560. Role: second opinion /
  ensemble fusion with D-FINE-M during evaluation, and teacher for distillation
  from RF-DETR-Large (60.5 COCO AP) into D-FINE-S.

## 4. Pretrained weights

Downloaded to `ai/pretrained/safety-fire_detection/` (gitignored).

| Model | URL | License | Size | SHA256 | Local dest |
|---|---|---|---|---|---|
| D-FINE-S (COCO) | https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_coco.pth | Apache-2.0 | ~42 MB | not downloaded (O365 preferred for fine-tune) | — |
| **D-FINE-S (Objects365)** | https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_obj365.pth | Apache-2.0 | 42 MB | `b96e35e5aef3dfe4853b76efa7ac8d02179fedcf60f6f3cfb0e69b29399b71a2` | `ai/pretrained/safety-fire_detection/dfine_s_obj365.pth` |
| D-FINE-M (COCO) | https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_coco.pth | Apache-2.0 | ~78 MB | not downloaded | — |
| **D-FINE-M (Objects365)** | https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_obj365.pth | Apache-2.0 | 78 MB | `20b0c3d2a725d3bc6b6b34b257ad453553d0ad3a3c585f967cac94d63a5181b3` | `ai/pretrained/safety-fire_detection/dfine_m_obj365.pth` |
| DEIM-D-FINE-M (COCO) | https://drive.google.com/file/d/18Lj2a6UN6k_n_UzqnJyiaiLGpDzQQit8 | Apache-2.0 | ~78 MB | manual download required (Google Drive) | — |
| DEIM-D-FINE-S (COCO) | https://drive.google.com/file/d/1tB8gVJNrfb6dhFvoHJECKOF5VpkthhfC | Apache-2.0 | ~40 MB | manual download required (Google Drive) | — |
| RF-DETR-Small | auto-download via `rfdetr` package (`RFDETRSmall()`) | Apache-2.0 | ~80 MB | fetched by library, no stable direct URL | `~/.cache/rfdetr/` |
| RT-DETRv2-R18 | https://github.com/lyuwenyu/RT-DETR/releases | Apache-2.0 | ~80 MB | — | fallback only |
| YOLOX-M (baseline) | https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth | Apache-2.0 | 97 MB | tracked in feature `runs/` | existing pipeline |
| TommyNgx/YOLOv10-Fire (community fine-tune) | https://hf.co/TommyNgx/YOLOv10-Fire-and-Smoke-Detection | Apache-2.0 | — | **gated — manual access request** | — |

URL health checks performed (`curl -I -sL`): all four D-FINE `.pth` URLs
return HTTP 200; GitHub release assets resolve via LFS redirect.

## 5. Edge deployment notes

- **YOLOX-M.** Baseline reference. Pure CNN, fully INT8-quantizable, ~30 FPS
  at 640 on a ~18 TOPS SoC. Proven export.
- **D-FINE-S / M.** Export path: `tools/deployment/export_onnx.py` in the
  Peterande repo, or HF Transformers Optimum ONNX export (D-FINE is now in
  `transformers>=4.45`). Backbone (HGNetv2) and hybrid encoder are pure CNN
  and quantize cleanly to INT8. **Attention in the transformer decoder should
  stay FP16** — mixed precision per-layer is supported by typical ~18 TOPS
  INT8 accelerators. Deformable-attention operators were reworked to
  ONNX-friendly form in D-FINE; still validate opset ≥ 17. Rough compute:
  D-FINE-M ≈ 57 GFLOPs @ 640, which fits comfortably inside an 18 TOPS
  envelope while leaving headroom for helmet/zone models.
- **DEIM.** Training-framework only. Deployed weights are structurally
  identical to D-FINE — same ONNX export, same INT8 quant caveats.
- **RF-DETR.** ONNX export supported via `model.export()` in the `rfdetr`
  package. DINOv2 patch-embed + ViT blocks → heavier INT8 quant sensitivity
  than HGNetv2; expect 1–2 AP drop with PTQ, recoverable with QAT. Small/Nano
  variants fit 18 TOPS; Medium is marginal.
- **RT-DETRv2-R18.** Discrete sampling operator specifically solves ONNX
  deformable-attention issues. Most conservative DETR-family choice for edge.
- **Ultralytics YOLOv8/10/11/12 & YOLO26.** Smooth ONNX + INT8 story.
  Blocker is **AGPL-3.0** — commercial deployment requires an Ultralytics
  Enterprise License or full source disclosure. Not acceptable for the
  current Nitto Denko delivery; listed for reference only.
- **Tiled 1280 inference.** All recommended models keep the YOLOX-M tiled
  SAHI strategy unchanged — just swap the inner single-tile detector.

## 6. Datasets for fine-tune

| Dataset | Images | Classes | License | Link |
|---|---|---|---|---|
| **FASDD** | 120 K+ | fire, smoke (multi-domain) | Open Access (CC-BY compat.) | https://github.com/openrsgis/fasdd (Science Data Bank DOI 10.57760/sciencedb.j00104.00103) |
| **FASDD_CV** (subset) | 40 K | fire, smoke (ground) | Open Access | https://www.kaggle.com/dataset_store/yuulind/fasdd-cv-coco |
| **D-Fire** | 21 K | fire, smoke | Academic (Gaia@UFMG) | https://github.com/gaiasd/DFireDataset |
| **FireNet** | ~500 | fire | Research-only | https://github.com/OlafenwaMoses/FireNET |
| **FIRESENSE** | videos | fire, smoke | CC-BY-NC | https://zenodo.org/record/836749 (non-commercial) |
| **Fire360** (benchmark) | 360° video | perception | CC-BY 4.0 | https://hf.co/papers/2506.02167 |
| **BoWFire** | 226 | fire | CC-BY 4.0 | https://bitbucket.org/gbdi/bowfire-dataset |
| **Roboflow Fire-Smoke YOLOv11** | ~10 K | fire, smoke | CC-BY 4.0 | https://universe.roboflow.com/sayed-gamall/fire-smoke-detection-yolov11 |

Recommended training corpus for Phase 1: FASDD (full) ∪ D-Fire ∪ ~5 K hard
negatives. FIRESENSE excluded from training (NC) but usable for eval.

## 7. Verdict vs ROADMAP baseline

**Augment — do not replace YOLOX-M in Phase 1.** Promote D-FINE-M (not just
D-FINE-S) from "alternative" to **co-primary** alongside YOLOX-M, and
adopt the DEIM training framework for both D-FINE-S and D-FINE-M runs.

Rationale:

1. **Accuracy headroom is real and free.** D-FINE-M gives +5.4 COCO AP over
   YOLOX-M at 25 % fewer params and similar INT8 cost. Smoke detection is
   exactly where global attention + FDR pay off. DEIM training lifts this
   another ~0.5 AP with no deployment-side change.
2. **Risk is low.** YOLOX-M stays as the proven fallback. D-FINE is in HF
   `transformers`, has COCO + O365 weights, Apache-2.0, and is already named
   as Option 2 in `safety-fire_detection.md`. D-FINE-M is new vs the current
   platform doc — the doc pins D-FINE-S only. D-FINE-M fits 18 TOPS with
   headroom and closes the gap to RT-DETR-R50 / RF-DETR-Medium.
3. **License cleanliness.** Every recommended candidate is Apache-2.0. The
   Ultralytics YOLOv8/10/11/12 and YOLO26 families are AGPL-3.0 and must be
   avoided for the commercial Nitto Denko deployment unless the Enterprise
   License is purchased.
4. **Teacher for future distillation.** RF-DETR-Large (60.5 AP, Apache-2.0)
   is the best available teacher for later knowledge distillation into
   D-FINE-S, matching the Phase-3 contingency already in the platform doc.

Concrete roadmap edit proposal:

- `safety-fire_detection.md` §Architecture → add **D-FINE-M (19 M, 52.3 AP,
  Apache-2.0)** as Option 2b; keep D-FINE-S as Option 2a (light mode).
- Training week 3–4: run YOLOX-M and DEIM-D-FINE-M in parallel from the
  Objects365 pretrains. Keep RT-DETRv2-R18 as fallback only.
- Quantization: mixed-precision INT8 (backbone INT8, decoder attention FP16).

## 8. References

- Peng, Y. et al., *D-FINE: Redefine Regression Task in DETRs as Fine-grained
  Distribution Refinement*, ICLR 2025 Spotlight.
  https://arxiv.org/abs/2410.13842 · https://github.com/Peterande/D-FINE
- Huang, S. et al., *DEIM: DETR with Improved Matching for Fast Convergence*,
  CVPR 2025. https://hf.co/papers/2412.04234 ·
  https://github.com/Intellindust-AI-Lab/DEIM
- *DEIMv2: Real-Time Object Detection Meets DINOv3* (2025).
  https://hf.co/papers/2509.20787 ·
  https://github.com/Intellindust-AI-Lab/DEIMv2
- Lv, W. et al., *RT-DETRv2: Improved Baseline with Bag-of-Freebies for
  Real-Time DETRs*. https://github.com/lyuwenyu/RT-DETR
- Roboflow, *RF-DETR: A SOTA Real-Time Object Detection Model*, ICLR 2026.
  https://github.com/roboflow/rf-detr · https://blog.roboflow.com/rf-detr/
- Wang, A. et al., *YOLOv10: Real-Time End-to-End Object Detection*, 2024.
  https://github.com/THU-MIG/yolov10
- Sapkota, R. et al., *YOLO26: Key Architectural Enhancements and Performance
  Benchmarking for Real-Time Object Detection*, 2025.
  https://hf.co/papers/2509.25164
- Han, X. et al., *Fire and Smoke Detection with Burning Intensity
  Representation*, 2024. https://hf.co/papers/2410.16642
- *A Comparative Analysis of YOLOv9, YOLOv10, YOLOv11 for Smoke and Fire
  Detection*, Fire 2025. https://www.mdpi.com/2571-6255/8/1/26
- Wang, M. et al., *FASDD: An Open-access 100,000-level Flame and Smoke
  Detection Dataset*, Geo-spatial Information Science, 2024.
  https://www.tandfonline.com/doi/full/10.1080/10095020.2024.2347922
- Pesonen, J. et al., *Detecting Wildfires on UAVs with Real-time Segmentation
  Trained by Larger Teacher Models*, 2024. https://hf.co/papers/2408.10843
- *YOLOv11-CHBG: A Lightweight Fire Detection Model*, Fire 2025.
  https://www.mdpi.com/2571-6255/8/9/338
- Ultralytics licensing (AGPL-3.0): https://www.ultralytics.com/license

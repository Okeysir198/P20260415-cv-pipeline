# Helmet Detection — SOTA Pretrained Model Survey (2025–2026)

> Scope: pretrained CV models for the PPE Helmet Detection feature (Nitto Denko factory).
> Target edge: ~18 TOPS INT8, chip-agnostic.
> Baseline to challenge: **YOLOX-M @ 640/1280**, 4 classes (`person`, `head_with_helmet`, `head_without_helmet`, `head_with_nitto_hat`).

## 1. Task summary

Multi-class single-stage detector for industrial PPE compliance. Must:

- Discriminate 4 classes with strong small-object recall (helmets may be <10 px at distance).
- Stay robust in crowded scenes (NMS suppression risk) and across front/side/back/overhead views.
- Export cleanly to **generic ONNX** and tolerate **INT8 PTQ/QAT** on a ~18 TOPS INT8 NPU budget.
- Keep accuracy > speed — a stronger model at ≥10 FPS INT8 is acceptable.
- Use a commercially-usable license (Apache-2.0 / MIT / BSD). AGPL/GPL is **flagged**.

Evaluation axes: accuracy on small helmets, NMS-free behaviour (crowd overlaps), INT8 friendliness
(CNN ops / simple attention), and availability of strong pretrained checkpoints (COCO or
Objects365→COCO) to warm-start fine-tuning for the 4 PPE classes.

## 2. Candidate models

| Name | Arch | Params | Pretrained | License | Bench (COCO AP) | Edge-friendly? |
|---|---|---|---|---|---|---|
| YOLOX-M (baseline) | CNN anchor-free (CSPDarknet + PAFPN + decoupled head) | 25.3 M | COCO | Apache-2.0 | 46.9 AP | Yes — proven INT8 |
| YOLOX-S | CNN anchor-free | 9.0 M | COCO | Apache-2.0 | 40.5 AP | Yes — very fast |
| D-FINE-S | Real-time DETR (HGNetv2-S + hybrid encoder + FDR, NMS-free) | 10 M | COCO / Obj365→COCO | Apache-2.0 | 48.5 AP | Yes — CNN backbone, simple attn |
| D-FINE-M | Real-time DETR (HGNetv2-M) | 19 M | COCO / Obj365→COCO | Apache-2.0 | 52.3 AP | Yes (with care on INT8 for attn) |
| D-FINE-L | Real-time DETR (HGNetv2-L) | 31 M | COCO / Obj365→COCO | Apache-2.0 | 54.0 AP | Borderline at 640; likely <10 FPS at 1280 |
| RT-DETRv2-R18 | DETR + ResNet-18 + deformable attn | 20 M | COCO | Apache-2.0 | 47.9 AP | Yes — widely deployed |
| RT-DETRv2-R50 | DETR + ResNet-50 | 42 M | COCO | Apache-2.0 | 53.4 AP | Borderline at 640 |
| YOLOv10-S / -M | CNN, NMS-free dual assignment | 7.2 / 15.4 M | COCO | **AGPL-3.0** (THU-MIG) | 46.3 / 51.1 AP | Yes but license-risk |
| YOLOv11 / v12 | CNN | varies | COCO | **AGPL-3.0** (Ultralytics) | 47–54 AP | License-risk |
| YOLO-World-S/M | Open-vocab (YOLOv8 + CLIP text) | 13–48 M | O365+GoldG+LVIS | GPL-3.0 (code) / Apache (weights vary) | 35–45 AP (zero-shot) | Possible — text tower can be stripped after distillation |
| Grounding-DINO-T | Open-vocab DETR | 172 M | O365+GoldG+Cap4M | Apache-2.0 | 52.5 AP (zero-shot) | NOT for edge; use as **auto-labeller / distillation teacher** only |
| RF-DETR (base) | DINOv2-S + DETR head | 29 M | COCO+O365 | Apache-2.0 | 54.7 AP | Yes (Roboflow claims 25 ms CPU) |
| DEIM-D-FINE-S/M | DETR with dense O2O matching | 10 / 19 M | COCO | Apache-2.0 | 49.0 / 52.7 AP | Same class as D-FINE |
| YOLOv8/v11 + PPE finetunes | CNN | 3–25 M | SHWD / Hard-Hat | **AGPL-3.0** | mAP50 0.89–0.94 on SHWD | Yes but license-risk |

Notes:

- AGPL/GPL (Ultralytics YOLOv8/v10/v11/v12 code, YOLO-World code) is a **commercial-use blocker** unless a paid license is obtained. Listed only for benchmark context.
- D-FINE, RT-DETRv2, YOLOX, RF-DETR, Grounding-DINO, DEIM are Apache-2.0 / permissive — safe.

## 3. Top 3 recommendations

### #1 — **D-FINE-M (Obj365→COCO)** — *replace YOLOX-M as primary*

- **Why:** Highest accuracy in the ~20 M-param single-stage DETR class (COCO AP 52.3 vs YOLOX-M 46.9). NMS-free FDR head directly addresses the crowded-scene NMS-suppression risk noted in the platform doc. HGNetv2 backbone is CNN-heavy → PTQ-friendly. Obj365-pretrained variant gives a much stronger warm-start for small persons/heads than pure COCO.
- **Fine-tune path:** HF Transformers (`D-FINE` class already integrated). Remap head to 4 classes, freeze backbone for 5 epochs, then full fine-tune 60–100 epochs on the 62K merged dataset + custom nitto_hat. Existing `core/p06_models` registry key `dfine-m`.
- **Dataset:** SHWD + Hard Hat Workers + Construction-PPE + SH17 + Safety-Guard + custom factory (Nitto hat).
- **Est FPS @18 TOPS INT8:** 640² ≈ 30–45 FPS (GFLOPS ~57); 1280² ≈ 10–15 FPS. Within budget and above the 10 FPS floor.

### #2 — **RT-DETRv2-R18** — *lower-risk transformer alternative*

- **Why:** Most-deployed real-time DETR on HF (107K downloads). Smaller than D-FINE-M in practice on NPU due to ResNet-18 backbone — simpler conv graph quantizes cleanly. Apache-2.0. Good fallback if D-FINE-M INT8 regresses on the deformable-attention op.
- **Fine-tune path:** HF Transformers `RTDetrV2ForObjectDetection`. 4-class head swap, 50–80 epochs. Already in `core/p06_models` as `rtdetr-r18`.
- **Dataset:** same as #1.
- **Est FPS @18 TOPS INT8:** 640² ≈ 40–55 FPS; 1280² ≈ 12–18 FPS.

### #3 — **YOLOX-M (baseline, retained)** — *keep as safety net*

- **Why:** Lowest deployment risk — the pipeline (`features/ppe-helmet_detection/configs/06_training.yaml`) is already validated, INT8 quantization behavior is well-understood on generic 18 TOPS NPUs, and Apache-2.0. Train in parallel to D-FINE-M as the A/B comparator the platform doc already specifies.
- **Fine-tune path:** unchanged — existing config, 200 epochs, focal+ciou.
- **Est FPS @18 TOPS INT8:** 640² ≈ 45–60 FPS; 1280² ≈ 12–18 FPS.

**Bonus — distillation teacher:** **Grounding-DINO-T** (zero-shot) as a **pseudo-labeller** for nitto_hat and hard-negative mining on un-annotated factory footage — never deployed to edge.

## 4. Pretrained weights

Local destination: `ai/pretrained/ppe-helmet_detection/`.

| Model | URL | License | Size | SHA256 | Local dest |
|---|---|---|---|---|---|
| D-FINE-M (Obj365→COCO) | https://huggingface.co/ustc-community/dfine-medium-obj2coco/resolve/main/model.safetensors | Apache-2.0 | 78.7 MB | `eb418ee9bcbaf052f57e71a695bad9af653355cef8defd547c8be75e9ab90bd8` | `ai/pretrained/ppe-helmet_detection/dfine_medium_obj2coco.safetensors` |
| YOLOX-M (COCO) | https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth | Apache-2.0 | 203.1 MB | `60076992b32da82951c90cfa7bd6ab70eba9eda243e08b940a396f60ac2d19b6` | `ai/pretrained/ppe-helmet_detection/yolox_m.pth` |
| RT-DETRv2-R18 (COCO) | https://huggingface.co/PekingU/rtdetr_v2_r18vd | Apache-2.0 | ~78 MB | *(fetched on demand via HF)* | `ai/pretrained/ppe-helmet_detection/rtdetr_v2_r18vd/` |
| D-FINE-S (Obj365→COCO) | https://huggingface.co/ustc-community/dfine-small-obj2coco | Apache-2.0 | ~42 MB | *(fetched on demand via HF)* | `ai/pretrained/ppe-helmet_detection/dfine_small_obj2coco/` |

All four URLs returned HTTP 200/302 on `curl -I -L` verification (2026-04-14). D-FINE-M and YOLOX-M are downloaded and SHA256-verified. Files are gitignored (tracked via DVC).

## 5. Edge deployment notes

- **Export format:** generic ONNX opset 17+. D-FINE and RT-DETRv2 export cleanly via HF `optimum.exporters.onnx` with `model_type=d_fine` / `rt_detr_v2`. YOLOX exports via its built-in `tools/export_onnx.py`.
- **INT8 PTQ:** use a calibration set of ~500 representative factory frames. D-FINE's deformable attention and bbox-decoder softmax are the two quant-sensitive ops — keep them FP16 (mixed precision) if AP drops >1.5 pt on PTQ. YOLOX is fully CNN and typically loses <0.5 AP on INT8.
- **Chip-agnostic:** no vendor ops. Avoid custom plugins; stick to stock ONNX ops so the model is portable across any ~18 TOPS INT8 NPU.
- **Input size policy:** ship **640²** as default; flip to **1280²** only if nitto_hat/small-helmet recall <0.85 at 640² — all three models still clear 10 FPS at 1280² within the 18 TOPS budget.
- **NMS-free advantage:** D-FINE / RT-DETRv2 skip post-processing NMS, reducing CPU-side latency by 2–4 ms vs YOLOX in crowded frames.
- **Tracker:** ByteTrack (MIT) on CPU — unchanged from baseline.

## 6. Datasets for fine-tune

| Dataset | Images | Classes | License | Notes |
|---|---|---|---|---|
| SHWD (Safety-Helmet-Wearing) | 7,581 | helmet, person/head | Open (academic) | Standard benchmark. Repo: https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset |
| Hard Hat Workers (Roboflow public) | 5,000 | helmet, head, person | CC BY 4.0 | https://public.roboflow.com/object-detection/hard-hat-workers |
| Construction-PPE | 1,416 | 11 PPE classes | MIT | Already in baseline pipeline |
| Safety-Guard | 5,000+ | 8 PPE classes | MIT | — |
| SH17 | 8,099 | 17 PPE classes (manufacturing) | Academic | Stronger domain match than construction |
| GDUT-HWD | 3,200 | multi-color hard hats | Research | Color variety |
| PPE-Detection (Kaggle / Roboflow mirrors) | ~10 K | 11 PPE | Varied — check per-download | Use only CC/MIT subsets |
| **Custom Nitto factory** | 2,100–3,500 target | 4 classes + nitto_hat | Internal | **Critical gap** — no public substitute |
| Grounding-DINO pseudo-labels | ~5–10 K | auto-labelled | Apache-2.0 | Distillation / hard-negative mining on unlabelled footage |

## 7. Verdict vs ROADMAP baseline

**Recommendation: augment, don't abandon.**

- **Keep YOLOX-M** as the W3–W4 v1 baseline (already under training per ROADMAP).
- **Add D-FINE-M (Obj365→COCO)** as the parallel v1-alt track — this directly matches the platform doc's "Option 2: D-FINE-S" contingency, upgraded to -M for the extra ~4 AP headroom at negligible edge-FPS cost within 18 TOPS INT8.
- **Decision gate (end of W6):** pick the winner on `nitto_hat` mAP@0.5 and crowded-scene recall. If D-FINE-M wins by ≥2 AP on nitto_hat without dropping below 15 FPS @640² INT8 — promote to primary for W7–W8. Otherwise keep YOLOX-M.
- **Do not adopt** Ultralytics YOLOv8/v10/v11/v12 or YOLO-World as deployed models — AGPL-3.0 is a commercial blocker for Nitto Denko delivery. Use **only** as offline auxiliary tools (e.g. Grounding-DINO for pseudo-labels, with Apache-2.0 code).
- **Net result:** stronger detector (+5 AP on COCO, better small-object + crowded-scene behaviour) at ≥10 FPS INT8, same license posture, minimal pipeline change (D-FINE already wired into `core/p06_models`).

## 8. References

- D-FINE paper — https://arxiv.org/abs/2410.13842
- RT-DETRv2 paper — https://arxiv.org/abs/2407.17140
- D-FINE repo — https://github.com/Peterande/D-FINE
- RT-DETR repo — https://github.com/lyuwenyu/RT-DETR
- YOLOX repo — https://github.com/Megvii-BaseDetection/YOLOX
- RF-DETR repo — https://github.com/roboflow/rf-detr
- HF D-FINE checkpoints — https://huggingface.co/ustc-community
- HF RT-DETRv2 checkpoints — https://huggingface.co/PekingU
- SHWD — https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset
- Hard Hat Workers (Roboflow) — https://public.roboflow.com/object-detection/hard-hat-workers
- "Improved YOLOv10 helmet detection" (J. Real-Time Img Proc 2025) — https://link.springer.com/article/10.1007/s11554-025-01775-y
- "YOLOv8 helmet in complex environment" (Sci Rep 2025) — https://www.nature.com/articles/s41598-025-08828-z
- "PPE detection YOLOv10 + transformers" (Sci Rep 2025) — https://www.nature.com/articles/s41598-025-12468-8
- DigitalOcean SOTA detection guide 2025 — https://www.digitalocean.com/community/tutorials/best-object-detection-models-guide

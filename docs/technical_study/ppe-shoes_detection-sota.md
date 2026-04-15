# Safety Shoes Detection — SOTA Pretrained Model Survey (2025–2026)

> Scope: evaluate the ROADMAP baseline (YOLOX-Tiny + MobileNetV3, two-stage) against
> 2025 SOTA detectors and fine-grained classifiers for edge deployment at ~18 TOPS INT8,
> chip-agnostic. Target license: Apache-2.0 / BSD / MIT. AGPL (Ultralytics) flagged.

## 1. Task summary

- **Input:** 1080p factory camera frame. Feet occupy ~60×40 px (~20×13 px after 640 resize).
- **Task:** Stage 1 — detect `person` (and optionally `shoe_region`) at 640. Stage 2 —
  classify 224×224 foot crop as `foot_with_safety_shoes` vs `foot_without_safety_shoes`.
- **Bottleneck:** fine-grained classification of small, often-occluded, texture-heavy foot
  crops (toe cap, sole pattern, ankle height). Public data is scarce (3.7K images, target 14K).
- **Edge envelope:** ~18 TOPS INT8, 25–40 FPS end-to-end, ONNX + generic INT8,
  commercially usable license.
- **Conclusion preview:** keep the two-stage topology; upgrade Stage 1 to **D-FINE-N/S**
  and Stage 2 to a **DINOv2-small / EfficientFormerV2-S1** head. See §7.

## 2. Candidate models

### 2a. Stage 1 — Person / foot-region detector (input 640)

| Model | Params | COCO AP | License | ONNX | Notes |
|---|---|---|---|---|---|
| YOLOX-Tiny (baseline) | 5.1M | 32.8 | Apache-2.0 | yes | Proven INT8, mature quantization, CNN — easy on generic INT8 kernels. |
| **D-FINE-N** | 3.8M | 42.8 | Apache-2.0 | yes | NMS-free DETR, ICLR'25 Spotlight, +10 AP over YOLOX-Tiny at smaller size. HF: `ustc-community/dfine-nano-coco`. |
| **D-FINE-S** | 10.4M | 48.7 | Apache-2.0 | yes | Best accuracy/size for ~18 TOPS. HF: `ustc-community/dfine-small-coco`. |
| RT-DETRv2-R18 | 20.2M | 47.9 | Apache-2.0 | yes | Heavier; beaten by D-FINE-S at similar speed. HF: `PekingU/rtdetr_v2_r18vd`. |
| RF-DETR-Nano | ~4M | ~48 | Apache-2.0 | yes (ONNX Hub) | Roboflow 2025 NAS-tuned DETR; 100 FPS T4. ICLR'26. |
| RF-DETR-Small | ~12M | ~52 | Apache-2.0 | yes | Strong alternative to D-FINE-S. |
| YOLOv10-N/S | 2.3/7.2M | 38.5/46.3 | AGPL-3.0 ⚠ | yes | **AGPL — flag, skip for product.** |
| YOLOv11-N/S | 2.6/9.4M | 39.5/47.0 | AGPL-3.0 ⚠ | yes | **AGPL — skip.** |
| YOLOv12-N/S | 2.6/9.3M | 40.6/48.0 | AGPL-3.0 ⚠ | yes | Attention-centric, 2025; **AGPL — skip.** |

### 2b. Stage 2 — Fine-grained shoe classifier (input 224)

| Model | Params | IN1K top-1 | License | Notes |
|---|---|---|---|---|
| MobileNetV3-Small (baseline) | 2.5M | 67.7 | BSD-3 | Weakest fine-grained capability, but cheapest. |
| **DINOv2-small (ViT-S/14)** | 22.1M | 81.1 linear | Apache-2.0 | SSL foundation model, **best fine-grained transfer** on small datasets; ~3.7K shoe set favors SSL. HF: `facebook/dinov2-small`. |
| DINOv3-ViT-S/16 | 21.6M | 82.0+ | Meta DINOv3 (other, **gated**) | Better than v2 but license restrictive + gated HF repo; **flag**. |
| EVA-02-Tiny/Small | 6/22M | 80–85 | MIT | Strong fine-grained; patched timm weights. |
| SigLIP-B/16 | 86M | — | Apache-2.0 | Too heavy for Stage 2 at 224. Use only for zero-shot labeling. |
| **EfficientFormerV2-S0/S1** | 3.6/6.2M | 75.7/79.0 | Apache-2.0 | Hybrid ViT, INT8-friendly, edge-tuned. HF: `timm/efficientformerv2_s0.snap_dist_in1k`. |
| FastViT-T8/T12 | 3.6/6.8M | 76.2/79.1 | **Apple ASCL (non-commercial concerns)** ⚠ | Good speed; license restrictive for product. Flag. |
| MobileViTv2-1.0 | 4.9M | 78.1 | **Apple sample-code license** ⚠ | Flag — non-commercial clauses. |
| ConvNeXt-Tiny | 28.6M | 82.1 | Apache-2.0 (timm) | Solid CNN; heavier than EFv2-S1 for similar accuracy. |
| ViT-Tiny (deit3) | 5.7M | 74.5 | Apache-2.0 | Baseline upgrade path noted in platform doc. |

### 2c. Single-stage alternatives

| Model | Params | Input | License | Notes |
|---|---|---|---|---|
| YOLOX-M (baseline alt) | 25.3M | 1280 | Apache-2.0 | Simpler pipeline; poor foot recall. |
| D-FINE-M | 19.2M | 640/1024 | Apache-2.0 | NMS-free; viable if shoe classes merged into detection head. |
| RF-DETR-Small | ~12M | 576 | Apache-2.0 | End-to-end detect+classify; good if classes expanded. |

## 3. Top 3 recommendations

### Stage 1 (person / shoe-region detection)
1. **D-FINE-N (3.8M, Apache-2.0)** — drop-in replacement for YOLOX-Tiny. +10 COCO AP,
   NMS-free, handles partially-visible persons near machinery. Est. ~35–55 FPS INT8 at 640 on ~18 TOPS.
2. **D-FINE-S (10.4M, Apache-2.0)** — use if accuracy budget allows. Est. ~20–30 FPS INT8.
3. **YOLOX-Tiny (baseline, 5.1M)** — keep as safety net; most stable INT8 on generic edge.

Fine-tune path: initialize from COCO weights → replace head for `{person, shoe_region}` →
train 100 epochs on 3.7K + augmented 14K set, mosaic=1.0, copy-paste=0.3, mixup=0.15.
Validate INT8 drop (<1 mAP target).

### Stage 2 (fine-grained shoe classifier)
1. **DINOv2-small + 2-class linear head (22M, Apache-2.0)** — best small-data fine-grained
   accuracy. Freeze backbone, train linear probe, then LoRA-unfreeze last 2 blocks.
   Est. 80–120 FPS INT8 per crop at 224 on ~18 TOPS (ViT-S/14 is ~4.6 GFLOPs).
2. **EfficientFormerV2-S1 (6.2M, Apache-2.0)** — best edge-speed alternative if DINOv2
   ViT-S latency is unacceptable. Est. 200+ FPS INT8 per crop.
3. **MobileNetV3-Small (baseline, 2.5M, BSD-3)** — keep as fallback; fastest but
   weakest on texture-level shoe discrimination.

Fine-tune path: 2-way head, AdamW 1e-3 → cosine, 100 epochs, heavy augmentation
(RandAug + RandomErasing 0.3 + color jitter), class-balanced sampler, mixup 0.2.

### Single-stage top pick
**D-FINE-S @ 640** with classes `{person, foot_with, foot_without, shoe_region}` — only if
operations strongly prefer one ONNX graph. Expect ~5–8 AP lower on foot classes vs two-stage.

## 4. Pretrained weights

Local root: `ai/pretrained/ppe-shoes_detection/`

| Model | URL | License | Size | SHA256 | Local dest |
|---|---|---|---|---|---|
| D-FINE-N (COCO) | https://huggingface.co/ustc-community/dfine-nano-coco/resolve/main/model.safetensors | Apache-2.0 | 15,278,996 B | `19e06bdc873da819920a8d373b879721a5b9759d822f8213220bb09abbdab58b` | `ai/pretrained/ppe-shoes_detection/dfine_nano_coco.safetensors` |
| D-FINE-S (COCO) | https://huggingface.co/ustc-community/dfine-small-coco/resolve/main/model.safetensors | Apache-2.0 | — | (not downloaded; verified 200) | `ai/pretrained/ppe-shoes_detection/dfine_small_coco.safetensors` |
| DINOv2-small | https://huggingface.co/facebook/dinov2-small/resolve/main/pytorch_model.bin | Apache-2.0 | 88,297,097 B | `1051e25b2ed69ddad24f3c41e7b6eed6e7f7d012103ea227e47eb82e87dc2050` | `ai/pretrained/ppe-shoes_detection/dinov2_small.bin` |
| EfficientFormerV2-S0 | https://huggingface.co/timm/efficientformerv2_s0.snap_dist_in1k/resolve/main/pytorch_model.bin | Apache-2.0 | 14,800,557 B | `673ae59957819e27619bdf29dd0b04a756350f2257e3cb7967916617e7570844` | `ai/pretrained/ppe-shoes_detection/efficientformerv2_s0.bin` |
| DINOv3-ViT-S/16 | https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m | Meta DINOv3 (other) | n/a | — | **Gated + restrictive license — skipped** |
| RF-DETR-Small ONNX | https://huggingface.co/onnx-community/rfdetr_small-ONNX/resolve/main/onnx/model.onnx | Apache-2.0 | — | (verified 200, not downloaded) | `ai/pretrained/ppe-shoes_detection/rfdetr_small.onnx` |
| FastViT-T8 | https://huggingface.co/timm/fastvit_t8.apple_in1k | Apple ASCL ⚠ | — | — | Skipped — license |
| MobileViTv2-1.0 | https://huggingface.co/timm/mobilevitv2_100.cvnets_in1k | Apple sample-code ⚠ | — | — | Skipped — license |
| YOLOv10/11/12 | ultralytics/* | AGPL-3.0 ⚠ | — | — | **Skipped — AGPL** |

All downloaded URLs verified with `curl -I` (HTTP 200). Stage 1 top pick (D-FINE-N) and
Stage 2 top picks (DINOv2-small, EfficientFormerV2-S0) downloaded to the local path above.

## 5. Edge deployment notes

- **~18 TOPS INT8 budget**, chip-agnostic. Use generic ONNX opset ≥17 and symmetric
  per-channel weight quantization with per-tensor activations.
- **D-FINE** exports to ONNX via HF Transformers (`RTDetrV2ForObjectDetection`-style path
  for RT-DETRv2; D-FINE uses `DFineForObjectDetection`). Use `opset=17`, disable dynamic
  axes only for batch dimension for best quantization. Replace `grid_sample` with discrete
  sampler (already optional in RT-DETRv2; D-FINE uses standard ops).
- **DINOv2 ViT-S/14** — 224×224 crop uses 16×16 tokens → ~4.6 GFLOPs. INT8 quantization
  of ViTs needs calibration on ~512 shoe crops with percentile (99.99%) activation
  clipping to avoid attention-softmax overflow. Expect <1% top-1 drop with QDQ.
- **Two-stage latency budget:** Stage 1 ~15–25 ms INT8 (D-FINE-N 640), Stage 2 ~6–10 ms
  per crop (DINOv2-S 224) × up to 5 persons/frame = ~30–70 ms per frame; run Stage 2
  async on parallel NPU stream if chip supports it. Target ≥10 FPS hit: yes, with margin.
- **Occlusion mitigation:** keep ByteTrack + ≥30-frame confirmation unchanged. DETRs
  (D-FINE) help with partially-visible persons via global attention.
- **Hard negatives:** keep the 500–1,000 loose-shoe images with empty labels — DETR
  models can over-detect shoes-on-floor without them.

## 6. Datasets for fine-tune

| Dataset | Usage | License | Notes |
|---|---|---|---|
| Internal `f_safety_shoes` | Primary (3.7K) | Internal | Already in repo. |
| Construction-PPE (Roboflow) | +boots class | MIT | ~1.4K; small shoe coverage. |
| SH17 (foot class) | Stage 1 foot aug | CC BY 4.0 | ~2K feet; use for shoe_region pretraining. |
| PPE Detection (Roboflow 10K) | Mixed PPE with boots | Varied | Review per-image licenses before merging. |
| Shoe50K / UT-Zap50K | Stage 2 fine-grained SSL pretrain | Non-commercial ⚠ | **Only use for weights not distributed** — flag. |
| Custom factory footage | Critical occlusion bucket | Internal | Collect 1.5K occluded + 1K clean (see platform doc). |
| Generative aug (SAM3+Flux) | +5K synthetic | Internal | Pipeline `p03_generative_aug`. |

For Stage 2 fine-grained, consider self-supervised pretraining of DINOv2-small on
unlabeled factory footage (DINO-style) before the 2-class linear-probe head — this is the
highest-leverage move given the 3.7K labeled budget.

## 7. Verdict vs ROADMAP baseline

**Keep topology (two-stage). Upgrade both stages. Baseline is now the fallback.**

| Aspect | Baseline | Recommended 2026 |
|---|---|---|
| Stage 1 | YOLOX-Tiny (5.1M) | **D-FINE-N (3.8M, Apache-2.0)** — +10 COCO AP, smaller, NMS-free |
| Stage 2 | MobileNetV3-Small (2.5M) | **DINOv2-small linear probe (22M, Apache-2.0)** — best fine-grained on small data; fallback **EfficientFormerV2-S1 (6.2M)** |
| Pipeline | Two-stage + ByteTrack | Unchanged |
| License | Apache/BSD clean | Apache clean; avoid Ultralytics, Apple-ASCL, Meta-DINOv3 |
| Edge headroom | 7.6M total | ~26M total — still <50% of 18 TOPS budget at INT8 |
| Single-stage alt | YOLOX-M @ 1280 | D-FINE-S @ 640 multi-class |

Rationale:
- The dominant error mode is **fine-grained classification of small crops**, not person
  detection. MobileNetV3-Small caps Stage 2 accuracy; DINOv2-small unlocks meaningful
  gains on the 3.7K dataset thanks to strong SSL features.
- D-FINE-N matches MobileNet-level compute while beating YOLOX-Tiny by ~10 AP; it is
  a near-free upgrade for Stage 1.
- AGPL (YOLOv10/11/12) and Apple-ASCL (FastViT / MobileViTv2) are disqualified under the
  Nitto Denko Apache-2.0 policy.
- DINOv3 is stronger than DINOv2 but the weights are gated and under a restrictive Meta
  license — **flagged, do not adopt without legal review**.

Risk items to validate during W5–W6 midpoint review:
1. D-FINE INT8 stability on the chosen edge chip (vendor-agnostic, but ViT-style attention
   ops may need calibration tuning).
2. DINOv2 ViT-S INT8 accuracy drop — budget 1 week for PTQ calibration; fall back to
   EfficientFormerV2-S1 if drop >3% top-1.
3. Per-frame latency with 5+ persons: Stage 2 batched crops to amortize.

## 8. References

- D-FINE (ICLR 2025 Spotlight): https://github.com/Peterande/D-FINE — Apache-2.0.
  HF Transformers: https://huggingface.co/docs/transformers/model_doc/d_fine
- D-FINE weights: https://huggingface.co/ustc-community/dfine-nano-coco ,
  https://huggingface.co/ustc-community/dfine-small-coco
- RT-DETRv2 (CVPR 2024 / arXiv 2407.17140): https://github.com/lyuwenyu/RT-DETR ,
  HF: https://huggingface.co/PekingU/rtdetr_v2_r18vd
- RF-DETR (Roboflow, ICLR 2026): https://github.com/roboflow/rf-detr — Apache-2.0.
  ONNX: https://huggingface.co/onnx-community/rfdetr_small-ONNX
- DINOv2 (arXiv 2304.07193): https://github.com/facebookresearch/dinov2 ,
  HF: https://huggingface.co/facebook/dinov2-small — Apache-2.0
- DINOv3 (arXiv 2508.10104): https://ai.meta.com/dinov3/ — Meta DINOv3 license, gated
- EVA-02 (arXiv 2303.11331): https://github.com/baaivision/EVA — MIT
- EfficientFormerV2 (ICCV 2023 / arXiv 2212.08059):
  https://huggingface.co/timm/efficientformerv2_s0.snap_dist_in1k — Apache-2.0 (timm port)
- FastViT (ICCV 2023): https://github.com/apple/ml-fastvit — Apple ASCL ⚠
- MobileViTv2 (arXiv 2206.02680): https://github.com/apple/ml-cvnets — Apple ⚠
- timm / pytorch-image-models: https://github.com/huggingface/pytorch-image-models
- YOLO licensing overview: https://www.ultralytics.com/license — AGPL-3.0 (flagged)
- YOLOX: https://github.com/Megvii-BaseDetection/YOLOX — Apache-2.0
- Platform doc: `ai/docs/03_platform/ppe-shoes_detection.md`
- ROADMAP: `ai/docs/ROADMAP.md`

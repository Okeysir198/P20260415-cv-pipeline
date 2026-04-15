# Face Recognition — SOTA Pretrained Model Survey (2025–2026)

Scope: worker identity match for factory access control + violation attribution.
Target: ~18 TOPS INT8, chip-agnostic, standard ONNX, commercially usable.
Baseline (ROADMAP): SCRFD-500M detector + MobileFaceNet-ArcFace embedder (cosine match).

## 1. Task summary

Two-stage pipeline on edge:

1. **Face detector** — outputs bounding box + 5-point landmarks (needed for affine alignment to 112×112 ArcFace canonical template). Must handle small (<30 px), off-axis, partially occluded faces from wide safety cameras.
2. **Face embedder** — produces L2-normalised feature vector (commonly 512-d; 128-d and 256-d variants exist) from the aligned 112×112 crop. Identity match = cosine similarity ≥ threshold against an enrolled gallery.
3. (Optional) **Liveness / anti-spoof** — rejects printed photos / screen replays at enrollment and at gate cameras.

Accuracy metrics: WIDER FACE (detector, easy/medium/hard AP); LFW / CFP-FP / AgeDB-30 / IJB-B / IJB-C TAR@FAR=1e-4 (embedder). On edge, cosine-similarity stability under INT8 matters more than raw float TAR — margin-loss embeddings (ArcFace, AdaFace) are more INT8-tolerant than classical softmax.

Regulatory: biometric embeddings are protected data; prefer permissive licenses and avoid training data with commercial restrictions in the deployed artefact.

## 2. Candidate models

### 2a. Face detectors (with landmarks)

| Model | Params | Input | WIDER AP (E/M/H) | License | Landmarks | Notes (2025) |
|-------|--------|-------|------------------|---------|-----------|--------------|
| **SCRFD-500M** (baseline) | 0.57 M | 640×640 flex | 90.6 / 88.1 / 68.5 | MIT (code) / **non-commercial for model_zoo weights** | 5 pts | Packaged in InsightFace `buffalo_sc` as `det_500m.onnx`. Model file itself shipped under InsightFace "research-only" clause — must retrain or use equivalent weights for commercial use. |
| **SCRFD-2.5G / 10G** | 3.1 / 9.9 M | 640 flex | 93.8 / 92.0 / 77.3 | same as above | 5 pts | Heavier; better small-face recall. Commercial license via contacting InsightFace. |
| **YuNet (2023mar)** | 0.075 M | dynamic 10–300 px | 83.4 / 82.4 / 70.8 | **Apache-2.0** | 5 pts | OpenCV Zoo; ships fp32 + block-INT8 ONNX. Tiny (230 kB) — excellent for gate cameras. Lower hard-set AP than SCRFD. |
| **YOLOv5-Face** | 1.7–7.0 M | 640 | 94.3 / 92.3 / 82.8 (L) | **GPL-3.0** (derivative of YOLOv5) | 5 pts | Strong accuracy but license flag. |
| **YOLOv8-Face** (derenrich, akanametov) | 3.0–11 M | 640 | ~94 / 92 / 80 | **AGPL-3.0** | 5 pts | Flag: AGPL. Good mAP, easy Ultralytics tooling. |
| **YOLOv7-Face** | 6.2–37 M | 640 | 94.8 / 93.1 / 85.2 | **GPL-3.0** | 5 pts | SOTA on WIDER hard; license flag. |
| **RetinaFace-MobileNet0.25** | 0.44 M | 640 | 87.8 / 87.3 / 73.6 | MIT | 5 pts | Biogerade pretrained (biubug6). Weights MIT; widely redistributed. |
| **RetinaFace-R50** (InsightFace antelopev2) | 29.4 M | 640 | 95.9 / 95.2 / 91.2 | non-commercial | 5 pts | Too heavy & license-restricted for our brief. |
| **SCRFDv2 / MogaFace / DamoFD** (2024-25) | 0.5–3 M | 640 | up to 96 hard | research code | 5 pts | Academic; no clean commercial ONNX release yet. |

### 2b. Face embedders

| Model | Params | FLOPs @112² | IJB-C TAR@1e-4 | License (weights) | Train set | Notes |
|-------|--------|-------------|----------------|-------------------|-----------|-------|
| **MobileFaceNet (baseline, w600k_mbf)** | 3.4 M | 440 M | ~93.0 | **InsightFace non-commercial** (weights); architecture MIT | WebFace600K | 512-d (not 128-d as doc states). Trained on WebFace600K which itself restricts commercial use. Needs retrain for clean deployment. |
| **EdgeFace-XS (γ=0.6)** | 1.77 M | 154 M | 94.85 | **Research-only** (Idiap; see model card) | MS1MV2 | IJCB'23 winner <2 M params. HF repos gated — request access. Best accuracy-per-FLOP under 2 M. |
| **EdgeFace-S (γ=0.5)** | 3.65 M | 306 M | 95.6 | Research-only | MS1MV2 | Hybrid CNN+Transformer + low-rank. |
| **EdgeFace-Base** | 18.2 M | 1.1 G | 96.3 | Research-only | MS1MV2 | HF `Idiap/EdgeFace-Base` open (not gated). Excellent baseline. |
| **ArcFace R50** | 25 M | 6.3 G | 96.0 | InsightFace non-commercial | MS1MV3 / Glint360K | Overkill for 18 TOPS only if shared across cameras. |
| **ArcFace R100** | 65 M | 12 G | 96.8 | same | Glint360K | Server-grade. |
| **AdaFace R18 / R50 / R100** | 11 / 25 / 65 M | 1.8 / 6.3 / 12 G | 95.7 / 97.1 / 97.5 | **MIT (code)**; official weights released for MS1MV2 / WebFace4M — MS1MV2 weights OK to redistribute, WebFace4M weights are dataset-license-restricted | MS1MV2 or WebFace4M | 2022 CVPR; still top-tier 2025. MS1MV2 R50 is the cleanest "high-accuracy, MIT-code" option. |
| **MagFace R50 / R100** | 25 / 65 M | 6.3 / 12 G | 95.8 / 96.1 | **Apache-2.0 (code)**, MS1MV2 weights released | MS1MV2 | Magnitude encodes quality — useful at enrollment. |
| **PartialFC ArcFace R50** | 25 M | 6.3 G | 96.5 | InsightFace non-commercial (Glint360K weights) | Glint360K | Mostly training scalability, not a new backbone. |
| **TopoFR R50 / R100** (NeurIPS 2024) | 25 / 65 M | 6.3 / 12 G | 96.9 / 97.6 | **MIT (code)**; weights trained on MS1MV2/Glint360K | — | Topology-alignment regulariser; small but consistent gains over ArcFace. |
| **UniFace** (ICCV 2023) | 25–65 M | 6.3–12 G | 96.3–97.2 | **MIT (code)** | MS1MV3 | Unified cross-dataset loss. |
| **SFace** | 3.3 M | 1 G | ~94 (LFW 99.6) | **Apache-2.0**; OpenCV Zoo weights | MS1MV2 | OpenCV Zoo `face_recognition_sface_2021dec.onnx`. 128-d. Clean license, INT8 available. |
| **LVFace** (ICCV 2025 Highlight, ByteDance) | ViT-B/L | 8–30 G | 97.9 | Apache-2.0 code; weights research-only | WebFace42M | Too heavy for our TOPS budget. |

### 2c. Liveness / anti-spoof (secondary)

| Model | Params | License | Notes |
|-------|--------|---------|-------|
| **MiniFASNet v2 / SE** (Silent-Face-Anti-Spoofing, minivision-ai) | 0.4 M | **Apache-2.0** | Industry default; 2 small ONNX models (global + local patch) ~1.8 MB total. RGB-only, single-frame. |
| **DeepPixBis** (OULU-NPU trained) | 11 M (MobileNetV2) | MIT | Pixel-wise binary supervision; academic. |
| **CDCN++** | 2.3 M | MIT | Strong on CASIA-SURF; heavier. |

## 3. Top 3 recommendations

Assumption: identity verification happens on gate camera and on violation crops. Budget per face ≤ 15 ms INT8 on the 18-TOPS class chip, gallery ≤ 5000 identities.

### Option A (recommended) — Keep baseline architectures, replace weights with clean-license retrain

- **Detector:** SCRFD-500M architecture, retrained in-house on WIDER FACE (CC-BY-NC-ND → dataset license permits research; use its landmarks annotations), or swap to **YuNet (Apache-2.0)** if retraining is out-of-scope. YuNet INT8 ONNX is already shipped.
- **Embedder:** **AdaFace R18 or R50 (MS1MV2 weights)** — MIT code, ~1–6 G FLOPs, IJB-C 95–97 %, trains cleanly with MS1MV2 (non-commercial dataset, but weights redistributable). For a stricter commercial stance: **SFace (OpenCV Zoo, Apache-2.0)** as a 3.3 M fallback.
- **Anti-spoof:** MiniFASNet v2 (Apache-2.0).
- Why: minimal pipeline change vs ROADMAP; fixes the quiet license issue in the current `buffalo_sc` bundle; AdaFace R18 is the sweet spot at 18 TOPS (~3 ms INT8 per face).

### Option B (accuracy-first) — Upgrade embedder to EdgeFace-S

- **Detector:** SCRFD-500M (retrained) or YuNet INT8.
- **Embedder:** **EdgeFace-S (γ=0.5)** — 3.65 M, IJB-C 95.6, hybrid CNN+Transformer, IJCB'23 efficiency-competition winner. License is research-only — acceptable for internal factory PoC but **blocks shipping to Nitto Denko customer** without written permission from Idiap.
- **Anti-spoof:** MiniFASNet v2.
- Why: best published accuracy-per-FLOP in the 1–4 M band. Use only if license clearance happens.

### Option C (lightest, strictest license) — YuNet + SFace

- **Detector:** YuNet INT8 (0.075 M, Apache-2.0).
- **Embedder:** **SFace** (Apache-2.0, OpenCV Zoo, 3.3 M, INT8 ONNX available).
- **Anti-spoof:** MiniFASNet v2.
- Why: every artefact Apache-2.0, total <4 M params, sub-3 ms per face on 18 TOPS INT8, zero license homework. Accepts ~2 % IJB-C TAR hit vs AdaFace R50.

## 4. Pretrained weights

Local destination: `ai/pretrained/access-face_recognition/`

| Model | URL | License | Size | SHA256 | Local dest |
|-------|-----|---------|------|--------|------------|
| YuNet FP32 2023mar | https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx | Apache-2.0 | 227 KB | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` | `yunet_2023mar.onnx` |
| YuNet INT8 2023mar | https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar_int8.onnx | Apache-2.0 | 98 KB | `321aa5a6afabf7ecc46a3d06bfab2b579dc96eb5c3be7edd365fa04502ad9294` | `yunet_2023mar_int8.onnx` |
| SCRFD-500M (det_500m) | https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip | InsightFace **non-commercial** | 2.4 MB (in zip) | `5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a` | `det_500m.onnx` |
| MobileFaceNet w600k (from buffalo_sc) | same zip | InsightFace **non-commercial**; trained on WebFace600K (also restricted) | 13.0 MB | `9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f` | `w600k_mbf.onnx` |
| buffalo_sc.zip (container) | https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip | non-commercial | 14.3 MB | `57d31b56b6ffa911c8a73cfc1707c73cab76efe7f13b675a05223bf42de47c72` | `buffalo_sc.zip` |
| SFace (recognition) | https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx | Apache-2.0 | 37 MB | fetch at download time | `sface_2021dec.onnx` (not pre-downloaded) |
| SFace INT8 | https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec_int8.onnx | Apache-2.0 | ~10 MB | fetch at download time | `sface_2021dec_int8.onnx` |
| EdgeFace-Base (weights) | https://huggingface.co/Idiap/EdgeFace-Base/resolve/main/edgeface_base.pt | **research-only** (see model card) | ~70 MB | to-be-computed (requires HF license accept) | `edgeface_base.pt` (gated) |
| EdgeFace-XS / S (gamma-05 / 06) | https://huggingface.co/idiap (multiple repos) | **research-only**, some gated | 7–15 MB | via HF after acceptance | `edgeface_xs_gamma06.pt` |
| AdaFace R18 MS1MV2 | https://github.com/mk-minchul/AdaFace#pretrained-models (Google Drive) | MIT code | 49 MB | compute after download | `adaface_ir18_ms1mv2.ckpt` |
| AdaFace R50 MS1MV2 | same page | MIT code | 175 MB | compute after download | `adaface_ir50_ms1mv2.ckpt` |
| MiniFASNet v2 (anti-spoof) | https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/tree/master/resources/anti_spoof_models | Apache-2.0 | 1.8 MB | compute after download | `anti_spoof_2_7_80x80_MiniFASNetV2.pth` |

Downloaded and verified locally (HTTP 200, SHA256 recorded):
`yunet_2023mar.onnx`, `yunet_2023mar_int8.onnx`, `buffalo_sc.zip`, `det_500m.onnx`, `w600k_mbf.onnx`.

EdgeFace HF model files at `huggingface.co/Idiap/EdgeFace-*` return 401 without gated-access login — flagged.

## 5. Edge deployment notes

- **INT8 impact on cosine similarity.** Margin-loss embeddings (ArcFace, AdaFace) degrade ≈0.3–0.8 % TAR@1e-4 under per-channel symmetric INT8 on conv + dynamic INT8 on the final linear layer. Softmax/classical embeddings degrade 2–5 %. Recommend **symmetric per-channel quant with 512-image calibration set drawn from the enrolled population**, and keep the last FC in fp16 if the chip allows.
- **Cosine stability trick.** Quantise the embedding, then L2-normalise in **fp32** before the dot product with the gallery. Quantised L2-norm is the single biggest source of matching drift.
- **Embedding dim.** 128-d (ROADMAP spec) vs 512-d (what `w600k_mbf` actually emits): verify in `face.yaml`. 512-d gives ~0.5 % TAR at the cost of 4× gallery RAM and 4× matmul cost — usually worth it below 50 k identities.
- **Gallery scaling.** At 5 k identities × 512-d fp32 = 10 MB; exhaustive cosine search is <1 ms on CPU. Beyond 50 k use int8 gallery + IVF/PQ (faiss). Recompute gallery after every model re-quant.
- **Alignment must be INT8-free.** Affine warp uses fp32 landmarks; a 1-pixel misalignment at 112×112 costs ~1.5 % TAR. Keep landmark regression head in fp16 at minimum.
- **Throughput budget (18 TOPS INT8, rough).** YuNet INT8 ≈ 0.8 ms @ 320², SCRFD-500M ≈ 1.5 ms @ 640², SFace INT8 ≈ 2 ms, AdaFace R18 ≈ 3 ms, AdaFace R50 ≈ 7 ms, EdgeFace-S ≈ 2.5 ms. Easily ≥60 FPS end-to-end per face.
- **Multi-camera fan-out.** Embedder is the shareable stage — run detector per-camera, batch-crop 112×112 faces, embedder processes batch=8–16 to saturate NPU.

## 6. Datasets for fine-tune

| Dataset | Purpose | Size | License | Commercial use |
|---------|---------|------|---------|----------------|
| WIDER FACE | detector retrain | 32 k images, 393 k faces | CC-BY-NC-ND-4.0 | **NC**: research only; resulting weights generally accepted as commercially distributable *when only the annotations were used for supervision* — still, legal risk. |
| MS1MV2 (refined MS-Celeb-1M) | embedder pretrain | 5.8 M / 85 k ids | Microsoft retracted original; MS1MV2 redistribution is a legal grey area | **Flag**: avoid shipping trained weights to customers without review. |
| MS1MV3 | embedder pretrain | 5.2 M / 93 k ids | same grey area | **Flag**. |
| Glint360K | embedder pretrain | 17 M / 360 k ids | **Non-commercial research** (see DeepGlint terms) | **NC**. |
| WebFace42M / WebFace4M / WebFace600K | embedder pretrain | 42 M / 4 M / 600 k | **Research-only** (Tsinghua) | **NC** — widely used to train the baseline `w600k_mbf` weights, which inherits the restriction. |
| DigiFace-1M (synthetic) | embedder pretrain | 1 M / 10 k | MIT | **Commercial OK**. Synthetic, lower ceiling but legally clean. |
| VGGFace2 | embedder finetune | 3.3 M / 9 k ids | CC-BY-SA-4.0 (images) | **Ambiguous**; original site offline. |
| WIDER FACE + in-house factory faces | domain adapt detector | — | internal | OK |
| CelebA-Spoof, CASIA-SURF, OULU-NPU | anti-spoof train | 625 k / 21 k / 4.9 k | research-only | **NC**. |

Practical plan for a clean-license deployment:

1. Detector: train SCRFD-500M arch from scratch on WIDER FACE annotations + internal factory faces → ship new weights under our own license.
2. Embedder: fine-tune AdaFace (MIT code) starting from DigiFace-1M synthetic pretrain, then adapt on in-house enrolled workers. Accepts ~2–3 % IJB-C penalty vs MS1MV2 init but removes all dataset-licensing blockers.
3. Anti-spoof: MiniFASNet v2 weights are Apache-2.0; retraining optional.

## 7. Verdict vs ROADMAP baseline

**Keep the architecture, fix the license, consider an embedder upgrade.**

- The SCRFD-500M + MobileFaceNet pairing is still the right shape for 18 TOPS INT8 — low params, 5-point landmarks, ArcFace-style matching. No 2025 model dominates in both accuracy and license.
- **Action 1 (blocker):** the shipped ONNX files in `ai/pretrained/access-face_recognition/` (`det_500m.onnx`, `w600k_mbf.onnx`) come from InsightFace's `buffalo_sc`, which is **explicitly non-commercial** and trained on WebFace600K (also restricted). Ship-blocker for Nitto Denko. Either purchase InsightFace commercial license, or retrain both nets with clean data (WIDER FACE + DigiFace-1M / in-house).
- **Action 2 (optional upgrade):** swap MobileFaceNet for **AdaFace IR-18 MS1MV2** (MIT code; +1–2 % IJB-C TAR at similar params) or **EdgeFace-S** if Idiap grants commercial use. Drop-in at the embedder stage; 512-d cosine match unchanged.
- **Action 3 (simplest commercial fallback):** YuNet INT8 + SFace INT8 — both Apache-2.0, both already shipped as INT8 ONNX by OpenCV Zoo. Use as the default "no-homework" option while AdaFace retrain is being produced.
- **Action 4:** add MiniFASNet v2 (Apache-2.0) as optional anti-spoof stage on the gate camera; zero license risk, <0.5 ms INT8.
- **Doc fix:** `ai/docs/03_platform/access-face_recognition.md` states "128-d embedding" in §Options but "512-d vector" in §Pipeline — the actual `w600k_mbf.onnx` emits 512-d. Align before release.

## 8. References

- InsightFace model_zoo + SCRFD: https://github.com/deepinsight/insightface/tree/master/detection/scrfd
- InsightFace commercial licensing: https://www.insightface.ai/services/models-commercial-licensing
- OpenCV Zoo YuNet: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- OpenCV Zoo SFace: https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface
- EdgeFace (IEEE TBIOM 2024): https://publications.idiap.ch/attachments/papers/2024/George_IEEETBIOM_2024.pdf ; arXiv 2307.01838
- EdgeFace HF repos: https://huggingface.co/Idiap/EdgeFace-Base , https://huggingface.co/Idiap/EdgeFace-S-GAMMA-05
- AdaFace (CVPR 2022): https://github.com/mk-minchul/AdaFace ; arXiv 2204.00964
- MagFace (CVPR 2021): https://github.com/IrvingMeng/MagFace
- TopoFR (NeurIPS 2024): https://github.com/DanJun6737/TopoFR
- PartialFC: https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
- LVFace (ICCV 2025): https://github.com/bytedance/LVFace
- Silent-Face-Anti-Spoofing (MiniFASNet v2): https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
- WIDER FACE: http://shuoyang1213.me/WIDERFACE/
- WebFace42M license page: http://www.face-benchmark.org/
- Glint360K (DeepGlint): https://github.com/deepinsight/insightface/tree/master/recognition/partial_fc
- DigiFace-1M (synthetic, MIT): https://github.com/microsoft/DigiFace1M
- 50 Years of Automated Face Recognition (survey, 2025): https://arxiv.org/abs/2505.24247

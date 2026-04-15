# Fall Classification — SOTA Pretrained Model Survey (2025–2026)

Scope: evaluate whether the ROADMAP baseline **YOLOX-M** single-frame 2-class (`person` / `fallen_person`) detector-classifier should be kept, replaced, or augmented for the 24/7 worker-fall safety feature running on a ~18 INT8 TOPS, chip-agnostic edge target. Accuracy outweighs speed; ≥10 FPS INT8 is the floor.

## 1. Task summary

- Input: RGB frames (640×640 native; crops 224×224 for classifier variants; 16×224×224 clips for temporal).
- Output: per-person posture/fall decision. Baseline is detection+class-head (`person`, `fallen_person`); alternatives split into (a) detector → person crop → classifier, and (b) detector → tracker → temporal clip classifier.
- Acceptance targets (from platform doc): mAP@0.5 ≥ 0.85, Precision ≥ 0.90, Recall ≥ 0.88, FP < 3%, FN < 2%. Recall is prioritised over precision.
- Pain points baseline struggles with: sitting/crouching/kneeling vs fallen, elevated surfaces (no depth), single-frame ambiguity (person lying to work under equipment). Temporal context and richer visual features are where 2025 SOTA helps.

## 2. Candidate models

| Family | Model | Type | Params | Input | Pretrain | License | Bench (reference) | Edge / INT8 notes |
|---|---|---|---|---|---|---|---|---|
| Baseline | YOLOX-M (COCO→fall) | single-frame detector | 25.3M | 640 | COCO | Apache-2.0 | ~85–90% acc, ~15–20 ms @18 TOPS | Pure CNN, excellent INT8; current ROADMAP choice |
| timm CNN | EfficientNetV2-S (timm `efficientnetv2_rw_s.ra2_in1k`) | single-frame classifier on crops | 24M | 224–288 | ImageNet-1k | Apache-2.0 | 83.8% IN-1k top-1 | Pure CNN, near-lossless INT8; pair with person detector |
| timm CNN | MobileNetV4-Conv-Small | single-frame classifier on crops | 9.7M | 224 | ImageNet-1k | Apache-2.0 | 73.8% IN-1k | Mobile-grade, trivial INT8, <5 ms |
| timm ConvNeXt | ConvNeXt-V2 Tiny (`convnextv2_tiny.fcmae_ft_in22k_in1k`) | single-frame classifier | 28.6M | 224 | IN-22k→IN-1k | CC-BY-NC-4.0 (flag) | 83.9% IN-1k | Good INT8; **non-commercial — flag** |
| Self-supervised ViT | DINOv2 ViT-S/14 (`facebook/dinov2-small`) | frozen backbone + linear/MLP probe on crops | 22.1M | 224 | LVD-142M SSL | Apache-2.0 | Strong linear-probe transfer | INT8 OK for small ViT; attention slower than CNN |
| Self-supervised ViT | DINOv3 ViT-S/16 (`facebook/dinov3-vits16-pretrain-lvd1689m`) | frozen backbone + probe | 21.6M | 224 | LVD-1689M SSL | **"dinov3-license" (non-commercial research)** — flag | Best-in-class SSL features 2025 | Gated; **license blocks commercial use** |
| Supervised ViT | EVA-02-Small (`timm/eva02_small_patch14_224.mim_in22k`) | single-frame classifier | 22M | 224 | MIM IN-22k | MIT | ~85% IN-1k after FT | INT8 workable but attention ops need calibration |
| Video Transformer | VideoMAE-Small K400 (`MCG-NJU/videomae-small-finetuned-kinetics`) | 16-frame clip classifier | 22M | 16×224 | K400 | **CC-BY-NC-4.0** — flag | 79.0% K400 top-1 | Sliding-window ONNX needed; attention hurts INT8 |
| Video Transformer | VideoMAE-Base K400 (`MCG-NJU/videomae-base-finetuned-kinetics`) | 16-frame clip classifier | 86.5M | 16×224 | K400 | **CC-BY-NC-4.0** — flag | 81.5% K400 top-1 | Too heavy for 10 FPS at 18 TOPS unless strided |
| Video Transformer | VideoMAE-V2 (paper / OpenGVLab variants) | clip classifier | 86M–1B | 16×224 | UnlabeledHybrid→K710 | CC-BY-NC-4.0 (most ckpts) — flag | SOTA on K400/K600/K700 (~88% K400) | Non-commercial ckpts; paper arXiv:2303.16727 |
| Video CNN | X3D-M / X3D-S (PyTorchVideo / PySlowFast) | clip classifier | 3.8M / 3.3M | 16×224 | K400 | Apache-2.0 | 76.0 / 73.1% K400 | Pure 3D CNN, INT8-friendly, real-time at 18 TOPS |
| Video CNN | MoViNet-A2-Stream (TF Hub / torch ports) | streaming clip classifier | 4.8M | 224, streaming | K600 | Apache-2.0 | 78.6% K600 | Stream-friendly (causal 3D conv), fits edge |
| Temporal 2D | TSM ResNet-50 (MIT-HAN-Lab) | 8–16 frame clip classifier | 24M | 224 | K400/Sth-Sth | Apache-2.0 | 74.7% K400 | 2D-CNN + temporal-shift → excellent INT8 |
| Hybrid | UniFormerV2-Base K710→K400 (Sense-X) | clip classifier | 115M | 16×224 | K710 | **MIT code, CC-BY-NC-4.0 weights** — flag | ~89% K400 | Attention-heavy; edge only with distillation |
| Keypoint temporal | PoseC3D / ST-GCN++ on RTMPose kpts | skeleton clip classifier | 2–3M | T×V×C | NTU-RGBD / K400 | Apache-2.0 (mmaction2) | 94%+ on NTU fall subset | Very light; requires pose pipeline (see fall_pose_estimation doc) |

## 3. Top 3 recommendations

1. **Keep YOLOX-M as the first-stage person / fallen-person detector, add an EfficientNetV2-S (or MobileNetV4-Conv-Small) posture classifier on the person crop.** This augments the baseline rather than replacing it: YOLOX-M handles recall and localization; a dedicated 224-crop classifier trained on sitting/crouching/kneeling/lying/fallen hard negatives directly attacks the documented false-positive mode. Both are Apache-2.0, pure CNN, trivially INT8, well under the 18 TOPS envelope (expected ≥30 FPS end-to-end).
2. **Temporal augmentation: X3D-M or TSM-R50 clip head over ByteTrack-tracked person tubes (16 frames ≈ 0.5 s @30 FPS).** Both are Apache-2.0, INT8-friendly pure/near-pure CNNs. Sliding-window ONNX export is straightforward. Addresses the "single-frame ambiguity" and "lying-to-work" failure modes. X3D-M is the preferred option at 3.8M params — fits alongside YOLOX-M with budget to spare. MoViNet-A2-Stream is a secondary choice because its causal streaming design amortises cost across frames.
3. **DINOv2-Small frozen backbone + linear/MLP probe on person crops as a research/ablation head.** Apache-2.0, 22M params, very strong transfer with small labelled data — useful given the 111 keypoint-annotated factory images and modest 17K classification set. Also serves as a sanity check against EfficientNetV2-S.

Models explicitly **not** recommended for production due to licensing: DINOv3, VideoMAE/VideoMAE-V2 public checkpoints, UniFormerV2 K400 weights, ConvNeXt-V2 (FCMAE pretrain). All are CC-BY-NC or research-only. They can be used for internal benchmarking only.

## 4. Pretrained weights

Download destination: `ai/pretrained/safety-fall-detection/`

| Model | URL | License | Size | SHA256 | Local dest |
|---|---|---|---|---|---|
| EfficientNetV2-S (timm) | https://huggingface.co/timm/efficientnetv2_rw_s.ra2_in1k/resolve/main/pytorch_model.bin | Apache-2.0 | 96,674,637 B (≈92.2 MB) | `8bb2555726585abc07991848cfa0de5732cff2d8bf9f88d6317a687dbdc1b303` | `ai/pretrained/safety-fall-detection/efficientnetv2_rw_s.ra2_in1k.bin` |
| VideoMAE-Small K400 | https://huggingface.co/MCG-NJU/videomae-small-finetuned-kinetics/resolve/main/pytorch_model.bin | **CC-BY-NC-4.0 (flag, research-only)** | 88,197,173 B (≈84.1 MB) | `d69585904fec7e507bf2edfba4a7abe2b92def9afc76460d1dc14dbbf864bcfc` | `ai/pretrained/safety-fall-detection/videomae-small-finetuned-kinetics.bin` |
| MobileNetV4-Conv-Small (timm) | https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k/resolve/main/pytorch_model.bin | Apache-2.0 | ~39 MB | not downloaded | `ai/pretrained/safety-fall-detection/mobilenetv4_conv_small.e2400_r224_in1k.bin` |
| DINOv2-Small | https://huggingface.co/facebook/dinov2-small/resolve/main/pytorch_model.bin | Apache-2.0 | ~88 MB | not downloaded | `ai/pretrained/safety-fall-detection/dinov2-small.bin` |
| X3D-M (PyTorchVideo) | https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_M.pyth | Apache-2.0 | ~12 MB | not downloaded | `ai/pretrained/safety-fall-detection/x3d_m.pyth` |
| DINOv3 ViT-S/16 | https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m | **Gated + DINOv3 license (non-commercial)** | n/a | gated | — do not deploy |

All HF `resolve/main` URLs verified with `curl -I` → HTTP 302 to CDN (see worker log). The two downloaded files (`efficientnetv2_rw_s.ra2_in1k.bin`, `videomae-small-finetuned-kinetics.bin`) are on-disk.

## 5. Edge deployment notes

- **Budget at ~18 INT8 TOPS, chip-agnostic, ONNX + generic INT8.** YOLOX-M baseline already consumes ~1–2 TOPS-equivalent per frame at 640. A secondary 224-crop classifier (EfficientNetV2-S, ~8 GFLOPs) on 1–5 tracked persons per frame costs another ~1 TOPS. Plenty of headroom.
- **Temporal models need sliding-window ONNX.** VideoMAE / X3D / TSM export as `[B, C, T, H, W]` with T=16. Strategies: (a) recompute full clip every K frames (K=8 gives 4 Hz decisions, combine with tracker for smoothing), (b) MoViNet-style causal 3D with state caches exported as ONNX inputs/outputs, (c) TSM's temporal-shift is stateless per frame — export as 2D CNN with a ring buffer in postprocessing (best INT8 story).
- **Attention quantisation caveats.** ViT / VideoMAE / UniFormer softmax and LayerNorm need per-channel INT8 calibration or FP16 fallback on attention blocks. Pure-CNN candidates (EfficientNetV2, MobileNetV4, X3D, TSM, MoViNet) have no such issue.
- **Person-crop classifier path is INT8-trivial** and also dataset-efficient: factory sitting/crouching hard-negative mining maps directly onto a 5-class posture head (standing / sitting / crouching / lying-working / fallen) without re-annotating detection boxes.
- **Streaming vs clip.** For the 500 ms (15-frame) confirmation window already specified in the platform doc, X3D-M on a 16-frame sliding window aligns exactly; the temporal classifier's own hysteresis replaces (or reinforces) the ByteTrack frame-counter.

## 6. Datasets for fine-tune

| Dataset | Modality | Size | License | Use |
|---|---|---|---|---|
| UR Fall Detection (URFall) | RGB+Depth video | 70 seqs (30 falls, 40 ADL) | Academic (research) | Temporal benchmark, cross-subject eval |
| Le2i Fall Detection | RGB video | ~130 seqs, 4 scenarios | Academic (request Univ. Burgundy) | Cross-scene generalisation |
| Multicam Fall (Montreal) | RGB multi-view | 24 scenarios × 8 cams | Academic | View-invariance |
| UP-Fall Detection | RGB+IMU+EEG | 11 activities, 17 subjects | Academic | Hard negatives (lying, sitting) |
| FallAllD | IMU-only | 26 subjects | Academic | Not directly usable (no RGB), ref only |
| MPDD Fall Dataset | RGB images | 1,200+ multi-person | Academic (Nature SciData 2025) | Multi-person hard cases |
| NTU-RGBD 60 / 120 | RGB+D+skeleton | 114k clips | Academic (non-commercial) | Skeleton-based temporal pretrain (PoseC3D / ST-GCN++). **Flag non-commercial.** |
| Kinetics-400/600/700 | RGB video | 300k+ clips | CC-BY-4.0 annotations; YouTube videos | Temporal backbone pretrain only |
| `g_fall_classify` (internal) | RGB images | 17K | Proprietary | Primary training set |

Recommended mix for v2: internal 17K + hard-negative top-up (1.8–2.7k factory sitting/crouching/lying from the platform doc) for the classifier head; URFall + Le2i + UP-Fall for the temporal head; NTU-fall subset only for ablation (non-commercial).

## 7. Verdict vs ROADMAP baseline

**Augment, don't replace.** YOLOX-M stays as the Phase-1 detector — it is Apache-2.0, fits the budget, has an existing 17K training set, and the platform doc's acceptance metrics are achievable with it plus temporal filtering. The productive 2025/2026 upgrades are orthogonal second stages, not backbone swaps:

- **Phase 1 (now):** Ship YOLOX-M as-is with 15-frame ByteTrack hysteresis.
- **Phase 1.5 (low risk, high FP reduction):** Add an **EfficientNetV2-S** (or MobileNetV4-Conv-Small for tighter budgets) posture classifier on YOLOX person crops, trained with the 1.8–2.7k hard-negative augmentation set. Pure CNN, Apache-2.0, fits comfortably. Directly attacks sitting/crouching false positives — the dominant failure mode.
- **Phase 2 (accuracy push):** Add an **X3D-M** (Apache-2.0) temporal head over ByteTrack person tubes for the last 16 frames (≈0.5 s). Expected lift: +3–5 pts recall on transient / ambiguous falls, matches the 91% F1 "3D-CNN + LSTM" reference in the platform doc's industry benchmarks.
- **Do not adopt** VideoMAE-V2, UniFormerV2 K400 weights, ConvNeXt-V2, or DINOv3 for production: all carry CC-BY-NC or custom non-commercial licenses that conflict with the "commercially usable" constraint. They remain valid for internal benchmarking.
- **DINOv2 (Apache-2.0)** is the only 2025-grade SSL ViT that is licence-clean; keep it as a research head to validate the supervised EfficientNetV2-S classifier.

Net: the ROADMAP YOLOX-M choice is correct. The gap is not at the detector — it is at the posture disambiguation and temporal-context stages, which should be filled with Apache-2.0 CNN heads (EfficientNetV2-S → X3D-M), not by swapping the backbone.

## 8. References

- YOLOX: Ge et al., arXiv:2107.08430. Code: https://github.com/Megvii-BaseDetection/YOLOX (Apache-2.0).
- EfficientNetV2: Tan & Le, arXiv:2104.00298. timm: https://huggingface.co/timm/efficientnetv2_rw_s.ra2_in1k (Apache-2.0).
- MobileNetV4: Qin et al., arXiv:2404.10518. timm: https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k (Apache-2.0).
- DINOv2: Oquab et al., arXiv:2304.07193. https://huggingface.co/facebook/dinov2-small (Apache-2.0).
- DINOv3: arXiv:2508.10104. https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m (gated, non-commercial — flagged).
- EVA-02: Fang et al., arXiv:2303.11331 (MIT).
- VideoMAE: Tong et al., arXiv:2203.12602. https://huggingface.co/MCG-NJU/videomae-small-finetuned-kinetics (CC-BY-NC-4.0 — flagged).
- VideoMAE-V2: Wang et al., arXiv:2303.16727 (research-only checkpoints — flagged).
- UniFormerV2: Li et al., arXiv:2211.09552 (code MIT, K400 weights CC-BY-NC-4.0 — flagged).
- X3D: Feichtenhofer, CVPR 2020, PyTorchVideo model zoo: https://pytorchvideo.readthedocs.io/en/latest/model_zoo.html (Apache-2.0).
- MoViNet: Kondratyuk et al., arXiv:2103.11511 (Apache-2.0).
- TSM: Lin et al., arXiv:1811.08383, https://github.com/mit-han-lab/temporal-shift-module (Apache-2.0).
- PoseC3D / ST-GCN++: Duan et al., arXiv:2104.13586; mmaction2 Apache-2.0.
- PapersWithCode Fall Detection: https://paperswithcode.com/task/fall-detection.
- URFall: http://fenix.ur.edu.pl/~mkepski/ds/uf.html. Le2i: http://le2i.cnrs.fr/Fall-detection-Dataset. MPDD: Nature Scientific Data 2025.
- Platform doc: `ai/docs/03_platform/safety-fall-detection.md`. Companion pose approach: `ai/docs/03_platform/safety-fall_pose_estimation.md`.

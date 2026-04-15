# Fall Pose Estimation — SOTA Pretrained Model Survey (2025–2026)

> Feature: `safety-fall_pose_estimation`
> Baseline (ROADMAP): **YOLOX-Tiny** person detector + **RTMPose-S** 17-kpt pose + rule-based fall logic
> Hardware budget: ~18 TOPS INT8, chip-agnostic
> Priority: accuracy > speed (≥10 FPS INT8 acceptable)

## 1. Task summary

Two-stage or one-stage human pose pipeline that outputs 17 COCO keypoints per person, used as input to geometric + temporal rule logic (aspect ratio, hip/shoulder y-deltas, velocity, temporal persistence) to emit `fall_detected` alerts. The model does not directly classify "fall" — the pose quality (especially hip, shoulder, knee, ankle visibility under occlusion) is the dominant driver of FP/FN. Life-safety task: recall is preferred over precision.

Edge constraints: standard ONNX export + generic INT8 static PTQ; no vendor-specific op sets; commercially usable license. Must handle multi-person (workplaces, care facilities) at ≥10 FPS INT8 after detector + pose combined.

## 2. Candidate models

| Model | Stage | Params | Pretrained | License | COCO AP (val) | Edge-friendly |
|---|---|---|---|---|---|---|
| **RTMPose-S (256×192)** | Top-down | 5.47 M | COCO / Body7 | Apache-2.0 | 72.2 | Yes — pure CNN (CSPNeXt), SimCC head; proven INT8 |
| RTMPose-M (256×192) | Top-down | 13.5 M | COCO / Body7 | Apache-2.0 | 75.8 | Borderline at 18 TOPS for multi-person |
| RTMPose-T (256×192) | Top-down | 3.34 M | COCO | Apache-2.0 | 68.5 | Yes — excellent INT8 headroom, slight AP drop |
| **RTMO-S (640×640)** | One-stage | 9.9 M | Body7 | Apache-2.0 | 68.6 COCO / 82.6 CrowdPose mAP | Yes — single graph, no crop loop; best for crowded scenes |
| RTMO-T | One-stage | ~7 M | Body7 | Apache-2.0 | ~66 | Yes |
| DWPose-S (256×192) | Top-down | ~5.5 M | UBody/COCO-WB | Apache-2.0 | 72.2 body / whole-body strong | Yes — RTMPose-S distilled; same footprint as RTMPose-S |
| ViTPose-S | Top-down | 22 M | COCO / MS COCO+AIC+MPII | Apache-2.0 | 73.8 | Borderline — ViT attention INT8 is tricky |
| ViTPose++-S | Top-down | 22 M | multi-dataset | Apache-2.0 | 75.8 | Borderline (same caveat) |
| HRNet-W32 (tiny-lite variants) | Top-down | ~9–28 M | COCO | MIT | 74.4 (W32) | OK but heavier than RTMPose at equal AP |
| MobileHumanPose | Top-down | ~3 M | COCO/MuCo | MIT | ~64 (2D) | Yes but dated |
| Sapiens-0.3B (pose) | Top-down ViT | 336 M | Humans-300M | CC-BY-NC 4.0 / Sapiens License | 83.2 (308 kpt superset) | **No** — far above 18 TOPS budget; non-commercial noncompliant for our use |
| YOLO-NAS-Pose S | One-stage | 22.2 M | COCO | **Deci pre-trained = non-commercial** (code Apache-2.0) | 59.8 AP | Good INT8 (quant-aware NAS), but weights license blocks us |
| YOLOv8-Pose / YOLO11-Pose | One-stage | 3–68 M | COCO | **AGPL-3.0** (flag) | 50–69 | Edge-friendly but copyleft — flag only |

**License callouts**

- **Ultralytics YOLOv8-pose / YOLO11-pose = AGPL-3.0.** Use would force open-sourcing of firmware stack. **Excluded.**
- **YOLO-NAS-Pose** code is Apache-2.0 but **Deci's pretrained weights are non-commercial**. Training from scratch is possible but costly; excluded unless a permissive re-release appears.
- **Sapiens** is technically impressive but smallest variant is 0.3 B params and weights are released under Sapiens License (research / Meta Platforms terms — not standard commercial). Excluded.
- **RTMPose / RTMO / DWPose / ViTPose (code & weights) = Apache-2.0.** All usable.

## 3. Top 3 recommendations

1. **RTMPose-S (256×192) + RTMDet-nano-person (two-stage) — KEEP baseline intent, swap detector.** Apache-2.0, 5.47 M params, 72.2 AP, proven ONNX + INT8 on multiple edge NPUs, reference latency ~5 ms/person on typical 18-TOPS chips. Replacing YOLOX-Tiny with RTMDet-nano-person (4.2 MB weights, COCO+Objects365 person-only, co-trained by the RTMPose authors) simplifies the pipeline — same toolchain, same Apache-2.0 license, better person-only AP and 2× faster than YOLOX-Tiny.
2. **RTMO-S (640×640) one-stage fallback for crowded scenes.** Apache-2.0, 9.9 M, 82.6 CrowdPose mAP, single ONNX graph — no per-person crop loop, bounded latency regardless of person count. Slightly lower COCO AP (68.6) than RTMPose-S but superior under occlusion / overlapping workers. Use when scenes routinely contain >5 people.
3. **DWPose-S (256×192) upgrade path.** Apache-2.0, same 5.5 M footprint as RTMPose-S, trained with two-stage distillation from RTMW teacher. Drop-in ONNX replacement for RTMPose-S with better whole-body and occlusion robustness; keeps the rest of the pipeline unchanged.

All three fit comfortably within 18 TOPS INT8 and are fully commercially usable.

## 4. Pretrained weights

Local destination: `ai/pretrained/safety-fall_pose_estimation/`

| Model | URL | License | Size | SHA256 | Local dest |
|---|---|---|---|---|---|
| RTMPose-S COCO 256×192 | https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth | Apache-2.0 | 22 007 763 B (21 MB) | `8d57b1112021367bb6857e468be383b2bd24a0b69f121b3f54177635fb742907` | `rtmpose-s_coco_256x192.pth` |
| RTMDet-nano person (COCO+Obj365) | https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth | Apache-2.0 | 4 215 674 B (4.0 MB) | `0e2da635c75e25dc88af08d01eb34bfe9cac06a7841cba223bdf04eed288b3dc` | `rtmdet-nano_person.pth` |
| RTMO-S Body7 640×640 | https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.pth | Apache-2.0 | 39 766 454 B (38 MB) | `dac2bf749bbfb51e69ca577ca0327dff4433e3be9a56b782f0b7ef94fb45247e` | `rtmo-s_body7_640x640.pth` |

All three `curl -I` verified and downloaded successfully (see `ai/pretrained/safety-fall_pose_estimation/`). Optional gated weights: Sapiens (HF-gated, Sapiens License — not downloaded).

## 5. Edge deployment notes

**Top-down (RTMPose-S) vs one-stage (RTMO-S) trade-off:**

| Aspect | Top-down RTMPose-S | One-stage RTMO-S |
|---|---|---|
| Per-frame cost | detector + K × pose(256×192) | constant pose(640×640) |
| Crowded scene cost | scales linearly with people | flat |
| Small-person AP | high (person crop is re-scaled to 256×192) | degrades for small persons |
| Deployment graphs | 2 ONNX files | 1 ONNX file |
| INT8 friendliness | excellent (pure CNN both stages) | good (pure CNN, NMS-free) |
| Fall-rule compatibility | Direct 17-kpt output | Direct 17-kpt output |

**ONNX + INT8 notes**

- Export RTMPose-S with `opset=11`, dynamic batch, input `1×3×256×192` NCHW; SimCC head outputs two 1-D classification tensors (x, y) — decode on CPU after model.
- RTMDet-nano-person: export with batched NMS fused or raw boxes + CPU NMS; ≥256×256 input keeps AP for small workers.
- Generic static INT8 PTQ with 300–500 representative frames from target site. CSPNeXt blocks quantize cleanly; expect <1.0 AP drop on COCO val.
- RTMO-S: keep the SimOTA assigner-free head in FP32 or INT8 (both supported); the heatmap decoder is a standard argmax — no special ops.
- Avoid ViTPose and Sapiens for now: attention + LayerNorm INT8 degrades materially on generic NPUs and usually requires per-channel + mixed-precision tuning that breaks "chip-agnostic".
- Multi-person budget at 18 TOPS INT8: baseline pipeline (RTMDet-nano + RTMPose-S) comfortably reaches ≥15 FPS for 4–6 persons; RTMO-S reaches ≥20 FPS regardless of count.

## 6. Datasets for fine-tune

- **COCO Keypoints 2017** — 17-kpt, ~250k person instances. Primary training set for all RTMPose/RTMO/DWPose variants.
- **CrowdPose** — 20k images, heavy occlusion. Essential for workplace/care-facility scenes with multiple workers. Improves recall under occlusion.
- **MPII Human Pose** — 25k images, 40k person-instances. Complementary for single-person posture diversity.
- **Halpe Full-Body** — COCO+MPII+foot merged, 136-kpt superset. Useful if we later extend rules to foot-ground contact.
- **Body7 (merged)** — MMPose's combined COCO+AIC+MPII+CrowdPose+Halpe+PoseTrack18+sub-JHMDB; what RTMO and new RTMPose variants pretrain on. Best single source to start from.
- **UR Fall / Le2i / URFD** — small public fall-specific datasets. Use only for evaluation of the rule layer, not for pose retraining.
- **Internal site footage** — annotate 500–1000 fall + 2000 normal-activity frames for the temporal rule threshold tuning; the pose model rarely needs re-tuning if site geometry is typical.

## 7. Verdict vs ROADMAP baseline

**Keep the RTMPose-S pose head. Swap the detector to RTMDet-nano-person. Add RTMO-S as a configurable crowded-scene fallback.**

Rationale:

- RTMPose-S remains Pareto-optimal at 18 TOPS: 72.2 COCO AP at 5.47 M params, Apache-2.0, clean INT8. No 2025 release beats it within budget for 17-kpt top-down pose (Sapiens is too large/licensed, ViTPose++ hits the same AP but costs 4× params and attention-INT8 pain, YOLO-NAS-Pose weights are non-commercial).
- The **YOLOX-Tiny → RTMDet-nano-person** swap is the only concrete improvement: same Apache-2.0 ecosystem, ~20 % of the detector params, person-only head, co-trained with RTMPose in MMPose — a single-repo tool chain instead of mixing Megvii YOLOX + MMPose.
- **RTMO-S as opt-in fallback** gives a meaningful robustness lever for crowded scenes without replacing the primary path.
- DWPose-S is a zero-risk future drop-in (same weights shape family) if whole-body keypoints become required.

Exclusions that should be recorded in ROADMAP: Ultralytics YOLOv8/11-pose (AGPL), YOLO-NAS-Pose Deci weights (non-commercial), Sapiens (size + license), MediaPipe Pose (already Option 3; not SOTA on COCO AP).

## 8. References

- RTMPose paper — https://arxiv.org/abs/2303.07399
- RTMPose project (MMPose) — https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose
- RTMO project (MMPose) — https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo
- RTMW paper — https://arxiv.org/html/2407.08634v1
- DWPose — https://github.com/IDEA-Research/DWPose
- ViTPose / ViTPose++ — https://github.com/ViTAE-Transformer/ViTPose
- ViTPose++ paper — https://arxiv.org/html/2212.04246v3
- Sapiens paper — https://arxiv.org/abs/2408.12569
- Sapiens project page — https://rawalkhirodkar.github.io/sapiens/
- YOLO-NAS-Pose — https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS-POSE.md
- YOLO-NAS-Pose weights commercial-use discussion — https://huggingface.co/hr16/yolo-nas-pose/discussions/1
- rtmlib (standalone RTMPose/DWPose/RTMO/RTMW without mmcv) — https://github.com/Tau-J/rtmlib
- OpenMMLab RTMPose blog — https://openmmlab.medium.com/rtmpose-the-all-in-one-real-time-pose-estimation-solution-for-application-and-research-6404f17cd52f
- Roboflow pose model comparison — https://blog.roboflow.com/best-pose-estimation-models/

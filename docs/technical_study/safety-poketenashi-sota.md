# Poketenashi Violations — SOTA Pretrained Model Survey (2025–2026)

Target platform: generic ~18 TOPS INT8 edge accelerator, chip-agnostic, standard ONNX, commercially usable licenses only. Accuracy > speed, ≥10 FPS INT8 acceptable.

Baseline (see `ai/docs/03_platform/safety-poketenashi.md`): YOLOX-Tiny (person) + RTMPose-S (17-kpt COCO pose) + YOLOX-Nano/M (phone) + CPU rule engine. Phase-1 rules: `phone_usage`, `hands_in_pockets`, `no_handrail`, `unsafe_stair_crossing` (+ Gap G3 `yubisashi` pointing-and-calling).

---

## 1. Task summary

| Sub-rule | Signal required | Baseline logic |
|---|---|---|
| hands_in_pockets | wrist + hip keypoints, wrist confidence (occlusion proxy); ideally finger/hand keypoints | wrist-near-hip ∧ low wrist conf, sustained ≥30f |
| phone_usage | phone bbox + wrist/ear keypoints + trajectory | phone overlaps hand/ear region while walking, ≥15f |
| no_handrail | wrist kpt vs handrail polygon, stair-zone polygon | no wrist within proximity of handrail polyline, ≥45f |
| unsafe_stair_crossing | person centroid + trajectory angle vs stair direction | angle >30°, ≥15f |
| yubisashi (Gap G3) | shoulder-wrist angle, head/nose orientation | arm horizontal + gaze aligned with wrist |

Key observation: three of five rules are **pose-driven**. Hand-in-pocket and point-and-call benefit strongly from **wholebody keypoints** (hands + face), which the current 17-kpt RTMPose-S baseline lacks. Temporal-skeleton classifiers are a second-stage option if rules plateau.

---

## 2. Candidate models

### 2a. Pose backbones (2025)

| Model | Params | Keypoints | COCO-WB AP | License | INT8 ONNX practical? | Notes |
|---|---|---|---|---|---|---|
| **RTMPose-S (body17)** | 5.47 M | 17 | 72.2 AP (COCO-Pose) | Apache-2.0 | Yes | Current baseline. No hands/face. |
| **RTMPose-M / L (COCO-WholeBody)** | 13 M / 27 M | 133 | 59.1 / 64.8 | Apache-2.0 | Yes | Drop-in upgrade; 133 kpts cover hands+face. |
| **DWPose-L (distilled RTMPose-L WB)** | ~27 M | 133 | 66.5 | Apache-2.0 | Yes (official ONNX) | Best accuracy/FLOP tradeoff in the RTM family; widely deployed. |
| **RTMW-m / l (cocktail14)** | 13.3 / 28.1 M | 133 | 67.2 / 70.1 | Apache-2.0 | Yes (official ONNX SDK zips) | 2024 refresh; larger training cocktail, best open WB AP under Apache. |
| **RTMO-m** | 22.6 M | 17 | 68.6 AP (COCO) | Apache-2.0 | Yes | One-stage (no person detector) — simplifies pipeline but body-only. |
| **ViTPose++ Small (body+WB MoE)** | 25 M | 17 or 133 | 75.8 (body), ~65 (WB) | Apache-2.0 | Possible (custom exporter); heavier at 384² | Higher AP than RTMPose, but 3–5× latency at same size. |
| **Sapiens-0.3B Pose** | 336 M | 308 (goliath) | SOTA | CC-BY-NC 4.0 | **FLAG — non-commercial** | Excluded from recommendation set. |
| **MediaPipe Pose Heavy / Hands** | ~6 M + 1.8 M | 33 + 21/hand | n/a | Apache-2.0 | Yes (TFLite/ONNX) | Single-person; useful as CPU sidecar for finger detail. |

### 2b. Skeleton-action models (temporal, second stage)

Window typically 30–60 frames × 17 or 133 kpts; used for `hands_in_pockets`/`yubisashi` if geometric rules underperform.

| Model | Params | NTU-RGBD 120 X-Sub | License | ONNX | Notes |
|---|---|---|---|---|---|
| **ST-GCN++ (PYSKL)** | 1.4 M | 85.6 | Apache-2.0 | Yes (opset≥12, einsum rewrite) | Cheapest; great edge fit. |
| **PoseC3D (SlowOnly-R50, kpt heatmap)** | 2.0 M (3D heatmap head) | 86.9 | Apache-2.0 | Yes | 3D heatmap representation, more robust to pose noise than GCN. |
| **HyperFormer** | ~2 M | 87.2 | Apache-2.0 | Yes | 2024 transformer-GCN hybrid. |
| **MotionBERT (MB-Lite, action head)** | 16 M | 87.4 (downstream) | Apache-2.0 | Yes | Transformer, heavier; strong representation transfer. |

### 2c. End-to-end video/behaviour models

| Model | License | Verdict |
|---|---|---|
| VideoMAE v2 | CC-BY-NC 4.0 | **FLAG — non-commercial**, excluded. |
| InternVideo2 | Apache-2.0 code, weights mixed (some non-commercial) | Too large (>300M) and zero-shot action on factory video not reliable enough; excluded for on-device. |
| X3D-XS (from baseline Option 2) | Apache-2.0 | Viable but requires ~3K labelled clips that do not exist — keep as Phase-2 fallback. |
| SlowFast | Apache-2.0 | Same data gap as X3D-XS. |

---

## 3. Top 3 recommendations

### Recommendation A (preferred): **YOLOX-Tiny + DWPose-L (133-kpt wholebody) + CPU rule engine**

- Upgrades the pose stage from 17 → 133 keypoints (body 17 + hands 2×21 + face 68 + feet 6).
- Directly resolves the biggest accuracy ceiling in the baseline: **hand-in-pocket** can now use **palm + finger keypoint confidence** (not just wrist proximity + low-confidence heuristic), and **yubisashi** can use index-fingertip orientation instead of wrist-only vectors.
- `no_handrail` benefits from finger-tip proximity to handrail polyline instead of wrist-only.
- DWPose-L at 384×288 INT8 is ~8–12 ms on 18 TOPS-class NPUs; with frame skipping (pose every 2nd frame) the full pipeline fits in a 10–15 FPS budget alongside YOLOX-Tiny + YOLOX-M phone.
- Keeps baseline person detector and phone detector unchanged — minimal migration.
- All Apache-2.0.

### Recommendation B: **YOLOX-Tiny + RTMW-m (wholebody, lighter) + CPU rules + optional PoseC3D head**

- RTMW-m is the newest (2024) wholebody model in the RTM family under Apache-2.0, with cleaner training data than DWPose. At 13 M params it is ~2× cheaper than DWPose-L with ~2 AP drop on WB.
- Add PoseC3D only as a Phase-2 stage if rule calibration on 200–400 factory clips still yields >10% FP on hands-in-pockets. PoseC3D consumes 48-frame × 17-kpt heatmap stacks (~2 M params), runs ≥20 FPS INT8.
- Best if NPU budget is tight (e.g. when running alongside a second feature such as Zone Intrusion on the same chip).

### Recommendation C (bold swap): **RTMO-m (one-stage) + DWPose-L crop-pose + rules**

- RTMO removes YOLOX-Tiny: one-stage 17-kpt multi-person pose directly on the full frame (68.6 AP COCO). For trajectory / zone-crossing rules this is sufficient.
- DWPose-L is then run **only on the persons already inside stair/handrail zones or those whose wrist keypoint is near hip** — an N-reduction trick that typically cuts wholebody calls by 3–5×.
- Net: fewer model hops, higher accuracy on detection-pose interface, and the same wholebody rule quality as (A) where it matters.
- Higher implementation cost: two-path pose scheduling logic.

All three drop Sapiens and VideoMAE v2 on license grounds.

---

## 4. Pretrained weights

Local destination root: `ai/pretrained/safety-poketenashi/`

| File | URL | License | Size | SHA256 | Local dest | Status |
|---|---|---|---|---|---|---|
| `dw-ll_ucoco_384.onnx` (DWPose-L WB, 384×288, ONNX) | https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx | Apache-2.0 | 134.4 MB | `724f4ff2439ed61afb86fb8a1951ec39c6220682803b4a8bd4f598cd913b1843` | `ai/pretrained/safety-poketenashi/dw-ll_ucoco_384.onnx` | Downloaded |
| `rtmpose-m_coco-wholebody_256x192.pth` (RTMPose-M WB, reference PyTorch ckpt) | https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth | Apache-2.0 | 72.0 MB | `3da02694cd6479d3b333ff42ebd0723f96bfa06adac1db1e2e815ed2e9e1b02d` | `ai/pretrained/safety-poketenashi/rtmpose-s_coco-wholebody.pth` | Downloaded (reference; for fine-tune comparison) |
| RTMW-L WB ONNX SDK (256×192) | https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip | Apache-2.0 | 212.7 MB | n/a (not downloaded) | `ai/pretrained/safety-poketenashi/rtmw-l_256x192.zip` | HEAD 200 verified, not pulled (size) |
| RTMW-L WB ONNX SDK (384×288) | https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip | Apache-2.0 | ~220 MB | n/a | `ai/pretrained/safety-poketenashi/rtmw-l_384x288.zip` | HEAD 200 verified |
| PYSKL ST-GCN++ / PoseC3D weights | https://github.com/kennymckormick/pyskl (NTU60/NTU120 model zoo) | Apache-2.0 | 5–20 MB per ckpt | download-time | `ai/pretrained/safety-poketenashi/stgcnpp_ntu120.pth` | Pull only if Phase-2 action head is triggered |
| MotionBERT (action head) | https://github.com/Walter0807/MotionBERT | Apache-2.0 (code); weights gated via OneDrive | ~65 MB | n/a | — | **FLAG: manual download from OneDrive, verify redistribution** |
| Sapiens-Pose | https://huggingface.co/facebook/sapiens-pose-0.3b | **CC-BY-NC 4.0** | — | — | — | **EXCLUDED — non-commercial** |
| VideoMAE v2 | https://github.com/OpenGVLab/VideoMAEv2 | **CC-BY-NC 4.0** | — | — | — | **EXCLUDED — non-commercial** |

`curl -I` checks were performed for DWPose, RTMPose-M WB, and RTMW-L ONNX SDKs — all return 200. An earlier guess of `rtmpose-s_…coco-wholebody_pt-aic-coco_270e-256x192-9c07c1b1_…` returned 404 (wrong filename); RTMPose-M is the smallest official WB checkpoint on OpenMMLab.

---

## 5. Edge deployment notes

- **Wholebody cost vs body-only.** DWPose-L @ 384×288 ≈ 2.6 GFLOPs/person, INT8 ≈ 8–12 ms on 18 TOPS-class NPUs. Same model @ 256×192 ≈ 1.2 GFLOPs ≈ 4–6 ms. RTMW-m @ 256×192 ≈ 0.7 GFLOPs ≈ 3 ms. Pick 256×192 for ≥3 persons/frame, 384×288 if ≤2 persons/frame.
- **Per-person budget.** At 30 FPS with ~3 tracked persons, budget for pose = 33 ms × 0.6 = 20 ms → RTMW-m at 256×192 or DWPose-L every 2nd frame both fit.
- **Skeleton-action temporal window on device.** ST-GCN++ needs a ring buffer of 48 × 133 × 3 floats ≈ 76 KB per tracked person — trivial. PoseC3D needs 48 × 17 × 56 × 56 heatmap tensor ≈ 10 MB per person; only viable with ≤2 concurrent persons or frame-skip to 16 frames. Prefer ST-GCN++ or HyperFormer on edge.
- **Quantization gotchas.** SimCC head in RTMPose/DWPose/RTMW quantizes cleanly; avoid per-tensor quant on the 1-D classifier heads (use per-channel). MotionBERT and ViTPose++ need calibration with ≥500 factory-distribution samples due to attention outliers.
- **Rules run on CPU; keypoint confidence threshold tuning** (e.g. drop wrist/hand keypoints below 0.3 confidence) is essential post-INT8 because confidence distributions shift.

---

## 6. Datasets for fine-tune

| Dataset | Use | License | Notes |
|---|---|---|---|
| COCO-WholeBody (v1.0) | primary — already in DWPose/RTMW/RTMPose-WB weights | CC-BY 4.0 (annotations), images per COCO | No fine-tune needed for baseline. |
| **Halpe-136** | optional — alt WB schema (26 body + 21+21 hand + 68 face) | research use, verify commercial on AlphaPose repo | Use if tighter hand coverage on low-res crops is needed. |
| UBody / H3WB | additional hand+face in upper-body crops | Academic | Useful for `yubisashi` gaze-hand alignment. |
| NTU-RGBD 120 | ST-GCN++ / PoseC3D backbone pretraining | Research only (restricted redistribution) | Use for pretrain, never ship weights fine-tuned on NTU without checking. |
| Kinetics-400 skeleton (PYSKL kpt) | alt action pretrain | CC-BY 4.0 | Usable commercially. |
| **Custom factory video** | rule threshold calibration + Phase-2 action fine-tune | internal | 200–400 hand-in-pocket clips (5–60 s each); label binary (violation/not) + per-frame keypoints from pose inference. |

Annotation load is unchanged from baseline — only `phone_usage` requires bbox labels. Wholebody upgrade is purely a model swap.

---

## 7. Verdict vs ROADMAP baseline

**Augment, do not replace.** Keep the two-stage topology (person detector → pose → CPU rules) exactly as the ROADMAP prescribes, but **swap RTMPose-S (17-kpt) for DWPose-L or RTMW-m (133-kpt wholebody)**. This is the single highest-leverage change for poketenashi because:

1. Three of five rules (hands-in-pockets, no-handrail, yubisashi) are bottlenecked by the absence of hand/finger keypoints in the 17-point schema; wholebody directly unlocks them.
2. License remains Apache-2.0, ONNX is first-class, INT8 cost stays within the 18 TOPS budget.
3. No dataset collection required — COCO-WholeBody pretrain is sufficient; only rule-threshold recalibration (already planned).

**Do not** adopt Sapiens (CC-BY-NC) or VideoMAE v2 (CC-BY-NC) — both fail the commercial license constraint.

**Defer** skeleton-action (ST-GCN++/PoseC3D) to Phase-2, gated on measured FP rate after wholebody rollout. The baseline's Option 2 action-recognition path (X3D-XS) is unnecessary if wholebody + rules meets the 0.85 precision / 0.82 recall target.

---

## 8. References

- RTMPose paper: https://arxiv.org/abs/2303.07399
- DWPose (ICCV 2023 CV4Metaverse): https://github.com/IDEA-Research/DWPose
- RTMW model zoo: https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose
- rtmlib (deploy-friendly RTMPose/DWPose/RTMO/RTMW): https://github.com/Tau-J/rtmlib
- ViTPose++: https://github.com/ViTAE-Transformer/ViTPose
- Sapiens (Meta, non-commercial): https://huggingface.co/facebook/sapiens-pose-0.3b
- PYSKL (ST-GCN++, PoseC3D): https://github.com/kennymckormick/pyskl
- MotionBERT: https://github.com/Walter0807/MotionBERT
- COCO-WholeBody: https://github.com/jin-s13/COCO-WholeBody
- Halpe-136: https://github.com/Fang-Haoshu/Halpe-FullBody
- FPI-Det (baseline phone dataset): https://github.com/KvCgRv/FPI-Det
- Platform baseline: `ai/docs/03_platform/safety-poketenashi.md`

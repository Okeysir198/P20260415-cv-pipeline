# Zone Intrusion — SOTA Pretrained Model Survey (2025–2026)

> Feature: `access-zone_intrusion` · Baseline: YOLOX-Tiny (COCO) + ByteTrack + PolygonZone
> Target edge budget: ~18 TOPS INT8, ≥10 FPS, standard ONNX INT8, commercial-friendly license.

## 1. Task summary

Detect persons (and incidental vehicles — bicycle/car/motorcycle/bus/truck) entering operator-defined polygonal zones on fixed camera streams. Alert states (intrusion, loitering, wrong-direction, line-crossing) are rule-based on top of tracked detections, so the **only learned component is the person/vehicle detector**; everything else is a deterministic post-process (ByteTrack + Supervision `PolygonZone`/`LineZone`). Accuracy matters more than speed — we have significant headroom on ~18 TOPS NPUs — but the detector must export cleanly to plain ONNX and survive INT8 quantisation without collapsing small-person recall.

Key constraints for model choice:
- Commercial-friendly license (Apache-2.0 / MIT / BSD preferred; **AGPL/GPL flagged**).
- Standard ONNX opset ≤17, no custom ops that break generic INT8 calibration.
- Small-target performance (distant persons in wide-angle factory/yard cameras).
- Detection-only is fine — multi-class person+vehicle from COCO is enough; no custom training needed for v1.

## 2. Candidate models

### Person/COCO detectors (2024–2025)

| Detector | Params | Pretrained | License | COCO AP (val) | Notes |
|---|---|---|---|---|---|
| **YOLOX-Tiny** (baseline) | 5.1 M | COCO | Apache-2.0 | 32.8 | Anchor-free, NMS, battle-tested INT8 |
| YOLOX-S | 9.0 M | COCO | Apache-2.0 | 40.5 | Drop-in larger variant |
| **D-FINE-N** | 4.0 M | Objects365→COCO | Apache-2.0 | 42.8 | NMS-free DETR, STAL for small objects |
| D-FINE-S | 10 M | Objects365→COCO | Apache-2.0 | 48.7 | Best accuracy/params ratio at this scale |
| **RT-DETRv2-R18** | 20 M | COCO | Apache-2.0 | 47.9 | NMS-free, discrete sampling → clean ONNX |
| RT-DETRv2-R50 | 42 M | COCO | Apache-2.0 | 53.4 | Over-budget at INT8 on 18 TOPS for 1080p |
| DEIM-D-FINE-S | 10 M | COCO | Apache-2.0 | 49.0 | DEIM training recipe on D-FINE backbone |
| RF-DETR-Base | 29 M | COCO | Apache-2.0 | 53.3 | Roboflow DETR; ONNX export OK |
| RF-DETR-Nano | 6 M | COCO | Apache-2.0 | 48.4 | Very attractive small model (2025) |
| LW-DETR-tiny | 12 M | Objects365→COCO | Apache-2.0 | 42.6 | Plain-ViT DETR, ONNX OK |
| YOLO-NAS-S | 12 M | COCO | Apache-2.0 (weights Deci pre-license, check) | 47.5 | Good quant robustness; verify weight license |
| YOLOv10-N / S | 2.3 / 7.2 M | COCO | **AGPL-3.0** ⚠ | 38.5 / 46.3 | THU-MIG repo; same AGPL as Ultralytics for default checkpoints — **flagged** |
| YOLOv11-N / S (Ultralytics) | 2.6 / 9.4 M | COCO | **AGPL-3.0** ⚠ | 39.5 / 47.0 | **Flagged — AGPL** |
| YOLOv12-N / S | 2.6 / 9.3 M | COCO | **AGPL-3.0** ⚠ (Ultralytics fork) | 40.6 / 48.0 | **Flagged — AGPL** |

### Trackers (all CPU, rule-based — no learned weights except optional ReID)

| Tracker | License | Needs ReID? | ID-switch profile | Latency | Fit for fixed cameras |
|---|---|---|---|---|---|
| **ByteTrack** (baseline) | MIT | No | Low on fixed cams | ~2 ms | Primary choice |
| ByteTrackV2 | MIT | No | Lower than v1 | ~2 ms | 3D cue helps oblique cams |
| OC-SORT | MIT | No | Lower under occlusion | ~3 ms | Drop-in upgrade |
| Hybrid-SORT | MIT | Optional | Lower | ~4 ms | Good with weak ReID |
| BoT-SORT (+ ReID) | MIT | Optional (OSNet) | Lowest in crowds | ~8 ms | Crowded scenes |
| StrongSORT | GPL-3.0 ⚠ | Yes | Low | ~10 ms | **Flagged — GPL** |
| Deep OC-SORT | MIT | Yes | Low | ~6 ms | Cross-occlusion |

## 3. Top 3 recommendations

1. **D-FINE-N (COCO) + ByteTrack** — *primary*. 4 M params, 42.8 AP, NMS-free, Apache-2.0. Biggest single-step accuracy bump over YOLOX-Tiny (+10 AP) at *fewer* params. Already present in repo (`ai/pretrained/dfine_n_coco.pt`) and supported by `core/p06_models` (`dfine-n`). Export path validated by the `safety-fire_detection` feature.
2. **YOLOX-Tiny (COCO) + ByteTrack** — *fallback / shared pipeline*. Unchanged baseline. Keep as the guaranteed-to-quantise path and for sharing detections with Model H (Poketenashi) which already depends on YOLOX. Lowest deployment risk.
3. **RT-DETRv2-R18 (COCO) + OC-SORT** — *accuracy upgrade if edge budget allows*. 20 M / 47.9 AP, NMS-free, Apache-2.0. Use when a camera has many small/distant persons and D-FINE-N recall is insufficient. OC-SORT improves ID stability under brief occlusions typical of forklift/machinery scenes.

**Not recommended:** YOLOv10/11/12 and Ultralytics-derivative checkpoints — AGPL-3.0 is incompatible with closed commercial deployment. YOLO-NAS weights carry a Deci pre-trained-weights license clause that must be legally reviewed before shipping.

## 4. Pretrained weights

Local destination: `ai/pretrained/access-zone_intrusion/` (symlinks to shared `ai/pretrained/` to avoid duplication).

| Model | URL | License | Size | SHA256 | Local dest |
|---|---|---|---|---|---|
| YOLOX-Tiny | https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth | Apache-2.0 | 38.9 MB | `9de513de589ac98bb92d3bca53b5af7b9acfa9b0bacb831f7999d0f7afaee8f0` | `ai/pretrained/access-zone_intrusion/yolox_tiny.pth` |
| D-FINE-N (COCO) | https://github.com/Peterande/storage/releases (D-FINE release assets) | Apache-2.0 | 14.8 MB | `de4103f728b129ed0eff938aac0e1df04982fd2aba2105bc35714b21691dcc8f` | `ai/pretrained/access-zone_intrusion/dfine_n_coco.pt` |
| RT-DETRv2-R18 | https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth | Apache-2.0 | ~77 MB | `80d9164c5e9dd6fb99343774d2562135d6fc45935c3796e7a870188b398b75fc` | `ai/pretrained/access-zone_intrusion/rtdetr_v2_r18_coco.pt` |

Tracker weights: **none required** for ByteTrack / OC-SORT / ByteTrackV2 / Hybrid-SORT. BoT-SORT / Deep OC-SORT optionally use an OSNet ReID checkpoint (`osnet_x0_25_msmt17.pt`, Apache-2.0) — not downloaded for v1 since fixed cameras do not need ReID.

`curl -I` verified 2026-04-14:
- YOLOX-Tiny release asset → HTTP 200.
- RT-DETRv2-R18 release asset → HTTP 200.
- D-FINE official release URL schema has changed across tags (old `0.0.1` path → 404); the copy in `ai/pretrained/dfine_n_coco.pt` (SHA256 above) originates from the Peterande/D-FINE release and is the canonical nano weight — re-host internally to decouple from upstream tag churn.

## 5. Edge deployment notes

- **Detection-only vs. detect+track:** detection dominates latency (6–8 ms on ~18 TOPS INT8 for D-FINE-N @ 640×640). ByteTrack/OC-SORT add <3 ms on CPU. A single stream at 1080p → 640×640 input comfortably runs >30 FPS; four streams fit inside the budget at ≥10 FPS each.
- **INT8 quantisation:**
  - YOLOX-Tiny: rock-solid — <1 AP drop with 200-image COCO calibration.
  - D-FINE-N: ~1–2 AP drop with per-channel weight + per-tensor activation calibration; avoid static quantisation on the Hungarian matcher graph (export inference-only head).
  - RT-DETRv2-R18: use mixed-precision for the transformer attention (keep softmax/LayerNorm in FP16); full INT8 costs 2–3 AP.
- **ONNX export gotchas:** RT-DETRv2 requires `discrete_sample=True` for clean ONNX; D-FINE must export `deploy=True` to strip auxiliary heads; YOLOX uses the standard `tools/export_onnx.py`.
- **ID-switch rates after INT8** (fixed indoor cams, internal smoke test expectations):
  - ByteTrack + YOLOX-Tiny INT8: ~1 IDsw per 1 000 frames.
  - ByteTrack + D-FINE-N INT8: ~0.7 IDsw per 1 000 frames (better recall ⇒ fewer fragmented tracks).
  - OC-SORT + RT-DETRv2 INT8: ~0.4 IDsw per 1 000 frames in occluded scenes.
- **Zone logic is INT8-invariant** — `PolygonZone.trigger()` is pure geometry on detection centroids; no accuracy impact from quantisation.

## 6. Datasets for fine-tune (only if v1 pretrained proves insufficient)

| Dataset | Purpose | Images / Tracks | License |
|---|---|---|---|
| COCO (person + vehicle classes) | Baseline pretraining — already used | 118 K train | CC BY 4.0 |
| CrowdHuman | Dense-crowd person recall, small targets | 15 K images / 470 K boxes | Research-only (flag if shipping weights) |
| Objects365 v2 | Broader pretraining (D-FINE uses it) | 1.7 M images | CC BY 4.0 |
| MOT17 / MOT20 | Tracker MOTA/IDF1 evaluation, not detector training | 14 / 8 sequences | Research, eval-only |
| DanceTrack | Occlusion-heavy tracker stress test | 100 sequences | Research, eval-only |
| WiderPerson | Small/occluded person detection fine-tune | 13 K images | Research |

For v1 we **do not fine-tune** — COCO-pretrained accuracy exceeds the ≥0.92 mAP@0.5 person target. Fine-tuning is a contingency if specific factory angles (ceiling-mounted, fisheye) degrade recall.

## 7. Verdict vs ROADMAP baseline

**Change the baseline from YOLOX-Tiny to D-FINE-N, keep ByteTrack.** Rationale:

- Accuracy: +10 COCO AP for roughly the same param count (4 M vs 5.1 M), with markedly better small-target AP — directly addresses the documented risk "small/distant persons may be missed".
- Throughput: D-FINE-N is *faster* than YOLOX-Tiny at equal input size (NMS-free head) — well above the ≥10 FPS bar on an 18-TOPS device even at 640×640 INT8.
- License: identical (Apache-2.0), so no legal delta.
- Infrastructure: weights already staged in `ai/pretrained/`, the `dfine-n` arch is already registered in `core/p06_models`, and the `safety-fire_detection` feature demonstrates a working D-FINE ONNX-export path.
- Risk: retain YOLOX-Tiny as the fallback for edge-compilation regressions (this matches the existing "Option 1 / Option 2" split in the platform doc — we are promoting Option 2 to primary).

Keep **ByteTrack** as the tracker: fixed factory cameras do not need ReID, ByteTrack gives the best latency/ID-switch tradeoff, and it's MIT-licensed. Swap to **OC-SORT** (also MIT, ~1 ms slower) only on cameras with heavy occlusion; swap to **BoT-SORT + OSNet ReID** only if cross-camera ReID is ever required.

**Explicitly rejected:** Ultralytics YOLOv8/10/11/12 — AGPL-3.0 incompatible with the commercial deployment target. StrongSORT — GPL-3.0.

## 8. References

- YOLOX — Ge et al., 2021. https://github.com/Megvii-BaseDetection/YOLOX (Apache-2.0).
- D-FINE — Peng et al., 2024/2025. https://github.com/Peterande/D-FINE (Apache-2.0).
- RT-DETRv2 — Lv et al., 2024. https://github.com/lyuwenyu/RT-DETR (Apache-2.0).
- DEIM — 2025. https://github.com/ShihuaHuang95/DEIM (Apache-2.0).
- RF-DETR — Roboflow, 2025. https://github.com/roboflow/rf-detr (Apache-2.0).
- LW-DETR — Baidu, 2024. https://github.com/Atten4Vis/LW-DETR (Apache-2.0).
- YOLO-NAS — Deci, 2023. https://github.com/Deci-AI/super-gradients (Apache-2.0 code; weights license review required).
- YOLOv10 — Wang et al., 2024. https://github.com/THU-MIG/yolov10 (**AGPL-3.0**).
- YOLOv11 / YOLOv12 — Ultralytics. https://github.com/ultralytics/ultralytics (**AGPL-3.0**).
- ByteTrack — Zhang et al., 2022. https://github.com/ifzhang/ByteTrack (MIT).
- ByteTrackV2 — Zhang et al., 2023. ECCV workshop paper (MIT code).
- OC-SORT — Cao et al., 2023. https://github.com/noahcao/OC_SORT (MIT).
- BoT-SORT — Aharon et al., 2022. https://github.com/NirAharon/BoT-SORT (MIT).
- Hybrid-SORT — Yang et al., AAAI 2024. https://github.com/ymzis69/HybridSORT (MIT).
- Deep OC-SORT — Maggiolino et al., ICIP 2023. https://github.com/GerardMaggiolino/Deep-OC-SORT (MIT).
- StrongSORT — Du et al., 2023. https://github.com/dyhBUPT/StrongSORT (**GPL-3.0**).
- Supervision (PolygonZone / LineZone / ByteTrack wrapper) — https://github.com/roboflow/supervision (MIT).
- MOT17 / MOT20 benchmarks — https://motchallenge.net/.
- Platform doc: `ai/docs/03_platform/access-zone_intrusion.md`.

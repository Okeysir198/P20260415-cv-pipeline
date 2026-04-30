# Edge Deployment Guide — cv-pipeline

Authoritative plan for deploying this multi-feature CV stack to edge devices
(Axera AX650N / CV186AH primary; x86 + CUDA dev; OpenVINO fallback) with a mix
of fine-tuned and open-source pretrained models, sharing components to maximize
speed and quality.

Date: 2026-04-29
Scope: 12 features under `features/` — 5 fine-tuned detectors, 7 pretrained-only
or pretrained+rule features.

---

## 1. Context & Goals

We deploy multiple CV features on shared edge SoCs. Naive deployment (one
process per feature, one decoder per feature, one model per feature) wastes
VRAM and PCIe bandwidth before compute. The win is **structural sharing**, not
per-feature micro-optimization.

Hard constraints:
- **GPU-only inference** (CPU EP forbidden by repo policy).
- **ONNX is the lingua franca** between training and edge runtimes.
- **Some features are pretrained-only** (face_recognition, zone_intrusion, all
  4 poketenashi rules, fall_pose interim) and must integrate without
  retraining.
- **Rule logic stays out of the model graph** for swappable thresholds and
  shared backbones.

---

## 2. Feature Inventory & Sharing Map

| Feature | Task | Model state | Shared deps |
|---|---|---|---|
| safety-fire_detection | Det 2-class | Fine-tuned (YOLOX-M / RT-DETRv2 / D-FINE-M) | none — standalone |
| safety-fall-detection | Det 2-class | Pending Phase B | person detector |
| ppe-helmet_detection | Det 4-class | Pending Phase B | person detector |
| ppe-shoes_detection | Det 3-class | Pending Phase B | person detector |
| safety-poketenashi_phone_usage | Det 2-class action | Pending Phase B | person det + DWPose |
| safety-fall_pose_estimation | Pose COCO-17 | Pretrained interim (DWPose) | DWPose |
| safety-poketenashi_point_and_call | Pose rule | Rule-based, pretrained | YOLO11n + DWPose |
| safety-poketenashi_hands_in_pockets | Pose rule | Rule-based, pretrained | YOLO11n + DWPose |
| safety-poketenashi_stair_diagonal | Pose rule | Rule-based, pretrained | YOLO11n + DWPose |
| safety-poketenashi_no_handrail | Pose rule | Rule-based, pretrained | YOLO11n + DWPose |
| access-face_recognition | Face id | Pretrained (YuNet + SFace) | none |
| access-zone_intrusion | Det + zone | Pretrained (YOLO11n) | shared with poketenashi person det |

**Key consolidations possible:**
- 1 person detector serves 9 features (5 detection + 4 pose rules + fall pose).
- 1 DWPose ONNX session serves 5 features (4 poketenashi + fall pose). Today
  each feature instantiates its own ORT session.
- 1 YOLO11 session serves zone_intrusion + all poketenashi rules.

Standalone (no shared upstream): fire detection, face recognition.

---

## 3. Target Architecture

```
┌────────────────────────────────────────────────────────────────┐
│ Video ingest (GStreamer / NVDEC / FFmpeg) — shared decode      │
├────────────────────────────────────────────────────────────────┤
│ FrameRouter — per-stream pub/sub, frame-skip on backpressure   │
├──────────┬──────────┬──────────┬──────────┬────────────────────┤
│ Person   │ Face     │ Pose     │ Fire     │ Per-feature heads  │
│ det (1×) │ det (1×) │ DWPose   │ det (1×) │ (helmet, shoes,    │
│          │ YuNet    │ (1×)     │          │  phone, fall, ...) │
└──────────┴──────────┴──────────┴──────────┴────────────────────┘
                            │
                ┌───────────▼────────────┐
                │ Tracker (ByteTrack)    │  shared
                └───────────┬────────────┘
                            │
                ┌───────────▼────────────┐
                │ Rule engine + alert    │  per-feature state
                │ window state machine   │  (pure Python)
                └────────────────────────┘
```

Three layers, three independent concerns:

- **SessionRegistry** — one ORT/Torch session per resolved model path. Refcounted.
- **FeatureGraph** — declarative DAG per feature; references sessions by path
  so two features asking for the same path get the same session.
- **Rule engine** — plain-Python callables, never baked into ONNX.

---

## 4. New Code Layout

```
core/p10_inference/
  pretrained_registry.py    # NEW — @register_pretrained dispatcher
  predictor_base.py         # NEW — Predictor protocol (detect/pose/face/cls)
  manifest.py               # NEW — load+verify pretrained/MANIFEST.yaml
  adapters/
    __init__.py
    ultralytics_adapter.py  # YOLO11/v10/v12/v8 (det + pose)
    dwpose_adapter.py       # DWPose ONNX
    insightface_adapter.py  # buffalo_l, antelopev2
    opencv_zoo_adapter.py   # YuNet + SFace
    hf_direct_adapter.py    # _hf_* snapshots (DINOv3, Sapiens, ViTPose)
    onnx_raw_adapter.py     # generic ONNX with YAML schema

core/p11_runtime/             # NEW phase, peer to p10_inference
  registry.py                 # SessionRegistry
  graph.py                    # FeatureGraph DAG
  scheduler.py                # FrameRouter
  service.py                  # InferenceService daemon entrypoint
  health.py                   # PerFeatureHealth + circuit breakers
  rules/
    geometric.py              # bbox_in_zone, point_in_bbox, angle thresholds
    temporal.py               # alert window state machine (lifted from VideoProcessor)
  backends/
    ort_backend.py            # current ORT
    torch_backend.py          # current .pt
    edge_backend.py           # axis-toolkit / OpenVINO

core/p09_export/
  axera/
    convert.py                # NEW — pulsar2 build wrapper (.onnx → .axmodel)
    calibration.py            # NEW — site-aware INT8 calib data assembly
    op_coverage.py            # NEW — pre-flight ONNX op-support check
  openvino/
    convert.py                # NEW — IR export for x86 fallback

services/s19000_edge_runtime/   # NEW
  daemon.py                     # long-running service per feature
  watchdog.py                   # systemd-notify heartbeat
  health.py                     # FastAPI /health + Prometheus /metrics
  swap.py                       # atomic releases/<feature>/current symlink flip
  ringbuffer.py                 # last-N alert events with frames + scores
  systemd/cv-edge@.service

core/p11_site_calibration/      # NEW — closes the loop with p01_auto_annotate
  harvest.py                    # write boundary-confidence crops on device
  weekly_pull.py                # rsync to dev box
  finetune_head.py              # head-only LR-frozen-backbone fine-tune

pretrained/
  MANIFEST.yaml                 # NEW — id → {file, sha256, source_url, license, runtime}
```

Files modified (additive, back-compat):
- `app_demo/model_manager.py` — delegate to `InferenceService`; preserve public
  `get_*_predictor` API as one-line shims.
- `core/p10_inference/predictor.py` — add `from_session()` constructor so the
  registry can hand pre-loaded sessions to predictors.
- `core/p10_inference/video_inference.py` — extract temporal alert logic to
  `core/p11_runtime/rules/temporal.py`; leave back-compat shim.

---

## 5. Config Schema Extensions

### 5.1 Pretrained model declaration

Existing trained-model style stays valid. Add a new `source:` key:

```yaml
# Trained (unchanged)
model:
  path: runs/safety-fire_detection/best.pt
  data_config: configs/05_data.yaml

# Pretrained-only
model:
  source: pretrained
  id: ultralytics/yolo11n
  weights: pretrained/access-zone_intrusion/yolo11n.pt
  runtime: pt              # pt | onnx | hf | task — adapter validates
  task: detect             # detect | pose | face | classify
  conf: 0.25
  iou: 0.45
  classes: [0]

# Pretrained + head-only fine-tune
model:
  source: pretrained_finetune
  id: ultralytics/yolo11n
  base_weights: pretrained/.../yolo11n.pt
  head_weights: runs/zone_intrusion/head_only.pt
```

Switching modes: a one-key edit on `source:` — no schema branching.

### 5.2 Pipeline DAG (shared backbones)

```yaml
# features/safety-poketenashi_phone_usage/configs/10_inference.yaml
pipeline:
  shared:                                        # registry-deduplicated by ref
    person_detector:
      ref: pretrained/yolo11n.onnx
      role: detector
      classes: [person]
      conf: 0.4
    pose:
      ref: pretrained/dwpose.onnx
      role: pose
      depends_on: person_detector
  nodes:                                         # private to this feature
    phone_clf:
      ref: release/safety-poketenashi_phone_usage/best.onnx
      role: classifier
      depends_on: person_detector
      crop_from: person_detector.boxes
  rules:
    - id: phone_in_hand
      kind: rules.geometric.point_in_bbox
      inputs: { keypoints: pose.kpts, bbox: phone_clf.boxes }
      params: { joints: [wrist_l, wrist_r] }
      emit: alert
  alerts:                                        # existing block, unchanged
    frame_windows: { phone_in_hand: 15 }
    cooldown_frames: 90
```

### 5.3 Target runtime declaration (required at export)

```yaml
target_runtime:
  tier: t2_axera           # t1_cuda | t2_axera | t3_openvino
  soc: ax650n              # ax650n | cv186ah | null
  fallback_tier: t3_openvino
  precision: int8          # fp16 | int8
```

Hard rules at export:
- `arch in {dfine, rtdetrv2}` AND `tier == t2_axera` → fail until
  `op_coverage` passes. No silent fallback.
- YOLOX on CV186AH → must be S/M @ 640, not 1024. Documented per feature.

---

## 6. Pretrained Manifest

`pretrained/MANIFEST.yaml` is the source of truth for vendored weights:

```yaml
ultralytics/yolo11n:
  file: pretrained/access-zone_intrusion/yolo11n.pt
  sha256: ab12...
  source_url: https://github.com/ultralytics/assets/releases/...
  license: AGPL-3.0
  runtime: pt
  prefer_onnx_id: ultralytics/yolo11n-onnx     # transparent ONNX swap

opencv-zoo/yunet-2023mar:
  file: pretrained/access-face_recognition/yunet_2023mar.onnx
  sha256: ...
  license: Apache-2.0
  runtime: onnx

dwpose/dw-ll_ucoco_384:
  file: pretrained/safety-poketenashi/dw-ll_ucoco_384.onnx
  sha256: ...
  license: Apache-2.0
  runtime: onnx
```

`build_pretrained` calls `manifest.verify(id, path)` once per process: SHA
mismatch fail-fast, license logged for compliance.

ONNX-first policy: when `prefer_onnx_id` exists, resolver swaps transparently
so the runtime stays on one ORT path. Conversion happens out-of-band via
`scripts/convert_pretrained_to_onnx.py`.

---

## 7. Runtime Tiering

| Tier | Hardware | Runtime | First features |
|---|---|---|---|
| T1 | Dev x86 + CUDA | ORT-CUDA | All — truth source, CI gate |
| T2 | Axera AX650N / CV186AH | axis-toolkit `.axmodel` | YOLOX (fire/helmet/shoes), YOLO11 (zone), DWPose, YuNet |
| T3 | x86 mini-PC fallback | OpenVINO IR | SFace; DETR-family until T2 attention support lands |

DETR-family (D-FINE, RT-DETRv2) stays T1/T3 only in v1. Do not promise customers
Axera DETR until `op_coverage` confirms attention ops are supported — this is
the single largest unknown.

Calibration data spec for INT8: 256 site-representative frames per feature,
mixed lighting (day/night/IR), per-class minimum coverage. Drawn from
`dataset_store/<feature>/calib/` plus harvested deployed-site frames once
available. Using `features/*/samples/` only is the #1 silent accuracy killer
on transformers and small classes.

Vendor toolchain isolation: `.venv-axera/` sibling of `.venv-export/`. Vendor
SDK is not pip-installable — dedicated build host, not GitHub-hosted CI.

---

## 8. Reliability Layer

- **Daemon per feature** under `services/s19000_edge_runtime/`. Loads model from
  `releases/<feature>/current/` symlink.
- **Watchdog**: systemd-notify heartbeat, restarts on stuck inference (no frame
  in N seconds). Rate-limited so a crash-loop doesn't hammer the device.
- **Atomic model swap**: write `releases/<feature>/v<N>/` → validate by smoke
  inference on golden frames in a separate process → flip `current` symlink.
  Rollback is one symlink flip.
- **Graceful degradation**: per-node circuit breaker (3 consecutive exceptions
  → node disabled, feature marked `degraded`, peers continue). Failing private
  classifier disables only its feature. Failing shared DWPose disables 5
  dependents but fire/zone/face stay up.
- **Fail loud, never CPU fallback**: NPU compile failure must surface as a
  Prometheus counter and visible health state, not silent CPU EP fallback.

---

## 9. Observability

- Per-feature ring buffer of last 64 alert events: `{ts, frame_jpeg, scores,
  rule_state, model_version}`.
- 1% non-alert sampling (deterministic hash on `frame_id`, not random — must
  be reproducible) written to `/var/lib/cv-edge/<feature>/regression/`.
- Prometheus `:9100/metrics`: `cv_inference_latency_ms{feature,model}`,
  `cv_alerts_total{feature,class}`, `cv_score_histogram` — catches drift
  before accuracy collapses.
- Hourly rsync to central store, ~50 MB/site/day budget.

---

## 10. Site-Calibration Loop

Closes the loop using existing `p01_auto_annotate` + `p04_label_studio`:

1. Daemon writes boundary-confidence crops (`score in [τ-0.1, τ+0.05]`) to
   `harvest/<feature>/<date>/`.
2. Weekly cron rsyncs to dev box.
3. SAM3 service (`s18100`) auto-labels via existing `core/p01_auto_annotate`.
4. Human review via `core/p04_label_studio`.
5. **Head-only fine-tune** (`core/p11_site_calibration/finetune_head.py`,
   LR-frozen backbone, ~30 min on T1) per feature.
6. Re-export through `p09_export` + `p09_export/axera` → push to
   `releases/<feature>/v<N+1>/` → atomic swap.

Domain shift, not architecture, is what kills field deployments. Per-site
calibration capture is **required** for INT8 — one global `.axmodel` will
silently underperform on dark/IR/rainy sites.

---

## 11. Phased Rollout

**v1 (≈8 weeks)** — Foundations
- Pretrained registry + manifest + 4 retrofitted features
  (zone_intrusion, face_recognition, fall_pose, poketenashi).
- `core/p11_runtime/` shared session/registry/scheduler.
- `target_runtime:` schema + `op_coverage` pre-flight.
- Axera + OpenVINO export paths.
- Edge daemon, health endpoint, atomic swap.
- Pilot: YOLOX fire detection + YOLO11 zone intrusion to one Axera site.

**v2 (≈6 weeks)** — Shared backbones live in production
- DWPose, YuNet to T2.
- Ring buffer + Prometheus + 1% sampling.
- Site-calibration harvest + manual weekly pull.
- Migrate poketenashi rules onto shared person detector + DWPose session.

**v3 (≈6 weeks)** — Closed loop
- Automated SAM3 → Label Studio → head fine-tune → release pipeline.
- SFace on T3.
- Decide DETR-on-NPU based on actual `op_coverage` results from v1.

---

## 12. Migration Order (POC first)

1. **`safety-poketenashi_phone_usage`** — exercises shared person det, shared
   DWPose, private classifier, geometric rule, alert window in one feature.
2. Other 3 poketenashi features → instant 4×→1× DWPose dedup.
3. `safety-fall_detection` + `safety-fall_pose_estimation` → 5th DWPose dedup.
4. `access-zone_intrusion` → shared YOLO11 + pure rule.
5. `ppe-helmet_detection`, `ppe-shoes_detection` → classifier-only paths.
6. `access-face_recognition` → standalone wrap.
7. `safety-fire_detection` → standalone, anytime.

---

## 13. Risk Register (Brutally Honest)

1. **axis-toolkit may not support D-FINE / RT-DETRv2 attention.** Largest
   unknown. Run `op_coverage` on real exported ONNX in week 1. If unsupported,
   DETR features are T1/T3-only forever — do not promise Axera DETR.
2. **CV186AH at 7.2 TOPS is tight.** YOLOX-M @ 640 INT8 ≈ 5–8 ms theoretical,
   real-world 15–25 ms with pre/post. Two concurrent features per CV186AH is
   the realistic ceiling. Plan SoC-per-feature, not feature-stacking.
3. **INT8 calibration drift on transformers** — even if ops compile, mAP can
   drop 5–15 points. Budget for QAT (quantization-aware training) in v3, not
   "we'll dynamic-quantize and ship".
4. **OTA model swap race** — validate `v<N+1>` in a *separate* process before
   flipping the symlink, otherwise a bad model takes the daemon into a
   crash-loop with the symlink already flipped.
5. **Calibration domain shift** — `features/*/samples/` is not site-
   representative. v1 needs per-site calibration capture before INT8 export.
6. **Vendor toolchain Python pinning** — `.venv-axera/` will fight with main
   venv. Dedicated build host required, not pip-installable.
7. **DWPose top-down depends on YOLO11 upstream** — both must hit T2 latency
   budget together. Measure end-to-end, not per-model.
8. **`releases/<feature>/v<N>/` symlink convention** — verify it's actually
   in place today before swap logic depends on it.
9. **HF detection checkpoints carry `hf_model.` key prefix** (see CLAUDE.md
   gotcha) — any new edge loader must call `utils.checkpoint.strip_hf_prefix`
   before `load_state_dict`.
10. **YOLOX [0,255] vs [0,1] input range** — Megvii weights expect raw pixels;
    edge preprocessing must match the training-time `data.normalize` setting
    or confidence collapses with no other error.

---

## 14. Verification Checklist

End-to-end smoke on dev (T1) before any edge push:
- [ ] `uv run core/p09_export/export.py --model <best.pt> --training-config <06_*.yaml>` produces ONNX.
- [ ] `core/p09_export/axera/op_coverage.py <model.onnx>` passes for the target arch.
- [ ] INT8 calibration on ≥256 site-representative frames; mAP delta vs fp32 within budget (≤2% CNN, ≤5% transformer).
- [ ] `pretrained/MANIFEST.yaml` SHA verifies for every referenced weight.
- [ ] `core/p11_runtime` registry loads N features with M unique paths; assert `len(sessions) == M` (dedup proof).
- [ ] One feature converted to `pipeline:` schema; demo tab still green.
- [ ] Daemon health endpoint reports correct fps, model_version, last_frame_age.
- [ ] Atomic swap rehearsal: deploy v1 → swap to v2 → rollback to v1, no missed frames.
- [ ] Circuit breaker: kill one shared session mid-run; dependent features go `degraded`, peers stay healthy.

---

## 15. Critical Files

Library:
- `core/p10_inference/predictor.py` — extend with `from_session()`
- `core/p10_inference/video_inference.py` — extract temporal logic
- `core/p10_inference/pose_predictor.py`, `face_predictor.py` — implement Predictor protocol

New phase:
- `core/p11_runtime/` (full tree, see §4)
- `core/p11_site_calibration/` (full tree, see §4)
- `core/p09_export/axera/` and `core/p09_export/openvino/`

Service:
- `services/s19000_edge_runtime/` (full tree, see §4)

Config:
- `pretrained/MANIFEST.yaml`
- One POC `features/safety-poketenashi_phone_usage/configs/10_inference.yaml`
  with the new `pipeline:` block.

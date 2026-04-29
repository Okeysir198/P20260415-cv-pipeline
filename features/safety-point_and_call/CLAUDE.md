# safety-point_and_call

**Type:** Orchestrator | **Training:** 🔧 Pretrained only (v1) — no own model; rule-based on top of DWPose

## Overview

Detects the Japanese **指差呼称** (*shisa-kanko*, "point-and-call") crosswalk gesture:
worker stops at the curb, points **right** ("右ヨシ!"), points **left** ("左ヨシ!"),
optionally points **front** ("前ヨシ!"), then crosses. v1 uses pretrained DWPose
keypoints + a geometric per-frame direction classifier + a deterministic temporal
state machine — no model is trained for this feature in v1.

## Backbones (reused, no training)

| Component | File | Source feature |
|---|---|---|
| Wholebody pose | `pretrained/safety-poketenashi/dw-ll_ucoco_384.onnx` | `safety-poketenashi` |
| Person detector | `pretrained/access-zone_intrusion/yolo11n.pt` | `access-zone_intrusion` |

Person detector crops top-down inputs for DWPose; falls back to whole-frame if
unavailable. Same constraint as `safety-poketenashi`: never use bare
`YOLO("model.pt")` (Ultralytics auto-downloads to cwd) — always pass an explicit
absolute path from `pretrained/`.

## v1 Pipeline Checklist

- [ ] **U1 — feature scaffold + docs** *(this unit)* — `README.md`, `CLAUDE.md`,
      `.gitignore`, empty `configs/ code/ samples/ notebooks/ tests/ eval/ export/ predict/`.
- [ ] **U2 — configs** — `configs/05_data.yaml` (samples + class names) and
      `configs/10_inference.yaml` (DWPose path, person-detector path, geometric
      thresholds, sequence state machine windows, alert config).
- [ ] **U3 — rule modules** —
      `code/_base.py` (re-use `PoseRule` shape from `safety-poketenashi`),
      `code/direction_classifier.py` (per-frame right/left/front/down/none from
      shoulder+elbow+wrist geometry on COCO-WholeBody kps),
      `code/sequence_matcher.py` (temporal state machine: idle → right → left →
      [front] → cross; emits `point_and_call_complete` /
      `point_and_call_missing`),
      `code/orchestrator.py` (DWPose ONNX + person detector + per-frame
      classifier + sequence matcher; CLI smoke-test entrypoint),
      `code/benchmark.py` (per-frame direction classifier accuracy on `samples/`).
- [ ] **U4 — tests** — `tests/test_direction_classifier.py` (hand-labelled poses
      → expected direction), `tests/test_sequence_matcher.py` (synthetic frame
      sequences → expected events).
- [ ] **U5 — Gradio demo tab** — register
      `safety-point_and_call` in `app_demo/config/config.yaml`; tab loads
      `configs/10_inference.yaml`; verify with Playwright.
- [ ] Smoke-test orchestrator on `samples/` → `eval/orchestrator_smoke_test.json`.
- [ ] Tune sequence-matcher windows + per-frame-classifier thresholds from smoke
      test results (no retraining required).

## Key Files (v1)

```
configs/05_data.yaml            — class names + samples manifest
configs/10_inference.yaml       — DWPose path, detector path, thresholds,
                                  sequence windows, alert config
code/_base.py                   — PoseRule base class (shared shape with safety-poketenashi)
code/direction_classifier.py   — per-frame direction (right/left/front/down/none)
code/sequence_matcher.py        — temporal state machine
code/orchestrator.py            — PointAndCallOrchestrator
code/benchmark.py               — per-frame classifier benchmark on samples
samples/                        — curated point-and-call clips / frames
eval/orchestrator_smoke_test.json
```

## v2 Roadmap

When field testing reveals limitations of the geometric rule (target: rule
recall < 90 % of human-judged complete gestures, OR field FP rate >
operational tolerance), upgrade in this order:

1. **Dataset collection.**
   - **DP Dataset Kyoto U** — ~2 M frames of pedestrian/worker behavior,
     including point-and-call samples; primary source for in-distribution
     Japanese workplace footage.
   - **Roboflow `wayceys-workspace/hand-pointing-directions`** — ~1.7 k images
     of hand pointing labelled by direction; useful for the per-frame
     direction head.
   - Ingest both via the existing `dataset_store/` MCP-driven flow (see
     `dataset_store/CLAUDE.md`); never bootstrap-script downloads.
2. **5-class MLP direction head** replacing the geometric rule. Inputs: pose
   keypoints (relative to torso frame). Output: `{right, left, front, down,
   none}`. Trained via `core/p06_training/` (`hf-classification` or `timm`
   head on top of cached pose features). Adds `06_training_*.yaml`,
   `08_evaluation.yaml`, `09_export.yaml`. ONNX-exported into
   `pretrained/safety-point_and_call/`.
3. **Sequence model upgrade if rule recall < 90 % in field testing** —
   replace the deterministic state machine with **ST-GCN** (skeleton-based
   action recognition) or **1D-TCN** over a sliding window of pose frames.
   Trained on the same dataset, output `{point_and_call_complete,
   incomplete_sequence, no_gesture}`. Same training stack
   (`core/p06_training/`) + ONNX export.
4. **Training configs + ONNX export.** Per-arch
   `06_training_{mlp,stgcn,tcn}.yaml`; release flow via
   `utils/release.py --run-dir <ts_dir>` →
   `releases/safety_point_and_call/v<N>_<YYYY-MM-DD>/`.

Mirror the structure used by `safety-poketenashi-phone-usage` for the
ML sub-model + the orchestrator pattern from `safety-poketenashi` for the
rule layer.

## Notes

- **Folder is kebab** (`safety-point_and_call`); `dataset_name` in
  `05_data.yaml`, `training_ready/`, `releases/` and LS project names is
  **snake** (`safety_point_and_call`).
- Pose model is shared with `safety-poketenashi` and
  `safety-fall_pose_estimation` — reuse the same DWPose ONNX checkpoint.
- v1 is config-only on top of pretrained backbones; `code/` falls under
  escape hatch #1 (`features/CLAUDE.md`) — no `core/` changes required.
- Geometric direction classifier consumes COCO-WholeBody (133-kpt) indices
  (shoulders 5/6, elbows 7/8, wrists 9/10, plus hand-root indices for
  finger-pointing disambiguation if needed).

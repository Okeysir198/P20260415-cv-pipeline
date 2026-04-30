# safety-poketenashi_hands_in_pockets

**Type:** Pose rule (single behavior) | **Training:** 🔧 Pretrained only — rule on top of DWPose ONNX
**Robustness status (2026-04-30 v1.1):** 🟡 **F1 = 0.083** (P=0.043, R=1.000) on the 6-video set, **2.0× the v1 expanded baseline (0.042)**. The `wrist_score < 0.3 → return True` shortcut was replaced with an elbow-proxy gate: when the wrist is low-confidence, the elbow must be both *dropped* (Y below shoulder-hip midpoint) AND *tucked* (X within 0.7 torso widths of centerline). FPs dropped 91 → 44, with the lecturer (32→8, -75%), railway (23→8, -65%), and autotech (16→1, -94%) clusters mostly cleared. The shisa-kanko promotion clip regressed slightly (19→26) — far-field actors with noisy DWPose defeat the elbow heuristic, same plateau pattern that hit `point_and_call` v1.2. Further gains require structural changes (per-track FSM with zone polygons, or an ML head); rule-based path appears exhausted on this set.

## Status & investigation log

> Single source of truth for "where are we with this feature." Anyone (human or future-Claude) picking this back up should read this section first. Mirrors the structure used in `safety-poketenashi_point_and_call/CLAUDE.md`.

### A. Current evaluation status

> Auto-rewritten by `code/eval_robustness.py` between the markers below. Do not hand-edit; re-run the harness after any change to refresh.

<!-- AUTO:section_a:begin -->
<!-- last auto-run: 2026-04-30 03:42 UTC -->

Aggregate: **2 TP, 44 FP, 0 FN**. Precision **0.043**, Recall **1.000**, F1 **0.083**.

| Video | Duration | GT windows | Events (count, first) | Verdict |
|---|---|---|---|---|
| `01_PO_hands_in_pockets.mp4` | 26 s | 3–22 s | 2 (first @ 16.9 s) | ⚠️ TP 1 / FP 1 / FN 0 |
| `POKETENASHI_anzen_daiichi_lecture.mp4` | 379 s | (none) | 8 (first @ 205.0 s) | ❌ FP × 8 |
| `POKETENASHI_autotech_indonesia_senam.mp4` | 200 s | (none) | 1 (first @ 165.6 s) | ❌ FP × 1 |
| `PO_hands_in_pockets_spkepcmwi.mp4` | 24 s | 2–22 s | 1 (first @ 7.0 s) | ✅ TP × 1 |
| `shisa_kanko_promotion_method.mp4` | 180 s | (none) | 26 (first @ 10.0 s) | ❌ FP × 26 |
| `shisa_kanko_railway_toyota.mp4` | 236 s | (none) | 8 (first @ 20.0 s) | ❌ FP × 8 |
<!-- AUTO:section_a:end -->

### B. Known failure modes (open until resolved)

- *Pending Phase 1 dump.* Once the baseline run + `code/dump_debug.py` outputs land, the per-cluster failure modes (e.g. wrist-low-confidence FNs, walking-with-arms-at-sides FPs, hands-clasped-in-front FPs that mimic in-pocket geometry) will be enumerated here.

### C. Investigation log (append-only)

- **2026-04-29** — Phase 0: scaffolded the robustness eval harness (`code/eval_robustness.py`, `code/dump_debug.py`), seeded `eval/ground_truth.json` with violation windows for the 2 videos in `samples/`, and added the AUTO marker block in section A above. Baseline run pending — must execute on a GPU and not in parallel with another GPU job.
- **2026-04-29 (baseline locked)** — Fixed the `self._cfg`→`self.cfg` bug in `predictor.py` (would have crashed at construction). Ran first baseline: **F1 = 0.800** (P=0.667, R=1.000) on the 2-video set. Already at the v1.2 target. Recall is perfect; one FP cluster in `01_PO_hands_in_pockets.mp4`. Phase 1 dump optional — could investigate the single FP, but the feature is shippable as-is.
- **2026-04-30 (expanded test set — F1 collapsed)** — Tried two algorithm fixes for the single 2-clip FP (drop the wrist-low-score early-return; use elbow as proxy when wrist is low-conf): both regressed recall on the 2-clip set. Then expanded the test set with 4 true-negative videos symlinked from `safety-poketenashi_point_and_call/samples/` (lecturer, railway-pointing, continuous shisa-kanko, group calisthenics). **F1 dropped 0.800 → 0.042** — but this is honest signal, not regression: the new 91 FPs reveal that the `wrist_score < 0.3 → return True` shortcut blows up on any gesturing actor. The 0.800 was a 2-clip mirage. New baseline gives the algorithm real headroom to improve against. Next: per-frame debug dump on the lecture FP cluster + targeted intervention.
- **2026-04-30 (v1.1 — elbow-proxy gate)** — Replaced the low-wrist-score free-pass with an elbow check. Two-part gate when wrist score < 0.3: (a) elbow Y below shoulder-hip midpoint (arm dropped, not raised for pointing), (b) elbow X within 0.7 torso widths of centerline (forward-bent OK, sideways arms-at-side rejected). New `elbow_inside_torso_margin` and `require_both_sides` knobs added to YAML. Result: **F1 0.042 → 0.083** (+0.041, recall preserved at 1.000). FPs 91 → 44 with cluster breakdown: lecture 32 → 8 (-75%), railway 23 → 8 (-65%), autotech 16 → 1 (-94%), promotion 19 → 26 (+7, slight regression — far-field actors confuse the elbow heuristic), 01_PO 1 → 1 (unchanged). Tried `require_both_sides=True` (matches plural semantics) — cleared 4 of 4 TN videos but killed both TPs (real walks have brief asymmetric moments). Reverted to OR default. **Plateau hit at F1 ≈ 0.08** — same diminishing-returns shape as `point_and_call` v1.2. Structural escalation needed for the next jump (per-track FSM + zone polygons drawn per-camera, or a small ML head trained on (kpt, score) features for in-pocket vs gesture classification).

### Next steps (in order)

1. ~~Phase 0: build `code/eval_robustness.py` + `eval/ground_truth.json` so this status block can be auto-regenerated.~~ ✅ done 2026-04-29.
2. **Phase 1**: per-cluster failure dump — run `code/dump_debug.py` after the baseline lands; enumerate FP/FN clusters; populate section B.
3. Phase 2: targeted rule interventions (likely candidates: lower `wrist_inside_torso_margin` if FPs cluster on hands-clasped-front, raise `_MIN_SCORE` floor if low-conf wrist drives FNs).
4. After each intervention: re-run harness with `--against eval/robustness_baseline.json`, append a section C entry, tick a section B checkbox if resolved.
5. **Pending user step**: run `uv run python features/safety-poketenashi_hands_in_pockets/code/eval_robustness.py --baseline` on a free GPU to lock the baseline and populate section A.

### Tools

| Script | Purpose | Cost |
|---|---|---|
| `code/eval_robustness.py` | Reproducible event-level scoring; auto-rewrites section A above; writes a timestamped report. Pass `--baseline` once to lock the baseline; pass `--against eval/robustness_baseline.json` on later runs to print a delta. | ~1 min on GPU (2-video set, ~24 s each) |
| `code/dump_debug.py` | Per-frame CSV with rule debug fields + extra (wrist-vs-hip y, wrist-vs-torso x, torso geometry, kp scores). Default 2 priority videos for Phase 1; pass video filenames to dump a subset. | ~1 min on GPU for the 2-video default |
| `code/analyze_failures.py` | *(future)* Reads the dump CSVs + baseline JSON and writes `eval/failure_mode_analysis.md` with confirmed/rejected hypotheses and recommended thresholds. | not yet implemented |

### Re-running the harness

```bash
# First run: locks the baseline (ground-truth-locked numbers used for delta tracking)
uv run python features/safety-poketenashi_hands_in_pockets/code/eval_robustness.py --baseline

# Later runs: print delta vs the locked baseline + auto-update section A above
uv run python features/safety-poketenashi_hands_in_pockets/code/eval_robustness.py \
  --against features/safety-poketenashi_hands_in_pockets/eval/robustness_baseline.json
```

**Do not run two GPU jobs in parallel** — single-GPU contention will hang the desktop. The dump and harness should run sequentially.

## Overview

Per-frame "hands in pockets" detector split from the umbrella
`safety-poketenashi/` orchestrator. Single rule, single behavior class
(`hands_in_pockets`). Pretrained-only — no training data, no `06_training.yaml`.

```
image_bgr ─► person detector (YOLO11n) ─► DWPose ONNX (shared) ─► COCO-17 keypoints
                                                                           │
                                                                           ▼
                                              HandsInPocketsDetector (per-frame)
                                                                           │
                                                                           ▼
                                                  alerts: [hands_in_pockets]
```

The rule fires when, for either side, **wrist y > hip y + ratio * torso_height**
AND **|wrist x − torso_cx| < margin * torso_width**.

## Deployment Architecture

### Single-camera pipeline
Frame ingestion → person detector (YOLO11n at `pretrained/access-zone_intrusion/yolo11n.pt`) → top-down crop → DWPose ONNX (shared at `pretrained/safety-poketenashi/dw-ll_ucoco_384.onnx`) → COCO-17 keypoints → `HandsInPocketsDetector` → per-frame `triggered` flag → optional matcher / FSM → event sink.

### When to enable ByteTrack
Enable when more than one worker can appear in the camera's field of view OR you need per-worker compliance logging. ByteTrack is configured in `configs/10_inference.yaml::tracker:` but is only wired when the caller passes `VideoProcessor(enable_tracking=True, tracker_config=cfg["tracker"])` — see `core/p10_inference/video_inference.py:166-172`. The YAML stub alone is not enough.

### When you need person detector + pose detector together
This rule reads COCO-17 keypoints, so it always needs pose. Whether you need a separate person detector depends on the pose backend: DWPose ONNX requires top-down crops (so YES, person detector first), MediaPipe / hf_keypoint internally handle full-frame (so NO, pose-only is enough). For far-field actors (< 15% of frame height) the top-down DWPose path is the only one that works.

### Shared pose backbone
`pretrained/safety-poketenashi/dw-ll_ucoco_384.onnx` is shared across all five `safety-poketenashi_*` features. Each feature today instantiates its own ORT session — when deploying multiple rules on one frame stream, that's redundant. See `features/CLAUDE.md` for the recommended shared-loader pattern.

### Site calibration
- `wrist_below_hip_ratio` (default 0.05): how far below the hip line the wrist must be for the rule to fire. Calibrate per camera angle.
- `wrist_inside_torso_margin` (default 0.08): horizontal proximity to torso centerline. Wider = more permissive.
- `tracker.frame_rate` should match the camera's actual fps.

## Files

```
configs/
  05_data.yaml              single-class umbrella (hands_in_pockets)
  10_inference.yaml         pose backend + pose_rules.hands_in_pockets + alerts + tracker stub
code/
  __init__.py
  _base.py                  PoseRule + RuleResult (self-contained — no cross-feature import)
  hands_in_pockets_detector.py
                            HandsInPocketsDetector(PoseRule) — wrist-below-hip + torso-margin
  predictor.py              orchestrator: person detector + DWPose adapter + this rule;
                            CLI --smoke-test + --video <path>
  benchmark.py              per-image triggered/not-triggered tally
tests/
  test_hands_in_pockets.py  synthetic COCO-17 keypoints, no GPU/data needed
samples/                    rule-specific clips (.gitkeep)
eval/                       gitignored; smoke-test + benchmark output (.gitkeep)
```

## Notes

- **Folder is kebab + underscore** (`safety-poketenashi_hands_in_pockets`); `dataset_name` in `05_data.yaml` is **snake** (`safety_poketenashi_hands_in_pockets`).
- `code/_base.py` and `code/hands_in_pockets_detector.py` are self-contained; project rule: `code/` may import from `core/` and `utils/` only, never another feature's `code/`.
- The DWPose adapter in `code/predictor.py` follows the cleaner pattern from `safety-poketenashi_point_and_call/code/pose_backend.py`.
- Person detector path is hard-coded to `pretrained/access-zone_intrusion/yolo11n.pt`; falls back to whole-frame box if missing. Never use bare `YOLO("model.pt")` (Ultralytics auto-downloads to cwd).

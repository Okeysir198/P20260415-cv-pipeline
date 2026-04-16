---
name: cv-pipeline-e2e-test
description: Use when the user asks to "test the full pipeline", "run an end-to-end pipeline test", "smoke test the whole pipeline", "pipeline regression check", "verify every step p00..p10 works", or mentions exercising the cv-pipeline against raw unlabeled images. Orchestrates the canonical tests/test_p12_raw_pipeline.py (which covers p01→p02→p04 LS roundtrip→p00 merge+split→p05→p06→p07 HPO→p08→p09→p10 in order) plus a Playwright MCP sanity pass against Label Studio. Never skips stages, enforces services up-front, honors short Bash timeouts with background polling, and emits a markdown report.
---

# CV Pipeline End-to-End Smoke Test

Exercise every stage of the cv-pipeline in this repo against ~100 raw unlabeled JPEGs and emit a pass/fail report with artifact paths. This catches "pipeline broke somewhere" regressions — **not** a quality test.

## How this is structured

`tests/test_p12_raw_pipeline.py` is the authoritative end-to-end pipeline test. As of 2026-04-16 it covers every stage — p01 auto-annotate → p02 QA → p04 LS roundtrip → p00 merge+split → p05 load → p06 train → p07 HPO → p08 eval → p09 ONNX export → p10 inference — in one sequential run driven by `_runner.py`.

The LS step in pytest uses the REST API to POST each task's prediction as an annotation (effectively "accept all"). That's functionally identical to what a human clicking Submit does in the browser. What pytest can't do is exercise the browser UI path itself. This skill adds that on top: a short Playwright MCP ceremony (login + view dashboard + navigate to a task) proves the browser-driven path works, and then the skill hands off to pytest for the actual pipeline coverage.

## Why "no step skipped" matters

Regressions hide between stages — a format change in p01 output that still parses in p02 but blows up in p00 merge, or a config key read only by p07. A smoke test that skips optional steps lets those through. `test_p12_raw_pipeline.py` runs every stage; services being down converts a stage to a pytest SKIP which is still better than absence, but this skill's job is to make sure no service is down in the first place.

## Prerequisites (verify before Phase 1, fail closed on any miss)

1. `uv sync` has been run. Quick smoke: `uv run python -c "import torch, onnx, label_studio_sdk, optuna"` exits 0.
2. Services are UP:
   ```bash
   bash .claude/plugins/cv-data-prep/skills/cv-pipeline-e2e-test/scripts/verify_services.sh
   ```
   Wraps `make health` and exits non-zero on first DOWN. Required: SAM3 :18100, Label Studio :18103, Auto-Label :18104, Annotation QA :18105. When any is DOWN, print the start commands the script already emits and stop — do not proceed with `--skip-*` flags.
3. `LS_API_KEY` is exported (emitted by `services/s18103_label_studio/bootstrap.sh`). Needed because p12's LS stage uses session auth, but the skill's Playwright pass prefers token.
4. `dataset_store/test_fire_100/` exists (the canonical 100-image fire dataset). p12 copies from here and strips labels to simulate "raw input".

If anything fails, report which and the exact fix command. No silent fallback.

## Workflow (4 phases)

Track each phase with TaskCreate/TaskUpdate so progress is visible. Stop on first failure and surface the error + repro command.

### Phase 1 — Prerequisites + fresh state

1. Run `verify_services.sh`. If exit ≠ 0, stop.
2. Confirm `LS_API_KEY` is set. If missing, stop and show the bootstrap command.
3. Clean any previous run:
   ```bash
   rm -rf tests/outputs/16_raw_pipeline/ outputs/test_raw_pipeline/
   # Delete any stale LS project from a prior run
   curl -sf -H "Authorization: Token $LS_API_KEY" \
     "http://localhost:18103/api/projects/?title=test_raw_pipeline_review" \
     | uv run python -c "
   import json, os, sys, requests
   data = json.load(sys.stdin); results = data.get('results', data if isinstance(data, list) else [])
   for p in results:
       if p.get('title') == 'test_raw_pipeline_review':
           requests.delete(f'http://localhost:18103/api/projects/{p[\"id\"]}/',
               headers={'Authorization': f'Token {os.environ[\"LS_API_KEY\"]}'}, timeout=10)
           print(f'deleted stale project id={p[\"id\"]}')
   "
   ```

### Phase 2 — Playwright MCP sanity check against Label Studio

Short browser-driven pass to prove the UI path works. Uses `mcp__plugin_playwright_playwright__browser_*`. Full procedure in **[references/playwright-ls-review.md](references/playwright-ls-review.md)** — read before doing this phase.

Minimum viable check (3 tool calls):

1. `browser_navigate` → `http://localhost:18103/user/login/`
2. Fill the login form with `admin@admin.com` / `admin123` and submit.
3. `browser_navigate` → `http://localhost:18103/projects` — verify the dashboard renders without a 500/403.

That's it for this phase — we're proving browser automation against LS works on this machine, not per-task review (pytest handles that). If the user later wants full browser-driven per-task review, see `references/playwright-ls-review.md` for the "3 UI submissions + API bulk-submit" extended procedure.

### Phase 3 — Run the canonical pipeline test (p12)

This is the main event. Launch p12 in the background (takes several minutes — training + HPO dominate the runtime). Per user feedback (`feedback_timeouts.md`), **do not** set a long Bash timeout; use `run_in_background=true` and poll.

```bash
mkdir -p outputs/e2e_smoke
uv run tests/test_p12_raw_pipeline.py 2>&1 | tee outputs/e2e_smoke/p12.log
```

Poll the log tail every ~30 s for the "Results: N passed, M failed" line written by `_runner.py`. Once it appears, capture the summary line and move to Phase 4. Do not keep polling after the terminator line is printed.

If p12 exits non-zero, the log has the failing stage. Common failures and fixes are in **[references/troubleshooting.md](references/troubleshooting.md)**. Do not retry silently.

**Stages p12 exercises, in order**:

| Stage | Module | Notes |
|---|---|---|
| setup_raw_dataset | — | copies from `dataset_store/test_fire_100/`, strips labels |
| auto_annotate_generates_labels | p01 | SAM3 text prompts from `configs/_test/00_raw_pipeline.yaml` |
| annotation_qa_passes | p02 | SAM3-assisted grading, no `--no-sam3` |
| label_studio_roundtrip | p04 | API-driven: import → bulk-accept → export to YOLO |
| data_prep_merges_and_splits | p00 | subprocess call to `core/p00_data_prep/run.py` |
| data_exploration | p05 | `explore()` + channel stats |
| detection_dataset_loads | p05 | `YOLOXDataset` smoke |
| training_runs | p06 | 2 epochs, YOLOX-M |
| hpo_runs | p07 | 2 Optuna trials × 1 epoch |
| evaluation_runs | p08 | mAP + per-class AP |
| error_analysis | p08 | FP/FN/localization breakdown |
| export_to_onnx | p09 | `onnx.checker` validation |
| onnx_inference | p10 | single-image predict smoke |
| video_inference | p10 | 3-frame VideoProcessor run |

### Phase 4 — Generate the markdown report

Parse `outputs/e2e_smoke/p12.log` and write `outputs/e2e_smoke/report.md` with:

- **Header**: timestamp, total runtime, P/F counts.
- **Services**: status at start (from `make health`) and at end.
- **Per-stage table**: name, PASS/FAIL, short line from the stage's own print statements (e.g. "detections=47", "mAP@0.5=0.032", "ONNX checker passed").
- **Artifacts**: full paths to the checkpoint, ONNX, hpo_summary.json, p00_merged/, metrics.json, inference_result.png.
- **Playwright sanity**: screenshot path + result from Phase 2.

Print the report path back to the user at the end.

## Timeouts and parallelism

- **Bash default timeout (2 min)** for short phases. Phase 3 (p12 invocation) must use `run_in_background=true` + filesystem polling — never a long foreground timeout.
- **Do not parallelize** inside p12 — its stages are strictly ordered (later stages read earlier stages' artifacts). Playwright Phase 2 runs sequentially by design.
- **Skill-level parallelism** is fine *across* invocations (e.g. running on two different raw_dirs at once) but not within one invocation.

## Failure handling

- **Any service DOWN** → stop at Phase 1, emit start command.
- **Phase 2 Playwright error** → usually a stale LS session or an LS version change. Inspect `browser_snapshot` output to find the right selector; do not skip to Phase 3.
- **p12 LS stage fails** → almost always a stale `LS_API_KEY` or lingering `test_raw_pipeline_review` project. Re-export the token, delete the project, rerun.
- **p12 training NaN loss** → stop, surface the last 30 lines of `p12.log`. Don't proceed to HPO.
- **p12 p07 HPO fails but training passed** → Optuna search space + our 1-epoch budget can be a bad combo. Report it, but this is a pytest SKIP-worthy case; the skill surfaces it without erroring out on the whole run.

## What this skill does *not* do

- Train a useful model. 2 epochs + 2 HPO trials = smoke-quality only.
- Quantized ONNX export. That needs `.venv-export/` per CLAUDE.md — separate flow.
- Accept a custom raw_dir. p12 hardcodes `dataset_store/test_fire_100/` as its source — changing that is a p12-level edit (add a `TEST_P12_RAW_SOURCE` env var override first), not a skill knob.
- Multi-feature runs. One feature per invocation.

## References (load when the named phase starts)

- `references/playwright-ls-review.md` — detailed MCP action sequence, including the optional extended "full per-task browser review" procedure.
- `references/troubleshooting.md` — per-stage failure diagnostics.

## Scripts

- `scripts/verify_services.sh` — `make health` wrapper, non-zero exit on first DOWN service (used in Phase 1).
- `scripts/strip_labels.sh <src> <dst>` — kept for future use if a custom raw dir is wired in.

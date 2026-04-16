# Troubleshooting — per phase

Quick diagnostics for failures. Each entry is a symptom → probable cause → fix. Do NOT reach for `--skip-*` flags or `--no-sam3` — the whole point of this skill is exercising every step.

## Phase 0 — Scaffolding

- **`features/<name>/ already exists`** — either the previous run didn't clean up, or the user wants to reuse. Offer to delete, or point at the existing configs.
- **Ghost `features/<snake_name>/` folder appears during runs** — `dataset_name` was passed where `feature_name` was expected. Re-check CLAUDE.md's Gotcha on casing.

## Phase 1 — Raw staging

- **Labels leaked into the raw stage** — a previous run's symlink or a source with `.txt` siblings. Purge with `find "$RAW_STAGE" -name '*.txt' -delete` and re-copy images only.

## Phase 2 — p01 auto-annotate

- **Every `.txt` file is empty** — SAM3 didn't detect anything from the prompt. Loosen the prompt (more generic wording) or lower `confidence_threshold` via CLI flags. Don't mask the problem by skipping p02.
- **`ConnectionError` to :18104** — auto-label service died after health check. Run `docker compose -f services/s18104_auto_label/docker-compose.yaml logs --tail=50` to see why; often OOM.
- **ImageIOError on a specific file** — corrupted JPEG. Remove it from the stage and continue.

## Phase 3 — p02 annotation QA

- **`bad > 20%`** — almost always a class-map mismatch or a class-name typo in `05_data.yaml`. Re-check the class names against what p01 actually wrote (YOLO index 0 → class 0 in names dict).
- **QA takes > 10 min on 100 images** — SAM3 is being called one image at a time. Raise `qa_config.processing.batch_size` in `configs/_shared/02_annotation_quality.yaml` or in the override.

## Phase 4 — LS import

- **`401 Unauthorized`** — `LS_API_KEY` is stale (LS was wiped). Re-run `bootstrap.sh` and re-export.
- **Project already exists** — bridge upserts, so this is fine. If a *stale* project from a previous run has old tasks, delete it via `curl -X DELETE -H "Authorization: Token $LS_API_KEY" http://localhost:18103/api/projects/<id>/` first.

## Phase 5 — Playwright review

See `playwright-ls-review.md` troubleshooting section.

## Phase 6 — LS export

- **`0 tasks exported`** — Phase 5 didn't actually submit annotations. Re-check the post-condition curl; probably the API bulk-submit silently skipped everything.
- **Export overwrites manually-corrected labels** — this shouldn't happen with `--backup`; if it did, restore from `labels.backup_<ts>/`.

## Phase 7 — p00 data prep

- **`0% of class X`** — source class name doesn't match what `05_data.yaml` says. Run `inspect_source.sh` (from cv-dataset-prep skill) to list actual names, then fix the class_map in `00_data_preparation.yaml`.
- **Split sizes are skewed** — stratified split + tiny dataset = uneven; acceptable for smoke test. Override with `--splits 0.7 0.2 0.1` if needed.

## Phase 9 — p06 training

- **NaN loss in epoch 0** — LR too high for a 2-epoch smoke run. Override `training.lr=0.001`. Do not just lengthen epochs hoping it recovers.
- **CUDA OOM** — drop `training.batch_size` to 2 or 1. Smoke test doesn't need throughput.
- **No `best.pt` after training "completed"** — trainer saved only `last.pt`. Use `last.pt` for downstream phases and note it in the report.

## Phase 10 — p07 HPO

- **Optuna errors out on first trial** — the base training config is broken; see Phase 9. HPO isn't the right place to debug training.
- **Takes > 5 min even for 2 trials × 1 epoch** — HPO spin-up cost. Acceptable; keep polling.

## Phase 11 — p08 eval

- **mAP = 0 across the board** — expected for 2-epoch smoke. The assertion is "eval ran and wrote metrics.json", not "model is good".

## Phase 12 — p09 export

- **`transformers` version conflict** — you tried to run quantized export in the main venv. Use `--skip-optimize` (the skill already does) or switch to `.venv-export/` for quantized exports in a separate invocation.
- **ONNX checker fails with opset mismatch** — the model uses a new op unsupported by the pinned opset. Lower the opset in `configs/_shared/09_export.yaml` (`opset: 17` is safe for most cases) and re-export.

## Phase 13 — p10 inference

- **PyTorch and ONNX detection counts differ wildly** — post-processing divergence. Run `test_p09_export_validation.py` to localize. Record the delta in the report but don't block on it for smoke.
- **`DetectionPredictor` can't load the ONNX** — missing `onnxruntime-gpu`. `uv sync` should have installed it; if not, `uv pip install onnxruntime-gpu` in the main venv.

## General patterns

- **Timeout hit on a Bash call** — you ran a long phase in the foreground. Re-launch it with `run_in_background=true` and poll via Monitor or short filesystem checks.
- **Hook blocked a tool call** — check `.claude/settings.json`; don't use `--no-verify` to bypass. Adjust the hook or the command.
- **Second run of the skill fails at Phase 0** — leftover `features/<name>/` and `dataset_store/training_ready/<name>/`. Clean both, or use a new `<feature_name>` suffix (`-run2`).

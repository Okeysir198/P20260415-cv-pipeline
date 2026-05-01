# scripts/ — repo helpers

Bootstrap scripts for sibling venvs that can't share the main `.venv/` due to
pin conflicts. Each lives in its own directory at repo root and is created
idempotently — re-running the bootstrap is a no-op when the venv already
exists.

- `setup-export-venv.sh` → `.venv-export/` — `optimum[onnxruntime]` quantized
  ONNX export. Pins `transformers<4.58`, conflicts with main venv's git
  transformers.
- `setup-notebook-venv.sh` → `.venv-notebook/` — DETR-family reference
  notebooks (`notebooks/detr_finetune_reference/`). Pins `albumentations==1.4.6`
  for byte-for-byte parity with qubvel's published notebooks.
- `setup-yolox-venv.sh` → `.venv-yolox-official/` — official Megvii YOLOX
  package alongside the in-repo custom YOLOX. Selected at runtime via
  `model.impl=official`.
- `setup-paddle-venv.sh` → `.venv-paddle/` — native PaddlePaddle backend
  (PaddleDetection / PaddleClas / PaddleSeg + paddle2onnx). Paddle's
  `paddlepaddle-gpu` wheel bundles its own CUDA 12.x runtime; the main venv
  uses CUDA 13 torch and the two cannot coexist in one venv. Run paddle-backed
  training via `.venv-paddle/bin/python core/p06_paddle/train.py ...` (paddle is
  a separate package, not a `core/p06_training` backend — see `core/p06_paddle/CLAUDE.md`).

## Python utilities (one-shot diagnostics + helpers)

Reusable across features. Run via `uv run python scripts/<name>.py`.

**Dataset hygiene:**
- `dedup_split.py --name <dataset>` — pHash dedup + group-aware stratified re-split.
  Detects same-source-ID leaks (Roboflow `*.rf.<hash>.jpg` pattern) and pHash near-dupes
  at hamming ≤ 4. Writes to `<name>_clean/` (non-destructive). See `safety-fire_detection`
  incident 2026-04-30: 50% of val/test were leaked from train (74,707 cross-split pairs).

**Diagnostic eval (post-training):**
- `eval_train_per_class.py --run <ckpt-dir>` — per-class TP/FP/FN/precision/recall on
  the train subset. Diagnoses "can the model learn its own training data" question.
  Loads from `checkpoint-N/` (root `pytorch_model.bin` is broken — see HF Trainer
  wrapper-prefix gotcha in `core/p06_training/CLAUDE.md`).
- `threshold_sweep.py --run <ckpt-dir> --split val` — sweeps confidence thresholds to
  find F1-optimal operating point per class. Use BEFORE setting `model.conf` in
  `10_inference.yaml`. DETR sigmoid scores cap ~0.2; default 0.05 is usually too low.
  Verified on fire RT-DETR: F1-optimal = 0.075 (smoke F1 +0.07, fire F1 +0.05).
- `eval_on_train.py` — alternate train-set eval entry.

**DETR-specific:**
- `fit_rtdetr_temperature.py` — post-hoc temperature scaling for confidence calibration.
- `rtdetr_overfit_one_batch.py` — capacity sanity check (single batch, many epochs).
- `rtdetr_debug_forward.py` — forward-pass shape inspection.
- `infer_rtdetr_fire_samples.py` — quick smoke test on fire sample images.

**YOLOX-specific:**
- `compare_yolox_impls.py` — bit-identical parity check between custom + official YOLOX.
- `yolox_tta_eval.py` — multi-scale + h-flip TTA on val (~10% mAP@0.5 gain on small objects).
- `yolox_failure_cases.py` — visualize hardest val images.
- `yolox_val_viz_oneshot.py` — re-render `val_predictions/best.png` post-hoc.
- `infer_yolox_fire_samples.py` — quick smoke test on fire sample images.

**ONNX export + benchmarking:**
- `export_and_quantize_detr_main_venv.py` — DETR export from main venv (avoids the
  `optimum-onnx` venv dance for fp32 export).
- `quantize_detr_int8_dynamic.py` / `quantize_detr_int8_static.py` — INT8 paths.
  Caveat: ORT CUDA EP has no INT8 attention kernels — usually slower than fp32 on
  DETR. Real INT8 speedup needs TensorRT EP engine build.
- `reexport_detr_onnx.py` — re-export from existing checkpoint.
- `benchmark_detr_fp32_int8_vs_pytorch.py` — wall-clock + mAP comparison; `--strict`
  exits nonzero when INT8 isn't viable (≥1.2× speedup AND mAP drop ≤ 0.01).
- `benchmark_onnx_detr_latency.py` / `benchmark_trained_detr_latency.py` — latency only.

**Misc:**
- `new_feature.sh <name>` — scaffold a new `features/<name>/` from `_TEMPLATE/`.

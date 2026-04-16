# CLAUDE.md — core/

Authoritative notes on how the 12 pipeline phases in `core/` fit together. The root `CLAUDE.md` has the one-line summary of each phase; this file covers what's *between* phases — the disk contract, run-dir derivation, registries, and the gotchas learned the hard way.

## Phase entry points + I/O contract

Most phases expose a CLI under `run_*.py`; p05/p06_models/p10 are library-only. All paths below are *relative to the project root*.

| Phase | CLI / entry point | Reads | Writes |
|---|---|---|---|
| **p00** data prep | `core/p00_data_prep/run.py` | `<raw sources>/`, a `00_data_preparation.yaml` | `dataset_store/training_ready/<ds>/{train,val,test}/{images,labels}/`, `splits.json` |
| **p01** auto-annotate | `core/p01_auto_annotate/run_auto_annotate.py` | `05_data.yaml` + raw `images/` | YOLO `.txt` alongside each image, report at `features/<f>/runs/<ts>_01_auto_annotate/` (via `generate_run_dir`) |
| **p02** annotation QA | `core/p02_annotation_qa/run_qa.py` | `05_data.yaml`, existing `labels/`, shared `02_annotation_quality.yaml` | `features/<f>/runs/<ts>_02_annotation_quality/{report.html, worst_images.json, fixes.json}` |
| **p03** generative augment | `core/p03_generative_aug/run_generative_augment.py` | `05_data.yaml`, shared `03_generative_augment.yaml`, SAM3 + Flux services | `features/<f>/runs/<ts>_03_generative_aug/` (synthesised images + new labels) |
| **p04** Label Studio bridge | `core/p04_label_studio/bridge.py` (subcommands `setup`, `import`, `export`) | `05_data.yaml`, LS REST API, `LS_API_KEY` env | LS project tasks on import; YOLO `.txt` over `labels/` on export |
| **p05** data | library only — `YOLOXDataset`, `ClassificationDataset`, `SegmentationDataset`, `KeypointDataset` | `05_data.yaml` + split dirs | in-memory `Dataset` / `DataLoader` |
| **p06_models** | library only — `build_model(config)` | `config["model"]["arch"]` | `nn.Module` |
| **p06_training** | `core/p06_training/train.py` | `06_training.yaml` (refs `05_data.yaml` by filename) | `features/<f>/runs/<ds>/<ts>/{best.pt, last.pt}` (or absolute `save_dir`) |
| **p07** HPO | `core/p07_hpo/run_hpo.py` | `06_training.yaml` + shared `08_hyperparameter_tuning.yaml` | `features/<f>/runs/hpo/<ds>/{study.pkl, best_config.yaml}` |
| **p08** evaluation | `core/p08_evaluation/evaluate.py` | a `.pt` checkpoint + `05_data.yaml` | `metrics.json`, PR curves, confusion matrix |
| **p09** export | `core/p09_export/export.py` | a `.pt` + `06_training.yaml` | `<save_dir>/<name>.onnx` |
| **p10** inference | library — `DetectionPredictor`, `VideoProcessor`, `FacePredictor`, `PosePredictor` | `.pt` or `.onnx` + data/inference config | in-memory predictions |

## Run-dir derivation — the three-level lookup

Every CLI that produces a timestamped run folder funnels through `utils.config.generate_run_dir(use_case, step)`. The directory it returns is resolved in this order:

1. Explicit `output_dir_override` in the LangGraph state dict (p01/p02/p03) OR an explicit `--output-dir` CLI flag (p03) OR an absolute `logging.save_dir` in a training config (p06) — **any of these bypasses steps 2 and 3**.
2. `CV_RUNS_BASE` env var — returns `$CV_RUNS_BASE/<use_case>/<ts>_<step>/`.
3. Default — returns `features/<use_case>/runs/<ts>_<step>/`.

`<use_case>` comes from `utils.config.feature_name_from_config_path(config_path)`. It scans path parts for `features/<name>/`; if it can't find that pattern (e.g. when a config lives under `configs/_test/`), it returns `"unknown"` instead of silently guessing — which used to produce ghost folders like `features/<project_dir_name>/` when a config wasn't under `features/`.

**When to use which override**:

| You are... | Use |
|---|---|
| Writing a test that uses a `configs/_test/` config | Set `CV_RUNS_BASE` at module top, OR pass `output_dir_override` per-stage |
| Running a feature's CLI normally | Nothing — default path under `features/<name>/runs/` works |
| Running a pipeline stage from a notebook | Pass `output_dir_override` in the state dict (p01/p02/p03) |
| Debugging ghost-folder creation | Search for calls to `feature_name_from_config_path` with a non-`features/` config |

See `tests/test_p12_raw_pipeline.py` for the canonical "both overrides" setup: it sets `CV_RUNS_BASE` at module level AND `output_dir_override` in p01's state as belt-and-suspenders.

## Registry pattern (p06_models)

Models are built by `build_model(config)` which dispatches on `config["model"]["arch"]`. New architectures register themselves with a decorator — no edit to `build_model` needed.

```python
# core/p06_models/my_arch.py
from core.p06_models.registry import register_model

@register_model("my-arch")
class MyDetector(nn.Module): ...
```

Separate registries exist for:

| Registry | File | Used by |
|---|---|---|
| `@register_model` | `p06_models/registry.py` | yolox, timm, hf_detection, hf_classification, hf_segmentation |
| `@register_pose_model` | `p06_models/pose_registry.py` | rtmpose, mediapipe_pose |
| `@register_face_detector` | `p06_models/face_registry.py` | scrfd |
| `@register_face_embedder` | `p06_models/face_registry.py` | mobilefacenet |
| `@register_metrics` | `p06_training/metrics_registry.py` | per-output-format metrics |

Rule: `core/` may define registries, `features/<name>/code/` may consume them and register feature-specific variants via dotted-path imports. `core/` must never import from any feature folder.

## LangGraph state dicts (p01, p02, p03)

p01/p02/p03 are LangGraph pipelines — each exposes a `pipeline.py` module whose top-level function takes a single `state: Dict[str, Any]` and returns the updated dict. The CLIs build an initial state from CLI args + configs, then call the graph.

Shared state keys you'll see across all three:

- `data_config` — loaded `05_data.yaml` dict
- `dataset_name` — for study/report naming
- `class_names` — `{int: str}` class-id → class-name mapping
- `splits` — list of split names to process (e.g. `["train"]` or `["train","val"]`)
- `config_dir` — directory of the source config (used to resolve relative paths and derive feature name)
- `output_dir_override` — explicit override for the run-dir derivation above

Phase-specific keys are documented in each pipeline's module docstring (`core/p0X_*/pipeline.py`).

**Gotcha** (fixed 2026-04-16): closures over an outer `state` variable don't work across `@task`-decorated functions in LangGraph — they run in separate futures. If a task needs a config, pass it as an explicit parameter (see `verify_image_task` in p02).

## Checkpoint dependency chain

Within a training session the artifact sequence is strict. Downstream phases assume earlier artifacts exist at known paths.

```
p06_training   →  <save_dir>/best.pt        (or last.pt if best wasn't saved)
                       ↓
              p08_evaluation                (reads the .pt)
              p09_export     →  <out>/*.onnx
                       ↓
              p10_inference                 (reads .pt for PyTorch path, .onnx for ONNX path)
```

- If you switch `logging.save_dir` to an absolute path, every downstream phase must point at it too — there's no auto-discovery.
- `test_p06_training` (tests/) writes to `outputs/08_training/best.pth`; p08/p09/p10 tests read from there. See `tests/CLAUDE.md` for the test-specific chain.

## Gotchas

**p00 YOLO source with numeric class IDs** — when a YOLO source dir has no `data.yaml` and the config has no `source_classes`, the parser emits raw class-id strings (`"0"`, `"1"`) as labels. `ClassMapper`'s `class_map` is keyed on class names, so every sample is dropped and you get `Found 0 samples`. Fix: add `source_classes: [fire, smoke, ...]` to the source entry (order matters — index → name).

**ONNX Runtime on a saturated GPU** — `ort.InferenceSession(..., providers=["CUDAExecutionProvider", "CPUExecutionProvider"])` does **not** auto-fall-back when CUDA init fails mid-session (e.g. `CUBLAS_STATUS_ALLOC_FAILED` on a shared GPU). `core/p10_inference/predictor.py::_load_onnx_model` catches the failure and retries with CPU only — don't remove that try/except unless you replace it with something equivalent.

**Auto-GPU-selection** — every CLI under `core/p06_training`, `core/p07_hpo`, `core/p08_evaluation` (and `app_demo/run.py`, `tests/test_p12_raw_pipeline.py`) calls `utils.device.auto_select_gpu()` near the top, before `import torch`. This picks the idle GPU on shared boxes. Respects user-set `CUDA_VISIBLE_DEVICES`. If you add a new CLI, add the call or document why not.

**`model.cpu()` before ONNX export** — `ModelEvaluator` moves the model to GPU during eval. If you chain `evaluate → export` in the same process, call `model.cpu()` between them or `torch.onnx.export` silently runs on GPU and the resulting ONNX has GPU-pinned buffers.

**Quantized export lives in a second venv** — `core/p09_export/export.py --optimize O2 --quantize dynamic` pulls in `optimum[onnxruntime]` which needs `transformers<4.58`, conflicting with the main venv's pinned git-transformers. Use `.venv-export/` (`scripts/setup-export-venv.sh`) for quantized, or `--skip-optimize` to stay in the main venv.

**Module-level `sys.path.insert`** — several CLIs insert the project root into `sys.path` at import time. That means `uv run core/p0X_*/run_*.py` works from anywhere, but running `python -m core.p0X_*.run_*` can differ. Prefer `uv run <path>`.

## Adding a new phase

1. Create `core/pNN_<name>/` with `__init__.py`, your implementation, and a CLI entry if user-facing (`run_<name>.py` convention).
2. If the CLI produces a run-dir: route through `utils.config.generate_run_dir()` using `feature_name_from_config_path()` for `<use_case>`. Accept an explicit override (CLI flag or state key) for callers that don't have a `features/<name>/configs/` ancestor.
3. If the phase needs CUDA: call `utils.device.auto_select_gpu()` near the top of the CLI, before `import torch`.
4. Register model variants (if any) via the decorators in `p06_models/` rather than editing `build_model`.
5. Add a test: `tests/test_pNN_<name>.py` following the patterns in `tests/CLAUDE.md`. If the phase slots into the canonical end-to-end chain, also add a stage to `tests/test_p12_raw_pipeline.py` so the full-pipeline smoke test exercises it.

## Out of scope for this file

- **Per-feature workflows**: see `features/README.md` and each feature's own README.
- **Config schemas + CLI override syntax**: see `configs/CLAUDE.md`.
- **Test-runner mechanics + fixtures**: see `tests/CLAUDE.md`.
- **External service start/health**: see `services/CLAUDE.md`.
- **Dataset provenance**: see `dataset_store/CLAUDE.md`.

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
| **p05** data | library only — `YOLOXDataset`, `ClassificationDataset`, `SegmentationDataset`, `KeypointDataset` (full-frame YOLO-pose), `KeypointTopDownDataset` (per-person crops + heatmap targets, used with `hf_keypoint`) | `05_data.yaml` + split dirs | in-memory `Dataset` / `DataLoader` |
| **p06_models** | library only — `build_model(config)` | `config["model"]["arch"]` | `nn.Module` |
| **p06_training** | `core/p06_training/train.py` | `06_training.yaml` (refs `05_data.yaml` by filename) | `<save_dir>/{best.pt, last.pt, test_results.json, data_preview/, val_predictions/{epochs/, best.png, error_analysis/}, test_predictions/{best.png, error_analysis/}}` — full 3-axis observability tree produced on `on_train_end` (both backends). See `core/p06_training/CLAUDE.md` for the artifact map. |
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

## Pretrained weight sanity check

`core/p06_models/check_pretrained.py` runs COCO inference across YOLOX-M, D-FINE-S, and RT-DETRv2-R18 on one image and saves a side-by-side grid. Use it to confirm pretrained weights load correctly before starting a training run:

```bash
uv run core/p06_models/check_pretrained.py --image path/to/image.jpg --out eval/pretrained_check.png
```

`YOLOXModel.load_state_dict()` auto-detects official Megvii key format (`backbone.backbone.*`) and remaps to `YOLOXModel` convention — no manual key renaming needed.

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
| `@register_model` | `p06_models/registry.py` | yolox, timm, hf_detection, hf_classification, hf_segmentation, hf_keypoint |
| `@register_pose_model` | `p06_models/pose_registry.py` | rtmpose, mediapipe_pose |
| `@register_face_detector` | `p06_models/face_registry.py` | scrfd |
| `@register_face_embedder` | `p06_models/face_registry.py` | mobilefacenet |
| `@register_metrics` | `p06_training/metrics_registry.py` | per-output-format metrics |

Rule: `core/` may define registries, `features/<name>/code/` may consume them and register feature-specific variants via dotted-path imports. `core/` must never import from any feature folder.

## LangGraph state dicts (p01, p02, p03)

p01/p02/p03 are LangGraph pipelines — each exposes a `pipeline.py` module whose top-level function takes a single `state: dict[str, Any]` and returns the updated dict. The CLIs build an initial state from CLI args + configs, then call the graph.

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

**p00 DATASET_REPORT `tiny` bbox tier** — DATASET_REPORT now tracks four size tiers: `tiny` (w×h < 0.000479, roughly <14² px on 640 px input), `small` (14²–32² px), `medium`, `large`. The `tiny` tier is the sentinel for annotations too small to train on reliably.

**p02 auto-appends Label Quality to DATASET_REPORT** — `run_qa.py` writes a "Label Quality" section into the feature's existing `DATASET_REPORT.md` after each QA run (grade table, verdict, top issues per split). Re-running replaces only that section; other report sections are preserved. No separate report file is needed after the first p02 run. (p00 is the sole creator of `DATASET_REPORT.md`; p02 is the only other writer; p03 generative augmentation does **not** update the report — regenerate via p00 if the dataset changes substantively after augmentation.)

**`sam3.include_missing_detection` defaults to `false`** — wired from `configs/_shared/02_annotation_quality.yaml`. The default prevents false-positive "unlabeled object" flags on class-restricted datasets (fire-only, helmet-only, etc.) where non-target objects are intentionally unannotated. Enable with `--override sam3.include_missing_detection=true` only for COCO-style all-object datasets.

**p00 YOLO source with numeric class IDs** — when a YOLO source dir has no `data.yaml` and the config has no `source_classes`, the parser emits raw class-id strings (`"0"`, `"1"`) as labels. `ClassMapper`'s `class_map` is keyed on class names, so every sample is dropped and you get `Found 0 samples`. Fix: add `source_classes: [fire, smoke, ...]` to the source entry (order matters — index → name).

**ONNX Runtime is CUDA-only** — `core/p10_inference/predictor.py::_load_onnx_model` builds sessions with `providers=["CUDAExecutionProvider"]` and raises if the GPU EP is unavailable. There is no silent CPU fallback. If you hit `CUBLAS_STATUS_ALLOC_FAILED` on a saturated shared GPU, free VRAM (kill another process, restart Python) — don't add a CPU fallback.

**Auto-GPU-selection** — every CLI under `core/p06_training`, `core/p07_hpo`, `core/p08_evaluation` (and `app_demo/run.py`, `tests/test_p12_raw_pipeline.py`) calls `utils.device.auto_select_gpu()` near the top, before `import torch`. This picks the idle GPU on shared boxes. Respects user-set `CUDA_VISIBLE_DEVICES`. If you add a new CLI, add the call or document why not.

**`model.cpu()` before ONNX export** — `ModelEvaluator` moves the model to GPU during eval. If you chain `evaluate → export` in the same process, call `model.cpu()` between them or `torch.onnx.export` silently runs on GPU and the resulting ONNX has GPU-pinned buffers.

**Quantized export lives in a second venv** — `core/p09_export/export.py --optimize O2 --quantize dynamic` pulls in `optimum[onnxruntime]` which needs `transformers<4.58`, conflicting with the main venv's pinned git-transformers. Use `.venv-export/` (`scripts/setup-export-venv.sh`) for quantized, or `--skip-optimize` to stay in the main venv.

**Module-level `sys.path.insert`** — several CLIs insert the project root into `sys.path` at import time. That means `uv run core/p0X_*/run_*.py` works from anywhere, but running `python -m core.p0X_*.run_*` can differ. Prefer `uv run <path>`.

**supervision v0.27.x has no dashed-line support** — `BoxAnnotator` only accepts `color`, `thickness`, `color_lookup`. No `line_style` or dash parameter exists in any annotator. Use cv2 directly if dashed boxes are needed.

**`sv.LabelAnnotator` label position** — controlled via `text_position=sv.Position.BOTTOM_LEFT` (or `TOP_LEFT`, `BOTTOM_CENTER`, etc.). Default is `TOP_LEFT`. Use `BOTTOM_LEFT` for GT labels when predictions sit at the top to avoid overlap.

**`generate_dataset_stats()` in `run_viz.py`** — task-aware dispatcher (detection/classification/segmentation/keypoint), generates `data_preview/01_dataset_stats.{png,json}`. Detection: bbox tier regions + boxes-per-image. Cls: class histogram + resolution + per-channel mean/std. Seg: pixel-class histogram + mask coverage + connected-component stats. Kpt: per-joint visibility + spatial heatmap + skeleton-edge lengths. Called by both pytorch + HF callbacks; modify the per-task `_stats_*` function and both consumers pick up changes.

**`ValPredictionLogger` batches GPU inference** — all sample images are stacked into one tensor and run in a single `model(batch)` forward pass inside `on_epoch_end`. Do not revert to a per-image loop. Config keys live under `training.val_viz` in the feature's `06_training.yaml`. **Subset gotcha**: when `data.subset.*` is active, `val_loader.dataset` is a `torch.utils.data.Subset` with no `img_paths`. Any callback accessing `dataset.img_paths` or `dataset.get_raw_item(idx)` must call `_unwrap_subset(dataset)` (defined in `callbacks.py`) to get `(underlying_dataset, index_map_fn)` and map indices through `Subset.indices`. **Class names gotcha**: use `getattr(trainer, "_loaded_data_cfg", trainer._data_cfg)` to get class names — `trainer._data_cfg` is the `data:` section of `06_training.yaml` which has no `names` key; `_loaded_data_cfg` is the resolved `05_data.yaml` which does.

**HF DETR-family models require `amp: false`** — D-FINE and RT-DETRv2 decoders overflow in fp16, producing NaN `pred_boxes` that crash `generalized_box_iou` inside the HF loss on the first forward pass. The NaN guard in `_train_one_epoch` cannot catch this (crash happens inside `forward_with_loss` before `loss.item()` is reached). Set `training.amp: false` in all D-FINE and RT-DETR training configs. YOLOX is unaffected.

**GPU augmentation (`training.gpu_augment`)** — when enabled, `build_dataloader()` uses `build_cpu_transforms()` (Mosaic/MixUp/CopyPaste + Resize + ToDtype, no Normalize) and the trainer applies `GpuDetectionTransform` after each batch is moved to device. Stateless transforms (RandomAffine, ColorJitter, Flips, Normalize) run on GPU via TVTensor dispatch. Float fill for GPU RandomAffine must be `114/255.0` (not `114`) because images arrive as float32 [0,1]. Do not add disk-I/O transforms to `build_gpu_transforms()`. Additional behaviour: (1) **Letterbox resize on GPU** — if a batch arrives at a size other than `input_size` (e.g. non-Mosaic paths), `GpuDetectionTransform` scales uniformly to preserve aspect ratio and pads the remainder with grey (114/255), then adjusts box coordinates analytically. No-op when already `input_size`. (2) **Contrast augmentation** — `GpuDetectionTransform` accepts `contrast: <float>` (default 0.0) under `augmentation:` in `06_training.yaml` (trainer passes `config.get("augmentation", {})` to `build_gpu_transforms`); uses vectorized luma-weighted mean adjustment on GPU. (3) **ColorJitter order randomized per batch** — brightness/contrast/saturation+hue are shuffled via `random.shuffle` each forward call. (4) **GPU augmentation runs inside `autocast`** — the trainer calls `gpu_transform` inside the AMP block, so augmentation executes in fp16 when `training.amp: true`, matching the forward-pass dtype and eliminating a mid-batch dtype boundary. (5) **CPU resize skipped when `mosaic: true`** — `build_cpu_transforms()` omits `v2.Resize` for Mosaic training paths because Mosaic always outputs `input_size`; the resize is retained for non-Mosaic training and all validation. (6) **`prefetch_factor` default is 4** — configurable via `data.prefetch_factor` in `06_training.yaml`; applies to the custom PyTorch backend only. HF Trainer (`training.backend: hf`) uses its own data pipeline and is unaffected by all of the above.

## Adding a new phase

1. Create `core/pNN_<name>/` with `__init__.py`, your implementation, and a CLI entry if user-facing (`run_<name>.py` convention).
2. If the CLI produces a run-dir: route through `utils.config.generate_run_dir()` using `feature_name_from_config_path()` for `<use_case>`. Accept an explicit override (CLI flag or state key) for callers that don't have a `features/<name>/configs/` ancestor.
3. If the phase needs CUDA: call `utils.device.auto_select_gpu()` near the top of the CLI, before `import torch`.
4. Register model variants (if any) via the decorators in `p06_models/` rather than editing `build_model`.
5. Add a test: `tests/test_pNN_<name>.py` following the patterns in `tests/CLAUDE.md`. If the phase slots into the canonical end-to-end chain, also add a stage to `tests/test_p12_raw_pipeline.py` so the full-pipeline smoke test exercises it.

## Out of scope for this file

- **Per-feature workflows**: see `features/README.md` and each feature's own README.
- **Keypoint top-down metrics + offline COCO AP**: see `core/p08_evaluation/keypoint_metrics.py` (numpy PCK/OKS + `compute_oks_ap_pycocotools` wrapper).
- **Config schemas + CLI override syntax**: see `configs/CLAUDE.md`.
- **Test-runner mechanics + fixtures**: see `tests/CLAUDE.md`.
- **External service start/health**: see `services/CLAUDE.md`.
- **Dataset provenance**: see `dataset_store/CLAUDE.md`.

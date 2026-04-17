# CLAUDE.md — cv-pipeline (Edge AI Computer Vision)

## Quick Start

```bash
# Install (use uv, not pip)
uv sync                        # Full p00→p10 pipeline + QA + LS + HPO + Gradio + Jupyter + Playwright + dev tools
uv sync --extra analysis       # Add FiftyOne + Cleanlab (heavy, optional)
bash scripts/setup-export-venv.sh   # Create .venv-export/ for quantized ONNX (optional)

# Auto-label a raw dir of JPEGs (no existing labels) via SAM3 → YOLO txt
uv run core/p01_auto_annotate/run_auto_annotate.py \
  --image-dir /path/to/raw_images \
  --classes "0:hand_with_glove" \
  --mode text \
  --text-prompts "hand_with_glove=a person's hand wearing a work glove"
# Writes labels next to images; report at <image_dir>/../auto_annotate_report/

# Prepare training-ready dataset (merge multi-source raw data, class-remap, stratified split)
uv run core/p00_data_prep/run.py --config features/safety-fire_detection/configs/00_data_preparation.yaml
uv run core/p00_data_prep/run.py --config features/safety-fire_detection/configs/00_data_preparation.yaml --dry-run

# Train
uv run core/p06_training/train.py --config features/safety-fire_detection/configs/06_training.yaml
uv run core/p06_training/train.py --config features/safety-fire_detection/configs/06_training.yaml \
  --override training.lr=0.005 training.epochs=100

# Evaluate
uv run core/p08_evaluation/evaluate.py \
  --model features/safety-fire_detection/runs/best.pt --config features/safety-fire_detection/configs/05_data.yaml

# Export to ONNX
uv run core/p09_export/export.py \
  --model features/safety-fire_detection/runs/best.pt \
  --training-config features/safety-fire_detection/configs/06_training.yaml

# Inference via the multi-tab Gradio demo (per-feature alert config is
# loaded from 10_inference.yaml by the feature tab)
uv run demo

# Scaffold a new feature (copies features/_TEMPLATE/, substitutes <feature_name>)
bash scripts/new_feature.sh my_new_feature
# then edit features/my_new_feature/configs/{05_data,06_training,10_inference}.yaml

# Tests
uv run tests/run_all.py              # Full pipeline (sequential, stops on failure)
uv run -m pytest tests/ -v           # Via pytest
uv run tests/test_p06_training.py    # Single test file
# Benchmark pretrained models against val split (outputs eval/benchmark_results.json)
uv run features/<feature>/code/benchmark.py --split val
```

## Architecture

```
core/
  p00_data_prep/       Format conversion (COCO/YOLO/VOC → YOLO)
  p01_auto_annotate/   SAM3 + text prompt auto-labeling
  p02_annotation_qa/   SAM3 quality assessment + scoring
  p03_generative_aug/  Synthetic data via SAM3 + Flux diffusion
  p04_label_studio/    Human review bridge (import/export)
  p05_data/            Dataset loaders (detection, classification, segmentation, keypoint)
  p06_models/          Model registry (YOLOX, D-FINE, RT-DETRv2, timm, HF)
  p06_training/        Training loops (PyTorch native, HF Trainer, custom)
  p07_hpo/             Optuna hyperparameter optimization
  p08_evaluation/      Metrics (mAP, mIoU, accuracy) + error analysis
  p09_export/          ONNX export + INT8 quantization
  p10_inference/       PyTorch + ONNX inference, pose, face, tracking
features/              Self-contained per-use-case folders (see features/README.md + features/CLAUDE.md)
                       Each feature has its own CLAUDE.md with checklist, benchmark results,
                       and training config template.
  <category-name>/     Names follow a `<category>-<name>` convention matching
                       docs/03_platform/: access-, ppe-, safety-, traffic-
                       (e.g. safety-fire_detection, safety-fall-detection,
                       ppe-helmet_detection, ppe-shoes_detection,
                       safety-poketenashi-phone-usage, access-face_recognition,
                       access-zone_intrusion, detect_vehicle). Uniform layout:
                       configs/, code/, samples/, notebooks/, tests/, runs/,
                       eval/, export/, predict/, release/
  _TEMPLATE/           Copy via scripts/new_feature.sh to scaffold new features
configs/_shared/       Shared pipeline templates (non-authoritative)
configs/_test/         CI test fixtures
services/              Microservices: SAM3, Flux, auto-label, QA (see services/CLAUDE.md)
tests/                 Integration tests with real data (see below)
app_demo/              Gradio demo UI (see app_demo/CLAUDE.md)
dataset_store/         raw/ + site_collected/ + training_ready/ — all datasets.
                       Downloads via MCP (Roboflow/Kaggle/HF), not bootstrap scripts.
                       See dataset_store/CLAUDE.md for per-source registry + v1 plan.
releases/              Versioned model releases.
                       `utils/release.py --run-dir <ts_dir>` →
                       releases/<dataset_name>/v<N>_<YYYY-MM-DD>/
                         {best.pth, *.onnx, 05_data.yaml, 06_training.yaml, model_card.yaml}
../smart_parking/      Sibling repo (split from this tree)
```

## Key Design Principles

- **Config-driven**: All hyperparameters from YAML. No hardcoded values. CLI `--override` for tuning.
- **Registry pattern**: Models, losses, metrics, postprocessors registered via decorators. Add new ones without editing core.
- **No unnecessary abstractions**: Lean code, framework-native features preferred (torchvision v2, HF built-in loss).

## Model Registry

```python
from core.p06_models import build_model
model = build_model(config)  # Dispatches by config["model"]["arch"]
```

| Arch key | Task | Framework |
|----------|------|-----------|
| `yolox-nano/tiny/s/m/l` | Detection | PyTorch native |
| `dfine-s/n/m` | Detection | HF Transformers |
| `rtdetr-r18/r50` | Detection | HF Transformers |
| `timm` | Classification | timm (any architecture via `timm_name`) |
| `hf-classification` | Classification | HF Transformers |
| `hf-segformer/mask2former/dinov2-seg` | Segmentation | HF Transformers |

## Training Backends

Set `training.backend` in YAML:
- **`pytorch`** (default): Custom trainer with EMA, SimOTA, per-component LR. YOLOX uses custom loss; HF/timm models use `forward_with_loss()`.
- **`hf`**: HuggingFace Trainer with DDP/DeepSpeed.
- **`custom`**: Dynamic import via `training.custom_trainer_class`.

## Config System

- Each feature has its full phase YAMLs in `features/<name>/configs/`:
  `00_data_preparation, 05_data, 06_training, 08_evaluation, 09_export, 10_inference`
- `10_inference.yaml` carries per-feature `alerts:` (thresholds, frame windows)
  — loaded via `core.p10_inference.video_inference.load_alert_config()`
- `configs/_shared/` holds non-authoritative templates; features never fall back
- `configs/_test/` holds CI test fixtures
- No inheritance between files — each is self-contained
- Paths are relative from project root (`../../dataset_store/`)

## Custom feature code

Three escape hatches, least → most invasive:

1. **Config-only**: stock `core/` handles it, `features/<name>/code/` is empty.
2. **Registry override**: custom class in `features/<name>/code/*.py`, reference
   via dotted path in YAML (e.g. `training.custom_trainer_class`).
3. **Fully custom pipeline**: `features/<name>/code/train.py` (any framework —
   Ultralytics, PaddleDetection, mmdetection). Must still read
   `features/<name>/configs/` and write to `runs/ eval/ export/ predict/`.

Rule: `code/` may import from `core/` and `utils/`; **`core/` must never
import from any `features/<name>/code/`**.

## Tests

- **Real data only** — no mocks. Uses `test_fire_100` dataset (100 images). Services (SAM3 :18100, QA :18105, auto-label :18104) skip gracefully when down.
- **40 test files** in four groups: `utils` (independent), `p00–p04` (annotation), `p05–p11` (train/eval/export/infer), `p12` (raw pipeline end-to-end).
- Test configs in `configs/_test/` — includes `00_raw_pipeline.yaml` (created at runtime by p12).
- `tests/run_all.py` runs sequentially; p08/p09/p10 depend on checkpoint from `p06_training` (`outputs/08_training/best.pth`).
- Each file also runs standalone: `uv run tests/test_p06_training.py`
- See `tests/CLAUDE.md` for the full file map, output dirs, fixture API, and gotchas.

## Gotchas

- **`uv` not `pip`**: Project uses uv with custom PyTorch CUDA 13.0 index. Always `uv run` or `uv sync`.
- **Bare `uv sync` installs everything** for the full pipeline (p00→p10, QA, Label Studio, HPO, Gradio, Jupyter, MediaPipe, pytest, ruff, dvc, Playwright). Only `--extra analysis` is opt-in (FiftyOne ~1 GB).
- **Quantized ONNX export needs a separate venv**: `optimum[onnxruntime]` requires `transformers<4.58` which conflicts with the git transformers pinned in the main venv. Run `bash scripts/setup-export-venv.sh` once to create `.venv-export/`, then use it only for quantization: `.venv-export/bin/python core/p09_export/export.py --optimize O2 --quantize dynamic ...`. The main venv's default export (`--skip-optimize`) still works for unquantized ONNX.
- **Feature folder vs `dataset_name` casing**: feature folders use `kebab-case` (e.g. `ppe-helmet_detection`); `dataset_name` in `05_data.yaml`, `training_ready/` subdirs, `releases/` subdirs, and LS project names use `snake_case` (hyphens replaced with underscores). `scripts/new_feature.sh` derives both correctly from a single `<feature_name>` arg. Every pipeline-step run_dir is derived via `utils.config.feature_name_from_config_path()` — never pass `dataset_name` where a feature folder is expected or you'll create ghost `features/<snake_name>/` folders.
- **`text_prompts:` live per-feature, not in shared QA config**: each feature's `05_data.yaml::text_prompts:` is the authoritative SAM3 prompt source. `core/p02_annotation_qa` reads from the data config first. Never add feature-specific prompts back to `configs/_shared/02_annotation_quality.yaml` — that shared file intentionally omits them.
- **`include_missing_detection` defaults to `false`**: the shared QA config sets `sam3.include_missing_detection: false` to prevent false-positive "unlabeled object" flags on class-restricted datasets (fire-only, helmet-only, etc.) where many non-target objects are intentionally unannotated. Enable only for COCO-style all-object datasets: `--override sam3.include_missing_detection=true`.
- **p02 auto-appends to `DATASET_REPORT.md`**: after each QA run, `run_qa.py` writes a "Label Quality" section into the feature's existing `DATASET_REPORT.md` (grade table, verdict, top issues). Re-running replaces that section only — other sections are preserved.
- **DVC for large files**: `.pt`, `.pth`, `.onnx` files are gitignored, tracked via DVC.
- **`sys.path.insert`**: Many modules add project root to path. Use `uv run` to avoid issues.
- **Overrides are nested dicts**: `DetectionTrainer(overrides={"training": {"epochs": 2}})`, not `{"training.epochs": 2}`.
- **Model on GPU after eval**: `ModelEvaluator` moves model to GPU. Call `model.cpu()` before ONNX export.

## Code Style

- Python 3.12+, ruff enforced (`E`, `F`, `I`, `UP`), line length 100
- Type hints on function signatures
- Private prefix `_` for internal classes/functions
- Constants: `UPPER_SNAKE_CASE`

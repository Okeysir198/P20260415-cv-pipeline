# CLAUDE.md — cv-pipeline (Edge AI Computer Vision)

## Quick Start

```bash
# Install (use uv, not pip)
uv sync                        # Full p00→p10 pipeline + QA + LS + HPO + Gradio + Jupyter + Playwright + dev tools
uv sync --extra analysis       # Add FiftyOne + Cleanlab (heavy, optional)
bash scripts/setup-export-venv.sh   # Create .venv-export/ for quantized ONNX (optional)
bash scripts/setup-paddle-venv.sh   # Create .venv-paddle/ for paddle backend (optional)

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

# Train (each feature has arch-specific configs: 06_training_{yolox,rtdetr,dfine}.yaml)
#   Every run auto-produces a unified observability tree (all tasks, both backends):
#     <save_dir>/
#       best.pt, test_results.json,
#       data_preview/   00_dataset_info, 01_dataset_stats, 02_data_labels_{train,val,test},
#                       03_aug_labels_train, 04_transform_pipeline
#       val_predictions/{epochs/, best.png, error_analysis/}
#       test_predictions/{best.png, error_analysis/}
#   error_analysis/ is flat-numbered 01..20 — every diagnostic at depth 0:
#     01 overview · 02 data_distribution · 03 distribution_mismatch · 04 label_quality
#     05 duplicates_leakage · 06 learning_ability · 07 per_class_performance
#     08 confusion_matrix · 09 confidence_calibration · 10 failure_mode_contribution
#     11 failure_by_attribute · 12 hardest_images · 13 failure_mode_examples/
#     14 robustness_sweep (det/cls/seg/kpt; per-task corruptions)
#     15..19 detection-only · 20 pixel_confusion_matrix (seg) | bbox_padding_sweep (kpt)
#     keypoint adds: 08 confusion_left_right · 09 confidence_vs_error
#     10 failure_mode_contribution · 13 failure_mode_examples/{high_error,ghost,swapped_pair}/
#     summary.md auto-iterates 01→20 with description + signal + suggested next step.
#   Authoritative name map: `CHART_FILENAMES` in core/p08_evaluation/error_analysis_runner.py.
#   No more sibling distribution_mismatch/, learning_ability/, or LQ_/DM_/LA_ prefixed files.
#   Details + opt-out knobs: core/p06_training/CLAUDE.md
uv run core/p06_training/train.py --config features/safety-fire_detection/configs/06_training_yolox.yaml
uv run core/p06_training/train.py --config features/safety-fire_detection/configs/06_training_yolox.yaml \
  --override training.lr=0.005 training.epochs=100
# Paddle backend (detection: PicoDet/PP-YOLOE) — separate world, sibling .venv-paddle/
# Train in paddle venv → export to ONNX → all downstream phases use main-venv ORT path.
bash scripts/setup-paddle-venv.sh                                    # one-time
.venv-paddle/bin/python core/p06_paddle/train.py \
  --config features/safety-fire_detection/configs/06_training_picodet.yaml
.venv-paddle/bin/python core/p06_paddle/export.py \
  --config <same.yaml> --checkpoint <run>/best.pdparams --out <run>/model.onnx
# Then eval/infer/demo use the .onnx via the standard main-venv path:
uv run core/p08_evaluation/evaluate.py --model <run>/model.onnx --config <05_data>.yaml

# Evaluate
uv run core/p08_evaluation/evaluate.py \
  --model features/safety-fire_detection/runs/<ts>/best.pth --config features/safety-fire_detection/configs/05_data.yaml

# Export to ONNX
uv run core/p09_export/export.py \
  --model features/safety-fire_detection/runs/<ts>/best.pth \
  --training-config features/safety-fire_detection/configs/06_training_yolox.yaml

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
notebooks/             Reference fine-tune ports — runnable upstream baselines for
                       diff-against-our-pipeline debugging. Each holds a `reference_<arch>/`
                       (HF cookbook/docs as `.py`) + `our_<arch>/` (same recipe through
                       core/p06_training/). Currently: detr_finetune_reference/,
                       segformer_finetune_reference/, image_classification_finetune_reference/,
                       vitpose_finetune_reference/. Run via `.venv-notebook/bin/python`,
                       NOT `uv run`. See per-folder CLAUDE.md.
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
- **Registry pattern**: Models, losses, metrics, postprocessors registered via decorators on a shared `utils.registry.Registry` base. Add new ones without editing core.
- **No unnecessary abstractions**: Lean code, framework-native features preferred (torchvision v2, HF built-in loss).
- **One viz path**: All image annotation (boxes, labels, keypoints, polygons, banners, grids) goes through `utils/viz.py` — `supervision`-backed, styled via `VizStyle` (defined in `core/p10_inference/supervision_bridge.py`, re-exported from `utils.viz`; always import as `utils.viz.VizStyle`). `cv2.rectangle/putText/circle/line/polylines/fillPoly/addWeighted` are reserved for I/O, color conversion, and scene synthesis in `app_demo/demo_videos/`. Matplotlib stays for statistical plots (`p08_evaluation/`); a single `apply_plot_style()` unifies rcParams across all of them.

## Model Registry

```python
from core.p06_models import build_model
model = build_model(config)  # Dispatches by config["model"]["arch"]
```

| Arch key | Task | Framework |
|----------|------|-----------|
| `yolox-nano/tiny/s/m/l` | Detection | PyTorch native |
| `dfine-s/n/m/l` (aliases: `dfine-large`) | Detection | HF Transformers |
| `rtdetr-r18/r50` | Detection | HF Transformers |
| `timm` | Classification | timm (any architecture via `timm_name`) |
| `hf-classification` | Classification | HF Transformers |
| `hf-segformer/mask2former/dinov2-seg` | Segmentation | HF Transformers |
| `hf_keypoint` | Keypoint (top-down) | HF Transformers (ViTPose family) |

Paddle archs (PicoDet, PP-YOLOE) are NOT in this unified registry — they live
in their own package (`core/p06_paddle/`) and run in `.venv-paddle/`. See
`core/p06_paddle/CLAUDE.md`.

## Training Backends

Set `training.backend` in YAML:
- **`pytorch`** (default): Custom trainer with EMA, SimOTA, per-component LR. YOLOX uses custom loss; HF/timm models use `forward_with_loss()`.
- **`hf`**: HuggingFace Trainer with DDP/DeepSpeed.
- **`custom`**: Dynamic import via `training.custom_trainer_class`.
- **`paddle`**: Selecting `backend: paddle` in the unified dispatcher prints a redirect — paddle is NOT handled by `core/p06_training/train.py`. Train via `.venv-paddle/bin/python core/p06_paddle/train.py` (separate package, separate venv, drives upstream `ppdet.engine.Trainer` directly). Convergence with the rest of the pipeline happens at ONNX (`core/p06_paddle/export.py` → main-venv ORT path). Setup: `bash scripts/setup-paddle-venv.sh`. v1 = detection only (PicoDet, PP-YOLOE). See `core/p06_paddle/CLAUDE.md`.

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
- **Benchmark scripts need `if __name__ == '__main__':`**: The DataLoader uses `forkserver` multiprocessing. Any script that creates a DataLoader must guard top-level code with `if __name__ == '__main__':` or forkserver worker spawning fails with `BrokenPipeError`. Write benchmarks as `.py` files, not inline stdin heredocs.
- **GPU augmentation split**: Set `training.gpu_augment: true` in `06_training.yaml` to move RandomAffine/ColorJitter/Flips/Normalize to GPU (**2.8x–3.2x faster**, scales with batch size). Mosaic, MixUp, CopyPaste, IRSimulation **must stay on CPU** — they call `dataset.get_raw_item()` which does disk I/O. GPU augmentation uses batched `F.affine_grid + F.grid_sample` (one CUDA call for all B images); box correctness via `M_fwd = inv(M_inv)` corner-transform + area filter — no tv_tensors in the GPU path. Additional details: (1) `GpuDetectionTransform` accepts a `contrast` float (config key `augmentation.contrast`, default 0.0) for vectorized luma-weighted mean contrast on GPU. (2) If images arrive at ≠ `input_size`, a GPU letterbox resize runs first — uniform scale, grey padding (114/255), box coords adjusted analytically; no-op for Mosaic output which is already `input_size`. (3) ColorJitter order (brightness/contrast/saturation+hue) is randomized per batch. (4) When `mosaic: true`, CPU `v2.Resize` is skipped — Mosaic guarantees `input_size` output. (5) `prefetch_factor` defaults to **4** (set via `data.prefetch_factor`; important for Mosaic which does 4 disk reads/sample). (6) `gpu_transform` is inside the `autocast` block — GPU augmentation runs in **fp16 when AMP is enabled**, no dtype boundary mid-batch. **HF Trainer note**: all of the above applies only to `training.backend: pytorch`; HF Trainer uses its own data pipeline and is unaffected.
- **YOLOX input range is [0, 255]** — Megvii `yolox_*.pth` weights expect raw pixel values in [0, 255], NOT [0, 1]-normalized. Dividing by 255 produces near-zero confidence scores with no other error. All other models (D-FINE, RT-DETRv2, SCRFD, timm, HF) use normalized [0, 1] + mean/std. Affects `_preprocess_yolox()` in any inference script.
- **Dual YOLOX implementations**: `model.impl` in `06_training.yaml::model:` selects the YOLOX backend. `custom` (default) uses the in-repo `YOLOXModel` + `YOLOXLoss` (SimOTA) — works in the main `.venv/`. `official` uses the upstream Megvii `yolox` package via `_OfficialYOLOXAdapter` (`core/p06_models/yolox.py`) — requires a separate venv: `bash scripts/setup-yolox-venv.sh` creates `.venv-yolox-official/`; then `.venv-yolox-official/bin/python core/p06_training/train.py --config ... --override model.impl=official`. Trainer loop, GPU-aug, EMA, fast-val are unchanged — adapter implements the `forward_with_loss()` hook already used by HF detection models. Caveats: (1) set `augmentation.normalize: false` for official (expects [0,255] inputs); (2) val does two forwards per batch (loss vs predictions) — roughly 2x val wall-time; (3) adapter uses default 2-group param split, not the custom path's 6-group per-component-LR split; (4) ONNX export + `p08`/`p10` consumers still go through the custom layout — do not load official-trained `.pt` into those paths without an adapter-aware loader. **Parity verified**: `scripts/compare_yolox_impls.py` loads `pretrained/yolox_m.pth` into both paths and compares raw outputs — bit-identical on xy/wh/obj/cls, mean IoU 1.0 on top detections.
- **YOLOX decode formula — no half-grid offset** (fixed from earlier `+ 0.5`): `YOLOXModel.forward` decodes xy as `(grid + raw_xy) * stride`, matching upstream Megvii YOLOX. The previous `grid + 0.5 + raw_xy` was self-consistent when training from scratch (the reg head absorbed the 0.5 bias) but produced a half-grid-cell shift (up to 16 px on stride 32) when loading Megvii pretrained weights for inference. Any custom-YOLOX checkpoint trained on the old `+0.5` formula will produce boxes shifted by up to 16 px on stride 32 under the new code — retrain if accurate boxes matter. wh/obj/cls are unaffected.
- **Multi-GPU training/HPO**: Use `CUDA_VISIBLE_DEVICES=N uv run ...` to target a specific GPU. Do NOT use `--device cuda:N` — `auto_select_gpu()` restricts `CUDA_VISIBLE_DEVICES` before that flag is parsed, causing "invalid device ordinal" errors.
- **HPO speed**: Disable all viz callbacks and subset both splits to keep trials fast: `--override training.data_viz.enabled=false training.aug_viz.enabled=false training.val_viz.enabled=false data.subset.train=0.05 data.subset.val=0.10`. For YOLOX HPO also add `augmentation.mosaic=false` — mosaic is CPU-bound (4 disk reads/sample) and dominates trial time.
- **Fast validation during training**: `training.val_full_interval: N` runs full val every N epochs (drives checkpoint/ES/scheduler); quick val (20% subset via `training.val_subset_fraction`) runs every epoch for logging. Set `val_full_interval: 0` to full-val every epoch. Both keys are present in all `06_training_*.yaml` configs.
- **Quick val overstates mAP vs full val — use `val_full_interval: 0` for production runs that select a released checkpoint**: best.pth is saved on whichever val runs that epoch. At `val_full_interval: 5` (the default), intermediate epochs only see the 20% quick-val subset, which typically runs a few mAP points higher than full val (smaller sample → easier/noisier). Best-epoch selection can land on a lucky quick-val peak that is *not* the true full-val best. For arch comparison / HPO where relative ordering matters and wall-time is tight, fast val is fine; for the *final* training run that produces the release checkpoint, set `val_full_interval: 0` so checkpoint selection uses ground-truth numbers. Accept ~2× per-epoch wall time in exchange.
- **`p08/evaluate.py` preprocessing parity**: the evaluator now uses `YOLOXDataset` + `build_transforms(is_train=False)` — same pipeline as training val (stretch-resize + ImageNet normalize). Previously used a `_MinimalDetectionDataset` that letterboxed + only `/255`, producing a 5× mAP gap vs training's internal val (0.09 instead of 0.45). Targets are tensors `(N, 5)` with `[cls, cx, cy, w, h]` normalized — `_gt_to_dict` in `_get_predictions_detection` converts them to the `{boxes, labels}` xyxy-pixel dicts `compute_map` expects. If you add a new detection arch that returns raw tensors, make sure `_dispatch_postprocess` knows about it or p08 will still silently mis-evaluate.
- **TTA available via `scripts/yolox_tta_eval.py`**: multi-scale (default 512/640/768) × optional h-flip inference over val, reports mAP@0.5. Typical gain on scale-variant classes (smoke, small objects) is ~10% mAP@0.5; deployment cost is 6 inference passes per image (~6× slower). Invariants: use the SAME `build_transforms(is_train=False)` as the evaluator so inputs are ImageNet-normalized; downscale the normalized tensor with `F.interpolate` (no need to renormalize after resize); unflip boxes with `x ← W - x`, then scale back to `ref_size` before merging, then class-wise NMS.
- **RT-DETRv2: HPO-derived LR often does not generalize from subset → full data**. An LR tuned on 5%-data short runs can be 1.5–2× too hot for full-data long runs, producing post-warmup divergence (val mAP peaks early then collapses). Use conservative settings for small-class fine-tune: `lr=5e-5`, `warmup_steps=300`, `epochs=30–40`, `patience=50`, `val_full_interval=0`, `bf16=true`, `amp=false`. This matches HF/Roboflow canonical recipes and the reference notebooks in `notebooks/detr_finetune_reference/`. Confirm with a short smoke (`data.subset.train=0.2`) before committing to full data.
- **YOLOX postprocess must not re-sigmoid** (regression guard): both the custom `_DecoupledHead` (`yolox.py:555-559`) and the official adapter (`decode_in_inference=True`) sigmoid obj+cls inside the model's eval-mode forward. `_postprocess_yolox` (`p06_training/postprocess.py`) must **not** apply sigmoid again — doing so squashes every score into [0.25, 0.55] and makes `conf_threshold` meaningless. mAP is unaffected (sigmoid is monotonic → preserves PR ordering) but val-prediction visualizations become "green walls" (thousands of boxes per image) and production deployment confidences stop being real [0, 1] probabilities. If you add a new YOLOX-style head, respect this invariant.
- **Dataset stats cache**: `generate_dataset_stats()` (now task-aware: detection/cls/seg/kpt) skips recompute if `<save_dir>/data_preview/01_dataset_stats.{json,png}` already exist. Delete those files to force a refresh.
- **Visualization defaults live in one place**: `utils.viz.VizStyle` is the single source of truth — palette, auto-thickness, label scale, error colors (TP/FP/FN/duplicate/bg_FP), grid cell size, banner style, keypoint radius, zone fill alpha. Class is defined at `core/p10_inference/supervision_bridge.py:241` and re-exported through `utils/viz.py`; always reference it via `utils.viz.VizStyle` from caller code. Override per-run via the `visualization:` (alias `viz:`) block in any training/inference config. Do not edit callsites to change colors/thickness; add a knob to `VizStyle` instead.
- **Viz invariants worth knowing** (load-bearing defaults that look optional but aren't): (1) `_smart_class_xticks()` in `core/p05_data/run_viz.py` auto-rotates x-tick labels to 90° when `n_classes > 12` — applied at every per-class bar chart (det class instances, cls images-per-class, seg pixel-count/components/area, kpt skeleton-edge). Bypassing it produces unreadable overlap on COCO-style 80-class datasets. (2) `classification_banner` in `utils/viz.py` upscales any image < 192 px wide before drawing and auto-shrinks/ellipsizes the banner text — required because EuroSAT-style 64-px thumbnails would otherwise have the banner cover ~38% of the cell. (3) `04_transform_pipeline.png` for segmentation overlays the *transformed mask* per-stage (`_overlay_segmentation` per-class palette), not just the image — that's the whole point of the chart for seg. (4) Top-down keypoint walker in `transform_pipeline_viz.py` runs `KeypointTopDownDataset.__getitem__` to get the post-transform tensor; the bottom-up `KeypointTransform` path expects `{boxes, keypoints}` and will silently skip every stage after Raw if fed a top-down dict. (5) `confident_disagreement_gallery` in `core/p08_evaluation/label_quality.py` uses `cols=min(4, …)` for cls/seg/det — 2 columns squeeze the GT|Pred banners into illegibility on small thumbnails.
- **DETR-family reference notebooks live in `.venv-notebook/`** — `notebooks/detr_finetune_reference/` contains byte-for-byte ports of qubvel's RT-DETRv2 + D-FINE fine-tune notebooks as runnable `.py` scripts, used as a known-good baseline when debugging our in-repo DETR training. Setup: `bash scripts/setup-notebook-venv.sh` creates `.venv-notebook/` with `albumentations==1.4.6` pin. Do **NOT** run via `uv run` (main venv has a newer albumentations with different box-clip semantics). See `notebooks/detr_finetune_reference/CLAUDE.md` for setup, phase 1/2 plan, and conversion gotchas (Jupyter magics stripped, `display()` commented, `datasets` 4.x access-pattern fix).
- **D-FINE requires `bf16: false`, RT-DETRv2 is bf16-neutral** — D-FINE's distribution-focused loss stalls val mAP at ~0.15 through ep11 under bf16 (observed on `our_dfine_albumentations/` CPPE-5 runs; eval_loss climbs 2.2→2.9 = divergence). Fp32 restores convergence to test mAP ≈ 0.44. RT-DETRv2's vanilla regression head is bf16-safe (verified ±0.01 on CPPE-5). **Now enforced**: `_validate_hf_backend_config` (`core/p06_training/hf_trainer.py`) hard-errors when `arch.startswith("dfine")` AND `training.bf16: true`. `training.amp: true` (fp16) still overflows both — stay on bf16 (RT-DETRv2 only) or fp32.
- **D-FINE is under-trained at qubvel's 30 epochs — default to 50 on small datasets**: on CPPE-5, bumping `training.epochs: 30 → 50` under otherwise-identical recipe lifted test mAP **0.430 → 0.492 (+0.062)**, with Goggles (rare class) nearly tripling 0.096 → 0.307. Qubvel's published 0.449 is beaten at a single seed once D-FINE is allowed to fully converge. RT-DETRv2 does not need this — it peaks in 30-40 ep and over-trains thereafter. D-FINE's distribution-focused regression head continues refining the box-corner distributions for many epochs after the classification head has plateaued, so the gain comes mostly from rare/hard classes. Both `notebooks/detr_finetune_reference/our_dfine_{torchvision,albumentations}/06_training.yaml` default to 50 epochs for this reason.
- **HF detection models auto-populate `id2label`/`label2id` from `data.names`** — `core/p06_models/hf_model.py::build_hf_model` reads the resolved 05_data.yaml's `names:` dict and passes `id2label`/`label2id` to `from_pretrained`. Required by HF cookbook convention; missing these dicts prevented class-embedding memorization in single-batch overfit tests (fire class top-1 score jumped 0.118 → 0.993 after the fix). Defaults to `class_0, class_1, …` if no names are present. Override via `model.names: {0: foo, 1: bar}` in the training config or rely on the data config.
- **HF-backend checkpoints save with `hf_model.` key prefix** — `_DetectionTrainer._save` writes `hf_model.model.*` keys; naive `load_state_dict(..., strict=False)` silently keeps random init. **Now handled centrally**: `utils.checkpoint.strip_hf_prefix(state_dict)` runs automatically inside `core/p08_evaluation/evaluate.py`, `core/p09_export/export.py::_load_model_from_checkpoint`, and `core/p10_inference/predictor.py::_load_pytorch_model` — each also warns when unexpected-key counts > 0. New reload sites must call this helper *before* `load_state_dict`. `AutoModelForObjectDetection.from_pretrained(run_dir)` is still affected (HF reads the bin without our helper); `core/p10_inference/model_adapter.py` strips into a temp dir before `from_pretrained` — copy that pattern for any new HF-direct loader. Optimum export through `main_export` has the same hazard.
- **HF Trainer ghost-folder footgun: `logging.run_name` becomes the feature folder**. `_config_to_training_args` (`core/p06_training/hf_trainer.py`) derives `output_dir` via `generate_run_dir(run_name or project or config_path.parent.parent.name, ...)` — `run_name` wins if set. A stale `logging.run_name: <anything>` in a `06_training_*.yaml` creates `features/<run_name>/runs/...` instead of landing under the actual feature folder. **Now warned, not silent**: a mismatch logs a `WARNING` at trainer startup pointing at the actual feature folder. Behavior is preserved (the run still lands under `features/<run_name>/runs/`) so callers who *want* a custom name aren't broken — but stale leftovers surface immediately. Pytorch backend uses `feature_name_from_config_path()` directly and is immune. All 10 DETR configs (5 RT-DETR + 5 D-FINE) had stale names dropped on 2026-04-21.
- **HF Trainer wandb fallback (auto-applied)**. Without `wandb login` the HF Trainer's wandb callback used to raise `UsageError` during `trainer.train()` setup, killing the run before epoch 1. **Now auto-handled**: `_wandb_credentials_available()` (`core/p06_training/hf_trainer.py`) checks `WANDB_API_KEY` env + netrc; if absent and wandb was in `report_to`, drops it and logs a warning, keeping tensorboard. To opt out of the fallback (i.e. force a hard failure on missing creds), set `logging.report_to` explicitly. Pytorch backend's `WandBLogger` was already lenient. Tensorboard `SummaryWriter` background-thread races on a missing `<output_dir>/runs/<date-host>/` subdir are unrelated and still occur — set `logging.report_to: none` if you want zero callbacks.
- **`optimum-onnx` (not `optimum` itself) pins `transformers==4.57`** — conflicts with main venv's git transformers (5.5). Keep optimum-onnx export work in `.venv-export/`. Cross-venv checkpoint reload is **not** transparent: transformers 5.5 renamed 52 detection-model layer keys vs 4.57, so a main-venv-trained DETR checkpoint won't fully load under export venv's tf 4.57 (52 missing + 52 unexpected) — Optimum then exports a partially-random ONNX. For DETR-family, prefer `torch.onnx.export(..., dynamo=False)` directly in the main venv (dynamo mode fails on tf 5.5's `aten._is_all_true`), followed by `onnxruntime.quantization.quantize_static` for INT8 — both are version-agnostic.
- **INT8 ONNX is slower than fp32 on ORT CUDA EP for DETR-family** — ORT CUDA has no real INT8 kernels for transformer attention/normalisation ops; emulation is often slower than fp32. Real INT8 speedup needs TensorRT EP engine build. Calibration runs on CPU EP and saturates cores for minutes on transformer models. **Now guarded**: `core/p09_export/quantize.py::quantize_static` defaults `calibration_method="percentile"`, auto-excludes `LayerNormalization/Softmax/Gather` from quantization when an arch hint or filename contains `dfine`/`rtdetr`/`detr`, and **hard-errors on `--calibration-method minmax` for DETR-family** (collapsed mAP→0 on D-FINE). `scripts/benchmark_detr_fp32_int8_vs_pytorch.py --strict` exits nonzero when INT8 isn't viable (≥1.2× speedup AND mAP drop ≤0.01). Use `percentile` (default) or `entropy`; YOLOX/timm/seg do not need the exclude list.
- **ONNX inference: TensorRT EP is opt-in, not default** — `core/p10_inference/predictor.py::DetectionPredictor` accepts `providers: list[str] | None`; default stays `["CUDAExecutionProvider"]` for backward compat. Pass `providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"]` to opt into TRT (ORT auto-falls-back to CUDA at session creation if TRT runtime libs aren't on `LD_LIBRARY_PATH`). Invalid provider names raise immediately. `onnxruntime-gpu` (already pinned) ships both EPs — no extra dep required. CPU EP remains forbidden by policy.
- **Keypoint top-down (`hf_keypoint`) emits a different sample shape** — `KeypointTopDownDataset.__getitem__` returns `{pixel_values, target_heatmap, target_weight}` (per-person crops with Gaussian heatmap targets), NOT the `{boxes, keypoints}` full-frame YOLO-pose dict that `KeypointDataset` returns. Anything new that consumes a keypoint dataset needs to branch on the type or you'll silently hit detection-shaped postprocess paths that produce zero predictions. The HF Trainer dispatcher (`_build_datasets`) and the post-train runner (`_run_topdown_keypoint_post_train`) are the canonical examples.
- **Paddle backend gotchas** (full overview in "Training Backends" above): never `uv add paddlepaddle-gpu` — its bundled CUDA clashes with main venv's CUDA 13 PyTorch wheels. `paddle2onnx` is `.venv-paddle/`-only (separate from `.venv-export/`'s optimum-onnx pin). For ORT INT8: PicoDet / PP-YOLOE (CNN) generally hit real speedup; any future transformer-y paddle arch should reuse the DETR-family exclusion list in `core/p09_export/quantize.py`. PP-TinyPose is top-down — feed `KeypointTopDownDataset`, not `KeypointDataset` (same constraint as `hf_keypoint`; bottom-up silently skips every transform stage past Raw).
- **`pycocotools` APm/APl needs real source bboxes** — every top-down crop has identical area (input_h × input_w), so passing the crop bbox to `compute_oks_ap_pycocotools` collapses every person into a single area bucket and APm/ARm come back as -1.0. `_collect_source_bboxes(...)` in `core/p06_training/hf_callbacks.py` reads the YOLO label row + source image dims out of `KeypointTopDownDataset._index` to recover real source-pixel bboxes. New keypoint metric callers must do the same or their challenge-shape AP numbers will be misleading.

## Code Style

- Python 3.12+, ruff enforced (`E`, `F`, `I`, `UP`), line length 100
- Type hints on function signatures
- Private prefix `_` for internal classes/functions
- Constants: `UPPER_SNAKE_CASE`

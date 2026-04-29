# CLAUDE.md — tests/

## Running Tests

```bash
# Full sequential pipeline (stops on first failure — later tests depend on earlier outputs)
uv run tests/run_all.py

# Single test file
uv run tests/test_p06_training.py

# Via pytest (alternative)
uv run -m pytest tests/test_p06_training.py -v
```

## Test Layout

52 files in two groups: independent **utils + viz** (fast) and **sequential pipeline** (p00→p12). **No mocks** — real data only. Full `run_all.py` takes ~17 min on one RTX-class GPU (verified 2026-04-14, 1021s).

### Utils (independent, always fast)

| File | Tests |
|------|-------|
| `test_utils00_config_device_metrics.py` | Config loading, device detection, bbox metrics, visualization |
| `test_utils01_supervision_metrics.py` | `compute_map`, precision/recall via supervision |
| `test_utils02_exploration.py` | `explore_dataset`, `compute_normalization`, image/annotation stats |
| `test_utils03_scaffold.py` | `build_05_data_yaml`, `build_06_training_yaml`, experiment script generation |
| `test_utils04_keypoint.py` | Keypoint geometry helpers (angle, distance, orientation, fall ratio) |
| `test_utils05_langgraph.py` | Reducers, batch state helpers, JSON serialisation of numpy types |
| `test_utils06_progress.py` | `ProgressBar`, `TrainingProgress` (best-metric tracking, context-manager) |
| `test_utils07_paddle_bridge.py` | YOLO → COCO JSON converter (end-to-end on synthetic tree) |
| `test_utils08_release.py` | Version bump, use-case detection, full release promotion |
| `test_utils09_yolo_io.py` | `pil_to_b64` round-trip, YOLO label parsing edge cases |
| `test_viz_helpers.py` | Smoke tests for `utils.viz` helpers (synthetic data, no GPU) |
| `test_data_preview_fidelity.py` | `generate_dataset_stats` + `write_dataset_info` fidelity across det/cls/seg/kpt on tmp_path fixtures |
| `test_transform_pipeline_viz.py` | `04_transform_pipeline.png` renderer + denormalize inverse invariant |
| `test_p06_aug_benchmark.py` | CPU vs GPU augmentation throughput — **standalone perf benchmark, not in `run_all.py`**; run with `uv run tests/test_p06_aug_benchmark.py` |

### Pipeline (sequential — run in order via `run_all.py`)

| File | Pipeline Stage | Service Deps | Outputs |
|------|---------------|--------------|---------|
| `test_p00_data_prep.py` | Format conversion (COCO/VOC/YOLO → YOLO) | — | — |
| `test_p01_auto_annotate.py` | Auto-annotation (NMS, label writer, scanner, graph) | :18104 (optional) | — |
| `test_p02_annotation_quality.py` | Annotation QA (sampler, scorer, SAM3 verify) | :18105, :18100 (optional) | `outputs/12_annotation_qa/` |
| `test_p03_generative_augment.py` | Generative augment (Inpainter, graph) | — | — |
| `test_p04_label_studio.py` | Label Studio bridge (YOLO↔LS format, API) | — | — |
| `test_p05_create_data.py` | Fixture data validation | — | — |
| `test_p05_data_exploration.py` | Dataset exploration (`explore()`, `compute_channel_stats`) | — | `outputs/01_exploration_stats.json` |
| `test_p05_detection_dataset.py` | `YOLOXDataset`, transforms, dataloader | — | `outputs/01_detection_dataset/` |
| `test_p05_classification_dataset.py` | Classification dataset (folder + label-file layouts) | — | — |
| `test_p05_segmentation_dataset.py` | Segmentation dataset, transforms, collate | — | — |
| `test_p05_keypoint_dataset.py` | Keypoint dataset (YOLO-pose format) | — | — |
| `test_p05_coco_dataset.py` | COCO detection dataset, getitem, dataloader | — | — |
| `test_p05_augmentation_preview.py` | Augmentation preview helpers | — | — |
| `test_p06_model_registry.py` | Model registry, `build_model()`, decorator registration | — | — |
| `test_p06_model_variants.py` | YOLOX/D-FINE/RT-DETRv2 — 3-epoch train, loss decreases | — | `outputs/07_model_variants/` |
| `test_p06_training.py` | `DetectionTrainer` — 2-epoch real training, checkpoint | — | **`outputs/08_training/`** ← p08/p09/p10 read from here |
| `test_p06_training_features.py` | Schedulers, grad accumulation, loss, metrics | — | — |
| `test_p06_training_hf_detection.py` | HF Trainer backend — 1 epoch end-to-end (validator, collator, viz bridge, test eval) | — | — |
| `test_p06_classification_training.py` | timm + HF classification training | — | — |
| `test_p06_segmentation_metrics.py` | mIoU, per-class metrics | — | — |
| `test_p06_segmentation_training.py` | Segmentation training (SegFormer) | — | — |
| `test_p06_training_paddle_det.py` | Paddle PicoDet-S full chain (setup -> train -> eval -> export -> infer) | — | — |
| `test_p06_training_paddle_cls.py` | Paddle PP-LCNet — **skipped in v1** (paddle backend = detection only) | — | — |
| `test_p06_training_paddle_seg.py` | Paddle PP-LiteSeg — **skipped in v1** | — | — |
| `test_p06_training_paddle_kpt.py` | Paddle PP-TinyPose — **skipped in v1** | — | — |
| `test_p06_val_prediction_logger.py` | `ValPredictionLogger` — grid viz of GT vs predictions | — | — |
| `test_p07_hpo.py` | Optuna HPO — 2 trials × 1 epoch | — | `outputs/06_hpo/` |
| `test_p08_evaluation.py` | `ModelEvaluator` — mAP, per-class AP | — | `outputs/10_evaluation/metrics.json` |
| `test_p08_error_analysis.py` | `ErrorAnalyzer` — FP/FN/localization breakdown | — | `outputs/11_error_analysis/` |
| `test_p09_export.py` | ONNX export, `onnx.checker` validation | — | **`outputs/12_export/model.onnx`** ← p10 reads from here |
| `test_p09_export_validation.py` | PyTorch vs ONNX numerical match | — | — |
| `test_p10_inference.py` | `DetectionPredictor` (.pt / .pth + .onnx), batch predict | — | `outputs/14_inference/` |
| `test_p10_video_inference.py` | `VideoProcessor`, frame counter, alert config | — | — |
| `test_p10_face_recognition.py` | Face registry, gallery, predictor (optional ONNX weights) | — | — |
| `test_p10_zone_intrusion.py` | Zone Intrusion detector — geometry helpers, dataclasses, full detect+draw | — | — |
| `test_p10_poketenashi.py` | Poketenashi pose rules (hands-in-pockets, stair safety, handrail, pointing-calling) on real COCO-17 keypoints | — | — |
| `test_p11_e2e_pipeline.py` | Full chain: train → eval → export → infer | — | `outputs/15_e2e_pipeline/` |
| `test_p12_raw_pipeline.py` | **Raw dataset → annotate → QA → LS roundtrip → p00 merge+split → train → HPO → eval → export → infer** | :18104, :18105, :18100, :18103 | `outputs/16_raw_pipeline/` |

### Checkpoint dependency chain

```
test_p06_training  →  outputs/08_training/best.pth
                           ↓
                  test_p08_evaluation
                  test_p08_error_analysis
                  test_p09_export  →  outputs/12_export/model.onnx
                                           ↓
                                  test_p09_export_validation
                                  test_p10_inference
                                  test_p10_video_inference
```

## Fixtures

```
tests/fixtures/
  __init__.py         real_image(), real_image_bgr_640(), real_image_with_targets(),
                      real_image_b64(), data_config(), train_config(), class_names()
  data/
    train/images/     10 fire images (checked into git)
    train/labels/     YOLO .txt files
    val/images/       5 fire images
    val/labels/
```

Use `from fixtures import real_image` — never `np.random` or `np.zeros` for inference tests.

## Test Configs (`configs/_test/`)

| Config | Dataset | Purpose |
|--------|---------|---------|
| `05_data.yaml` | `test_fire_100` (100 images) | Detection training/eval default |
| `05_data_ppe.yaml` | `test_ppe_100` | Rule-based auto-annotation (has `auto_label` section) |
| `05_data_shoes.yaml` | `test_shoes_detection_100` | Rule-based auto-annotation |
| `05_data_fire.yaml` | `test_fire_detection_100` | Fire, no `auto_label` section |
| `05_data_phone.yaml` | `test_phone_detection_100` | Phone, no `auto_label` section |
| `05_data_fall.yaml` | `test_fall_100` | Fall classification |
| `05_data_segmentation.yaml` | `test_segmentation` | Segmentation training |
| `00_raw_pipeline.yaml` | `outputs/test_raw_pipeline` | Created at runtime by p12; no labels initially |
| `06_training.yaml` | — | YOLOX-M training (640×640, fire) |
| `06_training_segmentation.yaml` | — | SegFormer segmentation |

## External Services (graceful skip)

Tests with service dependencies skip cleanly when the service is down — they `print("SKIP: ...")` and `return` (not raise). No test failure.

| Service | Port | Used by |
|---------|------|---------|
| SAM3 | :18100 | `test_p02_annotation_quality.py`, `test_p12_raw_pipeline.py` |
| QA service | :18105 | `test_p02_annotation_quality.py`, `test_p12_raw_pipeline.py` |
| Auto-label | :18104 | `test_p01_auto_annotate.py`, `test_p12_raw_pipeline.py` |

Health check pattern used in tests:
```python
import httpx
def has_service(url): return httpx.get(f"{url}/health", timeout=3).status_code == 200
```

## Writing New Tests

1. **Naming**: `test_p{NN}_{description}.py` (pipeline stage) or `test_utils{NN}_{description}.py`
2. **Runner**: include `if __name__ == "__main__": run_all([...], title="...")` — same file works standalone and via pytest
3. **Real data**: use fixture images or `dataset_store/test_fire_100/`; never synthetic arrays for inference
4. **Service deps**: check availability with `has_*_service()`, skip gracefully with `print("SKIP: ..."); return`
5. **Outputs**: write to `tests/outputs/{NN}_{name}/`; gitignored — later tests read from here
6. **Register**: add filename to `CORE_TESTS` list in `tests/run_all.py`
7. **Cross-stage state** (for multi-stage tests like p12): use module-level `_state = {}` dict — avoids pytest fixture dependency, works with standalone `_runner.py`

## Gotchas

- **`_runner.run_test()` catches exceptions** → a clean `return` marks PASS; skip is informational only
- **p08/p09/p10 skip if no checkpoint** — run `test_p06_training.py` first if running individually
- **p12 `_state`**: `test_p12_raw_pipeline.py` threads checkpoint and ONNX paths through a module-level `_state` dict across 11 sequential test functions — don't run functions out of order
- **`YOLOXDataset` not `DetectionDataset`** — the class is `core.p05_data.detection_dataset.YOLOXDataset`
- **`resolve_split_dir(config, split, config_dir)`** — takes the full config dict + split name + config dir Path, not a dataset path

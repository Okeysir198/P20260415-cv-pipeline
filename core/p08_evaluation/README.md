# p04_evaluation — Evaluation + Metrics + Visualization

## Purpose

Evaluate trained models on validation/test sets, compute mAP and per-class metrics, generate visualizations.

## Files

| File | Purpose |
|---|---|
| `evaluator.py` | `ModelEvaluator` — batched inference on a dataset split, collects predictions, computes mAP/per-class AP/confusion matrix/failure cases |
| `evaluate.py` | CLI entry point |
| `sv_metrics.py` | `compute_map()`, `compute_confusion_matrix()`, `compute_precision_recall()` — COCO-style metrics via `supervision.metrics.MeanAveragePrecision`. Drop-in replacement for the original numpy implementation. |
| `visualization.py` | `draw_bboxes()`, `plot_confusion_matrix()`, `plot_pr_curve()`, `plot_training_curves()` — matplotlib-based plotting, delegates bbox drawing to supervision bridge |
| `error_analysis.py` | `ErrorAnalyzer`, `ErrorCase`, `ErrorReport` — FP/FN classification, per-class optimal thresholds, hardest images |
| `error_analysis_runner.py` | `run_error_analysis()` — task-dispatched (detection / classification / segmentation / keypoint). Writes numbered chart PNGs (`01_…png` → `14_…png`) + `09_failure_mode_examples/` galleries. `CHART_FILENAMES` (module-level dict) is the authoritative logical-name → filename map; look up paths via `CHART_FILENAMES[key]` instead of hardcoding strings. Returned `artifacts` dict keys mirror those logical names (e.g. `per_class_performance`, `failure_mode_contribution`, `recoverable_map_vs_iou`, `hardest_images`, `failure_mode_examples_root`). |
| `analyze_errors.py` | Standalone CLI wrapper around `run_error_analysis` |

## CLI

```bash
uv run core/p08_evaluation/evaluate.py \
  --model runs/fire_detection/best.pth \
  --config features/safety-fire_detection/configs/01_data.yaml \
  --split test \
  --conf 0.5
```

## Config Reference

- `--conf` — confidence threshold (default 0.5)
- `--iou` — IoU threshold for mAP (default 0.5)
- `--split` — dataset split to evaluate: `val` or `test`

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

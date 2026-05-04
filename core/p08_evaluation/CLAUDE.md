# CLAUDE.md — core/p08_evaluation/

Companion to `README.md`. Covers the chart-by-chart semantics of
`run_error_analysis` and the threshold policy that gates which charts
respect a deploy cutoff vs which stay full-curve.

## Threshold policy (detection only)

`run_error_analysis(..., threshold_policy=...)` accepts three modes:

| Policy | Behavior | When to use |
|---|---|---|
| `f1_optimal_per_class` (default) | Pass 1 collects raw preds at `EPS_FLOOR = 1e-3`; pass 2 re-filters per class using `compute_f1_optimal_thresholds(...)` (3-point smoothed argmax of the same F1 curve chart 15 plots). Dict `{cls_id: thr}` lands in `summary.json::operating_point.thresholds`. | Deployment-relevant analysis — the post-train default. |
| `f1_optimal` | Single global F1-max threshold pooled across classes. | Multi-class models where ops want one cutoff. |
| `fixed` | Caller's `conf_threshold` applies to all charts (back-compat). | CI / deterministic tests. Forced on `split="train"` to avoid overfitting. |

Helper module: `core/p08_evaluation/threshold_policy.py`.

`conf_threshold` semantics flip with the policy:
- `fixed`: scalar deploy cutoff applied to every chart.
- `f1_optimal[_per_class]`: eps floor for raw collection (matters only when
  `conf_threshold > EPS_FLOOR` — the analyzer takes `min(caller, EPS_FLOOR)`
  in policy modes so the floor is honored).

## Threshold-sensitive vs threshold-agnostic charts

Per-class threshold dict applies to:

```
07_per_class_performance       # P/R/F1 bars at operating point
08_confusion_matrix            # TP/FP cells at operating point
08_top_confused_pairs          # same accumulators
10_failure_mode_contribution   # mode counts at operating point
11_failure_by_attribute        # miss-rate slices at operating point
12_hardest_images              # ranked by FP+FN at operating point
13_failure_mode_examples       # gallery cases at operating point
```

Threshold-agnostic (full-curve / GT-only — never gated):

```
01_overview                    # text card
02_data_distribution           # GT only
03_distribution_mismatch       # train↔val drift
04_label_quality               # confident-disagreement
05_duplicates_leakage          # pHash, ds-only
06_learning_ability            # train-vs-val regime
09_confidence_calibration      # TP/FP score histograms (eps floor)
14_robustness_sweep            # corruption sweep
15_threshold_analysis          # the sweep itself
16_recoverable_map_vs_iou      # mode-tagged at eps; full PR
17_confidence_attribution      # FN causality at eps
18_boxes_per_image             # GT crowdedness
19_bbox_aspect_ratio           # GT geometry
20_size_recall                 # GT size bands
21_pixel_confusion_matrix      # seg-only
21_bbox_padding_sweep          # kpt-only
```

`summary.json::operating_point.applied_to_charts` is the runtime-authoritative
list — keep it in sync with `core.p08_evaluation.threshold_policy.APPLIED_TO_CHARTS`.

## Two-pass detection analyzer (mechanics)

`_analyze_detection` runs:

1. **Pass 1 — forward + matching at eps floor.** One model forward per image
   (`_dispatch_postprocess` with `conf_threshold = min(caller, EPS_FLOOR)`).
   Populates `pred_cache`, threshold-INDEPENDENT GT counters, and an
   `eps_acc` accumulator set whose `confidence_tp/fp` and `fn_attribution`
   feed agnostic charts 09 + 17.
2. **Threshold derivation.** `compute_f1_optimal_thresholds` runs on the
   `detections` list emitted by Pass 1.
3. **Pass 2 — re-match at the operating point.** Iterates `pred_cache`,
   filters per-class via `threshold_for(thresholds, cls)`, and runs the
   same matcher (`_match_predictions_to_gt`) into a fresh `op_acc`. That
   accumulator drives the sensitive charts.

No second forward pass — both matches reuse the cached predictions.

## Adding a new sensitive chart

1. Compute it from `op_acc` (operating-point accumulators), not `eps_acc`.
2. Add the filename stem to `APPLIED_TO_CHARTS` in `threshold_policy.py`.
3. Document in this file's table.

## Adding a new agnostic chart

Compute from `eps_acc`, `detections` (eps + mode tags), or threshold-independent
GT counters. Do **not** add to `APPLIED_TO_CHARTS`.

## Chart decoration invariant

`_decorate_figure` (in `error_analysis_runner.py`) stamps the subtitle and the
`Suggested: …` footer onto every chart. The footer must NEVER use a bare
`fig.text(0.5, 0.01, …)` — `new_figure()` returns figures with
`constrained_layout=True`, which extends axes to the figure bottom and makes
that text overlap x-tick labels / x-axis labels (observed on charts
02/07/08/11/15 prior to 2026-05-04).

Rules:
- Constrained-layout figure, no glossary → `fig.supxlabel(text, …)` (matplotlib reserves space).
- Glossary charts (10/16/17) → caller already does `subplots_adjust(bottom=0.30+)`; place via `fig.text` at y=0.24.
- Non-constrained, no glossary → call `fig.subplots_adjust(bottom=max(current, 0.14))` BEFORE the `fig.text`.

## Standalone CLI

`core/p08_evaluation/run_error_analysis.py` exposes `--threshold-policy`
(default `f1_optimal_per_class`) + `--conf` (deploy cutoff for `fixed`
mode, eps floor otherwise). Adapter `conf_threshold` is forced to
`EPS_FLOOR` whenever the policy is non-fixed so raw predictions reach
the analyzer.

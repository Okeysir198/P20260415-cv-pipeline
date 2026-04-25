"""Chart-annotation registry — descriptions + next-step rules for every chart.

Every chart emitted by the error-analysis runner passes through a lookup here:

    meta = CHART_META.get(chart_filename_stem)
    if meta:
        desc     = meta.description
        next_step = evaluate_next_step(metrics_for_this_chart, meta)

``description`` is a short paragraph (< 80 words, plain English) explaining
what the chart shows. It goes inline under the chart image in ``summary.md``
and can be stamped as a matplotlib figure caption.

``next_step_rules`` is a list of ``Rule`` objects. Each rule has:
    - ``when(metrics)``  → bool
    - ``say``            → format string (can reference keys from ``metrics``)

The first rule whose ``when`` fires wins. If none fire,
``evaluate_next_step`` returns a neutral default (``"No action — signal
is within acceptable band."``).

Keys are the flat 01..20 filename stems used by
``error_analysis_runner.CHART_FILENAMES``. All diagnostics (distribution
mismatch, label quality, duplicates/leakage, learning ability, robustness)
now live under this same flat numbering.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Rule:
    when: Callable[[dict[str, Any]], bool]
    say: str


@dataclass(frozen=True)
class ChartMeta:
    title: str
    description: str
    next_step_rules: tuple[Rule, ...] = ()
    signal_template: str | None = None


_DEFAULT_NEXT_STEP = "No action — signal is within acceptable band."


def evaluate_next_step(
    metrics: dict[str, Any] | None, meta: ChartMeta | None
) -> str:
    if meta is None or not meta.next_step_rules:
        return _DEFAULT_NEXT_STEP
    m = metrics or {}
    for rule in meta.next_step_rules:
        try:
            if rule.when(m):
                return _format_safe(rule.say, m)
        except Exception:
            continue
    return _DEFAULT_NEXT_STEP


def render_signal(meta: ChartMeta | None, metrics: dict[str, Any] | None) -> str | None:
    if meta is None or not meta.signal_template:
        return None
    try:
        return meta.signal_template.format(**(metrics or {}))
    except (KeyError, IndexError, ValueError):
        return None


def _format_safe(template: str, values: dict[str, Any]) -> str:
    try:
        return template.format(**values)
    except (KeyError, IndexError, ValueError):
        return template


# ============================================================================
# Chart catalog — flat 01..20 numbering.
# ============================================================================

CHART_META: dict[str, ChartMeta] = {
    "01_overview": ChartMeta(
        title="Overview",
        description=(
            "Headline metric for this checkpoint alongside a per-failure-mode "
            "ranking. Bars represent how much the primary metric (mAP50 / "
            "mIoU / accuracy) would improve if each failure mode were "
            "resolved; the longest bar is the highest-value target."
        ),
        signal_template="Headline: {primary_metric_name}={primary_metric:.3f}.",
        next_step_rules=(
            Rule(
                when=lambda m: m.get("primary_metric", 1.0) < 0.15,
                say="Primary metric is very low ({primary_metric:.3f}) — check for "
                    "a data loading / label / ignore-index bug before tuning further.",
            ),
            Rule(
                when=lambda m: m.get("top_mode_delta", 0) > 0.10,
                say="'{top_mode}' contributes the largest recoverable loss "
                    "(Δ {top_mode_delta:.3f}). Focus the next iteration there.",
            ),
        ),
    ),

    "02_data_distribution": ChartMeta(
        title="Data distribution",
        description=(
            "Per-class sample counts and (for detection) per-class × size-tier "
            "breakdown. Severe imbalance explains much of the per-class "
            "performance spread seen later in per-class performance."
        ),
        signal_template="Imbalance ratio (max/min): {imbalance_ratio:.1f}x across {n_classes} classes.",
        next_step_rules=(
            Rule(
                when=lambda m: m.get("imbalance_ratio", 1) > 20,
                say="Severe class imbalance ({imbalance_ratio:.0f}x) — consider "
                    "class-weighted loss, oversampling, or targeted data collection "
                    "for the rare classes.",
            ),
        ),
    ),

    "03_distribution_mismatch": ChartMeta(
        title="Distribution mismatch — train vs val",
        description=(
            "Two-panel drift diagnostic. Left: per-class prevalence across "
            "train/val/test, summarised by Jensen-Shannon divergence. Right: "
            "per-image brightness/contrast/aspect/area histograms with KS "
            "p-values. Healthy splits target JS < 0.1 and KS p > 0.01."
        ),
        signal_template="JS(train,val)={js_divergence:.3f}  |  min KS p={ks_min_pvalue:.4f}.",
        next_step_rules=(
            Rule(
                when=lambda m: m.get("js_divergence", 0) > 0.20,
                say="Class distributions differ substantially (JS={js_divergence:.2f}). "
                    "Reshuffle with stratified train/val split, or expect val metrics "
                    "to under-represent rare-class performance.",
            ),
            Rule(
                when=lambda m: m.get("ks_min_pvalue", 1) < 0.01,
                say="Strong image-statistics drift (KS p={ks_min_pvalue:.4f}). "
                    "Likely a different camera / scene / time-of-day distribution. "
                    "Re-split or augment to bridge the gap.",
            ),
            Rule(
                when=lambda m: m.get("js_divergence", 0) > 0.10,
                say="Moderate class-distribution drift (JS={js_divergence:.2f}). "
                    "Consider stratified sampling for the next training run.",
            ),
        ),
    ),

    "04_label_quality": ChartMeta(
        title="Estimated label noise per class",
        description=(
            "Per-class fraction of samples (or pixels, for segmentation) "
            "where the trained model is **confident** but disagrees with "
            "the GT label. Strong signal of probable mislabels — high-"
            "confidence model errors on the training set usually trace back "
            "to the labels, not the model."
        ),
        signal_template="Worst class: {worst_class} ({worst_rate:.1%}).",
        next_step_rules=(
            Rule(
                when=lambda m: m.get("worst_rate", 0) > 0.10,
                say="'{worst_class}' has an estimated {worst_rate:.0%} label-noise "
                    "rate. Run a `p02` annotation-QA pass on this class, or send "
                    "the suspected-mislabel CSV back through Label Studio for "
                    "human review.",
            ),
            Rule(
                when=lambda m: m.get("worst_rate", 0) > 0.03,
                say="'{worst_class}' shows {worst_rate:.1%} suspect labels — "
                    "moderate. Worth eyeballing the gallery before spending more "
                    "compute on hyperparameter tuning.",
            ),
        ),
    ),

    "04_label_quality_gallery": ChartMeta(
        title="Suspected mislabel gallery",
        description=(
            "GT | Pred side-by-side for the most-suspect samples — those "
            "where the model is highly confident in a different answer. "
            "Treat as candidates for human review; never auto-relabel. The "
            "exported `04_suspected_mislabels.csv` is consumable by Label "
            "Studio."
        ),
        signal_template="{n_suspected} of {n_total} samples flagged ({suspect_frac:.1%}).",
        next_step_rules=(
            Rule(
                when=lambda m: m.get("suspect_frac", 0) > 0.10,
                say="More than {suspect_frac:.0%} of samples flagged as suspect — "
                    "likely a systematic labeling issue (taxonomy mismatch, drifted "
                    "annotation guidelines). Audit the labeling SOP before more training.",
            ),
        ),
    ),

    "05_duplicates_leakage": ChartMeta(
        title="Near-duplicates & cross-split leakage",
        description=(
            "Left panel: count of near-duplicate image pairs within each "
            "split (pHash Hamming distance ≤ 6). Right panel: pairs of "
            "near-identical images that appear in *different* splits — "
            "the definition of held-out leakage. Any non-zero cross-split "
            "count inflates val / test metrics and should be resolved "
            "before trusting the numbers."
        ),
        signal_template=(
            "Within-split dup pairs: {n_within_duplicates}  |  "
            "cross-split leakage pairs: {n_cross_leakage}."
        ),
        next_step_rules=(
            Rule(
                when=lambda m: m.get("n_cross_leakage", 0) > 0,
                say="{n_cross_leakage} cross-split leakage pair(s) detected "
                    "(worst: {worst_cross_pair}). Remove the duplicates from one "
                    "side of the split before trusting val/test metrics.",
            ),
            Rule(
                when=lambda m: m.get("n_within_duplicates", 0) > 20,
                say="{n_within_duplicates} near-duplicate pairs within splits — "
                    "de-duplicate to avoid metric bias (val) and wasted "
                    "gradient steps (train).",
            ),
        ),
    ),

    "06_learning_ability": ChartMeta(
        title="Learning ability — bias / variance regime + curves",
        description=(
            "Two-panel capacity diagnostic. Left: primary metric on train vs "
            "val with regime classified (high_bias / high_variance / healthy / "
            "ambiguous). Right: per-epoch overlay of training loss and eval "
            "metric. Widening train-loss-down / eval-metric-flat gap is the "
            "canonical overfitting signature."
        ),
        signal_template="Train {train_metric:.3f}  vs  Val {val_metric:.3f}  (gap {gap:+.3f}, regime `{regime}`).",
        next_step_rules=(
            Rule(
                when=lambda m: m.get("regime") == "high_bias",
                say="High-bias / underfit: scale up model capacity, swap to a "
                    "stronger backbone, or audit label quality (the model can't "
                    "even fit train).",
            ),
            Rule(
                when=lambda m: m.get("regime") == "high_variance",
                say="High-variance / overfit: add regularization "
                    "(weight decay, dropout), strengthen augmentation, or "
                    "collect more training data.",
            ),
            Rule(
                when=lambda m: m.get("regime") == "ambiguous",
                say="Mixed signal — read the per-class delta and learning "
                    "curves before deciding what to change.",
            ),
        ),
    ),

    "07_per_class_performance": ChartMeta(
        title="Per-class performance",
        description=(
            "Precision / Recall / F1 bars per class (detection, classification) "
            "or IoU bars (segmentation). A wide gap between classes flags the "
            "long-tail end of the distribution; a uniform low value flags a "
            "systemic issue (architecture, label quality, or domain gap)."
        ),
        signal_template="Best class: {best_class} ({best_score:.3f})  |  worst: {worst_class} ({worst_score:.3f}).",
        next_step_rules=(
            Rule(
                when=lambda m: m.get("worst_score", 1.0) < 0.10,
                say="Worst class '{worst_class}' is near zero ({worst_score:.3f}) — "
                    "inspect its GT samples and check for label-noise or annotation "
                    "confusion with neighbour classes.",
            ),
            Rule(
                when=lambda m: (
                    m.get("best_score", 0) - m.get("worst_score", 0) > 0.5
                ),
                say="Very wide per-class spread (Δ {best_score:.2f} → {worst_score:.2f}). "
                    "Long-tail problem — more data for '{worst_class}' is usually "
                    "higher-leverage than model changes.",
            ),
        ),
    ),

    "08_confusion_matrix": ChartMeta(
        title="Confusion matrix",
        description=(
            "Rows = ground-truth class, columns = predicted class. Strong "
            "off-diagonal cells = systematic class confusions. For detection "
            "the last row/column is 'background'; for segmentation cells "
            "are aggregated pixel counts."
        ),
        next_step_rules=(
            Rule(
                when=lambda m: m.get("max_off_diagonal", 0) > 0.20,
                say="'{top_confused_from}' → '{top_confused_to}' confusion is high "
                    "({max_off_diagonal:.2f}). Consider merging the two classes or "
                    "improving labels for the boundary cases.",
            ),
        ),
    ),

    "08_top_confused_pairs": ChartMeta(
        title="Top confused class pairs",
        description=(
            "Ranked list of the N most-confused (gt, pred) class pairs when "
            "there are too many classes to render a full matrix. A pair "
            "dominating the ranking is usually either a true class-similarity "
            "problem or a label-taxonomy mismatch."
        ),
    ),

    "09_confidence_calibration": ChartMeta(
        title="Confidence calibration",
        description=(
            "Histograms of model scores for true-positive (green) and "
            "false-positive (red) predictions. Well-calibrated detectors put "
            "TPs at high scores and FPs at low scores with minimal overlap — "
            "that separation is what NMS thresholding exploits."
        ),
        signal_template="TP median {tp_median:.2f}, FP median {fp_median:.2f}, overlap {overlap:.2f}.",
        next_step_rules=(
            Rule(
                when=lambda m: m.get("overlap", 0) > 0.3,
                say="TP/FP score distributions overlap heavily (overlap={overlap:.2f}). "
                    "Lower NMS conf_threshold, or retrain with hard-negative mining.",
            ),
            Rule(
                when=lambda m: m.get("tp_median", 1.0) < 0.5,
                say="TP median confidence is {tp_median:.2f} (< 0.5). Model is "
                    "under-confident — longer training or lower label smoothing "
                    "often helps.",
            ),
        ),
    ),

    "09_confidence_vs_error": ChartMeta(
        title="Confidence vs error (keypoint)",
        description=(
            "Scatter / binned plot of predicted keypoint confidence vs pixel "
            "localization error. Well-calibrated pose models show "
            "monotonically-decreasing error as confidence climbs."
        ),
    ),

    "10_failure_mode_contribution": ChartMeta(
        title="Failure mode contribution",
        description=(
            "Breakdown of how much each failure category (missed, "
            "localization, class confusion, duplicate, background FP) "
            "contributes to the overall metric gap. Per-class heatmap to "
            "the right surfaces classes that are uniquely affected by one mode."
        ),
        signal_template="Dominant mode: {dominant_mode} (Δ {dominant_delta:.3f}).",
        next_step_rules=(
            Rule(
                when=lambda m: m.get("dominant_mode") == "missed",
                say="Most loss comes from missed detections. Increase recall via "
                    "lower conf threshold, more diverse training data, or better "
                    "small-object augmentations.",
            ),
            Rule(
                when=lambda m: m.get("dominant_mode") == "localization",
                say="Boxes land on the right class but miss the right location. "
                    "Increase bbox loss weight, use stronger geometric augs, or "
                    "check for coarse annotations.",
            ),
            Rule(
                when=lambda m: m.get("dominant_mode") == "class_confusion",
                say="Most loss is class confusion — labels may be ambiguous. "
                    "Check confusion matrix above and consider class merges or "
                    "a label-studio QA pass on the top-confused pair.",
            ),
            Rule(
                when=lambda m: m.get("dominant_mode") == "duplicate",
                say="Duplicates dominate — tune NMS IoU threshold, or inspect "
                    "for overlapping anchors / query collisions.",
            ),
            Rule(
                when=lambda m: m.get("dominant_mode") == "background_fp",
                say="Most loss is background false-positives. Raise the conf "
                    "threshold, or add hard-negative background images.",
            ),
        ),
    ),

    "11_failure_by_attribute": ChartMeta(
        title="Failure by attribute",
        description=(
            "Miss-rate breakdown by image attributes — object size, aspect "
            "ratio, and scene crowdedness. A steep climb at the small-size "
            "bucket means the model is scale-sensitive; a climb at high "
            "crowdedness means NMS / detector saturation."
        ),
    ),

    "12_hardest_images": ChartMeta(
        title="Hardest images",
        description=(
            "Top-12 validation images ranked by per-image metric loss. Each "
            "cell shows GT | Pred side-by-side. Patterns across the 12 (same "
            "class, same scene, same camera) usually point at a dataset "
            "gap rather than a model flaw."
        ),
    ),

    "13_failure_mode_examples": ChartMeta(
        title="Failure mode examples",
        description=(
            "Per-mode × per-class galleries of representative failure cases "
            "(GT | Pred). Use this to scan for common root causes — "
            "repeatedly-mislabeled scenes usually surface here first."
        ),
    ),

    "14_robustness_sweep": ChartMeta(
        title="Robustness sweep",
        description=(
            "Primary metric under four lightweight corruption families — "
            "gaussian blur, JPEG compression, brightness shift, rotation — "
            "at 3 severities each. Steep falloff on any family identifies a "
            "deployment-time failure mode before it hits production."
        ),
        signal_template="Clean {clean:.3f}; worst family `{worst_family}` drops to {worst_value:.3f}.",
        next_step_rules=(
            Rule(
                when=lambda m: m.get("worst_drop_frac", 0) > 0.30,
                say="Metric drops >{worst_drop_frac:.0%} under `{worst_family}`. "
                    "Add the corresponding augmentation to training and re-evaluate.",
            ),
        ),
    ),

    # ---------------- Detection-specific (15..19) ---------------------
    "15_recoverable_map_vs_iou": ChartMeta(
        title="Recoverable mAP across IoU",
        description=(
            "How much mAP each failure mode would recover across the COCO "
            "IoU sweep (0.5 → 0.95). Modes whose curves are flat are truly "
            "unrecoverable at stricter IoU; steep drops signal localization-"
            "sensitive modes where better box regression pays off."
        ),
    ),

    "16_confidence_attribution": ChartMeta(
        title="Confidence attribution of FN",
        description=(
            "Decomposes false negatives into three causal buckets: "
            "true misses (no detection at all), under-confidence (matching "
            "detection below threshold), and localization failure (matching "
            "detection but wrong IoU). Splits 'your threshold is wrong' from "
            "'your model is wrong'."
        ),
        signal_template="FN: true_miss={true_miss:.0%}  under_conf={under_conf:.0%}  loc_fail={loc_fail:.0%}.",
        next_step_rules=(
            Rule(
                when=lambda m: m.get("under_conf", 0) > 0.3,
                say="{under_conf:.0%} of FNs are under-confidence detections. "
                    "Lower the detection threshold before changing the model.",
            ),
        ),
    ),

    "17_boxes_per_image": ChartMeta(
        title="Boxes per image (crowdedness)",
        description=(
            "Distribution of GT boxes per image — mean / median / p95 / max. "
            "High tails mean the val set has crowded scenes; consider bumping "
            "detector query / proposal counts to match."
        ),
    ),

    "18_bbox_aspect_ratio": ChartMeta(
        title="Bbox aspect ratio",
        description=(
            "Per-class log-scale width / height distribution. Extreme tails "
            "(very thin or very wide) indicate classes where default anchor / "
            "query priors mismatch the data — a known source of localization FP."
        ),
    ),

    "19_size_recall": ChartMeta(
        title="Size recall",
        description=(
            "Recall broken down by COCO area bands: small (< 32² px), "
            "medium (32²–96² px), large (> 96² px). Small-object recall is "
            "the most commonly gappy — augmentations or input-size bumps are "
            "the usual levers."
        ),
    ),

    # ---------------- Segmentation-specific (20) ----------------------
    "20_pixel_confusion_matrix": ChartMeta(
        title="Pixel confusion matrix",
        description=(
            "Class × class pixel-confusion matrix for semantic segmentation: "
            "row = GT class, column = predicted class. Off-diagonal cells "
            "reveal which classes get systematically swapped at the pixel "
            "level. Strong pairs are usually annotation-taxonomy ambiguity "
            "(e.g. road vs parking-driveway) rather than model failure."
        ),
        signal_template="Worst confusion: {top_pair_gt} → {top_pair_pred} ({top_pair_pct:.1%} of GT pixels).",
        next_step_rules=(
            Rule(
                when=lambda m: m.get("top_pair_pct", 0) > 0.30,
                say="'{top_pair_gt}' is mostly predicted as '{top_pair_pred}' "
                    "({top_pair_pct:.0%} of pixels). Check the labeling SOP for "
                    "this class boundary or consider merging the two classes.",
            ),
        ),
    ),
}

# Label Quality Grades (p02 interpretation)

`core/p02_annotation_qa/scorer.py::QualityScorer` computes a score in `[0, 1]` and assigns a grade.

## Scoring formula

```
score = 1.0 − Σ(weight_i × penalty_i)
```

Default weights (configurable in `02_annotation_quality.yaml → scoring.weights`):

| Component | Weight | Penalty source |
|---|---|---|
| `structural` | 0.3 | out-of-bounds bbox, duplicate bbox, degenerate (zero-area), aspect-ratio outlier |
| `bbox_quality` | 0.4 | SAM3 mask vs. labeled bbox → `1 − mean_IoU` |
| `classification` | 0.2 | SAM3 predicted class ≠ labeled class |
| `coverage` | 0.1 | SAM3 detected objects the label missed |

## Grades (configurable thresholds)

| Grade | Condition (default) | What it means |
|---|---|---|
| `good` | `score ≥ 0.80` | Trust for training. |
| `review` | `0.50 ≤ score < 0.80` | Usable but has issues; include in training with light weight or after manual spot-check. |
| `bad` | `score < 0.50` | Do not use as-is — either re-label (p01) or drop. |

Override thresholds via `--override scoring.thresholds.good=0.75 scoring.thresholds.review=0.45` on the p02 CLI.

## Interpreting grade distribution

| Distribution | Action |
|---|---|
| `good ≥ 80%`, `bad ≤ 5%` | Accept. Ready for training. |
| `good 60–80%`, `bad 5–20%` | Run p01 `--filter bad` to re-label, then re-QA. |
| `bad > 20%` | **Stop.** Almost always a class-map bug, not bad labels. Re-check class names with `inspect_source.sh`. Common culprit: case-sensitive mismatch (`"Helmet"` vs `"helmet"`). |
| `good < 50%` | Source quality is genuinely poor — drop the source and pick another. |
| All classes grade `bad` | Text prompts are too broad or wrong. Tune `text_prompts:` in `05_data.yaml` and re-run p02. |

## When to lower the `good` threshold

Lower it to 0.70 if:
- Domain is inherently hard (small objects at distance, heavy occlusion, low-light).
- SAM3 struggles with the object type (e.g. thin harness straps, glasses on faces).

Don't lower it to "make the numbers look good" — that defeats the QA purpose.

## When to run p02 without SAM3 (`--no-sam3`)

Use `--no-sam3` if:
- SAM3 service (:18100) is down and can't be started quickly.
- You only need structural validation (bbox geometry, duplicates).

Limitations: no `bbox_quality`/`classification`/`coverage` penalties — you'll miss mislabeled bboxes that are geometrically valid.

## Acting on `worst_images.json`

This file lists the N worst-scored samples with per-issue breakdown. Sort by `score ascending`; for each:

1. Open `runs/qa/<name>/visualizations/<image>.png` (labeled bbox overlaid).
2. If label is clearly wrong → mark for p01 re-labeling (`bad` grade).
3. If label is right but score is low → tune SAM3 prompts or lower thresholds.
4. If SAM3 caught a **missing** object → label is incomplete; re-run p01 with `--mode hybrid` to add missed detections.

## Applying fixes automatically (`--apply-fixes`)

`core/p02_annotation_qa/run_qa.py --apply-fixes` backs up originals (timestamped) then applies structural fixes from `fixes.json`. Does not handle classification or coverage fixes — those need p01.

Always spot-check the backup vs. new labels on a few samples before training.

## When to move a sample between splits (LS review)

During Label Studio review the reviewer can reassign each sample via the **Split** radio button (`train / val / test / drop`). Rules of thumb:

- **val is too homogeneous** — missing a class or lacking scene variety → pull a few samples from `train` → `val` so the evaluation signal covers the real distribution.
- **val has dubious labels** — move them to `train`; noisy val labels inflate reported error more than they help.
- **Class underrepresented in `test`** — rebalance a few samples into `test` so the test set exercises every class.
- **Sample is unrecoverable** (wrong subject, duplicate, corrupt image) → choose `drop`. It moves to `dropped/` (recoverable); only delete outright via `--hard-drop`.

Export writes the corrected labels **and** physically moves the files. The `splits.json` audit snapshot is rewritten from the filesystem afterwards.

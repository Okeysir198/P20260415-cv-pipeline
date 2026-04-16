---
name: cv-dataset-prep
description: This skill should be used when the user asks to "prepare training data", "build a training dataset", "merge datasets for a CV model", "audit label quality", "check dataset labels", "curate training set", "create 00_data_preparation.yaml", "run annotation QA", or wants to turn raw/ folders into a training_ready/ split in this cv-pipeline repo. Also triggers on mentions of p00_data_prep, p02_annotation_qa, p01_auto_annotate, SAM3 label verification, or label grading.
---

# CV Dataset Prep

Orchestrates `core/p00_data_prep` → `core/p02_annotation_qa` → `core/p01_auto_annotate` → `core/p04_label_studio`. All logic lives in `core/` — do not reimplement it.

## Prerequisites

```bash
curl -sf http://localhost:18105/health && echo QA_OK
curl -sf http://localhost:18100/health && echo SAM3_OK
curl -sf http://localhost:18104/health && echo AUTOLABEL_OK
```
p02 requires `:18105`. p01 requires `:18100` + `:18104`. If a required service is down, stop and tell the user which to start — do not skip the step silently.

---

## Step 1 — Clarify

Ask: **feature name** (`<category>-<name>`) and **canonical classes** (2–5 classes including negatives).
If `features/<name>/configs/00_data_preparation.yaml` already exists, offer to refresh it instead.

## Step 2 — Discover sources

```bash
bash .claude/plugins/cv-data-prep/skills/cv-dataset-prep/scripts/inspect_source.sh <raw_path>
```

Run for each candidate folder under `dataset_store/raw/<category>/`. Cross-reference `dataset_store/CLAUDE.md` for license and quality notes. Rank by: **label provenance > negative coverage > scene diversity > volume**. See `references/source-ranking.md`.

## Step 3 — Design class map

Map each source's raw class names onto the canonical list. Classes not in `class_map:` are silently dropped — list them in `dropped_classes:`. See `references/class-remap-patterns.md` for common patterns.

## Step 4 — Write `00_data_preparation.yaml`

Fill `templates/00_data_preparation.yaml.template` → `features/<name>/configs/00_data_preparation.yaml`.
Every source needs `license:`, `notes:`, `dropped_classes:`. Excluded datasets go in the top-level `held_back:` list.

## Step 5 — Dry-run, then merge

```bash
uv run core/p00_data_prep/run.py --config features/<name>/configs/00_data_preparation.yaml --dry-run
```

Check class distribution. **Any class at 0%** → class_map key mismatch — fix before continuing (re-run `inspect_source.sh` for exact string). **Any class < 3%** → warn the user, offer to add a source.

```bash
uv run core/p00_data_prep/run.py --config features/<name>/configs/00_data_preparation.yaml
```

## Step 6 — Write `05_data.yaml`

Fill `templates/05_data.yaml.template` → `features/<name>/configs/05_data.yaml`.
`path:` must match `output_dir:` from step 4. Fill `names:`, `num_classes:`, `text_prompts:` (required by p01).

## Step 7 — Annotation QA

```bash
uv run core/p02_annotation_qa/run_qa.py --data-config features/<name>/configs/05_data.yaml
# SAM3 down → add --no-sam3 (structural checks only)
```

Read grade distribution from `runs/.../summary.txt`. Decide:

| Result | Action |
|--------|--------|
| `good ≥ 80%` and `bad ≤ 5%` | Accept → skip to step 9 |
| `bad 5–20%` | Re-label with p01 → step 8 |
| `bad > 20%` | Stop — almost always a class_map bug, not bad labels |

See `references/label-quality-grades.md` for threshold tuning and worst_images interpretation.

## Step 8 — Re-label bad samples (p01)

```bash
uv run core/p01_auto_annotate/run_auto_annotate.py \
  --data-config features/<name>/configs/05_data.yaml --mode text --filter bad
```

Re-run step 7. If `bad` is still > 5% after one cycle, show the user `worst_images.json` — do not loop again automatically.

## Step 8b — Human review in Label Studio (optional)

Use when p01 can't fix issues (missing annotations, wrong class, split rebalancing needed).

```bash
# Setup + import
uv run core/p04_label_studio/bridge.py --email $LS_EMAIL --password $LS_PASSWORD setup \
  --data-config features/<name>/configs/05_data.yaml
uv run core/p04_label_studio/bridge.py --email $LS_EMAIL --password $LS_PASSWORD import \
  --data-config features/<name>/configs/05_data.yaml

# After human review is complete
uv run core/p04_label_studio/bridge.py --email $LS_EMAIL --password $LS_PASSWORD export \
  --data-config features/<name>/configs/05_data.yaml --project <dataset_name>_review
```

Then re-run step 7.

## Step 9 — Hand off

Confirm `05_data.yaml` `path:` matches `training_ready/<name>/` and `num_classes:` is correct.

Print the smoke-test training command:
```bash
uv run core/p06_training/train.py \
  --config features/<name>/configs/06_training.yaml --override training.epochs=5
```

Do not start training — hand off here.

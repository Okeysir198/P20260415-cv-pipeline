---
name: cv-dataset-prep
description: This skill should be used when the user asks to "prepare training data", "build a training dataset", "merge datasets for a CV model", "audit label quality", "check dataset labels", "curate training set", "create 00_data_preparation.yaml", "run annotation QA", or wants to turn raw/ folders into a training_ready/ split in this cv-pipeline repo. Also triggers on mentions of p00_data_prep, p02_annotation_qa, p01_auto_annotate, SAM3 label verification, or label grading (good/review/bad). Runs the quality-first workflow end-to-end: source selection → class-map → merge → QA → re-label bad → re-QA.
---

# CV Dataset Prep

Prepare a training-ready YOLO dataset for a CV feature in this repo, then audit and fix label quality. Wraps the existing `core/p00_data_prep`, `core/p02_annotation_qa`, `core/p01_auto_annotate` tools — do not rewrite them.

Always produce the outputs under the feature's existing folder, not a new parallel location.

## Prerequisites

Before step 1, verify:

1. `uv sync --extra all` has been run at the repo root.
2. Services for QA + re-labeling are up. Run:
   ```bash
   curl -sf http://localhost:18105/health && echo QA_OK
   curl -sf http://localhost:18100/health && echo SAM3_OK
   curl -sf http://localhost:18104/health && echo AUTOLABEL_OK
   ```
   - QA :18105 must be up for step 9.
   - SAM3 :18100 is optional for QA (pass `--no-sam3` to skip) but required for step 10.
   - Auto-label :18104 is required for step 10.
3. Raw datasets are present under `dataset_store/raw/<category>/`. See `dataset_store/CLAUDE.md` for the source registry.

If a service is missing, stop and tell the user which service to start — do not proceed.

## Workflow (12 steps)

### Step 1 — clarify the goal

Ask the user (with AskUserQuestion if multiple unknowns, otherwise inline):

- **Feature name** — `<category>-<name>` convention (e.g. `ppe-gloves_detection`, `safety-smoke_detection`). This becomes `features/<name>/`.
- **Task** — currently only `detection` is supported by `p00_data_prep`; classification/segmentation will be rejected.
- **Canonical classes** — the target class list (e.g. `[person, hand_with_glove, hand_without_glove]`). Keep it small; 2–6 classes is typical.

If the feature folder already exists with `00_data_preparation.yaml`, offer to refresh it rather than create from scratch.

### Step 2 — source discovery

List candidate raw folders under `dataset_store/raw/<relevant_category>/`. For each, call:

```bash
bash .claude/plugins/cv-data-prep/skills/cv-dataset-prep/scripts/inspect_source.sh <path>
```

This prints image count, label format (YOLO/COCO/VOC), and unique class names. Also cross-reference `dataset_store/CLAUDE.md` which has per-source license + quality notes + URLs for manual spot-check.

Rank candidates using `references/source-ranking.md` (quality-first — label provenance beats volume).

### Step 3 — pick the v1 source set

Select 2–4 sources that together cover all canonical classes (including negatives) with diverse scenes. Explain the pick to the user. Dropped sources go in a comment at the top of the YAML for v2 consideration.

Flag and drop:
- Sources with opaque class names (e.g. `"0"`, `"14"`, `"mobile_dataset - v9 ..."`) unless the user insists.
- Sources marked "spot-check" in `dataset_store/CLAUDE.md`.
- Near-duplicates of other picked sources (community-scraped Kaggle mirrors of Roboflow exports).

### Step 4 — class-map design

For each selected source, map its raw class names onto the canonical list. Rules:

- Be explicit about every source class you **keep**; list dropped ones in a YAML comment.
- If a canonical class has zero contributing sources, warn the user and suggest adding another source or deferring the class to `site_collected/`.
- Use `references/class-remap-patterns.md` for common PPE/fire/fall/phone taxonomies.

### Step 5 — author `00_data_preparation.yaml`

Render `templates/00_data_preparation.yaml.template` into `features/<name>/configs/00_data_preparation.yaml`. Critical path convention:

- Relative paths are relative to the **config file's directory**.
- `configs/` is 3 levels below repo root, so all repo-relative paths use `../../../`.
- Example: `../../../dataset_store/raw/helmet_detection/sh17_ppe`.

Before moving on, ensure `05_data.yaml`'s `path:` field matches the `output_dir:` in this file.

Format-per-source is decided by `references/format-decision.md`. When a source has both YOLO and VOC labels (e.g. sh17_ppe), prefer VOC because class names are explicit and can't go index-out-of-order.

### Step 6 — dry-run p00

```bash
uv run python core/p00_data_prep/run.py \
  --config features/<name>/configs/00_data_preparation.yaml --dry-run
```

Report the sample count and class distribution. Flag imbalances:

- Any class `<3%` of total instances — under-represented; offer to add a source.
- Any class `>75%` — dominant; consider down-sampling at training time (note in `05_data.yaml`).

If a class shows `0%`, something is wrong with the class_map — re-check step 4 against the actual class names reported by `inspect_source.sh`.

### Step 7 — full merge

Same command without `--dry-run`. p00 writes directly into `<training_ready>/{train,val,test}/{images,labels}/` — no flat `images/` or `labels/` dir, no symlinks. A tiny audit snapshot lands at `<training_ready>/splits.json` (just ratios + seed + counts, not filename lists). Stratified 80/10/10 by default; override with `--splits 0.85 0.1 0.05` if needed.

### Step 8 — author `05_data.yaml`

Render `templates/05_data.yaml.template` into `features/<name>/configs/05_data.yaml`. Fill `names:`, `num_classes:`, `input_size:`, and `text_prompts:`. The text prompts are **required** by step 10 (p01 auto-annotate) — propose per-class SAM3-friendly prompts and confirm with the user.

Path must be `../../../dataset_store/training_ready/<name>` (same relative convention as step 5).

### Step 9 — annotation QA

```bash
uv run python core/p02_annotation_qa/run_qa.py \
  --data-config features/<name>/configs/05_data.yaml
```

If SAM3 :18100 is down: add `--no-sam3` for structural-only validation (still useful but won't catch IoU-level errors).

Outputs land in `runs/qa/<name>/`:
- `report.html` — visual summary
- `worst_images.json` — worst-scored samples with issue types
- `fixes.json` — suggested auto-fixes

Parse the JSON sidecars to extract grade distribution: `good` / `review` / `bad` percentages. See `references/label-quality-grades.md` for how to interpret thresholds.

**Decision**:
- `good ≥ 80%` and `bad ≤ 5%`: accept the dataset; skip to step 12.
- `bad > 5%` or `good < 80%`: proceed to step 10.
- `bad > 20%`: stop and investigate — likely a class-map bug or a source with wrong labels. Do not try to auto-fix past this threshold.

### Step 9b — human review in Label Studio (optional, skip if grades already good)

For datasets where p01 auto-fix won't suffice (missing annotations, ambiguous classes, or samples that need moving between splits), pull the human into the loop:

1. **Start Label Studio** (once per machine):
   ```bash
   cd services/s18103_label_studio && docker compose up -d && ./bootstrap.sh
   # Copies ADMIN_TOKEN to the user's clipboard path; also prints it.
   export LS_API_KEY=<token from bootstrap.sh>
   ```

2. **Create the project + import tasks**:
   ```bash
   uv run python core/p04_label_studio/bridge.py \
     --email admin@admin.com --password admin123 setup \
     --data-config features/<name>/configs/05_data.yaml
   uv run python core/p04_label_studio/bridge.py \
     --email admin@admin.com --password admin123 import \
     --data-config features/<name>/configs/05_data.yaml
   ```
   Each imported task carries its current `split` in `data.split` and pre-selects the Choices UI radio button.

3. **Reviewer opens** `http://localhost:18103/projects/<id>` and:
   - Prioritises images from `runs/qa/<name>/summary.txt`'s worst list.
   - Fixes bounding boxes.
   - Optionally changes the **Split** radio button (`train / val / test / drop`) to rebalance the splits or quarantine the sample.
   - Submits the annotation.

4. **Export back**:
   ```bash
   uv run python core/p04_label_studio/bridge.py \
     --email admin@admin.com --password admin123 export \
     --data-config features/<name>/configs/05_data.yaml \
     --project <dataset_name>_review
   ```
   Export is **split-aware**: it writes corrected labels into the current split's `labels/`, then physically moves any sample whose split changed (via `os.rename`). `splits.json` is regenerated from the filesystem.

5. **Re-run step 9** (annotation QA) to confirm grades improved.

### Step 10 — re-label bad samples with p01

```bash
uv run python core/p01_auto_annotate/run_auto_annotate.py \
  --data-config features/<name>/configs/05_data.yaml \
  --mode text --filter bad
```

SAM3 re-generates labels for bad-graded images using the `text_prompts:` from `05_data.yaml`. The original labels are replaced in place. Preview HTML lands in `runs/auto_annotate/<name>/report.html` — offer to open it.

If auto-label :18104 is down, stop and ask the user to start it; do not skip this step silently.

### Step 11 — re-QA

Re-run step 9. Compare grade distributions before/after. Acceptable if `bad ≤ 5%` now. If still too many bad samples, two options:

- **Drop them**: use `--apply-fixes` in a new p02 run (it backs up labels first and excludes unrecoverable samples from splits).
- **Manual review**: point the user at `worst_images.json` and the samples in `runs/qa/<name>/visualizations/`.

### Step 12 — finalize

- Confirm `features/<name>/configs/05_data.yaml` points at `training_ready/<name>/` and has the correct `num_classes:`.
- Print the baseline training command for the user:

  ```bash
  uv run python core/p06_training/train.py \
    --config features/<name>/configs/06_training.yaml \
    --override training.epochs=5   # smoke test first
  ```

- Append a short one-paragraph summary to `features/<name>/README.md` describing the v1 dataset sources, sample count, and grade distribution.

Do not start training yourself — the skill hands off here.

## References (load when needed)

- `references/source-ranking.md` — quality-first selection (label provenance, scene diversity, class coverage). Read before step 3.
- `references/class-remap-patterns.md` — pre-baked taxonomies for PPE/fire/fall/phone. Read before step 4.
- `references/label-quality-grades.md` — interpreting p02 `good/review/bad`, when to adjust thresholds. Read before step 9.
- `references/format-decision.md` — picking YOLO vs COCO vs VOC parser. Read before step 5 when a source has multiple label formats.

## Templates

- `templates/00_data_preparation.yaml.template` — filled in step 5.
- `templates/05_data.yaml.template` — filled in step 8.

## Scripts

- `scripts/inspect_source.sh <path>` — run in step 2 to probe a raw folder.

## Common failure modes

- **`ModuleNotFoundError`** — user forgot `uv sync --extra all`; stop and tell them.
- **`No such file or directory` in p00** — relative path wrong. Config-file-relative, not repo-relative. Rebuild paths with `../../../dataset_store/...`.
- **`0% of class X`** — class_map key doesn't match the source's actual class name. Re-run `inspect_source.sh` and fix the exact string (case-sensitive, spaces matter).
- **MCP roboflow download returns success but folder empty** — harmless race; retry or check with `find <path> -type f | head`.
- **p02 reports huge `bad%`** — almost always a class-map bug, not bad labels. Check step 4.

## Out of scope

- Classification or segmentation datasets (p00 adapter is detection-only).
- Parquet / TFRecord sources (unsupported by p00).
- Training — hand off to `core/p06_training/train.py` in step 12.
- Dataset download — use the Roboflow/Kaggle/HF MCP tools documented in `dataset_store/CLAUDE.md`.

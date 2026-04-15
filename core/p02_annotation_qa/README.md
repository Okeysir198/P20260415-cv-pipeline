# Step 00 — Annotation QA

Automated annotation quality checking and improvement for YOLO datasets using **LangGraph** (agentic orchestration) and **SAM3** (Segment Anything Model 3) as a vision oracle.

Runs _before_ training as a data quality gate. Validates structural correctness, verifies semantic accuracy with SAM3, scores each image 0-1, and generates actionable reports.

## Architecture

Five-node LangGraph pipeline processing images in configurable batches:

```
START -> [sample] -> [validate_batch] -> [sam3_verify_batch] -> [score_batch] --has_more?--> loop
                                                                     |
                                                                 [aggregate] -> END
```

| Node | GPU | Purpose |
|------|-----|---------|
| `sample` | No | Stratified sampling (2000/split for large datasets, all for <5K) |
| `validate_batch` | No | 8 structural checks: missing/empty labels, OOB coords, invalid class, degenerate/large boxes, duplicates, extreme aspect ratio |
| `sam3_verify_batch` | Yes | Box-prompted bbox verification, text-prompted class verification (selective), auto-mask missing annotation detection (10% subsample) |
| `score_batch` | No | Weighted quality score (0-1), grade assignment (good/review/bad), fix suggestions |
| `aggregate` | No | Dataset-level statistics, worst-image visualization, report generation |

## Quick Start

```bash
# Install QA dependencies (includes LangGraph, SAM3, torch, transformers)
uv sync --extra qa

# Structural validation only (fast, no GPU)
uv run core/p02_annotation_qa/run_qa.py --data-config features/ppe-shoes_detection/configs/05_data.yaml --no-sam3

# Full QA with SAM3 verification
uv run core/p02_annotation_qa/run_qa.py --data-config features/safety-fire_detection/configs/05_data.yaml

# Custom sample size
uv run core/p02_annotation_qa/run_qa.py --data-config features/safety-fire_detection/configs/05_data.yaml \
    --override sampling.sample_size=500

# Resume interrupted run
uv run core/p02_annotation_qa/run_qa.py --data-config features/safety-fire_detection/configs/05_data.yaml \
    --resume runs/qa/fire/checkpoint.json

# Run all datasets
for cfg in configs/<usecase>/05_data.yaml; do
    uv run core/p02_annotation_qa/run_qa.py --data-config "$cfg" --no-sam3
done
```

## CLI Options

| Flag | Description |
|------|-------------|
| `--data-config` | **(required)** Path to data YAML config (e.g., `features/safety-fire_detection/configs/05_data.yaml`) |
| `--qa-config` | Path to QA config (default: `configs/_shared/02_annotation_quality.yaml`) |
| `--no-sam3` | Skip SAM3 verification — structural validation only |
| `--resume` | Resume from a checkpoint file |
| `--override` | Override config values (e.g., `sampling.sample_size=500 scoring.thresholds.good=0.9`) |

## Files

```
core/p02_annotation_qa/
    run_qa.py           # CLI entry point
    graph.py            # LangGraph StateGraph definition (5 nodes, batch loop)
    nodes.py            # All node implementations (sample, validate, verify, score, aggregate)
    sam3_client.py      # SAM3 wrapper with lazy model loading (box/text/auto-mask)
    sampler.py          # Stratified sampling with min-per-class guarantee
    scorer.py           # Quality scoring formula + grade assignment + fix generation
    reporter.py         # JSON/text report + worst-image visualization + fixes.json

configs/_shared/
    02_annotation_quality.yaml  # Default QA config (sampling, SAM3, scoring, validation thresholds)
```

## Validation Checks

Structural checks (no GPU, runs on every image):

| # | Check | Severity | Description |
|---|-------|----------|-------------|
| 1 | Missing label | High | Image has no corresponding `.txt` file |
| 2 | Empty label | Medium | `.txt` exists but is empty |
| 3 | Out-of-bounds | High | cx, cy, w, h outside [0, 1] |
| 4 | Invalid class ID | High | Class ID not in the dataset's names mapping |
| 5 | Degenerate box | Medium | Width or height < 0.005 |
| 6 | Large box | Low | Width or height > 0.95 |
| 7 | Duplicate | Medium | Same class + IoU > 0.95 |
| 8 | Extreme aspect | Low | w/h ratio > 20 or < 0.05 |

## SAM3 Verification

Three modes, applied selectively to control GPU cost:

| Mode | When | What |
|------|------|------|
| **Box-prompted** | Always | Feeds each YOLO bbox to SAM3, derives tight mask, computes IoU vs original |
| **Text-prompted** | Box IoU < 0.6 | Segments by class name text prompt, checks if annotation overlaps any found instance |
| **Auto-mask** | 10% subsample | Finds all objects, flags SAM3 masks with no YOLO annotation as potential misses |

## Scoring

```
score = 1.0
  - 0.3 * (structural_issues / annotations)
  - 0.4 * (1.0 - mean_sam3_iou)
  - 0.2 * (misclassified / annotations)
  - 0.1 * (missing / (annotations + missing))

score = clamp(score, 0.0, 1.0)
```

| Grade | Threshold | Action |
|-------|-----------|--------|
| good | score >= 0.8 | No action needed |
| review | 0.5 <= score < 0.8 | Flag for human review |
| bad | score < 0.5 | Likely needs re-annotation |

## Output

Reports are saved to `runs/qa/{dataset_name}/`:

```
runs/qa/fire/
    report.json          # Full machine-readable results (all images)
    summary.txt          # Human-readable summary with grade distribution
    fixes.json           # Auto-fixable corrections (review before applying)
    checkpoint.json      # Processing state for resume
    worst_images/        # Visualizations of worst-scoring images
        001_image_name.png   # Original annotations (green) + SAM3 suggestions (red)
        ...
```

## Config Reference

All values in `configs/_shared/02_annotation_quality.yaml`:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `sampling` | `strategy` | `stratified` | `stratified`, `random`, or `all` |
| `sampling` | `sample_size` | `2000` | Images per split |
| `sampling` | `min_per_class` | `100` | Minimum images per class in sample |
| `sam3` | `model` | `facebook/sam3` | HuggingFace model ID |
| `sam3` | `text_verify_threshold` | `0.6` | Box IoU below this triggers text verification |
| `sam3` | `auto_mask_sample_rate` | `0.1` | Fraction of batch for auto-mask |
| `scoring.weights` | `structural` | `0.3` | Weight for structural validation penalty |
| `scoring.weights` | `bbox_quality` | `0.4` | Weight for SAM3 IoU penalty |
| `scoring.thresholds` | `good` | `0.8` | Minimum score for "good" grade |
| `scoring.thresholds` | `review` | `0.5` | Minimum score for "review" grade |
| `validation` | `duplicate_iou_threshold` | `0.95` | IoU above this = duplicate |
| `text_prompts` | per class | — | SAM3-friendly text prompts for each class name |

## Dependencies

All managed via [uv](https://docs.astral.sh/uv/) at the project root (`pyproject.toml`, `[project.optional-dependencies.qa]`):

- **Required**: `langgraph`, `numpy`, `pyyaml`
- **For SAM3**: `torch`, `torchvision`, `transformers`, `accelerate`, `Pillow`
- **For visualization**: `opencv-python`, `matplotlib`
- **Reused from pipeline**: `utils/config.py`, `utils/metrics.py`, `core/p08_evaluation/visualization.py`, `utils/device.py`

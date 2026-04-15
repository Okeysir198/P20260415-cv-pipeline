# {Category}: {Use Case Name}
> ID: {letter} | Owner: {engineer_id} | Phase: {1|2} | Status: {planning|data_collection|training|evaluation|deployed}

## Customer Requirements

**Source:** {customer meeting / specification document / email — date}

**Explicit Requirements:**
- {Requirement 1 — what the customer explicitly asked for}
- {Requirement 2}
- {Requirement 3}

**Reference Data:**

| {Metric} | {Value} |
|---|---|
| {spec name} | {spec value} |

**Customer Reference:** {Link to reference product/paper if applicable}

## Business Problem Statement

- **{Problem area}:** {What real-world problem exists — injuries, compliance, liability}
- **{Stakeholder impact}:** {Who is affected and what do they expect from the solution}
- **{Regulatory / standards}:** {Applicable industry standards, compliance requirements, or legal obligations}
- **{Impact of inaction}:** {What happens if this problem is not solved — safety, legal, operational consequences}

## Technical Problem Statement

- **{Business problem ref} → {Technical challenge}:** {Translation into ML/CV difficulty — small objects, occlusion, class ambiguity, limited data, etc.}
- **{Business problem ref} → {Technical challenge}:** {Translation — data constraints, domain gap, annotation difficulty}
- **{Business problem ref} → {Technical challenge}:** {Translation — edge deployment constraints, latency, power, multi-model orchestration}
- **{Additional constraint}:** {Any technical constraint not directly from business (e.g., license restrictions, hardware limits)}

## Technical Solution Options

### Option 1: {Solution name} (Recommended)

- **Approach:** {Brief description of the solution}
- **Addresses:** {Which technical problems from above it solves}
- **Pros:** {Advantages — accuracy, speed, license, deployment ease}
- **Cons:** {Trade-offs — limitations, risks, resource requirements}

### Option 2: {Alternative name}

- **Approach:** {Brief description}
- **Addresses:** {Which technical problems it solves}
- **Pros:** {Advantages over Option 1 in specific scenarios}
- **Cons:** {Trade-offs compared to Option 1}

### Option 3: {Fallback name}

- **Approach:** {Brief description}
- **When to use:** {Conditions under which this becomes the preferred option}

**Decision:** Option 1 selected because {rationale}. See Architecture section for implementation details.

## Detection Classes

| Class ID | Name | Description |
|----------|------|-------------|
| 0 | {class_name} | {what it detects} |

## Alert Logic

| Model | Min Confidence | Min Duration | Tracking Required |
|---|---|---|---|
| {name} | {0.XX} | {N frames (Xms)} | {Yes/No} |

## Dataset

- **Sources:** {list each dataset with size, format, license, and link}
- **Size:** {total images}
- **Split:** {train/val/test percentages}
- **DVC tag:** {tag name or TBD}

### Custom Data Requirements

| Scenario | Images Needed | Collection Method | Priority |
|---|---|---|---|
| {scenario} | {count} | {method} | {High/Medium/Low} |

## Annotation Guidelines

{Bounding box rules, quality checks, inter-annotator agreement targets.}

## Architecture

- **Primary model:** {name} ({params}, {key specs}) -- {license}
- **Alternative:** {name} ({params}) -- {license}
- **Fallback:** {name} -- {license}, {why fallback}
- **Input size:** {WxH}
- **Key config:** `configs/{usecase}/06_training.yaml`

{Architecture diagrams (ASCII art) for primary and alternative models.}

## Training Strategy

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | {N} |
| Batch size | {N} |
| Optimizer | {type} (lr={X}, momentum={X}, wd={X}) |
| Scheduler | {type} with {N}-epoch warmup |
| Loss | {type} |
| EMA | decay={X} |
| AMP | {Enabled/Disabled} |
| Grad clip | {X} |

### Augmentation

{Table or list of augmentations with settings.}

### Phased Training

{v1/v2/v3 plan with dataset sizes and goals per phase.}

## Training Results

{Filled after training. Per-class mAP, precision, recall, confusion matrix reference.}

| Metric | v1 | v2 | v3 | Target |
|--------|----|----|-----|--------|
| mAP@0.5 | — | — | — | {X} |
| Precision | — | — | — | {X} |
| Recall | — | — | — | {X} |

## Edge Deployment

### Performance Budget

| Chip | Model | FPS (est.) | Power | Latency |
|------|-------|-----------|-------|---------|
| AX650N | {model} INT8 | {N} | {N}W | {N}ms |
| CV186AH | {model} INT8 | {N} | {N}W | {N}ms |

### Multi-Model Deployment

| Zone | Models | Combined FPS |
|------|--------|-------------|
| {zone_name} | {model list} | ~{N} each |

## Development Plan

| Week | Activity | Deliverable |
|------|----------|-------------|
| 1 | {activity} | {deliverable} |

### Model Card (Deliverables)

Each release produces a model card at `docs/model_cards/<model_id>.md` and a YAML card at `releases/<use_case>/v<N>/model_card.yaml`.

**Model artifacts:**

| Artifact | Format | Path |
|---|---|---|
| PyTorch model | `.pth` | `runs/{usecase}/best.pt` |
| ONNX model | `.onnx` | `runs/{usecase}/export/{id}_{arch}_{imgsz}_v{N}.onnx` |
| Training config | `.yaml` | `configs/{usecase}/06_training.yaml` |
| Metrics | `.json` | `runs/{usecase}/metrics.json` |

**Model card fields (auto-populated by `utils/release.py`):**

| Field | Source |
|---|---|
| Architecture, classes, input size | `06_training.yaml` |
| Dataset version (DVC tag) | `dvc tags list` |
| Training run path | `runs/{usecase}/` |
| mAP@0.5, Precision, Recall | `metrics.json` / `eval_results.json` |
| Per-class breakdown | Evaluation output |

## Key Commands

```bash
# Train
uv run core/p06_training/train.py --config configs/{usecase}/06_training.yaml

# Resume
uv run core/p06_training/train.py --config configs/{usecase}/06_training.yaml --resume runs/{usecase}/last.pth

# Evaluate
uv run core/p08_evaluation/evaluate.py --model runs/{usecase}/best.pt --config configs/{usecase}/05_data.yaml --split test --conf 0.25

# Export
uv run core/p09_export/export.py --model runs/{usecase}/best.pt --training-config configs/{usecase}/06_training.yaml --export-config configs/_shared/09_export.yaml
```

## Contingency Plan

{What to do if primary approach fails. Alternative architectures, data strategies, scope changes.}

## Limitations & Known Issues

{Honest assessment of failure modes, edge cases, environmental constraints.}

## Changelog

| Date | Change |
|------|--------|
| {YYYY-MM-DD} | Initial spec |

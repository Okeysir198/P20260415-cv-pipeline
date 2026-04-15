# Datasets

Image and label data for all use cases. **Data files are gitignored** — only folder structure and documentation are tracked in git. Use DVC for data versioning.

## Directory Layout

```
dataset_store/
├── training_ready/                 # Training-ready (deduplicated, class-mapped, split)
│   └── <use_case>/
│       ├── train/
│       │   ├── images/             # .jpg or .png
│       │   └── labels/             # .txt (YOLO format)
│       ├── val/
│       │   ├── images/
│       │   └── labels/
│       └── test/
│           ├── images/
│           └── labels/
│
├── raw/                            # Original downloads (unmodified)
│   ├── <use_case>/
│   │   └── <source_dataset>/       # One folder per source
│   └── ...
│
├── site_collected/                 # On-site images from deployment locations (flat layout)
│   ├── <use_case>/
│   │   ├── images/
│   │   └── labels/
│   └── ...
│
└── README.md
```

## Use Cases

| Use Case | Config | Path | Classes | Images |
|---|---|---|---|---|
| Fire Detection | `features/safety-fire_detection/configs/05_data.yaml` | `training_ready/fire_detection/` | `0:fire, 1:smoke` | ~122K |
| Helmet Detection | `features/ppe-helmet_detection/configs/05_data.yaml` | `training_ready/helmet_detection/` | `0:person, 1:head_with_helmet, 2:head_without_helmet, 3:head_with_nitto_hat` | ~62K |
| Shoes Detection | `features/ppe-shoes_detection/configs/05_data.yaml` | `training_ready/shoes_detection/` | `0:person, 1:foot_with_safety_shoes, 2:foot_without_safety_shoes` | ~3.7K |
| Fall (Classify) | `features/safety-fall_classification/configs/05_data.yaml` | `training_ready/fall_classification/` | `0:person, 1:fallen_person` | ~17K |
| Fall (Pose) | `features/safety-fall_pose_estimation/configs/05_data.yaml` | `training_ready/fall_pose_estimation/` | `0:person` (17 keypoints) | ~111 |
| Zone Intrusion | — | `zone_intrusion/` | Pretrained person detection (no custom data) | — |

## Label Format

YOLO detection: one `.txt` per image, each line = `class_id cx cy w h` (normalized 0-1).

YOLO pose: `class_id cx cy w h x1 y1 v1 x2 y2 v2 ... x17 y17 v17` (17 COCO keypoints, visibility 0/1/2).

## Adding a New Use Case

1. **Create the folder structure:**
   ```bash
   mkdir -p dataset_store/training_ready/<use_case>/{train,val,test}/{images,labels}
   ```

2. **Add `.gitkeep` to each leaf directory** so the structure is tracked:
   ```bash
   for d in dataset_store/training_ready/<use_case>/{train,val,test}/{images,labels}; do
     touch "$d/.gitkeep"
   done
   ```

3. **Create a data config** at `configs/<use_case>/05_data.yaml`:
   ```yaml
   dataset_name: "<use_case>"
   path: "../../dataset_store/training_ready/<use_case>"
   train: "train/images"
   val: "val/images"
   test: "test/images"

   names:
     0: class_a
     1: class_b
   num_classes: 2
   input_size: [640, 640]

   mean: [0.485, 0.456, 0.406]
   std: [0.229, 0.224, 0.225]
   ```

4. **Place raw source data** in `dataset_store/raw/<use_case>/<source_name>/` (one folder per source; original format, unmodified).

5. **Write a prepare script** in the raw source folder or `scripts/` to merge raw data into the training-ready layout with class remapping and train/val/test splits.

6. **Version with DVC** after merging:
   ```bash
   dvc add dataset_store/training_ready/<use_case>/
   git add dataset_store/training_ready/<use_case>.dvc
   git commit -m "<use_case> dataset v1: <N> images"
   git tag <use_case>-data-v1
   ```

## Training-Ready Dataset Preparation

End-to-end workflow to produce high-quality labels for training. All tools write directly to the dataset folder and automatically back up existing labels before overwriting.

### Prerequisites (start services)

```bash
cd services/s18100_sam3_service && docker compose up -d     # SAM3 :18100 (GPU)
cd services/s18104_auto_label && docker compose up -d       # Auto-Label :18104
cd services/s18105_annotation_quality_assessment && docker compose up -d  # QA :18105 (optional)
cd services/s18103_label_studio && docker compose up -d     # Label Studio :18103
```

### Step 1: Auto-Label (generates YOLO labels in dataset folder)

Auto-annotate writes labels directly to `<dataset>/<split>/labels/`. If labels already exist, they are backed up to `<split>/labels_backup_<timestamp>/` first.

Two pipeline modes, selected automatically based on config:

**Direct detection** (fire, phone, fall — no `auto_label` in config):
SAM3 detects final classes directly using text prompts.

**Rule-based detection** (helmet, shoes — `auto_label` in config):
SAM3 detects intermediate objects (e.g., `head`, `helmet`), then IoU overlap rules derive final classes (e.g., `head_with_helmet`, `head_without_helmet`). Optional VLM (Qwen3.5 via Ollama) verifies uncertain detections using priority-based sampling.

```bash
# Dry-run first to preview (no files written)
uv run core/p01_auto_annotate/run_auto_annotate.py \
  --data-config configs/<use_case>/05_data.yaml --mode text --dry-run

# Run auto-annotation (writes to dataset_store/<use_case>/<split>/labels/)
uv run core/p01_auto_annotate/run_auto_annotate.py \
  --data-config configs/<use_case>/05_data.yaml --mode text

# For a flat image directory with ad-hoc classes
uv run core/p01_auto_annotate/run_auto_annotate.py \
  --image-dir dataset_store/site_collected/<use_case>/images \
  --classes "0:person,1:helmet" --mode text
```

Modes: `text` (text-prompted, best quality), `auto` (automatic), `hybrid` (both).

#### Rule-Based Config (for "with/without" classes)

Add an `auto_label` section to `05_data.yaml` to enable rule-based detection:

```yaml
auto_label:
  detection_classes:                    # intermediate objects for SAM3
    person: "a person standing or walking"
    head: "a person's head, face"
    helmet: "a hard hat, safety helmet"
  class_rules:                          # IoU overlap → final class mapping
    - { output_class_id: 0, source: person, condition: direct }
    - { output_class_id: 1, source: head, condition: overlap, target: helmet, min_iou: 0.3 }
    - { output_class_id: 2, source: head, condition: no_overlap, target: helmet, min_iou: 0.3 }
  vlm_verify:                           # optional VLM quality check
    enabled: true
    model: "qwen3.5:9b"                # Ollama model name
    verify_classes: [1, 2]              # which classes to verify
    priority:
      low_confidence_threshold: 0.5     # detections below this → higher VLM priority
      small_box_threshold: 0.02         # area < 2% of image → higher VLM priority
      prioritize_derived: true          # overlap/no_overlap classes → higher VLM priority
    budget:
      sample_rate: 0.10                 # verify up to 10% of detections
      max_samples: 100                  # hard cap on VLM calls
```

Rule conditions: `direct` (pass-through), `overlap` (source IoU ≥ min_iou with target), `no_overlap` (source not overlapping any target). VLM requires Ollama running (`ollama serve` + `ollama pull qwen3.5:9b`).

### Step 2: Quality Check (validates and fixes labels)

Run structural validation and optional SAM3 verification. Use `--apply-fixes` to automatically correct labels (backs up originals first).

```bash
# Structural-only QA (no SAM3 required)
uv run core/p02_annotation_qa/run_qa.py \
  --data-config configs/<use_case>/05_data.yaml --no-sam3

# Full QA with SAM3 verification
uv run core/p02_annotation_qa/run_qa.py \
  --data-config configs/<use_case>/05_data.yaml

# Auto-apply fixes (clips out-of-bounds, removes duplicates/degenerates)
uv run core/p02_annotation_qa/run_qa.py \
  --data-config configs/<use_case>/05_data.yaml --apply-fixes
```

Fix types applied automatically:
- `clip_bbox` — coordinates outside [0, 1] clipped to valid range
- `remove_duplicate` — near-identical annotations removed
- `remove_degenerate` — zero-area or tiny boxes removed

### Step 3: Human Review (Label Studio)

Import labels into Label Studio for visual review and correction. Reviewers see the actual dataset images with bounding box overlays.

```bash
# Create a review project (once per use case)
uv run core/p04_label_studio/bridge.py setup \
  --data-config configs/<use_case>/05_data.yaml

# Import current labels for review
uv run core/p04_label_studio/bridge.py import \
  --data-config configs/<use_case>/05_data.yaml

# Or import from auto-annotate output specifically
uv run core/p04_label_studio/bridge.py import \
  --data-config configs/<use_case>/05_data.yaml \
  --from-auto-annotate runs/<use_case>/*_06_auto_annotate/

# Or import QA-flagged images for targeted review
uv run core/p04_label_studio/bridge.py import \
  --data-config configs/<use_case>/05_data.yaml \
  --from-qa-fixes runs/<use_case>/*_05_annotation_quality/fixes.json
```

Review annotations in the browser at **http://localhost:18103** (default account: `nthanhtrung198@gmail.com` / `Trung123`).

### Step 4: Export Reviewed Labels

Export reviewed annotations back to the dataset. When `--data-config` is provided without `--output-dir`, labels are written to `<dataset>/train/labels/` with automatic backup.

```bash
# Export to dataset (auto-derives output dir and enables backup)
uv run core/p04_label_studio/bridge.py export \
  --project <use_case>_review \
  --data-config configs/<use_case>/05_data.yaml

# Or export to a specific directory
uv run core/p04_label_studio/bridge.py export \
  --project <use_case>_review \
  --output-dir dataset_store/<use_case>/val/labels \
  --data-config configs/<use_case>/05_data.yaml --backup

# Export only human-reviewed tasks (skip unreviewed)
uv run core/p04_label_studio/bridge.py export \
  --project <use_case>_review \
  --data-config configs/<use_case>/05_data.yaml --only-reviewed
```

### Step 5: Train

Labels are now in the dataset folder — train directly.

```bash
uv run core/p06_training/train.py \
  --config configs/<use_case>/06_training.yaml
```

### Backup and Recovery

All tools back up existing labels before overwriting:
- **Auto-annotate**: `<split>/labels_backup_<YYYYMMDD_HHMMSS>/`
- **QA apply-fixes**: `<split>/labels_backup_<YYYYMMDDTHHMMSSZ>/`
- **Bridge export**: `<output_dir>/.backup_<YYYYMMDD_HHMMSS>/`

To restore from backup:
```bash
# Remove current labels and restore from backup
rm dataset_store/<use_case>/train/labels/*.txt
cp dataset_store/<use_case>/train/labels_backup_<timestamp>/*.txt \
   dataset_store/<use_case>/train/labels/
```

### Quick Reference

| Step | Tool | Writes to | Backup |
|------|------|-----------|--------|
| Auto-label (direct) | `run_auto_annotate.py` (no `auto_label` in config) | `<dataset>/<split>/labels/` | `labels_backup_<ts>/` |
| Auto-label (rule-based) | `run_auto_annotate.py` (with `auto_label` in config) | `<dataset>/<split>/labels/` | `labels_backup_<ts>/` |
| + VLM verify | Qwen3.5 via Ollama (auto, priority sampled) | Filters detections in-memory | N/A |
| QA check | `run_qa.py` | Report in `runs/` | N/A (read-only) |
| QA fix | `run_qa.py --apply-fixes` | `<dataset>/<split>/labels/` | `labels_backup_<ts>/` |
| LS import | `bridge.py import` | Label Studio DB | N/A |
| LS export | `bridge.py export` | `<dataset>/train/labels/` | `.backup_<ts>/` |
| Train | `train.py` | `runs/<use_case>/` | N/A |

### Pipeline Selection

| Config has `auto_label`? | Pipeline used | VLM | Use cases |
|--------------------------|---------------|-----|-----------|
| No | Standard (StateGraph) | No | fire, phone, fall |
| Yes | Rule-based (Functional API) | Auto if Ollama available | helmet, shoes |

## Raw Data Sources

Download raw datasets into `dataset_store/raw/<use_case>/<source_name>/`. Each source keeps its original format and structure.

| Use Case | Source | Download |
|---|---|---|
| Fire | D-Fire (21K images, YOLO) | `kaggle: sayedgamal99/smoke-fire-detection-yolo` |
| Fire | FASDD_CV (95K images, COCO JSON) | `kaggle: yuulind/fasdd-cv-coco` |
| Fire | Zenodo Indoor Fire (5K, YOLO) | `zenodo: 15826133` |
| Helmet | Hard Hat Workers (VOC XML) | `kaggle: andrewmvd/hard-hat-detection` |
| Helmet | SH17 PPE (YOLO+VOC) | `kaggle: mugheesahmad/sh17-dataset-for-ppe-detection` |
| Helmet | SFCHD (12K, YOLO) | `github: ZijianL/SFCHD-SCALE` |
| Helmet | HardHat-Vest v3 (YOLO) | `kaggle: muhammetzahitaydn/hardhat-vest-dataset-v3` |
| Shoes | keremberke PPE (COCO JSON) | `huggingface: keremberke/protective-equipment-detection` |
| Shoes | shoe_ppe (1.5K, YOLO) | `roboflow: harnessafesite/shoe_ppe` |
| Fall | Roboflow Fall (10K, YOLO) | `roboflow: roboflow-universe-projects/fall-detection-ca3o8` |
| Fall | CCTV Fall (112 images, YOLO Pose) | `kaggle: simuletic/cctv-incident-dataset-fall-and-lying-down-detection` |
| Fall | COCO Keypoints (58K, pretrain) | `kaggle: asad11914/coco-2017-keypoints` |
| Phone | FPI-Det (45K, YOLO) | `github: AstraBert/FPI-Det` |
| Phone | Phone Call Usage (3K, YOLO) | `roboflow: phoneusagedetection/phone-call-usage` |

## Source Traceability

All filenames carry a prefix that traces back to the raw source:

| Prefix | Source | Raw Location |
|---|---|---|
| `dfire_` | D-Fire | `raw/fire_detection/d_fire/` |
| `fasdd_` | FASDD_CV | `raw/fire_detection/fasdd_cv/` |
| `hhvest_` | HardHat-Vest v3 | `raw/helmet_detection/hardhat_vest_v3/` |
| `ppev8_` | PPE YOLOv8 | `raw/helmet_detection/ppe_yolov8/` |
| `sfchd_` | SFCHD | `raw/helmet_detection/sfchd/` |
| `sh17_` | SH17 PPE | `raw/helmet_detection/sh17_ppe/` |
| `shoeppe_` | shoe_ppe | `raw/shoes_detection/shoe_ppe/` |
| `kbppe_` | keremberke PPE | `raw/shoes_detection/keremberke_ppe/` |
| `rffall_` | Roboflow Fall | `raw/fall_detection/roboflow_fall/` |
| `cctvfall_` | CCTV Fall | `raw/fall_detection/cctv_fall/` |
| `falldet_` | Fall Detection Imgs | `raw/fall_detection/fall_detection_imgs/` |
| `cocostand_` | COCO standing persons | `raw/fall_detection/coco_keypoints/` |
| `fpi_` | FPI-Det | `raw/phone_detection/fpi_det/` |
| `phonecall_` | Phone Call Usage | `raw/phone_detection/phone_call_usage/` |

## Site-Collected Data

Images collected on-site at deployment locations (factories, construction sites). Flat layout (no train/val/test split) — merge into the training-ready layout using prepare scripts.

```
dataset_store/site_collected/
├── fire_detection/         {images, labels}
├── helmet_detection/       {images, labels}
├── shoes_detection/        {images, labels}
├── fall_detection/         {images, labels}  (covers both classify + pose)
└── phone_detection/        {images, labels}
```

Place site-collected images and YOLO labels here. Run the appropriate prepare script to merge into the training-ready layout.

## DVC Versioning

```bash
# Track a dataset
dvc add dataset_store/training_ready/fire_detection/
git add dataset_store/training_ready/fire_detection.dvc
git commit -m "fire dataset v2: +5K images"
git tag fire-data-v2

# Reproduce exact version
git checkout fire-data-v2 && dvc checkout
```

Tag convention: `<use_case>-data-v<N>` for datasets.

## Validation

Run data exploration to validate a dataset after merging:

```bash
uv run python utils/exploration.py --config configs/<use_case>/05_data.yaml
```

Update `mean`/`std` in the data config with the computed values.

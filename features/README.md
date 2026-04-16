# Features — Self-Contained Per-Feature Use Cases

Every directory under `features/` is a self-contained feature. All assets for
a use-case — configs, custom code, samples, notebooks, tests, training runs,
eval reports, exports, inference outputs, and tagged releases — live in one
folder. `core/` stays use-case-agnostic and is driven entirely by these
configs.

## Uniform Layout

```
features/<name>/
  README.md        Purpose, classes, dataset, commands
  configs/         Phase YAMLs: 00_data_preparation, 05_data, 06_training,
                   08_evaluation, 09_export, 10_inference
  code/            OPTIONAL — custom trainer / dataset / model / train.py
                   Referenced from configs via dotted path. See `ai/CLAUDE.md`.
  samples/         Small smoke-test images & clips (tracked)
  notebooks/       Exploration / one-off scripts (*.ipynb, *.py)
  tests/           Per-feature pytest (optional)
  runs/            Training checkpoints (gitignored; DVC tracks best.pt)
  eval/            Evaluation reports & metrics (gitignored)
  export/          Exported ONNX / TFLite / INT8 (gitignored)
  predict/         Inference outputs (images, videos, JSON) (gitignored)
  .gitignore       Scoped to this feature
```

Shared pipeline templates live at `configs/_shared/` (non-authoritative).
CI fixtures live at `configs/_test/`. Features **never** fall back to
`_shared` at runtime — every feature is authoritative for itself.

## Naming Contract

| Concept | Convention | Example |
|---|---|---|
| Feature folder | `<category>-<name>` (kebab-hyphen + snake) | `safety-fire_detection` |
| Categories | `access-`, `ppe-`, `safety-`, `traffic-` (per `docs/03_platform/`) | `ppe-helmet_detection` |
| `dataset_name` in 05_data.yaml | snake_case — **folder name with `-` → `_`** | `safety_fire_detection` |
| `training_ready/` subdir | equals `dataset_name` | `dataset_store/training_ready/safety_fire_detection/` |

`scripts/new_feature.sh` derives `dataset_name` automatically from the
folder name, so you never edit the mapping by hand. The one legacy
exception is `detect_vehicle` (no platform doc yet).

## Current Features

| Folder | Task | Notes |
|---|---|---|
| `safety-fire_detection` | Detection | Fire + smoke bounding boxes |
| `safety-fall-detection` | Detection | Fallen-person bounding boxes |
| `safety-fall_pose_estimation` | Keypoint | Fall detection via pose keypoints |
| `safety-poketenashi-phone-usage` | Detection | Phone-use violation detection |
| `safety-poketenashi` | Detection | Poketenashi (umbrella container) |
| `ppe-helmet_detection` | Detection | Hard-hat PPE compliance |
| `ppe-shoes_detection` | Detection | Safety-shoes PPE detection |
| `ppe-gloves_detection` | Detection | Glove PPE compliance |
| `access-face_recognition` | Face | Enrollment + verification |
| `access-zone_intrusion` | Detection + logic | Person/vehicle intrusion into restricted zones |
| `detect_vehicle` | Detection | Generic vehicle detection (legacy name) |

## Add a New Feature

```bash
bash scripts/new_feature.sh ppe-my_new_feature
# edit features/ppe-my_new_feature/configs/{05_data,06_training}.yaml
```

The scaffold copies `features/_TEMPLATE/`, substitutes the name, and
derives the snake-case `dataset_name`. No changes to `core/` or
`app_demo/tabs/` are required — everything resolves from configs.

For the end-to-end **train / evaluate / export / deploy** workflow and
the **raw-unlabeled → SAM3 auto-label → QA → train** flow, see the root
[`README.md`](../README.md) — commands are identical for every feature,
only the config paths change.

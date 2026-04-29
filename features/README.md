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
folder name, so you never edit the mapping by hand.

## Phase 1 Features

### Safety detection

| Folder | Task | Detects |
|---|---|---|
| `safety-fire_detection` | Detection | Fire + smoke (ML model) |
| `safety-fall-detection` | Detection | Fallen person (bounding-box model) |
| `safety-fall_pose_estimation` | Pose keypoints | Fall via torso angle; also backs poketenashi pose rules |

### Poketenashi — Prohibited / Required Actions

The poketenashi rule family is now split into one feature folder per
rule (each with its own configs/, code/, samples/, tests/). All five
share the DWPose ONNX in `pretrained/safety-poketenashi/` and follow
the same pose-rule pattern.

| Folder | Task | Behavior / Signal |
|---|---|---|
| `safety-poketenashi_phone_usage` | Detection (sub-model) | ❌ Using mobile phone while walking |
| `safety-poketenashi_hands_in_pockets` | Pose rule | ❌ Wrists inside torso band |
| `safety-poketenashi_stair_diagonal` | Pose + tracking | ❌ Trajectory angle vs stair axis |
| `safety-poketenashi_no_handrail` | Pose + zone rule | ❌ Hand keypoint outside railing zone |
| `safety-poketenashi_point_and_call` | Pose rule + sequence FSM | ❌ No pointing-and-calling at crosswalk checkpoint |

Only `safety-poketenashi_phone_usage` requires its own training data;
the other four are pretrained-only (DWPose) + per-rule logic in the
feature's `code/` folder.

### PPE compliance

| Folder | Task | Detects |
|---|---|---|
| `ppe-helmet_detection` | Detection | Hard-hat PPE compliance |
| `ppe-shoes_detection` | Detection | Safety-shoes compliance |

### Access control

| Folder | Task | Detects |
|---|---|---|
| `access-face_recognition` | Face | Enrollment + identity verification |
| `access-zone_intrusion` | Detection + zone logic | Person / vehicle in restricted zones |

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

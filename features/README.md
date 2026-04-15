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
  release/         Versioned deploy bundles (v<semver>/, symlink latest/)
  .gitignore       Scoped to this feature
```

Shared pipeline templates live at `configs/_shared/` (non-authoritative).
CI fixtures live at `configs/_test/`. Features **never** fall back to
`_shared` at runtime — every feature is authoritative for itself.

## Current Features (8)

Folder names follow a **`<category>-<name>`** convention aligned with
`docs/03_platform/`. Categories: `access-`, `ppe-`, `safety-`, `traffic-`.
The sole exception is `detect_vehicle`, which has no platform doc yet.

| Folder | Task | Notes |
|---|---|---|
| `safety-fire_detection` | Detection | Fire + smoke bounding boxes |
| `ppe-helmet_detection` | Detection | Hard-hat PPE compliance |
| `detect_vehicle` | Detection | Generic vehicle detection |
| `ppe-shoes_detection` | Detection | Safety-shoes PPE detection |
| `access-zone_intrusion` | Detection + logic | Person/vehicle intrusion into restricted zones |
| `safety-fall-detection` | Classification | Fall event classification from crops |
| `safety-fall_pose_estimation` | Keypoint | Fall detection via pose keypoints |
| `access-face_recognition` | Face | Enrollment + verification |

## Add a New Feature

```bash
bash scripts/new_feature.sh my_new_feature
# then edit features/my_new_feature/configs/{05_data,06_training}.yaml
uv run python core/p06_training/train.py \
  --config features/my_new_feature/configs/06_training.yaml
```

The scaffold copies `features/_TEMPLATE/` and substitutes the name. No
changes to `core/` or `app_demo/tabs/` are required — tabs and training
pipeline resolve everything from configs.

See `features/_TEMPLATE/README.md` for the feature README template.

# <feature_name>

One-line purpose (what does this feature detect / classify / predict?).

| Field | Value |
|-------|-------|
| Task | Detection / Classification / Segmentation / Keypoint |
| Recommended model | e.g. YOLOX-M, D-FINE-S, timm-mobilenetv3 |
| Classes | `class_a`, `class_b` |
| Dataset | `dataset_store/training_ready/<dataset_name>/` |

## Feature-specific notes

Document anything that is unique to this feature: non-standard classes,
custom alert logic, special augmentation, dataset caveats, known
failure modes. Skip sections that don't apply.

## Custom code (optional)

When the stock `core/` pipeline isn't enough, drop code into `code/` and
reference it from your config:

```yaml
# features/<name>/configs/06_training.yaml
training:
  backend: custom
  custom_trainer_class: features.<name>.code.custom_trainer.MyTrainer
```

See root [`README.md`](../../README.md) → "Bringing Your Own Code" for
the three escape hatches (config-only, registry override, fully custom).

---

**Layout, workflow, commands.** Every feature uses the same folder
layout and CLI. Don't duplicate them here — see:

- [`features/README.md`](../README.md) — uniform layout + naming contract
- [root `README.md`](../../README.md) — train / evaluate / export / infer
  commands (swap `<name>` into the config paths)

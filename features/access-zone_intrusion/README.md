# Detect Zone Intrusion

Person / vehicle intrusion into restricted zones using polygon logic.

| Field | Value |
|-------|-------|
| Task | Detection + logic |
| Recommended model | YOLOX-S (COCO-pretrained) + zone polygons |
| Classes | `person`, `vehicle` |
| Dataset | `dataset_store/training_ready/zone_intrusion/` |
| Dataset report | `../../dataset_store/training_ready/zone_intrusion/DATASET_REPORT.md` |

---

**Layout, workflow, commands.** Every feature shares the same folder
layout and CLI — don't duplicate them here. See:

- [`features/README.md`](../README.md) — uniform layout + naming contract
- [root `README.md`](../../README.md) — end-to-end train / evaluate /
  export / infer, plus the raw-unlabeled → SAM3 auto-label → QA flow.
  Swap `<name>` into the config paths shown there.

Custom code (optional): drop code into `code/` and reference it from a
config, e.g.

```yaml
# configs/06_training.yaml
training:
  backend: custom
  custom_trainer_class: features.<name>.code.custom_trainer.MyTrainer
```

See root README → "Bringing Your Own Code" for the three escape hatches.

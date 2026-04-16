# ppe-gloves_detection

One-line purpose (what does this feature detect / classify / predict?).

| Field | Value |
|-------|-------|
| Task | Detection / Classification / Segmentation / Keypoint |
| Recommended model | e.g. YOLOX-M, D-FINE-S, timm-mobilenetv3 |
| Classes | `class_a`, `class_b` |
| Dataset | `dataset_store/training_ready/<dataset_name>/` |

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

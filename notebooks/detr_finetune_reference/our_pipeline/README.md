# our_pipeline — CPPE-5 via our in-repo training code

A controlled experiment: **same data, same hyperparameters, different trainer
pipeline.** Uses our `core/p06_training/train.py` + `core/p05_data/` transforms
through the HF Trainer backend (`core/p06_training/hf_trainer.py`). The result
gets diffed against the reference notebook's test mAP 0.5585 (seed=42).

## Prereq: dump CPPE-5 to disk

```bash
.venv-notebook/bin/python notebooks/detr_finetune_reference/data_loader.py --dump-cppe5
```

Writes `dataset_store/training_ready/cppe5/{train,val,test}/{images,labels}/`
in YOLO format. Split is 850/150/29 matching qubvel's `seed=1337` split.

## Run (from repo root, main `.venv`)

```bash
CUDA_VISIBLE_DEVICES=1 uv run core/p06_training/train.py \
  --config notebooks/detr_finetune_reference/our_pipeline/06_training.yaml
```

Outputs land in `runs/<timestamp>_p06_training/` next to this README.
HF-Trainer-standard contents:

```
runs/<ts>/
├── checkpoint-<step>/           # model + optimizer + scheduler + rng
├── runs/<ts>_<host>/            # tensorboard events
├── trainer_state.json
└── all_results.json
```

## Expected result

Test mAP in **[0.50, 0.56]**. Anything ≥ 0.50 means our pipeline is
structurally equivalent to the reference for DETR-family fine-tune. The
residual 0.02-0.05 delta vs the reference's 0.5585 is expected from:

- torchvision v2 augments ≠ Albumentations 1.4.6 (different numerics)
- Two non-deterministic CUDA kernels under `warn_only=True`
- `subset` / shuffle order differences (mitigated by `seed=data_seed=42`)

## What is *not* a fair apples-to-apples

- Different aug libraries ⇒ we don't expect bit-exact numbers, only same
  trajectory shape and similar final mAP.
- If test mAP < 0.45 we have a real pipeline bug, not library drift.

Compare per-class APs (Coverall / Face_Shield / Gloves / Goggles / Mask)
against the reference notebook's seed=42 Bundle B run (0.6286 / 0.6894 /
0.5391 / 0.4180 / 0.5176) to confirm the class-level pattern matches.

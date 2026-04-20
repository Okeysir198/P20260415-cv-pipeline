# our_dfine_torchvision — D-FINE-large via our in-repo pipeline (torchvision v2 CPU aug)

**Arch**: D-FINE-large (`ustc-community/dfine-large-coco`, ~31M params).
**Aug backend**: torchvision v2 transforms.

Companion to `../our_dfine_albumentations/` — the **only** semantic difference
is `augmentation.library: torchvision`. Same model, same hyperparameters
(qubvel's recipe: bs=8, lr=5e-5, warmup=300, linear, WD=0, 30 epochs,
seed=42, `bf16: false`), same dataset (CPPE-5 seed=1337 split).

Purpose: mirrors the `our_rtdetr_v2_albumentations` vs `our_rtdetr_v2_torchvision`
experiment — measures whether our torchvision v2 transform path produces
the same convergence as the Albumentations fast path, for D-FINE specifically.

## Prereq

Same CPPE-5 dump as the other `our_*` experiments:

```bash
.venv-notebook/bin/python notebooks/detr_finetune_reference/data_loader.py --dump-cppe5
```

## Run

```bash
CUDA_VISIBLE_DEVICES=1 uv run core/p06_training/train.py \
  --config notebooks/detr_finetune_reference/our_dfine_torchvision/06_training.yaml
```

Outputs land in `runs/seed42/`.

## Expected

Based on the RT-DETRv2 comparison (torchvision landed within ±0.02 of
Albumentations on test mAP), D-FINE torchvision should produce similar
per-epoch val mAP and test mAP to `our_dfine_albumentations` within the
0.047σ seed-variance band measured there. Wall time will be ~2× slower
per epoch — torchvision v2 runs color/perspective on pre-resize uint8
images (see `../our_rtdetr_v2_torchvision/README.md` for the breakdown).

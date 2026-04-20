# our_rtdetr_v2_torchvision — RT-DETRv2 via our in-repo pipeline (torchvision v2 CPU aug)

**Planned.** Same as `our_rtdetr_v2_albumentations/` but with
`augmentation.library: torchvision` in `06_training.yaml` — measures
the performance cost of torchvision v2 vs Albumentations for the same
augmentation set.

## What will go here (once filled in)

```
our_rtdetr_v2_torchvision/
├── 05_data.yaml                # shared with albumentations variant
├── 06_training.yaml            # augmentation.library: torchvision, rest identical
├── README.md                   # same format as albumentations variant
└── runs/                       # HF-Trainer-standard outputs
```

## Expected comparison vs `our_rtdetr_v2_albumentations/`

- Test mAP: within noise (≈ 0.559 ± 0.01). Augmentation library choice
  is numerically different but statistically equivalent at 40 epochs.
- Wall time: ~2× slower per epoch (measured on CPPE-5:
  torchvision 27.8 s/ep vs Albumentations 15.7 s/ep at bs=16, 2 workers).
- Per-class AP pattern: similar shape; fringe differences on rare
  classes (Goggles, Face_Shield) due to different color/perspective
  transform numerics.

To create: copy `../our_rtdetr_v2_albumentations/` and flip
`augmentation.library` in `06_training.yaml`.

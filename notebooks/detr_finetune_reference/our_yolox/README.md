# our_yolox — YOLOX-M via our in-repo pipeline (CPPE-5)

**Planned.** Uses `core/p06_training/train.py --backend pytorch` for
`model.arch: yolox-m` on CPPE-5. Unlike the RT-DETRv2 / D-FINE variants,
YOLOX goes through our custom pytorch trainer — HF Trainer does not
support YOLOX in our pipeline.

## What will go here

```
our_yolox/
├── 05_data.yaml                # CPPE-5 data config
├── 06_training.yaml            # backend: pytorch, YOLOX hyperparams
├── README.md                   # setup + expected numbers
└── runs/                       # `features/<feature>/runs/` style outputs (custom trainer)
```

## Notes for this variant

- **Backend: `pytorch`** (not `hf`). Our HF backend config validator
  hard-fails on `output_format='yolox'`.
- **Normalization: `[0, 255]` raw** — Megvii YOLOX weights expect
  unnormalized pixel inputs. Set `augmentation.normalize: false` AND
  skip the ImageNet rescale.
- **Mosaic + MixUp are essential** for YOLOX and supported on the
  pytorch backend; they're forced off for DETR-family.
- **`model.impl: custom`** uses our in-repo YOLOX reimplementation.
  `model.impl: official` requires the separate `.venv-yolox-official/`
  venv (see repo root `CLAUDE.md`).

Reference numbers: qubvel's notebooks don't cover YOLOX. Compare
directly against our `features/safety-fire_detection/` YOLOX-M runs
(test mAP50 ≈ 0.49 on fire_detection with TTA).

To create: copy `features/safety-fire_detection/configs/06_training_yolox.yaml`
into `06_training.yaml` here, swap `dataset_config: 05_data.yaml` to point
at CPPE-5, and set the feature's save_dir to this folder's `runs/`.

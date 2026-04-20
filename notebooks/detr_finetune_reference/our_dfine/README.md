# our_dfine — D-FINE-large via our in-repo pipeline (CPPE-5)

**Planned.** Uses `core/p06_training/train.py --backend hf` for
`model.arch: dfine-large` on CPPE-5, to compare against
`../reference_dfine/` (qubvel's D-FINE recipe with lr=2e-5 fix).

## What will go here

```
our_dfine/
├── 05_data.yaml                # CPPE-5 data config
├── 06_training.yaml            # backend: hf, dfine-large hyperparams
├── README.md                   # setup + expected numbers
└── runs/                       # HF-Trainer-standard outputs
```

## Hyperparameter recipe to match (applied from `reference_dfine/`)

| Knob | Value |
|---|---|
| `model.arch` | `dfine-large` |
| `model.pretrained` | `ustc-community/dfine-large-coco` |
| `lr` | 2e-5 (halved vs RT-DETRv2 — larger backbone) |
| `warmup_steps` | 500 |
| `weight_decay` | 1e-4 |
| `bs` | 8 (memory-bound for dfine-large) |
| `epochs` | 30 |

To create: copy `../our_rtdetr_v2_albumentations/06_training.yaml`, flip
`model.arch` + `model.pretrained` + the three LR-related knobs, and
point `logging.save_dir` at this folder's `runs/`.

# reference_dfine — qubvel's D-FINE fine-tune reference (CPPE-5)

Runnable `.py` ports of qubvel's D-FINE HF notebooks, plus the
byte-for-byte original `.ipynb` files in the same folder.

## Contents

| File | Purpose |
|---|---|
| `finetune.py` | D-FINE-large fine-tune on CPPE-5. Takes `--seed`, `--tag`, `--aug` flags. Writes to `runs/dfine_large_cppe5_seed{SEED}{_TAG}/`. |
| `inference.py` | Single-image inference using the saved best checkpoint. |
| `DFine_finetune_on_a_custom_dataset.ipynb` | Upstream original notebook. |
| `DFine_inference.ipynb` | Upstream original inference notebook. |
| `runs/` | Training outputs. |

## Status

**D-FINE reference has hyperparameter adjustments** relative to qubvel's
raw recipe. The naïve port (lr=5e-5) plateaus val mAP at ep3 ≈ 0.20
because `ustc-community/dfine-large-coco` is ~3× the parameter count
of `rtdetr_v2_r50vd` — the same LR is too hot for the larger backbone.

Applied fix (baked into `finetune.py`):

| Knob | qubvel | **this script** | Why |
|---|---|---|---|
| `learning_rate` | 5e-5 | **2e-5** | halved for the larger backbone |
| `warmup_steps` | 300 | **500** | gentler rampup |
| `lr_scheduler` | linear (default) | **cosine** | |
| `weight_decay` | 0 (default) | **1e-4** | DETR canonical |
| `bf16` | — | **True** | RTX 5090 tensor cores |

## Run

```bash
CUDA_VISIBLE_DEVICES=1 .venv-notebook/bin/python \
  notebooks/detr_finetune_reference/reference_dfine/finetune.py \
  --seed 42 --tag lr2e5_warmup500_cosine_wd_bf16
```

Target val mAP: should climb past 0.25 (vs 0.20 ceiling of naïve port).
Target test mAP: somewhere in 0.35-0.45 (qubvel's published: 0.4485).

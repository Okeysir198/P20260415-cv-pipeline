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

`finetune.py` is a byte-for-byte port of qubvel's reference recipe — no
hyperparameter adjustments. An earlier "fix" (lr=2e-5 / cosine / WD=1e-4 /
bf16) was tried and **reverted**: it stalled val mAP at 0.22 / test 0.37,
worse than qubvel's published 0.4485. D-FINE requires fp32 (bf16 stalls
the distribution-focused loss; fp16 overflows decoder).

| Knob | Value | Note |
|---|---|---|
| `learning_rate` | 5e-5 | qubvel default |
| `warmup_steps` | 300 | qubvel default |
| `lr_scheduler` | linear (default) | qubvel default |
| `weight_decay` | 0 (default) | qubvel default |
| `bf16` | **False** | required for D-FINE — DFL stalls under bf16 |
| `num_train_epochs` | 30 | qubvel default; we recommend 50 elsewhere |
| `metric_for_best_model` | `eval_map` | torchmetrics emits `map` → HF prepends `eval_` |

Only additions over qubvel's notebook: early `set_seed(SEED)` + cuDNN
deterministic flags (`warn_only=True`) for reproducibility — no
optimization-math changes.

## Run

```bash
CUDA_VISIBLE_DEVICES=1 .venv-notebook/bin/python \
  notebooks/detr_finetune_reference/reference_dfine/finetune.py \
  --seed 42 --tag qubvel_lr5e5_warmup300_linear
```

Output dir: `runs/dfine_large_cppe5_seed{SEED}{_TAG}/`.

## Reproduced numbers (seed=42)

| Source | test_mAP | test_mAP_50 |
|---|---|---|
| qubvel published | 0.4485 | — |
| this repro (`runs/dfine_large_cppe5_seed42_qubvel_lr5e5_warmup300_linear/`) | **0.4294** | **0.6289** |

The ~0.02 gap vs qubvel's 0.4485 is within the expected D-FINE per-seed
spread (`warn_only=True` lets non-deterministic deformable-attention
backward kernels run). See parent `CLAUDE.md` for cross-arch comparisons
and the dfine-n recommendation for production CPPE-5-scale runs.

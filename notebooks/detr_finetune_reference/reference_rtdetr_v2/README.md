# reference_rtdetr_v2 — qubvel's RT-DETRv2 fine-tune reference (CPPE-5)

Runnable `.py` ports of qubvel's RT-DETRv2 HF notebooks, plus the
byte-for-byte original `.ipynb` files in the same folder. Used as the
known-good baseline to diff against our in-repo training pipeline
(`our_rtdetr_v2_*/`).

## Contents

| File | Purpose |
|---|---|
| `finetune.py` | RT-DETRv2-R50 fine-tune on CPPE-5. Takes `--seed`, `--tag`, `--aug` flags. Writes to `runs/rtdetr_v2_r50_cppe5_seed{SEED}{_TAG}/`. |
| `inference.py` | **Post-training val + test grid generator** — loads the best ckpt and emits `{split}_predictions/final.png` grids in the exact visual style our in-repo `HFValPredictionCallback` uses, so the two pipelines are directly comparable. |
| `RT_DETR_v2_finetune_on_a_custom_dataset.ipynb` | Upstream original notebook — frozen reference for diffing. |
| `RT_DETR_v2_inference.ipynb` | Upstream original inference notebook (single-image Flickr demo). |
| `runs/` | Training + viz outputs (HF-Trainer-standard layout: `checkpoint-N/`, `trainer_state.json`, tensorboard, `{val,test}_predictions/final.png`). |

## Run

Two steps. Training writes the best checkpoint under `_BEST_DIR`;
inference loads it to produce the val + test grids.

```bash
# Step 1 — train (~10 min on RTX 5090)
CUDA_VISIBLE_DEVICES=1 .venv-notebook/bin/python \
  notebooks/detr_finetune_reference/reference_rtdetr_v2/finetune.py \
  --seed 42 --tag bs16_lr1e4_cosine_wd_bf16

# Step 2 — generate val + test GT-vs-Pred grids (~20 s on RTX 5090)
.venv-notebook/bin/python \
  notebooks/detr_finetune_reference/reference_rtdetr_v2/inference.py \
  --seed 42 --tag bs16_lr1e4_cosine_wd_bf16
```

Expected **test mAP ≈ 0.559** on CPPE-5 (seed=42, Bundle B recipe),
within 0.02 of qubvel's published 0.5789.

Inference produces:

- `runs/rtdetr_v2_r50_cppe5_seed42_bs16_lr1e4_cosine_wd_bf16/val_predictions/final.png`
- `runs/rtdetr_v2_r50_cppe5_seed42_bs16_lr1e4_cosine_wd_bf16/test_predictions/final.png`

Each: 12 images (configurable via `--num-samples`), 2-col × 6-row grid,
GT boxes in purple, predictions in green with confidence labels. Same
`annotate_gt_pred` helper our in-repo `HFValPredictionCallback` uses, so
these files are directly comparable against any `our_*_albumentations/`
run's `val_predictions/epoch_NN.png` outputs.

Other `inference.py` flags:
- `--conf 0.15` — DETR sigmoid scores cap at ~0.2; 0.15 matches the
  callback default. Use `--conf 0.1` to show more low-score detections.
- `--num-samples 12` — grid size per split.
- `--splits val test` — restrict to a single split if desired.
- `--sample-seed 42` — the RNG for sampled image indices (so reruns are
  reproducible). Independent of the training `--seed`.

## Hyperparameter recipe (Bundle B, baked into the script)

| Knob | Value | Note |
|---|---|---|
| `model` | `PekingU/rtdetr_v2_r50vd` | |
| `image_size` | 480 | qubvel's default |
| `epochs` | 40 | |
| `lr` | 1e-4 | linear-scaled for bs=16 |
| `weight_decay` | 1e-4 | DETR canonical |
| `warmup_steps` | 300 | |
| `lr_scheduler` | cosine | |
| `max_grad_norm` | 0.1 | aggressive clip, DETR canonical |
| `bs` | 16 | |
| `bf16` | True | RTX 5090 tensor cores |
| Determinism | `use_deterministic_algorithms(warn_only=True)` | |

See top-level `../CLAUDE.md` for full hyperparameter progression
(non-det baseline → deterministic → Bundle A → Bundle B) and the
three-seed variance study.

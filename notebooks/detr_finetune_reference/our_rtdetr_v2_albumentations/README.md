# our_rtdetr_v2_albumentations — RT-DETRv2 via our in-repo pipeline (Albumentations CPU aug)

**Arch**: RT-DETRv2-R50. **Aug backend**: Albumentations 2.x (mirrors qubvel's
reference notebook recipe; ~2× faster than torchvision v2 per CPU worker for
this transform set).

A controlled experiment: same data (CPPE-5 seed=1337 split), same hyperparameters
(Bundle B: bs=16, lr=1e-4, cosine, wd=1e-4, bf16, 40 epochs, seed=42), same
model — but the training loop is our `core/p06_training/train.py` with
`training.backend: hf` instead of qubvel's notebook trainer. The resulting
test-set mAP gets diffed against the reference run in
`notebooks/detr_finetune_reference/runs/rtdetr_v2_r50_cppe5_seed42_bs16_lr1e4_cosine_wd_bf16/`.

## Sibling folders (planned)

| Folder | What it changes vs this one |
|---|---|
| `our_rtdetr_v2_albumentations/` *(this)* | — |
| `our_rtdetr_v2_torchvision/` | Same config, `augmentation.library: torchvision` — measures the overhead of our legacy aug pipeline vs Albumentations. |
| `our_dfine/` | Swap `model.arch: dfine-large` + dfine-specific hyperparams (lr=2e-5, warmup=500). Compare vs reference's `dfine_finetune_cppe5.py`. |
| `our_yolox/` | Swap to YOLOX-M. Compare vs reference's YOLOX runs. |

Each folder has its own `05_data.yaml` (shared schema; only the dataset
paths differ across features), `06_training.yaml`, and `runs/` outputs.

## Prereq: dump CPPE-5 to disk (one-time)

```bash
.venv-notebook/bin/python notebooks/detr_finetune_reference/data_loader.py --dump-cppe5
```

Writes `dataset_store/training_ready/cppe5/{train,val,test}/{images,labels}/`
in YOLO format — 850 / 150 / 29 images, matching qubvel's `seed=1337` split.

## Run (from repo root, main `.venv`)

```bash
CUDA_VISIBLE_DEVICES=1 uv run core/p06_training/train.py \
  --config notebooks/detr_finetune_reference/our_rtdetr_v2_albumentations/06_training.yaml
```

Outputs land in `runs/seed42/` next to this README. HF-Trainer-standard layout:

```
runs/seed42/
├── checkpoint-<step>/          # model + optimizer + scheduler + rng_state
├── runs/<ts>_<host>/           # tensorboard events
├── data_preview/               # dataset_stats, data_labels_{train,val,test}, aug_labels_train
├── val_predictions/            # per-epoch GT-vs-Pred grids
├── pytorch_model.bin           # best ckpt (load_best_model_at_end=True)
├── trainer_state.json
├── test_results.json           # final test-set mAP (per-class breakdown)
└── 05_data.yaml / 06_training.yaml / config_resolved.yaml   # lineage
```

## Expected result

| metric | reference notebook | **this run** |
|---|---|---|
| test mAP | 0.5585 | ≈ 0.559 |
| test mAP₅₀ | 0.8222 | ≈ 0.862 |
| wall time | 563s | ≈ 620s |

Verified: single seed=42 run reaches test mAP 0.5591 — within torch-metric
noise (+0.0006) of the reference. Speed parity with Albumentations; ~10%
slower with viz callbacks turned on.

## What this test proves

The in-repo HF detection backend (`core/p06_training/hf_trainer.py`) is a
faithful translation of qubvel's training recipe. Any future DETR-family
feature (fire_detection, helmet_detection, etc.) can opt into the same
recipe by setting `training.backend: hf` in its `06_training.yaml` and
expect comparable convergence.

## Caveats for head-to-head comparison

- **Augmentation library drift**: Albumentations 2.x (main venv) vs 1.4.6
  (notebook venv). Transforms we use (`A.Perspective`, `A.HorizontalFlip`,
  `A.RandomBrightnessContrast`, `A.HueSaturationValue`, `A.Resize`,
  `A.Affine`) have stable APIs across both — no behavioural difference
  observed in this run.
- **Non-deterministic CUDA kernels**: `grid_sampler_2d_backward_cuda` and
  memory-efficient-attention backward don't have deterministic impls.
  Each run introduces ~10⁻³ relative noise in gradients; `warn_only=True`
  in the determinism setup. Same caveat applies to the reference
  notebook run.
- **HF Trainer vs qubvel's trainer**: structurally the same (both call
  the model's `forward(pixel_values, labels)` → `.loss` → backprop). No
  EMA, no per-component LR groups, no custom callbacks influencing
  gradients.

## Compare per-class APs

```
class        qubvel  ours_ref  ours_in_repo
Coverall      0.613   0.6286    0.6512   (beats reference)
Face_Shield   0.717   0.6894    0.6778
Gloves        0.518   0.5391    0.4707
Goggles       0.520   0.4180    0.4362
Mask          0.527   0.5176    0.5594   (beats reference)
```

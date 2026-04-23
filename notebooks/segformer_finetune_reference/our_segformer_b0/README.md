# our_segformer_b0 — SegFormer-B0 via our in-repo pipeline (HF backend)

**Arch**: SegFormer-B0 (`nvidia/mit-b0`).
**Task**: Semantic segmentation on
[segments/sidewalk-semantic](https://huggingface.co/datasets/segments/sidewalk-semantic)
(35 classes, class 0 = `unlabeled` is ignored during loss + mIoU).

Companion to `../reference_segformer_b0/` — the upstream HF cookbook /
blog post run as a `.py`/`.ipynb` port. This folder reproduces the
exact same experiment through our pipeline
(`core/p06_training/train.py --backend hf`) so we can compare mIoU
trajectories and confirm there's no silent semantic regression from
our dataset loader + `build_transforms` + callbacks.

## Run

```bash
uv run core/p06_training/train.py \
  --config notebooks/segformer_finetune_reference/our_segformer_b0/06_training.yaml
```

Outputs land in `runs/seed42/` — HF-Trainer-standard layout
(`checkpoint-*/`, `runs/<ts>_<host>/` tensorboard, `trainer_state.json`)
plus our 3-axis observability tree
(`data_preview/`, `val_predictions/`, `test_predictions/`).

## Dataset

Upstream uses `load_dataset("segments/sidewalk-semantic")` (single
`train` split, 1000 images) then `train_test_split(0.2, seed=1)`. Our
pipeline expects an on-disk images+masks layout at

```
dataset_store/training_ready/sidewalk_semantic/
  train/{images,masks}/
  val/{images,masks}/
```

Masks are grayscale PNGs with pixel value == class id (0..34).
The dump is produced by a separate data-loader unit (not part of this
config's responsibility).

## Recipe (matches upstream cookbook)

| Knob | Value | Source |
|---|---|---|
| Pretrained | `nvidia/mit-b0` | cookbook cell 3 |
| Epochs | 50 | cookbook cell 35 |
| Optimizer | AdamW | HF Trainer default |
| LR | 6.0e-5 | cookbook cell 35 |
| Per-device batch (train/eval) | 2 / 2 | cookbook cell 35 |
| Precision | fp32 (`bf16: false, amp: false`) | cookbook default |
| Augmentation | ColorJitter(b=0.25, c=0.25, s=0.25, h=0.1) + HFlip on train only | cookbook cell 29 |
| Normalize | ImageNet mean/std (owned by `SegformerImageProcessor`) | cookbook cell 28 |
| `ignore_index` | 0 (= `unlabeled`) | cookbook cell 23/37 |
| Metric | `mean_iou` (best-ckpt selection) | cookbook cell 37 |
| Report to | `none` (no wandb auth) | HF-backend invariant |

## Result (pending)

| metric | upstream cookbook | **this** | Δ |
|---|---|---|---|
| Test mean_iou | TBD | TBD | TBD |
| Test overall_accuracy | TBD | TBD | TBD |

Fill in after first full run lands under `runs/seed42/`.

## Config invariants

- **No `logging.run_name`** — HF Trainer uses it as the feature folder
  name; `feature_name_from_config_path()` derives the correct path from
  the config path. Stray `run_name` creates ghost `notebooks/<name>/`
  dirs.
- **`logging.report_to: none`** — without `wandb login` the wandb
  callback raises `UsageError` during `trainer.train()` setup and kills
  the run before epoch 1.
- **`training.ignore_index: 0`** — mirrors upstream's masking of the
  `unlabeled` class in both loss and mIoU computation.
- **`training.metric_for_best_model: mean_iou`** — must match the key
  that `evaluate`/`torchmetrics` emits verbatim or HF Trainer silently
  fails to save the best checkpoint.

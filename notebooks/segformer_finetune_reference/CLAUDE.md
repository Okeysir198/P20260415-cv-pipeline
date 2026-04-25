# CLAUDE.md — notebooks/segformer_finetune_reference/

Isolated reproduction of the HF cookbook / blog SegFormer-B0
fine-tune (`nvidia/mit-b0`) on `segments/sidewalk-semantic` (35
classes, gated — requires `HF_TOKEN`).

**Purpose**: known-good baseline to diff against when our in-repo
semantic-segmentation pipeline (HF backend, `hf-segformer`) can't
reproduce the numbers.

Upstream:
- Notebook: https://github.com/huggingface/notebooks/blob/main/examples/semantic_segmentation.ipynb
- Blog post: https://huggingface.co/blog/fine-tune-segformer

## Layout

```
.
├── CLAUDE.md                       (this file)
├── README.md                       (human-facing overview)
├── .gitignore
│
├── reference_segformer_b0/         upstream .py port (finetune.py + inference.py + runs/)
└── our_segformer_b0/               same experiment via core/p06_training/ (05_data.yaml + 06_training.yaml + runs/)
```

Same two-kind folder convention as `notebooks/detr_finetune_reference/`
and `notebooks/image_classification_finetune_reference/`:

- `reference_<arch>/` = upstream baseline (HF cookbook / blog as runnable `.py`).
- `our_<arch>/` = same experiment through our pipeline.

## Venv

> **⚠️ CRITICAL:** Reference `.py` scripts run in `.venv-notebook/`, NOT
> the main `.venv/`. Keeps HF `transformers`/`datasets` pinned
> independently of the main repo. Always invoke via
> `.venv-notebook/bin/python ...`.

Setup via `scripts/setup-notebook-venv.sh`.

## Dataset

### sidewalk-semantic (gated)
- Upstream dataset used by the cookbook. Requires accepting terms on the
  Hub and an `HF_TOKEN`. The repo's `.env` carries one; launch scripts
  source it automatically.
- 35 classes, index 0 = `unlabeled` (ignored for loss + mIoU).
- Single `train` split (1000 images) → `train_test_split(0.2, seed=1)`.

## Key config invariants (for `our_segformer_b0/`)

```yaml
seed: 42

model:
  arch: hf-segformer
  pretrained: nvidia/mit-b0
  input_size: [512, 512]
  ignore_mismatched_sizes: true    # required — decoder head reshapes per num_classes

training:
  # bf16 works on SegFormer on Ampere+ in practice with negligible metric
  # drift. Flip to false for bit-identical cookbook parity (cookbook is fp32).
  bf16: true
  amp:  false
  ignore_index: 0      # unlabeled class — applied to both loss and mIoU
  metric_for_best_model: mean_iou

logging:
  report_to: none      # no wandb auth on HF Trainer (known footgun)
  # Do NOT set run_name — HF Trainer uses it as the output folder name.
```

## Observability

Both sides produce `data_preview/`, `val_predictions/{epochs/,best.png}`,
`val_predictions/error_analysis/`, `test_predictions/`, and
`test_results.json` — same tree as the detection reference runs. See
`core/p06_training/CLAUDE.md` → "Post-train observability" for the full
artefact map. All viz blocks default to enabled; flip off per-block in
`06_training.yaml::training.{data_viz,aug_viz,val_viz,best_viz,error_analysis}`.

## Conversion gotchas (when porting the `.ipynb` → `.py`)

1. Remove shell installs (`!pip`, `!git lfs install`), `notebook_login()`,
   `push_to_hub=True`, `trainer.push_to_hub(...)`.
2. `SegformerFeatureExtractor` → `SegformerImageProcessor` (upstream alias deprecated).
3. `feature_extractor.reduce_labels` → `feature_extractor.do_reduce_labels`.
4. `display(...)` commented out.

## Speed notes (reference-side)

Cookbook defaults (`batch_size=2`, fp32, `num_workers=0`,
`eval_steps=20`) run at ~1 it/s on a 5090 — GPU <5% utilized. The
reference `finetune.py` exposes CLI overrides that close the gap without
changing the recipe's effective behaviour meaningfully:

```bash
.venv-notebook/bin/python .../finetune.py \
  --seed 42 --batch-size 16 --num-workers 8 --bf16
```

Result: ~15× speedup (50 epochs in ~8 min instead of ~2.5h). Set
`HF_MODULES_CACHE=<writable path>` if `~/.cache/huggingface/modules` is
root-owned (blocks `evaluate.load`).

## Invariants for reference-vs-ours comparison

- Same dataset (same HF path + split seed).
- Same pretrained checkpoint (`nvidia/mit-b0`).
- Same input size (`[512, 512]`) + ImageNet normalize via the
  processor (not a manual `Normalize` on top).
- Same batch / lr / epochs / warmup.
- Same `ignore_index` and metric key (`mean_iou`).

## Known gotchas (carry-overs from repo-level CLAUDE.md)

- **`report_to: none`** on HF Trainer. Without `wandb login` the wandb
  callback raises `UsageError` during setup.
- **Do not set `logging.run_name`** — HF Trainer derives `output_dir`
  from it and creates ghost `notebooks/<run_name>/runs/` dirs.
- **HF-backend checkpoints save with `hf_model.` key prefix** — strip
  before `from_pretrained` reload / Optimum export.
- **`ignore_mismatched_sizes: true`** is required because the decoder
  head is reinitialized per `num_labels`; without it `from_pretrained`
  errors on the final conv's shape mismatch.

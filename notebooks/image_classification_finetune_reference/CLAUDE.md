# CLAUDE.md — notebooks/image_classification_finetune_reference/

Isolated reproduction of the HF cookbook image-classification notebook
(`microsoft/swin-tiny-patch4-window7-224` on EuroSAT).
**Purpose**: known-good baseline to diff against when our in-repo
classification pipeline (timm / HF `AutoModelForImageClassification`) can't
reproduce the numbers.

Upstream: https://github.com/huggingface/notebooks/blob/main/examples/image_classification.ipynb

## Layout

```
.
├── CLAUDE.md                       (this file)
├── README.md                       (human-facing overview)
├── data_loader.py                  EuroSAT → ImageFolder dump CLI
├── .gitignore
│
├── reference_swin_tiny/            upstream .py port of the cookbook notebook (finetune.py + inference.py + runs/)
└── our_swin_tiny/                  same experiment via core/p06_training/ (05_data.yaml + 06_training.yaml + runs/)
```

Two-kind folder convention — identical to `notebooks/detr_finetune_reference/`:
- `reference_<arch>/` = upstream baseline (HF cookbook as runnable `.py`).
- `our_<arch>/` = same experiment through our pipeline
  (`core/p06_training/train.py`). Each self-contained with
  `05_data.yaml`, `06_training.yaml`, README, and `runs/`.

## Venv

> **⚠️ CRITICAL:** Reference `.py` scripts run in `.venv-notebook/`, NOT the
> main `.venv/`. Keeps HF `transformers`/`datasets` pinned independently of
> the main repo so cookbook numbers are reproducible across toolchain churn.
> Always invoke via `.venv-notebook/bin/python ...`.

Setup via `scripts/setup-notebook-venv.sh` (same venv as the DETR reference
folder — the albumentations pin is irrelevant here but reusing the venv keeps
everything on one Python toolchain).

## Dataset — EuroSAT

- **Public**, no gated terms. Just `datasets.load_dataset("jonathan-roberts1/EuroSAT")`.
  No HF login token required, no accept-terms step (unlike CPPE-5).
- 27,000 RGB JPEGs, **64×64** native resolution.
- 10 classes, ordered from the dataset's ClassLabel feature:
  `AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture,
  PermanentCrop, Residential, River, SeaLake`.
- Ships as a single "train" split. Our `data_loader.py` does a deterministic
  **80/20 stratified** split (per-class shuffle with `seed=42`) and writes
  ImageFolder layout to `dataset_store/training_ready/eurosat/{train,val}/<class>/`.
- **Keep on-disk images at native 64×64.** Upscaling to 224 happens in the
  transform pipeline (the Swin-T processor does `Resize(224)` + ImageNet
  normalize). Writing pre-upscaled copies wastes disk and bakes in resize
  semantics we may want to change later.

## Key config invariants (for `our_swin_tiny/`)

Inherited from the HF cookbook notebook + our DETR-reference conventions:

```yaml
seed: 42

model:
  # Swin-T expects 224×224 input, ImageNet normalize — the checkpoint's
  # AutoImageProcessor handles this; do not re-normalize manually.
  arch: hf-classification
  hf_name: microsoft/swin-tiny-patch4-window7-224

training:
  # Classification is fp16/bf16-safe (no DFL-like numerical gotcha).
  bf16: true
  amp: false
  patience: 5

logging:
  # Skip wandb hard-fail on HF Trainer — same as DETR-reference configs.
  report_to: none
  # Do NOT set run_name here — HF Trainer's generate_run_dir() would use it
  # as the feature folder name, creating a ghost features/<run_name>/ dir
  # under our tree. Let it derive from the config path.
```

## Conversion gotchas (when porting the cookbook .ipynb → .py)

Same checklist as DETR-reference:

1. Remove shell installs (`get_ipython().system(...)`, `!pip ...`).
2. Comment out `display(...)` — Jupyter-only.
3. If the notebook uses `datasets` 2.x access patterns:
   ```python
   # 2.x:
   # label_names = ds["train"].features["label"].names
   # 4.x (same API for ClassLabel — keep as-is):
   label_names = ds["train"].features["label"].names
   ```
4. Cookbook uses `AutoImageProcessor` — **do not** add a manual
   `Normalize((0.485, ...), (0.229, ...))` on top. The processor already
   does it once.
5. If the notebook pushes to the Hub (`push_to_hub=True`), set it to
   `False` in the `.py` port or supply a dummy repo id — HF Trainer hangs
   on the hub-auth prompt otherwise.

## Invariants for reference-vs-ours comparison

- Same dataset dump (ImageFolder from `jonathan-roberts1/EuroSAT`,
  seed=42, val_split=0.2).
- Same model checkpoint (`microsoft/swin-tiny-patch4-window7-224`).
- Same input size (224) + ImageNet normalize via the processor.
- Same batch size, lr, epochs, warmup.
- Same seed + cuDNN deterministic flags (accept ±0.005 variance).

## Known gotchas

- **EuroSAT native 64×64**: keep disk images at 64 — resize lives in the
  model's image processor, not `data_loader.py`.
- **ImageFolder vs HF Dataset load**: both reference cookbook and our
  pipeline can read ImageFolder via `datasets.load_dataset("imagefolder",
  data_dir=...)` or `torchvision.datasets.ImageFolder`. Keep the dump
  format to one source of truth.
- **`report_to: none`** on HF Trainer. Without `wandb login` the HF
  `wandb` callback raises `UsageError` during setup — same footgun as
  DETR-reference configs.
- **Do not set `logging.run_name`** — HF Trainer derives
  `output_dir` from it and will create ghost `features/<run_name>/runs/`
  dirs. Drop the key or set it to exactly the containing folder name.
- **ClassLabel order must match `id2label.json`** — both reference and
  `our_*` must read `id2label.json` from the dataset root to guarantee
  consistent class-index ↔ name mapping across runs.

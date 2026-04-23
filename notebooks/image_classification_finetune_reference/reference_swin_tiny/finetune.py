#!/usr/bin/env python
"""Swin-tiny fine-tuning on EuroSAT — port of HF's reference notebook.

Upstream:
    https://github.com/huggingface/notebooks/blob/main/examples/image_classification.ipynb

Run in the isolated notebook env (NOT via `uv run` — we standardise on
`.venv-notebook/` across all reference ports; albumentations pin lives there):

    .venv-notebook/bin/python \\
      notebooks/image_classification_finetune_reference/reference_swin_tiny/finetune.py \\
      --seed 42

Deps pinned in notebooks/detr_finetune_reference/pyproject.toml (installed by
scripts/setup-notebook-venv.sh).

Self-contained: all checkpoints + tensorboard logs land under
`notebooks/image_classification_finetune_reference/reference_swin_tiny/runs/seed{SEED}/`
(or `runs/{TAG}_seed{SEED}/` when `--tag` is given) regardless of invoking cwd,
resolved via `__file__`.

Conversion notes vs upstream notebook:
- Shell installs (`!pip`, `!sudo apt`, `!git config`) stripped.
- `notebook_login()` + `push_to_hub=True` removed — local reproduction only.
- `display()` calls commented out (Jupyter-only).
- Image display hooks (`example['image']`, `.resize((200,200))`) dropped.
- `report_to="none"` (upstream relies on `push_to_hub` for artefact storage;
  no tensorboard/wandb dependency here — matches CLAUDE.md guidance for HF
  Trainer without `wandb login`).

Hyperparams preserved from upstream:
    model = "microsoft/swin-tiny-patch4-window7-224"
    lr = 5e-5
    epochs = 3
    per_device_train_batch_size = 32
    per_device_eval_batch_size = 32
    gradient_accumulation_steps = 4
    warmup_ratio = 0.1
    train transform: RandomResizedCrop + RandomHorizontalFlip + ToTensor + Normalize
    val transform:   Resize + CenterCrop + ToTensor + Normalize
    compute_metrics: evaluate.load("accuracy")
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# Parse args (and SEED env) before torch import — matches the RT-DETRv2 port's
# determinism convention.
_argp = argparse.ArgumentParser(add_help=False)
_argp.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 42)))
_argp.add_argument("--tag", type=str, default=os.environ.get("RUN_TAG", ""),
                   help="optional suffix on the run dir, e.g. 'swin_tiny_eurosat'")
_argp.add_argument("--epochs", type=int, default=None,
                   help="override upstream default of 3 epochs")
_argp.add_argument("--output-dir", type=str, default=None,
                   help="override default runs/<tag>_seed<SEED>/ location")
_args, _ = _argp.parse_known_args()
SEED = _args.seed
TAG = _args.tag
EPOCHS_OVERRIDE = _args.epochs
OUTPUT_DIR_OVERRIDE = _args.output_dir
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
from transformers import set_seed  # noqa: E402

set_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

_HERE = Path(__file__).resolve().parent
if OUTPUT_DIR_OVERRIDE:
    _RUN_DIR = Path(OUTPUT_DIR_OVERRIDE).resolve()
else:
    _dir_name = (f"{TAG}_seed{SEED}" if TAG else f"seed{SEED}")
    _RUN_DIR = _HERE / "runs" / _dir_name

# ## Model + batch
model_checkpoint = "microsoft/swin-tiny-patch4-window7-224"
batch_size = 32  # upstream default (per_device_{train,eval}_batch_size)

# ## Loading the dataset
# Upstream uses `jonathan-roberts1/EuroSAT` — a 10-class land-cover dataset
# shipped on HF Hub as an ImageFolder split (train only, we carve val off below).
from datasets import load_dataset  # noqa: E402

dataset = load_dataset("jonathan-roberts1/EuroSAT")

import evaluate  # noqa: E402

metric = evaluate.load("accuracy")

# example = dataset["train"][10]  # (Jupyter display no-op)
# example["image"]
# example["image"].resize((200, 200))

labels = dataset["train"].features["label"].names
label2id, id2label = {}, {}
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

# ## Preprocessing — AutoImageProcessor + torchvision transforms
from transformers import AutoImageProcessor  # noqa: E402

image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)

from torchvision.transforms import (  # noqa: E402
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
if "height" in image_processor.size:
    size = (image_processor.size["height"], image_processor.size["width"])
    crop_size = size
    max_size = None
elif "shortest_edge" in image_processor.size:
    size = image_processor.size["shortest_edge"]
    crop_size = (size, size)
    max_size = image_processor.size.get("longest_edge")

train_transforms = Compose(
    [
        RandomResizedCrop(crop_size),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)

val_transforms = Compose(
    [
        Resize(size),
        CenterCrop(crop_size),
        ToTensor(),
        normalize,
    ]
)


def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [
        train_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch


def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [
        val_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch


# Split training into training + validation (90/10), seeded.
splits = dataset["train"].train_test_split(test_size=0.1, seed=SEED)
train_ds = splits["train"]
val_ds = splits["test"]
train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)

# ## Model
from transformers import AutoModelForImageClassification, Trainer, TrainingArguments  # noqa: E402

model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint,
    label2id=label2id,
    id2label=id2label,
    # Replace pretrained 1000-class ImageNet head with our 10-class EuroSAT head.
    ignore_mismatched_sizes=True,
)

model_name = model_checkpoint.split("/")[-1]

epochs = EPOCHS_OVERRIDE if EPOCHS_OVERRIDE is not None else 3
lr = 5e-5

args = TrainingArguments(
    output_dir=str(_RUN_DIR),
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,  # upstream pushes to Hub; reference port writes locally only
    report_to="none",   # no wandb/tb auth required — see CLAUDE.md
    seed=SEED,
    data_seed=SEED,
)

import numpy as np  # noqa: E402


def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions."""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels_tensor = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels_tensor}


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

if __name__ == "__main__":
    train_results = trainer.train()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # Persist best checkpoint to a stable subdir so `inference.py` can find it.
    _BEST_DIR = _RUN_DIR / "best"
    trainer.save_model(str(_BEST_DIR))
    image_processor.save_pretrained(str(_BEST_DIR))
    print(f"Best model saved to: {_BEST_DIR}")

    # Upstream then pushes to Hub — skipped, local reproduction only.
    # trainer.push_to_hub()

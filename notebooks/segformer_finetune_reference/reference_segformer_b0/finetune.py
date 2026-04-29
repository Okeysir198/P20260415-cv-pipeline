#!/usr/bin/env python
"""SegFormer-B0 fine-tuning on sidewalk-semantic — port of HF's reference notebook.

Upstream:
    https://github.com/huggingface/notebooks/blob/main/examples/semantic_segmentation.ipynb
Blog post:
    https://huggingface.co/blog/fine-tune-segformer

Run in the isolated notebook env (NOT via `uv run` — main venv has a newer
albumentations with different box-clip semantics; SegFormer does not use it
directly, but we standardize on `.venv-notebook/` across all reference ports):

    .venv-notebook/bin/python \\
      notebooks/segformer_finetune_reference/reference_segformer_b0/finetune.py \\
      --seed 42

Deps pinned in notebooks/detr_finetune_reference/pyproject.toml (installed by
scripts/setup-notebook-venv.sh).

Self-contained: all checkpoints + tensorboard logs land under
`notebooks/segformer_finetune_reference/reference_segformer_b0/runs/seed{SEED}/`
(or `runs/{TAG}_seed{SEED}/` when `--tag` is given) regardless of invoking cwd,
resolved via `__file__`.

Conversion notes vs upstream notebook:
- Shell installs (`!pip`, `!git lfs install`) stripped.
- `notebook_login()` + `push_to_hub=True` removed — local reproduction only.
- `display()` calls commented out (Jupyter-only).
- `datasets` 4.x access pattern: `ds.features["objects"]["category"].feature.names`
  not 2.x-style `ds.features["objects"].feature["category"].names`. The
  segmentation dataset here doesn't use that path but the convention is
  documented in notebooks/detr_finetune_reference/CLAUDE.md.
- Dataset kept as `segments/sidewalk-semantic` (gated — requires HF_TOKEN);
  matches CLAUDE.md and is what `runs/seed42/` was trained on.
- `hub_strategy` + `hub_model_id` dropped (no Hub push).
- `report_to="none"` (upstream relies on `push_to_hub` for artefact storage;
  no tensorboard/wandb dependency here — matches CLAUDE.md guidance for HF
  Trainer without `wandb login`).

Hyperparams preserved from upstream:
    model = "nvidia/mit-b0"
    lr = 6e-5
    epochs = 50
    per_device_train_batch_size = 2
    per_device_eval_batch_size = 2
    ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1) on train only
    compute_metrics: evaluate.load("mean_iou") with ignore_index=0
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    p.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 42)))
    p.add_argument("--tag", type=str, default=os.environ.get("RUN_TAG", ""))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=8,
                   help="upstream default was 2; raised to 8 (GPU was <5%% util at bs=2)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--logging-steps", type=int, default=20)
    p.add_argument("--bf16", action="store_true",
                   help="use bf16 mixed precision (Ampere+); cookbook is fp32")
    return p


_HERE = Path(__file__).resolve().parent


def main() -> None:
    args = _build_argparser().parse_args()
    SEED = args.seed

    from transformers import set_seed

    set_seed(SEED)

    if args.output_dir:
        run_dir = Path(args.output_dir).resolve()
    else:
        dir_name = f"{args.tag}_seed{SEED}" if args.tag else f"seed{SEED}"
        run_dir = _HERE / "runs" / dir_name

    # ## Model

    model_checkpoint = "nvidia/mit-b0"  # pre-trained model from which to fine-tune
    batch_size = args.batch_size

    # ## Loading the dataset — gated `segments/sidewalk-semantic` (needs HF_TOKEN)
    from datasets import load_dataset

    hf_dataset_identifier = "segments/sidewalk-semantic"  # gated — requires HF_TOKEN
    ds = load_dataset(hf_dataset_identifier)

    # ## id2label / label2id
    # The sidewalk-semantic datasets bundle an `id2label.json` in the repo.
    import json

    import evaluate
    import torch
    from huggingface_hub import hf_hub_download
    from torch import nn
    from torchvision.transforms import ColorJitter
    from transformers import (
        SegformerForSemanticSegmentation,
        SegformerImageProcessor,
        Trainer,
        TrainingArguments,
    )

    filename = "id2label.json"
    id2label = json.load(
        open(hf_hub_download(hf_dataset_identifier, filename, repo_type="dataset"), "r")
    )
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)
    print(f"num_labels: {num_labels}")

    # **Note**: index 0 is `unlabeled` — we ignore it in the mIoU loss/metric.

    # Shuffle + 80/20 train/test split (upstream uses `seed=1`).
    ds = ds.shuffle(seed=1)
    ds = ds["train"].train_test_split(test_size=0.2)
    train_ds = ds["train"]
    test_ds = ds["test"]

    # ## Preprocessing — torchvision ColorJitter + SegformerImageProcessor
    #
    # Upstream uses `SegformerFeatureExtractor` (deprecated alias); current
    # transformers prefer `SegformerImageProcessor` — behaviourally identical.
    # Normalize uses ImageNet mean/std (processor default).
    feature_extractor = SegformerImageProcessor()
    jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)

    def train_transforms(example_batch):
        images = [jitter(x) for x in example_batch["pixel_values"]]
        labels = [x for x in example_batch["label"]]
        inputs = feature_extractor(images, labels)
        return inputs

    def val_transforms(example_batch):
        images = [x for x in example_batch["pixel_values"]]
        labels = [x for x in example_batch["label"]]
        inputs = feature_extractor(images, labels)
        return inputs

    train_ds.set_transform(train_transforms)
    test_ds.set_transform(val_transforms)

    # ## Model
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # ## TrainingArguments
    epochs = args.epochs if args.epochs is not None else 50
    lr = 6e-5

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_total_limit=3,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        eval_accumulation_steps=5,
        load_best_model_at_end=True,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        bf16=args.bf16,
        push_to_hub=False,         # upstream pushes to Hub; reference port writes locally only
        report_to="none",          # no wandb/tb auth required — see CLAUDE.md
        seed=SEED,
        data_seed=SEED,
    )

    # ## compute_metrics — mean IoU with ignore_index=0
    metric = evaluate.load("mean_iou")

    def compute_metrics(eval_pred):
        with torch.no_grad():
            logits, labels = eval_pred
            logits_tensor = torch.from_numpy(logits)
            # upscale logits (H/4, W/4) to label resolution
            logits_tensor = nn.functional.interpolate(
                logits_tensor,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).argmax(dim=1)

            pred_labels = logits_tensor.detach().cpu().numpy()
            # `_compute` vs `compute`: the evaluate library had a bug handling
            # kwargs in `compute` — upstream notebook uses `_compute` directly.
            # See: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
            metrics = metric._compute(
                predictions=pred_labels,
                references=labels,
                num_labels=len(id2label),
                ignore_index=0,
                reduce_labels=feature_extractor.do_reduce_labels,
            )

            per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
            per_category_iou = metrics.pop("per_category_iou").tolist()

            metrics.update(
                {f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)}
            )
            metrics.update(
                {f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)}
            )

            return metrics

    # ## Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=feature_extractor,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Persist best checkpoint to a stable subdir so `inference.py` can find it.
    best_dir = run_dir / "best"
    trainer.save_model(str(best_dir))
    feature_extractor.save_pretrained(str(best_dir))
    print(f"Best model saved to: {best_dir}")

    # Upstream then pushes to Hub — skipped, local reproduction only.
    # trainer.push_to_hub(...)


if __name__ == "__main__":
    main()

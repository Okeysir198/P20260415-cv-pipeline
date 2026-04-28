#!/usr/bin/env python
"""ViTPose-base fine-tuning on COCO-keypoints — reference port.

Upstream sources:
    Docs:      https://huggingface.co/docs/transformers/en/model_doc/vitpose
    Model:     https://huggingface.co/usyd-community/vitpose-base-simple
    Paper:     https://arxiv.org/abs/2204.12484

ViTPose is a TOP-DOWN 2D pose estimator: input is a single-person crop
(taken from a detector or GT bbox), output is 17 COCO keypoint heatmaps.
This script trains the heatmap head on COCO-keypoints using GT person
boxes for cropping (apples-to-apples training; swap to detector boxes
only when measuring full-pipeline OKS-AP).

Run in the isolated notebook env (NOT via `uv run`):

    .venv-notebook/bin/python \\
      notebooks/vitpose_finetune_reference/reference_vitpose_base/finetune.py \\
      --seed 42 --subset 5000

Pass `--subset 0` for the full COCO train (149k person instances).

Self-contained: all checkpoints land under
`notebooks/vitpose_finetune_reference/reference_vitpose_base/runs/seed{SEED}/`.

Conversion notes vs upstream:
- Shell installs (`!pip`), `notebook_login()`, `push_to_hub=True` stripped.
- `display()` calls removed (Jupyter-only).
- `report_to="none"` (matches CLAUDE.md guidance for HF Trainer without `wandb login`).
- Dataset: `rom1x38/COCO_keypoints` (public parquet mirror of COCO 2017
  person keypoints; multi-person per image, flattened to person crops
  at load time). No HF token required.

Hyperparams:
    model = "usyd-community/vitpose-base-simple"
    input_size = (256, 192)              # H x W, ViTPose default
    lr = 5e-4
    epochs = 30
    per_device_train_batch_size = 32
    AdamW, linear warmup 500 steps, cosine decay
    Loss = per-joint MSE on heatmaps weighted by visibility mask
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

_argp = argparse.ArgumentParser(add_help=False)
_argp.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 42)))
_argp.add_argument("--tag", type=str, default=os.environ.get("RUN_TAG", ""))
_argp.add_argument("--epochs", type=int, default=30)
_argp.add_argument("--lr", type=float, default=5e-4)
_argp.add_argument("--batch-size", type=int, default=32)
_argp.add_argument("--num-workers", type=int, default=8)
_argp.add_argument("--subset", type=int, default=5000,
                   help="Limit train set to N person crops (0 = full set).")
_argp.add_argument("--bf16", action="store_true", default=True)
_argp.add_argument("--output-dir", type=str, default=None)
_args, _ = _argp.parse_known_args()
SEED = _args.seed

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import set_seed  # noqa: E402

set_seed(SEED)

_HERE = Path(__file__).resolve().parent
if _args.output_dir:
    _RUN_DIR = Path(_args.output_dir).resolve()
else:
    _dir_name = f"{_args.tag}_seed{SEED}" if _args.tag else f"seed{SEED}"
    _RUN_DIR = _HERE / "runs" / _dir_name
_RUN_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model + processor
# ---------------------------------------------------------------------------
from transformers import (  # noqa: E402
    AutoImageProcessor,
    Trainer,
    TrainingArguments,
    VitPoseForPoseEstimation,
)

CKPT = "usyd-community/vitpose-base-simple"
NUM_KPTS = 17
INPUT_HW = (256, 192)  # (H, W) ViTPose default

# Lazy: created inside __main__ so `from finetune import ...` (used by
# inference.py) doesn't trigger the model + dataset download as a side
# effect of importing the module.
processor = None
model = None


def _build_model():
    global processor, model
    if processor is None:
        processor = AutoImageProcessor.from_pretrained(CKPT)
    if model is None:
        model = VitPoseForPoseEstimation.from_pretrained(
            CKPT,
            num_labels=NUM_KPTS,
            ignore_mismatched_sizes=True,
        )
    return processor, model

# ---------------------------------------------------------------------------
# Dataset — rom1x38/COCO_keypoints (public parquet mirror of COCO 2017
# person keypoints). Schema:
#   image: PIL JPEG
#   image_id: int
#   bboxes: list of [x, y, w, h]   (one per person)
#   keypoints: list of list[[x,y,v]*17]
# Multi-person per image → flattened to (image, single bbox, single
# keypoints) tuples for top-down ViTPose training.
# ---------------------------------------------------------------------------
from datasets import load_dataset  # noqa: E402

_DATASET_ID = "rom1x38/COCO_keypoints"


def _load_split(split: str, streaming: bool):
    """Load `train` or `val` parquet split."""
    name = "train" if split == "train" else "val"
    return load_dataset(_DATASET_ID, split=name, streaming=streaming)


def _flatten_persons(ds_split, max_samples: int | None = None):
    """Yield one row per person (drop persons with 0 visible keypoints).

    `ds_split` may be streaming (IterableDataset) or in-memory (Dataset).
    Returns a list of {image, bbox, keypoints(flat 51)} dicts.
    """
    out = []
    for row in ds_split:
        bboxes = row["bboxes"]
        kpts = row["keypoints"]
        for b, k in zip(bboxes, kpts):
            arr = np.asarray(k, dtype=np.float32)
            if arr.shape != (NUM_KPTS, 3) or (arr[:, 2] > 0).sum() == 0:
                continue
            out.append({"image": row["image"], "bbox": list(b),
                        "keypoints": arr.flatten().tolist()})
            if max_samples is not None and len(out) >= max_samples:
                return out
    return out


from datasets import Dataset  # noqa: E402


# ---------------------------------------------------------------------------
# Heatmap target encoder (per-joint 2D Gaussian) — standard top-down recipe.
# ---------------------------------------------------------------------------
HEATMAP_HW = (INPUT_HW[0] // 4, INPUT_HW[1] // 4)
HEATMAP_SIGMA = 2.0


def _encode_heatmap(kpts_in_crop: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """kpts_in_crop: (17, 2) in input-pixel coords; vis: (17,) {0,1,2}.

    Returns (17, H/4, W/4) float32 + (17,) target_weight float32.
    """
    H, W = HEATMAP_HW
    target = np.zeros((NUM_KPTS, H, W), dtype=np.float32)
    weight = (vis > 0).astype(np.float32)
    for j in range(NUM_KPTS):
        if weight[j] == 0:
            continue
        mu_x = kpts_in_crop[j, 0] / 4.0
        mu_y = kpts_in_crop[j, 1] / 4.0
        if not (0 <= mu_x < W and 0 <= mu_y < H):
            weight[j] = 0
            continue
        x = np.arange(W, dtype=np.float32)
        y = np.arange(H, dtype=np.float32)[:, None]
        target[j] = np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * HEATMAP_SIGMA ** 2))
    return target, weight


def _expand_bbox(bbox, img_w, img_h, ratio_hw=INPUT_HW, padding=1.25):
    """Expand COCO xywh bbox to ViTPose's aspect ratio + padding."""
    x, y, w, h = bbox
    cx, cy = x + w / 2.0, y + h / 2.0
    aspect = ratio_hw[1] / ratio_hw[0]  # W/H
    if w > aspect * h:
        h = w / aspect
    elif w < aspect * h:
        w = aspect * h
    w *= padding
    h *= padding
    x = max(0.0, cx - w / 2.0)
    y = max(0.0, cy - h / 2.0)
    w = min(img_w - x, w)
    h = min(img_h - y, h)
    return [x, y, w, h]


def _row_to_features(row):
    """Returns dict with `pixel_values`, `target_heatmap`, `target_weight`."""
    img = row["image"].convert("RGB")
    iw, ih = img.size
    bbox = _expand_bbox(row["bbox"], iw, ih)
    proc, _ = _build_model()
    inputs = proc(images=img, boxes=[[bbox]], return_tensors="np")
    pixel_values = inputs["pixel_values"][0]  # (3, 256, 192)

    # Map keypoints into crop space.
    kpts = np.asarray(row["keypoints"], dtype=np.float32).reshape(NUM_KPTS, 3)
    bx, by, bw, bh = bbox
    sx = INPUT_HW[1] / max(bw, 1e-6)
    sy = INPUT_HW[0] / max(bh, 1e-6)
    in_crop = np.zeros((NUM_KPTS, 2), dtype=np.float32)
    in_crop[:, 0] = (kpts[:, 0] - bx) * sx
    in_crop[:, 1] = (kpts[:, 1] - by) * sy
    vis = kpts[:, 2].astype(np.int64)
    target_hm, target_w = _encode_heatmap(in_crop, vis)
    return {
        "pixel_values": pixel_values.astype(np.float32),
        "target_heatmap": target_hm,
        "target_weight": target_w,
    }


# ---------------------------------------------------------------------------
# Loss + Trainer
# ---------------------------------------------------------------------------
class _PoseTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        pixel_values = inputs["pixel_values"]
        target = inputs["target_heatmap"]              # (B, 17, H/4, W/4)
        weight = inputs["target_weight"][:, :, None, None]  # (B, 17, 1, 1)
        out = model(pixel_values=pixel_values)
        pred = out.heatmaps if hasattr(out, "heatmaps") else out.logits
        loss = ((pred - target) ** 2 * weight).mean()
        return (loss, out) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only,
                        ignore_keys=None):
        # ViTPose's output object has no `.loss`, so the default
        # prediction_step would never report eval_loss. Compute it
        # manually here so `metric_for_best_model="loss"` works.
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, out = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return (loss.detach(), None, None)
        preds = out.heatmaps if hasattr(out, "heatmaps") else out.logits
        labels = inputs["target_heatmap"]
        return (loss.detach(), preds.detach(), labels.detach())


def _collator(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "target_heatmap": torch.stack([b["target_heatmap"] for b in batch]),
        "target_weight": torch.stack([b["target_weight"] for b in batch]),
    }


def _main():
    proc, mdl = _build_model()

    use_stream = _args.subset > 0
    print(f"[info] loading {_DATASET_ID} (streaming={use_stream}) ...")
    train_split = _load_split("train", streaming=use_stream)
    val_split = _load_split("val", streaming=use_stream)
    train_cap = _args.subset if _args.subset > 0 else None
    val_cap = min(2000, _args.subset) if _args.subset > 0 else 2000
    train_rows = _flatten_persons(train_split, max_samples=train_cap)
    val_rows = _flatten_persons(val_split, max_samples=val_cap)
    print(f"[info] flattened persons — train={len(train_rows)} val={len(val_rows)}")
    train_ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(val_rows)

    train = train_ds.map(_row_to_features, remove_columns=train_ds.column_names,
                         load_from_cache_file=False)
    val = val_ds.map(_row_to_features, remove_columns=val_ds.column_names,
                     load_from_cache_file=False)
    train.set_format("torch", columns=["pixel_values", "target_heatmap", "target_weight"])
    val.set_format("torch", columns=["pixel_values", "target_heatmap", "target_weight"])

    training_args = TrainingArguments(
        output_dir=str(_RUN_DIR),
        seed=SEED,
        num_train_epochs=_args.epochs,
        per_device_train_batch_size=_args.batch_size,
        per_device_eval_batch_size=_args.batch_size,
        learning_rate=_args.lr,
        weight_decay=0.01,
        warmup_steps=500,
        lr_scheduler_type="cosine",
        bf16=_args.bf16,
        fp16=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        # eval_loss == heatmap MSE on val. Real metric is OKS-AP via offline
        # pycocotools eval (see inference.py); eval_loss is monotonic enough
        # for in-loop best-checkpoint selection.
        metric_for_best_model="loss",
        greater_is_better=False,
        logging_steps=50,
        dataloader_num_workers=_args.num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="none",
    )
    trainer = _PoseTrainer(
        model=mdl,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=_collator,
    )
    trainer.train()
    trainer.save_model(str(_RUN_DIR / "best"))
    proc.save_pretrained(str(_RUN_DIR / "best"))
    print(f"[done] best model + processor saved to {_RUN_DIR / 'best'}")


if __name__ == "__main__":
    _main()

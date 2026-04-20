#!/usr/bin/env python
"""Run inference with the fine-tuned RT-DETRv2 best-checkpoint against CPPE-5.

Samples N images each from the train and validation splits and writes a
GT-vs-prediction side-by-side PNG per sample under:
    notebooks/detr_finetune_reference/inference/{train,val}/

Usage:
    .venv-notebook/bin/python notebooks/detr_finetune_reference/run_inference.py \
        [--n-samples 8] [--threshold 0.4] [--seed 0]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForObjectDetection

_HERE = Path(__file__).resolve().parent
_CKPT = _HERE / "runs" / "rtdetr_v2_r50_cppe5" / "checkpoint-2033"  # best (ep19)
_OUT_ROOT = _HERE / "inference"

# Consistent CPPE-5 class colour palette.
_CLASS_COLOURS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]


def _draw_boxes(ax, image: np.ndarray, boxes_xyxy, labels, scores, id2label, title: str):
    """Draw one panel: image + boxes (optionally with scores) + legend."""
    ax.imshow(image)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for box, label, score in zip(boxes_xyxy, labels, scores if scores is not None else [None] * len(labels)):
        x1, y1, x2, y2 = box
        colour = _CLASS_COLOURS[int(label) % len(_CLASS_COLOURS)]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor=colour, facecolor="none")
        ax.add_patch(rect)
        caption = id2label[int(label)]
        if score is not None:
            caption += f" {score:.2f}"
        ax.text(
            x1, max(0, y1 - 2), caption,
            fontsize=7, color="white",
            bbox={"facecolor": colour, "edgecolor": "none", "pad": 1, "alpha": 0.85},
        )


def _coco_xywh_to_xyxy(boxes):
    out = []
    for x, y, w, h in boxes:
        out.append([x, y, x + w, y + h])
    return np.asarray(out, dtype=np.float32) if out else np.zeros((0, 4), dtype=np.float32)


def _run_one(model, processor, device, sample, id2label, threshold: float):
    image = sample["image"].convert("RGB")
    img_np = np.asarray(image)
    h, w = img_np.shape[:2]

    gt_boxes = _coco_xywh_to_xyxy(sample["objects"]["bbox"])
    gt_labels = np.asarray(sample["objects"]["category"], dtype=np.int64)

    inputs = processor(images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([[h, w]], device=device)
    result = processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]

    pred_boxes = result["boxes"].cpu().numpy() if len(result["boxes"]) else np.zeros((0, 4), dtype=np.float32)
    pred_labels = result["labels"].cpu().numpy() if len(result["labels"]) else np.zeros((0,), dtype=np.int64)
    pred_scores = result["scores"].cpu().numpy() if len(result["scores"]) else np.zeros((0,), dtype=np.float32)

    return img_np, (gt_boxes, gt_labels), (pred_boxes, pred_labels, pred_scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=8, help="samples per split")
    # DETR-family sigmoid scores cap around ~0.2–0.3 (unlike YOLO-style 0.5+),
    # so the conventional 0.4 threshold hides nearly everything. 0.15 matches
    # what the training-time MAPEvaluator (threshold=0.0) treats as "real".
    ap.add_argument("--threshold", type=float, default=0.15, help="confidence threshold for preds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", type=Path, default=_CKPT)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from: {args.ckpt}")
    processor = AutoImageProcessor.from_pretrained(str(args.ckpt), use_fast=True)
    model = AutoModelForObjectDetection.from_pretrained(str(args.ckpt)).to(device).eval()
    id2label = model.config.id2label

    print("Loading CPPE-5 dataset...")
    dataset = load_dataset("cppe-5")
    if "validation" not in dataset:
        split = dataset["train"].train_test_split(0.15, seed=1337)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    rng = np.random.default_rng(args.seed)
    for split_name in ("train", "validation"):
        ds = dataset[split_name]
        out_dir = _OUT_ROOT / ("train" if split_name == "train" else "val")
        out_dir.mkdir(parents=True, exist_ok=True)
        idxs = rng.choice(len(ds), size=min(args.n_samples, len(ds)), replace=False)
        print(f"[{split_name}] sampling {len(idxs)} of {len(ds)} → {out_dir}")

        for k, i in enumerate(idxs):
            sample = ds[int(i)]
            img, (gt_boxes, gt_labels), (pred_boxes, pred_labels, pred_scores) = _run_one(
                model, processor, device, sample, id2label, args.threshold
            )
            fig, (ax_gt, ax_pr) = plt.subplots(1, 2, figsize=(12, 6))
            _draw_boxes(
                ax_gt, img, gt_boxes, gt_labels, scores=None,
                id2label=id2label,
                title=f"GT — {split_name}[{i}] — {len(gt_labels)} boxes",
            )
            _draw_boxes(
                ax_pr, img, pred_boxes, pred_labels, scores=pred_scores,
                id2label=id2label,
                title=f"Pred (thr={args.threshold}) — {len(pred_labels)} boxes",
            )
            fig.suptitle(f"CPPE-5 {split_name}[{i}]  ·  RT-DETRv2-R50 @ ep19 (best)", fontsize=10)
            fig.tight_layout()
            out_path = out_dir / f"{split_name}_{k:02d}_idx{int(i)}.png"
            fig.savefig(out_path, dpi=110, bbox_inches="tight")
            plt.close(fig)
            print(f"  wrote {out_path.name}   gt={len(gt_labels):>2}  pred={len(pred_labels):>2}")

    print(f"\nDone. Outputs under: {_OUT_ROOT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""SegFormer-B0 reference post-training visualization.

Loads the best checkpoint produced by `finetune.py --seed N [--tag T]` and
writes a GT-vs-Pred grid PNG showing model predictions on the val split.
Unlike the RT-DETRv2 port (which reuses `supervision_bridge.annotate_gt_pred`),
SegFormer outputs pixel-dense masks — `supervision`'s box-oriented helpers
don't apply — so this script uses matplotlib directly: for each sample we
draw a 3-column row [image | GT mask overlay | Pred mask overlay].

Usage (from repo root):

    .venv-notebook/bin/python \\
      notebooks/segformer_finetune_reference/reference_segformer_b0/inference.py \\
      --run-dir notebooks/segformer_finetune_reference/reference_segformer_b0/runs/seed42 \\
      --n 16

Outputs:
    <run-dir>/val_predictions_grid.png
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import matplotlib
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch import nn
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = Path(__file__).resolve().parent


def _sidewalk_palette(num_labels: int) -> np.ndarray:
    """Deterministic RGB palette — index 0 = black (unlabeled)."""
    rng = np.random.default_rng(seed=0)
    palette = rng.integers(0, 255, size=(num_labels, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0]  # unlabeled → black
    return palette


def _colorize(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Map (H, W) int label mask → (H, W, 3) uint8 RGB."""
    mask_clipped = np.clip(mask, 0, len(palette) - 1).astype(np.int64)
    return palette[mask_clipped]


def _overlay(image: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend image (H, W, 3 uint8) with colorized mask (same shape)."""
    return (image.astype(np.float32) * (1 - alpha) + mask_rgb.astype(np.float32) * alpha).astype(
        np.uint8
    )


def _resolve_run_dir(run_dir_arg: str | None, seed: int) -> Path:
    if run_dir_arg:
        p = Path(run_dir_arg).resolve()
    else:
        p = _HERE / "runs" / f"seed{seed}"
    if not p.exists():
        raise FileNotFoundError(f"run dir {p} does not exist — did you run finetune.py?")
    return p


def _find_best_dir(run_dir: Path) -> Path:
    # Matches finetune.py's convention: trainer.save_model(_RUN_DIR / "best")
    best = run_dir / "best"
    if best.exists():
        return best
    # Fallback: pick newest checkpoint-*
    checkpoints = sorted(
        run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])
    )
    if not checkpoints:
        raise FileNotFoundError(
            f"No best/ dir or checkpoint-* under {run_dir} — nothing to evaluate."
        )
    return checkpoints[-1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-dir", type=str, default=None,
                   help="path to a finetune.py run dir; defaults to ./runs/seed<SEED>")
    p.add_argument("--seed", type=int, default=42,
                   help="used only when --run-dir not given")
    p.add_argument("--n", type=int, default=16, help="number of samples in the grid")
    p.add_argument("--sample-seed", type=int, default=42,
                   help="RNG for index sampling (reproducible grids)")
    p.add_argument("--dataset", type=str, default="segments/sidewalk-semantic",
                   help="HF dataset id; match finetune.py's choice (gated — needs HF_TOKEN)")
    p.add_argument("--split-seed", type=int, default=1,
                   help="seed used by finetune.py's ds.shuffle() — 1 matches upstream")
    p.add_argument("--test-size", type=float, default=0.2,
                   help="val/test split fraction — matches upstream")
    args = p.parse_args()

    run_dir = _resolve_run_dir(args.run_dir, args.seed)
    best_dir = _find_best_dir(run_dir)
    print(f"Loading best checkpoint from: {best_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_extractor = SegformerImageProcessor.from_pretrained(str(best_dir))
    model = SegformerForSemanticSegmentation.from_pretrained(str(best_dir)).to(device).eval()

    # Reproduce finetune.py's split
    print(f"Loading dataset {args.dataset} (split_seed={args.split_seed}, "
          f"test_size={args.test_size})...")
    ds = load_dataset(args.dataset)
    ds = ds.shuffle(seed=args.split_seed)
    ds = ds["train"].train_test_split(test_size=args.test_size)
    test_ds = ds["test"]

    n = min(args.n, len(test_ds))
    rng = random.Random(args.sample_seed)
    indices = sorted(rng.sample(range(len(test_ds)), n))

    num_labels = model.config.num_labels
    palette = _sidewalk_palette(num_labels)

    ncols = 3  # image | GT | Pred
    nrows = n
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_i, idx in enumerate(indices):
        sample = test_ds[int(idx)]
        image: Image.Image = sample["pixel_values"].convert("RGB")
        gt_mask = np.asarray(sample["label"], dtype=np.int64)
        image_np = np.asarray(image, dtype=np.uint8)

        inputs = feature_extractor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits  # (1, C, H/4, W/4)
        upsampled = nn.functional.interpolate(
            logits,
            size=image.size[::-1],  # (H, W)
            mode="bilinear",
            align_corners=False,
        )
        pred_mask = upsampled.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int64)

        gt_rgb = _colorize(gt_mask, palette)
        pred_rgb = _colorize(pred_mask, palette)

        # Align GT dims to image (datasets sometimes store labels at a
        # different size — resize with nearest for label correctness).
        if gt_rgb.shape[:2] != image_np.shape[:2]:
            gt_rgb = np.asarray(
                Image.fromarray(gt_rgb).resize(image.size, resample=Image.NEAREST),
                dtype=np.uint8,
            )

        gt_overlay = _overlay(image_np, gt_rgb, alpha=0.5)
        pred_overlay = _overlay(image_np, pred_rgb, alpha=0.5)

        axes[row_i, 0].imshow(image_np)
        axes[row_i, 0].set_title(f"image (idx={idx})")
        axes[row_i, 1].imshow(gt_overlay)
        axes[row_i, 1].set_title("GT")
        axes[row_i, 2].imshow(pred_overlay)
        axes[row_i, 2].set_title("Pred")
        for col_i in range(ncols):
            axes[row_i, col_i].axis("off")

    fig.suptitle(f"SegFormer-B0 val predictions — {n} samples", fontsize=14)
    fig.tight_layout()

    out_path = run_dir / "val_predictions_grid.png"
    fig.savefig(str(out_path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

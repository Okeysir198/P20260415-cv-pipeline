#!/usr/bin/env python
"""RT-DETRv2 reference post-training visualization.

Loads the best checkpoint produced by `finetune.py --seed N --tag T` and
writes GT-vs-Pred grid PNGs for the CPPE-5 val + test splits, matching
the visual style our in-repo
`HFValPredictionCallback`
(`core/p06_training/hf_callbacks.py::HFValPredictionCallback`) emits
per-epoch during training. Same `annotate_gt_pred` helper, same
GT-purple / Pred-green conventions, same grid layout — so reference
and in-repo runs produce directly-comparable artefacts for visual diff.

Usage (from repo root):
    .venv-notebook/bin/python \\
      notebooks/detr_finetune_reference/reference_rtdetr_v2/inference.py \\
      --seed 42 --tag bs16_lr1e4_cosine_wd_bf16

Outputs:
    runs/rtdetr_v2_r50_cppe5_seed{SEED}_{TAG}/val_predictions/final.png
    runs/rtdetr_v2_r50_cppe5_seed{SEED}_{TAG}/test_predictions/final.png

Replaces the earlier one-image Flickr-URL demo (the upstream
RT_DETR_v2_inference.ipynb is preserved unchanged in this folder for
reference).
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import supervision as sv
import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.p10_inference.supervision_bridge import annotate_gt_pred  # noqa: E402

_HERE = Path(__file__).resolve().parent  # notebooks/detr_finetune_reference/reference_rtdetr_v2/


def _resolve_paths(seed: int, tag: str) -> tuple[Path, Path]:
    run_dir = _HERE / "runs" / (
        f"rtdetr_v2_r50_cppe5_seed{seed}" + (f"_{tag}" if tag else "")
    )
    best_dir = run_dir / "best"
    if not best_dir.exists():
        raise FileNotFoundError(
            f"No best checkpoint under {best_dir} — run finetune.py first with "
            f"the same --seed/--tag (produces this directory via "
            f"`trainer.save_model(_BEST_DIR)` at the end of training)."
        )
    return run_dir, best_dir


def _load_cppe5_splits(split_seed: int = 1337, split_ratio: float = 0.15):
    """Reproduce qubvel's CPPE-5 train/val/test split used by finetune.py."""
    ds = load_dataset("cppe-5")
    if "validation" not in ds:
        split = ds["train"].train_test_split(split_ratio, seed=split_seed)
        ds["train"] = split["train"]
        ds["validation"] = split["test"]
    return ds


def _pil_to_bgr(pil_img) -> np.ndarray:
    arr = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _render_split_grid(
    samples_ds,
    model,
    image_processor,
    device: str,
    split_name: str,
    class_names: dict,
    conf_threshold: float,
    num_samples: int,
    sample_seed: int,
    out_path: Path,
    grid_cols: int = 2,
    gt_thickness: int = 2,
    pred_thickness: int = 1,
    text_scale: float = 0.4,
    dpi: int = 150,
) -> None:
    n = len(samples_ds)
    k = min(num_samples, n)
    if k == 0:
        print(f"  [{split_name}] split is empty — skipping.")
        return

    rng = random.Random(sample_seed)
    indices = sorted(rng.sample(range(n), k))

    rows: list[np.ndarray] = []
    for idx in indices:
        row = samples_ds[int(idx)]
        # Normalise to RGB PIL up-front. HF's `AutoImageProcessor` chokes on
        # palette-mode PNGs ("Unable to infer channel dimension format"),
        # which CPPE-5 has a handful of.
        pil_img = row["image"].convert("RGB")
        image_bgr = _pil_to_bgr(pil_img)
        orig_h, orig_w = image_bgr.shape[:2]

        # GT: CPPE-5 stores bbox as COCO pixel [x, y, w, h] relative to actual image dims.
        gt_xyxy = None
        gt_class_ids = None
        boxes_coco = row["objects"]["bbox"]
        if boxes_coco:
            gt_arr = np.asarray(boxes_coco, dtype=np.float32).reshape(-1, 4)
            gt_xyxy = np.stack([
                gt_arr[:, 0],
                gt_arr[:, 1],
                gt_arr[:, 0] + gt_arr[:, 2],
                gt_arr[:, 1] + gt_arr[:, 3],
            ], axis=1)
            gt_class_ids = np.asarray(row["objects"]["category"], dtype=np.int64)

        # Predictions: image_processor resizes to 480×480 + normalizes;
        # post_process_object_detection(target_sizes=[(H, W)]) scales back.
        inputs = image_processor(images=pil_img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        post = image_processor.post_process_object_detection(
            outputs,
            threshold=conf_threshold,
            target_sizes=[(orig_h, orig_w)],
        )[0]

        pred_boxes = post["boxes"].detach().cpu().numpy().astype(np.float64).reshape(-1, 4)
        pred_labels = post["labels"].detach().cpu().numpy().astype(np.int64).ravel()
        pred_scores = post["scores"].detach().cpu().numpy().astype(np.float64).ravel()
        pred_dets = sv.Detections(
            xyxy=pred_boxes, class_id=pred_labels, confidence=pred_scores,
        )

        rows.append(annotate_gt_pred(
            image_bgr, gt_xyxy, gt_class_ids, pred_dets, class_names,
            gt_thickness=gt_thickness, pred_thickness=pred_thickness,
            text_scale=text_scale, draw_legend=True,
        ))

    # Grid — same layout as HFValPredictionCallback (2-col default, figsize=6×5 per cell).
    ncols = grid_cols
    nrows = math.ceil(len(rows) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5))
    axes = np.asarray(axes).ravel()
    for i in range(nrows * ncols):
        axes[i].axis("off")
        if i < len(rows):
            axes[i].imshow(cv2.cvtColor(rows[i], cv2.COLOR_BGR2RGB))
    fig.suptitle(
        f"{split_name} set — {len(rows)} samples (conf ≥ {conf_threshold})",
        fontsize=14,
    )
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{split_name}] wrote {out_path} ({len(rows)} images)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=42,
                   help="Training seed used by finetune.py (selects the run dir).")
    p.add_argument("--tag", type=str, default="bs16_lr1e4_cosine_wd_bf16",
                   help="Run tag used by finetune.py (selects the run dir).")
    p.add_argument("--conf", type=float, default=0.15,
                   help=("Confidence threshold for drawn predictions. "
                         "DETR sigmoid scores cap around ~0.2 — 0.15 matches our "
                         "in-repo val_predictions callback default."))
    p.add_argument("--num-samples", type=int, default=12,
                   help="Images per split grid.")
    p.add_argument("--splits", nargs="+", default=["val", "test"],
                   help="Which splits to render.")
    p.add_argument("--sample-seed", type=int, default=42,
                   help="Seed for the sampled-indices RNG (reproducible grids).")
    args = p.parse_args()

    run_dir, best_dir = _resolve_paths(args.seed, args.tag)
    print(f"Loading best checkpoint from: {best_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_processor = AutoImageProcessor.from_pretrained(str(best_dir), use_fast=True)
    model = AutoModelForObjectDetection.from_pretrained(str(best_dir)).to(device).eval()
    id2label = model.config.id2label  # {int: str}
    class_names = {int(k): str(v) for k, v in id2label.items()}

    print(f"Loading CPPE-5 dataset (split seed=1337, matches finetune.py)...")
    ds = _load_cppe5_splits()
    split_map = {"train": "train", "val": "validation", "test": "test"}

    for split_name in args.splits:
        hf_key = split_map.get(split_name, split_name)
        if hf_key not in ds:
            print(f"  [{split_name}] not in dataset — skipping.")
            continue
        out_path = run_dir / f"{split_name}_predictions" / "final.png"
        _render_split_grid(
            samples_ds=ds[hf_key],
            model=model, image_processor=image_processor, device=device,
            split_name=split_name, class_names=class_names,
            conf_threshold=args.conf, num_samples=args.num_samples,
            sample_seed=args.sample_seed, out_path=out_path,
        )

    print(f"\nDone. Grids under: {run_dir}")


if __name__ == "__main__":
    main()

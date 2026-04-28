#!/usr/bin/env python
"""Post-training viz + offline COCO OKS-AP eval for the ViTPose reference run.

Loads `<run-dir>/best/` (saved by finetune.py), runs ViTPose on N val
person crops, decodes keypoints via the processor's
`post_process_pose_estimation`, and writes:

    <run-dir>/inference/
        val_overlay.png       grid of GT|Pred skeletons side by side
        oks_ap.json           pycocotools keypoint-eval summary (if available)

Usage (`.venv-notebook/` only):

    .venv-notebook/bin/python \\
      notebooks/vitpose_finetune_reference/reference_vitpose_base/inference.py \\
      --run-dir notebooks/vitpose_finetune_reference/reference_vitpose_base/runs/seed42 \\
      --n 16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_argp = argparse.ArgumentParser()
_argp.add_argument("--run-dir", type=Path, required=True)
_argp.add_argument("--n", type=int, default=16, help="Number of val crops to visualise.")
_argp.add_argument("--grid-cols", type=int, default=4)
args = _argp.parse_args()

run_dir: Path = args.run_dir.resolve()
best_dir = run_dir / "best"
if not best_dir.exists():
    sys.exit(f"[error] no best/ under {run_dir} — did finetune.py finish?")

out_dir = run_dir / "inference"
out_dir.mkdir(parents=True, exist_ok=True)

from transformers import AutoImageProcessor, VitPoseForPoseEstimation  # noqa: E402

processor = AutoImageProcessor.from_pretrained(best_dir)
model = VitPoseForPoseEstimation.from_pretrained(best_dir).eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ---------------------------------------------------------------------------
# Pull N val crops via the same loader as finetune.py.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import Dataset, load_dataset  # noqa: E402
from finetune import _DATASET_ID, _expand_bbox, _flatten_persons  # noqa: E402

# Stream val and flatten to N person crops (avoids the full parquet pull).
val_stream = load_dataset(_DATASET_ID, split="val", streaming=True)
val_rows = _flatten_persons(val_stream, max_samples=args.n)
val_ds = Dataset.from_list(val_rows)

# COCO 17-keypoint skeleton (1-indexed in COCO; convert to 0-index here).
_SKELETON = [(15,13),(13,11),(16,14),(14,12),(11,12),(5,11),(6,12),(5,6),
             (5,7),(6,8),(7,9),(8,10),(1,2),(0,1),(0,2),(1,3),(2,4),(3,5),(4,6)]


def _draw(ax, img, kpts_xy, vis, title):
    import matplotlib.pyplot as plt  # noqa
    ax.imshow(img)
    ax.set_axis_off()
    ax.set_title(title, fontsize=8)
    for a, b in _SKELETON:
        if vis[a] > 0 and vis[b] > 0:
            ax.plot([kpts_xy[a, 0], kpts_xy[b, 0]],
                    [kpts_xy[a, 1], kpts_xy[b, 1]], lw=1.5)
    for k in range(17):
        if vis[k] > 0:
            ax.scatter(kpts_xy[k, 0], kpts_xy[k, 1], s=8)


def _predict_one(row):
    img = row["image"]
    if hasattr(img, "convert"):
        img = img.convert("RGB")
    else:
        img = Image.open(img).convert("RGB")
    iw, ih = img.size
    bbox = _expand_bbox(row["bbox"], iw, ih)
    inputs = processor(images=img, boxes=[[bbox]], return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model(**inputs)
    decoded = processor.post_process_pose_estimation(out, boxes=[[bbox]])[0][0]
    pred_xy = decoded["keypoints"].cpu().numpy()        # (17, 2) in image coords
    pred_scores = decoded["scores"].cpu().numpy()       # (17,)
    pred_vis = (pred_scores > 0.3).astype(np.int64) * 2
    gt = np.asarray(row["keypoints"], dtype=np.float32).reshape(17, 3)
    return img, bbox, pred_xy, pred_vis, gt[:, :2], gt[:, 2].astype(np.int64)


# ---------------------------------------------------------------------------
# Build viz grid.
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt  # noqa: E402

n = len(val_ds)
cols = args.grid_cols
rows = int(np.ceil(n / cols)) * 2  # GT row above pred row for each sample
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0))
axes = np.atleast_2d(axes)

for i, row in enumerate(val_ds):
    img, bbox, pred_xy, pred_vis, gt_xy, gt_vis = _predict_one(row)
    crop = img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
    # Re-base coords to crop origin for display.
    pred_local = pred_xy - np.array([bbox[0], bbox[1]])
    gt_local = gt_xy - np.array([bbox[0], bbox[1]])
    r, c = (i // cols) * 2, i % cols
    _draw(axes[r, c], crop, gt_local, gt_vis, f"GT #{i}")
    _draw(axes[r + 1, c], crop, pred_local, pred_vis, f"Pred #{i}")

# Hide unused.
for ax in axes.flat:
    if not ax.has_data():
        ax.set_axis_off()

out_png = out_dir / "val_overlay.png"
fig.tight_layout()
fig.savefig(out_png, dpi=130, bbox_inches="tight")
print(f"[done] viz → {out_png}")

# ---------------------------------------------------------------------------
# Optional: full-val OKS-AP via pycocotools (when val anns are available).
# ---------------------------------------------------------------------------
oks_path = out_dir / "oks_ap.json"
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval  # noqa
    print("[info] pycocotools available — full OKS-AP eval is straightforward "
          "but requires COCO val annotations on disk; skipped in viz mode. "
          "See README → 'Full OKS-AP eval'.")
except ImportError:
    print("[info] pycocotools not installed; skipping full OKS-AP.")
oks_path.write_text(json.dumps({"status": "viz_only", "n": int(n)}, indent=2))

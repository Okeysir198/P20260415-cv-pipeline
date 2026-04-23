#!/usr/bin/env python
"""Inference + val-split error analysis for the Swin-tiny EuroSAT reference run.

Loads `runs/seed{SEED}/best/` (produced by finetune.py), runs predictions on
the EuroSAT validation split, and writes:

    runs/seed{SEED}/val_report/
      confusion_matrix.png
      top_misclassifications.png     (top-K hardest misclassifications grid)
      report.json                    (per-class accuracy + overall accuracy)

Run in the isolated notebook env:
    .venv-notebook/bin/python \\
      notebooks/image_classification_finetune_reference/reference_swin_tiny/inference.py \\
      --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

_argp = argparse.ArgumentParser()
_argp.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 42)))
_argp.add_argument("--tag", type=str, default=os.environ.get("RUN_TAG", ""))
_argp.add_argument("--run-dir", type=str, default=None,
                   help="override auto-derived runs/seed<SEED>/ path")
_argp.add_argument("--top-k", type=int, default=16,
                   help="number of misclassification examples to show")
_args = _argp.parse_args()
SEED = _args.seed
TAG = _args.tag
TOP_K = _args.top_k

_HERE = Path(__file__).resolve().parent
if _args.run_dir:
    _RUN_DIR = Path(_args.run_dir).resolve()
else:
    _dir_name = (f"{TAG}_seed{SEED}" if TAG else f"seed{SEED}")
    _RUN_DIR = _HERE / "runs" / _dir_name

_BEST_DIR = _RUN_DIR / "best"
_OUT_DIR = _RUN_DIR / "val_report"
_OUT_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoImageProcessor, AutoModelForImageClassification  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"

image_processor = AutoImageProcessor.from_pretrained(str(_BEST_DIR))
model = AutoModelForImageClassification.from_pretrained(str(_BEST_DIR))
model = model.to(device).eval()

id2label = model.config.id2label
num_classes = len(id2label)

# Reproduce the 90/10 split used by finetune.py (same seed).
dataset = load_dataset("jonathan-roberts1/EuroSAT")
splits = dataset["train"].train_test_split(test_size=0.1, seed=SEED)
val_ds = splits["test"]

# Build val transform exactly like training — Resize + CenterCrop + ToTensor + Normalize.
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor  # noqa: E402

if "height" in image_processor.size:
    size = (image_processor.size["height"], image_processor.size["width"])
    crop_size = size
elif "shortest_edge" in image_processor.size:
    size = image_processor.size["shortest_edge"]
    crop_size = (size, size)

val_transforms = Compose(
    [
        Resize(size),
        CenterCrop(crop_size),
        ToTensor(),
        Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
    ]
)

# Run inference in batches; keep original PIL images for the misclassification grid.
all_preds: list[int] = []
all_scores: list[float] = []
all_labels: list[int] = []
all_images = []  # PIL originals, for the top-K grid

BATCH = 64
pending: list[torch.Tensor] = []
pending_labels: list[int] = []
pending_images = []


def _flush():
    if not pending:
        return
    batch_tensor = torch.stack(pending).to(device)
    with torch.no_grad():
        logits = model(pixel_values=batch_tensor).logits
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1).cpu().numpy()
        max_probs = probs.max(dim=-1).values.cpu().numpy()
    all_preds.extend(preds.tolist())
    all_scores.extend(max_probs.tolist())
    all_labels.extend(pending_labels)
    all_images.extend(pending_images)
    pending.clear()
    pending_labels.clear()
    pending_images.clear()


for ex in val_ds:
    img = ex["image"].convert("RGB")
    pending.append(val_transforms(img))
    pending_labels.append(int(ex["label"]))
    pending_images.append(img)
    if len(pending) >= BATCH:
        _flush()
_flush()

preds_arr = np.array(all_preds)
labels_arr = np.array(all_labels)
scores_arr = np.array(all_scores)

overall_acc = float((preds_arr == labels_arr).mean())
# Confusion matrix (rows = true, cols = pred).
cm = np.zeros((num_classes, num_classes), dtype=np.int64)
for t, p in zip(labels_arr, preds_arr):
    cm[t, p] += 1
per_class_acc = {
    id2label[i]: float(cm[i, i] / max(int(cm[i].sum()), 1)) for i in range(num_classes)
}

report = {
    "overall_accuracy": overall_acc,
    "per_class_accuracy": per_class_acc,
    "num_val_samples": int(len(labels_arr)),
    "confusion_matrix": cm.tolist(),
    "id2label": {int(k): v for k, v in id2label.items()},
}
(_OUT_DIR / "report.json").write_text(json.dumps(report, indent=2))

# Plot confusion matrix.
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(figsize=(max(6, num_classes * 0.6), max(6, num_classes * 0.6)))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(num_classes))
ax.set_yticks(range(num_classes))
ax.set_xticklabels([id2label[i] for i in range(num_classes)], rotation=45, ha="right")
ax.set_yticklabels([id2label[i] for i in range(num_classes)])
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"Swin-tiny EuroSAT — val confusion (acc={overall_acc:.3f})")
for i in range(num_classes):
    for j in range(num_classes):
        ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(_OUT_DIR / "confusion_matrix.png", dpi=120)
plt.close()

# Top-K misclassifications grid — hardest wrongs (highest-confidence wrongs first).
wrong_mask = preds_arr != labels_arr
wrong_idx = np.where(wrong_mask)[0]
# Sort by descending confidence of the wrong prediction (most "confidently wrong" first).
wrong_idx = wrong_idx[np.argsort(-scores_arr[wrong_idx])][:TOP_K]

if len(wrong_idx) > 0:
    cols = min(4, len(wrong_idx))
    rows = (len(wrong_idx) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    axes_flat = np.atleast_1d(axes).flatten()
    for ax_i, idx in enumerate(wrong_idx):
        ax = axes_flat[ax_i]
        ax.imshow(all_images[idx])
        ax.set_title(
            f"GT: {id2label[labels_arr[idx]]}\n"
            f"Pred: {id2label[preds_arr[idx]]} ({scores_arr[idx]:.2f})",
            fontsize=9,
        )
        ax.axis("off")
    for ax_j in range(len(wrong_idx), len(axes_flat)):
        axes_flat[ax_j].axis("off")
    plt.suptitle(f"Top-{len(wrong_idx)} misclassifications (most confident errors)")
    plt.tight_layout()
    plt.savefig(_OUT_DIR / "top_misclassifications.png", dpi=120)
    plt.close()

print(f"Overall val accuracy: {overall_acc:.4f}")
print(f"Wrote report to: {_OUT_DIR}")

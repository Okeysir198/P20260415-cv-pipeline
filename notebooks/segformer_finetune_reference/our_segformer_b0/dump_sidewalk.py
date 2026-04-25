"""Dump segments/sidewalk-semantic (gated HF dataset) into the images/masks
layout that our SegmentationDataset expects.

Output:
    dataset_store/training_ready/sidewalk_semantic/
        train/{images,masks}/
        val/{images,masks}/

Uses the same seed=1 shuffle + 80/20 split as the upstream cookbook
(and reference_segformer_b0/finetune.py), so val indices match across
both sides.

Run from repo root (needs HF_TOKEN in .env):
    set -a; source .env; set +a
    .venv-notebook/bin/python notebooks/segformer_finetune_reference/our_segformer_b0/dump_sidewalk.py
"""
from __future__ import annotations
from pathlib import Path

from datasets import load_dataset

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "dataset_store" / "training_ready" / "sidewalk_semantic"

ds = load_dataset("segments/sidewalk-semantic")
ds = ds.shuffle(seed=1)["train"].train_test_split(test_size=0.2)

for split_src, split_dst in [("train", "train"), ("test", "val")]:
    img_dir = OUT / split_dst / "images"
    msk_dir = OUT / split_dst / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)
    for i, ex in enumerate(ds[split_src]):
        ex["pixel_values"].convert("RGB").save(img_dir / f"{i:05d}.jpg", quality=95)
        ex["label"].save(msk_dir / f"{i:05d}.png")
    print(f"{split_dst}: {i + 1} images -> {img_dir.parent}")

print(f"Done -> {OUT}")

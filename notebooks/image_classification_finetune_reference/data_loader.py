"""Bridge: HuggingFace EuroSAT dataset → on-disk ImageFolder layout.

Materialises `jonathan-roberts1/EuroSAT` to
``dataset_store/training_ready/eurosat/{train,val}/<class>/*.jpg`` so the
reference and `our_*` training scripts can consume it via
``torchvision.datasets.ImageFolder`` or ``datasets.load_dataset("imagefolder", ...)``.

EuroSAT ships as a single split of 27,000 RGB JPEGs (64×64) across 10 classes:
AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture,
PermanentCrop, Residential, River, SeaLake. When only one split is present we
do an 80/20 stratified train/val split (per-class) with a fixed seed.

Upstream reference notebook:
    https://github.com/huggingface/notebooks/blob/main/examples/image_classification.ipynb

Layout written:

    <output_root>/eurosat/
        id2label.json                 {"0": "AnnualCrop", ...}
        train/
            AnnualCrop/  <stem>.jpg
            Forest/      <stem>.jpg
            ...
        val/
            AnnualCrop/  <stem>.jpg
            ...

CLI:
    .venv-notebook/bin/python data_loader.py --dump-eurosat
    .venv-notebook/bin/python data_loader.py --dump-eurosat --limit 500
    .venv-notebook/bin/python data_loader.py --dump-eurosat --val-split 0.2 --seed 42 --force
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_ROOT = _REPO_ROOT / "dataset_store" / "training_ready"
_HF_DATASET_ID = "jonathan-roberts1/EuroSAT"
_DATASET_DIR_NAME = "eurosat"


def _resolve_class_names(ds) -> List[str]:
    """Return ordered class names from a HF dataset's ``label`` ClassLabel.

    Falls back to a sorted-by-id list of stringified labels if the feature
    isn't a ClassLabel (shouldn't happen for EuroSAT but defend anyway).
    """
    feat = ds.features.get("label")
    names = getattr(feat, "names", None)
    if names:
        return list(names)
    # Fallback: derive from unique label ints.
    unique = sorted({int(x) for x in ds["label"]})
    return [f"class_{i}" for i in unique]


def _stratified_split(
    indices_per_class: Dict[int, List[int]],
    val_split: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """Return ``(train_idx, val_idx)`` lists stratified by class.

    Per-class: shuffle deterministically with ``seed``, take ``round(n * val_split)``
    to val (at least 1 if the class has ≥2 samples), rest to train.
    """
    rng = random.Random(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for cls in sorted(indices_per_class.keys()):
        idxs = list(indices_per_class[cls])
        rng.shuffle(idxs)
        n = len(idxs)
        n_val = int(round(n * val_split))
        if n >= 2:
            n_val = max(1, min(n - 1, n_val))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    return train_idx, val_idx


def _count_existing_images(split_dir: Path) -> int:
    if not split_dir.is_dir():
        return 0
    return sum(1 for p in split_dir.rglob("*.jpg"))


def dump_eurosat_to_imagefolder(
    output_root: Optional[Path] = None,
    val_split: float = 0.2,
    seed: int = 42,
    limit: Optional[int] = None,
    force: bool = False,
) -> Dict[str, int]:
    """Download EuroSAT and write ImageFolder layout.

    Args:
        output_root: directory that will hold ``eurosat/``. Defaults to
            ``<repo>/dataset_store/training_ready``.
        val_split: fraction of each class that goes to ``val/``. Default 0.2.
        seed: RNG seed for the stratified shuffle. Default 42.
        limit: if set, take the first ``limit`` rows of the HF dataset before
            splitting (useful for smoke tests).
        force: re-dump even if the target appears populated.

    Returns:
        ``{"train": N_train, "val": N_val}`` image counts.
    """
    from datasets import load_dataset

    root = Path(output_root) if output_root else _DEFAULT_DATA_ROOT
    base = root / _DATASET_DIR_NAME
    train_dir = base / "train"
    val_dir = base / "val"

    ds_all = load_dataset(_HF_DATASET_ID)
    # EuroSAT ships with a single "train" split; fall back to the first key.
    split_key = "train" if "train" in ds_all else next(iter(ds_all.keys()))
    ds = ds_all[split_key]
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    class_names = _resolve_class_names(ds)
    id2label = {str(i): name for i, name in enumerate(class_names)}

    # Idempotent early-exit — only when not limited/forced.
    if not force and limit is None:
        n_train = _count_existing_images(train_dir)
        n_val = _count_existing_images(val_dir)
        if n_train + n_val == len(ds) and n_train > 0 and n_val > 0:
            # Ensure id2label is on disk.
            base.mkdir(parents=True, exist_ok=True)
            (base / "id2label.json").write_text(json.dumps(id2label, indent=2))
            return {"train": n_train, "val": n_val}

    # Stratified split.
    by_class: Dict[int, List[int]] = defaultdict(list)
    for i, row_label in enumerate(ds["label"]):
        by_class[int(row_label)].append(i)
    train_idx, val_idx = _stratified_split(by_class, val_split=val_split, seed=seed)

    # Prepare class subdirs.
    base.mkdir(parents=True, exist_ok=True)
    for split_dir in (train_dir, val_dir):
        for name in class_names:
            (split_dir / name).mkdir(parents=True, exist_ok=True)

    (base / "id2label.json").write_text(json.dumps(id2label, indent=2))

    def _dump(indices: List[int], split_dir: Path) -> int:
        n = 0
        for i in indices:
            row = ds[int(i)]
            img = row["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            cls_name = class_names[int(row["label"])]
            stem = f"{i:06d}"
            img.save(split_dir / cls_name / f"{stem}.jpg", "JPEG", quality=95)
            n += 1
        return n

    counts = {
        "train": _dump(train_idx, train_dir),
        "val": _dump(val_idx, val_dir),
    }
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description="EuroSAT → ImageFolder dump utility.")
    p.add_argument(
        "--dump-eurosat",
        action="store_true",
        help=f"Download {_HF_DATASET_ID} and write "
             f"dataset_store/training_ready/{_DATASET_DIR_NAME}/ as ImageFolder.",
    )
    p.add_argument("--limit", type=int, default=None,
                   help="Cap number of source rows before splitting (smoke test).")
    p.add_argument("--val-split", type=float, default=0.2,
                   help="Fraction of each class routed to val/. Default 0.2.")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for the stratified split. Default 42.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite even if target already has the expected file count.")
    p.add_argument("--output-root", type=Path, default=None,
                   help="Override parent of eurosat/. Defaults to dataset_store/training_ready.")
    args = p.parse_args()

    if args.dump_eurosat:
        counts = dump_eurosat_to_imagefolder(
            output_root=args.output_root,
            val_split=args.val_split,
            seed=args.seed,
            limit=args.limit,
            force=args.force,
        )
        dest = (args.output_root or _DEFAULT_DATA_ROOT) / _DATASET_DIR_NAME
        print(f"\nDumped EuroSAT → {dest}")
        for split, n in counts.items():
            print(f"  {split}: {n} images")
    else:
        p.print_help()


if __name__ == "__main__":
    main()

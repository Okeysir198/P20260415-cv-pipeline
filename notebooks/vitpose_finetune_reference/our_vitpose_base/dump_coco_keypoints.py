"""Dump rom1x38/COCO_keypoints (HF parquet mirror of COCO 2017 person
keypoints) into the YOLO-pose `images/labels` layout that our
`KeypointTopDownDataset` expects.

Output:
    dataset_store/training_ready/coco_keypoints/
        train/images/<image_id>.jpg
        train/labels/<image_id>.txt
        val/images/<image_id>.jpg
        val/labels/<image_id>.txt

Each `.txt` line:
    class_id cx cy w h kx1 ky1 v1 ... kx17 ky17 v17    (all normalised 0–1)

`class_id` is always 0 (COCO-keypoints is single-class — `person`).

Run from repo root (HF token only needed if rate-limited):
    set -a; source .env; set +a
    .venv-notebook/bin/python \\
      notebooks/vitpose_finetune_reference/our_vitpose_base/dump_coco_keypoints.py \\
      --max-train 5000 --max-val 1000

Pass `--max-train 0 --max-val 0` for the full set (149k+ persons,
~19 GB train images). Use `--max-train N --max-val M` for a smoke
dump that completes in minutes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset

_REPO = Path(__file__).resolve().parents[3]
_OUT = _REPO / "dataset_store" / "training_ready" / "coco_keypoints"
_DATASET_ID = "rom1x38/COCO_keypoints"
_NUM_KPTS = 17


def _ensure_dirs(split: str) -> tuple[Path, Path]:
    img_dir = _OUT / split / "images"
    lbl_dir = _OUT / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    return img_dir, lbl_dir


def _yolo_pose_line(bbox_xywh, kpts_xy_v, img_w: int, img_h: int) -> str | None:
    """Convert COCO-style bbox + keypoints to a normalised YOLO-pose row.

    Returns None if the bbox is invalid (zero size).
    """
    x, y, w, h = bbox_xywh
    if w <= 0 or h <= 0:
        return None
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))
    parts = [f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"]
    arr = np.asarray(kpts_xy_v, dtype=np.float32).reshape(_NUM_KPTS, 3)
    for kx, ky, v in arr:
        nkx = max(0.0, min(1.0, kx / img_w))
        nky = max(0.0, min(1.0, ky / img_h))
        parts.append(f"{nkx:.6f} {nky:.6f} {int(v)}")
    return " ".join(parts)


def dump_split(split: str, max_persons: int | None) -> dict:
    """Stream a split, write one image + label file per source image.

    A single .txt may contain multiple rows (one per labeled person).
    Returns a stats dict for logging.
    """
    name = "train" if split == "train" else "val"
    print(f"[info] streaming {_DATASET_ID} split={name} max_persons={max_persons}")
    ds = load_dataset(_DATASET_ID, split=name, streaming=True)

    img_dir, lbl_dir = _ensure_dirs(split)
    n_images = 0
    n_persons = 0
    skipped_empty = 0

    for row in ds:
        if max_persons is not None and n_persons >= max_persons:
            break

        bboxes = row.get("bboxes") or []
        kpts_per_person = row.get("keypoints") or []
        if not bboxes or not kpts_per_person:
            skipped_empty += 1
            continue

        # Determine which persons in this image have any visible keypoints.
        good_rows: list[str] = []
        img = row["image"].convert("RGB")
        iw, ih = img.size
        for b, k in zip(bboxes, kpts_per_person):
            arr = np.asarray(k, dtype=np.float32).reshape(_NUM_KPTS, 3)
            if (arr[:, 2] > 0).sum() == 0:
                continue
            line = _yolo_pose_line(b, k, iw, ih)
            if line is not None:
                good_rows.append(line)

        if not good_rows:
            skipped_empty += 1
            continue

        image_id = row.get("image_id", n_images)
        stem = f"{int(image_id):012d}"
        img_path = img_dir / f"{stem}.jpg"
        lbl_path = lbl_dir / f"{stem}.txt"

        if not img_path.exists():
            img.save(img_path, format="JPEG", quality=92)
        lbl_path.write_text("\n".join(good_rows) + "\n")

        n_images += 1
        n_persons += len(good_rows)

    print(
        f"[done] {split}: wrote {n_images} images / {n_persons} person rows"
        f" — skipped {skipped_empty} (no labeled person)"
    )
    return {"images": n_images, "persons": n_persons, "skipped": skipped_empty}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--max-train", type=int, default=5000,
                    help="Cap on labeled-person rows in train split (0 = full).")
    ap.add_argument("--max-val", type=int, default=1000,
                    help="Cap on labeled-person rows in val split (0 = full).")
    args = ap.parse_args()

    _OUT.mkdir(parents=True, exist_ok=True)
    stats = {
        "train": dump_split(
            "train", max_persons=args.max_train if args.max_train > 0 else None,
        ),
        "val": dump_split(
            "val", max_persons=args.max_val if args.max_val > 0 else None,
        ),
        "source": _DATASET_ID,
    }
    (_OUT / "_dump_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[done] stats → {_OUT / '_dump_stats.json'}")


if __name__ == "__main__":
    main()

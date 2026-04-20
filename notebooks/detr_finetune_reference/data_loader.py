"""Bridge: dataset_store/training_ready/<name>/ → HuggingFace Dataset.

Converts our YOLO-format datasets into the exact column layout that the
`qubvel/RT_DETR_v2_finetune_on_a_custom_dataset.ipynb` (and HF cookbook's DETR
notebook) expect, so the notebooks can run verbatim on any of our features.

Expected on-disk layout:
    dataset_store/training_ready/<dataset_name>/
        train/{images,labels}/
        val/{images,labels}/
        test/{images,labels}/

YOLO label format: one line per box, "<cls> <cx_norm> <cy_norm> <w_norm> <h_norm>"
with coords normalized to [0, 1] relative to image dimensions.

Output HF Dataset columns (matches CPPE-5 convention — the format used by
qubvel's RT-DETRv2 notebook and the HF cookbook):
    image_id:  int
    image:     PIL.Image
    width:     int
    height:    int
    objects:   dict{
        id:       list[int]      (unique per-box id within dataset)
        area:     list[float]
        bbox:     list[list[float]]   # COCO pixel [x_top_left, y_top_left, width, height]
        category: list[int]
    }

Note on format: qubvel's notebook uses ``A.BboxParams(format="coco", ...)``
which expects boxes as ``[x, y, w, h]`` in pixel space. HF cookbook uses
PASCAL VOC internally but converts. Output COCO so the notebooks run verbatim.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from PIL import Image
from datasets import ClassLabel, Dataset, Features, Image as HFImage, Sequence, Value

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_ROOT = _REPO_ROOT / "dataset_store" / "training_ready"


def _iter_split(split_dir: Path):
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Missing {img_dir}")
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in _IMG_EXTS:
            continue
        yield img_path, lbl_dir / f"{img_path.stem}.txt"


def _parse_yolo_label(label_path: Path, img_w: int, img_h: int):
    """Parse a YOLO label file → lists of (category, bbox_coco_pixel, area).

    Output boxes are COCO format ``[x_top_left, y_top_left, width, height]`` in
    pixel coords, matching ``A.BboxParams(format="coco", ...)``.
    """
    categories, bboxes, areas = [], [], []
    if not label_path.exists():
        return categories, bboxes, areas
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(p) for p in parts[1:])
        except ValueError:
            continue
        if w <= 0 or h <= 0:
            continue
        xmin = max(0.0, (cx - w / 2) * img_w)
        ymin = max(0.0, (cy - h / 2) * img_h)
        xmax = min(float(img_w), (cx + w / 2) * img_w)
        ymax = min(float(img_h), (cy + h / 2) * img_h)
        bw, bh = xmax - xmin, ymax - ymin
        if bw <= 0 or bh <= 0:
            continue
        categories.append(cls)
        bboxes.append([xmin, ymin, bw, bh])   # COCO: [x, y, w, h]
        areas.append(bw * bh)
    return categories, bboxes, areas


def load_yolo_as_hf_dataset(
    dataset_name: str,
    split: str = "train",
    subset: Optional[float] = None,
    seed: int = 42,
    data_root: Optional[Path] = None,
    class_names: Optional[Dict[int, str]] = None,
) -> Dataset:
    """Load a YOLO-format split as an HF `Dataset`.

    Args:
        dataset_name: folder name under ``dataset_store/training_ready/``
            (e.g. ``"fire_detection"``).
        split: one of ``"train" | "val" | "test"``.
        subset: fraction in ``(0, 1]`` or ``None`` for full split.
            Shuffled with ``seed`` before slicing.
        seed: RNG seed for subset shuffle.
        data_root: override for testing; defaults to
            ``<repo>/dataset_store/training_ready``.
        class_names: ``{int: str}`` — if provided, the ``category`` feature is
            typed as ``ClassLabel`` so the notebook can read names via
            ``dataset.features["objects"].feature["category"].names``. Defaults
            to ``load_class_names(dataset_name)``.

    Returns:
        HuggingFace ``Dataset`` with COCO-pixel boxes ``[x, y, w, h]``.
    """
    root = Path(data_root) if data_root else _DEFAULT_DATA_ROOT
    split_dir = root / dataset_name / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    if class_names is None:
        class_names = load_class_names(dataset_name)

    rows = {"image_id": [], "image": [], "width": [], "height": [], "objects": []}
    bbox_counter = 0
    for i, (img_path, lbl_path) in enumerate(_iter_split(split_dir)):
        with Image.open(img_path) as im:
            w, h = im.size
        cats, boxes, areas = _parse_yolo_label(lbl_path, w, h)
        ids = list(range(bbox_counter, bbox_counter + len(cats)))
        bbox_counter += len(cats)
        rows["image_id"].append(i)
        rows["image"].append(str(img_path))
        rows["width"].append(w)
        rows["height"].append(h)
        rows["objects"].append({
            "id": ids,
            "area": areas,
            "bbox": boxes,
            "category": cats,
        })

    # Prefer ClassLabel so notebook can do `.feature["category"].names`.
    if class_names:
        max_id = max(class_names.keys())
        ordered = [class_names.get(i, f"class_{i}") for i in range(max_id + 1)]
        category_feature: object = ClassLabel(names=ordered)
    else:
        category_feature = Value("int64")

    # `objects` as Sequence of a dict-of-scalar-Values — HF represents this as
    # parallel-lists on disk, but `.features["objects"]` returns a Sequence with
    # `.feature[...]` giving the inner Features dict. That matches CPPE-5's
    # schema so the reference notebook can do:
    #   ds.features["objects"].feature["category"].names
    features = Features({
        "image_id": Value("int32"),
        "image": HFImage(),
        "width": Value("int32"),
        "height": Value("int32"),
        "objects": Sequence({
            "id": Value("int64"),
            "area": Value("float32"),
            "bbox": Sequence(Value("float32"), length=4),
            "category": category_feature,
        }),
    })
    ds = Dataset.from_dict(rows, features=features)

    if subset is not None and 0.0 < subset < 1.0:
        n = max(1, int(round(len(ds) * subset)))
        ds = ds.shuffle(seed=seed).select(range(n))

    return ds


def load_feature_dataset(
    dataset_name: str,
    subset: Optional[float] = None,
    seed: int = 42,
    data_root: Optional[Path] = None,
) -> Dict[str, Dataset]:
    """Convenience wrapper — returns `{"train", "validation", "test"}` dict,
    matching the shape of HF `load_dataset("cppe-5")`.

    Intended as a one-line swap for the dataset-loading cell of the reference
    notebooks:

    ```python
    # Original cell:
    # dataset = load_dataset("cppe-5")

    # Replacement:
    from data_loader import load_feature_dataset
    dataset = load_feature_dataset("fire_detection", subset=0.05)
    ```
    """
    class_names = load_class_names(dataset_name)
    return {
        "train":      load_yolo_as_hf_dataset(dataset_name, "train", subset, seed, data_root, class_names),
        "validation": load_yolo_as_hf_dataset(dataset_name, "val",   subset, seed, data_root, class_names),
        "test":       load_yolo_as_hf_dataset(dataset_name, "test",  subset, seed, data_root, class_names),
    }


def load_class_names(dataset_name: str) -> Dict[int, str]:
    """Read class names from the feature's ``05_data.yaml`` if present."""
    import yaml

    feature_dir_map = {
        "fire_detection": "safety-fire_detection",
        "helmet_detection": "ppe-helmet_detection",
        "shoes_detection": "ppe-shoes_detection",
        "fall_detection": "safety-fall-detection",
        "phone_usage": "safety-poketenashi-phone-usage",
    }
    feature = feature_dir_map.get(dataset_name, f"safety-{dataset_name}")
    data_yaml = _REPO_ROOT / "features" / feature / "configs" / "05_data.yaml"
    if not data_yaml.exists():
        return {}
    cfg = yaml.safe_load(data_yaml.read_text())
    names = cfg.get("names") or {}
    return {int(k): str(v) for k, v in names.items()}


def dump_cppe5_to_yolo(
    output_root: Optional[Path] = None,
    split_seed: int = 1337,
    split_ratio: float = 0.15,
) -> Dict[str, int]:
    """Write the HF `cppe-5` dataset to our on-disk YOLO layout.

    Inverse of `load_yolo_as_hf_dataset` for this one specific source. The
    split matches qubvel's reference notebook (``train_test_split(0.15,
    seed=1337)`` on the HF ``train`` split to manufacture a ``validation``).

    Output layout (uses ``val`` directory name to match our convention):
        <output_root>/cppe5/
            train/{images, labels}/
            val/{images, labels}/
            test/{images, labels}/

    Args:
        output_root: directory that will hold ``cppe5/``. Defaults to
            ``<repo>/dataset_store/training_ready``.
        split_seed: RNG seed for the 85/15 train→val split (qubvel: 1337).
        split_ratio: validation fraction out of HF ``train`` (qubvel: 0.15).

    Returns:
        ``{"train": N, "val": N, "test": N}`` — sample counts per split.
    """
    from datasets import load_dataset

    root = Path(output_root) if output_root else _DEFAULT_DATA_ROOT
    base = root / "cppe5"

    ds = load_dataset("cppe-5")
    if "validation" not in ds:
        split = ds["train"].train_test_split(split_ratio, seed=split_seed)
        ds["train"] = split["train"]
        ds["validation"] = split["test"]

    counts: Dict[str, int] = {}
    for hf_split, disk_split in (("train", "train"), ("validation", "val"), ("test", "test")):
        split_dir = base / disk_split
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        n = 0
        for row in ds[hf_split]:
            img = row["image"].convert("RGB")
            # Use actual PIL image dims, NOT row["width"]/row["height"]. The
            # upstream CPPE-5 HF dataset has a known metadata-mismatch bug
            # (~6% of validation rows: e.g. image_id=70 claims 300×300 but
            # the real image is ~2000×1500). The reference notebook avoids
            # this bug by keeping bboxes in pixel space all the way to
            # Albumentations (which reads dims from `image.shape`). We do
            # the same here before normalizing-for-YOLO on disk.
            w, h = img.size
            image_id = int(row["image_id"])
            stem = f"{image_id:06d}"

            img.save(img_dir / f"{stem}.jpg", "JPEG", quality=95)

            lines = []
            for bbox_coco, cls in zip(row["objects"]["bbox"], row["objects"]["category"]):
                x, y, bw, bh = bbox_coco  # COCO pixel [x_top_left, y_top_left, w, h]
                if bw <= 0 or bh <= 0:
                    continue
                # Clip to image bounds (matches A.BboxParams(clip=True) in the
                # reference recipe — handles boxes that poke slightly past the
                # edge due to annotator rounding without dropping the row).
                x2 = min(float(w), float(x) + float(bw))
                y2 = min(float(h), float(y) + float(bh))
                x1 = max(0.0, float(x))
                y1 = max(0.0, float(y))
                bw_c, bh_c = x2 - x1, y2 - y1
                if bw_c <= 0 or bh_c <= 0:
                    continue
                cx = (x1 + bw_c / 2) / w
                cy = (y1 + bh_c / 2) / h
                wn = bw_c / w
                hn = bh_c / h
                lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            n += 1
        counts[disk_split] = n

    return counts


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="YOLO ↔ HF Dataset bridge utilities.")
    p.add_argument("--dump-cppe5", action="store_true",
                   help="Download HF cppe-5 and dump to dataset_store/training_ready/cppe5/ in YOLO format.")
    p.add_argument("--dataset", default="fire_detection",
                   help="(smoke-test mode) dataset name under training_ready/")
    p.add_argument("--split", default="train")
    p.add_argument("--subset", type=float, default=0.05)
    args = p.parse_args()

    if args.dump_cppe5:
        counts = dump_cppe5_to_yolo()
        dest = _DEFAULT_DATA_ROOT / "cppe5"
        print(f"\nDumped CPPE-5 → {dest}")
        for s, n in counts.items():
            print(f"  {s}: {n} images")
    else:
        ds = load_yolo_as_hf_dataset(args.dataset, args.split, subset=args.subset)
        names = load_class_names(args.dataset)
        print(f"Loaded {len(ds)} {args.split} samples from {args.dataset}")
        print(f"Class names: {names}")
        print(f"First row keys: {list(ds[0].keys())}")
        print(f"First row objects: {ds[0]['objects']}")

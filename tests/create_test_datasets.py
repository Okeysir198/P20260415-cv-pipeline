"""
Create small test datasets (80 train + 20 val) for CI smoke tests.

Samples from real raw data in dataset_store/raw/ where available;
derives from existing test_fire_100 for datasets without a clear source.

Usage:
    uv run tests/create_test_datasets.py           # skip existing
    uv run tests/create_test_datasets.py --force   # overwrite existing
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

# Project root on sys.path so core parsers import cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.p00_data_prep.parsers.voc import parse_voc
from core.p00_data_prep.parsers.coco import parse_coco

STORE = PROJECT_ROOT / "dataset_store"
RAW = STORE / "raw"

N_TRAIN = 80
N_VAL = 20
SEED = 42


def _split_paths(all_paths: list[Path], n_train: int, n_val: int) -> tuple:
    """Shuffle and split paths into train/val groups."""
    rng = random.Random(SEED)
    sampled = rng.sample(all_paths, min(len(all_paths), n_train + n_val))
    return sampled[:n_train], sampled[n_train : n_train + n_val]


def _setup_dirs(out_dir: Path) -> tuple[Path, Path, Path, Path]:
    """Create train/val images+labels dirs, return (train_img, train_lbl, val_img, val_lbl)."""
    for p in ["train/images", "train/labels", "val/images", "val/labels"]:
        (out_dir / p).mkdir(parents=True, exist_ok=True)
    return (
        out_dir / "train" / "images",
        out_dir / "train" / "labels",
        out_dir / "val" / "images",
        out_dir / "val" / "labels",
    )


def _copy_images_and_labels(
    samples: list[dict],
    class_map: dict[str, int],
    train_img_dir: Path,
    train_lbl_dir: Path,
    val_img_dir: Path,
    val_lbl_dir: Path,
) -> int:
    """Copy images, remap labels via class_map, write YOLO txt files.
    Returns total images written."""
    train_samples, val_samples = _split_paths(
        [s["image_path"] for s in samples], N_TRAIN, N_VAL
    )

    img_path_to_sample = {s["image_path"]: s for s in samples}
    total = 0

    for split_samples, img_dir, lbl_dir in [
        (train_samples, train_img_dir, train_lbl_dir),
        (val_samples, val_img_dir, val_lbl_dir),
    ]:
        for img_path in split_samples:
            sample = img_path_to_sample[img_path]
            dst_img = img_dir / img_path.name
            shutil.copy2(img_path, dst_img)

            lbl_path = lbl_dir / (img_path.stem + ".txt")
            lines: list[str] = []
            for label, bbox in zip(sample["labels"], sample["bboxes"]):
                cid = class_map.get(label)
                if cid is None:
                    continue
                lines.append(f"{cid} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")
            lbl_path.write_text("\n".join(lines) if lines else "", encoding="utf-8")
            total += 1

    return total


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------


def build_ppe(out_dir: Path, force: bool):
    """test_ppe_100 from hard_hat_workers (VOC)."""
    print("[1/6] test_ppe_100  (helmet detection, VOC source)")
    if out_dir.exists() and not force:
        print(f"  SKIP: {out_dir} exists")
        return

    if out_dir.exists():
        shutil.rmtree(out_dir)

    samples = parse_voc(
        {"path": str(RAW / "helmet_detection" / "hard_hat_workers"), "name": "hard_hat_workers"},
        PROJECT_ROOT,
    )
    print(f"  Parsed {len(samples)} VOC samples")

    # Source classes: helmet, head, person -> target: 0=person, 1=head_with_helmet, 2=head_without_helmet
    class_map = {"person": 0, "helmet": 1, "head": 2}
    ti, tl, vi, vl = _setup_dirs(out_dir)
    total = _copy_images_and_labels(samples, class_map, ti, tl, vi, vl)
    print(f"  Wrote {total} images to {out_dir}")


def build_shoes(out_dir: Path, force: bool):
    """test_shoes_detection_100 from keremberke_ppe (COCO)."""
    print("[2/6] test_shoes_detection_100  (shoes detection, COCO source)")
    if out_dir.exists() and not force:
        print(f"  SKIP: {out_dir} exists")
        return

    if out_dir.exists():
        shutil.rmtree(out_dir)

    samples = parse_coco(
        {
            "path": str(RAW / "shoes_detection" / "keremberke_ppe"),
            "name": "keremberke_ppe",
            "has_splits": True,
            "splits_to_use": ["train", "valid"],
        },
        PROJECT_ROOT,
    )
    # Filter to samples that have shoe-related annotations
    shoe_cats = {"shoes", "no_shoes"}
    shoe_samples = []
    for s in samples:
        has_shoe = any(l in shoe_cats for l in s["labels"])
        if has_shoe:
            shoe_samples.append(s)
    print(f"  Parsed {len(samples)} COCO samples, {len(shoe_samples)} with shoe annotations")

    # Target: 0=person, 1=foot_with_safety_shoes, 2=foot_without_safety_shoes
    # Source: shoes -> foot_with_safety_shoes, no_shoes -> foot_without_safety_shoes, person -> person
    class_map = {"shoes": 1, "no_shoes": 2, "person": 0}
    ti, tl, vi, vl = _setup_dirs(out_dir)
    total = _copy_images_and_labels(shoe_samples, class_map, ti, tl, vi, vl)
    print(f"  Wrote {total} images to {out_dir}")


def build_fire(out_dir: Path, force: bool):
    """test_fire_detection_100 from d_fire (YOLO format)."""
    print("[3/6] test_fire_detection_100  (fire detection, YOLO source)")
    if out_dir.exists() and not force:
        print(f"  SKIP: {out_dir} exists")
        return

    if out_dir.exists():
        shutil.rmtree(out_dir)

    src_dir = RAW / "fire_detection" / "d_fire" / "data"
    src_img_dir = src_dir / "train" / "images"
    src_lbl_dir = src_dir / "train" / "labels"

    img_paths = sorted(src_img_dir.glob("*"))
    img_paths = [p for p in img_paths if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    print(f"  Found {len(img_paths)} source images")

    # Source data.yaml: names: ['smoke', 'fire'] -> 0=smoke, 1=fire
    # Target config: 0=fire, 1=smoke  -> remap 0->1, 1->0
    train_imgs, val_imgs = _split_paths(img_paths, N_TRAIN, N_VAL)
    ti, tl, vi, vl = _setup_dirs(out_dir)
    total = 0

    for split_imgs, img_dir, lbl_dir in [
        (train_imgs, ti, tl),
        (val_imgs, vi, vl),
    ]:
        for img_path in split_imgs:
            shutil.copy2(img_path, img_dir / img_path.name)

            lbl_path = src_lbl_dir / (img_path.stem + ".txt")
            out_lbl = lbl_dir / (img_path.stem + ".txt")
            lines: list[str] = []
            if lbl_path.exists():
                for line in lbl_path.read_text(encoding="utf-8").strip().splitlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    src_cid = int(parts[0])
                    # Remap: 0(smoke)->1, 1(fire)->0
                    dst_cid = 1 - src_cid
                    lines.append(f"{dst_cid} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
            out_lbl.write_text("\n".join(lines) if lines else "", encoding="utf-8")
            total += 1

    print(f"  Wrote {total} images to {out_dir}")


def build_phone(out_dir: Path, force: bool):
    """test_phone_detection_100 derived from test_fire_100, relabel class 0 -> phone."""
    print("[4/6] test_phone_detection_100  (derived from test_fire_100)")
    if out_dir.exists() and not force:
        print(f"  SKIP: {out_dir} exists")
        return

    if out_dir.exists():
        shutil.rmtree(out_dir)

    src = STORE / "test_fire_100"
    ti, tl, vi, vl = _setup_dirs(out_dir)
    total = 0

    for split in ["train", "val"]:
        dst_img = ti if split == "train" else vi
        dst_lbl = tl if split == "train" else vl
        src_img = src / split / "images"
        src_lbl = src / split / "labels"

        for img_path in sorted(src_img.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            shutil.copy2(img_path, dst_img / img_path.name)

            lbl_path = src_lbl / (img_path.stem + ".txt")
            out_lbl = dst_lbl / (img_path.stem + ".txt")
            lines: list[str] = []
            if lbl_path.exists():
                for line in lbl_path.read_text(encoding="utf-8").strip().splitlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    # All classes -> 0 (phone_usage), keep bbox
                    lines.append(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
            out_lbl.write_text("\n".join(lines) if lines else "", encoding="utf-8")
            total += 1

    print(f"  Wrote {total} images to {out_dir}")


def build_fall(out_dir: Path, force: bool):
    """test_fall_detection_100 derived from test_fire_100, relabel for fall detection."""
    print("[5/6] test_fall_detection_100  (derived from test_fire_100)")
    if out_dir.exists() and not force:
        print(f"  SKIP: {out_dir} exists")
        return

    if out_dir.exists():
        shutil.rmtree(out_dir)

    src = STORE / "test_fire_100"
    ti, tl, vi, vl = _setup_dirs(out_dir)
    total = 0

    for split in ["train", "val"]:
        dst_img = ti if split == "train" else vi
        dst_lbl = tl if split == "train" else vl
        src_img = src / split / "images"
        src_lbl = src / split / "labels"

        for img_path in sorted(src_img.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            shutil.copy2(img_path, dst_img / img_path.name)

            lbl_path = src_lbl / (img_path.stem + ".txt")
            out_lbl = dst_lbl / (img_path.stem + ".txt")
            lines: list[str] = []
            if lbl_path.exists():
                for line in lbl_path.read_text(encoding="utf-8").strip().splitlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    src_cid = int(parts[0])
                    # source: 0=smoke, 1=fire -> target: 0=person, 1=fallen_person
                    dst_cid = src_cid  # 0->0 person, 1->1 fallen_person
                    lines.append(f"{dst_cid} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
            out_lbl.write_text("\n".join(lines) if lines else "", encoding="utf-8")
            total += 1

    print(f"  Wrote {total} images to {out_dir}")


def build_segmentation(out_dir: Path, force: bool):
    """test_segmentation derived from test_fire_100, converting bboxes to polygon masks."""
    print("[6/6] test_segmentation  (derived from test_fire_100, polygon masks)")
    if out_dir.exists() and not force:
        print(f"  SKIP: {out_dir} exists")
        return

    if out_dir.exists():
        shutil.rmtree(out_dir)

    src = STORE / "test_fire_100"
    ti, tl, vi, vl = _setup_dirs(out_dir)
    total = 0

    for split in ["train", "val"]:
        dst_img = ti if split == "train" else vi
        dst_lbl = tl if split == "train" else vl
        src_img = src / split / "images"
        src_lbl = src / split / "labels"

        for img_path in sorted(src_img.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            shutil.copy2(img_path, dst_img / img_path.name)

            lbl_path = src_lbl / (img_path.stem + ".txt")
            out_lbl = dst_lbl / (img_path.stem + ".txt")
            lines: list[str] = []
            if lbl_path.exists():
                for line in lbl_path.read_text(encoding="utf-8").strip().splitlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    src_cid = int(parts[0])
                    # Remap: 0(smoke)->1, 1(fire)->0 to match target 0=fire, 1=smoke
                    dst_cid = 1 - src_cid
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    # Convert YOLO bbox [cx,cy,w,h] to 4-point polygon
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    # YOLO segmentation format: class_id x1 y1 x2 y2 x3 y3 x4 y4
                    poly = f"{dst_cid} {x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}"
                    lines.append(poly)
            out_lbl.write_text("\n".join(lines) if lines else "", encoding="utf-8")
            total += 1

    print(f"  Wrote {total} images to {out_dir}")


# ---------------------------------------------------------------------------
# Config updaters
# ---------------------------------------------------------------------------


def update_phone_config():
    """Update 05_data_phone.yaml: rename class 'phone' -> 'phone_usage'."""
    config_path = PROJECT_ROOT / "configs" / "_test" / "05_data_phone.yaml"
    content = config_path.read_text(encoding="utf-8")
    content = content.replace("  0: phone", "  0: phone_usage")
    content = content.replace(
        '  phone: "a mobile phone, smartphone, cellphone held in hand or near face"',
        '  phone_usage: "a mobile phone, smartphone, cellphone held in hand or near face"',
    )
    config_path.write_text(content, encoding="utf-8")
    print("  Updated configs/_test/05_data_phone.yaml")


def update_segmentation_config():
    """Update 05_data_segmentation.yaml for fire/smoke segmentation with polygon labels."""
    config_path = PROJECT_ROOT / "configs" / "_test" / "05_data_segmentation.yaml"
    new_content = """# Dataset config - Segmentation (100 images for pipeline testing)
dataset_name: "test_segmentation"
path: "../../dataset_store/test_segmentation"
train: "train/images"
val: "val/images"
test: "val/images"

names:
  0: fire
  1: smoke
num_classes: 2
input_size: [640, 640]

mean: [0.485, 0.456, 0.406]
std: [0.229, 0.224, 0.225]
"""
    config_path.write_text(new_content, encoding="utf-8")
    print("  Updated configs/_test/05_data_segmentation.yaml")


def update_configs():
    """Update test configs that need changes to match the generated datasets.

    PPE, shoes, fire, and fall configs already have correct class names.
    Phone needs class rename; segmentation needs full rewrite.
    """
    print("\nUpdating test configs...")
    update_phone_config()
    update_segmentation_config()
    print("  (ppe, shoes, fire, fall configs already correct -- no changes needed)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Create small test datasets for CI smoke tests")
    parser.add_argument("--force", action="store_true", help="Overwrite existing datasets")
    args = parser.parse_args()

    print(f"Creating test datasets in {STORE}/")
    print(f"  Train: {N_TRAIN} images, Val: {N_VAL} images, Seed: {SEED}")
    if args.force:
        print("  --force: will overwrite existing datasets\n")
    else:
        print("  (skip existing, use --force to overwrite)\n")

    builders = [
        (STORE / "test_ppe_100", build_ppe),
        (STORE / "test_shoes_detection_100", build_shoes),
        (STORE / "test_fire_detection_100", build_fire),
        (STORE / "test_phone_detection_100", build_phone),
        (STORE / "test_fall_detection_100", build_fall),
        (STORE / "test_segmentation", build_segmentation),
    ]

    for out_dir, builder in builders:
        try:
            builder(out_dir, args.force)
        except Exception as e:
            print(f"  ERROR creating {out_dir.name}: {e}")
            import traceback
            traceback.print_exc()

    update_configs()
    print("\nDone.")


if __name__ == "__main__":
    main()

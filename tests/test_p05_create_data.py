"""Test 01: Dataset Preparation — verify fixture data exists and is valid."""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from _runner import passed, failed, errors, run_all

OUTPUTS = Path(__file__).resolve().parent / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


def run_test(name, func):
    global passed, failed
    try:
        result = func()
        # Check if function explicitly returned a value (pytest functions should return None)
        if result is not None:
            print(f"  WARNING: {name} returned {type(result).__name__} instead of None")
        print(f"  PASS: {name}")
        passed += 1
    except AssertionError as e:
        print(f"  FAIL: {name} — {e}")
        failed += 1
        errors.append((name, str(e)))
    except Exception as e:
        print(f"  FAIL: {name} — {e}")
        failed += 1
        errors.append((name, str(e)))


def test_fixture_data_exists():
    """Verify fixture data directory structure exists."""
    fixture_dir = ROOT / "tests" / "fixtures" / "data"

    assert fixture_dir.exists(), f"Fixture dir not found: {fixture_dir}"
    for split in ["train", "val"]:
        for sub in ["images", "labels"]:
            d = fixture_dir / split / sub
            assert d.exists(), f"Missing: {d}"


def test_image_count():
    """Verify fixture data has ~10 train / ~5 val images."""
    fixture_dir = ROOT / "tests" / "fixtures" / "data"
    train_imgs = list((fixture_dir / "train" / "images").glob("*"))
    val_imgs = list((fixture_dir / "val" / "images").glob("*"))

    assert len(train_imgs) >= 8, f"Too few train images: {len(train_imgs)}"
    assert len(val_imgs) >= 3, f"Too few val images: {len(val_imgs)}"
    assert len(train_imgs) + len(val_imgs) >= 10, (
        f"Total images too low: {len(train_imgs) + len(val_imgs)}"
    )
    print(f"    Train: {len(train_imgs)}, Val: {len(val_imgs)}")


def test_label_format():
    """Read label files and verify YOLO format (class_id cx cy w h, all in [0,1])."""
    fixture_dir = ROOT / "tests" / "fixtures" / "data"
    label_files = list((fixture_dir / "train" / "labels").glob("*.txt"))
    assert len(label_files) > 0, "No label files found"

    checked = 0
    for lf in label_files[:10]:
        text = lf.read_text().strip()
        if not text:
            continue
        for line in text.split("\n"):
            parts = line.strip().split()
            assert len(parts) == 5, f"Bad label line in {lf.name}: {line}"
            cls_id = int(parts[0])
            assert cls_id in (0, 1), f"Invalid class {cls_id} in {lf.name}"
            coords = [float(x) for x in parts[1:]]
            for c in coords:
                assert 0.0 <= c <= 1.0, f"Coord out of range in {lf.name}: {c}"
            checked += 1
    assert checked > 0, "No valid labels checked"
    print(f"    Checked {checked} label lines across {min(10, len(label_files))} files")


def save_outputs():
    """Save fixture dataset info to outputs."""
    fixture_dir = ROOT / "tests" / "fixtures" / "data"
    info = {}
    for split in ["train", "val"]:
        imgs = list((fixture_dir / split / "images").glob("*"))
        lbls = list((fixture_dir / split / "labels").glob("*.txt"))
        # Count class distribution
        class_counts = {0: 0, 1: 0}
        for lf in lbls:
            text = lf.read_text().strip()
            if not text:
                continue
            for line in text.split("\n"):
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
        info[split] = {
            "images": len(imgs),
            "labels": len(lbls),
            "class_distribution": {
                f"{k} ({'fire' if k == 0 else 'smoke'})": v for k, v in class_counts.items()
            },
        }

    out_file = OUTPUTS / "00_dataset_info.txt"
    with open(out_file, "w") as f:
        json.dump(info, f, indent=2)
    print(f"    Saved: {out_file}")


if __name__ == "__main__":
    print("=" * 60)
    print("Test 01: Dataset Preparation")
    print("=" * 60)

    run_all([
        ("fixture_data_exists", test_fixture_data_exists),
        ("image_count", test_image_count),
        ("label_format", test_label_format),
    ], title=None, exit_on_fail=False)

    try:
        save_outputs()
    except Exception as e:
        print(f"  WARNING: Could not save outputs — {e}")

    if failed > 0:
        sys.exit(1)

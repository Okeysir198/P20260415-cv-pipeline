"""Test 07: Utils — paddle_bridge YOLO→COCO converter.

Real end-to-end conversion on a minimal synthetic YOLO dataset, then
verify COCO JSON structure. No mocks.
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from _runner import run_all  # noqa: E402
from utils.paddle_bridge import get_image_size, yolo_to_coco  # noqa: E402


def _make_yolo_tree(root: Path) -> Path:
    """Build a tiny YOLO-format dataset + 05_data.yaml, return config path."""
    dataset = root / "dset"
    for split in ["train", "val"]:
        (dataset / split / "images").mkdir(parents=True)
        (dataset / split / "labels").mkdir(parents=True)

    # Two train images, one with two boxes, one with no label file
    Image.new("RGB", (200, 100), color=(128, 64, 0)).save(
        dataset / "train" / "images" / "a.jpg"
    )
    (dataset / "train" / "labels" / "a.txt").write_text(
        "0 0.5 0.5 0.4 0.2\n1 0.25 0.25 0.1 0.1\n"
    )
    Image.new("RGB", (100, 100), color=(0, 128, 0)).save(
        dataset / "train" / "images" / "b.jpg"
    )
    # b.txt intentionally missing → should count as image with 0 annotations

    # One val image
    Image.new("RGB", (50, 50), color=(0, 0, 200)).save(
        dataset / "val" / "images" / "v.jpg"
    )
    (dataset / "val" / "labels" / "v.txt").write_text("0 0.5 0.5 1.0 1.0\n")

    config = {
        "dataset_name": "tiny",
        "path": "dset",
        "train": "train/images",
        "val": "val/images",
        "names": {0: "fire", 1: "smoke"},
    }
    config_path = root / "05_data.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def test_get_image_size_pil():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "img.png"
        Image.new("RGB", (320, 240)).save(p)
        assert get_image_size(p) == (320, 240)


def test_yolo_to_coco_end_to_end():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        config_path = _make_yolo_tree(root)
        out_dir = root / "ann"

        paths = yolo_to_coco(str(config_path), str(out_dir), ["train", "val"])
        assert set(paths.keys()) == {"train", "val"}

        train_coco = json.loads(paths["train"].read_text())
        assert len(train_coco["images"]) == 2
        # 2 annotations from a.txt, 0 from missing b.txt
        assert len(train_coco["annotations"]) == 2

        # Categories must be 1-indexed (PaddleDetection convention)
        cat_ids = sorted(c["id"] for c in train_coco["categories"])
        assert cat_ids == [1, 2]

        # Verify a 200x100 image w/ YOLO box (0.5 0.5 0.4 0.2) → COCO bbox
        # w=0.4*200=80, h=0.2*100=20, x=(0.5*200)-40=60, y=(0.5*100)-10=40
        img_a = next(i for i in train_coco["images"] if i["file_name"].endswith("a.jpg"))
        anns_a = [a for a in train_coco["annotations"] if a["image_id"] == img_a["id"]]
        assert len(anns_a) == 2
        first = next(a for a in anns_a if a["category_id"] == 1)
        assert first["bbox"] == [60.0, 40.0, 80.0, 20.0]
        assert first["area"] == 1600.0

        val_coco = json.loads(paths["val"].read_text())
        assert len(val_coco["images"]) == 1
        assert len(val_coco["annotations"]) == 1
        assert val_coco["annotations"][0]["bbox"] == [0.0, 0.0, 50.0, 50.0]


def test_yolo_to_coco_skips_missing_split():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        config_path = _make_yolo_tree(root)
        out_dir = root / "ann"
        # request "test" which wasn't created
        paths = yolo_to_coco(str(config_path), str(out_dir), ["test"])
        assert paths == {}


if __name__ == "__main__":
    run_all([
        ("get_image_size", test_get_image_size_pil),
        ("yolo_to_coco_end_to_end", test_yolo_to_coco_end_to_end),
        ("yolo_to_coco_missing_split", test_yolo_to_coco_skips_missing_split),
    ], title="Test utils07: paddle_bridge")

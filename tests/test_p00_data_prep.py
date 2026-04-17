#!/usr/bin/env python3
"""
Test suite for Generic CV Data Preparation Tool.

Tests use REAL datasets from dataset_store/raw/ to validate:
- Format parsers (YOLO, VOC, COCO)
- Class mapper
- Split generator
- Detection adapter
- File operations
"""

import json
import sys
import tempfile
import traceback
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.p00_data_prep.core.splitter import SplitGenerator
from core.p00_data_prep.utils.class_mapper import ClassMapper
from core.p00_data_prep.utils.file_ops import FileOps
from core.p00_data_prep.adapters.detection import DetectionAdapter
from core.p00_data_prep.parsers import yolo, voc, coco


# Real dataset paths (relative to project root)
REAL_DATASETS = {
    "yolo_flat": "dataset_store/raw/fire_detection/d_fire/data/data/train",
    "voc": "dataset_store/raw/helmet_detection/hard_hat_workers/data",
}


def check_dataset_exists(path: str) -> bool:
    return (ROOT / path).exists()


def test_yolo_parser_real():
    """Test YOLO format parser with real dataset."""
    print("\n=== Testing YOLO Parser (Real Data) ===")

    dataset_path = REAL_DATASETS["yolo_flat"]

    if not check_dataset_exists(dataset_path):
        print(f"⚠️  Skipping - dataset not found: {dataset_path}")
        return

    source_config = {
        "name": "d_fire",
        "path": dataset_path,
        "has_splits": False
    }

    samples = yolo.parse_yolo(source_config, ROOT)

    print(f"   Found {len(samples)} samples from d_fire dataset")

    assert len(samples) > 0, "No samples found"

    sample = samples[0]
    assert "filename" in sample
    assert "image_path" in sample
    assert "labels" in sample
    assert "source" in sample
    assert sample["source"] == "d_fire"

    if sample["labels"]:
        print(f"   Sample labels: {sample['labels'][:3]}...")

    print("✅ YOLO parser tests passed")


def test_voc_parser_real():
    """Test VOC format parser with real dataset."""
    print("\n=== Testing VOC Parser (Real Data) ===")

    dataset_path = REAL_DATASETS["voc"]

    if not check_dataset_exists(dataset_path):
        print(f"⚠️  Skipping - dataset not found: {dataset_path}")
        return

    source_config = {
        "name": "hard_hat_workers",
        "path": dataset_path,
        "has_splits": False
    }

    samples = voc.parse_voc(source_config, ROOT)

    print(f"   Found {len(samples)} samples from hard_hat_workers dataset")

    assert len(samples) > 0, "No samples found"

    sample = samples[0]
    assert "filename" in sample
    assert "image_path" in sample
    assert "labels" in sample
    assert "source" in sample

    if sample["labels"]:
        unique_labels = {label for s in samples for label in s["labels"]}
        print(f"   Unique classes in dataset: {unique_labels}")
        assert "helmet" in unique_labels or "head" in unique_labels

    print("✅ VOC parser tests passed")


def test_coco_parser_real():
    """Test COCO format parser with real dataset."""
    print("\n=== Testing COCO Parser (Real Data) ===")

    dataset_path = "dataset_store/raw/shoes_detection/keremberke_ppe/data"

    if not check_dataset_exists(dataset_path):
        print(f"⚠️  Skipping - dataset not found: {dataset_path}")
        return

    # Roboflow COCO exports put the annotation file inside each split folder
    source_config = {
        "name": "keremberke_ppe",
        "path": dataset_path + "/train",
        "format": "coco",
        "has_splits": False
    }

    samples = coco.parse_coco(source_config, ROOT)

    print(f"   Found {len(samples)} samples from keremberke_ppe dataset")

    assert len(samples) > 0, f"No samples found from {dataset_path}"

    sample = samples[0]
    assert "filename" in sample
    assert "image_path" in sample
    assert "labels" in sample

    if sample["labels"]:
        print(f"   Sample labels: {sample['labels'][:3]}...")

    print("✅ COCO parser tests passed")


def test_split_generator_real():
    """Test split generator with real labels."""
    print("\n=== Testing SplitGenerator (Real Labels) ===")

    samples = []
    for i in range(100):
        if i < 60:
            labels = ["helmet"]
        elif i < 90:
            labels = ["head"]
        else:
            labels = ["helmet", "head"]
        samples.append({"filename": f"img{i:04d}.jpg", "labels": labels})

    splitter = SplitGenerator(ratios=(0.8, 0.1, 0.1), seed=42, stratified=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        splits_file = Path(tmpdir) / "splits.json"
        splits = splitter.generate_splits(samples, splits_file, task_type="detection")

        print(f"   Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

        assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == 100
        assert splits_file.exists()

        with open(splits_file) as f:
            data = json.load(f)

        assert data["metadata"]["stratified"] == True
        assert data["metadata"]["total_samples"] == 100
        assert "counts" in data
        assert sum(data["counts"].values()) == 100

    print("✅ SplitGenerator tests passed")


def test_class_mapper_real():
    """Test class mapper with real class names."""
    print("\n=== Testing ClassMapper (Real Classes) ===")

    target_classes = ["person", "head_with_helmet", "head_without_helmet"]

    class_map = {
        "Hardhat": "head_with_helmet",
        "NO-Hardhat": "head_without_helmet",
        "Person": "person",
        "helmet": "head_with_helmet",
        "no_helmet": "head_without_helmet",
        "person": "person",
    }

    mapper = ClassMapper(target_classes, class_map)

    assert mapper.get_target_id("Hardhat") == 1  # head_with_helmet
    assert mapper.get_target_id("NO-Hardhat") == 2  # head_without_helmet
    assert mapper.get_target_id("Person") == 0  # person
    assert mapper.get_target_id("helmet") == 1  # head_with_helmet

    source_classes = {"Hardhat", "NO-Hardhat", "Person", "helmet", "unknown_class"}
    unmapped = mapper.validate_mapping(source_classes)

    assert "unknown_class" in unmapped
    assert "Hardhat" not in unmapped

    print(f"   Unmapped classes: {unmapped}")
    print("✅ ClassMapper tests passed")


def test_file_ops_real():
    """Test file operations with real image detection."""
    print("\n=== Testing FileOps (Real Images) ===")

    dataset_path = ROOT / REAL_DATASETS["yolo_flat"]

    if not dataset_path.exists():
        print(f"⚠️  Skipping - dataset not found: {REAL_DATASETS['yolo_flat']}")
        return

    ops = FileOps(handle_duplicates="rename")

    img_files = ops.get_image_files(dataset_path / "images")

    print(f"   Found {len(img_files)} image files")

    assert len(img_files) > 0

    for img_path in img_files[:5]:
        assert ops.is_image_file(img_path)

    assert not ops.is_image_file(dataset_path / "labels" / "data.yaml")

    print("✅ FileOps tests passed")


def test_detection_adapter_real():
    """Test detection adapter with real dataset config."""
    print("\n=== Testing DetectionAdapter (Real Config) ===")

    config = {
        "task": "detection",
        "dataset_name": "helmet_detection_test",
        "classes": ["person", "head_with_helmet", "head_without_helmet"],
        "sources": [
            {
                "name": "hard_hat_workers",
                "path": REAL_DATASETS["voc"],
                "format": "voc",
                "has_splits": False,
                "class_map": {
                    "helmet": "head_with_helmet",
                    "head": "head_without_helmet",
                    "person": "person"
                }
            }
        ]
    }

    if not (ROOT / REAL_DATASETS["voc"]).exists():
        print(f"⚠️  Skipping - dataset not found: {REAL_DATASETS['voc']}")
        return

    adapter = DetectionAdapter(config)

    print(f"   Parsing {REAL_DATASETS['voc']}...")
    samples = adapter.parse_source(config["sources"][0], ROOT)

    print(f"   Found {len(samples)} samples")
    assert len(samples) > 0

    converted = adapter.convert_annotations(
        samples,
        config["classes"],
        adapter.class_mapper
    )

    print(f"   Converted {len(converted)} samples")
    assert len(converted) > 0

    sample = converted[0]
    assert "objects" in sample

    if sample["objects"]:
        obj = sample["objects"][0]
        assert "class_id" in obj
        assert "class_name" in obj
        print(f"   Sample object: {obj}")

    stats = adapter.get_class_statistics(converted)
    print(f"   Class stats: {stats}")
    assert sum(stats.values()) > 0

    print("✅ DetectionAdapter tests passed")


def test_full_pipeline_real():
    """Test end-to-end pipeline with real data (small subset)."""
    print("\n=== Testing Full Pipeline (Real Data Subset) ===")

    dataset_path = REAL_DATASETS["voc"]

    if not check_dataset_exists(dataset_path):
        print(f"⚠️  Skipping - dataset not found: {dataset_path}")
        return

    config = {
        "task": "detection",
        "dataset_name": "test_pipeline",
        "classes": ["head_with_helmet", "head_without_helmet"],
        "sources": [
            {
                "name": "hard_hat_workers",
                "path": dataset_path,
                "format": "voc",
                "has_splits": False,
                "class_map": {
                    "helmet": "head_with_helmet",
                    "head": "head_without_helmet"
                }
            }
        ],
        "splits": {
            "train": 0.7,
            "val": 0.15,
            "test": 0.15,
            "seed": 42
        }
    }

    adapter = DetectionAdapter(config)

    print("   Merging sources...")
    all_samples = []
    for source_config in config["sources"]:
        samples = adapter.parse_source(source_config, ROOT)
        all_samples.extend(samples[:50])  # limit to 50 for speed

    print(f"   Collected {len(all_samples)} samples")

    converted = adapter.convert_annotations(
        all_samples,
        config["classes"],
        adapter.class_mapper
    )

    print(f"   Converted {len(converted)} samples")

    splitter = SplitGenerator(ratios=(0.7, 0.15, 0.15), seed=42, stratified=True)

    split_samples = [
        {"filename": s["filename"], "labels": s.get("labels", [])}
        for s in converted
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        splits_file = Path(tmpdir) / "splits.json"
        splits = splitter.generate_splits(split_samples, splits_file, task_type="detection")

        print(f"   Splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

        assert splits_file.exists()

        with open(splits_file) as f:
            data = json.load(f)

        assert data["task_type"] == "detection"
        assert data["metadata"]["stratified"] == True

    stats = adapter.get_class_statistics(converted)
    assert len(stats) > 0
    print(f"   Final class stats: {stats}")

    print("✅ Full pipeline tests passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Testing Generic CV Data Preparation Tool")
    print("Using REAL datasets from dataset_store/raw/")
    print("=" * 60)

    try:
        test_yolo_parser_real()
        test_voc_parser_real()
        test_coco_parser_real()
        test_split_generator_real()
        test_class_mapper_real()
        test_file_ops_real()
        test_detection_adapter_real()
        test_full_pipeline_real()

        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()

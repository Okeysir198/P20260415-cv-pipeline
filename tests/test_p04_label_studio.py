"""Test 08: Label Studio Bridge — format conversion, I/O, config, path mapping, gather, build, CLI parser, integration."""

import argparse
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import passed, failed, errors, run_test, run_all
from core.p04_label_studio.bridge import (
    LabelStudioAPI,
    _image_path_to_ls_url,
    _label_path_for_image,
    _ls_url_to_label_path,
    build_parser,
    build_task,
    cmd_import,
    cmd_setup,
    generate_label_config,
    gather_auto_annotate_pairs,
    gather_dataset_pairs,
    gather_qa_fixes_pairs,
    load_ls_config,
    ls_to_yolo,
    read_yolo_labels,
    resolve_api_key,
    write_yolo_labels,
    yolo_to_ls,
)

OUTPUTS = Path(__file__).resolve().parent / "outputs" / "04_label_studio_bridge"
OUTPUTS.mkdir(parents=True, exist_ok=True)

DATA_CONFIG_PATH = str(ROOT / "configs" / "_test" / "05_data.yaml")
FIXTURES_DIR = ROOT / "tests" / "fixtures" / "data"
CLASS_NAMES = {0: "fire", 1: "smoke"}
CLASS_NAME_TO_ID = {"fire": 0, "smoke": 1}


# ---------------------------------------------------------------------------
# Service check (for integration tests)
# ---------------------------------------------------------------------------


def _check_ls_service() -> bool:
    try:
        resp = requests.get("http://localhost:18103/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


HAS_LS_SERVICE = _check_ls_service()


# ===================================================================
# Section A: Unit Tests (no service needed)
# ===================================================================


# --- A1-A4: yolo_to_ls format conversion ---


def test_yolo_to_ls_basic():
    """Convert a fire bbox and verify LS rectanglelabels structure."""
    result = yolo_to_ls(class_id=0, cx=0.5, cy=0.5, w=0.3, h=0.2, class_names=CLASS_NAMES)
    assert result["type"] == "rectanglelabels", f"Expected type rectanglelabels, got {result['type']}"
    assert result["from_name"] == "label", f"Expected from_name label, got {result['from_name']}"
    assert result["to_name"] == "image", f"Expected to_name image, got {result['to_name']}"
    value = result["value"]
    assert 0 <= value["x"] <= 100, f"x out of range: {value['x']}"
    assert 0 <= value["y"] <= 100, f"y out of range: {value['y']}"
    assert 0 < value["width"] <= 100, f"width out of range: {value['width']}"
    assert 0 < value["height"] <= 100, f"height out of range: {value['height']}"
    assert value["rectanglelabels"] == ["fire"], f"Expected ['fire'], got {value['rectanglelabels']}"
    print(f"    result: x={value['x']}, y={value['y']}, w={value['width']}, h={value['height']}")


def test_yolo_to_ls_smoke():
    """Convert a smoke bbox and verify label."""
    result = yolo_to_ls(class_id=1, cx=0.25, cy=0.75, w=0.4, h=0.15, class_names=CLASS_NAMES)
    assert result["value"]["rectanglelabels"] == ["smoke"], (
        f"Expected ['smoke'], got {result['value']['rectanglelabels']}"
    )
    print(f"    smoke label confirmed: {result['value']['rectanglelabels']}")


def test_yolo_to_ls_clamping():
    """Negative center coords should be clamped to valid range."""
    result = yolo_to_ls(class_id=0, cx=-0.1, cy=-0.1, w=0.5, h=0.5, class_names=CLASS_NAMES)
    value = result["value"]
    assert value["x"] >= 0, f"x should be clamped to >= 0, got {value['x']}"
    assert value["y"] >= 0, f"y should be clamped to >= 0, got {value['y']}"
    print(f"    clamped: x={value['x']}, y={value['y']}, w={value['width']}, h={value['height']}")


def test_yolo_to_ls_unknown_class():
    """Unknown class_id should produce 'class_{id}' fallback label."""
    result = yolo_to_ls(class_id=99, cx=0.5, cy=0.5, w=0.3, h=0.2, class_names=CLASS_NAMES)
    assert result["value"]["rectanglelabels"] == ["class_99"], (
        f"Expected ['class_99'], got {result['value']['rectanglelabels']}"
    )


# --- A5-A8: ls_to_yolo format conversion ---


def test_ls_to_yolo_basic():
    """Convert a valid LS rectanglelabels result back to YOLO format."""
    result = {
        "type": "rectanglelabels",
        "value": {
            "x": 20.0,
            "y": 30.0,
            "width": 40.0,
            "height": 50.0,
            "rectanglelabels": ["fire"],
        },
    }
    converted = ls_to_yolo(result, CLASS_NAME_TO_ID)
    assert converted is not None, "Expected non-None result"
    class_id, cx, cy, w, h = converted
    assert class_id == 0, f"Expected class_id=0, got {class_id}"
    assert cx == 0.4, f"Expected cx=0.4, got {cx}"
    assert cy == 0.55, f"Expected cy=0.55, got {cy}"
    assert w == 0.4, f"Expected w=0.4, got {w}"
    assert h == 0.5, f"Expected h=0.5, got {h}"
    print(f"    converted: ({class_id}, {cx}, {cy}, {w}, {h})")


def test_ls_to_yolo_unknown_label():
    """Unknown label name should return None."""
    result = {
        "type": "rectanglelabels",
        "value": {
            "x": 20.0,
            "y": 30.0,
            "width": 40.0,
            "height": 50.0,
            "rectanglelabels": ["unknown_class"],
        },
    }
    assert ls_to_yolo(result, CLASS_NAME_TO_ID) is None, "Unknown label should return None"


def test_ls_to_yolo_wrong_type():
    """Non-rectanglelabels type should return None."""
    result = {
        "type": "polygonlabels",
        "value": {
            "x": 20.0,
            "y": 30.0,
            "width": 40.0,
            "height": 50.0,
            "rectanglelabels": ["fire"],
        },
    }
    assert ls_to_yolo(result, CLASS_NAME_TO_ID) is None, "Wrong type should return None"


def test_ls_to_yolo_empty_labels():
    """Empty rectanglelabels list should return None."""
    result = {
        "type": "rectanglelabels",
        "value": {
            "x": 20.0,
            "y": 30.0,
            "width": 40.0,
            "height": 50.0,
            "rectanglelabels": [],
        },
    }
    assert ls_to_yolo(result, CLASS_NAME_TO_ID) is None, "Empty labels should return None"


# --- A9: Round-trip ---


def test_yolo_ls_roundtrip():
    """YOLO -> LS -> YOLO round-trip should preserve values within tolerance.

    Uses values that won't be clamped by yolo_to_ls (center must be far
    enough from edges so x_pct >= 0 and w_pct <= 100 - x_pct).
    """
    test_cases = [
        (0.5, 0.5, 0.3, 0.2),
        (0.25, 0.75, 0.4, 0.15),
        (0.5, 0.5, 0.6, 0.6),
    ]
    for cx, cy, w, h in test_cases:
        ls_result = yolo_to_ls(class_id=0, cx=cx, cy=cy, w=w, h=h, class_names=CLASS_NAMES)
        # Build the LS result dict from the yolo_to_ls output for ls_to_yolo
        ls_input = {
            "type": ls_result["type"],
            "value": ls_result["value"],
        }
        converted = ls_to_yolo(ls_input, CLASS_NAME_TO_ID)
        assert converted is not None, f"Round-trip returned None for ({cx}, {cy}, {w}, {h})"
        _, rt_cx, rt_cy, rt_w, rt_h = converted
        assert abs(rt_cx - cx) < 0.001, f"cx mismatch: {cx} vs {rt_cx}"
        assert abs(rt_cy - cy) < 0.001, f"cy mismatch: {cy} vs {rt_cy}"
        assert abs(rt_w - w) < 0.001, f"w mismatch: {w} vs {rt_w}"
        assert abs(rt_h - h) < 0.001, f"h mismatch: {h} vs {rt_h}"
    print(f"    all {len(test_cases)} round-trip cases passed")


# --- A10-A12: read_yolo_labels ---


def test_read_yolo_labels_basic():
    """Read a real fixture label file and verify structure."""
    label_dir = FIXTURES_DIR / "val" / "labels"
    label_files = sorted(label_dir.glob("*.txt"))
    assert len(label_files) > 0, f"No label files found in {label_dir}"
    label_path = label_files[0]
    annotations = read_yolo_labels(label_path)
    assert isinstance(annotations, list), f"Expected list, got {type(annotations)}"
    assert len(annotations) > 0, f"Expected at least 1 annotation, got {len(annotations)}"
    for ann in annotations:
        assert len(ann) == 5, f"Expected 5-tuple, got {len(ann)}-tuple"
        class_id, cx, cy, w, h = ann
        assert isinstance(class_id, int), f"class_id should be int, got {type(class_id)}"
        assert all(isinstance(v, float) for v in (cx, cy, w, h)), "cx,cy,w,h should be float"
    print(f"    read {len(annotations)} annotations from {label_path.name}")


def test_read_yolo_labels_missing_file():
    """Reading a non-existent file should return empty list."""
    result = read_yolo_labels(Path("/nonexistent/path/label.txt"))
    assert result == [], f"Expected [], got {result}"


def test_read_yolo_labels_malformed():
    """Malformed lines should be skipped; valid lines preserved."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        # Line 1: only 4 fields (< 5) -> skipped by field count check
        f.write("0 0.5 0.5 0.3\n")
        # Line 2: not parseable as numbers -> skipped by ValueError
        f.write("bad line\n")
        # Line 3: valid 5-field annotation
        f.write("1 0.2 0.3 0.1 0.15\n")
        tmp_path = Path(f.name)
    try:
        annotations = read_yolo_labels(tmp_path)
        assert len(annotations) == 1, (
            f"Expected 1 valid annotation (bad lines skipped), got {len(annotations)}"
        )
        cid, cx, cy, w, h = annotations[0]
        assert cid == 1, f"Expected class_id=1, got {cid}"
        print(f"    got {len(annotations)} valid annotations from malformed file")
    finally:
        tmp_path.unlink()


# --- A13-A14: write_yolo_labels ---


def test_write_yolo_labels_basic():
    """Write a single annotation and verify file content format."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        tmp_path = Path(f.name)
    try:
        write_yolo_labels(tmp_path, [(0, 0.5, 0.5, 0.3, 0.2)])
        content = tmp_path.read_text()
        assert "0 0.500000 0.500000 0.300000 0.200000" in content, (
            f"Unexpected content: {content}"
        )
        print(f"    content: {content.strip()}")
    finally:
        tmp_path.unlink()


def test_write_read_roundtrip():
    """Write two annotations, read back, verify exact match."""
    annotations = [
        (0, 0.123456, 0.234567, 0.345678, 0.456789),
        (1, 0.987654, 0.876543, 0.765432, 0.654321),
    ]
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        tmp_path = Path(f.name)
    try:
        write_yolo_labels(tmp_path, annotations)
        read_back = read_yolo_labels(tmp_path)
        assert len(read_back) == len(annotations), (
            f"Expected {len(annotations)} annotations, got {len(read_back)}"
        )
        for (o_cid, o_cx, o_cy, o_w, o_h), (r_cid, r_cx, r_cy, r_w, r_h) in zip(annotations, read_back):
            assert o_cid == r_cid, f"class_id mismatch: {o_cid} vs {r_cid}"
            assert o_cx == r_cx, f"cx mismatch: {o_cx} vs {r_cx}"
            assert o_cy == r_cy, f"cy mismatch: {o_cy} vs {r_cy}"
            assert o_w == r_w, f"w mismatch: {o_w} vs {r_w}"
            assert o_h == r_h, f"h mismatch: {o_h} vs {r_h}"
        print(f"    write/read round-trip matched for {len(annotations)} annotations")
    finally:
        tmp_path.unlink()


# --- A15-A16: resolve_api_key ---


def test_resolve_api_key_cli_priority():
    """CLI key should take priority over env var and config."""
    old_env = os.environ.pop("LS_API_KEY", None)
    try:
        os.environ["LS_API_KEY"] = "env_key"
        config = {"label_studio": {"api_key": "config_key"}}
        result = resolve_api_key("cli_key", config)
        assert result == "cli_key", f"Expected 'cli_key', got '{result}'"
        print(f"    CLI key correctly prioritized: {result}")
    finally:
        if old_env is not None:
            os.environ["LS_API_KEY"] = old_env
        else:
            os.environ.pop("LS_API_KEY", None)


def test_resolve_api_key_missing():
    """No key from any source should raise ValueError."""
    old_env = os.environ.pop("LS_API_KEY", None)
    try:
        config = {"label_studio": {}}
        raised = False
        try:
            resolve_api_key(None, config)
        except ValueError:
            raised = True
        assert raised, "Expected ValueError when no API key is found"
        print("    ValueError raised as expected")
    finally:
        if old_env is not None:
            os.environ["LS_API_KEY"] = old_env


# --- A17-A18: load_ls_config ---


def test_load_ls_config_default():
    """Default config should contain url and local_files_root."""
    config = load_ls_config()
    assert isinstance(config, dict), f"Expected dict, got {type(config)}"
    ls = config.get("label_studio", {})
    assert "url" in ls, "Missing 'url' in label_studio config"
    assert "local_files_root" in ls, "Missing 'local_files_root' in label_studio config"
    print(f"    default url={ls['url']}, local_files_root={ls['local_files_root']}")


def test_load_ls_config_missing_file():
    """Non-existent config file should return sensible defaults."""
    config = load_ls_config("/nonexistent/path.yaml")
    assert isinstance(config, dict), f"Expected dict, got {type(config)}"
    ls = config.get("label_studio", {})
    assert ls.get("url") == "http://localhost:18103", (
        f"Expected default url, got {ls.get('url')}"
    )
    print(f"    missing file fallback: url={ls.get('url')}")


# --- A19-A20: generate_label_config ---


def test_generate_label_config_basic():
    """Generated XML should contain class names and expected tags."""
    xml = generate_label_config(CLASS_NAMES)
    assert "fire" in xml, "Missing 'fire' in label config"
    assert "smoke" in xml, "Missing 'smoke' in label config"
    assert "<Image" in xml, "Missing <Image tag"
    assert "<RectangleLabels" in xml, "Missing <RectangleLabels tag"
    print(f"    XML config generated ({len(xml)} chars)")


def test_generate_label_config_many_classes():
    """Config with 25 classes should include all class names."""
    many_classes = {i: f"class_{i}" for i in range(25)}
    xml = generate_label_config(many_classes)
    for i in range(25):
        assert f"class_{i}" in xml, f"Missing class_{i} in label config"
    print(f"    all 25 class names present in XML config")


# --- A21-A22: _image_path_to_ls_url ---


def test_image_path_to_ls_url_basic():
    """Image under dataset_base should produce a valid local-files URL."""
    image_path = FIXTURES_DIR / "val" / "images" / "test.jpg"
    url = _image_path_to_ls_url(image_path, local_files_root="/datasets", dataset_base=FIXTURES_DIR)
    # Code no longer prefixes with local_files_root — LS prepends DOCUMENT_ROOT itself.
    expected_prefix = f"/data/local-files/?d={FIXTURES_DIR.name}/"
    assert url.startswith(expected_prefix), (
        f"URL should start with {expected_prefix}, got {url}"
    )
    assert "test.jpg" in url, f"URL should contain test.jpg, got {url}"
    print(f"    url={url}")


def test_image_path_to_ls_url_outside_base():
    """Image outside dataset_base should fall back to filename."""
    image_path = Path("/tmp/other/test.jpg")
    url = _image_path_to_ls_url(image_path, local_files_root="/datasets", dataset_base=FIXTURES_DIR)
    assert url.startswith("/data/local-files/?d="), f"URL should start with /data/local-files/?d=, got {url}"
    assert "test.jpg" in url, f"URL should contain test.jpg, got {url}"
    print(f"    fallback url={url}")


# --- A23-A24: Path helpers ---


def test_label_path_for_image():
    """Should replace images/ with labels/ and .jpg with .txt."""
    result = _label_path_for_image(Path("train/images/foo.jpg"))
    assert result == Path("train/labels/foo.txt"), f"Expected train/labels/foo.txt, got {result}"


def test_ls_url_to_label_path():
    """Should extract filename from LS URL and place in output_dir."""
    url = "/data/local-files/?d=dataset/train/images/foo.jpg"
    result = _ls_url_to_label_path(url, output_dir=Path("/tmp/out"))
    assert result == Path("/tmp/out/foo.txt"), f"Expected /tmp/out/foo.txt, got {result}"


# --- A25: gather_dataset_pairs ---


def test_gather_dataset_pairs():
    """Should gather pairs from train and val splits of fixture dataset.

    Note: bridge.py uses its own _PROJECT_ROOT (parent.parent from bridge.py,
    which resolves to ai/tools/) to resolve paths. We build a config dict
    manually with absolute paths to the fixture data to avoid this issue.
    """
    # Build a config with absolute paths pointing to fixture data
    config = {
        "path": str(FIXTURES_DIR),
        "train": "train/images",
        "val": "val/images",
    }
    pairs = gather_dataset_pairs(config, ["train", "val"])
    assert isinstance(pairs, list), f"Expected list, got {type(pairs)}"
    assert len(pairs) >= 10 + 5, (
        f"Expected at least 15 pairs (10 train + 5 val), got {len(pairs)}"
    )
    for img, lbl, split in pairs:
        assert isinstance(img, Path), f"Image should be Path, got {type(img)}"
        assert isinstance(lbl, Path), f"Label should be Path, got {type(lbl)}"
        assert split in ("train", "val"), f"Unexpected split tag: {split}"
    print(f"    gathered {len(pairs)} pairs from train+val")


# --- A26-A27: gather_auto_annotate_pairs ---


def test_gather_auto_annotate_pairs():
    """Should find pairs from symlinked images/ and labels/ directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "images").symlink_to(FIXTURES_DIR / "val" / "images")
        (tmpdir / "labels").symlink_to(FIXTURES_DIR / "val" / "labels")
        pairs = gather_auto_annotate_pairs(tmpdir)
        assert len(pairs) >= 3, f"Expected at least 3 pairs (5 val images), got {len(pairs)}"
        for img, lbl in pairs:
            assert isinstance(img, Path) and isinstance(lbl, Path)
        print(f"    gathered {len(pairs)} auto-annotate pairs")


def test_gather_auto_annotate_pairs_no_labels():
    """Directory without labels/ should return empty list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pairs = gather_auto_annotate_pairs(Path(tmpdir))
        assert pairs == [], f"Expected [], got {pairs}"


# --- A28-A30: gather_qa_fixes_pairs ---


def test_gather_qa_fixes_pairs_valid():
    """Valid fixes.json should produce correct pairs."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(
            {
                "fixes": [
                    {"image_path": "/a/img1.jpg", "label_path": "/a/img1.txt"},
                    {"image_path": "/a/img2.jpg", "label_path": "/a/img2.txt"},
                ]
            },
            f,
        )
        tmp_path = Path(f.name)
    try:
        pairs = gather_qa_fixes_pairs(tmp_path)
        assert len(pairs) == 2, f"Expected 2 pairs, got {len(pairs)}"
        assert pairs[0] == (Path("/a/img1.jpg"), Path("/a/img1.txt"))
        assert pairs[1] == (Path("/a/img2.jpg"), Path("/a/img2.txt"))
        print(f"    got {len(pairs)} QA fix pairs")
    finally:
        tmp_path.unlink()


def test_gather_qa_fixes_pairs_duplicates():
    """Duplicate image_paths should be deduplicated."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(
            {
                "fixes": [
                    {"image_path": "/a/img1.jpg", "label_path": "/a/img1.txt"},
                    {"image_path": "/a/img1.jpg", "label_path": "/a/img1_v2.txt"},
                ]
            },
            f,
        )
        tmp_path = Path(f.name)
    try:
        pairs = gather_qa_fixes_pairs(tmp_path)
        assert len(pairs) == 1, f"Expected 1 pair (deduplicated), got {len(pairs)}"
        print(f"    deduplicated to {len(pairs)} pair")
    finally:
        tmp_path.unlink()


def test_gather_qa_fixes_pairs_missing_file():
    """Non-existent fixes.json should return empty list."""
    pairs = gather_qa_fixes_pairs(Path("/nonexistent/fixes.json"))
    assert pairs == [], f"Expected [], got {pairs}"


# --- A31-A33: build_task ---


def test_build_task_basic():
    """Build task from a real fixture image+label pair."""
    label_dir = FIXTURES_DIR / "val" / "labels"
    image_dir = FIXTURES_DIR / "val" / "images"
    label_files = sorted(label_dir.glob("*.txt"))
    assert len(label_files) > 0, "No label files found"
    label_path = label_files[0]
    # Find matching image
    img_path = image_dir / (label_path.stem + ".jpg")
    assert img_path.exists(), f"Image not found: {img_path}"

    task = build_task(
        image_path=img_path,
        label_path=label_path,
        class_names=CLASS_NAMES,
        local_files_root="/datasets",
        dataset_base=FIXTURES_DIR,
    )

    assert isinstance(task, dict), f"Expected dict, got {type(task)}"
    assert "data" in task, "Missing 'data' key in task"
    assert "image" in task["data"], "Missing 'image' in task['data']"
    assert task["data"]["image"].startswith("/data/local-files/"), (
        f"Image URL should be local-files URL, got {task['data']['image']}"
    )
    assert "predictions" in task, "Missing 'predictions' key in task"
    assert len(task["predictions"]) > 0, "predictions list should not be empty"
    assert len(task["predictions"][0]["result"]) > 0, "result list should not be empty"
    print(f"    task built: image={task['data']['image']}, bboxes={len(task['predictions'][0]['result'])}")


def test_build_task_no_labels():
    """Task with non-existent label should not have predictions."""
    img_path = FIXTURES_DIR / "val" / "images" / "nonexistent.jpg"
    # Create a dummy image so we can reference it
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp_img = Path(f.name)
    try:
        task = build_task(
            image_path=tmp_img,
            label_path=Path("/nonexistent/label.txt"),
            class_names=CLASS_NAMES,
            local_files_root="/datasets",
            dataset_base=FIXTURES_DIR,
        )
        # Should not have predictions key or should have empty result list
        has_preds = "predictions" in task
        if has_preds:
            assert len(task["predictions"]) == 0 or len(task["predictions"][0].get("result", [])) == 0, (
                "Expected no predictions for missing label file"
            )
        print(f"    no predictions for missing label (has_predictions={has_preds})")
    finally:
        tmp_img.unlink()


def test_build_task_multiple_bboxes():
    """Image with multiple annotations should produce multiple bbox results."""
    label_dir = FIXTURES_DIR / "val" / "labels"
    image_dir = FIXTURES_DIR / "val" / "images"
    # Find a label file with > 1 line
    multi_bbox_label = None
    for lbl in sorted(label_dir.glob("*.txt")):
        lines = lbl.read_text().strip().splitlines()
        if len(lines) > 1:
            multi_bbox_label = lbl
            break
    assert multi_bbox_label is not None, "No label file with >1 annotation found"

    img_path = image_dir / (multi_bbox_label.stem + ".jpg")
    assert img_path.exists(), f"Image not found: {img_path}"

    task = build_task(
        image_path=img_path,
        label_path=multi_bbox_label,
        class_names=CLASS_NAMES,
        local_files_root="/datasets",
        dataset_base=FIXTURES_DIR,
    )

    result_list = task["predictions"][0]["result"]
    assert len(result_list) > 1, (
        f"Expected >1 bbox result, got {len(result_list)} for {multi_bbox_label.name}"
    )
    print(f"    {multi_bbox_label.name}: {len(result_list)} bboxes in task")


# --- A34-A36: build_parser ---


def test_build_parser_import():
    """Parser should recognize 'import' subcommand with correct attributes."""
    parser = build_parser()
    args = parser.parse_args(["import", "--data-config", "test.yaml", "--from-auto-annotate", "/tmp/aa"])
    assert args.command == "import", f"Expected command='import', got {args.command}"
    assert args.data_config == "test.yaml", f"Expected data_config='test.yaml', got {args.data_config}"
    assert args.from_auto_annotate == "/tmp/aa", (
        f"Expected from_auto_annotate='/tmp/aa', got {args.from_auto_annotate}"
    )


def test_build_parser_export():
    """Parser should recognize 'export' subcommand with project and output-dir."""
    parser = build_parser()
    args = parser.parse_args(["export", "--project", "my_project", "--output-dir", "/tmp/out"])
    assert args.command == "export", f"Expected command='export', got {args.command}"
    assert args.project == "my_project", f"Expected project='my_project', got {args.project}"
    assert args.output_dir == "/tmp/out", f"Expected output_dir='/tmp/out', got {args.output_dir}"


def test_build_parser_setup():
    """Parser should recognize 'setup' subcommand with data-config."""
    parser = build_parser()
    args = parser.parse_args(["setup", "--data-config", "test.yaml"])
    assert args.command == "setup", f"Expected command='setup', got {args.command}"
    assert args.data_config == "test.yaml", f"Expected data_config='test.yaml', got {args.data_config}"


# ===================================================================
# Section B: Integration Tests (require LS service)
# ===================================================================


def test_cmd_import_dry_run():
    """Dry-run import should print summary without making API calls."""
    args = argparse.Namespace(
        data_config=DATA_CONFIG_PATH,
        from_auto_annotate=None,
        from_qa_fixes=None,
        ls_config=None,
        api_key="test-key",
        project=None,
        splits="train val",
        dry_run=True,
        email="",
        password="",
    )
    cmd_import(args)
    print("    dry-run import completed without error")


def test_cmd_setup_dry_run():
    """Dry-run setup should print label config without creating a project."""
    args = argparse.Namespace(
        data_config=DATA_CONFIG_PATH,
        ls_config=None,
        api_key="test-key",
        project=None,
        dry_run=True,
        email="",
        password="",
    )
    cmd_setup(args)
    print("    dry-run setup completed without error")


def _ensure_ls_session(ls_url: str, email: str, password: str) -> requests.Session:
    """Create or reuse a Label Studio account and return an authenticated session."""
    session = requests.Session()

    # Login (or signup if account doesn't exist)
    r = session.get(f"{ls_url}/user/login/", timeout=10)
    csrf = session.cookies.get("csrftoken", "")
    r = session.post(
        f"{ls_url}/user/login/",
        data={"email": email, "password": password, "csrfmiddlewaretoken": csrf},
        headers={"Referer": f"{ls_url}/user/login/"},
        timeout=10,
    )

    # If still on login page, account doesn't exist — sign up
    if "/user/login" in r.url:
        session = requests.Session()
        r = session.get(f"{ls_url}/user/signup/", timeout=10)
        csrf = session.cookies.get("csrftoken", "")
        r = session.post(
            f"{ls_url}/user/signup/",
            data={
                "email": email,
                "password": password,
                "csrfmiddlewaretoken": csrf,
            },
            headers={"Referer": f"{ls_url}/user/signup/"},
            timeout=10,
        )
        # Login after signup
        session = requests.Session()
        r = session.get(f"{ls_url}/user/login/", timeout=10)
        csrf = session.cookies.get("csrftoken", "")
        session.post(
            f"{ls_url}/user/login/",
            data={"email": email, "password": password, "csrfmiddlewaretoken": csrf},
            headers={"Referer": f"{ls_url}/user/login/"},
            timeout=10,
        )

    return session


_LS_TEST_EMAIL = "bridge_test@test.local"
_LS_TEST_PASSWORD = "BridgeTest123!"


def test_full_pipeline_roundtrip():
    """Full import -> annotate -> export round-trip with Label Studio."""
    if not HAS_LS_SERVICE:
        print("    SKIP: Label Studio service not available at localhost:18103")
        return

    project_name = "test_bridge_roundtrip"
    ls_url = "http://localhost:18103"

    # Get authenticated session (auto-creates account if needed)
    session = _ensure_ls_session(ls_url, _LS_TEST_EMAIL, _LS_TEST_PASSWORD)

    # Build API client with session-cookie auth (LS 1.23+ disabled legacy tokens)
    api = LabelStudioAPI(
        url=ls_url,
        api_key="unused",
        email=_LS_TEST_EMAIL,
        password=_LS_TEST_PASSWORD,
    )

    # Step 1: Delete existing test project if present
    existing = api.find_project(project_name)
    if existing:
        session.delete(f"{ls_url}/api/projects/{existing['id']}", timeout=10)

    # Step 2: Create project
    label_xml = generate_label_config(CLASS_NAMES)
    project = api.create_project(title=project_name, label_config=label_xml)
    project_id = project["id"]

    # Step 3: Import val fixtures (use absolute paths to avoid resolve_path issues)
    # bridge.py's _PROJECT_ROOT resolves to ai/tools/, so relative paths in
    # configs don't resolve correctly when running from ai/. Build config directly.
    val_data_config = {
        "path": str(FIXTURES_DIR),
        "val": "val/images",
    }
    pairs = gather_dataset_pairs(val_data_config, ["val"])
    tasks = []
    for img_path, lbl_path, _split in pairs:
        task = build_task(
            image_path=img_path,
            label_path=lbl_path,
            class_names=CLASS_NAMES,
            local_files_root="/datasets",
            dataset_base=FIXTURES_DIR,
            model_version="test_v1",
        )
        tasks.append(task)

    imported = api.import_tasks(project_id, tasks)
    assert imported > 0, f"Expected to import tasks, got {imported}"

    # Step 4: Simulate human review — copy predictions to annotations
    fetched_tasks = api.get_tasks(project_id)
    assert len(fetched_tasks) > 0, "No tasks fetched after import"

    for task in fetched_tasks:
        predictions = task.get("predictions", [])
        if not predictions:
            continue
        result = predictions[0].get("result", [])
        if not result:
            continue
        resp = session.post(
            f"{ls_url}/api/tasks/{task['id']}/annotations/",
            json={"result": result},
            timeout=10,
        )
        assert resp.status_code in (200, 201), f"Failed to create annotation: {resp.status_code}"

    # Step 5: Export reviewed annotations
    reviewed_tasks = api.get_tasks(project_id, only_reviewed=True)
    assert len(reviewed_tasks) > 0, "No reviewed tasks found after annotation"

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "exported"
        output_dir.mkdir()

        for task in reviewed_tasks:
            annotations_list = task.get("annotations", [])
            if not annotations_list:
                continue
            annotation = annotations_list[-1]
            results = annotation.get("result", [])

            yolo_anns = []
            for r in results:
                converted = ls_to_yolo(r, CLASS_NAME_TO_ID)
                if converted is not None:
                    yolo_anns.append(converted)

            image_url = task.get("data", {}).get("image", "")
            label_path = _ls_url_to_label_path(image_url, output_dir)
            write_yolo_labels(label_path, yolo_anns)

        # Step 6: Verify round-trip
        exported_files = list(output_dir.glob("*.txt"))
        assert len(exported_files) > 0, "No label files exported"

        match_count = 0
        for exported_file in exported_files:
            exported_anns = read_yolo_labels(exported_file)
            original_label = None
            for orig_img, orig_lbl, _split in pairs:
                if orig_img.stem == exported_file.stem:
                    original_label = orig_lbl
                    break
            if original_label is None:
                continue

            original_anns = read_yolo_labels(original_label)
            if len(original_anns) != len(exported_anns):
                print(
                    f"    MISMATCH {exported_file.name}: "
                    f"original={len(original_anns)} exported={len(exported_anns)}"
                )
                continue

            all_match = True
            for (o_cid, o_cx, o_cy, o_w, o_h), (e_cid, e_cx, e_cy, e_w, e_h) in zip(
                original_anns, exported_anns
            ):
                if o_cid != e_cid:
                    all_match = False
                    break
                for orig, exp in [(o_cx, e_cx), (o_cy, e_cy), (o_w, e_w), (o_h, e_h)]:
                    if abs(orig - exp) > 0.001:
                        all_match = False
                        break
                if not all_match:
                    break

            if all_match:
                match_count += 1
            else:
                print(f"    MISMATCH {exported_file.name}: coordinate values differ")

        assert match_count > 0, "No labels matched after round-trip"
        print(f"    Round-trip verified: {match_count}/{len(exported_files)} labels match")

    # Cleanup: delete test project
    try:
        session.delete(f"{ls_url}/api/projects/{project_id}", timeout=10)
    except Exception:
        pass


# ===================================================================
# Main runner
# ===================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Test 08: Label Studio Bridge — unit + integration tests")
    print("=" * 60)

    # Section A: Unit tests (always run)
    print("\n--- Section A: Unit Tests ---")
    run_all([
        ("yolo_to_ls_basic", test_yolo_to_ls_basic),
        ("yolo_to_ls_smoke", test_yolo_to_ls_smoke),
        ("yolo_to_ls_clamping", test_yolo_to_ls_clamping),
        ("yolo_to_ls_unknown_class", test_yolo_to_ls_unknown_class),
        ("ls_to_yolo_basic", test_ls_to_yolo_basic),
        ("ls_to_yolo_unknown_label", test_ls_to_yolo_unknown_label),
        ("ls_to_yolo_wrong_type", test_ls_to_yolo_wrong_type),
        ("ls_to_yolo_empty_labels", test_ls_to_yolo_empty_labels),
        ("yolo_ls_roundtrip", test_yolo_ls_roundtrip),
        ("read_yolo_labels_basic", test_read_yolo_labels_basic),
        ("read_yolo_labels_missing_file", test_read_yolo_labels_missing_file),
        ("read_yolo_labels_malformed", test_read_yolo_labels_malformed),
        ("write_yolo_labels_basic", test_write_yolo_labels_basic),
        ("write_read_roundtrip", test_write_read_roundtrip),
        ("resolve_api_key_cli_priority", test_resolve_api_key_cli_priority),
        ("resolve_api_key_missing", test_resolve_api_key_missing),
        ("load_ls_config_default", test_load_ls_config_default),
        ("load_ls_config_missing_file", test_load_ls_config_missing_file),
        ("generate_label_config_basic", test_generate_label_config_basic),
        ("generate_label_config_many_classes", test_generate_label_config_many_classes),
        ("image_path_to_ls_url_basic", test_image_path_to_ls_url_basic),
        ("image_path_to_ls_url_outside_base", test_image_path_to_ls_url_outside_base),
        ("label_path_for_image", test_label_path_for_image),
        ("ls_url_to_label_path", test_ls_url_to_label_path),
        ("gather_dataset_pairs", test_gather_dataset_pairs),
        ("gather_auto_annotate_pairs", test_gather_auto_annotate_pairs),
        ("gather_auto_annotate_pairs_no_labels", test_gather_auto_annotate_pairs_no_labels),
        ("gather_qa_fixes_pairs_valid", test_gather_qa_fixes_pairs_valid),
        ("gather_qa_fixes_pairs_duplicates", test_gather_qa_fixes_pairs_duplicates),
        ("gather_qa_fixes_pairs_missing_file", test_gather_qa_fixes_pairs_missing_file),
        ("build_task_basic", test_build_task_basic),
        ("build_task_no_labels", test_build_task_no_labels),
        ("build_task_multiple_bboxes", test_build_task_multiple_bboxes),
        ("build_parser_import", test_build_parser_import),
        ("build_parser_export", test_build_parser_export),
        ("build_parser_setup", test_build_parser_setup),
    ], title=None, exit_on_fail=False)

    # Section B: Integration tests
    print("\n--- Section B: Integration Tests (requires LS service) ---")
    run_all([
        ("cmd_import_dry_run", test_cmd_import_dry_run),
        ("cmd_setup_dry_run", test_cmd_setup_dry_run),
        ("full_pipeline_roundtrip", test_full_pipeline_roundtrip),
    ], title=None, exit_on_fail=False)

    # Summary
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    if errors:
        print("  Failures:")
        for name, err in errors:
            print(f"    - {name}: {err}")
    print(f"{'=' * 60}")
    sys.exit(1 if failed > 0 else 0)

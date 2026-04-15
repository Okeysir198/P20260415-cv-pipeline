"""Test 07: Scaffold — verify create_experiment.py generates correct workspace."""

import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import run_all
from utils.yolo_io import parse_classes
from utils.scaffold import (
    build_05_data_yaml,
    build_06_training_yaml,
    build_evaluate_py,
    build_export_py,
    build_inference_py,
    build_train_py,
    write_file,
    PRETRAINED_WEIGHTS,
    YOLOX_PARAMS,
)

OUTPUTS = Path(__file__).resolve().parent / "outputs" / "07_scaffold"
OUTPUTS.mkdir(parents=True, exist_ok=True)


def test_parse_classes():
    """Verify parse_classes parses comma-separated ID:name pairs correctly."""
    result = parse_classes("0:car,1:truck,2:bus")
    assert result == {0: "car", 1: "truck", 2: "bus"}, f"Unexpected: {result}"
    print(f"    Parsed: {result}")

    # Single class
    single = parse_classes("0:person")
    assert single == {0: "person"}, f"Unexpected single: {single}"

    # Whitespace tolerance
    spaced = parse_classes(" 0 : fire , 1 : smoke ")
    assert spaced == {0: "fire", 1: "smoke"}, f"Unexpected spaced: {spaced}"
    print("    Whitespace-tolerant parsing OK")


def test_parse_classes_invalid():
    """Verify parse_classes raises ValueError for invalid format."""
    try:
        parse_classes("car,truck")
        raise AssertionError("Should have raised ValueError for missing ':'")
    except ValueError:
        pass
    print("    Invalid format raises ValueError as expected")


def test_scaffold_creates_configs():
    """Run scaffold in dry-run mode, verify config content is valid YAML."""
    usecase = "test_vehicle_detection"
    classes = parse_classes("0:car,1:truck,2:bus")

    data_yaml_str = build_05_data_yaml(usecase, classes)
    training_yaml_str = build_06_training_yaml(usecase, "yolox-m", classes)

    # Parse as YAML to verify validity
    data_cfg = yaml.safe_load(data_yaml_str)
    training_cfg = yaml.safe_load(training_yaml_str)

    # Validate data config structure
    assert data_cfg["dataset_name"] == usecase
    assert data_cfg["num_classes"] == 3, f"Expected 3, got {data_cfg['num_classes']}"
    assert data_cfg["names"] == {0: "car", 1: "truck", 2: "bus"}
    assert data_cfg["train"] == "train/images"
    assert data_cfg["val"] == "val/images"
    assert data_cfg["test"] == "test/images"
    assert "input_size" in data_cfg
    assert "mean" in data_cfg
    assert "std" in data_cfg
    print(f"    Data config: {data_cfg['num_classes']} classes")

    # Validate training config structure
    assert training_cfg["model"]["arch"] == "yolox-m"
    assert training_cfg["model"]["num_classes"] == 3
    assert training_cfg["model"]["depth"] == 0.67
    assert training_cfg["model"]["width"] == 0.75
    assert training_cfg["training"]["epochs"] == 200
    assert training_cfg["training"]["optimizer"] == "sgd"
    assert training_cfg["loss"]["type"] == "yolox"
    print(f"    Training config: arch={training_cfg['model']['arch']}, "
          f"epochs={training_cfg['training']['epochs']}")


def test_scaffold_dfine_config():
    """Verify scaffold generates correct config for D-FINE model (non-YOLOX)."""
    usecase = "test_fire"
    classes = parse_classes("0:fire,1:smoke")

    training_yaml_str = build_06_training_yaml(usecase, "dfine-s", classes)
    training_cfg = yaml.safe_load(training_yaml_str)

    assert training_cfg["model"]["arch"] == "dfine-s"
    assert training_cfg["model"]["num_classes"] == 2
    assert training_cfg["model"]["pretrained"] is True  # "true" in YAML = bool True
    assert training_cfg["loss"]["type"] == "detr-passthrough"
    # D-FINE should NOT have depth/width
    assert "depth" not in training_cfg["model"]
    assert "width" not in training_cfg["model"]
    print(f"    D-FINE config: arch={training_cfg['model']['arch']}, "
          f"loss={training_cfg['loss']['type']}")


def test_scaffold_experiment_scripts():
    """Verify generated experiment scripts have correct structure."""
    usecase = "test_scaffold_exp"

    train_py = build_train_py(usecase)
    evaluate_py = build_evaluate_py(usecase)
    export_py = build_export_py(usecase)
    inference_py = build_inference_py(usecase)

    for script_name, content in [
        ("train.py", train_py),
        ("evaluate.py", evaluate_py),
        ("export.py", export_py),
        ("inference.py", inference_py),
    ]:
        # Verify shebang
        assert content.startswith("#!/usr/bin/env python3"), (
            f"{script_name} missing shebang"
        )
        # Verify sys.path setup
        assert "sys.path.insert(0," in content, (
            f"{script_name} missing sys.path setup"
        )
        # Verify main guard
        assert 'if __name__ == "__main__":' in content, (
            f"{script_name} missing __main__ guard"
        )
        # Verify it references the correct usecase
        assert usecase in content, (
            f"{script_name} does not reference usecase '{usecase}'"
        )
        print(f"    {script_name}: structure OK")

    # Train script should import DetectionTrainer
    assert "DetectionTrainer" in train_py
    # Evaluate script should import ModelEvaluator
    assert "ModelEvaluator" in evaluate_py
    # Export script should import ModelExporter
    assert "ModelExporter" in export_py
    # Inference script should import DetectionPredictor
    assert "DetectionPredictor" in inference_py
    print("    All scripts import correct pipeline modules")


def test_scaffold_write_dry_run():
    """Verify write_file in dry-run mode does not create files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "subdir" / "test.yaml"

        result = write_file(target, "content: true\n", dry_run=True, force=False)
        assert result is True, "dry-run should return True"
        assert not target.exists(), "dry-run should not create file"
        print(f"    Dry-run: no file created at {target}")


def test_scaffold_write_real():
    """Verify write_file creates files with correct content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "configs" / "test" / "05_data.yaml"
        content = "dataset_name: test\nnum_classes: 2\n"

        result = write_file(target, content, dry_run=False, force=False)
        assert result is True, "Should return True for new file"
        assert target.exists(), "File should be created"
        assert target.read_text() == content, "File content mismatch"
        print(f"    Created: {target}")

        # Writing again without force should skip
        result2 = write_file(target, "new content\n", dry_run=False, force=False)
        assert result2 is False, "Should return False for existing file without force"
        assert target.read_text() == content, "Content should not change without force"
        print("    Skip existing (no force): OK")

        # Writing with force should overwrite
        new_content = "overwritten: true\n"
        result3 = write_file(target, new_content, dry_run=False, force=True)
        assert result3 is True, "Should return True with force"
        assert target.read_text() == new_content, "Content should be overwritten with force"
        print("    Force overwrite: OK")


def test_scaffold_full_workspace():
    """Scaffold a complete workspace into a temp directory and verify all files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        usecase = "smoke_detection"
        model = "yolox-m"
        classes = parse_classes("0:fire,1:smoke")

        files = {
            root / "configs" / usecase / "05_data.yaml": build_05_data_yaml(usecase, classes),
            root / "configs" / usecase / "06_training.yaml": build_06_training_yaml(
                usecase, model, classes),
            root / "experiments" / usecase / "train.py": build_train_py(usecase),
            root / "experiments" / usecase / "evaluate.py": build_evaluate_py(usecase),
            root / "experiments" / usecase / "export.py": build_export_py(usecase),
            root / "experiments" / usecase / "inference.py": build_inference_py(usecase),
        }

        for path, content in files.items():
            write_file(path, content, dry_run=False, force=False)

        # Verify all 6 files exist
        for path in files:
            assert path.exists(), f"Missing: {path}"

        # Verify configs are valid YAML
        data_cfg = yaml.safe_load((root / "configs" / usecase / "05_data.yaml").read_text())
        train_cfg = yaml.safe_load((root / "configs" / usecase / "06_training.yaml").read_text())
        assert data_cfg["num_classes"] == 2
        assert train_cfg["model"]["arch"] == "yolox-m"

        print(f"    Full workspace: 6 files created in {root}")
        print(f"    configs/{usecase}/05_data.yaml: {data_cfg['num_classes']} classes")
        print(f"    configs/{usecase}/06_training.yaml: arch={train_cfg['model']['arch']}")


def test_pretrained_weights_mapping():
    """Verify PRETRAINED_WEIGHTS covers all supported architectures."""
    expected_archs = ["yolox-m", "yolox-tiny", "dfine-s", "dfine-n", "dfine-m", "rtdetrv2-r18"]
    for arch in expected_archs:
        assert arch in PRETRAINED_WEIGHTS, f"Missing pretrained mapping for {arch}"

    # YOLOX should point to .pth files
    assert PRETRAINED_WEIGHTS["yolox-m"].endswith(".pth")
    # HF models should use "true" (auto-download)
    assert PRETRAINED_WEIGHTS["dfine-s"] == "true"
    print(f"    Pretrained weights mapped for {len(PRETRAINED_WEIGHTS)} architectures")


def test_yolox_params():
    """Verify YOLOX_PARAMS has correct depth/width for known variants."""
    assert YOLOX_PARAMS["yolox-m"]["depth"] == 0.67
    assert YOLOX_PARAMS["yolox-m"]["width"] == 0.75
    assert YOLOX_PARAMS["yolox-tiny"]["depth"] == 0.33
    assert YOLOX_PARAMS["yolox-tiny"]["width"] == 0.375
    print(f"    YOLOX params: M=({YOLOX_PARAMS['yolox-m']['depth']}, "
          f"{YOLOX_PARAMS['yolox-m']['width']}), "
          f"Tiny=({YOLOX_PARAMS['yolox-tiny']['depth']}, "
          f"{YOLOX_PARAMS['yolox-tiny']['width']})")


if __name__ == "__main__":
    run_all([
        ("parse_classes", test_parse_classes),
        ("parse_classes_invalid", test_parse_classes_invalid),
        ("scaffold_creates_configs", test_scaffold_creates_configs),
        ("scaffold_dfine_config", test_scaffold_dfine_config),
        ("scaffold_experiment_scripts", test_scaffold_experiment_scripts),
        ("scaffold_write_dry_run", test_scaffold_write_dry_run),
        ("scaffold_write_real", test_scaffold_write_real),
        ("scaffold_full_workspace", test_scaffold_full_workspace),
        ("pretrained_weights_mapping", test_pretrained_weights_mapping),
        ("yolox_params", test_yolox_params),
    ], title="Test 08 (tools): Scaffold")

"""Test 08: Utils — release pipeline helpers (real filesystem, no mocks)."""

import json
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import run_all
from utils.release import _detect_use_case, _load_metrics, _next_version, release


def test_next_version_empty_dir():
    with tempfile.TemporaryDirectory() as td:
        assert _next_version(Path(td), "fire") == 1


def test_next_version_picks_max_plus_one():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "fire"
        (base / "v1_2026-01-01").mkdir(parents=True)
        (base / "v3_2026-02-02").mkdir(parents=True)
        (base / "v2_2026-01-15").mkdir(parents=True)
        # non-version dir should be ignored
        (base / "notes").mkdir()
        assert _next_version(Path(td), "fire") == 4


def test_next_version_ignores_malformed():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "fire"
        (base / "vABC_bad").mkdir(parents=True)
        (base / "v_2026").mkdir(parents=True)  # no digits
        assert _next_version(Path(td), "fire") == 1


def test_detect_use_case_from_data_config():
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        (run_dir / "05_data.yaml").write_text(
            yaml.safe_dump({"dataset_name": "helmet_detection", "num_classes": 1})
        )
        assert _detect_use_case(run_dir) == "helmet_detection"


def test_detect_use_case_fallback_parent_name():
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td) / "phone_detection" / "2026-04-01_1200"
        run_dir.mkdir(parents=True)
        assert _detect_use_case(run_dir) == "phone_detection"


def test_load_metrics_from_metrics_json():
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        payload = {"mAP50": 0.812, "mAP": 0.5}
        (run_dir / "metrics.json").write_text(json.dumps(payload))
        assert _load_metrics(run_dir) == payload


def test_load_metrics_missing_returns_empty():
    with tempfile.TemporaryDirectory() as td:
        assert _load_metrics(Path(td)) == {}


def test_release_creates_versioned_dir_and_model_card():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        run_dir = root / "runs" / "fire_det" / "2026-04-01_1200"
        run_dir.mkdir(parents=True)
        # Minimal run artefacts
        (run_dir / "best.pth").write_bytes(b"\x00\x01weights")
        (run_dir / "05_data.yaml").write_text(
            yaml.safe_dump({"dataset_name": "fire_det", "num_classes": 2})
        )
        (run_dir / "06_training.yaml").write_text(
            yaml.safe_dump({
                "model": {"arch": "yolox-m", "num_classes": 2, "input_size": [640, 640]},
                "data": {"dataset_config": "features/fire_det/configs/05_data.yaml"},
            })
        )
        (run_dir / "metrics.json").write_text(json.dumps({"mAP50": 0.77}))

        releases = root / "releases"
        rdir = release(run_dir=run_dir, releases_dir=releases, notes="unit test")

        assert rdir.exists()
        assert rdir.parent.name == "fire_det"
        assert rdir.name.startswith("v1_")
        assert (rdir / "best.pth").exists()
        assert (rdir / "05_data.yaml").exists()
        assert (rdir / "06_training.yaml").exists()

        card = yaml.safe_load((rdir / "model_card.yaml").read_text())
        assert card["model"]["use_case"] == "fire_det"
        assert card["model"]["version"] == 1
        assert card["model"]["notes"] == "unit test"
        assert card["metrics"]["mAP50"] == 0.77
        assert card["architecture"]["arch"] == "yolox-m"
        assert card["architecture"]["num_classes"] == 2

        # Second release → v2
        rdir2 = release(run_dir=run_dir, releases_dir=releases)
        assert rdir2.name.startswith("v2_")


if __name__ == "__main__":
    run_all([
        ("next_version_empty", test_next_version_empty_dir),
        ("next_version_max", test_next_version_picks_max_plus_one),
        ("next_version_malformed", test_next_version_ignores_malformed),
        ("detect_use_case_from_config", test_detect_use_case_from_data_config),
        ("detect_use_case_fallback", test_detect_use_case_fallback_parent_name),
        ("load_metrics_json", test_load_metrics_from_metrics_json),
        ("load_metrics_missing", test_load_metrics_missing_returns_empty),
        ("release_e2e", test_release_creates_versioned_dir_and_model_card),
    ], title="Test utils08: release")

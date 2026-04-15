"""Test 14: Generative Augment — module imports, Inpainter init, graph build, service tests."""

import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import passed, failed, errors, run_test, run_all
from core.p03_generative_aug.graph import build_graph
from core.p03_generative_aug.inpainter import Inpainter
import core.p03_generative_aug.nodes as nodes
from core.p03_generative_aug.nodes import scan_node
from utils.config import load_config

OUTPUTS = Path(__file__).resolve().parent / "outputs" / "14_generative_augment"
OUTPUTS.mkdir(parents=True, exist_ok=True)

TRAIN_CONFIG_PATH = str(ROOT / "configs" / "_test" / "06_training.yaml")


# ===========================================================================
# Section 1: Client-side Inpainter tests (no services needed)
# ===========================================================================

def test_modules_importable():
    """Verify core generative augment modules can be imported."""
    assert Inpainter is not None, "Inpainter import is None"
    assert nodes is not None, "nodes import is None"
    print(f"    Inpainter and nodes imported successfully")


def test_inpainter_init():
    """Create Inpainter in service mode (no GPU model loading)."""
    config = {
        "inpainting": {
            "mode": "service",
            "service_url": "http://localhost:8002",
        }
    }
    inpainter = Inpainter(config)

    assert inpainter._mode == "service", (
        f"Expected _mode='service', got '{inpainter._mode}'"
    )
    assert inpainter._service_url == "http://localhost:8002", (
        f"Expected service_url='http://localhost:8002', got '{inpainter._service_url}'"
    )
    assert inpainter._model_id == "black-forest-labs/FLUX.2-klein-4B", (
        f"Expected default model='black-forest-labs/FLUX.2-klein-4B', got '{inpainter._model_id}'"
    )
    assert inpainter._torch_dtype_str == "bfloat16", (
        f"Expected torch_dtype='bfloat16', got '{inpainter._torch_dtype_str}'"
    )
    assert hasattr(inpainter, "_default_steps"), (
        "Inpainter missing '_default_steps' attribute"
    )
    assert inpainter._default_steps == 4, (
        f"Expected _default_steps=4, got {inpainter._default_steps}"
    )
    assert inpainter._default_guidance_scale == 3.5, (
        f"Expected _default_guidance_scale=3.5, got {inpainter._default_guidance_scale}"
    )
    print(f"    Inpainter created: mode='{inpainter._mode}', model='{inpainter._model_id}', "
          f"steps={inpainter._default_steps}, guidance={inpainter._default_guidance_scale}")


def test_inpainter_mask_composite():
    """Verify _mask_composite blends images correctly."""
    # White original, black edited, half-mask
    original = PILImage.new("RGB", (64, 64), (255, 255, 255))  # type: ignore[arg-type]  # white
    edited = PILImage.new("RGB", (64, 64), (0, 0, 0))  # type: ignore[arg-type]  # black

    # Left half = white mask (use edited), right half = black (keep original)
    mask_arr = np.zeros((64, 64), dtype=np.uint8)
    mask_arr[:, :32] = 255
    mask_pil = PILImage.fromarray(mask_arr, mode="L")

    result = Inpainter._mask_composite(original, edited, mask_pil)
    result_arr = np.array(result)

    # Left half should be black (from edited)
    assert np.mean(result_arr[:, :32, :]) < 10, "Left half should be ~black (from edited)"
    # Right half should be white (from original)
    assert np.mean(result_arr[:, 32:, :]) > 245, "Right half should be ~white (from original)"
    print(f"    _mask_composite blended correctly: left={np.mean(result_arr[:, :32, :]):.0f}, "
          f"right={np.mean(result_arr[:, 32:, :]):.0f}")


def test_graph_build():
    """Build the generative augment LangGraph, verify it has invoke method."""
    graph = build_graph()

    assert graph is not None, "build_graph() returned None"
    assert hasattr(graph, "invoke"), (
        f"Graph object missing 'invoke' method, has: {dir(graph)}"
    )
    print(f"    Graph built: {type(graph).__name__} with invoke method")


def test_inpainter_local_init():
    """Create Inpainter in 'local' mode, verify _mode='local' and _pipeline is None."""
    config = {
        "inpainting": {
            "mode": "local",
            "model": "black-forest-labs/FLUX.2-klein-4B",
        }
    }
    inpainter = Inpainter(config)

    assert inpainter._mode == "local", f"Expected _mode='local', got '{inpainter._mode}'"
    assert inpainter._pipeline is None, (
        f"Expected _pipeline=None (lazy loading), got {type(inpainter._pipeline)}"
    )
    print(f"    Inpainter local mode: _mode='{inpainter._mode}', _pipeline={inpainter._pipeline}")


def test_inpainter_unload():
    """Create Inpainter, call unload(), verify _pipeline is still None (no crash)."""
    config = {
        "inpainting": {
            "mode": "local",
            "model": "black-forest-labs/FLUX.2-klein-4B",
        }
    }
    inpainter = Inpainter(config)
    assert inpainter._pipeline is None, "Pipeline should be None before unload"

    # unload() should not crash even when pipeline was never loaded
    inpainter.unload()
    assert inpainter._pipeline is None, "Pipeline should still be None after unload"
    print(f"    unload() succeeded without crash, _pipeline={inpainter._pipeline}")


def test_inpainter_config_from_yaml():
    """Load the default generative augment config and verify Inpainter picks up YAML values."""
    config = load_config(str(ROOT / "configs" / "_shared" / "03_generative_augment.yaml"))
    inpainter = Inpainter(config)

    assert inpainter._model_id == "black-forest-labs/FLUX.2-klein-4B", (
        f"Config model mismatch: {inpainter._model_id}"
    )
    assert inpainter._torch_dtype_str == "bfloat16", (
        f"Config dtype mismatch: {inpainter._torch_dtype_str}"
    )
    assert inpainter._service_url == "http://localhost:8002", (
        f"Config service_url mismatch: {inpainter._service_url}"
    )
    assert inpainter._default_guidance_scale == 3.5, (
        f"Config guidance_scale mismatch: {inpainter._default_guidance_scale}"
    )
    assert inpainter._default_steps == 4, (
        f"Config steps mismatch: {inpainter._default_steps}"
    )
    print(f"    Config loaded correctly: model={inpainter._model_id}, "
          f"steps={inpainter._default_steps}, guidance={inpainter._default_guidance_scale}")


def test_scan_node_with_real_data():
    """Call scan_node() with test_fire_100 dataset, source_class_id=0 (fire)."""
    train_config = load_config(TRAIN_CONFIG_PATH)

    # Load the data config referenced by the training config
    train_config_dir = Path(TRAIN_CONFIG_PATH).parent
    data_config_rel = train_config["data"]["dataset_config"]
    data_config_path = (train_config_dir / data_config_rel).resolve()
    data_config = load_config(str(data_config_path))

    class_names = {int(k): v for k, v in data_config["names"].items()}

    # Build the augment config with minimal settings
    augment_config = {
        "generative_augment": {
            "output_dir": str(OUTPUTS / "scan_test"),
            "splits": ["train"],
        },
        "processing": {
            "batch_size": 16,
        },
    }

    # Construct state dict matching GenAugmentState TypedDict
    state = {
        "data_config": data_config,
        "augment_config": augment_config,
        "dataset_name": data_config["dataset_name"],
        "class_names": class_names,
        "config_dir": str(data_config_path.parent),
        "source_class_id": 0,  # fire class
        "target_class_id": 1,
    }

    result = scan_node(state)

    assert "image_paths" in result, f"scan_node missing 'image_paths', got keys: {list(result.keys())}"
    assert "total_images" in result, f"scan_node missing 'total_images'"
    assert "total_batches" in result, f"scan_node missing 'total_batches'"
    assert "current_batch_idx" in result, f"scan_node missing 'current_batch_idx'"
    assert result["current_batch_idx"] == 0, f"current_batch_idx should be 0, got {result['current_batch_idx']}"

    image_paths = result["image_paths"]
    assert isinstance(image_paths, dict), f"Expected dict, got {type(image_paths)}"

    total = result["total_images"]
    assert total > 0, f"Expected at least 1 image with fire class, got {total}"

    # Verify paths are real files
    for _, paths in image_paths.items():
        for p in paths[:3]:  # Check first few
            assert Path(p).exists(), f"Image path does not exist: {p}"

    print(f"    scan_node found {total} images with source_class_id=0 across splits: "
          f"{ {s: len(p) for s, p in image_paths.items()} }")



# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    run_all([
        ("modules_importable", test_modules_importable),
        ("inpainter_init", test_inpainter_init),
        ("inpainter_mask_composite", test_inpainter_mask_composite),
        ("graph_build", test_graph_build),
        ("inpainter_local_init", test_inpainter_local_init),
        ("inpainter_unload", test_inpainter_unload),
        ("inpainter_config_from_yaml", test_inpainter_config_from_yaml),
        ("scan_node_with_real_data", test_scan_node_with_real_data),
    ], title="Test 14: Generative Augment", header_char="=" * 70)

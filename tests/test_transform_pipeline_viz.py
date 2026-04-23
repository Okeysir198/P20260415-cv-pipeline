"""Test: 04_normalize_check.png renderer + callback.

Primary invariant: the denormalize step is an exact inverse of
``v2.Normalize + ToDtype(scale=True)`` (within L∞ ≤ 2 due to uint8 rounding).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import run_all  # noqa: E402

from core.p05_data.transform_pipeline_viz import (  # noqa: E402
    denormalize_chw,
    render_normalize_check,
)

OUTPUTS = Path(__file__).resolve().parent / "outputs" / "transform_pipeline_viz"
OUTPUTS.mkdir(parents=True, exist_ok=True)

_FIXTURE_DATASET = Path(__file__).resolve().parent / "fixtures" / "data"


def _minimal_configs() -> tuple[dict, dict]:
    """Data/training configs pointing at the fixture fire dataset."""
    data_config = {
        "dataset_name": "fixture_fire",
        "path": str(_FIXTURE_DATASET),
        "train": "train/images",
        "val": "val/images",
        "input_size": [64, 64],
        "num_classes": 2,
        "names": {0: "fire", 1: "smoke"},
        "mean": [0.5, 0.5, 0.5],
        "std": [0.25, 0.25, 0.25],
    }
    training_config = {
        "augmentation": {
            "library": "torchvision",
            "mosaic": False, "mixup": False, "copypaste": False,
            "fliplr": 0.0, "flipud": 0.0,
            "degrees": 0.0, "scale": [1.0, 1.0], "translate": 0.0, "shear": 0.0,
            "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
            "normalize": True,
        }
    }
    return data_config, training_config


def test_renders_png():
    data_cfg, train_cfg = _minimal_configs()
    out = OUTPUTS / "normalize_check.png"
    if out.exists():
        out.unlink()
    ret = render_normalize_check(
        out_path=out,
        data_config=data_cfg,
        training_config=train_cfg,
        base_dir=".",
        class_names={0: "fire", 1: "smoke"},
        num_samples=3,
    )
    assert ret is not None and out.exists(), "PNG was not created"
    from PIL import Image
    with Image.open(out) as im:
        w, h = im.size
    assert h > 500, f"PNG too short: {h} px"
    assert w > 500, f"PNG too narrow: {w} px"
    print(f"    normalize_check.png: {w}x{h} px")


def test_normalize_denormalize_roundtrip():
    """denormalize_chw must invert v2.Normalize+ToDtype within L∞ ≤ 2."""
    import torchvision.transforms.v2 as v2

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    rng = np.random.default_rng(0)
    img_hwc = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

    normalize = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])
    normalized = normalize(img_hwc)
    restored = denormalize_chw(normalized, mean, std)
    l_inf = int(np.abs(img_hwc.astype(np.int32) - restored.astype(np.int32)).max())
    assert l_inf <= 2, f"normalize/denormalize round-trip L∞ too large: {l_inf}"
    print(f"    denormalize round-trip L∞: {l_inf} (≤ 2 ok)")


def test_callback_import():
    """NormalizeCheckCallback imports and instantiates cleanly."""
    from core.p06_training.callbacks_viz import NormalizeCheckCallback
    data_cfg, train_cfg = _minimal_configs()
    cb = NormalizeCheckCallback(
        save_dir=str(OUTPUTS),
        data_config=data_cfg,
        training_config=train_cfg,
        base_dir=".",
        class_names={0: "fire", 1: "smoke"},
    )
    assert cb.save_dir == OUTPUTS
    print("    NormalizeCheckCallback instantiates ok")


if __name__ == "__main__":
    run_all(
        [
            ("renders_png", test_renders_png),
            ("normalize_denormalize_roundtrip", test_normalize_denormalize_roundtrip),
            ("callback_import", test_callback_import),
        ],
        title="normalize_check_viz",
    )

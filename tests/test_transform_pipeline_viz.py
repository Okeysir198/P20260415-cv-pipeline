"""Test: 05_transform_pipeline.png renderer + callback.

Primary invariant: the denormalize step is an exact inverse of
``v2.Normalize + ToDtype(scale=True)`` (within L∞ ≤ 2 due to uint8 rounding).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import run_all  # noqa: E402

from core.p05_data.transform_pipeline_viz import render_transform_pipeline  # noqa: E402

OUTPUTS = Path(__file__).resolve().parent / "outputs" / "transform_pipeline_viz"
OUTPUTS.mkdir(parents=True, exist_ok=True)


class _SyntheticDataset:
    """Tiny in-memory dataset — 4 x 32x32 BGR uint8 images + known boxes."""

    def __init__(self) -> None:
        rng = np.random.default_rng(0)
        self._items = []
        for _ in range(4):
            img = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
            targets = np.array(
                [[0, 0.5, 0.5, 0.4, 0.4]], dtype=np.float32
            )  # one center box
            self._items.append({"image": img, "targets": targets})

    def __len__(self) -> int:
        return len(self._items)

    def get_raw_item(self, idx: int) -> dict:
        it = self._items[idx]
        return {"image": it["image"].copy(), "targets": it["targets"].copy()}


def _minimal_configs() -> tuple[dict, dict]:
    data_config = {
        "dataset_name": "synthetic",
        "input_size": [64, 64],
        "num_classes": 1,
        "names": {0: "thing"},
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


def test_png_renders_and_is_nonempty():
    data_cfg, train_cfg = _minimal_configs()
    ds = _SyntheticDataset()
    out = OUTPUTS / "render_smoke.png"
    if out.exists():
        out.unlink()
    render_transform_pipeline(
        out_path=out,
        dataset=ds,
        data_config=data_cfg,
        training_config=train_cfg,
        base_dir=".",
        class_names={0: "thing"},
    )
    assert out.exists(), "PNG was not created"
    from PIL import Image
    with Image.open(out) as im:
        w, h = im.size
    assert h > 500, f"PNG too short: {h} px"
    assert w > 500, f"PNG too narrow: {w} px"
    print(f"    render_smoke.png: {w}x{h} px")


def test_normalize_denormalize_is_exact_inverse():
    """Pre-normalize tensor and post-denormalize must match within L∞ ≤ 2."""
    data_cfg, train_cfg = _minimal_configs()
    ds = _SyntheticDataset()
    out = OUTPUTS / "inverse_check.png"
    info = render_transform_pipeline(
        out_path=out,
        dataset=ds,
        data_config=data_cfg,
        training_config=train_cfg,
        base_dir=".",
        class_names={0: "thing"},
        _return_snapshots=True,
    )
    assert info is not None, "renderer did not return snapshots"
    l_inf = info["denorm_l_inf"]
    assert l_inf is not None, "pre/post-normalize tensors were not captured"
    assert l_inf <= 2, f"normalize/denormalize round-trip L∞ too large: {l_inf}"
    print(f"    denormalize round-trip L∞: {l_inf} (≤ 2 ok)")


def test_callback_constructs():
    """TransformPipelineCallback imports and instantiates cleanly."""
    from core.p06_training.callbacks_viz import TransformPipelineCallback
    data_cfg, train_cfg = _minimal_configs()
    cb = TransformPipelineCallback(
        save_dir=str(OUTPUTS),
        data_config=data_cfg,
        training_config=train_cfg,
        base_dir=".",
        class_names={0: "thing"},
    )
    assert cb.save_dir == OUTPUTS
    print("    TransformPipelineCallback instantiates ok")


if __name__ == "__main__":
    run_all(
        [
            ("renders_and_is_nonempty", test_png_renders_and_is_nonempty),
            ("normalize_denormalize_exact_inverse",
             test_normalize_denormalize_is_exact_inverse),
            ("callback_constructs", test_callback_constructs),
        ],
        title="transform_pipeline_viz",
    )

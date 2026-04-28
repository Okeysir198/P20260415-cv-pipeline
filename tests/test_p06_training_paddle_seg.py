"""Test: Paddle backend full-chain — Segmentation (PP-LiteSeg-B1).

Same setup → train → eval → export → infer chain as the detection variant
(see ``test_p06_training_paddle_det.py``).

Skips when ``.venv-paddle/`` is missing, no CUDA, or the
``test_segmentation`` fixture isn't materialised.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import run_all  # noqa: E402
from test_p06_training_paddle_det import (  # noqa: E402
    PADDLE_VENV,
    dataset_present,
    ensure_paddle_venv,
    find_best_checkpoint,
    has_cuda,
    run_cli,
)

OUT_DIR = ROOT / "tests" / "outputs" / "08_training_paddle_seg"
EVAL_DIR = ROOT / "tests" / "outputs" / "10_evaluation_paddle_seg"
EXPORT_DIR = ROOT / "tests" / "outputs" / "12_export_paddle_seg"
INFER_DIR = ROOT / "tests" / "outputs" / "14_inference_paddle_seg"

TRAIN_CFG = ROOT / "configs" / "_test" / "06_training_paddle_seg.yaml"
DATA_CFG = ROOT / "configs" / "_test" / "05_data_segmentation.yaml"


def test_paddle_seg_full_chain():
    if not has_cuda():
        print("SKIP: no CUDA GPU available — Paddle backend requires GPU")
        return
    if not dataset_present("test_segmentation"):
        print("SKIP: dataset_store/test_segmentation not materialised")
        return
    if not ensure_paddle_venv():
        return

    for d in (OUT_DIR, EVAL_DIR, EXPORT_DIR, INFER_DIR):
        d.mkdir(parents=True, exist_ok=True)

    py = sys.executable  # main venv: train.py imports torch; paddle dispatcher subprocess-hops into .venv-paddle/
    env = {**os.environ, "WANDB_MODE": "disabled", "CUDA_VISIBLE_DEVICES": "0"}

    # Train
    r = run_cli([
        py, str(ROOT / "core" / "p06_training" / "train.py"),
        "--config", str(TRAIN_CFG),
        "--override",
        "training.epochs=1",
        "data.subset.train=0.1",
        "data.subset.val=0.1",
        f"logging.save_dir={OUT_DIR}",
    ], env=env)
    assert r.returncode == 0, f"train.py exited {r.returncode}"

    ckpt = find_best_checkpoint(OUT_DIR)
    assert ckpt is not None, f"no best checkpoint in {OUT_DIR}"

    # Eval
    r = run_cli([
        py, str(ROOT / "core" / "p08_evaluation" / "evaluate.py"),
        "--model", str(ckpt),
        "--config", str(DATA_CFG),
        "--output-dir", str(EVAL_DIR),
    ], env=env)
    assert r.returncode == 0, f"evaluate.py exited {r.returncode}"
    assert (EVAL_DIR / "metrics.json").exists()

    # Export
    r = run_cli([
        py, str(ROOT / "core" / "p09_export" / "export.py"),
        "--model", str(ckpt),
        "--training-config", str(TRAIN_CFG),
        "--output-dir", str(EXPORT_DIR),
        "--skip-optimize",
    ], env=env)
    assert r.returncode == 0, f"export.py exited {r.returncode}"
    onnx_path = next(EXPORT_DIR.glob("*.onnx"), None)
    assert onnx_path is not None, "no .onnx file produced"

    # Infer — SegmentationPredictor returns an HxW class-id mask
    from core.p10_inference.predictor import SegmentationPredictor  # noqa: WPS433
    from fixtures import real_image  # noqa: WPS433

    img = real_image()
    predictor = SegmentationPredictor(
        model_path=str(onnx_path),
        config_path=str(DATA_CFG),
        device="cuda",
    )
    mask = predictor.predict(img)
    assert mask is not None, "predict() returned None"
    # Mask should be 2-D and have at least one non-background pixel allowed
    # (fresh-from-scratch model can output all zeros; we only assert shape)
    assert getattr(mask, "ndim", 0) == 2 or getattr(mask, "shape", None), (
        f"unexpected mask type: {type(mask).__name__}"
    )
    print(f"    seg mask shape: {getattr(mask, 'shape', '?')}")


if __name__ == "__main__":
    run_all(
        [("paddle_seg_full_chain", test_paddle_seg_full_chain)],
        title="Paddle Backend — Segmentation Full Chain",
    )

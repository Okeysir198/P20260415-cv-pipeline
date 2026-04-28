"""Test: Paddle backend full-chain — Keypoint (PP-TinyPose 128x96).

Mirrors the detection/cls/seg variants but for top-down keypoint.

There is currently **no keypoint test fixture** in ``configs/_test/`` or
``tests/fixtures/`` — when one is checked in (expected at
``configs/_test/05_data_keypoint.yaml`` + ``dataset_store/test_keypoint_100/``),
this test will run the full chain. Until then it skips-with-reason and
remains green so the registration is non-fatal in ``run_all.py``.

Also skips when ``.venv-paddle/`` or CUDA is unavailable.
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

OUT_DIR = ROOT / "tests" / "outputs" / "08_training_paddle_kpt"
EVAL_DIR = ROOT / "tests" / "outputs" / "10_evaluation_paddle_kpt"
EXPORT_DIR = ROOT / "tests" / "outputs" / "12_export_paddle_kpt"
INFER_DIR = ROOT / "tests" / "outputs" / "14_inference_paddle_kpt"

TRAIN_CFG = ROOT / "configs" / "_test" / "06_training_paddle_kpt.yaml"
DATA_CFG = ROOT / "configs" / "_test" / "05_data_keypoint.yaml"


def test_paddle_kpt_full_chain():
    if not has_cuda():
        print("SKIP: no CUDA GPU available — Paddle backend requires GPU")
        return
    if not DATA_CFG.exists():
        print(f"SKIP: keypoint test fixture missing — "
              f"add {DATA_CFG.relative_to(ROOT)} + "
              f"dataset_store/test_keypoint_100/ to enable this test")
        return
    if not dataset_present("test_keypoint_100"):
        print("SKIP: dataset_store/test_keypoint_100 not materialised")
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

    # Infer — PosePredictor on a fixture image
    from core.p10_inference.pose_predictor import PosePredictor  # noqa: WPS433
    from fixtures import real_image  # noqa: WPS433

    img = real_image()
    predictor = PosePredictor(
        model_path=str(onnx_path),
        config_path=str(DATA_CFG),
        device="cuda",
    )
    out = predictor.predict(img)
    assert out is not None, "predict() returned None"
    # Pose output is typically a list of {keypoints, scores, bbox} dicts
    print(f"    pose inference: {type(out).__name__}")


if __name__ == "__main__":
    run_all(
        [("paddle_kpt_full_chain", test_paddle_kpt_full_chain)],
        title="Paddle Backend — Keypoint Full Chain",
    )

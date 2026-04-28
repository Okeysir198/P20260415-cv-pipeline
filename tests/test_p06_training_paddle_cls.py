"""Test: Paddle backend full-chain — Classification (PP-LCNet).

Same setup → train → eval → export → infer chain as the detection variant
(see ``test_p06_training_paddle_det.py``); see that file for the rationale
behind each step and the skip-with-reason policy.

Skips when ``.venv-paddle/`` is missing, no CUDA, or the
``test_fall_detection_100`` fixture isn't materialised.
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

OUT_DIR = ROOT / "tests" / "outputs" / "08_training_paddle_cls"
EVAL_DIR = ROOT / "tests" / "outputs" / "10_evaluation_paddle_cls"
EXPORT_DIR = ROOT / "tests" / "outputs" / "12_export_paddle_cls"
INFER_DIR = ROOT / "tests" / "outputs" / "14_inference_paddle_cls"

TRAIN_CFG = ROOT / "configs" / "_test" / "06_training_paddle_cls.yaml"
DATA_CFG = ROOT / "configs" / "_test" / "05_data_fall.yaml"


def test_paddle_cls_full_chain():
    if not has_cuda():
        print("SKIP: no CUDA GPU available — Paddle backend requires GPU")
        return
    if not dataset_present("test_fall_detection_100"):
        print("SKIP: dataset_store/test_fall_detection_100 not materialised")
        return
    if not ensure_paddle_venv():
        return

    for d in (OUT_DIR, EVAL_DIR, EXPORT_DIR, INFER_DIR):
        d.mkdir(parents=True, exist_ok=True)

    py = str(PADDLE_VENV)
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

    # Infer — ClassificationPredictor on a fixture image
    from core.p10_inference.predictor import ClassificationPredictor  # noqa: WPS433
    from fixtures import real_image  # noqa: WPS433

    img = real_image()
    predictor = ClassificationPredictor(
        model_path=str(onnx_path),
        config_path=str(DATA_CFG),
        device="cuda",
    )
    out = predictor.predict(img)
    # Accept dict {label, score, …} or top-k list
    assert out is not None, "predict() returned None"
    if isinstance(out, dict):
        assert "label" in out or "scores" in out or "logits" in out, f"unexpected keys: {list(out)}"
    else:
        assert hasattr(out, "__len__") and len(out) >= 1
    print(f"    inference output: {type(out).__name__}")


if __name__ == "__main__":
    run_all(
        [("paddle_cls_full_chain", test_paddle_cls_full_chain)],
        title="Paddle Backend — Classification Full Chain",
    )

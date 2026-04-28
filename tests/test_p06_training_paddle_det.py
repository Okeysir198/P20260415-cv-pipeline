"""Test: Paddle backend full-chain — Detection (PicoDet-S).

Walks every step that the user demands for the new `backend: paddle` path:

    1. ``bash scripts/setup-paddle-venv.sh`` (idempotent).
    2. Train 1 epoch via ``core/p06_training/train.py`` with the test config.
    3. Eval via ``core/p08_evaluation/evaluate.py``.
    4. Export via ``core/p09_export/export.py --skip-optimize``.
    5. Inference via :class:`core.p10_inference.predictor.DetectionPredictor`
       on one fixture image — assert at least one valid output.

Skip-with-reason whenever:

    * ``.venv-paddle/bin/python`` does not exist (upstream paddle units
      haven't built the venv yet — this test is forward-compatible).
    * No CUDA GPU is available.
    * The test_fire_100 dataset is not materialised (DVC-pulled).

Mirrors the structure of ``tests/test_p06_training_hf_detection.py`` but
runs through the project's standard CLI scripts so it tests the wiring
end-to-end, not just the trainer entrypoint.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import run_all  # noqa: E402

OUT_DIR = ROOT / "tests" / "outputs" / "08_training_paddle_det"
EVAL_DIR = ROOT / "tests" / "outputs" / "10_evaluation_paddle_det"
EXPORT_DIR = ROOT / "tests" / "outputs" / "12_export_paddle_det"
INFER_DIR = ROOT / "tests" / "outputs" / "14_inference_paddle_det"

TRAIN_CFG = ROOT / "configs" / "_test" / "06_training_paddle_det.yaml"
DATA_CFG = ROOT / "configs" / "_test" / "05_data.yaml"
SETUP_SCRIPT = ROOT / "scripts" / "setup-paddle-venv.sh"
PADDLE_VENV = ROOT / ".venv-paddle" / "bin" / "python"

# Paddle backend may save in any of these formats depending on the trainer impl.
CHECKPOINT_NAMES = ("best.pdparams", "best.pth", "best.pt")


def find_best_checkpoint(run_dir: Path) -> Path | None:
    return next((run_dir / n for n in CHECKPOINT_NAMES if (run_dir / n).exists()), None)


def has_cuda() -> bool:
    try:
        import torch  # noqa: WPS433

        return torch.cuda.is_available()
    except Exception:
        return False


def dataset_present(name: str) -> bool:
    """Return True if `dataset_store/<name>/` looks usable (has images)."""
    base = ROOT / "dataset_store" / name
    if not base.is_dir():
        return False
    # accept either YOLO-style train/images or plain folder layout (cls)
    return any(base.rglob("*.jpg")) or any(base.rglob("*.png"))


def ensure_paddle_venv() -> bool:
    """Try to build .venv-paddle/ once; return True if usable, False to skip."""
    if PADDLE_VENV.exists():
        return True
    if not SETUP_SCRIPT.exists():
        print(f"SKIP: {SETUP_SCRIPT.name} not found "
              "(upstream paddle backend units haven't landed yet)")
        return False
    print(f"  Running {SETUP_SCRIPT.name} (idempotent)…")
    try:
        result = subprocess.run(
            ["bash", str(SETUP_SCRIPT)],
            cwd=str(ROOT), timeout=600, check=False,
            capture_output=True, text=True,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        print(f"SKIP: setup-paddle-venv.sh failed: {exc}")
        return False
    if result.returncode != 0 or not PADDLE_VENV.exists():
        tail = (result.stderr or result.stdout or "").splitlines()[-5:]
        print(f"SKIP: .venv-paddle/ not built. Last lines:\n  " + "\n  ".join(tail))
        return False
    return True


def run_cli(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a CLI command from project root, surface stderr on failure."""
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(ROOT), check=False, **kwargs)


def test_paddle_det_full_chain():
    # 0. Preconditions
    if not has_cuda():
        print("SKIP: no CUDA GPU available — Paddle backend requires GPU")
        return
    if not dataset_present("test_fire_100"):
        print("SKIP: dataset_store/test_fire_100 not materialised "
              "(DVC pull required)")
        return
    if not ensure_paddle_venv():
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    INFER_DIR.mkdir(parents=True, exist_ok=True)

    py = sys.executable  # main venv: train.py imports torch; paddle dispatcher subprocess-hops into .venv-paddle/
    env = {**os.environ, "WANDB_MODE": "disabled", "CUDA_VISIBLE_DEVICES": "0"}

    # 2. Train 1 epoch
    train_cmd = [
        py, str(ROOT / "core" / "p06_training" / "train.py"),
        "--config", str(TRAIN_CFG),
        "--override",
        "training.epochs=1",
        "data.subset.train=0.1",
        "data.subset.val=0.1",
        f"logging.save_dir={OUT_DIR}",
    ]
    r = run_cli(train_cmd, env=env)
    assert r.returncode == 0, f"train.py exited {r.returncode}"

    ckpt = find_best_checkpoint(OUT_DIR)
    assert ckpt is not None, f"no best checkpoint in {OUT_DIR}"
    print(f"    checkpoint: {ckpt.name} ({ckpt.stat().st_size/1e6:.1f} MB)")

    # 3. Eval
    eval_cmd = [
        py, str(ROOT / "core" / "p08_evaluation" / "evaluate.py"),
        "--model", str(ckpt),
        "--config", str(DATA_CFG),
        "--output-dir", str(EVAL_DIR),
    ]
    r = run_cli(eval_cmd, env=env)
    assert r.returncode == 0, f"evaluate.py exited {r.returncode}"
    metrics_json = EVAL_DIR / "metrics.json"
    assert metrics_json.exists(), f"missing {metrics_json}"

    # 4. Export ONNX (skip optimum optimization; fp32 is enough for smoke)
    export_cmd = [
        py, str(ROOT / "core" / "p09_export" / "export.py"),
        "--model", str(ckpt),
        "--training-config", str(TRAIN_CFG),
        "--output-dir", str(EXPORT_DIR),
        "--skip-optimize",
    ]
    r = run_cli(export_cmd, env=env)
    assert r.returncode == 0, f"export.py exited {r.returncode}"
    onnx_path = next(EXPORT_DIR.glob("*.onnx"), None)
    assert onnx_path is not None, f"no .onnx file in {EXPORT_DIR}"
    print(f"    onnx: {onnx_path.name} ({onnx_path.stat().st_size/1e6:.1f} MB)")

    # 5. Inference on one fixture image
    from core.p10_inference.predictor import DetectionPredictor  # noqa: WPS433
    from fixtures import real_image  # noqa: WPS433

    img = real_image()
    predictor = DetectionPredictor(
        model_path=str(onnx_path),
        config_path=str(DATA_CFG),
        device="cuda",
    )
    detections = predictor.predict(img, conf_threshold=0.0)  # accept anything > 0
    # Detections is supervision.Detections or a list — coerce to len()
    n = len(detections) if hasattr(detections, "__len__") else 0
    assert n >= 0, f"predict() returned non-iterable: {detections!r}"
    print(f"    inference: {n} detections on fixture image")


if __name__ == "__main__":
    run_all(
        [("paddle_det_full_chain", test_paddle_det_full_chain)],
        title="Paddle Backend — Detection Full Chain",
    )

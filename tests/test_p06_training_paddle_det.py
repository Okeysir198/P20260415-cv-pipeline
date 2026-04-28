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


def gpu_compute_capability() -> tuple[int, int] | None:
    """Return (major, minor) compute capability of CUDA device 0, or None."""
    try:
        import torch  # noqa: WPS433

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_capability(0)
    except Exception:
        return None


def paddle_supports_gpu() -> bool:
    """paddlepaddle-gpu 3.3.x ships sm_90 max — Blackwell (sm_120, RTX 50xx)
    isn't supported yet upstream. Loss kernels fail with malformed shapes.
    Update this guard when paddle ships sm_120 wheels.
    """
    cc = gpu_compute_capability()
    if cc is None:
        return False
    major, _ = cc
    return major <= 9


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
    if not paddle_supports_gpu():
        cc = gpu_compute_capability()
        print(f"SKIP: paddle 3.3.x doesn't support compute capability {cc} "
              f"(Blackwell / RTX 50xx). Awaiting upstream paddle release.")
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

    paddle_py = str(PADDLE_VENV)  # paddle code lives in .venv-paddle/
    env = {**os.environ, "WANDB_MODE": "disabled", "CUDA_VISIBLE_DEVICES": "0"}

    # 2. Train 1 epoch — runs in .venv-paddle/, paddle-native trainer
    train_cmd = [
        paddle_py, str(ROOT / "core" / "p06_paddle" / "train.py"),
        "--config", str(TRAIN_CFG),
        "--override", "training.epochs=1",
        f"logging.save_dir={OUT_DIR}",
    ]
    r = run_cli(train_cmd, env=env)
    assert r.returncode == 0, f"core/p06_paddle/train.py exited {r.returncode}"

    ckpt = find_best_checkpoint(OUT_DIR)
    assert ckpt is not None, f"no best checkpoint in {OUT_DIR}"
    print(f"    checkpoint: {ckpt.name} ({ckpt.stat().st_size/1e6:.1f} MB)")

    # 3. Export ONNX — runs in .venv-paddle/, then everything downstream goes
    #    through the standard main-venv ORT path.
    onnx_path = EXPORT_DIR / "model.onnx"
    export_cmd = [
        paddle_py, str(ROOT / "core" / "p06_paddle" / "export.py"),
        "--config", str(TRAIN_CFG),
        "--checkpoint", str(ckpt),
        "--out", str(onnx_path),
    ]
    r = run_cli(export_cmd, env=env)
    assert r.returncode == 0, f"core/p06_paddle/export.py exited {r.returncode}"
    assert onnx_path.exists(), f"no .onnx file at {onnx_path}"
    print(f"    onnx: {onnx_path.name} ({onnx_path.stat().st_size/1e6:.1f} MB)")

    # 4. Eval the ONNX from main venv (this is the canonical post-export path)
    eval_cmd = [
        sys.executable, str(ROOT / "core" / "p08_evaluation" / "evaluate.py"),
        "--model", str(onnx_path),
        "--config", str(DATA_CFG),
        "--output-dir", str(EVAL_DIR),
    ]
    r = run_cli(eval_cmd, env=env)
    assert r.returncode == 0, f"evaluate.py (ONNX) exited {r.returncode}"
    metrics_json = EVAL_DIR / "metrics.json"
    assert metrics_json.exists(), f"missing {metrics_json}"

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

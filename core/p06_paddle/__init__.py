"""PaddlePaddle backend (paddle-native, runs in .venv-paddle/).

Architecturally separate from core/p06_training (the torch/HF backend dispatcher).
Paddle has its own training loops (ppdet.engine.Trainer, PaddleClas Engine,
PaddleSeg Trainer) — we drive them directly rather than try to bridge tensors.

Convergence happens at ONNX: `core/p06_paddle/export.py` writes `model.onnx`,
then everything downstream (eval, error analysis, inference, demo) goes through
the standard main-venv ORT path that already handles YOLOX/HF ONNX.

Entry points (all run from `.venv-paddle/bin/python`):
- `core/p06_paddle/train.py`  — `Trainer.train()` + observability tree skeleton
- `core/p06_paddle/eval.py`   — `Trainer.evaluate()` on a checkpoint
- `core/p06_paddle/export.py` — `paddle2onnx` wrapper

This package must NOT be imported from the main venv. Doing so will fail with
ImportError on `paddle`. The pipeline guards against this at the dispatcher
(`core/p06_training/train.py` prints a redirect when `backend: paddle`).
"""

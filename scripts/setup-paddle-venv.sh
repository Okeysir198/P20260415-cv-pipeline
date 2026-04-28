#!/usr/bin/env bash
# Set up .venv-paddle — a dedicated venv for the native PaddlePaddle backend
# (PaddleDetection / PaddleClas / PaddleSeg + paddle2onnx).
#
# Why: paddlepaddle-gpu ships its own CUDA runtime bundle (CUDA 12.x at the time
# of writing) and pins numpy/protobuf/onnx versions that conflict with the main
# .venv (CUDA 13 torch + git transformers + numpy 2.x). Keeping it isolated
# avoids polluting the main environment. Paddle's bundled CUDA does not need to
# match the host CUDA used by torch — they live in different process venvs.
#
# Usage:
#   bash scripts/setup-paddle-venv.sh           # create .venv-paddle (idempotent)
#
# Then run paddle-backed training/export via:
#   .venv-paddle/bin/python core/p06_training/train.py \
#     --config features/<feat>/configs/06_training_paddle.yaml
#
# Idempotency: if .venv-paddle/ already exists, the script exits 0 with a
# "already exists, skipping" message. Delete the directory to force a refresh.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv-paddle"

if [ -d "$VENV_DIR" ]; then
  echo "$VENV_DIR already exists, skipping."
  echo "Delete it (rm -rf $VENV_DIR) to force a refresh."
  exit 0
fi

echo "Creating $VENV_DIR (Python 3.12)..."
uv venv --python 3.12 "$VENV_DIR"

PYTHON="$VENV_DIR/bin/python"

echo "Installing paddlepaddle-gpu (bundled CUDA 12.x) into $VENV_DIR..."
# Paddle ships GPU wheels via its own index. CUDA 12.x build is ABI-compatible
# with most modern hosts; the wheel bundles its own CUDA runtime libs so it
# does not need to match the host CUDA used by torch in the main venv.
uv pip install --python "$PYTHON" \
  --index-strategy unsafe-best-match \
  --extra-index-url "https://www.paddlepaddle.org.cn/packages/stable/cu126/" \
  "paddlepaddle-gpu>=3.0.0"

echo "Installing paddle2onnx + PaddleSeg from PyPI..."
# paddle2onnx has working Python 3.12 wheels; paddleseg's PyPI release is also
# 3.12-clean. PaddleDetection / PaddleClas PyPI releases are stale (pin
# numpy<1.24 / opencv==4.6.0 / faiss-cpu==1.7.1) and cannot resolve on
# Python 3.12 — install those from git source below instead.
uv pip install --python "$PYTHON" \
  "paddleseg>=2.8" \
  "paddle2onnx>=1.2"

echo "Installing PaddleDetection + PaddleClas from git (--no-deps)..."
# --no-deps avoids their stale numpy/opencv/faiss pins; the runtime deps
# they need are already covered by the shared block below.
uv pip install --python "$PYTHON" --no-deps \
  "git+https://github.com/PaddlePaddle/PaddleDetection.git" \
  "git+https://github.com/PaddlePaddle/PaddleClas.git"

echo "Installing shared runtime deps used by core/ + utils/..."
uv pip install --python "$PYTHON" \
  "numpy>=1.26,<2.0" \
  "pyyaml>=6.0.3" \
  "tqdm>=4.67" \
  "Pillow>=10" \
  "opencv-python>=4.9" \
  "matplotlib>=3.8" \
  "supervision>=0.27.0" \
  "pycocotools>=2.0" \
  "tensorboard>=2.17" \
  "scikit-learn>=1.5" \
  "loguru>=0.7" \
  "tabulate>=0.9" \
  "psutil>=6.0" \
  "scikit-image>=0.24" \
  "onnx>=1.17" \
  "onnxruntime-gpu>=1.20; platform_machine == 'x86_64'" \
  "onnxruntime>=1.20; platform_machine != 'x86_64'" \
  "pandas>=2.2"

echo "Installing repo as editable so core/ + utils/ are importable..."
uv pip install --python "$PYTHON" -e "$REPO_ROOT" --no-deps

echo
echo "Done. Test with:"
echo "  $PYTHON -c 'import paddle; print(paddle.__version__); paddle.utils.run_check()'"
echo
echo "Then train with:"
echo "  $PYTHON core/p06_training/train.py \\"
echo "    --config features/<feat>/configs/06_training_paddle.yaml"

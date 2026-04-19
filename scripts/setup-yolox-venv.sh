#!/usr/bin/env bash
# Set up .venv-yolox-official — a dedicated venv for training/running YOLOX via
# the official Megvii-BaseDetection/YOLOX package.
#
# Why: the main venv ships a *custom* YOLOX reimplementation (see
# core/p06_models/yolox.py). Installing the upstream `yolox` package alongside
# would pin older torchvision/onnx/thop versions that clash with the main
# venv's CUDA 13 torch + git transformers. Keeping it isolated avoids
# polluting the main environment.
#
# Usage:
#   bash scripts/setup-yolox-venv.sh          # create / refresh .venv-yolox-official
#
# Then train against the official implementation via:
#   .venv-yolox-official/bin/python core/p06_training/train.py \
#     --config features/<feat>/configs/06_training.yaml \
#     --override model.impl=official
#
# The `model.impl` config key selects custom (default) or official at build time;
# this venv just makes the `yolox` import resolvable.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv-yolox-official"

echo "Creating $VENV_DIR (Python 3.12)..."
uv venv --python 3.12 "$VENV_DIR"

PYTHON="$VENV_DIR/bin/python"

echo "Installing torch + build deps into $VENV_DIR..."
uv pip install --python "$PYTHON" \
  --index-strategy unsafe-best-match \
  --extra-index-url "https://download.pytorch.org/whl/cu130" \
  "torch>=2.10.0" \
  "torchvision>=0.25.0"

# YOLOX's setup.py imports torch at build time. Also, its dependency list
# pins `onnx-simplifier==0.4.10` (old, fails to build on Python 3.12). We
# don't use YOLOX's own export path — the repo has its own ONNX export in
# core/p09_export/, so we install YOLOX with --no-deps and provide the
# runtime deps it needs explicitly below.
uv pip install --python "$PYTHON" "setuptools>=70" "wheel>=0.44" "ninja>=1.11"

echo "Installing official YOLOX from git (--no-deps, --no-build-isolation)..."
uv pip install --python "$PYTHON" --no-deps --no-build-isolation \
  "git+https://github.com/Megvii-BaseDetection/YOLOX.git"

echo "Installing YOLOX runtime deps + this repo's deps..."
uv pip install --python "$PYTHON" \
  "numpy>=2.4" \
  "pyyaml>=6.0.3" \
  "tqdm>=4.67" \
  "Pillow>=12.1" \
  "opencv-python>=4.13" \
  "matplotlib>=3.10" \
  "supervision>=0.27.0" \
  "pycocotools>=2.0" \
  "tensorboard>=2.17" \
  "scikit-learn>=1.5" \
  "optuna>=4.0" \
  "loguru>=0.7" \
  "tabulate>=0.9" \
  "psutil>=6.0" \
  "thop>=0.1" \
  "scikit-image>=0.24" \
  "onnxruntime-gpu>=1.24; platform_machine == 'x86_64'" \
  "onnxruntime>=1.24; platform_machine != 'x86_64'" \
  "onnx>=1.20" \
  "pandas>=2.2"

echo "Installing repo as editable so core/ + utils/ are importable..."
uv pip install --python "$PYTHON" -e "$REPO_ROOT" --no-deps

echo
echo "Done. Test with:"
echo "  $PYTHON -c 'from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead; print(\"official yolox ok\")'"
echo
echo "Then train with:"
echo "  $PYTHON core/p06_training/train.py \\"
echo "    --config features/<feat>/configs/06_training.yaml \\"
echo "    --override model.impl=official"

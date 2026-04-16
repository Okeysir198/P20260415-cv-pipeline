#!/usr/bin/env bash
# Set up .venv-export — a dedicated venv for ONNX optimization + INT8 quantization
# via optimum[onnxruntime].
#
# Why: optimum[onnxruntime] requires transformers<4.58, which conflicts with the
# git transformers pinned in the main .venv for SAM3/QA. Keeping it in a separate
# venv avoids polluting the main environment.
#
# Usage:
#   bash scripts/setup-export-venv.sh            # create / refresh .venv-export
#
# Then run quantized export via:
#   .venv-export/bin/python core/p09_export/export.py \
#     --model features/<feat>/runs/<ts>/best.pth \
#     --training-config features/<feat>/configs/06_training.yaml \
#     --output-dir features/<feat>/export \
#     --optimize O2 --quantize dynamic

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv-export"

echo "Creating $VENV_DIR (Python 3.12)..."
uv venv --python 3.12 "$VENV_DIR"

PYTHON="$VENV_DIR/bin/python"

echo "Installing ONNX export + optimum stack into $VENV_DIR..."
uv pip install --python "$PYTHON" \
  "torch>=2.10.0" \
  "torchvision>=0.25.0" \
  "transformers>=4.51,<4.58" \
  "optimum[onnxruntime]>=2.1" \
  "onnx>=1.20" \
  "onnxruntime-gpu>=1.24; platform_machine == 'x86_64'" \
  "onnxruntime>=1.24; platform_machine != 'x86_64'" \
  "onnxsim>=0.6" \
  "onnxscript>=0.6" \
  "thop>=0.1" \
  "opencv-python>=4.13" \
  "matplotlib>=3.10" \
  "supervision>=0.27.0" \
  "numpy>=2.4" \
  "pyyaml>=6.0.3" \
  "tqdm>=4.67" \
  "Pillow>=12.1"

echo "Installing repo as editable so core/ + utils/ are importable..."
uv pip install --python "$PYTHON" -e "$REPO_ROOT" --no-deps

echo
echo "Done. Test with:"
echo "  $PYTHON -c 'from optimum.onnxruntime import ORTOptimizer; print(\"optimum ok\")'"

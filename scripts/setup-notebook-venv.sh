#!/usr/bin/env bash
# Set up .venv-notebook — a dedicated venv for running the HF reference
# fine-tuning notebooks at notebooks/detr_finetune_reference/.
#
# Why: the notebooks pin specific dep versions (albumentations==1.4.6) and use
# the HF Trainer API; we keep them isolated from the main venv so the
# reference setup is byte-for-byte reproducible against upstream notebooks.
#
# Usage:
#   bash scripts/setup-notebook-venv.sh
#
# Then:
#   .venv-notebook/bin/jupyter lab notebooks/detr_finetune_reference/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv-notebook"

echo "Creating $VENV_DIR (Python 3.12)..."
uv venv --python 3.12 "$VENV_DIR"

PYTHON="$VENV_DIR/bin/python"

echo "Installing torch CUDA 13 + transformers + notebook deps..."
uv pip install --python "$PYTHON" \
  --index-strategy unsafe-best-match \
  --extra-index-url "https://download.pytorch.org/whl/cu130" \
  "torch>=2.10.0" \
  "torchvision>=0.25.0"

uv pip install --python "$PYTHON" \
  "transformers @ git+https://github.com/huggingface/transformers.git" \
  "albumentations==1.4.6" \
  "torchmetrics>=1.4" \
  "datasets>=2.21" \
  "pycocotools>=2.0" \
  "timm>=1.0" \
  "jupyterlab>=4.2" \
  "ipykernel>=6.29" \
  "matplotlib>=3.10" \
  "Pillow>=12.1" \
  "wandb>=0.17" \
  "numpy>=1.26" \
  "opencv-python>=4.13"

echo "Installing repo as editable so utils/ is importable from data_loader..."
uv pip install --python "$PYTHON" -e "$REPO_ROOT" --no-deps

echo "Registering Jupyter kernel 'detr-reference'..."
"$PYTHON" -m ipykernel install --user --name detr-reference \
  --display-name "Python (detr-reference)"

echo
echo "Done. Launch notebooks with:"
echo "  $VENV_DIR/bin/jupyter lab notebooks/detr_finetune_reference/"
echo
echo "Or run a single notebook headless:"
echo "  $VENV_DIR/bin/jupyter nbconvert --to notebook --execute \\"
echo "    notebooks/detr_finetune_reference/RT_DETR_v2_finetune_on_a_custom_dataset.ipynb"

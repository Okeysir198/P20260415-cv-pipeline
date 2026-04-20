#!/usr/bin/env bash
# Set up .venv-notebook — a dedicated uv-managed venv for running the HF
# reference fine-tuning notebooks at notebooks/detr_finetune_reference/.
#
# Why: the notebooks pin specific dep versions (albumentations==1.4.6) and use
# a git HF transformers; we keep them isolated from the main venv so the
# reference setup is byte-for-byte reproducible against upstream notebooks.
# Deps are declared in notebooks/detr_finetune_reference/pyproject.toml.
#
# Usage:
#   bash scripts/setup-notebook-venv.sh
#
# Then:
#   .venv-notebook/bin/jupyter lab notebooks/detr_finetune_reference/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_DIR="$REPO_ROOT/notebooks/detr_finetune_reference"
VENV_DIR="$REPO_ROOT/.venv-notebook"

echo "Syncing $PROJECT_DIR into $VENV_DIR (Python 3.12)..."
UV_PROJECT_ENVIRONMENT="$VENV_DIR" uv sync \
  --project "$PROJECT_DIR" \
  --python 3.12

PYTHON="$VENV_DIR/bin/python"

echo "Installing main repo as editable so utils/ is importable from data_loader..."
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
echo "    notebooks/detr_finetune_reference/reference/RT_DETR_v2_finetune_on_a_custom_dataset.ipynb"

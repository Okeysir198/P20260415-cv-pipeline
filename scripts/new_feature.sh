#!/usr/bin/env bash
# Scaffold a new feature folder from features/_TEMPLATE/.
# Usage: bash scripts/new_feature.sh <feature_name>
#
# Naming convention:
#   feature_name (folder): kebab-case with hyphens, e.g. "ppe-helmet_detection"
#   dataset_name (config): snake_case (underscores), e.g. "ppe_helmet_detection"
# The script derives <dataset_name> from <feature_name> by replacing every
# hyphen with an underscore so both stay consistent with every downstream
# consumer (training_ready/, runs/, LS project, release dir).

set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <feature_name>" >&2
  exit 1
fi
NAME="$1"
DATASET_NAME="${NAME//-/_}"   # kebab → snake

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO_ROOT/features/_TEMPLATE"
DST="$REPO_ROOT/features/$NAME"

if [ -e "$DST" ]; then
  echo "Error: $DST already exists." >&2
  exit 1
fi

cp -r "$SRC" "$DST"

# Replace <feature_name> and <dataset_name> placeholders in tracked text files
grep -rlE '<feature_name>|<dataset_name>' "$DST" 2>/dev/null \
  | xargs -r sed -i -e "s/<feature_name>/$NAME/g" -e "s/<dataset_name>/$DATASET_NAME/g"

echo "Created $DST"
echo "  feature_name: $NAME"
echo "  dataset_name: $DATASET_NAME"
echo "Next steps:"
echo "  1. Edit $DST/configs/05_data.yaml (classes)"
echo "  2. Edit $DST/configs/06_training.yaml (model, hyperparams)"
echo "  3. uv run python core/p06_training/train.py --config $DST/configs/06_training.yaml"

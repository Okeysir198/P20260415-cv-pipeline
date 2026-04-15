#!/usr/bin/env bash
# Scaffold a new feature folder from features/_TEMPLATE/.
# Usage: bash scripts/new_feature.sh <feature_name>
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <feature_name>" >&2
  exit 1
fi
NAME="$1"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO_ROOT/features/_TEMPLATE"
DST="$REPO_ROOT/features/$NAME"

if [ -e "$DST" ]; then
  echo "Error: $DST already exists." >&2
  exit 1
fi

cp -r "$SRC" "$DST"

# Replace <feature_name> placeholders in tracked text files
grep -rl '<feature_name>' "$DST" 2>/dev/null \
  | xargs -r sed -i "s/<feature_name>/$NAME/g"

echo "Created $DST"
echo "Next steps:"
echo "  1. Edit $DST/configs/05_data.yaml (classes, dataset path)"
echo "  2. Edit $DST/configs/06_training.yaml (model, hyperparams)"
echo "  3. uv run python core/p06_training/train.py --config $DST/configs/06_training.yaml"

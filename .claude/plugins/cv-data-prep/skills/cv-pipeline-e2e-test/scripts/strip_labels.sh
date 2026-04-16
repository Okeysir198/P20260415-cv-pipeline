#!/usr/bin/env bash
# Copy images from <src> into <dst>/{train,val}/images without any labels.
# Usage: strip_labels.sh <src_dataset_with_splits> <dst_raw_stage> [n_train] [n_val]

set -euo pipefail

SRC="${1:?src dir}"
DST="${2:?dst dir}"
N_TRAIN="${3:-15}"
N_VAL="${4:-5}"

if [ ! -d "$SRC/train/images" ] || [ ! -d "$SRC/val/images" ]; then
  echo "Source $SRC must have train/images/ and val/images/ subdirs." >&2
  exit 1
fi

mkdir -p "$DST"/{train,val}/images

copy_n() {
  local split="$1" n="$2"
  local i=0
  # Prefer .jpg then .png
  for f in "$SRC/$split/images"/*.jpg "$SRC/$split/images"/*.jpeg "$SRC/$split/images"/*.png; do
    [ -f "$f" ] || continue
    cp "$f" "$DST/$split/images/"
    i=$((i+1))
    [ "$i" -ge "$n" ] && break
  done
  echo "  $split: copied $i images"
}

copy_n train "$N_TRAIN"
copy_n val   "$N_VAL"

# Sanity: no label files sneaked in.
leaked=$(find "$DST" \( -name '*.txt' -o -name '*.json' -o -name '*.xml' \) -type f | head -5)
if [ -n "$leaked" ]; then
  echo "Label files leaked into $DST — aborting:" >&2
  echo "$leaked" >&2
  exit 2
fi

echo "Raw stage ready at $DST (no labels)."

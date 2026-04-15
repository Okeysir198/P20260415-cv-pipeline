#!/usr/bin/env bash
# inspect_source.sh <raw_dataset_path>
# Probe a raw dataset folder and report: image count, label format, unique class names.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <raw_dataset_path>" >&2
  exit 2
fi

P="$1"
if [[ ! -d "$P" ]]; then
  echo "Not a directory: $P" >&2
  exit 1
fi

echo "=== $P ==="

# 1. Image count + size
imgs=$(find "$P" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) 2>/dev/null | wc -l)
size=$(du -sh "$P" 2>/dev/null | cut -f1)
echo "size=$size  images=$imgs"

# 2. Detect format
yolo_yaml=$(find "$P" -maxdepth 3 -name 'data.yaml' 2>/dev/null | head -1)
coco_json=$(find "$P" -maxdepth 3 \( -name '_annotations.coco.json' -o -name 'instances_*.json' \) 2>/dev/null | head -1)
voc_dir=$(find "$P" -maxdepth 3 -type d \( -name 'voc_labels' -o -name 'Annotations' \) 2>/dev/null | head -1)

format="unknown"
if [[ -n "$yolo_yaml" ]]; then format="yolo"; fi
if [[ -n "$coco_json" && -z "$yolo_yaml" ]]; then format="coco"; fi
if [[ -n "$voc_dir" && -z "$yolo_yaml" && -z "$coco_json" ]]; then format="voc"; fi
if [[ -n "$voc_dir" && -n "$yolo_yaml" ]]; then format="voc+yolo (prefer voc)"; fi

echo "format=$format"

# 3. Splits
for split in train val valid test; do
  n=$(find "$P" -maxdepth 3 -type d -name "$split" 2>/dev/null | head -1)
  if [[ -n "$n" ]]; then
    c=$(find "$n" -type f \( -iname '*.jpg' -o -iname '*.png' \) 2>/dev/null | wc -l)
    [[ "$c" -gt 0 ]] && echo "  split '$split' -> $c imgs"
  fi
done

# 4. Class names
echo "classes:"
case "$format" in
  yolo|'voc+yolo (prefer voc)')
    if [[ -n "$yolo_yaml" ]]; then
      # names: can be a list ("- Boots") or a dict ("0: Boots") or an inline array (["a","b"]).
      python3 -c "
import yaml, sys
d = yaml.safe_load(open('$yolo_yaml'))
n = d.get('names')
if isinstance(n, list):
    for i, x in enumerate(n): print(f'  - {i}: {x}')
elif isinstance(n, dict):
    for i, x in sorted(n.items()): print(f'  - {i}: {x}')
" 2>/dev/null | head -30
    fi
    ;;
esac
case "$format" in
  coco)
    python3 -c "
import json
with open('$coco_json') as f: d = json.load(f)
for c in d.get('categories', []):
    print(f'  - {c[\"id\"]}: {c[\"name\"]}')
" 2>/dev/null | head -30
    ;;
  voc|'voc+yolo (prefer voc)')
    if [[ -n "$voc_dir" ]]; then
      find "$voc_dir" -name '*.xml' -exec grep -hoP '<name>\K[^<]+' {} + 2>/dev/null | sort -u | head -30 | sed 's/^/  - /'
    fi
    ;;
esac

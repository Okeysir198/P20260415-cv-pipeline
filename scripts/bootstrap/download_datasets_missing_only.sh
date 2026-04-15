#!/usr/bin/env bash
# Download only missing datasets (skip what's already present on current machine).
# Analyzes manifest and filters out rows where dest already exists.
#
# Usage:
#   bash ai/scripts/bootstrap/download_datasets_missing_only.sh                 # all missing
#   bash ai/scripts/bootstrap/download_datasets_missing_only.sh --only <feat>   # missing for feature
#   bash ai/scripts/bootstrap/download_datasets_missing_only.sh --dry-run
#   bash ai/scripts/bootstrap/download_datasets_missing_only.sh --analyze       # show missing without downloading
#
# Secrets (KAGGLE_KEY, ROBOFLOW_API_KEY, HF_TOKEN) loaded from $PROJECT_ROOT/.env.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

MANIFEST="$SCRIPT_DIR/manifests/datasets.tsv"
DEST_ROOT="$PROJECT_ROOT/ai/dataset_store"

ANALYZE_ONLY=0

# Extended arg parser
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only) FILTER_FEATURE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --analyze) ANALYZE_ONLY=1; shift ;;
    -h|--help)
      sed -n '1,/^$/p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

process_row() {
  local source_type="$1" url_or_id="$2" dest_rel="$3" size_gb="$4" license="$5" features="$6" notes="$7"

  # Skip comments and header
  [[ -z "${source_type:-}" || "${source_type:0:1}" == "#" ]] && return 0
  [[ "$source_type" == "source_type" ]] && return 0

  # Filter by feature if specified
  if [[ -n "$FILTER_FEATURE" && "$features" != *"$FILTER_FEATURE"* ]]; then
    return 0
  fi

  local dest="$DEST_ROOT/$dest_rel"

  # Skip if already present
  if already_present "$dest"; then
    if [[ "$ANALYZE_ONLY" -eq 1 ]]; then
      return 0  # Silent skip for analyze mode
    fi
    N_SKIP=$((N_SKIP+1))
    printf '  ✓ SKIP %-10s %s (%.2f GB already present)\n' "$source_type" "$dest_rel" "$(( $(du -sb "$dest" 2>/dev/null | awk '{print $1}') / (1024*1024*1024) ))" 2>/dev/null || true
    log_row SKIP "$features" "$dest" "$(fsize "$dest")"
    return 0
  fi

  # Print missing item
  if [[ "$ANALYZE_ONLY" -eq 1 ]]; then
    # Handle size_gb that might be "—" or other non-numeric values
    local size_display="${size_gb}"
    if [[ ! "$size_gb" =~ ^[0-9] ]]; then
      size_display="?"
    fi
    echo "  ✗ MISSING $source_type $dest_rel (${size_display} GB) — $features"
    return 0
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '  [dry-run] %-10s %-6s %s -> %s\n' "$source_type" "${size_gb}G" "$url_or_id" "$dest_rel"
    return 0
  fi

  printf '[%s] %s (%sGB) -> %s\n' "$source_type" "$features" "$size_gb" "$dest_rel"

  case "$source_type" in
    manual)
      echo "  MANUAL: fetch via $url_or_id"
      [[ -n "$notes" ]] && echo "  notes: $notes"
      N_MANUAL=$((N_MANUAL+1))
      log_row MANUAL "$features" "$dest" 0
      return 0 ;;
    rsync_only)
      echo "  RSYNC-ONLY ($notes): source=$url_or_id"
      N_RSYNC=$((N_RSYNC+1))
      log_row RSYNC "$features" "$dest" 0
      return 0 ;;
  esac

  local ok=1
  dispatch_download "$source_type" "$url_or_id" "$dest" || ok=0

  if [[ "$ok" -eq 1 ]]; then
    N_DL=$((N_DL+1))
    log_row OK "$features" "$dest" "$(fsize "$dest")"
  else
    N_FAIL=$((N_FAIL+1))
    log_row FAIL "$features" "$dest" "$(fsize "$dest")"
  fi
}

# ============================================================================
# Main
# ============================================================================

if [[ "$ANALYZE_ONLY" -eq 1 ]]; then
  echo "=== Analyzing missing datasets ==="
  echo "Filter: ${FILTER_FEATURE:-all}"
  echo ""
  while IFS=$'\t' read -r source_type url_or_id dest_rel size_gb license features notes; do
    process_row "$source_type" "$url_or_id" "$dest_rel" "$size_gb" "$license" "$features" "${notes:-}"
  done <"$MANIFEST"
  exit 0
fi

mkdir -p "$(dirname "$LOG_FILE")"
echo "# datasets bootstrap missing-only $(date -u +%FT%TZ) filter=${FILTER_FEATURE:-all} dry=${DRY_RUN}" >>"$LOG_FILE"

while IFS=$'\t' read -r source_type url_or_id dest_rel size_gb license features notes; do
  process_row "$source_type" "$url_or_id" "$dest_rel" "$size_gb" "$license" "$features" "${notes:-}"
done <"$MANIFEST"

print_summary

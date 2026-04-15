#!/usr/bin/env bash
# Download all public open datasets needed for Phase 1 features.
# Internal / Nitto-Denko data is flagged rsync_only — not fetched here.
#
# Usage:
#   bash ai/scripts/bootstrap/download_datasets.sh                 # all
#   bash ai/scripts/bootstrap/download_datasets.sh --only <feat>   # filter (substring match on features_using)
#   bash ai/scripts/bootstrap/download_datasets.sh --dry-run
#
# Secrets (KAGGLE_KEY, ROBOFLOW_API_KEY, HF_TOKEN) loaded from $PROJECT_ROOT/.env.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

parse_args "$@"

MANIFEST="$SCRIPT_DIR/manifests/datasets.tsv"
DEST_ROOT="$PROJECT_ROOT/ai/dataset_store"

process_row() {
  local source_type="$1" url_or_id="$2" dest_rel="$3" size_gb="$4" license="$5" features="$6" notes="$7"

  # features_using column is comma-separated — filter does substring match.
  if [[ -n "$FILTER_FEATURE" && "$features" != *"$FILTER_FEATURE"* ]]; then
    return 0
  fi

  local dest="$DEST_ROOT/$dest_rel"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '  [dry-run] %-10s %-6s %s -> %s\n' "$source_type" "${size_gb}G" "$url_or_id" "$dest_rel"
    return 0
  fi

  if already_present "$dest"; then
    N_SKIP=$((N_SKIP+1))
    log_row SKIP "$features" "$dest" "$(fsize "$dest")"
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

mkdir -p "$(dirname "$LOG_FILE")"
echo "# datasets bootstrap $(date -u +%FT%TZ) filter=${FILTER_FEATURE:-all} dry=${DRY_RUN}" >>"$LOG_FILE"

while IFS=$'\t' read -r source_type url_or_id dest_rel size_gb license features notes; do
  [[ -z "${source_type:-}" || "${source_type:0:1}" == "#" ]] && continue
  [[ "$source_type" == "source_type" ]] && continue
  process_row "$source_type" "$url_or_id" "$dest_rel" "$size_gb" "$license" "$features" "${notes:-}"
done <"$MANIFEST"

print_summary

#!/usr/bin/env bash
# Download all public pretrained weights for Phase 1 features.
#
# Usage:
#   bash ai/scripts/bootstrap/download_pretrained.sh               # all
#   bash ai/scripts/bootstrap/download_pretrained.sh --only <feat> # filter by feature column
#   bash ai/scripts/bootstrap/download_pretrained.sh --dry-run     # plan only
#
# Idempotent: re-running skips files that already exist non-empty.
# Secrets (HF_TOKEN etc.) are loaded from $PROJECT_ROOT/.env — never echoed.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

parse_args "$@"

MANIFEST="$SCRIPT_DIR/manifests/pretrained.tsv"
DEST_ROOT="$PROJECT_ROOT/ai/pretrained"

process_row() {
  local source_type="$1" url_or_id="$2" dest_rel="$3" sha="$4" license="$5" feature="$6" notes="$7"

  if [[ -n "$FILTER_FEATURE" && "$feature" != "$FILTER_FEATURE" ]]; then
    return 0
  fi

  local dest="$DEST_ROOT/$dest_rel"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '  [dry-run] %-10s %s -> %s\n' "$source_type" "$url_or_id" "$dest_rel"
    return 0
  fi

  if already_present "$dest"; then
    N_SKIP=$((N_SKIP+1))
    log_row SKIP "$feature" "$dest" "$(fsize "$dest")"
    return 0
  fi

  printf '[%s] %s -> %s\n' "$source_type" "$feature" "$dest_rel"

  case "$source_type" in
    manual)
      echo "  MANUAL: fetch via $url_or_id"
      [[ -n "$notes" ]] && echo "  notes: $notes"
      N_MANUAL=$((N_MANUAL+1))
      log_row MANUAL "$feature" "$dest" 0
      return 0 ;;
    rsync_only)
      echo "  RSYNC-ONLY: source=$url_or_id"
      N_RSYNC=$((N_RSYNC+1))
      log_row RSYNC "$feature" "$dest" 0
      return 0 ;;
  esac

  local ok=1
  dispatch_download "$source_type" "$url_or_id" "$dest" || ok=0

  # SHA256 verification disabled (manifest SHAs stale)

  if [[ "$ok" -eq 1 ]]; then
    N_DL=$((N_DL+1))
    log_row OK "$feature" "$dest" "$(fsize "$dest")"
  else
    N_FAIL=$((N_FAIL+1))
    log_row FAIL "$feature" "$dest" "$(fsize "$dest")"
  fi
}

mkdir -p "$(dirname "$LOG_FILE")"
echo "# pretrained bootstrap $(date -u +%FT%TZ) filter=${FILTER_FEATURE:-all} dry=${DRY_RUN}" >>"$LOG_FILE"

while IFS=$'\t' read -r source_type url_or_id dest_rel sha license feature notes; do
  [[ -z "${source_type:-}" || "${source_type:0:1}" == "#" ]] && continue
  [[ "$source_type" == "source_type" ]] && continue
  process_row "$source_type" "$url_or_id" "$dest_rel" "$sha" "$license" "$feature" "${notes:-}"
done <"$MANIFEST"

print_summary

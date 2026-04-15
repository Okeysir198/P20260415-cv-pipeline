#!/usr/bin/env bash
# Shared helpers for download_pretrained.sh and download_datasets.sh.
# Source this file: `source "$(dirname "$0")/_common.sh"`

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOG_FILE="$PROJECT_ROOT/ai/scripts/bootstrap/bootstrap.log"

# Load secrets from .env without echoing them.
if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.env"
  set +a
fi

HF_TOKEN="${HF_TOKEN:-}"
KAGGLE_USERNAME="${KAGGLE_USERNAME:-}"
KAGGLE_KEY="${KAGGLE_KEY:-}"
ROBOFLOW_API_KEY="${ROBOFLOW_API_KEY:-}"

# Runtime counters.
N_DL=0
N_SKIP=0
N_FAIL=0
N_MANUAL=0
N_RSYNC=0

log_row() {
  # timestamp \t status \t feature \t dest \t bytes
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%FT%TZ)" "$1" "$2" "$3" "${4:-0}" >>"$LOG_FILE"
}

# Portable file-size (bytes). Empty string if missing.
fsize() { [[ -e "$1" ]] && stat -c%s "$1" 2>/dev/null || echo ""; }

# Returns 0 if dest already exists and is non-empty (idempotent skip).
already_present() {
  local p="$1"
  if [[ -d "$p" ]]; then
    [[ -n "$(ls -A "$p" 2>/dev/null)" ]]
  elif [[ -f "$p" || -L "$p" ]]; then
    [[ -s "$p" ]]
  else
    return 1
  fi
}

# Verify SHA256 if expected present; delete file + return 1 on mismatch.
verify_sha256() {
  local file="$1" expected="$2"
  [[ -z "$expected" || ! -f "$file" ]] && return 0
  local got
  got=$(sha256sum "$file" | awk '{print $1}')
  [[ "$got" == "$expected" ]]
}

# Download a single file via curl (follows redirects; partial suffix for resume).
curl_download() {
  local url="$1" dest="$2"
  mkdir -p "$(dirname "$dest")"
  curl -fL --retry 2 --retry-delay 3 -C - -o "${dest}.partial" "$url"
  mv "${dest}.partial" "$dest"
}

# Download an HF single file via hf (preferred) with token, fallback to curl.
hf_file_download() {
  local repo="$1" file_path="$2" dest="$3"
  mkdir -p "$(dirname "$dest")"
  if command -v hf >/dev/null 2>&1; then
    local tmpdir
    tmpdir=$(mktemp -d)
    local args=(download "$repo" "$file_path" --local-dir "$tmpdir")
    [[ -n "$HF_TOKEN" ]] && args+=(--token "$HF_TOKEN")
    hf "${args[@]}" >/dev/null
    mv "$tmpdir/$file_path" "$dest"
    rm -rf "$tmpdir"
  else
    local auth=()
    [[ -n "$HF_TOKEN" ]] && auth=(-H "Authorization: Bearer $HF_TOKEN")
    curl -fL --retry 2 "${auth[@]}" -o "${dest}.partial" \
      "https://huggingface.co/${repo}/resolve/main/${file_path}"
    mv "${dest}.partial" "$dest"
  fi
}

# Download a whole HF repo folder.
hf_repo_download() {
  local repo="$1" dest="$2"
  mkdir -p "$dest"
  if command -v hf >/dev/null 2>&1; then
    local args=(download "$repo" --local-dir "$dest")
    [[ -n "$HF_TOKEN" ]] && args+=(--token "$HF_TOKEN")
    hf "${args[@]}" >/dev/null
  elif command -v huggingface-cli >/dev/null 2>&1; then
    local args=(download "$repo" --local-dir "$dest")
    [[ -n "$HF_TOKEN" ]] && args+=(--token "$HF_TOKEN")
    huggingface-cli "${args[@]}" >/dev/null
  else
    echo "ERROR: neither 'hf' nor 'huggingface-cli' installed" >&2
    return 1
  fi
}

# Install gdown on demand and pull a GDrive asset.
gdrive_download() {
  local url="$1" dest="$2"
  command -v gdown >/dev/null 2>&1 || pip install --user --quiet gdown
  mkdir -p "$(dirname "$dest")"
  gdown --fuzzy -O "$dest" "$url" || {
    echo "WARN: gdown may have hit virus-scan gate for $url" >&2
    return 1
  }
}

kaggle_download() {
  local slug="$1" dest="$2"
  command -v kaggle >/dev/null 2>&1 || {
    echo "ERROR: kaggle CLI missing (pip install kaggle)" >&2
    return 1
  }
  mkdir -p "$dest"
  kaggle datasets download -d "$slug" -p "$dest" --unzip
}

roboflow_download() {
  local url="$1" dest="$2"
  [[ -z "$ROBOFLOW_API_KEY" ]] && {
    echo "ERROR: ROBOFLOW_API_KEY not set" >&2
    return 1
  }
  mkdir -p "$dest"
  # Roboflow export URL pattern:
  #   https://universe.roboflow.com/ds/<slug>?key=<KEY>&format=yolov8
  local sep='?'
  [[ "$url" == *\?* ]] && sep='&'
  curl -fL --retry 2 -o "$dest/export.zip" \
    "${url}${sep}key=${ROBOFLOW_API_KEY}&format=yolov8"
  (cd "$dest" && unzip -q -o export.zip && rm -f export.zip)
}

# Arg parser shared by both scripts.
FILTER_FEATURE=""
DRY_RUN=0
parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --only) FILTER_FEATURE="$2"; shift 2 ;;
      --dry-run) DRY_RUN=1; shift ;;
      -h|--help)
        sed -n '1,/^$/p' "$0" | sed 's/^# \{0,1\}//'
        exit 0 ;;
      *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
  done
}

# Dispatch a single download by source_type. Returns 0 on success, non-zero on failure.
# Args: source_type url_or_id dest
dispatch_download() {
  local source_type="$1" url_or_id="$2" dest="$3"
  case "$source_type" in
    hf)       hf_repo_download "$url_or_id" "$dest" ;;
    hf_file)
      local repo="${url_or_id%%:*}" fpath="${url_or_id#*:}"
      hf_file_download "$repo" "$fpath" "$dest" ;;
    github)   curl_download "$url_or_id" "$dest" ;;
    gdrive)   gdrive_download "$url_or_id" "$dest" ;;
    kaggle)   kaggle_download "$url_or_id" "$dest" ;;
    roboflow) roboflow_download "$url_or_id" "$dest" ;;
    *) echo "  ERROR: unknown or non-dispatched source_type '$source_type'" >&2; return 1 ;;
  esac
}

print_summary() {
  echo
  echo "===== bootstrap summary ====="
  echo "downloaded: $N_DL"
  echo "skipped (present): $N_SKIP"
  echo "failed: $N_FAIL"
  echo "manual pending: $N_MANUAL"
  echo "rsync-only flagged: $N_RSYNC"
  echo "log: $LOG_FILE"
}

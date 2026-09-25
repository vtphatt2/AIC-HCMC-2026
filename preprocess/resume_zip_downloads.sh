#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URLS_FILE="$SCRIPT_DIR/download_urls_M01-M10.txt"
DEST_DIR="/workspace/AIC-HCMC-2026/challenge_resources/data/raw_zip_videos"
JOBS=2

usage() {
  cat <<'EOF'
Usage: resume_zip_downloads.sh [--urls-file FILE] [--dest DIR] [--jobs N]

Downloads every URL in the manifest. Existing partial files are resumed and
complete ZIPs are validated and skipped.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --urls-file) URLS_FILE="$2"; shift 2 ;;
    --dest) DEST_DIR="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$URLS_FILE" ]] || { echo "ERROR: URL manifest not found: $URLS_FILE" >&2; exit 1; }
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --jobs must be positive" >&2; exit 2; }
mkdir -p "$DEST_DIR"

download_one() {
  local url="$1" name output remote_size local_size final_size
  name="$(basename "${url%%\?*}")"
  output="$DEST_DIR/$name"

  remote_size="$(
    curl --fail --silent --show-error --location --head \
      --connect-timeout 30 "$url" \
      | tr -d '\r' \
      | awk 'tolower($1) == "content-length:" { value=$2 } END { print value }'
  )" || return 1
  [[ "$remote_size" =~ ^[0-9]+$ ]] || {
    echo "[failed] $name: server did not report Content-Length" >&2
    return 1
  }

  local_size=0
  [[ -f "$output" ]] && local_size="$(stat -c %s "$output")"
  if (( local_size == remote_size )); then
    if python3 - "$output" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as archive:
    archive.infolist()
PY
    then
      echo "[skip complete] $name ($local_size bytes)"
      return 0
    fi
    echo "[failed] $name has complete size but an invalid ZIP directory; remove it before retrying" >&2
    return 1
  fi
  if (( local_size > remote_size )); then
    echo "[failed] $name local size $local_size exceeds remote size $remote_size" >&2
    return 1
  fi

  echo "[resume] $name: $local_size/$remote_size bytes"
  curl --fail --location --continue-at - \
    --retry 20 --retry-all-errors --retry-delay 10 \
    --connect-timeout 30 --speed-limit 1024 --speed-time 120 \
    --silent --show-error --output "$output" "$url" || {
      echo "[failed] $name retained at $(stat -c %s "$output" 2>/dev/null || echo 0) bytes" >&2
      return 1
    }

  final_size="$(stat -c %s "$output")"
  if (( final_size != remote_size )); then
    echo "[failed] $name size $final_size; expected $remote_size" >&2
    return 1
  fi
  python3 - "$output" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as archive:
    count = len(archive.infolist())
print(f"[complete] {sys.argv[1]}: {count} entries")
PY
}

export DEST_DIR
export -f download_one

mapfile -t URLS < <(
  sed 's/#.*//' "$URLS_FILE" | awk 'NF { print $0 }'
)
((${#URLS[@]} > 0)) || { echo "ERROR: URL manifest is empty" >&2; exit 1; }

echo "Downloading ${#URLS[@]} archive(s) with $JOBS concurrent job(s) into $DEST_DIR"
printf '%s\0' "${URLS[@]}" \
  | xargs -0 -r -n 1 -P "$JOBS" bash -c 'download_one "$1"' _

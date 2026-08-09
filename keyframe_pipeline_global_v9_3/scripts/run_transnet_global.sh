#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/runtime_env.sh"

# Phase 1 only (TransNet scene detection, global GPU batching) from a URL.
# Downloads the full ZIP first (streaming download runs TransNet per-video,
# not batched), then runs process_video_zip_gpu.py --phase transnet.
usage() {
  cat <<'USAGE'
Usage: scripts/run_transnet_global.sh --url URL [--out-dir DIR] [--zip PATH] [extra process_video_zip_gpu.py flags...]

Example (smoke test on first 5 videos):
  scripts/run_transnet_global.sh --url https://host/Videos_L28_a.zip --limit 5
USAGE
}

URL=""
OUT_DIR="$ROOT/out"
ZIP_PATH="$ROOT/source.zip"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url) URL="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --zip) ZIP_PATH="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

[[ -n "$URL" ]] || { echo "ERROR: --url is required" >&2; usage >&2; exit 2; }

mkdir -p "$OUT_DIR" "$(dirname "$ZIP_PATH")"

echo "=== downloading $URL -> $ZIP_PATH ==="
if command -v aria2c >/dev/null 2>&1; then
  aria2c -x16 -s16 -k16M --file-allocation=none --continue=true \
    --console-log-level=warn --summary-interval=5 \
    -d "$(dirname "$ZIP_PATH")" -o "$(basename "$ZIP_PATH")" "$URL"
else
  curl -L --fail -o "$ZIP_PATH" "$URL"
fi

echo "=== running TransNet phase (global batching) ==="
"$PYTHON_BIN" "$ROOT/src/process_video_zip_gpu.py" \
  --zip "$ZIP_PATH" \
  --out-dir "$OUT_DIR" \
  --phase transnet \
  --transnet-mode global \
  --no-jsonl \
  "${EXTRA_ARGS[@]}"

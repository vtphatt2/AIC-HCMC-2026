#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'USAGE'
Usage: scripts/fetch_model_url.sh --url URL [--out-dir DIR] [--connections N]

Downloads a model checkpoint file (e.g. model.safetensors) from any plain
HTTP(S) URL via aria2c (multi-connection) -- for hosting your own copy
somewhere with better routing to a given rental than HF/Kaggle's CDNs
(e.g. the same host you already use for source video ZIPs, a Cloudflare
R2/S3/Backblaze bucket, etc.). No auth handling -- upload the file
somewhere with plain/public HTTPS access first.

Options:
  --url URL          required
  --out-dir DIR       default: ./local_model
  --connections N        aria2 connections (default 16)

After it finishes, run the pipeline with:
  --model-source local --model-dir <out-dir> --model-arch vit_pe_core_gigantic_patch14_448
USAGE
}

URL=""
OUT_DIR="$ROOT/local_model"
CONNECTIONS="16"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url) URL="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --connections) CONNECTIONS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$URL" ]] || { echo "ERROR: --url is required" >&2; usage >&2; exit 2; }
command -v aria2c >/dev/null 2>&1 || { echo "ERROR: aria2c not found (bash setup.sh installs it)" >&2; exit 1; }

FILE="$(basename "${URL%%\?*}")"
mkdir -p "$OUT_DIR"

echo "=== downloading $URL ($CONNECTIONS connections) ==="
aria2c -x "$CONNECTIONS" -s "$CONNECTIONS" -k 16M --file-allocation=none --continue=true \
  --console-log-level=warn --summary-interval=5 \
  -d "$OUT_DIR" -o "$FILE" "$URL"

echo ""
echo "Done. Run the pipeline with:"
echo "  --model-source local --model-dir $OUT_DIR --model-arch vit_pe_core_gigantic_patch14_448"

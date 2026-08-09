#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'USAGE'
Usage: scripts/fetch_kaggle_model_fast.sh [options]

Downloads a Kaggle model version via the public download API and, if Kaggle
redirects to a signed direct URL (it does for GCS-backed models), fetches
that URL with aria2c (multi-connection) instead of kagglehub's single-stream
per-file downloader -- much faster for one large checkpoint file.

Requires Kaggle API credentials: either $KAGGLE_USERNAME/$KAGGLE_KEY, or
~/.kaggle/kaggle.json (same credentials kagglehub/the kaggle CLI already use).

Options:
  --owner OWNER        default: meowluvmatcha
  --model MODEL        default: lufina
  --framework FW        default: transformers
  --variation VAR        default: default
  --version N              default: 1
  --out-dir DIR              default: ./kaggle_model (extracted here)
  --connections N               aria2 connections (default 16)

After it finishes, run the pipeline with:
  --model-source local --model-dir <out-dir>
USAGE
}

OWNER="meowluvmatcha"
MODEL="lufina"
FRAMEWORK="transformers"
VARIATION="default"
VERSION="1"
OUT_DIR="$ROOT/kaggle_model"
CONNECTIONS="16"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --owner) OWNER="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --framework) FRAMEWORK="$2"; shift 2 ;;
    --variation) VARIATION="$2"; shift 2 ;;
    --version) VERSION="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --connections) CONNECTIONS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ]]; then
  if [[ -f "$HOME/.kaggle/kaggle.json" ]]; then
    KAGGLE_USERNAME="$(python3 -c "import json; print(json.load(open('$HOME/.kaggle/kaggle.json'))['username'])")"
    KAGGLE_KEY="$(python3 -c "import json; print(json.load(open('$HOME/.kaggle/kaggle.json'))['key'])")"
  else
    echo "ERROR: set \$KAGGLE_USERNAME/\$KAGGLE_KEY or create ~/.kaggle/kaggle.json" >&2
    exit 1
  fi
fi

API_URL="https://www.kaggle.com/api/v1/models/$OWNER/$MODEL/$FRAMEWORK/$VARIATION/$VERSION/download"
ARCHIVE="$(mktemp -u --suffix=.tar.gz)"
mkdir -p "$OUT_DIR"

echo "=== resolving download URL ==="
HEADERS="$(curl -s -D - -o /dev/null -u "$KAGGLE_USERNAME:$KAGGLE_KEY" "$API_URL")"
REAL_URL="$(printf '%s' "$HEADERS" | grep -i '^location:' | tail -1 | tr -d '\r' | sed -E 's/^[Ll]ocation: *//')"

if [[ -n "$REAL_URL" ]] && command -v aria2c >/dev/null 2>&1; then
  echo "=== got signed direct URL; downloading with aria2c ($CONNECTIONS connections) ==="
  aria2c -x "$CONNECTIONS" -s "$CONNECTIONS" -k 16M --file-allocation=none --continue=true \
    --console-log-level=warn --summary-interval=5 \
    -d "$(dirname "$ARCHIVE")" -o "$(basename "$ARCHIVE")" "$REAL_URL"
else
  echo "=== no redirect resolved (or aria2c missing); falling back to authenticated single-stream download ==="
  curl -L -u "$KAGGLE_USERNAME:$KAGGLE_KEY" -o "$ARCHIVE" "$API_URL"
fi

echo "=== extracting into $OUT_DIR ==="
tar -xzf "$ARCHIVE" -C "$OUT_DIR"
rm -f "$ARCHIVE"

echo ""
echo "Done. Run the pipeline with:"
echo "  --model-source local --model-dir $OUT_DIR"

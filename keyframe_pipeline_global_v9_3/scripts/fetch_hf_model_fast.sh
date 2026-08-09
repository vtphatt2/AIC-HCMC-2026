#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'USAGE'
Usage: scripts/fetch_hf_model_fast.sh [options]

Downloads the PE-Core checkpoint straight from Hugging Face's public resolve
URL (https://huggingface.co/{repo}/resolve/main/{file}) via aria2c
(multi-connection). This is the same file huggingface_hub/timm would
download, just fetched with real parallel connections instead of relying on
hf_transfer actually being active -- useful if downloads are still slow
despite HF_HUB_ENABLE_HF_TRANSFER=1 (stale venv without the hf_transfer
package, or a throttled route to HF's CDN).

No auth needed for this public model; only model.safetensors is fetched
(config.json isn't needed for --model-source local, which builds the
architecture from --model-arch and only reads weights off disk).

Options:
  --repo REPO        default: timm/vit_pe_core_gigantic_patch14_448.fb
  --file FILE          default: model.safetensors
  --out-dir DIR          default: ./hf_model
  --connections N          aria2 connections (default 16)

After it finishes, run the pipeline with:
  --model-source local --model-dir <out-dir> --model-arch vit_pe_core_gigantic_patch14_448
USAGE
}

REPO="timm/vit_pe_core_gigantic_patch14_448.fb"
FILE="model.safetensors"
OUT_DIR="$ROOT/hf_model"
CONNECTIONS="16"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --file) FILE="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --connections) CONNECTIONS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

command -v aria2c >/dev/null 2>&1 || { echo "ERROR: aria2c not found (bash setup.sh installs it)" >&2; exit 1; }

URL="https://huggingface.co/$REPO/resolve/main/$FILE"
mkdir -p "$OUT_DIR"

echo "=== downloading $URL ($CONNECTIONS connections) ==="
aria2c -x "$CONNECTIONS" -s "$CONNECTIONS" -k 16M --file-allocation=none --continue=true \
  --console-log-level=warn --summary-interval=5 \
  -d "$OUT_DIR" -o "$FILE" "$URL"

echo ""
echo "Done. Run the pipeline with:"
echo "  --model-source local --model-dir $OUT_DIR --model-arch vit_pe_core_gigantic_patch14_448"

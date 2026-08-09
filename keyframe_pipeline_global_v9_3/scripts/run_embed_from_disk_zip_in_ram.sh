#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/runtime_env.sh"
ZIP_SOURCE="${1:-/workspace/Videos_L28_a.zip}"
OUT="${2:-/workspace/output_L28_a}"
shift $(( $# >= 2 ? 2 : $# )) || true
exec "$ROOT/scripts/run_ram_pipeline.sh" \
  --zip-source "$ZIP_SOURCE" \
  --out-dir "$OUT" \
  --skip-transnet \
  "$@"

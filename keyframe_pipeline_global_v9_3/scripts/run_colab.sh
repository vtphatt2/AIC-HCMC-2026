#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/runtime_env.sh"

# Colab convenience wrapper. All core logic remains in run_pipeline.sh.
# If neither --url nor --zip is supplied, use the current AIC L28_a dataset.
DEFAULT_URL="${AIC_VIDEO_URL:-https://aic-data.ledo.io.vn/Videos_L28_a.zip}"
BATCH_SIZE="${COLAB_BATCH_SIZE:-16}"
PREFETCH="${COLAB_PREFETCH_BATCHES:-3}"

has_source=0
for arg in "$@"; do
  case "$arg" in
    --url|--zip) has_source=1 ;;
  esac
done

SOURCE_ARGS=()
if [[ "$has_source" -eq 0 ]]; then
  SOURCE_ARGS=(--url "$DEFAULT_URL")
fi

exec "$ROOT/run_pipeline.sh" \
  "${SOURCE_ARGS[@]}" \
  --profile colab \
  --amp auto \
  --tf32 \
  --batch-size "$BATCH_SIZE" \
  --prefetch-batches "$PREFETCH" \
  "$@"

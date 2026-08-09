#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/runtime_env.sh"

# Sweep --batch-size for the PE-Core embed phase and report throughput + avg
# GPU utilization per size, so you can pick a batch size by evidence instead
# of guessing. Requires TransNet metadata to already exist in --out-dir
# (run it once automatically if missing).
usage() {
  cat <<'USAGE'
Usage: scripts/bench_batch_size.sh --zip PATH --out-dir DIR [options]

Options:
  --zip PATH            Source ZIP (required)
  --out-dir DIR          Working dir; reused across batch sizes (required)
  --limit N              Videos to use for the benchmark (default: 5)
  --batch-sizes "A B C"  Space-separated sizes to try (default: "16 32 64 96 128")
  extra flags (--amp, --tf32, --device, ...) are forwarded to process_video_zip_gpu.py

Example:
  scripts/bench_batch_size.sh --zip ./source.zip --out-dir ./bench_out --limit 5
USAGE
}

ZIP=""
OUT_DIR=""
LIMIT=5
BATCH_SIZES="16 32 64 96 128"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zip) ZIP="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --batch-sizes) BATCH_SIZES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

[[ -n "$ZIP" && -n "$OUT_DIR" ]] || { echo "ERROR: --zip and --out-dir are required" >&2; usage >&2; exit 2; }

mkdir -p "$OUT_DIR"

if ! ls "$OUT_DIR"/*/keyframes.json >/dev/null 2>&1; then
  echo "=== no TransNet metadata found in $OUT_DIR; running Phase 1 once ==="
  "$PYTHON_BIN" "$ROOT/src/process_video_zip_gpu.py" \
    --zip "$ZIP" --out-dir "$OUT_DIR" --phase transnet \
    --limit "$LIMIT" --no-jsonl
fi

HAVE_NVIDIA_SMI=0
command -v nvidia-smi >/dev/null 2>&1 && HAVE_NVIDIA_SMI=1

RESULTS_FILE="$(mktemp)"
trap 'rm -f "$RESULTS_FILE"' EXIT

for BS in $BATCH_SIZES; do
  echo ""
  echo "=== batch-size=$BS ==="

  UTIL_LOG=""
  SAMPLER_PID=""
  if [[ "$HAVE_NVIDIA_SMI" -eq 1 ]]; then
    UTIL_LOG="$(mktemp)"
    ( while true; do
        nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits >> "$UTIL_LOG"
        sleep 1
      done ) &
    SAMPLER_PID=$!
  fi

  "$PYTHON_BIN" "$ROOT/src/process_video_zip_gpu.py" \
    --zip "$ZIP" --out-dir "$OUT_DIR" --phase embed \
    --limit "$LIMIT" --batch-size "$BS" --overwrite --no-jsonl \
    "${EXTRA_ARGS[@]}"

  AVG_UTIL="n/a"
  if [[ -n "$SAMPLER_PID" ]]; then
    kill "$SAMPLER_PID" 2>/dev/null || true
    wait "$SAMPLER_PID" 2>/dev/null || true
    if [[ -s "$UTIL_LOG" ]]; then
      AVG_UTIL="$(awk '{sum+=$1; n++} END {if (n>0) printf "%.0f", sum/n; else print "n/a"}' "$UTIL_LOG")"
    fi
    rm -f "$UTIL_LOG"
  fi

  IMG_PER_S="$("$PYTHON_BIN" - "$OUT_DIR" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
total_kf = 0
gpu_s = 0.0
for p in out.glob("*/embedding_stats.json"):
    try:
        d = json.loads(p.read_text())
    except Exception:
        continue
    total_kf += int(d.get("num_keyframes") or 0)
    gpu_s += float((d.get("timing_seconds") or {}).get("gpu_embedding") or 0)
print(f"{total_kf/gpu_s:.2f}" if gpu_s > 0 else "n/a")
PY
)"

  echo "batch=$BS gpu_util_avg=${AVG_UTIL}% embedding_gpu_images_per_s=$IMG_PER_S" | tee -a "$RESULTS_FILE"
done

echo ""
echo "=== summary ==="
printf "%-8s %-14s %-28s\n" "batch" "gpu_util_avg%" "embedding_gpu_images_per_s"
while read -r line; do
  bs="$(sed -n 's/.*batch=\([0-9]*\).*/\1/p' <<<"$line")"
  util="$(sed -n 's/.*gpu_util_avg=\([^ %]*\)%.*/\1/p' <<<"$line")"
  ips="$(sed -n 's/.*images_per_s=\(.*\)/\1/p' <<<"$line")"
  printf "%-8s %-14s %-28s\n" "$bs" "$util" "$ips"
done < "$RESULTS_FILE"

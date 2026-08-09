#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/runtime_env.sh"
URL="https://aic-data.ledo.io.vn/Videos_L28_a.zip"
OUT="/workspace/output_L28_a"
RAM_ROOT="/dev/shm/keyframe_pipeline_runtime"
AMP="auto"
TF32=1
COMPILE=0
BATCH_SIZE="2"
PREFETCH_BATCHES="2"
DEVICE="cuda"
KEEP_RAM=0
SKIP_TRANSNET=0
ZIP_SOURCE=""

usage() {
  cat <<'USAGE'
Usage: run_ram_pipeline.sh [options]

Designed for a small disk + large RAM GPU VM (e.g. 16GB disk, 64GB RAM, RTX 4090 24GB).
The video ZIP and Hugging Face cache live in RAM; only metadata/embeddings go to --out-dir.

Options:
  --url URL                    ZIP URL (default: AIC Videos_L28_a.zip)
  --zip-source PATH            Existing complete ZIP to COPY into RAM instead of downloading
  --out-dir PATH               Persistent output directory (default: /workspace/output_L28_a)
  --ram-dir PATH               RAM working dir (default: /dev/shm/keyframe_pipeline_runtime)
  --amp auto|off|bf16|fp16     autocast compute (default auto), FP32 weights
  --[no-]tf32                  TF32 enabled by default
  --compile                    optional torch.compile
  --batch-size N               PE-Core batch size (default: 2)
  --prefetch-batches N         CPU batches queued ahead (default: 2)
  --device DEVICE              default: cuda
  --skip-transnet              Reuse existing scenes.json/keyframes.json and run embed only
  --keep-ram                   Keep ZIP + HF cache in RAM after completion
  -h, --help

Examples:
  ./run_ram_pipeline.sh --amp auto --tf32 --batch-size 2
  ./run_ram_pipeline.sh --zip-source /workspace/Videos_L28_a.zip --skip-transnet
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url) URL="$2"; shift 2 ;;
    --zip-source) ZIP_SOURCE="$2"; shift 2 ;;
    --out-dir) OUT="$2"; shift 2 ;;
    --ram-dir) RAM_ROOT="$2"; shift 2 ;;
    --amp) AMP="$2"; shift 2 ;;
    --precision) case "$2" in fp32) AMP="off";; fp16|bf16) AMP="$2";; esac; shift 2 ;;
    --tf32) TF32=1; shift ;;
    --no-tf32) TF32=0; shift ;;
    --compile) COMPILE=1; shift ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --prefetch-batches) PREFETCH_BATCHES="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --skip-transnet) SKIP_TRANSNET=1; shift ;;
    --keep-ram) KEEP_RAM=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$AMP" in auto|off|fp16|bf16) ;; *) echo "Invalid --amp: $AMP" >&2; exit 2 ;; esac

mkdir -p "$RAM_ROOT" "$OUT"
RAM_ROOT="$(cd "$RAM_ROOT" && pwd)"
ZIP_RAM="$RAM_ROOT/Videos_L28_a.zip"
HF_HOME_RAM="$RAM_ROOT/hf_home"
TMP_RAM="$RAM_ROOT/tmp"
mkdir -p "$HF_HOME_RAM" "$TMP_RAM"

# Put all transient caches in RAM, not the 16GB overlay disk.
export HF_HOME="$HF_HOME_RAM"
export HF_HUB_CACHE="$HF_HOME_RAM/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME_RAM/hub"
export XDG_CACHE_HOME="$RAM_ROOT/xdg_cache"
export TMPDIR="$TMP_RAM"
export PIP_CACHE_DIR="$RAM_ROOT/pip_cache"
export PYTHONUNBUFFERED=1

# If this package has an isolated Pillow/deps dir, use it first.
if [[ -d "$ROOT/.python_deps" ]]; then
  export PYTHONPATH="$ROOT/.python_deps${PYTHONPATH:+:$PYTHONPATH}"
fi

avail_bytes=$(df -PB1 "$RAM_ROOT" | awk 'NR==2 {print $4}')
# ZIP ~4.1GB + PE-Core ~7.6GB + HF temp/cache + preprocess queues. 16GiB is a practical floor.
min_bytes=$((16 * 1024 * 1024 * 1024))
if (( avail_bytes < min_bytes )); then
  echo "ERROR: RAM filesystem has only $((avail_bytes/1024/1024/1024)) GiB free; need ~16 GiB or more." >&2
  echo "Check: df -h /dev/shm ; free -h" >&2
  exit 1
fi

echo "=== RAM pipeline configuration ==="
echo "RAM root:   $RAM_ROOT"
echo "RAM free:   $(df -h "$RAM_ROOT" | awk 'NR==2 {print $4}')"
echo "HF cache:   $HF_HOME_RAM"
echo "ZIP in RAM: $ZIP_RAM"
echo "Output:     $OUT"
echo "GPU:        $DEVICE"
echo "PE-Core:    weights=fp32 amp=$AMP tf32=$TF32 compile=$COMPILE batch=$BATCH_SIZE prefetch=$PREFETCH_BATCHES"

cleanup() {
  if [[ "$KEEP_RAM" -eq 0 ]]; then
    echo "[cleanup] deleting RAM runtime: $RAM_ROOT"
    rm -rf "$RAM_ROOT"
  else
    echo "[keep-ram] runtime kept at $RAM_ROOT"
  fi
}
trap cleanup EXIT

if [[ -n "$ZIP_SOURCE" ]]; then
  if [[ ! -f "$ZIP_SOURCE" ]]; then echo "ZIP source not found: $ZIP_SOURCE" >&2; exit 1; fi
  echo "=== Copy existing ZIP to RAM ==="
  cp --reflink=auto --sparse=always "$ZIP_SOURCE" "$ZIP_RAM" 2>/dev/null || cp "$ZIP_SOURCE" "$ZIP_RAM"
  ls -lh "$ZIP_RAM"

  if [[ "$SKIP_TRANSNET" -eq 0 ]]; then
    echo "=== TransNet phase (ZIP already complete in RAM) ==="
    "$PYTHON_BIN" "$ROOT/src/process_video_zip_gpu.py" \
      --zip "$ZIP_RAM" --out-dir "$OUT" --phase transnet --device "$DEVICE"
  fi
else
  if [[ "$SKIP_TRANSNET" -eq 1 ]]; then
    echo "ERROR: --skip-transnet with no --zip-source is not useful because embedding still needs the complete ZIP." >&2
    exit 2
  fi
  echo "=== Streaming download -> RAM + TransNet ==="
  "$PYTHON_BIN" "$ROOT/src/stream_download_transnet.py" \
    --url "$URL" \
    --zip "$ZIP_RAM" \
    --out-dir "$OUT" \
    --device "$DEVICE"
fi

echo "=== PE-Core vision-only embedding ==="
echo "Weights/cache are downloaded into RAM: $HF_HOME_RAM"
EMBED_ARGS=(
  --zip "$ZIP_RAM" --out-dir "$OUT" --phase embed
  --amp "$AMP" --batch-size "$BATCH_SIZE" --prefetch-batches "$PREFETCH_BATCHES"
  --device "$DEVICE" --no-jsonl
)
if [[ "$TF32" -eq 1 ]]; then EMBED_ARGS+=(--tf32); else EMBED_ARGS+=(--no-tf32); fi
if [[ "$COMPILE" -eq 1 ]]; then EMBED_ARGS+=(--compile); fi
"$PYTHON_BIN" "$ROOT/src/process_video_zip_gpu.py" "${EMBED_ARGS[@]}"

echo "=== Finished ==="
find "$OUT" -name embeddings.npy | wc -l | xargs echo "embedding files:"

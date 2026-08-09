#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/runtime_env.sh"

ZIP_PATH=""
OUT_DIR=""
HF_CACHE_DIR="/dev/shm/aic_keyframe_pipeline/hf_home"
TEMP_DIR="/dev/shm/aic_keyframe_pipeline/tmp"
BATCH_SIZE="64"
PREFETCH_BATCHES="3"
EMBED_FFMPEG_PREPROCESS=1
EMBED_DECODE_MODE="auto"
EMBED_SEEK_GAP_SECONDS="2.0"
EMBED_SEEK_MIN_SAVINGS="0.30"
EMBED_SEEK_MAX_GROUPS="128"
AMP="auto"
TF32=1
COMPILE=0
DEVICE="cuda"
LOCAL_FILES_ONLY=0
MODEL_SOURCE="huggingface"
KAGGLE_MODEL="meowluvmatcha/lufina/transformers/default"
MODEL_DIR=""
KAGGLE_CACHE_DIR="/dev/shm/aic_keyframe_pipeline/kagglehub"

usage() {
  cat <<'USAGE'
Usage: run_embed_only.sh --zip PATH --out-dir PATH [options]

Options:
  --model-source huggingface|kaggle|local   default huggingface
  --kaggle-model HANDLE     default meowluvmatcha/lufina/transformers/default
  --model-dir PATH          required for --model-source local
  --hf-cache-dir PATH       default /dev/shm/aic_keyframe_pipeline/hf_home
  --kaggle-cache-dir PATH   default /dev/shm/aic_keyframe_pipeline/kagglehub
  --temp-dir PATH           default /dev/shm/aic_keyframe_pipeline/tmp
  --amp auto|off|bf16|fp16   default auto
  --[no-]tf32                 default tf32 enabled
  --compile                    optional torch.compile
  --batch-size N            default 64
  --prefetch-batches N      default 3
  --[no-]embed-ffmpeg-preprocess
  --embed-decode-mode auto|sequential|seek
  --embed-seek-gap-seconds N
  --embed-seek-min-savings F
  --embed-seek-max-groups N
  --device DEVICE           default cuda
  --local-files-only
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zip) ZIP_PATH="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --model-source) MODEL_SOURCE="$2"; shift 2 ;;
    --kaggle-model) KAGGLE_MODEL="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --hf-cache-dir) HF_CACHE_DIR="$2"; shift 2 ;;
    --kaggle-cache-dir) KAGGLE_CACHE_DIR="$2"; shift 2 ;;
    --temp-dir) TEMP_DIR="$2"; shift 2 ;;
    --amp) AMP="$2"; shift 2 ;;
    --precision) case "$2" in fp32) AMP="off";; fp16|bf16) AMP="$2";; esac; shift 2 ;;
    --tf32) TF32=1; shift ;;
    --no-tf32) TF32=0; shift ;;
    --compile) COMPILE=1; shift ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --prefetch-batches) PREFETCH_BATCHES="$2"; shift 2 ;;
    --embed-ffmpeg-preprocess) EMBED_FFMPEG_PREPROCESS=1; shift ;;
    --no-embed-ffmpeg-preprocess) EMBED_FFMPEG_PREPROCESS=0; shift ;;
    --embed-decode-mode) EMBED_DECODE_MODE="$2"; shift 2 ;;
    --embed-seek-gap-seconds) EMBED_SEEK_GAP_SECONDS="$2"; shift 2 ;;
    --embed-seek-min-savings) EMBED_SEEK_MIN_SAVINGS="$2"; shift 2 ;;
    --embed-seek-max-groups) EMBED_SEEK_MAX_GROUPS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --local-files-only) LOCAL_FILES_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$ZIP_PATH" && -n "$OUT_DIR" ]] || { usage >&2; exit 2; }
[[ -f "$ZIP_PATH" ]] || { echo "ERROR: ZIP not found: $ZIP_PATH" >&2; exit 1; }
mkdir -p "$HF_CACHE_DIR" "$TEMP_DIR" "$OUT_DIR" "$KAGGLE_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"
export HF_HUB_CACHE="$HF_CACHE_DIR/hub"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR/hub"
export HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME="$HF_CACHE_DIR/xdg"
export KAGGLEHUB_CACHE="$KAGGLE_CACHE_DIR"
export TMPDIR="$TEMP_DIR"
export PYTHONUNBUFFERED=1
if [[ -d "$ROOT/.python_deps" ]]; then
  export PYTHONPATH="$ROOT/.python_deps${PYTHONPATH:+:$PYTHONPATH}"
fi

ARGS=(
  --zip "$ZIP_PATH" --out-dir "$OUT_DIR" --phase embed
  --amp "$AMP" --batch-size "$BATCH_SIZE"
  --prefetch-batches "$PREFETCH_BATCHES" --device "$DEVICE" --no-jsonl
  --embed-decode-mode "$EMBED_DECODE_MODE"
  --embed-seek-gap-seconds "$EMBED_SEEK_GAP_SECONDS"
  --embed-seek-min-savings "$EMBED_SEEK_MIN_SAVINGS"
  --embed-seek-max-groups "$EMBED_SEEK_MAX_GROUPS"
  --model-source "$MODEL_SOURCE" --kaggle-model "$KAGGLE_MODEL" --kaggle-cache-dir "$KAGGLE_CACHE_DIR"
)
if [[ -n "$MODEL_DIR" ]]; then ARGS+=(--model-dir "$MODEL_DIR"); fi
if [[ "$EMBED_FFMPEG_PREPROCESS" -eq 1 ]]; then ARGS+=(--embed-ffmpeg-preprocess); else ARGS+=(--no-embed-ffmpeg-preprocess); fi
if [[ "$TF32" -eq 1 ]]; then ARGS+=(--tf32); else ARGS+=(--no-tf32); fi
if [[ "$COMPILE" -eq 1 ]]; then ARGS+=(--compile); fi
if [[ "$LOCAL_FILES_ONLY" -eq 1 ]]; then ARGS+=(--local-files-only); fi
exec "$PYTHON_BIN" "$ROOT/src/process_video_zip_gpu.py" "${ARGS[@]}"

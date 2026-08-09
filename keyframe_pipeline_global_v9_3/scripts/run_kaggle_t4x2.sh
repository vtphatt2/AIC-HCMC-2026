#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/runtime_env.sh"
URL=""
ZIP_PATH=""
OUT_DIR=""
ARCHIVE=""
SOURCE_DIR="/kaggle/working/aic_source"
KAGGLE_CACHE_DIR="/kaggle/working/kagglehub"
HF_CACHE_DIR="/kaggle/working/hf_cache"
TEMP_DIR="/kaggle/working/tmp"
LOG_DIR="/kaggle/working/keyframe_pipeline_logs"
WORKERS=2
BATCH_SIZE=16
PREFETCH_BATCHES=2
AMP="auto"
COMPILE=0
TF32=0
LIMIT=""
ENTRY_REGEX=""
MODEL_SOURCE="kaggle"
KAGGLE_MODEL="meowluvmatcha/lufina/transformers/default"
MODEL_DIR=""
MODEL_ID="hf_hub:timm/vit_pe_core_gigantic_patch14_448.fb"

usage() {
  cat <<'USAGE'
Usage:
  run_kaggle_t4x2.sh --url URL [options]
  run_kaggle_t4x2.sh --zip /kaggle/input/.../Videos_*.zip [options]

Optimized launcher for Kaggle 2x T4. It shards videos across two independent
processes; each process owns one GPU and one PE-Core copy. No DDP is used.

Main options:
  --url URL                    Download the source ZIP once to /kaggle/working.
  --zip PATH                   Use an existing ZIP (recommended if attached as Kaggle Dataset).
  --out-dir PATH               Shared output directory; auto-derived from ZIP name.
  --archive PATH               Final result ZIP; auto-derived from ZIP name.
  --workers N                  GPU workers/shards (default 2).
  --batch-size N               Per-GPU global batch size (default 16 for T4 16GB).
  --prefetch-batches N         Per-worker CPU prefetch queue (default 2).
  --amp auto|fp16|bf16|off     default auto; T4 auto resolves to fp16.
  --compile                    Optional torch.compile per worker (off by default).
  --tf32 / --no-tf32           default off for T4 (T4 has no TF32 acceleration).
  --limit N                    Limit global sorted video list before sharding.
  --entry-regex REGEX          Process only matching ZIP entries.

Model options:
  --model-source kaggle|local|huggingface   default kaggle
  --kaggle-model HANDLE                    default meowluvmatcha/lufina/transformers/default
  --model-dir PATH                         for --model-source local
  --model-id ID                            for --model-source huggingface
  --kaggle-cache-dir PATH                  default /kaggle/working/kagglehub
  --hf-cache-dir PATH                      default /kaggle/working/hf_cache
  --temp-dir PATH                          default /kaggle/working/tmp
  --log-dir PATH                           worker logs

Examples:
  # URL source
  bash run_kaggle_t4x2.sh \
    --url "https://aic-data.ledo.io.vn/Videos_L28_a.zip"

  # Existing Kaggle Dataset ZIP (avoids re-download)
  bash run_kaggle_t4x2.sh \
    --zip /kaggle/input/videos-l28-a/Videos_L28_a.zip

  # Smoke test: first 2 videos -> one video per T4
  bash run_kaggle_t4x2.sh --zip /path/Videos_L28_a.zip --limit 2
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url) URL="$2"; shift 2 ;;
    --zip) ZIP_PATH="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --archive) ARCHIVE="$2"; shift 2 ;;
    --source-dir) SOURCE_DIR="$2"; shift 2 ;;
    --kaggle-cache-dir) KAGGLE_CACHE_DIR="$2"; shift 2 ;;
    --hf-cache-dir) HF_CACHE_DIR="$2"; shift 2 ;;
    --temp-dir) TEMP_DIR="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --prefetch-batches) PREFETCH_BATCHES="$2"; shift 2 ;;
    --amp) AMP="$2"; shift 2 ;;
    --compile) COMPILE=1; shift ;;
    --tf32) TF32=1; shift ;;
    --no-tf32) TF32=0; shift ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --entry-regex) ENTRY_REGEX="$2"; shift 2 ;;
    --model-source) MODEL_SOURCE="$2"; shift 2 ;;
    --kaggle-model) KAGGLE_MODEL="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --model-id) MODEL_ID="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$URL" || -n "$ZIP_PATH" ]] || { echo "ERROR: one of --url or --zip is required" >&2; usage >&2; exit 2; }
[[ -z "$URL" || -z "$ZIP_PATH" ]] || { echo "ERROR: use either --url or --zip, not both" >&2; exit 2; }
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --workers must be positive" >&2; exit 2; }
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --batch-size must be positive" >&2; exit 2; }
[[ "$PREFETCH_BATCHES" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --prefetch-batches must be positive" >&2; exit 2; }
case "$AMP" in auto|fp16|bf16|off) ;; *) echo "ERROR: invalid --amp: $AMP" >&2; exit 2 ;; esac
case "$MODEL_SOURCE" in kaggle|local|huggingface) ;; *) echo "ERROR: invalid --model-source: $MODEL_SOURCE" >&2; exit 2 ;; esac
if [[ "$MODEL_SOURCE" == "local" && -z "$MODEL_DIR" ]]; then
  echo "ERROR: --model-dir is required for --model-source local" >&2; exit 2
fi

mkdir -p "$SOURCE_DIR" "$KAGGLE_CACHE_DIR" "$HF_CACHE_DIR" "$TEMP_DIR" "$LOG_DIR"
export HF_HOME="$HF_CACHE_DIR"
export HF_HUB_CACHE="$HF_CACHE_DIR/hub"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR/hub"
export XDG_CACHE_HOME="$HF_CACHE_DIR/xdg"
export KAGGLEHUB_CACHE="$KAGGLE_CACHE_DIR"
export TMPDIR="$TEMP_DIR"
export PYTHONUNBUFFERED=1

GPU_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
if (( GPU_COUNT < WORKERS )); then
  echo "ERROR: requested $WORKERS workers but PyTorch sees only $GPU_COUNT CUDA GPU(s)." >&2
  echo "Enable Kaggle accelerator 'GPU T4 x2' and restart the session." >&2
  exit 1
fi

echo "=== Kaggle multi-GPU configuration ==="
echo "GPUs visible:     $GPU_COUNT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo "Workers:          $WORKERS"
echo "Per-GPU batch:    $BATCH_SIZE"
echo "Per-worker queue: $PREFETCH_BATCHES"
echo "AMP:              $AMP"
echo "TF32:             $TF32 (off is recommended for T4)"
echo "Compile:          $COMPILE"

if [[ -n "$URL" ]]; then
  BASENAME="$(basename "${URL%%\?*}")"
  [[ "$BASENAME" == *.zip ]] || BASENAME="input.zip"
  ZIP_PATH="$SOURCE_DIR/$BASENAME"
  echo "=== Download source ZIP once ==="
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x 16 -s 16 -k 16M --file-allocation=none --continue=true \
      --console-log-level=warn --summary-interval=5 -d "$SOURCE_DIR" -o "$BASENAME" "$URL"
  else
    "$PYTHON_BIN" - "$URL" "$ZIP_PATH" <<'PYDOWNLOAD'
import sys, requests
url, out = sys.argv[1], sys.argv[2]
with requests.get(url, stream=True, timeout=(30, 300)) as r:
    r.raise_for_status()
    with open(out, 'wb', buffering=16*1024*1024) as f:
        for chunk in r.iter_content(8*1024*1024):
            if chunk: f.write(chunk)
print(out)
PYDOWNLOAD
  fi
else
  ZIP_PATH="$(readlink -f "$ZIP_PATH")"
  BASENAME="$(basename "$ZIP_PATH")"
fi
[[ -f "$ZIP_PATH" ]] || { echo "ERROR: ZIP not found: $ZIP_PATH" >&2; exit 1; }

STEM="${BASENAME%.zip}"
DATA_ID="${STEM#Videos_}"
[[ -n "$OUT_DIR" ]] || OUT_DIR="/kaggle/working/output_${DATA_ID}"
[[ -n "$ARCHIVE" ]] || ARCHIVE="/kaggle/working/${DATA_ID}_results.zip"
mkdir -p "$OUT_DIR" "$(dirname "$ARCHIVE")"

"$PYTHON_BIN" - "$ZIP_PATH" <<'PYZIP'
import sys, zipfile
p=sys.argv[1]
with zipfile.ZipFile(p) as z:
    bad=z.testzip()
    if bad: raise RuntimeError(f"Corrupt ZIP member: {bad}")
    vids=[i for i in z.infolist() if i.filename.lower().endswith(('.mp4','.mov','.mkv','.avi','.webm','.m4v'))]
    stored=sum(i.compress_type == zipfile.ZIP_STORED for i in vids)
print(f"ZIP OK: {len(vids)} videos, {stored} ZIP_STORED")
PYZIP

# Warm shared model cache once before workers start, avoiding two simultaneous downloads.
if [[ "$MODEL_SOURCE" == "kaggle" ]]; then
  echo "=== Warm shared Kaggle model cache ==="
  "$PYTHON_BIN" - "$KAGGLE_MODEL" <<'PYMODEL'
import sys, kagglehub
p = kagglehub.model_download(sys.argv[1])
print(f"Kaggle model ready: {p}")
PYMODEL
fi

echo "=== Launch $WORKERS independent GPU shards ==="
pids=()
for ((i=0; i<WORKERS; i++)); do
  args=(
    --zip "$ZIP_PATH"
    --out-dir "$OUT_DIR"
    --phase all
    --num-shards "$WORKERS"
    --shard-index "$i"
    --device cuda
    --amp "$AMP"
    --batch-size "$BATCH_SIZE"
    --prefetch-batches "$PREFETCH_BATCHES"
    --model-source "$MODEL_SOURCE"
    --kaggle-model "$KAGGLE_MODEL"
    --model-id "$MODEL_ID"
    --kaggle-cache-dir "$KAGGLE_CACHE_DIR"
    --no-jsonl
  )
  if [[ "$TF32" -eq 1 ]]; then args+=(--tf32); else args+=(--no-tf32); fi
  if [[ "$COMPILE" -eq 1 ]]; then args+=(--compile); fi
  if [[ -n "$MODEL_DIR" ]]; then args+=(--model-dir "$MODEL_DIR"); fi
  if [[ -n "$LIMIT" ]]; then args+=(--limit "$LIMIT"); fi
  if [[ -n "$ENTRY_REGEX" ]]; then args+=(--entry-regex "$ENTRY_REGEX"); fi

  log="$LOG_DIR/worker_${i}.log"
  echo "[launch] physical GPU $i -> shard $i/$WORKERS ; log=$log"
  (
    set -o pipefail
    CUDA_VISIBLE_DEVICES="$i" "$PYTHON_BIN" "$ROOT/src/process_video_zip_gpu.py" "${args[@]}" 2>&1 \
      | sed -u "s/^/[gpu${i}] /" \
      | tee "$log"
  ) &
  pids+=("$!")
done

failed=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "ERROR: worker $i failed; inspect $LOG_DIR/worker_${i}.log" >&2
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

echo "=== Package merged results ==="
bash "$ROOT/package_results.sh" "$OUT_DIR" "$ARCHIVE"

if [[ -n "$URL" ]]; then
  echo "=== Remove downloaded source ZIP (frees disk before the next dataset) ==="
  rm -f "$ZIP_PATH"
fi

echo "=== DONE ==="
echo "Result:            $ARCHIVE"
echo "Output:            $OUT_DIR"
echo "Worker summaries:  $OUT_DIR/run_summary.shard_*_of_${WORKERS}.json"
echo "Logs:              $LOG_DIR"
echo "embedding files:   $(find "$OUT_DIR" -name embeddings.npy | wc -l)"

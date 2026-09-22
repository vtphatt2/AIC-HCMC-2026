#!/usr/bin/env bash
set -euo pipefail
RUN_STARTED_EPOCH=$(date +%s)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/runtime_env.sh"
URL=""
ZIP_PATH=""
OUT_DIR=""
ARCHIVE=""
PROFILE="balanced"
WORK_ROOT=""
SOURCE_DIR=""
HF_CACHE_DIR=""
TEMP_DIR=""
AMP="auto"
TF32=1
COMPILE=0
BATCH_SIZE="64"
PREFETCH_BATCHES="3"
EMBED_FFMPEG_PREPROCESS=1
EMBED_DECODE_MODE="auto"
EMBED_SEEK_GAP_SECONDS="2.0"
EMBED_SEEK_MIN_SAVINGS="0.30"
EMBED_SEEK_MAX_GROUPS="128"
TRANSNET_MODE="global"
TRANSNET_BATCH_SIZE="16"
TRANSNET_DECODE_WORKERS="4"
TRANSNET_ACTIVE_VIDEOS="8"
TRANSNET_PREFETCH_VIDEOS="8"
TRANSNET_PREFETCH_WINDOWS="256"
TRANSNET_BATCH_TIMEOUT_MS="20"
KEYFRAME_STRATEGY="tiered"
KEYFRAMES_PER_SECOND="0.3"
MIN_KEYFRAMES_PER_SCENE="1"
MAX_KEYFRAMES_PER_SCENE="20"
PARALLEL_STAGES=1
DEVICE="cuda"
LIMIT=""
DOWNLOAD_ENGINE="auto"
CONNECTIONS="16"
DELETE_SOURCE_BEFORE_PACKAGE=0
KEEP_TEMP=0
LOCAL_FILES_ONLY=0
MODEL_SOURCE="huggingface"
KAGGLE_MODEL="meowluvmatcha/lufina/transformers/default"
MODEL_DIR=""
KAGGLE_CACHE_DIR=""

usage() {
  cat <<'USAGE'
Usage:
  run_pipeline.sh --url URL [options]
  run_pipeline.sh --zip /path/existing.zip [options]

Storage profiles:
  balanced   (default) ZIP on /workspace disk, HF cache + temp on /dev/shm.
             Best for Vast.ai-style 15G shm + ~17G disk.
  ram-rich   ZIP + HF cache + temp on /dev/shm. Use when shm is genuinely large.
  disk-rich  ZIP + HF cache + temp on /workspace. Use when disk is large.
  colab      ZIP/cache/temp on /content; safe for Google Colab.

Main options:
  --url URL                    Download a source ZIP.
  --zip PATH                   Use an already-downloaded ZIP; skips download.
  --profile balanced|ram-rich|disk-rich|colab
  --work-root PATH              Root for persistent source/output/archive. Auto-detected safely.
  --out-dir PATH               Persistent phase outputs. Auto-derived from ZIP name.
  --archive PATH               Final result ZIP. Auto-derived from ZIP name.

Storage overrides (override the selected profile):
  --source-dir PATH            Directory for downloaded source ZIP.
  --hf-cache-dir PATH          Hugging Face/PE-Core cache directory.
  --temp-dir PATH              Temporary directory used by ffmpeg/Python/HF.

Model source:
  --model-source huggingface|kaggle|local   default huggingface
  --kaggle-model HANDLE                    default meowluvmatcha/lufina/transformers/default
  --model-dir PATH                         required for --model-source local
  --kaggle-cache-dir PATH                  default alongside profile temp/cache

Compute:
  --amp auto|off|bf16|fp16     Autocast compute mode (default auto); weights stay FP32.
  --[no-]tf32                  TF32 for remaining FP32 ops (default enabled).
  --compile                    Optional torch.compile for PE-Core.
  --batch-size N               PE-Core global batch size (default 64).
  --prefetch-batches N         Prepared CPU batches queued ahead (default 3).
  --[no-]embed-ffmpeg-preprocess  Resize/crop selected frames in ffmpeg from encoder config.
  --embed-decode-mode auto|sequential|seek  Adaptive PE-Core keyframe decode mode.
  --embed-seek-gap-seconds N   Group sparse seeks across gaps <= N seconds (default 2.0).
  --embed-seek-min-savings F   Auto seek only above estimated fractional savings (default 0.30).
  --embed-seek-max-groups N    Max seek groups allowed in auto mode (default 128).
  --transnet-mode global|sequential
                               Global temporal batching (default) or v8 legacy path.
  --transnet-batch-size N      100-frame windows per TransNet GPU forward (default 16).
  --transnet-decode-workers N  Parallel ffmpeg TransNet decoders (default 4).
  --transnet-active-videos N   Legacy compatibility flag; ignored by streaming-global mode.
  --transnet-prefetch-videos N Legacy compatibility flag; ignored by streaming-global mode.
  --transnet-prefetch-windows N 100-frame windows queued ahead of GPU (default 256).
  --transnet-batch-timeout-ms N Flush partial GPU batch after N ms (default 20).
  --keyframe-strategy tiered|linear
                               Per-scene sample count policy (default tiered).
  --keyframes-per-second N     Linear sampling rate (default 0.3).
  --min-keyframes-per-scene N  Linear lower bound per scene (default 1).
  --max-keyframes-per-scene N  Linear upper bound per scene (default 20).
  --parallel-stages           Start PE-Core immediately and consume each video as
                              soon as TransNet publishes it (default).
  --sequential-stages         Wait for all TransNet videos before starting PE-Core.
  --device DEVICE              Default cuda.
  --limit N                    Process only the first N videos (useful for smoke tests).
  --local-files-only           Never download PE-Core; require local cache.

Download:
  --download-engine auto|aria2|python
  --connections N              aria2 parallel connections (default 16).

Space/cleanup:
  --delete-source-before-package
                               Delete downloaded source ZIP after embedding, before
                               creating result archive. Useful on very small disks.
  --keep-temp                  Keep temp directory after completion.

Examples:
  # Small disk + 15G shm (recommended default)
  bash run_pipeline.sh --url https://host/Videos_L28_a.zip \
    --profile balanced --batch-size 64

  # Large /dev/shm
  bash run_pipeline.sh --url https://host/Videos_L28_a.zip \
    --profile ram-rich --batch-size 64

  # Explicit custom layout
  bash run_pipeline.sh --url https://host/Videos_L28_a.zip \
    --source-dir /workspace/source \
    --hf-cache-dir /dev/shm/aic_hf \
    --temp-dir /dev/shm/aic_tmp
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url) URL="$2"; shift 2 ;;
    --zip) ZIP_PATH="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --work-root) WORK_ROOT="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --archive) ARCHIVE="$2"; shift 2 ;;
    --source-dir) SOURCE_DIR="$2"; shift 2 ;;
    --hf-cache-dir) HF_CACHE_DIR="$2"; shift 2 ;;
    --temp-dir) TEMP_DIR="$2"; shift 2 ;;
    --model-source) MODEL_SOURCE="$2"; shift 2 ;;
    --kaggle-model) KAGGLE_MODEL="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --kaggle-cache-dir) KAGGLE_CACHE_DIR="$2"; shift 2 ;;
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
    --transnet-mode) TRANSNET_MODE="$2"; shift 2 ;;
    --transnet-batch-size) TRANSNET_BATCH_SIZE="$2"; shift 2 ;;
    --transnet-decode-workers) TRANSNET_DECODE_WORKERS="$2"; shift 2 ;;
    --transnet-active-videos) TRANSNET_ACTIVE_VIDEOS="$2"; shift 2 ;;
    --transnet-prefetch-videos) TRANSNET_PREFETCH_VIDEOS="$2"; shift 2 ;;
    --transnet-prefetch-windows) TRANSNET_PREFETCH_WINDOWS="$2"; shift 2 ;;
    --transnet-batch-timeout-ms) TRANSNET_BATCH_TIMEOUT_MS="$2"; shift 2 ;;
    --keyframe-strategy) KEYFRAME_STRATEGY="$2"; shift 2 ;;
    --keyframes-per-second) KEYFRAMES_PER_SECOND="$2"; shift 2 ;;
    --min-keyframes-per-scene) MIN_KEYFRAMES_PER_SCENE="$2"; shift 2 ;;
    --max-keyframes-per-scene) MAX_KEYFRAMES_PER_SCENE="$2"; shift 2 ;;
    --parallel-stages) PARALLEL_STAGES=1; shift ;;
    --sequential-stages) PARALLEL_STAGES=0; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --download-engine) DOWNLOAD_ENGINE="$2"; shift 2 ;;
    --connections) CONNECTIONS="$2"; shift 2 ;;
    --delete-source-before-package) DELETE_SOURCE_BEFORE_PACKAGE=1; shift ;;
    --keep-temp) KEEP_TEMP=1; shift ;;
    --local-files-only) LOCAL_FILES_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -n "$URL" && -n "$ZIP_PATH" ]]; then
  echo "ERROR: use either --url or --zip, not both" >&2
  exit 2
fi
if [[ -z "$URL" && -z "$ZIP_PATH" ]]; then
  echo "ERROR: one of --url or --zip is required" >&2
  usage >&2
  exit 2
fi
case "$PROFILE" in
  balanced|ram-rich|disk-rich|colab) ;;
  *) echo "ERROR: invalid --profile: $PROFILE" >&2; exit 2 ;;
esac
case "$AMP" in
  auto|off|fp16|bf16) ;;
  *) echo "ERROR: invalid --amp: $AMP" >&2; exit 2 ;;
esac
case "$TRANSNET_MODE" in
  global|sequential) ;;
  *) echo "ERROR: invalid --transnet-mode: $TRANSNET_MODE" >&2; exit 2 ;;
esac
case "$KEYFRAME_STRATEGY" in
  tiered|linear) ;;
  *) echo "ERROR: invalid --keyframe-strategy: $KEYFRAME_STRATEGY" >&2; exit 2 ;;
esac
case "$DOWNLOAD_ENGINE" in
  auto|aria2|python) ;;
  *) echo "ERROR: invalid --download-engine: $DOWNLOAD_ENGINE" >&2; exit 2 ;;
esac
case "$MODEL_SOURCE" in
  huggingface|kaggle|local) ;;
  *) echo "ERROR: invalid --model-source: $MODEL_SOURCE" >&2; exit 2 ;;
esac

# Resolve a writable persistent work root before any download/GPU work.
resolve_work_root() {
  local candidate
  if [[ -n "$WORK_ROOT" ]]; then
    candidate="$WORK_ROOT"
  elif [[ -n "${KAGGLE_KERNEL_RUN_TYPE:-}" || -d /kaggle/working ]]; then
    candidate="/kaggle/working"
  elif [[ -d /content && -w /content ]]; then
    candidate="/content"
  elif [[ -d /workspace && -w /workspace ]]; then
    candidate="/workspace"
  else
    candidate="${HOME:-$PWD}/workspace"
    echo "[workspace] /workspace is unavailable; falling back to: $candidate"
  fi
  mkdir -p "$candidate" || { echo "ERROR: cannot create work root: $candidate" >&2; exit 1; }
  local probe="$candidate/.keyframe_pipeline_write_test.$$"
  if ! ( : > "$probe" && rm -f "$probe" ); then
    echo "ERROR: work root is not writable: $candidate" >&2
    echo "Use --work-root PATH pointing to a persistent writable directory." >&2
    exit 1
  fi
  WORK_ROOT="$(cd "$candidate" && pwd)"
}
resolve_work_root

# PE-Core checkpoints run ~7GB; Colab/Kaggle/etc. size /dev/shm very
# inconsistently (observed as low as ~5.7G), well under what the README's
# "15G /dev/shm" assumption expects. Writability alone doesn't catch this --
# the model download fails deep inside kagglehub/huggingface_hub with a bare
# ENOSPC once shm fills, which looks like a random crash. Require real room.
MIN_SHM_GIB_FOR_CACHE=12
shm_has_room() {
  [[ -d /dev/shm && -w /dev/shm ]] || return 1
  local avail_bytes min_bytes
  avail_bytes=$(df -PB1 /dev/shm | awk 'NR==2 {print $4}')
  min_bytes=$((MIN_SHM_GIB_FOR_CACHE * 1024 * 1024 * 1024))
  (( avail_bytes >= min_bytes ))
}

# Resolve profile defaults. Persistent paths derive from WORK_ROOT, never an assumed /workspace.
case "$PROFILE" in
  balanced)
    : "${SOURCE_DIR:=$WORK_ROOT/aic_source}"
    if shm_has_room; then
      : "${HF_CACHE_DIR:=/dev/shm/aic_keyframe_pipeline/hf_home}"
      : "${TEMP_DIR:=/dev/shm/aic_keyframe_pipeline/tmp}"
      : "${KAGGLE_CACHE_DIR:=/dev/shm/aic_keyframe_pipeline/kagglehub}"
    else
      if [[ -d /dev/shm && -w /dev/shm ]]; then
        echo "[profile] /dev/shm has less than ${MIN_SHM_GIB_FOR_CACHE}G free (model checkpoints run ~7-8G); using disk for HF/Kaggle cache instead." >&2
      fi
      : "${HF_CACHE_DIR:=$WORK_ROOT/aic_runtime/hf_home}"
      : "${TEMP_DIR:=$WORK_ROOT/aic_runtime/tmp}"
      : "${KAGGLE_CACHE_DIR:=$WORK_ROOT/aic_runtime/kagglehub}"
    fi
    ;;
  ram-rich)
    if ! shm_has_room; then
      echo "ERROR: --profile ram-rich requires /dev/shm with at least ${MIN_SHM_GIB_FOR_CACHE}G free (for model cache alone; ZIP+temp need more on top). Use --profile balanced or disk-rich instead." >&2
      exit 1
    fi
    : "${SOURCE_DIR:=/dev/shm/aic_keyframe_pipeline/source}"
    : "${HF_CACHE_DIR:=/dev/shm/aic_keyframe_pipeline/hf_home}"
    : "${TEMP_DIR:=/dev/shm/aic_keyframe_pipeline/tmp}"
    : "${KAGGLE_CACHE_DIR:=/dev/shm/aic_keyframe_pipeline/kagglehub}"
    ;;
  disk-rich)
    : "${SOURCE_DIR:=$WORK_ROOT/aic_runtime/source}"
    : "${HF_CACHE_DIR:=$WORK_ROOT/aic_runtime/hf_home}"
    : "${TEMP_DIR:=$WORK_ROOT/aic_runtime/tmp}"
    : "${KAGGLE_CACHE_DIR:=$WORK_ROOT/aic_runtime/kagglehub}"
    ;;
  colab)
    : "${SOURCE_DIR:=$WORK_ROOT/aic_source}"
    : "${HF_CACHE_DIR:=$WORK_ROOT/aic_runtime/hf_home}"
    : "${TEMP_DIR:=$WORK_ROOT/aic_runtime/tmp}"
    : "${KAGGLE_CACHE_DIR:=$WORK_ROOT/aic_runtime/kagglehub}"
    ;;
esac

mkdir -p "$SOURCE_DIR" "$HF_CACHE_DIR" "$TEMP_DIR" "$KAGGLE_CACHE_DIR"
SOURCE_DIR="$(cd "$SOURCE_DIR" && pwd)"
HF_CACHE_DIR="$(cd "$HF_CACHE_DIR" && pwd)"
TEMP_DIR="$(cd "$TEMP_DIR" && pwd)"
KAGGLE_CACHE_DIR="$(cd "$KAGGLE_CACHE_DIR" && pwd)"

# Derive data name from URL/existing ZIP. Both historical ``Videos_L28_a.zip``
# and current range ``Video_N001-N010.zip`` transport prefixes are omitted.
if [[ -n "$URL" ]]; then
  BASENAME="$(basename "${URL%%\?*}")"
  [[ "$BASENAME" == *.zip ]] || BASENAME="input.zip"
else
  ZIP_PATH="$(readlink -f "$ZIP_PATH")"
  BASENAME="$(basename "$ZIP_PATH")"
fi
STEM="${BASENAME%.zip}"
DATA_ID="${STEM#Videos_}"
DATA_ID="${DATA_ID#Video_}"
[[ -n "$OUT_DIR" ]] || OUT_DIR="$WORK_ROOT/output_${DATA_ID}"
[[ -n "$ARCHIVE" ]] || ARCHIVE="$WORK_ROOT/${DATA_ID}_results.zip"
mkdir -p "$OUT_DIR" "$(dirname "$ARCHIVE")"

# Refuse to start expensive work if persistent destinations are not writable.
for d in "$SOURCE_DIR" "$OUT_DIR" "$(dirname "$ARCHIVE")"; do
  probe="$d/.keyframe_pipeline_write_test.$$"
  if ! ( : > "$probe" && rm -f "$probe" ); then
    echo "ERROR: destination is not writable: $d" >&2; exit 1
  fi
done

if [[ -n "$URL" ]]; then
  ZIP_PATH="$SOURCE_DIR/$BASENAME"
  DOWNLOADED_BY_US=1
else
  DOWNLOADED_BY_US=0
fi

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

TRANSNET_PID=""
cleanup() {
  if [[ -n "$TRANSNET_PID" ]] && kill -0 "$TRANSNET_PID" 2>/dev/null; then
    kill "$TRANSNET_PID" 2>/dev/null || true
    wait "$TRANSNET_PID" 2>/dev/null || true
  fi
  if [[ "$KEEP_TEMP" -eq 0 ]]; then
    rm -rf "$TEMP_DIR" 2>/dev/null || true
  else
    echo "[keep-temp] $TEMP_DIR"
  fi
}
trap cleanup EXIT

free_h() { df -h "$1" 2>/dev/null | awk 'NR==2 {print $4 " free on " $1}'; }

echo "=== Configuration ==="
echo "Profile:      $PROFILE"
echo "WORK ROOT:    $WORK_ROOT"
echo "Source ZIP:   $ZIP_PATH"
echo "Source fs:    $(free_h "$SOURCE_DIR")"
echo "HF cache:     $HF_CACHE_DIR"
echo "HF cache fs:  $(free_h "$HF_CACHE_DIR")"
echo "Temp:         $TEMP_DIR"
echo "Temp fs:      $(free_h "$TEMP_DIR")"
echo "Kaggle cache: $KAGGLE_CACHE_DIR"
echo "Kaggle fs:    $(free_h "$KAGGLE_CACHE_DIR")"
echo "Output:       $OUT_DIR"
echo "Archive:      $ARCHIVE"
echo "PE-Core:      weights=fp32 amp=$AMP tf32=$TF32 compile=$COMPILE batch=$BATCH_SIZE prefetch=$PREFETCH_BATCHES"
echo "TransNet:     mode=$TRANSNET_MODE batch=$TRANSNET_BATCH_SIZE decode_workers=$TRANSNET_DECODE_WORKERS active_videos=$TRANSNET_ACTIVE_VIDEOS prefetch_videos=$TRANSNET_PREFETCH_VIDEOS"
echo "Stage flow:   $([[ "$PARALLEL_STAGES" -eq 1 ]] && echo streaming-parallel || echo sequential)"
echo "Model src:    $MODEL_SOURCE"
echo "Kaggle mdl:   $KAGGLE_MODEL"
echo "Downloader:   $DOWNLOAD_ENGINE connections=$CONNECTIONS"
[[ -n "$LIMIT" ]] && echo "Video limit:  $LIMIT"

if [[ -n "$URL" ]]; then
  echo ""
  echo "=== preflight: disk space for source ZIP ==="
  REMOTE_SIZE_BYTES="$("$PYTHON_BIN" - "$URL" <<'PYSIZE' 2>/dev/null || true
import sys, requests
try:
    r = requests.head(sys.argv[1], allow_redirects=True, timeout=15)
    print(int(r.headers.get('content-length', 0)))
except Exception:
    print(0)
PYSIZE
)"
  AVAIL_BYTES="$(df -PB1 "$SOURCE_DIR" | awk 'NR==2 {print $4}')"
  if [[ "${REMOTE_SIZE_BYTES:-0}" -gt 0 ]]; then
    # Require the ZIP size plus a 20% safety margin (temp extraction, output metadata).
    NEED_BYTES=$(( REMOTE_SIZE_BYTES + REMOTE_SIZE_BYTES / 5 ))
    echo "Remote ZIP size: $((REMOTE_SIZE_BYTES/1024/1024)) MiB; free on $SOURCE_DIR: $((AVAIL_BYTES/1024/1024)) MiB"
    if (( AVAIL_BYTES < NEED_BYTES )); then
      echo "ERROR: not enough disk space for this ZIP. Need ~$((NEED_BYTES/1024/1024)) MiB, have $((AVAIL_BYTES/1024/1024)) MiB free on $SOURCE_DIR." >&2
      echo "Free space (delete old ZIPs in $SOURCE_DIR, pass --delete-source-before-package for future runs) or use --work-root/--source-dir on a bigger disk." >&2
      exit 1
    fi
  else
    echo "Could not determine remote ZIP size (HEAD request failed/no content-length); skipping size check. Free on $SOURCE_DIR: $((AVAIL_BYTES/1024/1024)) MiB" >&2
  fi

  echo ""
  echo "=== STEP 1/4: download complete ZIP ==="
  rm -f "$ZIP_PATH"

  ENGINE="$DOWNLOAD_ENGINE"
  if [[ "$ENGINE" == "auto" ]]; then
    if command -v aria2c >/dev/null 2>&1; then ENGINE="aria2"; else ENGINE="python"; fi
  fi

  if [[ "$ENGINE" == "aria2" ]]; then
    aria2c \
      -x "$CONNECTIONS" -s "$CONNECTIONS" -k 16M \
      --file-allocation=none --continue=true \
      --console-log-level=warn --summary-interval=5 \
      -d "$SOURCE_DIR" -o "$BASENAME" "$URL"
  else
    "$PYTHON_BIN" - "$URL" "$ZIP_PATH" <<'PYDOWNLOAD'
import sys, time, requests
from tqdm import tqdm
url, out = sys.argv[1], sys.argv[2]
chunk_size = 8 * 1024 * 1024
t0 = time.perf_counter()
with requests.get(url, stream=True, timeout=(30, 300)) as r:
    r.raise_for_status()
    total = int(r.headers.get('content-length', 0))
    done = 0
    with open(out, 'wb', buffering=16*1024*1024) as f, tqdm(total=total or None, unit='B', unit_scale=True, unit_divisor=1024, desc='ZIP') as bar:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk); done += len(chunk); bar.update(len(chunk))
elapsed = time.perf_counter() - t0
print(f"Downloaded {done/1024**3:.3f} GiB in {elapsed:.1f}s ({done/1024**2/max(elapsed,1e-9):.1f} MiB/s)")
PYDOWNLOAD
  fi
else
  echo ""
  echo "=== STEP 1/4: using existing ZIP ==="
fi

if [[ ! -f "$ZIP_PATH" ]]; then
  echo "ERROR: ZIP not found after download/setup: $ZIP_PATH" >&2
  exit 1
fi

"$PYTHON_BIN" - "$ZIP_PATH" <<'PYZIPCHECK'
import sys, zipfile
p=sys.argv[1]
with zipfile.ZipFile(p) as z:
    bad=z.testzip()
    if bad: raise RuntimeError(f"Corrupt ZIP member: {bad}")
    videos=[i for i in z.infolist() if i.filename.lower().endswith(('.mp4','.mov','.mkv','.avi','.webm','.m4v'))]
    stored=sum(i.compress_type == zipfile.ZIP_STORED for i in videos)
print(f"ZIP OK: {len(videos)} videos, {stored} ZIP_STORED")
if stored != len(videos):
    print("WARNING: compressed video entries will be materialized one-at-a-time under TMPDIR")
PYZIPCHECK

echo ""
echo "=== STEP 2/4: TransNetV2 metadata ==="
TRANSNET_ARGS=(
  --zip "$ZIP_PATH"
  --out-dir "$OUT_DIR"
  --phase transnet
  --device "$DEVICE"
  --transnet-mode "$TRANSNET_MODE"
  --transnet-batch-size "$TRANSNET_BATCH_SIZE"
  --transnet-decode-workers "$TRANSNET_DECODE_WORKERS"
  --transnet-active-videos "$TRANSNET_ACTIVE_VIDEOS"
  --transnet-prefetch-videos "$TRANSNET_PREFETCH_VIDEOS"
  --transnet-prefetch-windows "$TRANSNET_PREFETCH_WINDOWS"
  --transnet-batch-timeout-ms "$TRANSNET_BATCH_TIMEOUT_MS"
  --keyframe-strategy "$KEYFRAME_STRATEGY"
  --keyframes-per-second "$KEYFRAMES_PER_SECOND"
  --min-keyframes-per-scene "$MIN_KEYFRAMES_PER_SCENE"
  --max-keyframes-per-scene "$MAX_KEYFRAMES_PER_SCENE"
  --no-jsonl
)
if [[ -n "$LIMIT" ]]; then TRANSNET_ARGS+=(--limit "$LIMIT"); fi
TRANSNET_RC=0
TRANSNET_SENTINEL="$OUT_DIR/.transnet.complete"
rm -f "$TRANSNET_SENTINEL"
if [[ "$PARALLEL_STAGES" -eq 1 ]]; then
  TRANSNET_ARGS+=(--summary-name run_summary.transnet.json)
  (
    set +e
    "$PYTHON_BIN" "$ROOT/src/process_video_zip_gpu.py" "${TRANSNET_ARGS[@]}"
    rc=$?
    sentinel_tmp="${TRANSNET_SENTINEL}.tmp.$$"
    printf '%s\n' "$rc" > "$sentinel_tmp"
    mv "$sentinel_tmp" "$TRANSNET_SENTINEL"
    exit "$rc"
  ) &
  TRANSNET_PID=$!
  echo "[parallel] TransNet producer pid=$TRANSNET_PID; starting PE-Core consumer without an archive-wide barrier."
else
  "$PYTHON_BIN" "$ROOT/src/process_video_zip_gpu.py" "${TRANSNET_ARGS[@]}" || TRANSNET_RC=$?
fi

echo ""
echo "=== STEP 3/4: adaptive ffmpeg keyframe decode + encoder-aware preprocess + PE-Core global embed ==="
EMBED_ARGS=(
  --zip "$ZIP_PATH"
  --out-dir "$OUT_DIR"
  --phase embed
  --amp "$AMP"
  --batch-size "$BATCH_SIZE"
  --prefetch-batches "$PREFETCH_BATCHES"
  --embed-decode-mode "$EMBED_DECODE_MODE"
  --embed-seek-gap-seconds "$EMBED_SEEK_GAP_SECONDS"
  --embed-seek-min-savings "$EMBED_SEEK_MIN_SAVINGS"
  --embed-seek-max-groups "$EMBED_SEEK_MAX_GROUPS"
  --device "$DEVICE"
  --model-source "$MODEL_SOURCE"
  --kaggle-model "$KAGGLE_MODEL"
  --kaggle-cache-dir "$KAGGLE_CACHE_DIR"
  --no-jsonl
)
if [[ "$TF32" -eq 1 ]]; then EMBED_ARGS+=(--tf32); else EMBED_ARGS+=(--no-tf32); fi
if [[ "$EMBED_FFMPEG_PREPROCESS" -eq 1 ]]; then EMBED_ARGS+=(--embed-ffmpeg-preprocess); else EMBED_ARGS+=(--no-embed-ffmpeg-preprocess); fi
if [[ "$COMPILE" -eq 1 ]]; then EMBED_ARGS+=(--compile); fi
if [[ -n "$MODEL_DIR" ]]; then EMBED_ARGS+=(--model-dir "$MODEL_DIR"); fi
if [[ "$LOCAL_FILES_ONLY" -eq 1 ]]; then EMBED_ARGS+=(--local-files-only); fi
if [[ -n "$LIMIT" ]]; then EMBED_ARGS+=(--limit "$LIMIT"); fi
if [[ "$PARALLEL_STAGES" -eq 1 ]]; then
  EMBED_ARGS+=(--wait-for-transnet-sentinel "$TRANSNET_SENTINEL" --summary-name run_summary.embed.json)
fi
EMBED_RC=0
"$PYTHON_BIN" "$ROOT/src/process_video_zip_gpu.py" "${EMBED_ARGS[@]}" || EMBED_RC=$?
if [[ "$PARALLEL_STAGES" -eq 1 ]]; then
  wait "$TRANSNET_PID" || TRANSNET_RC=$?
  TRANSNET_PID=""
fi
if [[ "$TRANSNET_RC" -ne 0 ]]; then
  echo "WARNING: TransNet phase reported failure(s) (exit $TRANSNET_RC); embedding continued for videos that succeeded." >&2
fi
if [[ "$EMBED_RC" -ne 0 ]]; then
  echo "WARNING: Embed phase reported failure(s) (exit $EMBED_RC)." >&2
fi

if [[ "$PARALLEL_STAGES" -eq 1 ]]; then
  "$PYTHON_BIN" - "$OUT_DIR" <<'PYMERGE'
from pathlib import Path
import json, os, sys

out = Path(sys.argv[1])
def load(name):
    path = out / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

transnet = load("run_summary.transnet.json")
embed = load("run_summary.embed.json")
payload = {
    "zip": embed.get("zip") or transnet.get("zip"),
    "stage_flow": "streaming-parallel",
    "transnet_ready": transnet.get("transnet_ready", []),
    "embedding_results": embed.get("embedding_results", []),
    "failures": transnet.get("failures", []) + embed.get("failures", []),
}
target = out / "run_summary.json"
temporary = out / "run_summary.json.tmp"
temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
os.replace(temporary, target)
PYMERGE
fi

if [[ "$DELETE_SOURCE_BEFORE_PACKAGE" -eq 1 && "$DOWNLOADED_BY_US" -eq 1 ]]; then
  echo ""
  echo "=== freeing source space before packaging ==="
  rm -f "$ZIP_PATH"
fi

echo ""
echo "=== STEP 4/4: package Phase 1 + Phase 2 results ==="
bash "$ROOT/package_results.sh" "$OUT_DIR" "$ARCHIVE"

RUN_ENDED_EPOCH=$(date +%s)
RUN_WALL_SECONDS=$((RUN_ENDED_EPOCH-RUN_STARTED_EPOCH))
"$PYTHON_BIN" - "$OUT_DIR" "$RUN_WALL_SECONDS" "$BATCH_SIZE" "$PREFETCH_BATCHES" "$AMP" "$TF32" "$PARALLEL_STAGES" <<'PYPERF'
from pathlib import Path
import json, sys
out=Path(sys.argv[1]); wall=float(sys.argv[2]); batch=int(sys.argv[3]); prefetch=int(sys.argv[4]); amp=sys.argv[5]; tf32=bool(int(sys.argv[6])); parallel=bool(int(sys.argv[7]))
video_stats=[]; total_kf=0; cpu_s=0.0; gpu_s=0.0
for p in sorted(out.glob('*/embedding_stats.json')):
    try: d=json.loads(p.read_text()); video_stats.append(d)
    except Exception: continue
for d in video_stats:
    n=int(d.get('num_keyframes') or 0); total_kf+=n
    t=d.get('timing_seconds') or {}; cpu_s+=float(t.get('cpu_decode_and_preprocess') or 0); gpu_s+=float(t.get('gpu_embedding') or 0)
transnet=[]
for p in sorted(out.glob('*/transnet_stats.json')):
    try: transnet.append(json.loads(p.read_text()))
    except Exception: pass
transnet_decode=sum(float((d or {}).get('decode_seconds') or 0) for d in transnet)
transnet_infer=sum(float((d or {}).get('inference_seconds') or 0) for d in transnet)
summary={
 'videos_complete':len(video_stats),'total_keyframes':total_kf,'wall_seconds':round(wall,3),
 'end_to_end_keyframes_per_s':round(total_kf/wall,3) if wall>0 else None,
 'embedding_gpu_seconds_sum':round(gpu_s,3),'embedding_gpu_images_per_s':round(total_kf/gpu_s,3) if gpu_s>0 else None,
 'decode_preprocess_seconds_sum':round(cpu_s,3),'decode_preprocess_keyframes_per_s':round(total_kf/cpu_s,3) if cpu_s>0 else None,
 'transnet_decode_seconds_sum':round(transnet_decode,3),'transnet_inference_seconds_sum':round(transnet_infer,3),
 'batch_size':batch,'prefetch_batches':prefetch,'amp':amp,'tf32':tf32,
 'stage_flow':'streaming-parallel' if parallel else 'sequential',
}
(out/'performance_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
print('[performance] videos=%d keyframes=%d wall=%.1fs e2e=%.2f keyframes/s gpu=%.2f img/s decode_preprocess=%.2f keyframes/s' % (
 len(video_stats), total_kf, wall, summary['end_to_end_keyframes_per_s'] or 0, summary['embedding_gpu_images_per_s'] or 0, summary['decode_preprocess_keyframes_per_s'] or 0))
PYPERF

# Refresh archive once so performance_summary.json is included.
bash "$ROOT/package_results.sh" "$OUT_DIR" "$ARCHIVE" >/dev/null

echo ""
echo "=== DONE ==="
echo "OUTPUT DIR: $OUT_DIR"
echo "RESULT ZIP: $ARCHIVE"
echo "scene files:     $(find "$OUT_DIR" -name scenes.json | wc -l)"
echo "keyframe files:  $(find "$OUT_DIR" -name keyframes.json | wc -l)"
echo "embedding files: $(find "$OUT_DIR" -name embeddings.npy | wc -l)"
echo "Performance: $OUT_DIR/performance_summary.json"

if [[ "$TRANSNET_RC" -ne 0 || "$EMBED_RC" -ne 0 ]]; then
  echo "WARNING: this dataset had per-video failures (transnet_rc=$TRANSNET_RC embed_rc=$EMBED_RC); see run_summary.json." >&2
  exit 1
fi

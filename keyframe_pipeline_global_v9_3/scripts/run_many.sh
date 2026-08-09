#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/runtime_env.sh"
URLS=()
URLS_FILE=""
WORK_ROOT=""
CONTINUE_ON_ERROR=0
COMMON_ARGS=()
usage(){ cat <<'USAGE'
Usage:
  run_many.sh --url URL --url URL2 [common run_pipeline options]
  run_many.sh --urls-file urls.txt [common run_pipeline options]

Options handled here:
  --url URL              Repeatable dataset URL.
  --urls-file FILE       One URL per line; blank lines/# comments ignored.
  --work-root PATH       Persistent root shared by all datasets.
  --continue-on-error    Continue to the next dataset if one fails.
  --keep-source-zips     Don't delete each dataset's downloaded ZIP after use
                         (default: deleted after embedding, before packaging,
                         since multiple full-size ZIPs left on disk across
                         datasets is the most common cause of an unattended
                         multi-dataset run running out of disk space).
All other options are forwarded to run_pipeline.sh.
USAGE
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --url) URLS+=("$2"); shift 2 ;;
    --urls-file) URLS_FILE="$2"; shift 2 ;;
    --work-root) WORK_ROOT="$2"; shift 2 ;;
    --continue-on-error) CONTINUE_ON_ERROR=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) COMMON_ARGS+=("$1"); if [[ $# -ge 2 && "$2" != --* ]]; then COMMON_ARGS+=("$2"); shift 2; else shift; fi ;;
  esac
done
if [[ -n "$URLS_FILE" ]]; then
  [[ -f "$URLS_FILE" ]] || { echo "ERROR: URL file not found: $URLS_FILE" >&2; exit 1; }
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"; line="$(echo "$line" | xargs)"; [[ -n "$line" ]] && URLS+=("$line")
  done < "$URLS_FILE"
fi
((${#URLS[@]})) || { echo "ERROR: provide at least one --url or --urls-file" >&2; exit 2; }
# Running many datasets sequentially accumulates one full source ZIP per
# dataset on disk (several GB each) unless deleted after use -- the most
# common cause of an unattended multi-dataset run silently running out of
# disk space partway through. Delete by default; pass --keep-source-zips to
# disable (e.g. to reuse ZIPs across repeated runs).
keep_source=0
FILTERED_ARGS=()
for arg in "${COMMON_ARGS[@]}"; do
  case "$arg" in
    --delete-source-before-package) keep_source=1; FILTERED_ARGS+=("$arg") ;;
    --keep-source-zips) keep_source=1 ;;
    *) FILTERED_ARGS+=("$arg") ;;
  esac
done
COMMON_ARGS=("${FILTERED_ARGS[@]}")
if [[ "$keep_source" -eq 0 ]]; then
  COMMON_ARGS+=(--delete-source-before-package)
fi
if [[ -z "$WORK_ROOT" ]]; then
  if [[ -n "${KAGGLE_KERNEL_RUN_TYPE:-}" || -d /kaggle/working ]]; then WORK_ROOT=/kaggle/working
  elif [[ -d /content && -w /content ]]; then WORK_ROOT=/content
  elif [[ -d /workspace && -w /workspace ]]; then WORK_ROOT=/workspace
  else WORK_ROOT="${HOME:-$PWD}/workspace"; fi
fi
mkdir -p "$WORK_ROOT"
SUMMARY="$WORK_ROOT/multi_run_summary.jsonl"
: > "$SUMMARY"
echo "=== MULTI DATASET RUN ==="
echo "Datasets: ${#URLS[@]}"
echo "Work root: $WORK_ROOT"
idx=0
for url in "${URLS[@]}"; do
  idx=$((idx+1)); base="$(basename "${url%%\?*}")"; stem="${base%.zip}"; data_id="${stem#Videos_}"
  echo; echo "=== DATASET $idx/${#URLS[@]}: $data_id ==="; echo "$url"
  t0=$(date +%s)
  set +e
  bash "$ROOT/run_pipeline.sh" --url "$url" --work-root "$WORK_ROOT" "${COMMON_ARGS[@]}"
  rc=$?
  set -e
  t1=$(date +%s); elapsed=$((t1-t0))
  "$PYTHON_BIN" - "$SUMMARY" "$data_id" "$url" "$rc" "$elapsed" "$WORK_ROOT" <<'PY'
import json,sys
p,data_id,url,rc,elapsed,root=sys.argv[1:]
obj={"dataset":data_id,"url":url,"returncode":int(rc),"elapsed_seconds":int(elapsed),
     "output_dir":f"{root}/output_{data_id}","archive":f"{root}/{data_id}_results.zip"}
with open(p,'a',encoding='utf8') as f: f.write(json.dumps(obj,ensure_ascii=False)+'\n')
PY
  if (( rc != 0 )); then
    echo "[FAILED] $data_id (rc=$rc)"
    (( CONTINUE_ON_ERROR )) || exit "$rc"
  else
    echo "[OK] $data_id in ${elapsed}s -> $WORK_ROOT/${data_id}_results.zip"
  fi
done
echo; echo "=== MULTI RUN COMPLETE ==="; echo "Summary: $SUMMARY"

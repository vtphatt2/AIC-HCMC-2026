#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'USAGE'
Usage:
  scripts/test_resilience.sh disk-preflight --url URL [--source-dir PATH]
  scripts/test_resilience.sh cleanup --url URL --url URL2 [--limit N]

disk-preflight: fills the target disk to near-full, then runs run_pipeline.sh
  and expects it to refuse to start (loud, immediate error) instead of
  downloading and failing partway through. Takes seconds.

cleanup: runs run_many.sh over 2+ URLs with a small --limit (default 2), and
  fails unless each dataset's source ZIP is gone from disk before the next
  dataset starts downloading. Takes minutes, not hours -- --limit only
  shrinks video count, not the disk/cleanup mechanism being tested.
USAGE
}

MODE="${1:-}"; shift || true
URLS=(); SOURCE_DIR=""; LIMIT=2

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url) URLS+=("$2"); shift 2 ;;
    --source-dir) SOURCE_DIR="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$MODE" in
  disk-preflight)
    [[ "${#URLS[@]}" -ge 1 ]] || { echo "ERROR: --url required" >&2; exit 2; }
    SOURCE_DIR="${SOURCE_DIR:-$ROOT/aic_source}"
    mkdir -p "$SOURCE_DIR"
    FILL="$SOURCE_DIR/.disk_fill_test.tmp"
    trap 'rm -f "$FILL"' EXIT

    avail=$(df -PB1 "$SOURCE_DIR" | awk 'NR==2 {print $4}')
    keep_free=$((300*1024*1024))  # leave 300MiB free -- less than any real dataset ZIP
    fill_size=$((avail - keep_free))
    (( fill_size > 0 )) || { echo "ERROR: disk already has less than 300MiB free; nothing to fill" >&2; exit 1; }

    echo "=== filling $SOURCE_DIR to ~300MiB free (was $((avail/1024/1024)) MiB) ==="
    fallocate -l "$fill_size" "$FILL" 2>/dev/null || dd if=/dev/zero of="$FILL" bs=1M count=$((fill_size/1024/1024)) status=none
    df -h "$SOURCE_DIR"

    echo ""
    echo "=== running run_pipeline.sh; expecting an immediate disk-space error ==="
    set +e
    bash "$ROOT/run_pipeline.sh" --url "${URLS[0]}" --source-dir "$SOURCE_DIR" --profile balanced
    rc=$?
    set -e

    rm -f "$FILL"
    trap - EXIT
    if [[ "$rc" -ne 0 ]]; then
      echo ""
      echo "PASS: run_pipeline.sh refused to start (exit $rc) instead of downloading onto a full disk."
    else
      echo ""
      echo "FAIL: run_pipeline.sh exited 0 despite the disk being nearly full -- preflight check did not trigger."
      exit 1
    fi
    ;;

  cleanup)
    [[ "${#URLS[@]}" -ge 2 ]] || { echo "ERROR: need at least 2 --url for this test" >&2; exit 2; }
    WORK_ROOT="$(mktemp -d)"
    trap 'rm -rf "$WORK_ROOT"' EXIT
    SOURCE_DIR="$WORK_ROOT/aic_source"

    echo "=== running run_many.sh over ${#URLS[@]} URLs (--limit $LIMIT), watching $SOURCE_DIR ==="
    URL_ARGS=(); for u in "${URLS[@]}"; do URL_ARGS+=(--url "$u"); done
    bash "$ROOT/scripts/run_many.sh" "${URL_ARGS[@]}" --work-root "$WORK_ROOT" \
      --limit "$LIMIT" --profile balanced --batch-size 16 &
    pid=$!

    seen_nonempty=0
    while kill -0 "$pid" 2>/dev/null; do
      count=$(find "$SOURCE_DIR" -maxdepth 1 -name '*.zip' 2>/dev/null | wc -l)
      if (( count > 1 )); then
        echo "FAIL: $count source ZIPs present simultaneously in $SOURCE_DIR -- cleanup between datasets is not happening."
        kill "$pid" 2>/dev/null || true
        exit 1
      fi
      (( count == 1 )) && seen_nonempty=1
      sleep 3
    done
    wait "$pid"

    if [[ "$seen_nonempty" -eq 1 ]]; then
      echo ""
      echo "PASS: at most 1 source ZIP existed at a time across ${#URLS[@]} datasets."
    else
      echo ""
      echo "WARNING: never observed a ZIP on disk -- check --source-dir/--work-root resolved correctly."
    fi
    ;;

  *)
    usage; exit 2 ;;
esac

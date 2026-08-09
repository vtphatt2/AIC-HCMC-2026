#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'USAGE'
Usage: scripts/test_multi_link_fake.sh [--count N] [--videos-per-zip N] [--duration SECONDS] [extra run_many.sh flags]

End-to-end test of the multi-dataset flow (run_many.sh -> run_pipeline.sh,
disk preflight, ZIP cleanup between datasets) using entirely fake/local
data -- no real dataset, no real URL, no network dependency:

  1. Generates N tiny synthetic ZIPs (scripts/make_fake_zip.sh).
  2. Serves them from a local HTTP server (python -m http.server), so
     --url http://127.0.0.1:PORT/fakeN.zip exercises the real download path
     (aria2/curl, disk preflight) exactly like a real dataset URL would.
  3. Runs scripts/run_many.sh over those fake URLs.
  4. Tears down the HTTP server and temp files on exit.

Still loads the real TransNet/PE-Core models and does real GPU work -- it's
"fake" in that the video content and URLs are synthetic/local, not that the
pipeline behavior is mocked. First run still pays the one-time PE-Core
download cost.

Example:
  bash scripts/test_multi_link_fake.sh --count 3 --continue-on-error
USAGE
}

COUNT=3
VIDEOS_PER_ZIP=2
DURATION=5
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --count) COUNT="$2"; shift 2 ;;
    --videos-per-zip) VIDEOS_PER_ZIP="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

WORKDIR="$(mktemp -d)"
SERVER_PID=""
cleanup() {
  [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

echo "=== generating $COUNT fake ZIP(s) ($VIDEOS_PER_ZIP video(s) x ${DURATION}s each) ==="
for i in $(seq 1 "$COUNT"); do
  bash "$ROOT/scripts/make_fake_zip.sh" "$WORKDIR/serve/Videos_FAKE_$i.zip" "$VIDEOS_PER_ZIP" "$DURATION"
done

PORT=$(( (RANDOM % 20000) + 20000 ))
echo "=== serving $WORKDIR/serve on http://127.0.0.1:$PORT ==="
( cd "$WORKDIR/serve" && python3 -m http.server "$PORT" --bind 127.0.0.1 >/dev/null 2>&1 ) &
SERVER_PID=$!
sleep 1
kill -0 "$SERVER_PID" 2>/dev/null || { echo "ERROR: local HTTP server failed to start" >&2; exit 1; }

URL_ARGS=()
for i in $(seq 1 "$COUNT"); do
  URL_ARGS+=(--url "http://127.0.0.1:$PORT/Videos_FAKE_$i.zip")
done

echo ""
echo "=== running run_many.sh over $COUNT fake dataset(s) ==="
bash "$ROOT/scripts/run_many.sh" "${URL_ARGS[@]}" \
  --work-root "$WORKDIR/work" --batch-size 4 --prefetch-batches 2 \
  "${EXTRA_ARGS[@]}"

echo ""
echo "=== done; multi_run_summary.jsonl ==="
cat "$WORKDIR/work/multi_run_summary.jsonl" 2>/dev/null || echo "(not found)"

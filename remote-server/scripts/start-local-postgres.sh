#!/usr/bin/env bash
# Start/stop the portable, no-Docker PostgreSQL used for local dev
# (installed with `scoop install postgresql`; matches POSTGRES_URL in .env).
#
# First-time setup (already done once on this machine, kept here for a
# fresh machine / after `scoop uninstall postgresql`):
#   PG_BIN="$HOME/scoop/apps/postgresql/<version>/bin"
#   "$PG_BIN/initdb.exe" -D "$DATA_DIR" -U aic2026 --pwfile=<(printf 'aic2026') -E UTF8
#   "$PG_BIN/pg_ctl.exe" -D "$DATA_DIR" -l "$DATA_DIR.log" -o "-p 15432" start
#   "$PG_BIN/createdb.exe" -p 15432 -U aic2026 aic2026
#
# Usage: ./scripts/start-local-postgres.sh [start|stop|status]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PG_VERSION_DIR="$(find "$HOME/scoop/apps/postgresql" -maxdepth 1 -type d -name '[0-9]*' | sort -V | tail -1)"
PG_BIN="$PG_VERSION_DIR/bin"
DATA_DIR="$REPO_ROOT/challenge_resources/data/postgres_data"
LOG="$REPO_ROOT/challenge_resources/data/postgres_data.log"

ACTION="${1:-start}"

case "$ACTION" in
  start)
    "$PG_BIN/pg_ctl.exe" -D "$DATA_DIR" -l "$LOG" -o "-p 15432" start
    ;;
  stop)
    "$PG_BIN/pg_ctl.exe" -D "$DATA_DIR" stop
    ;;
  status)
    "$PG_BIN/pg_isready.exe" -p 15432
    ;;
  *)
    echo "Usage: $0 [start|stop|status]" >&2
    exit 1
    ;;
esac

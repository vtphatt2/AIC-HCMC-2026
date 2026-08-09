#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[v7] install.sh is kept for compatibility; delegating to setup.sh"
exec bash "$ROOT/setup.sh" "$@"

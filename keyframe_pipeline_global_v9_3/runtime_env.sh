#!/usr/bin/env bash
# Shared runtime selection for all launchers.
# Prefer the project-local venv created by setup.sh, then fall back to python3/python.
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
RUNTIME_FILE="$ROOT/.runtime_python"
if [[ -f "$RUNTIME_FILE" ]]; then
  _RUNTIME_PY="$(head -n 1 "$RUNTIME_FILE" | tr -d '\r\n')"
else
  _RUNTIME_PY=""
fi
if [[ -n "$_RUNTIME_PY" && -x "$_RUNTIME_PY" ]]; then
  PYTHON_BIN="$_RUNTIME_PY"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "ERROR: Python 3 not found. Run: bash $ROOT/setup.sh" >&2
  return 1 2>/dev/null || exit 1
fi
export PYTHON_BIN

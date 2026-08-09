#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
PYTHON3="${PYTHON3:-}"
INSTALL_TORCH_IF_MISSING=1
SKIP_APT=0

usage() {
  cat <<'USAGE'
Usage: bash setup.sh [options]

Creates a PEP-668-safe project virtual environment and installs runtime deps.
The venv uses --system-site-packages so preinstalled CUDA PyTorch is reused.

Options:
  --venv PATH              venv directory (default: <project>/.venv)
  --python PATH            Python 3 interpreter to use
  --skip-apt               do not try apt-get for python3-venv/ffmpeg/aria2
  --no-install-torch       fail instead of pip-installing torch if torch is unavailable
  -h, --help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv) VENV_DIR="$2"; shift 2 ;;
    --python) PYTHON3="$2"; shift 2 ;;
    --skip-apt) SKIP_APT=1; shift ;;
    --no-install-torch) INSTALL_TORCH_IF_MISSING=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$PYTHON3" ]]; then
  if command -v python3 >/dev/null 2>&1; then PYTHON3="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then PYTHON3="$(command -v python)"
  else echo "ERROR: Python 3 not found." >&2; exit 1
  fi
fi

run_apt() {
  if [[ "$SKIP_APT" -eq 1 ]] || ! command -v apt-get >/dev/null 2>&1; then return 1; fi
  if [[ "$(id -u)" -eq 0 ]]; then
    apt-get "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo apt-get "$@"
  else
    return 1
  fi
}

need_apt=0
"$PYTHON3" -m venv --help >/dev/null 2>&1 || need_apt=1
command -v ffmpeg >/dev/null 2>&1 || need_apt=1
command -v ffprobe >/dev/null 2>&1 || need_apt=1
command -v aria2c >/dev/null 2>&1 || need_apt=1
if [[ "$need_apt" -eq 1 && "$SKIP_APT" -eq 0 && -x "$(command -v apt-get 2>/dev/null || true)" ]]; then
  # python3-venv (generic) sometimes doesn't match the interpreter's exact
  # minor version (common on Colab); also try the version-pinned package.
  pyminor="$("$PYTHON3" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  echo "[setup] installing OS dependencies (python3-venv, python${pyminor}-venv, ffmpeg, aria2)"
  run_apt update -qq || true
  run_apt install -y python3-venv "python${pyminor}-venv" ffmpeg aria2 || true
fi

if ! "$PYTHON3" -m venv --help >/dev/null 2>&1; then
  echo "ERROR: Python venv module is unavailable. Install python3-venv (or python3-full), then rerun." >&2
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  echo "ERROR: ffmpeg/ffprobe not found. Install ffmpeg or rerun without --skip-apt." >&2
  exit 1
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "[setup] creating venv: $VENV_DIR"
  if ! "$PYTHON3" -m venv --system-site-packages "$VENV_DIR"; then
    echo "[setup] normal venv creation failed (often ensurepip on Colab)."
    echo "[setup] retrying with --without-pip while exposing system site packages..."
    rm -rf "$VENV_DIR"
    "$PYTHON3" -m venv --system-site-packages --without-pip "$VENV_DIR"
  fi
else
  echo "[setup] reusing venv: $VENV_DIR"
fi
PY="$VENV_DIR/bin/python"

if ! "$PY" -m pip --version >/dev/null 2>&1; then
  echo "ERROR: pip is unavailable inside the venv even through system-site-packages." >&2
  echo "On Colab, run: python3 -m pip install -U pip, then rerun setup.sh." >&2
  echo "On Ubuntu, install python3-pip/python3-venv and rerun." >&2
  exit 1
fi

"$PY" -m pip install --upgrade pip setuptools wheel
"$PY" -m pip install --upgrade \
  numpy Pillow timm huggingface_hub "huggingface_hub[hf_transfer]" safetensors transnetv2-pytorch requests kagglehub tqdm

if ! "$PY" - <<'PY' >/dev/null 2>&1
import torch
print(torch.__version__)
PY
then
  if [[ "$INSTALL_TORCH_IF_MISSING" -eq 0 ]]; then
    echo "ERROR: torch is unavailable. Install a CUDA-enabled PyTorch build, then rerun setup.sh." >&2
    exit 1
  fi
  echo "[setup] PyTorch not found; installing torch into the venv."
  "$PY" -m pip install torch
fi

cat > "$ROOT/.runtime_python" <<EOF2
$PY
EOF2

echo ""
echo "=== Runtime check ==="
"$PY" - <<'PY'
import sys
print("python:", sys.executable)
print("version:", sys.version.split()[0])
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"gpu {i}:", torch.cuda.get_device_name(i))
else:
    print("WARNING: CUDA is not available to PyTorch. GPU pipeline will not run until this is fixed.")
import timm, kagglehub
from transnetv2_pytorch import TransNetV2
print("timm:", timm.__version__)
print("imports: OK")
PY

echo "ffmpeg: $(command -v ffmpeg)"
echo "aria2c: $(command -v aria2c 2>/dev/null || echo 'not installed (optional)')"
echo ""
echo "Setup complete. Launchers automatically use: $PY"
echo "No 'source .venv/bin/activate' is required."

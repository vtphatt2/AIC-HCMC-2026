#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/runtime_env.sh"
echo '=== RAM ==='
free -h
echo
echo '=== Filesystems ==='
df -h /workspace /content /dev/shm /tmp 2>/dev/null || true
echo
echo '=== GPU ==='
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || nvidia-smi || true

echo
echo '=== AMP capability ==='
"$PYTHON_BIN" - <<'PYAMP' 2>/dev/null || true
import torch
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('BF16 supported:', torch.cuda.is_bf16_supported())
    print('v6 --amp auto ->', 'bf16' if torch.cuda.is_bf16_supported() else 'fp16')
else:
    print('CUDA unavailable')
PYAMP

#!/usr/bin/env bash
# Mac / Linux / WSL launcher for local-backend + frontend.
# Windows (non-WSL): use scripts/start-local.ps1 instead.
# See docs/launch_scripts.md for which --backend to pick per machine.
set -euo pipefail

backend_port=8002
frontend_port=3002
backend=onnx-cpu

usage() {
    echo "Usage: $0 [--backend onnx-cpu|torch-cpu|torch-cuda|torch-mps] [--backend-port PORT] [--frontend-port PORT]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend) backend="$2"; shift 2 ;;
        --backend-port) backend_port="$2"; shift 2 ;;
        --frontend-port) frontend_port="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

case "$backend" in
    onnx-cpu)   export PECORE_BACKEND=onnx ;;
    torch-cpu)  export PECORE_BACKEND=torch; export PECORE_DEVICE=cpu ;;
    torch-cuda) export PECORE_BACKEND=torch; export PECORE_DEVICE=cuda ;;
    torch-mps)  export PECORE_BACKEND=torch; export PECORE_DEVICE=mps; export PECORE_PRECISION=fp32 ;;
    *)
        echo "Unknown --backend '$backend'. Valid: onnx-cpu, torch-cpu, torch-cuda, torch-mps" >&2
        exit 1 ;;
esac

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
backend_dir="$root/local-client/local-backend"
frontend_dir="$root/local-client/frontend"
log_dir="$root/runtime-logs"
mkdir -p "$log_dir"

port_listening() {
    (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null
}

if ! port_listening "$backend_port"; then
    export CORS_ORIGINS="http://localhost:$frontend_port,http://127.0.0.1:$frontend_port"
    (cd "$backend_dir" && nohup .venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port "$backend_port" \
        >"$log_dir/backend.log" 2>"$log_dir/backend.err.log" &)
else
    echo "Backend already listening on $backend_port"
fi

api_url="http://127.0.0.1:$backend_port"
if ! port_listening "$frontend_port"; then
    (cd "$frontend_dir" && NEXT_PUBLIC_API_URL="$api_url" nohup npm run dev -- -p "$frontend_port" \
        >"$log_dir/frontend.log" 2>"$log_dir/frontend.err.log" &)
else
    echo "Frontend already listening on $frontend_port"
fi

echo "Backend:  $api_url (--backend $backend)"
echo "Frontend: http://127.0.0.1:$frontend_port"

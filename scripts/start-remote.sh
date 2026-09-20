#!/usr/bin/env bash
set -euo pipefail

port=8000
skip_databases=false
reload=false

usage() {
    echo "Usage: $0 [--port PORT] [--skip-databases] [--reload]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) port="$2"; shift 2 ;;
        --skip-databases) skip_databases=true; shift ;;
        --reload) reload=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
remote_dir="$root/remote-server"
python="$remote_dir/.venv/bin/python"

[[ -x "$python" ]] || {
    echo "Missing remote venv. Run: cd remote-server && python -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    exit 1
}
if [[ ! -f "$remote_dir/.env" ]]; then
    cp "$remote_dir/.env.example" "$remote_dir/.env"
    echo "Created remote-server/.env from .env.example; edit it for GPU/CORS settings."
fi

port_listening() {
    (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null
}

wait_port() {
    local target="$1"
    for _ in {1..90}; do
        port_listening "$target" && return
        sleep 1
    done
    echo "Port $target did not become ready within 90 seconds." >&2
    exit 1
}

if [[ "$skip_databases" == false ]]; then
    command -v docker >/dev/null 2>&1 || {
        echo "docker was not found. Install Docker or use --skip-databases for external databases." >&2
        exit 1
    }
    (cd "$remote_dir" && docker compose up -d --wait --wait-timeout 120)
    wait_port 19530
    wait_port 15432
fi

port_listening "$port" && { echo "Port $port is already in use." >&2; exit 1; }
args=(-m uvicorn main:app --env-file "$remote_dir/.env" --host 0.0.0.0 --port "$port")
[[ "$reload" == true ]] && args+=(--reload)
echo "Remote backend: http://127.0.0.1:$port (Ctrl+C to stop; Docker databases stay running)"
cd "$remote_dir"
exec "$python" "${args[@]}"

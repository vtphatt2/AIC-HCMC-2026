#!/usr/bin/env bash
# Mac / Linux / WSL launcher for local-backend + frontend.
# Windows (non-WSL): use scripts/start-local.ps1 instead.
# See docs/launch_scripts.md for which --backend to pick per machine.
set -euo pipefail

backend_port=8000
frontend_port=3000
backend=onnx-cpu
lan_address=""

usage() {
    echo "Usage: $0 [--backend onnx-cpu|torch-cpu|torch-cuda|torch-mps] [--backend-port PORT] [--frontend-port PORT] [--lan-address IPv4]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend) backend="$2"; shift 2 ;;
        --backend-port) backend_port="$2"; shift 2 ;;
        --frontend-port) frontend_port="$2"; shift 2 ;;
        --lan-address) lan_address="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

case "$backend" in
    onnx-cpu)   backend_env_vars="PECORE_BACKEND=onnx" ;;
    torch-cpu)  backend_env_vars="PECORE_BACKEND=torch PECORE_DEVICE=cpu" ;;
    torch-cuda) backend_env_vars="PECORE_BACKEND=torch PECORE_DEVICE=cuda" ;;
    torch-mps)  backend_env_vars="PECORE_BACKEND=torch PECORE_DEVICE=mps PECORE_PRECISION=fp32" ;;
    *)
        echo "Unknown --backend '$backend'. Valid: onnx-cpu, torch-cpu, torch-cuda, torch-mps" >&2
        exit 1 ;;
esac

bind_host=127.0.0.1
public_host=127.0.0.1
if [[ -n "$lan_address" ]]; then
    if [[ ! "$lan_address" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]]; then
        echo "--lan-address must be an IPv4 address, for example 192.168.0.102" >&2
        exit 1
    fi
    bind_host=0.0.0.0
    public_host="$lan_address"
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
backend_dir="$root/local-client/local-backend"
frontend_dir="$root/local-client/frontend"
log_dir="$root/runtime-logs"

[[ -x "$backend_dir/.venv/bin/python" ]] || {
    echo "Missing backend venv. Run: cd local-client/local-backend && python -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    exit 1
}
command -v npm >/dev/null 2>&1 || { echo "npm was not found. Install Node.js first." >&2; exit 1; }

port_listening() {
    (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null
}

if [[ -n "$lan_address" ]] && { port_listening "$backend_port" || port_listening "$frontend_port"; }; then
    echo "LAN mode needs fresh processes. Stop services on ports $backend_port and $frontend_port, then run this command again." >&2
    exit 1
fi

# Runs $cmd in its own terminal window (Terminal.app on Mac, gnome-terminal
# or xterm on Linux/WSL) so you can watch it live. Falls back to a background
# process logging to runtime-logs/ if no terminal emulator is available.
run_service() {
    local title="$1" cmd="$2" log_name="$3"
    local keep_open="$cmd; echo; echo '--- $title exited, press Enter to close ---'; read -r _"

    if [[ "${OSTYPE:-}" == darwin* ]] && command -v osascript >/dev/null 2>&1; then
        local escaped=${keep_open//\\/\\\\}
        escaped=${escaped//\"/\\\"}
        osascript -e "tell application \"Terminal\" to do script \"$escaped\"" >/dev/null
        return
    fi
    if command -v gnome-terminal >/dev/null 2>&1; then
        gnome-terminal --title="$title" -- bash -c "$keep_open"
        return
    fi
    if command -v xterm >/dev/null 2>&1; then
        (xterm -T "$title" -e bash -c "$keep_open" &)
        return
    fi

    echo "No terminal emulator found (tried Terminal.app/gnome-terminal/xterm) for $title; running in background. Logs: $log_dir/$log_name.log" >&2
    mkdir -p "$log_dir"
    (nohup bash -c "$cmd" >"$log_dir/$log_name.log" 2>"$log_dir/$log_name.err.log" &)
}

if ! port_listening "$backend_port"; then
    origins="http://localhost:$frontend_port,http://127.0.0.1:$frontend_port,http://$public_host:$frontend_port"
    backend_cmd="cd \"$backend_dir\" && $backend_env_vars CORS_ORIGINS=\"$origins\" .venv/bin/python -m uvicorn main:app --host $bind_host --port $backend_port"
    run_service "Backend ($backend)" "$backend_cmd" "backend"
    backend_note="opened in its own terminal window"
else
    echo "Backend already listening on $backend_port"
    backend_note="already running"
fi

api_url="http://$public_host:$backend_port"
if ! port_listening "$frontend_port"; then
    frontend_cmd="cd \"$frontend_dir\" && NEXT_PUBLIC_API_URL=\"$api_url\" npm run dev -- -H $bind_host -p $frontend_port"
    run_service "Frontend" "$frontend_cmd" "frontend"
    frontend_note="opened in its own terminal window"
else
    echo "Frontend already listening on $frontend_port"
    frontend_note="already running"
fi

echo "Backend:  $api_url (--backend $backend) - $backend_note"
echo "Frontend: http://$public_host:$frontend_port - $frontend_note"

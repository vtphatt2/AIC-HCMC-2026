# Launch scripts

`scripts/start-local.ps1` (Windows) and `scripts/start-local.sh` (Mac, Linux,
WSL) start the local backend and frontend, each in its own terminal window,
so you can watch their output live. They're a shortcut for local dev only —
see [running.md](running.md) / [setup.md](setup.md) for what each `ENV_MODE`
and `.env` value means.

- Windows: opens two `cmd` windows (titled "Backend (...)" / "Frontend").
- Mac: opens two Terminal.app windows.
- Linux/WSL: opens two `gnome-terminal` or `xterm` windows, whichever is
  installed (tried in that order). If neither is available, falls back to a
  background process logging to `runtime-logs/`.

Both scripts skip a service that's already listening on its port, and take
the same `--backend` choice, which sets `PECORE_BACKEND` / `PECORE_DEVICE`
for the PE-Core text encoder ([text_encoder.py](../local-client/local-backend/app/services/text_encoder.py)):

| `--backend` | Needs | Use on |
|---|---|---|
| `onnx-cpu` (default) | nothing extra — `requirements.txt` already has `onnxruntime`, CPU only | any machine, including CPU-only ones |
| `torch-cpu` | `pip install -r requirements-torch.txt` | any machine, slower than onnx-cpu, no GPU used |
| `torch-cuda` | `requirements-torch.txt` + an NVIDIA GPU/driver | Linux or WSL2 with an NVIDIA GPU, or Windows with an NVIDIA GPU |
| `torch-mps` | `requirements-torch.txt` | Apple Silicon Mac (not available on `start-local.ps1`) |

The env var a `.env` file sets for the same key still wins (`main.py` loads
`.env` with `override=True`), so leave `PECORE_BACKEND` / `PECORE_DEVICE`
unset in `.env` if you want `--backend` to control it.

## Windows

```powershell
scripts\start-local.ps1                       # onnx-cpu, ports 8000/3000
scripts\start-local.ps1 -Backend torch-cuda
scripts\start-local.ps1 -Backend torch-cpu -BackendPort 8010 -FrontendPort 3010
scripts\start-local.ps1 -LanAddress 192.168.0.102
```

## Mac / Linux / WSL

```bash
bash scripts/start-local.sh                   # onnx-cpu, ports 8000/3000
scripts/start-local.sh --backend torch-mps     # Apple Silicon
scripts/start-local.sh --backend torch-cuda    # Linux/WSL2 with an NVIDIA GPU
bash scripts/start-local.sh --lan-address 192.168.0.102
```

`-LanAddress` / `--lan-address` exposes both dev servers on the local network
and gives the frontend a backend URL that phones can reach. Stop any existing
processes on ports 8000/3000 first, run the LAN command, then open
`http://192.168.0.102:3000/tuning` on a phone connected to the same trusted
Wi-Fi. The query page checks the selected config revision once per second and
automatically searches again after a tuning save.

Requires the backend venv at `local-client/local-backend/.venv` (`python -m
venv .venv && .venv/bin/pip install -r requirements.txt`, see
[setup.md](setup.md)) and `npm install` already run in `local-client/frontend`.

## Remote server

`start-remote` starts Milvus/PostgreSQL with Docker Compose, waits for their
ports, then runs the remote FastAPI backend in the current terminal:

```powershell
scripts\start-remote.ps1
scripts\start-remote.ps1 -SkipDatabases -Reload
```

```bash
bash scripts/start-remote.sh
bash scripts/start-remote.sh --skip-databases --reload
```

It creates `remote-server/.env` from `.env.example` only when missing. The
remote launcher does not start the frontend; clients point
`NEXT_PUBLIC_API_URL` or `REMOTE_SERVER_URL` at port `8000`.
It expects dependencies in `remote-server/.venv`; setup remains explicit
because the GPU/CPU dependency choice is machine-specific.

## Stopping

Close the terminal window (or Ctrl+C inside it, then close). If a service
fell back to running in the background (no terminal emulator found), kill it
by port instead:

- Windows: `Get-NetTCPConnection -LocalPort 8000,3000 | Select-Object -Expand OwningProcess | Stop-Process`
- Mac/Linux/WSL: `lsof -ti:8000,3000 | xargs kill`

## Not covered by `--backend`

The learned reranker's ONNX cross-encoder
([basic_learned_reranker.py](../local-client/local-backend/app/strategies/basic_learned_reranker.py))
always runs on `CPUExecutionProvider`, regardless of `--backend` — it has no
GPU path yet.

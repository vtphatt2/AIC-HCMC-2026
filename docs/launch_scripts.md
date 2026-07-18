# Launch scripts

`scripts/start-local.ps1` (Windows) and `scripts/start-local.sh` (Mac, Linux,
WSL) start the local backend and frontend in the background, with logs under
`runtime-logs/`. They're a shortcut for local dev only — see
[running.md](running.md) / [setup.md](setup.md) for what each `ENV_MODE` and
`.env` value means.

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
scripts\start-local.ps1                       # onnx-cpu, ports 8002/3002
scripts\start-local.ps1 -Backend torch-cuda
scripts\start-local.ps1 -Backend torch-cpu -BackendPort 8010 -FrontendPort 3010
```

## Mac / Linux / WSL

```bash
scripts/start-local.sh                        # onnx-cpu, ports 8002/3002
scripts/start-local.sh --backend torch-mps     # Apple Silicon
scripts/start-local.sh --backend torch-cuda    # Linux/WSL2 with an NVIDIA GPU
```

Requires the backend venv at `local-client/local-backend/.venv` (`python -m
venv .venv && .venv/bin/pip install -r requirements.txt`, see
[setup.md](setup.md)) and `npm install` already run in `local-client/frontend`.

## Stopping

Both scripts start detached background processes; there's no stop
subcommand. Kill by port:

- Windows: `Get-NetTCPConnection -LocalPort 8002,3002 | Select-Object -Expand OwningProcess | Stop-Process`
- Mac/Linux/WSL: `lsof -ti:8002,3002 | xargs kill`

## Not covered by `--backend`

The learned reranker's ONNX cross-encoder
([basic_learned_reranker.py](../local-client/local-backend/app/strategies/basic_learned_reranker.py))
always runs on `CPUExecutionProvider`, regardless of `--backend` — it has no
GPU path yet.

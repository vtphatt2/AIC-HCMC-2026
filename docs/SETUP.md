# Setup & Running

Everything needed to get the system running, on one machine or across
several. Deeper technical writeups (backend internals, DB schema, why a
given bug happened) moved to [docs/archive/](archive/) — this page stays
focused on running it.

> **One-time migration:** `challenge_resources/data/zip_file/` and
> `raw_zip/` were renamed to `zip_embeddings/` and `raw_zip_videos/`. These
> folders are gitignored — pulling this change does **not** rename the real
> data on your disk. On every machine that already has this data (including
> the GPU server), rename them by hand once:
> ```bash
> cd challenge_resources/data
> mv zip_file zip_embeddings   # or: ren zip_file zip_embeddings   (Windows cmd)
> mv raw_zip raw_zip_videos    # or: ren raw_zip raw_zip_videos
> ```
> Alternatively, without touching the folders, set `RESULTS_ZIP_DIR` /
> `RAW_ZIP_DIR` in your `.env` to the old paths — see
> [Environment variables](#environment-variables) below.

## Prerequisites

| Tool | Minimum | Needed for |
|---|---|---|
| Python | 3.10+ | any backend |
| Node.js + npm | 18+ | frontend |
| Docker Desktop | latest | `remote-server` with real Milvus (optional — see below) |
| ngrok or cloudflared | any | sharing across machines without a shared network |

---

## Which scenario am I in?

One dataset — the lot archives in `challenge_resources/data/zip_embeddings/` —
reached three ways:

| I have… | Run this | Where data comes from |
|---|---|---|
| The lot archives on this machine, no GPU needed | `local-backend`, `ENV_MODE=ZIP` | Vectors exported from the archives |
| Access to a teammate's running GPU server | `local-backend`, `ENV_MODE=LOCAL` | Proxied to `REMOTE_SERVER_URL` |
| The GPU workstation itself | `remote-server`, `ENV_MODE=SERVER` | Real Milvus + PostgreSQL |

---

## 1. ZIP mode — local dev, no GPU

```bash
cd local-client/local-backend
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate elsewhere)
pip install -r requirements.txt   # base: no torch, ONNX text encoder by default
cp .env.example .env
```

You need `challenge_resources/data/zip_embeddings/*_results.zip` (the lot
archives) plus two derived files, both required and cheap to rebuild:

```bash
cd ../../remote-server
python scripts/ingest_zip_pipeline_results.py --skip-postgres   # → Milvus records
python scripts/export_video_fps.py                              # → video_fps.json

cd ../local-client/local-backend
python scripts/export_vectors_npy.py                             # → vectors.f32.npy (~11s)
```

`.env`:

```env
ENV_MODE=ZIP
AIC_SAMPLE_ROOT=D:\path\to\AIC-HCMC-2026\challenge_resources\data
MILVUS_LITE_PATH=D:\path\to\challenge_resources\data\milvus_lite.db
```

```bash
uvicorn main:app --reload --port 8000
```

Startup log should say `DataProvider: ZIP mode -> numpy memmap (exact),
193508 vectors`. If it refuses to start, the two export commands above are
what it's asking for.

**Thumbnails and playback** — two options:

- **Organizer archives not on this machine** (the common case): one-time
  build, then frames/video stream over HTTP Range from the organizers' host.
  Without it every frame 404s and the UI falls back to YouTube.
  ```bash
  python -m scripts.build_zip_video_index \
      --urls-file ../../challenge_resources/data/zip_video_links.txt \
      --output   ../../challenge_resources/data/zip_video_index.json
  ```
- **`raw_zip_videos/Videos_L*.zip` archives already on this machine**
  (e.g. running `local-backend` directly on what used to be the
  `remote-server` host): skip the build above and read them straight off
  disk instead — no `zip_video_index.json`, no network fetch:
  ```env
  ZIP_MEDIA_SOURCE=local
  RAW_ZIP_DIR=D:\path\to\challenge_resources\data\raw_zip_videos   # optional, this is the default
  ```
  `/api/health` reports `"zip_media_source":"local"` and the video count
  found there. Same routes, same behavior otherwise — including client-side
  decode (`NEXT_PUBLIC_FRAME_DECODE=client`) when sharing with teammates.

Then the frontend, in a second terminal:

```bash
cd local-client/frontend
npm install
cp .env.local.example .env.local     # NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev
```

Open http://localhost:3000.

---

## 2. Remote server — GPU workstation

### Restart the configured WSL workstation

Use this procedure for the existing installation at
`/workspace/AIC-HCMC-2026`. Docker must be running. The Python environment,
Node dependencies, dataset, and `.env` files are
already present on this workstation; restarting does not require installing
packages or ingesting the data again. On another machine, substitute its
repository path and complete the first-time setup below first.

The primary system command is the remote-server launcher below. Run the web
components in separate WSL terminals and keep those terminals open.

**Terminal 1 — backend and databases (port 8000):**

```bash
cd /workspace/AIC-HCMC-2026/remote-server
bash ../scripts/start-remote.sh --skip-databases
```

This reuses the Docker databases that are already running, reads
`remote-server/.env`, and runs the optimized `remote-server` backend. The
current `.env` supplies `WEB_CONCURRENCY=2` and the hardware/search/media
tuning, so no environment prefix is required. Wait for `Application startup
complete`. If the database containers are not already running, omit
`--skip-databases` once so the launcher starts and checks them. Do not add
`--reload` for a demo: reload mode ignores the two-worker setting.

**Terminal 2 — frontend (port 3000):**

```bash
cd /workspace/AIC-HCMC-2026/local-client/frontend
npm run build   # repeat after frontend source or .env.local changes
npm run start
```

`local-client/frontend/.env.local` uses `NEXT_PUBLIC_API_URL=/`, so API
requests go through the same hostname as the page. The production server
avoids development compilation during searches. Rebuild after frontend code
or public environment variables change. Use `npm run dev` only while editing
the frontend.

**Terminal 3 — shared web proxy (port 3001):**

```bash
cd /workspace/AIC-HCMC-2026
node scripts/share-proxy.cjs
```

Open **http://localhost:3001** on this machine. The proxy sends page requests
to the frontend and backend API/media requests to port 8000. These first
three terminals are sufficient for local use.

**Terminal 4 — public Cloudflare tunnel (optional for local use):**

```bash
cloudflared tunnel --url http://127.0.0.1:3001
```

Share the printed `https://…trycloudflare.com` URL with other users, including
those on Wi-Fi or mobile data. Both local and public access use the proxy:

```text
Local browser  → localhost:3001 → proxy → frontend / backend
Public browser → Cloudflare URL → tunnel → localhost:3001 → proxy → frontend / backend
```

Check `http://localhost:3001/api/health` and
`https://<your-tunnel-hostname>/api/health` in a browser; both should return
JSON with `"status":"ok"`. Then try a search and confirm the images load.
The temporary public URL usually changes when the tunnel restarts; see §5b
for named tunnels. Closing a service terminal or shutting down WSL stops that
service. Use Ctrl+C to stop it; Docker databases stay running when the backend
exits. If a port is already occupied, use the existing service or stop it
before starting another copy.

Exact 640px result cards are stored separately from full-size images for all
indexed L/M/N/S archives. The optional background warmer prioritizes N/S/M,
then fills older L cards. It is installed as the user
service `aic-prewarm-cards.service`; it resumes missing cards after a reboot and
first builds exact decoded-frame timestamps for N/S. It uses low-priority
workers with a twelve-core CPU quota, 8 GiB soft
memory limit and 10 GiB hard limit so live searches retain priority. It checks completed cards without rereading their
JPEG bytes on restart. Search remains available while it runs. Check it with
`systemctl --user status aic-prewarm-cards.service` and
`tail -f challenge_resources/data/zip_embeddings/prewarm_new_thumbnails.log`
from the repository root. A successful run writes
`/home/collab/.cache/aic2026/cards/.prewarm-result-thumbnails-complete`.
When a browser abandons an image from a previous search, the backend cancels
that decode unless another viewer is still waiting for the same frame. The
warmer pauses new decodes while live image requests are active, then resumes
after two seconds of inactivity.

For N sources where a short seek cannot reproduce the verified frame, the
warmer performs one sequential pass per video (at most two concurrently) and
saves exact selected JPEGs under
`challenge_resources/data/zip_embeddings/exact_n_frames/`. Every published
image must match the decoded timeline's pixel checksum. The live route then
reads that image instead of repeating a full-video decode. This does not
rewrite source videos, change their FPS/audio, or change vector IDs, ranking,
sampling, or model settings. Unrecoverable images are recorded in
`/home/collab/.cache/aic2026/cards/.prewarm-result-thumbnails-failures.json`;
exit status 2 stops automatic retries so bad sources can be reviewed.

### First-time setup

```bash
cd remote-server
python -m venv .venv
source .venv/bin/activate      # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

```env
ENV_MODE=SERVER
MILVUS_HOST=localhost
MILVUS_PORT=19530
POSTGRES_URL=postgresql://aic2026:aic2026@localhost:15432/aic2026
```

**Databases — pick one:**

- **Docker (real Milvus server):**
  ```bash
  docker compose up -d
  docker compose ps    # wait until MinIO is healthy, others Up
  ```
- **No Docker (dev/test only — embedded Milvus Lite + portable Postgres):**
  ```bash
  pip install -r requirements-milvus-lite.txt
  # set MILVUS_LITE_PATH=../challenge_resources/data/milvus_lite.db in .env
  bash scripts/start-local-postgres.sh start   # port 15432
  ```

**Ingest the dataset:**

```bash
python scripts/ingest_zip_pipeline_results.py --dry-run   # preview counts
python scripts/ingest_zip_pipeline_results.py
python scripts/export_video_fps.py                        # required: frame ↔ timestamp
```

Optional GPU CAGRA index:

```bash
pip install -r requirements-cagra.txt
python scripts/build_cagra_index.py
```

Then add `VECTOR_SEARCH_BACKEND=cagra`, `PECORE_DEVICE=cuda`,
`PECORE_PRECISION=fp16` to `.env`. CAGRA is CUDA-only; on Apple silicon keep
`VECTOR_SEARCH_BACKEND=milvus`, `PECORE_DEVICE=mps`, `PECORE_PRECISION=fp32`.

**Media**: drop the organizers' `Videos_L*.zip` into
`challenge_resources/data/raw_zip_videos/` — nothing to build, `/api/health`
reports how many were found. Lots without an archive there fall back to
YouTube playback.

**Start it:**

```bash
bash ../scripts/start-remote.sh          # Windows: ..\scripts\start-remote.ps1
```

This also starts Docker Milvus/PostgreSQL unless `--skip-databases` /
`-SkipDatabases` is passed. It does **not** start the frontend.

**Verify:**

```bash
curl http://localhost:8000/api/health
curl -X POST http://localhost:8000/api/warmup_text_encoder   # pay model load before a demo
```

**Frontend, anywhere with network access to this machine:**

```bash
cd local-client/frontend && npm install && cp .env.local.example .env.local
```

`.env.local`: `NEXT_PUBLIC_API_URL=http://<this-machine-ip>:8000`, and add
that origin (`http://<frontend-host>:3000`) to `CORS_ORIGINS` in
`remote-server/.env` if the frontend runs on a different machine.

---

## 3. LOCAL mode — your laptop, someone else's GPU server

Ask whoever runs `remote-server` for its URL (LAN IP or ngrok tunnel — see
[§5](#5-sharing-across-machines-ngrok)). Then, in
`local-client/local-backend/.env`:

```env
ENV_MODE=LOCAL
REMOTE_SERVER_URL=https://their-tunnel.ngrok-free.dev
CORS_ORIGINS=http://localhost:3000
```

```bash
uvicorn main:app --reload --port 8000
```

Startup log: `DataProvider: LOCAL mode -> https://their-tunnel.ngrok-free.dev`.
Frontend setup is identical to ZIP mode (§1). All four retrieval channels
come from the server in this mode, including the two the local lot archives
can't provide (`subtitled.semantic`, `transcript.semantic`).

---

## 4. One-command launch scripts

Shortcuts for local dev only — everything above still applies underneath.

```powershell
scripts\start-local.ps1                       # Windows: onnx-cpu, ports 8000/3000
scripts\start-local.ps1 -Backend torch-cuda
scripts\start-local.ps1 -LanAddress 192.168.0.102
```

```bash
bash scripts/start-local.sh                   # Mac/Linux/WSL
scripts/start-local.sh --backend torch-mps     # Apple Silicon
scripts/start-local.sh --lan-address 192.168.0.102
```

`--backend` picks the PE-Core text encoder: `onnx-cpu` (default, no torch
needed) / `torch-cpu` / `torch-cuda` / `torch-mps` (Mac only). The Mac/Linux/WSL
script opens backend, frontend and the optional agent service; the Windows
script opens backend and frontend, with agent started separately. Both skip
services already listening on their ports. See [Agent usage](agent/USAGE.md)
for the agent connection and Windows command.

For `remote-server`:

```powershell
scripts\start-remote.ps1
scripts\start-remote.ps1 -SkipDatabases -Reload
```
```bash
bash scripts/start-remote.sh
bash scripts/start-remote.sh --skip-databases --reload
```

**Stopping:** close the terminal window (or Ctrl+C, then close). If a
service fell back to running in the background (no terminal emulator
found): Windows — `Get-NetTCPConnection -LocalPort 8000,3000 | Select-Object
-Expand OwningProcess | Stop-Process`; Mac/Linux — `lsof -ti:8000,3000 |
xargs kill`.

---

## 5. Sharing across machines (ngrok or Cloudflare Tunnel)

Two different situations — pick the one that matches what you're doing:

| Sharing… | Command |
|---|---|
| A frontend + backend running together on **your** machine | [§5a](#5a-frontend--backend-on-the-same-machine) |
| Only a `remote-server` backend, for teammates' own `local-backend` (LOCAL mode) | [§5c](#5c-only-a-backend-remote-server) |

Choose either ngrok or Cloudflare Tunnel. ngrok has a PowerShell launcher;
Cloudflare Tunnel is started manually because it needs no application-specific
adapter. `scripts/share-proxy.cjs` is tunnel-agnostic: it presents the
frontend and backend on one local port. Its two agent API paths go to Next.js,
which calls the optional agent service; other search/media API paths go to the
VORTA backend.

One-time ngrok setup: `ngrok config add-authtoken <token>` (from
[dashboard.ngrok.com](https://dashboard.ngrok.com)). Claiming a free static
domain there means the URL doesn't change every restart.

### 5a. Frontend + backend on the same machine

Works whether the backend is `local-backend` or `remote-server` —
`scripts/share-proxy.cjs` forwards purely by port (3000 frontend, 8000
backend), it doesn't care which one is actually listening.

**Started via `start-local.ps1`?** (only ever starts `local-backend`) —
restart with `-Ngrok`:

```powershell
scripts\start-local.ps1 -Ngrok
```

Set `NGROK_DOMAIN=your-domain.ngrok-free.dev` in the repo-root `.env` first,
or pass `-NgrokDomain`. This wires up the proxy and tunnel for manual search.
To use agents, start the agent service as described in [Agent usage](agent/USAGE.md)
and ensure the frontend inherits `AGENT_SERVER_URL` before it starts.

**Already running in their own terminals** (the only path for
`remote-server`, since the launch script can't start it):

1. ⚠️ **Restart the frontend with `NEXT_PUBLIC_API_URL=/`.** This is the
   single most common failure — the page loads fine but every search fails,
   because `NEXT_PUBLIC_*` values are baked in when Next.js *starts*, not
   read live. If the frontend was already running before this was set, it's
   still shipping the old value to every visitor.
   ```bash
   cd local-client/frontend
   NEXT_PUBLIC_API_URL=/ npm run dev
   ```
   PowerShell equivalent:
   ```powershell
   $env:NEXT_PUBLIC_API_URL = "/"
   npm run dev
   ```
2. Start the proxy and point the tunnel at **it**, not at 3000 or 8000 directly:
   ```bash
   node scripts/share-proxy.cjs          # port 3001 by default
   ngrok http 3001
   ```

**Verify before sending the link** — from your own machine, hit the tunnel
like a teammate would:

```bash
curl https://your-domain.ngrok-free.dev/api/health
```

JSON back → good. HTML back → tunnel is pointed at 3000 instead of 3001.
Connection refused → `share-proxy.cjs` isn't running or its `BACKEND_PORT`
doesn't match. Then open the tunnel URL in an **incognito window** and run
one real search — that's the only check that also catches the
`NEXT_PUBLIC_API_URL` mistake above.

### 5b. Frontend + backend through Cloudflare Tunnel

Cloudflare Tunnel uses the same topology as ngrok, but points `cloudflared`
at the local share proxy. The proxy binds only to `127.0.0.1`; Cloudflare
connects out from the machine, so no inbound port needs to be opened.
Install [`cloudflared`](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)
on the machine that runs the frontend/backend before continuing.

1. Start the backend on port 8000 and restart the frontend with
   `NEXT_PUBLIC_API_URL=/`, exactly as in §5a. The relative API URL is
   essential: visitors then call `/api/...` on the same public hostname.
2. Start the proxy from the repository root:
   ```bash
   SHARE_PORT=3001 FRONTEND_PORT=3000 BACKEND_PORT=8000 \
     node scripts/share-proxy.cjs
   ```
3. For a temporary development link (no Cloudflare account or DNS setup):
   ```bash
   cloudflared tunnel --url http://127.0.0.1:3001
   ```
   Share the generated `https://*.trycloudflare.com` URL. Quick tunnels are
   for testing only; use a named tunnel for a stable team or production URL.
   They also cannot start while `~/.cloudflared/config.yml` exists.
4. For a named tunnel and a domain managed in Cloudflare:
   ```bash
   cloudflared tunnel login
   cloudflared tunnel create aic-frontend
   cloudflared tunnel route dns aic-frontend app.example.com
   ```
   Create `~/.cloudflared/config.yml`, substituting the UUID and credentials
   path printed by `tunnel create`:
   ```yaml
   tunnel: <tunnel-uuid>
   credentials-file: /home/<user>/.cloudflared/<tunnel-uuid>.json
   ingress:
     - hostname: app.example.com
       service: http://127.0.0.1:3001
     - service: http_status:404
   ```
   Validate and run it:
   ```bash
   cloudflared tunnel ingress validate
   cloudflared tunnel run aic-frontend
   ```

Verify the deployed hostname before sharing it:

```bash
curl https://app.example.com/api/health
```

It must return JSON. As with ngrok, HTML means the tunnel was pointed at the
Next dev server instead of the share proxy. This application has no built-in
authentication, so protect any non-private hostname with Cloudflare Access or
another auth layer before exposing it beyond the team.

### 5c. Only a backend (`remote-server`)

```bash
ngrok http 8000
```

Teammates set the printed URL as `REMOTE_SERVER_URL` in their own
`local-backend/.env` (§3 above). If instead a teammate's **browser** calls
this URL directly (no `local-backend` in between), add the tunnel URL to
`CORS_ORIGINS` in `remote-server/.env` and restart the backend.

The equivalent Cloudflare command for a temporary backend-only tunnel is:

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

For a named tunnel, map its public hostname to `http://127.0.0.1:8000` rather
than port 3001. Teammates using `ENV_MODE=LOCAL` set that HTTPS hostname as
`REMOTE_SERVER_URL`. Browser-direct use still requires that browser origin in
`CORS_ORIGINS`.

### Troubleshooting

| Symptom | Fix |
|---|---|
| Page loads, every search fails | Restart frontend with `NEXT_PUBLIC_API_URL=/` (§5a step 1) |
| `curl .../api/health` returns HTML | Tunnel points at 3000, not 3001 — `ngrok http 3001` |
| `curl .../api/health` refused/502 | `share-proxy.cjs` down or wrong `BACKEND_PORT` |
| Interstitial "you are about to visit" page | Normal on free ngrok — click through once, a cookie skips it after |
| CORS error in browser console | Only relevant to §5c browser-direct access — add the tunnel URL to `CORS_ORIGINS`, restart |
| `agent already running (port 4040 in use)` | A previous `ngrok` is still up — fine to reuse, or close it first |
| Cloudflare URL returns 502 | Confirm `share-proxy.cjs` is running on 3001 and the tunnel ingress service is `http://127.0.0.1:3001` |
| Cloudflare quick tunnel exits or is rate-limited | It is a development-only `trycloudflare.com` tunnel; create a named tunnel instead |

---

## After an update

```bash
git pull
cd local-client/local-backend && pip install -r requirements.txt   # only if requirements changed
cd ../frontend && npm install                                      # only if package.json changed
```

Restart the backend — required for new/edited strategies and any `.env`
change (`--reload` only covers `.py` edits).

**Ingested new lot archives?** Stop the backend first (Milvus Lite is
one-process-at-a-time), then:

```bash
cd remote-server
python scripts/ingest_zip_pipeline_results.py --skip-postgres
cd ../local-client/local-backend
python scripts/export_vectors_npy.py       # NOT optional — skipping it leaves search on stale vectors
cd ../../remote-server
python scripts/export_video_fps.py
```

Sanity check after any of the above:

```bash
curl http://localhost:8000/api/health
curl http://localhost:8000/api/strategies
```

---

## Using the app once it's running

Search by image (semantic scene description), by text (OCR/transcript), or
both, with optional temporal steps to find a sequence rather than one
frame. Click **?** in the app's left panel for the full in-app guide,
including the command-bar (`>_ Chat`) slash-commands and keyboard shortcuts.
Full standalone writeup: [archive/USAGE.md](archive/USAGE.md).

---

## Environment variables

Both backends load the repo-root `.env` first as shared defaults
(`CORS_ORIGINS`, PE-Core/ZIP tuning knobs, `NGROK_DOMAIN`), then their own
`.env` second, which wins on any key set in both. The frontend does not
participate — Next.js only reads `local-client/frontend/.env.local`.

Full variable-by-variable reference lives as comments in each
`.env.example`: [`.env.example`](../.env.example),
[`local-client/local-backend/.env.example`](../local-client/local-backend/.env.example),
[`remote-server/.env.example`](../remote-server/.env.example),
[`local-client/frontend/.env.local.example`](../local-client/frontend/.env.local.example).

---

## Common issues

For source-picture disagreements during offline processing, follow the
[verified decoder recovery workflow](DECODER_RECOVERY.md). Recovery keeps the
video release-blocked until its source and derived artifacts pass validation.

| Symptom | Fix |
|---|---|
| `Failed building wheel for pydantic-core` | No pre-built wheel for your Python version — ensure `requirements.txt` specifies `pydantic==2.10.6`+, retry |
| `Cannot connect to backend` | Backend not running, crashed on startup, or `NEXT_PUBLIC_API_URL` wrong in `.env.local` |
| `Strategy 'xxx' not found` | Check startup log for a `✗` line — syntax error, missing `name`/`description`/`author`, or failed import |
| Refuses to start in ZIP mode | Missing `vectors.f32.npy` — run the two export commands in §1 |
| Thumbnails 404, video falls back to YouTube | Missing `zip_video_index.json` — build it (§1) |

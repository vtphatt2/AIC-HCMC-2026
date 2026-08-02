# Setup Guide

## Choose Your Setup

| Goal | Backend | Data source | Section |
|---|---|---|---|
| Full demo on one GPU machine | `remote-server` | CAGRA/HNSW + PostgreSQL | **Recommended Full Demo** |
| UI or strategy smoke test | `local-backend` | Mock JSON | Option A |
| Search sample vectors without Docker | `local-backend` | `AIC2026_sample` | SAMPLE mode |
| Develop against another GPU server | `local-backend` | Remote raw-data proxy | Option B |

For the current complete demo with **CAGRA + local CTranslate2 INT8**, use the recommended
path below. You do not need `local-client/local-backend`.

## Recommended Full Demo

This runs all components on one GPU machine:

```text
Next.js frontend :3000
        |
        v
remote-server :8000
  +-- Local CTranslate2 INT8 VI→EN
  +-- PE-Core + CAGRA
  +-- PostgreSQL + Milvus
```

Complete Option C once for environment configuration, dataset ingestion, and
CAGRA index creation. For normal demo startup after that:

### 1. Start databases

```bash
cd remote-server
docker compose up -d
docker compose ps
```

Wait until MinIO is healthy and the other services are `Up`.

### 2. Start the remote backend

```bash
cd remote-server
source .venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000
```

Verify the active configuration:

```bash
curl http://localhost:8000/api/health
```

For CAGRA, the response should report `"vector_search_backend":"cagra"` and
`"pecore_device":"cuda"` with `"pecore_precision":"fp16"`. The VI→EN button
runs CTranslate2 INT8 on CPU and replaces the query input.

### 3. Start the frontend

In a second terminal:

```bash
cd local-client/frontend
npm install
npm run dev
```

Ensure `local-client/frontend/.env.local` contains:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Open http://localhost:3000.

Use Option A, SAMPLE mode, and Option B only for their specific development
workflows. The sections below contain first-time setup and troubleshooting
details.

### Teammate Frontend-Only Access

Teammates don't need the dataset, Milvus, PostgreSQL, CAGRA, or Google
credentials on their laptops — just the frontend, pointed at the host.

Host machine: run `remote-server` as above, and make sure `CORS_ORIGINS` in
its `.env` includes the frontend origin (`http://localhost:3000,http://127.0.0.1:3000`).

Teammate machine:

```bash
cd local-client/frontend && npm install && cp .env.local.example .env.local
```

Set `NEXT_PUBLIC_API_URL=http://<host-lan-ip>:8000` in `.env.local`, then
`npm run dev`. Verify with `curl http://<host-ip>:8000/api/health`.

Both machines need the same network, or a tunnel (Ngrok, Tailscale, VPN,
Cloudflare Tunnel) with `NEXT_PUBLIC_API_URL` set to the tunnel URL.

---

## Prerequisites

| Tool | Minimum version | Install |
|---|---|---|
| Python | 3.10+ | [python.org](https://python.org) |
| Node.js | 18+ | [nodejs.org](https://nodejs.org) |
| npm | 9+ | Included with Node.js |
| Docker Desktop | Latest | [docker.com](https://docker.com) — GPU server only |

---

## Option A — Local Development (Mock Data)

This is the default. No GPU server needed. All data comes from JSON files.

### 1. Clone the repo

```bash
git clone <repo-url>
cd AIC-HCMC-2026
```

### 2. Set up the local backend

```bash
cd local-client/local-backend

# Create a virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy the env file (MOCK mode is the default)
cp .env.example .env
```

`.env` contents (no changes needed for mock mode):
```env
ENV_MODE=MOCK
CORS_ORIGINS=http://localhost:3000
```

Start the backend:
```bash
uvicorn main:app --reload --port 8000
```

Expected startup output:
```
Starting local backend…
DataProvider: MOCK mode — 3 videos, 105 frames loaded
Discovering strategies…
  OK nam_visual_search_v1  [Nam Visual Search v1]  by Nam
  OK transcript_search  [Transcript Search v1]  by Team AIC 2026
Ready — 2 strategy/strategies available.
```

Verify: open http://localhost:8000/api/strategies — you should see a JSON list.

### 3. Set up the frontend

Open a **second terminal**:

```bash
cd local-client/frontend

# Install dependencies
npm install

# Copy the env file
cp .env.local.example .env.local
```

`.env.local` contents (no changes needed):
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Start the frontend:
```bash
npm run dev
```

Open http://localhost:3000. You should see the search UI with the strategy dropdown populated.

This MOCK setup is for UI and strategy development. VI→EN translation runs in
the local backend; point `NEXT_PUBLIC_API_URL` to `remote-server` only for the
complete GPU-search demo.

---

## Optional — Local Sample Search (SAMPLE mode)

To search local `AIC2026_sample` PE-Core vectors without Milvus/PostgreSQL,
set `ENV_MODE=SAMPLE`. The dataset is detected inside or beside the repository;
otherwise set `AIC_SAMPLE_ROOT`.

---

## Option B — Local Backend Connected to GPU Server (LOCAL mode)

Use this when you want to test strategies against real data from the GPU workstation.

### 1. Get the server URL

Ask whoever is running the GPU server for the Ngrok/LAN URL. It looks like `https://xxxx.ngrok.io` or `http://192.168.x.x:8000`.

### 2. Update your `.env`

Edit `local-client/local-backend/.env`:

```env
ENV_MODE=LOCAL
REMOTE_SERVER_URL=https://xxxx.ngrok.io
CORS_ORIGINS=http://localhost:3000
```

### 3. Restart the backend

```bash
uvicorn main:app --reload --port 8000
```

Expected startup output:
```
DataProvider: LOCAL mode → https://xxxx.ngrok.io
Discovering strategies…
  OK nam_visual_search_v1  [Nam Visual Search v1]  by Nam
  OK transcript_search  [Transcript Search v1]  by Team AIC 2026
Ready — 2 strategy/strategies available.
```

The frontend setup is identical to Option A.

---

## Option C — GPU Server (Production)

Run this on the GPU workstation. Requires Docker and the full dataset.

### 1. Start the databases

```bash
cd remote-server
docker compose up -d
```

Wait ~30 seconds for Milvus to initialise. Check with:
```bash
docker compose ps
```

All services should show `Up`.

### 2. Set up the backend

```bash
cd remote-server

python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

cp .env.example .env
```

Edit `.env`:
```env
ENV_MODE=SERVER

MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=video_frames
VECTOR_DIM=1280

POSTGRES_URL=postgresql://aic2026:aic2026@localhost:15432/aic2026

# Allow the contestant laptops to call this server
CORS_ORIGINS=http://localhost:3000,https://your-ngrok-url.ngrok.io
```

The portable default is Milvus HNSW. For the optional CAGRA GPU path:

```bash
pip install -r requirements-cagra.txt
python scripts/build_cagra_index.py
```

Then add:

```env
VECTOR_SEARCH_BACKEND=cagra
PECORE_DEVICE=cuda
PECORE_PRECISION=fp16
WARMUP_TEXT_ENCODER=true
```

CAGRA is CUDA-only. On an Apple silicon MacBook, use Milvus HNSW with MPS text
encoding instead:

```env
VECTOR_SEARCH_BACKEND=milvus
PECORE_DEVICE=mps
PECORE_PRECISION=fp32
```

For local Vietnamese-to-English translation (the ~78 MB INT8 model downloads once, then uses the local cache):

```env
TRANSLATION_MODEL_ID=dekthedev/opus-mt-vi-en-ct2-int8
TRANSLATION_MODEL_REVISION=14a921f3c4b7238b2b49d247e53810f0f7c78236
TRANSLATION_CPU_THREADS=4
WARMUP_TRANSLATION=true
```

Start the server (the launcher also starts Milvus/PostgreSQL):
```bash
bash ../scripts/start-remote.sh
```

On Windows use `..\scripts\start-remote.ps1`. Pass `--skip-databases` /
`-SkipDatabases` when Milvus and PostgreSQL are managed elsewhere.

### 3. Expose via Ngrok (for LOCAL mode contestants)

```bash
ngrok http 8000
```

Share the Ngrok URL with the team. They set it as `REMOTE_SERVER_URL` in their local `.env`.

### 4. Ingest sample data

Place `AIC2026_sample` inside or beside the repository. The scripts detect
both nested dataset folders (`keyframes/keyframes`, `metadata/metadata`,
`PECore-features/PECore-features`) and flat folders (`keyframes`, `metadata`,
`PECore-features`). For any other location, pass its path explicitly:

```bash
python scripts/ingest_embeddings_to_milvus.py \
  --sample-root /path/to/AIC2026_sample \
  --copy-keyframes
```

To enable runtime HNSW/FLAT/ScaNN selection, build all three Milvus collections:

```bash
python scripts/ingest_embeddings_to_milvus.py \
  --sample-root /path/to/AIC2026_sample \
  --copy-keyframes \
  --vector-index all \
  --recreate-milvus
```

The script upserts video metadata into PostgreSQL and frame embeddings into
Milvus, so it is safe to rerun after correcting metadata.

After ingestion, verify:
```bash
curl http://localhost:8000/api/health
# {"status":"ok","env_mode":"SERVER","strategies":2}
```

Before the first demo search, load the text model once:

```bash
curl -X POST http://localhost:8000/api/warmup_text_encoder
```

---

## Adding a New Strategy

Copy the template in [strategy_template_v2.md](strategy_template_v2.md) into
`local-client/local-backend/app/strategies/`; see [strategy_v2.md](strategy_v2.md)
for the data contract.

---

## Common Issues

### `Failed building wheel for pydantic-core`

Your Python version doesn't have a pre-built pydantic wheel. Make sure `requirements.txt` specifies `pydantic==2.10.6` or later, then retry.

### `Cannot connect to backend. Is the local backend running?`

The frontend can't reach `localhost:8000`. Check:
- Is the backend terminal still running?
- Did it start without errors?
- Is `NEXT_PUBLIC_API_URL` in `.env.local` set to `http://localhost:8000`?

### `Strategy 'xxx' not found`

The backend didn't load your strategy. Check the startup log for a `✗` line with an error message. Common causes: syntax error in the file, missing `name`/`description`/`author` attributes, or a failed import.

### Backend loads 30 frames instead of 105 after updating mock_frames.json

Uvicorn's `--reload` only watches `.py` files. Restart the backend manually (Ctrl+C, then `uvicorn main:app --reload --port 8000`).

### YouTube player doesn't seek to the right time

The `seekTo()` call requires the video to be loaded. Make sure the video ID in the mock data (`youtube_id` field) is a real, publicly available YouTube video.

---

## Environment Variable Reference

### `local-client/local-backend/.env`

| Variable | Default | Description |
|---|---|---|
| `ENV_MODE` | `MOCK` | `MOCK`, `SAMPLE`, or `LOCAL` |
| `AIC_SAMPLE_ROOT` | auto-detected | Optional dataset path for `SAMPLE` mode |
| `REMOTE_SERVER_URL` | _(empty)_ | Required when `ENV_MODE=LOCAL` |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed origins |
| `PECORE_BACKEND` | `torch` | `torch` (full model) or `onnx` (lightweight, no torch download) — see [PE-Core-bigG-14-448-Text-Encoder.README.md](PE-Core-bigG-14-448-Text-Encoder.README.md) |
| `FRAME_IMAGE_SOURCE` | `local` | `local` (JPG), `local_video` (ffmpeg over `data/videos`), or `youtube_storyboard` fallback |

See [running.md](running.md) for the full scenario matrix and exact commands.

### `remote-server/.env`

| Variable | Default | Description |
|---|---|---|
| `ENV_MODE` | `SERVER` | Should always be `SERVER` on the GPU machine |
| `MILVUS_HOST` | `localhost` | Milvus hostname |
| `MILVUS_PORT` | `19530` | Milvus port |
| `MILVUS_COLLECTION` | `video_frames` | Default Milvus HNSW collection name |
| `MILVUS_COLLECTION_HNSW` | `video_frames` | HNSW collection used for runtime selection |
| `MILVUS_COLLECTION_FLAT` | `video_frames_flat` | FLAT collection used for runtime selection |
| `MILVUS_COLLECTION_SCANN` | `video_frames_scann` | ScaNN collection used for runtime selection |
| `VECTOR_DIM` | `1280` | Embedding dimension (PE-Core-bigG-14-448) |
| `VECTOR_SEARCH_BACKEND` | `milvus` | `milvus`/`hnsw`, `flat`, `scann`, or `cagra` |
| `POSTGRES_URL` | _(see .env.example)_ | Full asyncpg connection string |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed origins |
| `PECORE_BACKEND` | `torch` | `torch` (full model) or `onnx` (lightweight, no torch download) |
| `PECORE_DEVICE` | `cpu` | `cpu`, `cuda`, or `mps`; use `mps` on Apple silicon |
| `PECORE_PRECISION` | `fp32` | Use `fp16` for CUDA/CAGRA; keep `fp32` for CPU/MPS |
| `WARMUP_TEXT_ENCODER` | `false` | Load and warm PE-Core during startup |
| `TRANSLATION_MODEL_ID` | `dekthedev/opus-mt-vi-en-ct2-int8` | Local CTranslate2 INT8 translation model |
| `TRANSLATION_MODEL_REVISION` | pinned commit | Reproducible model revision |
| `TRANSLATION_CPU_THREADS` | `4` | CPU threads used by CTranslate2 |
| `TRANSLATION_MODEL_PATH` | _(empty)_ | Optional pre-downloaded local model directory |
| `WARMUP_TRANSLATION` | `false` | Initialize translation during startup |

### `local-client/frontend/.env.local`

| Variable | Default | Description |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | FastAPI backend used by the frontend |

# Setup Guide

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
DataProvider: MOCK mode — 3 videos, 105 frames loaded
Discovering strategies…
  ✓ example_strategy  [Example (Mock)]  by Team AIC 2026
Ready — 1 strategy/strategies available.
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
  ✓ example_strategy  [Example (Mock)]  by Team AIC 2026
Ready — 1 strategy/strategies available.
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

Start the server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

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

The script upserts video metadata into PostgreSQL and frame embeddings into
Milvus, so it is safe to rerun after correcting metadata.

After ingestion, verify:
```bash
curl http://localhost:8000/api/health
# {"status":"ok","env_mode":"SERVER","strategies":1}
```

Before the first demo search, load the text model once:

```bash
curl -X POST http://localhost:8000/api/warmup_text_encoder
```

---

## Adding a New Strategy

```bash
# 1. Copy the template
cp local-client/local-backend/app/strategies/example_strategy.py \
   local-client/local-backend/app/strategies/yourname_v1.py

# 2. Edit the file — set name, description, author, implement fusion_and_temporal()

# 3. Restart the backend (Ctrl+C then uvicorn again, or save any .py file if --reload is on)
```

Your strategy now appears in the frontend dropdown. See [strategy_guide.md](strategy_guide.md) for full documentation.

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

### `remote-server/.env`

| Variable | Default | Description |
|---|---|---|
| `ENV_MODE` | `SERVER` | Should always be `SERVER` on the GPU machine |
| `MILVUS_HOST` | `localhost` | Milvus hostname |
| `MILVUS_PORT` | `19530` | Milvus port |
| `MILVUS_COLLECTION` | `video_frames` | Milvus collection name |
| `VECTOR_DIM` | `1280` | Embedding dimension (PE-Core-bigG-14-448) |
| `POSTGRES_URL` | _(see .env.example)_ | Full asyncpg connection string |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed origins |
| `PECORE_DEVICE` | `cpu` | Set to `cuda` on a CUDA-capable server |

### `local-client/frontend/.env.local`

| Variable | Default | Description |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Local backend URL |

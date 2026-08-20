# Setup Guide

## Choose Your Setup

There is one dataset — the lot archives in `challenge_resources/data/zip_file/`.

| Goal | Backend | Reads | Section |
|---|---|---|---|
| Full demo on one GPU machine | `remote-server` | Milvus/CAGRA + PostgreSQL | **Recommended Full Demo** |
| Develop strategies on your own machine | `local-backend` | `vectors.f32.npy` exported from the archives | Option A |
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
  +-- Google Translate (free) VI→EN
  +-- PE-Core + CAGRA
  +-- PostgreSQL + Milvus (or embedded Milvus Lite)
```

Complete Option C once for environment configuration, dataset ingestion, and
CAGRA index creation. For normal demo startup after that:

### 1. Start databases

Two options — pick one:

**Docker (real Milvus server):**
```bash
cd remote-server
docker compose up -d
docker compose ps
```

Wait until MinIO is healthy and the other services are `Up`.

**No Docker, dev/test-only** (embedded Milvus Lite + portable PostgreSQL):
`remote-server` is meant to run against a real Milvus server in production —
this path is only for developing/testing `remote-server` code on a laptop.
Needs `pymilvus>=2.4` (the base `requirements.txt` pins `2.3.7` for the real
server), so also install `pip install -r requirements-milvus-lite.txt`. Then
set `MILVUS_LITE_PATH=../challenge_resources/data/milvus_lite.db` in
`remote-server/.env` (skips `MILVUS_HOST`/`MILVUS_PORT` entirely — no server
process, just a file), and start PostgreSQL via the bundled helper script
instead of Docker:

```bash
cd remote-server
bash scripts/start-local-postgres.sh start   # scoop-installed portable postgres, port 15432
bash scripts/start-local-postgres.sh status
```

`local-client/local-backend` points `MILVUS_LITE_PATH` at the **same** file —
it is also where the exported `vectors.f32.npy` lives. See
[running.md](running.md).

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
calls Google Translate's free web endpoint (no local model, no API key) and
replaces the query input.

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

Use Option A and Option B only for their specific development workflows. The
sections below contain first-time setup and troubleshooting details.

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

## Option A — Local Development (ZIP mode)

No GPU server needed. Search runs against vectors exported from the lot
archives, on this machine.

### 1. Clone the repo

```bash
git clone <repo-url>
cd AIC-HCMC-2026
```

### 2. Get the data

You need `challenge_resources/data/zip_file/*_results.zip` (the lot archives),
then two derived files. Both are cheap and both must be rebuilt after every
ingest:

```bash
cd remote-server
python scripts/ingest_zip_pipeline_results.py --skip-postgres   # → Milvus records
python scripts/export_video_fps.py                              # → video_fps.json  (~0.1s)

cd ../local-client/local-backend
python scripts/export_vectors_npy.py                            # → vectors.f32.npy (~11s)
```

See [Readme-Ingest.md](../challenge_resources/data/zip_file/Readme-Ingest.md)
for flags and the one-process-at-a-time caveat.

### 3. Set up the local backend

```bash
cd local-client/local-backend

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

`.env` needs the data path — the vector files sit next to `MILVUS_LITE_PATH`:

```env
ENV_MODE=ZIP
AIC_SAMPLE_ROOT=/abs/path/to/AIC-HCMC-2026/challenge_resources/data
MILVUS_LITE_PATH=/abs/path/to/AIC-HCMC-2026/challenge_resources/data/milvus_lite.db
CORS_ORIGINS=http://localhost:3000
```

Start it:

```bash
uvicorn main:app --reload --port 8000
```

Expected startup output:

```
Starting local backend...
DataProvider: ZIP mode -> numpy memmap (exact), 193508 vectors
Discovering strategies...
  OK raw_visual  [Raw visual]  by AIC HCMC
  ...
Ready - 8 strategy/strategies available.
```

If it refuses to start, the two export commands above are what it is asking for.

### 4. Thumbnails and playback

Frames live in the organizers' *video* archives, not the results archives. Build
the byte-offset index once:

```bash
python -m scripts.build_zip_video_index \
    --urls-file ../../challenge_resources/data/zip_video_links.txt \
    --output   ../../challenge_resources/data/zip_video_index.json
```

Without it every thumbnail 404s and the UI falls back to YouTube — usable, but
you lose the frame grid. Full detail: [zip_media.md](zip_media.md).

### 5. Set up the frontend

Open a **second terminal**:

```bash
cd local-client/frontend
npm install
cp .env.local.example .env.local
```

`.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

```bash
npm run dev
```

Open http://localhost:3000 — the strategy dropdown should be populated.

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
DataProvider: LOCAL mode -> https://xxxx.ngrok.io
Discovering strategies...
Ready - 8 strategy/strategies available.
```

The frontend setup is identical to Option A. In this mode all four retrieval
channels come from the server, including the two the lot archives do not carry
(`subtitled.semantic`, `transcript.semantic`).

---

## Option C — GPU Server (Production)

Run this on the GPU workstation, with the full dataset. Docker is the default
path; a no-Docker path (embedded Milvus Lite + portable PostgreSQL) also
works for lighter/offline dev — see the "No Docker" callout under
**Recommended Full Demo** above.

This section is first-time setup. Once it is set up, the day-to-day start,
restart, and post-update commands are in
[running.md → Remote server](running.md#remote-server-server-mode).

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

Vietnamese-to-English translation uses Google Translate's free web endpoint
(`deep-translator`, no API key, no model download) — nothing to configure
beyond the optional warmup flag:

```env
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
Full account/domain setup and troubleshooting: [ngrok.md](ngrok.md#exposing-the-remote-server).

### 4. Ingest the dataset

```bash
cd remote-server
python scripts/ingest_zip_pipeline_results.py --dry-run   # preview counts
python scripts/ingest_zip_pipeline_results.py             # --vector-index all for runtime HNSW/FLAT/ScaNN switching
python scripts/export_video_fps.py                        # required: frame ↔ timestamp
```

Safe to rerun — it upserts by `frame_id`/`video_id`. Full flag reference:
[../remote-server/README_INDEXING_SEARCH.md](../remote-server/README_INDEXING_SEARCH.md#4-ingest-the-lot-archives).

### 5. Media

Drop the organizers' `Videos_L*.zip` into `challenge_resources/data/raw_zip/`.
Nothing to build — `/api/zip-frame` and `/api/zip-video` read byte ranges out of
them directly, and `/api/health` reports how many videos were found. Lots whose
archive is absent fall back to YouTube playback with no thumbnail.

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

### YouTube player doesn't seek to the right time

The `seekTo()` call requires the video to be loaded. Make sure the video ID in the mock data (`youtube_id` field) is a real, publicly available YouTube video.

---

## Environment Variable Reference

### `local-client/local-backend/.env`

| Variable | Default | Description |
|---|---|---|
| `ENV_MODE` | `ZIP` | `ZIP` (search the exported lot vectors) or `LOCAL` (proxy to the GPU server) |
| `AIC_SAMPLE_ROOT` | auto-detected | `challenge_resources/data`; where transcripts and the vector files live |
| `REMOTE_SERVER_URL` | _(empty)_ | Required when `ENV_MODE=LOCAL` |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed origins |
| `PECORE_BACKEND` | `torch` | `torch` (full model) or `onnx` (lightweight, no torch download) — see [PE-Core-bigG-14-448-Text-Encoder.README.md](PE-Core-bigG-14-448-Text-Encoder.README.md) |
| `PECORE_DEVICE` | `cpu` | `cpu` or `mps` (Apple silicon) |
| `PECORE_PRECISION` | `fp32` | Keep `fp32` on CPU/MPS |
| `PECORE_ONNX_THREADS` | `14` | ONNX Runtime intra-op threads per text-encode. Measured on 20 cores: 105 ms at 14, 115 ms at 20 — past bandwidth saturation more threads only contend |
| `WARMUP_TEXT_ENCODER` | `false` | Load PE-Core at startup instead of on the first query (~3.3 s) |
| `MILVUS_LITE_PATH` | _(unset)_ | Path to the data directory's `milvus_lite.db`. Also tells the backend where `vectors.f32.npy`/`.meta.npz` live — **set this even though the fast path does not query Milvus**. Search priority: numpy memmap → Milvus Lite FLAT → linear `.npy` scan |
| `VECTOR_SEARCH_BACKEND` | `milvus` | Only consulted on the Milvus Lite fallback path. Must be `flat` — its HNSW index returns wrong neighbours ([doc](milvus-lite-hnsw-recall-bug.md)) |
| `NUMPY_SEARCH_CHUNK_ROWS` | `65536` | Rows scanned per block in the exact-search path. Bounds peak memory; barely affects speed |
| `ZIP_FRAME_CONCURRENCY` | `12` | In-flight requests on `/api/zip-frame` |
| `ZIP_FRAME_DECODE_CONCURRENCY` | `6` | Concurrent ffmpeg processes (CPU-bound) |
| `ZIP_REGION_CACHE_MB` | `192` | In-memory GOP byte cache |
| `ZIP_MOOV_CACHE_DIR` | `cache/zip_moov/` | On-disk MP4 `moov` cache |
| `GEMINI_API_KEY` | _(unset)_ | Enables `QueryParser`; only strategies calling `context.parse_json()` need it |
| `GEMINI_QUERY_MODEL` | `gemini-3.1-flash-lite` | Model used by that parser |

The local backend also exposes the organizer-ZIP media routes
(`/api/zip-video`, `/api/zip-frame`, `/api/zip-frame-plan`, `/api/zip-bytes`).
They need no env vars beyond the tuning knobs above — just a
`challenge_resources/data/zip_video_index.json` manifest, built once. Full
walkthrough: [zip_media.md](zip_media.md).

See [running.md](running.md) for the full scenario matrix and exact commands.

### `remote-server/.env`

| Variable | Default | Description |
|---|---|---|
| `ENV_MODE` | `SERVER` | Should always be `SERVER` on the GPU machine |
| `MILVUS_LITE_PATH` | _(unset)_ | Optional: path to an embedded Milvus Lite file. When set, skips `MILVUS_HOST`/`MILVUS_PORT`/Docker entirely and uses that file instead. |
| `MILVUS_HOST` | `localhost` | Milvus hostname (ignored when `MILVUS_LITE_PATH` is set) |
| `MILVUS_PORT` | `19530` | Milvus port (ignored when `MILVUS_LITE_PATH` is set) |
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
| `WARMUP_TRANSLATION` | `false` | Initialize the Google Translate client during startup (no model to download) |
| `FRAME_STATIC_DIR` | `remote-server/static/frames` | Directory served at `/static/frames`. Lot archives ship no JPGs, so this is normally empty and thumbnails come from `/api/zip-frame` |
| `RAW_ZIP_DIR` | `challenge_resources/data/raw_zip` | Where `Videos_L*.zip` archives live, for `/api/zip-frame` and `/api/zip-video` |
| `VIDEO_FPS_MAP` | `challenge_resources/data/video_fps.json` | Precomputed `video_id → fps`, built by `scripts/export_video_fps.py` |
| `ZIP_FRAME_TIMEOUT_SEC` | `45` | Wall-clock bound on one frame request |
| `ZIP_FRAME_DECODE_CONCURRENCY` | `6` | Concurrent ffmpeg processes |
| `ALLOW_STRATEGY_CONFIG_WRITES` | `false` | Keep `false` on a shared server — clients tune via per-request `config_overrides` instead |
| `TRANSCRIPT_CHUNK_SEARCH_ENABLED` | `true` | Topic-based transcript chunk search ([doc](search_by_transcript.md)) |
| `TRANSCRIPT_MODEL_ID` | `intfloat/multilingual-e5-small` | Sentence-transformer for chunk embeddings |
| `TRANSCRIPT_MODEL_DEVICE` | `cpu` | Device for that model |
| `MILVUS_TRANSCRIPT_COLLECTION` | `transcript_chunks` | Transcript chunk collection |
| `TRANSCRIPT_VECTOR_DIM` | `384` | Dimension of that collection |
| `WARMUP_TRANSCRIPT_SEARCH` | `true` | Load the transcript model at startup |
| `GEMINI_API_KEY` / `GEMINI_QUERY_MODEL` | _(unset)_ | Structured query parser, used only by strategies calling `context.parse_json()` |

remote-server serves thumbnails and playback from `raw_zip/Videos_L*.zip`
directly ([zip_media.md](zip_media.md)); lots whose archive is not on its disk
return 404 and the UI falls back to YouTube.

### `local-client/frontend/.env.local`

| Variable | Default | Description |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | FastAPI backend used by the frontend |
| `NEXT_PUBLIC_FRAME_DECODE` | `server` | `client` decodes thumbnails in the browser with WebCodecs — decode cost then scales with viewers instead of stacking on the backend ([doc](zip_media.md#4-client-side-decode-optional-recommended-when-sharing)) |

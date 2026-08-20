# Running — Quick Reference

One page to pick your scenario and get a command to run. Full walkthroughs and
rationale live in the linked docs — this page is deliberately just the decision
table plus commands.

- Already running and just pulled/ingested? → [After an update](#after-an-update).
- Want to know what the code does, not how to start it? → [backend_flow.md](backend_flow.md).

## Which scenario am I in?

There is one dataset — the lot archives in `challenge_resources/data/zip_file/` —
and three ways to reach it:

| I have… | Use | Backend command |
|---|---|---|
| The lot archives on this machine | ZIP | [below](#zip-mode--the-common-local-dev-case) |
| Access to a teammate's running GPU server (ngrok/LAN URL) | LOCAL | [setup.md Option B](setup.md#option-b--local-backend-connected-to-gpu-server-local-mode) |
| The GPU workstation itself | SERVER | [Remote server](#remote-server-server-mode) |

`MOCK` and `SAMPLE` are gone — see [architecture.md](architecture.md#env_mode).

All local-backend scenarios need the UI too:

```bash
cd local-client/frontend && npm install && cp .env.local.example .env.local && npm run dev
```

Or start backend + frontend together: [launch_scripts.md](launch_scripts.md).

---

## ZIP mode — the common local dev case

```bash
cd local-client/local-backend
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate elsewhere)
pip install -r requirements.txt   # base: no torch, ONNX text encoder by default
```

`.env`:

```env
ENV_MODE=ZIP
AIC_SAMPLE_ROOT=D:\path\to\AIC-HCMC-2026\challenge_resources\data
PECORE_BACKEND=onnx

# Where the vector files live. Required for the fast exact-search path even
# though search does not go through Milvus Lite itself — the .npy files sit
# next to this database.
MILVUS_LITE_PATH=D:\path\to\challenge_resources\data\milvus_lite.db
```

```bash
uvicorn main:app --reload --port 8000
```

The startup log states which search backend was selected — read it, it is the
fastest way to catch a misconfigured path:

```
DataProvider: ZIP mode -> numpy memmap (exact), 193508 vectors
```

### Decision points inside ZIP mode

| Situation | What to do |
|---|---|
| Weak machine, no GPU | Keep `PECORE_BACKEND=onnx` (the default in `requirements.txt`). No torch installed. |
| Want the full-precision OpenCLIP model | `pip install -r requirements-torch.txt`, set `PECORE_BACKEND=torch`. See [PE-Core-bigG-14-448-Text-Encoder.README.md](PE-Core-bigG-14-448-Text-Encoder.README.md). |
| Thumbnails and playback | Build `zip_video_index.json` once → [zip_media.md](zip_media.md). Without it every frame 404s and the UI falls back to YouTube. |
| Frames look close but not exact | Run `remote-server/scripts/export_video_fps.py` → [zip_media.md §5](zip_media.md#5-getting-the-frame-right). |
| Sharing one backend with teammates | `NEXT_PUBLIC_FRAME_DECODE=client` in the frontend `.env.local` moves thumbnail decoding into each browser ([zip_media.md §4](zip_media.md#4-client-side-decode-optional-recommended-when-sharing)). |

### Which vector backend answers a search

Priority, first available wins:

1. **numpy memmap** (`vectors.f32.npy` next to `MILVUS_LITE_PATH`) — exact,
   ~35 ms at `top_k=1000`. This is the intended path.
2. **Milvus Lite** — fallback if the export was never run. ~45× slower, and it
   must stay on the FLAT index ([milvus-lite-hnsw-recall-bug.md](milvus-lite-hnsw-recall-bug.md)).

`pip install -r requirements-milvus-lite.txt` is only needed for path 2. If
neither exists the backend refuses to start and prints the commands that build
them.

---

## After an update

### Pulled new code

```bash
git pull
cd local-client/local-backend && pip install -r requirements.txt   # only if requirements changed
cd ../frontend && npm install                                      # only if package.json changed
```

Then restart the backend. Restart is required, not optional, for new or edited
strategies and for any `.env` change; `uvicorn --reload` covers `.py` edits only.

Check the `.env.example` diff after a pull — new tuning knobs land there first:

```bash
git diff HEAD@{1} -- local-client/local-backend/.env.example remote-server/.env.example .env.example
```

### Ingested new lot archives

Two steps, both required, in this order — and **stop the backend first**
(Milvus Lite allows one process at a time, and Windows refuses to overwrite a
memory-mapped `.npy`):

```bash
cd remote-server
python scripts/ingest_zip_pipeline_results.py --dry-run --skip-postgres   # preview counts
python scripts/ingest_zip_pipeline_results.py --skip-postgres

cd ../local-client/local-backend
python scripts/export_vectors_npy.py                                      # ~11s, NOT optional

cd ../../remote-server
python scripts/export_video_fps.py                                        # fps map, ~0.1s
```

Skipping the export is silent: search keeps running on the previous ingest's
vectors — a valid file, just older. The backend prints a warning at startup when
it detects that. Full detail and flags:
[Readme-Ingest.md](../challenge_resources/data/zip_file/Readme-Ingest.md).

Restart the backend once all three finish.

### Added new video archives to the ZIP index

```bash
cd local-client/local-backend
python -m scripts.build_zip_video_index \
    --urls-file ../../challenge_resources/data/zip_video_links.txt \
    --output   ../../challenge_resources/data/zip_video_index.json
```

The index is read once at startup, so restart afterwards. See
[zip_media.md](zip_media.md).

### Sanity check after any of the above

```bash
curl http://localhost:8000/api/health
curl http://localhost:8000/api/strategies
```

---

## Remote server (SERVER mode)

On the GPU workstation. One command, from the repo root:

```powershell
scripts\start-remote.ps1                      # Docker Milvus+Postgres, then the backend
scripts\start-remote.ps1 -SkipDatabases -Reload
```

```bash
bash scripts/start-remote.sh
bash scripts/start-remote.sh --skip-databases --reload
```

It expects a venv at `remote-server/.venv` (dependency choice is machine-specific,
so it is not created for you), creates `remote-server/.env` from `.env.example`
when missing, waits for ports 19530/15432, then runs uvicorn on 0.0.0.0:8000 in
the current terminal. It does **not** start the frontend.

First-time setup, ingestion, CAGRA, and the validation scripts:
[README_INDEXING_SEARCH.md](../remote-server/README_INDEXING_SEARCH.md).

### Without Docker

```env
MILVUS_LITE_PATH=../challenge_resources/data/milvus_lite.db
```

plus `bash remote-server/scripts/start-local-postgres.sh start` for a portable
Postgres, then `--skip-databases`. This is a dev/test convenience — the
production path is a real Milvus server.

### Before a demo

```bash
curl http://localhost:8000/api/health
curl -X POST http://localhost:8000/api/warmup_text_encoder    # or WARMUP_TEXT_ENCODER=true
```

Or set `WARMUP_TEXT_ENCODER=true` / `WARMUP_TRANSLATION=true` in
`remote-server/.env` so startup pays the cost instead of the first query.

### Serving clients

| Client | Setting |
|---|---|
| Frontend straight at the server | `NEXT_PUBLIC_API_URL=http://<server>:8000` |
| Teammate's local-backend proxying | `ENV_MODE=LOCAL` + `REMOTE_SERVER_URL=<url>` |
| Over ngrok | expose port 8000; `CORS_ORIGINS` must list the caller's origin |

### Thumbnails and playback

remote-server serves `/api/zip-frame` and `/api/zip-video` from its own copy of
the video archives in `challenge_resources/data/raw_zip/`. Drop `Videos_L*.zip`
in there — nothing to build — and check `local_zip_videos` in `/api/health`.
Lots whose archive is missing return 404 and the UI falls back to YouTube.
Details: [zip_media.md](zip_media.md).

---

## Everything else

See the [documentation map](../README.md#documentation-map) in the README.

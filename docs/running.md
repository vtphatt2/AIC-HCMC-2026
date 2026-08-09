# Running & Setup — Quick Reference

One page to pick your scenario and get a command to run. Full walkthroughs
and rationale live in the linked docs — this page is deliberately just the
decision table + commands.

## Which scenario am I in?

| I have... | Use | Backend command |
|---|---|---|
| Nothing, just want the UI working | MOCK | `ENV_MODE=MOCK` → [setup.md Option A](setup.md#option-a--local-development-mock-data) |
| `AIC2026_sample` (or a partial copy: `metadata/` + `PECore-features/`, no `keyframes/`) | SAMPLE | see below |
| Access to a teammate's running GPU server (ngrok/LAN URL) | LOCAL | [setup.md Option B](setup.md#option-b--local-backend-connected-to-gpu-server-local-mode) |
| The GPU workstation itself, full dataset | SERVER | `bash scripts/start-remote.sh` from repo root (Docker Milvus+Postgres by default; set `MILVUS_LITE_PATH` + `scripts/start-local-postgres.sh` to skip Docker) |

All local-backend scenarios: `cd local-client/frontend && npm install && cp .env.local.example .env.local && npm run dev` for the UI (port 3000).

## SAMPLE mode — the common local dev case

```bash
cd local-client/local-backend
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt   # base: no torch, ONNX text encoder by default
```

`.env`:
```env
ENV_MODE=SAMPLE
AIC_SAMPLE_ROOT=D:\path\to\your\dataset
PECORE_BACKEND=onnx
```

```bash
uvicorn main:app --reload --port 8000
```

Decision points inside SAMPLE mode:

| Situation | What to do |
|---|---|
| Weak machine, no GPU, want fast/light setup | Keep `PECORE_BACKEND=onnx` (default in requirements.txt). No torch installed. |
| Want the full-precision OpenCLIP model instead | `pip install -r requirements-torch.txt`, set `PECORE_BACKEND=torch`. See [PE-Core-bigG-14-448-Text-Encoder.README.md](PE-Core-bigG-14-448-Text-Encoder.README.md). |
| Have `videos/<video_id>.mp4` but no keyframe JPGs | Set `FRAME_IMAGE_SOURCE=local_video`; the backend decodes the requested timestamp locally with ffmpeg. |
| No local videos or keyframe JPGs | `FRAME_IMAGE_SOURCE=youtube_storyboard` is an approximate dev-only fallback. See [youtube-storyboard-thumbnails-workaround.md](youtube-storyboard-thumbnails-workaround.md). |
| Have real `keyframes/` images | Leave `FRAME_IMAGE_SOURCE=local` (default). |
| Have `challenge_resources/data/zip_video_index.json` (organizer video ZIPs) | Video playback and frame thumbnails can come straight from `/api/zip-video`, `/api/zip-frame` — no local `keyframes/`/`videos/` needed. Build the index once: `python scripts/build_zip_video_index.py`. |
| Ingested via `ingest_zip_pipeline_results.py` and want faster repeated search | `pip install -r requirements-milvus-lite.txt`, set `MILVUS_LITE_PATH=<repo>/challenge_resources/data/milvus_lite.db`. Falls back to linear search automatically if unset. |

## Full env var reference

See [setup.md → Environment Variable Reference](setup.md#environment-variable-reference).

## Everything else

| Topic | Doc |
|---|---|
| One-command launch scripts (Windows/Mac/Linux/WSL, backend/GPU choice) | [launch_scripts.md](launch_scripts.md) |
| System design, ENV_MODE switching, data flow | [architecture.md](architecture.md) |
| Strategy/data contract | [strategy_v2.md](strategy_v2.md) |
| Writing a new strategy | [strategy_template_v2.md](strategy_template_v2.md) |
| Milvus/Postgres schema | [db_schema.md](db_schema.md) |
| PE-Core ingestion, CAGRA, translation, validation (remote-server) | [../remote-server/README_INDEXING_SEARCH.md](../remote-server/README_INDEXING_SEARCH.md) |
| Transcript search | [search_by_transcript.md](search_by_transcript.md) |
| Repo layout | [../README.md#repository-layout](../README.md#repository-layout) |
| PE-Core ONNX text encoder specs | [PE-Core-bigG-14-448-Text-Encoder.README.md](PE-Core-bigG-14-448-Text-Encoder.README.md) |
| Keyframe sampling rule | [keyframe_selection.md](keyframe_selection.md) |
| Vector search backend benchmark (HNSW vs CAGRA vs ScaNN) | [vector_search_benchmark_report.md](vector_search_benchmark_report.md) |

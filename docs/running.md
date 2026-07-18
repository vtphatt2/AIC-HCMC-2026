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
| The GPU workstation itself, full dataset, Docker | SERVER | [setup.md Option C](setup.md#option-c--gpu-server-production) |

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
| No `keyframes/` folder (only `metadata/` + `PECore-features/`) | Search still works (vectors are all that's needed). For images: `pip install -r requirements-youtube-thumbnail.txt` and set `FRAME_IMAGE_SOURCE=youtube_storyboard`. See [youtube-storyboard-thumbnails-workaround.md](youtube-storyboard-thumbnails-workaround.md). Not pixel-accurate — dev/test only. |
| Have real `keyframes/` images | Leave `FRAME_IMAGE_SOURCE=local` (default). |

## Full env var reference

See [setup.md → Environment Variable Reference](setup.md#environment-variable-reference).

## Everything else

| Topic | Doc |
|---|---|
| One-command launch scripts (Windows/Mac/Linux/WSL, backend/GPU choice) | [launch_scripts.md](launch_scripts.md) |
| System design, ENV_MODE switching, data flow | [architecture.md](architecture.md) |
| Writing a new strategy | [strategy_guide.md](strategy_guide.md) |
| Milvus/Postgres schema | [db_schema.md](db_schema.md) |
| PE-Core ingestion, CAGRA, translation, validation (remote-server) | [../remote-server/README_INDEXING_SEARCH.md](../remote-server/README_INDEXING_SEARCH.md) |
| Transcript search | [search_by_transcript.md](search_by_transcript.md) |
| Repo layout | [folder_tree.md](folder_tree.md) |

# AIC HCMC 2026 — Video Retrieval System

A video retrieval playground for the **Ho Chi Minh City AI Challenge 2026**
(Video Browser Showdown format), with lightweight local development modes and a
GPU-backed production mode.

One dataset — the organizers' lot archives — reached three ways: searched
locally from exported vectors, proxied to a teammate's GPU server, or served
by that server directly.

---

## Documentation Map

**New here?** [running.md](docs/running.md) → [architecture.md](docs/architecture.md)
→ [backend_flow.md](docs/backend_flow.md) → [strategy_template_v2.md](docs/strategy_template_v2.md).
That is the whole onboarding path; everything below is reference.

### Running it

| Document | Description |
|---|---|
| [docs/running.md](docs/running.md) | **Start here** — scenario picker, exact commands, what to re-run after an update |
| [docs/setup.md](docs/setup.md) | First-time setup, step by step, for local dev and the GPU server |
| [docs/launch_scripts.md](docs/launch_scripts.md) | One-command local/remote launch (Windows/Mac/Linux/WSL), ngrok sharing |
| [docs/ngrok.md](docs/ngrok.md) | Ngrok setup + both sharing scenarios: local dev instance, remote server |
| [docs/USAGE.md](docs/USAGE.md) | Using the search UI: query modes, command bar, results, player |
| [remote-server/README_INDEXING_SEARCH.md](remote-server/README_INDEXING_SEARCH.md) | remote-server: ingestion, HNSW/CAGRA, validation scripts |

### How it works

| Document | Description |
|---|---|
| [docs/architecture.md](docs/architecture.md) | System design, ENV_MODE switching, data flow, duplicate filtering |
| [docs/backend_flow.md](docs/backend_flow.md) | Backend code flow: startup, request path, module map, endpoint map |
| [docs/zip_media.md](docs/zip_media.md) | Frames and video playback out of the organizer ZIPs, without downloading them |
| [docs/db_schema.md](docs/db_schema.md) | PostgreSQL DDL + Milvus collection schema |
| [docs/gaps.md](docs/gaps.md) | What is documented or implied but **not implemented** |

### Writing strategies

| Document | Description |
|---|---|
| [docs/strategy_template_v2.md](docs/strategy_template_v2.md) | **How to write your own strategy** — copy this |
| [docs/strategy_v2.md](docs/strategy_v2.md) | Strategy/data contract: channels, SearchContext, hit shape |
| [docs/search_by_transcript.md](docs/search_by_transcript.md) | Topic-based transcript chunking + vector search |
| [docs/reranking_hybrid_notes.md](docs/reranking_hybrid_notes.md) | Ideas and measurements for hybrid reranking |

### Models, data, and measurements

| Document | Description |
|---|---|
| [docs/PE-Core-bigG-14-448-Text-Encoder.README.md](docs/PE-Core-bigG-14-448-Text-Encoder.README.md) | Lightweight ONNX text encoder (no torch) |
| [docs/keyframe_selection.md](docs/keyframe_selection.md) | Keyframe sampling rule |
| [challenge_resources/data/zip_file/Readme-Ingest.md](challenge_resources/data/zip_file/Readme-Ingest.md) | What the lot archives contain and how to ingest them |
| [docs/performance_pain_points.md](docs/performance_pain_points.md) | Every measured bottleneck, what fixed it, and the wrong turns |
| [docs/vector_search_benchmark_report.md](docs/vector_search_benchmark_report.md) | HNSW vs CAGRA vs ScaNN |
| [docs/milvus-lite-hnsw-recall-bug.md](docs/milvus-lite-hnsw-recall-bug.md) | Why local search does not use Milvus Lite's HNSW |

### Sub-projects (own docs, own lifecycle)

| Directory | What it is |
|---|---|
| [keyframe_pipeline_global_v9_3/](keyframe_pipeline_global_v9_3/README.md) | The GPU keyframe + embedding pipeline that produces `*_results.zip` |
| [preprocess/](preprocess/README.md) | Dataset preprocessing utilities |
| [notebooks/](notebooks/NOTEBOOK_AUDIT.md) | EDA and evaluation notebooks |
| [paper_draft/](paper_draft/README.md) | Write-up drafts |
| [scripts/synthetic_query_generator/](scripts/synthetic_query_generator/README.md) | Synthetic query generation |

---

## Repository Layout

```
AIC-HCMC-2026/
├── challenge_resources/         # All dataset/model/log assets, gitignored except manifests
│   ├── data/                    # zip_file/ lot archives (the dataset), raw_zip/ video archives,
│   │                            #   vectors.f32.npy + .meta.npz (search), video_fps.json,
│   │                            #   milvus_lite.db, postgres_data/, zip_video_index.json,
│   │                            #   strategy-configs/, transcripts/
│   │                            #   _legacy_aic2026_sample/ — unread by any code path
│   ├── onnx-models/             # PE-Core ONNX text encoder + tokenizer
│   ├── generated_queries/, queries_1/, queries_2/, runtime-logs/
│
├── remote-server/               # Runs on the GPU workstation (ENV_MODE=SERVER)
│   ├── docker-compose.yml       # Milvus + PostgreSQL (optional — MILVUS_LITE_PATH skips Docker)
│   ├── requirements.txt
│   ├── main.py                  # FastAPI entry point
│   ├── scripts/
│   │   ├── ingest_zip_pipeline_results.py  # lot archives → Milvus (+ PostgreSQL)
│   │   ├── export_video_fps.py             # lot archives → video_fps.json (frame ↔ timestamp)
│   │   ├── ingest_embeddings_to_milvus.py  # legacy AIC2026_sample ingest
│   │   └── start-local-postgres.sh         # portable Postgres (scoop, no Docker) start/stop/status
│   └── app/
│       ├── db/
│       │   ├── milvus_client.py      # Milvus HNSW search (server or embedded Milvus Lite)
│       │   ├── cagra_client.py       # Optional cuVS CAGRA GPU search
│       │   └── postgres_client.py    # DDL, OCR full-text, transcript interval queries
│       ├── services/
│       │   ├── text_encoder.py       # PE-Core text encoder + cache
│       │   ├── local_zip_media.py    # Frames/playback from raw_zip/Videos_L*.zip, no unpacking
│       │   └── translation.py        # VI/mixed → English via free Google Translate (deep-translator)
│       ├── data_provider.py          # Reads directly from local DBs
│       └── strategies/               # Mirrors local-backend/app/strategies/ — same contract
│           ├── base_strategy.py      # SearchContext + BaseStrategy V2 (+ duplicate-result filtering)
│           ├── _similarity_filter.py # Greedy sequence-aware near-duplicate result filter
│           ├── _fusion.py, _duy_temporal_core.py   # Shared helpers (RRF, temporal DP)
│           ├── raw_visual.py         # Single-channel baseline
│           ├── temporal_visual.py    # Multi-step temporal baseline
│           ├── multi_source.py       # Four-channel RRF example
│           ├── multi_source_temporal.py
│           └── duy_temporal_search*.py  # Temporal variants per offset window
│
└── local-client/
    ├── frontend/                # Next.js UI (Pages Router, Tailwind, TS strict=false)
    │   └── src/
    │       ├── pages/index.tsx       # Main search page (+ duplicate-threshold slider)
    │       ├── pages/tuning.tsx      # Strategy weight tuning (phone-friendly, draft under .runtime/)
    │       ├── lib/
    │       │   ├── useFrameImage.ts     # Server vs browser thumbnail decode, transparently
    │       │   └── zipFrameDecoder.ts   # WebCodecs decode of a GOP byte range
    │       └── components/
    │           ├── QueryGroup.tsx    # Semantic + Text search box pair
    │           ├── CommandPanel.tsx  # Chat-mode command bar
    │           ├── ResultCard.tsx    # Frame thumbnail card
    │           ├── ResultGrid.tsx    # Confidence-sorted grid
    │           ├── VideoGroupGrid.tsx, TranscriptChunkCard.tsx, HelpModal.tsx
    │           └── VideoModal.tsx    # YouTube-first player, zip-video proxy fallback, live frame counter
    │
    └── local-backend/           # Dev playground (ENV_MODE=ZIP or LOCAL)
        ├── main.py                   # + /api/zip-video, /api/zip-frame, /api/zip-frame-plan, /api/zip-bytes
        ├── scripts/
        │   ├── export_vectors_npy.py     # *_results.zip → vectors.f32.npy (run after every ingest)
        │   └── build_zip_video_index.py  # organizer video ZIPs → zip_video_index.json
        └── app/
            ├── data_provider.py      # ZIP: exported vectors | LOCAL: proxy to the server
            ├── db/
            │   ├── numpy_vector_store.py # Exact search over a memmapped array — the SAMPLE default
            │   └── milvus_client.py      # Milvus Lite fallback (FLAT only — see the recall-bug doc)
            ├── services/
            │   ├── remote_zip_proxy.py   # Range-translating proxy for organizer video ZIPs
            │   ├── zip_frame_source.py   # Precise single-frame extraction via MP4 sample-table index
            │   ├── mp4_box_parser.py     # moov/sample-table parsing
            │   └── range_http_client.py  # Retry/backoff wrapper for upstream Range requests
            └── strategies/           # Same V2 strategy contract as server
```

---

## Core System Concept

Every retrieval strategy is a Python file that subclasses `BaseStrategy` and
implements `async run(context)`. It chooses data channels through
`context.retrieve()` and owns fusion, temporal matching, and reranking.
Local mode searches vectors exported from the lot archives, or proxies to the
server. Production mode uses PE-Core text embeddings with Milvus HNSW or
optional cuVS CAGRA, plus PostgreSQL for metadata and text retrieval.

---

## The Search UI at a Glance

```
┌── Sticky header ────────────────────────────────────────────────────┐
│  AIC 2026   Strategy: [Example (Mock) ▼]                            │
│  [▼ Hide Search]  [+ Add Temporal Step]   Top K [100]  [Search]    │
│  ┌ Query Group 1 ───────────────────────────────────────────────┐   │
│  │ [Describe what you want to find…─────────] [VI→EN]           │   │
│  └───────────────────────────────────────────────────────────────┘  │
│  ┌ Temporal Step 2 (5s after step 1) ──────────────────────────┐   │
│  │ [Describe what you want to find…─────────] [VI→EN] [Remove] │   │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
  Result grid (sorted by confidence ↓)
  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
  │ frame │ │ frame │ │ frame │ │ frame │ │ frame │ │ frame │
  │ vid   │ │ vid   │ │ vid   │ │ vid   │ │ vid   │ │ vid   │
  │ 00:42 │ │ 01:23 │ │ 02:05 │ │ 03:47 │ │ 04:11 │ │ 05:02 │
  │ 97.3% │ │ 94.1% │ │ 88.6% │ │ 85.2% │ │ 81.0% │ │ 77.4% │
  └───────┘ └───────┘ └───────┘ └───────┘ └───────┘ └───────┘
  Click any card → YouTube modal opens at that exact timestamp
                   Frame counter updates live as video plays
```

---

## Strategy Development Quickstart

Copy the template in [docs/strategy_template_v2.md](docs/strategy_template_v2.md)
into `local-client/local-backend/app/strategies/`; backend discovery adds it to
the dropdown automatically.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Visual embeddings | `timm/PE-Core-bigG-14-448` on CPU, CUDA, or Apple MPS |
| Visual search | Milvus HNSW by default; optional cuVS CAGRA on NVIDIA GPU |
| Text DB | PostgreSQL 15 — full-text OCR + interval transcripts |
| Backend | Python 3.10+ / FastAPI / uvicorn |
| Frontend | Next.js 14 (Pages Router) / Tailwind CSS / TypeScript |
| Video player | YouTube IFrame API, falling back to a Range-proxied organizer-ZIP `<video>` stream on embed failure |
| Translation | Google Translate (free web endpoint via `deep-translator`); button replaces the editable query |
| Local transport | HTTP via `httpx` (LOCAL mode) |
| Local vector search | Exact search over a memmapped `vectors.f32.npy` (BLAS, ~35 ms at `top_k=1000`); Milvus Lite FLAT as fallback — [why not HNSW](docs/milvus-lite-hnsw-recall-bug.md) |
| Media | Frames and playback read straight out of the organizers' video ZIPs — HTTP Range on local-backend, a file seek on remote-server, no unpacking either way ([zip_media.md](docs/zip_media.md)) |

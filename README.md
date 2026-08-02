# AIC HCMC 2026 — Video Retrieval System

A video retrieval playground for the **Ho Chi Minh City AI Challenge 2026**
(Video Browser Showdown format), with lightweight local development modes and a
GPU-backed production mode.

Team members can develop against mock or sample data, proxy to the GPU server,
or run the complete PE-Core retrieval pipeline directly on the server.

---

## Quick Links

| Document | Description |
|---|---|
| [docs/running.md](docs/running.md) | **Start here** — scenario picker + exact commands |
| [docs/setup.md](docs/setup.md) | Step-by-step setup for local dev and GPU server |
| [docs/launch_scripts.md](docs/launch_scripts.md) | One-command local and remote launch (Windows/Mac/Linux/WSL) |
| [docs/architecture.md](docs/architecture.md) | System design, data flow, ENV_MODE switching |
| [docs/strategy_v2.md](docs/strategy_v2.md) | Strategy/data contract |
| [docs/strategy_template_v2.md](docs/strategy_template_v2.md) | **How to write your own strategy** |
| [docs/db_schema.md](docs/db_schema.md) | PostgreSQL DDL + Milvus collection schema |
| [remote-server/README_INDEXING_SEARCH.md](remote-server/README_INDEXING_SEARCH.md) | PE-Core ingestion, HNSW/CAGRA, translation, validation, and performance |
| [docs/PE-Core-bigG-14-448-Text-Encoder.README.md](docs/PE-Core-bigG-14-448-Text-Encoder.README.md) | Lightweight ONNX text encoder (no torch) |
| [docs/youtube-storyboard-thumbnails-workaround.md](docs/youtube-storyboard-thumbnails-workaround.md) | Dev-only workaround for missing keyframe images |

For a fast handoff, read `README.md` → `docs/architecture.md` →
`docs/setup.md` → `remote-server/README_INDEXING_SEARCH.md`.

---

## Repository Layout

```
AIC-HCMC-2026/
├── remote-server/               # Runs on the GPU workstation (ENV_MODE=SERVER)
│   ├── docker-compose.yml       # Milvus + PostgreSQL
│   ├── requirements.txt
│   ├── main.py                  # FastAPI entry point
│   └── app/
│       ├── db/
│       │   ├── milvus_client.py      # Milvus HNSW search
│       │   ├── cagra_client.py       # Optional cuVS CAGRA GPU search
│       │   └── postgres_client.py    # DDL, OCR full-text, transcript interval queries
│       ├── services/
│       │   ├── text_encoder.py       # PE-Core text encoder + cache
│       │   └── translation.py        # Optional VI/mixed → English translation
│       ├── data_provider.py          # Reads directly from local DBs
│       └── strategies/
│           ├── base_strategy.py      # SearchContext + BaseStrategy V2
│           ├── raw_visual.py         # Single-channel baseline
│           ├── temporal_visual.py    # Multi-step temporal baseline
│           └── multi_source.py       # Four-channel RRF example
│
└── local-client/
    ├── frontend/                # Next.js UI (Pages Router, Tailwind, TS strict=false)
    │   └── src/
    │       ├── pages/index.tsx       # Main search page
    │       └── components/
    │           ├── QueryGroup.tsx    # Semantic + Text search box pair
    │           ├── ResultCard.tsx    # Frame thumbnail card
    │           ├── ResultGrid.tsx    # Confidence-sorted grid
    │           └── VideoModal.tsx    # YouTube player modal with live frame counter
    │
    └── local-backend/           # Dev playground (ENV_MODE=MOCK, SAMPLE, or LOCAL)
        ├── main.py
        └── app/
            ├── data_provider.py      # MOCK JSON | SAMPLE vectors | LOCAL proxy
            ├── mock/                 # 105 sample frames across 3 videos
            └── strategies/           # Same V2 strategy contract as server
```

---

## Core System Concept

Every retrieval strategy is a Python file that subclasses `BaseStrategy` and
implements `async run(context)`. It chooses data channels through
`context.retrieve()` and owns fusion, temporal matching, and reranking.
Local modes use mock/sample data or proxy to the server. Production mode uses
PE-Core text embeddings with Milvus HNSW or optional cuVS CAGRA, plus
PostgreSQL for metadata and text retrieval.

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
| Video player | YouTube IFrame API |
| Translation | Local CTranslate2 INT8 VI→EN; button replaces the editable query |
| Local transport | HTTP via `httpx` (LOCAL mode) |

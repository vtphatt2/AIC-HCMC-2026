# AIC HCMC 2026 — Video Retrieval System

A dual-mode video retrieval playground for the **Ho Chi Minh City AI Challenge 2026** (Video Browser Showdown format).

Team members develop and test retrieval strategies on their laptops against mock data, then promote the same files—unchanged—to the GPU server for competition.

## Quick Links

| Document | Description |
|---|---|
| [docs/setup.md](docs/setup.md) | Step-by-step setup for local dev and GPU server |
| [docs/architecture.md](docs/architecture.md) | System design, data flow, ENV_MODE switching |
| [docs/strategy_guide.md](docs/strategy_guide.md) | **How to write your own strategy** |
| [docs/db_schema.md](docs/db_schema.md) | PostgreSQL DDL + Milvus collection schema |
---

## (How to Run)

### 1. Prerequisites

- **Python 3.10+**
- **Node.js 18+** + npm
- **Git Bash** (tren Windows, dung Git Bash de co shell Unix)

### 2. Backend (FastAPI)

```bash
cd local-client/local-backend

# Tao virtual environment (lan dau tien)
python -m venv .venv
source .venv/Scripts/activate   # Git Bash / Windows
# .venv\Scripts\activate        # CMD / PowerShell

# Cai dat dependencies
pip install -r requirements.txt

# Chay backend o MOCK mode (khong can database)
set ENV_MODE=MOCK              # Windows CMD
# $env:ENV_MODE="MOCK"         # PowerShell
# export ENV_MODE=MOCK          # Git Bash / Linux

uvicorn main:app --reload --port 8000
```

Backend: `http://localhost:8000`.

**CHECK:**
```bash
curl http://localhost:8000/api/health
# -> {"status":"ok","env_mode":"MOCK","strategies":2}

curl http://localhost:8000/api/strategies
# -> Danh sach cac strategy co san trong dropdown
```

**Endpoints:**

| Endpoint | Method | Mo ta |
|---|---|---|
| `/api/health` | GET | Kiem tra backend dang song |
| `/api/strategies` | GET | Danh sach strategy dang ky |
| `/api/search` | POST | Tim kiem video frames |

### 3. Frontend (Next.js)

```bash
cd local-client/frontend

npm install

npm run dev
```

Frontend: `http://localhost:3000`.

File `.env.local` da duoc cau hinh san tro toi `http://localhost:8000`. Neu doi port backend, sua lai file nay.

### 4. Test Strategy

```bash
cd local-client/local-backend

# Test tat ca strategy
python scripts/test_strategy_mock.py

# Test mot strategy cu the
python scripts/test_strategy_mock.py --strategy nam_defensive_fusion_v1
```

Test script se in ra tung test case, expected behavior, top results, va assertion pass/fail. Cuoi cung la dong `Summary: N passed, M failed out of X`.

### 5. Create New Strategy 

```bash
cd local-client/local-backend/app/strategies

# Copy template
cp example_strategy.py yourname_idea_v1.py

# Sua file: name, description, author, fusion_and_temporal()
# Restart backend -> strategy tu dong xuat hien trong dropdown
```

### 6. RUN

```bash
# Terminal 1: Backend
cd local-client/local-backend
set ENV_MODE=MOCK
uvicorn main:app --reload --port 8000

# Terminal 2: Frontend
cd local-client/frontend
npm run dev

# Terminal 3: Test (tuy chon)
cd local-client/local-backend
python scripts/test_strategy_mock.py --strategy nam_defensive_fusion_v1
```

Mo browser vao `http://localhost:3000` de su dung UI.

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
│       │   ├── milvus_client.py      # HNSW vector index, 1280-dim COSINE
│       │   └── postgres_client.py    # DDL, OCR full-text, transcript interval queries
│       ├── data_provider.py          # Reads directly from local DBs
│       └── strategies/
│           ├── base_strategy.py      # Abstract base — guardrails live here
│           └── stable_fusion.py      # Current production strategy
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
    └── local-backend/           # Dev playground (ENV_MODE=MOCK or LOCAL)
        ├── main.py
        ├── requirements.txt
        ├── scripts/
        │   └── test_strategy_mock.py  # 11 test cases, zero dependencies
        └── app/
            ├── data_provider.py      # MOCK: reads JSON  |  LOCAL: proxies to GPU server
            ├── mock/                 # 110 frames + 15 OCR + 12 transcripts (3 videos)
            │   ├── mock_frames.json
            │   ├── mock_ocr.json
            │   ├── mock_transcripts.json
            │   └── mock_videos.json
            └── strategies/
                ├── base_strategy.py              # Identical to server — copy strategies freely
                ├── example_strategy.py           # Template / UI smoke-test strategy
                └── nam_defensive_fusion_v1.py    # Adaptive 3-modal fusion + dedup
```

---

## The Core Idea in One Paragraph

Every retrieval strategy is a single Python file that subclasses `BaseStrategy` and implements one method: `fusion_and_temporal()`. That method receives raw multi-modal data (visual vectors, OCR text, transcript intervals) already fetched from the database, and returns a ranked list of frames. A `DataProvider` abstraction handles *where* that data comes from—local mock JSON files during development, or live Milvus/PostgreSQL on the GPU server during competition—without touching the strategy file at all.

---

## The Search UI at a Glance

```
┌── Sticky header ────────────────────────────────────────────────────┐
│  AIC 2026   Strategy: [Example (Mock) ▼]                            │
│  [▼ Hide Search]  [+ Add Temporal Step]   Top K [100]  [Search]    │
│  ┌ Query Group 1 ───────────────────────────────────────────────┐   │
│  │ [Semantic search…────────────────────────] [VI→EN]           │   │
│  │ [Text / OCR search…──────────────────────]                   │   │
│  └───────────────────────────────────────────────────────────────┘  │
│  ┌ Temporal Step 2 (5s after step 1) ──────────────────────────┐   │
│  │ [Semantic search…────────────────────────] [VI→EN]           │   │
│  │ [Text / OCR search…──────────────────────]          [Remove] │   │
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

## Adding Your Own Strategy (TL;DR)

1. Copy `local-client/local-backend/app/strategies/example_strategy.py`
2. Rename to `yourname_idea_v1.py`
3. Fill in `name`, `description`, `author`
4. Implement `fusion_and_temporal()`
5. Restart the backend — your strategy appears in the dropdown

See [.skills/strategy-development/](.skills/strategy-development/) for full details.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Visual embeddings | `timm/PE-Core-bigG-14-448` — 1280-dim vectors |
| Vector DB | Milvus 2.3 — HNSW index, COSINE metric |
| Text DB | PostgreSQL 15 — full-text OCR + interval transcripts |
| Backend | Python 3.10+ / FastAPI / uvicorn |
| Frontend | Next.js 14 (Pages Router) / Tailwind CSS / TypeScript |
| Video player | YouTube IFrame API |
| Local transport | HTTP via `httpx` (LOCAL mode) |

# System Architecture

## Overview

The system has two deployment targets that share the same strategy code:

```
┌────────────────────────────────────────────────────────────────────┐
│  Contestant's Laptop                                               │
│                                                                    │
│  ┌──────────────────┐      HTTP       ┌────────────────────────┐  │
│  │  Next.js Frontend│ ──────────────► │  Local Backend         │  │
│  │  localhost:3000   │ ◄────────────── │  FastAPI :8000         │  │
│  └──────────────────┘   JSON results  │                        │  │
│                                       │  DataProvider          │  │
│                                       │  ┌──────────────────┐  │  │
│                                       │  │ MOCK mode        │  │  │
│                                       │  │ reads mock/*.json│  │  │
│                                       │  └──────────────────┘  │  │
│                                       │  ┌──────────────────┐  │  │
│                                       │  │ LOCAL mode       │  │  │
│                                       │  │ proxies → Server │  │  │
│                                       │  └──────────────────┘  │  │
│                                       └────────────────────────┘  │
└──────────────────────────────────────────────┬─────────────────────┘
                                               │ HTTP /api/raw-data
                                               │ (LOCAL mode only)
                                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  GPU Workstation (remote-server)                                 │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  FastAPI :8000  (ENV_MODE=SERVER)                          │ │
│  │                                                            │ │
│  │  DataProvider ──► Milvus (visual vectors, 1280-dim HNSW)  │ │
│  │               ──► PostgreSQL (OCR, transcripts)           │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## ENV_MODE

The `ENV_MODE` environment variable controls where data comes from. Set it in the `.env` file.

| Value | Where it runs | What DataProvider does |
|---|---|---|
| `MOCK` | Contestant laptop | Reads `app/mock/*.json` — no server needed |
| `LOCAL` | Contestant laptop | Proxies HTTP requests to `REMOTE_SERVER_URL` to fetch raw data |
| `SERVER` | GPU workstation | Connects directly to local Milvus + PostgreSQL |

### Switching from MOCK to LOCAL

Edit `local-client/local-backend/.env`:

```env
ENV_MODE=LOCAL
REMOTE_SERVER_URL=https://your-ngrok-url.ngrok.io
```

The strategy files require **zero changes** when switching modes. Only the `.env` file changes.

---

## Request Lifecycle

### Search request (MOCK or LOCAL mode)

```
Frontend (browser)
  │
  │  POST /api/search  { strategy_id, query_groups, top_k }
  ▼
Local Backend (FastAPI)
  │
  ├─ Looks up strategy by strategy_id
  ├─ Calls strategy.search(query_groups)
  │     │
  │     ├─ pre_process(query_groups)       ← optional override
  │     │
  │     ├─ DataProvider.get_raw_data()     ← MOCK: reads JSON
  │     │                                    LOCAL: HTTP to GPU server
  │     │
  │     ├─ fusion_and_temporal(raw_data)   ← YOUR ALGORITHM (2s timeout)
  │     │
  │     └─ post_filter(results)            ← optional override
  │
  ├─ results[:top_k]
  │
  └─ Returns { results, total, execution_time_ms }
  │
  ▼
Frontend renders result grid
  Click card → YouTube modal seeks to timestamp_ms / 1000 seconds
```

### Raw data shape

Every strategy receives the same `raw_data` dict regardless of mode:

```python
{
    "frames": [
        {
            "frame_id":     "dQw4w9WgXcQ_000025",
            "video_id":     "dQw4w9WgXcQ",
            "frame_number": 25,
            "timestamp_ms": 1000,
            "image_url":    "https://..."   # or /static/frames/... on server
        },
        ...
    ],
    "ocr": [
        {
            "frame_id":     "dQw4w9WgXcQ_000025",
            "video_id":     "dQw4w9WgXcQ",
            "frame_number": 25,
            "timestamp_ms": 1000,
            "ocr_text":     "BREAKING NEWS: City Center"
        },
        ...
    ],
    "transcripts": [
        {
            "video_id":      "dQw4w9WgXcQ",
            "start_time_ms": 0,
            "end_time_ms":   4000,
            "text":          "Floodwaters have reached the city centre..."
        },
        ...
    ],
    "videos": {
        "dQw4w9WgXcQ": {
            "video_id":    "dQw4w9WgXcQ",
            "title":       "Mock Video — City Street Scene",
            "youtube_id":  "dQw4w9WgXcQ",
            "fps":         25.0,
            "duration_ms": 212000,
            "frame_count": 5300
        },
        ...
    }
}
```

---

## Strategy Auto-Discovery

On startup the backend scans `app/strategies/*.py` and auto-registers any class that:
- Inherits from `BaseStrategy`
- Is **not** `BaseStrategy` itself
- Has non-empty `name`, `description`, and `author` class attributes

The file stem becomes the `strategy_id` (e.g. `duy_temporal_v1.py` → `"duy_temporal_v1"`).

A reload of the backend (or `--reload` watching the `.py` file) is all that's needed to pick up a new strategy.

---

## Guardrails (always enforced)

| Guardrail | Value | Where enforced |
|---|---|---|
| Fetch cap | 1000 records max per query | `BaseStrategy.search()` → `DataProvider.get_raw_data(limit=FETCH_CAP)` |
| Execution timeout | 2.0 seconds | `asyncio.wait_for(asyncio.to_thread(fusion_and_temporal), timeout=2.0)` |
| Top K cap | User-controlled (default 100, max 1000) | `main.py` slices `results[:top_k]` before returning |

The timeout cancels the HTTP response but does not forcibly kill the worker thread. If a strategy has a true infinite loop the thread will continue in the background until the process restarts. This is acceptable for a development playground.

---

## Promoting a Strategy to Production

1. Test your strategy locally until satisfied with the score
2. Copy the file verbatim to `remote-server/app/strategies/yourname_v1.py`
3. Restart the remote server
4. Select it in the frontend dropdown

No other changes needed. The `DataProvider` on the server handles all DB connections transparently.

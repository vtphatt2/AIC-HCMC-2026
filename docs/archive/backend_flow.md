# Backend Code Flow

What actually happens inside `local-client/local-backend` and `remote-server`,
file by file, from process start to JSON response. [architecture.md](architecture.md)
covers *why* the design is shaped this way; this page covers *where the code is*.

Both backends are the same three layers with different bottom halves:

```
main.py                      HTTP routes, strategy discovery, guardrails, media routes
  └─ app/strategies/         ranking logic (yours) — identical contract on both sides
       └─ SearchContext      the only door strategies use to reach data
            └─ app/data_provider.py    picks the data source for this ENV_MODE
                 ├─ app/db/*           numpy memmap | Milvus (Lite/server) | PostgreSQL
                 └─ HTTP               LOCAL mode: proxies to remote-server /api/retrieve
```

---

## 1. Startup

`lifespan()` in `main.py` runs once, in this order.

### local-backend (`local-client/local-backend/main.py`)

| Step | Code | Notes |
|---|---|---|
| Load `.env` | `load_dotenv(override=True)` | `.env` beats any shell/launcher env var |
| Build `DataProvider()` | `app/data_provider.py` | Branches on `ENV_MODE`; prints which search backend it picked |
| Optional encoder warmup | `WARMUP_TEXT_ENCODER=true` | ~3.3 s of model load moved off the first query |
| Optional query parser | `QueryParser()` if `GEMINI_API_KEY` is set | Only strategies calling `context.parse_json()` use it |
| Discover strategies | `discover_strategies()` | Imports every `app/strategies/*.py`, skips `_`-prefixed files and `base_strategy` |

`DataProvider.__init__` in `ZIP` mode is where the search backend is chosen, in
this priority (first available wins — the startup log says which one it took):

1. **numpy memmap** — `app/db/numpy_vector_store.py`, used when `MILVUS_LITE_PATH`
   is set *and* `vectors.f32.npy` + `vectors.meta.npz` exist next to it. Exact,
   ~35 ms at `top_k=1000`. Prints a warning when the `.npy` is older than the
   newest `zip_file/*_results.zip` — i.e. you ingested and forgot to re-export.
2. **Milvus Lite** — `app/db/milvus_client.py`, the fallback for a machine that
   has not run `scripts/export_vectors_npy.py`. Must stay on the FLAT index; its
   HNSW path returns wrong neighbours ([milvus-lite-hnsw-recall-bug.md](milvus-lite-hnsw-recall-bug.md)).

Nothing else. If neither exists the backend refuses to start and prints the two
commands that produce them, rather than coming up with an empty index.

### remote-server (`remote-server/main.py`)

Same shape plus the infrastructure it owns:

| Step | Code |
|---|---|
| `postgres_client.init_schema()` | Creates tables if missing ([db_schema.md](db_schema.md)) |
| `milvus_client.connect()` + `create_collection_if_missing()` | Real Milvus server, or Milvus Lite when `MILVUS_LITE_PATH` is set (dev/test only) |
| `DataProvider()` | Reads Milvus/PostgreSQL directly — no HTTP hop |
| Optional translation warmup | `WARMUP_TRANSLATION=true` |
| Transcript chunk search | `TRANSCRIPT_CHUNK_SEARCH_ENABLED` (default true) → `TranscriptSearchService` ([search_by_transcript.md](search_by_transcript.md)) |
| Discover strategies | identical to local |

Strategy discovery is why "add a strategy" is a one-file operation: drop a
`BaseStrategy` subclass into `app/strategies/`, restart (or run with `--reload`),
and it shows up in the frontend dropdown under its filename stem.

---

## 2. A search request, end to end

`POST /api/search` → `main.py::search()`:

1. **Resolve strategy** — 404 when `strategy_id` is not in the discovered dict.
2. **Resolve config** — `StrategyConfigStore` (`app/services/strategy_config.py`)
   loads the named preset from `challenge_resources/data/strategy-configs/`,
   then `.resolve()` layers request-scoped `config_overrides` on top and
   validates the result against the strategy's `config_schema`. On remote-server
   writing presets needs `ALLOW_STRATEGY_CONFIG_WRITES=true`; the tuning UI
   instead sends overrides per request so nothing shared gets mutated.
3. **Clamp** — `top_k = min(max(top_k, 1), FETCH_CAP)`, `FETCH_CAP = 1000`.
4. **Run** — `BaseStrategy.search()` (`app/strategies/base_strategy.py`):
   - builds a `SearchContext` (query groups, options, top_k, genre, algorithm),
   - wraps `_run_with_filter(context)` in `asyncio.wait_for(..., 30 s)`,
   - `_run_with_filter` calls your `run(context)`, applies near-duplicate
     filtering, and re-runs via `request_more_results()` until the page is full
     or the channel is exhausted (see
     [architecture.md → Duplicate-Result Filtering](architecture.md#duplicate-result-filtering)).
5. **Respond** — results sliced to `top_k`, plus `config_id`, `config_revision`,
   `effective_config`, `duplicate_threshold`, `execution_time_ms`.

Errors become status codes at this boundary only: `TimeoutError` → 408,
`ValueError` → 400, anything else → 500.

### Inside `SearchContext`

`context.retrieve(channel, query, top_k=…)` is the single data door. Four channels:

| Channel | local-backend (ZIP) | remote-server |
|---|---|---|
| `raw.semantic` | numpy memmap, else Milvus Lite FLAT | Milvus (HNSW/FLAT/ScaNN) or CAGRA |
| `subtitled.semantic` | **not in the lot archives — raises** | the Milvus collection for that channel |
| `transcript.lexical` | local transcript scan (`transcript_index.py`) | PostgreSQL full-text |
| `transcript.semantic` | **not in the lot archives — raises** | `transcript_chunks` Milvus collection |

In `ENV_MODE=LOCAL` all four channels are the remote-server's, proxied. In `ZIP`
the two remote-only channels raise a `RuntimeError` naming the reason — an empty
list would read as "no matches" for data that was never there.

`POST /api/search/transcript` (the Transcripts search tab, a separate
endpoint from these four retrieval channels) is not affected by that
`RuntimeError` gap — as of 2026-08-21 it fuzzy-matches locally in `ZIP`
mode too, via `transcript_index.py`'s `search_all_transcripts` (see
[gaps.md §4](gaps.md#4-transcriptsemantic-is-remote-only)).

`retrieve()` caches per channel inside one request and pages forward using
`exclude_frame_ids`, so a second call on the same channel returns *new* hits
rather than the same page again. `context.frame_embeddings(ids)` batch-fetches
raw vectors — free on the numpy path (the row is already mapped), a gRPC round
trip of 1280 floats per hit on the Milvus path.

`context.results(hits)` is the only supported way to shape the response: it
fills `frame_image_url` and strips `_`-prefixed private keys.

### `ENV_MODE=LOCAL`

`DataProvider.retrieve()` does no searching at all — it POSTs the same arguments
to `REMOTE_SERVER_URL/api/retrieve` and returns the hits. Every other layer is
unchanged, which is why a strategy file needs zero edits to move between modes.

---

## 3. Media routes (thumbnails and playback)

Search results carry `video_id` + `timestamp_ms`; the *image* is resolved
separately. Lot archives ship no JPGs at all, so `data_provider.retrieve()`
rewrites every `image_url` to `/api/zip-frame/{video_id}/{timestamp_ms}` at read
time and the ZIP media routes take over — that whole path is
[zip_media.md](zip_media.md). Both backends now serve those routes.

`ENV_MODE=LOCAL` is the one exception: `/static/frames/...` is passed through to
the remote server, for lots that do have JPGs on its disk.

---

## 4. Endpoint map

| Endpoint | Where | Code |
|---|---|---|
| `GET /api/health` | both | env mode, strategy count, selected backends |
| `GET /api/strategies` | both | dropdown contents |
| `GET/PUT/DELETE /api/strategies/{id}/configs[/{cid}]` | both | `strategy_config.py`; writes gated on remote |
| `GET /api/vector-search-algorithms` | both | local returns `linear`, or proxies in LOCAL mode |
| `POST /api/translate` | both | `translation.py`, Google Translate free web endpoint, 256-entry LRU |
| `POST /api/search` | both | the flow above |
| `POST /api/search/transcript` | both | local: `data_provider.search_transcript_chunks`; remote: `TranscriptSearchService` |
| `GET /api/transcript/{video_id}` | both | local: `transcript_index.py`; remote: `transcript_jsonl_reader.py` |
| `POST /api/warmup_text_encoder` | both | run once before a demo |
| `POST /api/retrieve`, `/api/frame-embeddings`, `/api/keyframes` | **remote only** | the raw-data API that `ENV_MODE=LOCAL` proxies to |
| `/api/zip-video`, `/api/zip-frame` | both | [zip_media.md](zip_media.md) |
| `/api/zip-frame-plan`, `/api/zip-bytes` | local only | browser-side decode ([doc](zip_media.md#4-client-side-decode-optional-recommended-when-sharing)) |

local-backend reads the archives over HTTP Range from the organizers' host;
remote-server seeks in its own copy under `raw_zip/`. Same math, different fetch
layer.

---

## 5. Guardrails

| Guardrail | Value | Enforced in |
|---|---|---|
| Fetch cap | 1000 hits per retrieval and per response | `SearchContext`, `BaseStrategy.search()` |
| Execution timeout | 30 s | `asyncio.wait_for` in `BaseStrategy.search()` |
| Top-K | request-controlled, clamped to `FETCH_CAP` | `main.py` |
| ZIP frame timeout | 45 s (`ZIP_FRAME_TIMEOUT_SEC`) | `main.py::zip_frame` |
| ZIP concurrency | 6 ffmpeg (`ZIP_FRAME_DECODE_CONCURRENCY`), 12 in flight (`ZIP_FRAME_CONCURRENCY`) | `zip_frame_source.py` / `main.py` |

The execution timeout ends the HTTP response but does not kill the worker
thread — a genuinely infinite strategy loop keeps running until the process
restarts. Acceptable for a dev playground; worth knowing during a demo.

---

## 6. Reading the logs

Both backends emit `[TIMER]` lines with a fixed vocabulary. Grep these first
when something is slow:

```
[TIMER] request_received strategy=… top_k=… query_groups=…
[TIMER] text_encode      … ms text_len=…
[TIMER] vector_search    … ms backend=numpy|milvus channel=… top_k=… hits=…
[TIMER] total_request    … ms strategy=… status=ok|timeout|error results=…
```

Measured baselines, and the wrong turns taken while getting them, are in
[performance_pain_points.md](performance_pain_points.md).

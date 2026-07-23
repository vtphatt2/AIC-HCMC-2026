# Backend Indexing & Visual Search (remote-server)

How to ingest `AIC2026_sample` into Milvus + PostgreSQL and validate the
result. For bringing up the whole GPU server (env vars, CAGRA/MPS,
translation, Docker), see [../docs/setup.md → Option
C](../docs/setup.md#option-c--gpu-server-production). For the Milvus/Postgres
schema and the `/api/search` request/response contract, see
[../docs/db_schema.md](../docs/db_schema.md) and
[../docs/architecture.md](../docs/architecture.md).

## 1. Dataset layout

The scripts auto-detect the repo's `data/` directory (and still accept the old
`AIC2026_sample/` layout):

```text
AIC-HCMC-2026/
  data/
    keyframes/L01_V001/...                         # frame images
    metadata/L01_V001.json                         # fps, youtube_id
    PECore-features/raw_keyframe_embeddings/L01_V001/...
    PECore-features/subtitled_keyframe_embeddings/L01_V001/...
```

Flat layout (`keyframes/`, `metadata/`, `PECore-features/` containing video
folders/files directly) also works. For any other location, pass
`--sample-root /path/to/AIC2026_sample` to every script below.

Minimum required for Search by Text: `PECore-features` (or `--features-subdir
embeddings` for a different embedding source), `keyframes`, `metadata`.

## 2. Start databases

```bash
cd remote-server
docker compose up -d
docker compose ps   # wait for all services Up
```

## 3. Install dependencies

```bash
cd remote-server
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt                      # base: PECore ONNX, Milvus, PostgreSQL
pip install -r requirements-torch.txt                 # optional: full OpenCLIP torch backend
pip install --extra-index-url https://pypi.nvidia.com -r requirements-cagra.txt  # optional: CUDA 13 CAGRA
pip install -r requirements-youtube-thumbnail.txt      # optional: YouTube thumbnail workaround
```

## 4. Create the Milvus index and ingest embeddings

```bash
python scripts/create_milvus_index.py                 # --recreate to drop+rebuild
python scripts/ingest_embeddings_to_milvus.py
```

`ingest_embeddings_to_milvus.py` flags:

| Flag | Default | Description |
|---|---|---|
| `--sample-root` | auto-detect | Path to `AIC2026_sample` |
| `--features-subdir` | `PECore-features` | Embedding subfolder (e.g. `embeddings`) |
| `--channel` | `raw.semantic` | `raw.semantic` or `subtitled.semantic` |
| `--batch-size` | 256 | Vectors per Milvus insert batch |
| `--vector-index` | `hnsw` | `hnsw`, `flat`, `scann`, or `all` (build all three for runtime switching) |
| `--recreate-milvus` | false | Drop collection before inserting |
| `--copy-keyframes` | false | Copy jpgs into `remote-server/static/frames` |
| `--skip-postgres` | false | Skip PostgreSQL video metadata upsert |
| `--dry-run` | false | Scan only, no writes |

```bash
# Full re-ingest: recreate + all three indexes + copy keyframes to static/
python scripts/ingest_embeddings_to_milvus.py \
  --channel raw.semantic --recreate-milvus --copy-keyframes --vector-index all
python scripts/ingest_embeddings_to_milvus.py \
  --channel subtitled.semantic --recreate-milvus --vector-index all
```

Safe to rerun after correcting metadata — it upserts rather than duplicating.
Frame images are served from `AIC2026_sample/keyframes` by default; set
`FRAME_STATIC_DIR` to override, or use `--copy-keyframes` to stage a copy
under `remote-server/static/frames`.

## 5. Run the backend

```bash
bash ../scripts/start-remote.sh                 # starts Docker + backend
curl http://localhost:8000/api/health
curl -X POST http://localhost:8000/api/warmup_text_encoder   # load the text model once before a demo
```

For `VECTOR_SEARCH_BACKEND=cagra` / Apple MPS / translation env vars, see
[../docs/setup.md → Option C](../docs/setup.md#option-c--gpu-server-production).
Building the CAGRA index itself is part of this ingestion pipeline:

```bash
pip install -r requirements-cagra.txt
python scripts/build_cagra_index.py
```

CAGRA artifacts live under `remote-server/cache/` (not committed) — rebuild
after changing the cuVS version, since its serialized index format is
experimental.

## 6. Validation & evaluation scripts

**`check_index_integrity.py`** — verifies vector count, dim (1280), required
metadata fields, duplicate `frame_id`, and missing image paths. Reports
`PASS` / `WARN` (index usable, some images missing) / `FAIL` (dim/count/dupe
error).

```bash
python scripts/check_index_integrity.py
```

**`compare_linear_vs_milvus.py`** — validates HNSW recall against brute-force
search over `AIC2026_sample/PECore-features`. Reports `overlap@5`/`overlap@10`
(how many frame IDs match) and `score_difference_on_shared` (large values
suggest a normalization/indexing bug). Add `--include-cagra` once a CAGRA
index exists to compare all three.

```bash
python scripts/compare_linear_vs_milvus.py --query "a busy street with people" --top-k 10
```

**`smoke_search_queries.py`** — end-to-end sanity check against a running
backend. Empty output usually means the backend isn't up, the strategy wasn't
discovered, or the PE-Core model/Milvus index is unavailable.

```bash
python scripts/smoke_search_queries.py --backend-url http://localhost:8000 --query "cars on a road" --top-k 5
```

**`evaluate_query_set.py`** — runs a JSON query set (`{"queries": [{"id":
"q1", "semantic_query": "..."}]}`) and writes `evaluation_report.json` /
`.csv` with top-k results, latency, and score distribution. There's no ground
truth yet, so this doesn't compute recall/MAP — use it for spotting retrieval
regressions via score distribution and visual inspection of top-k frames.

```bash
python scripts/evaluate_query_set.py queries.json --top-k 10
```

## Runtime notes

- The PE-Core text embedding cache is a 128-entry in-process LRU keyed on the
  exact normalized query text — it speeds up repeated identical queries only,
  resets on restart, and never changes retrieval scores.
- Backend logs `[TIMER] request_received / model_load / text_encode /
  milvus_search / fusion / total_request / warmup_text_encoder` — use these to
  find latency bottlenecks.
- Local-client proxy mode (`ENV_MODE=LOCAL` pointed at this server) and
  `ENV_MODE=SAMPLE` (searches `AIC2026_sample` directly, no Milvus/Postgres)
  are documented in [../docs/setup.md](../docs/setup.md).

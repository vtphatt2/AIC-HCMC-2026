# Backend Indexing & Visual Search (remote-server)

How to ingest the organizers' lot archives into Milvus + PostgreSQL and
validate the result. For bringing up the whole GPU server (env vars, CAGRA/MPS,
translation, Docker), see [../docs/SETUP.md § Remote
server](../docs/SETUP.md#2-remote-server--gpu-workstation). For the
Milvus/Postgres schema and the `/api/search` request/response contract, see
[../docs/archive/db_schema.md](../docs/archive/db_schema.md) and
[../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md).

## 1. Dataset

`challenge_resources/data/zip_file/*_results.zip` — one archive per organizer
"lot", holding per-video scene metadata and PE-Core embeddings, no JPGs. This is
the only dataset the system indexes; ingestion is [§4](#4-ingest-the-lot-archives).

Frames and video playback do not come from here — they are read from the
*video* archives (`raw_zip/Videos_L*.zip`) at request time, see
[../docs/ARCHITECTURE.md § Media](../docs/ARCHITECTURE.md#media-youtube-first-zip-proxy-fallback)
(full mechanism: [../docs/archive/zip_media.md](../docs/archive/zip_media.md)).

<details>
<summary>Legacy: the <code>AIC2026_sample</code> layout</summary>

`ingest_embeddings_to_milvus.py`'s CLI reads an older layout (`keyframes/`,
`metadata/`, `PECore-features/` — nine L01–L03 videos with keyframe JPGs, one
`.npy` per frame). That dataset is no longer part of the repo; it sits unread in
`challenge_resources/data/_legacy_aic2026_sample/`.

**The file itself stays**, because `ingest_zip_pipeline_results.py` imports
`upsert_vectors()` and `upsert_videos()` from it — those are the shared
Milvus/PostgreSQL writers, not sample-specific code. Only its `main()` and
`iter_video_records()` are legacy, and it cannot read a `*_results.zip`: no zip
handling, and the archives hold one batched `embeddings.npy` per video where it
expects one `.npy` per frame.

</details>

## 2. Start databases

Docker (real Milvus + PostgreSQL):
```bash
cd remote-server
docker compose up -d
docker compose ps   # wait for all services Up
```

**No Docker, dev-only** (embedded Milvus Lite + portable PostgreSQL) — this
server is meant to carry the heavier production workload (real Milvus, full
dataset); Milvus Lite here is only for developing/testing `remote-server`
code on a laptop without Docker or a real Milvus server, same as
`local-backend`'s own Milvus Lite path. Needs `pymilvus>=2.4` (the base
`requirements.txt` pins `2.3.7` for the production path), so install the
extra requirements file too:
```bash
pip install -r requirements-milvus-lite.txt
```
Set `MILVUS_LITE_PATH=../challenge_resources/data/milvus_lite.db` in `.env`, then:
```bash
cd remote-server
bash scripts/start-local-postgres.sh start
```
`local-backend` can point at the same `milvus_lite.db` file for its own
lightweight search — see [../docs/SETUP.md](../docs/SETUP.md). Only one
process can hold the Milvus Lite file open at a time, so stop the backend
before running an ingestion script against it.

## 3. Install dependencies

```bash
cd remote-server
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt                      # base: PECore ONNX, real Milvus server, PostgreSQL
pip install -r requirements-torch.txt                 # optional: full OpenCLIP torch backend
pip install --extra-index-url https://pypi.nvidia.com -r requirements-cagra.txt  # optional: CUDA 13 CAGRA
pip install -r requirements-milvus-lite.txt            # optional: dev-only, see "No Docker" above
```

## 4. Ingest the lot archives

Each `challenge_resources/data/zip_file/*_results.zip` archive is one
organizer "lot" (e.g. `L26_c_results.zip`, produced from `Videos_L26_c.zip`)
and contains, per video, `phase1_transnet/video__<id>/{scenes.json,keyframes.json}`
(fps, selected frame numbers) and `phase2_embeddings/video__<id>/embeddings.npy`
((num_keyframes, 1280) float32, L2-normalized PE-Core-bigG vectors) — no JPGs.

```bash
cd remote-server
python scripts/ingest_zip_pipeline_results.py --dry-run   # scan only, prints counts
python scripts/ingest_zip_pipeline_results.py
```

| Flag | Default | Description |
|---|---|---|
| `--zip-dir` | `challenge_resources/data/zip_file` | Directory containing `*_results.zip` archives |
| `--batch-size` | 256 | Vectors per Milvus insert batch |
| `--vector-index` | `hnsw` | `hnsw`, `flat`, `scann`, or `all` |
| `--recreate-milvus` | false | Drop collection(s) before inserting |
| `--skip-postgres` | false | Only ingest Milvus vectors |
| `--dry-run` | false | Scan only, no writes |

`youtube_id`/`title` are resolved from the organizers' media-info archive
(`media-info/<video_id>.json` entries with a `watch_url`, e.g.
`media-info-aic25-b1.zip`, dropped in the same `--zip-dir`) and denormalized
straight into the Milvus record for that frame — this is what lets both
backends prioritize YouTube playback without a PostgreSQL round trip. A
video missing from the media-info archive just gets no `youtube_id`, and
playback for it falls back to the `/api/zip-video`/`/api/zip-frame` proxy
like any other video would.

`image_url` is intentionally left blank at ingest time — every hit already
carries `video_id` + `frame_number`/`timestamp_ms`, which is enough for a
consumer to derive its own thumbnail URL at read time (`local-backend`
rewrites it to `/api/zip-frame/{video_id}/{timestamp_ms}`). Baking a URL in
at ingest time risks it going stale the moment a consumer's serving
mechanism changes.

Safe to rerun — it upserts by `frame_id`/`video_id`.

Two exports must follow every ingest, or search and the frame routes drift out
of sync with what was just indexed:

```bash
python scripts/export_video_fps.py                       # video_fps.json, ~0.1s
cd ../local-client/local-backend
python scripts/export_vectors_npy.py                     # vectors.f32.npy, ~11s
```

`export_vectors_npy.py` is what local search actually reads; `export_video_fps.py`
is what both backends use to turn a `timestamp_ms` back into the right frame
([../docs/archive/zip_media.md §5](../docs/archive/zip_media.md#5-getting-the-frame-right)).

## 5. Run the backend

```bash
bash ../scripts/start-remote.sh                 # starts Docker + backend
curl http://localhost:8000/api/health
curl -X POST http://localhost:8000/api/warmup_text_encoder   # load the text model once before a demo
```

For `VECTOR_SEARCH_BACKEND=cagra` / Apple MPS / translation env vars, see
[../docs/SETUP.md § Remote server](../docs/SETUP.md#2-remote-server--gpu-workstation).
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

Recall validation used to live in `compare_linear_vs_milvus.py`, which
brute-forced the legacy `PECore-features` layout and compared it with Milvus.
Deleted: the two corpora are now disjoint (nine L01–L03 videos versus the
indexed L21–L30), so its overlap was structurally zero. For the recall question
it existed to answer, `docs/repro/milvus_lite_hnsw_recall.py` is self-contained
and needs no dataset at all.

**`smoke_search_queries.py`** — end-to-end sanity check against a running
backend. Empty output usually means the backend isn't up, the strategy wasn't
discovered, or the PE-Core model/Milvus index is unavailable.

```bash
python scripts/smoke_search_queries.py --backend-url http://localhost:8000 --query "cars on a road" --top-k 5
```

Batch evaluation over a query set is **not** a script here — it lives in the
notebooks (`notebooks/NOTEBOOK_EVALUATION_INPUT_SPEC.md`). An earlier version of
this page documented an `evaluate_query_set.py` that was never written; see
[../docs/archive/gaps.md](../docs/archive/gaps.md#3-evaluate_query_setpy-is-documented-but-does-not-exist).

## Runtime notes

- The PE-Core text embedding cache is a 128-entry in-process LRU keyed on the
  exact normalized query text — it speeds up repeated identical queries only,
  resets on restart, and never changes retrieval scores.
- Backend logs `[TIMER] request_received / model_load / text_encode /
  milvus_search / fusion / total_request / warmup_text_encoder` — use these to
  find latency bottlenecks.
- Local-client modes (`ENV_MODE=ZIP` searching the exported vectors,
  `ENV_MODE=LOCAL` proxying to this server) are documented in
  [../docs/SETUP.md](../docs/SETUP.md).
- `/api/frame-embeddings` (`query_frame_vectors()` in `milvus_client.py`)
  batch-fetches raw vectors by `frame_id`, used by the near-duplicate result
  filter (`duplicate_threshold` in `/api/search` — see
  [../docs/archive/architecture.md](../docs/archive/architecture.md#duplicate-result-filtering)).

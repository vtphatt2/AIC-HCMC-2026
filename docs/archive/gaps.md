# Known Gaps

Things the docs describe, imply, or would reasonably lead you to expect, that
are **not implemented** today, so a decision can be made per item. Nothing here
is claimed to be urgent.

Items 1 and 2 have since been built — kept, struck through, because what remains
of them is a *data* prerequisite that still bites. Verified against the tree at
the time of writing; each entry names the file to check.

---

## 1. ~~remote-server cannot serve ZIP-only lots~~ — BUILT

`remote-server` now serves `/api/zip-video` and `/api/zip-frame` from a local
copy of `Videos_L*.zip` under `challenge_resources/data/raw_zip/`
(`app/services/local_zip_media.py`), and `data_provider._hydrate_frames()`
derives `image_url` for any hit that has none. See [zip_media.md](zip_media.md).

**Still blocked on data, not code:** only `Videos_L30_a.zip` has been downloaded
to `raw_zip/`. Videos from the other thirteen lots return 404 on those routes
until their archives are there — the frontend falls back to YouTube, exactly as
it does for any video it cannot stream.

## 2. ~~Extracting frames from a local ZIP without unpacking it~~ — BUILT

Same module as item 1. Entries are `ZIP_STORED`, so an offset plus a file seek
reaches any byte of any MP4 inside; nothing is unpacked and nothing is cached to
disk. Verified against ffmpeg decoding the unpacked MP4
(`remote-server/tests/test_local_zip_media.py`).

## 3. `evaluate_query_set.py` is documented but does not exist

`remote-server/README_INDEXING_SEARCH.md` §6 describes a script that writes
`evaluation_report.json` / `.csv`. There is no such file in
`remote-server/scripts/`. The two other validation scripts in that section
(`check_index_integrity.py`, `smoke_search_queries.py`) do exist and work.

The reference has been removed from that section as part of this pass. Decide
whether to write the script or leave evaluation to the notebooks
(`notebooks/NOTEBOOK_EVALUATION_INPUT_SPEC.md`).

---

## 4. `transcript.semantic` is remote-only

`SearchContext.retrieve(…)` raises a `RuntimeError` for `transcript.semantic`
*and* `subtitled.semantic` in `ENV_MODE=ZIP`, and `POST /api/search/transcript`
returns an empty list there. Only `ENV_MODE=LOCAL` (proxying) and `SERVER` have
them.

Both are absent for the same reason: the lot archives carry one embedding set
per keyframe and no transcript index. The chunk index and its
sentence-transformer live on the server ([search_by_transcript.md](search_by_transcript.md)).
Worth knowing before writing a strategy that assumes four channels everywhere —
`multi_source.py` is one, and it runs only in `LOCAL` or on the server.

---

## 5. Ingest and vector export are two manual steps that must stay in sync

Not a missing feature so much as a standing footgun, listed because it silently
produces *wrong results* rather than an error.

`ingest_zip_pipeline_results.py` writes Milvus; `export_vectors_npy.py` writes
the `vectors.f32.npy` that local search actually reads. Running the first
without the second leaves local search on the previous ingest's vectors — a
valid file, just older. The backend prints a staleness warning at startup
(`numpy_vector_store.staleness_warning()`), which only helps if someone reads
the startup log.

Options, in ascending effort: leave it (a warning exists), have the ingest
script call the export at the end, or fail startup instead of warning.

---

## 6. Smaller stale/missing items

| Item | Where | Note |
|---|---|---|
| `/api/vector-search-algorithms` on local-backend always reports `linear` | `local-backend/main.py` | Cosmetic: it says `linear` even when the numpy memmap or Milvus Lite path is the one actually serving. The frontend dropdown is correct in effect (there is one option), just mislabelled. |
| Strategy timeout does not kill the thread | `base_strategy.py` | A runaway strategy keeps a core busy until restart. Documented in [backend_flow.md](backend_flow.md#5-guardrails); acceptable for a playground, worth knowing during a demo. |
| Only one lot's video archive is downloaded | `challenge_resources/data/raw_zip/` | `Videos_L30_a.zip` only. Thumbnails and playback work for L30; every other lot 404s on remote-server's media routes and falls back to YouTube. local-backend still reaches all lots over HTTP Range. |
| `basic_learned_reranker` is CPU-only and archived | `local-backend/app/archive_v1/strategies/` | Moved out of the active strategy set; the notes in [reranking_hybrid_notes.md](reranking_hybrid_notes.md) still describe it as a live experiment. |
| `/api/zip-video` has no overall timeout | `local-backend/main.py` | A stalled upstream *stream* is not bounded (only connection-open is retried). The player's `onError` covers it in practice. |

# zip_file/ — organizer archives and how to ingest them

This directory holds the raw inputs for `remote-server/scripts/ingest_zip_pipeline_results.py`.
Everything here is gitignored except this file — the archives are too heavy to push
(tens to ~85 MB each) and are meant to be re-downloaded per machine, not versioned.

## What's in here

| File pattern | Contents |
|---|---|
| `L{lot}_{sublot}_results.zip` (e.g. `L21_a_results.zip`) | Output of `keyframe_pipeline_global_v9_3` for one organizer "lot" (produced from the matching `Videos_L{lot}_{sublot}.zip`). Per video: `phase1_transnet/video__<video_id>/{scenes.json,keyframes.json}` (fps, selected frame numbers) and `phase2_embeddings/video__<video_id>/embeddings.npy` — `(num_keyframes, 1280)` float32, L2-normalized PE-Core-bigG vectors. No JPGs — metadata + embeddings only. |
| `media-info-*.zip` (e.g. `media-info-aic25-b1.zip`) | One `media-info/<video_id>.json` per video, with a `watch_url` — this is where `youtube_id`/title come from at ingest time. |

Every `*_results.zip` present in this directory gets picked up automatically — there's
no manifest to edit, just drop the archive in and ingest.

## Ingesting

Ingestion is **manual** — dropping a new `*_results.zip` here does not index it by
itself. Run this every time you add archives:

```bash
cd remote-server
python scripts/ingest_zip_pipeline_results.py --dry-run --skip-postgres   # preview counts first
python scripts/ingest_zip_pipeline_results.py --skip-postgres
python scripts/export_video_fps.py                                        # required, ~0.1s

cd ../local-client/local-backend
python scripts/export_vectors_npy.py                                      # required, ~11s
```

`export_video_fps.py` writes `video_fps.json` — the fps each video was ingested
with, which both backends need to turn a `timestamp_ms` back into the right
frame. Deriving it from the MP4 instead is wrong for the 91 videos that are not
25 fps, and the error grows with the timestamp
([../../../docs/archive/zip_media.md](../../../docs/archive/zip_media.md#5-getting-the-frame-right)).

**The export step is not optional.** `local-backend` searches a flat
`vectors.f32.npy` rather than Milvus, because `milvus_lite`'s HNSW path returns
wrong neighbours (see [milvus-lite-hnsw-recall-bug.md](../../../docs/archive/milvus-lite-hnsw-recall-bug.md))
and its brute-force path is ~45x slower than a memmapped BLAS scan (~1570 ms vs
~35 ms at `top_k=1000`). Skipping the export leaves search on the *previous*
ingest's vectors — the file is still valid, just older, so nothing would fail.
The backend prints a warning at startup when it spots that, but re-running the
export is the fix. It reads these archives directly and takes ~11 seconds.

- **Stop any running backend first** (`local-backend` and/or `remote-server`) if they're
  pointed at the same `MILVUS_LITE_PATH` — the embedded Milvus Lite file only allows one
  process to hold it open at a time; the ingest script will fail to connect otherwise.
  The same applies to the export step: a running backend keeps `vectors.f32.npy`
  memory-mapped, and Windows refuses to overwrite a mapped file (writing to a temp name
  and swapping does not help — the rename is refused too). Restart the backend once both
  steps finish.
- **`--skip-postgres`**: `local-backend` never reads PostgreSQL (per the local/remote
  weight split — see `docs/ARCHITECTURE.md`), and `youtube_id`/title are
  already denormalized straight into each Milvus frame record. Only drop this flag if
  you're running `remote-server` for real and want the PostgreSQL `videos` table
  populated too.
- **Safe to rerun over everything**: the script globs *every* `*_results.zip` in this
  directory each run and upserts by `frame_id`/`video_id` — already-ingested archives
  just get re-upserted (harmless, only costs time). There's no "only ingest what's new"
  flag; if that matters, move already-ingested archives out of this directory first.
- **`--vector-index all`** builds HNSW + FLAT + ScaNN collections for runtime algorithm
  switching; default is HNSW only.
- Full flag reference and troubleshooting:
  [`../../../remote-server/README_INDEXING_SEARCH.md`](../../../remote-server/README_INDEXING_SEARCH.md#4-ingest-the-lot-archives).

## Verifying what's actually indexed

A video existing here as a `.zip` does not mean it's searchable — check Milvus directly
if a query for a specific video/lot comes up empty:

```python
from pymilvus import connections, Collection
connections.connect(uri=r"<repo>\challenge_resources\data\milvus_lite.db")
col = Collection("video_frames"); col.load()
print(col.num_entities)
rows = col.query(expr='video_id == "L21_V006"', output_fields=["timestamp_ms"], limit=10000)
print(len(rows))
```
(Same one-process-at-a-time caveat as above — stop the backend first.)

## Lot history (informational, not authoritative — check Milvus for ground truth)

| Lots | Ingested |
|---|---|
| L26_c, L26_d, L26_e, L27_a, L28_a | Initial ingest |
| L21_a, L22_a, L23_a, L24_a, L25_a, L26_a, L26_b | Added 2026-08-10 |
| L29_a, L30_a | Added 2026-08-10 |

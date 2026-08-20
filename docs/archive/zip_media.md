# Frames and Video from the Organizer ZIPs

Lot data (`*_results.zip`) is **embeddings and metadata only — no JPGs, no
MP4s**, and it is the only dataset either backend has. A search returns
`video_id` + `timestamp_ms` and nothing to show. This page is how a picture and
a playable video get produced from those IDs.

Both backends serve the same two routes from the same MP4 sample-table logic;
only where the bytes come from differs:

| | local-backend | remote-server |
|---|---|---|
| Source | organizers' host, HTTP Range | `challenge_resources/data/raw_zip/Videos_L*.zip`, file seek |
| Setup | build `zip_video_index.json` (below) | drop the archive in `raw_zip/`, nothing to build |
| Code | `app/services/zip_frame_source.py` | `app/services/local_zip_media.py` |
| Caches | moov on disk, GOP in memory, retry/backoff | none — a local seek is free to repeat |
| Coverage | every lot with a URL in `zip_video_links.txt` | every lot whose archive is on this disk |

There are two archives per lot, and they are not the same file:

| Archive | Lives in | Holds | Used by |
|---|---|---|---|
| `L{lot}_{sub}_results.zip` | `challenge_resources/data/zip_file/` | `scenes.json`, `keyframes.json`, `embeddings.npy` | ingest + vector export ([Readme-Ingest.md](../challenge_resources/data/zip_file/Readme-Ingest.md)) |
| `Videos_L{lot}_{sub}.zip` | organizer host, or `challenge_resources/data/raw_zip/` | the actual `video/*.mp4` files | everything on this page |

The video archives are ~10–100 GB per lot. Nothing here downloads one.

---

## 1. Setup

### remote-server — nothing to build

Put `Videos_L*.zip` in `challenge_resources/data/raw_zip/` (or point `RAW_ZIP_DIR`
elsewhere). On first request the central directories of every archive there are
scanned into a `video_id → (archive, offset, size)` map — 0.09 s for a 4 GB
archive, since only the directory is read. `GET /api/health` reports
`local_zip_videos`, which is the check for whether it found anything.

### local-backend — build the ZIP index once

```bash
cd local-client/local-backend
python -m scripts.build_zip_video_index \
    --urls-file ../../challenge_resources/data/zip_video_links.txt \
    --output   ../../challenge_resources/data/zip_video_index.json
```

`zip_video_links.txt` is one archive URL per line (`#` comments allowed). The
script reads only ZIP central-directory and local-file-header bytes — a handful
of Range requests per archive, no video content — by handing a Range-backed
file-like object to stdlib `zipfile`, so ZIP64 handling comes for free.

Output is a flat manifest:

```json
{"L21_V001": {"zip_url": "https://…/Videos_L21_a.zip",
              "entry_name": "video/L21_V001.mp4",
              "data_offset": 76,
              "size": 130322332}}
```

`data_offset` is the absolute byte position where that MP4's raw bytes start
inside the archive. That is the whole trick: entries are stored uncompressed
(`ZIP_STORED`), so byte *N* of the MP4 is byte `data_offset + N` of the ZIP, and
any range inside one file is reachable with one HTTP Range request. Entries that
are not `ZIP_STORED` are skipped with a warning — they cannot be served this way.

No `zip_video_index.json` means every lookup 404s and the UI silently falls back
to YouTube. Leaving the routes on with an empty index is safe.

---

## 2. Video playback — `GET /api/zip-video/{video_id}`

`app/services/remote_zip_proxy.py`. Pure byte translation, no decode:

```
browser  Range: bytes=1048576-2097151        (its own seek, on the MP4)
   ↓
proxy    look up data_offset for video_id
         Range: bytes={data_offset+1048576}-{data_offset+2097151}   (on the ZIP)
   ↓
upstream 206 Partial Content  →  streamed through unmodified, as 206 + Content-Range
```

The response advertises `Accept-Ranges: bytes` and a `Content-Range` sized to
the *MP4*, not the archive, so the `<video>` element seeks and buffers exactly as
it would against a plain MP4 URL. `VideoModal.tsx` only reaches this route when
there is no `youtube_id` or the YouTube player reported an error; a 404 here is
the expected signal to fall back.

Only the connection-open step is retried (3 attempts, exponential backoff).
Retrying mid-stream is meaningless once bytes are already flowing to the browser
— the response just ends early and the player's own `onError` handles it.

On **remote-server** (`app/services/local_zip_media.py`) the same route seeks in
the local archive and streams 1 MB chunks off disk. No upstream, so no retry
layer; the response headers are identical, so the browser cannot tell.

---

## 3. Thumbnails — `GET /api/zip-frame/{video_id}/{timestamp_ms}`

`app/services/zip_frame_source.py`, ported from the organizers'
`remote_zip_video_toolkit` index design. The naive alternative — point ffmpeg at
the URL and let it probe the container — costs several round trips per frame and
collapses under a grid load.

```
timestamp_ms
  → frame_id           = round(ts/1000 × fps)
  → moov box           downloaded once per video (~1 MB), cached on disk
  → sample table       stts/ctts/stss/stsz/stsc/stco parsed by mp4_box_parser.py
  → GOP byte range     nearest preceding keyframe … target sample + 4 (B-frame margin)
  → ONE Range request  on the ZIP, at data_offset + region_start
  → AVCC → Annex-B     SPS/PPS prepended from avcC
  → ffmpeg             -f h264 -i pipe:0 -vf select=eq(n,target) → one JPEG
```

Three caches, each fixing a different cost:

| Cache | Where | Why |
|---|---|---|
| `moov` box | disk, `cache/zip_moov/` (`ZIP_MOOV_CACHE_DIR`) | ~1 MB per video. A cold 30-video grid pulls ~30 MB before the first thumbnail; bandwidth, not concurrency, was the wall — a restart used to cost 45 s, now 18 s |
| sample index | memory, per video | parsing the same `moov` again is pointless |
| GOP byte region | memory, LRU by bytes (`ZIP_REGION_CACHE_MB`, default 192) | frames seconds apart usually sit in the *same* GOP, so they want the identical range |

Plus single-flight in `main.py`: the same `(video_id, timestamp_ms)` requested
concurrently — a duplicate result row, two temporal steps landing on one frame —
shares one in-flight decode instead of starting several.

Failure handling is deliberate: `get_frame_jpeg()` returns bytes or raises
`ZipFrameUnavailable`, never a raw network or subprocess error. The route turns
that into 502, a timeout into 504, and anything unexpected into 500 — a client
is never left waiting with nothing coming back.

**remote-server** runs the identical pipeline minus every line that exists to
survive a network: no moov cache, no GOP cache, no single-flight, no retries —
the region is one `seek`+`read`. Only the ffmpeg semaphore
(`ZIP_FRAME_DECODE_CONCURRENCY`) is kept, because that cost is CPU, not I/O.

---

## 4. Client-side decode (optional, recommended when sharing)

ffmpeg-per-thumbnail scales with *viewers*, not with data: 100 thumbnails per
person, all on the host. Setting `NEXT_PUBLIC_FRAME_DECODE=client` in
`local-client/frontend/.env.local` moves the decode into the browser:

```
GET /api/zip-frame-plan/{video_id}/{timestamp_ms}
    → { codec, parameter_sets, nal_length_size, samples[], bytes_url, target_pts }
GET /api/zip-bytes/{video_id}/{region_start}/{length}
    → raw GOP bytes (immutable, cached for a year)
browser: WebCodecs VideoDecoder → the target frame
```

The backend does the same parsing as `/api/zip-frame` but stops before ffmpeg,
so it costs pure arithmetic once a video's `moov` is cached. The frontend
(`src/lib/useFrameImage.ts`, `src/lib/zipFrameDecoder.ts`) falls back to the
server URL on anything — no WebCodecs, unsupported codec, malformed GOP — with
no error state, so a slow-path thumbnail looks identical.

`/api/zip-bytes` is not a general proxy: the requested range is validated
against that video's own extent in the archive. And the browser cannot bypass
us — the organizer's host sends no `Access-Control-Allow-Origin`, so the bytes
still travel through the backend; only the decoding leaves.

---

## 5. Getting the frame right

`timestamp_ms → frame` is the inverse of what the ingest did
(`timestamp_ms = frame_number / fps * 1000` from each video's `scenes.json`), and
two things about it are easy to get wrong in a way that returns a *plausible*
picture from the wrong moment:

**fps must be the one the ingest used.** Deriving it from the MP4's own `moov`
agrees for the 25.0 fps majority, but 91 of 873 videos are 29.97/30.0 and one is
a 26.438 VFR average — and an fps error grows with the timestamp instead of
staying bounded. `remote-server/scripts/export_video_fps.py` writes
`challenge_resources/data/video_fps.json` once; both backends read it, falling
back to scanning the archives, then to the `moov`.

```bash
cd remote-server && python scripts/export_video_fps.py    # rerun after ingesting new lots
```

**Decode order is not presentation order.** Samples sit in the file in decode
order; a decoder emits them in presentation order, and with B-frames those
differ. `select=eq(n,…)` counts what ffmpeg *emits*, so the sample's position in
the file is the wrong index — it was 3 frames off at 60 s on `L30_V001`. The
first presented frame's pts is also `base_pts`, not 0, whenever there is
reordering, which cost another frame. Both are handled in `_range_plan()`; the
browser decoder sidesteps them entirely by matching on pts.

`remote-server/tests/test_local_zip_media.py` checks the served frame against
ffmpeg decoding the unpacked MP4, so a regression names the frame it returned
instead of just failing.

## 6. If you would rather unpack

Extract `<video_id>.mp4` into `challenge_resources/data/videos/` and serve those
instead. Costs disk and buys nothing over the routes above, which is why the
`FRAME_IMAGE_SOURCE=local`/`local_video`/`youtube_storyboard` modes were removed
along with the `AIC2026_sample` layout they read.

---

## 7. Tuning and troubleshooting

| Env var | Default | Effect |
|---|---|---|
| `ZIP_FRAME_CONCURRENCY` | 12 | requests in flight on `/api/zip-frame` |
| `ZIP_FRAME_DECODE_CONCURRENCY` | 6 | concurrent ffmpeg processes (CPU-bound) |
| `ZIP_REGION_CACHE_MB` | 192 | GOP byte cache size |
| `ZIP_MOOV_CACHE_DIR` | `cache/zip_moov/` | relocate the `moov` cache (local-backend) |
| `NEXT_PUBLIC_FRAME_DECODE` | `server` | `client` = WebCodecs in the browser (local-backend) |
| `RAW_ZIP_DIR` | `challenge_resources/data/raw_zip` | where remote-server looks for archives |
| `VIDEO_FPS_MAP` | `challenge_resources/data/video_fps.json` | relocate the fps map |

Capping both concurrency knobs at one number is a mistake worth avoiding: it
queues per-video `moov` downloads behind ffmpeg processes, and a cold grid gets
slower. They throttle different resources.

| Symptom | Cause |
|---|---|
| Thumbnails 404 / player falls back to YouTube | local-backend: no `zip_video_index.json`, or this `video_id` is not in it. remote-server: that lot's archive is not in `raw_zip/` — check `local_zip_videos` in `/api/health` |
| The frame is close but not the one asked for | an fps mismatch; run `export_video_fps.py` and check the startup log for "using ingest fps … over moov-derived …" |
| 502 on every frame of one video | that entry is not `ZIP_STORED`, or its `moov` could not be located |
| 504 | upstream slow past `ZIP_FRAME_TIMEOUT_SEC` (45 s), queue wait included |
| `imageio-ffmpeg is required` | `pip install imageio-ffmpeg` in the backend venv |
| First grid after a restart is slow, later ones fast | expected — `moov` downloads. It is cached on disk, so this is once per video, not once per restart |

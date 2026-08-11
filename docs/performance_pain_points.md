# Performance pain points

Measured 2026-08-11 on an i7 (20 logical CPUs, **15.7 GB RAM**), local-backend
in `SAMPLE` mode against Milvus Lite, 193,508 vectors / 873 videos, frontend
shared over ngrok.

Every number came from a measurement. Where a conclusion was later overturned
by a better measurement, the correction is kept rather than quietly deleted —
two of the wrong turns below cost real time and are worth not repeating.

---

## 1. FIXED — repeated ingests left 56% dead rows, and that caused everything else

`scripts/ingest_zip_pipeline_results.py` globs **every** `*_results.zip` on each
run and upserts. Adding L29/L30 therefore rewrote all ~193k rows; the
superseded ones stayed on disk as tombstoned segments, complete with their
HNSW indexes, and were loaded into RAM on every startup.

Segment row-id ranges told the story — 9 segments spanning ids 1..438,837 for
193,508 live entities:

| | Before | After clean re-ingest |
|---|---:|---:|
| Segments | 9 | 5 |
| Data on disk | 2.26 GB | 0.93 GB |
| Indexes on disk | 2.32 GB | 0.95 GB |
| **RSS after `col.load()`** | **5.46 GB** | **2.82 GB** |
| Collection load time | 2.4 s | 0.9 s |

`col.compact()` is a **no-op in Milvus Lite** — it returns `job=None`,
`compaction id: 0`, and reclaims nothing. The only way to reclaim is to drop
the DB directory and re-ingest (4.4 minutes for all 14 lots).

**Effect on a 4-event query** (`top_k=60`, `duplicate_threshold=0.98`, freshly
worded so nothing hits `lru_cache`):

| | Before | After |
|---|---|---|
| Wall time | 14.2 s / 25.1 s / 4.9 s | **3.79 s / 4.09 s / 4.03 s** |
| Free RAM during | 1.0–2.5 GB | **6.4 GB** |
| Text encode | 100–1185 ms, unstable | **113–140 ms, stable** |
| vector_search | 190–870 ms, unstable | **~420 ms, stable** |

The headline is not that it got faster — it is that **the variance is gone**.

### Maintenance rule this implies

Re-ingest is not additive. After adding archives, either move the
already-ingested ones out of `zip_file/` first, or delete
`challenge_resources/data/milvus_lite.db` and rebuild from scratch. Doing
neither silently doubles resident memory. See
[Readme-Ingest.md](../challenge_resources/data/zip_file/Readme-Ingest.md).

### Wrong turn #1 — "the text encoder is 11x slower inside the backend"

The encoder measured 106 ms standalone and 1130–1185 ms inside the backend, so
it looked like a backend-specific defect. It was not. The standalone runs had
the machine to themselves *because Milvus Lite's file lock forces the backend
to be stopped first*. Re-measured properly — both processes running, both fed
strings never encoded anywhere:

| | encode (ms) |
|---|---|
| Standalone | 121, 105, 281, 433, 249, 104 |
| Backend, same moment | 552, 109, 272, 100 |

Same distribution. There was never a backend-specific problem; there was one
machine short on RAM. The lesson: never compare a process that owns the
machine against one that is sharing it.

### Wrong turn #2 — "Milvus Lite does not implement HNSW"

Inferred from `ef` having no effect and recall being 1.000 at every `ef`.
Wrong. Milvus Lite implements HNSW via faiss (`milvus_lite/index/faiss_hnsw.py`),
faiss-cpu 1.15.0 is installed, and 0.95 GB of real `.vector.hnsw.idx` files sit
on disk. The proposal that followed from this — replace Milvus Lite with a
numpy memmap — would have *removed* a working ANN index in exchange for a
linear scan. It was dropped before any code was written.

---

## 2. CLOSED (won't fix) — multi-event queries are sequential, but concurrency doesn't help

A 4-event query issues **two rounds** of four searches, not four searches:

```
round 0  4 x (encode ~115 ms + search top_k=450 ~420 ms) = 2422 ms  -> 60 candidates, 56 kept
round 1  4 x (encode 0.002 ms cached + search top_k=300 ~300 ms) = 1254 ms
                                                          total = 3740 ms
```

Round 1 exists because the duplicate filter under-delivered: oversampling at
1.5x produced 60 candidates and filtering kept 56, short of `top_k=60`, so
`request_more_results()` ran another full round.

The events are independent — DP chaining only runs once all four candidate
lists exist — so in principle a round could overlap them. Measurement says
don't: see below, the machine is already saturated by one event's work.

`lru_cache` behaves correctly here: round 1 re-encodes nothing (0.002 ms).

**The remaining prize is round 1 itself.** It cost 1.25 s and exists only
because filtering left 56 of a needed 60. Removing the need for it is worth
more than any concurrency change — see item 3.

### Measured: parallelising the encoder is not worth it, and is dangerous

`_duy_temporal_core.py:34` already wraps the four events in `asyncio.gather`,
but everything inside is blocking, so the loop never yields and they run one
after another. Before wrapping it in `asyncio.to_thread`, the question is
whether the work can overlap at all. Fixed workload — the four encodes one
4-event query needs:

| Configuration | Wall for 4 encodes |
|---|---:|
| Sequential, 1 session, default threads | **412 ms** |
| 4 threads, 4 sessions, **default** threads (20 each) | **2597 ms** (6.3x worse) |
| 4 threads, 4 sessions, 5 threads each | 373 ms (-9%) |
| 2 threads, 2 sessions, 10 threads each | 397 ms (-4%) |

**The hazard is the second row.** ONNX Runtime gives each session every core it
can find; four sessions on a 20-core machine means 80 threads thrashing. Any
future change that runs encodes concurrently — `asyncio.to_thread`, a worker
pool, or simply two teammates searching at once through a threaded handler —
**must** set `intra_op_num_threads`, or throughput collapses ~5x.

The best case is a 9% gain, because one inference already saturates the
machine. Encode is 0.41 s of a 3.7 s query (11%); search is 2.8 s (76%). The
encoder is not where the time is.

Threading encode *and* search together measures the same way: 1.20x at two
events, **1.03x at four**.

### Why: the encoder is memory-bandwidth bound, not CPU bound

Single-encode latency against the threads given to that one inference:

| threads | 1 | 2 | 4 | 8 | 14 | 20 |
|---|---:|---:|---:|---:|---:|---:|
| ms | 396 | 221 | 136 | 117 | **105** | 115 |
| speedup | 1.00x | 1.79x | 2.91x | 3.40x | 3.78x | 3.46x |

It plateaus at 8 threads and gets *worse* at 20. A compute-bound model would
keep scaling toward 20x. Cross-check: 0.54 GB of weights / 0.105 s =
**5.1 GB/s**, about what this machine's bus sustains (a single-threaded
streaming read measures 2.1 GB/s).

This explains the ~10 encodes/sec ceiling that every thread arrangement hit —
each encode drags 541 MB of weights across the bus, and the bus is the wall.
It also explains why concurrency cannot help: extra threads do not create extra
bandwidth.

**But it makes batching valuable.** A batch of 4 in one forward pass reads
those 541 MB *once* for all four queries. Compute goes up 4x, and compute is
not the limit:

| | now | projected with batch=4 |
|---|---:|---:|
| 4 events | 412 ms | ~150–200 ms |
| throughput ceiling | ~10 enc/s | ~35–40 enc/s |

The projection is inferred from the bandwidth analysis, not measured — it
cannot be measured until an export that accepts a batch exists.

### What the export needs to change

The root cause is exporting with **batch=1**: `batch x num_heads = 1 x 20 = 20`
is indistinguishable from `num_heads = 20`, so the tracer recorded the literal
20 and the batch dimension dissolved into the head dimension. That is the
`[72, 20, 64]` reshape; there are 210 constant-shaped Reshape nodes in total,
about six per attention block, so this is a re-export and not a patch.

1. Export with an example input of **batch >= 2** (4 is a good choice) — then
   `4 x 20 = 80` cannot be confused with 20.
2. `do_constant_folding=False` — constant folding is what turns shape
   arithmetic into literals.
3. Keep `dynamic_axes={'input_tokens': {0: 'batch'}, 'text_embeddings': {0: 'batch'}}`.
   It is already there (the I/O is declared `batch_size`) but could not
   propagate inward because of 1 and 2.
4. Better still, use the dynamo exporter with `torch.export.Dim("batch")`
   instead of the tracing exporter.
5. Export and verify in fp32 **first**, quantise to int8 after, then verify
   again — not both in one step.

To verify, re-run the graph inspection: the `[72, 1, 3, 1280]` and
`[72, 20, 64]` constants must be gone, replaced by `Concat(Shape(...))`
subgraphs. Then run batch 1/2/4/8 and check every row matches the batch=1
result.

### Serving several teammates

The ~10 encodes/sec ceiling is for the **whole machine**, shared. Running four
sessions in parallel does not raise it (412 ms for 4 encodes at 1x20 threads,
397 ms at 2x10, 373 ms at 4x5).

So the right shape for multi-user is not parallel sessions but **dynamic
micro-batching**: a queue that collects queries arriving within ~30 ms and
encodes them in one pass. That is the standard inference-server pattern and it
fits a bandwidth-bound model exactly — but it needs the batch-capable export
first.

## 2b. FIXED — the encoder now queues, and pins its thread count

`_encode_uncached` holds a lock, so callers wait rather than run side by side,
and the ONNX session is created with `intra_op_num_threads` from
`PECORE_ONNX_THREADS` (default 14) instead of ONNX Runtime's "every core".

Both follow from the bandwidth measurement above: serialising costs ~9% in the
best case and removes a 6.3x cliff, and 14 threads beat 20 (105 ms vs 115 ms).
Measured after the change: encode 101–106 ms per query, down from 113–140 ms.

## 2c. FIXED — duplicate and neighbouring thumbnail requests no longer repeat work

A grid fires ~100 thumbnail requests at once, and they overlap heavily: the
same frame appears at several ranks, and frames seconds apart in one video
usually sit in the same GOP and therefore want the identical byte range.

Two changes, neither of which adds latency:

- **Single-flight** (`_single_flight` in `main.py`) — concurrent requests for
  the same `(video_id, timestamp_ms)` share one decode.
- **Region cache** (`_fetch_region_cached` in `zip_frame_source.py`) — an
  LRU over fetched GOP byte ranges, bounded by `ZIP_REGION_CACHE_MB` (192).
  The per-key lock matters as much as the cache: without it, twelve thumbnails
  from one video all fire before any of them fills it.

Measured, twelve requests fired simultaneously:

| Pattern | Wall |
|---|---:|
| Same frame x12 | **130 ms** — one decode, not twelve |
| 12 nearby frames, one video | 872 ms |
| 12 different videos, cold | 2115 ms |
| the same 12 again, regions cached | **450 ms** |

A timed "wait 1–2 s to fill a batch" queue was considered and rejected: it
would add its full delay to every thumbnail, including those with no duplicate
to merge, while catching the same overlap this does for free.

## 2d. NEW — thumbnails can decode in the browser

`NEXT_PUBLIC_FRAME_DECODE=client` switches `ResultCard` from `<img
src=/api/zip-frame/...>` to: ask `/api/zip-frame-plan/...` where the bytes are,
fetch that range from `/api/zip-bytes/...`, decode with WebCodecs. Falls back
to the server route silently when WebCodecs is absent or a frame fails.

This is the one cost that scales with *viewers* rather than with data — each
person's grid is up to 100 thumbnails, and in server mode every one was an
ffmpeg process on the host.

Verified by replaying the client's byte assembly through ffmpeg and comparing
against the server route: 5/5 frames matched, across GOPs from 12 to 178
samples. Bandwidth barely changes for typical GOPs (113 KB region vs 106 KB
JPEG) but a long GOP costs more (800 KB vs 125 KB), so this trades some
bandwidth for all of the host's decode CPU.

**It cannot go further than that.** The organizer's host sends no
`Access-Control-Allow-Origin`, so the browser cannot fetch those ranges itself
— the bytes still pass through the backend, only the decoding leaves.
`/api/zip-bytes` validates every range against that video's own extent in the
archive, so it is not a general proxy.

## 2e. FIXED (correctness) — the DP let one frame satisfy every event

Not a performance bug, found while measuring one. With a group's
`temporal_offset_ms` at 0, the DP's ordering test read
`previous.timestamp <= current.timestamp`, which admits *equality* — so a chain
could use the same frame for all four events.

Those degenerate chains then won the ranking: the best-scoring frame counted
once per event, and a span of zero satisfies every interval window. Measured on
the 4-event asparagus query before the fix:

| Strategy | Results | Chains that repeat a frame |
|---|---:|---:|
| 0-5s | 51 | **51** |
| 5-10s | 30 | 26 |
| 10-20s | 39 | 33 |
| unbounded | 60 | **56** |

Because a result row renders `path[-1]`, every one of those displayed the same
picture — which is what "it all collapses onto event 4" looks like from the UI,
and why genuine chains never reached the top.

Fixed by requiring each step to land strictly after the one before
(`MIN_STEP_MS`). After: **0 repeated-frame chains** in all four strategies.

A second bug hid behind it: `temporal_results` built a chain's `steps` by
passing all of them to `data_provider.results()`, which **drops repeated
frame_ids**. Right for a result list, wrong for an ordered chain — a four-event
match arrived at the client with one step. Now rendered one step at a time, so
`steps` always mirrors `evidence`.

Consequence to know about: **0-5s with four events now legitimately returns
nothing** for queries like the asparagus one. Four *distinct* keyframes do fit
in 5 s (median keyframe gap is ~1.8 s, and each sampled video has 68–254 such
windows), but they must also be the top semantic match for four different
events in order. The unbounded strategy's best chain for that query spans
5680 ms — just outside the window. It used to "work" only by returning one
frame four times.

## 2f. FIXED — one deep fetch instead of refill rounds

`PER_QUERY_LIMIT` 300 → **1000**. `FETCH_CAP` bounds total hits at 1000 either
way; the rounds were merely spending that budget down (450, 300, 250) and
re-running DP and duplicate filtering over everything gathered so far on each
pass.

| 4-event query | Before | After |
|---|---:|---:|
| Rounds | 3 | **1** |
| Total | 4003 ms | **2339 ms** |

Consistent with the per-event measurement: reaching 1000 hits costs 660 ms
across three searches but 486 ms in one.

## 3. OPEN — `top_k` is the real search cost, and the strategy asks for 450

`_duy_temporal_core.py:5` sets `PER_QUERY_LIMIT = 300`; `base_strategy.py`
applies `OVERSAMPLE_FACTOR = 1.5`, so a `top_k=60` request issues a 450-hit
search per event.

Measured on the clean collection (metadata-only output, ef=512):

| top_k | 10 | 60 | 100 | 200 | 300 | 450 | 600 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ms | 98 | 111 | 147 | 161 | 172 | **271** | 282 |

There is a **~95 ms floor** per search independent of `top_k` — gRPC through
Milvus Lite's in-process server plus the fan-out across 5 segments. Twelve
searches per query means over a second of pure floor.

### The refill rounds are draining one fixed budget, expensively

`FETCH_CAP = 1000` caps total hits per (channel, query). The rounds spend it
down: 450, then 300, then 250 — exactly 1000. Each round re-scans from scratch
with a longer exclude list, because HNSW cannot resume.

Full payload, on the clean collection:

| top_k | 250 | 300 | 450 | 600 | 800 | 1000 |
|---|---:|---:|---:|---:|---:|---:|
| clean, ms | 184 | 205 | 271 | 340 | 423 | 486 |
| with a 450-id exclude, ms | 198 | 223 | 280 | 340 | 428 | 474 |

The exclude expression is nearly free (+14 ms at 250, +9 ms at 450), so
exclusion is not the cost. But reaching 1000 hits costs:

| | per event |
|---|---:|
| three searches (450 + 300 + 250) | **660 ms** |
| one search (1000) | **486 ms** |

Same hits, 26% cheaper in one go. **But do not simply fetch deep always** — a
query satisfied by round 0 pays 271 ms today and would pay 486 ms, 79% worse.
The shape that fits the measurement is: keep the first round at 450, and when a
refill is first requested, jump straight to the remaining budget instead of
stepping down through it.

Observed round counts vary a lot by query (1 to 3+ in the samples taken), which
is why total query time still swings between ~2.4 s and ~16 s even after the
collection was rebuilt.

## 4. NOT a tuning issue — `ef` is discarded by the library

Every sweep of `ef` came back flat, at every `top_k`, for every query type:

| ef | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ms | 100 | 101 | 96 | 106 | 112 | 99 | 95 |

The explanation recorded here first — "`ef` is not where the time goes; the
~95 ms floor dominates" — was **wrong**. `ef` never reaches faiss:
`milvus_lite`'s search executor calls `index.search()` without `params`, so
`efSearch` is hard-wired to 64, and a second defect bounds the traversal
regardless. Under 0.7% of the collection is scored on any query.

Full diagnosis, evidence and reproduction:
[milvus-lite-hnsw-recall-bug.md](milvus-lite-hnsw-recall-bug.md).

This is a **correctness** problem, not a performance one, and it outranks
everything else in this document — searches have been returning results chosen
from less than 1% of the corpus. `DEEP_SEARCH_EF = 512` in
`milvus_client.py:43` is inert locally; it still matters on remote-server,
which runs real Milvus.

## 5. FIXED — cold thumbnails were bandwidth-bound

Every video needs its ~1 MB `moov` box before any of its frames can decode. A
grid spanning 30 videos pulled ~30 MB before the first thumbnail appeared.
Concurrency was the wrong lever:

| in-flight requests | cold 60-thumbnail grid |
|---|---|
| 6 | 45.3 s, 2 timeouts |
| 12 | 45.2 s, 2 timeouts |
| 32 | **60.3 s**, 3 timeouts |

Fixed by caching `moov` to disk (`cache/zip_moov/`), since it never changes:

| | |
|---|---|
| Cold, empty cache | 45.2 s, 2 failed |
| After restart, cache warm | **18.4 s, 60/60 ok** |
| Warm, through ngrok | **9.4 s, 60/60 ok** |

Decode (CPU) and requests in flight (network) are now throttled separately:
`ZIP_FRAME_DECODE_CONCURRENCY` (6) and `ZIP_FRAME_CONCURRENCY` (12).

## 6. FIXED — the Next.js dev server cannot proxy the thumbnail load

Serving `/api/*` through `next.config.js` rewrites killed the dev server: a grid
loads up to 100 thumbnails, the browser aborts the in-flight ones on every new
search, and the ECONNRESET storm ended the process. Replaced with
`scripts/share-proxy.cjs`. See [launch_scripts.md](launch_scripts.md).

## 7. FIXED — tuning draft polling

Both `index.tsx` and `tuning.tsx` polled `/api/tuning-draft` every second, per
open tab. `index.tsx` was worse: it called `setStrategyConfigDraft`
unconditionally each tick, and since `strategyConfigDraft?.revision` was in the
effect's deps the interval was torn down and rebuilt every tick. Its deps also
included `queryGroups`, so every keystroke, translate and added step fired
another fetch.

Replaced with a `storage`-event ping (`src/lib/tuningPing.ts`) — fires only on a
real save, and only for saves made on the same machine, so a teammate's slider
cannot re-trigger your search. Auto-search now happens **only** on that ping;
the page's own bookkeeping writes (syncing `event_weights` to the step count
when a temporal step is added) no longer trigger a query.

---

## Constraints found while exploring fixes

**The ONNX export cannot batch** — `batch=2` fails with
`input_shape_size == requested_shape_size was false`. 210 Reshape nodes carry
constant shapes; the batch dimension was folded into the head dimension at
export time. Fixable, since the model is exported in-house — see "What the
export needs to change" under item 2.

**The organizer's host sends no CORS headers.** A cross-origin ranged GET to
`https://aic-data.ledo.io.vn/...` returns `Accept-Ranges: bytes` but no
`Access-Control-Allow-Origin`, so a browser cannot fetch those byte ranges
itself. Any design where the client does its own Range requests against the
organizer's host is blocked by the browser. It becomes possible only on the
remote-server design, where we host the ZIPs and control the headers.

Frame decoding can still move to the browser without that: the backend already
builds the Annex-B GOP bytes before handing them to ffmpeg, so it could return
those and let WebCodecs' `VideoDecoder` do the work on the client. That trades
our CPU for bandwidth and needs an ffmpeg fallback for browsers without
WebCodecs.

---

## Suggested order

1. **Item 3** — the whole remaining budget is here. Eight searches at ~300–420 ms
   is 76% of a 3.7 s query, and four of those eight exist only because round 0
   came up four candidates short. Sweep `PER_QUERY_LIMIT` and
   `OVERSAMPLE_FACTOR` together against real queries and read total wall time —
   they pull in opposite directions, and this knob has already been guessed
   wrong once.
2. **Re-export the ONNX model with a working batch dimension.** Worth ~0.25 s
   on a single 4-event query, which is modest — but it lifts the machine-wide
   encode ceiling from ~10/s to ~35–40/s, and that ceiling is what a room full
   of teammates actually runs into. Do it for the second reason, not the first.
3. WebCodecs offload only if thumbnails still hurt once teammates are on it.

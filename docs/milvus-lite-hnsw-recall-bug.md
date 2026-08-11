# Milvus Lite: HNSW search misses true nearest neighbours

Found 2026-08-11 while investigating why a four-event temporal query returned
the wrong video. Not a problem with our data, our ingest, or our index build.
Written up separately from
[performance_pain_points.md](performance_pain_points.md) because it is
upstream's to fix and because it invalidates several conclusions recorded
there.

## Which component, exactly

The defect is in **`milvus_lite`** — the embedded server — and **not** in
`pymilvus`, the client. They are separate pip packages:

| Package | Role | Verdict |
|---|---|---|
| `pymilvus` 3.0.1 | client we call `Collection.search()` on | **correct** |
| `milvus_lite` 3.2.0 | Milvus reimplemented in-process, no Docker | **defective** |

The client is exonerated by the server's own code: `milvus_lite`'s gRPC
servicer *successfully parses* the `search_params` we send. A malformed client
request would not have got that far. The parameter completes the client→server
hop and then dies **inside the server**, between its gRPC layer and its index
layer.

**This does not affect `remote-server`.** Real Milvus is an entirely different
implementation (Go + C++/knowhere); `milvus_lite` is a Python reimplementation
for running without Docker. `ef` and `DEEP_SEARCH_EF` still do what they say
there — only the local dev path is broken.

## Standalone reproduction (for the upstream report)

[`repro/milvus_lite_hnsw_recall.py`](repro/milvus_lite_hnsw_recall.py) — no
private data, synthetic vectors, temporary database:

```
pymilvus 3.0.1   milvus_lite 3.2.0
50000 vectors, dim 128, HNSW M=16 efConstruction=256, COSINE

--- query from the data's own cluster ---
  exact best score: 0.3986
       ef  hnsw best  recall@10   rows
       16     0.3235         0%     10
       64     0.3235         0%     10
      256     0.3235         0%     10
     1024     0.3235         0%     10
     4096     0.3235         0%     10
    16384     0.3235         0%     10
  reachable at limit=50000: 609 of 50000 (1.2%)

--- query from elsewhere on the sphere ---
  exact best score: 0.3436
       ef  hnsw best  recall@10   rows
       16     0.2976         0%     10
      ...     0.2976         0%     10
    16384     0.2976         0%     10
  reachable at limit=50000: 529 of 50000 (1.1%)

--- with params forwarded AND valid_mask dropped (patched) ---
  from cluster     exact 0.3986  hnsw 0.3986  recall@10 100%
  from elsewhere   exact 0.3436  hnsw 0.3436  recall@10 100%
```

**recall@10 is 0% at every `ef`**, on a plain 50k-vector collection with default
index settings, for queries both inside and outside the data's own cluster.
Roughly 1% of the collection is reachable. Patching the two dropped inputs
restores 100%.

The distribution of the query turns out not to matter at all — an earlier
theory here blamed the CLIP modality gap, and the synthetic repro disproves it.

## Suggested fix

Two changes in `milvus_lite`:

```diff
--- a/milvus_lite/search/executor_indexed.py
+++ b/milvus_lite/search/executor_indexed.py
@@
-        local_ids, dists = index.search(
-            query_vectors, top_k, valid_mask=local_mask
-        )
+        # Forward the caller's search params (ef / nprobe / ...); the gRPC
+        # servicer already parses them and the index classes already accept
+        # them, so today they are silently dropped here.
+        # Skip the IDSelector path when nothing is actually excluded --
+        # semantically identical (valid_mask=None means "all rows valid") but
+        # it avoids the traversal bound the selector imposes.
+        local_ids, dists = index.search(
+            query_vectors, top_k,
+            valid_mask=None if local_mask.all() else local_mask,
+            params=search_params,
+        )
```

`search_params` needs threading in from `Collection.search` through
`execute_search_with_index`, which currently declares no argument for it.

The `local_mask.all()` guard is not a workaround: `FaissHnswIndex.search`'s own
docstring states that `valid_mask=None` means "all rows valid", so the two are
equivalent by definition when the mask is all-true. It simply avoids paying the
selector path when there is nothing to select.

## How this was established, in order

The conclusion does not rest on reading library source. Source was only opened
after the behaviour had already been pinned from outside, and it was opened to
explain the numbers rather than to find them:

1. Scored the collection's own vectors against the query with numpy — no Milvus
   involved. Found a vector at cosine **0.2518**.
2. Milvus reports rank-1 = **0.2210** for the same query and never returns the
   0.2518 vector at any depth.
3. For a frame Milvus *does* return, its score and ours **agree exactly**
   (0.1174) — so the metric is not the difference.
4. Searching the index *with the missing vector* returns it at rank 1, score
   **1.0000** — so it is indexed and reachable in principle.

Only then: read the source, traced the call at runtime, and reproduced the
correct answer by forcing the two dropped inputs.

## Versions

| | |
|---|---|
| `milvus_lite` | **3.2.0** |
| `pymilvus` | 3.0.1 |
| `faiss` (via faiss-cpu) | 1.15.0 |
| Collection | `video_frames`, 193,508 vectors, dim 1280, COSINE |
| Index | HNSW, `M=16`, `efConstruction=256`, 5 segments |

## Symptom

`Collection.search()` on an HNSW index returns results that are not the nearest
neighbours, and `ef` has no effect at any value.

On the synthetic reproduction below: **recall@10 is 0%** and about 1% of the
collection is reachable.

In this repository, where it was found: a text query returns a rank-1 score of
**0.2210** while the collection contains a vector scoring **0.2518** against the
same query. The better vector is never returned at any depth — not at
`limit=1000`, not at `limit=16384`. Across six unrelated queries, search only
ever returns 829–1175 rows out of 193,508 — **under 0.7% of the corpus is
scored**.

## The missing vector is present and correct

Ruling out a corrupt or partial index, for `L26_V194` frame `L26_V194_002643`
(ts=105720):

| Check | Result |
|---|---|
| `query()` returns it | yes, `‖v‖ = 1.000000`, dim 1280 |
| Searching the index *with that vector* | returns itself at rank 1, **score 1.0000** |
| Milvus's score for a frame it does return | 0.1174 — **identical** to our own dot product |
| Our dot product for the missing frame | 0.2518, computed two ways |
| Segments actually searched | all 5, covering 10240+100000+63840+10240+9188 = **193,508** |

So: the vector is indexed, the metric agrees, every segment is visited. The
graph traversal simply never reaches it.

## Root cause — two defects that compound

### 1. Search parameters never reach the index

`milvus_lite/search/executor_indexed.py:201`

```python
local_ids, dists = index.search(
    query_vectors, top_k, valid_mask=local_mask
)
```

`FaissHnswIndex.search` accepts `params` — and this call site omits it.

`milvus_lite/index/faiss_hnsw.py:169-173`

```python
params = params or {}
ef = int(params.get("ef", 64))
self._index.hnsw.efSearch = ef
```

So `efSearch` is **hard-wired to 64 for every search**, whatever the client
sends.

This is not us passing the parameter in the wrong shape. Both ends of the
library handle it correctly and only the middle link drops it:

| Layer | Handles search params? |
|---|---|
| `adapter/grpc/servicer.py:640,1271` | **yes** — parses `search_params` off the request |
| `search/executor_indexed.py:201` | **no** — the function does not even declare an argument to receive them |
| `index/faiss_hnsw.py:188` | **yes** — forwards `params=sp` to faiss when given |

`executor_indexed.py:201` is the only call to `index.search()` on the vector
path, so there is no alternative route. No client-side call shape can restore
the parameter, because the break happens after the request has already been
parsed successfully.

Confirmed at runtime by tracing the call:

```
FaissHnsw  top_k=1000  params=None  num_vectors=10240
FaissHnsw  top_k=1000  params=None  num_vectors=100000
FaissHnsw  top_k=1000  params=None  num_vectors=63840
FaissHnsw  top_k=1000  params=None  num_vectors=10240
FaissHnsw  top_k=1000  params=None  num_vectors=9188
```

`params=None` on all five.

### 2. The IDSelector path caps traversal, making `ef` irrelevant anyway

The executor always passes `valid_mask` — the per-segment liveness bitmap used
for tombstones and cross-segment dedup — even when nothing is excluded. That
routes faiss through its `IDSelectorBatch` path.

Forcing `ef` through while the mask is still passed changes **nothing at all**:

| `ef` forced | `valid_mask` | rows returned | top score | best score for `L26_V194` |
|---:|---|---:|---:|---:|
| — | kept | 1136 | 0.2210 | 0.1174 |
| — | dropped | 2000 | 0.2430 | 0.1174 |
| 64 | kept | 1136 | 0.2210 | 0.1174 |
| 64 | dropped | 2000 | 0.2430 | 0.1174 |
| 4096 | kept | 1136 | 0.2210 | 0.1174 |
| **4096** | **dropped** | 2000 | **0.2518** | **0.2518** |

Only the last row recovers the true nearest neighbour, and it matches the
brute-force ground truth exactly. **Both defects have to be worked around at
once**, which is why fixing either alone shows no improvement — and why the
first several hypotheses in this investigation looked disproven when they were
merely incomplete.

Latency for that last row is 751 ms against 392 ms today: correct results cost
roughly twice as much on this path.

## Minimal reproduction

```python
import numpy as np
from pymilvus import Collection, connections
from milvus_lite.index.faiss_hnsw import FaissHnswIndex

connections.connect(uri="<path>/milvus_lite.db")
col = Collection("video_frames"); col.load()

q = <a query vector far from the data, e.g. a CLIP text embedding>

# 1. what the index returns
res = col.search(data=[q], anns_field="vector",
                 param={"metric_type": "COSINE", "params": {"ef": 16384}},
                 limit=16384, output_fields=["frame_id"])[0]
print(len(res), res[0].score)          # ~1136 rows, top 0.2210

# 2. what is actually in there
rows = col.query(expr='video_id == "L26_V194"', output_fields=["vector"], limit=16000)
mat = np.asarray([r["vector"] for r in rows], dtype="float32")
print((mat @ q).max())                 # 0.2518 -- higher than the rank-1 above
```

## Why we are not patching it

Monkey-patching `FaissHnswIndex.search` to forward `params` **and** drop
`valid_mask` does restore correct results — and is unsafe. That bitmap is what
hides rows superseded by a later ingest. Our own collection carried **245,000
dead rows** until it was rebuilt (see performance_pain_points.md item 1); with
the mask dropped, those would have been returned as live results. That trades a
silent recall bug for a silent staleness bug.

## Result after switching to exact search

Built with `--vector-index flat` (3.8 min) and switched
`VECTOR_SEARCH_BACKEND=flat`. `data_provider.py` also had `"hnsw"` hard-coded
at three call sites, so the env var had never taken effect — now it reads
`milvus_client.DEFAULT_ALGORITHM`.

The query that started this, four events, asparagus:

| Variant | HNSW | flat |
|---|---|---|
| 4 events | correct video **absent** | **rank 1** |
| 4 events, E4 shortened | rank 1 | rank 1 |
| 4 events, E4 weight 0 | correct video **absent** | **rank 1** |
| first 3 events only | rank 1 | rank 1 |

All four now agree. Two reported oddities were symptoms of this bug, not
separate defects:

- *"Dropping four words from E4 changes which video wins."* Both wordings now
  return the same video.
- *"Setting event 4's weight to 0 isn't the same as removing event 4."* Now it
  is, in effect. The mechanism described earlier is still real — a weight of
  zero removes an event's contribution to the score but not its requirement to
  exist and to fall after the previous step — it simply stopped mattering once
  the video had enough candidate frames to form an ordered chain.

`0-5s` with four events also went from 0 results to 5 (spans 3600–4800 ms):
those chains existed all along and were never visible.

Cost, same query end to end:

| | HNSW | flat |
|---|---:|---:|
| per search (`top_k=1000`) | ~500 ms | ~1570 ms |
| 4-event query, one round | 2339 ms | **6416 ms** |
| correct | no | yes |

An earlier estimate here put exact search at ~200 ms per query based on a
1 GB scan at memory bandwidth. Measured, `BruteForceIndex` runs at closer to
1 GB/s, so it is ~3x the HNSW path rather than free. A numpy `memmap` + BLAS
GEMV over the same 990 MB should reach 3–5 GB/s and is the obvious way to buy
the latency back without giving up exactness — not attempted yet.

## What we are doing instead

Switching the local path to exact search: `VECTOR_SEARCH_BACKEND=flat`, which
the repo already supports (`milvus_client.py:28`, collection
`video_frames_flat`, built with
`ingest_zip_pipeline_results.py --vector-index flat`).

`BruteForceIndex` takes no search parameters, so there is nothing to drop, and
it does not use the graph traversal that the IDSelector path constrains. At
193,508 x 1280 float32 that is a ~1 GB scan per query — on this hardware about
the same as the incorrect HNSW path already costs.

Real Milvus on remote-server is unaffected; this is specific to the embedded
`milvus_lite` build.

## Conclusions this invalidates

Recorded so they are not repeated:

- **"`ef` has no effect because the ~95 ms floor dominates."** Wrong. `ef` has
  no effect because it is discarded before reaching faiss, and because the
  IDSelector path bounds the traversal independently.
- **"Recall is 1.000 at every `ef`."** That first sweep used query vectors
  *taken from the collection*, so each query was itself a node in the graph.
  Every configuration was equally capped, so they agreed with each other and
  with nothing else. Never measure ANN recall with in-distribution queries.
- **"L26_V194 was dropped because no ordered chain existed."** True of the data
  the DP received, but that data was already missing the six frames that would
  have formed the chain.

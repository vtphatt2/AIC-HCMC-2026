# Vector Search Benchmark Report

Evaluation of vector search backends for the PE-Core text-to-frame search
stage only — bypasses UI, API routing, metadata reranking, and temporal
fusion. See [db_schema.md](db_schema.md) for the Milvus collection schema and
[architecture.md](architecture.md#production-search-backends) for how
`VECTOR_SEARCH_BACKEND` selects between these at runtime.

## Setup

- 4,088 PE-Core visual embeddings (1,280-dim), 160 Vietnamese scene-description queries (avg. 20 words).
- Ground truth: target keyframe + visually-close same-scene neighbors, kept tight to avoid inflating recall.
- HNSW M=16/M=32 built from the same FLAT source vectors, so only graph degree differs. All indexes hold the same 4,088 frame IDs.

## Recommendation

```env
VECTOR_SEARCH_BACKEND=hnsw   # CPU default
VECTOR_SEARCH_BACKEND=cagra  # GPU default (CUDA only)
```

| Backend | Verdict |
|---|---|
| **CAGRA** (GPU) | 100% overlap with FLAT exact search, p50 2.8ms — default GPU engine. |
| **HNSW M=16** (CPU) | 99.2% overlap with FLAT, p50 2.7ms, lowest CPU memory — default CPU engine. |
| **HNSW M=32** (CPU) | 99.3% overlap, same latency, more RAM for +0.1% overlap — alternative if memory isn't a constraint. |
| **ScaNN** (CPU) | 96.6% overlap@100, fastest CPU latency (p50 2.1ms) — supported, not a default; the `with_raw_data=true` config tested doesn't prove real memory savings, needs larger-scale tuning first. |
| **FLAT** (CPU) | Exact search — kept as the validation baseline, not for production query traffic. |
| IVF_FLAT (CPU/GPU) | Benchmark-only; noticeably lower overlap (89–98%), not recommended. |

The vector search backend is not the retrieval bottleneck: FLAT exact, GPU
brute force, and CAGRA all return identical results, so quality is limited by
query design and visual embedding alignment, not index choice.

## Accuracy & latency

Recall@K against ground truth, exact-overlap % vs. FLAT, and search latency
(vector search only — PE-Core query encoding is separate: p50 9.9ms, p95
14.2ms):

| Backend | R@10 | R@100 | MRR@100 | Overlap@10 | Overlap@100 | p50 (ms) | p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| FLAT CPU (exact) | 36.25 | 55.00 | 0.2553 | 100.0 | 100.0 | 3.29 | 4.06 |
| CAGRA GPU | 36.25 | 55.00 | 0.2553 | 100.0 | 100.0 | 2.84 | 3.63 |
| GPU brute force | 36.25 | 55.00 | 0.2553 | 100.0 | 100.0 | 1.58 | 2.27 |
| HNSW CPU M=16 | 35.63 | 54.38 | 0.2543 | 99.19 | 99.23 | 2.67 | 3.24 |
| HNSW CPU M=32 | 35.63 | 55.00 | 0.2509 | 99.38 | 99.34 | 2.65 | 3.19 |
| ScaNN CPU | 35.63 | 55.63 | 0.2489 | 98.25 | 96.59 | 2.09 | 2.65 |
| IVF_FLAT CPU | 35.63 | 55.63 | 0.2489 | 98.25 | 96.71 | 2.21 | 2.79 |
| IVF_FLAT GPU | 31.88 | 48.75 | 0.2315 | 93.19 | 91.64 | 0.60 | 0.98 |

`R@100` for IVF_FLAT/ScaNN reads slightly higher than exact FLAT purely from
boundary cases around rank 100 (three extra top-100 hits gained, two stronger
FLAT hits lost, including a rank-1). Exact-overlap is the cleaner signal for
comparing backends — raw `R@100` is noisy here.

## Build time & memory

| Backend | Build time | Runtime RAM/VRAM (4K vectors) | Estimated at 1M vectors |
|---|---:|---:|---|
| FLAT CPU | — | ~3.8 GB | ~5.1–6.0 GB |
| HNSW CPU M=16 | 9.4s | ~3.8 GB | ~5.5–7.0 GB |
| HNSW CPU M=32 | 7.4s | ~1.7 GB | ~6.0–8.5 GB |
| ScaNN CPU | 10.9s | ~3.9 GB | ~5.5–7.0 GB (raw=true, not compressed) |
| IVF_FLAT CPU | 7.4s | ~1.7 GB | ~5.2–6.5 GB |
| CAGRA GPU | — | ~6.1 GB VRAM | ~6.0–9.0 GB VRAM |
| GPU brute force | 0.6ms | ~2.0 GB VRAM | ~5.1–6.5 GB VRAM |
| IVF_FLAT GPU | 111ms | ~2.0 GB VRAM | ~5.3–7.0 GB VRAM (needs large-scale tuning) |

Raw float32 vectors alone are ~5.1 GB at 1M × 1280-dim — most of the estimate
above is that baseline plus each index's graph/centroid overhead.

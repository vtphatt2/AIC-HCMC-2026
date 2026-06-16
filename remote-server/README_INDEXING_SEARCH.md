# Backend Indexing And Visual Search

This backend indexes PE-Core image embeddings from `AIC2026_sample` into Milvus and serves PE-Core text-to-image search through the existing DataProvider/Strategy architecture.

## 1. Download And Place Data

Download the AIC sample dataset from the team-provided storage link, then extract it at the repository root:

```text
AIC-HCMC-2026/
  AIC2026_sample/
    keyframes/
      keyframes/
        L01_V001/
        L01_V002/
        ...
    metadata/
      metadata/
        L01_V001.json
        L01_V002.json
        ...
    PECore-features/
      PECore-features/
        L01_V001/
        L01_V002/
        ...
```

The scripts also accept the flat layout where `keyframes/`, `metadata/`, and
`PECore-features/` contain the video folders/files directly.

The scripts auto-detect either:

```text
AIC-HCMC-2026/AIC2026_sample
AIC2026_sample beside AIC-HCMC-2026
```

If the dataset is stored somewhere else, pass it explicitly:

```bash
python scripts/ingest_embeddings_to_milvus.py --sample-root /path/to/AIC2026_sample
python scripts/check_index_integrity.py --sample-root /path/to/AIC2026_sample
python scripts/compare_linear_vs_milvus.py --sample-root /path/to/AIC2026_sample --query "a busy street"
```

On Windows PowerShell:

```powershell
python scripts/ingest_embeddings_to_milvus.py --sample-root "D:\path\to\AIC2026_sample"
```

Minimum required folders for Search by Text:
- `PECore-features/PECore-features` or `PECore-features`: precomputed PE-Core image embeddings (`.npy`)
- `keyframes/keyframes` or `keyframes`: frame images served through `/static/frames/...`
- `metadata/metadata` or `metadata`: video metadata used for FPS, timestamps, and YouTube IDs

## 2. Start databases

```bash
cd remote-server
docker compose up -d
```

## 3. Create Milvus index

```bash
cd remote-server
python scripts/create_milvus_index.py
```

Use `--recreate` only when you want to drop and rebuild `video_frames`.

## 4. Ingest embeddings

```bash
cd remote-server
python scripts/ingest_embeddings_to_milvus.py --copy-keyframes
```

Defaults:
- sample root: first existing `AIC2026_sample` inside or beside the repository
- Milvus collection: `video_frames`
- vector dim: `1280`
- metric/index: `COSINE` + `HNSW`, `M=16`, `efConstruction=256`
- search `ef`: `256` below top-50, `512` from top-50, and always at least `top_k`

## 5. Run backend

```bash
cd remote-server
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Optional CUDA 13 CAGRA search

HNSW remains the default. On an NVIDIA GPU with CUDA 13 support, CAGRA can use
the same API and strategies without changing the frontend.
It runs through cuVS in the Python backend, so the existing Milvus Docker image
does not need GPU or CUDA changes.

Stop the backend, install the optional dependencies, and build the ignored
local index:

```bash
pip install -r requirements-cagra.txt
python scripts/build_cagra_index.py
```

Then set:

```env
VECTOR_SEARCH_BACKEND=cagra
PECORE_DEVICE=cuda
PECORE_PRECISION=fp16
WARMUP_TEXT_ENCODER=true
CAGRA_SEARCH_WIDTH=32
```

Start the backend normally. Startup waits for the text encoder warmup, so the
first search does not pay model-loading and initial CUDA setup costs.
`/api/health` reports the selected search backend, encoder device, and encoder
precision.
PostgreSQL and Milvus still run because OCR, transcript, and temporal text
workflows use the existing databases.

Use `VECTOR_SEARCH_BACKEND=milvus` to return to HNSW. CAGRA artifacts are stored
under `remote-server/cache/` and are not committed. Rebuild them after changing
the cuVS version because its serialized index format is experimental.

### Optional Apple silicon MPS text encoding

On a MacBook with Apple silicon, PyTorch MPS can accelerate PE-Core text
encoding while Milvus HNSW still handles vector search:

```env
VECTOR_SEARCH_BACKEND=milvus
PECORE_DEVICE=mps
PECORE_PRECISION=fp32
```

CAGRA remains CUDA-only and should not be enabled on MPS.

### Optional Vietnamese-to-English translation

The search UI can translate selected semantic queries before sending them to
PE-Core. The provider is configured on the backend and is not exposed in the
UI. Queries are not saved to a database or disk; translated results use a small
in-process cache and otherwise remain in the current browser session.

Configure the ignored `remote-server/.env`:

```env
GOOGLE_CLOUD_PROJECT=your-project-id
GEMINI_API_KEY=your-key
GEMINI_TRANSLATION_MODEL=gemini-3.1-flash-lite
TRANSLATION_PROVIDER=nmt
WARMUP_TRANSLATION=true
```

Cloud Translation uses Application Default Credentials:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project your-project-id
```

The UI uses the same endpoint regardless of the configured provider:

```http
POST /api/translate
{"texts":["một người đang đi bộ"]}
```

`WARMUP_TRANSLATION=true` performs one translation during startup so the first
user search does not pay client initialization and connection setup costs.

For local-client proxy mode, set `local-client/local-backend/.env`:

```env
ENV_MODE=LOCAL
REMOTE_SERVER_URL=http://localhost:8000
```

Then run:

```bash
cd local-client/local-backend
uvicorn main:app --reload --port 8001
```

## 6. Test search

Remote server:

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d "{\"strategy_id\":\"nam_visual_search_v1\",\"query_groups\":[{\"semantic_query\":\"a street scene with people\",\"temporal_offset_ms\":0}],\"top_k\":10}"
```

Local sample mode can also test against `AIC2026_sample` without Milvus by setting `ENV_MODE=SAMPLE` in `local-client/local-backend/.env` and using the same `strategy_id`.

## 7. Validation And Evaluation

### Smoke test backend search

Start the backend first, then run:

```bash
cd remote-server
python scripts/smoke_search_queries.py --top-k 5
```

Useful options:

```bash
python scripts/smoke_search_queries.py \
  --backend-url http://localhost:8000 \
  --strategy-id nam_visual_search_v1 \
  --query "a busy street with people" \
  --query "cars on a road" \
  --top-k 10
```

Read the output as a quick correctness check: each query should return ranked rows with `video_id`, `frame_id`, `timestamp_ms`, `confidence`, and `frame_image_url`. Empty output usually means the backend is not running, the strategy was not discovered, or the PE-Core text model/Milvus index is unavailable.

### Compare linear vs Milvus

This validates HNSW search quality against brute-force search over `AIC2026_sample/PECore-features`.

```bash
cd remote-server
python scripts/compare_linear_vs_milvus.py --query "a busy street with people" --top-k 10
```

Add `--include-cagra` after preparing the optional CAGRA index to compare all
three result sets with the same query embedding.

Read the output:
- `overlap@5` and `overlap@10`: how many frame IDs match between exact linear search and Milvus HNSW in the top results.
- `score_difference_on_shared`: absolute cosine score difference for shared frame IDs. Large differences can indicate inconsistent normalization or indexing.
- The rank table shows linear and Milvus results side by side.

### Check index integrity

```bash
cd remote-server
python scripts/check_index_integrity.py
```

Read the output:
- `status: PASS`: schema/count/metadata look correct.
- `status: WARN`: core index is usable, but some image paths are missing.
- `status: FAIL`: vector dim, row count, duplicate `frame_id`, or required metadata is wrong.

The check verifies:
- vector count in Milvus
- vector dim is `1280`
- required metadata: `frame_id`, `video_id`, `frame_number`, `timestamp_ms`
- duplicate `frame_id`
- missing image paths under `remote-server/static` or `../AIC2026_sample/keyframes[/keyframes]`

### Evaluate a query set

Create a `queries.json` file:

```json
{
  "queries": [
    {"id": "q1", "semantic_query": "a busy street with people"},
    {"id": "q2", "semantic_query": "cars on a road"}
  ]
}
```

Run:

```bash
cd remote-server
python scripts/evaluate_query_set.py queries.json --top-k 10
```

Outputs:
- `evaluation_report.json`: query groups, top-k results, backend execution time, score distribution.
- `evaluation_report.csv`: flat table for spreadsheet inspection.

Because there is no ground truth yet, this report does not compute recall/MAP. Use `score_distribution.mean`, `score_distribution.max`, and visual inspection of top-k frame URLs to catch retrieval regressions.

---

# Search By Text Backend (PE-Core + Milvus)

## Overview

This section documents the backend Search by Text implementation for AIC HCMC 2026.

Goal:
- Input: text query
- Output: top-k relevant video frames
- Embedding model: `PE-Core-bigG-14-448`
- Vector database: Milvus
- Similarity: cosine similarity
- Index: HNSW

## Architecture

### Offline Indexing

Pipeline:

```text
Video
  ↓
TransNetV2
  ↓
Keyframes
  ↓
PE-Core Image Encoder
  ↓
1280-dim Image Embeddings
  ↓
Milvus (HNSW)
```

Each frame is stored with:

```json
{
  "frame_id": "L01_V001_000123",
  "video_id": "L01_V001",
  "frame_number": 123,
  "timestamp_ms": 4920,
  "image_url": "/static/frames/L01_V001/000123.jpg",
  "embedding": [1280]
}
```

`video_id` is the internal dataset identifier. Video metadata in PostgreSQL
stores a separate `youtube_id` used by the frontend player.

## Milvus Collection

Collection:

```text
video_frames
```

Schema:

```text
frame_id        VARCHAR (PK)
video_id        VARCHAR
frame_number    INT64
timestamp_ms    INT64
image_url       VARCHAR
vector          FLOAT_VECTOR(1280)
```

Index:

```python
metric_type = "COSINE"
index_type = "HNSW"

M = 16
efConstruction = 256
```

Search:

```python
base_ef = 512 if top_k >= 50 else 256
ef = max(base_ef, top_k)
```

Milvus requires:

```text
ef >= top_k
```

The API caps `top_k` at 1000. Deeper result sets use a broader HNSW search for
better recall, and `ef` still grows when `top_k` exceeds 512.

## Index Integrity Validation

Current verified result:

```text
status: PASS

vector_count: 3349
queried_rows: 3349

dim: 1280
expected_dim: 1280

duplicate_frame_ids: 0
missing_metadata_count: 0
missing_image_count: 0
```

Meaning:
- all vectors were inserted
- vector dimension is correct
- required metadata is present
- no duplicate frame IDs
- all image URLs resolve to available frame images

## Query Pipeline

Search flow:

```text
Text Query
  ↓
PE-Core Text Encoder
  ↓
1280-dim Text Embedding
  ↓
Normalize
  ↓
Milvus Search
  ↓
Top-K Frames
  ↓
Strategy Ranking
  ↓
API Response
```

## PE-Core Text Encoder

Model:

```text
hf-hub:timm/PE-Core-bigG-14-448
```

Output:

```text
shape = (1280,)
dtype = float32
L2 norm = 1.0
```

## Cold Start Behavior

PE-Core model loading on CPU is slow.

Observed timing:

```text
model_load: hardware and local model-cache dependent
first unique query after warmup: typically dominated by one text encode
repeated identical query: text embedding is served from a 128-entry memory cache
```

The cache stores the exact normalized embedding, so it improves repeated-query
latency without changing retrieval scores. It is memory-only and resets when
the backend restarts.

## Warmup Endpoint

To avoid first-request timeout:

```http
GET /api/warmup_text_encoder
```

or:

```http
POST /api/warmup_text_encoder
```

Example response:

```json
{
  "status": "ok",
  "model_load_ms": 0.0,
  "encode_ms": 200.0,
  "device": "cpu"
}
```

Actual values vary by CPU/GPU and whether the model was already loaded. Warmup
performs ten uncached text forward passes on the same dedicated worker used
by searches. This settles initial CUDA execution and wakes an idle GPU instead
of returning the cached `"warmup query"` embedding.

Recommendation:

```text
Set WARMUP_TEXT_ENCODER=true for demo and competition servers. Call the warmup
endpoint again immediately before a latency-sensitive demo if the GPU has been
idle for a long time.
```

## Search API

Endpoint:

```http
POST /api/search
```

Request:

```json
{
  "strategy_id": "nam_visual_search_v1",
  "query_groups": [
    {
      "semantic_query": "người đàn ông đang phát biểu",
      "text_query": "",
      "temporal_offset_ms": 0
    }
  ],
  "top_k": 10
}
```

Response example:

```json
{
  "results": [
    {
      "video_id": "L01_V004",
      "youtube_id": "YwRMKNp17ro",
      "frame_id": "L01_V004_002703",
      "frame_number": 2703,
      "timestamp_ms": 108120,
      "confidence": 0.2097,
      "frame_image_url": "/static/frames/L01_V004/002703.jpg",
      "fps": 25.0
    }
  ],
  "strategy_id": "nam_visual_search_v1",
  "total": 10,
  "execution_time_ms": 1543
}
```

## Example Smoke Test

```bash
python scripts/smoke_search_queries.py \
  --backend-url http://localhost:8000 \
  --query "người đàn ông đang phát biểu" \
  --top-k 10 \
  --timeout 30
```

Observed output:

```text
1  L01_V004  L01_V004_002703  108120   0.2097
2  L01_V005  L01_V005_031414  1256560  0.2062
3  L01_V005  L01_V005_008093  323720   0.2040
4  L01_V005  L01_V005_038690  1547600  0.2028
5  L01_V002  L01_V002_015470  618800   0.2027
```

## Timing Logs

Backend logs:

```text
[TIMER] request_received
[TIMER] model_load
[TIMER] text_encode
[TIMER] milvus_search
[TIMER] fusion
[TIMER] total_request
[TIMER] warmup_text_encoder
```

Purpose:
- identify bottlenecks
- debug timeout behavior
- benchmark search latency

## Current Status

Completed:
- PE-Core text encoder
- Milvus HNSW index
- optional cuVS CAGRA index
- embedding ingestion
- Search by Text API
- warmup endpoint
- index integrity validation
- smoke testing
- timing instrumentation
- validation scripts, including `compare_linear_vs_milvus.py`

Future work:
- run full linear-vs-Milvus benchmark after warmup
- OCR reranking
- transcript reranking
- temporal search expansion
- multimodal fusion
- query benchmark using AIC 2024/2025 query sets

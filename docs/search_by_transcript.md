# Search by Transcript — Topic-Based Chunking & Vector Search

## Tổng quan

Tính năng tìm kiếm transcript bằng **vector embedding** + **phân loại chủ đề tự động**, thay thế hoàn toàn phương pháp keyword-based cũ.

```
Query tiếng Việt → multilingual-e5-small encode → 384-dim vector
  → Dự đoán topic (Ẩm thực / Công nghệ / ...) từ 14 chủ đề
  → Milvus HNSW search trong collection transcript_chunks, filter theo topic
  → PostgreSQL hydrate raw_text
  → Trả về ranked chunks kèm topic, timestamp, text
```

---

## 1. Cài đặt

```bash
cd remote-server
pip install -r requirements.txt   # đã có sentence-transformers>=2.2.0
```

Kiểm tra `.env`:
```env
TRANSCRIPT_CHUNK_SEARCH_ENABLED=true
TRANSCRIPT_MODEL_ID=intfloat/multilingual-e5-small
TRANSCRIPT_MODEL_DEVICE=cpu        # hoặc cuda nếu có GPU
TRANSCRIPT_VECTOR_DIM=384
MILVUS_TRANSCRIPT_COLLECTION=transcript_chunks
```

---

## 2. Index dữ liệu transcript

### 2.1 Dry-run (kiểm tra trước)

```bash
cd remote-server
$env:PYTHONIOENCODING='utf-8'
python scripts/index_transcripts.py --sample-root ../AIC2026_sample --dry-run
```

Output mẫu:
```
DRY RUN — first 10 chunks:
  [    1] L01_V001 | Thời sự    |     8.0s →    51.0s | Kính chào và cảm ơn...
  [    2] L01_V001 | Công nghệ  |    30.0s →    70.0s | hơn 300 nhạc công...
  [    3] L01_V001 | Kinh tế    |    47.0s →    88.0s | giữ nhịp tăng trưởng...

Video Genres (dominant topic per video):
  L01_V001: Thời sự
  L03_V001: Ẩm thực
  L03_V002: Ẩm thực

Would index 531 chunks and tag 9 video genres.
```

### 2.2 Index thực tế

```bash
# Lần đầu — tạo collection + nạp toàn bộ
python scripts/index_transcripts.py --sample-root ../AIC2026_sample

# Cập nhật lại (xóa cũ trước)
python scripts/index_transcripts.py --sample-root ../AIC2026_sample --clear-existing
```

Output:
```
Indexing Milvus vectors: 100%|██████████| 531/531
PostgreSQL: upserted 531 chunk metadata rows
Milvus: indexed 531 chunk vectors
PostgreSQL: updated genre for 9 videos

Topic distribution:
  Thời sự      127  ██████████████████████████████████████████████████
  Sức khỏe      99  ██████████████████████████████████████
  Đời sống      90  ███████████████████████████████████
  Kinh tế       53  ████████████████████████
  ...
```

---

## 3. Khởi động

```bash
# Terminal 1: Backend
cd remote-server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Frontend
cd local-client/frontend
npm run dev
```

Server log sẽ hiện:
```
Transcript chunk search service ready
Discovering strategies...
  ✓ transcript_fusion_strategy  (Transcript Fusion v1  by Team AIC 2026)
Server ready — 4 strategy/strategies loaded.
```

---

## 4. Sử dụng API

### Endpoint

```
POST /api/search/transcript
Content-Type: application/json

{
  "query":        "cách nấu phở bò",    // required
  "top_k":        20,                    // optional, default 100, max 1000
  "topic_filter": ""                     // optional: "Ẩm thực" | "Công nghệ" | ...
                                         // "" = auto-predict từ nội dung query
}
```

### Response

```json
{
  "results": [
    {
      "chunk_id":      42,
      "video_id":      "L03_V001",
      "topic":         "Ẩm thực",
      "start_time_ms": 30000,
      "end_time_ms":   70000,
      "text":          "cách nấu phở bò truyền thống Nam Định...",
      "score":         0.952
    }
  ],
  "total":             15,
  "execution_time_ms": 234
}
```

### Ví dụ curl

```bash
# Auto-predict topic
curl -X POST http://localhost:8000/api/search/transcript \
  -H "Content-Type: application/json" \
  -d '{"query":"cách nấu phở bò","top_k":5}'

# Ép topic thủ công
curl -X POST http://localhost:8000/api/search/transcript \
  -H "Content-Type: application/json" \
  -d '{"query":"nguyên liệu","top_k":5,"topic_filter":"Kinh tế"}'
```

---

## 5. Sử dụng UI

Vào http://localhost:3000, chọn tab **Transcripts** (xanh lá).

```
┌──────────────────────────────────────────────────────────────────┐
│ [Search transcript chunks...]  Top K [20] Topic [Auto ▼] [Search Transcripts] │
└──────────────────────────────────────────────────────────────────┘
```

| Thành phần | Chức năng |
|---|---|
| **Search input** | Nhập query tiếng Việt hoặc Anh. Enter để search |
| **Top K** | Số chunk kết quả (1–200). Default 20 |
| **Topic** | **Auto** = backend tự dự đoán. Chọn cụ thể để ép filter |
| **Search Transcripts** | Thực thi search |
| **Topic badge** | Màu riêng cho 14 chủ đề (cam = Ẩm thực, xanh dương = Công nghệ...) |
| **Timestamp** | `00:30 – 01:10` — click để mở YouTube tại giây đó |
| **Score** | Xanh ≥80%, vàng ≥60%, đỏ <60% |

---

## 6. 14 Chủ đề

| Chủ đề | Màu badge | Mô tả |
|---|---|---|
| Ẩm thực | Cam | Nấu ăn, món ăn, nguyên liệu |
| Công nghệ | Xanh dương | AI, robot, chip, phần mềm |
| Du lịch | Cyan | Địa điểm, khách sạn, tour |
| Thể thao | Đỏ | Bóng đá, điền kinh, giải đấu |
| Giáo dục | Tím | Bài giảng, trường học, thi cử |
| Kinh tế | Vàng đậm | Giá cả, thị trường, doanh nghiệp |
| Sức khỏe | Xanh lá | Bệnh viện, thuốc, dinh dưỡng |
| Giải trí | Hồng | Phim ảnh, âm nhạc, gameshow |
| Thời sự | Xám | Tin tức, chính trị, xã hội |
| Văn hóa | Indigo | Nghệ thuật, lễ hội, truyền thống |
| Đời sống | Teal | Gia đình, mua sắm, sinh hoạt |
| Môi trường | Xanh lá đậm | Khí hậu, ô nhiễm, thiên nhiên |
| Giao thông | Vàng | Đường xá, kẹt xe, cầu đường |
| Pháp luật | Nâu | Tòa án, luật, quy định |

---

## 7. Strategy tích hợp

Strategy **Transcript Fusion v1** trong **Frames mode** tự động dùng transcript chunk search:

```
Frames mode → Strategy: Transcript Fusion v1
  Semantic: "phóng viên"        → visual PE-Core search
  Text:     "bản tin thời sự"   → transcript chunk vector search
  Genre:    Thời sự             → filter theo genre
  ↓
RRF fusion (visual + transcript chunks + OCR)
  ↓
Ranked frames kèm confidence
```

Chọn strategy này từ dropdown trong Frames mode để kết hợp visual + transcript.

---

## 8. Genre filter trong Frames mode

Genre của mỗi video được tính tự động từ dominant chunk topic khi chạy `index_transcripts.py`.

Trong Frames mode, dropdown **Genre** cho phép lọc video theo thể loại:
- **All** — không filter
- **Ẩm thực** — chỉ search trong video nấu ăn
- **Thời sự** — chỉ search trong video tin tức
- ...

Genre filter hoạt động ở cả visual search (Milvus `video_id in [...]`) và strategy fusion (dynamic weight tuning).

---

## 9. Kiến trúc dữ liệu

### Milvus: `transcript_chunks`

| Field | Type | Mô tả |
|---|---|---|
| chunk_id | INT64 (PK) | ID tự tăng |
| video_id | VARCHAR(64) | Mã video |
| topic | VARCHAR(64) | Chủ đề dự đoán |
| start_time_ms | INT64 | Thời gian bắt đầu chunk |
| end_time_ms | INT64 | Thời gian kết thúc chunk |
| vector | FLOAT_VECTOR(384) | Embedding từ multilingual-e5-small |

Index: HNSW, COSINE metric, M=16, efConstruction=256

### PostgreSQL: `transcript_chunks_metadata`

| Column | Type | Mô tả |
|---|---|---|
| chunk_id | BIGINT (PK) | Khớp với Milvus chunk_id |
| video_id | VARCHAR(64) | Mã video |
| topic | VARCHAR(64) | Chủ đề |
| start_time_ms | BIGINT | |
| end_time_ms | BIGINT | |
| raw_text | TEXT | Nội dung đầy đủ của chunk |

### PostgreSQL: `videos.genre`

| Column | Type | Mô tả |
|---|---|---|
| genre | VARCHAR(64) | Thể loại chính của video (dominant topic) |

---

## 10. Lịch sử phát triển ý tưởng Transcript Search

### 10.1 Giai đoạn 1: Keyword-based Search (Ban đầu)

**Ý tưởng:** Tìm kiếm transcript bằng cách match từ khóa trực tiếp.

```
Query: "phở bò"
  → PostgreSQL full-text search: WHERE text @@ plainto_tsquery('phở bò')
  → Trả về các transcript intervals chứa từ khóa
  → Map sang frames trong khoảng thời gian đó
```

**Hạn chế:**
- Chỉ tìm được exact match, không hiểu ngữ nghĩa
- "cách nấu phở" ≠ "làm phở" ≠ "phở truyền thống"
- Không tận dụng được vector embeddings đã có

---

### 10.2 Giai đoạn 2: Vector-based Transcript Search (Hiện tại)

**Ý tưởng:** Encode transcript thành vector, dùng semantic search.

```
Query: "cách nấu phở bò truyền thống"
  → multilingual-e5-small encode → 384-dim vector
  → Milvus HNSW search trong transcript_chunks collection
  → Trả về top-K chunks có cosine similarity cao nhất
  → Phân loại topic tự động (14 chủ đề)
  → PostgreSQL hydrate raw_text
```

**Cải tiến:**
- Hiểu ngữ nghĩa: "nấu phở" ≈ "làm phở" ≈ "chế biến phở"
- Topic classification giúp filter nhanh
- Tận dụng multilingual-e5-small hỗ trợ cả tiếng Việt và Anh

**Kiến trúc dữ liệu:**
- Milvus `transcript_chunks`: lưu vector 384-dim + metadata
- PostgreSQL `transcript_chunks_metadata`: lưu raw_text để hydrate
- PostgreSQL `videos.genre`: dominant topic per video

---

### 10.3 Giai đoạn 3: Two-Stage Late Fusion Pipeline (Mới nhất)

**Vấn đề của Giai đoạn 2:**
- Transcript search và Visual search chạy độc lập
- Không có sự tương tác giữa "what is said" và "what is shown"
- Khi user query cả semantic + text, kết quả không được fusion tối ưu

**Ý tưởng Two-Stage:**
Thay vì search toàn cục, dùng transcript để **thu hẹp không gian search** trước, sau đó mới chạy visual search trong không gian hẹp đó.

```
User Query: Semantic="cách nấu phở bò", Text="phở bò"
  │
  ├─► Stage 1 (Coarse): Transcript vector search
  │     E5 encode "phở bò" → 384-dim
  │     Milvus search transcript_chunks → Top 20 chunks
  │     Extract candidate video_ids: {L03_V001, L03_V002, ...}
  │
  ├─► Build Milvus expr: 'video_id in ["L03_V001", "L03_V002"]'
  │
  ├─► Stage 2 (Fine): PE-Core visual search với filter
  │     PE-Core encode "cách nấu phở bò" → 1280-dim
  │     Milvus search video_frames WITH expr filter
  │     Chỉ search frames trong candidate videos
  │
  └─► Fusion: Dynamic α weighting
        Score_final = α · Score_transcript + (1-α) · Score_visual
        α điều chỉnh theo genre và signal quality
```

**Lợi ích:**
- **Hiệu quả:** Stage 2 chỉ search trong ~5-10 videos thay vì toàn bộ dataset
- **Chính xác:** Kết hợp được "what is said" (transcript) và "what is shown" (visual)
- **Linh hoạt:** Dynamic α tự điều chỉnh dựa trên độ tin cậy của mỗi signal
- **Fallback an toàn:** Nếu Stage 1 fail → fallback về visual-only; nếu Stage 2 fail → transcript-only

**So sánh với RRF (Reciprocal Rank Fusion) cũ:**

| Tiêu chí | RRF (TranscriptFusionStrategy) | Two-Stage Late Fusion |
|----------|--------------------------------|------------------------|
| Search space | Toàn cục cho cả transcript và visual | Transcript toàn cục → Visual thu hẹp |
| Fusion method | Rank-based (1/(k+rank)) | Score-based (α·transcript + (1-α)·visual) |
| Vector space | Không so sánh trực tiếp (rank-only) | Không so sánh trực tiếp (weighted score) |
| Hiệu suất | 2 searches độc lập | Stage 2 nhanh hơn nhờ expr filter |
| Độ chính xác | Tốt cho rank aggregation | Tốt hơn khi transcript và visual bổ sung nhau |

---

## 11. Các thay đổi code cho Two-Stage Late Fusion

### 11.1 Files đã sửa

| File | Thay đổi | Dòng |
|------|----------|------|
| `remote-server/app/services/transcript_search.py` | Thêm method `search_for_candidates()` | +32 lines |
| `remote-server/app/data_provider.py` | Thêm method `get_raw_data_two_stage()` | +170 lines |
| `remote-server/app/data_provider.py` | Fix bug `self._collection` → `self._metadata_collection()` | 1 line |
| `remote-server/app/strategies/two_stage_fusion_strategy.py` | **Tạo mới** — Two-Stage Fusion strategy | 484 lines |
| `remote-server/scripts/test_two_stage_fusion.py` | **Tạo mới** — Test suite 37 tests | 484 lines |
| `docs/test_two_stage_fusion_guide.md` | **Tạo mới** — Test matrix & troubleshooting | 229 lines |

### 11.2 Chi tiết thay đổi

#### `transcript_search.py` — Thêm `search_for_candidates()`

```python
async def search_for_candidates(
    self,
    query: str,
    top_k: int = 20,
    topic_filter: str | None = None,
) -> tuple[list[dict], set[str]]:
    """
    Stage 1: Tìm candidate video_ids từ transcript search.
    
    Returns:
        (chunks, candidate_video_ids) — danh sách chunks và tập video_ids
    """
    # Encode query bằng E5 model
    query_vector = self.encode_query(query)
    
    # Search trong Milvus transcript_chunks collection
    chunk_hits = milvus_client.search_transcript_chunks(
        self._collection,
        query_vector.tolist(),
        top_k=top_k,
        topic_filter=topic_filter,
    )
    
    # Extract unique video_ids
    candidate_video_ids = {h["video_id"] for h in chunk_hits}
    return chunk_hits, candidate_video_ids
```

**Mục đích:** Tách riêng logic "tìm candidate videos" để Stage 2 có thể dùng.

---

#### `data_provider.py` — Thêm `get_raw_data_two_stage()`

```python
async def get_raw_data_two_stage(
    self,
    query_groups: list[dict],
    limit: int = 1000,
    video_genre: str = "All",
    stage1_top_k: int = 20,
) -> dict:
    """
    Two-stage late fusion retrieval pipeline.
    
    Stage 1: Transcript search → candidate video_ids
    Stage 2: Visual search với Milvus expr filter
    """
    # Stage 1: Transcript search
    if self._transcript_search and combined_text_query:
        chunks, candidate_vids = await self._transcript_search.search_for_candidates(
            combined_text_query, top_k=stage1_top_k
        )
        stage1_chunks = chunks
        stage1_video_ids = candidate_vids
    
    # Build Milvus expr cho Stage 2
    if stage1_video_ids:
        quoted_vids = ", ".join(f'"{vid}"' for vid in stage1_video_ids)
        stage2_expr = f"video_id in [{quoted_vids}]"
    
    # Stage 2: Visual search với filter
    for group in query_groups:
        if semantic_query:
            query_vector = await self._encode_text(semantic_query)
            hits = milvus_client.vector_search(
                collection, query_vector.tolist(),
                top_k=search_limit,
                expr=stage2_expr,  # ← Filter chỉ search trong candidate videos
            )
    
    # Fallback nếu Stage 2 rỗng
    if not stage2_frames and combined_semantic_query:
        # Chạy global search không có filter
        ...
    
    return {
        "frames": frames,
        "stage1_chunks": stage1_chunks,
        "stage1_video_ids": stage1_video_ids,
        "stage2_frames": stage2_frames,
        ...
    }
```

**Mục đích:** Implement logic 2-stage, build Milvus expr, xử lý fallback.

**Bug fix:** `_frames_for_transcript_chunks()` reference `self._collection` (không tồn tại) → sửa thành `self._metadata_collection()`.

---

#### `two_stage_fusion_strategy.py` — Strategy mới

```python
class TwoStageFusionStrategy(BaseStrategy):
    """
    Two-Stage Late Fusion Retrieval Pipeline.
    
    Stage 1: Transcript vector search → candidate videos
    Stage 2: PE-Core visual search filtered to candidates
    Fusion: Dynamic α = f(signal_quality, genre)
    """
    
    name = "Two-Stage Fusion v1"
    
    async def search(self, query_groups, limit, video_genre):
        # Override để dùng get_raw_data_two_stage() thay vì get_raw_data()
        raw_data = await self.data_provider.get_raw_data_two_stage(
            query_groups, limit=limit, video_genre=video_genre
        )
        return self.fusion_and_temporal(raw_data, query_groups)
    
    def fusion_and_temporal(self, raw_data, query_groups):
        # Kiểm tra stage nào có kết quả
        has_stage1 = bool(stage1_chunks)
        has_stage2 = bool(stage2_frames)
        
        # Fallback nếu thiếu stage
        if not has_stage1: return visual_only_results
        if not has_stage2: return transcript_only_results
        
        # Build rankings
        transcript_ranked = self._build_transcript_ranking(stage1_chunks)
        visual_ranked = self._build_visual_ranking(stage2_frames)
        
        # Compute dynamic α
        alpha = _compute_dynamic_alpha(
            avg_transcript_score, avg_visual_score,
            has_semantic, video_genre
        )
        
        # Weighted fusion
        fused = self._weighted_fusion(
            transcript_ranked, visual_ranked, alpha
        )
        
        # Apply boosts
        if ocr_ranked: fused = self._apply_ocr_boost(fused, ocr_ranked)
        if len(query_groups) > 1: fused = self._apply_temporal_boost(fused, query_groups)
        
        return fused
```

**Các hàm helper:**

```python
def _compute_dynamic_alpha(transcript_score, visual_score, has_semantic, genre):
    """
    Dynamic α dựa trên signal quality và genre.
    
    - Nếu transcript_score < 0.05 → α = 0 (chỉ dùng visual)
    - Nếu visual_score < 0.05 → α = 1 (chỉ dùng transcript)
    - Ngược lại: α = base_alpha(genre) + 0.2 * (ratio - 0.5)
    """
    ...

GENRE_ALPHA_MAP = {
    "Thời sự": 0.50,    # Cân bằng speech + visual
    "Giáo dục": 0.35,   # Visual (slides) quan trọng hơn
    "Ẩm thực": 0.25,    # Visual (cooking actions) rất quan trọng
    "Pháp luật": 0.45,  # Speech + documents
    ...
}
```

---

### 11.3 Luồng dữ liệu chi tiết

```
User Query: Semantic="cách nấu phở bò", Text="phở bò"
  │
  ▼
TwoStageFusionStrategy.search()
  │
  ├─► DataProvider.get_raw_data_two_stage()
  │     │
  │     ├─► Stage 1: TranscriptSearchService.search_for_candidates("phở bò")
  │     │     │
  │     │     ├─► E5 encode "phở bò" → 384-dim vector
  │     │     ├─► Milvus search transcript_chunks → Top 20 chunks
  │     │     └─► Return (chunks, {L03_V001, L03_V002, L01_V005})
  │     │
  │     ├─► Build expr: 'video_id in ["L03_V001", "L03_V002", "L01_V005"]'
  │     │
  │     ├─► Stage 2: PE-Core encode "cách nấu phở bò" → 1280-dim
  │     │     │
  │     │     └─► Milvus search video_frames WITH expr
  │     │           → Chỉ trả về frames từ 3 videos trên
  │     │
  │     ├─► Fetch metadata từ PostgreSQL
  │     │     ├─► transcript_chunks_metadata (raw_text)
  │     │     └─► videos (youtube_id, fps, genre)
  │     │
  │     └─► Return raw_data với stage1_chunks, stage2_frames, ...
  │
  ├─► fusion_and_temporal(raw_data, query_groups)
  │     │
  │     ├─► _build_transcript_ranking()
  │     │     └─► Group by video, lấy max score per video
  │     │         [{video_id: L03_V001, _score: 0.92}, ...]
  │     │
  │     ├─► _build_visual_ranking()
  │     │     └─► Sort frames by score descending
  │     │         [{frame_id: f1, _score: 0.85}, ...]
  │     │
  │     ├─► _compute_dynamic_alpha()
  │     │     └─► avg_transcript=0.85, avg_visual=0.75, genre=Ẩm thực
  │     │         → α = 0.25 (base) + 0.2 * (0.53 - 0.5) = 0.256
  │     │
  │     ├─► _weighted_fusion()
  │     │     └─► For each transcript video:
  │     │           confidence = 0.256 * 0.92 + (1-0.256) * 0.85 = 0.87
  │     │
  │     ├─► _apply_ocr_boost()
  │     │     └─► If frame has OCR match: confidence += 0.05
  │     │
  │     └─► _apply_temporal_boost()
  │           └─► If multi-step query: +0.05 for nearby frames
  │
  └─► Return sorted results by confidence descending
```

---

### 11.4 Test suite

File: `remote-server/scripts/test_two_stage_fusion.py`

**37 tests chia thành 13 nhóm:**

| Nhóm | Số test | Kiểm tra gì |
|------|---------|-------------|
| `TestDynamicAlpha` | 8 | Tính toán α, giới hạn [0,1], genre mapping |
| `TestFusionMath` | 3 | Weighted fusion, α=0 (visual-only), α=1 (transcript-only) |
| `TestOCRBoost` | 3 | +5% boost, cap tại 1.0, không match thì không boost |
| `TestTemporalBoost` | 3 | +5% boost, tolerance boundary, không self-boost |
| `TestFallbackPaths` | 3 | Stage 1 rỗng, Stage 2 rỗng, cả hai rỗng |
| `TestMilvusExprFormatting` | 4 | Build expr string, single/multi/empty video_ids |
| `TestResultStructure` | 2 | Schema compliance (8 fields), sort order |
| `TestRankingBuilders` | 3 | Best-per-video, sort descending, no-semantic zeros |
| `TestDataProviderTwoStageAsync` | 3 | Stage 1 called/skipped, expr injection |
| `TestScenarioA_HappyPath` | 1 | End-to-end với strong match |
| `TestScenarioB_VisualHeavy` | 2 | Visual-heavy query, α giảm |
| `TestScenarioC_ZeroTranscript` | 1 | Gibberish query → fallback |
| `TestScenarioD_OCRAndTemporal` | 1 | Multi-step + OCR boost |

**Chạy test:**
```bash
cd remote-server

# Smoke test nhanh
python scripts/test_two_stage_fusion.py --smoke

# Unit tests đầy đủ
python scripts/test_two_stage_fusion.py --unit -v

# Tất cả tests (bao gồm async)
python scripts/test_two_stage_fusion.py -v
```

---

## 12. Troubleshooting

| Vấn đề | Giải pháp |
|---|---|
| `Transcript chunk search is not available` | Chạy `index_transcripts.py` để nạp dữ liệu. Kiểm tra `.env` có `TRANSCRIPT_CHUNK_SEARCH_ENABLED=true` |
| `sentence-transformers not found` | `pip install sentence-transformers>=2.2.0` |
| `collection not found` | Backend tự tạo collection khi khởi động nếu `TRANSCRIPT_CHUNK_SEARCH_ENABLED=true` |
| Search chậm | Chưa có INVERTED index trên `topic` (Milvus 2.3.4 không hỗ trợ). Scalar filter vẫn hoạt động với default index |
| Windows console lỗi Unicode | Set `$env:PYTHONIOENCODING='utf-8'` trước khi chạy script |
| NumPy version conflict | `pip install "numpy<2"` nếu gặp lỗi `_ARRAY_API not found` |
| `TwoStageFusionStrategy` không xuất hiện | Restart backend, kiểm tra log có lỗi import không |
| `Stage 2 returned 0 results` | Candidate videos từ Stage 1 không có frames trong Milvus. Kiểm tra `ingest_embeddings_to_milvus.py` đã chạy chưa |
| `alpha=0.000` cho mọi query | Transcript scores quá thấp (<0.05). Kiểm tra E5 model và transcript chunks có meaningful scores không |
| Milvus `expr` syntax error | Video_id chứa ký tự đặc biệt. Chỉ cho phép alphanumeric + underscore |

---

## 13. Tài liệu liên quan

| Tài liệu | Mô tả |
|----------|-------|
| [test_two_stage_fusion_guide.md](test_two_stage_fusion_guide.md) | Ma trận test 4 kịch bản, log verification, troubleshooting chi tiết |
| [strategy_guide.md](strategy_guide.md) | Hướng dẫn viết strategy mới |
| [architecture.md](architecture.md) | Kiến trúc hệ thống tổng thể |
| [db_schema.md](db_schema.md) | Schema PostgreSQL + Milvus |
| [vector_search_benchmark_report.md](vector_search_benchmark_report.md) | Benchmark HNSW vs CAGRA vs ScaNN |
